"""Event tools: events."""

import asyncio
import json
import logging
from collections import defaultdict

from fastmcp import FastMCP

from .._state import _check_session, _get_project_dbs, _log_audit

logger = logging.getLogger("tessera.server")


def register_event_tools(mcp: FastMCP) -> None:
    """Register event/hook analysis tools."""

    @mcp.tool()
    async def events(
        query: str = "",
        direction: str = "all",
        detect_mismatches: bool = False,
        mismatch_filter: str = "all",
        limit: int = 50,
        offset: int = 0,
        session_id: str = "",
    ) -> str:
        """Analyze event/hook registrations and emissions across the codebase.

        Tracks directional event edges: which functions REGISTER listeners on events
        and which functions FIRE/EMIT events. Works across WordPress hooks (PHP),
        EventEmitter/DOM events (JS/TS), and Django signals (Python).

        **Common patterns:**
        - All events in the project: events()
        - Who listens to an event: events("pum_popup_saved", direction="registers_on")
        - Who fires an event: events("pum_popup_saved", direction="fires")
        - Find events with prefix: events("pum_%")
        - All mismatches: events(detect_mismatches=True)
        - Only orphaned listeners: events(detect_mismatches=True, mismatch_filter="orphaned")
        - Only unfired events: events(detect_mismatches=True, mismatch_filter="unfired")
        - Page through results: events("pum_%", limit=20, offset=40)

        **Direction values:**
        - "all" (default): Show both registrations and emissions
        - "registers_on": Only listener registrations (add_action, on, connect)
        - "fires": Only event emissions (do_action, emit, send)

        **Mismatch detection (detect_mismatches=True):**
        - ORPHANED_LISTENER: Registered but never fired in indexed code
        - UNFIRED_EVENT: Fired but no listener registered in indexed code

        **Mismatch filter (requires detect_mismatches=True):**
        - "all" (default): Show both orphaned listeners and unfired events
        - "orphaned": Only ORPHANED_LISTENER mismatches
        - "unfired": Only UNFIRED_EVENT mismatches

        Args:
            query: Event/hook name or pattern (supports % wildcard). Empty = all events.
            direction: Filter by direction — "all", "registers_on", or "fires".
            detect_mismatches: When true, analyze event registration/emission balance.
            mismatch_filter: Filter mismatches — "all", "orphaned", or "unfired".
            limit: Max results to return (default 50). Use 0 for unlimited.
            offset: Skip this many results for pagination (default 0).
        """
        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"

        dbs = _get_project_dbs(scope)
        if not dbs:
            _log_audit("events", 0, agent_id=agent_id)
            return "Error: No accessible projects"

        try:
            event_name = query if query else None

            tasks = [
                asyncio.to_thread(db.get_events, event_name=event_name, direction=direction)
                for pid, pname, db in dbs
            ]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            all_events = []
            for (pid, pname, _db), result in zip(dbs, results_list, strict=False):
                if isinstance(result, Exception):
                    logger.warning("Events query on project %d failed: %s", pid, result)
                    continue
                for r in result:
                    entry = {
                        "event_name": r["event_name"],
                        "direction": r["direction"],
                        "from_symbol": r["from_symbol"],
                        "from_scope": r.get("from_scope", ""),
                        "file": r.get("from_file", ""),
                        "line": r.get("line", 0),
                        "project_id": pid,
                        "project_name": pname,
                    }
                    subtype = r.get("subtype", "")
                    if subtype:
                        entry["event_subtype"] = subtype
                    all_events.append(entry)

            output: dict = {}

            effective_limit = limit if limit > 0 else None

            if detect_mismatches:
                # Group by event name to detect mismatches
                by_event: dict[str, dict] = defaultdict(lambda: {
                    "registers": [], "fires": [],
                })
                for e in all_events:
                    key = e["event_name"]
                    if e["direction"] == "registers_on":
                        by_event[key]["registers"].append(e)
                    elif e["direction"] == "fires":
                        by_event[key]["fires"].append(e)

                mismatches = []
                for name, data in sorted(by_event.items()):
                    if data["registers"] and not data["fires"]:
                        mismatches.append({
                            "event_name": name,
                            "issue": "ORPHANED_LISTENER",
                            "detail": f"{len(data['registers'])} listener(s), never fired",
                            "locations": [
                                {
                                    "symbol": r["from_symbol"],
                                    "file": r["file"],
                                    "line": r["line"],
                                    "project": r["project_name"],
                                }
                                for r in data["registers"]
                            ],
                        })
                    elif data["fires"] and not data["registers"]:
                        mismatches.append({
                            "event_name": name,
                            "issue": "UNFIRED_EVENT",
                            "detail": f"Fired {len(data['fires'])} time(s), no listeners",
                            "locations": [
                                {
                                    "symbol": r["from_symbol"],
                                    "file": r["file"],
                                    "line": r["line"],
                                    "project": r["project_name"],
                                }
                                for r in data["fires"]
                            ],
                        })

                if mismatch_filter == "orphaned":
                    mismatches = [m for m in mismatches if m["issue"] == "ORPHANED_LISTENER"]
                elif mismatch_filter == "unfired":
                    mismatches = [m for m in mismatches if m["issue"] == "UNFIRED_EVENT"]

                total_mismatches = len(mismatches)
                paged = mismatches[offset:offset + effective_limit] if effective_limit else mismatches[offset:]
                output["mismatches"] = paged
                output["summary"] = {
                    "total_events": len(by_event),
                    "total_mismatches": total_mismatches,
                    "orphaned_listeners": sum(1 for m in mismatches if m["issue"] == "ORPHANED_LISTENER"),
                    "unfired_events": sum(1 for m in mismatches if m["issue"] == "UNFIRED_EVENT"),
                    "healthy": len(by_event) - total_mismatches,
                    "showing": len(paged),
                    "offset": offset,
                    "filter": mismatch_filter,
                }
            else:
                total = len(all_events)
                paged = all_events[offset:offset + effective_limit] if effective_limit else all_events[offset:]
                output["events"] = paged
                output["total"] = total
                output["showing"] = len(paged)
                output["offset"] = offset

            _log_audit("events", len(all_events), agent_id=agent_id)
            return json.dumps(output, indent=2)
        except Exception as e:
            logger.exception("Events tool error")
            _log_audit("events", 0, agent_id=agent_id)
            return f"Error querying events: {e!s}"
