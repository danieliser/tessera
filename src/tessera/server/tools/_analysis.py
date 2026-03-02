"""Analysis tools: impact, cross_refs, collection_map."""

import asyncio
import json
import logging

from fastmcp import FastMCP

from .._state import _check_session, _get_project_dbs, _log_audit

logger = logging.getLogger("tessera.server")


def register_analysis_tools(mcp: FastMCP) -> None:
    """Register analysis-related tools."""

    @mcp.tool()
    async def impact(symbol_name: str, depth: int = 3, include_types: bool = True, session_id: str = "") -> str:
        """Analyze the blast radius of changing a symbol — what depends on it and what breaks.

        Traverses the dependency graph outward from the named symbol up to `depth` hops,
        collecting every function, class, and module that directly or transitively depends
        on it. Use this before refactoring to understand the full impact of a change.

        **Common patterns:**
        - Before renaming a function: impact("handle_request") → see all callers at every depth
        - Before changing a class interface: impact("ProjectDB", depth=2)
        - Assess risk of a change: impact("normalize_bm25_score") → if few dependents, safe to modify

        **Depth guide:**
        - depth=1: Direct callers only. Fast. Good for quick "who uses this?" checks.
        - depth=2-3: Typical refactoring scope. Shows indirect dependents.
        - depth=5+: Full transitive closure. Slower, but shows cascading impact.

        Args:
            symbol_name: Exact symbol name to analyze.
            depth: How many hops to traverse (default 3). Higher = more complete but slower.
            include_types: Include type references in the graph (default True).
        """
        from .._state import _project_graphs

        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"

        dbs = _get_project_dbs(scope)
        if not dbs:
            _log_audit("impact", 0, agent_id=agent_id)
            return "Error: No accessible projects"

        try:
            # Parallel query pattern
            tasks = [
                asyncio.to_thread(db.get_impact, symbol_name, depth, include_types)
                for pid, pname, db in dbs
            ]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            all_results = []
            for (pid, pname, _db), result in zip(dbs, results_list, strict=False):
                if isinstance(result, Exception):
                    logger.warning("Query on project %d failed: %s", pid, result)
                    continue
                for r in result:
                    r["project_id"] = pid
                    r["project_name"] = pname

                # Rank by PPR if graph available
                graph = _project_graphs.get(pid)
                if graph and not graph.is_sparse_fallback():
                    try:
                        affected_ids = [r['id'] for r in result if 'id' in r]
                        if affected_ids:
                            ppr_scores = graph.personalized_pagerank(affected_ids)
                            for r in result:
                                r['ppr_relevance'] = ppr_scores.get(r.get('id', -1), 0.0)
                            result = sorted(result, key=lambda x: -x.get('ppr_relevance', 0.0))
                    except Exception as e:
                        logger.warning("PPR ranking failed for impact on project %d: %s", pid, e)

                all_results.extend(result)

            _log_audit("impact", len(all_results), agent_id=agent_id)
            return json.dumps(all_results, indent=2)
        except Exception as e:
            logger.exception("Impact tool error")
            _log_audit("impact", 0, agent_id=agent_id)
            return f"Error analyzing impact: {str(e)}"

    @mcp.tool()
    async def cross_refs(symbol_name: str, session_id: str = "") -> str:
        """Find cross-project references for a symbol.

        Returns references where the symbol is defined in one project
        and referenced in another.
        """
        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"

        dbs = _get_project_dbs(scope)
        if not dbs:
            _log_audit("cross_refs", 0, agent_id=agent_id)
            return "Error: No accessible projects"

        try:
            # Parallel query: find definitions in each project
            def_tasks = [
                asyncio.to_thread(db.lookup_symbols, symbol_name)
                for pid, pname, db in dbs
            ]
            def_results = await asyncio.gather(*def_tasks, return_exceptions=True)

            # Collect which projects define this symbol
            definition_projects = {}
            for (pid, pname, _db), result in zip(dbs, def_results, strict=False):
                if isinstance(result, Exception):
                    logger.warning("Query on project %d failed: %s", pid, result)
                    continue
                if result:
                    definition_projects[pid] = pname

            if not definition_projects:
                result_dict = {"symbol": symbol_name, "definition_projects": {}, "cross_refs": []}
                _log_audit("cross_refs", 0, agent_id=agent_id)
                return json.dumps(result_dict, indent=2)

            # Parallel query: find callers of this symbol in each project
            caller_tasks = [
                asyncio.to_thread(db.get_callers, symbol_name=symbol_name)
                for pid, pname, db in dbs
            ]
            caller_results = await asyncio.gather(*caller_tasks, return_exceptions=True)

            # Build cross-project references
            cross_ref_list = []
            for (pid, pname, _db), result in zip(dbs, caller_results, strict=False):
                if isinstance(result, Exception):
                    logger.warning("Query on project %d failed: %s", pid, result)
                    continue
                for caller in result:
                    for def_pid, def_pname in definition_projects.items():
                        if def_pid != pid:
                            cross_ref_list.append({
                                "from_project_id": pid,
                                "from_project_name": pname,
                                "to_project_id": def_pid,
                                "to_project_name": def_pname,
                                "file": caller.get("file_id", ""),
                                "line": caller.get("line", 0),
                                "kind": caller.get("kind", "function"),
                            })

            result_dict = {
                "symbol": symbol_name,
                "definition_projects": {str(k): {"id": k, "name": v} for k, v in definition_projects.items()},
                "cross_refs": cross_ref_list,
            }

            _log_audit("cross_refs", len(cross_ref_list), agent_id=agent_id)
            return json.dumps(result_dict, indent=2)
        except Exception as e:
            logger.exception("cross_refs error")
            _log_audit("cross_refs", 0, agent_id=agent_id)
            return f"Error: {str(e)}"

    @mcp.tool()
    async def collection_map(collection_id: int = 0, session_id: str = "") -> str:
        """Get inter-project dependency graph.

        If collection_id is 0, use all allowed projects in scope.
        """
        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"

        # Get allowed projects
        dbs = _get_project_dbs(scope)
        if not dbs:
            _log_audit("collection_map", 0, agent_id=agent_id)
            return "Error: No accessible projects"

        try:
            # Query symbol definitions for each project in parallel
            async def _query_symbols(db):
                """Query all symbols for a database."""
                symbols = await asyncio.to_thread(db.lookup_symbols, "*")
                return symbols

            tasks = [
                _query_symbols(db)
                for pid, pname, db in dbs
            ]
            symbols_list = await asyncio.gather(*tasks, return_exceptions=True)

            # Build project node info and symbol index (name → project_ids)
            projects_dict = {}
            symbol_definitions = {}  # symbol_name → set of project_ids that define it

            for (pid, pname, _db), symbols_result in zip(dbs, symbols_list, strict=False):
                if isinstance(symbols_result, Exception):
                    logger.warning("Symbol query on project %d failed: %s", pid, symbols_result)
                    symbol_count = 0
                    all_symbols = []
                else:
                    all_symbols = symbols_result
                    symbol_count = len(all_symbols)

                projects_dict[pname] = {"id": pid, "symbol_count": symbol_count}

                # Index symbol definitions
                for symbol in all_symbols:
                    sym_name = symbol.get("name", "")
                    if sym_name:
                        if sym_name not in symbol_definitions:
                            symbol_definitions[sym_name] = set()
                        symbol_definitions[sym_name].add(pid)

            # Find edges: for each reference from project A to symbol in project B
            edges = {}  # (from_pid, to_pid) → {cross_refs: count, symbols: set}

            for (pid, _pname, db), symbols_result in zip(dbs, symbols_list, strict=False):
                if isinstance(symbols_result, Exception):
                    continue

                # Get all references from this project
                all_symbols = symbols_result
                for symbol in all_symbols:
                    refs = await asyncio.to_thread(
                        db.get_refs, symbol_id=symbol["id"], kind="all"
                    )
                    for ref in refs:
                        to_symbol_name = ref.get("to_symbol_name", "")
                        # Check if this symbol is defined in another project
                        if to_symbol_name in symbol_definitions:
                            for target_pid in symbol_definitions[to_symbol_name]:
                                if target_pid != pid:  # Cross-project reference
                                    edge_key = (pid, target_pid)
                                    if edge_key not in edges:
                                        edges[edge_key] = {"cross_refs": 0, "symbols": set()}
                                    edges[edge_key]["cross_refs"] += 1
                                    edges[edge_key]["symbols"].add(to_symbol_name)

            # Convert edges to list format, mapping project IDs to names
            pid_to_name = {pid: pname for pid, pname, _ in dbs}
            edges_list = [
                {
                    "from": pid_to_name.get(from_pid, f"project-{from_pid}"),
                    "to": pid_to_name.get(to_pid, f"project-{to_pid}"),
                    "cross_refs": v["cross_refs"],
                    "symbols": sorted(list(v["symbols"])),
                }
                for (from_pid, to_pid), v in edges.items()
            ]

            result_dict = {
                "collection_id": collection_id,
                "projects": projects_dict,
                "edges": edges_list,
            }

            _log_audit("collection_map", len(edges_list), agent_id=agent_id)
            return json.dumps(result_dict, indent=2)
        except Exception as e:
            logger.exception("collection_map error")
            _log_audit("collection_map", 0, agent_id=agent_id)
            return f"Error: {str(e)}"
