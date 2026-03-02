"""Symbol tools: symbols, references, file_context."""

import asyncio
import json
import logging

from fastmcp import FastMCP

from .._state import _check_session, _get_project_dbs, _log_audit

logger = logging.getLogger("tessera.server")


def register_symbol_tools(mcp: FastMCP) -> None:
    """Register symbol-related tools."""

    @mcp.tool()
    async def symbols(query: str = "*", kind: str = "", language: str = "", session_id: str = "") -> str:
        """Find functions, classes, methods, and imports across the indexed codebase.

        Returns symbol definitions with file paths, line numbers, signatures, and scope.
        Use this to locate where something is defined before reading the full file.

        **Common patterns:**
        - Find a specific function: symbols("hybrid_search")
        - List all classes: symbols("*", kind="class")
        - Find Python functions: symbols("*", kind="function", language="python")
        - Find by prefix: symbols("test_*")

        **Symbol kinds:** function, class, method, import, variable, interface, type_alias.

        Args:
            query: Symbol name pattern. Use "*" for all, or a name/prefix to filter.
            kind: Filter by symbol kind (e.g., "function", "class", "method", "import").
            language: Filter by language (e.g., "python", "typescript", "php").
        """
        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"

        dbs = _get_project_dbs(scope)
        if not dbs:
            _log_audit("symbols", 0, agent_id=agent_id)
            return "Error: No accessible projects"

        try:
            # Parallel query pattern
            tasks = [
                asyncio.to_thread(
                    db.lookup_symbols, query,
                    kind or None, language or None
                )
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
                all_results.extend(result)

            _log_audit("symbols", len(all_results), agent_id=agent_id)
            return json.dumps(all_results, indent=2)
        except Exception as e:
            logger.exception("Symbols tool error")
            _log_audit("symbols", 0, agent_id=agent_id)
            return f"Error querying symbols: {str(e)}"

    @mcp.tool()
    async def references(symbol_name: str, kind: str = "all", session_id: str = "") -> str:
        """Find all references to and from a specific symbol.

        Returns two lists: "outgoing" (what this symbol calls/uses) and "callers"
        (what calls/uses this symbol). Essential for understanding dependency chains
        and call graphs before making changes.

        **Common patterns:**
        - Who calls this function: references("handle_request") → check callers list
        - What does this function depend on: references("handle_request") → check outgoing list
        - Find all usages of a class: references("ProjectDB")
        - Filter by reference type: references("MyClass", kind="import")

        **Reference kinds:** call, import, inherit, type_ref, attribute, or "all" (default).

        Args:
            symbol_name: Exact symbol name to look up.
            kind: Filter reference kind — "all" (default), "call", "import", "inherit", etc.
        """
        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"

        dbs = _get_project_dbs(scope)
        if not dbs:
            _log_audit("references", 0, agent_id=agent_id)
            return "Error: No accessible projects"

        try:
            # Parallel query pattern: fetch both outgoing and callers for each DB in parallel
            async def _query_references(db, symbol_name, kind):
                """Query both references and callers for a database."""
                outgoing = await asyncio.to_thread(db.get_refs, symbol_name=symbol_name, kind=kind)
                callers = await asyncio.to_thread(db.get_callers, symbol_name=symbol_name, kind=kind)
                return (outgoing, callers)

            tasks = [
                _query_references(db, symbol_name, kind)
                for pid, pname, db in dbs
            ]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            all_outgoing = []
            all_callers = []
            for (pid, pname, _db), result in zip(dbs, results_list, strict=False):
                if isinstance(result, Exception):
                    logger.warning("Query on project %d failed: %s", pid, result)
                    continue
                outgoing, callers = result
                for r in outgoing:
                    all_outgoing.append({
                        "to_symbol": r.get("to_symbol_name", ""),
                        "kind": r.get("kind", ""),
                        "line": r.get("line", 0),
                        "project_id": pid,
                        "project_name": pname,
                    })
                for c in callers:
                    all_callers.append({
                        "name": c.get("name", ""),
                        "kind": c.get("kind", ""),
                        "file_id": c.get("file_id"),
                        "line": c.get("line", 0),
                        "scope": c.get("scope", ""),
                        "project_id": pid,
                        "project_name": pname,
                    })

            results = {"outgoing": all_outgoing, "callers": all_callers}
            total = len(all_outgoing) + len(all_callers)
            _log_audit("references", total, agent_id=agent_id)
            return json.dumps(results, indent=2)
        except Exception as e:
            logger.exception("References tool error")
            _log_audit("references", 0, agent_id=agent_id)
            return f"Error finding references: {str(e)}"

    @mcp.tool()
    async def file_context(file_path: str, session_id: str = "") -> str:
        """Get the structural skeleton of a file — all symbols, imports, and internal references.

        Returns the file's complete symbol table without reading the full source code.
        Use this to understand a file's structure before deciding which parts to read,
        or to map dependencies between symbols within a file.

        **Common patterns:**
        - Understand a module's API: file_context("src/tessera/search.py")
        - Find all classes in a file: file_context("src/models.py") → filter by kind="class"
        - Map internal call graph: check references list for intra-file calls

        **Returns:** JSON with {file: {path, language, lines}, symbols: [{name, kind, line,
        signature, scope}], references: [{from_symbol_id, to_symbol_id, kind, line}]}.

        Args:
            file_path: Relative path from project root (e.g., "src/tessera/search.py").
        """
        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"

        if ".." in file_path or file_path.startswith("/"):
            _log_audit("file_context", 0, agent_id=agent_id)
            return "Error: Invalid file path (path traversal detected)"

        dbs = _get_project_dbs(scope)
        if not dbs:
            _log_audit("file_context", 0, agent_id=agent_id)
            return "Error: No accessible projects"

        try:
            # Try each project until file is found
            for pid, pname, db in dbs:
                file_info = await asyncio.to_thread(db.get_file, path=file_path)
                if not file_info:
                    continue

                file_id = file_info["id"]
                file_info_dict = dict(file_info)
                file_info_dict["project_id"] = pid
                file_info_dict["project_name"] = pname

                all_symbols = await asyncio.to_thread(
                    lambda _db=db, _fid=file_id: [dict(r) for r in _db.conn.execute(
                        "SELECT id, name, kind, line, col, scope, signature FROM symbols WHERE file_id = ? ORDER BY line",
                        (_fid,)
                    ).fetchall()]
                )

                symbol_ids = [s["id"] for s in all_symbols]
                file_refs = []
                if symbol_ids:
                    placeholders = ",".join("?" * len(symbol_ids))
                    file_refs = await asyncio.to_thread(
                        lambda _db=db, _ph=placeholders, _sids=symbol_ids: [dict(r) for r in _db.conn.execute(
                            f"SELECT from_symbol_id, to_symbol_id, kind, line FROM refs WHERE from_symbol_id IN ({_ph}) ORDER BY line",
                            _sids
                        ).fetchall()]
                    )

                result = {
                    "file": file_info_dict,
                    "symbols": all_symbols,
                    "references": file_refs,
                }
                _log_audit("file_context", 1, agent_id=agent_id)
                return json.dumps(result, indent=2)

            _log_audit("file_context", 0, agent_id=agent_id)
            return json.dumps(None, indent=2)
        except Exception as e:
            logger.exception("File context tool error")
            _log_audit("file_context", 0, agent_id=agent_id)
            return f"Error retrieving file context: {str(e)}"
