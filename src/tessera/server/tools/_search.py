"""Search tools: hybrid search and document search."""

import asyncio
import logging

from fastmcp import FastMCP

from ...embeddings import EmbeddingUnavailableError
from ...search import (
    SearchType,
    doc_search,
    enrich_with_docid,
    extract_snippet,
    format_results,
    hybrid_search,
)
from .._state import (
    _check_session,
    _get_project_dbs,
    _log_audit,
)

logger = logging.getLogger("tessera.server")


def register_search_tools(mcp: FastMCP) -> None:
    """Register search-related tools."""

    @mcp.tool()
    async def search(query: str, limit: int = 10, filter_language: str = "", source_type: str = "", output_format: str = "json", search_mode: str = "", advanced_fts: bool = False, weights: str = "", session_id: str = "") -> str:
        """Hybrid semantic + keyword search across indexed codebase.

        Combines FTS5 keyword matching, FAISS vector similarity, and PageRank
        graph ranking via weighted Reciprocal Rank Fusion (RRF). Returns
        results with file paths, line numbers, relevance scores, and focused
        code snippets.

        **When to use which mode:**
        - Default (no search_mode): Best for most queries. Merges keyword + semantic signals.
        - "lex": Exact identifier lookup. Use for function names, class names, variable names.
          Example: search("lex:hybrid_search") — fast, precise, FTS5-only.
        - "vec": Conceptual/semantic search. Use for "how does X work" questions.
          Example: search("vec:error handling strategy") — finds semantically similar code.
        - "hyde": Hypothetical document embedding. Best for abstract concepts where you want
          to find code that *describes* the concept, not code that *uses* the terms.
          Example: search("hyde:retry with exponential backoff")

        **FTS5 operators** (requires advanced_fts=True):
        - Phrases: search('"def hybrid_search"', advanced_fts=True) — exact phrase match
        - Negation: search("error NOT warning", advanced_fts=True) — exclude terms
        - Prefix: search("hybrid*", advanced_fts=True) — prefix matching
        - NEAR: search("NEAR(search query, 5)", advanced_fts=True) — proximity

        **Output formats:**
        - "json": Full metadata (default). Best for programmatic processing.
        - "markdown": Formatted with code blocks. Best for displaying to users.
        - "csv": Tabular. Best for analysis or spreadsheet export.
        - "files": Deduplicated file paths only. Best for knowing which files to read.

        Args:
            query: Search query. Supports prefix syntax: "lex:query", "vec:query", "hyde:query".
            limit: Max results (default 10).
            filter_language: Filter by programming language (e.g., "python", "typescript").
            source_type: Filter by content type — "code" or "markdown".
            output_format: Result format — "json" (default), "csv", "markdown", or "files".
            search_mode: Override search list routing — "lex", "vec", "hyde", or "lex,vec".
            advanced_fts: Enable FTS5 operators (phrases, NOT, *, NEAR). Default False.
            weights: Custom RRF fusion weights. Format: "keyword=1.5,semantic=1.0,graph=0.8".
                Omit to use defaults. Increase a weight to boost that signal; decrease to dampen it.
                Example: "keyword=3.0" for identifier-heavy searches, "semantic=2.0" for conceptual.
        """
        # Import state at call time to get current values (globals are mutable)
        from .._state import _embedding_client, _project_graphs

        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"

        dbs = _get_project_dbs(scope)
        if not dbs:
            _log_audit("search", 0, agent_id=agent_id)
            return "Error: No accessible projects"

        source_type_filter = [source_type] if source_type else None

        # Parse RRF weights: "keyword=2.0,semantic=0.5" or "" for defaults
        rrf_weights = None
        if weights:
            try:
                rrf_weights = {}
                for pair in weights.split(","):
                    key, val = pair.strip().split("=")
                    rrf_weights[key.strip()] = float(val.strip())
            except (ValueError, TypeError):
                return f"Error: Invalid weights format '{weights}'. Use 'keyword=1.5,semantic=1.0,graph=0.8'."

        # Resolve search types: explicit param > inline prefix > default
        search_types = None
        if search_mode:
            try:
                search_types = [SearchType(t.strip()) for t in search_mode.split(",")]
            except ValueError:
                return f"Error: Invalid search_mode '{search_mode}'. Use lex, vec, hyde, or comma-separated."
        # If search_types is still None, hybrid_search will parse prefix from query

        try:
            # Determine if we need embeddings based on search types
            need_embedding = True
            if search_types and all(t == SearchType.LEX for t in search_types):
                need_embedding = False

            # Determine embedding method: embed_query (VEC) vs embed_single (HYDE)
            use_hyde = search_types and SearchType.HYDE in search_types

            # Embed query if embedding client is available and needed
            query_embedding = None
            if need_embedding and _embedding_client:
                try:
                    import numpy as np
                    if use_hyde:
                        raw = await asyncio.to_thread(_embedding_client.embed_single, query)
                    else:
                        raw = await asyncio.to_thread(_embedding_client.embed_query, query)
                    query_embedding = np.array(raw, dtype=np.float32)
                except EmbeddingUnavailableError:
                    logger.debug("Embedding endpoint unavailable, falling back to keyword-only search")

            # Parallel query across all accessible projects
            ppr_used = False
            for pid, _pname, _db in dbs:
                g = _project_graphs.get(pid)
                if g:
                    ppr_used = True
                    logger.debug("Search on project %d using graph version %.1f", pid, g.loaded_at)

            tasks = [
                asyncio.to_thread(
                    hybrid_search, query, query_embedding, db,
                    _project_graphs.get(pid), limit, source_type_filter,
                    search_types, advanced_fts, rrf_weights,
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

            # Sort by score descending, cap at limit
            all_results.sort(key=lambda r: r.get("score", 0), reverse=True)
            all_results = all_results[:limit]

            # Add stable document IDs
            all_results = enrich_with_docid(all_results)

            # Extract best-matching snippets
            for r in all_results:
                if r.get("content"):
                    snippet_info = extract_snippet(r["content"], query)
                    r.update(snippet_info)

            _log_audit("search", len(all_results), agent_id=agent_id, ppr_used=ppr_used)
            return format_results(all_results, format=output_format)
        except Exception as e:
            logger.exception("Search tool error")
            _log_audit("search", 0, agent_id=agent_id)
            return f"Error during search: {str(e)}"

    @mcp.tool()
    async def doc_search_tool(query: str, limit: int = 10, formats: str = "", source_type: str = "", output_format: str = "json", session_id: str = "") -> str:
        """Search non-code documents only — markdown, PDF, YAML, JSON, config files, and assets.

        Use this instead of `search` when you specifically need documentation, specs,
        config files, or other non-code content. Excludes source code files from results.

        **Supported document types:** markdown, pdf, yaml, json, html, xml, text, txt,
        rst, csv, tsv, log, ini, cfg, toml, conf.

        **When to use:**
        - Finding documentation: doc_search("authentication flow")
        - Searching config files: doc_search("database connection", formats="yaml,toml")
        - Finding specs: doc_search("API design", formats="markdown")
        - Searching logs: doc_search("error timeout", formats="log")

        Args:
            query: Search query (same syntax as search tool).
            limit: Max results (default 10).
            formats: Comma-separated document types to search (e.g., "markdown,yaml").
                Leave empty to search all document types.
            source_type: Filter by a single source type (e.g., "markdown").
            output_format: Result format — "json" (default), "csv", "markdown", or "files".
        """
        from .._state import _embedding_client

        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"

        dbs = _get_project_dbs(scope)
        if not dbs:
            _log_audit("doc_search", 0, agent_id=agent_id)
            return "Error: No accessible projects"

        try:
            format_list = [f.strip() for f in formats.split(",") if f.strip()] if formats else None
            source_type_filter = [source_type] if source_type else None

            # Embed query if embedding client is available
            query_embedding = None
            if _embedding_client:
                try:
                    import numpy as np
                    raw = await asyncio.to_thread(_embedding_client.embed_query, query)
                    query_embedding = np.array(raw, dtype=np.float32)
                except EmbeddingUnavailableError:
                    logger.debug("Embedding endpoint unavailable, falling back to keyword-only doc_search")

            # Parallel query across projects
            # source_type overrides format_list when provided (e.g., source_type="asset")
            effective_formats = source_type_filter if source_type_filter else format_list
            tasks = [
                asyncio.to_thread(
                    doc_search, query, query_embedding, db, graph=None, limit=limit, formats=effective_formats
                )
                for pid, pname, db in dbs
            ]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            all_results = []
            for (pid, pname, _db), result in zip(dbs, results_list, strict=False):
                if isinstance(result, Exception):
                    logger.warning("doc_search on project %d failed: %s", pid, result)
                    continue
                for r in result:
                    r["project_id"] = pid
                    r["project_name"] = pname
                all_results.extend(result)

            all_results.sort(key=lambda r: r.get("score", 0), reverse=True)
            all_results = all_results[:limit]

            # Add stable document IDs
            all_results = enrich_with_docid(all_results)

            # Extract best-matching snippets
            for r in all_results:
                if r.get("content"):
                    snippet_info = extract_snippet(r["content"], query)
                    r.update(snippet_info)

            _log_audit("doc_search", len(all_results), agent_id=agent_id)
            return format_results(all_results, format=output_format)
        except Exception as e:
            logger.exception("doc_search error")
            _log_audit("doc_search", 0, agent_id=agent_id)
            return f"Error: {str(e)}"
