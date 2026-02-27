"""Real-world federation tests: index the Popup Maker plugin ecosystem and validate
cross-project queries, collection management, scope enforcement, and latency.

Requires PM plugins at Local Sites path. Skips gracefully if not found.
No embedding endpoint needed — structural indexing only.
"""

import os
import json
import time
import asyncio
import statistics
import tempfile

import pytest

from tessera.db import ProjectDB, GlobalDB
from tessera.indexer import IndexerPipeline
from tessera.server import create_server
from tessera.auth import create_scope, validate_session


WP_PLUGINS = "/Users/danieliser/Local Sites/popup-maker/app/public/wp-content/plugins"

PM_PLUGINS = [
    ("popup-maker", ["php", "typescript", "javascript"], 50),
    ("popup-maker-pro", ["php", "typescript", "javascript"], 50),
    ("popup-maker-ecommerce-popups", ["php", "typescript"], 10),
    ("popup-maker-lms-popups", ["php", "typescript"], 10),
]


def _plugins_available():
    return all(
        os.path.isdir(os.path.join(WP_PLUGINS, name))
        for name, _, _ in PM_PLUGINS
    )


pytestmark = pytest.mark.skipif(
    not _plugins_available(),
    reason=f"PM plugins not found at {WP_PLUGINS}",
)


async def get_json(server, tool_name, args=None):
    """Call a tool and parse the JSON result."""
    content, _ = await server.call_tool(tool_name, args or {})
    text = content[0].text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pytest.fail(f"Tool {tool_name} returned non-JSON: {text[:200]}")


async def get_text(server, tool_name, args=None):
    """Call a tool and return raw text."""
    content, _ = await server.call_tool(tool_name, args or {})
    return content[0].text


# Session-scoped temp dir for ProjectDB.base_dir
_session_base_dir = None


@pytest.fixture(scope="session")
def pm_ecosystem(tmp_path_factory):
    """Index all 4 PM plugins into a shared GlobalDB, create collection, return server.

    Session-scoped to avoid re-indexing on every test (~minutes for 1700+ files).
    Sets ProjectDB.base_dir to a session-scoped temp dir so indexer and server
    resolve to the same DB files.
    """
    global _session_base_dir

    base = tmp_path_factory.mktemp("pm_federation")
    data_dir = str(base / "tessera-data")
    _session_base_dir = data_dir

    # Set base_dir so ProjectDB(plugin_path) resolves to our temp dir
    old_base_dir = ProjectDB.base_dir
    ProjectDB.base_dir = data_dir

    global_db_path = str(base / "global.db")
    global_db = GlobalDB(global_db_path)

    project_ids = {}
    project_dbs = {}
    stats_map = {}

    for name, languages, min_files in PM_PLUGINS:
        plugin_path = os.path.join(WP_PLUGINS, name)

        # Use ProjectDB(plugin_path) so it goes through _get_data_dir
        # with our base_dir — same path the server will resolve to
        project_db = ProjectDB(plugin_path)

        pipeline = IndexerPipeline(
            project_path=plugin_path,
            project_db=project_db,
            global_db=global_db,
            embedding_client=None,
            languages=languages,
        )
        pid = pipeline.register(name)
        stats = pipeline.index_project()

        project_ids[name] = pid
        project_dbs[name] = project_db
        stats_map[name] = stats

    # Create collection with all 4 plugins
    coll_id = global_db.create_collection(
        "pm-ecosystem",
        list(project_ids.values()),
    )

    # Create multi-project server (will resolve DBs via same base_dir)
    server = create_server(None, global_db_path)

    ProjectDB.base_dir = old_base_dir

    return {
        "server": server,
        "global_db": global_db,
        "global_db_path": global_db_path,
        "project_ids": project_ids,
        "project_dbs": project_dbs,
        "stats": stats_map,
        "collection_id": coll_id,
        "base_dir": data_dir,
    }


@pytest.fixture(autouse=True)
def _restore_base_dir_for_federation(pm_ecosystem):
    """Ensure ProjectDB.base_dir points to session fixture's data dir during tests.

    The conftest autouse fixture resets base_dir per function; we override it here
    so the server can resolve ProjectDBs to the session-indexed data.
    """
    old = ProjectDB.base_dir
    ProjectDB.base_dir = pm_ecosystem["base_dir"]
    yield
    ProjectDB.base_dir = old


# ---------------------------------------------------------------------------
# TestRealIndexing — sanity checks that indexing produced data
# ---------------------------------------------------------------------------


class TestRealIndexing:
    """Verify indexing worked for each plugin."""

    @pytest.fixture(params=[p[0] for p in PM_PLUGINS])
    def plugin_stats(self, request, pm_ecosystem):
        name = request.param
        return name, pm_ecosystem["stats"][name], pm_ecosystem["project_dbs"][name], pm_ecosystem["project_ids"][name]

    def test_files_indexed(self, plugin_stats):
        name, stats, db, pid = plugin_stats
        total = stats.files_processed + stats.files_skipped
        assert total > 0, f"{name}: no files discovered"

    def test_no_crashes(self, plugin_stats):
        name, stats, db, pid = plugin_stats
        assert stats.files_failed == 0, f"{name}: {stats.files_failed} files failed indexing"

    def test_symbols_extracted(self, plugin_stats):
        name, stats, db, pid = plugin_stats
        assert stats.symbols_extracted > 0, f"{name}: no symbols extracted"

    def test_refs_extracted(self, plugin_stats):
        name, stats, db, pid = plugin_stats
        count = db.conn.execute(
            "SELECT COUNT(*) FROM refs WHERE project_id = ?", (pid,)
        ).fetchone()[0]
        assert count > 0, f"{name}: no references extracted"

    def test_known_hooks_in_core(self, pm_ecosystem):
        """Core plugin should contain known WP hook symbols."""
        core_db = pm_ecosystem["project_dbs"]["popup-maker"]
        core_pid = pm_ecosystem["project_ids"]["popup-maker"]

        # Check for at least some pum_ symbols
        pum_symbols = core_db.lookup_symbols(query="pum_*", project_id=core_pid)
        assert len(pum_symbols) > 0, "No pum_* symbols found in popup-maker core"

    def test_print_indexing_stats(self, pm_ecosystem, capsys):
        """Print stats for manual review."""
        print(f"\n{'='*70}")
        print("  Popup Maker Ecosystem — Indexing Stats")
        print(f"{'='*70}")
        for name, _, _ in PM_PLUGINS:
            s = pm_ecosystem["stats"][name]
            db = pm_ecosystem["project_dbs"][name]
            pid = pm_ecosystem["project_ids"][name]

            sym_count = db.conn.execute(
                "SELECT COUNT(*) FROM symbols WHERE project_id = ?", (pid,)
            ).fetchone()[0]
            ref_count = db.conn.execute(
                "SELECT COUNT(*) FROM refs WHERE project_id = ?", (pid,)
            ).fetchone()[0]
            edge_count = db.conn.execute(
                "SELECT COUNT(*) FROM edges WHERE project_id = ?", (pid,)
            ).fetchone()[0]

            print(f"\n  {name}")
            print(f"    Files: {s.files_processed} indexed, {s.files_skipped} skipped, {s.files_failed} failed")
            print(f"    Symbols: {sym_count}  Refs: {ref_count}  Edges: {edge_count}")
            print(f"    Chunks: {s.chunks_created}  Time: {s.time_elapsed:.1f}s")
        print(f"{'='*70}")


# ---------------------------------------------------------------------------
# TestRealFederationQueries — all 5 query tools across the collection
# ---------------------------------------------------------------------------


class TestRealFederationQueries:
    """Exercise query tools in multi-project mode with real data."""

    async def test_search_returns_multi_project(self, pm_ecosystem):
        """search() should return results from multiple projects."""
        server = pm_ecosystem["server"]
        result = await get_json(server, "search", {
            "query": "popup",
            "limit": 20,
            "session_id": "",
        })
        assert len(result) > 0, "search('popup') returned no results"

        project_names = {r.get("project_name") for r in result}
        assert len(project_names) >= 2, (
            f"Expected results from 2+ projects, got: {project_names}"
        )

    async def test_symbols_returns_multi_project(self, pm_ecosystem):
        """symbols() should return pum_ symbols from multiple projects."""
        server = pm_ecosystem["server"]
        result = await get_json(server, "symbols", {
            "query": "pum_*",
            "session_id": "",
        })
        assert len(result) > 0, "symbols('pum_*') returned no results"

        project_names = {r.get("project_name") for r in result}
        # Core at minimum; Pro likely also has pum_ symbols
        assert "popup-maker" in project_names, "Core plugin missing from pum_* symbols"

    async def test_references_returns_data(self, pm_ecosystem):
        """references() for a known symbol returns data."""
        server = pm_ecosystem["server"]
        # Find a symbol that exists in core
        core_db = pm_ecosystem["project_dbs"]["popup-maker"]
        core_pid = pm_ecosystem["project_ids"]["popup-maker"]
        some_sym = core_db.lookup_symbols(query="*", kind="function", project_id=core_pid)
        if not some_sym:
            pytest.skip("No function symbols in core")

        sym_name = some_sym[0]["name"]
        result = await get_json(server, "references", {
            "symbol_name": sym_name,
            "session_id": "",
        })
        # references() returns a dict with callers/outgoing keys
        assert isinstance(result, dict)
        assert "callers" in result or "outgoing" in result

    async def test_impact_returns_data(self, pm_ecosystem):
        """impact() for a symbol returns without error."""
        server = pm_ecosystem["server"]
        core_db = pm_ecosystem["project_dbs"]["popup-maker"]
        core_pid = pm_ecosystem["project_ids"]["popup-maker"]
        some_sym = core_db.lookup_symbols(query="*", kind="function", project_id=core_pid)
        if not some_sym:
            pytest.skip("No function symbols in core")

        sym_name = some_sym[0]["name"]
        result = await get_json(server, "impact", {
            "symbol_name": sym_name,
            "session_id": "",
        })
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# TestRealCrossRefs — the payoff: cross-plugin dependency detection
# ---------------------------------------------------------------------------


class TestRealCrossRefs:
    """Validate cross_refs finds real inter-plugin dependencies."""

    async def test_cross_refs_for_common_symbols(self, pm_ecosystem):
        """Find symbols defined in core that are referenced in other plugins."""
        server = pm_ecosystem["server"]
        core_db = pm_ecosystem["project_dbs"]["popup-maker"]
        core_pid = pm_ecosystem["project_ids"]["popup-maker"]

        # Get function symbols from core to try
        core_functions = core_db.lookup_symbols(
            query="*", kind="function", project_id=core_pid
        )
        if not core_functions:
            pytest.skip("No function symbols in core")

        # Try the first 20 core symbols to find at least one cross-ref
        found_cross_ref = False
        cross_ref_details = []

        for sym in core_functions[:20]:
            result = await get_json(server, "cross_refs", {
                "symbol_name": sym["name"],
                "session_id": "",
            })
            if result.get("cross_refs"):
                found_cross_ref = True
                cross_ref_details.append({
                    "symbol": sym["name"],
                    "count": len(result["cross_refs"]),
                    "from_projects": list({r["from_project_name"] for r in result["cross_refs"]}),
                })
                if len(cross_ref_details) >= 3:
                    break

        # Print what we found for manual review
        if cross_ref_details:
            print(f"\n  Cross-refs found:")
            for d in cross_ref_details:
                print(f"    {d['symbol']}: {d['count']} refs from {d['from_projects']}")

        assert found_cross_ref, (
            "No cross-project references found among first 20 core function symbols. "
            "This may indicate cross-project reference resolution isn't working."
        )

    async def test_cross_ref_structure(self, pm_ecosystem):
        """Cross-ref results have required fields and correct project separation."""
        server = pm_ecosystem["server"]
        core_db = pm_ecosystem["project_dbs"]["popup-maker"]
        core_pid = pm_ecosystem["project_ids"]["popup-maker"]

        # Find any symbol with cross-refs
        core_functions = core_db.lookup_symbols(
            query="*", kind="function", project_id=core_pid
        )
        for sym in core_functions[:30]:
            result = await get_json(server, "cross_refs", {
                "symbol_name": sym["name"],
                "session_id": "",
            })
            if result.get("cross_refs"):
                for ref in result["cross_refs"]:
                    assert "from_project_id" in ref
                    assert "from_project_name" in ref
                    assert "to_project_id" in ref
                    assert "to_project_name" in ref
                    assert ref["from_project_id"] != ref["to_project_id"], (
                        f"Cross-ref should be between different projects: {ref}"
                    )
                return  # One valid cross-ref is enough

        pytest.skip("No cross-refs found to validate structure")


# ---------------------------------------------------------------------------
# TestRealCollectionMap — inter-plugin dependency graph
# ---------------------------------------------------------------------------


class TestRealCollectionMap:
    """Validate collection_map builds correct dependency graph."""

    async def test_collection_map_has_all_projects(self, pm_ecosystem):
        """collection_map returns all 4 projects."""
        server = pm_ecosystem["server"]
        result = await get_json(server, "collection_map", {
            "collection_id": pm_ecosystem["collection_id"],
            "session_id": "",
        })

        assert "projects" in result
        for name, _, _ in PM_PLUGINS:
            assert name in result["projects"], f"{name} missing from collection_map"

    async def test_collection_map_has_edges(self, pm_ecosystem):
        """collection_map should show dependency edges between plugins."""
        server = pm_ecosystem["server"]
        result = await get_json(server, "collection_map", {
            "collection_id": pm_ecosystem["collection_id"],
            "session_id": "",
        })

        edges = result.get("edges", [])
        # Print for manual review
        print(f"\n  Collection map: {len(edges)} edges")
        for e in edges[:10]:
            syms = e.get("symbols", [])
            sym_preview = ", ".join(syms[:3])
            if len(syms) > 3:
                sym_preview += "..."
            print(f"    {e['from']} -> {e['to']}: {e.get('cross_refs', 0)} refs ({sym_preview})")

        # We expect at least some edges — Pro/Ecommerce/LMS depend on core
        assert len(edges) > 0, "No dependency edges found between PM plugins"

    async def test_collection_map_edge_structure(self, pm_ecosystem):
        """Each edge has required fields."""
        server = pm_ecosystem["server"]
        result = await get_json(server, "collection_map", {
            "collection_id": pm_ecosystem["collection_id"],
            "session_id": "",
        })

        for edge in result.get("edges", []):
            assert "from" in edge
            assert "to" in edge
            assert "cross_refs" in edge
            assert "symbols" in edge
            assert isinstance(edge["symbols"], list)
            assert edge["from"] != edge["to"]


# ---------------------------------------------------------------------------
# TestRealScopeEnforcement — verify scope gating with real data
# ---------------------------------------------------------------------------


class TestRealScopeEnforcement:
    """Scope gating restricts which projects are visible."""

    async def test_subset_collection_restricts_search(self, pm_ecosystem):
        """A collection with 2 plugins should only return results from those 2."""
        global_db = pm_ecosystem["global_db"]
        pids = pm_ecosystem["project_ids"]

        # Create a second collection with only core + pro
        subset_coll_id = global_db.create_collection(
            "core-and-pro",
            [pids["popup-maker"], pids["popup-maker-pro"]],
        )

        # Create collection-scoped session
        session_id = create_scope(
            global_db.conn,
            agent_id="test-restricted",
            level="collection",
            projects=[],
            collections=[str(subset_coll_id)],
            ttl_minutes=30,
        )

        # Use the same server — scope is enforced per-request via session
        server = pm_ecosystem["server"]

        result = await get_json(server, "search", {
            "query": "popup",
            "limit": 50,
            "session_id": session_id,
        })

        project_names = {r.get("project_name") for r in result}
        allowed = {"popup-maker", "popup-maker-pro"}
        assert project_names.issubset(allowed), (
            f"Scope leak: got results from {project_names - allowed}"
        )

    async def test_subset_collection_restricts_symbols(self, pm_ecosystem):
        """Collection-scoped symbols() only returns symbols from allowed projects."""
        global_db = pm_ecosystem["global_db"]
        pids = pm_ecosystem["project_ids"]

        subset_coll_id = global_db.create_collection(
            "core-only-for-scope-test",
            [pids["popup-maker"]],
        )

        session_id = create_scope(
            global_db.conn,
            agent_id="test-core-only",
            level="collection",
            projects=[],
            collections=[str(subset_coll_id)],
            ttl_minutes=30,
        )

        server = pm_ecosystem["server"]

        result = await get_json(server, "symbols", {
            "query": "*",
            "session_id": session_id,
        })

        project_names = {r.get("project_name") for r in result}
        assert project_names.issubset({"popup-maker"}), (
            f"Scope leak: got results from {project_names}, expected only popup-maker"
        )


# ---------------------------------------------------------------------------
# TestLatencyBenchmarks — measure query performance
# ---------------------------------------------------------------------------


class TestLatencyBenchmarks:
    """Measure query latency across the 4-project collection."""

    async def _time_calls(self, server, tool_name, args, n=10):
        """Run a tool n times and return timing stats."""
        times = []
        for _ in range(n):
            start = time.perf_counter()
            await server.call_tool(tool_name, args)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
        return {
            "min": min(times),
            "p50": statistics.median(times),
            "p95": sorted(times)[int(len(times) * 0.95)],
            "max": max(times),
        }

    async def test_query_latency(self, pm_ecosystem, capsys):
        """Benchmark all query tools and print timing table."""
        server = pm_ecosystem["server"]

        benchmarks = {}

        benchmarks["search"] = await self._time_calls(
            server, "search", {"query": "popup", "limit": 10, "session_id": ""}, n=10
        )
        benchmarks["symbols"] = await self._time_calls(
            server, "symbols", {"query": "pum_*", "session_id": ""}, n=10
        )
        benchmarks["cross_refs"] = await self._time_calls(
            server, "cross_refs", {"symbol_name": "pum_popup_before_title", "session_id": ""}, n=5
        )
        benchmarks["collection_map"] = await self._time_calls(
            server, "collection_map", {"collection_id": pm_ecosystem["collection_id"], "session_id": ""}, n=5
        )

        # Print results
        print(f"\n{'='*70}")
        print("  Latency Benchmarks (4 projects, ~1700 files)")
        print(f"{'='*70}")
        print(f"  {'Tool':<20} {'Min':>8} {'P50':>8} {'P95':>8} {'Max':>8}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for tool, t in benchmarks.items():
            print(f"  {tool:<20} {t['min']:>7.1f}ms {t['p50']:>7.1f}ms {t['p95']:>7.1f}ms {t['max']:>7.1f}ms")
        print(f"{'='*70}")

        # Assert p95 under 500ms for point query tools (search, symbols, cross_refs).
        # collection_map is inherently O(N*M) across all project pairs — no hard limit.
        for tool in ("search", "symbols", "cross_refs"):
            t = benchmarks[tool]
            assert t["p95"] < 500, (
                f"{tool} p95 latency {t['p95']:.0f}ms exceeds 500ms target"
            )
