"""Structured validation tests against frozen fixture files.

Each test class indexes a known fixture file through the real pipeline
(parse → index → query) and asserts exact symbol counts, reference targets,
edge correctness, and query behavior. Every expected value is hand-verified
against actual parser output — failures here mean the extraction pipeline
has a real bug, not a test bug.

Known limitations documented inline:
- Parser uses '<module>' as from_symbol for file-scope code; these refs are
  dropped because '<module>' has no symbol ID. Affects TS arrow fns and PHP hooks.
- Parser attributes method-body refs to the class name, not the method name.
"""

import os
import shutil

import pytest

from tessera.indexer import IndexerPipeline


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _index_fixtures(tmpdir: str, fixture_files: list[str]) -> IndexerPipeline:
    """Copy fixture files into a temp project dir, index, and return the pipeline."""
    for fname in fixture_files:
        src = os.path.join(FIXTURES_DIR, fname)
        dst = os.path.join(tmpdir, fname)
        shutil.copy2(src, dst)

    pipeline = IndexerPipeline(tmpdir)
    pipeline.index_project()
    return pipeline


def _get_edges(db) -> list[dict]:
    """Query edges table joined with symbol names (not exposed as a method)."""
    cursor = db.conn.execute("""
        SELECT e.type, s1.name as from_name, s2.name as to_name, e.weight
        FROM edges e
        JOIN symbols s1 ON e.from_id = s1.id
        JOIN symbols s2 ON e.to_id = s2.id
        ORDER BY e.type, s1.name, s2.name
    """)
    return [dict(row) for row in cursor.fetchall()]


def _get_all_refs(db) -> list[dict]:
    """Query all refs in the DB (get_refs requires a filter)."""
    cursor = db.conn.execute("SELECT * FROM refs ORDER BY line")
    return [dict(row) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Python validation
# ---------------------------------------------------------------------------

class TestPythonValidation:
    """Validate Python extraction against tests/fixtures/python_sample.py.

    Expected (hand-verified from parser output):
      11 symbols (10 + 1 <module> pseudo-symbol), 7 refs, 9 edges
    """

    @pytest.fixture
    def db(self, tmp_path):
        pipeline = _index_fixtures(str(tmp_path), ["python_sample.py"])
        return pipeline.project_db

    # --- Symbol extraction ---

    def test_symbol_count(self, db):
        syms = db.lookup_symbols("*")
        assert len(syms) == 11

    def test_import_symbols(self, db):
        imports = db.lookup_symbols("*", kind="import")
        assert len(imports) == 2

    def test_class_symbols(self, db):
        classes = db.lookup_symbols("*", kind="class")
        names = {s["name"] for s in classes}
        assert names == {"Event", "ClickEvent"}

    def test_method_symbols_with_scope(self, db):
        methods = db.lookup_symbols("*", kind="method")
        scoped = {(s["name"], s["scope"]) for s in methods}
        assert ("__init__", "Event") in scoped
        assert ("serialize", "Event") in scoped
        assert ("__init__", "ClickEvent") in scoped
        assert ("distance_from_origin", "ClickEvent") in scoped
        assert len(methods) == 4

    def test_function_symbols(self, db):
        funcs = db.lookup_symbols("*", kind="function")
        names = {s["name"] for s in funcs}
        assert names == {"process_event", "create_click"}

    def test_async_function_extracted(self, db):
        """process_event is async — must not be skipped."""
        results = db.lookup_symbols("process_event")
        assert len(results) == 1
        assert results[0]["kind"] == "function"

    # --- Reference extraction ---

    def test_extends_reference(self, db):
        refs = _get_all_refs(db)
        extends = [r for r in refs if r["kind"] == "extends"]
        assert len(extends) == 1
        assert extends[0]["to_symbol_name"] == "Event"

    def test_function_call_references(self, db):
        """create_click calls ClickEvent and distance_from_origin."""
        create_click = db.lookup_symbols("create_click")[0]
        refs = db.get_refs(symbol_id=create_click["id"])
        targets = {r["to_symbol_name"] for r in refs if r["kind"] == "calls"}
        assert "ClickEvent" in targets
        assert "distance_from_origin" in targets

    def test_method_call_references(self, db):
        """process_event calls serialize."""
        pe = db.lookup_symbols("process_event")[0]
        refs = db.get_refs(symbol_id=pe["id"])
        targets = {r["to_symbol_name"] for r in refs if r["kind"] == "calls"}
        assert "serialize" in targets

    def test_super_call_reference(self, db):
        """ClickEvent.__init__ calls super().__init__."""
        inits = db.lookup_symbols("__init__", kind="method")
        click_init = [s for s in inits if s["scope"] == "ClickEvent"][0]
        refs = db.get_refs(symbol_id=click_init["id"])
        targets = {r["to_symbol_name"] for r in refs}
        assert "__init__" in targets or "super" in targets

    def test_total_ref_count(self, db):
        refs = _get_all_refs(db)
        assert len(refs) == 7

    # --- Edge correctness ---

    def test_contains_edges(self, db):
        edges = _get_edges(db)
        contains = [(e["from_name"], e["to_name"]) for e in edges if e["type"] == "contains"]
        assert ("Event", "__init__") in contains
        assert ("Event", "serialize") in contains
        assert ("ClickEvent", "__init__") in contains
        assert ("ClickEvent", "distance_from_origin") in contains
        assert len(contains) == 4

    def test_calls_edges(self, db):
        edges = _get_edges(db)
        calls = {(e["from_name"], e["to_name"]) for e in edges if e["type"] == "calls"}
        assert ("create_click", "ClickEvent") in calls
        assert ("create_click", "distance_from_origin") in calls
        assert ("process_event", "serialize") in calls
        assert ("__init__", "__init__") in calls  # super().__init__

    def test_extends_edge(self, db):
        edges = _get_edges(db)
        extends = [(e["from_name"], e["to_name"]) for e in edges if e["type"] == "extends"]
        assert ("ClickEvent", "Event") in extends
        assert len(extends) == 1

    def test_no_duplicate_edges(self, db):
        edges = _get_edges(db)
        tuples = [(e["from_name"], e["to_name"], e["type"]) for e in edges]
        assert len(tuples) == len(set(tuples)), f"Duplicate edges: {tuples}"

    def test_total_edge_count(self, db):
        edges = _get_edges(db)
        assert len(edges) == 9

    # --- Query correctness ---

    def test_lookup_exact_match(self, db):
        results = db.lookup_symbols("Event")
        assert len(results) == 1
        assert results[0]["kind"] == "class"

    def test_lookup_substring_fallback(self, db):
        """Looking up 'click' should find ClickEvent via substring fallback."""
        results = db.lookup_symbols("click")
        names = {s["name"] for s in results}
        assert "ClickEvent" in names or "create_click" in names

    def test_lookup_wildcard(self, db):
        results = db.lookup_symbols("*Event*")
        names = {s["name"] for s in results}
        assert "Event" in names
        assert "ClickEvent" in names

    def test_lookup_by_kind_filter(self, db):
        results = db.lookup_symbols("*", kind="class")
        assert all(s["kind"] == "class" for s in results)

    def test_get_callers(self, db):
        """ClickEvent should be called by create_click."""
        ce = db.lookup_symbols("ClickEvent")[0]
        callers = db.get_callers(ce["id"])
        caller_names = {c["name"] for c in callers}
        assert "create_click" in caller_names

    def test_get_forward_refs(self, db):
        """create_click at depth 1 should reach ClickEvent and distance_from_origin."""
        cc = db.lookup_symbols("create_click")[0]
        fwd = db.get_forward_refs(cc["id"], depth=1)
        names = {s["name"] for s in fwd}
        assert "ClickEvent" in names
        assert "distance_from_origin" in names

    def test_impact_depth_1(self, db):
        """Impact on Event at depth 1: contains __init__ and serialize."""
        event = db.lookup_symbols("Event")[0]
        fwd = db.get_forward_refs(event["id"], depth=1)
        names = {s["name"] for s in fwd}
        assert "__init__" in names
        assert "serialize" in names

    def test_impact_depth_2(self, db):
        """Impact on Event at depth 2: forward refs follow outgoing edges.
        extends goes FROM ClickEvent TO Event, so it's inbound, not outbound.
        At depth 2 we traverse __init__→__init__ (super call)."""
        event = db.lookup_symbols("Event")[0]
        fwd = db.get_forward_refs(event["id"], depth=2)
        names = {s["name"] for s in fwd}
        assert "__init__" in names
        assert "serialize" in names

    def test_keyword_search(self, db):
        results = db.keyword_search("event")
        assert len(results) > 0


# ---------------------------------------------------------------------------
# TypeScript validation
# ---------------------------------------------------------------------------

class TestTypeScriptValidation:
    """Validate TypeScript extraction against tests/fixtures/typescript_sample.ts.

    Note: Tessera uses tree-sitter JS grammar for TypeScript, so the fixture
    is written as valid JavaScript (no interfaces, no type annotations).

    Expected (hand-verified from parser output):
      11 symbols (10 + 1 <module> pseudo-symbol), 9 refs stored (was 8, 3 from '<module>')
      11 edges stored (was 10, 1 from '<module>')
    """

    @pytest.fixture
    def db(self, tmp_path):
        pipeline = _index_fixtures(str(tmp_path), ["typescript_sample.ts"])
        return pipeline.project_db

    # --- Symbol extraction ---

    def test_symbol_count(self, db):
        syms = db.lookup_symbols("*")
        assert len(syms) == 11

    def test_import_symbols(self, db):
        imports = db.lookup_symbols("*", kind="import")
        assert len(imports) == 1

    def test_class_symbols(self, db):
        classes = db.lookup_symbols("*", kind="class")
        names = {s["name"] for s in classes}
        assert names == {"BaseModel", "User"}

    def test_method_symbols_with_scope(self, db):
        methods = db.lookup_symbols("*", kind="method")
        scoped = {(s["name"], s["scope"]) for s in methods}
        assert ("constructor", "BaseModel") in scoped
        assert ("validate", "BaseModel") in scoped
        assert ("constructor", "User") in scoped
        assert ("toJSON", "User") in scoped
        assert ("log", "User") in scoped
        assert len(methods) == 5

    def test_function_symbols(self, db):
        funcs = db.lookup_symbols("*", kind="function")
        names = {s["name"] for s in funcs}
        assert "createUser" in names
        assert "loadUser" in names

    # --- Reference extraction ---

    def test_extends_reference(self, db):
        refs = _get_all_refs(db)
        extends = [r for r in refs if r["kind"] == "extends"]
        assert len(extends) == 1
        assert extends[0]["to_symbol_name"] == "BaseModel"

    def test_createUser_calls(self, db):
        cu = db.lookup_symbols("createUser")[0]
        refs = db.get_refs(symbol_id=cu["id"])
        targets = {r["to_symbol_name"] for r in refs if r["kind"] == "calls"}
        assert "validate" in targets

    def test_log_calls_toJSON(self, db):
        log_sym = [s for s in db.lookup_symbols("log", kind="method")
                   if s["scope"] == "User"][0]
        refs = db.get_refs(symbol_id=log_sym["id"])
        targets = {r["to_symbol_name"] for r in refs if r["kind"] == "calls"}
        assert "toJSON" in targets

    def test_stored_ref_count(self, db):
        """Parser produces 8 refs, 3 are from '<module>' (loadUser arrow fn
        calling readFile, parse, createUser). With <module> pseudo-symbol fix, all are stored. 9 refs stored."""
        refs = _get_all_refs(db)
        assert len(refs) == 9

    # --- Edge correctness ---

    def test_contains_edges(self, db):
        edges = _get_edges(db)
        contains = {(e["from_name"], e["to_name"]) for e in edges if e["type"] == "contains"}
        assert ("BaseModel", "constructor") in contains
        assert ("BaseModel", "validate") in contains
        assert ("User", "constructor") in contains
        assert ("User", "toJSON") in contains
        assert ("User", "log") in contains
        assert len(contains) == 5

    def test_extends_edge(self, db):
        edges = _get_edges(db)
        extends = [(e["from_name"], e["to_name"]) for e in edges if e["type"] == "extends"]
        assert ("User", "BaseModel") in extends
        assert len(extends) == 1

    def test_calls_edges(self, db):
        edges = _get_edges(db)
        calls = {(e["from_name"], e["to_name"]) for e in edges if e["type"] == "calls"}
        assert ("log", "toJSON") in calls
        assert ("createUser", "validate") in calls

    def test_no_duplicate_edges(self, db):
        edges = _get_edges(db)
        tuples = [(e["from_name"], e["to_name"], e["type"]) for e in edges]
        assert len(tuples) == len(set(tuples)), f"Duplicate edges: {tuples}"

    def test_total_edge_count(self, db):
        """Parser produces 10 edges, 1 from '<module>' (→createUser calls).
        With <module> pseudo-symbol fix, all are stored. 11 edges stored."""
        edges = _get_edges(db)
        assert len(edges) == 11

    # --- Query correctness ---

    def test_lookup_exact(self, db):
        results = db.lookup_symbols("User")
        assert len(results) == 1
        assert results[0]["kind"] == "class"

    def test_get_callers(self, db):
        validate = [s for s in db.lookup_symbols("validate", kind="method")
                    if s["scope"] == "BaseModel"][0]
        callers = db.get_callers(validate["id"])
        caller_names = {c["name"] for c in callers}
        assert "createUser" in caller_names

    def test_get_forward_refs_user(self, db):
        user = db.lookup_symbols("User")[0]
        fwd = db.get_forward_refs(user["id"], depth=1)
        names = {s["name"] for s in fwd}
        assert "constructor" in names
        assert "toJSON" in names
        assert "log" in names


# ---------------------------------------------------------------------------
# PHP validation
# ---------------------------------------------------------------------------

class TestPHPValidation:
    """Validate PHP extraction against tests/fixtures/php_sample.php.

    Expected (hand-verified from parser output):
      8 symbols (7 + 1 <module> pseudo-symbol), 10 refs stored
      (3 calls, 1 extends, 2 hooks_into, 2 type_reference)
    """

    @pytest.fixture
    def db(self, tmp_path):
        pipeline = _index_fixtures(str(tmp_path), ["php_sample.php"])
        return pipeline.project_db

    # --- Symbol extraction ---

    def test_symbol_count(self, db):
        syms = db.lookup_symbols("*")
        assert len(syms) == 8

    def test_import_symbols(self, db):
        imports = db.lookup_symbols("*", kind="import")
        assert len(imports) == 1

    def test_class_symbols_with_namespace(self, db):
        classes = db.lookup_symbols("*", kind="class")
        names = {s["name"] for s in classes}
        assert "App\\Analytics\\Tracker" in names
        assert "App\\Analytics\\ClickTracker" in names

    def test_method_symbols(self, db):
        methods = db.lookup_symbols("*", kind="method")
        scoped = {(s["name"], s["scope"]) for s in methods}
        assert ("track", "App\\Analytics\\Tracker") in scoped
        assert ("flush", "App\\Analytics\\Tracker") in scoped
        assert ("trackClick", "App\\Analytics\\ClickTracker") in scoped
        assert len(methods) == 3

    def test_function_with_namespace(self, db):
        funcs = db.lookup_symbols("*", kind="function")
        names = {s["name"] for s in funcs}
        assert "App\\Analytics\\create_tracker" in names

    # --- Reference extraction ---

    def test_extends_reference(self, db):
        refs = _get_all_refs(db)
        extends = [r for r in refs if r["kind"] == "extends"]
        assert len(extends) == 1
        assert "Tracker" in extends[0]["to_symbol_name"]

    def test_method_call_references(self, db):
        """Stored refs include: Tracker→flush, Tracker→dispatch, ClickTracker→track.
        Note: Parser attributes these to class names, not method names."""
        refs = _get_all_refs(db)
        call_refs = [r for r in refs if r["kind"] == "calls"]
        targets = {r["to_symbol_name"] for r in call_refs}
        assert "flush" in targets
        assert "dispatch" in targets
        assert "track" in targets

    def test_stored_ref_count(self, db):
        """10 refs: 3 calls, 1 extends, 2 hooks_into, 2 type_reference (BaseEvent param, Tracker return)."""
        refs = _get_all_refs(db)
        assert len(refs) == 10

    # --- Edge correctness ---

    def test_contains_edges(self, db):
        edges = _get_edges(db)
        contains = {(e["from_name"], e["to_name"]) for e in edges if e["type"] == "contains"}
        assert any("Tracker" in f and "track" == t for f, t in contains)
        assert any("Tracker" in f and "flush" == t for f, t in contains)
        assert any("ClickTracker" in f and "trackClick" == t for f, t in contains)
        assert len(contains) == 3

    def test_calls_edges(self, db):
        edges = _get_edges(db)
        calls = {(e["from_name"], e["to_name"]) for e in edges if e["type"] == "calls"}
        # ClickTracker→track is stored (both symbols resolved)
        assert any("track" == t for _, t in calls)

    def test_no_duplicate_edges(self, db):
        edges = _get_edges(db)
        tuples = [(e["from_name"], e["to_name"], e["type"]) for e in edges]
        assert len(tuples) == len(set(tuples)), f"Duplicate edges: {tuples}"

    def test_total_edge_count(self, db):
        """Parser produces 7 edges from parser, but 2 hooks_into edges target external
        hooks (init, analytics_enabled) that have no symbol IDs, so they can't be stored.
        5 edges stored (3 contains + 2 calls)."""
        edges = _get_edges(db)
        assert len(edges) == 5

    # --- Query correctness ---

    def test_lookup_namespaced(self, db):
        """Can look up by full namespace or substring."""
        results = db.lookup_symbols("*Tracker*")
        names = {s["name"] for s in results}
        assert any("Tracker" in n for n in names)

    def test_get_callers_track(self, db):
        """track is called by trackClick (method-level attribution)."""
        track = db.lookup_symbols("track", kind="method")[0]
        callers = db.get_callers(track["id"])
        caller_names = {c["name"] for c in callers}
        assert "trackClick" in caller_names


# ---------------------------------------------------------------------------
# Cross-file validation
# ---------------------------------------------------------------------------

class TestCrossFileValidation:
    """Validate cross-file name resolution using cross_file_a.py + cross_file_b.py.

    cross_file_a.py defines Logger + get_logger.
    cross_file_b.py imports get_logger, calls it and logger.info().

    Expected:
      File A: 3 symbols + 1 <module> = 4 symbols, 1 ref, 2 edges
      File B: 2 symbols + 1 <module> = 3 symbols, 2 refs, 0 edges
      Total: 7 symbols, 3 refs, 2 edges
    """

    @pytest.fixture
    def db(self, tmp_path):
        pipeline = _index_fixtures(
            str(tmp_path), ["cross_file_a.py", "cross_file_b.py"]
        )
        return pipeline.project_db

    def test_total_symbols(self, db):
        syms = db.lookup_symbols("*")
        assert len(syms) == 7  # (3 from A + 1 <module>) + (2 from B + 1 <module>)

    def test_file_a_symbols(self, db):
        syms = db.lookup_symbols("*", kind="class")
        assert any(s["name"] == "Logger" for s in syms)

        funcs = db.lookup_symbols("get_logger", kind="function")
        assert len(funcs) == 1

        methods = db.lookup_symbols("info", kind="method")
        assert len(methods) == 1
        assert methods[0]["scope"] == "Logger"

    def test_file_b_symbols(self, db):
        imports = db.lookup_symbols("*", kind="import")
        assert len(imports) == 1

        funcs = db.lookup_symbols("process", kind="function")
        assert len(funcs) == 1

    def test_cross_file_call_refs(self, db):
        """process() calls get_logger and info — stored with unresolved to_symbol_id."""
        proc = db.lookup_symbols("process", kind="function")[0]
        refs = db.get_refs(symbol_id=proc["id"])
        targets = {r["to_symbol_name"] for r in refs if r["kind"] == "calls"}
        assert "get_logger" in targets
        assert "info" in targets

    def test_get_logger_calls_logger(self, db):
        """get_logger calls Logger (constructor)."""
        gl = db.lookup_symbols("get_logger")[0]
        refs = db.get_refs(symbol_id=gl["id"])
        targets = {r["to_symbol_name"] for r in refs}
        assert "Logger" in targets

    def test_cross_file_callers(self, db):
        """get_logger should be called by process (cross-file name resolution)."""
        gl = db.lookup_symbols("get_logger")[0]
        callers = db.get_callers(gl["id"])
        caller_names = {c["name"] for c in callers}
        assert "process" in caller_names

    def test_cross_file_callers_info(self, db):
        """info should be called by process (cross-file method call)."""
        info = db.lookup_symbols("info", kind="method")[0]
        callers = db.get_callers(info["id"])
        caller_names = {c["name"] for c in callers}
        assert "process" in caller_names

    def test_cross_file_impact(self, db):
        """Impact on Logger at depth 1 reaches info (contained)."""
        logger = db.lookup_symbols("Logger")[0]
        fwd = db.get_forward_refs(logger["id"], depth=1)
        names = {s["name"] for s in fwd}
        assert "info" in names

    def test_total_ref_count(self, db):
        refs = _get_all_refs(db)
        assert len(refs) == 3

    def test_contains_edges(self, db):
        edges = _get_edges(db)
        contains = {(e["from_name"], e["to_name"]) for e in edges if e["type"] == "contains"}
        assert ("Logger", "info") in contains

    def test_calls_edges(self, db):
        edges = _get_edges(db)
        calls = {(e["from_name"], e["to_name"]) for e in edges if e["type"] == "calls"}
        assert ("get_logger", "Logger") in calls

    def test_total_edge_count(self, db):
        edges = _get_edges(db)
        assert len(edges) == 2
