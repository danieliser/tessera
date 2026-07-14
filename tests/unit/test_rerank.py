"""Two-stage retrieval budget, diversity, and document invariants."""

from tessera.rerank import (
    MAX_RERANK_DOCUMENT_CHARS,
    build_rerank_document,
    rerank_candidate_budget,
    rerank_document_budget,
    rerank_retrieval_budget,
    select_rerank_candidates,
)


def _candidate(
    chunk_id: int,
    path: str,
    *,
    project_id: int = 1,
    file_id: int | None = None,
) -> dict:
    return {
        "id": chunk_id,
        "project_id": project_id,
        "file_id": file_id,
        "file_path": path,
        "content": f"chunk {chunk_id}",
        "score": 1.0 / chunk_id,
    }


def test_candidate_budget_expands_only_for_active_reranker() -> None:
    assert rerank_candidate_budget(10, reranker_active=False) == 10
    assert rerank_candidate_budget(10, reranker_active=True) == 30
    assert rerank_candidate_budget(20, reranker_active=True) == 50
    assert rerank_candidate_budget(100, reranker_active=True) == 100
    assert rerank_candidate_budget(0, reranker_active=True) == 0
    assert rerank_retrieval_budget(30, reranker_active=True) == 150
    assert rerank_retrieval_budget(50, reranker_active=True) == 250
    assert rerank_retrieval_budget(100, reranker_active=True) == 250
    assert rerank_retrieval_budget(10, reranker_active=False) == 10


def test_document_budget_conserves_visible_page_context() -> None:
    assert rerank_document_budget(10, 10) == MAX_RERANK_DOCUMENT_CHARS
    assert rerank_document_budget(10, 30) == 682
    assert rerank_document_budget(2, 6) == 682
    assert rerank_document_budget(10, 0) == 0


def test_coverage_first_pool_preserves_order_and_then_allows_second_chunks() -> None:
    results = [
        _candidate(1, "src/a.py", file_id=10),
        _candidate(2, "src/a.py", file_id=10),
        _candidate(3, "src/b.py", file_id=20),
        _candidate(4, "src/c.py", file_id=30),
        _candidate(5, "src/b.py", file_id=20),
    ]

    assert [item["id"] for item in select_rerank_candidates(results, 3)] == [1, 3, 4]
    assert [item["id"] for item in select_rerank_candidates(results, 5)] == [1, 3, 4, 2, 5]


def test_file_identity_is_project_local_and_unknown_paths_do_not_collapse() -> None:
    results = [
        _candidate(1, "src/shared.py", project_id=10),
        _candidate(1, "src/shared.py", project_id=20),
        _candidate(2, "", project_id=10),
        _candidate(3, "", project_id=10),
    ]

    assert [item["id"] for item in select_rerank_candidates(results, 4)] == [1, 1, 2, 3]


class SymbolDB:
    def get_chunk(self, _chunk_id):
        return {"symbol_ids": '[4, 4, "bad", 9]'}

    def get_symbol(self, symbol_id):
        return {
            4: {
                "name": "search",
                "kind": "method",
                "scope": "SearchService",
                "signature": "def search(self, query: str)",
            },
            9: {
                "name": "SearchService",
                "kind": "class",
                "scope": "(module)",
                "signature": "class SearchService",
            },
        }.get(symbol_id)


def test_structured_document_includes_path_symbol_and_bounded_content() -> None:
    result = {
        "id": 7,
        "project_name": "fixture",
        "file_path": "src/search.py",
        "source_type": "code",
        "start_line": 9,
        "end_line": 19,
        "content": "x" * (MAX_RERANK_DOCUMENT_CHARS * 2),
    }

    document = build_rerank_document(result, SymbolDB())

    assert len(document) == MAX_RERANK_DOCUMENT_CHARS
    assert "Project: fixture" in document
    assert "Path: src/search.py" in document
    assert "Lines: 10-20" in document
    assert "method SearchService.search :: def search(self, query: str)" in document
    assert "class SearchService :: class SearchService" in document
    assert "Content:\n" in document


def test_structured_document_tolerates_missing_and_malformed_symbol_metadata() -> None:
    class MalformedDB:
        def get_chunk(self, _chunk_id):
            return {"symbol_ids": "not-json"}

    document = build_rerank_document(
        {"id": 1, "content": "fallback", "file_path": "README.md"},
        MalformedDB(),
    )

    assert "Path: README.md" in document
    assert "Symbols:" not in document
    assert document.endswith("fallback")
