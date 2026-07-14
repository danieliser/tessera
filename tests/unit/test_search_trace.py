"""Deterministic candidate-trace invariants and diagnostics."""

from __future__ import annotations

import json

import numpy as np

from tessera.search import SearchType, hybrid_search
from tessera.search_trace import (
    FederatedSearchTrace,
    SearchTrace,
    aggregate_candidate_diagnostics,
    candidate_retrieval_diagnostics,
    cross_project_diagnostics,
    federated_candidate_diagnostics,
    stratified_candidate_diagnostics,
)


class HybridDB:
    chunks = {
        1: {
            "id": 1,
            "file_id": 10,
            "file_path": "src/a.py",
            "source_type": "code",
            "content": "first chunk",
        },
        2: {
            "id": 2,
            "file_id": 10,
            "file_path": "src/a.py",
            "source_type": "code",
            "content": "second chunk",
        },
        3: {
            "id": 3,
            "file_id": 30,
            "file_path": "docs/guide.md",
            "source_type": "markdown",
            "content": "guide",
        },
        4: {
            "id": 4,
            "file_id": 40,
            "file_path": "src/b.py",
            "source_type": "code",
            "content": "other file",
        },
    }

    def keyword_search(self, *_args, limit: int, source_type=None, **_kwargs):
        rows = [
            {"id": 1, "score": -0.3},
            {"id": 2, "score": -0.2},
            {"id": 3, "score": -0.15},
            {"id": 4, "score": -0.1},
        ]
        if source_type:
            rows = [row for row in rows if self.chunks[row["id"]]["source_type"] in source_type]
        return [dict(row) for row in rows[:limit]]

    def get_all_embeddings(self):
        return [1, 2, 3, 4], np.array(
            [
                [0.8, 0.2],
                [0.9, 0.1],
                [1.0, 0.0],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        )

    def get_chunk(self, chunk_id):
        return dict(self.chunks[chunk_id])


def _run_hybrid(db, trace=None):
    return hybrid_search(
        "find handler",
        np.array([1.0, 0.0], dtype=np.float32),
        db,
        limit=2,
        source_type=["code"],
        file_dedup=True,
        trace=trace,
    )


def test_trace_is_observational_and_attributes_filter_dedup_and_fusion() -> None:
    untraced = _run_hybrid(HybridDB())
    trace = SearchTrace(project_id=7, project_name="fixture")
    traced = _run_hybrid(HybridDB(), trace)

    assert traced == untraced
    assert [result["id"] for result in traced] == [2, 4]
    assert trace.validate_complete_attribution() == []

    filtered = trace.candidates["7:3"]
    assert filtered["channels"]["semantic"]["eligible_for_fusion"] is False
    assert any(event["reason"] == "source_type_not_allowed" for event in filtered["events"])

    duplicate = trace.candidates["7:1"]
    assert any(event["reason"] == "duplicate_file" for event in duplicate["events"])
    winner = trace.candidates["7:2"]
    assert winner["channels"]["semantic"]["rank"] == 2
    assert winner["channels"]["semantic"]["fusion_rank"] == 1
    assert set(winner["fusion"]["contributions"]) == {"keyword", "semantic"}

    payload = trace.to_dict()
    assert payload["attribution_complete"] is True
    assert "content" not in json.dumps(payload)


def test_trace_records_bm25_shortcut_without_running_semantic_search() -> None:
    class ShortcutDB(HybridDB):
        def keyword_search(self, *_args, **_kwargs):
            return [{"id": 1, "score": -10.0}, {"id": 4, "score": -0.1}]

        def get_all_embeddings(self):
            raise AssertionError("semantic search must be skipped")

    trace = SearchTrace()
    results = hybrid_search(
        "exact_name",
        np.array([1.0, 0.0], dtype=np.float32),
        ShortcutDB(),
        limit=2,
        trace=trace,
    )

    assert [result["id"] for result in results] == [1, 4]
    assert all(result["short_circuited"] for result in results)
    assert trace.decisions["bm25_short_circuit"]["taken"] is True
    assert trace.channels["semantic"]["reason"] == "bm25_short_circuit"
    assert trace.channels["graph"]["reason"] == "bm25_short_circuit"
    assert trace.validate_complete_attribution() == []


def test_trace_records_graph_seed_expansion_and_channel_contribution() -> None:
    class Cursor:
        def fetchone(self):
            return (1,)

    class Connection:
        def execute(self, *_args, **_kwargs):
            return Cursor()

    class GraphDB(HybridDB):
        conn = Connection()

        def keyword_search(self, *_args, **_kwargs):
            return [{"id": 1, "score": -0.2}]

        def get_chunk(self, chunk_id):
            chunk = super().get_chunk(chunk_id)
            chunk["symbol_ids"] = "[10]" if chunk_id == 1 else "[]"
            return chunk

        def get_symbol_to_chunks_mapping(self):
            return {10: [1], 11: [4]}

    class Graph:
        loaded_at = 123.0

        def is_sparse_fallback(self):
            return False

        def personalized_pagerank(self, _seeds):
            return {10: 0.8, 11: 0.3}

    trace = SearchTrace(project_name="graph-fixture")
    results = hybrid_search(
        "target",
        None,
        GraphDB(),
        graph=Graph(),
        limit=2,
        search_types=[SearchType.LEX],
        trace=trace,
    )

    assert [result["id"] for result in results] == [1, 4]
    assert trace.channels["graph"]["status"] == "used"
    assert trace.decisions["graph_expansion"]["seed_symbol_ids"] == [10]
    assert set(trace.candidates["graph-fixture:1"]["fusion"]["contributions"]) == {
        "keyword",
        "graph",
    }
    assert trace.validate_complete_attribution() == []


def test_candidate_metrics_report_channel_recall_and_duplicate_crowding() -> None:
    trace = SearchTrace(project_name="fixture")
    _run_hybrid(HybridDB(), trace)
    diagnostics = candidate_retrieval_diagnostics(
        trace,
        ["src/b.py"],
        cutoffs=(1, 2, 3),
    )

    assert diagnostics["union"]["recall_at_k"] == {"1": 0.0, "2": 0.0, "3": 1.0}
    assert diagnostics["union"]["duplicate_file_count"] == 1
    assert diagnostics["union"]["file_diversity"] == 0.666667
    assert diagnostics["attribution_complete"] is True

    aggregate = aggregate_candidate_diagnostics([
        {"candidate_diagnostics": diagnostics},
        {"candidate_diagnostics": diagnostics},
    ])
    assert aggregate["union"]["queries"] == 2
    assert aggregate["union"]["recall_at_k"]["3"] == 1.0

    stratified = stratified_candidate_diagnostics([
        {
            "repository": "one",
            "language": "python",
            "content_type": "code",
            "candidate_diagnostics": diagnostics,
        },
        {
            "repository": "two",
            "language": "go",
            "content_type": "document",
            "candidate_diagnostics": diagnostics,
        },
    ])
    assert stratified["macro_per_repository"]["union"]["groups"] == 2
    assert set(stratified["per_repository"]) == {"one", "two"}
    assert set(stratified["protected_segments"]["language"]) == {"go", "python"}


def test_federated_trace_attributes_projects_scores_and_rerank_pool() -> None:
    project_a = {
        "id": 1,
        "project_id": 10,
        "project_name": "alpha",
        "file_path": "alpha/a.py",
        "score": 0.03,
    }
    project_a_second = {
        "id": 2,
        "project_id": 10,
        "project_name": "alpha",
        "file_path": "alpha/other.py",
        "score": 0.02,
    }
    project_b = {
        "id": 1,
        "project_id": 20,
        "project_name": "beta",
        "file_path": "beta/b.py",
        "score": 0.025,
    }
    union = [project_a, project_b, project_a_second]
    pool = union[:2]
    reranked = [(1, 0.9), (0, 0.7)]
    final = [project_b, project_a]

    trace = FederatedSearchTrace(query="handler", limit=2)
    trace.record_project_union(union)
    trace.record_rerank_pool(
        pool,
        requested_size=2,
        retrieval_size=10,
        selection_policy="file_coverage_first",
        document_char_limit=4096,
    )
    trace.record_rerank_result(pool, reranked, final)
    trace.record_final(final)

    attribution = cross_project_diagnostics(trace)
    assert attribution["project_count"] == 2
    assert attribution["projects"]["alpha"]["candidate_count"] == 2
    assert attribution["projects"]["beta"]["best_global_rank"] == 2
    assert attribution["source_concentration_hhi"] == 0.555556
    assert attribution["scores_are_probabilities"] is False
    assert trace.reranker["results"][0] == {
        "key": "20:1",
        "input_rank": 2,
        "output_rank": 1,
        "reranker_score": 0.9,
    }
    assert trace.reranker["requested_pool_size"] == 2
    assert trace.reranker["project_retrieval_size"] == 10
    assert trace.reranker["selection_policy"] == "file_coverage_first"
    assert trace.reranker["document_char_limit"] == 4096
    candidate_attribution = {
        candidate["key"]: candidate
        for candidate in trace.candidate_attribution()
    }
    assert candidate_attribution["20:1"]["rerank_pool_rank"] == 2
    assert candidate_attribution["20:1"]["final_rank"] == 1
    assert candidate_attribution["10:2"]["exclusion_reason"] == "rerank_pool_limit"
    assert trace.validate_complete_attribution() == []

    metrics = federated_candidate_diagnostics(
        trace,
        ["beta/b.py"],
        cutoffs=(1, 2),
    )
    assert metrics["project_union"]["recall_at_k"] == {"1": 0.0, "2": 1.0}
    assert metrics["rerank_pool"]["recall_at_k"] == {"1": 0.0, "2": 1.0}
    assert metrics["final"]["recall_at_k"] == {"1": 1.0, "2": 1.0}
    serialized = json.loads(trace.to_json())
    assert serialized["reranker"]["status"] == "used"
    assert serialized["attribution_complete"] is True
