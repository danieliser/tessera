"""Tests for search.py module."""

import pytest
import numpy as np
from codemem.search import rrf_merge, cosine_search


class TestRRFMerge:
    """Test Reciprocal Rank Fusion merging."""

    def test_rrf_merge_single_list(self):
        """Test RRF with a single ranked list."""
        ranked_lists = [
            [
                {"id": 1, "score": 0.9},
                {"id": 2, "score": 0.8},
            ]
        ]

        result = rrf_merge(ranked_lists, k=60)
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2
        assert "rrf_score" in result[0]

    def test_rrf_merge_multiple_lists(self):
        """Test RRF with multiple ranked lists."""
        ranked_lists = [
            [
                {"id": 1, "score": 0.9},
                {"id": 2, "score": 0.8},
                {"id": 3, "score": 0.7},
            ],
            [
                {"id": 2, "score": 0.95},
                {"id": 1, "score": 0.85},
                {"id": 4, "score": 0.75},
            ]
        ]

        result = rrf_merge(ranked_lists, k=60)
        assert len(result) == 4
        # Both items should have rrf_score
        assert all("rrf_score" in item for item in result)
        # Items appearing in both lists should have higher scores than items in only one list
        # ID 1 and 2 appear in both, so they should be top 2
        top_two_ids = {result[0]["id"], result[1]["id"]}
        assert top_two_ids == {1, 2}

    def test_rrf_merge_disjoint_sets(self):
        """Test RRF with disjoint item sets."""
        ranked_lists = [
            [{"id": 1}, {"id": 2}],
            [{"id": 3}, {"id": 4}]
        ]

        result = rrf_merge(ranked_lists, k=60)
        assert len(result) == 4
        ids = {item["id"] for item in result}
        assert ids == {1, 2, 3, 4}

    def test_rrf_merge_empty_lists(self):
        """Test RRF with empty list."""
        result = rrf_merge([], k=60)
        assert result == []

    def test_rrf_score_calculation(self):
        """Test that RRF scores are calculated correctly."""
        # For k=60, rank 1 score = 1/(60+1) = 1/61 ≈ 0.01639
        ranked_lists = [
            [{"id": 1}]
        ]

        result = rrf_merge(ranked_lists, k=60)
        assert len(result) == 1
        expected_score = 1.0 / (60 + 1)
        assert abs(result[0]["rrf_score"] - expected_score) < 0.0001

    def test_rrf_merge_enriches_items(self):
        """Test that RRF merge preserves and enriches item data."""
        ranked_lists = [
            [{"id": 1, "text": "hello"}],
            [{"id": 1, "score": 0.9}]
        ]

        result = rrf_merge(ranked_lists, k=60)
        assert len(result) == 1
        assert result[0]["id"] == 1
        assert "text" in result[0]
        assert "score" in result[0]
        assert "rrf_score" in result[0]


class TestCosineSearch:
    """Test cosine similarity search."""

    def test_cosine_search_empty_embeddings(self):
        """Test cosine search with empty embeddings."""
        query = np.zeros(768, dtype=np.float32)
        chunk_ids = []
        embeddings = np.array([], dtype=np.float32).reshape(0, 768)

        result = cosine_search(query, chunk_ids, embeddings)
        assert result == []

    def test_cosine_search_single_embedding(self):
        """Test cosine search with single embedding."""
        query = np.ones(768, dtype=np.float32)
        chunk_ids = [1]
        embeddings = np.ones((1, 768), dtype=np.float32)

        result = cosine_search(query, chunk_ids, embeddings, limit=10)
        assert len(result) == 1
        assert result[0]["id"] == 1
        assert 0.9 < result[0]["score"] <= 1.0  # Cosine similarity near 1

    def test_cosine_search_returns_sorted(self):
        """Test that results are sorted by score descending."""
        query = np.ones(768, dtype=np.float32)
        chunk_ids = [1, 2, 3]
        embeddings = np.array([
            np.ones(768, dtype=np.float32),
            np.zeros(768, dtype=np.float32),
            0.5 * np.ones(768, dtype=np.float32)
        ], dtype=np.float32)

        result = cosine_search(query, chunk_ids, embeddings)
        # Results should be sorted by score descending
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_cosine_search_limit(self):
        """Test that limit parameter works."""
        query = np.ones(768, dtype=np.float32)
        chunk_ids = [1, 2, 3, 4, 5]
        embeddings = np.random.randn(5, 768).astype(np.float32)

        result = cosine_search(query, chunk_ids, embeddings, limit=2)
        assert len(result) == 2

    def test_cosine_search_zero_query(self):
        """Test cosine search with zero query vector."""
        query = np.zeros(768, dtype=np.float32)
        chunk_ids = [1]
        embeddings = np.ones((1, 768), dtype=np.float32)

        result = cosine_search(query, chunk_ids, embeddings)
        assert result == []  # Zero vector has no meaningful similarity

    def test_cosine_search_normalized_vectors(self):
        """Test that cosine search works with normalized vectors."""
        # Vectors at 90 degrees should have ~0 similarity
        query = np.zeros(768, dtype=np.float32)
        query[0] = 1.0  # [1, 0, 0, ..., 0]

        chunk_ids = [1]
        embeddings = np.zeros((1, 768), dtype=np.float32)
        embeddings[0, 1] = 1.0  # [0, 1, 0, ..., 0]

        result = cosine_search(query, chunk_ids, embeddings)
        assert len(result) == 1
        assert abs(result[0]["score"]) < 0.01  # Nearly orthogonal

    def test_cosine_search_result_structure(self):
        """Test that results have correct structure."""
        query = np.ones(768, dtype=np.float32)
        chunk_ids = [42]
        embeddings = np.ones((1, 768), dtype=np.float32)

        result = cosine_search(query, chunk_ids, embeddings)
        assert len(result) == 1
        assert "id" in result[0]
        assert "score" in result[0]
        assert result[0]["id"] == 42
        assert isinstance(result[0]["score"], float)

    def test_cosine_search_multiple_chunks(self):
        """Test cosine search with multiple chunks."""
        query = np.ones(768, dtype=np.float32)
        chunk_ids = [10, 20, 30, 40, 50]
        embeddings = np.random.randn(5, 768).astype(np.float32)

        result = cosine_search(query, chunk_ids, embeddings, limit=5)
        assert len(result) <= 5
        result_ids = [r["id"] for r in result]
        assert set(result_ids).issubset(set(chunk_ids))
