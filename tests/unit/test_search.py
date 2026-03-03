"""Tests for search.py module."""

import numpy as np

from tessera.search import cosine_search, rrf_merge


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


class TestExtractSnippet:
    """Test extract_snippet with collapsed ancestry."""

    def test_flat_snippet_no_ancestors(self):
        """Without ancestors, returns flat snippet with line numbers."""
        from tessera.search import extract_snippet

        content = "line0\nline1\nfoo_bar\nline3\nline4"
        result = extract_snippet(content, "foo_bar", context_lines=1, chunk_start_line=10)
        assert result["best_match_line"] == 12  # absolute: 10 + 2
        assert "12 |" in result["snippet"]
        assert result["ancestors"] == []

    def test_flat_snippet_backward_compat(self):
        """Calling with no extra params works like the old version."""
        from tessera.search import extract_snippet

        content = "alpha\nbeta\ngamma\ndelta\nepsilon"
        result = extract_snippet(content, "gamma")
        # chunk_start_line defaults to 0, so lines are 0-based
        assert result["best_match_line"] == 2
        assert "snippet" in result

    def test_collapsed_ancestry_single_parent(self):
        """Lines mode with one ancestor shows definition + collapse + match."""
        from tessera.search import extract_snippet

        content = (
            "    def helper(self):\n"  # line 0 (abs 20)
            "        x = 1\n"          # line 1 (abs 21)
            "        y = 2\n"          # line 2 (abs 22)
            "        foo_match = 3\n"  # line 3 (abs 23)
            "        z = 4\n"          # line 4 (abs 24)
            "        return z"         # line 5 (abs 25)
        )
        ancestors = [{
            "name": "MyClass", "kind": "class",
            "line": 10, "end_line": 30,
            "signature": "class MyClass:",
        }]
        result = extract_snippet(
            content, "foo_match", context_lines=1,
            ancestors=ancestors, chunk_start_line=20,
        )
        snippet = result["snippet"]
        # Should contain the class definition line
        assert "10 |" in snippet or "10 | " in snippet
        assert "class MyClass:" in snippet
        # Should contain collapse marker
        assert "..." in snippet
        assert "lines)" in snippet
        # Should contain the match with line numbers
        assert "23 |" in snippet or "23 | " in snippet
        assert "foo_match" in snippet
        # Ancestors metadata
        assert len(result["ancestors"]) == 1
        assert result["ancestors"][0]["name"] == "MyClass"

    def test_collapsed_ancestry_nested(self):
        """Lines mode with class → method nesting."""
        from tessera.search import extract_snippet

        content = "        target = True\n        other = False"
        ancestors = [
            {"name": "Processor", "kind": "class", "line": 5, "end_line": 50,
             "signature": "class Processor:"},
            {"name": "run", "kind": "method", "line": 15, "end_line": 40,
             "signature": "    def run(self):"},
        ]
        result = extract_snippet(
            content, "target", context_lines=1,
            ancestors=ancestors, chunk_start_line=30,
        )
        snippet = result["snippet"]
        # Both ancestors should appear
        assert "class Processor:" in snippet
        assert "def run(self):" in snippet
        # Match window
        assert "target" in snippet
        assert len(result["ancestors"]) == 2

    def test_full_mode(self):
        """Full mode expands entire outermost ancestor range."""
        from tessera.search import extract_snippet

        content = "line0\nline1\ntarget\nline3\nline4\nline5"
        ancestors = [{
            "name": "foo", "kind": "function",
            "line": 100, "end_line": 105,
            "signature": "def foo():",
        }]
        result = extract_snippet(
            content, "target", context_lines=1,
            ancestors=ancestors, chunk_start_line=100,
            mode="full",
        )
        # Full mode should include all lines from ancestor start to end
        assert result["snippet_start_line"] == 100
        assert result["snippet_end_line"] == 105
        assert len(result["ancestors"]) == 1

    def test_max_depth_limits_ancestors(self):
        """max_depth should trim ancestor list from outermost."""
        from tessera.search import extract_snippet

        content = "            deep_match = 1"
        ancestors = [
            {"name": "Module", "kind": "class", "line": 1, "end_line": 100,
             "signature": "class Module:"},
            {"name": "Inner", "kind": "class", "line": 10, "end_line": 80,
             "signature": "    class Inner:"},
            {"name": "method", "kind": "method", "line": 20, "end_line": 60,
             "signature": "        def method(self):"},
        ]
        result = extract_snippet(
            content, "deep_match", context_lines=0,
            ancestors=ancestors, chunk_start_line=40,
            max_depth=1,
        )
        # Only innermost ancestor should remain
        assert len(result["ancestors"]) == 1
        assert result["ancestors"][0]["name"] == "method"

    def test_empty_content(self):
        """Empty content returns empty result."""
        from tessera.search import extract_snippet

        result = extract_snippet("", "query")
        assert result["snippet"] == ""
        assert result["ancestors"] == []

    def test_no_ancestors_match_outside_symbols(self):
        """Top-level code with no containing symbols returns flat snippet."""
        from tessera.search import extract_snippet

        content = "import os\nimport sys\nfoo = 1"
        result = extract_snippet(
            content, "foo", context_lines=1,
            ancestors=[], chunk_start_line=1,
        )
        assert result["ancestors"] == []
        assert "foo" in result["snippet"]

    def test_semantic_scoring_finds_try_except(self):
        """Semantic scoring picks try/except over a comment containing query words."""
        import numpy as np
        from tessera.search import extract_snippet

        # Lines 0-4 are filler, lines 5-7 are the error handling block.
        # Keyword scoring would pick line 0 ("error handling" words match).
        # Semantic scoring should pick the try/except area.
        content = (
            "# error handling comment here\n"            # line 0
            "import os\n"                                # line 1
            "import sys\n"                               # line 2
            "import logging\n"                           # line 3
            "logger = logging.getLogger()\n"             # line 4
            "def process():\n"                           # line 5
            "    try:\n"                                  # line 6
            "        result = transform(data)\n"          # line 7
            "    except ValueError as e:\n"               # line 8
            "        logger.warning(e)\n"                 # line 9
            "        raise\n"                             # line 10
            "    return result"                           # line 11
        )

        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Mock embed_fn: only windows containing try/except score high
        def mock_embed(texts: list[str]) -> list[list[float]]:
            results = []
            for text in texts:
                if "try:" in text and "except" in text:
                    results.append([0.95, 0.05, 0.0])
                elif "try:" in text or "except" in text:
                    results.append([0.8, 0.1, 0.0])
                else:
                    results.append([0.1, 0.8, 0.2])
            return results

        result = extract_snippet(
            content, "error handling",
            context_lines=1,
            query_embedding=query_emb,
            embed_fn=mock_embed,
        )

        # Semantic scoring should pick the try/except area (lines 6-8),
        # not line 0 where "error handling" appears as text
        assert result["best_match_line"] in (6, 7, 8), (
            f"Expected try/except area (6-8), got line {result['best_match_line']}"
        )

    def test_semantic_scoring_fallback_on_error(self):
        """Falls back to keyword scoring when embed_fn raises."""
        import numpy as np
        from tessera.search import extract_snippet

        content = "alpha\nbeta\nerror_handler\ndelta"
        query_emb = np.array([1.0, 0.0], dtype=np.float32)

        def broken_embed(texts):
            raise RuntimeError("embedding server down")

        result = extract_snippet(
            content, "error_handler",
            query_embedding=query_emb,
            embed_fn=broken_embed,
        )
        # Should fall back to keyword and find "error_handler" on line 2
        assert result["best_match_line"] == 2


class TestGetAncestorSymbols:
    """Test ProjectDB.get_ancestor_symbols()."""

    def test_finds_containing_symbols(self, tmp_path):
        from tessera.db import ProjectDB

        db = ProjectDB(str(tmp_path))
        fid = db.upsert_file(1, "test.py", "python", "hash1")
        db.insert_symbols([
            {"project_id": 1, "file_id": fid, "name": "MyClass", "kind": "class",
             "line": 5, "col": 0, "scope": "", "signature": "class MyClass:", "end_line": 50},
            {"project_id": 1, "file_id": fid, "name": "do_work", "kind": "method",
             "line": 10, "col": 4, "scope": "MyClass", "signature": "def do_work(self):", "end_line": 30},
            {"project_id": 1, "file_id": fid, "name": "other", "kind": "function",
             "line": 55, "col": 0, "scope": "", "signature": "def other():", "end_line": 60},
        ])

        # Line 20 is inside MyClass.do_work
        ancestors = db.get_ancestor_symbols(fid, 20)
        names = [a["name"] for a in ancestors]
        assert "MyClass" in names
        assert "do_work" in names
        assert "other" not in names
        # Outermost first
        assert names.index("MyClass") < names.index("do_work")

    def test_no_containing_symbols(self, tmp_path):
        from tessera.db import ProjectDB

        db = ProjectDB(str(tmp_path))
        fid = db.upsert_file(1, "test.py", "python", "hash1")
        db.insert_symbols([
            {"project_id": 1, "file_id": fid, "name": "func", "kind": "function",
             "line": 10, "col": 0, "scope": "", "signature": "def func():", "end_line": 20},
        ])

        # Line 5 is before the function
        ancestors = db.get_ancestor_symbols(fid, 5)
        assert ancestors == []

    def test_exact_boundary_lines(self, tmp_path):
        from tessera.db import ProjectDB

        db = ProjectDB(str(tmp_path))
        fid = db.upsert_file(1, "test.py", "python", "hash1")
        db.insert_symbols([
            {"project_id": 1, "file_id": fid, "name": "func", "kind": "function",
             "line": 10, "col": 0, "scope": "", "signature": "def func():", "end_line": 20},
        ])

        # Exact start line
        assert len(db.get_ancestor_symbols(fid, 10)) == 1
        # Exact end line
        assert len(db.get_ancestor_symbols(fid, 20)) == 1
        # One past end
        assert len(db.get_ancestor_symbols(fid, 21)) == 0
