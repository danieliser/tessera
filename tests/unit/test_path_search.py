"""Invariant tests for the independent file/path candidate channel."""

from __future__ import annotations

import numpy as np

from tessera.db import ProjectDB
from tessera.search import SearchType, hybrid_search
from tessera.search_trace import SearchTrace


def _add_chunk(
    db: ProjectDB,
    path: str,
    *,
    content: str = "opaque implementation body",
    source_type: str = "code",
) -> int:
    file_id = db.upsert_file(1, path, "document" if source_type == "markdown" else "python", "hash")
    return db.insert_chunks(
        [
            {
                "project_id": 1,
                "file_id": file_id,
                "start_line": 1,
                "end_line": 2,
                "symbol_ids": [],
                "ast_type": "document" if source_type == "markdown" else "module",
                "chunk_type": "document" if source_type == "markdown" else "code",
                "content": content,
                "source_type": source_type,
                "file_path": path,
            }
        ]
    )[0]


def test_path_only_identifier_aliases_are_searchable(tmp_path) -> None:
    db = ProjectDB(str(tmp_path))
    target_id = _add_chunk(db, "src/network/HTTPResponseCache.ts")

    assert db.keyword_search("http response cache") == []
    for query in (
        "http response cache",
        "HTTPResponseCache",
        "http-response-cache",
        "http_response_cache",
        "SRC/NETWORK/httpresponsecache.TS",
    ):
        results = db.path_search(query, limit=5)
        assert results and results[0]["id"] == target_id
        assert results[0]["file_path"] == "src/network/HTTPResponseCache.ts"


def test_duplicate_basenames_and_generated_vendor_paths_remain_distinct(tmp_path) -> None:
    db = ProjectDB(str(tmp_path))
    source_id = _add_chunk(db, "src/api/ConfigLoader.py")
    generated_id = _add_chunk(db, "vendor/generated/ConfigLoader.py")

    basename_results = db.path_search("config loader", limit=10)
    assert [row["id"] for row in basename_results] == [source_id, generated_id]

    generated_results = db.path_search("generated config loader", limit=10)
    assert generated_results[0]["id"] == generated_id
    assert {row["id"] for row in generated_results} == {source_id, generated_id}


def test_path_search_respects_source_type_and_selects_one_chunk_per_file(tmp_path) -> None:
    db = ProjectDB(str(tmp_path))
    code_id = _add_chunk(db, "src/RunBook.py", source_type="code")
    markdown_id = _add_chunk(db, "docs/RunBook.md", source_type="markdown")
    markdown_file = db.get_chunk(markdown_id)["file_id"]
    second_markdown_id = db.insert_chunks(
        [
            {
                "project_id": 1,
                "file_id": markdown_file,
                "start_line": 20,
                "end_line": 25,
                "symbol_ids": [],
                "ast_type": "document",
                "chunk_type": "document",
                "content": "another opaque section",
                "source_type": "markdown",
                "file_path": "docs/RunBook.md",
            }
        ]
    )[0]

    code = db.path_search("run book", limit=10, source_type=["code"])
    docs = db.path_search("run book", limit=10, source_type=["markdown"])

    assert [row["id"] for row in code] == [code_id]
    assert [row["id"] for row in docs] == [markdown_id]
    assert second_markdown_id not in {row["id"] for row in docs}


def test_path_index_is_idempotent_and_deleted_with_file(tmp_path) -> None:
    db = ProjectDB(str(tmp_path))
    path = "src/RequestRouter.go"
    file_id = db.upsert_file(1, path, "go", "first")
    db.upsert_file(1, path, "go", "second")

    rows = db.conn.execute(
        "SELECT file_id FROM file_paths_fts WHERE file_id = ?",
        (file_id,),
    ).fetchall()
    assert len(rows) == 1

    _add_chunk(db, "src/OtherRouter.go")
    db.delete_file_data(path)
    rows = db.conn.execute(
        "SELECT file_id FROM file_paths_fts WHERE file_id = ?",
        (file_id,),
    ).fetchall()
    assert rows == []


def test_v3_database_is_backfilled_on_open(tmp_path) -> None:
    db = ProjectDB(str(tmp_path))
    target_id = _add_chunk(db, "docs/DeploymentPlaybook.md", source_type="markdown")
    db.conn.execute("DROP TABLE file_paths_fts")
    db.conn.execute("INSERT OR REPLACE INTO _meta (key, value) VALUES ('schema_version', '3')")
    db.conn.commit()
    db.close()

    migrated = ProjectDB(str(tmp_path))
    assert migrated.get_meta("schema_version") == "4"
    results = migrated.path_search("deployment playbook", limit=5)
    assert [row["id"] for row in results] == [target_id]


def test_hybrid_path_channel_recovers_candidate_absent_from_content_lists() -> None:
    class FakeDB:
        chunks = {
            **{
                chunk_id: {
                    "id": chunk_id,
                    "file_id": chunk_id,
                    "file_path": f"src/distractor_{chunk_id}.py",
                    "source_type": "code",
                    "content": "distractor",
                }
                for chunk_id in range(1, 11)
            },
            99: {
                "id": 99,
                "file_id": 99,
                "file_path": "src/HTTPResponseCache.ts",
                "source_type": "code",
                "content": "opaque",
            },
        }

        def keyword_search(self, *_args, **_kwargs):
            return []

        def path_search(self, *_args, **_kwargs):
            return [{"id": 99, "score": -8.0}]

        def get_all_embeddings(self):
            return list(range(1, 11)), np.array(
                [[1.0, 0.0] for _ in range(10)],
                dtype=np.float32,
            )

        def get_chunk(self, chunk_id):
            return dict(self.chunks[chunk_id])

    control = hybrid_search(
        "http response cache",
        np.array([1.0, 0.0], dtype=np.float32),
        FakeDB(),
        limit=10,
        enable_path_search=False,
    )
    trace = SearchTrace(project_name="path-fixture")
    treatment = hybrid_search(
        "http response cache",
        np.array([1.0, 0.0], dtype=np.float32),
        FakeDB(),
        limit=10,
        trace=trace,
    )

    assert 99 not in {row["id"] for row in control}
    assert treatment[0]["id"] == 99
    assert trace.channels["path"]["status"] == "used"
    assert trace.candidates["path-fixture:99"]["channels"]["path"]["rank"] == 1
    assert trace.validate_complete_attribution() == []


def test_no_path_match_preserves_control_order_and_legacy_db_capability() -> None:
    class LegacyDB:
        chunks = {
            1: {"id": 1, "file_id": 1, "file_path": "a.py", "content": "a"},
            2: {"id": 2, "file_id": 2, "file_path": "b.py", "content": "b"},
        }

        def keyword_search(self, *_args, **_kwargs):
            return [{"id": 1, "score": -0.2}, {"id": 2, "score": -0.1}]

        def get_chunk(self, chunk_id):
            return dict(self.chunks[chunk_id])

    class EmptyPathDB(LegacyDB):
        def path_search(self, *_args, **_kwargs):
            return []

    expected = hybrid_search(
        "unrelated query",
        None,
        EmptyPathDB(),
        limit=2,
        search_types=[SearchType.LEX],
        enable_path_search=False,
    )
    actual = hybrid_search(
        "unrelated query",
        None,
        EmptyPathDB(),
        limit=2,
        search_types=[SearchType.LEX],
    )
    legacy = hybrid_search(
        "unrelated query",
        None,
        LegacyDB(),
        limit=2,
        search_types=[SearchType.LEX],
    )

    assert actual == expected
    assert legacy == expected
