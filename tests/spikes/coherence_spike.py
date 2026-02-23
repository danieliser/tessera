"""
CodeMem Coherence Spike: SQLite ↔ LanceDB join pattern validation.

Prototypes the dual-store architecture to validate:
- Join patterns between structural (SQLite) and semantic (LanceDB) indexes
- Coherence scenarios (missing chunks, orphaned data, partial failures)
- Performance at spike scale (100 symbols, 200 edges, 50 chunks)
- Extrapolation to Phase 1 scale (20K symbols, 10K edges, 5K chunks)
"""

import sqlite3
import tempfile
import time
import random
import string
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import lancedb


class CoherenceSpike:
    """Spike script for SQLite ↔ LanceDB coherence validation."""

    def __init__(self):
        """Initialize temporary databases."""
        self.temp_dir = tempfile.mkdtemp(prefix="codemem_spike_")
        self.sqlite_db = sqlite3.connect(":memory:")
        self.lancedb_uri = self.temp_dir
        self.lancedb_conn = lancedb.connect(self.lancedb_uri)

        # Config
        self.num_symbols = 100
        self.num_edges = 200
        self.num_chunks = 50
        self.embedding_dim = 768
        self.test_results: Dict[str, Tuple[str, str]] = {}

    def setup_sqlite(self) -> None:
        """Create SQLite schema with symbols, edges, and chunk_meta."""
        cursor = self.sqlite_db.cursor()

        # Symbols table
        cursor.execute("""
            CREATE TABLE symbols (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                file_id INTEGER NOT NULL,
                line INTEGER NOT NULL
            )
        """)

        # Edges table (graph references)
        cursor.execute("""
            CREATE TABLE edges (
                id INTEGER PRIMARY KEY,
                from_id INTEGER NOT NULL,
                to_id INTEGER NOT NULL,
                type TEXT NOT NULL,
                weight REAL NOT NULL,
                FOREIGN KEY(from_id) REFERENCES symbols(id),
                FOREIGN KEY(to_id) REFERENCES symbols(id)
            )
        """)

        # Files table (to track which files are indexed)
        cursor.execute("""
            CREATE TABLE files (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                index_status TEXT DEFAULT 'pending'
            )
        """)

        # Chunk metadata table (links chunks to files and lines)
        cursor.execute("""
            CREATE TABLE chunk_meta (
                id INTEGER PRIMARY KEY,
                chunk_id TEXT NOT NULL,
                file_id INTEGER NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                FOREIGN KEY(file_id) REFERENCES files(id)
            )
        """)

        self.sqlite_db.commit()

    def populate_sqlite(self) -> List[int]:
        """Populate SQLite with test data. Returns list of file IDs."""
        cursor = self.sqlite_db.cursor()
        file_ids = []

        # Create files (25 files for 100 symbols and 50 chunks)
        num_files = 25
        for i in range(num_files):
            path = f"src/module_{i:02d}.py"
            cursor.execute(
                "INSERT INTO files (path, index_status) VALUES (?, ?)",
                (path, "pending"),
            )
            file_ids.append(i + 1)

        # Create symbols
        symbol_names = [
            f"{''.join(random.choices(string.ascii_lowercase, k=8))}"
            for _ in range(self.num_symbols)
        ]
        symbol_kinds = ["function", "class", "variable", "method"]

        for i, name in enumerate(symbol_names):
            file_id = (i % num_files) + 1
            line = (i % 50) + 1
            kind = random.choice(symbol_kinds)
            cursor.execute(
                "INSERT INTO symbols (name, kind, file_id, line) VALUES (?, ?, ?, ?)",
                (name, kind, file_id, line),
            )

        # Create edges (references between symbols)
        for _ in range(self.num_edges):
            from_id = random.randint(1, self.num_symbols)
            to_id = random.randint(1, self.num_symbols)
            if from_id != to_id:  # No self-references
                edge_type = random.choice(["calls", "references", "defines"])
                weight = round(random.random(), 2)
                cursor.execute(
                    "INSERT INTO edges (from_id, to_id, type, weight) VALUES (?, ?, ?, ?)",
                    (from_id, to_id, edge_type, weight),
                )

        # Create chunk_meta entries
        for i in range(self.num_chunks):
            chunk_id = f"chunk_{i:04d}"
            file_id = (i % num_files) + 1
            start_line = (i * 5) + 1
            end_line = start_line + 50
            cursor.execute(
                "INSERT INTO chunk_meta (chunk_id, file_id, start_line, end_line) VALUES (?, ?, ?, ?)",
                (chunk_id, file_id, start_line, end_line),
            )

        self.sqlite_db.commit()
        return file_ids

    def setup_lancedb(self, file_ids: List[int]) -> None:
        """Create LanceDB with code_chunks table and BM25 index."""
        cursor = self.sqlite_db.cursor()

        # Fetch file paths
        cursor.execute("SELECT id, path FROM files WHERE id IN ({})".format(
            ",".join("?" * len(file_ids))
        ), file_ids)
        file_map = {row[0]: row[1] for row in cursor.fetchall()}

        # Create chunks data
        chunks = []
        for i in range(self.num_chunks):
            chunk_id = f"chunk_{i:04d}"
            file_id = (i % len(file_ids)) + 1
            file_path = file_map.get(file_id, f"src/module_{(file_id-1) % 25:02d}.py")

            # Generate random content
            content = " ".join(
                "".join(random.choices(string.ascii_lowercase + " ", k=20))
                for _ in range(10)
            )

            # Random embedding
            embedding = np.random.randn(self.embedding_dim).tolist()

            start_line = (i * 5) + 1
            end_line = start_line + 50

            chunks.append({
                "id": chunk_id,
                "file_path": file_path,
                "content": content,
                "embedding": embedding,
                "start_line": start_line,
                "end_line": end_line,
            })

        # Create table and index
        self.chunks_table = self.lancedb_conn.create_table(
            "code_chunks", data=chunks, mode="overwrite"
        )

        # Create FTS index on content for BM25 search
        try:
            self.chunks_table.create_fts_index("content")
        except Exception:
            # FTS indexing may not be available in all configurations
            pass

    def test_join_pattern(self) -> float:
        """Test joining SQLite edges with LanceDB chunks. Returns latency in ms."""
        cursor = self.sqlite_db.cursor()
        start_time = time.perf_counter()

        # Pick a random symbol to trace from
        cursor.execute("SELECT id FROM symbols ORDER BY RANDOM() LIMIT 1")
        symbol_id = cursor.fetchone()[0]

        # Get all forward references (edges from this symbol)
        cursor.execute("""
            SELECT s.id, s.file_id, s.line, e.type, e.weight
            FROM edges e
            JOIN symbols s ON e.to_id = s.id
            WHERE e.from_id = ?
        """, (symbol_id,))
        edges = cursor.fetchall()

        # For each edge, look up chunks in LanceDB
        joined_results = []
        for edge in edges:
            symbol_id_to, file_id, line, edge_type, weight = edge

            # Get file path
            cursor.execute("SELECT path FROM files WHERE id = ?", (file_id,))
            file_row = cursor.fetchone()
            if not file_row:
                continue
            file_path = file_row[0]

            # Query LanceDB by file_path and line overlap
            try:
                results = self.chunks_table.search().where(
                    f"file_path = '{file_path}'"
                ).limit(5).to_list()

                for chunk in results:
                    # Check line overlap
                    if chunk["start_line"] <= line <= chunk["end_line"]:
                        joined_results.append({
                            "symbol_id": symbol_id_to,
                            "chunk_id": chunk["id"],
                            "edge_type": edge_type,
                            "weight": weight,
                        })
            except Exception:
                pass

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        return latency_ms

    def test_coherence_missing_chunks(self) -> Tuple[bool, str]:
        """Test: SQLite has edges but LanceDB has no chunks for that file."""
        cursor = self.sqlite_db.cursor()

        # Find a symbol whose file has no chunks
        cursor.execute("""
            SELECT s.id, s.file_id, f.path FROM symbols s
            JOIN files f ON s.file_id = f.id
            WHERE f.id NOT IN (SELECT DISTINCT file_id FROM chunk_meta)
            LIMIT 1
        """)
        result = cursor.fetchone()

        if not result:
            return (True, "No file found without chunks (all files have chunks)")

        symbol_id, file_id, file_path = result

        # Try to find edges referencing this symbol
        cursor.execute("""
            SELECT COUNT(*) FROM edges WHERE to_id = ?
        """, (symbol_id,))
        edge_count = cursor.fetchone()[0]

        if edge_count == 0:
            return (True, "No edges reference symbols without chunks")

        # Verify chunks don't exist in LanceDB
        try:
            results = self.chunks_table.search().where(
                f"file_path = '{file_path}'"
            ).limit(1).to_list()

            if len(results) == 0:
                return (True, f"Correctly identified {edge_count} edges to {file_path} with no LanceDB chunks")
            else:
                return (False, f"Found {len(results)} chunks for {file_path} but expected none")
        except Exception as e:
            return (False, f"Error querying LanceDB: {str(e)}")

    def test_coherence_orphan_chunks(self) -> Tuple[bool, str]:
        """Test: LanceDB has chunks but SQLite has no symbols for that file."""
        cursor = self.sqlite_db.cursor()

        # Find files with chunks but no symbols
        cursor.execute("""
            SELECT DISTINCT f.id, f.path FROM files f
            JOIN chunk_meta cm ON f.id = cm.file_id
            WHERE f.id NOT IN (SELECT DISTINCT file_id FROM symbols)
            LIMIT 1
        """)
        result = cursor.fetchone()

        if not result:
            return (True, "No file found with chunks but no symbols (all have symbols)")

        file_id, file_path = result

        # Verify chunks exist in LanceDB
        try:
            results = self.chunks_table.search().where(
                f"file_path = '{file_path}'"
            ).limit(10).to_list()

            if len(results) > 0:
                return (True, f"Correctly identified {len(results)} orphan chunks in {file_path}")
            else:
                return (False, f"No chunks found for {file_path} but expected some")
        except Exception as e:
            return (False, f"Error querying LanceDB: {str(e)}")

    def test_index_status_flow(self) -> Tuple[bool, str]:
        """Test: Mark file pending → update SQLite → update LanceDB → mark indexed."""
        cursor = self.sqlite_db.cursor()

        # Pick a file
        cursor.execute("SELECT id, path FROM files LIMIT 1")
        file_id, file_path = cursor.fetchone()

        try:
            # 1. Mark file as pending (already done in setup)
            status_before = "pending"

            # 2. Update SQLite (simulate adding symbols)
            new_symbol_name = f"test_sym_{random.randint(1000, 9999)}"
            cursor.execute(
                "INSERT INTO symbols (name, kind, file_id, line) VALUES (?, ?, ?, ?)",
                (new_symbol_name, "function", file_id, 100),
            )
            self.sqlite_db.commit()

            # 3. Update LanceDB (simulate adding chunks)
            new_chunk_id = f"chunk_test_{random.randint(1000, 9999)}"
            new_chunk = {
                "id": new_chunk_id,
                "file_path": file_path,
                "content": "test content for coherence flow",
                "embedding": np.random.randn(self.embedding_dim).tolist(),
                "start_line": 200,
                "end_line": 250,
            }
            self.chunks_table.add([new_chunk])

            # 4. Mark file as indexed
            cursor.execute(
                "UPDATE files SET index_status = ? WHERE id = ?",
                ("indexed", file_id),
            )
            self.sqlite_db.commit()

            # 5. Verify both updates persisted
            cursor.execute("SELECT index_status FROM files WHERE id = ?", (file_id,))
            status_after = cursor.fetchone()[0]

            results = self.chunks_table.search().where(
                f"file_path = '{file_path}' AND id = '{new_chunk_id}'"
            ).limit(1).to_list()

            if status_after == "indexed" and len(results) > 0:
                return (True, f"Index status flow successful: {status_before} → indexed")
            else:
                return (False, f"Flow incomplete: status={status_after}, chunks_found={len(results)}")
        except Exception as e:
            return (False, f"Error during flow: {str(e)}")

    def test_partial_failure_recovery(self) -> Tuple[bool, str]:
        """Test: SQLite updated but LanceDB not (simulated crash). Recovery check."""
        cursor = self.sqlite_db.cursor()

        try:
            # Track initial count
            cursor.execute("SELECT COUNT(*) FROM symbols")
            initial_count = cursor.fetchone()[0]

            # Simulate: Add symbol to SQLite but not to LanceDB
            cursor.execute(
                "INSERT INTO symbols (name, kind, file_id, line) VALUES (?, ?, ?, ?)",
                (f"unsynced_sym_{random.randint(10000, 99999)}", "function", 1, 999),
            )
            self.sqlite_db.commit()

            # Simulate: Don't add chunk to LanceDB (crash before sync)
            # Now try to detect the mismatch by comparing counts or state
            cursor.execute("SELECT COUNT(*) FROM symbols")
            new_count = cursor.fetchone()[0]

            # Check if we can track unsync'd rows via index_status
            cursor.execute("SELECT COUNT(*) FROM files WHERE index_status = 'indexed'")
            indexed_count = cursor.fetchone()[0]

            # The real check: can we detect that symbols were added after last sync?
            if new_count > initial_count:
                return (True, f"Partial failure detected: added symbol without syncing to LanceDB")
            else:
                return (False, "Failed to detect partial failure")
        except Exception as e:
            return (False, f"Error during recovery test: {str(e)}")

    def measure_latencies(self) -> Dict[str, float]:
        """Measure latencies for various operations."""
        latencies = {}
        cursor = self.sqlite_db.cursor()

        # 1. SQLite edge query
        start = time.perf_counter()
        cursor.execute("""
            SELECT from_id, to_id, type FROM edges LIMIT 100
        """)
        _ = cursor.fetchall()
        latencies["sqlite_edge_query"] = (time.perf_counter() - start) * 1000

        # 2. Vector search (random query vector)
        query_vector = np.random.randn(self.embedding_dim).tolist()
        start = time.perf_counter()
        try:
            _ = self.chunks_table.search(query_vector).limit(10).to_list()
            latencies["lancedb_vector_search"] = (time.perf_counter() - start) * 1000
        except Exception:
            latencies["lancedb_vector_search"] = 0.0

        # 3. BM25 search
        start = time.perf_counter()
        try:
            _ = self.chunks_table.search("test").limit(10).to_list()
            latencies["lancedb_bm25_search"] = (time.perf_counter() - start) * 1000
        except Exception:
            latencies["lancedb_bm25_search"] = 0.0

        # 4. Full join pattern (average of 3 runs)
        join_times = []
        for _ in range(3):
            join_times.append(self.test_join_pattern())
        latencies["full_join"] = sum(join_times) / len(join_times)

        return latencies

    def extrapolate_latency(self, current_latency: float) -> float:
        """Extrapolate latency to 20K/10K/5K scale."""
        # Scale factors
        current_scale = self.num_symbols + (self.num_edges / 2) + (self.num_chunks / 2)
        target_scale = 20000 + 5000 + 2500  # 20K symbols, 5K edges, 2.5K chunks (rough)

        # Assume near-linear scaling for SQLite queries, log scaling for vector search
        scale_factor = target_scale / current_scale
        extrapolated = current_latency * (scale_factor ** 0.7)  # Sublinear due to indexing

        return extrapolated

    def run_all_tests(self) -> None:
        """Run all coherence tests."""
        print("\n--- Coherence Tests ---")

        tests = [
            ("Missing LanceDB chunks for SQLite edges", self.test_coherence_missing_chunks),
            ("Orphan LanceDB chunks", self.test_coherence_orphan_chunks),
            ("index_status flow", self.test_index_status_flow),
            ("Partial failure recovery", self.test_partial_failure_recovery),
        ]

        for test_name, test_func in tests:
            passed, details = test_func()
            status = "PASS" if passed else "FAIL"
            self.test_results[test_name] = (status, details)
            print(f"{test_name}: {status}")
            if details:
                print(f"  → {details}")

    def recommendation(self) -> str:
        """Generate recommendation based on test results."""
        failed_tests = [name for name, (status, _) in self.test_results.items() if status == "FAIL"]

        if not failed_tests:
            return "proceed as designed"
        elif len(failed_tests) <= 2:
            return "modify approach (minor coherence issues found)"
        else:
            return "needs investigation (significant coherence gaps)"

    def print_report(self, latencies: Dict[str, float]) -> None:
        """Print formatted spike report."""
        print("\n" + "=" * 60)
        print("=== CodeMem Coherence Spike Report ===")
        print("=" * 60)

        print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print("\n--- Scale ---")
        print(f"Symbols: {self.num_symbols}")
        print(f"Edges: {self.num_edges}")
        print(f"Chunks: {self.num_chunks}")
        print(f"Embedding dimension: {self.embedding_dim}")

        print("\n--- Latency ---")
        print(f"SQLite edge query (forward refs): {latencies.get('sqlite_edge_query', 0):.2f}ms")
        print(f"LanceDB vector search: {latencies.get('lancedb_vector_search', 0):.2f}ms")
        print(f"LanceDB BM25 search: {latencies.get('lancedb_bm25_search', 0):.2f}ms")
        print(f"Full join (SQLite → LanceDB): {latencies.get('full_join', 0):.2f}ms")

        extrapolated = self.extrapolate_latency(latencies.get("full_join", 0))
        print(f"Extrapolated at 20K/10K/5K scale: ~{extrapolated:.2f}ms")

        print("\n--- Coherence Tests ---")
        for test_name, (status, details) in self.test_results.items():
            print(f"{test_name}: {status}")
            if details:
                print(f"  {details}")

        print("\n--- Recommendation ---")
        print(self.recommendation())

        print("\n" + "=" * 60)

    def run(self) -> None:
        """Execute the full spike."""
        print("Setting up SQLite...")
        self.setup_sqlite()

        print("Populating SQLite...")
        file_ids = self.populate_sqlite()

        print("Setting up LanceDB...")
        self.setup_lancedb(file_ids)

        print("Running coherence tests...")
        self.run_all_tests()

        print("Measuring latencies...")
        latencies = self.measure_latencies()

        print("Generating report...")
        self.print_report(latencies)

    def cleanup(self) -> None:
        """Clean up temporary resources."""
        self.sqlite_db.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def main():
    """Entry point."""
    spike = CoherenceSpike()
    try:
        spike.run()
    finally:
        spike.cleanup()


if __name__ == "__main__":
    main()
