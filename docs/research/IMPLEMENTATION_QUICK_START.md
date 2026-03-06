# Quick Start: Implementing Code Chunking Improvements for Tessera

## Diagnostic First (Do This Week 1)

Before implementing anything, diagnose where the noise is coming from:

```python
# In your benchmark evaluation script, add this diagnostic:

def diagnose_retrieval_quality(queries, results, chunks_db):
    """Analyze where retrieval breaks down."""

    for query, retrieved_chunks in zip(queries, results):
        # Q1: How many chunks retrieved before first correct answer?
        correct_idx = next(i for i, c in enumerate(retrieved_chunks)
                          if c['is_relevant'])
        print(f"Query '{query}': correct answer at rank {correct_idx}")

        # Q2: Are top-ranked chunks noise (wrong file, wrong function)?
        for i, chunk in enumerate(retrieved_chunks[:10]):
            relevance = chunk['semantic_relevance']
            file = chunk['file_path']
            parent_func = chunk['parent_function']
            print(f"  Rank {i}: {file}::{parent_func} (score: {relevance:.3f})")

        # Q3: If we filtered to top-3 files, would we catch it?
        files_in_top_10 = set(c['file_path'] for c in retrieved_chunks[:10])
        correct_file = retrieved_chunks[correct_idx]['file_path']
        print(f"  File-level pre-filtering: {'PASS' if correct_file in files_in_top_10 else 'FAIL'}")

        print()

# Run on your 20 benchmark queries to find patterns
```

**What to look for:**
- If correct answer is at rank 20+: Noise problem → Pre-filtering + Contextual embeddings help
- If correct answer is at rank 5-10: Reranking problem → Late chunking or better embedding model
- If multiple different relevant chunks: Deduplication may help

---

## Phase 1 Implementation (Weeks 1-2)

### Step 1A: Pre-Filtering (File-Level Search)

**File:** `/Users/danieliser/Toolkit/codemem/src/tessera/search.py`

Replace or wrap the main search function:

```python
async def search_with_prefiltering(
    query: str,
    k_chunks: int = 20,
    k_files: int = 20,  # NEW: file-level candidate limit
    embedding_model,
    db_connection,
) -> list[dict]:
    """
    Two-stage retrieval:
    1. File-level: FTS5 + FAISS on file-level summaries/docstrings
    2. Chunk-level: FAISS + FTS5 within top-K files
    """

    # Stage 1: Find relevant files
    file_level_query = build_file_query(query)  # Extract file-related terms
    relevant_files = await fts5_search(
        file_level_query,
        limit=k_files,
        index_type='files'  # Search on file names + docstrings only
    )
    file_ids = [f['id'] for f in relevant_files]

    # Stage 2: Search chunks within those files
    chunk_results = await faiss_search(
        query_embedding=embedding_model.embed(query),
        k=k_chunks,
        file_filter=file_ids  # CONSTRAINT: only these files
    )

    # Stage 3: RRF fusion (existing logic)
    fts5_results = await fts5_search(query, limit=k_chunks, file_filter=file_ids)
    final_results = rrf_fusion(faiss_results, fts5_results)

    return final_results

# Backward-compatible wrapper:
def search(query: str, k: int = 20, **kwargs):
    """Use pre-filtering by default."""
    return search_with_prefiltering(
        query,
        k_chunks=k,
        k_files=max(5, k // 3),  # e.g., k=20 → k_files=6
        **kwargs
    )
```

**Database Changes:** Tag chunks with file_id. Query should be:

```sql
-- In db.py, add file-level index if not exists
CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
    file_path, file_summary, language
);

-- During indexing:
INSERT INTO files_fts (file_path, file_summary, language)
SELECT file_path, /* first docstring or first 200 chars */, language
FROM chunks WHERE depth = 0;  -- Top-level chunks = file level
```

**Validation:** Benchmark on 5 queries. Expected: +5-10% MRR (immediate noise reduction).

---

### Step 1B: Contextual Embeddings (Chunk + Parent Context)

**File:** `/Users/danieliser/Toolkit/codemem/src/tessera/chunker.py`

Modify chunk creation to inject parent context:

```python
class ContextualChunk:
    """Chunk with parent context prepended."""

    def __init__(self, node, parent_context: str, original_chunker):
        self.chunk = original_chunker.chunk(node)
        self.parent_context = parent_context  # e.g., "[CLASS: UserService]"
        self.original_text = self.chunk.text
        self.contextual_text = self._build_contextual_text()

    def _build_contextual_text(self) -> str:
        """Return text with parent context for embedding."""
        context_parts = []

        # Add class context if inside a class
        if self.chunk.parent_class:
            context_parts.append(f"[CLASS: {self.chunk.parent_class}]")

        # Add function context
        if self.chunk.parent_function:
            context_parts.append(f"[FUNC: {self.chunk.parent_function}]")

        # Add import summary for file context
        if self.chunk.file_imports:
            context_parts.append(f"[IMPORTS: {self.chunk.file_imports[:100]}]")

        return " ".join(context_parts) + "\n" + self.original_text

    def for_embedding(self) -> str:
        """Return text to embed (includes context)."""
        return self.contextual_text

    def for_storage(self) -> str:
        """Return original text for display (no context)."""
        return self.original_text


# During indexing:
async def index_with_context(project_path: str, embedding_model, db):
    """Re-index existing chunks with contextual text."""

    for chunk in db.get_all_chunks():
        # Re-embed with context
        contextual_chunk = ContextualChunk(chunk.ast_node,
                                         chunk.parent_context,
                                         chunker)
        embedding = embedding_model.embed(contextual_chunk.for_embedding())

        # Store original + contextual embedding separately
        db.update_chunk(
            chunk.id,
            embedding=embedding,
            display_text=contextual_chunk.for_storage(),
            contextual_text=contextual_chunk.contextual_text
        )
```

**Cost:** One-time re-indexing on Popup Maker (~30 min). Then identical cost per rebuild (no change to re-indexing speed).

**Validation:** Benchmark full 20 queries. Expected: +15-25% MRR combined with pre-filtering.

---

## Phase 2 Implementation (Week 3, Optional)

### Step 2A: Late Chunking (If Using Jina API)

**File:** `/Users/danieliser/Toolkit/codemem/src/tessera/embeddings.py`

Add Jina support:

```python
class JinaEmbeddingClient(EmbeddingClient):
    """Late chunking support via Jina API."""

    def __init__(self, api_key: str, model: str = "jina-embeddings-v3"):
        self.client = JinaClient(api_key=api_key)
        self.model = model
        self.supports_late_chunking = True

    async def embed(self, texts: list[str], use_late_chunking: bool = True) -> list[list[float]]:
        """
        Embed texts with late chunking:
        - Full document embeddings (8K token context)
        - Chunking applied after embedding
        """

        if use_late_chunking:
            # Send full chunks; Jina will handle late chunking internally
            response = self.client.embed(
                texts,
                model=self.model,
                task="retrieval",  # Code-specific task
                input_type="document"
            )
        else:
            # Standard early chunking
            response = self.client.embed(texts, model=self.model)

        return response.embeddings


# In search.py, switch embedding model:
embedding_model = JinaEmbeddingClient(api_key=os.getenv("JINA_API_KEY"))

# Or keep Nomic, use Jina for validation:
embedding_model_nomic = NomicEmbeddingClient()
embedding_model_jina = JinaEmbeddingClient()

# A/B test on 5 queries:
results_nomic = search(query, embedding_model=embedding_model_nomic)
results_jina = search(query, embedding_model=embedding_model_jina)
# Compare MRR, latency
```

**Cost:** Jina API (~$0.02/1M tokens). Test on small sample first.

**Decision Point:** If Jina shows >3% improvement over Nomic, switch. Otherwise, stick with on-device Nomic.

---

## Phase 2B: Diagnostic Reranker Tuning

Before implementing more complex logic, verify your reranker is configured correctly:

```python
# In search.py, log reranker scores:

def search(query: str, k: int = 20, debug: bool = False):
    candidates = retrieve_candidates(query, k=k*2)  # Over-retrieve

    if debug:
        print(f"PRE-RERANK (top 5):")
        for i, c in enumerate(candidates[:5]):
            print(f"  {i}: {c['file']} score={c['score']:.3f}")

    reranked = reranker.rerank(query, candidates[:k])

    if debug:
        print(f"POST-RERANK (top 5):")
        for i, c in enumerate(reranked[:5]):
            print(f"  {i}: {c['file']} score={c['rerank_score']:.3f}")

    return reranked[:k]

# Run on 3 queries where MRR is currently bad
# If reranker scores don't change ranking much: reranker is weak
# If top-5 all wrong: retrieval quality is the issue (not reranking)
```

**Action:** If reranker weak, upgrade to Jina Reranker v3 (63.28 on CoIR code benchmark).

---

## Validation Checklist

- [ ] Diagnostic run: Identify bottleneck (file-level? chunk-level? reranking?)
- [ ] Pre-filtering: Implement + benchmark (target: +5-10% MRR)
- [ ] Contextual embeddings: Implement + benchmark (target: +15-25% combined)
- [ ] Combined MRR target: 0.691 → 0.80+
- [ ] Latency impact < 50ms per query
- [ ] Re-indexing time < 1 hour (Popup Maker)
- [ ] Storage increase < 20%

---

## File Locations to Modify

```
src/tessera/
  search.py          → Add pre-filtering stage
  chunker.py         → Add contextual text injection
  embeddings.py      → Add Jina client (optional)
  db.py              → Add file-level FTS5 index
  indexer.py         → Update re-indexing to include contextual embed

tests/
  test_search.py     → Add pre-filtering validation tests
```

---

## Rollback Plan

All changes are additive. If performance degrades:

1. **Pre-filtering hurting?** Set `k_files = unlimited` (disables file-level filtering)
2. **Contextual embeddings bad?** Keep both embeddings, query on original (contextual optional)
3. **Late chunking slow?** Fall back to Nomic, disable Jina

---

## Next: Monitor in Production

Once deployed, log:
- Chunk count per query (should stabilize around 20-30 with pre-filtering)
- MRR per query type (code search vs cross-file dependencies)
- Reranker score distribution (outliers = quality issues)
- Re-indexing times per codebase size

Adjust `k_files` ratio based on metrics.

