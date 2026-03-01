# Future: Embedding & Indexing Optimization

## Problem
Embedding large codebases (8K+ chunks) loads the MLX embedding model into unified memory (~45GB virtual address space on Apple Silicon) and pins CPU/GPU at 100% until complete. On machines with less memory or slower GPUs this could be a serious UX issue.

## Optimization Tasks

### 1. Throttled Batch Embedding
- Add configurable batch size and inter-batch delay
- Default: 64 chunks per batch, 100ms delay between batches
- Prevents CPU/GPU from pegging at 100% continuously
- Config option: `embedding_batch_size`, `embedding_batch_delay_ms`

### 2. Progressive Embedding
- Embed chunks incrementally during `reindex_incremental()` — only new/changed chunks
- Skip re-embedding unchanged files (hash comparison)
- Currently all chunks get embedded on every full index

### 3. Memory-Aware Embedding
- Check available system memory before starting
- If low memory: reduce batch size, add longer delays
- Warn user if embedding model will consume >50% of available RAM

### 4. Background/Async Embedding
- Option to embed in background after structural index completes
- Structural search (keyword + graph) works immediately
- Semantic search becomes available as embeddings complete
- Progress indicator via MCP status

### 5. Embedding Persistence
- Embeddings currently stored in SQLite (chunk_embeddings table) — already persistent
- Ensure reindex doesn't re-embed chunks that haven't changed
- Add `content_hash` column to chunk_meta for change detection

### 6. Model Idle Timeout Coordination
- When Tessera finishes embedding, signal the gateway to unload the model sooner
- Don't wait for the full 15-minute idle timeout
- Or: expose a "done embedding" hint to the gateway
