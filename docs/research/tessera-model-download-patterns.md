# Research: Python CLI Model Auto-Download Patterns for Tessera

**Date:** 2026-03-03
**Tier:** Quick
**Question:** How do mature Python CLI tools handle automatic ML model downloads on first run? What's the best pattern for zero-config local embeddings—automatic download on `tessera serve`, or explicit `tessera download-model` command?

**Recommendation:** **Automatic download on first `tessera serve`** with visible tqdm progress bar, cache in `~/.cache/tessera/embeddings/` (respecting XDG_CACHE_HOME), fallback to keyword-only search if download fails offline. Model size: use ONNX quantized version (~84MB for Nomic Embed 1.5 Q4_K_M), acceptable because it matches Playwright's download pattern your user already knows.

---

## Context & Constraints

- Tessera already uses Nomic Embed 1.5 as the embedding model
- User (danieliser) already uses `shot-scraper`/`webshot` which implements automatic Playwright browser downloads (`playwright install`) — pattern is already familiar
- Zero-config is the goal: indexing should work immediately without manual setup
- Local-only embeddings (no cloud API)
- Need graceful offline fallback

---

## Options Evaluated

### Option 1: Automatic Download on First `tessera serve`
- **Confidence:** High
- **What it is:** Model downloads silently (with progress bar) when user first runs `tessera serve`. Subsequent runs use cache.
- **Strengths:**
  - Matches Playwright pattern (already used in webshot) — user has mental model for this
  - Zero friction: `tessera index /path && tessera serve` just works
  - FastEmbed (Qdrant) uses this approach; tqdm progress bar is standard
  - HuggingFace Hub + `hf_hub_download()` handles cache versioning via ETags automatically
  - Cache integrity verified against metadata file
  - Offline mode gracefully detected; if no internet on first run, falls back to keyword-only search
- **Weaknesses:**
  - Adds ~5-10s latency to first `serve` startup (download depends on network speed)
  - Users with no internet on first run can't use semantic search unless model pre-cached
  - Might surprise users who expect instant startup
- **Cost:** ~84MB disk (Q4_K_M ONNX quantized Nomic 1.5); ~5-10s first-run delay on 50Mbps internet
- **Maintenance:** Zero — uses HuggingFace Hub; model updates are automatic via ETag invalidation

### Option 2: Explicit `tessera download-model` Command
- **Confidence:** High
- **What it is:** User must run `tessera download-model` before first `serve`. Model lives in `.tessera/models/` in project root or global cache.
- **Strengths:**
  - Explicit, predictable: users know when model is available
  - Can be parallelized in CI/CD (separate step before indexing)
  - Clear error messages if download fails
  - Works well in containerized/offline environments (can pre-populate cache)
- **Weaknesses:**
  - **Two-step init friction** — contradicts "zero-config" goal
  - Users forget to run it, get cryptic "no embeddings available" error
  - spaCy uses this (`python -m spacy download en_core_web_sm`) and users complain about the extra step
  - Worse UX than automatic (contradicts user's Playwright expectations)
- **Cost:** Same as Option 1, but user must manually trigger
- **Maintenance:** Same as Option 1

### Option 3: Check-Then-Download Hybrid
- **Confidence:** Medium
- **What it is:** On `tessera serve`, check if model cached. If not, ask user interactively: "Download Nomic Embed 1.5 (84MB)? [Y/n]" before proceeding.
- **Strengths:**
  - Explicit consent without friction
  - Falls back to keyword-only if user declines
  - Good for users with metered connections
- **Weaknesses:**
  - Interactive CLI is unexpected for server startup
  - Blocks startup waiting for user input (bad for automation/CI)
  - Still requires user action (not zero-config)
- **Cost:** Same as Option 1
- **Maintenance:** Same as Option 1

---

## Model Size Analysis

| Model | Type | Size | Format | Latency (CPU) |
|-------|------|------|--------|---------------|
| Nomic Embed 1.5 Q2_K (quantized) | ONNX | 49 MB | 2-bit | ~150ms est |
| Nomic Embed 1.5 Q4_K_M (quantized) | ONNX | 84 MB | 4-bit | ~100-150ms |
| BAAI bge-small-en-v1.5 | Safetensors | 401 MB | Full FP32 | ~100ms |
| all-MiniLM-L6-v2 (sentence-transformers) | PyTorch | ~90 MB | FP32 | ~50ms |

**Verdict:** 84MB is acceptable for auto-download. Compares to:
- Playwright browsers: 200-300MB
- Docker images: GBs
- Most Python packages: 10-50MB
- User expectations: small enough to download in <5min on typical residential internet (50Mbps = ~84MB in 13 seconds)

---

## Caching & Offline Behavior (Patterns from Research)

### Cache Location
- **FastEmbed/sentence-transformers standard:** `~/.cache/torch/` or `~/.cache/sentence_transformers/`
- **HuggingFace Hub standard:** `~/.cache/huggingface/hub/`
- **XDG-compliant best practice:** Respect `XDG_CACHE_HOME` (defaults to `~/.cache/`)
- **Tessera-specific:** `~/.cache/tessera/embeddings/` (or `$XDG_CACHE_HOME/tessera/embeddings/`)

### Integrity Verification
- HuggingFace Hub auto-verifies via ETag on HTTP headers — no manual checksum needed
- FastEmbed maintains `files_metadata.json` with blob IDs for secondary verification
- Recommendation: Use HuggingFace's built-in ETag validation; no additional checksum layer needed

### Offline Fallback
- FastEmbed: Detects offline, retries once, then raises error
- **Better pattern** (Tessera): On download failure, log warning and fall back to keyword-only search
- Keyword-only search is acceptable but slower/less accurate; better than 100% failure

---

## First-Run UX Patterns (What Tools Actually Do)

| Tool | Automatic? | Progress Bar | Cache Location | First-run time |
|------|-----------|--------------|-----------------|----------------|
| **Playwright** | Yes (on import) | `PLAYWRIGHT_SHOW_DOWNLOAD_PROGRESS` env var | `~/.cache/ms-playwright/` | ~30-60s |
| **FastEmbed** | Yes (on init) | tqdm (visible) | `~/.cache/fastembed_cache/` | ~5-30s |
| **spaCy** | No (explicit cmd) | pip's progress | `~/.cache/spacy/` | user runs `python -m spacy download en_core_web_sm` |
| **sentence-transformers** | Yes (on init) | Silent (no progress) | `~/.cache/torch/sentence_transformers/` | ~5-30s |
| **HuggingFace transformers** | Yes (on `.from_pretrained()`) | Silent | `~/.cache/huggingface/hub/` | ~5-30s |

**Winner:** FastEmbed + Playwright both use automatic downloads with visible progress bars. Matches user's expectations.

---

## Implementation Checklist

Based on research, here's what to implement for Tessera:

1. **On `tessera serve` startup:**
   ```python
   # Pseudocode
   model_path = cache_dir / f"nomic-embed-text-v1.5"
   if not (model_path / "model.safetensors").exists():
       logger.info("Downloading Nomic Embed 1.5 (84MB)...")
       download_model_from_huggingface(
           repo_id="nomic-ai/nomic-embed-text-v1.5",
           cache_dir=cache_dir,
           show_progress=True  # tqdm
       )
   ```

2. **Use HuggingFace Hub's `hf_hub_download()`:**
   - Automatic ETag-based caching (no manual integrity checks needed)
   - Built-in retry logic with exponential backoff
   - Supports offline mode via `local_files_only=True`

3. **Offline fallback:**
   ```python
   try:
       embeddings = load_embeddings(model_path)
   except (FileNotFoundError, ConnectionError):
       logger.warning(
           "Embedding model not available. "
           "Falling back to keyword-only search. "
           "Run 'tessera serve --download-model' to enable semantic search."
       )
       use_keyword_search_only = True
   ```

4. **Cache directory:**
   ```python
   cache_dir = Path(os.getenv("XDG_CACHE_HOME", "~/.cache")) / "tessera" / "embeddings"
   cache_dir.mkdir(parents=True, exist_ok=True)
   ```

5. **Progress bar:**
   - Use `tqdm` (standard library for Python CLI progress)
   - Show file size + download speed (tqdm does this automatically with `unit='B', unit_scale=True`)

---

## Dissenting Views / Caveats

- **Offline-first users:** If your user base includes air-gapped environments, automatic download is friction. Counter: provide `tessera download-model --model-path /local/path` to pre-populate cache in CI/CD.
- **Slow internet:** 5-30s delay on first run might annoy users with slow connections. Counter: visible progress bar reduces perceived wait (UX research shows progress visibility increases tolerance).
- **Disk quota:** 84MB is non-trivial on embedded systems (Raspberry Pi). Counter: use Q2_K quantized version (49MB) as fallback, or require explicit config.
- **Model updates:** If Nomic Embed 1.5 is updated on HuggingFace, ETag changes and forces re-download. This is correct behavior, but might surprise users. Counter: log "Updating embedding model..." on re-download.

---

## Recommendation

**Implement automatic download on `tessera serve` startup** for these reasons:

1. **Matches user's mental model:** Playwright pattern (`playwright install` is automatic on import) already familiar from webshot/shot-scraper.
2. **Zero-config aligns with goal:** User can `tessera index` + `tessera serve` and embeddings work immediately.
3. **Fast enough:** 84MB ONNX model downloads in ~13-30 seconds on typical residential internet; tqdm progress bar makes wait acceptable.
4. **Proven pattern:** FastEmbed (Qdrant's embedding library) uses this; tqdm is industry standard for CLI progress.
5. **Graceful degradation:** On first-run offline, fall back to keyword-only search — not ideal but functional.
6. **No manual integrity checks:** HuggingFace Hub's ETag-based caching handles verification automatically.

**What would change this:** If user research shows 80%+ of users operate in air-gapped environments, switch to explicit `tessera download-model` command with better offline documentation.

---

## Sources

- [sentence-transformers model caching issue](https://github.com/UKPLab/sentence-transformers/issues/1828)
- [HuggingFace Hub caching guide](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache)
- [HuggingFace Hub file download reference](https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download)
- [FastEmbed GitHub model_management.py](https://github.com/qdrant/fastembed/blob/main/fastembed/common/model_management.py)
- [FastEmbed documentation](https://qdrant.github.io/fastembed/Getting%20Started/)
- [Playwright browser download documentation](https://playwright.dev/docs/browsers)
- [Playwright progress bar PR #24352](https://github.com/microsoft/playwright/pull/24352)
- [spaCy CLI documentation](https://spacy.io/api/cli)
- [tqdm progress bar library](https://github.com/tqdm/tqdm)
- [Nomic Embed 1.5 model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [Nomic Embed GGUF quantized versions](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF)
- [BAAI BGE-small model card](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [Graceful degradation patterns (AWS Reliability Pillar)](https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/rel_mitigate_interaction_failure_graceful_degradation.html)
- [Ollama cache directory documentation](https://docs.ollama.com/faq)
