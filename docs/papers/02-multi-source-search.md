# Paper 2: Unified Multi-Source Code Search Across Code, Documentation, and Configuration

**Status:** IDEA
**Target venue:** ICSE 2027 / ASE 2026 (full paper, 10 pages)

## Abstract (Draft)

Modern software projects are not just code. Architecture decisions live in markdown. API contracts live in OpenAPI YAML. Build configuration lives in JSON and TOML. Deployment rules live in Dockerfiles and CI configs. Yet code search systems index only source code, and documentation search systems index only prose — forcing developers and AI agents to maintain separate mental models of where knowledge lives.

We present a unified retrieval architecture that indexes code, documentation, configuration, and structured data into a single searchable corpus with source-type-aware ranking. Our approach introduces three techniques: (1) AST-aware chunking for code that respects function and class boundaries, heading-aware chunking for documentation that preserves section hierarchy, and path-aware chunking for structured formats that maintains key provenance; (2) source-type-aware RRF weight shifting that automatically adjusts the balance between keyword, semantic, and graph signals based on detected content type; and (3) file-level deduplication that prevents result fragmentation when multiple chunks from the same file match.

Evaluated on a curated benchmark spanning code, documentation, and cross-reference queries against a production Next.js application (1,729 files across 8 content types), our unified approach achieves Top-10 recall of 100% on documentation queries and 100% on cross-reference queries, with a blend MRR of 0.87. We demonstrate that source-type routing improves code MRR by 0.10 while maintaining documentation quality, and that unified search enables a new class of cross-reference queries (e.g., "find the implementation described in the architecture doc") that are impossible in siloed systems.

## Key Claims

1. **Single-index multi-source search** is not just convenient — it enables cross-reference queries that siloed systems cannot answer.

2. **Source-type-aware chunking** is necessary because code, prose, and config have fundamentally different structural boundaries that affect retrieval quality.

3. **RRF weight shifting by source type** improves retrieval without requiring the user to specify what type of content they want.

4. **File-level dedup** is critical for multi-source corpora where a single file (e.g., a long markdown doc) may produce dozens of matching chunks.

## Novelty vs. Prior Art

| Prior work | What they do | What we add |
|---|---|---|
| Sourcegraph | Code-only search | No documentation, no config |
| Glean | Enterprise search across SaaS tools | Not code-aware, no AST chunking |
| Phind | AI-powered code search | No multi-source ranking architecture |
| Algolia DocSearch | Documentation-only | No code, no structural awareness |
| DevDocs | Aggregated API docs | No project-specific content, no search fusion |

**Gap:** No published system addresses the architecture for ranking code alongside its documentation in a unified result set with source-type-aware signals.

## Required Literature Review

- [ ] Multi-modal information retrieval
- [ ] Document chunking strategies for RAG
- [ ] Code-specific chunking (AST-aware approaches)
- [ ] File-level vs. passage-level retrieval tradeoffs
- [ ] Enterprise search architectures
- [ ] Developer information seeking behavior studies

## Data / Experiments Needed

- Breakdown of PM38 by source type (code vs doc vs cross-ref) — already have
- Ablation: unified index vs. separate indexes queried independently
- Ablation: with vs. without source-type-aware weight shifting
- Ablation: with vs. without file-level dedup
- Chunk boundary quality analysis (AST-aware vs. fixed-size vs. sliding window)
- User study or agent task completion comparison (stretch)

## Risks

- "You just index everything" is a hard sell without strong ablation data
- Need compelling cross-reference query examples beyond our PM corpus
- Reviewers may want larger-scale evaluation (multiple projects, diverse stacks)
