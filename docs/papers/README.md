# Tessera Research Papers

Potential academic publications derived from the Tessera project. Each paper targets a distinct contribution with minimal overlap. Papers are ordered by assessed strength of contribution.

## Status Key

- **IDEA** — Concept documented, needs literature review
- **RESEARCH** — Literature review in progress
- **ABSTRACT** — Abstract drafted, ready for co-author review
- **DRAFTING** — Paper in progress
- **SUBMITTED** — Under review
- **ACCEPTED** — Accepted for publication

## Papers

### Paper 1: Deterministic Graph-Augmented Retrieval for Multi-Source Code Intelligence

**File:** [01-graph-augmented-retrieval.md](01-graph-augmented-retrieval.md)
**Status:** IDEA
**Target venue:** ICSE 2027 or FSE 2027 (full paper, 12 pages)
**Strength:** Flagship contribution — novel fusion architecture with deterministic graphs

The core architecture paper. Weighted RRF fusion of FTS5 keyword matching, FAISS vector similarity, and PageRank graph ranking — where the graph signal is deterministic (tree-sitter AST), not learned. Demonstrated to outperform learned graph representations on structural code queries while maintaining semantic flexibility.

---

### Paper 2: Unified Multi-Source Code Search Across Code, Documentation, and Configuration

**File:** [02-multi-source-search.md](02-multi-source-search.md)
**Status:** IDEA
**Target venue:** ICSE 2027 or ASE 2026 (full paper, 10 pages)
**Strength:** High commercial relevance, clear gap in literature

No existing system unifies code, documentation, config, and assets in a single ranked result set. Covers source-type-aware chunking strategies (AST-aware for code, heading-aware for docs, path-aware for config), source-type-aware RRF weight shifting, and file-level deduplication across content types.

---

### Paper 3: Benchmarking Agent-Oriented Code Retrieval

**File:** [03-agent-retrieval-benchmark.md](03-agent-retrieval-benchmark.md)
**Status:** IDEA
**Target venue:** MSR 2027 (benchmark/data track) or NeurIPS Datasets & Benchmarks
**Strength:** Highest citation potential — fills a gap everyone will need

CodeSearchNet evaluates NL→code with docstring-derived queries. CoIR evaluates retrieval models. Neither evaluates *search systems* the way AI agents use them — structural queries across code and documentation categories. Documents the methodology for building PM20/PM38 and proposes a standard evaluation framework.

---

### Paper 4: Static Detection of Event-Driven Architecture Patterns Across Language Boundaries

**File:** [04-event-edge-analysis.md](04-event-edge-analysis.md)
**Status:** IDEA
**Target venue:** MSR 2027 (tool track, 4 pages) or ICPC 2027
**Strength:** Contained, quickest to write, clear contribution

Cross-language static analysis of event registrations and emissions using directional graph edges. Unifies WordPress hooks (PHP), EventEmitter/DOM (JS/TS), and Django signals (Python) under a single `registers_on`/`fires` model. Includes mismatch detection (orphaned listeners, unfired events) and semantic subtyping (action vs filter).

---

### Paper 5: Model-Agnostic Embedding Evaluation for Practical Code Retrieval

**File:** [05-embedding-evaluation.md](05-embedding-evaluation.md)
**Status:** IDEA
**Target venue:** EMNLP 2026 (short paper) or ACL Findings
**Strength:** Useful data point — challenges assumptions about code-specialized models

Controlled comparison of embedding models (BGE-small, Jina-Code-v2, Nomic, CodeRankEmbed) on identical real-world code navigation tasks. Finding: a 67MB general-purpose model outperformed 560MB code-specialized models. Challenges the assumption that code-specialized embeddings are necessary for code retrieval.

---

### Paper 6: Scope-Gated Search Federation for Multi-Agent Code Intelligence

**File:** [06-scope-gated-federation.md](06-scope-gated-federation.md)
**Status:** IDEA
**Target venue:** Workshop at ASE 2026 or CCS 2026 (position paper, 4 pages)
**Strength:** Emerging area (MCP security), hold for maturity

Capability-based access control for AI agent tool use. Project → collection → global scope tiers with search-time federation. Prevents agent data leakage across project boundaries during MCP sessions. Touches on under-explored intersection of access control semantics and AI agent frameworks.

---

## Potential Merges

- Papers 1 + 2 could merge into a single flagship ICSE submission covering the full architecture
- Papers 3 + 5 could merge into a comprehensive benchmark contribution
- Paper 4 could become a case study within Paper 1

## Next Steps

1. Literature review for each paper — identify prior art, confirm novelty claims
2. Draft abstracts for top 3
3. Decide on merge strategy vs. portfolio approach
4. Identify co-authors and venue deadlines
