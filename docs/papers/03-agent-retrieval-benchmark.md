# Paper 3: Benchmarking Agent-Oriented Code Retrieval

**Status:** IDEA
**Target venue:** MSR 2027 (benchmark/data track) or NeurIPS Datasets & Benchmarks 2026
**Strength:** Highest citation potential

## Abstract (Draft)

The rise of AI coding agents — tools that autonomously navigate, understand, and modify codebases — has created a new class of code retrieval requirements that existing benchmarks fail to measure. CodeSearchNet evaluates natural language to code retrieval using docstring-derived queries, testing whether a model can match a description to its implementation. CoIR extends this to broader information retrieval but still evaluates embedding models in isolation. Neither benchmark evaluates *search systems* — the full pipeline of indexing, chunking, multi-signal ranking, and result presentation that an agent actually interacts with.

We identify three critical gaps: (1) existing benchmarks use synthetic queries derived from code artifacts (docstrings, comments), while agents issue structural queries derived from task context ("where is the popup save handler," "what calls this function"); (2) existing benchmarks evaluate single content types, while agents need to find code, documentation, and configuration in the same query; (3) existing benchmarks measure retrieval model quality, while agents depend on system-level properties like chunking granularity, context window fit, and result deduplication.

We present AgentCodeBench, a methodology and initial dataset for evaluating code retrieval systems as agents use them. Our benchmark comprises curated query sets across three categories — code navigation, documentation retrieval, and cross-reference resolution — evaluated against real production codebases with human-annotated expected results. We report MRR, Top-K recall, and a novel "agent utility" metric that accounts for context window budget. We demonstrate that system-level differences (chunking strategy, fusion weights, result dedup) produce larger quality swings than model-level differences (embedding architecture, vector dimensionality), suggesting the research community's focus on embedding models may be misallocated for agent-facing retrieval.

## Key Claims

1. **Agent queries differ fundamentally from developer queries.** Agents issue structural, compositional queries; developers issue keyword lookups. Existing benchmarks test the latter.

2. **System-level properties dominate model-level properties** for agent retrieval quality. Chunking granularity, fusion strategy, and dedup have more impact than embedding model choice.

3. **Multi-category evaluation is essential.** A system that scores well on code but poorly on documentation fails agents that need both. Single-category benchmarks hide this.

4. **Context window budget matters.** An agent can only consume N tokens of search results. A metric that ignores this overvalues systems that return many low-quality results.

## Benchmark Design

### Query Categories

| Category | Description | Example |
|---|---|---|
| Code navigation | Find implementation of a specific concept | "popup save handler," "authentication middleware" |
| Documentation retrieval | Find docs describing a feature or API | "how popup conditions work," "theming configuration" |
| Cross-reference | Find code described by/describing a doc | "implementation of the architecture described in docs/popup-lifecycle.md" |
| Structural | Find by relationship, not content | "functions that call render_popup," "classes extending BaseModel" |

### Evaluation Metrics

- **MRR** — standard, for comparability with existing benchmarks
- **Top-K recall** (K=1,3,5,10) — measures how quickly the right result appears
- **Agent utility score** — weighted recall that penalizes results exceeding a context budget (e.g., 4K tokens)
- **Source diversity** — whether results span content types appropriately for the query

### Dataset Construction Methodology

1. Select a real, production codebase with diverse content types
2. Define queries from actual agent interaction patterns (not derived from code)
3. Human-annotate expected files/functions per query
4. Categorize queries by type (code, doc, cross-ref, structural)
5. Validate: each query must be answerable from the indexed content

## Novelty vs. Prior Art

| Benchmark | Queries | Content | Evaluates |
|---|---|---|---|
| CodeSearchNet (Husain et al., 2019) | Docstring-derived | Code only | Retrieval models |
| CoIR (Li et al., 2024) | NL descriptions | Code only | Retrieval models |
| SWE-bench (Jimenez et al., 2024) | Bug reports | Code only | Agent task completion |
| DevBench (Li et al., 2024) | Dev tasks | Code only | Agent task completion |
| HumanEval (Chen et al., 2021) | Function specs | Code only | Code generation |
| **AgentCodeBench (ours)** | **Agent-style structural** | **Code + docs + config** | **Search systems** |

**Gap:** No benchmark evaluates code *search systems* (not models) on *agent-style queries* (not NL descriptions) across *multiple content types* (not code only).

## Required Literature Review

- [ ] CodeSearchNet, CoIR, and code retrieval benchmarks
- [ ] SWE-bench and agent evaluation frameworks
- [ ] Information retrieval evaluation methodology (TREC, MS MARCO)
- [ ] RAG evaluation frameworks (RAGAS, ARES)
- [ ] Developer information seeking studies (Ko et al., Xia et al.)
- [ ] Context window optimization for LLM agents

## Data / Experiments Needed

- PM20/PM38 as initial dataset (already have)
- Expand to 2-3 additional codebases (different stacks, sizes)
- Cross-system evaluation: Tessera vs. Sourcegraph vs. GitHub search vs. raw embedding retrieval
- Model swap experiments: same system, different embeddings, measure delta
- System swap experiments: same embeddings, different chunking/fusion, measure delta
- Inter-annotator agreement on expected results

## Risks

- "Your benchmark is too small" — need to expand beyond PM
- "Your benchmark is biased toward your system" — need adversarial evaluation
- Dataset construction is labor-intensive
- Need buy-in from at least one other group to validate methodology
