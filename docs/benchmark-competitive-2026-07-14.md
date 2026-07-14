# Tessera vs CodeGraph: Python, TypeScript, and Documentation Benchmark

*Generated 2026-07-14T05:10:25.144042+00:00 from pinned Flask and Next.js revisions.*

## Technical Summary

**Tessera leads the normalized source-code comparison on MRR@10.** Tessera with BGE-small + Jina-tiny scored 0.577; CodeGraph scored 0.413 across the same Python and TypeScript query set. Tessera also covers document and mixed-source queries, which CodeGraph does not index and which are therefore reported separately.

Across all Tessera-supported content, reranking every query is counterproductive: base hybrid retrieval scored 0.610, global Jina-tiny reranking scored 0.588, and the post-hoc code-only reranking policy scored 0.638.

The exact-edge guardrail remains 4/4 for Tessera versus 3/4 for CodeGraph. These graph cases measure exact target correctness, not retrieval ranking.

## Normalized Code Retrieval

Both engines are reduced to ordered source-file paths and scored against the same expected files.

| Engine | Queries | MRR@10 | Top-1 | Top-3 | Top-10 | Mean latency | P95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| tessera_bge_small | 23 | 0.506 | 43% | 57% | 61% | 50 ms | 120 ms |
| tessera_bge_small_jina_tiny | 23 | 0.577 | 48% | 65% | 74% | 769 ms | 1038 ms |
| codegraph | 23 | 0.413 | 35% | 48% | 48% | 2931 ms | 5550 ms |

Latency is directional: Tessera uses an in-process cached index, while CodeGraph is invoked as a fresh CLI process for each query.

## Ranker Ablation Across All Content

The routed policy is derived from the same measured rows: use Jina-tiny for code queries and preserve base hybrid ordering for document and mixed queries.

| Tessera policy | Queries | MRR@10 | Top-1 | Top-3 | Top-10 |
|---|---:|---:|---:|---:|---:|
| BGE-small hybrid | 58 | 0.610 | 50% | 67% | 81% |
| BGE-small + Jina-tiny globally | 58 | 0.588 | 47% | 66% | 83% |
| BGE-small + Jina-tiny on code only (post-hoc) | 58 | 0.638 | 52% | 71% | 86% |

## Benchmark-Driven Regression Fix

The initial 14-case smoke run exposed a retrieval defect: the high-confidence BM25 shortcut could return chunks from only one file despite `file_dedup=True`, then skip semantic retrieval. Flask document Top-10 was 0/2 even though both targets were indexed.

The fix over-fetches chunk candidates before file deduplication and declines the BM25 shortcut when it cannot fill the requested unique-file limit. Two red/green unit cases cover duplicate-heavy and single-hit pools.

| Smoke configuration | MRR@10 before | MRR@10 after | Top-10 before | Top-10 after |
|---|---:|---:|---:|---:|
| BGE-small hybrid | 0.536 | 0.604 | 71% | 86% |
| BGE-small + Jina-tiny | 0.554 | 0.648 | 71% | 93% |

The Flask document subset improved from 0/2 to 2/2 Top-10 under both configurations. This smoke comparison uses the same queries and cached indexes before and after the search patch.

## Language and Content Segments

| Engine | Repository / segment | Queries | MRR@10 | Top-3 | Top-10 |
|---|---|---:|---:|---:|---:|
| codegraph | flask / code | 10 | 0.650 | 80% | 80% |
| codegraph | nextjs / code | 13 | 0.231 | 23% | 23% |
| tessera_bge_small | flask / code | 10 | 0.764 | 80% | 90% |
| tessera_bge_small | flask / cross | 2 | 0.625 | 50% | 100% |
| tessera_bge_small | flask / document | 8 | 0.679 | 88% | 100% |
| tessera_bge_small | nextjs / code | 13 | 0.308 | 38% | 38% |
| tessera_bge_small | nextjs / cross | 14 | 0.469 | 50% | 86% |
| tessera_bge_small | nextjs / document | 11 | 0.955 | 100% | 100% |
| tessera_bge_small_jina_tiny | flask / code | 10 | 0.650 | 70% | 70% |
| tessera_bge_small_jina_tiny | flask / cross | 2 | 1.000 | 100% | 100% |
| tessera_bge_small_jina_tiny | flask / document | 8 | 0.666 | 75% | 100% |
| tessera_bge_small_jina_tiny | nextjs / code | 13 | 0.520 | 62% | 77% |
| tessera_bge_small_jina_tiny | nextjs / cross | 14 | 0.370 | 43% | 71% |
| tessera_bge_small_jina_tiny | nextjs / document | 11 | 0.756 | 82% | 100% |

## Engineering Priorities

1. **Gate reranking by content and language.** Jina-tiny improves Next.js code MRR from 0.308 to 0.520, but reduces Flask code from 0.764 to 0.650 and combined document retrieval from 0.839 to 0.718. Validate a TypeScript/code-only gate on an independent set before changing defaults.
2. **Improve TypeScript candidate generation.** Even after reranking, Next.js code Top-10 is 77%; the remaining misses are webpack configuration, HMR, and static generation. These should become regression cases before tuning weights.
3. **Sweep rankers behind the routed baseline.** Compare Jina tiny, turbo, and v3 (and code-specialized embedders such as CodeRankEmbed) per language/content segment rather than selecting on one aggregate.
4. **Finish the broader benchmark program.** Expand the exact graph track beyond four collision cases, add independently authored relevance labels, and run process-normalized latency plus repeated agent-task evaluations.

## Method and Definitions

- **Corpus:** Next.js `adf8c612addd` and Flask `2c1b30d0503c`.
- **Queries:** 58 unique ground-truth questions; each Tessera configuration runs all questions, while CodeGraph runs the source-code subset.
- **MRR@10:** reciprocal rank of the first expected file, averaged across supported queries; a miss contributes zero.
- **Top-k:** fraction of supported queries with any expected file in the first k unique file results.
- **Documents:** Markdown, MDX, and reStructuredText are grouped as document retrieval. Unsupported CodeGraph rows are excluded rather than scored as failures.
- **Tessera configuration:** BGE-small embeddings, hybrid retrieval, file deduplication; one run without reranking and one with Jina-tiny reranking.
- **Routing ablation:** post-hoc selection from measured rows, not a third model execution; it should be confirmed on an independent query set before becoming a default.
- **CodeGraph adapter:** ordered source-code file blocks emitted by `codegraph explore --max-files 10`.

## Validation Assessment

**Share with caveats.** File-level relevance labels and calculations are deterministic and machine-readable. The sample is broad enough to guide engineering priorities, but it is curated rather than independently blinded, and latency is not a process-normalized performance comparison.

Recommended next step: inspect the per-query misses in the JSON artifact, then add the highest-impact failures as permanent regression cases before changing models or ranker weights.
