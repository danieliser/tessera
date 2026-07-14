# Candidate tracing benchmark — 2026-07-14

Candidate tracing passed its observational and operational gates on the full
six-repository development corpus. This is a diagnostic result, not a ranking
improvement: the traced run matched the frozen PR-01 baseline on all 30 query
ranks and every ordered top-file list.

## Frozen inputs

| Item | Value |
|---|---|
| Tessera revision | `a73c373ec93246b1fd22b57df0b0ab6799a8e0e0` |
| Corpus manifest SHA-256 | `0e84423ee5f116554af104908b0ba845cee25bc69bfecc38cb813e3deab1c701` |
| Repositories | attrs, Hono, FastRoute, pflag, swift-argument-parser, dry-configurable |
| Languages | Python, TypeScript, PHP, Go, Swift, Ruby |
| Content | code, document, mixed |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Queries | 30 development queries; no holdout access |

## Invariance and overhead

| Gate | Result |
|---|---:|
| Paired traced/untraced calls | 300 / 300 exactly equal |
| Frozen baseline query ranks changed | 0 / 30 |
| Frozen ordered top-file lists changed | 0 / 30 |
| Complete candidate attribution | 30 / 30 queries |
| Untraced warm search p95 | 4.9694 ms |
| Traced warm search p95 | 5.8670 ms |
| Added p95 | 0.8976 ms |
| p95 ratio | 1.1806x |
| Largest serialized query trace | 48,898 bytes |

The predeclared limits were +25 ms, 1.5x p95, and 250 KiB per query trace.
Tracing remains disabled by default and does not enter normal search payloads.

Focused validation passed 544 unit, graph-search, server, and federation tests,
plus strict documentation and lint. An additional default-suite run reached
255 passes and one skip before a large real-repository fixture exhausted the
host volume (`sqlite3.OperationalError: database or disk is full`). The 674
subsequent errors were fixture setup failures because pytest could no longer
create temporary directories; there were no assertion failures. Its generated
1.7 GiB pytest directory was removed after the run.

## Candidate findings

The primary values below are macro-averaged across repositories.

| Candidate set | Mean candidates | Recall@10 | Recall@20 | Recall@50 | Duplicate-file rate | File diversity |
|---|---:|---:|---:|---:|---:|---:|
| Keyword | 0.0333 | 0.0333 | 0.0333 | 0.0333 | 0.0000 | 0.0333 |
| Semantic | 34.7333 | 0.9667 | 1.0000 | 1.0000 | 0.4482 | 0.5518 |
| Graph | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Fused union | 34.7333 | 0.9667 | 1.0000 | 1.0000 | 0.4482 | 0.5518 |

Graph is explicitly recorded as skipped because the benchmark passes no graph,
matching the current server default. Its zero is an availability observation,
not evidence that graph retrieval is ineffective.

The lexical channel produced one candidate across the 30 natural-language
queries, so the current fused union was effectively semantic-only. This does
not justify a query-specific lexical rewrite. It strengthens the general case
for independently searchable path and symbol channels in PR-04/05.

The expected FastRoute result missed the visible top 10 but was present by
candidate rank 20. Across all repositories, Recall@20 reached 100%. This is the
mechanism PR-03 is intended to test: decouple retrieval depth from the visible
result limit and give a reranker a real candidate pool. No pool size or ranking
constant is selected from this result; PR-03 retains its predeclared latency
budget and generic duplicate-starvation fixtures.

Duplicate chunks from the same file occupied 44.8% of the pre-dedup semantic
and union candidates on average. The protected-language range was 4.5% for
TypeScript through 70.0% for Python. This makes file diversity a required PR-03
guardrail rather than a benchmark-specific heuristic.

## Cross-project diagnostics

The server's debug trace now preserves project-local identity and rank through
global sorting and reranking. It reports per-project candidate share, score
range/mean/standard deviation, concentration (HHI), pairwise score-range
overlap, reranker input/output rank, and final project sequence. Deterministic
two-project fixtures validate this path. The current development labels are
repository-local, so no unsupported cross-project relevance claim is made.

## Artifacts and reproduction

- Machine-readable traces and candidate metrics:
  [`benchmarks/development-candidate-trace-2026-07-14.json`](/benchmarks/development-candidate-trace-2026-07-14.json)
- Paired overhead/equality run:
  [`benchmarks/development-trace-overhead-2026-07-14.json`](/benchmarks/development-trace-overhead-2026-07-14.json)
- Frozen untraced baseline:
  [`benchmarks/development-baseline-2026-07-14.json`](/benchmarks/development-baseline-2026-07-14.json)

```bash
uv run python scripts/benchmark_development.py \
  --tier quick --trace \
  --compare-baseline benchmarks/development-baseline-2026-07-14.json \
  --output benchmarks/development-candidate-trace-2026-07-14.json

uv run python scripts/benchmark_trace_overhead.py \
  --tier quick --repetitions 10 --warmups 2 \
  --output benchmarks/development-trace-overhead-2026-07-14.json
```
