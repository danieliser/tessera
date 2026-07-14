# Two-stage retrieval experiment — 2026-07-14

PR-03 is not safe to merge. A real expanded rerank pool improved some file-level
quality metrics and recovered the one current Top-10 miss, but neither the
initial full-context treatment nor a principled fixed-total-context remediation
met the predeclared latency gates. The remediation's macro MRR gain was small,
uncertain, and uneven across repositories.

## Frozen inputs

| Item | Value |
|---|---|
| Initial revision | `c36c093883909945d0e5f2418e7664ffacfe8473` |
| Remediation revision | `e8dbc0a414da6dd9857a693a746b644871a08818` |
| Corpus manifest SHA-256 | `0e84423ee5f116554af104908b0ba845cee25bc69bfecc38cb813e3deab1c701` |
| Repositories | attrs, Hono, FastRoute, pflag, swift-argument-parser, dry-configurable |
| Languages | Python, TypeScript, PHP, Go, Swift, Ruby |
| Content | code, document/Markdown, mixed |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Reranker | `Xenova/ms-marco-MiniLM-L-6-v2` |
| Queries | 30 development queries; no holdout access |

The experiment did not sweep pool sizes, context limits, queries, or models.
The pool multiplier of three was inherited from the existing server's nominal
pool. The raw over-fetch factor of five reused the existing unique-file
over-fetch policy.

## Final remediation result

| Engine | Macro MRR@10 | Top-1 | Top-10 | p95 | Mean pool files | Duplicate-file rate |
|---|---:|---:|---:|---:|---:|---:|
| Current page/content | 0.750000 | 60.0% | 96.7% | 2,824.68 ms | 6.17 | 27.2% |
| Expanded score-order/content | 0.790000 | 66.7% | 96.7% | 11,988.71 ms | 14.23 | 43.5% |
| Expanded coverage-first/content | 0.719762 | 56.7% | 100% | 13,418.29 ms | 21.77 | 18.4% |
| Two-stage structured/remediated | 0.758095 | 66.7% | 100% | 5,824.85 ms | 21.77 | 18.4% |

The final treatment's macro MRR delta was `+0.008095`; its repository-paired
95% bootstrap interval was `[-0.157381, +0.132778]`. This is non-negative at
the point estimate but not persuasive evidence of a general quality gain.

| Repository | Current MRR | Treatment MRR | Delta |
|---|---:|---:|---:|
| attrs | 0.866667 | 1.000000 | +0.133333 |
| Hono | 0.900000 | 0.545238 | -0.354762 |
| FastRoute | 0.600000 | 0.750000 | +0.150000 |
| pflag | 0.650000 | 0.833333 | +0.183333 |
| swift-argument-parser | 0.766667 | 0.740000 | -0.026667 |
| dry-configurable | 0.716667 | 0.680000 | -0.036667 |

Code MRR improved from 0.625000 to 0.688889 and document MRR remained 1.0,
but the six mixed code/document queries fell from 0.875000 to 0.723810. With
five cases per repository and six per content guardrail, these movements are
diagnostic rather than stable estimates; Hono's regression is nevertheless too
large to dismiss.

## Initial result and remediation

The initial structured treatment capped each of 30 candidates at 4,096
characters. It increased macro MRR from 0.750000 to 0.801190 and Top-1 from
60.0% to 73.3%, but p95 was 8,093.41 ms against 3,406.24 ms current. It failed
the 500 ms absolute and +300 ms added-p95 gates.

That result was rejected rather than used to loosen the gates. The remediation
kept the incumbent visible page's total model-context budget fixed: one
512-token window was approximated as 2,048 characters, and the total
`visible_limit * 2,048` characters was divided across the actual pool. The
default 10-visible/30-candidate path therefore received 682 characters per
candidate. The model, pool, raw retrieval, queries, and gates were unchanged.

A label-free warm synthetic check isolated the intended mechanism:

| Generic reranker input | Median | Maximum over 5 measured runs |
|---|---:|---:|
| 10 × 4,096 characters | 1,544.71 ms | 1,765.00 ms |
| 30 × 4,096 characters | 4,013.84 ms | 4,606.32 ms |
| 30 × 1,365 characters | 574.93 ms | 590.54 ms |
| 30 × 682 characters | 351.20 ms | 444.04 ms |
| 30 × 341 characters | 148.98 ms | 173.65 ms |

The exact remediation corpus run occurred while the host was heavily
contended: another local model held roughly 31% of RAM, swap reached about
30.7 GiB, and unrelated CPU-heavy processes were active. Its absolute latency
is therefore a conservative observation, but the experiment still fails its
declared wall-clock gate. No post-result context reduction is adopted to chase
the public corpus.

## Deterministic evidence

The old server failed the promotion fixture because `hybrid_search` received
only the visible `limit=2`; candidate three never reached the reranker. The new
fixture requests the declared raw depth, limits the actual cross-encoder pool,
and promotes candidate three to rank one. Separate invariants cover:

- unchanged `limit` and ordering when reranking is disabled;
- bounded pool and raw-fetch budgets, including visible limits above the cap;
- coverage-first ordering across projects/files and safe unknown-path identity;
- second-chunk backfill after each available file receives one candidate;
- structured path, source, line, section/key, and indexed-symbol context;
- malformed symbol metadata and dynamic total-context bounds;
- trace attribution for requested pool, raw retrieval depth, selection policy,
  and actual document cap.

## Decision and follow-ups

Do not merge the production change from this experiment. Three findings remain
useful:

1. The nominal server pool was indeed not real; the deterministic defect and
   FastRoute recovery validate the candidate-eligibility hypothesis.
2. Hard coverage-first selection is too blunt. It reduces duplicate crowding
   but can lower MRR; file diversity should be relevance-aware or implemented
   in the hierarchical file-to-chunk stage planned for PR-06.
3. Cross-encoder throughput is the binding constraint. A separate model/cascade
   experiment should compare supported rerankers under a fixed total token and
   latency budget, without coupling model selection to this retrieval PR.

The sealed holdout remains unopened. A future replacement for PR-03 needs a new
predeclared experiment and independent validation before merge.

## Artifacts and reproduction

- Initial failed run:
  [`benchmarks/development-two-stage-initial-2026-07-14.json`](/benchmarks/development-two-stage-initial-2026-07-14.json)
- Context-conserving remediation:
  [`benchmarks/development-two-stage-2026-07-14.json`](/benchmarks/development-two-stage-2026-07-14.json)

```bash
uv run pytest -q \
  tests/unit/test_rerank.py \
  tests/unit/test_search_trace.py \
  tests/integration/test_server.py

uv run python scripts/benchmark_two_stage.py \
  --tier quick \
  --output benchmarks/development-two-stage-2026-07-14.json
```
