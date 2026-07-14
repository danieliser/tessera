# Repository-separated development baseline

**Result:** Tessera's frozen BGE-small hybrid baseline reached **0.730 macro
MRR@10**, **83.3% Top-3**, and **96.7% Top-10** across 30 public development
queries in six equally weighted repositories. Warm query latency was 27 ms mean
and 70 ms p95 on the recorded machine.

This is `development` evidence, not a sealed holdout and not a model/routing
selection experiment. It establishes the starting point for predeclared
structural retrieval changes. The labels are visible, so no result here may be
presented as final release generalization evidence.

## Repository results

| Repository | Language | Queries | MRR@10 | Top-1 | Top-3 | Top-10 | Files / chunks | Cold index |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attrs | Python | 5 | 0.867 | 80% | 100% | 100% | 52 / 245 | 27.8 s |
| Hono | TypeScript/JavaScript | 5 | 0.800 | 60% | 100% | 100% | 315 / 356 | 75.7 s |
| pflag | Go | 5 | 0.729 | 60% | 80% | 100% | 79 / 1,576 | 114.7 s |
| swift-argument-parser | Swift | 5 | 0.717 | 60% | 80% | 100% | 75 / 447 | 67.5 s |
| dry-configurable | Ruby | 5 | 0.669 | 60% | 60% | 100% | 16 / 30 | 2.2 s |
| FastRoute | PHP | 5 | 0.600 | 40% | 80% | 80% | 32 / 65 | 6.7 s |
| **Macro** | — | **30** | **0.730** | **60%** | **83.3%** | **96.7%** | **569 / 2,719** | **294.6 s** |

Each repository contributes five queries, so pooled and macro metrics happen to
match in this first version. The harness still computes per-repository values
first; later label expansion cannot silently make a larger repository dominate.

## Protected content segments

| Content type | Queries | MRR@10 | Top-1 | Top-3 | Top-10 |
|---|---:|---:|---:|---:|---:|
| Code | 18 | 0.667 | 50.0% | 83.3% | 94.4% |
| Document | 6 | 0.875 | 83.3% | 83.3% | 100% |
| Mixed code/document | 6 | 0.774 | 66.7% | 83.3% | 100% |

One FastRoute code query missed Top-10. That case is retained as development
evidence, but it does not justify a query-specific synonym, filename boost,
language route, or regression constant. Candidate tracing in PR-02 should first
classify whether the expected file was absent from retrieval channels or lost
during fusion.

## Corpus-scope before/after check

The first dry run revealed that Git sparse-checkout cone mode leaves root files
present even when the manifest declares only source and documentation paths.
That caused a Ruby `.rubocop.yml` containing ERB to be treated as YAML. The
governance layer now emits a deterministic `.tesseraignore` from each manifest
entry and validates labels only inside the declared paths.

The correction removed 38 out-of-scope files and 411 chunks (607 / 3,130 to
569 / 2,719), eliminated the extraction error, and changed macro MRR@10 from
0.658 to 0.730. This is corpus-method correction, not a product ranking gain;
the discarded run is reported only to demonstrate why path isolation is part
of reproducibility.

## Reproduction

The machine-readable artifact binds the run to Tessera commit
`c85a1866571f0d5e8f7d81664c9246d7250a76bb`, manifest SHA-256
`0e84423ee5f116554af104908b0ba845cee25bc69bfecc38cb813e3deab1c701`,
all six repository revisions, the model, platform, command, and random seed.

```bash
uv run python scripts/benchmark_governance.py validate --verify-repositories
uv run python scripts/benchmark_development.py \
  --tier quick --output benchmarks/development-baseline-2026-07-14.json
```

The first command validates all public development and legacy-regression
repositories. The second runs only the development split. Sealed holdout
identities and labels remain outside the checkout.
