# Retrieval corpus governance

The public corpus manifest is [`v1/manifest.yaml`](v1/manifest.yaml). It is the
source of truth for repository identity, exact revision, license, content
coverage, label provenance, and evaluation split.

## Evaluation classes

- `legacy_regression` is diagnostic only. These cases have already influenced
  hypotheses or implementation and cannot select ranking constants, models, or
  routes.
- `development` is the public, repository-separated working set. It may be used
  to compare predeclared changes, with macro-per-repository results reported
  before pooled query metrics.
- `sealed_holdout` is opened once for a frozen milestone. Its repository
  identities and labels do not live in this repository and are not available to
  implementation agents.

The development corpus covers Python, TypeScript/JavaScript, PHP, Go, Swift,
Ruby, Markdown, reStructuredText, and DocC content. Every public repository is
pinned to a full commit and uses a permissive open-source license.

## Sealed holdout handling

Set `TESSERA_SEALED_HOLDOUT_MANIFEST` only in the isolated evaluation
environment. The referenced file must be private or encrypted, must use a
disjoint set of repositories, and must never be copied below the public
worktree. `benchmarks/corpora/private/` is ignored as a final local safeguard;
the preferred location is outside the checkout entirely.

Before opening the holdout, record the frozen code revision, model/profile,
index format, metric, tolerance, and operational budgets. Run it once without
inspecting intermediate labels. After results are revealed, move that corpus to
`legacy_regression` and commission a new sealed corpus for the next milestone.

## Validation and runs

Schema and label validation is local and fast:

```bash
uv run python scripts/benchmark_governance.py validate
```

Repository validation clones the exact public revisions and verifies every
label against category-compatible files:

```bash
uv run python scripts/benchmark_governance.py validate --verify-repositories
```

Benchmark artifacts must record the manifest digest, split, exact repository
revisions, command, random seed, environment, and whether the run may be used
for selection. Paired comparisons use the same query cases and report the 95%
repository-level paired bootstrap interval for the macro MRR@10 delta.
