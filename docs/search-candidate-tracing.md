# Candidate tracing

Tessera keeps search traces out of normal results. A trace is an observational
debug artifact: it records existing decisions and candidates but must not
alter retrieval or ranking.

## Development benchmark

Add `--trace` to the repository-separated development benchmark:

```bash
uv run python scripts/benchmark_development.py \
  --tier quick --trace --output benchmarks/development-candidate-trace.json
```

Each query then includes `candidate_trace` and `candidate_diagnostics`. The
summary includes macro query-level Recall@10/20/50, duplicate-file rate, and
file diversity for each active source channel and the fused union.

## Server debugging

The MCP server creates and logs a federated trace only when debug logging is
enabled (the CLI's verbose/debug mode). Normal JSON, Markdown, CSV, and file
outputs do not contain traces.

The trace schema is versioned as `1.0` and contains:

- project and chunk identity, file path, and source type;
- raw and post-filter channel rank/score;
- filter, shortcut, graph-expansion, deduplication, and result-limit decisions;
- weighted-RRF contribution and union rank;
- project-local and global rank/score attribution;
- rerank-pool membership and reranker input/output rank;
- final rank or a concrete exclusion reason.

Chunk contents are intentionally excluded. RRF and reranker scores are not
probabilities; cross-project diagnostics report score ranges and source
concentration without claiming probabilistic calibration.
