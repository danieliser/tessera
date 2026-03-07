# Tessera Validation Test Set

Multi-tier benchmark for evaluating search quality across diverse codebases.

## Target Codebases

| Codebase | Language | Domain | Size | Why |
|----------|----------|--------|------|-----|
| **PM20** (existing) | PHP | WordPress plugin | ~200 files | Known baseline, code + docs |
| **Next.js (vercel/next.js)** | TypeScript | Web framework | ~5000 files | Large TS monorepo, deep imports |
| **Payload CMS** | TypeScript | Headless CMS | ~1500 files | Full-stack Next.js, docs + code |

## Tiers

### Quick (5-10 queries per codebase, ~30s)
Fast A/B comparison between models. "Does model X beat model Y in general?"
- 3 code queries (find function, find class, find pattern)
- 2 doc queries (find concept, find how-to)
- 2 cross queries (code mentioned in docs, doc referenced in code)

### Standard (20-30 queries per codebase, ~2min)
Default for benchmark runs. Covers all query categories with enough samples for stable MRR.
- 10 code queries (symbols, patterns, imports, error handling)
- 10 doc queries (concepts, tutorials, config, API docs)
- 10 cross queries (implementations of documented features)

### Full (50-100 queries per codebase, ~10min)
Deep evaluation with edge cases and adversarial queries.
- All Standard queries plus:
- Fuzzy/misspelled queries
- Natural language paraphrases of code concepts
- Multi-hop queries (find X that uses Y from Z)
- Negative queries (things that don't exist)

## Query Format

```python
# (query_text, expected_files, description, category, difficulty)
QUERIES = [
    (
        "server-side rendering middleware",
        ["middleware.ts", "server.ts", "render.ts"],
        "Find SSR middleware implementation",
        "code",
        "standard",
    ),
]
```

## Running

```bash
# Quick comparison
uv run python scripts/benchmark_validation/run.py --tier quick

# Standard benchmark
uv run python scripts/benchmark_validation/run.py --tier standard

# Full evaluation
uv run python scripts/benchmark_validation/run.py --tier full

# Specific codebase
uv run python scripts/benchmark_validation/run.py --codebase nextjs --tier standard
```
