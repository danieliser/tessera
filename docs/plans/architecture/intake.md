# CodeMem — Intake & Scoping

**Date**: 2026-02-21
**Tier**: Full (with `--red-team`)
**Project type**: Infrastructure
**Plan directory**: `docs/plans/architecture/`

## Idea

Build **CodeMem**: a hierarchical, scope-gated codebase indexing and persistent memory system for always-on AI agents. Exposed as an MCP server. Supports multi-language codebases (PHP, TypeScript, Python, Swift) plus non-code documents (PDF, Markdown). Provides project/collection/global scope tiers with capability-based access control.

## The Right Question

Not "how do we build a code indexer" — but **"how do we give AI agents persistent, scoped memory about codebases that scales from one project to dozens of repos across multiple ecosystems?"**

The real decision: What's the thinnest viable system that provides hierarchical codebase memory without becoming a framework we're stuck maintaining?

### Adjacent questions

1. **Build order vs. value curve** — Which phase delivers the first "wow" moment? If Phase 1 alone isn't useful, the project risks dying before Phase 3.
2. **Embedding model lock-in** — "Local model endpoint" sounds cheap, but model changes stale all embeddings. What's the re-index cost at 50 repos?
3. **MCP session lifecycle** — Scope is locked per session, but who starts the server? What happens when the persistence agent dies mid-session?

### Wrong questions to avoid

- "Should we use Neo4j vs SQLite?" — Already validated by research. Don't relitigate.
- "What's the best embedding model?" — It's swappable. Pick one and move.

### Hidden assumptions to validate

1. ~2-3K lines of glue is realistic (every thin wrapper grows 5-10x with error handling, migrations, edge cases)
2. SQLite adjacency queries perform at collection scale (10+ repos, thousands of hook references)
3. Search-time federation doesn't add noticeable latency vs. unified index
4. Tree-sitter's inability to resolve dynamic dispatch (PHP/Python/JS) won't materially degrade cross-reference quality

## Premise Validation

**Status: VALID** (validated 2026-02-21)

No existing tool covers >40% of requirements:
- **CodeRLM** (Feb 2026) — Closest philosophy (tree-sitter, lightweight, agent-driven) but session-scoped, no persistence, no vectors, no graph, no hierarchy
- **Octocode** — Has "knowledge graphs" but single-project, no scope gating, no documents
- **tree-sitter-mcp servers** — Structural queries only, explicitly "don't create durable index"
- **Code Pathfinder** — Python-only, call graphs only
- Prior landscape (CodeGraphContext, Continue.dev, Sourcegraph, Greptile, Zilliz, Khoj) — unchanged, each ≤40%

The gap between "tree-sitter symbol lookup" and "persistent hierarchical codebase memory with scope gating" remains wide. No new entrant bridges it.

## Constraints

- **No external servers/daemons**: Fully embedded (SQLite, LanceDB). No Neo4j/Milvus/Postgres.
- **Local model endpoint available**: OpenAI-compatible API for embeddings — cost is zero.
- **Multi-language from day 1**: PHP, TypeScript, Python, Swift minimum.
- **MCP interface**: Works with Claude Code, Cursor, any MCP client.
- **Thin glue layer**: ~2-3K lines orchestrating proven libraries.
- **Open-sourceable**: Core system separable from proprietary integrations.

## Success Criteria

1. Task agent scoped to one project can search, navigate symbols, trace references — within that project only
2. Collection-scoped agent can trace cross-project references (hooks, imports, dependencies)
3. Persistence agent can register projects, trigger indexing, assign scopes, query globally
4. Indexing a medium PHP/TS project completes in under 60 seconds
5. Incremental re-indexing (git-diff-based) completes in under 5 seconds
6. Hybrid search quality matches or exceeds Continue.dev's indexer
7. Non-code documents (PDF, Markdown) searchable within same interface

## Non-Goals

- **Not an IDE plugin**: MCP is the interface. No VS Code extension.
- **Not static analysis**: No linting, type checking. That's LSP's job.
- **Not an LLM framework**: No prompt construction, agent orchestration. Just memory.
- **Not a git replacement**: No version control, diffing, merge resolution.
- **Not SCIP/LSP precision**: Tree-sitter gives syntax, not semantics. Accepted trade-off.

## Red Team Focus

All 7 risks from BRIEFING.md, with extra pressure on:
1. **SQLite-only graph at scale** — adjacency table performance at collection size
2. **Scope token security** — can agents fabricate global scope?
3. **"2-3K lines" realism** — thin glue → inevitable framework growth?

Plus:
4. Federation latency overhead
5. Tree-sitter type resolution limits in dynamic languages
6. Embedding model drift / re-index cost
7. Cross-language references (PHP REST API → TypeScript)
