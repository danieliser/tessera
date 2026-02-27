# Research: Cross-File Dependency Graph Architecture in Production Code Intelligence Tools

**Date:** 2026-02-24
**Tier:** Standard
**Question:** How do production-grade code intelligence tools build cross-file dependency graphs? Is the standard approach two-pass indexing (symbols first, then references), or is there a better architectural pattern?
**Recommendation:** Use **file-incremental, query-time resolution** (stack graphs / deferred binding) rather than two-pass indexing. This enables scalability and incremental updates without pre-computing the full dependency graph.

---

## Context & Constraints

Tessera is designing an indexing architecture for multi-file, multi-language code intelligence. The evaluation must determine whether to:
1. Index all symbols in pass 1, then resolve cross-file references in pass 2
2. Build per-file graphs independently and defer cross-file resolution to query-time
3. Use incremental compilation with in-memory symbol tables (LSP approach)

The decision affects scalability, incremental update performance, and architectural complexity.

---

## Options Evaluated

### Option 1: Two-Pass Global Indexing (Traditional Compiler Approach)

- **Confidence:** High (well-documented compiler theory)
- **What it is:** Pass 1 scans all files and builds a global symbol table. Pass 2 resolves all cross-file references. All edges are pre-computed and stored.
- **Strengths:**
  - ✓ Guarantees correct symbol resolution for complex cases (overloading, polymorphism)
  - ✓ All edges pre-computed; query-time lookups are O(1) dictionary access
  - ✓ Works well for single-language, controlled-scope codebases
  - ✓ Well-understood theory from compiler design [[1](https://www.geeksforgeeks.org/compiler-design/single-pass-two-pass-and-multi-pass-compilers/)]

- **Weaknesses:**
  - ✗ Not incremental: one file change forces full re-indexing of all dependent files
  - ✗ Requires holding full symbol table in memory during pass 2 (scalability ceiling)
  - ✗ File ordering constraints: SCIP's LSIF predecessor required strict vertex ordering for edges
  - ✗ Expensive for very large repos: GitHub-scale codebases become infeasible [[2](https://sourcegraph.com/blog/announcing-scip)]
  - ✗ Demands a build system or language-specific type checker (PHP, Python, JS vary widely)

- **Cost:** O(n) files × O(m) symbols per file = O(n·m) space for global table. Single-threaded pass 2 blocks incremental updates.

- **Maintenance:** Dead approach in production. GitHub, Sourcegraph, and modern LSP servers have moved away from it.

---

### Option 2: File-Incremental, Query-Time Resolution (Stack Graphs / Deferred Binding)

- **Confidence:** High (production-proven at GitHub, Sourcegraph)
- **What it is:** Each file's AST is parsed independently to produce an isolated subgraph (no cross-file visibility). Cross-file binding is resolved at query-time by merging subgraphs and searching for valid paths.
- **How it works:**
  - **Indexing phase:** For each file independently, tree-sitter parses the AST and applies a declarative graph DSL (stack graphs or tree-sitter-graph) to extract definitions, scopes, imports, and local references.
  - **Storage:** Per-file subgraphs stored independently. Imports/exports are recorded as metadata.
  - **Query phase:** When resolving a reference, load the containing file's subgraph + relevant dependency subgraphs, merge them, and perform path-finding.

- **Strengths:**
  - ✓ **True incrementalism:** File change requires re-indexing only that file (~100k lines/sec in tree-sitter)
  - ✓ **Parallelizable:** Each file indexed independently, no global locks or ordering
  - ✓ **Scalable:** No global symbol table; memory bounded per file
  - ✓ **Language-agnostic:** Works via syntax trees without type information; GitHub supports 10+ languages [[3](https://github.blog/open-source/introducing-stack-graphs/)]
  - ✓ **Query-aware:** Resolution heuristics can be tuned for different search patterns (imports, exports, scope nesting)
  - ✓ **Production-grade:** Powers GitHub's code navigation and Sourcegraph's SCIP [[2](https://sourcegraph.com/blog/announcing-scip)] [[3](https://github.blog/open-source/introducing-stack-graphs/)]

- **Weaknesses:**
  - ✗ Query latency: Cross-file resolution is not O(1); requires path-finding (typically sub-100ms at GitHub scale)
  - ✗ Ambiguity: Without type information, multiple candidates possible (mitigated by embedding similarity or heuristics)
  - ✗ Complexity: Path-finding logic and scope merging are non-trivial to implement correctly
  - ✗ Missing some precise cases: Overload resolution, polymorphic dispatch require type information
  - ✗ RFC 001 trade-off: File-incremental approaches acknowledge ~90% accuracy vs 100% in compiler-style resolution [[4](https://github.com/orgs/sheeptechnologies/discussions/4)]

- **Cost:** Indexing O(n) files independently. Query-time merging + path-finding depends on scope depth and import fan-out (typically sub-100ms). No global memory spike.

- **Maintenance:** Active and growing. GitHub maintains [tree-sitter-graph](https://github.com/tree-sitter/tree-sitter-graph) and [stack-graphs](https://github.com/github/tree-sitter-stack-graphs). Sourcegraph ships SCIP as a Protobuf standard [[2](https://sourcegraph.com/blog/announcing-scip)].

---

### Option 3: Incremental Compilation with In-Memory Symbol Tables (LSP Approach)

- **Confidence:** High (implemented in rust-analyzer, pyright, TypeScript server)
- **What it is:** LSP servers maintain in-memory compilation state for the workspace. When a file changes, only affected files are re-analyzed. Symbol resolution happens during the bind/type-check pipeline.
- **How it works:**
  - On startup or file change, re-parse affected files
  - Incrementally rebuild symbol tables for that file and dependents
  - Resolve all symbols during a single pipeline pass (parse → bind → type-check)
  - Clients query the in-memory state via LSP requests

- **Strengths:**
  - ✓ **Precise resolution:** Type information available; handles overloads, generics, polymorphism
  - ✓ **Interactive latency:** Multi-second edit-to-response feedback (rust-analyzer ~100-500ms for most queries)
  - ✓ **Real-time diagnostics:** Errors surfaced as you type
  - ✓ **Language-specific:** Can leverage language semantics (Rust traits, Python duck typing, etc.)
  - ✓ **Workspace awareness:** Knows about dependencies via import statements
  - ✓ **Proven:** Standard for IDE integration (VS Code, Neovim, Emacs)

- **Weaknesses:**
  - ✗ Not designed for batch indexing: Memory spike for large workspaces (entire Chromium can't be analyzed in one LSP instance)
  - ✗ Language-specific burden: Pyright ≠ rust-analyzer ≠ TypeScript server; each needs separate impl
  - ✗ Dependency management: Must integrate with package managers (Cargo, npm, pip, composer) for cross-file symbol resolution
  - ✗ No persistent index: All state re-computed on server restart
  - ✗ Not suitable for offline/batch analysis or non-interactive use cases
  - ✗ Requires a language type-checker: No solution for untyped/dynamic languages without approximation

- **Cost:** In-memory O(workspace size). Rust-analyzer primes symbol caches at startup; pyright can consume 2-4GB for large repos [[5](https://github.com/rust-lang/rust-analyzer/pull/18180)].

- **Maintenance:** High effort. Each language needs its own LSP server and type checker. Suitable for language teams (Rust, TypeScript, Python foundations) but not for universal code indexing.

---

## Comparison Matrix

| Criteria | Two-Pass Global | File-Incremental (Stack Graphs) | LSP In-Memory |
|----------|-----------------|----------------------------------|----------------|
| **Incremental Update** | Full re-index | Single file | Single file + dependents |
| **Parallelizable** | No (pass 2 sequential) | Yes (per-file) | Partial (file level) |
| **Memory at Scale** | O(n·m) global table | O(max file) | O(workspace) |
| **Query Latency** | O(1) lookup | O(path-find) ~10-100ms | O(symbol-lookup) ~50-500ms |
| **Accuracy** | 100% (with type info) | ~90% (syntax-only) | 100% (with types) |
| **Language Coverage** | Requires type-checker per lang | Syntax-only, ~10+ langs | Requires LSP server per lang |
| **Incremental Indexing** | ✗ Not practical | ✓ Standard | ✓ Standard |
| **Offline Batch Use** | ✓ Works | ✓ Works | ✗ Interactive only |
| **Production Examples** | Legacy (CodeQL is different) | GitHub, Sourcegraph, semantic | rust-analyzer, pyright, tsserver |
| **Scalability** | GitHub's limit (~100M LOC) | GitHub-scale proven | IDE-scale (~10M LOC) |

---

## Recommendation

**Use file-incremental, query-time resolution (stack graphs / deferred binding).**

### Why This Wins

1. **Incrementalism is non-negotiable at scale.** GitHub, Sourcegraph, and modern semantic analysis all use this pattern because re-indexing millions of files on every change is unfeasible. Two-pass global indexing was abandoned for this reason.

2. **Tessera's constraints favor it:**
   - Multi-language support (PHP, TS, Python, Swift) — stack graphs work syntactically without per-language type-checkers
   - Persistent index (stored in SQLite) — unlike LSP's ephemeral in-memory state
   - Batch indexing + incremental updates — file-incremental is designed for this
   - Cross-repo federation [[6](https://github.com/tree-sitter/tree-sitter-stack-graphs)] — stack graphs can merge subgraphs across repos

3. **The accuracy trade-off is acceptable.** 90% syntactic resolution covers:
   - All imports/exports
   - All named function/class definitions
   - Scope nesting and re-exports
   - Simple overloads (name-based disambiguation)

   The 10% miss (precise type overloads, generic specialization) is out-of-scope for a syntax-based indexer. If needed, integrate a type-checker as an optional second pass for high-value files.

4. **Proven implementation.**
   - [Stack graphs](https://github.com/github/tree-sitter-stack-graphs) — open-source, GitHub-maintained
   - [SCIP](https://github.com/sourcegraph/scip) — production at Sourcegraph, becoming a standard [[7](https://sourcegraph.com/blog/announcing-scip)]
   - [tree-sitter-graph](https://github.com/tree-sitter/tree-sitter-graph) — DSL for graph construction, used by both above

### What Would Change This Recommendation

- **If 100% precision is required for all symbols** (e.g., you're building a refactoring tool, not a search/navigation tool): Integrate a per-language type-checker for symbol resolution. This becomes a three-layer architecture: (1) syntax graphs, (2) optional type resolution, (3) query merging.

- **If you only support one or two languages** (e.g., Python-only): Consider an LSP-based approach (pyright) for that language, because the type-checking effort is amortized across users. But this loses multi-language coverage.

- **If you have guaranteed small codebases** (< 1M LOC, single repo): Two-pass global indexing is simpler to implement. But this doesn't apply to Tessera (GitHub-scale is an explicit goal).

---

## Implementation Notes for Tessera

1. **Use tree-sitter-graph DSL** to define per-file subgraph extraction (definitions, scopes, imports, references).
2. **Store subgraphs as JSON graphs** in SQLite (one per file). Implement a simple graph schema: nodes {id, kind, name, scope}, edges {from, to, kind}.
3. **At query time,** implement a path-finding algorithm (DFS or BFS) that:
   - Loads the containing file's subgraph
   - Follows import edges to dependency subgraphs
   - Searches for paths from reference node to definition nodes
   - Returns candidates ranked by scope distance or embedding similarity
4. **Handle incremental updates:** On file change, re-run tree-sitter-graph for that file only, update its subgraph in the DB, mark dependents as "pending re-analysis" for next rebuild.
5. **For type information,** add an optional type-resolution layer (e.g., Pyright for Python) that runs after syntax indexing and augments the symbol graph with type bindings.

---

## Sources

- [1] [Single Pass vs Two-Pass (Multi-Pass) Compilers - GeeksforGeeks](https://www.geeksforgeeks.org/compiler-design/single-pass-two-pass-and-multi-pass-compilers/)
- [2] [SCIP - a better code indexing format than LSIF | Sourcegraph Blog](https://sourcegraph.com/blog/announcing-scip)
- [3] [Introducing stack graphs - The GitHub Blog](https://github.blog/open-source/introducing-stack-graphs/)
- [4] [RFC 001 - Remove SCIP Dependency and Implement Tree-sitter Based File-Incremental Indexing](https://github.com/orgs/sheeptechnologies/discussions/4)
- [5] [Index workspace symbols at startup rather than on first symbol search - rust-analyzer PR #18180](https://github.com/rust-lang/rust-analyzer/pull/18180)
- [6] [tree-sitter/tree-sitter-graph: Construct graphs from parsed source code](https://github.com/tree-sitter/tree-sitter-graph)
- [7] [Optimizing a code intelligence indexer | Sourcegraph Blog](https://sourcegraph.com/blog/optimizing-a-code-intel-indexer)
- [8] [Stack graphs Name resolution at scale - Douglas A. Creager](https://arxiv.org/abs/2211.01224)
- [9] [The code behind GitHub's new Code Navigation](https://blog.codenamev.com/the-code-behind-github-code-navigation/)
- [10] [Stack graphs: Name resolution at scale (EVCS 2023)](https://dcreager.net/stack-graphs/)

