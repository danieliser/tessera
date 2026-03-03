# Multi-Project Federation in Tessera

Tessera is built for managing multiple related projects — WordPress plugin ecosystems, microservices, shared library + consumer projects, or any codebase landscape where you need to search, analyze, and understand dependencies across project boundaries.

This guide walks you through registering multiple projects, grouping them into collections, and running searches and cross-project analyses across all of them.

## Concepts: Projects, Collections, and Federation

**Projects** are individual codebases you want to index. Each project gets its own SQLite database and FAISS vector index at `~/.tessera/data/{project-slug}/index.db`.

**Collections** are logical groupings of projects. Use them to organize related projects — e.g., a WordPress plugin ecosystem (core plugin + add-ons + shared utilities), or a backend with multiple microservices.

**Federation** is how Tessera handles multi-project queries. When you search or analyze across projects:
1. Each project DB is queried in parallel (via `asyncio.gather`)
2. Results are merged and sorted by relevance score
3. Each result includes `project_id` and `project_name` so you know where it came from
4. No data is duplicated — everything stays at project level

This "search-time federation" keeps your data clean and queries fast.

## Quick Reference: Tool Signatures

| Tool | Purpose | Scope |
|------|---------|-------|
| `register_project(path, name, language="", collection_id=0)` | Register a project | Global |
| `create_collection_tool(name, project_ids=[])` | Create a collection | Global |
| `add_to_collection_tool(collection_id, project_id)` | Add a project to collection | Global |
| `list_collections_tool()` | List all collections | Global |
| `delete_collection_tool(collection_id)` | Delete a collection (doesn't delete projects) | Global |
| `search(query, limit=10, ...)` | Hybrid search across accessible projects | Project |
| `cross_refs(symbol_name)` | Find where symbols cross projects | Project |
| `collection_map(collection_id=0)` | Dependency overview of projects | Project |

All administrative tools require global scope access. See [Security & Access Control](./security.md) for session tokens and scoped agent access.

---

## Registering Projects

Projects must be registered before they can be indexed and searched. Registration happens once per project.

### Basic Registration

```python
# From MCP client, call the register_project tool
result = client.call_tool("register_project", {
    "path": "/path/to/my-project",
    "name": "my-project"
})
# Returns: {"id": 1, "name": "my-project", "path": "/path/to/my-project", ...}
```

The tool returns project metadata including the assigned `id`. You'll use this `id` later when adding projects to collections.

**Parameters:**
- `path` (required): Absolute path to the project root
- `name` (required): Human-readable name (used in search results and logs)
- `language` (optional): Programming language hint — `php`, `typescript`, `python`, `swift`. If omitted, auto-detected during indexing
- `collection_id` (optional): Assign to a collection at registration time (or add later with `add_to_collection_tool`)

### Example: Register Four Related Projects

Imagine a WordPress plugin ecosystem: a core plugin, a pro add-on, an ecommerce extension, and a shared utilities package.

```python
# Core plugin
result_a = client.call_tool("register_project", {
    "path": "/Users/dev/plugins/core-plugin",
    "name": "core-plugin",
    "language": "php"
})
project_a_id = result_a["id"]  # e.g., 1

# Pro add-on
result_b = client.call_tool("register_project", {
    "path": "/Users/dev/plugins/pro-addon",
    "name": "pro-addon",
    "language": "php"
})
project_b_id = result_b["id"]  # e.g., 2

# Ecommerce add-on
result_c = client.call_tool("register_project", {
    "path": "/Users/dev/plugins/ecommerce-addon",
    "name": "ecommerce-addon",
    "language": "php"
})
project_c_id = result_c["id"]  # e.g., 3

# Shared utilities
result_d = client.call_tool("register_project", {
    "path": "/Users/dev/plugins/shared-utils",
    "name": "shared-utils",
    "language": "php"
})
project_d_id = result_d["id"]  # e.g., 4
```

Each project is now registered and ready to be indexed. The registration doesn't trigger indexing — you can index on-demand via the `reindex` tool.

### Indexing After Registration

After registration, trigger the first index:

```python
result = client.call_tool("reindex", {
    "project_id": project_a_id,
    "mode": "full"
})
# Returns: {"files_processed": 120, "symbols_extracted": 847, "time_elapsed": 2.34, ...}
```

Index all four projects. You can do this in parallel by queuing all four `reindex` calls at once.

---

## Collections: Grouping Projects

Collections organize related projects and make multi-project queries cleaner. You can group projects at registration time or add them later.

### Create a Collection

```python
result = client.call_tool("create_collection_tool", {
    "name": "plugin-ecosystem",
    "project_ids": [project_a_id, project_b_id, project_c_id, project_d_id]
})
# Returns: {"id": 1, "name": "plugin-ecosystem", "projects": [1, 2, 3, 4]}
collection_id = result["id"]
```

### Add a Project to a Collection Later

```python
# If you register a new add-on after creating the collection:
result = client.call_tool("register_project", {
    "path": "/Users/dev/plugins/new-addon",
    "name": "new-addon",
    "language": "php"
})
new_project_id = result["id"]

# Add it to the existing collection
client.call_tool("add_to_collection_tool", {
    "collection_id": collection_id,
    "project_id": new_project_id
})
```

### List Collections

```python
result = client.call_tool("list_collections_tool", {})
# Returns: {"collections": [{"id": 1, "name": "plugin-ecosystem", "projects": [1, 2, 3, 4, 5]}]}
```

### Delete a Collection

```python
client.call_tool("delete_collection_tool", {
    "collection_id": collection_id
})
# Returns: {"deleted": true, "collection_id": 1}
```

**Important:** Deleting a collection removes the grouping but does NOT delete the projects. The projects remain registered and indexed.

---

## Searching Across Projects

When you run a search, Tessera queries all accessible projects in parallel and merges results.

### Basic Multi-Project Search

```python
result = client.call_tool("search", {
    "query": "filter payment hooks",
    "limit": 20
})

# Results include project context
# [
#   {"file": "hooks/payment.php", "project_id": 1, "project_name": "core-plugin", "score": 0.92, ...},
#   {"file": "hooks/filter.php", "project_id": 3, "project_name": "ecommerce-addon", "score": 0.87, ...},
#   {"file": "utils/hooks.php", "project_id": 4, "project_name": "shared-utils", "score": 0.81, ...},
# ]
```

Each result includes:
- `project_id`: The project this result came from
- `project_name`: Human-readable project name
- `score`: Relevance score (higher = more relevant)
- `file`: File path relative to project root
- `line`: Line number
- `snippet`: Code or text context

Results are sorted by score descending across all projects. The search respects both keyword (FTS5) and semantic (embeddings) signals.

### Syntax Variants

Tessera supports several search modes. Use them the same way across multi-project queries:

**Lexical search (fast, exact identifier matching):**
```python
result = client.call_tool("search", {
    "query": "lex:apply_filters",  # Prefix with "lex:" for FTS5 only
    "limit": 10
})
```

**Semantic search (conceptual, slower but finds related code):**
```python
result = client.call_tool("search", {
    "query": "vec:caching strategy with TTL",
    "limit": 10
})
```

**Document/config search (non-code only):**
```python
result = client.call_tool("doc_search_tool", {
    "query": "API authentication endpoints",
    "formats": "markdown,yaml",
    "limit": 10
})
```

See [Search Guide](./search-guide.md) for full syntax, operators, and weight customization.

---

## Cross-Project References: Finding Dependencies

Use `cross_refs` to find where a symbol (function, class, constant) defined in one project is used in another.

### Basic Usage

```python
result = client.call_tool("cross_refs", {
    "symbol_name": "apply_filters"  # WordPress hook function
})
```

**Returns:**
```json
{
  "symbol": "apply_filters",
  "definition_projects": {
    "1": {"id": 1, "name": "core-plugin"}
  },
  "cross_refs": [
    {
      "from_project_id": 2,
      "from_project_name": "pro-addon",
      "to_project_id": 1,
      "to_project_name": "core-plugin",
      "file": "hooks/main.php",
      "line": 42,
      "kind": "function"
    },
    {
      "from_project_id": 3,
      "from_project_name": "ecommerce-addon",
      "to_project_id": 1,
      "to_project_name": "core-plugin",
      "file": "filters/products.php",
      "line": 18,
      "kind": "function"
    }
  ]
}
```

**Interpreting the result:**
- `definition_projects`: Projects that define `apply_filters`
- `cross_refs`: Each reference shows which project called it, from where, and at what line
- An empty `cross_refs` array means the symbol is only used within its defining project (no cross-project dependencies)

### Use Cases

**Identify reverse dependencies before refactoring:**
```python
# Before changing shared utility, see who depends on it
result = client.call_tool("cross_refs", {
    "symbol_name": "format_price"
})

if result["cross_refs"]:
    print(f"Changing format_price will affect {len(result['cross_refs'])} call sites across projects")
    for ref in result["cross_refs"]:
        print(f"  - {ref['from_project_name']}: {ref['file']}:{ref['line']}")
```

**Check if a shared utility is used at all:**
```python
result = client.call_tool("cross_refs", {
    "symbol_name": "deprecated_util"
})

if not result["cross_refs"]:
    print("Safe to delete — no cross-project dependencies")
```

---

## Collection Map: Dependency Overview

Use `collection_map` to get a bird's-eye view of all projects in a collection and how they depend on each other.

### Basic Usage

```python
result = client.call_tool("collection_map", {
    "collection_id": 1  # or 0 for all accessible projects
})
```

**Returns:**
```json
{
  "collection_id": 1,
  "projects": {
    "core-plugin": {"id": 1, "symbol_count": 847},
    "pro-addon": {"id": 2, "symbol_count": 234},
    "ecommerce-addon": {"id": 3, "symbol_count": 456},
    "shared-utils": {"id": 4, "symbol_count": 89}
  },
  "edges": [
    {
      "from": "pro-addon",
      "to": "core-plugin",
      "cross_refs": 12,
      "symbols": ["apply_filters", "do_action", "get_option"]
    },
    {
      "from": "ecommerce-addon",
      "to": "core-plugin",
      "cross_refs": 8,
      "symbols": ["apply_filters", "register_post_type"]
    },
    {
      "from": "ecommerce-addon",
      "to": "shared-utils",
      "cross_refs": 5,
      "symbols": ["format_price", "validate_license"]
    }
  ]
}
```

**Interpreting the result:**
- `projects`: Each project and its symbol count (a proxy for size)
- `edges`: Cross-project references. Each edge shows:
  - `from` → `to`: Direction of dependency
  - `cross_refs`: Number of references
  - `symbols`: Which symbols are being called

### Use Cases

**Visualize ecosystem structure:**
```python
result = client.call_tool("collection_map", {"collection_id": 1})

for edge in result["edges"]:
    print(f"{edge['from']} → {edge['to']}: {edge['cross_refs']} refs")
    print(f"  Symbols used: {', '.join(edge['symbols'][:3])}...")
```

**Find circular dependencies:**
```python
edges_by_direction = {}
for edge in result["edges"]:
    key = (edge["from"], edge["to"])
    edges_by_direction[key] = edge["cross_refs"]

# Check for reverse edges (A → B and B → A)
for (a, b), count_ab in edges_by_direction.items():
    if (b, a) in edges_by_direction:
        count_ba = edges_by_direction[(b, a)]
        print(f"Circular dependency: {a} ↔ {b}")
        print(f"  {a} → {b}: {count_ab} refs")
        print(f"  {b} → {a}: {count_ba} refs")
```

**Identify high-value shared utilities:**
```python
result = client.call_tool("collection_map", {"collection_id": 1})

# Count how many projects import from each project
import_counts = {}
for edge in result["edges"]:
    to_proj = edge["to"]
    import_counts[to_proj] = import_counts.get(to_proj, 0) + 1

# Sort by dependency count
sorted_deps = sorted(import_counts.items(), key=lambda x: -x[1])
for proj, count in sorted_deps:
    print(f"{proj} is imported by {count} other projects")
```

---

## Scoped Access: Granting Multi-Project Access to Sub-Agents

The federation tools are designed for orchestrator agents that manage many projects. If you want to delegate work to sub-agents, you can create **scoped session tokens** that grant access to only specific projects or collections.

### Grant Access to a Specific Collection

```python
# Orchestrator creates a scoped token for a sub-agent
# that can only see projects in the "plugin-ecosystem" collection

result = client.call_tool("create_scope_tool", {
    "agent_id": "sub-agent-1",
    "scope_level": "collection",
    "collection_ids": [1]  # Only the "plugin-ecosystem" collection
})
# Returns: {"session_id": "abc123...", "valid_until": "2026-03-10T...", ...}
```

Pass the token to the sub-agent via the `TESSERA_SESSION_ID` environment variable:

```bash
# Spawn a sub-agent with scoped access to the collection
TESSERA_SESSION_ID=abc123... claude "analyze plugin compatibility"
```

Or bake it into an MCP config file (see [Passing Tokens to Sub-Agents](security.md#passing-tokens-to-sub-agents)).

The sub-agent can now search and analyze across all projects in that collection, but cannot see projects outside it.

### Grant Access to Specific Projects

```python
# Grant access to only the core plugin and shared utilities
result = client.call_tool("create_scope_tool", {
    "agent_id": "code-cleanup-bot",
    "scope_level": "project",
    "project_ids": [1, 4]  # core-plugin and shared-utils
})
```

```bash
TESSERA_SESSION_ID=<returned_session_id> claude "clean up dead code"
```

The sub-agent can now search and analyze only these two projects.

See [Security & Access Control](security.md) for full details on scope tokens, environment variables, and session management.

---

## Real Example: WordPress Plugin Ecosystem

Here's a complete walkthrough of registering, organizing, and analyzing a WordPress plugin ecosystem.

### Setup: Register Four Projects

```python
# Assume client is initialized with Tessera MCP server

# 1. Register core plugin
core = client.call_tool("register_project", {
    "path": "/path/to/plugins/my-plugin",
    "name": "my-plugin",
    "language": "php"
})
core_id = core["id"]

# 2. Register pro add-on
pro = client.call_tool("register_project", {
    "path": "/path/to/plugins/my-plugin-pro",
    "name": "my-plugin-pro",
    "language": "php"
})
pro_id = pro["id"]

# 3. Register ecommerce add-on
ecom = client.call_tool("register_project", {
    "path": "/path/to/plugins/my-plugin-ecom",
    "name": "my-plugin-ecom",
    "language": "php"
})
ecom_id = ecom["id"]

# 4. Register shared utilities (lightweight package both depend on)
utils = client.call_tool("register_project", {
    "path": "/path/to/plugins/my-plugin-utils",
    "name": "my-plugin-utils",
    "language": "php"
})
utils_id = utils["id"]

# Create a collection to group them
ecosystem = client.call_tool("create_collection_tool", {
    "name": "my-plugin-ecosystem",
    "project_ids": [core_id, pro_id, ecom_id, utils_id]
})
ecosystem_id = ecosystem["id"]
```

### Index All Projects

```python
project_ids = [core_id, pro_id, ecom_id, utils_id]

for pid in project_ids:
    result = client.call_tool("reindex", {
        "project_id": pid,
        "mode": "full"
    })
    print(f"Project {pid}: {result['files_processed']} files, {result['symbols_extracted']} symbols")
```

### Search Across the Ecosystem

Find all uses of a payment filter hook across all projects:

```python
result = client.call_tool("search", {
    "query": "payment filter",
    "limit": 20
})

print(f"Found {len(result)} matches across {len(set(r['project_id'] for r in result))} projects\n")

for r in result:
    print(f"{r['project_name']}: {r['file']}:{r['line']}")
    print(f"  {r['snippet'][:100]}...\n")
```

### Trace a Utility Function's Cross-Project Usage

The utilities package exports a `format_price()` function. Who uses it?

```python
result = client.call_tool("cross_refs", {
    "symbol_name": "format_price"
})

print(f"Defined in: {', '.join(p['name'] for p in result['definition_projects'].values())}\n")
print(f"Cross-project references: {len(result['cross_refs'])}\n")

by_project = {}
for ref in result["cross_refs"]:
    proj = ref["from_project_name"]
    if proj not in by_project:
        by_project[proj] = []
    by_project[proj].append(ref)

for proj, refs in sorted(by_project.items()):
    print(f"{proj}: {len(refs)} calls")
    for ref in refs[:2]:
        print(f"  {ref['file']}:{ref['line']}")
    if len(refs) > 2:
        print(f"  ... and {len(refs) - 2} more")
```

### View the Dependency Graph

```python
result = client.call_tool("collection_map", {
    "collection_id": ecosystem_id
})

print("Ecosystem Structure:")
print("====================\n")

for name, proj in sorted(result["projects"].items()):
    print(f"{name} (ID {proj['id']}): {proj['symbol_count']} symbols")

print("\nDependencies:")
print("--------------\n")

for edge in sorted(result["edges"], key=lambda e: -e["cross_refs"]):
    print(f"{edge['from']} → {edge['to']}: {edge['cross_refs']} cross-refs")
    if edge["symbols"]:
        print(f"  Key symbols: {', '.join(edge['symbols'][:5])}")
```

**Example output:**
```
Ecosystem Structure
====================

my-plugin (ID 1): 847 symbols
my-plugin-pro (ID 2): 234 symbols
my-plugin-ecom (ID 3): 456 symbols
my-plugin-utils (ID 4): 89 symbols

Dependencies:
--------------

my-plugin-pro → my-plugin: 12 cross-refs
  Key symbols: apply_filters, get_option, register_activation_hook
my-plugin-ecom → my-plugin: 8 cross-refs
  Key symbols: apply_filters, register_post_type
my-plugin-ecom → my-plugin-utils: 5 cross-refs
  Key symbols: format_price, validate_license
my-plugin-pro → my-plugin-utils: 3 cross-refs
  Key symbols: format_price
```

---

## Troubleshooting

### Projects Not Showing in Search Results

**Problem:** You registered projects A and B, but search only returns results from A.

**Diagnosis:**
1. Check that both projects are registered: `list_projects` (view via `status` tool)
2. Check that both have been indexed: `reindex` with `mode="full"` for each
3. Check for stale indexes: Look for warnings in search results mentioning stale file hashes

**Fix:**
```python
# Force a full re-index on the missing project
result = client.call_tool("reindex", {
    "project_id": project_b_id,
    "mode": "full",
    "force": True
})
```

### Cross-Project References Not Found

**Problem:** `cross_refs` returns an empty list even though you're sure project B calls a function from project A.

**Diagnosis:**
1. Check that the symbol is defined in project A: `search` for the symbol name in project A only
2. Check that project B has been indexed: `status` tool shows file counts
3. The reference may not be extracted by the parser (dynamic calls, string-based hooks, etc.)

**About dynamic references:**
Tessera uses tree-sitter for parsing. It cannot extract:
- Function names passed as strings: `call_user_func($func_name)` (PHP)
- Hook names in variables: `do_action($hook)` (WordPress)
- Dynamically loaded modules

For WordPress hooks specifically, add comments to help the parser:
```php
// @tessera-ref: my_custom_hook
do_action($hook);  // Parser will now associate this with "my_custom_hook"
```

### Cross-Refs Show References But No Definitions

**Problem:** `cross_refs` finds project B calling a symbol, but `definition_projects` is empty.

**Diagnosis:** The symbol is referenced but not defined in any indexed project.

**Possible causes:**
- The symbol is defined in an unindexed project (e.g., WordPress core, external library)
- The symbol is imported from an external library and re-exported; the original definition isn't indexed

**Workaround:** Use `cross_refs` to find all references, then manually check their definitions. Or register and index the library project if it's part of your codebase.

### Search Results Mixing Unrelated Projects

**Problem:** You searched for something and got results from a project you didn't intend to include.

**Diagnosis:** You're in "global" or "project" scope mode. All accessible projects are searched.

**Fix:** Use scoped session tokens to limit agent access. See [Security & Access Control](./security.md).

---

## Best Practices

### Organize Ecosystems into Collections

If you're managing related projects (plugin ecosystem, microservices, etc.), create a collection immediately. It makes documentation clearer and it's easier for tooling to understand the boundary.

```python
client.call_tool("create_collection_tool", {
    "name": "my-ecosystem",
    "project_ids": [1, 2, 3, 4]
})
```

### Index Incrementally After Registration

Don't wait to index all projects at once. Register, then index project by project:

```python
# Register all
projects = [...]
for path, name in projects:
    client.call_tool("register_project", {"path": path, "name": name})

# Index all (can parallelize this)
for pid in all_project_ids:
    client.call_tool("reindex", {"project_id": pid, "mode": "full"})
```

### Use Language Hints at Registration

Tessera auto-detects language during indexing, but providing a hint at registration time makes logs clearer:

```python
client.call_tool("register_project", {
    "path": "/path/to/project",
    "name": "my-project",
    "language": "php"  # Hint at registration
})
```

### Pin Sub-Agents to Collections

If you delegate analysis to a sub-agent, create a scoped token that limits it to a specific collection or set of projects:

```python
# Create a token for a code review bot that only sees the plugin ecosystem
scope = client.call_tool("create_scope_tool", {
    "agent_id": "code-review-bot",
    "scope_level": "collection",
    "collection_ids": [1]
})
```

```bash
# Pass the token via environment variable
TESSERA_SESSION_ID=<scope_session_id> claude "review code quality"
```

### Monitor Cross-Project Dependencies

Run `collection_map` periodically (e.g., in CI) to detect new or unexpected dependencies:

```python
def check_for_circular_deps(collection_id):
    result = client.call_tool("collection_map", {
        "collection_id": collection_id
    })

    edges_by_direction = {}
    for edge in result["edges"]:
        key = (edge["from"], edge["to"])
        edges_by_direction[key] = edge["cross_refs"]

    circulars = []
    for (a, b), count_ab in edges_by_direction.items():
        if (b, a) in edges_by_direction:
            circulars.append((a, b))

    if circulars:
        raise Exception(f"Circular dependencies detected: {circulars}")
```

---

## See Also

- [Search Guide](./search-guide.md) — Full search syntax and semantic search tuning
- [Security & Access Control](./security.md) — Session tokens and scope gating
- [Indexing & Maintenance](./indexing.md) — How federation works under the hood
