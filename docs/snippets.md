# Snippet Rendering in Tessera

## Overview

When you search Tessera and get results, each result includes a **snippet** — a focused excerpt from the code chunk that shows the most relevant lines in context. Rather than displaying an entire chunk (often 20-40 lines), the snippet system identifies the single best-matching line and renders it with structural context to help you understand what you're looking at.

This guide explains how snippets work, how Tessera decides which lines matter, and how to control the output with `expand_context` and `max_depth` parameters.

## What Is a Snippet?

A snippet is the output of `extract_snippet()`, which takes three inputs:
1. **The full chunk content** — Text from a parsed symbol (function, class, module section)
2. **Your search query** — The words or concepts you're searching for
3. **Optional ancestry** — Symbols that contain the matched line (e.g., the function containing a matched statement)

The function returns a focused excerpt with these properties:
- **The best-matching line** — The line that best answers your query
- **Context window** — 3 lines before and after the match (by default)
- **Ancestor skeleton** — If the match is nested (e.g., inside a method inside a class), the snippet shows where it sits in that structure
- **Line numbers** — Every line is numbered with its absolute file position

Here's a concrete example. Your search: `"error handling"`. The chunk contains:

```
10 | def process():
11 |     data = load_file()
12 |     try:
13 |         result = transform(data)
14 |     except ValueError as e:
15 |         logger.error(str(e))
16 |     return result
```

The snippet would be:

```
10 | def process():
     ...  (2 lines)
12 |     try:
13 |         result = transform(data)
14 |     except ValueError as e:
15 |         logger.error(str(e))
16 |     return result
```

The `...  (2 lines)` marker is a **collapse** — it hides lines 11 (and the start of the function body) between the function definition and the actual error handling block.

---

## Line Scoring: How Snippets Pick the "Best" Line

Tessera uses two strategies to find the most relevant line in a chunk: **semantic scoring** and **keyword scoring**.

### Semantic Scoring (Preferred)

When embeddings are available, Tessera scores **3-line sliding windows** by semantic similarity to your query embedding.

**How it works:**
1. Build sliding windows: for each line index i, create a window of lines [i-1, i, i+1]
2. Embed each window (batch operation)
3. Compute cosine similarity between each window embedding and your query embedding
4. Pick the line at the center of the highest-scoring window

**Why this matters:** Semantic scoring finds *conceptual matches*, not just keyword matches.

**Concrete example:**
```
Query: "error handling"

Line 0: # error handling in the request/response pipeline
Line 1: import os
Line 2: import sys
Line 3: def process():
Line 4:     try:
Line 5:         result = transform(data)
Line 6:     except ValueError:
Line 7:         log_error(e)
Line 8:     return result
```

- **Keyword scoring** would pick line 0 (it literally says "error handling").
- **Semantic scoring** would pick lines 4-7 (the actual `try/except` block that *implements* error handling).

Semantic scoring wins because it understands semantics, not just surface text.

### Keyword Scoring (Fallback)

If embeddings are unavailable or semantic scoring fails, Tessera falls back to **keyword overlap**.

**How it works:**
1. Tokenize the query (split into words, lowercase)
2. For each line, count how many query tokens appear in it
3. Pick the line with the highest overlap

**Why it's reliable:** Identifier lookups almost always work. Query `"ProjectDB"` will find the line where `ProjectDB` is used or defined.

### Fallback Chain

The actual flow is:
1. **Try semantic scoring** (if query embedding + embed_fn available)
2. **On any error** (embedding server down, embed_fn fails), fall back to keyword
3. **Always return a result** — worst case, keyword scoring works for any query

This means snippets are always generated, even if semantic embeddings fail.

---

## Context Modes: `expand_context` Parameter

The `expand_context` parameter (in the search tool) controls how much structural context is shown around the best match.

### `"lines"` Mode (Default)

**Rendering strategy:** Show the match window (best line ±3 lines) surrounded by a **collapsed ancestry skeleton**.

**What you see:**
- Each ancestor's definition line (class declaration, function signature)
- Collapse markers (`...  (N lines)`) between ancestors and between the last ancestor and the match window
- The match window itself (full ±3 lines)
- Collapse markers after the match window to the end of the innermost ancestor

**Example:**

```
5 | class Processor:
     ...  (5 lines)
11 |     def run(self):
         ...  (2 lines)
14 |         self.status = "running"
15 |         try:
16 |             result = self.execute()
17 |         except Exception as e:
18 |             self.log_error(e)
19 |         self.status = "done"
         ...  (1 lines)
```

In this example:
- Line 5: class definition
- `...  (5 lines)`: collapses lines 6-10
- Line 11: method definition
- `...  (2 lines)`: collapses lines 12-13
- Lines 14-19: the actual match window (error handling code)
- `...  (1 lines)`: collapses lines 20 to the end of the method

**Use case:** "I want to see the match with just enough context to understand where it sits in the class/module structure."

### `"full"` Mode

**Rendering strategy:** Expand the entire containing symbol (the outermost ancestor) from start to end.

**What you see:**
- Every line from the outermost ancestor's definition to its end
- No collapse markers
- Line numbers for all lines

**Example (same nesting as above, but full mode):**

```
5 | class Processor:
6 |     def __init__(self):
7 |         self.status = None
8 |         self.cache = {}
9 |
10 |     def setup(self):
11 |         self.status = "setup"
12 |
13 |     def run(self):
14 |         self.status = "running"
15 |         try:
16 |             result = self.execute()
17 |         except Exception as e:
18 |             self.log_error(e)
19 |         self.status = "done"
20 |
21 |     def execute(self):
22 |         return None
```

**Use case:** "I need to understand the complete containing symbol, not just the matched snippet."

---

## Ancestry Skeletons: Understanding Nesting

### How Ancestry Works

When a search result's best-matching line is inside a symbol (e.g., a method inside a class), Tessera looks up all the symbols that contain that line. The `get_ancestor_symbols()` method returns them ordered **outermost first**.

For example, if the matched line is inside a method inside a class inside a module:
```
Ancestors (in order): [Module, Class, Method]
```

### Rendering Rules (Lines Mode)

In `"lines"` mode, the snippet renderer uses these rules:

1. **Render ancestor definitions** (class and function signatures) that appear *before* the match window
2. **Skip ancestors whose definition line is inside the match window** (they're already visible)
3. **Show collapse markers** between each rendered element
4. **Indent collapse markers** based on nesting depth (more indentation = deeper nesting)

**Example:**

Here's actual code (file lines 5-25):
```
5 | class MyClass:
6 |     def helper(self):
7 |         x = 1
8 |         y = 2
9 |         foo_match = 3
10 |        z = 4
11 |        return z
```

Your search: `"foo_match"`. The chunk starts at line 5.

Ancestors for line 9: `[{"name": "MyClass", "kind": "class", "line": 5, "end_line": 25, "signature": "class MyClass:"}, {"name": "helper", "kind": "method", "line": 6, "end_line": 11, "signature": "    def helper(self):"}]`

Rendered snippet in `"lines"` mode (context_lines=1):

```
5 | class MyClass:
     ...  (1 lines)
6 |     def helper(self):
         ...  (2 lines)
8 |         y = 2
9 |         foo_match = 3
10 |        z = 4
     ...  (1 lines)
```

Breaking this down:
- Line 5: Class definition rendered
- `...  (1 lines)`: collapses the space between the class definition (line 5) and the method definition (line 6). Indentation is one level (4 spaces) because we're inside a class.
- Line 6: Method definition rendered
- `...  (2 lines)`: collapses lines 7-8. Indentation is two levels (8 spaces) because we're inside a method inside a class.
- Lines 8-10: match window
- `...  (1 lines)`: collapses the final line (11) of the method

The **indentation of collapse markers** encodes nesting depth: `"    " * num_ancestors`.

---

## Controlling Snippet Depth: `max_depth` Parameter

The `max_depth` parameter limits how many ancestor levels are shown. This is useful when deeply nested code produces excessive ancestor chains.

### Example

Code structure:
```
Module: Config
  Class: Settings
    Class: Database
      Method: connect()
        Inside: Line 150 (matched)
```

Ancestors (outermost first): `[Config, Settings, Database, connect]` (4 levels)

**Without max_depth (show all 4 levels):**
```
1 | # Module: Config
     ...  (10 lines)
12 | class Settings:
     ...  (5 lines)
18 |     class Database:
         ...  (3 lines)
22 |         def connect(self):
             ...  (2 lines)
25 |             conn = establish()
26 |             return conn
```

**With max_depth=1 (show only innermost):**
```
22 |         def connect(self):
             ...  (2 lines)
25 |             conn = establish()
26 |             return conn
```

The `max_depth` parameter **trims ancestors from the outermost end**. So `max_depth=1` keeps only the innermost ancestor (the function), `max_depth=2` keeps class + method, etc.

**When to use:**
- `max_depth=None` (default): Show all nesting. Good for understanding full context.
- `max_depth=1`: Show only the immediate containing symbol. Good for deeply nested code where full ancestry is overwhelming.
- `max_depth=2`: Show two levels (e.g., class + method). Common for object-oriented code.

---

## Practical Guidance

### Choosing Context Mode

**Use `expand_context="lines"` (default) when:**
- You want a focused view of the matched code
- You're scanning results quickly
- The matching symbol is medium to large
- You just need to confirm this is the right place

**Use `expand_context="full"` when:**
- You need to understand the complete containing function/class
- The matched line is near the start of its symbol and you want to see the whole implementation
- You're reviewing code for a PR or audit
- Context is short (<100 lines)

### Tuning max_depth

**Use `max_depth=1` when:**
- Code is deeply nested (3+ levels)
- You only care about the immediate container (e.g., the method, not the class above it)

**Use `max_depth=2` or higher when:**
- You need to understand class + method relationships
- The nesting is shallow (2 levels)

**Omit (default: all levels) when:**
- Nesting is moderate (1-3 levels)
- You want the full structural picture

---

## Example Workflows

### Workflow 1: Locating a Function Call

**Search:** `"lex:ProjectDB"`

**Result snippet (default: lines mode):**
```
45 | class QueryRunner:
     ...  (8 lines)
54 |     def build_query(self):
         ...  (3 lines)
57 |         db = ProjectDB(path)
58 |         return db.find_symbol("foo")
```

**What you learn:** `ProjectDB` is used inside the `build_query` method of `QueryRunner` class, around line 57.

---

### Workflow 2: Understanding Error Handling

**Search:** `"vec:error handling"`

**Result snippet with semantic scoring:**
```
120 | def process_request():
      ...  (8 lines)
128 |     try:
129 |         response = call_api()
130 |     except NetworkError as e:
131 |         return retry_with_backoff(e)
132 |     return response
```

**What you learn:** The best semantic match for "error handling" is the actual try/except block (lines 128-131), not a comment that happens to mention the words.

---

### Workflow 3: Deep Nesting with max_depth

**Search:** `"cache hit"`

**Code structure:** `Module → Config → Cache → Strategy → find_hit()`

**With max_depth=1:**
```
250 |         def find_hit(self):
      ...  (4 lines)
254 |             if key in self.cache:
255 |                 return self.cache[key]
```

**What you learn:** Only the innermost method is shown. You don't see the 3 parent layers.

**With max_depth=None (default):**
```
1 | # Cache strategy module
    ...  (20 lines)
22 | class Config:
    ...  (5 lines)
28 |     class Cache:
        ...  (3 lines)
32 |         class Strategy:
             ...  (2 lines)
35 |             def find_hit(self):
                 ...  (4 lines)
39 |                 if key in self.cache:
40 |                 return self.cache[key]
```

**What you learn:** The full nesting structure, from module to method.

---

## Implementation Details

### Where Snippets Are Generated

Snippets are generated by the `search` tool in `/src/tessera/server/tools/_search.py`. The flow is:

1. Run hybrid search (keyword + semantic + graph ranking)
2. For each result with content:
   - Compute the best-matching line (semantic or keyword)
   - Look up ancestor symbols for that line
   - Call `extract_snippet()` with the selected mode
   - Embed the result in the search result dict

### The extract_snippet Function

Located in `/src/tessera/search.py`:

```python
extract_snippet(
    chunk_content: str,
    query: str,
    context_lines: int = 3,
    ancestors: list[dict] | None = None,
    chunk_start_line: int = 0,
    mode: str = "lines",
    max_depth: int | None = None,
    query_embedding: Optional[np.ndarray] = None,
    embed_fn: Optional[Callable[..., list]] = None,
) -> dict:
```

**Key functions:**
- `_find_best_match_line()`: Picks the best line (semantic or keyword)
- `_semantic_best_line()`: Scores 3-line windows with embeddings
- `_render_line()`: Formats a single line with its number
- `_render_collapse()`: Formats a collapse marker

### Ancestor Lookup

The `get_ancestor_symbols()` method in `/src/tessera/db/_project.py` returns all symbols whose line range contains the matched line:

```python
SELECT * FROM symbols
WHERE file_id = ? AND line <= ? AND end_line >= ?
ORDER BY line ASC, end_line DESC
```

Result: symbols ordered outermost first (ascending start line, then descending end line for tiebreakers).

---

## Troubleshooting

### Snippet Shows the Wrong Line

**Cause:** Keyword scoring picked a comment that mentions your words instead of the actual code.

**Solution:** Enable semantic search (default) if you're using `"lex"` mode. Semantic scoring finds the conceptual match, not just the keyword match.

### Snippet Is Too Large or Too Small

**Too large:** Use `expand_context="lines"` (default) instead of `"full"`. Or reduce context_lines (not a search tool parameter, but available in direct API calls).

**Too small:** Use `expand_context="full"` to see the entire containing symbol.

### Ancestry Chain Is Overwhelming

**Use `max_depth=1` or `max_depth=2`** to reduce nesting levels shown. You'll still see the matched line with immediate context.

### Embeddings Not Working

**Symptom:** Snippets seem to pick random lines, not semantically relevant ones.

**Cause:** Embedding client is down or unavailable.

**Solution:** Keyword scoring (fallback) still works, but it's less smart. Restart the embedding server or check logs for `EmbeddingUnavailableError`.

---

## API Summary

### Search Tool Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expand_context` | string | `"lines"` | Snippet mode: `"lines"` (collapsed ancestry) or `"full"` (expand all) |
| `max_depth` | int \| None | None | Max ancestor nesting levels to show. None = all. |

### Result Snippet Fields

Each search result includes:

| Field | Type | Description |
|-------|------|-------------|
| `snippet` | string | Rendered snippet with line numbers and collapse markers |
| `snippet_start_line` | int | First line of snippet (absolute file line) |
| `snippet_end_line` | int | Last line of snippet (absolute file line) |
| `best_match_line` | int | Line Tessera identified as the best match |
| `ancestors` | list[dict] | Ancestor symbols shown in snippet (name, kind, line, end_line) |

---

## See Also

- **Search tool documentation** in `/docs/api.md`
- **Tree-sitter parser** used for symbol extraction (see `/src/tessera/parser.py`)
- **FAISS vector search** for semantic scoring (see `/src/tessera/search.py`)
