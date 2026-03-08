# Tessera Now Maps Event-Driven Architecture Across Your Entire Codebase

*v0.10.1 release — March 2026*

## The Problem Nobody Talks About

Every WordPress plugin is an event-driven system. Hundreds of hooks — `add_action`, `add_filter`, `do_action`, `apply_filters` — wiring together functionality that no call graph can trace. The same pattern shows up in Node.js with EventEmitter, in frontend code with DOM events, and in Django with signals.

Static analysis tools don't see any of it. Your IDE can find function calls. It can't tell you that `pum_popup_content` is fired in exactly one place (`Popup::get_content`) and has 15 listeners spread across `Site.php`, `Bricks.php`, and a legacy compatibility layer.

Until now.

## One Query, Complete Visibility

Tessera v0.10.1 introduces the `events()` tool — a dedicated MCP interface for querying event registrations and emissions across your indexed codebase.

Ask it who listens to a hook:

```
events("pum_popup_content", direction="registers_on")
```

You get back every listener, the function that registered it, the file, and the line number. Ask who fires it:

```
events("pum_popup_content", direction="fires")
```

One fire point. `Popup::get_content`, line 159. That's the complete picture of this hook's lifecycle, assembled from static analysis in milliseconds.

## The Real Power: Mismatch Detection

Here's where it gets interesting. Ask Tessera to analyze registration/emission balance:

```
events(detect_mismatches=True, mismatch_filter="unfired")
```

On Popup Maker — a production WordPress plugin with 837 files and 4,986 symbols — this returns 157 `pum_`-prefixed hooks that are fired but have zero listeners. Plus 46 `popmake_`-prefixed legacy hooks that nobody's using.

That's dead code detection for event-driven systems. No existing tool does this.

Some of those "unfired" hooks are intentional extensibility points — they're designed for add-on plugins to hook into. But 46 deprecated `popmake_*` hooks from a legacy API? Those are confirmed removal candidates. One query surfaced cleanup work that would have taken hours of manual code review.

## Cross-Language, Same Model

This isn't WordPress-specific. The same `registers_on` / `fires` model works across:

- **PHP**: `add_action`, `add_filter`, `do_action`, `apply_filters`, `do_action_ref_array`, `apply_filters_ref_array`
- **JavaScript/TypeScript**: `on`, `once`, `addEventListener`, `emit`, `dispatchEvent`, `trigger`
- **@wordpress/hooks (JS)**: `addAction`, `addFilter`, `doAction`, `applyFilters`
- **Python**: `signal.connect`, `signal.send`, `signal.send_robust`

A polyglot codebase — PHP backend with a JS frontend sharing the same hook namespace via `@wordpress/hooks` — gets unified analysis across both languages in a single query.

## Actions vs Filters: Semantic Subtyping

WordPress distinguishes between actions (fire-and-forget) and filters (value-returning chains). This matters — a filter's return value flows through every listener in sequence. Tessera preserves this distinction:

```json
{
  "event_name": "pum_popup_content",
  "direction": "registers_on",
  "event_subtype": "filter",
  "from_symbol": "add_core_content_filters",
  "file": "classes/Site.php",
  "line": 37
}
```

The `event_subtype` field appears only when the language has meaningful subtypes. Generic event systems (EventEmitter, DOM, Django signals) omit it entirely. No noise for codebases that don't need it.

## The Numbers

Popup Maker analysis (837 files, 4,986 symbols):

| Metric | Count |
|--------|-------|
| Total event edges | 1,011 |
| Unique events | 603 |
| Actions | 300 |
| Filters | 495 |
| Generic JS events | 216 |
| Healthy (both sides present) | 77 |
| Orphaned listeners | 237 |
| Unfired events | 157 + 46 legacy |

Most-used hook: `pum_init` — 22 references (12 listeners, 10 fire points). The central nervous system of the plugin, visible in one query.

## How It Works

Under the hood: tree-sitter AST analysis with pluggable per-language extractors. Each language module declares event function patterns via simple dictionaries. The base class handles AST walking. Adding a new event pattern is adding a dictionary entry. Adding a new language is dropping a file in `parser/languages/`.

No LLMs. No training data. No hallucinated edges. Deterministic, reproducible, language-agnostic.

## Try It

```bash
pip install tessera-idx
tessera index /path/to/your/project
tessera serve --project /path/to/your/project
```

Then ask your AI agent to call `events()`.

---

*Tessera is open source under the MIT license. [GitHub](https://github.com/danieliser/tessera)*
