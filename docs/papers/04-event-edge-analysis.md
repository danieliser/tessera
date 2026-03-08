# Paper 4: Static Detection of Event-Driven Architecture Patterns Across Language Boundaries

**Status:** IDEA
**Target venue:** MSR 2027 (tool track, 4 pages) or ICPC 2027

## Abstract (Draft)

Event-driven architectures — WordPress hooks, Node.js EventEmitter, Django signals, DOM events — are pervasive in modern software yet invisible to traditional static analysis tools. Call graphs show direct invocations but miss indirect coupling through event registration and emission, creating blind spots in dependency analysis, dead code detection, and impact assessment.

We present a cross-language static analysis approach that detects event-driven patterns using directional graph edges: `registers_on` (listener registration) and `fires` (event emission). Our system uses a pluggable extractor architecture built on tree-sitter AST analysis to unify event patterns across PHP (WordPress hooks), JavaScript/TypeScript (EventEmitter, DOM events, @wordpress/hooks), and Python (Django signals) under a single queryable model.

We introduce two novel analyses enabled by directional event edges: (1) mismatch detection, identifying orphaned listeners (registered but never fired) and unfired events (fired but never listened to) as indicators of dead code or missing integrations; and (2) semantic subtyping, distinguishing WordPress actions (fire-and-forget) from filters (value-returning chains) without runtime analysis.

Evaluated against Popup Maker, a production WordPress plugin (837 files, 4,986 symbols), our system extracts 1,011 event edges across 603 unique events, classifies them into 300 actions and 495 filters, and identifies 157 unfired extensibility hooks and 237 orphaned listeners — including 46 deprecated legacy hooks confirmed as removal candidates by the maintainer.

## Key Claims

1. **Directional event edges** (`registers_on` / `fires`) enable analyses impossible with undirected "hooks_into" edges — specifically mismatch detection and event flow tracing.

2. **Cross-language unification** allows event analysis across polyglot codebases (PHP backend + JS frontend sharing a hook namespace via @wordpress/hooks).

3. **Semantic subtyping** (action vs. filter) preserves domain-specific meaning without runtime analysis, enabling richer static reasoning about event contracts.

4. **Mismatch detection is practical dead code analysis** for event-driven systems — a category of dead code that existing tools miss entirely.

## Required Literature Review

- [ ] Static analysis of event-driven systems
- [ ] WordPress hook analysis tools (if any exist in literature)
- [ ] Pub/sub pattern detection in static analysis
- [ ] Dead code detection approaches
- [ ] Cross-language static analysis
- [ ] Tree-sitter in research tooling

## Data / Experiments Needed

- Popup Maker analysis (already have: 1,011 edges, 603 events)
- Second WordPress plugin for validation (PM Pro or WooCommerce)
- Node.js EventEmitter-heavy project (e.g., Socket.io)
- Django signals project (e.g., Django Oscar)
- False positive/negative analysis on mismatch detection
- Comparison with manual hook documentation (if available)

## Risks

- Tool papers have lower impact factor
- "This is just pattern matching" — need to emphasize the cross-language unification and mismatch analysis as contributions
- WordPress-heavy evaluation may limit perceived generality
