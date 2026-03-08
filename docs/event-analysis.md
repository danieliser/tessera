# Event Analysis Architecture

Tessera tracks event-driven patterns across languages — WordPress hooks, Node.js EventEmitter, Django signals, DOM events — using directional graph edges stored alongside the standard call graph.

## How It Works

### Directional Edges

Every event interaction produces one of two edge types:

- **`registers_on`** — a function registers a listener on an event
  - PHP: `add_action('init', 'my_init')` → `my_init registers_on init`
  - JS: `emitter.on('change', handler)` → `handler registers_on change`
  - Python: `post_save.connect(my_handler)` → `my_handler registers_on post_save`

- **`fires`** — a function emits/fires an event
  - PHP: `do_action('init')` → `current_function fires init`
  - JS: `emitter.emit('change', data)` → `current_function fires change`
  - Python: `post_save.send(sender=MyModel)` → `current_function fires post_save`

### Event Subtypes

Some event systems distinguish between event categories. Tessera preserves this via the `subtype` field:

| Language | Function | Direction | Subtype |
|----------|----------|-----------|---------|
| PHP | `add_action` | registers_on | `action` |
| PHP | `add_filter` | registers_on | `filter` |
| PHP | `do_action` | fires | `action` |
| PHP | `apply_filters` | fires | `filter` |
| JS | `addAction` (@wordpress/hooks) | registers_on | `action` |
| JS | `addFilter` (@wordpress/hooks) | registers_on | `filter` |
| JS | `doAction` (@wordpress/hooks) | fires | `action` |
| JS | `applyFilters` (@wordpress/hooks) | fires | `filter` |
| JS | `on` / `addEventListener` | registers_on | *(none)* |
| JS | `emit` / `dispatchEvent` | fires | *(none)* |
| Python | `signal.connect` | registers_on | *(none)* |
| Python | `signal.send` | fires | *(none)* |

Generic event systems (EventEmitter, DOM, Django signals) have no subtype — the field is omitted from output.

### Plugin Extractor Architecture

Each language has a dedicated extractor module in `src/tessera/parser/languages/`. Extractors declare event patterns via class-level dictionaries:

```python
class PHPExtractor(LanguageExtractor):
    EVENT_REGISTERS = {
        "add_action": "registers_on",
        "add_filter": "registers_on",
    }
    EVENT_FIRES = {
        "do_action": "fires",
        "apply_filters": "fires",
    }
    EVENT_SUBTYPES = {
        "add_action": "action",
        "do_action": "action",
        "add_filter": "filter",
        "apply_filters": "filter",
    }
```

The base class provides default AST walking for event extraction. Languages with non-standard patterns (e.g., Python signals using `signal.connect()` where the event name comes from the object, not a string argument) override `extract_events()`.

Adding a new event pattern for an existing language: add entries to the dictionaries. Adding a new language: create a module in `parser/languages/` — it auto-discovers on import.

### Storage

Event edges are stored in the `refs` table alongside standard references (calls, imports, extends):

```sql
SELECT r.to_symbol_name AS event_name,
       r.kind AS direction,        -- 'registers_on' or 'fires'
       r.subtype,                  -- 'action', 'filter', or ''
       s.name AS from_symbol,
       f.path AS from_file,
       r.line
FROM refs r
JOIN symbols s ON r.from_symbol_id = s.id
LEFT JOIN files f ON s.file_id = f.id
WHERE r.kind IN ('registers_on', 'fires')
```

## The `events()` MCP Tool

### Basic Queries

```
events()                                          # All events
events("pum_popup_saved")                         # Specific event
events("pum_%")                                   # Wildcard pattern
events("pum_popup_saved", direction="registers_on")  # Who listens
events("pum_popup_saved", direction="fires")         # Who fires
```

### Mismatch Detection

```
events(detect_mismatches=True)                    # All mismatches
events(detect_mismatches=True, mismatch_filter="orphaned")  # Registered, never fired
events(detect_mismatches=True, mismatch_filter="unfired")   # Fired, no listeners
```

Mismatch types:
- **ORPHANED_LISTENER** — event is registered but never fired in indexed code. Common for WordPress core hooks (fired by WP itself, not indexed).
- **UNFIRED_EVENT** — event is fired but no listener exists in indexed code. Indicates dead extensibility points or hooks consumed by external add-ons.

### Pagination

```
events("pum_%", limit=20, offset=40)   # Page 3 of results
events("pum_%", limit=0)               # Unlimited (use with care)
```

Default limit is 50 results per call.

## Interpreting Results

### Healthy Events
Both `registers_on` and `fires` edges exist → the event is actively used.

### Orphaned Listeners in WordPress
Many "orphaned" listeners hook into WordPress core actions (`admin_init`, `wp_enqueue_scripts`, etc.). These fire from WordPress itself, which isn't in the index. They're healthy — the mismatch is expected. Cross-referencing with WP core documentation can distinguish true orphans.

### Unfired Events in Plugin Ecosystems
Extensibility hooks that appear "unfired" may have listeners in *other* plugins. Indexing related plugins in the same collection resolves this. For example, `pum_popup_saved` has listeners in Popup Maker Pro that only appear when PM Pro is indexed alongside PM.

### Dynamic Hook Names
PHP patterns like `do_action("pum_{$field_type}_render")` produce edges with the literal string `pum_` — the dynamic portion can't be resolved statically. These will always show as unfired. The presence of these edges still indicates the extensibility pattern exists.
