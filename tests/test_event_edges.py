"""Tests for directional event/hook edge extraction across languages."""

import pytest

from tessera.parser import extract_references, extract_symbols, build_edges
from tessera.parser._patterns import Reference


# ---------------------------------------------------------------------------
# PHP: WordPress hooks
# ---------------------------------------------------------------------------

PHP_HOOK_CODE = """<?php
function setup_plugin() {
    add_action('popup_maker/init', ['Site', 'actions']);
    add_filter('pum_popup_content', [$embed, 'run_shortcode'], 8);
    do_action('popup_maker/init');
    apply_filters('pum_popup_is_loadable', true);
}
"""


def test_php_add_action_registers_on():
    refs = extract_references(PHP_HOOK_CODE, "php")
    hook_refs = [r for r in refs if r.to_symbol == "popup_maker/init" and r.kind == "registers_on"]
    assert len(hook_refs) == 1
    assert hook_refs[0].from_symbol == "setup_plugin"


def test_php_add_filter_registers_on():
    refs = extract_references(PHP_HOOK_CODE, "php")
    hook_refs = [r for r in refs if r.to_symbol == "pum_popup_content" and r.kind == "registers_on"]
    assert len(hook_refs) == 1


def test_php_do_action_fires():
    refs = extract_references(PHP_HOOK_CODE, "php")
    hook_refs = [r for r in refs if r.to_symbol == "popup_maker/init" and r.kind == "fires"]
    assert len(hook_refs) == 1
    assert hook_refs[0].from_symbol == "setup_plugin"


def test_php_apply_filters_fires():
    refs = extract_references(PHP_HOOK_CODE, "php")
    hook_refs = [r for r in refs if r.to_symbol == "pum_popup_is_loadable" and r.kind == "fires"]
    assert len(hook_refs) == 1


def test_php_no_hooks_into_kind():
    """The old 'hooks_into' kind should no longer be produced."""
    refs = extract_references(PHP_HOOK_CODE, "php")
    hooks_into = [r for r in refs if r.kind == "hooks_into"]
    assert len(hooks_into) == 0


def test_php_event_edges_in_build_edges():
    """Event references should produce edges even when target isn't a known symbol."""
    symbols = extract_symbols(PHP_HOOK_CODE, "php")
    refs = extract_references(PHP_HOOK_CODE, "php")
    edges = build_edges(symbols, refs)
    event_edges = [e for e in edges if e.type in ("registers_on", "fires")]
    assert len(event_edges) >= 3  # at least: 2 registers_on + 2 fires (deduped by from+to+type)


# ---------------------------------------------------------------------------
# TypeScript/JavaScript: EventEmitter + DOM
# ---------------------------------------------------------------------------

JS_EVENT_CODE = """
function setupListeners(emitter) {
    emitter.on('user_created', handleUser);
    emitter.once('shutdown', cleanup);
    document.addEventListener('click', handleClick);
}

function notifyAll(emitter) {
    emitter.emit('user_created', userData);
}
"""


def test_js_on_registers_on():
    refs = extract_references(JS_EVENT_CODE, "javascript")
    hook_refs = [r for r in refs if r.to_symbol == "user_created" and r.kind == "registers_on"]
    assert len(hook_refs) == 1
    assert hook_refs[0].from_symbol == "setupListeners"


def test_js_once_registers_on():
    refs = extract_references(JS_EVENT_CODE, "javascript")
    hook_refs = [r for r in refs if r.to_symbol == "shutdown" and r.kind == "registers_on"]
    assert len(hook_refs) == 1


def test_js_addeventlistener_registers_on():
    refs = extract_references(JS_EVENT_CODE, "javascript")
    hook_refs = [r for r in refs if r.to_symbol == "click" and r.kind == "registers_on"]
    assert len(hook_refs) == 1


def test_js_emit_fires():
    refs = extract_references(JS_EVENT_CODE, "javascript")
    hook_refs = [r for r in refs if r.to_symbol == "user_created" and r.kind == "fires"]
    assert len(hook_refs) == 1
    assert hook_refs[0].from_symbol == "notifyAll"


TS_EVENT_CODE = """
class EventBus {
    subscribe(event: string, handler: Function) {
        this.on(event, handler);
    }

    publish(event: string, data: unknown) {
        this.emit(event, data);
    }
}

function init() {
    const bus = new EventBus();
    bus.on('ready', onReady);
    bus.emit('starting', {});
}
"""


def test_ts_method_on_registers_on():
    refs = extract_references(TS_EVENT_CODE, "typescript")
    hook_refs = [r for r in refs if r.to_symbol == "ready" and r.kind == "registers_on"]
    assert len(hook_refs) == 1


def test_ts_method_emit_fires():
    refs = extract_references(TS_EVENT_CODE, "typescript")
    hook_refs = [r for r in refs if r.to_symbol == "starting" and r.kind == "fires"]
    assert len(hook_refs) == 1


# ---------------------------------------------------------------------------
# Python: Django-style signals
# ---------------------------------------------------------------------------

PY_SIGNAL_CODE = """
from django.dispatch import Signal

user_created = Signal()

def setup_handlers():
    user_created.connect(handle_user_created)
    post_save.connect(audit_log)

def create_user(data):
    user = User.objects.create(**data)
    user_created.send(sender=User, instance=user)
"""


def test_python_connect_registers_on():
    refs = extract_references(PY_SIGNAL_CODE, "python")
    hook_refs = [r for r in refs if r.kind == "registers_on"]
    assert len(hook_refs) >= 1


def test_python_send_fires():
    refs = extract_references(PY_SIGNAL_CODE, "python")
    hook_refs = [r for r in refs if r.kind == "fires"]
    assert len(hook_refs) >= 1


# ---------------------------------------------------------------------------
# Edge kind documentation
# ---------------------------------------------------------------------------

def test_reference_kinds_documented():
    """Verify the Reference dataclass documents event edge kinds."""
    import inspect
    source = inspect.getsource(Reference)
    assert "registers_on" in source
    assert "fires" in source
