"""Tests for the hooks system and model-profile-driven hook registration."""

import asyncio
from unittest.mock import MagicMock

from tessera.hooks import (
    EMBEDDING_TEXTS,
    _scope_prefix_filter,
    get_hooks,
    reset_hooks,
    setup_model_hooks,
)
from tessera.model_profiles import ModelProfile


def _make_profile(**overrides) -> ModelProfile:
    defaults = dict(
        key="test",
        model_id="test/model",
        display_name="Test-384d",
        dimensions=384,
        max_tokens=512,
        size_mb=67,
        architecture="bert",
        provider="fastembed",
    )
    defaults.update(overrides)
    return ModelProfile(**defaults)


class TestHookRegistration:

    def setup_method(self):
        reset_hooks()

    def test_scope_prefix_registered_when_flag_true(self):
        profile = _make_profile(scope_prefix=True)
        setup_model_hooks(profile)
        hooks = get_hooks()
        assert hooks.has_filter(EMBEDDING_TEXTS) > 0

    def test_scope_prefix_not_registered_when_flag_false(self):
        profile = _make_profile(scope_prefix=False)
        setup_model_hooks(profile)
        hooks = get_hooks()
        assert hooks.has_filter(EMBEDDING_TEXTS) == 0

    def test_setup_with_none_profile(self):
        setup_model_hooks(None)
        hooks = get_hooks()
        assert hooks.has_filter(EMBEDDING_TEXTS) == 0

    def test_setup_clears_previous_hooks(self):
        profile_on = _make_profile(scope_prefix=True)
        profile_off = _make_profile(scope_prefix=False)

        setup_model_hooks(profile_on)
        hooks = get_hooks()
        assert hooks.has_filter(EMBEDDING_TEXTS) > 0

        setup_model_hooks(profile_off)
        assert hooks.has_filter(EMBEDDING_TEXTS) == 0

    def test_reset_hooks_clears_all(self):
        profile = _make_profile(scope_prefix=True)
        setup_model_hooks(profile)
        reset_hooks()
        hooks = get_hooks()
        assert hooks.has_filter(EMBEDDING_TEXTS) == 0


class TestScopePrefixFilter:

    def _make_chunk(self, start_line: int, end_line: int):
        chunk = MagicMock()
        chunk.start_line = start_line
        chunk.end_line = end_line
        return chunk

    def _make_symbol(self, name: str, kind: str, line: int, end_line: int, scope: str = ""):
        sym = MagicMock()
        sym.name = name
        sym.kind = kind
        sym.line = line
        sym.end_line = end_line
        sym.scope = scope
        return sym

    def test_no_symbols_returns_original(self):
        texts = ["some code here"]
        chunks = [self._make_chunk(0, 10)]
        result = _scope_prefix_filter(texts, chunks=chunks, symbols=[], rel_path="test.py")
        assert result == texts

    def test_adds_class_prefix(self):
        texts = ["class body code"]
        chunks = [self._make_chunk(0, 20)]
        symbols = [self._make_symbol("MyClass", "class", 0, 20)]
        result = _scope_prefix_filter(texts, chunks=chunks, symbols=symbols, rel_path="src/test.py")
        assert result[0].startswith("MyClass class in test.py: ")

    def test_adds_method_prefix(self):
        texts = ["method body"]
        chunks = [self._make_chunk(5, 10)]
        symbols = [
            self._make_symbol("MyClass", "class", 0, 20),
            self._make_symbol("do_thing", "method", 5, 10, scope="MyClass"),
        ]
        result = _scope_prefix_filter(texts, chunks=chunks, symbols=symbols, rel_path="util.php")
        assert "MyClass class" in result[0]
        assert "do_thing method" in result[0]
        assert "in util.php: " in result[0]

    def test_non_overlapping_symbol_ignored(self):
        texts = ["standalone code"]
        chunks = [self._make_chunk(50, 60)]
        symbols = [self._make_symbol("FarAway", "class", 0, 10)]
        result = _scope_prefix_filter(texts, chunks=chunks, symbols=symbols, rel_path="x.py")
        assert result == texts

    def test_multiple_chunks(self):
        texts = ["chunk1 code", "chunk2 code"]
        chunks = [self._make_chunk(0, 10), self._make_chunk(20, 30)]
        symbols = [
            self._make_symbol("A", "class", 0, 10),
            self._make_symbol("B", "function", 20, 30),
        ]
        result = _scope_prefix_filter(texts, chunks=chunks, symbols=symbols, rel_path="f.py")
        assert "A class" in result[0]
        assert "B method" in result[1]


class TestHookExecution:

    def setup_method(self):
        reset_hooks()

    def test_filter_executes_through_hooks(self):
        profile = _make_profile(scope_prefix=True)
        setup_model_hooks(profile)
        hooks = get_hooks()

        texts = ["code here"]
        chunk = MagicMock()
        chunk.start_line = 0
        chunk.end_line = 10

        sym = MagicMock()
        sym.name = "Foo"
        sym.kind = "class"
        sym.line = 0
        sym.end_line = 10
        sym.scope = ""

        result = asyncio.run(
            hooks.apply_filters(
                EMBEDDING_TEXTS,
                texts,
                chunks=[chunk],
                symbols=[sym],
                rel_path="bar.py",
            )
        )
        assert "Foo class" in result[0]
        assert "in bar.py: " in result[0]
