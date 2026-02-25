"""Tests for the tree-sitter based parser module."""

import pytest
from tessera.parser import (
    detect_language,
    parse_file,
    extract_symbols,
    extract_references,
    build_edges,
    parse_and_extract,
    Symbol,
    Reference,
    Edge,
)


class TestLanguageDetection:
    """Test language detection from file extensions."""

    def test_detect_python(self):
        assert detect_language("module.py") == "python"
        assert detect_language("/path/to/script.py") == "python"

    def test_detect_typescript(self):
        assert detect_language("index.ts") == "typescript"
        assert detect_language("app.tsx") == "typescript"

    def test_detect_javascript(self):
        assert detect_language("script.js") == "javascript"
        assert detect_language("app.jsx") == "javascript"

    def test_detect_php(self):
        assert detect_language("index.php") == "php"
        assert detect_language("/path/to/class.php") == "php"

    def test_detect_go(self):
        assert detect_language("main.go") == "go"

    def test_detect_rust(self):
        assert detect_language("main.rs") == "rust"

    def test_detect_java(self):
        assert detect_language("Main.java") == "java"

    def test_detect_csharp(self):
        assert detect_language("Program.cs") == "csharp"

    def test_detect_ruby(self):
        assert detect_language("script.rb") == "ruby"

    def test_detect_swift(self):
        assert detect_language("main.swift") == "swift"

    def test_detect_kotlin(self):
        assert detect_language("main.kt") == "kotlin"
        assert detect_language("build.kts") == "kotlin"

    def test_detect_c(self):
        assert detect_language("main.c") == "c"
        assert detect_language("header.h") == "c"

    def test_detect_cpp(self):
        assert detect_language("main.cpp") == "cpp"
        assert detect_language("main.cc") == "cpp"
        assert detect_language("main.cxx") == "cpp"
        assert detect_language("header.hpp") == "cpp"

    def test_detect_unknown(self):
        assert detect_language("file.txt") is None
        assert detect_language("file") is None


class TestPythonExtraction:
    """Test symbol and reference extraction for Python."""

    def test_extract_function(self):
        code = """
def hello(name):
    return f"Hello {name}"
"""
        symbols = extract_symbols(code, "python")
        assert len(symbols) == 1
        assert symbols[0].name == "hello"
        assert symbols[0].kind == "function"
        assert symbols[0].signature == "hello(name)"

    def test_extract_class(self):
        code = """
class Calculator:
    def add(self, a, b):
        return a + b
"""
        symbols = extract_symbols(code, "python")
        assert len(symbols) == 2
        class_sym = next(s for s in symbols if s.kind == "class")
        method_sym = next(s for s in symbols if s.kind == "method")
        assert class_sym.name == "Calculator"
        assert method_sym.name == "add"
        assert method_sym.scope == "Calculator"

    def test_extract_import(self):
        code = """
import os
from pathlib import Path
"""
        symbols = extract_symbols(code, "python")
        imports = [s for s in symbols if s.kind == "import"]
        assert len(imports) == 2

    def test_function_call_reference(self):
        code = """
def helper():
    pass

def main():
    helper()
"""
        symbols = extract_symbols(code, "python")
        refs = extract_references(code, "python")
        assert any(r.kind == "calls" and r.to_symbol == "helper" for r in refs)

    def test_class_inheritance(self):
        code = """
class Base:
    pass

class Derived(Base):
    pass
"""
        refs = extract_references(code, "python")
        assert any(
            r.kind == "extends" and r.to_symbol == "Base" and r.from_symbol == "Derived"
            for r in refs
        )


class TestTypeScriptExtraction:
    """Test symbol and reference extraction for TypeScript."""

    def test_extract_function(self):
        code = """
function greet(name: string): string {
    return `Hello ${name}`;
}
"""
        symbols = extract_symbols(code, "typescript")
        assert len(symbols) >= 1
        func = next(s for s in symbols if s.name == "greet")
        assert func.kind == "function"

    def test_extract_class(self):
        code = """
class Calculator {
    add(a: number, b: number): number {
        return a + b;
    }
}
"""
        symbols = extract_symbols(code, "typescript")
        class_sym = next((s for s in symbols if s.kind == "class"), None)
        method_sym = next((s for s in symbols if s.kind == "method"), None)
        assert class_sym is not None and class_sym.name == "Calculator"
        assert method_sym is not None and method_sym.name == "add"

    def test_extract_import(self):
        code = """
import { useState } from 'react';
import axios from 'axios';
"""
        symbols = extract_symbols(code, "typescript")
        imports = [s for s in symbols if s.kind == "import"]
        assert len(imports) >= 2

    def test_class_extends(self):
        code = """
class Animal {
    name: string;
}

class Dog extends Animal {
    bark() {}
}
"""
        refs = extract_references(code, "typescript")
        assert any(
            r.kind == "extends"
            and r.from_symbol == "Dog"
            and r.to_symbol == "Animal"
            for r in refs
        )


class TestPHPExtraction:
    """Test symbol and reference extraction for PHP."""

    def test_extract_function(self):
        code = """<?php
function greet($name) {
    return "Hello " . $name;
}
"""
        symbols = extract_symbols(code, "php")
        func = next((s for s in symbols if s.name == "greet"), None)
        assert func is not None
        assert func.kind == "function"

    def test_extract_class(self):
        code = """<?php
class Calculator {
    public function add($a, $b) {
        return $a + $b;
    }
}
"""
        symbols = extract_symbols(code, "php")
        class_sym = next((s for s in symbols if s.kind == "class"), None)
        method_sym = next((s for s in symbols if s.kind == "method"), None)
        assert class_sym is not None and class_sym.name == "Calculator"
        assert method_sym is not None and method_sym.name == "add"

    def test_extract_use_statement(self):
        code = """<?php
use Symfony\\Component\\HttpFoundation\\Request;
use PDO;
"""
        symbols = extract_symbols(code, "php")
        imports = [s for s in symbols if s.kind == "import"]
        assert len(imports) >= 2

    def test_wordpress_add_action(self):
        code = """<?php
function my_plugin_init() {
    echo "Plugin loaded";
}
add_action('wp_loaded', 'my_plugin_init');
"""
        refs = extract_references(code, "php")
        hook_refs = [r for r in refs if r.kind == "hooks_into"]
        assert any(r.to_symbol == "wp_loaded" for r in hook_refs)

    def test_wordpress_add_filter(self):
        code = """<?php
function my_filter($content) {
    return strtoupper($content);
}
add_filter('the_content', 'my_filter');
"""
        refs = extract_references(code, "php")
        filter_refs = [r for r in refs if r.kind == "hooks_into"]
        assert any(r.to_symbol == "the_content" for r in filter_refs)


class TestPythonAsyncFunctions:
    """Test async function extraction in Python."""

    def test_extract_async_function(self):
        code = """
async def fetch_data(url):
    return await get(url)
"""
        symbols = extract_symbols(code, "python")
        funcs = [s for s in symbols if s.kind == "function"]
        assert len(funcs) == 1
        assert funcs[0].name == "fetch_data"

    def test_extract_async_method(self):
        code = """
class Client:
    async def connect(self):
        pass
"""
        symbols = extract_symbols(code, "python")
        methods = [s for s in symbols if s.kind == "method"]
        assert len(methods) == 1
        assert methods[0].name == "connect"

    def test_async_function_refs_tracked(self):
        code = """
async def handler():
    await process()

def process():
    pass
"""
        refs = extract_references(code, "python")
        assert any(r.from_symbol == "handler" and r.to_symbol == "process" for r in refs)


class TestDecoratedDefinitions:
    """Test decorated function and class extraction in Python."""

    def test_extract_decorated_function(self):
        code = """
@cache
def decorated_func():
    pass
"""
        symbols = extract_symbols(code, "python")
        funcs = [s for s in symbols if s.kind == "function"]
        assert len(funcs) == 1
        assert funcs[0].name == "decorated_func"

    def test_extract_decorated_class(self):
        code = """
@dataclass
class DecoratedClass:
    pass
"""
        symbols = extract_symbols(code, "python")
        classes = [s for s in symbols if s.kind == "class"]
        assert len(classes) == 1
        assert classes[0].name == "DecoratedClass"

    def test_decorated_function_with_body(self):
        code = """
@cache
def cached_func(x):
    return compute(x)

def compute(x):
    return x * 2
"""
        symbols = extract_symbols(code, "python")
        funcs = [s for s in symbols if s.kind == "function"]
        func_names = {f.name for f in funcs}
        assert "cached_func" in func_names
        assert "compute" in func_names

        refs = extract_references(code, "python")
        assert any(
            r.from_symbol == "cached_func" and r.to_symbol == "compute"
            for r in refs
        )


class TestMethodCallExtraction:
    """Test method call (obj.method()) extraction across languages."""

    def test_python_method_call(self):
        code = """
class Dog:
    def speak(self):
        return "woof"

def main():
    dog = Dog()
    dog.speak()
"""
        refs = extract_references(code, "python")
        assert any(r.to_symbol == "speak" and r.kind == "calls" for r in refs)

    def test_python_chained_call(self):
        code = """
def process():
    result = obj.method().chain()
"""
        refs = extract_references(code, "python")
        calls = [r.to_symbol for r in refs if r.kind == "calls"]
        assert "method" in calls
        assert "chain" in calls

    def test_typescript_method_call(self):
        code = """
function main() {
    const dog = new Dog();
    dog.speak();
}
"""
        refs = extract_references(code, "typescript")
        assert any(r.to_symbol == "speak" and r.kind == "calls" for r in refs)

    def test_php_method_call(self):
        code = """<?php
function main() {
    $dog = new Dog();
    $dog->speak();
}
"""
        refs = extract_references(code, "php")
        assert any(r.to_symbol == "speak" and r.kind == "calls" for r in refs)


class TestMethodChainCapture:
    """Test full method chain extraction."""

    def test_python_three_method_chain(self):
        code = '''
def caller():
    obj.alpha().beta().gamma()
'''
        refs = extract_references(code, "python")
        calls = [r.to_symbol for r in refs if r.kind == "calls"]
        assert "alpha" in calls
        assert "beta" in calls
        assert "gamma" in calls

    def test_python_single_method_call_unchanged(self):
        """Single method call should still work."""
        code = '''
def caller():
    obj.method()
'''
        refs = extract_references(code, "python")
        calls = [r.to_symbol for r in refs if r.kind == "calls"]
        assert "method" in calls

    def test_python_simple_function_call_unchanged(self):
        """Direct function call (no chain) should still work."""
        code = '''
def caller():
    helper()
'''
        refs = extract_references(code, "python")
        assert any(r.to_symbol == "helper" and r.kind == "calls" for r in refs)

    def test_typescript_chain(self):
        code = '''
function caller() {
    arr.filter(x => x > 0).map(x => x * 2).reduce((a, b) => a + b);
}
'''
        refs = extract_references(code, "typescript")
        calls = [r.to_symbol for r in refs if r.kind == "calls"]
        assert "filter" in calls
        assert "map" in calls
        assert "reduce" in calls

    def test_php_chain(self):
        code = '''<?php
function caller() {
    $db->query()->fetch()->validate();
}
'''
        refs = extract_references(code, "php")
        calls = [r.to_symbol for r in refs if r.kind == "calls"]
        assert "query" in calls
        assert "fetch" in calls
        assert "validate" in calls

    def test_python_chain_with_initial_call(self):
        """get_db().query().fetch() — initial function call + chain."""
        code = '''
def caller():
    get_db().query().fetch()
'''
        refs = extract_references(code, "python")
        calls = [r.to_symbol for r in refs if r.kind == "calls"]
        assert "get_db" in calls
        assert "query" in calls
        assert "fetch" in calls


class TestEdgeDeduplication:
    """Test that build_edges deduplicates identical edges."""

    def test_duplicate_calls_deduplicated(self):
        symbols = [
            Symbol(name="func_a", kind="function", line=1, col=0),
            Symbol(name="func_b", kind="function", line=5, col=0),
        ]
        refs = [
            Reference(from_symbol="func_a", to_symbol="func_b", kind="calls", line=2),
            Reference(from_symbol="func_a", to_symbol="func_b", kind="calls", line=3),
            Reference(from_symbol="func_a", to_symbol="func_b", kind="calls", line=4),
        ]
        edges = build_edges(symbols, refs)
        call_edges = [e for e in edges if e.type == "calls"]
        assert len(call_edges) == 1

    def test_different_targets_not_deduplicated(self):
        symbols = [
            Symbol(name="a", kind="function", line=1, col=0),
            Symbol(name="b", kind="function", line=5, col=0),
            Symbol(name="c", kind="function", line=9, col=0),
        ]
        refs = [
            Reference(from_symbol="a", to_symbol="b", kind="calls", line=2),
            Reference(from_symbol="a", to_symbol="c", kind="calls", line=3),
        ]
        edges = build_edges(symbols, refs)
        assert len(edges) == 2


class TestTypeScriptImplements:
    """Test TypeScript implements clause extraction."""

    def test_class_implements_interface(self):
        """Note: tree-sitter-javascript doesn't have implements_clause.
        This tests what happens with JS grammar (which TS uses here)."""
        code = """
class Animal {
    speak() {}
}

class Dog extends Animal {
    bark() {}
}
"""
        refs = extract_references(code, "typescript")
        assert any(r.kind == "extends" and r.to_symbol == "Animal" for r in refs)


class TestPHPNamespaces:
    """Test PHP namespace-qualified symbol names."""

    def test_namespaced_class(self):
        code = """<?php
namespace App\\Model;

class User {
    public function getName() {
        return $this->name;
    }
}
"""
        symbols = extract_symbols(code, "php")
        classes = [s for s in symbols if s.kind == "class"]
        assert len(classes) == 1
        assert "App\\Model\\User" == classes[0].name

    def test_namespaced_function(self):
        code = """<?php
namespace App\\Utils;

function helper() {
    return true;
}
"""
        symbols = extract_symbols(code, "php")
        funcs = [s for s in symbols if s.kind == "function"]
        assert len(funcs) == 1
        assert "App\\Utils\\helper" == funcs[0].name

    def test_no_namespace_unchanged(self):
        code = """<?php
function standalone() {
    return true;
}
"""
        symbols = extract_symbols(code, "php")
        funcs = [s for s in symbols if s.kind == "function"]
        assert funcs[0].name == "standalone"


class TestEdgeBuilding:
    """Test graph edge construction."""

    def test_build_edges_from_symbols_and_refs(self):
        symbols = [
            Symbol(name="func_a", kind="function", line=1, col=0),
            Symbol(name="func_b", kind="function", line=5, col=0),
        ]
        refs = [
            Reference(from_symbol="func_a", to_symbol="func_b", kind="calls", line=2),
        ]
        edges = build_edges(symbols, refs)
        assert len(edges) == 1
        assert edges[0].from_name == "func_a"
        assert edges[0].to_name == "func_b"
        assert edges[0].type == "calls"

    def test_edge_weight_default(self):
        symbols = [
            Symbol(name="a", kind="function", line=1, col=0),
            Symbol(name="b", kind="function", line=5, col=0),
        ]
        refs = [Reference(from_symbol="a", to_symbol="b", kind="calls", line=2)]
        edges = build_edges(symbols, refs)
        assert edges[0].weight == 1.0


class TestParseAndExtract:
    """Test the convenience parse_and_extract function."""

    def test_parse_python_file(self):
        code = """
def process(data):
    return transform(data)

def transform(d):
    return d
"""
        symbols, refs, edges = parse_and_extract("test.py", code)
        assert len(symbols) >= 2
        assert any(s.name == "process" for s in symbols)
        assert any(s.name == "transform" for s in symbols)

    def test_parse_typescript_file(self):
        code = """
function process(data: any) {
    return transform(data);
}

function transform(d: any) {
    return d;
}
"""
        symbols, refs, edges = parse_and_extract("test.ts", code)
        assert len(symbols) >= 2

    def test_parse_php_file(self):
        code = """<?php
function process($data) {
    return transform($data);
}

function transform($d) {
    return $d;
}
"""
        symbols, refs, edges = parse_and_extract("test.php", code)
        assert len(symbols) >= 2


class TestLazyGrammarLoading:
    """Test lazy loading of tree-sitter grammars."""

    def test_python_loads(self):
        from tessera.parser import _load_language
        lang = _load_language("python")
        assert lang is not None

    def test_typescript_loads(self):
        from tessera.parser import _load_language
        lang = _load_language("typescript")
        assert lang is not None

    def test_javascript_loads(self):
        from tessera.parser import _load_language
        lang = _load_language("javascript")
        assert lang is not None

    def test_php_loads(self):
        from tessera.parser import _load_language
        lang = _load_language("php")
        assert lang is not None

    def test_grammar_caching(self):
        from tessera.parser import _load_language, _grammar_cache
        _grammar_cache.clear()
        lang1 = _load_language("python")
        lang2 = _load_language("python")
        assert lang1 is lang2  # Should be the same cached instance

    def test_unsupported_raises(self):
        from tessera.parser import _load_language
        with pytest.raises(ValueError, match="not installed"):
            _load_language("brainfuck")


class TestGenericExtractors:
    """Test generic symbol and reference extractors for unsupported languages."""

    def test_generic_extract_symbols_basic(self):
        # Test that generic extractor doesn't crash (approximate extraction)
        from tessera.parser import extract_symbols

        # Use a language that would fall back to generic extractor
        # Since we don't have Go/Rust installed, test via PHP to verify structure
        code = """<?php
function helper() {}
class Calculator {}
"""
        symbols = extract_symbols(code, "php")
        assert len(symbols) >= 2

    def test_generic_extract_references_basic(self):
        # Test that generic extractor doesn't crash
        from tessera.parser import extract_references

        code = """<?php
function main() {
    helper();
}
function helper() {}
"""
        refs = extract_references(code, "php")
        # Generic extraction may not catch all references, but shouldn't crash
        assert isinstance(refs, list)


class TestTypeScriptTypeReferences:
    """Test type reference extraction from TypeScript code."""

    def _type_refs(self, code: str) -> list[Reference]:
        """Helper: extract references and return only type_reference kind."""
        refs = extract_references(code, "typescript")
        return [r for r in refs if r.kind == "type_reference"]

    def _all_refs(self, code: str) -> list[Reference]:
        """Helper: extract all references."""
        return extract_references(code, "typescript")

    def test_variable_type_annotation(self):
        code = "const a: Foo = {}"
        refs = self._type_refs(code)
        assert len(refs) == 1
        assert refs[0].to_symbol == "Foo"
        assert refs[0].from_symbol == "<module>"

    def test_parameter_type(self):
        code = "function bar(x: Foo) {}"
        refs = self._type_refs(code)
        assert len(refs) == 1
        assert refs[0].to_symbol == "Foo"
        assert refs[0].from_symbol == "bar"

    def test_return_type(self):
        code = "function bar(): Foo { return {} as any }"
        refs = self._type_refs(code)
        assert len(refs) == 1
        assert refs[0].to_symbol == "Foo"
        assert refs[0].from_symbol == "bar"

    def test_generic_type(self):
        code = "const a: Array<Foo> = []"
        refs = self._type_refs(code)
        names = {r.to_symbol for r in refs}
        assert "Array" in names
        assert "Foo" in names

    def test_union_type(self):
        code = "type X = Foo | Bar"
        refs = self._type_refs(code)
        names = {r.to_symbol for r in refs}
        assert names == {"Foo", "Bar"}

    def test_intersection_type(self):
        code = "type X = Foo & Bar"
        refs = self._type_refs(code)
        names = {r.to_symbol for r in refs}
        assert names == {"Foo", "Bar"}

    def test_as_expression(self):
        code = "const x = {} as Foo"
        refs = self._type_refs(code)
        assert len(refs) == 1
        assert refs[0].to_symbol == "Foo"

    def test_satisfies_expression(self):
        code = "const x = {} satisfies Foo"
        refs = self._type_refs(code)
        assert len(refs) == 1
        assert refs[0].to_symbol == "Foo"

    def test_type_predicate(self):
        code = "function isFoo(x: unknown): x is Foo { return true }"
        refs = self._type_refs(code)
        names = {r.to_symbol for r in refs}
        assert "Foo" in names

    def test_conditional_type(self):
        code = "type Cond = Foo extends Bar ? Baz : never"
        refs = self._type_refs(code)
        names = {r.to_symbol for r in refs}
        assert "Foo" in names
        assert "Bar" in names
        assert "Baz" in names
        # 'never' is predefined_type, should NOT be in refs
        assert "never" not in names

    def test_generic_constraint(self):
        code = "function foo<T extends Foo>(x: T): void {}"
        refs = self._type_refs(code)
        names = {r.to_symbol for r in refs}
        assert "Foo" in names
        # T should be skipped (single letter)
        assert "T" not in names

    def test_interface_extends_generic(self):
        code = "interface X extends Array<Foo> {}"
        refs = self._type_refs(code)
        names = {r.to_symbol for r in refs}
        assert "Array" in names
        assert "Foo" in names

    def test_nested_type_identifier(self):
        code = "type NS = Namespace.Foo"
        refs = self._type_refs(code)
        assert len(refs) == 1
        assert refs[0].to_symbol == "Namespace.Foo"

    def test_no_primitive_refs(self):
        code = "const a: string = ''; const b: number = 0; const c: boolean = true"
        refs = self._type_refs(code)
        assert len(refs) == 0

    def test_no_self_ref_in_alias(self):
        code = "type Foo = Foo[]"
        refs = self._type_refs(code)
        # declaration_name filter skips all occurrences of 'Foo' in the RHS
        # (recursive type references are self-referential, not cross-type)
        assert len(refs) == 0

    def test_alias_refs_other_types(self):
        code = "type Foo = Bar & Baz"
        refs = self._type_refs(code)
        names = {r.to_symbol for r in refs}
        # Foo (the declaration) should NOT appear; Bar and Baz should
        assert "Foo" not in names
        assert names == {"Bar", "Baz"}

    def test_skips_single_letter_type_params(self):
        code = "function id<T>(x: T): T { return x }"
        refs = self._type_refs(code)
        names = {r.to_symbol for r in refs}
        assert "T" not in names

    def test_existing_call_refs_preserved(self):
        code = """
function main() {
    const x: Foo = bar()
}
"""
        all_refs = self._all_refs(code)
        call_refs = [r for r in all_refs if r.kind == "calls"]
        type_refs = [r for r in all_refs if r.kind == "type_reference"]
        assert any(r.to_symbol == "bar" for r in call_refs)
        assert any(r.to_symbol == "Foo" for r in type_refs)
