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
        assert detect_language("script.js") == "typescript"  # JS uses TS grammar
        assert detect_language("app.jsx") == "typescript"

    def test_detect_php(self):
        assert detect_language("index.php") == "php"
        assert detect_language("/path/to/class.php") == "php"

    def test_detect_unknown(self):
        assert detect_language("file.txt") is None
        assert detect_language("file.rb") is None
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
        # Should extract at least "chain" (rightmost of the outermost call)
        method_calls = [r for r in refs if r.kind == "calls"]
        assert len(method_calls) > 0

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
