"""Tests for Ruby language extractor: symbols, references, and graph edges."""

from tessera.parser import (
    build_edges,
    detect_language,
    extract_references,
    extract_symbols,
    parse_and_extract,
)


class TestRubyLanguageDetection:

    def test_detect_ruby(self):
        assert detect_language("app.rb") == "ruby"
        assert detect_language("/path/to/model.rb") == "ruby"


class TestRubySymbolExtraction:

    def test_top_level_method(self):
        code = """
def helper(name)
  puts name
end
"""
        symbols = extract_symbols(code, "ruby")
        funcs = [s for s in symbols if s.kind == "function"]
        assert len(funcs) == 1
        assert funcs[0].name == "helper"
        assert funcs[0].scope == ""

    def test_class_method(self):
        code = """
class Gateway
  def charge(amount)
    amount * 100
  end
end
"""
        symbols = extract_symbols(code, "ruby")
        classes = [s for s in symbols if s.kind == "class"]
        methods = [s for s in symbols if s.kind == "method"]
        assert len(classes) == 1
        assert classes[0].name == "Gateway"
        assert len(methods) == 1
        assert methods[0].name == "charge"
        assert methods[0].scope == "Gateway"

    def test_module(self):
        code = """
module Payments
  class Gateway
  end
end
"""
        symbols = extract_symbols(code, "ruby")
        modules = [s for s in symbols if s.name == "Payments"]
        assert len(modules) == 1
        assert modules[0].kind == "class"  # Module treated as class
        # Nested class should have qualified scope
        nested = [s for s in symbols if "Gateway" in s.name]
        assert len(nested) == 1
        assert nested[0].scope == "Payments"

    def test_constant(self):
        code = """
TIMEOUT = 30
MAX_RETRIES = 3
"""
        symbols = extract_symbols(code, "ruby")
        consts = [s for s in symbols if s.kind == "variable"]
        assert len(consts) == 2
        names = {s.name for s in consts}
        assert names == {"TIMEOUT", "MAX_RETRIES"}

    def test_require_as_import(self):
        code = """
require "json"
require_relative "./helpers"
"""
        symbols = extract_symbols(code, "ruby")
        imports = [s for s in symbols if s.kind == "import"]
        assert len(imports) == 2

    def test_multiple_methods_in_class(self):
        code = """
class Service
  def initialize(config)
    @config = config
  end

  def start
    run
  end

  def stop
    cleanup
  end
end
"""
        symbols = extract_symbols(code, "ruby")
        methods = [s for s in symbols if s.kind == "method"]
        assert len(methods) == 3
        names = {s.name for s in methods}
        assert names == {"initialize", "start", "stop"}
        assert all(s.scope == "Service" for s in methods)


class TestRubyReferenceExtraction:

    def test_simple_call_with_parens(self):
        code = """
def helper
end

def main
  helper()
end
"""
        refs = extract_references(code, "ruby")
        assert any(r.kind == "calls" and r.to_symbol == "helper" for r in refs)

    def test_bare_call_not_detected(self):
        """Ruby bare calls (no parens, no receiver) are ambiguous at AST level."""
        code = """
def helper
end

def main
  helper
end
"""
        refs = extract_references(code, "ruby")
        # Bare identifier without parens is not a `call` node in tree-sitter
        call_names = {r.to_symbol for r in refs if r.kind == "calls"}
        assert "helper" not in call_names  # Known limitation

    def test_method_call_on_object(self):
        code = """
class Foo
  def bar
    response.body
  end
end
"""
        refs = extract_references(code, "ruby")
        assert any(r.kind == "calls" and r.to_symbol == "body" for r in refs)

    def test_class_inheritance(self):
        code = """
class Transaction < BaseRecord
end
"""
        refs = extract_references(code, "ruby")
        extends = [r for r in refs if r.kind == "extends"]
        assert len(extends) == 1
        assert extends[0].to_symbol == "BaseRecord"
        assert extends[0].from_symbol == "Transaction"

    def test_include_mixin(self):
        code = """
class Model
  include Serializable
end
"""
        refs = extract_references(code, "ruby")
        impl = [r for r in refs if r.kind == "implements"]
        assert len(impl) == 1
        assert impl[0].to_symbol == "Serializable"

    def test_extend_mixin(self):
        code = """
class Model
  extend ClassMethods
end
"""
        refs = extract_references(code, "ruby")
        impl = [r for r in refs if r.kind == "implements"]
        assert len(impl) == 1
        assert impl[0].to_symbol == "ClassMethods"

    def test_builtin_calls_skipped(self):
        code = """
def test
  puts "hello"
  raise "error"
end
"""
        refs = extract_references(code, "ruby")
        call_names = {r.to_symbol for r in refs if r.kind == "calls"}
        assert "puts" not in call_names
        assert "raise" not in call_names


class TestRubyGraphEdges:

    def test_method_containment_edge(self):
        code = """
class Gateway
  def charge
  end
end
"""
        symbols, refs, edges = parse_and_extract("test.rb", code)
        contains = [e for e in edges if e.type == "contains"]
        assert any(e.from_name == "Gateway" and e.to_name == "charge" for e in contains)

    def test_call_edge(self):
        code = """
def helper
end

def main
  helper()
end
"""
        symbols, refs, edges = parse_and_extract("test.rb", code)
        call_edges = [e for e in edges if e.type == "calls"]
        assert any(e.from_name == "main" and e.to_name == "helper" for e in call_edges)

    def test_extends_edge(self):
        code = """
class Base
end

class Child < Base
end
"""
        symbols, refs, edges = parse_and_extract("test.rb", code)
        extends_edges = [e for e in edges if e.type == "extends"]
        assert any(e.from_name == "Child" and e.to_name == "Base" for e in extends_edges)

    def test_implements_edge(self):
        code = """
module Serializable
end

class Model
  include Serializable
end
"""
        symbols, refs, edges = parse_and_extract("test.rb", code)
        impl_edges = [e for e in edges if e.type == "implements"]
        assert any(e.from_name == "Model" and e.to_name == "Serializable" for e in impl_edges)


class TestRubyParseAndExtract:

    def test_full_file(self):
        code = """
require "json"

module Payments
  class Gateway
    def initialize(key)
      @key = key
    end

    def charge(amount)
      Transaction.new(amount)
    end
  end

  class Transaction < BaseRecord
    include Serializable

    STATUSES = %w[pending completed].freeze

    def complete!
      update(status: :completed)
    end
  end
end

def helper
end
"""
        symbols, refs, edges = parse_and_extract("test.rb", code)

        sym_names = {s.name for s in symbols if s.name}
        assert "Payments" in sym_names
        assert any("Gateway" in n for n in sym_names)
        assert any("Transaction" in n for n in sym_names)
        assert "helper" in sym_names
        assert "STATUSES" in sym_names

        ref_kinds = {r.kind for r in refs}
        assert "calls" in ref_kinds
        assert "extends" in ref_kinds
        assert "implements" in ref_kinds

        edge_types = {e.type for e in edges}
        assert "contains" in edge_types
