"""Tests for Swift language extractor: symbols, references, and graph edges."""

from tessera.parser import (
    build_edges,
    detect_language,
    extract_references,
    extract_symbols,
    parse_and_extract,
)


class TestSwiftLanguageDetection:

    def test_detect_swift(self):
        assert detect_language("main.swift") == "swift"
        assert detect_language("/path/to/ViewController.swift") == "swift"


class TestSwiftSymbolExtraction:

    def test_function(self):
        code = """
func greet(name: String) -> String {
    return "Hello \\(name)"
}
"""
        symbols = extract_symbols(code, "swift")
        funcs = [s for s in symbols if s.kind == "function"]
        assert len(funcs) == 1
        assert funcs[0].name == "greet"

    def test_class(self):
        code = """
class Manager {
    func add() {}
}
"""
        symbols = extract_symbols(code, "swift")
        classes = [s for s in symbols if s.kind == "class"]
        methods = [s for s in symbols if s.kind == "method"]
        assert len(classes) == 1
        assert classes[0].name == "Manager"
        assert len(methods) == 1
        assert methods[0].name == "add"
        assert methods[0].scope == "Manager"

    def test_struct(self):
        code = """
struct Point {
    let x: Double
    let y: Double
}
"""
        symbols = extract_symbols(code, "swift")
        structs = [s for s in symbols if s.kind == "class"]
        assert len(structs) == 1
        assert structs[0].name == "Point"

    def test_protocol(self):
        code = """
protocol Agent {
    func spawn() throws
    var name: String { get }
}
"""
        symbols = extract_symbols(code, "swift")
        protocols = [s for s in symbols if s.kind == "interface"]
        assert len(protocols) == 1
        assert protocols[0].name == "Agent"
        # Protocol method signatures
        methods = [s for s in symbols if s.kind == "method"]
        assert any(m.name == "spawn" for m in methods)

    def test_enum(self):
        code = """
enum Status {
    case active
    case inactive
}
"""
        symbols = extract_symbols(code, "swift")
        enums = [s for s in symbols if s.kind == "class"]
        assert len(enums) == 1
        assert enums[0].name == "Status"

    def test_extension_adds_to_scope(self):
        """Extension methods should use the extended type as scope."""
        code = """
struct Runtime {}

extension Runtime {
    func spawn() {}
}
"""
        symbols = extract_symbols(code, "swift")
        methods = [s for s in symbols if s.kind == "method"]
        assert len(methods) == 1
        assert methods[0].name == "spawn"
        assert methods[0].scope == "Runtime"

    def test_import(self):
        code = """
import Foundation
import SwiftUI
"""
        symbols = extract_symbols(code, "swift")
        imports = [s for s in symbols if s.kind == "import"]
        assert len(imports) == 2

    def test_property(self):
        code = """
let defaultTimeout: Int = 30
var globalState: String = ""
"""
        symbols = extract_symbols(code, "swift")
        vars_ = [s for s in symbols if s.kind == "variable"]
        assert len(vars_) == 2
        names = {s.name for s in vars_}
        assert names == {"defaultTimeout", "globalState"}


class TestSwiftReferenceExtraction:

    def test_simple_call(self):
        code = """
func helper() {}

func main() {
    helper()
}
"""
        refs = extract_references(code, "swift")
        assert any(r.kind == "calls" and r.to_symbol == "helper" for r in refs)

    def test_navigation_call(self):
        code = """
func main() {
    sessions.append(item)
}
"""
        refs = extract_references(code, "swift")
        assert any(r.kind == "calls" and r.to_symbol == "append" for r in refs)

    def test_class_inheritance(self):
        code = """
class Manager: ObservableObject {
}
"""
        refs = extract_references(code, "swift")
        extends = [r for r in refs if r.kind == "extends"]
        assert len(extends) == 1
        assert extends[0].to_symbol == "ObservableObject"
        assert extends[0].from_symbol == "Manager"

    def test_extension_conformance(self):
        code = """
protocol Agent {}

extension Runtime: Agent {
}
"""
        refs = extract_references(code, "swift")
        impl = [r for r in refs if r.kind == "implements"]
        assert len(impl) == 1
        assert impl[0].to_symbol == "Agent"
        assert impl[0].from_symbol == "Runtime"

    def test_param_type_references(self):
        code = """
func process(cfg: Config, handler: Handler) -> Result {
    return Result()
}
"""
        refs = extract_references(code, "swift")
        type_refs = [r for r in refs if r.kind == "type_reference"]
        type_names = {r.to_symbol for r in type_refs}
        assert "Config" in type_names
        assert "Handler" in type_names
        assert "Result" in type_names

    def test_builtin_types_skipped(self):
        code = """
func add(a: Int, b: String) -> Bool {
    return true
}
"""
        refs = extract_references(code, "swift")
        type_refs = [r for r in refs if r.kind == "type_reference"]
        type_names = {r.to_symbol for r in type_refs}
        assert "Int" not in type_names
        assert "String" not in type_names
        assert "Bool" not in type_names


class TestSwiftGraphEdges:

    def test_method_containment_edge(self):
        code = """
class Manager {
    func add() {}
}
"""
        symbols, refs, edges = parse_and_extract("test.swift", code)
        contains = [e for e in edges if e.type == "contains"]
        assert any(e.from_name == "Manager" and e.to_name == "add" for e in contains)

    def test_extension_containment_edge(self):
        code = """
struct Runtime {}

extension Runtime {
    func spawn() {}
}
"""
        symbols, refs, edges = parse_and_extract("test.swift", code)
        contains = [e for e in edges if e.type == "contains"]
        assert any(e.from_name == "Runtime" and e.to_name == "spawn" for e in contains)

    def test_call_edge(self):
        code = """
func helper() {}

func main() {
    helper()
}
"""
        symbols, refs, edges = parse_and_extract("test.swift", code)
        call_edges = [e for e in edges if e.type == "calls"]
        assert any(e.from_name == "main" and e.to_name == "helper" for e in call_edges)

    def test_extends_edge(self):
        code = """
class Base {}

class Manager: Base {
}
"""
        symbols, refs, edges = parse_and_extract("test.swift", code)
        extends_edges = [e for e in edges if e.type == "extends"]
        assert any(e.from_name == "Manager" and e.to_name == "Base" for e in extends_edges)

    def test_implements_edge(self):
        code = """
protocol Agent {}

struct Runtime {}

extension Runtime: Agent {
}
"""
        symbols, refs, edges = parse_and_extract("test.swift", code)
        impl_edges = [e for e in edges if e.type == "implements"]
        assert any(e.from_name == "Runtime" and e.to_name == "Agent" for e in impl_edges)


class TestSwiftParseAndExtract:

    def test_full_file(self):
        code = """
import Foundation

protocol Agent {
    func spawn() throws
}

class Base {}

class Runtime: Base {
    let name: String = ""
    func helper() {}
}

extension Runtime: Agent {
    func spawn() throws {}
}

func freeFunction() {
    let r = Runtime()
}
"""
        symbols, refs, edges = parse_and_extract("test.swift", code)

        sym_names = {s.name for s in symbols if s.name}
        assert "Agent" in sym_names
        assert "Base" in sym_names
        assert "Runtime" in sym_names
        assert "freeFunction" in sym_names

        ref_kinds = {r.kind for r in refs}
        assert "calls" in ref_kinds
        assert "extends" in ref_kinds
        assert "implements" in ref_kinds

        edge_types = {e.type for e in edges}
        assert "contains" in edge_types
        assert "extends" in edge_types
        assert "implements" in edge_types
