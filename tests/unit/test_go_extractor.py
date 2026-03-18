"""Tests for Go language extractor: symbols, references, and graph edges."""

import pytest

from tessera.parser import (
    Reference,
    Symbol,
    build_edges,
    detect_language,
    extract_references,
    extract_symbols,
    parse_and_extract,
)


class TestGoLanguageDetection:

    def test_detect_go(self):
        assert detect_language("main.go") == "go"
        assert detect_language("/path/to/handler.go") == "go"

    def test_detect_go_test(self):
        assert detect_language("handler_test.go") == "go"


class TestGoSymbolExtraction:

    def test_function(self):
        code = """
package main

func NewRuntime(name string) *Runtime {
    return &Runtime{name: name}
}
"""
        symbols = extract_symbols(code, "go")
        funcs = [s for s in symbols if s.kind == "function"]
        assert len(funcs) == 1
        assert funcs[0].name == "NewRuntime"

    def test_method_with_pointer_receiver(self):
        code = """
package main

func (r *Runtime) Spawn(ctx context.Context) error {
    return nil
}
"""
        symbols = extract_symbols(code, "go")
        methods = [s for s in symbols if s.kind == "method"]
        assert len(methods) == 1
        assert methods[0].name == "Spawn"
        assert methods[0].scope == "Runtime"

    def test_method_with_value_receiver(self):
        code = """
package main

func (r Runtime) Name() string {
    return r.name
}
"""
        symbols = extract_symbols(code, "go")
        methods = [s for s in symbols if s.kind == "method"]
        assert len(methods) == 1
        assert methods[0].name == "Name"
        assert methods[0].scope == "Runtime"

    def test_struct_type(self):
        code = """
package main

type Runtime struct {
    name string
}
"""
        symbols = extract_symbols(code, "go")
        structs = [s for s in symbols if s.kind == "class"]
        assert len(structs) == 1
        assert structs[0].name == "Runtime"

    def test_interface_type(self):
        code = """
package main

type Agent interface {
    Name() string
    Spawn(ctx context.Context) error
}
"""
        symbols = extract_symbols(code, "go")
        ifaces = [s for s in symbols if s.kind == "interface"]
        assert len(ifaces) == 1
        assert ifaces[0].name == "Agent"

    def test_type_alias(self):
        code = """
package main

type MyInt int
"""
        symbols = extract_symbols(code, "go")
        aliases = [s for s in symbols if s.kind == "type_alias"]
        assert len(aliases) == 1
        assert aliases[0].name == "MyInt"

    def test_const_block(self):
        code = """
package main

const (
    MaxRetries = 3
    DefaultTimeout = 30
)
"""
        symbols = extract_symbols(code, "go")
        consts = [s for s in symbols if s.kind == "variable"]
        assert len(consts) == 2
        names = {s.name for s in consts}
        assert names == {"MaxRetries", "DefaultTimeout"}

    def test_var_block(self):
        code = """
package main

var (
    ErrNotFound = fmt.Errorf("not found")
    globalState string
)
"""
        symbols = extract_symbols(code, "go")
        vars_ = [s for s in symbols if s.kind == "variable"]
        assert len(vars_) == 2
        names = {s.name for s in vars_}
        assert names == {"ErrNotFound", "globalState"}

    def test_blank_var_skipped(self):
        """var _ Agent = (*Runtime)(nil) should not create a variable symbol."""
        code = """
package main

var _ Agent = (*Runtime)(nil)
"""
        symbols = extract_symbols(code, "go")
        vars_ = [s for s in symbols if s.kind == "variable"]
        assert len(vars_) == 0

    def test_import_declaration(self):
        code = """
package main

import (
    "fmt"
    "context"
)
"""
        symbols = extract_symbols(code, "go")
        imports = [s for s in symbols if s.kind == "import"]
        assert len(imports) == 1  # One import declaration block

    def test_multiple_types(self):
        """Full file with mixed declarations."""
        code = """
package main

import "fmt"

type Runtime struct {
    name string
}

type Agent interface {
    Name() string
}

func NewRuntime() *Runtime {
    return &Runtime{}
}

func (r *Runtime) Name() string {
    return r.name
}

const Version = "1.0"
"""
        symbols = extract_symbols(code, "go")
        kinds = {s.kind for s in symbols}
        assert "import" in kinds
        assert "class" in kinds
        assert "interface" in kinds
        assert "function" in kinds
        assert "method" in kinds
        assert "variable" in kinds


class TestGoReferenceExtraction:

    def test_simple_function_call(self):
        code = """
package main

func helper() {}

func main() {
    helper()
}
"""
        refs = extract_references(code, "go")
        assert any(r.kind == "calls" and r.to_symbol == "helper" for r in refs)

    def test_package_qualified_call(self):
        code = """
package main

import "fmt"

func main() {
    fmt.Println("hello")
}
"""
        refs = extract_references(code, "go")
        assert any(r.kind == "calls" and r.to_symbol == "Println" for r in refs)

    def test_method_call_via_selector(self):
        code = """
package main

func main() {
    r := &Runtime{}
    r.Spawn(ctx)
}
"""
        refs = extract_references(code, "go")
        assert any(r.kind == "calls" and r.to_symbol == "Spawn" for r in refs)

    def test_struct_embedding_extends(self):
        code = """
package main

type Base struct {}

type Runtime struct {
    Base
    name string
}
"""
        refs = extract_references(code, "go")
        extends = [r for r in refs if r.kind == "extends"]
        assert len(extends) == 1
        assert extends[0].to_symbol == "Base"
        assert extends[0].from_symbol == "Runtime"

    def test_pointer_embedding(self):
        code = """
package main

type Runtime struct {
    *Base
    name string
}
"""
        refs = extract_references(code, "go")
        extends = [r for r in refs if r.kind == "extends"]
        assert len(extends) == 1
        assert extends[0].to_symbol == "Base"

    def test_interface_assertion(self):
        code = """
package main

type Agent interface {
    Name() string
}

type Runtime struct{}

var _ Agent = (*Runtime)(nil)
"""
        refs = extract_references(code, "go")
        impl = [r for r in refs if r.kind == "implements"]
        assert len(impl) == 1
        assert impl[0].from_symbol == "Runtime"
        assert impl[0].to_symbol == "Agent"

    def test_receiver_type_reference(self):
        code = """
package main

func (r *Runtime) Spawn() error {
    return nil
}
"""
        refs = extract_references(code, "go")
        type_refs = [r for r in refs if r.kind == "type_reference" and r.to_symbol == "Runtime"]
        assert len(type_refs) >= 1

    def test_param_type_references(self):
        code = """
package main

func Process(cfg Config, handler Handler) Result {
    return Result{}
}
"""
        refs = extract_references(code, "go")
        type_refs = [r for r in refs if r.kind == "type_reference"]
        type_names = {r.to_symbol for r in type_refs}
        assert "Config" in type_names
        assert "Handler" in type_names
        assert "Result" in type_names

    def test_builtin_types_skipped(self):
        code = """
package main

func Add(a int, b string) bool {
    return true
}
"""
        refs = extract_references(code, "go")
        type_refs = [r for r in refs if r.kind == "type_reference"]
        type_names = {r.to_symbol for r in type_refs}
        assert "int" not in type_names
        assert "string" not in type_names
        assert "bool" not in type_names

    def test_builtin_calls_skipped(self):
        code = """
package main

func main() {
    s := make([]int, 10)
    l := len(s)
    _ = append(s, 1)
}
"""
        refs = extract_references(code, "go")
        call_names = {r.to_symbol for r in refs if r.kind == "calls"}
        assert "make" not in call_names
        assert "len" not in call_names
        assert "append" not in call_names

    def test_struct_field_type_refs(self):
        code = """
package main

type Server struct {
    handler Handler
    config  *Config
}
"""
        refs = extract_references(code, "go")
        type_refs = [r for r in refs if r.kind == "type_reference"]
        type_names = {r.to_symbol for r in type_refs}
        assert "Handler" in type_names
        assert "Config" in type_names


class TestGoGraphEdges:
    """Test that Go references produce correct graph edges via build_edges()."""

    def test_method_containment_edge(self):
        code = """
package main

type Runtime struct{}

func (r *Runtime) Spawn() {}
"""
        symbols, refs, edges = parse_and_extract("test.go", code)
        contains = [e for e in edges if e.type == "contains"]
        assert any(
            e.from_name == "Runtime" and e.to_name == "Spawn"
            for e in contains
        )

    def test_call_edge(self):
        code = """
package main

func helper() {}

func main() {
    helper()
}
"""
        symbols, refs, edges = parse_and_extract("test.go", code)
        call_edges = [e for e in edges if e.type == "calls"]
        assert any(
            e.from_name == "main" and e.to_name == "helper"
            for e in call_edges
        )

    def test_extends_edge_from_embedding(self):
        code = """
package main

type Base struct{}

type Runtime struct {
    Base
    name string
}
"""
        symbols, refs, edges = parse_and_extract("test.go", code)
        extends_edges = [e for e in edges if e.type == "extends"]
        assert any(
            e.from_name == "Runtime" and e.to_name == "Base"
            for e in extends_edges
        )

    def test_implements_edge(self):
        code = """
package main

type Agent interface {
    Name() string
}

type Runtime struct{}

var _ Agent = (*Runtime)(nil)
"""
        symbols, refs, edges = parse_and_extract("test.go", code)
        impl_edges = [e for e in edges if e.type == "implements"]
        assert any(
            e.from_name == "Runtime" and e.to_name == "Agent"
            for e in impl_edges
        )

    def test_no_duplicate_edges(self):
        code = """
package main

func helper() {}

func main() {
    helper()
    helper()
}
"""
        symbols, refs, edges = parse_and_extract("test.go", code)
        call_edges = [
            (e.from_name, e.to_name, e.type)
            for e in edges if e.type == "calls"
        ]
        # build_edges deduplicates
        assert len(call_edges) == len(set(call_edges))


class TestGoParseAndExtract:
    """Integration: parse_and_extract returns all three outputs."""

    def test_full_file(self):
        code = """
package main

import "fmt"

type Agent interface {
    Name() string
}

type Base struct{}

type Runtime struct {
    Base
    name string
}

func NewRuntime() *Runtime {
    return &Runtime{}
}

func (r *Runtime) Name() string {
    fmt.Println(r.name)
    return r.name
}

func main() {
    r := NewRuntime()
    _ = r.Name()
}

var _ Agent = (*Runtime)(nil)
"""
        symbols, refs, edges = parse_and_extract("test.go", code)

        # Symbols
        sym_names = {s.name for s in symbols if s.name}
        assert "Agent" in sym_names
        assert "Runtime" in sym_names
        assert "Base" in sym_names
        assert "NewRuntime" in sym_names
        assert "Name" in sym_names
        assert "main" in sym_names

        # References should include calls, extends, implements, type_reference
        ref_kinds = {r.kind for r in refs}
        assert "calls" in ref_kinds
        assert "extends" in ref_kinds
        assert "implements" in ref_kinds
        assert "type_reference" in ref_kinds

        # Edges: contains, calls (NewRuntime is local), extends, implements
        edge_types = {e.type for e in edges}
        assert "contains" in edge_types
        assert "calls" in edge_types  # main → NewRuntime (both local symbols)
        assert "extends" in edge_types
        assert "implements" in edge_types
