"""Import-binding extraction for cross-file reference resolution."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class ImportBinding:
    """A local import name and the in-project module that defines it."""

    local_name: str
    target_name: str
    module_path: PurePosixPath
    language: str
    is_module: bool = False


_TYPESCRIPT_NAMED_IMPORT = re.compile(
    r"\bimport\s+(?:type\s+)?\{(?P<names>[^}]+)\}\s+from\s*"
    r"(?P<quote>['\"])(?P<module>[^'\"]+)(?P=quote)",
    re.MULTILINE,
)


def extract_import_bindings(
    file_path: str,
    language: str,
    source: str,
) -> dict[str, ImportBinding]:
    """Return local names that can be resolved to an in-project import target.

    Named imports and Python module imports are included. Python module imports
    are only used when a parser records a simple ``module.member()`` receiver.
    """
    if language == "python":
        return _extract_python_bindings(file_path, source)
    if language in {"typescript", "javascript"}:
        return _extract_typescript_bindings(file_path, language, source)
    return {}


def candidate_module_paths(binding: ImportBinding) -> set[str]:
    """Return normalized indexed-file paths that may implement a binding."""
    module = binding.module_path.as_posix()
    if binding.language == "python":
        return {f"{module}.py", f"{module}/__init__.py"}
    return {
        f"{module}.ts",
        f"{module}.tsx",
        f"{module}.js",
        f"{module}.jsx",
        f"{module}/index.ts",
        f"{module}/index.tsx",
        f"{module}/index.js",
        f"{module}/index.jsx",
    }


def _extract_python_bindings(file_path: str, source: str) -> dict[str, ImportBinding]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    bindings: dict[str, ImportBinding] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module_path = _python_module_path(file_path, node.module, node.level)
            if module_path is None:
                continue
            for imported in node.names:
                if imported.name == "*":
                    continue
                local_name = imported.asname or imported.name
                bindings[local_name] = ImportBinding(
                    local_name=local_name,
                    target_name=imported.name,
                    module_path=module_path,
                    language="python",
                )
        elif isinstance(node, ast.Import):
            for imported in node.names:
                # Without ``as``, ``import package.module`` binds ``package``.
                # Do not pretend that it binds the nested module; that would
                # turn uncertain resolution into a false-positive edge.
                if imported.asname:
                    local_name = imported.asname
                    module_name = imported.name
                elif "." not in imported.name:
                    local_name = imported.name
                    module_name = imported.name
                else:
                    continue
                bindings[local_name] = ImportBinding(
                    local_name=local_name,
                    target_name="",
                    module_path=PurePosixPath(*module_name.split(".")),
                    language="python",
                    is_module=True,
                )
    return bindings


def _python_module_path(
    file_path: str,
    module: str | None,
    level: int,
) -> PurePosixPath | None:
    current_dir = PurePosixPath(file_path).parent
    if level:
        base = current_dir
        for _ in range(level - 1):
            base = base.parent
        return base.joinpath(*(module.split(".") if module else ()))
    if not module:
        return None
    return PurePosixPath(*module.split("."))


def _extract_typescript_bindings(
    file_path: str,
    language: str,
    source: str,
) -> dict[str, ImportBinding]:
    bindings: dict[str, ImportBinding] = {}
    base_dir = PurePosixPath(file_path).parent
    for match in _TYPESCRIPT_NAMED_IMPORT.finditer(source):
        module = match.group("module")
        if not module.startswith("."):
            continue
        module_path = base_dir.joinpath(module)
        for raw_specifier in match.group("names").split(","):
            specifier = raw_specifier.strip().removeprefix("type ").strip()
            if not specifier:
                continue
            parts = re.split(r"\s+as\s+", specifier, maxsplit=1)
            target_name = parts[0].strip()
            local_name = parts[-1].strip()
            if not target_name or not local_name:
                continue
            bindings[local_name] = ImportBinding(
                local_name=local_name,
                target_name=target_name,
                module_path=module_path,
                language=language,
            )
    return bindings
