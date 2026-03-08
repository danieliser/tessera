"""Auto-discovery of language extractor modules.

Drop a .py file in this directory with a class that subclasses LanguageExtractor,
and it will be automatically registered. No dispatch chains to update.
"""

import importlib
import pkgutil
from pathlib import Path

from tessera.parser._base import LanguageExtractor

# Registry: language name → extractor instance
_EXTRACTORS: dict[str, LanguageExtractor] = {}

# Extension map: file extension → language name
_EXTENSION_MAP: dict[str, str] = {}


def _discover():
    """Scan this package for LanguageExtractor subclasses and register them."""
    package_path = str(Path(__file__).parent)
    for finder, module_name, is_pkg in pkgutil.iter_modules([package_path]):
        if module_name.startswith("_"):
            continue  # Skip private modules like _generic
        try:
            mod = importlib.import_module(f".{module_name}", __package__)
        except ImportError:
            continue
        for attr_name in dir(mod):
            cls = getattr(mod, attr_name)
            if (
                isinstance(cls, type)
                and issubclass(cls, LanguageExtractor)
                and cls is not LanguageExtractor
                and hasattr(cls, "language")
            ):
                instance = cls()
                _EXTRACTORS[instance.language] = instance
                for ext in instance.extensions:
                    _EXTENSION_MAP[ext] = instance.language

    # Also load _generic as fallback (not auto-registered since it starts with _)
    try:
        mod = importlib.import_module("._generic", __package__)
        for attr_name in dir(mod):
            cls = getattr(mod, attr_name)
            if (
                isinstance(cls, type)
                and issubclass(cls, LanguageExtractor)
                and cls is not LanguageExtractor
                and getattr(cls, "language", None) == "_generic"
            ):
                _EXTRACTORS["_generic"] = cls()
    except ImportError:
        pass


def get_extractor(language: str) -> LanguageExtractor | None:
    """Get the extractor for a language, falling back to generic."""
    if not _EXTRACTORS:
        _discover()
    return _EXTRACTORS.get(language) or _EXTRACTORS.get("_generic")


def detect_language_from_ext(file_path: str) -> str | None:
    """Detect language from file extension using registered extractors."""
    if not _EXTENSION_MAP:
        _discover()
    for ext, lang in _EXTENSION_MAP.items():
        if file_path.endswith(ext):
            return lang
    return None


def get_all_extractors() -> dict[str, LanguageExtractor]:
    """Return all registered extractors."""
    if not _EXTRACTORS:
        _discover()
    return dict(_EXTRACTORS)
