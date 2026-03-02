"""Shared exceptions for the Tessera package."""


class PathTraversalError(Exception):
    """Raised when a file path escapes the project root."""
