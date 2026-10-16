"""Shared algorithm-plugin interfaces (guide 4.1)."""

from .plugin import (  # noqa: F401
    MethodContext,
    MethodSpec,
    BusinessPlugin,
    load_method,
    list_method_entries,
)

__all__ = ["MethodContext", "MethodSpec", "BusinessPlugin", "load_method", "list_method_entries"]
