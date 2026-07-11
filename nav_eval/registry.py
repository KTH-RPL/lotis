"""Method registry for navigation evaluation runners."""

from __future__ import annotations

from typing import Dict, Type

from .interfaces import NavigationMethod


_METHODS: Dict[str, Type[NavigationMethod]] = {}


def register_method(name: str, method_cls: Type[NavigationMethod]) -> None:
    """Register an evaluation method class."""
    if not name:
        raise ValueError("Method name must be non-empty")
    _METHODS[name] = method_cls


def get_method(name: str) -> Type[NavigationMethod]:
    """Return a registered method class."""
    try:
        return _METHODS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_METHODS)) or "<none>"
        raise ValueError(f"Unknown method '{name}'. Available methods: {available}") from exc


def registered_methods() -> Dict[str, Type[NavigationMethod]]:
    """Return a copy of the method registry."""
    return dict(_METHODS)
