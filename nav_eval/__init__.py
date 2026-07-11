"""Navigation evaluation for trajectory-following methods."""

from .interfaces import Action, NavigationMethod
from .registry import get_method, register_method

# Register built-in methods.
from . import methods as _methods  # noqa: F401

__all__ = ["Action", "NavigationMethod", "get_method", "register_method"]
