"""Loader subpackage initialiser.

Expose the main loader implementations under the `pils.loader` namespace
so callers can import `from pils.loader import StoutLoader, PathLoader`.
"""

from .path import PathLoader
from .stout import StoutLoader

# Backwards compatibility: some code imports `StoutDataLoader` from `pils.loader`
StoutDataLoader = StoutLoader

__all__ = ["StoutLoader", "PathLoader"]
