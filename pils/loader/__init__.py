"""Loader subpackage initialiser.

Expose the main loader implementations under the `pils.loader` namespace
so callers can import `from pils.loader import StoutLoader, PathLoader`.
"""

from pils.loader.path import PathLoader
from pils.loader.stout import StoutLoader

__all__ = ["StoutLoader", "PathLoader"]
