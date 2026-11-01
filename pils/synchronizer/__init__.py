"""
Synchronizer - GPS-based correlation synchronization.

This module provides correlation-based TIME synchronization with GPS payload
as the single source of truth. Uses cross-correlation to detect time offsets
between data sources.

Key Features:
- GPS payload as mandatory reference timebase
- NED position correlation for GPS sources (3D signal)
- Pitch angle correlation for inclinometer (1D signal)
- Sub-sample precision using parabolic interpolation
- Rich offset metadata for all sources
- Support for non-correlation sensors (ADC, IMU, etc.) that are already time-aligned

The package splits :class:`Synchronizer` by responsibility:

- :mod:`pils.synchronizer.core`: :class:`Synchronizer` and ``synchronize``
- :mod:`pils.synchronizer.sources`: the ``add_*`` methods
- :mod:`pils.synchronizer.offsets`: time offsets by correlation
- :mod:`pils.synchronizer.resample`: resampling tables on a time grid

Every name of the former ``pils/synchronizer.py`` module is re-exported here.
"""

from pils.synchronizer.core import Synchronizer
from pils.synchronizer.resample import (  # noqa: F401
    _ANGLE_PATTERNS,
    _CLOCK_FIELDS,
    _HELD_COLUMNS,
    REDUNDANT_TIME_COLUMNS,
    angle_period,
    interp_angle,
    interp_column,
    is_held_column,
    resample_table,
)

__all__ = [
    "REDUNDANT_TIME_COLUMNS",
    "Synchronizer",
    "angle_period",
    "interp_angle",
    "interp_column",
    "is_held_column",
    "resample_table",
]
