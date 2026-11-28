"""
PPK Analysis - Standalone Post-Processed Kinematic GPS Analysis.

This module provides standalone PPK analysis using RTKLIB with smart
execution logic, versioned storage, and parquet persistence.

Key Features
------------
- Smart re-run: Only executes if config changed or forced
- Auto-timestamp versioning (rev_YYYYMMDD_HHMMSS)
- Per-revision folder organization
- Parquet storage for all versions
- Complete separation from Flight class

Examples
--------
>>> from pils.analyze.ppk import PPKAnalysis
>>> from pils.flight import Flight
>>> # Create Flight object
>>> flight_info = {
...     "drone_data_folder_path": "/path/to/flight/drone",
...     "aux_data_folder_path": "/path/to/flight/aux"
... }
>>> flight = Flight(flight_info)
>>> # Create new analysis with Flight object
>>> ppk = PPKAnalysis(flight)
>>> ppk.run_analysis('config.conf')  # Only runs if config changed
>>> # Load existing
>>> ppk = PPKAnalysis.load(flight)
>>> latest = ppk.get_latest_version()
>>> print(latest.pos_data)  # Polars DataFrame

The package:

- :mod:`~pils.analyze.ppk.core`: :class:`PPKAnalysis` and running RTKLIB
- :mod:`~pils.analyze.ppk.inputs`: RINEX time spans, base station, overlap
- :mod:`~pils.analyze.ppk.storage`: saving, loading and listing revisions
- :mod:`~pils.analyze.ppk.versions`: :class:`PPKVersion`, legacy tables
"""

from pils.analyze.ppk.core import PPKAnalysis
from pils.analyze.ppk.versions import PPKVersion, legacy_ppk_tables

__all__ = ["PPKAnalysis", "PPKVersion", "legacy_ppk_tables"]
