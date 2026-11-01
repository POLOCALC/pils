"""Quick look at a flight: one Markdown report of its data and their quality.

:class:`QuickLook` loads a flight's raw files and writes, in a new revision::

    flight_YYYYMMDD_hhmm/proc/quicklook/rev_YYYYMMDD_hhmmss/
    ├── report.md            # the report (links the plots below)
    ├── plots/*.png
    ├── <table>.parquet      # the report's tables (inventory, sensors, ...)
    └── manifest.json        # written last, as in pils.store

The report has a warnings summary first, then one section per topic: the
flight (STOUT), the files, the drone, the payload sensors, the
synchronisation, the GNSS data for PPK (rover and base) and the AZEL of the
telescopes. A section whose data is missing says why instead.

Nothing is changed in the flight folder except ``proc/quicklook``; with
``output_dir`` the report goes to ``<output_dir>/<flight>/proc/quicklook``.
The PPK tuning (``ppk_template``) runs in a temporary folder and only its
results are kept in the report's revision.

Examples
--------
>>> from pils.analyze.quicklook import QuickLook
>>> report = QuickLook.from_flight("flight_20251208_1927").run()  # doctest: +SKIP
>>> print(report)  # doctest: +SKIP
.../flight_20251208_1927/proc/quicklook/rev_20261017_180102/report.md

The package:

- :mod:`~pils.analyze.quicklook.report`: :class:`QuickLook`, the report and
  its sections
- :mod:`~pils.analyze.quicklook.gnss`: the GNSS, PPK and AZEL sections
- :mod:`~pils.analyze.quicklook.rinex`: reading RINEX observations
- :mod:`~pils.analyze.quicklook.common`: thresholds, :class:`Section`,
  formatting helpers and data checks

Every name of the former ``pils/analyze/quicklook.py`` module is re-exported
here.
"""

from pils.analyze.quicklook.common import (  # noqa: F401
    AIRBORNE_HEIGHT_M,
    COLORS,
    DJI_RTK_FLAGS,
    MAX_CLOCK_OFFSET_S,
    MAX_GAP_S,
    MIN_CORRELATION,
    MIN_COVERAGE_PCT,
    MIN_LITCHI_OVERLAP_PCT,
    QUICKLOOK_DIR,
    REPORT_FILE,
    SNR_LEVELS,
    Section,
    _fmt,
    _seconds,
    _size,
    _utc,
    airborne_interval,
    local_en,
    md_table,
    offset_rows,
    offset_warnings,
    overlap_pct,
    rtk_flag_summary,
    table_summary,
)
from pils.analyze.quicklook.gnss import QuickLookGNSSMixin  # noqa: F401
from pils.analyze.quicklook.report import (  # noqa: F401
    QuickLook,
    _collect_warnings,
    _WarningCollector,
)
from pils.analyze.quicklook.rinex import (  # noqa: F401
    _rinex_time,
    band_table,
    common_satellites,
    read_rinex,
)
