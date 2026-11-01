"""RINEX Quality Analyzer v5 - Polars Core (Feature Complete).

High-performance GNSS analysis with 100% geodetic logic fidelity.
"""

import logging
from pathlib import Path

import polars as pl

from ..utils import (  # noqa: F401  (re-exported, as before the split)
    GNSS_FREQUENCIES,
    RTKLIB_BANDS,
    C,
    GNSSColors,
    get_dual_freq_bands,
    get_frequency_band,
)
from .parsing import RINEXParsingMixin
from .quality import RINEXQualityMixin
from .signals import RINEXSignalMixin

logger = logging.getLogger(__name__)

# Backwards compatibility aliases
CONSTELLATION_COLORS = GNSSColors.CONSTELLATION_COLORS
RTKLIB_bands = RTKLIB_BANDS


# =============================================================================
# RINEX ANALYZER CLASS
# =============================================================================


class RINEXAnalyzer(RINEXParsingMixin, RINEXSignalMixin, RINEXQualityMixin):
    """High-performance RINEX quality analyzer using Polars.

    Analyzes GNSS RINEX observation files for data quality assessment,
    including SNR analysis, multipath detection, cycle slip detection,
    and satellite geometry metrics.

    Attributes
    ----------
    obspath : str
        Path to RINEX file
    filename : str
        Name of RINEX file
    df : pl.DataFrame
        Parsed observations data
    header_info : dict
        Header metadata from RINEX file
    obs_types : dict
        Observation types by constellation
    epochs : list
        List of observation epochs
    azel_df : pl.DataFrame
        Azimuth-elevation data for satellites
    glo_slots : dict
        GLONASS frequency slot assignments

    Examples
    --------
    >>> analyzer = RINEXAnalyzer('station.obs')
    >>> analyzer.parse()
    >>> quality = analyzer.assess_data_quality()
    >>> print(f"Quality score: {quality['score']}")
    """

    def __init__(self, obspath: Path, navpath: Path | None = None) -> None:
        """Initialize RINEX analyzer.

        Args:
            obspath: Path to RINEX observation file
        """
        self.obspath = obspath
        self.obsname = Path(obspath).name

        if navpath is not None:
            self.navpath = navpath
            self.navname = Path(navpath).name

        self.df_obs = pl.DataFrame()
        self.header_info = {}
        self.obs_types = {}
        self.epochs = []
        self.azel_df = pl.DataFrame()
        self.glo_slots = {}
