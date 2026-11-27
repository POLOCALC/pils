"""RINEXAnalyzer: a RINEX observation file (and its navigation file), parsed and assessed."""

import logging
from pathlib import Path

import polars as pl

from .parsing import RINEXParsingMixin
from .quality import RINEXQualityMixin
from .signals import RINEXSignalMixin

logger = logging.getLogger(__name__)


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

        self.df_obs = pl.DataFrame()
        self.header_info = {}
        self.obs_types = {}
        self.epochs = []
        self.azel_df = pl.DataFrame()
        self.glo_slots = {}
