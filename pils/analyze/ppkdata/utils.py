"""Utilities for RTK data analysis and plotting."""

import datetime

import polars as pl

# =============================================================================
# GNSS CONSTANTS
# =============================================================================

# Speed of light (m/s)
C = 299792458.0

# GNSS Frequencies in MHz
GNSS_FREQUENCIES = {
    "G": {"L1": 1575.42, "L2": 1227.60, "L5": 1176.45},
    "R": {"G1": 1602.0, "G2": 1246.0},  # Nominal, ignoring FDMA channels for MP
    "E": {
        "E1": 1575.42,
        "E5a": 1176.45,
        "E5b": 1207.14,
        "E6": 1278.75,
        "E5ab": 1191.795,
    },
    "C": {"B1": 1561.098, "B2": 1207.14, "B3": 1268.52},
    "S": {"L1": 1575.42},
    "J": {"L1": 1575.42, "L2": 1227.60, "L5": 1176.45},
}

# Constellation names for reporting/plotting
CONSTELLATION_NAMES = {
    "G": "GPS",
    "R": "GLONASS",
    "E": "Galileo",
    "C": "BeiDou",
    "S": "SBAS",
    "J": "QZSS",
    "I": "IRNSS",
}

# RTKLIB band classification
RTKLIB_BANDS = {
    "single": ["L1", "G1", "E1", "B1"],  # Primary bands
    "dual": ["L2", "G2", "E5b", "B2"],  # Secondary bands
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_frequency_band(constellation: str, freq_code: str) -> str:
    """Map constellation and frequency code to band name.

    Args:
        constellation: GNSS constellation code (G, R, E, C, J, S)
        freq_code: Frequency code from RINEX observation

    Returns:
        Band name (e.g., 'L1', 'E5a', 'B2')

    Examples:
        >>> get_frequency_band('G', '1')
        'L1'
        >>> get_frequency_band('E', '5')
        'E5a'
    """
    if constellation == "C":
        return {"2": "B1", "7": "B2", "6": "B3", "1": "B1C", "5": "B2a"}.get(
            freq_code, f"B{freq_code}"
        )
    if constellation == "E":
        return {"1": "E1", "5": "E5a", "7": "E5b", "8": "E5ab", "6": "E6"}.get(
            freq_code, f"E{freq_code}"
        )
    if constellation == "R":
        return {"1": "G1", "2": "G2"}.get(freq_code, f"G{freq_code}")
    return {"1": "L1", "2": "L2", "5": "L5"}.get(freq_code, f"L{freq_code}")


def get_dual_freq_bands(constellation: str) -> tuple[str, str | None]:
    """Get primary and secondary frequency bands for a constellation.

    Args:
        constellation: GNSS constellation code (G, R, E, C, J, S)

    Returns:
        Tuple of (primary_band, secondary_band) or (primary_band, None)

    Examples:
        >>> get_dual_freq_bands('G')
        ('L1', 'L2')
        >>> get_dual_freq_bands('E')
        ('E1', 'E5b')
    """
    return {
        "G": ("L1", "L2"),
        "R": ("G1", "G2"),
        "E": ("E1", "E5b"),
        "C": ("B1", "B2"),
        "S": ("L1", None),
        "J": ("L1", "L2"),
    }.get(constellation, (None, None))


# =============================================================================
# COLOR SCHEME
# =============================================================================


class GNSSColors:
    """GNSS constellation color scheme for visualization.

    Provides consistent color coding for GNSS constellations, solution
    quality indicators, and frequency bands across all plots.
    """

    # Constellation colors (matching industry standards)
    GPS = "#1f77b4"  # Blue
    GLONASS = "#ff7f0e"  # Orange
    GALILEO = "#2ca02c"  # Green
    BEIDOU = "#d62728"  # Red
    QZSS = "#9467bd"  # Purple
    IRNSS = "#8c564b"  # Brown
    SBAS = "#e377c2"  # Pink

    # Solution quality colors
    FIX = "#2ca02c"  # Green
    FLOAT = "#ff7f0e"  # Orange
    SINGLE = "#d62728"  # Red

    # Band colors
    BAND_PRIMARY = "#1f77b4"  # Blue (L1/G1/E1/B1)
    BAND_SECONDARY = "#ff7f0e"  # Orange (L2/G2/E5b/B2)
    BAND_TERTIARY = "#2ca02c"  # Green (L5/E5a/B2a)

    CONSTELLATION_MAP = {
        "G": GPS,
        "R": GLONASS,
        "E": GALILEO,
        "C": BEIDOU,
        "J": QZSS,
        "I": IRNSS,
        "S": SBAS,
    }

    # Backwards compatibility alias
    CONSTELLATION_COLORS = CONSTELLATION_MAP

    @classmethod
    def get_constellation_color(cls, constellation: str) -> str:
        """Get color for a constellation code (first character of satellite ID).

        Args:
            constellation: GNSS constellation code ('G', 'R', 'E', 'C', etc.)

        Returns:
            Hex color code for the constellation

        Examples:
            >>> GNSSColors.get_constellation_color('G')
            '#1f77b4'
            >>> GNSSColors.get_constellation_color('E')
            '#2ca02c'
            >>> GNSSColors.get_constellation_color('X')  # Unknown
            '#808080'
        """
        return cls.CONSTELLATION_MAP.get(constellation, "#808080")

    @classmethod
    def apply_theme(cls, ax) -> None:
        """Apply consistent theme to plot axes.

        Adds grid, background color, and styling for consistent plot appearance.

        Args:
            ax: Matplotlib axes object

        Examples:
            >>> import matplotlib.pyplot as plt
            >>> fig, ax = plt.subplots()
            >>> ax.plot([1, 2, 3], [1, 4, 9])
            >>> GNSSColors.apply_theme(ax)
            >>> ax.set_title('Themed Plot')
        """
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_facecolor("#f5f5f5")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "C",
    "GNSS_FREQUENCIES",
    "CONSTELLATION_NAMES",
    "RTKLIB_BANDS",
    # Functions
    "get_frequency_band",
    "get_dual_freq_bands",
    # Classes
    "GNSSColors",
]


# =============================================================================
# TIME SYSTEMS
# =============================================================================

# GPS - UTC (leap seconds), from the date each value starts
GPS_UTC_LEAP_SECONDS = (
    ("2017-01-01", 18),
    ("2015-07-01", 17),
    ("2012-07-01", 16),
    ("2009-01-01", 15),
    ("2006-01-01", 14),
    ("1999-01-01", 13),
)
# Start of GPS time, as a UNIX timestamp (1980-01-06 00:00:00)
GPS_EPOCH_UNIX = 315964800.0


def gps_utc_offset(gps_time: pl.Expr) -> pl.Expr:
    """GPS - UTC in seconds for a GPS-time datetime expression.

    Examples
    --------
    >>> import polars as pl, datetime
    >>> df = pl.DataFrame({"t": [datetime.datetime(2025, 12, 6)]})
    >>> df.select(gps_utc_offset(pl.col("t")))["literal"][0]
    18
    """
    expr = pl.lit(0)
    for start, leap in reversed(GPS_UTC_LEAP_SECONDS):
        expr = (
            pl.when(gps_time >= datetime.datetime.fromisoformat(start))
            .then(pl.lit(leap))
            .otherwise(expr)
        )
    return expr


def to_utc(time: pl.Expr, system: str) -> pl.Expr:
    """A naive datetime in ``system`` (GPST, UTC or JST) as UTC (us, UTC zone)."""
    system = system.upper()
    if system == "GPST":
        time = time - pl.duration(seconds=gps_utc_offset(time))
    elif system == "JST":
        time = time - pl.duration(hours=9)
    elif system != "UTC":
        raise ValueError(f"Unknown time system {system!r}")
    return time.dt.cast_time_unit("us").dt.replace_time_zone("UTC")
