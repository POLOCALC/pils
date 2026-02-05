"""Utilities for RTK data analysis and plotting."""

import matplotlib.pyplot as plt


class GNSSColors:
    """GNSS constellation color scheme for visualization."""

    # Constellation colors
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

    @classmethod
    def get_constellation_color(cls, constellation: str) -> str:
        """Get color for a constellation code (first character of satellite ID)."""
        return cls.CONSTELLATION_MAP.get(constellation, "#808080")

    @classmethod
    def apply_theme(cls, ax) -> None:
        """Apply consistent theme to plot axes."""
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_facecolor("#f5f5f5")
