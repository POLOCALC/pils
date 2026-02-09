"""RTKLIB Statistics File Analyzer."""

from pathlib import Path

import polars as pl


class STATAnalyzer:
    """Polars-based analyzer for RTKLIB .stat residual files.

    Parses RTKLIB processing statistics including phase/code residuals,
    SNR values, cycle slips, and rejection statistics.

    Attributes
    ----------
    filepath : str
        Path to .pos.stat file
    df : pl.DataFrame
        Parsed statistics data

    Examples
    --------
    >>> analyzer = STATAnalyzer('solution.pos.stat')
    >>> df = analyzer.parse()
    >>> sat_stats = analyzer.get_satellite_stats()
    >>> print(sat_stats.head())
    """

    def __init__(self, filepath: str | Path) -> None:
        """Initialize statistics analyzer.

        Args:
            filepath: Path to RTKLIB .stat file
        """
        self.filepath = filepath
        self.df = pl.DataFrame()

    def parse(self) -> pl.DataFrame:
        """Parse $SAT lines from .stat file into Polars DataFrame.

        Extracts satellite-level processing statistics including:
        - Phase and code residuals
        - SNR measurements
        - Cycle slip counts
        - Rejection counts
        - Lock status

        Returns:
            DataFrame with columns: tow, satellite, frequency, azimuth,
            elevation, resid_code, resid_phase, snr, slip, lock, reject

        Raises:
            FileNotFoundError: If .stat file doesn't exist

        Examples:
            >>> analyzer = STATAnalyzer('rtk.pos.stat')
            >>> df = analyzer.parse()
            >>> gps_stats = df.filter(pl.col('satellite').str.starts_with('G'))
            >>> print(f"GPS observations: {len(gps_stats)}")
        """
        if not Path(self.filepath).exists():
            raise FileNotFoundError(f"STAT file not found: {self.filepath}")

        # Use polars to read CSV-like $SAT lines
        # RTKLIB $SAT Format: 0:$SAT, 1:week, 2:tow, 3:sat, 4:freq, 5:az, 6:el, 7:resp, 8:resc, 9:vs, 10:snr, 11:fix, 12:slip, 13:lock, 14:outc, 15:slipc, 16:rejc

        sat_data = []
        with open(self.filepath) as f:
            for line in f:
                if line.startswith("$SAT"):
                    parts = line.strip().split(",")
                    try:
                        sat_data.append(
                            {
                                "tow": float(parts[2]),
                                "satellite": parts[3],
                                "frequency": int(parts[4]),
                                "azimuth": float(parts[5]),
                                "elevation": float(parts[6]),
                                "resid_code": float(parts[7]),
                                "resid_phase": float(parts[8]),
                                "snr": float(parts[10]),
                                "slip": int(parts[12]),
                                "lock": int(parts[13]),
                                "reject": int(parts[16]),
                            }
                        )
                    except (ValueError, IndexError):
                        continue

        if not sat_data:
            return pl.DataFrame()

        self.df = pl.DataFrame(sat_data).with_columns(
            [pl.col("satellite").str.slice(0, 1).alias("constellation")]
        )

        return self.df

    def get_satellite_stats(self):
        """Aggregates residuals and SNR per satellite.

        Returns:
            DataFrame with per-satellite statistics including:
            - avg_snr: Mean signal-to-noise ratio
            - avg_resid_phase: Mean absolute phase residual
            - avg_resid_code: Mean absolute code residual
            - p95_resid_phase: 95th percentile phase residual
            - total_slips: Total cycle slips detected
            - total_rejects: Total rejected observations

        Examples:
            >>> analyzer = STATAnalyzer('solution.pos.stat')
            >>> analyzer.parse()
            >>> sat_stats = analyzer.get_satellite_stats()
            >>> gps_stats = sat_stats.filter(pl.col('satellite').str.starts_with('G'))
            >>> print(f"GPS satellites: {gps_stats.height}")
            >>> high_residual = sat_stats.filter(pl.col('avg_resid_phase') > 0.05)
        """
        """
        Aggregates residuals and SNR per satellite.
        """
        if self.df.is_empty():
            return pl.DataFrame()

        return (
            self.df.group_by(["satellite", "frequency"])
            .agg(
                [
                    pl.col("snr").mean().alias("avg_snr"),
                    pl.col("resid_phase").abs().mean().alias("avg_resid_phase"),
                    pl.col("resid_code").abs().mean().alias("avg_resid_code"),
                    pl.col("resid_phase").abs().quantile(0.95).alias("p95_resid_phase"),
                    pl.col("slip").sum().alias("total_slips"),
                    pl.col("reject").sum().alias("total_rejects"),
                ]
            )
            .sort(["satellite", "frequency"])
        )

    def get_global_stats(self):
        """Aggregates residuals and SNR per frequency band.

        Returns
        -------
        pl.DataFrame
            DataFrame with global statistics by frequency:
            - mean_snr: Average SNR for the frequency
            - mean_resid_phase: Average phase residual
            - mean_resid_code: Average code residual

        Examples
        --------
        >>> analyzer = STATAnalyzer('solution.pos.stat')
        >>> analyzer.parse()
        >>> global_stats = analyzer.get_global_stats()
        >>> print(global_stats)
        """
        if self.df.is_empty():
            return pl.DataFrame()

        return (
            self.df.group_by("frequency")
            .agg(
                [
                    pl.col("snr").mean().alias("mean_snr"),
                    pl.col("resid_phase").abs().mean().alias("mean_resid_phase"),
                    pl.col("resid_code").abs().mean().alias("mean_resid_code"),
                ]
            )
            .sort("frequency")
        )
