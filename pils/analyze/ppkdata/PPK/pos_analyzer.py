"""RTKLIB Position Solution Analyzer."""

from pathlib import Path

import numpy as np
import polars as pl


class POSAnalyzer:
    """Polars-based analyzer for RTKLIB .pos solution files.

    Parses and analyzes RTK position solutions from RTKLIB,
    computing statistics and ENU offsets.

    Attributes
    ----------
    filepath : str
        Path to .pos file
    df : pl.DataFrame
        Parsed position data
    header : list
        Header lines from .pos file

    Examples
    --------
    >>> analyzer = POSAnalyzer('solution.pos')
    >>> df = analyzer.parse()
    >>> stats = analyzer.get_statistics()
    >>> print(f"Fix rate: {stats['fix_rate']:.1f}%")
    """

    def __init__(self, filepath: str | Path) -> None:
        """Initialize position analyzer.

        Args:
            filepath: Path to RTKLIB .pos file
        """
        self.filepath = filepath
        self.df = pl.DataFrame()
        self.header = []

    def parse(self) -> pl.DataFrame:
        """Parse .pos file into Polars DataFrame.

        Reads RTKLIB position solution file and converts to structured
        DataFrame with time, coordinates, quality indicators, and
        computed ENU offsets.

        Returns:
            DataFrame with columns: time, lat, lon, height, Q, ns,
            sdn, sde, sdu, age, ratio, E, N, U

        Raises:
            FileNotFoundError: If .pos file doesn't exist

        Examples:
            >>> analyzer = POSAnalyzer('rtk.pos')
            >>> df = analyzer.parse()
            >>> fix_epochs = df.filter(pl.col('Q') == 1)
            >>> print(f"Fix epochs: {len(fix_epochs)}")
        """
        if not Path(self.filepath).exists():
            raise FileNotFoundError(f"POS file not found: {self.filepath}")

        with open(self.filepath) as f:
            lines = f.readlines()

        # Extract header and data
        data_lines = [line for line in lines if not line.startswith("%")]

        # Simple parser for the standard RTKLIB POS format
        # Format: GPST, latitude(deg), longitude(deg), height(m), Q, ns, sdn(m), sde(m), sdu(m), sdne(m), sdeu(m), sdun(m), age(s), ratio

        records = []
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 15:
                try:
                    records.append(
                        {
                            "time": f"{parts[0]} {parts[1]}",
                            "lat": float(parts[2]),
                            "lon": float(parts[3]),
                            "height": float(parts[4]),
                            "Q": int(parts[5]),
                            "ns": int(parts[6]),
                            "sdn": float(parts[7]),
                            "sde": float(parts[8]),
                            "sdu": float(parts[9]),
                            "age": float(parts[13]),
                            "ratio": float(parts[14]),
                        }
                    )
                except (ValueError, IndexError):
                    continue

        if not records:
            return pl.DataFrame()

        self.df = pl.DataFrame(records).with_columns(
            [pl.col("time").str.to_datetime("%Y/%m/%d %H:%M:%S%.f")]
        )

        # Compute ENU Offsets if we have data
        if not self.df.is_empty():
            self._compute_enu()

        return self.df

    def _compute_enu(self):
        """Converts Lat/Lon/Height to ENU offsets relative to the mean position.

        Internal method that transforms geodetic coordinates to local
        East-North-Up frame for position analysis.

        Examples:
            >>> analyzer = POSAnalyzer('solution.pos')
            >>> df = analyzer.parse()
            >>> # ENU columns automatically computed after parse()
            >>> print(df.select(['east', 'north', 'up']).head())
        """
        # Mean position as origin
        lat_mean = self.df["lat"].mean()
        lon_mean = self.df["lon"].mean()
        if lat_mean is None or lon_mean is None:
            return
        # Convert to float explicitly
        lat0 = np.radians(float(lat_mean)) if lat_mean is not None else 0.0  # type: ignore
        lon0 = np.radians(float(lon_mean)) if lon_mean is not None else 0.0  # type: ignore
        h0_val = self.df["height"].mean()
        h0 = float(h0_val) if h0_val is not None else 0.0  # type: ignore

        # WGS84 Constants
        a = 6378137.0
        b = 6356752.314245
        e2 = 1 - (b**2 / a**2)

        def llh_to_ecef(lat, lon, h):
            lat_rad = np.deg2rad(lat)
            lon_rad = np.deg2rad(lon)
            N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)
            X = (N + h) * np.cos(lat_rad) * np.cos(lon_rad)
            Y = (N + h) * np.cos(lat_rad) * np.sin(lon_rad)
            Z = (N * (1 - e2) + h) * np.sin(lat_rad)
            return X, Y, Z

        X, Y, Z = llh_to_ecef(
            self.df["lat"].to_numpy(),
            self.df["lon"].to_numpy(),
            self.df["height"].to_numpy(),
        )
        X0, Y0, Z0 = llh_to_ecef(np.rad2deg(lat0), np.rad2deg(lon0), h0)

        dx = X - X0
        dy = Y - Y0
        dz = Z - Z0

        # ECEF to ENU rotation matrix
        e = -np.sin(lon0) * dx + np.cos(lon0) * dy
        n = (
            -np.sin(lat0) * np.cos(lon0) * dx
            - np.sin(lat0) * np.sin(lon0) * dy
            + np.cos(lat0) * dz
        )
        u = (
            np.cos(lat0) * np.cos(lon0) * dx
            + np.cos(lat0) * np.sin(lon0) * dy
            + np.sin(lat0) * dz
        )

        self.df = self.df.with_columns(
            [pl.Series("east", e), pl.Series("north", n), pl.Series("up", u)]
        )

    def get_statistics(self):
        """Calculate position statistics (fix rate, avg ratio, etc.).

        Returns:
            Dictionary with processing quality metrics:
            - total_epochs: Total number of position solutions
            - fix_epochs: Number of fixed ambiguity solutions
            - float_epochs: Number of float ambiguity solutions
            - single_epochs: Number of single point solutions
            - fix_rate: Percentage of fixed solutions
            - avg_ratio: Average ambiguity ratio
            - max_ratio: Maximum ambiguity ratio

        Examples:
            >>> analyzer = POSAnalyzer('solution.pos')
            >>> analyzer.parse()
            >>> stats = analyzer.get_statistics()
            >>> print(f"Fix rate: {stats['fix_rate']:.1f}%")
            >>> print(f"Average ratio: {stats['avg_ratio']:.2f}")
            >>> if stats['fix_rate'] < 95:
            ...     print("Warning: Low fix rate detected")
        """
        """
        Calculates position statistics (Fix rate, etc.)
        """
        if self.df.is_empty():
            return {}

        total_epochs = len(self.df)
        q_counts = self.df["Q"].value_counts()

        fix_count = (
            q_counts.filter(pl.col("Q") == 1)["count"].sum()
            if 1 in q_counts["Q"]
            else 0
        )
        float_count = (
            q_counts.filter(pl.col("Q") == 2)["count"].sum()
            if 2 in q_counts["Q"]
            else 0
        )
        single_count = (
            q_counts.filter(pl.col("Q") == 5)["count"].sum()
            if 5 in q_counts["Q"]
            else 0
        )

        return {
            "total_epochs": total_epochs,
            "fix_epochs": fix_count,
            "float_epochs": float_count,
            "single_epochs": single_count,
            "fix_rate": (fix_count / total_epochs) * 100 if total_epochs > 0 else 0,
            "avg_ratio": self.df["ratio"].mean(),
            "avg_ns": self.df["ns"].mean(),
            "max_ratio": self.df["ratio"].max(),
            "min_ratio": self.df["ratio"].min(),
        }
