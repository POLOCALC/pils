"""RTKLIB Position Solution Analyzer."""

from pathlib import Path

import numpy as np
import polars as pl

from pils.analyze.ppkdata.utils import to_utc


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
        self.header: list[str] = []
        self.time_system = "GPST"
        self.height_column = "h_ell_m"

    def parse(self) -> pl.DataFrame:
        """Parse .pos file into Polars DataFrame.

        Reads an RTKLIB position solution (``llh`` output, ``hms`` time) into
        a table with the names of :mod:`pils.schema`, plus ENU offsets from
        the mean position.

        Returns
        -------
            DataFrame with columns: datetime (UTC), timestamp (UNIX s, UTC),
            lat_deg, lon_deg, h_ell_m (h_msl_m if RTKLIB wrote geodetic
            heights), quality (1 fix, 2 float, 3 sbas, 4 dgps, 5 single,
            6 ppp), num_sats, sd_n_m, sd_e_m, sd_u_m, age_s, ratio, e_m, n_m,
            u_m. The time system of the file (GPST, UTC or JST, from its
            header) is converted to UTC.

        Raises
        ------
            FileNotFoundError: If .pos file doesn't exist

        Examples
        --------
            >>> analyzer = POSAnalyzer('rtk.pos')
            >>> df = analyzer.parse()
            >>> fix_epochs = df.filter(pl.col('quality') == 1)
            >>> print(f"Fix epochs: {len(fix_epochs)}")
        """
        if not Path(self.filepath).exists():
            raise FileNotFoundError(f"POS file not found: {self.filepath}")

        with open(self.filepath) as f:
            lines = f.readlines()

        self.header = [line for line in lines if line.startswith("%")]
        data_lines = [line for line in lines if not line.startswith("%")]
        self.time_system, self.height_column = self._header_info(self.header)

        # Format: time (2 fields), latitude(deg), longitude(deg), height(m), Q,
        # ns, sdn(m), sde(m), sdu(m), sdne(m), sdeu(m), sdun(m), age(s), ratio
        records = []
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 15:
                try:
                    records.append(
                        {
                            "time": f"{parts[0]} {parts[1]}",
                            "lat_deg": float(parts[2]),
                            "lon_deg": float(parts[3]),
                            self.height_column: float(parts[4]),
                            "quality": int(parts[5]),
                            "num_sats": int(parts[6]),
                            "sd_n_m": float(parts[7]),
                            "sd_e_m": float(parts[8]),
                            "sd_u_m": float(parts[9]),
                            "age_s": float(parts[13]),
                            "ratio": float(parts[14]),
                        }
                    )
                except (ValueError, IndexError):
                    continue

        if not records:
            return pl.DataFrame()

        utc = to_utc(
            pl.col("time").str.to_datetime("%Y/%m/%d %H:%M:%S%.f"), self.time_system
        )
        self.df = (
            pl.DataFrame(records)
            .with_columns(utc.alias("datetime"))
            .with_columns((pl.col("datetime").dt.epoch("us") / 1e6).alias("timestamp"))
            .select(
                "datetime", "timestamp", pl.exclude("time", "datetime", "timestamp")
            )
        )

        # Compute ENU Offsets if we have data
        if not self.df.is_empty():
            self._compute_enu()

        return self.df

    @staticmethod
    def _header_info(header: list[str]) -> tuple[str, str]:
        """Time system and height column from the .pos header.

        The column line starts with the time system (``GPST``, ``UTC`` or
        ``JST``); the legend says ``height=WGS84/ellipsoidal`` or
        ``geodetic`` (above mean sea level).
        """
        time_system = "GPST"
        height = "h_ell_m"
        for line in header:
            words = line.lstrip("%").split()
            if words and words[0] in ("GPST", "UTC", "JST"):
                time_system = words[0]
            if "height=" in line and "geodetic" in line.split("height=")[1]:
                height = "h_msl_m"
        return time_system, height

    def _compute_enu(self):
        """Convert Lat/Lon/Height to ENU offsets relative to the mean position.

        Internal method that transforms geodetic coordinates to local
        East-North-Up frame for position analysis.

        Examples
        --------
            >>> analyzer = POSAnalyzer('solution.pos')
            >>> df = analyzer.parse()
            >>> # ENU columns automatically computed after parse()
            >>> print(df.select(['e_m', 'n_m', 'u_m']).head())
        """
        # Mean position as origin
        lat_mean = self.df["lat_deg"].mean()
        lon_mean = self.df["lon_deg"].mean()
        if lat_mean is None or lon_mean is None:
            return
        # Convert to float explicitly
        lat0 = np.radians(float(lat_mean)) if lat_mean is not None else 0.0  # type: ignore
        lon0 = np.radians(float(lon_mean)) if lon_mean is not None else 0.0  # type: ignore
        height = self.df[self.height_column]
        h0_val = height.mean()
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
            self.df["lat_deg"].to_numpy(),
            self.df["lon_deg"].to_numpy(),
            height.to_numpy(),
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
            [pl.Series("e_m", e), pl.Series("n_m", n), pl.Series("u_m", u)]
        )

    def get_statistics(self):
        """Calculate position statistics (fix rate, avg ratio, etc.).

        Returns
        -------
            Dictionary with processing quality metrics:
            - total_epochs: Total number of position solutions
            - fix_epochs: Number of fixed ambiguity solutions
            - float_epochs: Number of float ambiguity solutions
            - single_epochs: Number of single point solutions
            - fix_rate: Percentage of fixed solutions
            - avg_ratio: Average ambiguity ratio
            - max_ratio: Maximum ambiguity ratio

        Examples
        --------
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
        quality = self.df["quality"]
        fix_count = int((quality == 1).sum())
        float_count = int((quality == 2).sum())
        single_count = int((quality == 5).sum())

        return {
            "total_epochs": total_epochs,
            "fix_epochs": fix_count,
            "float_epochs": float_count,
            "single_epochs": single_count,
            "fix_rate": (fix_count / total_epochs) * 100 if total_epochs > 0 else 0,
            "avg_ratio": self.df["ratio"].mean(),
            "avg_ns": self.df["num_sats"].mean(),
            "max_ratio": self.df["ratio"].max(),
            "min_ratio": self.df["ratio"].min(),
        }
