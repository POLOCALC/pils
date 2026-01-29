import polars as pl
import os
import numpy as np
from datetime import datetime

class POSAnalyzer:
    """
    Polars-based analyzer for RTKLIB .pos solution files.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pl.DataFrame()
        self.header = []

    def parse(self):
        """
        Parses the .pos file into a Polars DataFrame.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"POS file not found: {self.filepath}")

        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        # Extract header and data
        data_lines = [line for line in lines if not line.startswith('%')]
        header_lines = [line for line in lines if line.startswith('%')]
        
        # Simple parser for the standard RTKLIB POS format
        # Format: GPST, latitude(deg), longitude(deg), height(m), Q, ns, sdn(m), sde(m), sdu(m), sdne(m), sdeu(m), sdun(m), age(s), ratio
        
        records = []
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 15:
                try:
                    records.append({
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
                        "ratio": float(parts[14])
                    })
                except (ValueError, IndexError):
                    continue

        if not records:
            return pl.DataFrame()

        self.df = pl.DataFrame(records).with_columns([
            pl.col("time").str.to_datetime("%Y/%m/%d %H:%M:%S%.f")
        ])
        
        # Compute ENU Offsets if we have data
        if not self.df.is_empty():
            self._compute_enu()
            
        return self.df

    def _compute_enu(self):
        """Converts Lat/Lon/Height to ENU offsets relative to the mean position."""
        # Mean position as origin
        lat0 = np.deg2rad(self.df["lat"].mean())
        lon0 = np.deg2rad(self.df["lon"].mean())
        h0 = self.df["height"].mean()

        # WGS84 Constants
        a = 6378137.0
        b = 6356752.314245
        f = (a - b) / a
        e2 = 1 - (b**2 / a**2)

        def llh_to_ecef(lat, lon, h):
            lat_rad = np.deg2rad(lat)
            lon_rad = np.deg2rad(lon)
            N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
            X = (N + h) * np.cos(lat_rad) * np.cos(lon_rad)
            Y = (N + h) * np.cos(lat_rad) * np.sin(lon_rad)
            Z = (N * (1 - e2) + h) * np.sin(lat_rad)
            return X, Y, Z

        X, Y, Z = llh_to_ecef(self.df["lat"].to_numpy(), self.df["lon"].to_numpy(), self.df["height"].to_numpy())
        X0, Y0, Z0 = llh_to_ecef(np.rad2deg(lat0), np.rad2deg(lon0), h0)

        dx = X - X0
        dy = Y - Y0
        dz = Z - Z0

        # ECEF to ENU rotation matrix
        e = -np.sin(lon0) * dx + np.cos(lon0) * dy
        n = -np.sin(lat0) * np.cos(lon0) * dx - np.sin(lat0) * np.sin(lon0) * dy + np.cos(lat0) * dz
        u = np.cos(lat0) * np.cos(lon0) * dx + np.cos(lat0) * np.sin(lon0) * dy + np.sin(lat0) * dz

        self.df = self.df.with_columns([
            pl.Series("east", e),
            pl.Series("north", n),
            pl.Series("up", u)
        ])

    def get_statistics(self):
        """
        Calculates position statistics (Fix rate, etc.)
        """
        if self.df.is_empty():
            return {}

        total_epochs = len(self.df)
        q_counts = self.df["Q"].value_counts()
        
        fix_count = q_counts.filter(pl.col("Q") == 1)["count"].sum() if 1 in q_counts["Q"] else 0
        float_count = q_counts.filter(pl.col("Q") == 2)["count"].sum() if 2 in q_counts["Q"] else 0
        single_count = q_counts.filter(pl.col("Q") == 5)["count"].sum() if 5 in q_counts["Q"] else 0
        
        return {
            "total_epochs": total_epochs,
            "fix_epochs": fix_count,
            "float_epochs": float_count,
            "single_epochs": single_count,
            "fix_rate": (fix_count / total_epochs) * 100 if total_epochs > 0 else 0,
            "avg_ratio": self.df["ratio"].mean(),
            "avg_ns": self.df["ns"].mean(),
            "max_ratio": self.df["ratio"].max(),
            "min_ratio": self.df["ratio"].min()
        }
