import os

import polars as pl


class STATAnalyzer:
    """
    Polars-based analyzer for RTKLIB .stat files (Residuals and SNR).
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pl.DataFrame()

    def parse(self):
        """
        Parses $SAT lines from the .stat file into a Polars DataFrame.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"STAT file not found: {self.filepath}")

        # Use polars to read CSV-like $SAT lines
        # RTKLIB $SAT Format: 0:$SAT, 1:week, 2:tow, 3:sat, 4:freq, 5:az, 6:el, 7:resp, 8:resc, 9:vs, 10:snr, 11:fix, 12:slip, 13:lock, 14:outc, 15:slipc, 16:rejc

        sat_data = []
        with open(self.filepath, "r") as f:
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
        """
        Aggregates residuals and SNR per frequency band.
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
