"""Signal metrics of RINEX observations: SNR, multipath, cycle slips."""

import logging

import polars as pl

from ..utils import (
    GNSS_FREQUENCIES,
    C,
    get_dual_freq_bands,
)

logger = logging.getLogger(__name__)


class RINEXSignalMixin:
    """SNR, multipath and cycle slips for :class:`~pils.analyze.ppkdata.RINEX.analyzer.RINEXAnalyzer`."""

    # STATISTICS
    def get_snr(self):
        """Extract SNR observations from parsed data.

        Returns
        -------
            DataFrame containing only SNR ('S') observations

        Examples
        --------
            >>> analyzer = RINEXAnalyzer('file.obs')
            >>> analyzer.parse_obs_file()
            >>> snr_df = analyzer.get_snr()
            >>> print(f"SNR observations: {len(snr_df)}")
        """
        return self.df_obs.filter(pl.col("obs_type") == "S")

    def get_snr_statistics(self):
        """Calculate SNR statistics per satellite and frequency.

        Returns
        -------
            DataFrame with mean, std, and count grouped by satellite and frequency

        Examples
        --------
            >>> analyzer = RINEXAnalyzer('file.obs')
            >>> analyzer.parse_obs_file()
            >>> stats = analyzer.get_snr_statistics()
            >>> gps_l1 = stats.filter(pl.col('frequency') == 'L1')
            >>> print(gps_l1)
        """
        return (
            self.get_snr()
            .group_by(["satellite", "frequency"])
            .agg(
                [
                    pl.col("value").mean().alias("mean"),
                    pl.col("value").std().alias("std"),
                    pl.col("value").count().alias("count"),
                ]
            )
        )

    def get_global_frequency_summary(self) -> pl.DataFrame:
        """Compute global statistics summary for all frequency bands.

        Calculates mean/std SNR, multipath RMS, satellite count,
        and observation count for each frequency band.

        Returns
        -------
            DataFrame with columns: frequency, mean, std, mean_MP_RMS,
            n_satellites, count

        Examples
        --------
            >>> analyzer = RINEXAnalyzer('file.obs')
            >>> analyzer.parse()
            >>> summary = analyzer.get_global_frequency_summary()
            >>> print(summary.filter(pl.col('frequency') == 'L1'))
        """
        snr_summary = (
            self.get_snr()
            .group_by(["constellation", "frequency"])
            .agg(
                [
                    pl.col("value").mean().alias("mean"),
                    pl.col("value").std().alias("std"),
                    pl.col("satellite").n_unique().alias("n_satellites"),
                    pl.col("value").count().alias("count"),
                ]
            )
        )
        mp_rms = self.get_multipath_rms()
        if not mp_rms.is_empty():
            mp_sum = mp_rms.group_by(["constellation", "frequency"]).agg(
                pl.col("MP_RMS").mean().alias("mean_MP_RMS")
            )
            return snr_summary.join(
                mp_sum, on=["constellation", "frequency"], how="left"
            ).sort(["constellation", "frequency"])
        return snr_summary.with_columns(pl.lit(None).alias("mean_MP_RMS")).sort(
            ["constellation", "frequency"]
        )

    # ADVANCED GEODETIC ANALYSIS
    def estimate_multipath(self):
        """Estimate multipath error using code-phase combinations.

        Implements RTKLIB multipath algorithm using dual-frequency
        data to isolate multipath effects from other errors.

        Returns
        -------
            DataFrame with multipath estimates (MP) for each observation

        Examples
        --------
            >>> analyzer = RINEXAnalyzer('file.obs')
            >>> analyzer.parse_obs_file()
            >>> mp_df = analyzer.estimate_multipath()
            >>> high_mp = mp_df.filter(pl.col('MP').abs() > 1.0)
            >>> print(f"High multipath: {len(high_mp)} observations")
        """
        """
        Implements the RTKLIB multipath algorithm (Plot::updateMp).
        1. Calculate reference ionosphere I from available dual-frequency data.
        2. Compute raw MP for all frequencies.
        3. Remove mean bias per arc (separated by jumps or cycle slips).
        """
        if self.df_obs.is_empty():
            return pl.DataFrame()

        # 1. Prepare frequency information
        # Get all phase (L) and code (P) obs
        obs = self.df_obs.filter(pl.col("obs_type").is_in(["L", "C", "P"]))
        if obs.is_empty():
            return pl.DataFrame()

        # Pivot obs_type to columns. Ensure unique rows first.
        pivoted = obs.unique(
            subset=["time", "satellite", "frequency", "obs_type"]
        ).pivot(
            on="obs_type",
            index=["time", "satellite", "constellation", "frequency"],
            values="value",
        )

        # Merge Code types (P and C)
        code_cols = [c for c in ["P", "C"] if c in pivoted.columns]
        if not code_cols:
            return pl.DataFrame()

        if len(code_cols) == 2:
            pivoted = pivoted.with_columns(
                pl.col("P").fill_null(pl.col("C")).alias("P_val")
            )
        else:
            pivoted = pivoted.with_columns(pl.col(code_cols[0]).alias("P_val"))

        if "L" not in pivoted.columns:
            return pl.DataFrame()

        # Filter rows with both P and L
        data = pivoted.filter(pl.col("P_val").is_not_null() & pl.col("L").is_not_null())
        if data.is_empty():
            return pl.DataFrame()

        # Join with GNSS frequencies to get Hz values
        freq_rows = []
        unique_sats = data.select(["satellite", "constellation"]).unique()

        for row in unique_sats.iter_rows(named=True):
            sat = row["satellite"]
            const = row["constellation"]
            if const not in GNSS_FREQUENCIES:
                continue

            for band, f_mhz in GNSS_FREQUENCIES[const].items():
                f_hz = f_mhz * 1e6
                # GLONASS FDMA adjustment
                if const == "R":
                    slot = self.glo_slots.get(sat)
                    if slot is not None:
                        if band == "G1":
                            f_hz = (1602.0 + slot * 0.5625) * 1e6
                        elif band == "G2":
                            f_hz = (1246.0 + slot * 0.4375) * 1e6

                freq_rows.append(
                    {
                        "satellite": sat,
                        "constellation": const,
                        "frequency": band,
                        "f_hz": f_hz,
                    }
                )

        freq_df = pl.DataFrame(freq_rows)

        data = data.join(freq_df, on=["satellite", "constellation", "frequency"])

        # Phase L in meters
        data = data.with_columns((pl.col("L") * (C / pl.col("f_hz"))).alias("L_m"))

        # 2. Reference Ionosphere I per (time, satellite)
        # Group by (time, sat) and pick first two available frequencies for the reference ionosphere
        dual = (
            data.sort(
                ["time", "satellite", "f_hz"], descending=[False, False, True]
            )  # Sort to pick primary bands
            .group_by(["time", "satellite"])
            .agg(
                [
                    pl.col("f_hz").head(2).alias("f_pair"),
                    pl.col("L_m").head(2).alias("L_pair"),
                ]
            )
            .filter(pl.col("f_pair").list.len() >= 2)
        )

        if dual.is_empty():
            return pl.DataFrame()

        dual = dual.with_columns(
            [
                pl.col("f_pair").list.get(0).alias("f1"),
                pl.col("f_pair").list.get(1).alias("f2"),
                pl.col("L_pair").list.get(0).alias("L1_m"),
                pl.col("L_pair").list.get(1).alias("L2_m"),
            ]
        )

        # Reference I at f1: I1 = (L1 - L2) / ((f1/f2)**2 - 1)
        dual = dual.with_columns(
            (
                (pl.col("L1_m") - pl.col("L2_m"))
                / ((pl.col("f1") / pl.col("f2")) ** 2 - 1)
            ).alias("I1")
        )

        # Join back to calculate raw MP for ALL frequencies in that epoch
        res = data.join(
            dual.select(["time", "satellite", "f1", "I1"]), on=["time", "satellite"]
        )

        # Raw MP_j = P_j - L_j - 2 * (f1/f_j)^2 * I1
        res = res.with_columns(
            (
                pl.col("P_val")
                - pl.col("L_m")
                - 2 * (pl.col("f1") / pl.col("f_hz")) ** 2 * pl.col("I1")
            ).alias("MP_raw")
        )

        # 3. Bias Removal per Arc (Sequence of continuous observations)
        res = res.sort(["satellite", "frequency", "time"])

        # Break arcs on: Time gap (> 60s) OR Raw MP jump (> 5.0m)
        res = res.with_columns(
            [
                pl.col("time")
                .diff()
                .dt.total_seconds()
                .over(["satellite", "frequency"])
                .fill_null(0)
                .alias("dt"),
                pl.col("MP_raw")
                .diff()
                .abs()
                .over(["satellite", "frequency"])
                .fill_null(0)
                .alias("jump"),
            ]
        )

        res = res.with_columns(
            ((pl.col("dt") > 60) | (pl.col("jump") > 5.0)).alias("is_break")
        )

        res = res.with_columns(
            pl.col("is_break")
            .cast(pl.Int32)
            .cum_sum()
            .over(["satellite", "frequency"])
            .alias("arc_id")
        )

        # Final MP: Subtract mean per arc to remove ambiguity bias
        res = res.with_columns(
            (
                pl.col("MP_raw")
                - pl.col("MP_raw").mean().over(["satellite", "frequency", "arc_id"])
            ).alias("MP")
        )

        return res.select(["time", "satellite", "constellation", "frequency", "MP"])

    def get_multipath_rms(self):
        """Calculate RMS multipath per satellite and frequency.

        Returns
        -------
            DataFrame with MP_RMS values grouped by satellite, constellation, frequency

        Examples
        --------
            >>> analyzer = RINEXAnalyzer('file.obs')
            >>> analyzer.parse_obs_file()
            >>> mp_rms = analyzer.get_multipath_rms()
            >>> print(f"Average L1 multipath: {mp_rms.filter(pl.col('frequency')=='L1')['MP_RMS'].mean():.2f}m")
        """
        mp = self.estimate_multipath()
        if mp.is_empty():
            return pl.DataFrame()
        return mp.group_by(["satellite", "constellation", "frequency"]).agg(
            (pl.col("MP") ** 2).mean().sqrt().alias("MP_RMS")
        )

    def detect_cycle_slips(self, threshold_gf=0.08, threshold_mw=2.5):
        """Detect cycle slips using dual-frequency combinations.

        Uses geometry-free (GF) and Melbourne-Wübbena (MW) combinations
        to identify carrier phase cycle slips.

        Args:
            threshold_gf: Geometry-free jump threshold in meters (default: 0.08)
            threshold_mw: Melbourne-Wübbena jump threshold in cycles (default: 2.5)

        Returns
        -------
            DataFrame with detected slips including time, satellite, and type

        Examples
        --------
            >>> analyzer = RINEXAnalyzer('file.obs')
            >>> analyzer.parse_obs_file()
            >>> slips = analyzer.detect_cycle_slips(threshold_gf=0.1, threshold_mw=3.0)
            >>> print(f"Total cycle slips detected: {len(slips)}")
            >>> gps_slips = slips.filter(pl.col('satellite').str.starts_with('G'))
        """
        """Restores dual-combination (GF + MW) Slip Detection."""
        slips = []
        for sat in self.df_obs["satellite"].unique().to_list():
            const = sat[0]
            b1, b2 = get_dual_freq_bands(const)
            if not b2 or const not in GNSS_FREQUENCIES:
                continue

            f1 = GNSS_FREQUENCIES[const][b1]
            f2 = GNSS_FREQUENCIES[const][b2]
            l1_wl = C / (f1 * 1e6)
            l2_wl = C / (f2 * 1e6)
            wl_wl = C / ((f1 - f2) * 1e6)

            sub = self.df_obs.filter(pl.col("satellite") == sat)

            def get_t(data, kind, band):
                return data.filter(
                    (pl.col("obs_type") == kind) & (pl.col("frequency") == band)
                ).select(["time", "value"])

            c1, c2, l1, l2 = (
                get_t(sub, "C", b1),
                get_t(sub, "C", b2),
                get_t(sub, "L", b1),
                get_t(sub, "L", b2),
            )
            if any(d.is_empty() for d in [c1, c2, l1, l2]):
                continue

            comb = (
                l1.rename({"value": "L1_cyc"})
                .join(l2.rename({"value": "L2_cyc"}), on="time")
                .join(c1.rename({"value": "C1"}), on="time")
                .join(c2.rename({"value": "C2"}), on="time")
            )

            # GF in meters
            comb = comb.with_columns(
                (pl.col("L1_cyc") * l1_wl - pl.col("L2_cyc") * l2_wl).alias("GF")
            )
            # MW in cycles
            comb = comb.with_columns(
                (
                    (
                        (f1 * pl.col("L1_cyc") * l1_wl - f2 * pl.col("L2_cyc") * l2_wl)
                        / (f1 - f2)
                        - (f1 * pl.col("C1") + f2 * pl.col("C2")) / (f1 + f2)
                    )
                    / wl_wl
                ).alias("MW")
            )

            # Jump detection
            comb = comb.sort("time").with_columns(
                [
                    pl.col("GF").diff().abs().alias("jump_gf"),
                    pl.col("MW").diff().abs().alias("jump_mw"),
                ]
            )

            hit = comb.filter(
                (pl.col("jump_gf") > threshold_gf) | (pl.col("jump_mw") > threshold_mw)
            )
            if not hit.is_empty():
                hit = hit.with_columns(
                    (
                        pl.when(pl.col("jump_gf") > threshold_gf)
                        .then(pl.lit("GF"))
                        .otherwise(pl.lit(""))
                        + pl.when(pl.col("jump_mw") > threshold_mw)
                        .then(pl.lit("MW"))
                        .otherwise(pl.lit(""))
                    ).alias("type")
                )
                slips.append(
                    hit.select(["time", "type"]).with_columns(
                        pl.lit(sat).alias("satellite")
                    )
                )

        return pl.concat(slips) if slips else pl.DataFrame()
