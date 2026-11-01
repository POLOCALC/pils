"""Data quality of RINEX observations: completeness, gaps, integrity, geometry, score."""

import logging
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class RINEXQualityMixin:
    """Quality metrics and score for :class:`~pils.analyze.ppkdata.RINEX.analyzer.RINEXAnalyzer`."""

    def get_completeness_metrics(self):
        """Calculate observation completeness percentage.

        Computes ratio of actual observations to expected observations
        assuming dual-frequency tracking for all satellites.

        Returns
        -------
            Completeness percentage (0-100)

        Examples
        --------
            >>> analyzer = RINEXAnalyzer('file.obs')
            >>> analyzer.parse_obs_file()
            >>> completeness = analyzer.get_completeness_metrics()
            >>> print(f"Data completeness: {completeness:.1f}%")
        """
        """Calculates observation rate vs expected capacity (Obs / (Sats * Epochs * 2))."""
        if not self.epochs or self.df_obs.is_empty():
            return 0.0
        n_epochs = len(self.epochs)
        n_sats = self.df_obs["satellite"].n_unique()
        # Expecting at least 2 bands (L1/L2) per satellite per epoch for RTK
        expected = n_epochs * n_sats * 2
        actual = self.df_obs.filter(pl.col("obs_type").is_in(["L", "C", "P"])).shape[0]
        return min(100.0, (actual / expected) * 100.0) if expected > 0 else 0.0

    def get_gap_metrics(self):
        """Detect data gaps in observation epochs.

        Identifies periods where expected observations are missing
        based on median epoch interval.

        Returns
        -------
            Dictionary with max_gap, n_gaps, and expected_interval

        Examples
        --------
            >>> analyzer = RINEXAnalyzer('file.obs')
            >>> analyzer.parse_obs_file()
            >>> gaps = analyzer.get_gap_metrics()
            >>> print(f"Maximum gap: {gaps['max_gap']:.1f} seconds")
            >>> print(f"Number of gaps: {gaps['n_gaps']}")
        """
        """Detects periods with no observations."""
        if len(self.epochs) < 2:
            return {"max_gap": 0, "n_gaps": 0}
        diffs = [
            (self.epochs[i + 1] - self.epochs[i]).total_seconds()
            for i in range(len(self.epochs) - 1)
        ]
        # Assume expected interval is the most common difference
        expected_interval = np.median(diffs)
        gaps = [d for d in diffs if d > expected_interval + 0.1]
        return {
            "max_gap": max(diffs) if diffs else 0,
            "n_gaps": len(gaps),
            "expected_interval": expected_interval,
        }

    def get_integrity_metrics(self):
        """Calculate cycle slip rates for data integrity assessment.

        Returns
        -------
            Dictionary with slip_rate (per satellite per hour) and total_slips

        Examples
        --------
            >>> analyzer = RINEXAnalyzer('file.obs')
            >>> analyzer.parse_obs_file()
            >>> integrity = analyzer.get_integrity_metrics()
            >>> print(f"Slip rate: {integrity['slip_rate']:.2f} per sat/hour")
        """
        """Calculates LLI/Slip rates."""
        slips = self.detect_cycle_slips()
        if self.df_obs.is_empty():
            return {"slip_rate": 0, "total_slips": 0}

        n_sats = self.df_obs["satellite"].n_unique()
        duration_hours = (
            (max(self.epochs) - min(self.epochs)).total_seconds() / 3600.0
            if len(self.epochs) > 1
            else 1.0
        )

        total_slips = len(slips)
        slip_rate_per_sat_hour = (
            (total_slips / n_sats / duration_hours) if n_sats > 0 else 0
        )

        return {"slip_rate": slip_rate_per_sat_hour, "total_slips": total_slips}

    def get_geometric_metrics(self):
        """Assess satellite geometric distribution.

        Evaluates azimuth quadrant coverage and elevation spread
        for current satellite constellation.

        Returns
        -------
            Dictionary with quadrants, el_spread, and diversity_score

        Examples
        --------
            >>> analyzer = RINEXAnalyzer('file.obs')
            >>> analyzer.parse_obs_file()
            >>> analyzer.compute_satellite_azel()
            >>> geom = analyzer.get_geometric_metrics()
            >>> print(f"Quadrant coverage: {geom['quadrants']}/4")
            >>> print(f"Diversity score: {geom['diversity_score']:.1f}/100")
        """
        """Assesses geometric distribution (quadrants and elevation spread)."""
        if self.azel_df.is_empty():
            return {"quadrants": 0, "el_spread": 0, "diversity_score": 0}

        # Quadrant coverage (0-90, 90-180, 180-270, 270-360)
        # We take the latest epoch for geometry assessment
        latest_time = self.azel_df["time"].max()
        current_sky = self.azel_df.filter(pl.col("time") == latest_time)

        quads = (
            current_sky.with_columns(
                (pl.col("azimuth") // 90).cast(pl.Int32).alias("quad")
            )["quad"]
            .unique()
            .to_list()
        )

        n_quads = len(quads)
        el_min = current_sky["elevation"].min()
        el_max = current_sky["elevation"].max()

        # Ensure both are numbers and not None
        if el_min is None or el_max is None:
            el_spread = 0.0
        else:
            # Cast to float to ensure proper numeric operations
            el_min_val = float(el_min) if el_min is not None else 0.0  # type: ignore
            el_max_val = float(el_max) if el_max is not None else 0.0  # type: ignore
            el_spread = abs(el_max_val - el_min_val)

        # Diversity score: 0-100 based on quadrants (60%) and el spread (40%)
        quad_p = (n_quads / 4.0) * 100
        el_p = min(100, (float(el_spread) / 60.0) * 100)  # 60 deg spread is good

        diversity = float(quad_p * 0.6 + el_p * 0.4)

        return {
            "quadrants": n_quads,
            "el_spread": el_spread,
            "diversity_score": diversity,
        }

    def get_per_sat_quality_scores(self):
        """Calculate quality scores for each satellite from the % of 'Good' epochs."""
        # This is now handled within assess_data_quality for efficiency and consistency
        res = self.assess_data_quality()
        return res["sat_scores"]

    def assess_data_quality(self) -> dict[str, Any]:
        """Comprehensive GNSS data quality assessment.

        Implements 4-step quality algorithm evaluating:
        1. Good satellite count (>35 dBHz SNR)
        2. Sky cell coverage (12 cells)
        3. Elevation span (0-90°)
        4. Azimuth balance

        Returns
        -------
            Dictionary containing:
            - score: Overall quality score (0-100)
            - status_icon: Visual status indicator
            - metrics: Detailed metric values
            - red_flags: List of identified issues
            - sat_scores: Per-satellite quality ratings
            - epoch_df: Time-series quality metrics

        Examples
        --------
            >>> quality = analyzer.assess_data_quality()
            >>> print(f"Score: {quality['score']:.1f}/100")
            >>> for flag in quality['red_flags']:
            ...     print(f"Warning: {flag}")
        """
        """
        Calculates session quality based on the 4-Step Algorithm in quality.md.
        Strictly epoch-based per-satellite and session evaluation.
        """
        if self.df_obs.is_empty():
            return {
                "status": "UNCERTAIN",
                "status_icon": "UNCERTAIN",
                "score": 0,
                "reason": "No observation data",
                "metrics": {
                    "avg_good_sats": 0,
                    "avg_cells": 0,
                    "avg_el_span": 0,
                    "avg_balance": 0,
                },
                "epoch_df": pl.DataFrame(),
                "sat_scores": pl.DataFrame(),
                "red_flags": ["No observation data available"],
            }

        if self.azel_df.is_empty():
            logger.warning(
                "NAV file not provided - cannot compute geometric quality metrics"
            )
            # Provide basic SNR/MP quality assessment without geometry
            snr = self.get_snr()
            if snr.is_empty():
                basic_score = 0
                sat_count = 0
            else:
                avg_snr = snr.group_by("time").agg(
                    pl.col("value").mean().alias("avg_snr")
                )

                mean_snr = avg_snr.select(pl.col("avg_snr").mean()).item()

                basic_score = (
                    float(min(100, (mean_snr / 45.0) * 100))
                    if avg_snr["avg_snr"].mean()
                    else 0
                )
                sat_count = snr["satellite"].n_unique()

            return {
                "status": "UNCERTAIN",
                "status_icon": "UNCERTAIN",
                "score": basic_score,
                "reason": "NAV file missing - limited quality assessment",
                "metrics": {
                    "avg_good_sats": sat_count,
                    "avg_cells": 0,
                    "avg_el_span": 0,
                    "avg_balance": 0,
                },
                "epoch_df": pl.DataFrame(),
                "sat_scores": pl.DataFrame(),
                "red_flags": [
                    "NAV file not provided - geometric quality metrics unavailable"
                ],
            }

        # 1. Prepare Data
        snr = self.get_snr()
        mp_est = self.estimate_multipath()
        lli_df = self.df_obs.filter(pl.col("obs_type") == "L")

        primary_bands = ["L1", "G1", "E1", "B1"]
        secondary_bands = ["L2", "G2", "E5b", "B2"]

        obs_l1 = (
            snr.filter(pl.col("frequency").is_in(primary_bands))
            .join(
                mp_est.filter(pl.col("frequency").is_in(primary_bands)).select(
                    ["time", "satellite", "MP"]
                ),
                on=["time", "satellite"],
                how="inner",
            )
            .join(
                lli_df.filter(pl.col("frequency").is_in(primary_bands)).select(
                    ["time", "satellite", "lli"]
                ),
                on=["time", "satellite"],
                how="inner",
            )
            .rename({"value": "snr_l1", "MP": "mp_l1", "lli": "lli_l1"})
        )

        obs_l2 = (
            snr.filter(pl.col("frequency").is_in(secondary_bands))
            .join(
                lli_df.filter(pl.col("frequency").is_in(secondary_bands)).select(
                    ["time", "satellite", "lli"]
                ),
                on=["time", "satellite"],
                how="left",
            )
            .rename({"value": "snr_l2", "lli": "lli_l2"})
            .select(["time", "satellite", "snr_l2", "lli_l2"])
        )

        obs_data = obs_l1.join(obs_l2, on=["time", "satellite"], how="left").join(
            self.azel_df, on=["time", "satellite"], how="inner"
        )

        # 2. Mark "GOOD" satellites (Epoch Level)
        # Criteria: SNR > 35 (L1) / 30 (L2), MP < 1.0, LLI == 0, Elevation > 15
        obs_data = obs_data.with_columns(
            (
                (pl.col("snr_l1") > 35)
                & (pl.col("lli_l1") == 0)
                & (
                    (pl.col("snr_l2").is_null())
                    | ((pl.col("snr_l2") > 30) & (pl.col("lli_l2").fill_null(0) == 0))
                )
                & (pl.col("mp_l1").abs() < 1.0)
                & (pl.col("elevation") > 15)
            ).alias("is_good")
        )

        # 3. Calculate Session Metrics (Epoch by Epoch)
        epoch_stats = []
        for t in self.epochs:
            epoch_obs = obs_data.filter(pl.col("time") == t)
            good_sats = epoch_obs.filter(pl.col("is_good"))
            n_good = good_sats.shape[0]

            if n_good > 0:
                # Step 2: Coverage
                quads = good_sats.with_columns(
                    (pl.col("azimuth") // 90).cast(pl.Int32).alias("quad")
                )
                bins = quads.with_columns(
                    pl.when(pl.col("elevation") < 30)
                    .then(pl.lit("low"))
                    .when(pl.col("elevation") < 50)
                    .then(pl.lit("mid"))
                    .otherwise(pl.lit("high"))
                    .alias("ebin")
                )
                cells_covered = bins.select(["quad", "ebin"]).unique().shape[0]

                # Step 3: Geometry
                el_min = good_sats["elevation"].min()
                el_max = good_sats["elevation"].max()

                if el_min is None or el_max is None:
                    el_span = 0.0
                else:
                    # Cast to float to ensure proper numeric operations
                    el_min_val = float(el_min) if el_min is not None else 0.0  # type: ignore
                    el_max_val = float(el_max) if el_max is not None else 0.0  # type: ignore
                    el_span = abs(el_max_val - el_min_val)

                q_counts = bins.group_by("quad").count()
                if q_counts.shape[0] > 0:
                    min_count = q_counts["count"].min()
                    max_count = q_counts["count"].max()
                    if (
                        min_count is not None
                        and max_count is not None
                        and max_count > 0  # type: ignore
                    ):
                        # Cast to float to ensure proper numeric operations
                        min_val = float(min_count) if min_count is not None else 0.0  # type: ignore
                        max_val = float(max_count) if max_count is not None else 1.0  # type: ignore
                        balance = min_val / max_val if max_val > 0 else 0.0
                    else:
                        balance = 0.0
                else:
                    balance = 0.0
            else:
                cells_covered = 0
                el_span = 0
                balance = 0

            # Step 4: Final Score per epoch
            s_count = float(min(100.0, (n_good / 20.0) * 100))
            s_cov = float((cells_covered / 12.0) * 100)
            s_el = float(min(100.0, (float(el_span) / 45.0) * 100))
            s_az = float(balance * 100)

            epoch_score = float(
                s_count * 0.40 + s_cov * 0.30 + s_el * 0.15 + s_az * 0.15
            )

            epoch_stats.append(
                {
                    "time": t,
                    "n_good": n_good,
                    "cells": cells_covered,
                    "el_span": el_span,
                    "balance": balance,
                    "score": epoch_score,
                }
            )

        epoch_df = pl.DataFrame(epoch_stats)

        # Handle empty epoch_df case
        if epoch_df.is_empty():
            return {
                "status": "UNCERTAIN",
                "status_icon": "UNCERTAIN",
                "score": 0,
                "metrics": {
                    "avg_good_sats": 0,
                    "avg_cells": 0,
                    "avg_el_span": 0,
                    "avg_balance": 0,
                },
                "epoch_df": epoch_df,
                "sat_scores": pl.DataFrame(),
                "red_flags": ["No valid epochs found"],
            }

        session_score = epoch_df["score"].mean()

        # 4. Calculate Per-Satellite Session Scores
        # Defined as the % of epochs where the satellite was marked "GOOD"
        duration_hours = (
            (max(self.epochs) - min(self.epochs)).total_seconds() / 3600.0
            if len(self.epochs) > 1
            else 1.0
        )
        sat_quality = obs_data.group_by("satellite").agg(
            [
                (pl.col("is_good").sum() / pl.col("is_good").count() * 100).alias(
                    "total_score"
                ),
                pl.col("snr_l1").mean().alias("snr_l1"),
                pl.col("snr_l2").mean().alias("snr_l2"),
                pl.col("mp_l1").abs().mean().alias("mp_val"),
                # Placeholder for slip counts in the table (LLI is already checked per-epoch)
                (pl.col("lli_l1").sum() + pl.col("lli_l2").fill_null(0).sum()).alias(
                    "slip_count"
                ),
            ]
        )

        sat_quality = sat_quality.with_columns(
            pl.when(pl.col("total_score") >= 80)
            .then(pl.lit("Excellent"))
            .when(pl.col("total_score") >= 55)
            .then(pl.lit("Fair"))
            .otherwise(pl.lit("Poor"))
            .alias("rating"),
            (pl.col("slip_count") / duration_hours).alias("slip_rate"),
        ).sort("satellite")

        # 5. Result
        def get_grade(s):
            if s >= 85:
                return "Excellent"
            if s >= 70:
                return "Good"
            if s >= 55:
                return "Fair"
            return "Poor"

        status = get_grade(session_score)

        return {
            "status": status,
            "status_icon": {
                "Excellent": "EXCELLENT",
                "Good": "GOOD",
                "Fair": "FAIR",
                "Poor": "POOR",
            }.get(status),
            "score": session_score if session_score is not None else 0,
            "metrics": {
                "avg_good_sats": epoch_df["n_good"].mean() or 0,
                "avg_cells": epoch_df["cells"].mean() or 0,
                "avg_el_span": epoch_df["el_span"].mean() or 0,
                "avg_balance": epoch_df["balance"].mean() or 0,
            },
            "epoch_df": epoch_df,
            "sat_scores": sat_quality,
            "red_flags": (
                [
                    f"Critical quality drop in {len(epoch_df.filter(pl.col('score') < 55))} epochs"
                ]
                if len(epoch_df.filter(pl.col("score") < 55)) > 0
                else []
            ),
        }

    def get_time_span(self):
        """Get observation time span.

        Returns
        -------
            Tuple of (start_datetime, end_datetime)

        Examples
        --------
            >>> analyzer = RINEXAnalyzer('file.obs')
            >>> analyzer.parse_obs_file()
            >>> start, end = analyzer.get_time_span()
            >>> duration = (end - start).total_seconds() / 3600
            >>> print(f"Session duration: {duration:.1f} hours")
        """
        if not self.epochs:
            return None, None
        return min(self.epochs), max(self.epochs)
