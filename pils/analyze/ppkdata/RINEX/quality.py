"""Data quality of RINEX observations: completeness, gaps, integrity, geometry, score."""

import logging
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


class RINEXQualityMixin:
    """Quality metrics and score for :class:`~pils.analyze.ppkdata.RINEX.analyzer.RINEXAnalyzer`."""

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
                el_span = (
                    0.0
                    if el_min is None or el_max is None
                    else abs(float(el_max) - float(el_min))  # type: ignore[arg-type]
                )

                q_counts = bins.group_by("quad").count()["count"]
                min_count, max_count = q_counts.min(), q_counts.max()
                balance = (
                    float(min_count) / float(max_count)  # type: ignore[arg-type]
                    if min_count is not None and max_count
                    else 0.0
                )
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
