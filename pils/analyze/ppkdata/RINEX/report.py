"""
RINEX Report Generator v5
Feature-Complete Orchestrator for RINEXAnalyzer and RINEXPlotter.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import polars as pl

from ..utils import CONSTELLATION_NAMES, get_dual_freq_bands
from .analyzer import RINEXAnalyzer
from .plotter import RINEXPlotter

logger = logging.getLogger(__name__)


class RINEXReport:
    """Generate comprehensive RINEX quality analysis reports.

    Orchestrates RINEXAnalyzer and RINEXPlotter to produce detailed
    markdown reports with plots analyzing GNSS data quality.

    Attributes
    ----------
    analyzer : RINEXAnalyzer
        Analyzer instance for data processing
    plotter : RINEXPlotter
        Plotter instance for visualization

    Examples
    --------
    >>> report = RINEXReport(rinex_obs=Path('file.obs'), rinex_nav=Path('file.nav'))
    >>> report.generate('quality_report.md', plot_folder='plots')
    >>> # Creates quality_report.md with plots in plots/ subfolder
    """

    def __init__(
        self,
        rinex_obs: Optional[Path] = None,
        rinex_nav: Optional[Path] = None,
        analyzer: Optional[RINEXAnalyzer] = None,
        plotter: Optional[RINEXPlotter] = None,
    ) -> None:
        """Initialize RINEX report generator.

        Args:
            rinex_obs: Path to RINEX observation file (if analyzer not provided)
            rinex_nav: Path to RINEX navigation file (if analyzer not provided)
            analyzer: Existing RINEXAnalyzer instance
            plotter: Existing RINEXPlotter instance

        Examples:
            >>> # From file
            >>> report = RINEXReport(rinex_obs=Path('station.obs'), rinex_nav=Path('station.nav'))
            >>> # From existing analyzer
            >>> analyzer = RINEXAnalyzer('station.obs')
            >>> analyzer.parse_obs_file()
            >>> report = RINEXReport(analyzer=analyzer)
        """
        if analyzer is not None:
            self.analyzer = analyzer
        else:
            if rinex_obs is not None and rinex_nav is not None:
                self.analyzer = RINEXAnalyzer(rinex_obs, navpath=rinex_nav)
                self.analyzer.parse_obs_file()  # Parse the RINEX file
                self.analyzer.parse_nav_file()
            else:
                logger.info("No analyzer nor RINEX file provided")
                self.analyzer = None  # type: ignore

        if plotter is not None:
            self.plotter = plotter
        else:
            if self.analyzer is not None:
                self.plotter = RINEXPlotter(self.analyzer)
            else:
                self.plotter = None  # type: ignore

    def generate(
        self, report_name: Union[str, Path] = "rinex_report.md", plot_folder: str = "assets"
    ) -> str:
        """Generate complete RINEX quality analysis report.

        Args:
            report_name: Output path for markdown report
            plot_folder: Subfolder name for plot assets

        Returns:
            Path to generated report file

        Raises:
            ValueError: If analyzer or plotter not initialized

        Examples:
            >>> report = RINEXReport(rinex_obs=Path('file.obs'), rinex_nav=Path('file.nav'))
            >>> report_path = report.generate('rinex_quality.md', plot_folder='figures')
            >>> print(f"Report generated at: {report_path}")
        """
        if self.analyzer is None:
            raise ValueError("Analyzer not initialized. Provide rinex file or analyzer instance.")
        if self.plotter is None:
            raise ValueError("Plotter not initialized.")

        report_path = Path(report_name)
        output_dir = report_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        assets_dir = output_dir / plot_folder
        assets_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating RINEX quality report in '{output_dir}'")

        # 0. Data Preparation
        self.analyzer.compute_satellite_azel()
        freq_summary = self.analyzer.get_global_frequency_summary()

        # 1. Header & Quality Scoreboard
        start_time, end_time = self.analyzer.get_time_span()
        quality = self.analyzer.assess_data_quality()

        report = f"# GNSS Quality Analysis: {self.analyzer.obsname}\n\n"
        report += f"**Analysis Date:** {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        if start_time and end_time:
            report += f"**Session Start:** {start_time:%Y-%m-%d %H:%M:%S}\n"
            report += f"**Session End:**   {end_time:%Y-%m-%d %H:%M:%S}\n"
            report += f"**Duration:**      {end_time - start_time}\n\n"

        report += "## Executive Quality Scoreboard\n"
        score = quality["score"]
        status_icon = quality["status_icon"]

        report += f"### Overall Score: **{score:.1f}/100** ({status_icon})\n\n"

        # Red Flags Alert
        if quality["red_flags"]:
            report += "### Red Flags Detected\n"
            for flag in quality["red_flags"]:
                report += f"- {flag}\n"
            report += "\n"

        # 4-Step Algorithm Summary
        m = quality["metrics"]
        report += "#### 🛰️ 4-Step Algorithm Metrics (Session Avg)\n"
        report += f"| Good Sats (40%) | Cell Coverage (30%) | Elevation Span (15%) | Azimuth Balance (15%) |\n"
        report += f"|---|---|---|---|\n"
        avg_sats = f"{m['avg_good_sats']:.1f}" if m["avg_good_sats"] is not None else "N/A"
        avg_cells = f"{m['avg_cells']:.1f}" if m["avg_cells"] is not None else "N/A"
        avg_el_span = f"{m['avg_el_span']:.1f}°" if m["avg_el_span"] is not None else "N/A"
        avg_balance = f"{m['avg_balance']:.2f}" if m["avg_balance"] is not None else "N/A"
        report += f"| {avg_sats} / 20 | {avg_cells} / 12 | {avg_el_span} | {avg_balance} |\n\n"

        # Good Satellites Trend Plot
        trend_path = assets_dir / "good_sats_trend.png"
        logger.debug("Generating Good Satellites trend plot")
        self.plotter.plot_good_satellites_trend(quality["epoch_df"], str(trend_path))
        if trend_path.exists():
            report += f"![Good Satellites Trend]({plot_folder}/good_sats_trend.png)\n\n"

        # Fleet Review Table
        report += "### Satellite Quality Fleet Review\n"
        report += "| Sat | Rating | Score | SNR L1 | SNR L2 | MP RMS | Slips/h |\n"
        report += "|---|---|---|---|---|---|---|\n"
        for row in quality["sat_scores"].iter_rows(named=True):
            s1 = f"{row['snr_l1']:.1f}" if row["snr_l1"] is not None and row["snr_l1"] > 0 else "-"
            s2 = f"{row['snr_l2']:.1f}" if row["snr_l2"] is not None and row["snr_l2"] > 0 else "-"
            score_val = f"{row['total_score']:.1f}" if row["total_score"] is not None else "N/A"
            mp_val = f"{row['mp_val']:.3f}" if row["mp_val"] is not None else "N/A"
            slip_val = f"{row['slip_rate']:.1f}" if row["slip_rate"] is not None else "N/A"
            report += f"| {row['satellite']} | {row['rating']} | {score_val} | {s1} | {s2} | {mp_val} | {slip_val} |\n"
        report += "\n"

        if score > 75:
            report += "> [!NOTE]\n> The data pool is solid. Major constellations are reliable for high-precision GNSS processing.\n\n"
        else:
            report += "> [!CAUTION]\n> High degree of satellite degradation. RTK positions may be biased or suffer from long fix times. Review Fleet Review Table.\n\n"

        # Global Dashboard
        dash_path = assets_dir / "dashboard_global.png"
        logger.debug("Building Global Dashboard")
        self.plotter.plot_all_frequencies_summary(str(dash_path))
        if dash_path.exists():
            report += "## Global Performance Dashboard\n"
            report += f"![Global Dashboard]({plot_folder}/dashboard_global.png)\n\n"

        # Band Comparison Plot
        comp_path = assets_dir / "band_comparison.png"
        logger.debug("Generating Primary vs Secondary comparison plot")
        self.plotter.plot_band_comparison(str(comp_path))
        if comp_path.exists():
            report += f"#### Multi-Band SNR Hierarchy\n![Band Comparison]({plot_folder}/band_comparison.png)\n\n"

        report += "### Frequency Band Metrics\n"
        report += "| Band | Mean SNR | Std SNR | MP RMS (m) | Sats | Observations |\n|---|---|---|---|---|---|\n"
        for row in freq_summary.iter_rows(named=True):
            mean_val = f"{row['mean']:.1f}" if row["mean"] is not None else "N/A"
            std_val = f"{row['std']:.2f}" if row["std"] is not None else "N/A"
            mp_val = f"{row['mean_MP_RMS']:.3f}" if row["mean_MP_RMS"] is not None else "N/A"
            report += f"| {row['frequency']} | {mean_val} | {std_val} | {mp_val} | {row['n_satellites']} | {row['count']} |\n"

        # 2. Pooled Distribution & Elevation Dependency
        pooled_path = assets_dir / "pooled_comparison.png"
        logger.debug("Generating Pooled Distributions")
        self.plotter.plot_global_l1_l2_comparison_hist(str(pooled_path))
        if pooled_path.exists():
            report += "\n## Multi-Constellation Quality Context\n"
            report += f"#### Global SNR Pooled Benchmarking\n![Comparison]({plot_folder}/pooled_comparison.png)\n\n"

        for pool in ["single", "dual"]:
            name = "L1-Band" if pool == "single" else "L2-Band"
            logger.debug(f"Generating {name} context plots")

            # Skyplot
            sky_path = assets_dir / f"sky_{pool}.png"
            self.plotter.plot_skyplot_snr(pool=pool, save_path=str(sky_path))
            if sky_path.exists():
                report += (
                    f"### {name} Tracking & Quality\n![Skyplot]({plot_folder}/sky_{pool}.png)\n\n"
                )

            # Elevation Dependence
            el_path = assets_dir / f"elevation_{pool}.png"
            self.plotter.plot_elevation_dependent_stats(pool=pool, save_path=str(el_path))
            if el_path.exists():
                report += f"#### Elevation Dependency (SNR & MP)\n![Elevation]({plot_folder}/elevation_{pool}.png)\n\n"

        # 3. Detailed Constellation Performance
        report += "## Constellation-Specific Analysis\n"
        constellations = sorted(self.analyzer.df_obs["constellation"].unique().to_list())
        for const in constellations:
            cname = CONSTELLATION_NAMES.get(const, const)

            # Try to generate per-constellation plots first to see if we should add the header
            bar_path = assets_dir / f"bar_{const}.png"
            hist_path = assets_dir / f"hist_{const}.png"

            self.plotter.plot_sat_avg_snr_bar(const, str(bar_path))
            self.plotter.plot_constellation_histograms(
                const,
                sorted(
                    self.analyzer.df_obs.filter(pl.col("constellation") == const)["frequency"]
                    .unique()
                    .to_list()
                ),
                str(hist_path),
            )

            if bar_path.exists() or hist_path.exists():
                report += f"### {cname} Performance Review\n"
                if bar_path.exists():
                    report += f"#### Average SNR per Spacecraft\n![Bar]({plot_folder}/bar_{const}.png)\n\n"
                if hist_path.exists():
                    report += f"#### Quality Distribution by Band\n![Hist]({plot_folder}/hist_{const}.png)\n\n"

            # Detailed Time Series
            bands = sorted(
                self.analyzer.df_obs.filter(pl.col("constellation") == const)["frequency"]
                .unique()
                .to_list()
            )
            for band in bands:
                logger.debug(f"Detailed plots for {cname} {band}")
                sats = sorted(
                    self.analyzer.df_obs.filter(
                        (pl.col("constellation") == const) & (pl.col("frequency") == band)
                    )["satellite"]
                    .unique()
                    .to_list()
                )

                # SNR Time Series
                img_snr = f"ts_snr_{const}_{band}.png"
                snr_path = assets_dir / img_snr
                self.plotter.plot_snr_time_series(sats, band, str(snr_path))
                if snr_path.exists():
                    report += f"#### Band {band} Time Series\n![SNR]({plot_folder}/{img_snr})\n\n"

                # Multipath Time Series
                img_mp = f"ts_mp_{const}_{band}.png"
                mp_path = assets_dir / img_mp
                self.plotter.plot_multipath_time_series(sats, band, str(mp_path))
                if mp_path.exists():
                    report += f"![MP]({plot_folder}/{img_mp})\n\n"

        # 4. Signal Integrity & Reliability
        slip_path = assets_dir / "cycle_slips.png"
        logger.debug("Generating Integrity Dashboards")
        self.plotter.plot_cycle_slips(str(slip_path))
        if slip_path.exists():
            report += "## Signal Integrity Monitoring\n"
            report += f"### Cycle Slip Event Detection (GF/MW Combined)\n![Slips]({plot_folder}/cycle_slips.png)\n"

        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Report generated: {report_path}")
        return str(report_path)
