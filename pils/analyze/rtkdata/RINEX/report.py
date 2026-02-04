"""
RINEX Report Generator v5
Feature-Complete Orchestrator for RINEXAnalyzer and RINEXPlotter.
"""

import os
from datetime import datetime

import polars as pl

from .analyzer import CONSTELLATION_NAMES, get_dual_freq_bands


class RINEXReport:
    def __init__(self, analyzer, plotter):
        self.analyzer = analyzer
        self.plotter = plotter

    def generate(self, output_dir="professional_rinex_report"):
        """Produce the full high-fidelity report with all dashboards and metrics."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        assets_dir = os.path.join(output_dir, "assets")
        if not os.path.exists(assets_dir):
            os.makedirs(assets_dir)

        print(f"📄 Mastering Professional GNSS Report in '{output_dir}'...")

        # 0. Data Preparation
        freq_summary = self.analyzer.get_global_frequency_summary()

        # 1. Header & Quality Scoreboard
        start_time, end_time = self.analyzer.get_time_span()
        quality = self.analyzer.assess_data_quality()

        report = f"# GNSS Quality Analysis: {self.analyzer.filename}\n\n"
        report += f"**Analysis Date:** {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        if start_time and end_time:
            report += f"**Session Start:** {start_time:%Y-%m-%d %H:%M:%S}\n"
            report += f"**Session End:**   {end_time:%Y-%m-%d %H:%M:%S}\n"
            report += f"**Duration:**      {end_time - start_time}\n\n"

        report += "## 🏆 Executive Quality Scoreboard\n"
        score = quality["score"]
        status_icon = quality["status_icon"]

        report += f"### Overall Score: **{score:.1f}/100** ({status_icon})\n\n"

        # Red Flags Alert
        if quality["red_flags"]:
            report += "### ⚠️ Red Flags Detected\n"
            for flag in quality["red_flags"]:
                report += f"- {flag}\n"
            report += "\n"

        # 4-Step Algorithm Summary
        m = quality["metrics"]
        report += "#### 🛰️ 4-Step Algorithm Metrics (Session Avg)\n"
        report += f"| Good Sats (40%) | Cell Coverage (30%) | Elevation Span (15%) | Azimuth Balance (15%) |\n"
        report += f"|---|---|---|---|\n"
        report += f"| {m['avg_good_sats']:.1f} / 20 | {m['avg_cells']:.1f} / 12 | {m['avg_el_span']:.1f}° | {m['avg_balance']:.2f} |\n\n"

        # Good Satellites Trend Plot
        trend_path = os.path.join(assets_dir, "good_sats_trend.png")
        print("  - Generating Good Satellites trend plot...")
        self.plotter.plot_good_satellites_trend(quality["epoch_df"], trend_path)
        if os.path.exists(trend_path):
            report += "![Good Satellites Trend](assets/good_sats_trend.png)\n\n"

        # Fleet Review Table
        report += "### 📋 Satellite Quality Fleet Review\n"
        report += "| Sat | Rating | Score | SNR L1 | SNR L2 | MP RMS | Slips/h |\n"
        report += "|---|---|---|---|---|---|---|\n"
        for row in quality["sat_scores"].iter_rows(named=True):
            icon = (
                "✅" if row["rating"] == "Excellent" else "⚠️" if row["rating"] == "Fair" else "❌"
            )
            s1 = f"{row['snr_l1']:.1f}" if row["snr_l1"] > 0 else "-"
            s2 = f"{row['snr_l2']:.1f}" if row["snr_l2"] > 0 else "-"
            report += f"| {row['satellite']} | {icon} {row['rating']} | {row['total_score']:.1f} | {s1} | {s2} | {row['mp_val']:.3f} | {row['slip_rate']:.1f} |\n"
        report += "\n"

        if score > 75:
            report += "> [!NOTE]\n> The data pool is solid. Major constellations are reliable for high-precision GNSS processing.\n\n"
        else:
            report += "> [!CAUTION]\n> High degree of satellite degradation. RTK positions may be biased or suffer from long fix times. Review Fleet Review Table.\n\n"

        # Global Dashboard
        dash_path = os.path.join(assets_dir, "dashboard_global.png")
        print("  - Building Global Dashboard...")
        self.plotter.plot_all_frequencies_summary(dash_path)
        if os.path.exists(dash_path):
            report += "## 📊 Global Performance Dashboard\n"
            report += "![Global Dashboard](assets/dashboard_global.png)\n\n"

        # Band Comparison Plot
        comp_path = os.path.join(assets_dir, "band_comparison.png")
        print("  - Generating Primary vs Secondary comparison plot...")
        self.plotter.plot_band_comparison(comp_path)
        if os.path.exists(comp_path):
            report += (
                "#### Multi-Band SNR Hierarchy\n![Band Comparison](assets/band_comparison.png)\n\n"
            )

        report += "### Frequency Band Metrics\n"
        report += "| Band | Mean SNR | Std SNR | MP RMS (m) | Sats | Observations |\n|---|---|---|---|---|---|\n"
        for row in freq_summary.iter_rows(named=True):
            mp_val = f"{row['mean_MP_RMS']:.3f}" if row["mean_MP_RMS"] is not None else "N/A"
            report += f"| {row['frequency']} | {row['mean']:.1f} | {row['std']:.2f} | {mp_val} | {row['n_satellites']} | {row['count']} |\n"

        # 2. Pooled Distribution & Elevation Dependency
        pooled_path = os.path.join(assets_dir, "pooled_comparison.png")
        print("  - Generating Pooled Distributions...")
        self.plotter.plot_global_l1_l2_comparison_hist(pooled_path)
        if os.path.exists(pooled_path):
            report += "\n## 🌐 Multi-Constellation Quality Context\n"
            report += "#### Global SNR Pooled Benchmarking\n![Comparison](assets/pooled_comparison.png)\n\n"

        for pool in ["single", "dual"]:
            name = "L1-Band" if pool == "single" else "L2-Band"
            print(f"  - Generating {name} context plots...")

            # Skyplot
            sky_path = os.path.join(assets_dir, f"sky_{pool}.png")
            self.plotter.plot_skyplot_snr(pool=pool, save_path=sky_path)
            if os.path.exists(sky_path):
                report += f"### {name} Tracking & Quality\n![Skyplot](assets/sky_{pool}.png)\n\n"

            # Elevation Dependence
            el_path = os.path.join(assets_dir, f"elevation_{pool}.png")
            self.plotter.plot_elevation_dependent_stats(pool=pool, save_path=el_path)
            if os.path.exists(el_path):
                report += f"#### Elevation Dependency (SNR & MP)\n![Elevation](assets/elevation_{pool}.png)\n\n"

        # 3. Detailed Constellation Performance
        report += "## 🛰️ Constellation-Specific Analysis\n"
        constellations = sorted(self.analyzer.df["constellation"].unique().to_list())
        for const in constellations:
            cname = CONSTELLATION_NAMES.get(const, const)

            # Try to generate per-constellation plots first to see if we should add the header
            bar_path = os.path.join(assets_dir, f"bar_{const}.png")
            hist_path = os.path.join(assets_dir, f"hist_{const}.png")

            self.plotter.plot_sat_avg_snr_bar(const, bar_path)
            self.plotter.plot_constellation_histograms(
                const,
                sorted(
                    self.analyzer.df.filter(pl.col("constellation") == const)["frequency"]
                    .unique()
                    .to_list()
                ),
                hist_path,
            )

            if os.path.exists(bar_path) or os.path.exists(hist_path):
                report += f"### {cname} Performance Review\n"
                if os.path.exists(bar_path):
                    report += f"#### Average SNR per Spacecraft\n![Bar](assets/bar_{const}.png)\n\n"
                if os.path.exists(hist_path):
                    report += (
                        f"#### Quality Distribution by Band\n![Hist](assets/hist_{const}.png)\n\n"
                    )

            # Detailed Time Series
            bands = sorted(
                self.analyzer.df.filter(pl.col("constellation") == const)["frequency"]
                .unique()
                .to_list()
            )
            for band in bands:
                print(f"  - Detailed plots for {cname} {band}...")
                sats = sorted(
                    self.analyzer.df.filter(
                        (pl.col("constellation") == const) & (pl.col("frequency") == band)
                    )["satellite"]
                    .unique()
                    .to_list()
                )

                # SNR Time Series
                img_snr = f"ts_snr_{const}_{band}.png"
                snr_path = os.path.join(assets_dir, img_snr)
                self.plotter.plot_snr_time_series(sats, band, snr_path)
                if os.path.exists(snr_path):
                    report += f"#### Band {band} Time Series\n![SNR](assets/{img_snr})\n\n"

                # Multipath Time Series
                img_mp = f"ts_mp_{const}_{band}.png"
                mp_path = os.path.join(assets_dir, img_mp)
                self.plotter.plot_multipath_time_series(sats, band, mp_path)
                if os.path.exists(mp_path):
                    report += f"![MP](assets/{img_mp})\n\n"

        # 4. Signal Integrity & Reliability
        slip_path = os.path.join(assets_dir, "cycle_slips.png")
        print("  - Generating Integrity Dashboards...")
        self.plotter.plot_cycle_slips(slip_path)
        if os.path.exists(slip_path):
            report += "## ⚡ Signal Integrity Monitoring\n"
            report += "### Cycle Slip Event Detection (GF/MW Combined)\n![Slips](assets/cycle_slips.png)\n"

        report += "\n\n---\n*Report generated by Antigravity high-performance GNSS suite. Optimized for rapid QA.*"

        with open(os.path.join(output_dir, "report.md"), "w") as f:
            f.write(report)

        print(f"✅ Masterpiece Report generated at: {output_dir}/report.md")
        return os.path.join(output_dir, "report.md")
