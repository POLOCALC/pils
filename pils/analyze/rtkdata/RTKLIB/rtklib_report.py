import os
from datetime import datetime

import polars as pl

from .pos_analyzer import POSAnalyzer
from .stat_analyzer import STATAnalyzer

CONSTELLATION_NAMES = {
    "G": "GPS",
    "R": "GLONASS",
    "E": "Galileo",
    "C": "BeiDou",
    "J": "QZSS",
    "I": "IRNSS",
    "S": "SBAS",
}


class RTKLIBReport:
    def __init__(self, pos_analyzer=None, stat_analyzer=None, plotter=None):
        self.pos = pos_analyzer
        self.stat = stat_analyzer
        self.plotter = plotter

    def generate(self, output_dir="rtklib_quality_report"):
        """Generates a high-fidelity Markdown report for RTKLIB outputs."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        assets_dir = os.path.join(output_dir, "assets")
        if not os.path.exists(assets_dir):
            os.makedirs(assets_dir)

        print(f"📄 Generating RTKLIB Quality Report in '{output_dir}'...")

        report = "# RTKLIB Solution Quality Analysis\n\n"
        report += f"**Analysis Date:** {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        report += "## 🏆 Executive Solution Scoreboard\n"

        if self.plotter and self.stat:
            # Skyplot at the very beginning
            sky_path = os.path.join(assets_dir, "skyplot_rtklib.png")
            print("  - Generating RTKLIB Skyplot...")
            self.plotter.plot_skyplot_snr(sky_path)
            if os.path.exists(sky_path):
                report += "![Skyplot](assets/skyplot_rtklib.png)\n\n"

        # 1. Solution Statistics (Fix Rate)
        if self.pos:
            stats = self.pos.get_statistics()

            fix_rate = stats.get("fix_rate", 0)
            status = (
                "🟢 EXCELLENT"
                if fix_rate > 95
                else "🟡 GOOD" if fix_rate > 80 else "🟠 FAIR" if fix_rate > 50 else "🔴 POOR"
            )

            report += f"### Fix Rate: **{fix_rate:.1f}%** ({status})\n\n"
            report += "#### Epoch Distribution\n"
            report += f"| Status | Epochs | Percentage |\n"
            report += f"|---|---|---|\n"
            report += f"| Fix (Q=1) | {stats['fix_epochs']} | {(stats['fix_epochs']/stats['total_epochs']*100):.1f}% |\n"
            report += f"| Float (Q=2) | {stats['float_epochs']} | {(stats['float_epochs']/stats['total_epochs']*100):.1f}% |\n"
            report += f"| Single (Q=5) | {stats['single_epochs']} | {(stats['single_epochs']/stats['total_epochs']*100):.1f}% |\n\n"

            report += f"**Total Epochs:** {stats['total_epochs']} | **Avg Ratio:** {stats['avg_ratio']:.2f} | **Avg Sat Count:** {stats['avg_ns']:.1f}\n\n"

        # 2. ENU & Trajectory Dashboards
        if self.plotter and self.pos:
            # ENU Time Series
            enu_path = os.path.join(assets_dir, "enu_stability.png")
            print("  - Generating ENU Stability plot...")
            self.plotter.plot_enu_time_series(enu_path)
            if os.path.exists(enu_path):
                report += "## 🌍 Coordinate Stability (ENU)\n"
                report += "![ENU](assets/enu_stability.png)\n\n"

            # Trajectory
            traj_path = os.path.join(assets_dir, "trajectory.png")
            print("  - Building Trajectory Map...")
            self.plotter.plot_trajectory_q(traj_path)
            if os.path.exists(traj_path):
                report += "## 🗺️ Solution Trajectory\n"
                report += "![Trajectory](assets/trajectory.png)\n\n"

            # Ratio
            ratio_path = os.path.join(assets_dir, "ratio_time.png")
            print("  - Generating Ratio stability plot...")
            self.plotter.plot_ratio_time(ratio_path)
            if os.path.exists(ratio_path):
                report += "## 📈 AR Ratio Stability\n"
                report += "![Ratio](assets/ratio_time.png)\n\n"

        # 3. Residual & Signal Analysis (from .stat)
        if self.stat:
            sat_stats = self.stat.get_satellite_stats()
            global_stats = self.stat.get_global_stats()

            report += "## 📡 Signal & Residual Analysis\n"

            if self.plotter:
                snr_trend_path = os.path.join(assets_dir, "snr_stability.png")
                print("  - Generating SNR stability trend...")
                self.plotter.plot_avg_snr_time_series(snr_trend_path)
                if os.path.exists(snr_trend_path):
                    report += "### Signal Strength Stability (SNR)\n"
                    report += "![SNR Stability](assets/snr_stability.png)\n\n"

            report += "### Global Per-Band Metrics\n"
            report += "| Band | Mean SNR | Mean Phase Resid (m) | Mean Code Resid (m) |\n"
            report += "|---|---|---|---|\n"
            for row in global_stats.iter_rows(named=True):
                report += f"| {row['frequency']} | {row['mean_snr']:.1f} | {row['mean_resid_phase']:.4f} | {row['mean_resid_code']:.3f} |\n"
            report += "\n"

            if self.plotter:
                resid_path = os.path.join(assets_dir, "residuals_multi.png")
                print("  - Generating Multi-Band residual distributions...")
                self.plotter.plot_residual_distribution_dual(resid_path)
                if os.path.exists(resid_path):
                    report += "### Localized Residual Distributions\n"
                    report += "![Residuals](assets/residuals_multi.png)\n\n"

                snr_el_path = os.path.join(assets_dir, "snr_vs_el.png")
                self.plotter.plot_snr_vs_elevation(snr_el_path)
                if os.path.exists(snr_el_path):
                    report += "### SNR vs Elevation\n"
                    report += "![SNR_EL](assets/snr_vs_el.png)\n\n"

            # Constellation-Specific Residuals
            constellations = sorted(self.stat.df["constellation"].unique().to_list())
            report += "## 🛰️ Constellation-Specific Performance\n"
            for const in constellations:
                c_full_name = CONSTELLATION_NAMES.get(const, const)
                if c_full_name:
                    c_full_name = c_full_name.upper()
                else:
                    c_full_name = const.upper()
                report += f"### {c_full_name} Analysis\n"

                # SNR Time Series
                if self.plotter:
                    snr_t_path = os.path.join(assets_dir, f"snr_ts_{const}.png")
                    if hasattr(self.plotter, "plot_constellation_snr_time_series"):
                        self.plotter.plot_constellation_snr_time_series(const, snr_t_path)
                    if os.path.exists(snr_t_path):
                        report += f"#### {c_full_name} SNR Stability over Time\n![SNR](assets/snr_ts_{const}.png)\n\n"

                # Histograms
                if self.plotter:
                    h_path = os.path.join(assets_dir, f"resid_hist_{const}.png")
                    if hasattr(self.plotter, "plot_stat_constellation_hists_dual"):
                        self.plotter.plot_stat_constellation_hists_dual(const, h_path)
                    if os.path.exists(h_path):
                        report += f"#### {c_full_name} Phase & Code Residuals\n![Hist](assets/resid_hist_{const}.png)\n\n"

                # Bar Chart
                if self.plotter:
                    b_path = os.path.join(assets_dir, f"resid_bar_{const}.png")
                    if hasattr(self.plotter, "plot_sat_residual_bar"):
                        self.plotter.plot_sat_residual_bar(const, b_path)
                    if os.path.exists(b_path):
                        report += f"#### {c_full_name} Per-Satellite Peak Residuals\n![Bar](assets/resid_bar_{const}.png)\n\n"

            report += "## 📋 Satellite Quality Audit\n"
            report += "Analyzed satellites ranked by typical Carrier Phase stability (P95 Phase Residual).\n\n"

            # Top 10 Best
            report += "### 🌟 Top 10 Best Performers (Lowest Error)\n"
            report += "| Sat | Band | Mean SNR | P95 Phase Resid (m) | Slips | Rejects |\n"
            report += "|---|---|---|---|---|---|\n"
            for row in (
                sat_stats.sort("p95_resid_phase", descending=False).head(10).iter_rows(named=True)
            ):
                report += f"| {row['satellite']} | {row['frequency']} | {row['avg_snr']:.1f} | {row['p95_resid_phase']:.4f} | {row['total_slips']} | {row['total_rejects']} |\n"
            report += "\n"

            # Top 10 Worst
            report += "### 🔴 Top 10 Worst Performers (Highest Error)\n"
            report += "| Sat | Band | Mean SNR | P95 Phase Resid (m) | Slips | Rejects |\n"
            report += "|---|---|---|---|---|---|\n"
            for row in (
                sat_stats.sort("p95_resid_phase", descending=True).head(10).iter_rows(named=True)
            ):
                report += f"| {row['satellite']} | {row['frequency']} | {row['avg_snr']:.1f} | {row['p95_resid_phase']:.4f} | {row['total_slips']} | {row['total_rejects']} |\n"
            report += "\n"

        report += "\n\n---\n*Report generated by RTKLIB Master Analysis suite.*"

        with open(os.path.join(output_dir, "report.md"), "w") as f:
            f.write(report)

        print(f"✅ RTKLIB Quality Report generated at: {output_dir}/report.md")
        return os.path.join(output_dir, "report.md")
