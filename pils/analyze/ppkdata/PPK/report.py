"""RTKLIB Report Generator.

Generates comprehensive quality analysis reports for RTKLIB PPK solutions.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import polars as pl

from pils.analyze.ppkdata.PPK.pos_analyzer import POSAnalyzer
from pils.analyze.ppkdata.PPK.stat_analyzer import STATAnalyzer
from pils.analyze.ppkdata.PPK.plotter import PPKPlotter
from ..utils import CONSTELLATION_NAMES

logger = logging.getLogger(__name__)


class RTKLIBReport:
    """Generate comprehensive RTKLIB solution quality reports.

    Orchestrates POSAnalyzer, STATAnalyzer, and PPKPlotter to produce
    detailed markdown reports analyzing RTK solution quality.

    Attributes
    ----------
    pos : POSAnalyzer
        Analyzer for position solution data
    stat : STATAnalyzer
        Analyzer for processing statistics
    plotter : PPKPlotter
        Plotter for visualization
    
    Examples
    --------
    >>> report = RTKLIBReport(pos_file='solution.pos', stat_file='solution.pos.stat')
    >>> report.generate('ppk_report', plot_dir='plots')
    >>> # Creates ppk_report.md with plots in plots/ subfolder
    """

    def __init__(
        self,
        pos_file: Optional[Union[str, Path]] = None,
        stat_file: Optional[Union[str, Path]] = None,
        pos_analyzer: Optional[POSAnalyzer] = None,
        stat_analyzer: Optional[STATAnalyzer] = None,
        plotter: Optional[PPKPlotter] = None,
    ) -> None:
        """Initialize RTKLIB report generator.

        Args:
            pos_file: Path to .pos file
            stat_file: Path to .pos.stat file
            pos_analyzer: Existing POSAnalyzer instance
            stat_analyzer: Existing STATAnalyzer instance
            plotter: Existing PPKPlotter instance
        
        Examples:
            >>> # From files
            >>> report = RTKLIBReport(pos_file='solution.pos', stat_file='solution.pos.stat')
            >>> # From existing analyzers
            >>> pos = POSAnalyzer('solution.pos')
            >>> pos.parse()
            >>> report = RTKLIBReport(pos_analyzer=pos)
        """
        if pos_analyzer is not None:
            self.pos = pos_analyzer
        else:
            if pos_file is not None:
                self.pos = POSAnalyzer(pos_file)
                self.pos.parse()  # Parse the .pos file
            else:
                logger.info(".pos file or pos_analyzer is necessary to generate the report")

        if stat_analyzer is not None:
            self.stat = stat_analyzer
        else:
            if stat_file is not None:
                self.stat = STATAnalyzer(stat_file)
                self.stat.parse()  # Parse the .stat file
            else:
                logger.info(".stat file or stat_analyzer is necessary to generate the report")

        if plotter is not None:
            self.plotter = plotter
        else:
            self.plotter = PPKPlotter(self.pos, self.stat)

    def generate(self, output_dir: str = "rtklib_quality_report", plot_dir: str = "assets") -> str:
        """Generate high-fidelity markdown report for RTKLIB outputs.

        Args:
            output_dir: Directory for report output
            plot_dir: Subdirectory name for plot assets

        Returns:
            Path to generated report file
        
        Examples:
            >>> report = RTKLIBReport(pos_file='solution.pos', stat_file='solution.pos.stat')
            >>> report_path = report.generate('my_report', plot_dir='figures')
            >>> print(f"Report saved to: {report_path}")
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        assets_dir = output_path / plot_dir
        assets_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating RTKLIB Quality Report in '{output_dir}'")

        report = "# RTKLIB Solution Quality Analysis\n\n"
        report += f"**Analysis Date:** {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        report += "## Executive Solution Scoreboard\n"

        if self.plotter and self.stat:
            # Skyplot at the very beginning
            sky_path = assets_dir / "skyplot_rtklib.png"
            logger.debug("Generating RTKLIB Skyplot")
            self.plotter.plot_skyplot_snr(str(sky_path))
            if sky_path.exists():
                report += f"![Skyplot]({plot_dir}/skyplot_rtklib.png)\n\n"

        # 1. Solution Statistics (Fix Rate)
        if self.pos:
            stats = self.pos.get_statistics()

            fix_rate = stats.get("fix_rate", 0)
            status = (
                "EXCELLENT"
                if fix_rate > 95
                else "GOOD" if fix_rate > 80 else "FAIR" if fix_rate > 50 else "POOR"
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
            enu_path = assets_dir / "enu_stability.png"
            logger.debug("Generating ENU Stability plot")
            self.plotter.plot_enu_time_series(str(enu_path))
            if enu_path.exists():
                report += "## Coordinate Stability (ENU)\n"
                report += f"![ENU]({plot_dir}/enu_stability.png)\n\n"

            # Trajectory
            traj_path = assets_dir / "trajectory.png"
            logger.debug("Building Trajectory Map")
            self.plotter.plot_trajectory_q(str(traj_path))
            if traj_path.exists():
                report += "## Solution Trajectory\n"
                report += f"![Trajectory]({plot_dir}/trajectory.png)\n\n"

            # Ratio
            ratio_path = assets_dir / "ratio_time.png"
            logger.debug("Generating Ratio stability plot")
            self.plotter.plot_ratio_time(str(ratio_path))
            if ratio_path.exists():
                report += "## AR Ratio Stability\n"
                report += f"![Ratio]({plot_dir}/ratio_time.png)\n\n"

        # 3. Residual & Signal Analysis (from .stat)
        if self.stat:
            sat_stats = self.stat.get_satellite_stats()
            global_stats = self.stat.get_global_stats()

            report += "## Signal & Residual Analysis\n"

            if self.plotter:
                snr_trend_path = assets_dir / "snr_stability.png"
                logger.debug("Generating SNR stability trend")
                self.plotter.plot_avg_snr_time_series(str(snr_trend_path))
                if snr_trend_path.exists():
                    report += "### Signal Strength Stability (SNR)\n"
                    report += f"![SNR Stability]({plot_dir}/snr_stability.png)\n\n"

            report += "### Global Per-Band Metrics\n"
            report += "| Band | Mean SNR | Mean Phase Resid (m) | Mean Code Resid (m) |\n"
            report += "|---|---|---|---|\n"
            for row in global_stats.iter_rows(named=True):
                report += f"| {row['frequency']} | {row['mean_snr']:.1f} | {row['mean_resid_phase']:.4f} | {row['mean_resid_code']:.3f} |\n"
            report += "\n"

            if self.plotter:
                resid_path = assets_dir / "residuals_multi.png"
                logger.debug("Generating Multi-Band residual distributions")
                self.plotter.plot_residual_distribution_dual(str(resid_path))
                if resid_path.exists():
                    report += "### Localized Residual Distributions\n"
                    report += f"![Residuals]({plot_dir}/residuals_multi.png)\n\n"

                snr_el_path = assets_dir / "snr_vs_el.png"
                self.plotter.plot_snr_vs_elevation(str(snr_el_path))
                if snr_el_path.exists():
                    report += "### SNR vs Elevation\n"
                    report += f"![SNR_EL]({plot_dir}/snr_vs_el.png)\n\n"

            # Constellation-Specific Residuals
            constellations = sorted(self.stat.df["constellation"].unique().to_list())
            report += "## Constellation-Specific Performance\n"
            for const in constellations:
                c_full_name = CONSTELLATION_NAMES.get(const, const)
                if c_full_name:
                    c_full_name = c_full_name.upper()
                else:
                    c_full_name = const.upper()
                report += f"### {c_full_name} Analysis\n"

                # SNR Time Series
                if self.plotter:
                    snr_t_path = assets_dir / f"snr_ts_{const}.png"
                    if hasattr(self.plotter, "plot_constellation_snr_time_series"):
                        self.plotter.plot_constellation_snr_time_series(const, str(snr_t_path))
                    if snr_t_path.exists():
                        report += f"#### {c_full_name} SNR Stability over Time\n![SNR]({plot_dir}/snr_ts_{const}.png)\n\n"

                # Histograms
                if self.plotter:
                    h_path = assets_dir / f"resid_hist_{const}.png"
                    if hasattr(self.plotter, "plot_stat_constellation_hists_dual"):
                        self.plotter.plot_stat_constellation_hists_dual(const, str(h_path))
                    if h_path.exists():
                        report += f"#### {c_full_name} Phase & Code Residuals\n![Hist]({plot_dir}/resid_hist_{const}.png)\n\n"

                # Bar Chart
                if self.plotter:
                    b_path = assets_dir / f"resid_bar_{const}.png"
                    if hasattr(self.plotter, "plot_sat_residual_bar"):
                        self.plotter.plot_sat_residual_bar(const, str(b_path))
                    if b_path.exists():
                        report += f"#### {c_full_name} Per-Satellite Peak Residuals\n![Bar]({plot_dir}/resid_bar_{const}.png)\n\n"

            report += "## Satellite Quality Audit\n"
            report += "Analyzed satellites ranked by typical Carrier Phase stability (P95 Phase Residual).\n\n"

            # Top 10 Best
            report += "### Top 10 Best Performers (Lowest Error)\n"
            report += "| Sat | Band | Mean SNR | P95 Phase Resid (m) | Slips | Rejects |\n"
            report += "|---|---|---|---|---|---|\n"
            for row in (
                sat_stats.sort("p95_resid_phase", descending=False).head(10).iter_rows(named=True)
            ):
                report += f"| {row['satellite']} | {row['frequency']} | {row['avg_snr']:.1f} | {row['p95_resid_phase']:.4f} | {row['total_slips']} | {row['total_rejects']} |\n"
            report += "\n"

            # Top 10 Worst
            report += "### Top 10 Worst Performers (Highest Error)\n"
            report += "| Sat | Band | Mean SNR | P95 Phase Resid (m) | Slips | Rejects |\n"
            report += "|---|---|---|---|---|---|\n"
            for row in (
                sat_stats.sort("p95_resid_phase", descending=True).head(10).iter_rows(named=True)
            ):
                report += f"| {row['satellite']} | {row['frequency']} | {row['avg_snr']:.1f} | {row['p95_resid_phase']:.4f} | {row['total_slips']} | {row['total_rejects']} |\n"
            report += "\n"

        report += "\n\n---\n*Report generated by RTKLIB Master Analysis suite.*"

        report_path = output_path / "report.md"
        report_path.write_text(report)

        logger.info(f"RTKLIB Quality Report generated: {report_path}")
        return str(report_path)
