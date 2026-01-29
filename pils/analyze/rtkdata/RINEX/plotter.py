"""
RINEX Quality Plotter v5
Feature-Complete Visualization engine for RINEXAnalyzer.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import polars as pl
from .analyzer import CONSTELLATION_NAMES, RTKLIB_bands
from ..utils import GNSSColors

class RINEXPlotter:
    def __init__(self, analyzer):
        self.analyzer = analyzer

    def _get_arr(self, col):
        """Extract numpy array from Polars Series."""
        return col.to_numpy()

    def plot_all_frequencies_summary(self, save_path=None):
        """The 'WOW' 2x2 Global Dashboard."""
        summary = self.analyzer.get_global_frequency_summary()
        if summary.is_empty(): return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.patch.set_alpha(0)
        
        # 1. Avg SNR by Band
        colors = [GNSSColors.get_constellation_color(b[0]) for b in summary["frequency"]]
        axes[0,0].bar(summary["frequency"].to_list(), summary["mean"].to_numpy(), color=colors, alpha=0.8)
        axes[0,0].axhline(35, color="red", ls="--")
        axes[0,0].set_title("Average SNR by Frequency Band", fontweight="bold")
        GNSSColors.apply_theme(axes[0,0])
        
        # 2. MP RMS by Band
        valid_mp = summary.filter(pl.col("mean_MP_RMS").is_not_null())
        if not valid_mp.is_empty():
            axes[0,1].bar(valid_mp["frequency"].to_list(), valid_mp["mean_MP_RMS"].to_numpy(), color="#e377c2", alpha=0.8)
            axes[0,1].set_title("Average Multipath RMS by Band", fontweight="bold")
            GNSSColors.apply_theme(axes[0,1])
        
        # 3. Observation Count
        axes[1,0].pie(summary["count"].to_numpy(), labels=summary["frequency"].to_list(), autopct='%1.1f%%', colors=plt.cm.Paired.colors)
        axes[1,0].set_title("Observation Distribution", fontweight="bold")
        
        # 4. Satellite Diversity
        axes[1,1].bar(summary["frequency"].to_list(), summary["n_satellites"].to_numpy(), color=GNSSColors.BEIDOU)
        axes[1,1].set_title("Satellites Tracked per Band", fontweight="bold")
        GNSSColors.apply_theme(axes[1,1])
        
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, transparent=True)
            plt.close()
        else: 
            plt.show()

    def plot_skyplot_snr(self, pool="single", save_path=None):
        """Polar skyplot with SNR tracks and Satellite Labels."""
        if self.analyzer.azel_df.is_empty(): return
        bands = RTKLIB_bands[pool]
        snr = self.analyzer.get_snr().filter(pl.col("frequency").is_in(bands))
        data = snr.join(self.analyzer.azel_df, on=["time", "satellite"])
        if data.is_empty(): return
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        fig.patch.set_alpha(0)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        # Use numpy directly
        az = np.deg2rad(data["azimuth"].to_numpy())
        dist = 90 - data["elevation"].to_numpy()
        sc = ax.scatter(az, dist, c=data["value"].to_numpy(), cmap='viridis', s=8, alpha=0.6)
        plt.colorbar(sc, ax=ax, label="SNR (dB-Hz)")
        
        # Labels - label the middle of each track
        for sat in data["satellite"].unique().to_list():
            sub = data.filter(pl.col("satellite") == sat)
            mid = len(sub) // 2
            ax.text(np.deg2rad(sub["azimuth"][mid]), 90 - sub["elevation"][mid], sat, fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5, boxstyle='round'))

        ax.set_title(f"Skyplot SNR Tracks: {pool.capitalize()} Band Pools", pad=30, fontweight='bold', fontsize=14)
        GNSSColors.apply_theme(ax)
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, transparent=True)
            plt.close()
        else: 
            plt.show()

    def plot_elevation_dependent_stats(self, pool="single", save_path=None):
        """Two panels: SNR and MP vs Elevation (Binned Averages, Color-coded by Sat Count)."""
        if self.analyzer.azel_df.is_empty(): return
        bands = RTKLIB_bands[pool]
        
        # 1. Prepare SNR Data
        snr = self.analyzer.get_snr().filter(pl.col("frequency").is_in(bands))
        merged = snr.join(self.analyzer.azel_df, on=["time", "satellite"])
        
        # Bin by elevation (1 degree bins)
        snr_binned = (merged.with_columns((pl.col("elevation").round(0)).alias("el_bin"))
                      .group_by("el_bin")
                      .agg([
                          pl.col("value").mean().alias("avg_snr"),
                          pl.col("satellite").n_unique().alias("n_sats")
                      ]).sort("el_bin"))

        # 2. Prepare MP Data
        mp = self.analyzer.estimate_multipath().filter(pl.col("frequency").is_in(bands))
        mp_binned = pl.DataFrame()
        if not mp.is_empty():
            mp_merged = mp.join(self.analyzer.azel_df, on=["time", "satellite"])
            mp_binned = (mp_merged.with_columns((pl.col("elevation").round(0)).alias("el_bin"))
                         .group_by("el_bin")
                         .agg([
                             pl.col("MP").abs().mean().alias("avg_mp"),
                             pl.col("satellite").n_unique().alias("n_sats")
                         ]).sort("el_bin"))

        # 3. Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_alpha(0)
        
        # Left: SNR vs Elevation
        sc1 = ax1.scatter(snr_binned["el_bin"].to_numpy(), snr_binned["avg_snr"].to_numpy(), 
                         c=snr_binned["n_sats"].to_numpy(), cmap="plasma", s=50, edgecolor='none')
        ax1.set_xlabel("Elevation (deg)", fontweight="bold")
        ax1.set_ylabel("AVG SNR (dB-Hz)", fontweight="bold")
        ax1.set_title(f"{pool.capitalize()} Pool: SNR vs Elevation", fontweight="bold")
        plt.colorbar(sc1, ax=ax1, label="# Satellites")
        GNSSColors.apply_theme(ax1)

        # Right: Multipath vs Elevation
        if not mp_binned.is_empty():
            sc2 = ax2.scatter(mp_binned["el_bin"].to_numpy(), mp_binned["avg_mp"].to_numpy(), 
                             c=mp_binned["n_sats"].to_numpy(), cmap="plasma", s=50, edgecolor='none')
            ax2.set_xlabel("Elevation (deg)", fontweight="bold")
            ax2.set_ylabel("AVG Multipath (m)", fontweight="bold")
            ax2.set_title(f"{pool.capitalize()} Pool: MP vs Elevation", fontweight="bold")
            plt.colorbar(sc2, ax=ax2, label="# Satellites")
            GNSSColors.apply_theme(ax2)
        else:
            ax2.text(0.5, 0.5, "No Multipath Data Available", ha='center', transform=ax2.transAxes)
            GNSSColors.apply_theme(ax2)

        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, transparent=True, bbox_inches='tight')
            plt.close()
        else: 
            plt.show()

    def plot_snr_time_series(self, satellites, freq_band, save_path=None):
        snr = self.analyzer.get_snr().filter((pl.col("satellite").is_in(satellites)) & (pl.col("frequency") == freq_band))
        if snr.is_empty(): return
        
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_alpha(0)
        for sat in satellites:
            sub = snr.filter(pl.col("satellite") == sat)
            if not sub.is_empty():
                ax.plot(sub["time"].to_numpy(), sub["value"].to_numpy(), label=sat, alpha=0.8, linewidth=1.5)
        
        ax.set_ylabel("SNR (dB-Hz)", fontweight="bold")
        ax.set_title(f"Signal Strength Time Series: {freq_band}", fontweight="bold", fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        GNSSColors.apply_theme(ax)
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, transparent=True)
            plt.close()

    def plot_multipath_time_series(self, satellites, freq_band, save_path=None):
        mp = self.analyzer.estimate_multipath().filter((pl.col("satellite").is_in(satellites)) & (pl.col("frequency") == freq_band))
        if mp.is_empty(): return
        
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_alpha(0)
        for sat in satellites:
            sub = mp.filter(pl.col("satellite") == sat)
            if not sub.is_empty():
                ax.plot(sub["time"].to_numpy(), sub["MP"].to_numpy(), label=sat, alpha=0.7)
        
        ax.set_ylabel("MP (meters)", fontweight="bold")
        ax.set_title(f"Multipath Time Series: {freq_band}", fontweight="bold", fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, transparent=True)
            plt.close()

    def plot_constellation_histograms(self, const, bands, save_path=None):
        snr = self.analyzer.get_snr().filter((pl.col("constellation") == const) & (pl.col("frequency").is_in(bands)))
        if snr.is_empty(): return
        
        fig, axes = plt.subplots(1, len(bands), figsize=(14, 5), squeeze=False, sharex=True)
        fig.patch.set_alpha(0)
        color = GNSSColors.get_constellation_color(const)
        for i, band in enumerate(bands):
            sub = snr.filter(pl.col("frequency") == band)
            axes[0, i].hist(sub["value"].to_numpy(), bins=30, color=color, alpha=0.7, edgecolor='black')
            axes[0, i].set_title(f"SNR Band {band}", fontweight="bold")
            axes[0, i].axvline(35, color="red", linestyle="--", alpha=0.5)
            GNSSColors.apply_theme(axes[0, i])
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, transparent=True)
            plt.close()

    def plot_sat_avg_snr_bar(self, const, save_path=None):
        stats = self.analyzer.get_snr_statistics().filter(pl.col("satellite").str.starts_with(const))
        if stats.is_empty(): return
        
        sats = sorted(stats["satellite"].unique().to_list())
        bands = sorted(stats["frequency"].unique().to_list())
        
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_alpha(0)
        
        x = np.arange(len(sats))
        width = 0.8 / len(bands)
        
        for i, band in enumerate(bands):
            vals = []
            for sat in sats:
                row = stats.filter((pl.col("satellite") == sat) & (pl.col("frequency") == band))
                vals.append(row["mean"][0] if not row.is_empty() else 0)
            ax.bar(x + i*width - 0.4 + width/2, vals, width, 
                   color=[GNSSColors.BAND_PRIMARY, GNSSColors.BAND_SECONDARY][i%2], 
                   label=band, alpha=0.8)
            
        ax.set_xticks(x)
        ax.set_xticklabels(sats, rotation=45)
        ax.set_ylabel("Mean SNR (dB-Hz)")
        ax.set_title(f"Average SNR per {CONSTELLATION_NAMES.get(const, const)} Satellite", fontweight="bold")
        ax.legend()
        GNSSColors.apply_theme(ax)
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, transparent=True)
            plt.close()

    def plot_cycle_slips(self, save_path=None):
        """Restores the detailed scatter plot for Cycle Slips."""
        slips = self.analyzer.detect_cycle_slips()
        if slips.is_empty(): return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_alpha(0)
        
        sats = sorted(slips["satellite"].unique().to_list(), key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 0))
        sat_map = {s: i for i, s in enumerate(sats)}
        
        # Plot by type
        for t, m, c in [("GF", "x", GNSSColors.SINGLE), ("MW", "+", GNSSColors.GPS), ("GFMW", "o", GNSSColors.FIX)]:
            subset = slips.filter(pl.col("type").str.contains(t) if t != "GFMW" else pl.col("type") == "GFMW")
            if not subset.is_empty():
                ax.scatter(subset["time"].to_numpy(), [sat_map[s] for s in subset["satellite"].to_list()], marker=m, color=c, label=f"{t} Slip", s=60, zorder=10)
        
        ax.set_yticks(range(len(sats)))
        ax.set_yticklabels(sats)
        ax.set_title("Detected Cycle Slips (Integrity Events)", fontweight="bold", fontsize=14)
        ax.legend()
        GNSSColors.apply_theme(ax)
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, transparent=True)
            plt.close()

    def plot_global_l1_l2_comparison_hist(self, save_path=None):
        snr = self.analyzer.get_snr()
        l1 = snr.filter(pl.col("frequency").is_in(RTKLIB_bands["single"]))["value"].to_numpy()
        l2 = snr.filter(pl.col("frequency").is_in(RTKLIB_bands["dual"]))["value"].to_numpy()
        
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_alpha(0)
        ax.hist(l1, bins=50, alpha=0.5, label="L1 (Primary)", color="#1f77b4")
        ax.hist(l2, bins=50, alpha=0.5, label="L2 (Secondary)", color="#ff7f0e")
        ax.axvline(35, color="green", ls="--", label="Target Quality")
        ax.set_title("Global Signal Distribution: L1 vs L2", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, transparent=True)
            plt.close()

    def plot_good_satellites_trend(self, epoch_df, save_path=None):
        """Plots the number of 'Good' satellites per epoch over time."""
        if epoch_df.is_empty(): return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_alpha(0)
        
        ax.fill_between(epoch_df["time"].to_numpy(), epoch_df["n_good"].to_numpy(), color=GNSSColors.FIX, alpha=0.3)
        ax.plot(epoch_df["time"].to_numpy(), epoch_df["n_good"].to_numpy(), color=GNSSColors.FIX, linewidth=2, label="Good Satellites")
        
        # Target threshold
        ax.axhline(15, color=GNSSColors.FLOAT, ls="--", alpha=0.6, label="Target (15+)")
        ax.axhline(8, color=GNSSColors.SINGLE, ls=":", alpha=0.6, label="Critical (8)")
        
        ax.set_ylabel("Satellite Count", fontweight="bold")
        ax.set_title("Constellation Health: 'Good' Satellites Over Time", fontweight="bold", fontsize=14)
        ax.legend(loc="upper right")
        GNSSColors.apply_theme(ax)
        
        # Format time axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, transparent=True)
            plt.close()
        else: 
            plt.show()

    def plot_band_comparison(self, save_path=None):
        """Grouped bar chart comparing Primary (L1) vs Secondary (L2) SNR for all constellations."""
        summary = self.analyzer.get_global_frequency_summary()
        if summary.is_empty(): return
        
        # Define Primary vs Secondary mapping
        bands_map = {
            "G": ("G1", "G2"),
            "E": ("E1", "E5b"),
            "C": ("B1", "B2"),
            "R": ("G1", "G2")
        }
        
        valid_constellations = []
        primary_vals = []
        secondary_vals = []
        
        for c, (p_band, s_band) in bands_map.items():
            p_val = summary.filter((pl.col("constellation") == c) & (pl.col("frequency") == p_band))["mean"]
            s_val = summary.filter((pl.col("constellation") == c) & (pl.col("frequency") == s_band))["mean"]
            
            if not p_val.is_empty():
                valid_constellations.append(CONSTELLATION_NAMES.get(c, c))
                primary_vals.append(p_val[0])
                secondary_vals.append(s_val[0] if not s_val.is_empty() else 0)
            
        if not valid_constellations: return

        x = np.arange(len(valid_constellations))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_alpha(0)
        
        ax.bar(x - width/2, primary_vals, width, label='Primary (L1/E1/B1)', color=GNSSColors.BAND_PRIMARY, alpha=0.9)
        ax.bar(x + width/2, secondary_vals, width, label='Secondary (L2/E5b/B2)', color=GNSSColors.BAND_SECONDARY, alpha=0.9)
        
        ax.set_ylabel('Mean SNR (dB-Hz)', fontweight="bold")
        ax.set_title('Primary vs Secondary Signal Strength Comparison', fontweight="bold", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_constellations, fontweight="bold")
        ax.legend()
        GNSSColors.apply_theme(ax)
        
        # Add values on top
        for i, v in enumerate(primary_vals):
            if v > 0: ax.text(i - width/2, v + 0.5, f"{v:.1f}", ha='center', fontweight='bold')
        for i, v in enumerate(secondary_vals):
            if v > 0: ax.text(i + width/2, v + 0.5, f"{v:.1f}", ha='center', fontweight='bold')

        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, transparent=True)
            plt.close()
        else: 
            plt.show()
