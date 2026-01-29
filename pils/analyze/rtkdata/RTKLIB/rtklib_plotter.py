import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl
import os
from ..utils import GNSSColors

class RTKLIBPlotter:
    def __init__(self, pos_analyzer=None, stat_analyzer=None):
        self.pos = pos_analyzer
        self.stat = stat_analyzer

    def plot_skyplot_snr(self, save_path=None):
        """Polar skyplot with SNR tracks."""
        if not self.stat or self.stat.df.is_empty(): return
        
        df = self.stat.df
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        fig.patch.set_alpha(0)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rlim(90, 0)
        
        # Plot all points
        sc = ax.scatter(np.deg2rad(df["azimuth"]), 90 - df["elevation"], 
                        c=df["snr"], cmap='viridis', s=5, alpha=0.5)
        plt.colorbar(sc, ax=ax, label="SNR (dB-Hz)")
        
        # Add labels for each satellite
        for sat in df["satellite"].unique().to_list():
            sub = df.filter(pl.col("satellite") == sat)
            if not sub.is_empty():
                ax.text(np.deg2rad(sub["azimuth"].mean()), 90 - sub["elevation"].mean(), 
                        sat, fontsize=8, fontweight='bold', ha='center',
                        bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))
        
        ax.set_title("Skyplot: Satellite Tracks (Colored by SNR)", fontweight='bold', fontsize=14, pad=30)
        GNSSColors.apply_theme(ax)
        plt.tight_layout()
        
        if save_path: 
            plt.savefig(save_path, transparent=True)
            plt.close()
        else: 
            plt.show()

    def plot_trajectory_q(self, save_path=None):
        """Plots trajectory (Lon vs Lat) colored by Q status."""
        if not self.pos or self.pos.df.is_empty(): return
        
        df = self.pos.df
        q_colors = {1: GNSSColors.FIX, 2: GNSSColors.FLOAT, 5: GNSSColors.SINGLE}
        q_labels = {1: 'Fix', 2: 'Float', 5: 'Single'}
        
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_alpha(0)
        
        for q in sorted(df["Q"].unique().to_list()):
            sub = df.filter(pl.col("Q") == q)
            ax.scatter(sub["lon"], sub["lat"], c=q_colors.get(q, 'black'), 
                       label=q_labels.get(q, f'Q={q}'), s=15, alpha=0.7)
            
        ax.set_title("Trajectory Map (Colored by Solution Quality)", fontweight="bold")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.legend()
        GNSSColors.apply_theme(ax)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, transparent=True)
            plt.close()
        else:
            plt.show()

    def plot_ratio_time(self, save_path=None):
        """Plots AR Ratio over time."""
        if not self.pos or self.pos.df.is_empty(): return
        
        df = self.pos.df
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_alpha(0)
        
        ax.plot(df["time"].to_numpy(), df["ratio"].to_numpy(), color='purple', linewidth=1.5)
        ax.axhline(3.0, color='red', linestyle='--', label='Fix Threshold (3.0)')
        
        ax.set_title("AR Ratio vs Time", fontweight="bold")
        ax.set_ylabel("Ratio")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.legend()
        GNSSColors.apply_theme(ax)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, transparent=True)
            plt.close()
        else:
            plt.show()

    def plot_snr_vs_elevation(self, save_path=None):
        """Plots SNR vs Elevation from .stat file, with band legends."""
        if not self.stat or self.stat.df.is_empty(): return
        
        df = self.stat.df
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_alpha(0)
        
        bands = sorted(df["frequency"].unique().to_list())
        colors = [GNSSColors.BAND_PRIMARY, GNSSColors.BAND_SECONDARY, GNSSColors.BAND_TERTIARY]
        
        for i, b in enumerate(bands):
            sub = df.filter(pl.col("frequency") == b)
            ax.scatter(sub["elevation"].to_numpy(), sub["snr"].to_numpy(), 
                       color=colors[i % len(colors)], s=10, alpha=0.3, label=f"Band {b}")
        
        ax.set_title("Signal Strength (SNR) vs Elevation", fontweight="bold")
        ax.set_xlabel("Elevation (deg)")
        ax.set_ylabel("SNR (dB-Hz)")
        ax.legend()
        GNSSColors.apply_theme(ax)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, transparent=True)
            plt.close()
        else:
            plt.show()

    def plot_enu_time_series(self, save_path=None):
        """Plots East, North, Up error time series colored by Q status."""
        if not self.pos or self.pos.df.is_empty(): return
        if "east" not in self.pos.df.columns: return
        
        df = self.pos.df
        q_colors = {1: GNSSColors.FIX, 2: GNSSColors.FLOAT, 5: GNSSColors.SINGLE}
        q_labels = {1: 'Fix', 2: 'Float', 5: 'Single'}
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.patch.set_alpha(0)
        
        cols = ['east', 'north', 'up']
        labels = ['East (m)', 'North (m)', 'Up (m)']
        
        for i, (col, label) in enumerate(zip(cols, labels)):
            axes[i].plot(df["time"].to_numpy(), df[col].to_numpy(), color='black', linewidth=0.5, alpha=0.2)
            for q in sorted(df["Q"].unique().to_list()):
                sub = df.filter(pl.col("Q") == q)
                axes[i].scatter(sub["time"].to_numpy(), sub[col].to_numpy(), 
                                c=q_colors.get(q, 'gray'), s=2, label=q_labels.get(q, f'Q={q}') if i == 0 else "")
            axes[i].set_ylabel(label, fontweight="bold")
            GNSSColors.apply_theme(axes[i])
            
        axes[0].set_title("ENU Position Deviations Over Time (Colored by Fix Q)", fontweight="bold", fontsize=14)
        axes[2].set_xlabel("Time", fontweight="bold")
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        if 1 in df["Q"]: axes[0].legend(loc='upper right')
        
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, transparent=True)
            plt.close()
        else: 
            plt.show()

    def plot_residual_distribution_dual(self, save_path=None):
        """Overimposed histograms for Phase and Pseudorange residuals split by band."""
        if not self.stat or self.stat.df.is_empty(): return
        
        df = self.stat.df
        bands = sorted(df["frequency"].unique().to_list())
        colors = [GNSSColors.BAND_PRIMARY, GNSSColors.BAND_SECONDARY]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_alpha(0)
        
        for i, b in enumerate(bands):
            sub = df.filter(pl.col("frequency") == b)
            ax1.hist(sub["resid_phase"].to_numpy(), bins=100, range=(-0.1, 0.1), 
                    color=colors[i % 2], alpha=0.5, label=f"Band {b}", edgecolor='black')
            ax2.hist(sub["resid_code"].to_numpy(), bins=100, range=(-5, 5), 
                    color=colors[i % 2], alpha=0.5, label=f"Band {b}", edgecolor='black')
            
        ax1.set_title("Carrier Phase Residual Distribution", fontweight="bold")
        ax1.set_xlabel("Residual (m)")
        ax1.legend()
        GNSSColors.apply_theme(ax1)
        
        ax2.set_title("Pseudorange Residual Distribution", fontweight="bold")
        ax2.set_xlabel("Residual (m)")
        ax2.legend()
        GNSSColors.apply_theme(ax2)
        
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, transparent=True)
            plt.close()
        else: 
            plt.show()

    def plot_avg_snr_time_series(self, save_path=None):
        """Average SNR per band as a function of time with STD shading, colored by satellite count."""
        if not self.stat or self.stat.df.is_empty(): return
        
        agg = (self.stat.df.group_by(["tow", "frequency"])
               .agg([
                   pl.col("snr").mean().alias("avg_snr"),
                   pl.col("snr").std().alias("std_snr"),
                   pl.col("satellite").count().alias("n_sats")
               ]).sort(["tow", "frequency"]))
        
        if agg.is_empty(): return
        
        bands = sorted(agg["frequency"].unique().to_list())
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_alpha(0)
        
        for i, b in enumerate(bands):
            sub = agg.filter(pl.col("frequency") == b)
            t = sub["tow"].to_numpy()
            y = sub["avg_snr"].to_numpy()
            s = sub["std_snr"].fill_null(0).to_numpy()
            n = sub["n_sats"].to_numpy()
            
            line_color = GNSSColors.BAND_PRIMARY if i == 0 else GNSSColors.BAND_SECONDARY
            ax.plot(t, y, color=line_color, label=f"Band {b} Mean", linewidth=2)
            # Increased alpha and better contrast for shading
            ax.fill_between(t, y - s, y + s, color=line_color, alpha=0.2)
            
            # Simple scatter for points, no colorbar as requested
            ax.scatter(t, y, color=line_color, s=15, alpha=0.6)
            
        ax.set_title("Average Signal Strength (SNR) Stability Over Time", fontweight="bold", fontsize=14)
        ax.set_xlabel("Time (TOW)")
        ax.set_ylabel("SNR (dB-Hz)")
        ax.legend()
        GNSSColors.apply_theme(ax)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, transparent=True)
            plt.close()
        else:
            plt.show()

    def plot_stat_constellation_hists_dual(self, const, save_path=None):
        """Carrier Phase and Pseudorange residuals split by band for a constellation."""
        if not self.stat or self.stat.df.is_empty(): return
        df = self.stat.df.filter(pl.col("constellation") == const)
        if df.is_empty(): return
        
        bands = sorted(df["frequency"].unique().to_list())
        colors = [GNSSColors.BAND_PRIMARY, GNSSColors.BAND_SECONDARY]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_alpha(0)
        
        for i, b in enumerate(bands):
            sub = df.filter(pl.col("frequency") == b)
            ax1.hist(sub["resid_phase"].to_numpy(), bins=50, range=(-0.1, 0.1), 
                    color=colors[i % 2], alpha=0.5, label=f"Band {b}", edgecolor='black')
            ax2.hist(sub["resid_code"].to_numpy(), bins=50, range=(-5, 5), 
                    color=colors[i % 2], alpha=0.5, label=f"Band {b}", edgecolor='black')
            
        ax1.set_title(f"Phase Residuals: {const}", fontweight="bold")
        ax1.set_xlabel("Residual (m)")
        ax1.legend()
        GNSSColors.apply_theme(ax1)
        
        ax2.set_title(f"Code Residuals: {const}", fontweight="bold")
        ax2.set_xlabel("Residual (m)")
        ax2.legend()
        GNSSColors.apply_theme(ax2)
        
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, transparent=True)
            plt.close()
        else: 
            plt.show()

    def plot_constellation_snr_time_series(self, const, save_path=None):
        """Split SNR vs Time into subplots per band, showing all satellites for that constellation."""
        if not self.stat or self.stat.df.is_empty(): return
        df = self.stat.df.filter(pl.col("constellation") == const)
        if df.is_empty(): return
        
        bands = sorted(df["frequency"].unique().to_list())
        fig, axes = plt.subplots(len(bands), 1, figsize=(14, 5 * len(bands)), sharex=True)
        if len(bands) == 1: axes = [axes]
        fig.patch.set_alpha(0)
        
        for i, b in enumerate(bands):
            sub_b = df.filter(pl.col("frequency") == b)
            sats = sorted(sub_b["satellite"].unique().to_list())
            
            for sat in sats:
                sat_data = sub_b.filter(pl.col("satellite") == sat).sort("tow")
                axes[i].plot(sat_data["tow"].to_numpy(), sat_data["snr"].to_numpy(), label=sat, alpha=0.7)
            
            axes[i].set_title(f"SNR stability - Band {b}", fontweight="bold")
            axes[i].set_ylabel("SNR (dB-Hz)")
            axes[i].legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='x-small', ncol=2)
            GNSSColors.apply_theme(axes[i])
            
        axes[-1].set_xlabel("Time (TOW)")
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, transparent=True)
            plt.close()
        else: 
            plt.show()

    def plot_sat_residual_bar(self, const, save_path=None):
        """Bar chart of P95 phase residuals per satellite."""
        if not self.stat or self.stat.df.is_empty(): return
        stats = self.stat.get_satellite_stats().filter(pl.col("satellite").str.starts_with(const))
        if stats.is_empty(): return
        
        sats = sorted(stats["satellite"].unique().to_list())
        bands = sorted(stats["frequency"].unique().to_list())
        
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_alpha(0)
        
        x = np.arange(len(sats))
        width = 0.8 / len(bands)
        
        for i, band in enumerate(bands):
            sub = stats.filter(pl.col("frequency") == band)
            vals = []
            for s in sats:
                match = sub.filter(pl.col("satellite") == s)
                vals.append(match["p95_resid_phase"].sum() if not match.is_empty() else 0)
                
            ax.bar(x + i*width - 0.4 + width/2, vals, width, 
                   color=[GNSSColors.BAND_PRIMARY, GNSSColors.BAND_SECONDARY][i % 2], 
                   label=f"Band {band}", alpha=0.8)
            
        ax.set_xticks(x)
        ax.set_xticklabels(sats, rotation=45)
        ax.set_ylabel("P95 Phase Residual (m)", fontweight="bold")
        ax.set_title(f"Satellite Peak Phase Errors: {const}", fontweight="bold")
        ax.axhline(0.05, color="red", ls="--", alpha=0.5, label="Target (5cm)")
        ax.legend()
        GNSSColors.apply_theme(ax)
        plt.tight_layout()
        
        if save_path: 
            plt.savefig(save_path, transparent=True)
            plt.close()
        else: 
            plt.show()

