import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import scipy
from pathlib import Path

import os
import argparse
import sys
sys.path.insert(0,"VNA_analysis/")
from VNA_data import VNA_data

if __name__== "__main__" :

    parser = argparse.ArgumentParser(description="Get VNA data analyzed for a given component to test.")
    parser.add_argument("-b","--band",type=str,help="Frequency band of the component",default="WR10")
    parser.add_argument("-d","--dirpath",type=str,help="Path to the VNA data .s2p")
    parser.add_argument("-m","--method",type=str,default="original",help="Decide to run the analysis with the original or corrected S params")
    parser.add_argument("-o","--outpath",default=None,type=str,help="Outpath to save results")
    parser.add_argument("-p","--plot",action="store_true",default=True,help="Boolean variable to decide to plot the analysis")
    
    args = parser.parse_args()

    data = VNA_data(args.dirpath)
    data.correct_S_params()

    if args.plot:
        data.plot_S_params()

    analysis_freqs = np.asarray([84,87,90,93,96,99,102])
    freq_interval = 0.1

    data.run_single_freqs(interest_freqs=analysis_freqs,method="original",freq_interval=freq_interval)

    if args.plot:
        n = len(analysis_freqs)
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
    
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        axes = axes.flatten()
    
        for i, f in enumerate(data.single_freqs_df["Frequency"].values):
            ax = axes[i]
            freq_mask = (data.freq > f - freq_interval) & (data.freq < f + freq_interval)
            ax.plot(data.freq[freq_mask], data.S12[freq_mask], ls="-", marker="o", color="black")
        
            avg = data.single_freqs_df["S12 avg [dB]"][data.single_freqs_df["Frequency"] == f].values[0]
            std = data.single_freqs_df["S12 std [dB]"][data.single_freqs_df["Frequency"] == f].values[0]
        
            ax.axhline(avg, color="green", ls="--")
            ax.axhspan(avg - std, avg + std, color="green", alpha=0.3)
            ax.set_title(f"S12 @ {f} GHz",size=16)
            ax.set_xlabel("Frequency [GHz]",size=14)
            ax.set_ylabel("Power [dB]",size=14)

            ax.tick_params(axis="both",labelsize=12)
    
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()

        if args.outpath is not None:
            if not os.path.exists(args.outpath):
                os.makedirs(args.outpath)
                
            plot_name = Path(args.outpath).name
            print(plot_name)
            
            plt.savefig(args.outpath+f"/{plot_name}_freq_zoom.png",dpi=300)
            
        plt.show()

        if args.outpath is not None:

            fig = plt.figure()

            plt.plot(data.freq,data.S12,color="black")
            plt.errorbar(data.single_freqs_df["Frequency"],data.single_freqs_df["S12 avg [dB]"].values,
                         yerr=data.single_freqs_df["S12 std [dB]"],
                         ls="",marker="D",color="green",label="Single freq. avg.")

            plt.xlabel("Frequency [GHz]",size=14)
            plt.ylabel("S12 power [dB]",size=14)
            plt.title(f"{plot_name}",size=16)
            plt.tick_params(axis="both",labelsize=12)
            plt.tight_layout()

            plt.savefig(args.outpath+f"/{plot_name}_VNA_meas.png",dpi=300)

            plt.show()

        
            
    if args.outpath is not None:
    
        data.save_to_csv(f"{args.outpath}/{plot_name}_VNA_test.csv")
    