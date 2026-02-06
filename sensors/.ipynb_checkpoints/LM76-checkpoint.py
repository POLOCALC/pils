import pandas as pd
import numpy as np
import struct
import matplotlib.pyplot as plt

from tools import is_ascii_file, get_logpath_from_datapath

class LM76:
    def __init__(self, path, logpath=None):
        self.path = path
        self.data = None

        if logpath is not None:
            self.logpath = logpath
        else:
            self.logpath = get_logpath_from_datapath(self.path)

        self.tstart = None
        
    def load_data(self):
        self.data = pd.read_csv(self.path)
        self.data["timestamp_ns"] = self.data["timestamp_ns"].values*1e-9 # correct units of time
    

    def plot(self):
        fig, axs = plt.subplots(2, 1)
        fig.suptitle("LM76 Termostat")
        axs[0].plot(self.data["timestamp_ns"], self.data["temperature_c"].values)
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel(r"Temperature [$^\circ$C]")
  
        axs[1].scatter(self.data["timestamp_ns"], self.data["status_crit"].values, label="status TCrit")
        axs[1].scatter(self.data["timestamp_ns"], self.data["status_high"].values, label="status THigh")
        axs[1].scatter(self.data["timestamp_ns"], self.data["status_low"].values, label="status TLow")
        axs[1].legend()

        plt.tight_layout()
        plt.show()