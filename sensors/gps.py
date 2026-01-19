import pandas as pd
from pyubx2 import UBXReader, UBX_PROTOCOL
import matplotlib.pyplot as plt

from tools import read_log_time, get_logpath_from_datapath

class GPS:
    def __init__(self, path, logpath=None):
        self.path = path
        if logpath is not None:
            self.logpath = logpath
        else:
            self.logpath = get_logpath_from_datapath(self.path)

        self.tstart = None

    def load_data(self):
        """
        Reads a GPS file written in binary format and returns its content as a pandas DataFrame.

        Parameters
        ----------
        gps_path : str
            Path to the GPS file to be read.
        logfile : str
            Path to the log file to be read.

        Returns
        -------
        gps_data : pandas.DataFrame
            DataFrame with columns ["iTOW", "ecefX", "ecefY", "ecefZ", "datetime"].
            iTOW is the time of week in milliseconds since Sunday midnight, ecefX, ecefY, ecefZ are the ECEF coordinates, and datetime is the converted timestamp in datetime format.

        """

        gps_data = []
        with open(self.path, 'rb') as stream:
            ubr = UBXReader(stream, protfilter=UBX_PROTOCOL, quitonerror=False)
            for raw_data, parsed_data in ubr:
                if parsed_data.identity == "NAV-POSLLH":
                    gps_data.append(pd.Series(parsed_data.__dict__))
            gps_data = pd.DataFrame(gps_data)[["iTOW", "lon", "lat", "height", "hMSL", "hAcc", "vAcc"]]

        # Time offset vectorized (relative to first iTOW)
        self.t_start = read_log_time(keyphrase="Sensor ZED-F9P started", logfile=self.logpath)
        gps_data["datetime"] = pd.to_datetime(self.t_start) + pd.to_timedelta(gps_data["iTOW"] - gps_data["iTOW"].iloc[0], unit="ms")
        gps_data["timestamp"] = gps_data["datetime"].astype('int64') / 10**9
        gps_data["height"] = gps_data["height"]/1000 # to get values in m

        self.data = gps_data

    def plot(self):
        if self.data is None:
            raise ValueError("Data not loaded. Run load_data() first.")
        fig, axs = plt.subplots(3, 1, sharex=True)
        axs[0].plot(self.data["timestamp"], self.data["lon"], color="mediumblue")
        axs[1].plot(self.data["timestamp"], self.data["lat"], color="darkorchid")
        axs[2].plot(self.data["timestamp"], self.data["height"], color="magenta")

        axs[0].set_ylabel("Longitude [°]")
        axs[1].set_ylabel("Latitude [°]")
        axs[2].set_ylabel("Height [m]")
        axs[-1].set_xlabel("Time [s]")
        plt.show()

