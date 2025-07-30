import pandas as pd
import matplotlib.pyplot as plt
from tools import drop_nan_and_zero_cols, read_log_time, get_logpath_from_datapath

from decoders import KERNEL_utils as kernel

def decode_inclino(inclino_path):
    """
    Decodes inclinometer data from a binary file and returns the decoded messages as a dictionary.

    Parameters
    ----------
    inclino_path : str
        Path to the binary file containing inclinometer data.

    Returns
    -------
    decoded_msg : dict
        A dictionary where keys are message field names and values are lists of field values extracted from the decoded messages.
    """

    with open(inclino_path, "rb") as fd:
        data = fd.read()

    #Define the starting sequence of a message
    sequence = b'\xaaU\x01\x81'
    msgs = data.split(sequence)[1:]

    decoded_msg = {}    
    for msg in msgs:
        try:
            msg = sequence + msg
            tmp = kernel.KernelMsg().decode_single(msg, return_dict=True)

            if not decoded_msg.keys():
                decoded_msg = {k:[] for k in tmp.keys()}

            for j in tmp.keys():
                decoded_msg[j].append(tmp[j])
        except:
            continue
    return decoded_msg

class Inclinometer:
    def __init__(self, path, logpath=None):
        self.path = path    
        if logpath is not None:
            self.logpath = logpath
        else:
            self.logpath = get_logpath_from_datapath(self.path)

        self.data = None
        self.tstart = None

    def read_log_time(self, logfile=None):
        keyphrases = ["Connected to KERNEL sensor Kernel-100", "Sensor Kernel-100 started"]
        for keyphrase in keyphrases:
            try:
                # Get start time from logfile
                tstart = read_log_time(keyphrase=keyphrase, logfile=logfile)
                if tstart is None:
                    continue
                else:
                    self.tstart = tstart
                    break
            except:
                print("Couldn't find start time from logfile. Skipping datetime conversion.")

    def load_data(self):
        """
        Reads a inclinometer file written in binary format and returns its content as a pandas DataFrame.

        Parameters
        ----------
        inclino_path : str
            Path to the inclinometer file to be read.
        logfile : str
            Path to the log file to be read.

        Returns
        -------
        inclino_data : pandas.DataFrame
            DataFrame with columns ["pitch", "roll", "yaw", "datetime"].
            pitch, roll, and yaw are the Euler angles of the inclinometer, and datetime is the converted timestamp in datetime format.

        """
        # Load data from binary decoder
        inclino_data = pd.DataFrame(decode_inclino(self.path))

        # Detect counter wrap-arounds (where counter resets)
        counter = inclino_data["Counter"]
        diff_counter = counter.diff()
        wraps = diff_counter.abs() > 60000
        wrap_cumsum = wraps.cumsum()
        new_counter = counter + wrap_cumsum * (counter.max() - counter.min())
        
        ind_good = (new_counter.diff() == 16) | (new_counter.diff() == 13) #must be updated to more flexibility
        new_counter = new_counter[ind_good]
        inclino_data = inclino_data[ind_good]

        # Convert counter to time (seconds)
        inclino_tst = new_counter / 2000.0  # assuming 2 kHz sampling
        inclino_data["timestamp"] = inclino_tst

        if self.logpath is not None:
            self.read_log_time(logfile=self.logpath)
            inclino_data["datetime"] = inclino_data["timestamp"].apply(lambda x: self.tstart + pd.Timedelta(seconds=x))
            inclino_data["tunix"] = inclino_data["datetime"].astype('int64') / 10**9

        # Rename Euler angles to match drone convention
        inclino_data = inclino_data.rename(columns={"Roll": "pitch", "Pitch": "roll", "Heading": "yaw"})
        inclino_data.loc[:, "pitch"] = -inclino_data["pitch"]
        inclino_data = drop_nan_and_zero_cols(inclino_data)

        self.data = inclino_data

    def plot(self):
        if self.data is None:
            raise ValueError("Data not loaded. Run load_data() first.")
        fig, axs = plt.subplots(3, 1, sharex=True)
        axs[0].plot(self.data["timestamp"], self.data["yaw"], color="cornflowerblue")
        axs[1].plot(self.data["timestamp"], self.data["pitch"], color="crimson")
        axs[2].plot(self.data["timestamp"], self.data["roll"], color="forestgreen")

        axs[0].set_ylabel("Yaw [°]")
        axs[1].set_ylabel("Pitch [°]")
        axs[2].set_ylabel("Roll [°]")
        axs[-1].set_xlabel("Time [s]")
        plt.show()

        