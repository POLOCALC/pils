import pandas as pd
import numpy as np
import struct
import matplotlib.pyplot as plt

from tools import is_ascii_file, get_logpath_from_datapath

ADS1015_VALUE_GAIN = {
    1: 4.096,
    2: 2.048,
    4: 1.024,
    8: 0.512,
    16: 0.256,
}  

def decode_adc_file_struct(adc_path):
    """
    Decodes the old ADC file written in a structured binary format and returns its content as a pandas DataFrame.

    Parameters
    ----------
    adc_path : str
        Path to the ADC file to be decoded.

    Returns
    -------
    adc_data : pandas.DataFrame
        DataFrame with columns ["time", "reading_time", "amplitude", "datetime"].
        Time is the timestamp in seconds since the epoch, reading_time is the time
        it took to take the measurement, amplitude is the measurement itself, and
        datetime is the converted timestamp in datetime format.
    """

    pattern = "dqf"

    with open(adc_path, "rb") as f:
        data = f.read()

    reps = int(len(data)/20)
    vals = struct.unpack("<" + pattern * reps, data)
    vals = np.reshape(np.array(vals), (reps, 3))
    adc_data = pd.DataFrame(vals, columns=["timestamp", "reading_time", "amplitude"])
    adc_data["datetime"] = pd.to_datetime(adc_data["timestamp"], unit="s")
    return adc_data

def decode_adc_file_ascii(adc_path, gain_config=16):
    """
    Decodes the last version of ADC file written in ASCII format and returns its content as a list of tuples.

    Parameters
    ----------
    adc_path : str
        Path to the ADC file to be decoded.

    Returns
    -------
    adc_data : list of tuples
        List containing tuples of (timestamp, value) where timestamp is the
        time in seconds since the epoch and value is the corresponding measurement.
    """

    gain = ADS1015_VALUE_GAIN[gain_config] #Need to be tracked from the config file

    adc_data = []
    with open(adc_path, "rb") as f:
        lines = f.readlines()

    for line in lines:
        try:
            # Decode the line from bytes to ASCII and split
            text_line = line.decode("ascii").strip()
            timestamp, value = text_line.split()
            adc_data.append((int(timestamp), int(value))) #Convert from digital readout to mV
        except Exception as e:
            print(f"Error decoding file: {e}")
            continue

    adc_data = pd.DataFrame(adc_data, columns=["timestamp", "amplitude"])
    adc_data["amplitude"] *= gain/2048*1e3
    # Normalize timestamps (optional but helpful for plotting)
    adc_data["timestamp"] = adc_data["timestamp"] / 1e6  # convert from us to seconds
    #adc_data["amplitude"] = adc_data["amplitude"].astype(float)

    return adc_data

class ADC:
    def __init__(self, path, logpath=None, gain_config=16):
        self.path = path
        self.data = None

        if logpath is not None:
            self.logpath = logpath
        else:
            self.logpath = get_logpath_from_datapath(self.path)

        self.tstart = None

        with open(self.path, "rb") as f:
            data = f.read()
            self.is_ascii = is_ascii_file(data)
            f.close()

        self.gain_config = gain_config
        self.gain = ADS1015_VALUE_GAIN[gain_config]
        
    def load_data(self):
        if self.is_ascii:
            self.data = decode_adc_file_ascii(self.path, self.gain_config)
        else:
            self.data = decode_adc_file_struct(self.path)

    def plot(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.data["timestamp"], self.data["amplitude"], color="crimson")
        plt.ylabel("ADC amplitude [mV]")
        plt.xlabel("Time [s]")
        plt.show()
