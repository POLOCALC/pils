import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd

from tools import get_logpath_from_datapath, read_log_time

class Camera:
    def __init__(self, path, logpath=None):
        self.path = path
        if logpath is not None:
            self.logpath = logpath
        else:
            self.logpath = get_logpath_from_datapath(self.path)

        self.tstart = None
        self.capture = None
        self.fps = None

    def load_data(self):
        self.capture = cv2.VideoCapture(self.path)
        self.tstart = read_log_time("INFO:Camera Sony starts recording", self.logpath)

        #Compute the fps
        frame_count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_sec = self.capture.get(cv2.CAP_PROP_FRAME_COUNT) / self.capture.get(cv2.CAP_PROP_FPS)
        self.fps = frame_count / duration_sec

    def get_frame(self, frame_number):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.capture.read()
        return frame
    
    def plot_frame(self, frame_number, color="rgb"):
        frame = self.get_frame(frame_number)

        if color == "rgb":
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif color == "hsv":
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif color == "gray":
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif color == "bgr":
            img = frame.copy()
        else:
            raise KeyError(f"{color} is not known")

        plt.figure()
        plt.imshow(img)
        plt.show()


######################################### FUNCTIONS TO DETECT THE SCANNING TIMESTAMPS #########################################

def get_flying_range(data, litchi):
    timerange = litchi.data[litchi.data["isflying"].astype("bool")]["datetime"]
    is_flying = data["datetime"].between(timerange.iloc[0], timerange.iloc[-1])
    return data[is_flying]

def detect_peaks(data, column, negate=False, width=1, prominence=1):
    sign = -1 if negate is True else 1    
    peaks, _ = sp.signal.find_peaks(sign*data[column], width=width, prominence=prominence)
    return [data["datetime"].iloc[p] for p in peaks]

def get_scanning_range(DH, margin=15, plot=False):
    # Drone peaks
    drone_data = get_flying_range(DH.drone.data, DH.litchi)
    drone_data = drone_data.groupby("datetime").first().reset_index()

    alt_peaks = detect_peaks(drone_data, "RTKdata:Hmsl_P",negate=True,  width=5, prominence=1e-7)
    lat_peaks = detect_peaks(drone_data, "RTKdata:Lat_P", negate=True, width=10, prominence=1e-4)
    lon_peaks = detect_peaks(drone_data, "RTKdata:Lon_P", negate=True, width=10, prominence=1e-4)

    # Inclino peaks
    inclino_data = get_flying_range(DH.payload.inclino.data, DH.litchi)
    inclino_peaks = detect_peaks(inclino_data, "pitch", negate=True, width=50, prominence=3)

    # Barometer peaks
    baro_data = get_flying_range(DH.payload.baro.data, DH.litchi)
    baro_peaks = detect_peaks(baro_data, "pressure", negate=False, width=1, prominence=2)

    all_peaks = alt_peaks + lat_peaks + lon_peaks + inclino_peaks + baro_peaks

    # Convert to pandas Series of datetime64[ns]
    ts_series = pd.Series(all_peaks).sort_values().reset_index(drop=True)

    # Convert to seconds since first timestamp
    seconds = (ts_series - ts_series.min()).dt.total_seconds().values

    # Find timestamps that have at least another timestamp within 1 second
    keep_mask = np.zeros(len(seconds), dtype=bool)

    for i, t in enumerate(seconds):
        # Check if any other timestamp is within 1 second
        twin = 5
        if np.any(np.abs(seconds - t) <= twin) and np.sum(np.abs(seconds - t) <= twin) > 1:
            keep_mask[i] = True

    # Filter consensus peaks
    consensus_peaks = ts_series[keep_mask]

    # Compute scanning range
    if not consensus_peaks.empty:
        scanning_range = (consensus_peaks.min(), consensus_peaks.max())
    else:
        scanning_range = None

    scan_start, scan_end = scanning_range[0] - pd.Timedelta(seconds=margin), scanning_range[1] + pd.Timedelta(seconds=margin)

    if plot:
        baro_data = get_flying_range(DH.payload.baro.data, DH.litchi)
        plt.figure()
        plt.plot(baro_data["datetime"], baro_data["pressure"])

        # Add vertical lines for each type of peak
        for dt in all_peaks:
            plt.axvline(dt, color="k", ls="--", alpha=0.2)
        plt.axvline(scan_start, color="r", ls="-")
        plt.axvline(scan_end, color="r", ls="-")
        plt.show()

    return scan_start, scan_end 