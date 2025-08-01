import os
import numpy as np
import pandas as pd
import datetime
from pathlib import Path

from tools import read_log_time, get_path_from_keyword
from drones.DJIDrone import DJIDrone
from drones.BlackSquareDrone import BlackSquareDrone

DATADIR = "/data/POLOCALC/campaigns"

def flightpath_from_num(num, dirpath=DATADIR):
    """
    Find the path to a flight log file with a given number.

    Parameters
    ----------
    num : int
        The number of the flight log to find.
    dir_path : str
        The root directory to search in. Defaults to
        ``/data/POLOCALC/campaigns/``.

    Returns
    -------
    str or None
        The path to the file if found, or ``None`` if the file is not found.
    """
    for root, _, _ in os.walk(dirpath):
        if os.path.isfile(os.path.join(root, f"FLY{num}.csv")):
            return os.path.join(root, f"FLY{num}.csv")
        
    print(f"[flightpath_from_num] No file found in {dirpath} for {num}.")
    return None 

def get_drone_timestamp(drone_path, model="dji"):
    """
    Retrieve the timestamp of the first GPS data entry for a given drone.

    Parameters
    ----------
    drone : DJIDrone or BlackSquareDrone
        An instance of a drone object containing GPS data.
    type : str, optional
        The type of drone, either 'dji' or 'blacksquare'. Default is 'dji'.

    Returns
    -------
    pd.Timestamp
        The timestamp of the first GPS data entry in UTC format.

    Raises
    ------
    ValueError
        If the `type` is neither 'dji' nor 'blacksquare'.
    """
    if model.lower() == "dji":
        drone = DJIDrone(drone_path)
        drone.load_data(cols=["GPS:dateTimeStamp"])
        drone_tst = pd.to_datetime(drone.data["GPS:dateTimeStamp"], utc=True).iloc[0]
    elif model.lower() == "blacksquare":
        drone = BlackSquareDrone(drone_path)
        drone.load_data()
        raise NotImplementedError("[get_drone_timestamp] BlackSquareDrone timestamp handling is not implemented yet")
    else:
        raise ValueError(f"[get_drone_timestamp] Unknown type {type}. Should be 'dji' or 'blacksquare'.")
    return drone_tst



def match_log_filename(drone_tst, sensors_root_dir=DATADIR, time_delta=pd.Timedelta(minutes=10)):
    """
    Matches and retrieves the most relevant sensor log directory and its corresponding log file
    based on the timestamp of the drone data.

    Parameters
    ----------
    drone : DJIDrone or BlackSquareDrone (not implemented yet)
        An instance of a drone object that has already loaded data with a "GPS:dateTimeStamp" column.
    sensors_root_dir : str, optional
        Root directory where the sensor log directories are located. Default is `DATADIR`.
    time_delta : pd.Timedelta, optional
        Maximum allowable difference between drone and sensor timestamps for a valid match.
        Default is 10 minutes.

    Returns
    -------
    matched_sensors_dir : str or None
        Path to the matching 'sensors_data' directory. None if no suitable match is found.
    matched_logfile : str or None
        Path to the corresponding 'file.log'. None if no suitable match is found.

    Notes
    -----
    - Assumes the sensor logs have a standard layout: each subdirectory contains a `file.log`
      and a `sensors_data/` subdirectory.
    - Uses a hardcoded keyphrase to extract timestamps from log files.
    - Returns the closest match only if the time difference is within the threshold.
    """
    matched_timestamps = []
    matched_dirs = []
    matched_logs = []

    for root, _, files in os.walk(sensors_root_dir):
        if "file.log" in files:
            logfile = os.path.join(root, "file.log")
            sensors_dir = os.path.join(root, "sensors_data")
            if not os.path.isdir(sensors_dir):
                continue
            # sensor_files = [f for f in os.listdir(sensors_dir) if f.startswith(SENSORS_DIR[sensor]["filename_prefix"])]
            # if not sensor_files:
            #     continue

            tst = read_log_time(keyphrase="INFO:Loaded configuration config/rover_config.yml", logfile=logfile)
            if tst is None:
                continue

            matched_timestamps.append(pd.to_datetime(tst, utc=True))
            matched_dirs.append(sensors_dir)
            matched_logs.append(logfile)

    if not matched_timestamps:
        print(f"[match_log_filename] No matching timestamps found.")
        return None, None

    deltas = [abs(t - drone_tst) for t in matched_timestamps]
    min_delta = min(deltas)

    if min_delta <= time_delta:
        idx = deltas.index(min_delta)
        return matched_dirs[idx], matched_logs[idx]
    else:
        print(f"[match_log_filename] Found no files with time delta <({time_delta}).")
        return None, None
    
def match_litchi_filename(drone_tst, dirpath, time_delta=pd.Timedelta(minutes=10)):
    """
    Matches and retrieves the Litchi flight log file closest in time to a given drone timestamp.

    Parameters
    ----------
    drone_tst : datetime.datetime
        The UTC timestamp of the drone log to match against.
    dirpath : str
        Directory containing Litchi log files.
    time_delta : pd.Timedelta, optional
        Maximum allowable time difference for a match. Default is 10 minutes.

    Returns
    -------
    str or None
        Full path to the matched Litchi file, or None if no match found.

    Notes
    -----
    - Litchi filenames must follow the pattern "YYYY-MM-DD_HH-MM-SS_v2.csv".
    - Timestamps are interpreted as local (America/Santiago) and converted to UTC.
    """
    # Gather candidate files
    litchi_files = get_path_from_keyword(dirpath, "_v2.csv")
    if not litchi_files:
        print(f"[match_litchi_filename] No Litchi files found in {dirpath}")
        return None

    # Parse timestamps
    valid_entries = []
    for f in litchi_files:
        try:
            ts_local = datetime.datetime.strptime(os.path.basename(f), "%Y-%m-%d_%H-%M-%S_v2.csv")
            valid_entries.append((f, ts_local))
        except ValueError:
            print(f"[match_litchi_filename] Skipping unparseable filename: {f}")
            continue

    if not valid_entries:
        print("[match_litchi_filename] No valid timestamps parsed from filenames.")
        return None

    # Build DataFrame with UTC timestamps
    litchi_df = pd.DataFrame(valid_entries, columns=["filename", "timestamp_local"])
    litchi_df["timestamp_utc"] = pd.to_datetime(litchi_df["timestamp_local"]).dt.tz_localize("America/Santiago").dt.tz_convert("UTC")

    # Find the file closest in time to the drone timestamp
    litchi_df["delta"] = np.abs(litchi_df["timestamp_utc"] - drone_tst)
    closest = litchi_df.loc[litchi_df["delta"].idxmin()]

    if closest["delta"] <= time_delta:
        return closest["filename"]
    else:
        print(f"[match_litchi_filename] No Litchi file within ({time_delta}).")
        return None

class PathHandler:

    def __init__(self, num, dirpath=DATADIR, drone_model="dji"):
        self.num = num 
        self.dirpath = dirpath
        self.drone_model = drone_model
        self.drone = None
        self.logfile = None
        self.inclino = None
        self.adc = None
        self.gps = None
        self.baro = None
        self.gyro = None
        self.accelero = None
        self.magneto = None
        self.imu_dir = None
        self.litchi = None

    def get_filenames(self, litchi=True):
        
        self.drone = flightpath_from_num(self.num, self.dirpath)
        drone_tst = get_drone_timestamp(self.drone, model=self.drone_model)
        sensors_dir, self.logfile = match_log_filename(drone_tst, self.dirpath)

        if self.logfile is None:
            print("[get_filenames] No matching logfile found. Stopped the process.")
            return

        if sensors_dir is None:
            print("[get_filenames] No matching logfile found. Stopped the process.")
        else:
            self.inclino = get_path_from_keyword(sensors_dir, "Kernel-100")
            self.adc = get_path_from_keyword(sensors_dir, "ADS1015")
            self.gps = get_path_from_keyword(sensors_dir, "ZED-F9P")
            self.baro = get_path_from_keyword(sensors_dir, "barometer")
            self.gyro = get_path_from_keyword(sensors_dir, "gyroscope")
            self.accelero = get_path_from_keyword(sensors_dir, "accelerometer")
            self.magneto = get_path_from_keyword(sensors_dir, "magnetometer")

            if litchi:
                self.litchi = match_litchi_filename(drone_tst, self.dirpath)

        





        

    
