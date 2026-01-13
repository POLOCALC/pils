import pandas as pd
import numpy as np
import datetime

import os
import struct

#Inclinometer reader scripts
import sys
from decoders import KERNEL_utils as kernel
from pyubx2 import UBXReader, UBX_PROTOCOL

from tools import drop_nan_and_zero_cols

import time

def get_drone_data(datapath):
    """
    Read the drone data from the given CSV file. The function filters the
    data to select only the rows in which both the RTK latitude and the
    computed latitude are valid. The function also filters the data to
    select only the rows in which the RTK latitude refreshed.

    Parameters
    ----------
    datapath : string
        The path to the CSV file containing the drone data.

    Returns
    -------
    drone_data : pandas.DataFrame
        A DataFrame containing the filtered drone data.
    """
    data = pd.read_csv(datapath, low_memory=False, usecols=["Clock:offsetTime", 'GPS:dateTimeStamp', "RTKdata:GpsState", "RTKdata:Lat_P", "RTKdata:Lon_P", "RTKdata:Hmsl_P"])
    ind = data["RTKdata:GpsState"].notna() & data["RTKdata:Lat_P"] != 0 & data["GPS:dateTimeStamp"].notna()
    data = data[ind].reset_index(drop=True)
    data["GPS:dateTimeStamp"] = pd.to_datetime(data["GPS:dateTimeStamp"].values)
    data = drop_nan_and_zero_cols(data)

    return data

def get_filenames(drone_data, sensors_root_dir="/home/tsouverin/polocalc/2024_12_campaign", litchi_dir="/home/tsouverin/polocalc/data/litchi_flightlogs/"):
    """
    Match and retrieve file paths for litchi, inclinometer, ADC, and GPS data
    based on the drone data timestamp.

    Parameters
    ----------
    drone_data : pandas.DataFrame
        DataFrame containing the drone data with a "GPS:dateTimeStamp" column.
    sensors_root_dir : str, optional
        Root directory where sensor log files are stored.
    litchi_dir : str, optional
        Directory where litchi flight log files are stored.

    Returns
    -------
    litchi_path : str
        Path to the matched litchi file or None if no match is found.
    inclino_path : str
        Path to the matched inclinometer file or None if no match is found.
    adc_path : str
        Path to the matched ADC file or None if no match is found.
    gps_path : str
        Path to the matched GPS file or None if no match is found.
    logfile : str
        Path to the logfile from which the matching files were derived, or
        None if no matches are found.
    """

    litchi_path = match_litchi_filename(drone_data, litchi_dir=litchi_dir)
    inclino_path, logfile1 = match_inclino_filenames(drone_data, sensors_root_dir=sensors_root_dir)
    adc_path, logfile2 = match_adc_filenames(drone_data, sensors_root_dir=sensors_root_dir)
    gps_path, logfile3 = match_gps_filenames(drone_data, sensors_root_dir=sensors_root_dir)

    if logfile1 != logfile2:
        print("Inclinometer and ADC files do not match")
    if logfile2 != logfile3:
        print("ADC and GPS files do not match")
    if logfile2 != logfile3:
        print("Inclinometer and GPS files do not match")

    if logfile1 is not None:
        logfile = logfile1
    elif logfile2 is not None:  
        logfile = logfile2
    elif logfile3 is not None:
        logfile = logfile3
    else:
        logfile = None

    return litchi_path, inclino_path, adc_path, gps_path, logfile


def match_litchi_filename(drone_cat, litchi_dir="/home/tsouverin/polocalc/data/litchi_flightlogs/"):    
    """
    Match and retrieve the file path for a Litchi flight log file based on
    the drone data timestamp.

    Parameters
    ----------
    drone_cat : pandas.DataFrame
        DataFrame containing the drone data with a "GPS:dateTimeStamp" column.
    litchi_dir : str, optional
        Directory where Litchi flight log files are stored.

    Returns
    -------
    str or None
        Path to the matched Litchi file if found, or None if no match is found.
    """
    drone_tst = pd.to_datetime(drone_cat["GPS:dateTimeStamp"].iloc[0])

    #Look for litchi files with matching datetime
    files = [f for f in os.listdir(litchi_dir) if f.endswith("_v2.csv") and f.startswith("20")]

    if not files:
        print("No files found.")
        return None

    #Look for the closest litchi datetime to the drone data
    litchi_data = pd.DataFrame({
        "filename": files,
        "timestamp": [datetime.datetime.strptime(f, "%Y-%m-%d_%H-%M-%S_v2.csv") for f in files]
    })
    
    litchi_data["timestamp_utc"] = pd.to_datetime(litchi_data["timestamp"]).dt.tz_localize("America/Santiago").dt.tz_convert("UTC")

    #Return the closest litchi file if the timedelta betwwen drone and litchi files are less than 20 minutes
    deltas = np.abs(litchi_data["timestamp_utc"] - drone_tst)

    min_delta = deltas.min()
    if min_delta < pd.Timedelta(minutes=10):
        best_match = litchi_data.loc[deltas.idxmin(), "filename"]
        return os.path.join(litchi_dir, best_match)
    else:
        print("No timing match.")
        return None

def read_log_time(keyphrase, logfile):
    """
    Read a log file and find the line containing the given keyphrase.
    Return the timestamp extracted from this line.

    Parameters
    ----------
    keyphrase : str
        The string to search in the log file.
    logfile : str
        Path to the log file.

    Returns
    -------
    tstart : datetime.datetime
        The timestamp extracted from the log file.
    """
    f = open(logfile, "r")
    lines = f.readlines()
    
    line_tstart = [line for line in lines if keyphrase in line]
    if len(line_tstart) != 0:
        tstart =  datetime.datetime.strptime(line_tstart[0].split("[")[0].replace(" ", ""), "%Y/%m/%d%H:%M:%S.%f")
        return tstart
    else:
        return None

def match_inclino_filenames(drone_cat, sensors_root_dir="/home/tsouverin/polocalc/2024_12_campaign"):
    """
    Match and retrieve the file path for an inclinometer data file based on
    the drone data timestamp.

    Parameters
    ----------
    drone_cat : pandas.DataFrame
        DataFrame containing the drone data with a "GPS:dateTimeStamp" column.
    sensors_root_dir : str, optional
        Root directory where sensor log files are stored.

    Returns
    -------
    inclino_path : str or None
        Path to the matched inclinometer file if found, or None if no match is found.
    logfile : str or None
        Path to the logfile from which the matching inclinometer file was derived,

        or None if no matches are found.
    """
    #Load all log files
    logfiles = [os.path.join(root, "file.log") for root, _, files in os.walk(sensors_root_dir) if "file.log" in files]

    #Fetch timestamps in log files and corresponding inclinometer filenames
    inclino_tst = []
    inclino_files = []
    return_logfiles = []
    for logfile in logfiles:
        f = open(logfile, "r")
        lines = f.readlines()
        if len(lines) > 0:
            sensors_dir = os.path.join(os.path.dirname(logfile), "sensors_data")
            sensors_files = [file for file in os.listdir(sensors_dir)]
            is_inclino_file = [f for f in sensors_files if f.startswith("Kernel")]
            if len(is_inclino_file) > 0:
                tst = read_log_time(keyphrase="Sensor Kernel-100 started", logfile=logfile)
                inclino_tst.append(tst) 
                inclino_files.append(os.path.join(sensors_dir, os.path.join(sensors_dir, is_inclino_file[0])))
                return_logfiles.append(logfile)
    #Return the closest inclinometer file if the timedelta betwwen drone and litchi files are less than 20 minutes
    drone_tst = pd.to_datetime(drone_cat["GPS:dateTimeStamp"], utc=True)[0]
    delta_t = ([pd.to_datetime(tst, utc=True) - drone_tst for tst in inclino_tst])
    if np.abs(np.array(delta_t)).min() < pd.Timedelta(minutes=10):
        drone_tst = pd.to_datetime(drone_cat["GPS:dateTimeStamp"], utc=True)[0]
        match_file = np.abs(np.array(delta_t)).argmin()
        inclino_path = inclino_files[match_file]
        logfile = return_logfiles[match_file]
    else:
        inclino_path = None
        logfile = None
    return inclino_path, logfile

def match_adc_filenames(drone_cat, sensors_root_dir="/home/tsouverin/polocalc/2024_12_campaign"):
    """
    Match and retrieve the file path for an ADC data file based on
    the drone data timestamp.

    Parameters
    ----------
    drone_cat : pandas.DataFrame
        DataFrame containing the drone data with a "GPS:dateTimeStamp" column.
    sensors_root_dir : str, optional
        Root directory where sensor log files are stored.

    Returns
    -------
    adc_path : str or None
        Path to the matched ADC file if found, or None if no match is found.
    logfile : str or None
        Path to the logfile from which the matching ADC file was derived,
        or None if no matches are found.
    """
    #Load all log files
    logfiles = [os.path.join(root, "file.log") for root, _, files in os.walk(sensors_root_dir) if "file.log" in files]

    #Fetch timestamps in log files and corresponding gps filenames
    adc_tst = []
    adc_files = []
    return_logfiles = []
    for logfile in logfiles:
        f = open(logfile, "r")
        lines = f.readlines()
        if len(lines) > 0:
            sensors_dir = os.path.join(os.path.dirname(logfile), "sensors_data")
            sensors_files = [file for file in os.listdir(sensors_dir)]
            is_adc_file = [f for f in sensors_files if f.startswith("ADS1015")]
            if len(is_adc_file) > 0:
                tst = read_log_time(keyphrase="Sensor ADS1015 started", logfile=logfile)
                adc_tst.append(tst) 
                adc_files.append(os.path.join(sensors_dir, os.path.join(sensors_dir, is_adc_file[0])))
                return_logfiles.append(logfile)

    #Return the closest adc file if the timedelta between drone and adc files are less than 20 minutes
    drone_tst = pd.to_datetime(drone_cat["GPS:dateTimeStamp"], utc=True)[0]
    delta_t = ([pd.to_datetime(tst, utc=True) - drone_tst for tst in adc_tst])
    if np.abs(np.array(delta_t)).min() < pd.Timedelta(minutes=10):
        match_file = np.abs(np.array(delta_t)).argmin()
        adc_path = adc_files[match_file]
        logfile = return_logfiles[match_file]
    else:
        adc_path = None
        logfile = None
    return adc_path, logfile

def match_gps_filenames(drone_cat, sensors_root_dir="/home/tsouverin/polocalc/2024_12_campaign"):
    """
    Match and retrieve the file path for a GPS data file based on
    the drone data timestamp.

    Parameters
    ----------
    drone_cat : pandas.DataFrame
        DataFrame containing the drone data with a "GPS:dateTimeStamp" column.
    sensors_root_dir : str, optional
        Root directory where sensor log files are stored.

    Returns
    -------
    gps_path : str or None
        Path to the matched GPS file if found, or None if no match is found.
    logfile : str or None
        Path to the logfile from which the matching GPS file was derived,
        or None if no matches are found.
    """    
    #Load all log files
    logfiles = [os.path.join(root, "file.log") for root, _, files in os.walk(sensors_root_dir) if "file.log" in files]

    #Fetch timestamps in log files and corresponding gps filenames
    gps_tst = []
    gps_files = []
    return_logfiles = []
    for logfile in logfiles:
        f = open(logfile, "r")
        lines = f.readlines()
        if len(lines) > 0:
            sensors_dir = os.path.join(os.path.dirname(logfile), "sensors_data")
            sensors_files = [file for file in os.listdir(sensors_dir)]
            is_gps_file = [f for f in sensors_files if f.startswith("ZED-F9P")]
            if len(is_gps_file) > 0:
                tst = read_log_time(keyphrase="Sensor ZED-F9P started", logfile=logfile)
                gps_tst.append(tst) 
                gps_files.append(os.path.join(sensors_dir, os.path.join(sensors_dir, is_gps_file[0])))
                return_logfiles.append(logfile)

    #Return the closest inclinometer file if the timedelta between drone and gps files are less than 20 minutes
    drone_tst = pd.to_datetime(drone_cat["GPS:dateTimeStamp"], utc=True)[0]
    delta_t = ([pd.to_datetime(tst, utc=True) - drone_tst for tst in gps_tst])
    if np.abs(np.array(delta_t)).min() < pd.Timedelta(minutes=10):
        match_file = np.abs(np.array(delta_t)).argmin()
        gps_path = gps_files[match_file]
        logfile = return_logfiles[match_file]
    else:
        gps_path = None
        logfile = None
    return gps_path, logfile

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

    reps = int(len(data) / 20)
    vals = struct.unpack("<" + pattern * reps, data)
    vals = np.reshape(np.array(vals), (reps, 3))
    adc_data = pd.DataFrame(vals, columns=["timestamp", "reading_time", "amplitude"])
    adc_data["datetime"] = pd.to_datetime(adc_data["timestamp"], unit="s")
    return adc_data

def decode_adc_file_ascii(adc_path, gain_value=0.256):
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

    adc_data = []
    adc_gain = gain_value #Need to be tracked from the config file
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
    adc_data["amplitude"] *= adc_gain/2048*1e3
    # Normalize timestamps (optional but helpful for plotting)
    adc_data["timestamp"] = adc_data["timestamp"] / 1e6  # convert from us to seconds
    adc_data["amplitude"] = adc_data["amplitude"]
    #adc_data["datetime"] = pd.to_datetime(adc_data["timestamp"], unit="s")

    return adc_data

def is_ascii_file(file_bytes):
    """
    Checks if a given file is written in ASCII.

    Parameters
    ----------
    file_bytes : bytes
        Bytes from the file to be checked.

    Returns
    -------
    is_ascii : bool
        True if the file is written in ASCII, False otherwise.
    """
    try:
        file_bytes.decode('ascii')
        return True
    except UnicodeDecodeError:
        return False
    

ADS1015_VALUE_GAIN = {
    1: 4.096,
    2: 2.048,
    4: 1.024,
    8: 0.512,
    16: 0.256,
}
    
def read_adc_file(adc_path, gain_config=16):
    """
    Reads an ADC file and returns its content as a pandas DataFrame.
    
    Parameters
    ----------
    adc_path : str
        Path to the ADC file to be read.
    
    Returns
    -------
    adc_data : pandas.DataFrame
        DataFrame with columns ["time", "reading_time", "amplitude", "datetime"].
        Time is the timestamp in seconds since the epoch, reading_time is the time
        it took to take the measurement, amplitude is the measurement itself, and
        datetime is the converted timestamp in datetime format.
    """
    with open(adc_path, "rb") as f:
        data = f.read()

    gain_value = ADS1015_VALUE_GAIN[gain_config]

    if is_ascii_file(data):
        adc_data = decode_adc_file_ascii(adc_path, gain_value)
    else:
        adc_data = decode_adc_file_struct(adc_path)
    
    return adc_data

def read_gps_file(gps_path, logfile):
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
    with open(gps_path, 'rb') as stream:
        ubr = UBXReader(stream, protfilter=UBX_PROTOCOL, quitonerror=False)
        for raw_data, parsed_data in ubr:
            if parsed_data.identity == "NAV-POSLLH":
                gps_data.append(pd.Series(parsed_data.__dict__))
        gps_data = pd.DataFrame(gps_data)[["iTOW", "lon", "lat", "height", "hMSL", "hAcc", "vAcc"]]


    # # Read binary GPS data
    # with open(gps_path, "rb") as fstream:
    #     data = ubx.read(fstream)

    # # Get NAV-POSLLH data directly if available
    # if "NAV-POSLLH" not in data:
    #     raise ValueError("NAV-POSLLH data not found in GPS file.")

    # gps_data = pd.DataFrame(data["NAV-POSLLH"])

    # Time offset vectorized (relative to first iTOW)
    t_start = read_log_time(keyphrase="Sensor ZED-F9P started", logfile=logfile)
    gps_data["datetime"] = pd.to_datetime(t_start) + pd.to_timedelta(gps_data["iTOW"] - gps_data["iTOW"].iloc[0], unit="ms")
    # gps_data["timestamp"] = (gps_data["datetime"] - gps_data["datetime"][0]) // pd.Timedelta('1s')

    return gps_data

def read_inclino_file(inclino_path, logfile=None):
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
    inclino_data = pd.DataFrame(decode_inclino(inclino_path))

    # Detect counter wrap-arounds (where counter resets)
    counter = inclino_data["Counter"]
    diff_counter = counter.diff()
    wraps = diff_counter.abs() > 60000
    wrap_cumsum = wraps.cumsum()
    new_counter = counter + wrap_cumsum * (counter.max() - counter.min())

    # Convert counter to time (seconds)
    inclino_tst = new_counter / 2000.0  # assuming 2 kHz sampling
    inclino_data["timestamp"] = inclino_tst

    if logfile is not None:
        keyphrases = ["Connected to KERNEL sensor Kernel-100", "Sensor Kernel-100 started"]
        for keyphrase in keyphrases:
            try:
                # Get start time from logfile
                tstart = read_log_time(keyphrase=keyphrase, logfile=logfile)
                if tstart is None:
                    continue
                else:
                    inclino_data["datetime"] = tstart + pd.to_timedelta(inclino_tst, unit="s")
                    break
            except:
                print("Couldn't find start time from logfile. Skipping datetime conversion.")
    else:
        print("No logfile given. Skipping datetime conversion.")


    # Rename Euler angles to match drone convention
    inclino_data = inclino_data.rename(columns={"Roll": "pitch", "Pitch": "roll", "Heading": "yaw"})
    inclino_data = drop_nan_and_zero_cols(inclino_data)

    return inclino_data

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

def read_litchi_file(litchi_path):
    """
    Reads a Litchi flight log file and returns its content as a pandas DataFrame.

    Parameters
    ----------
    litchi_path : str
        Path to the Litchi flight log file to be read.

    Returns
    -------
    litchi_data : pandas.DataFrame
        DataFrame containing the Litchi flight log data with columns 
        ["datetime(utc)", "datetime(local)"] and other relevant data columns.
        The columns contain timestamps in UTC and local time, respectively.
        DataFrame is cleaned of any columns with all NaN or zero values.
    """
    litchi_cols = ['latitude', 'longitude', 'altitude(m)', 'speed(mps)', 'distance(m)', 'velocityX(mps)', 'velocityY(mps)', 'velocityZ(mps)', 'pitch(deg)', 'roll(deg)', 'yaw(deg)',
                   'batteryTemperature', 'pitchRaw', 'rollRaw', 'yawRaw', 'gimbalPitchRaw', 'gimbalRollRaw', 'gimbalYawRaw', "datetime(utc)"]
    litchi_data = pd.read_csv(litchi_path, usecols=litchi_cols)
    litchi_data["datetime(utc)"] = pd.to_datetime(litchi_data["datetime(utc)"].values)
    litchi_data = litchi_data.rename(columns={"datetime(utc)": "datetime"})
    # litchi_data["datetime(local)"] = pd.to_datetime(litchi_data["datetime(local)"].values)
    litchi_data = drop_nan_and_zero_cols(litchi_data)
    return litchi_data

def flightpath_from_num(num, dir_path="/home/tsouverin/polocalc/data/dji_log_data/"):
    """
    Find the path to a flight log file with a given number.

    Parameters
    ----------
    num : int
        The number of the flight log to find.
    dir_path : str
        The root directory to search in. Defaults to
        ``/home/tsouverin/polocalc/data/dji_log_data/``.

    Returns
    -------
    str or None
        The path to the file if found, or ``None`` if the file is not found.
    """
    for root, _, _ in os.walk(dir_path):
        if os.path.isfile(os.path.join(root, f"FLY{num}.csv")):
            return os.path.join(root, f"FLY{num}.csv")
    return None 
