import os
import datetime

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
    with open(logfile, "r") as f:
        lines = f.readlines()
    
    line_tstart = [line for line in lines if keyphrase in line]
    if len(line_tstart) != 0:
        tstart =  datetime.datetime.strptime(line_tstart[0].split("[")[0].replace(" ", ""), "%Y/%m/%d%H:%M:%S.%f")
        return tstart
    else:
        return None

def drop_nan_and_zero_cols(df):
    """
    Drop any columns in the given DataFrame that consist entirely of NaN or zero values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to be cleaned.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with any columns consisting of entirely NaN or zero values removed.
    """
    all_nan = df.isna().all()
    all_zero = df.eq(0).all()
    to_drop = all_nan | all_zero
    return df.loc[:, ~to_drop]

def get_path_from_keyword(dirpath, keyword):
    paths = []
    for root, dir, files in os.walk(dirpath):
        for file in files:
            if keyword in file:
                paths.append(os.path.join(root, file))

    if len(paths) == 0:
        print(f"[get_path_from_keyword] No file found for {keyword}")
        return None
    elif len(paths) == 1:
        paths = paths[0]

    return paths

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


# def get_logpath_from_datapath(datapath):
#     """
#     Given a path to a video or data file, returns the corresponding log file path.
#     The log file is expected to match '*_file.log' in the same directory or parent.

#     Parameters
#     ----------
#     datapath : str
#         Path to a video or sensor file.

#     Returns
#     -------
#     logpath : str

#     Raises
#     ------
#     FileNotFoundError
#         If no log file is found
#     FileExistsError
#         If multiple log files are found
#     """
#     print(datapath)
#     ext = os.path.splitext(datapath)[-1].lower()

#     if ext == ".mp4":
#         search_dir = os.path.abspath(os.path.join(datapath, "../"))
#     elif ext == ".bin":
#         search_dir = os.path.abspath(os.path.join(datapath, "../"))  # parent of sensor
#     else:
#         search_dir = os.path.dirname(datapath)

#     # Look for any *_file.log
#     logfiles = [os.path.join(search_dir, f)
#                 for f in os.listdir(search_dir)
#                 if f.endswith("_file.log")]

#     print(search_dir[1])

#     if len(logfiles) == 0:
#         raise FileNotFoundError(f"[get_logpath_from_datapath] No log file found in {search_dir}")
#     if len(logfiles) > 1:
#         raise FileExistsError(f"[get_logpath_from_datapath] Multiple log files found in {search_dir}")

#     return logfiles[0]

def get_logpath_from_datapath(datapath):
    """
    Given a sensor or camera file path, return the *_file.log in the aux folder.
    """
    if not os.path.exists(datapath):
        raise FileNotFoundError(f"Datapath does not exist: {datapath}")

    # Go to parent folder(s)
    folder = os.path.dirname(datapath)          # sensor file → sensors/
    aux_dir = os.path.dirname(folder)           # sensors/ → aux/

    # Look for *_file.log
    logfiles = [f for f in os.listdir(aux_dir) if f.endswith("_file.log")]
    if not logfiles:
        raise FileNotFoundError(f"No log file found in {aux_dir}")
    if len(logfiles) > 1:
        raise FileExistsError(f"Multiple log files found in {aux_dir}")

    return os.path.join(aux_dir, logfiles[0])




def far_to_celcius(temp):
    return (temp - 32)*5/9
