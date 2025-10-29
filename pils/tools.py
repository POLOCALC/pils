import os
import datetime
import glob


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
        tstart = datetime.datetime.strptime(
            line_tstart[0].split("[")[0].replace(" ", ""), "%Y/%m/%d%H:%M:%S.%f"
        )
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
        file_bytes.decode("ascii")
        return True
    except UnicodeDecodeError:
        return False


def get_logpath(aux_path):
    """
    Given a path to a sensor data file anywhere under the aux folder,
    this function traverses up to find the aux folder and returns the
    path to the log file located there.

    The folder structure is expected to be:
    aux/
        file.log
        sensors_data/
            ZED-F9P_*.bin
            Kernel-100_*.bin
            Inertial_*/
                barometer.bin
                etc.

    Parameters
    ----------
    aux_path : str
        Path to a sensor data file (e.g., /home/data/aux/sensors_data/ZED-F9P_*.bin)
        or any file under the aux folder.

    Returns
    -------
    logpath : str
        Path to the log file in the aux folder.

    Raises
    ------
    FileNotFoundError
        If no log file is found in the aux folder.
    ValueError
        If the path doesn't contain an 'aux' folder.
    """
    if aux_path is None:
        raise ValueError("[get_logpath] Path is None")

    # Convert to absolute path
    aux_path = os.path.abspath(aux_path)

    # Split the path into components
    path_parts = aux_path.split(os.sep)

    # Find the 'aux' folder in the path
    try:
        aux_index = path_parts.index("aux")
    except ValueError:
        raise ValueError(f"[get_logpath] No 'aux' folder found in path: {aux_path}")

    # Reconstruct path up to and including 'aux'
    aux_folder = os.sep.join(path_parts[: aux_index + 1])

    # Look for .log files in the aux folder (not recursive)
    log_files = glob.glob(os.path.join(aux_folder, "*.log"))

    if len(log_files) == 0:
        raise FileNotFoundError(
            f"[get_logpath] No log file found in aux folder: {aux_folder}"
        )

    if len(log_files) > 1:
        # If multiple log files, try to find one that matches the date pattern
        # For now, just return the first one
        print(
            f"[get_logpath] Warning: Multiple log files found in {aux_folder}, using first one"
        )

    return log_files[0]


def far_to_celcius(temp):
    return (temp - 32) * 5 / 9
