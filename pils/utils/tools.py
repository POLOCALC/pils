"""
Utility functions for file handling, log parsing, and data processing.
"""

import datetime
import os
from typing import List, Optional, Union

import polars as pl


def read_log_time(
    keyphrase: str, logfile: str
) -> tuple[Optional[datetime.datetime], Optional[datetime.date]]:
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
    tstart : datetime.datetime or None
        The timestamp extracted from the log file, or None if not found.
    date : datetime.date or None
        The date (YYYY-MM-DD) extracted from the log file, or None if not found.
    """
    with open(logfile, "r") as f:
        lines = f.readlines()

    line_tstart = [line for line in lines if keyphrase in line]
    if len(line_tstart) != 0:
        tstart = datetime.datetime.strptime(
            line_tstart[0].split("[")[0].replace(" ", ""), "%Y/%m/%d%H:%M:%S.%f"
        )
        return tstart, tstart.date()
    return None, None


def drop_nan_and_zero_cols(df: pl.DataFrame) -> pl.DataFrame:
    """
    Drop any columns in the given DataFrame that consist entirely of NaN or zero values.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame to be cleaned.

    Returns
    -------
    df : polars.DataFrame
        DataFrame with any columns consisting of entirely NaN or zero values removed.
    """
    cols_to_keep = []
    for col in df.columns:
        series = df[col]
        # Check if all null
        all_null = series.is_null().all()
        # Check if all zero (only for numeric columns)
        if series.dtype in [
            pl.Float64,
            pl.Float32,
            pl.Int64,
            pl.Int32,
            pl.Int16,
            pl.Int8,
            pl.UInt64,
            pl.UInt32,
            pl.UInt16,
            pl.UInt8,
        ]:
            all_zero = (series == 0).all()
        else:
            all_zero = False

        if not (all_null or all_zero):
            cols_to_keep.append(col)

    return df.select(cols_to_keep)


def get_path_from_keyword(dirpath: str, keyword: str) -> Optional[Union[str, List[str]]]:
    """
    Find file(s) in directory tree matching a keyword.

    Parameters
    ----------
    dirpath : str
        Directory to search in.
    keyword : str
        Filename keyword to match.

    Returns
    -------
    paths : str, list of str, or None
        Single path if one match, list of paths if multiple, None if no matches.
    """
    paths = []
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            if keyword in file:
                paths.append(os.path.join(root, file))

    if len(paths) == 0:
        return None
    elif len(paths) == 1:
        return paths[0]

    return paths


def is_ascii_file(file_bytes: bytes) -> bool:
    """
    Check if a given file is written in ASCII.

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


def get_logpath_from_datapath(datapath: str) -> str:
    """
    Given a sensor or camera file path, return the *_file.log in the aux folder.

    Parameters
    ----------
    datapath : str
        Path to sensor or camera data file.

    Returns
    -------
    logpath : str
        Path to the log file.

    Raises
    ------
    FileNotFoundError
        If no log file found in parent directory.
    FileExistsError
        If multiple log files found.
    """
    if not os.path.exists(datapath):
        raise FileNotFoundError(f"Datapath does not exist: {datapath}")

    # Go to parent folder(s)
    folder = os.path.dirname(datapath)  # sensor file → sensors/
    aux_dir = os.path.dirname(folder)  # sensors/ → aux/

    # Look for *_file.log
    logfiles = [f for f in os.listdir(aux_dir) if f.endswith("_file.log")]
    if not logfiles:
        raise FileNotFoundError(f"No log file found in {aux_dir}")
    if len(logfiles) > 1:
        raise FileExistsError(f"Multiple log files found in {aux_dir}")

    return os.path.join(aux_dir, logfiles[0])


def fahrenheit_to_celsius(temp: float) -> float:
    """Convert temperature from Fahrenheit to Celsius."""
    return (temp - 32) * 5 / 9
