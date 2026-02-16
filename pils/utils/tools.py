"""
Utility functions for file handling, log parsing, and data processing.
"""

import datetime
import re
from pathlib import Path

import polars as pl


def read_log_time(
    keyphrase: str, logfile: str | Path
) -> tuple[datetime.datetime | None, datetime.date | None]:
    """
    Read a log file and find the line containing the given keyphrase.
    Return the timestamp extracted from this line.

    Parameters
    ----------
    keyphrase : str
        The string to search in the log file.
    logfile : str or Path
        Path to the log file.

    Returns
    -------
    tstart : datetime.datetime or None
        The timestamp extracted from the log file, or None if not found.
    date : datetime.date or None
        The date (YYYY-MM-DD) extracted from the log file, or None if not found.
    """
    logfile = Path(logfile)  # Convert to Path if string
    with open(logfile) as f:
        lines = f.readlines()

    line_tstart = [line for line in lines if keyphrase in line]
    if len(line_tstart) != 0:
        tstart = datetime.datetime.strptime(
            line_tstart[0].split("[")[0].replace(" ", ""), "%Y/%m/%d%H:%M:%S.%f"
        )
        return tstart, tstart.date()
    return None, None


def read_alvium_log_time(keyphrase: str, logfile: str | Path) -> pl.DataFrame:
    """
    Read Alvium camera log file and extract timestamps and frame numbers.

    Parameters
    ----------
    keyphrase : str
        The string to search in the log file (e.g., "Saving frame").
    logfile : str or Path
        Path to the log file.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns 'timestamp' (Float64) and 'frame_num' (Int64).
        Returns empty DataFrame if no matches found.
    """
    logfile = Path(logfile)  # Convert to Path if string

    # Pattern to extract datetime and frame number
    pattern = r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}).*frame_(\d+)\.raw"

    data = []
    with open(logfile) as f:
        for line in f:
            if keyphrase in line:
                match = re.search(pattern, line)
                if match:
                    # Parse datetime string to timestamp
                    dt_str = match.group(1)
                    dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
                    timestamp = dt.timestamp()

                    # Get frame number
                    frame_num = int(match.group(2))

                    data.append({"timestamp": timestamp, "frame_num": frame_num})

    # Return DataFrame (empty if no matches)
    return (
        pl.DataFrame(data) if data else pl.DataFrame({"timestamp": [], "frame_num": []})
    )


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


def get_path_from_keyword(dirpath: str | Path, keyword: str) -> str | list[str] | None:
    """
    Find file(s) in directory tree matching a keyword.

    Parameters
    ----------
    dirpath : str or Path
        Directory to search in.
    keyword : str
        Filename keyword to match.

    Returns
    -------
    paths : str, list of str, or None
        Single path if one match, list of paths if multiple, None if no matches.
    """
    dirpath = Path(dirpath)  # Convert to Path if string
    paths = []

    # Use rglob for recursive search
    for file_path in dirpath.rglob("*"):
        if file_path.is_file() and keyword in file_path.name:
            paths.append(str(file_path))

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


def get_logpath_from_datapath(datapath: str | Path) -> Path:
    """
    Given a sensor or camera file path, return the *_file.log in the aux folder.

    Parameters
    ----------
    datapath : str or Path
        Path to sensor or camera data file.

    Returns
    -------
    logpath : Path
        Path to the log file.

    Raises
    ------
    FileNotFoundError
        If no log file found in parent directory.
    FileExistsError
        If multiple log files found.
    """
    datapath = Path(datapath)  # Convert to Path if string

    if not datapath.exists():
        raise FileNotFoundError(f"Datapath does not exist: {datapath}")

    # Go to parent folder(s)
    folder = datapath.parent  # sensor file → sensors/
    aux_dir = folder.parent  # sensors/ → aux/

    # Look for *_file.log
    logfiles = [f for f in aux_dir.iterdir() if f.name.endswith("_file.log")]
    if not logfiles:
        raise FileNotFoundError(f"No log file found in {aux_dir}")
    if len(logfiles) > 1:
        raise FileExistsError(f"Multiple log files found in {aux_dir}")

    return logfiles[0]


def fahrenheit_to_celsius(temp: float) -> float:
    """Convert temperature from Fahrenheit to Celsius."""
    return (temp - 32) * 5 / 9
