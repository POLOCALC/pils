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
    with open(logfile, errors="replace") as f:
        lines = f.readlines()

    for line in lines:
        if keyphrase in line:
            tstart = parse_log_datetime(line)
            if tstart is not None:
                return tstart, tstart.date()
    return None, None


# Payload log time stamps: "2025/12/08 19:25:43.590", "2024/04/29 18:59:31"
# and, in 2023, "02/21/2023 19:06:54"
_LOG_TIME = re.compile(
    r"(?P<date>\d{4}/\d{2}/\d{2}|\d{2}/\d{2}/\d{4})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2})(?:\.(?P<frac>\d+))?"
)


def parse_log_datetime(line: str) -> datetime.datetime | None:
    """Time stamp at the start of a payload log line (UTC, no time zone).

    Parameters
    ----------
    line : str
        Log line, e.g. ``"2025/12/08 19:25:43.590  [Thread-1]  INFO:..."``.

    Returns
    -------
    datetime.datetime or None
        None if the line has no time stamp.
    """
    match = _LOG_TIME.search(line)
    if match is None:
        return None
    date = match["date"]
    date_format = "%Y/%m/%d" if date[4] == "/" else "%m/%d/%Y"
    moment = datetime.datetime.strptime(
        f"{date} {match['time']}", f"{date_format} %H:%M:%S"
    )
    if match["frac"]:
        moment += datetime.timedelta(seconds=float(f"0.{match['frac']}"))
    return moment


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

    # Look for *_file.log (in 2023: sensors/*_log.log)
    logfiles = [f for f in aux_dir.iterdir() if f.name.endswith("_file.log")]
    if not logfiles:
        logfiles = sorted(folder.glob("*_log.log"))[:1]
    if not logfiles:
        raise FileNotFoundError(f"No log file found in {aux_dir}")
    if len(logfiles) > 1:
        raise FileExistsError(f"Multiple log files found in {aux_dir}")

    return logfiles[0]


def fahrenheit_to_celsius(temp: float) -> float:
    """Convert temperature from Fahrenheit to Celsius."""
    return (temp - 32) * 5 / 9


def standardize_datetimes(df: pl.DataFrame) -> pl.DataFrame:
    """Datetime columns as ``Datetime("us", "UTC")``.

    Naive datetimes are taken as UTC (the payload and drone loaders build
    them from UTC epochs); aware ones are converted to UTC.

    Parameters
    ----------
    df : pl.DataFrame
        Any table.

    Returns
    -------
    pl.DataFrame
        The table with its datetime columns in microseconds, UTC.
    """
    exprs = []
    for name, dtype in df.schema.items():
        if isinstance(dtype, pl.Datetime):
            col = pl.col(name).dt.cast_time_unit("us")
            if dtype.time_zone is None:
                col = col.dt.replace_time_zone("UTC")
            elif dtype.time_zone != "UTC":
                col = col.dt.convert_time_zone("UTC")
            exprs.append(col)
    return df.with_columns(exprs) if exprs else df
