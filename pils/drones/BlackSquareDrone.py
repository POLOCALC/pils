from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from astropy.utils.iers import LeapSeconds

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

"""https://ardupilot.org/copter/docs/logmessages.html"""

ARDUTYPES = {
    "a": (np.int16, (32,)),  # int16_t[32]
    "b": np.int8,  # int8_t
    "B": np.uint8,  # uint8_t
    "h": np.int16,  # int16_t
    "H": np.uint16,  # uint16_t
    "i": np.int32,  # int32_t
    "I": np.uint32,  # uint32_t
    "f": np.float32,  # float
    "d": np.float64,  # double
    "n": "S4",  # char[4]
    "N": "S16",  # char[16]
    "Z": "S64",  # char[64]
    "c": np.float64,  # np.int16,                 # int16_t * 100, usually scaled
    "C": np.float64,  # np.uint16,                # uint16_t * 100, usually scaled
    "e": np.float64,  # np.int32,                 # int32_t * 100, usually scaled
    "E": np.float64,  # np.uint32,                # uint32_t * 100, usually scaled
    "L": np.float64,  # np.int32,                 # int32_t * 1e7 latitude/longitude
    "M": "S64",  # np.uint8,                 # uint8_t flight mode
    "q": np.int64,  # int64_t
    "Q": np.uint64,  # uint64_t
}
ARDUFACTOR = {
    "c": 100,
    "C": 100,
    "e": 100,
    "E": 100,
    "L": 1e7,
}
FLIGHTMODES = {
    "Stabilize": 0,
    "Acro": 1,
    "Altitude Hold": 2,
    "Auto": 3,
    "Guided": 4,
    "Loiter": 5,
    "RTL": 6,
    "Circle": 7,
    "Land": 9,
    "Drift": 11,
    "Sport": 13,
    "Flip": 14,
    "AutoTune": 15,
    "PosHold": 16,
    "Brake": 17,
    "Throw": 18,
    "Avoid_ADSB": 19,
    "Guided_NoGPS": 20,
    "Smart RTL": 21,
    "FlowHold": 22,
    "Follow": 23,
    "ZigZag": 24,
    "System Identification": 25,
    "Heli_Autorotate": 26,
    "Turtle": 27,
}


def messages_to_df(
    messages: list[list[str]], columns: list[str], format_str: str
) -> pl.DataFrame:
    """Convert ArduPilot log messages to Polars DataFrame.

    Parameters
    ----------
    messages : List[List[str]]
        List of message rows (each row is list of string values).
    columns : List[str]
        Column names for the DataFrame.
    format_str : str
        Format string specifying data types (ArduPilot format codes).

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with converted message data.
    """
    dtypes = []
    for col, _f in zip(columns, format_str, strict=False):
        np_dtype = ARDUTYPES.get(_f, object)
        dtypes.append((col, np_dtype))

    # Create structured array
    arr = np.array([tuple(row) for row in messages], dtype=dtypes)
    # Convert to polars DataFrame
    data_dict = {col: arr[col].tolist() for col in columns}
    return pl.DataFrame(data_dict)


def read_msgs(path: str | Path) -> dict[str, pl.DataFrame]:
    """Read ArduPilot log file and parse messages into DataFrames.

    Parameters
    ----------
    path : Union[str, Path]
        Path to ArduPilot log file.

    Returns
    -------
    Dict[str, pl.DataFrame]
        Dictionary mapping message types to DataFrames.

    Raises
    ------
    FileNotFoundError
        If log file not found.
    """
    with open(path) as f:
        lines = (line.strip() for line in f)

        # First pass: extract formats and group messages
        formats = {}
        grouped_msgs = defaultdict(list)

        for line in lines:
            # Extract all the formats
            if line.startswith("FMT"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    _, _, _, msg_type, format_str, *colnames = parts
                    formats[msg_type] = {"Format": format_str, "Columns": colnames}
            # Extract all the messages
            elif line and not line.startswith("FILE"):
                parts = [p.strip() for p in line.split(",")]
                msg_type, values = parts[0], parts[1:]
                grouped_msgs[msg_type].append(values)

    # Start parsing
    dfs = {}
    for msg_type, messages in grouped_msgs.items():
        if msg_type not in formats:
            continue

        columns = formats[msg_type]["Columns"]
        format_str = formats[msg_type]["Format"]

        try:
            df = messages_to_df(messages, columns, format_str)
            dfs[msg_type] = df
        except Exception as e:
            logger.warning(f"Failed to parse message type '{msg_type}': {e}")

    return dfs


def generate_log_file(lines: list[str]):
    """Generate log file from lines.

    Parameters
    ----------
    lines : List[str]
        List of log file lines.

    Returns
    -------
    Dict[str, Any]
        Dictionary with log file information.
    """
    msgs = []
    for line in lines:
        msg = line.strip().split(",")
        if msg[0] == "FILE":
            msgs.append([msg[1], msg[4][1:]])

    msg_df = pl.DataFrame(msgs, schema=["Name", "Log"])
    names = msg_df["Name"].unique().to_list()
    s = msg_df.filter(pl.col("Name") == names[0])["Log"].str.concat("")
    decoded_str = s[0].encode("utf-8").decode("unicode_escape")

    # Write to a text file with proper formatting
    with open("test.txt", "w", encoding="utf-8") as f:
        f.write(decoded_str)


def get_leapseconds(year: int, month: int) -> int:
    """Calculate number of leap seconds for given date.

    Parameters
    ----------
    year : int
        Year (e.g., 2024).
    month : int
        Month (1-12).

    Returns
    -------
    int
        Number of leap seconds to subtract from GPS time.
    """
    # Load and prepare DataFrame
    ls_table = LeapSeconds.auto_open()
    ls_df = pl.DataFrame(
        {
            "year": list(ls_table["year"]),  # type: ignore
            "month": list(ls_table["month"]),  # type: ignore
            "day": list(ls_table["day"]),  # type: ignore
            "tai_utc": list(ls_table["tai_utc"]),  # type: ignore
        }
    )
    ls_df = ls_df.with_columns(
        [pl.date(pl.col("year"), pl.col("month"), pl.col("day")).alias("date")]
    )
    ls_df = ls_df.sort("date")

    # Define dates
    start_date = datetime(1980, 1, 1).date()
    end_date = datetime(year, month, 1).date()

    # Find the leap seconds at start and end
    start_ls = ls_df.filter(pl.col("date") <= start_date).tail(1)["tai_utc"][0]
    end_ls = ls_df.filter(pl.col("date") <= end_date).tail(1)["tai_utc"][0]

    # Compute leap seconds
    leap_seconds = end_ls - start_ls
    return leap_seconds


class BlackSquareDrone:
    """Loader for BlackSquare drone ArduPilot log files.

    Parses ArduPilot binary log format and provides access to sensor data.
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize BlackSquareDrone loader.

        Parameters
        ----------
        path : Union[str, Path]
            Path to ArduPilot log file.
        """

        self.path = path

        self.data = None
        self.imu = None
        self.barometer = None
        self.magnetometer = None
        self.gps = None
        self.batteries = None
        self.attitude = None
        self.pwm = None
        self.position = None
        self.gpa = None
        self.gimbal = None
        self.params = None

        self.datetime = None

    def load_data(self) -> None:
        """Load all sensor data from ArduPilot log file.

        Populates instance attributes with DataFrames for each sensor type.
        """
        self.data = read_msgs(self.path)
        self.imu = self.data["IMU"]
        self.barometer = self.data["BARO"]
        self.magnetometer = self.data["MAG"]
        self.gps = self.data["GPS"]
        self.batteries = self.data["BAT"]
        self.attitude = self.data["ATT"]
        self.pwm = self.data["RCOU"]
        self.position = self.data["POS"]

        # GPA may not always be present
        if "GPA" in self.data:
            self.gpa = self.data["GPA"]
        else:
            self.gpa = None

        self.params = self.data["PARM"]
        self.params = self.params.with_columns(
            [pl.col("Name").cast(pl.Utf8).str.replace(r"^b'|'$", "").alias("Name")]
        )

        if "MNT" in self.data.keys():
            self.gimbal = self.data["MNT"]

    def compute_datetime(self) -> None:
        """Compute datetime from GPS week and milliseconds.

        Converts GPS time to UTC by subtracting leap seconds.
        """
        if self.gps is not None:
            gps = self.gps
            # GPS epoch is 1980-01-06, calculate datetime from GWk (week) and GMS (milliseconds)
            gps_epoch = datetime(1980, 1, 6)

            gps_dt = []
            for row in gps.iter_rows(named=True):
                dt = gps_epoch + timedelta(
                    weeks=int(row["GWk"]), milliseconds=int(row["GMS"])
                )
                gps_dt.append(dt)

            # Get leap seconds
            first_dt = gps_dt[0]
            leapseconds = get_leapseconds(first_dt.year, first_dt.month)

            # Subtract leap seconds
            gps_dt_corrected = [dt - timedelta(seconds=leapseconds) for dt in gps_dt]
            self.datetime = pl.Series("datetime", gps_dt_corrected)
