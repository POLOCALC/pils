"""BlackSquare (ArduPilot) drone logs."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

# The format codes of https://ardupilot.org/copter/docs/logmessages.html
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
    "c": np.float64,  # read as float: int16_t * 100, usually scaled
    "C": np.float64,  # read as float: uint16_t * 100, usually scaled
    "e": np.float64,  # read as float: int32_t * 100, usually scaled
    "E": np.float64,  # read as float: uint32_t * 100, usually scaled
    "L": np.float64,  # read as float: int32_t * 1e7 latitude/longitude
    "M": "S64",  # uint8_t flight mode
    "q": np.int64,  # int64_t
    "Q": np.uint64,  # uint64_t
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
        except Exception as e:  # one bad message type does not lose the others
            logger.warning(f"Failed to parse message type '{msg_type}': {e}")

    return dfs


class BlackSquare:
    """Loader for BlackSquare drone ArduPilot log files.

    Parses ArduPilot binary log format and provides access to sensor data.
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize BlackSquare loader.

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

        self.gpa = self.data.get("GPA")  # not always present

        self.params = self.data["PARM"]
        self.params = self.params.with_columns(
            [pl.col("Name").cast(pl.Utf8).str.replace(r"^b'|'$", "").alias("Name")]
        )

        if "MNT" in self.data:
            self.gimbal = self.data["MNT"]
