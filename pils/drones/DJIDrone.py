import datetime
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.interpolate import interp1d

from ..utils.logging_config import get_logger
from ..utils.tools import drop_nan_and_zero_cols

logger = get_logger(__name__)

CRC_TABLE = [
    0x0000,
    0x1189,
    0x2312,
    0x329B,
    0x4624,
    0x57AD,
    0x6536,
    0x74BF,
    0x8C48,
    0x9DC1,
    0xAF5A,
    0xBED3,
    0xCA6C,
    0xDBE5,
    0xE97E,
    0xF8F7,
    0x1081,
    0x0108,
    0x3393,
    0x221A,
    0x56A5,
    0x472C,
    0x75B7,
    0x643E,
    0x9CC9,
    0x8D40,
    0xBFDB,
    0xAE52,
    0xDAED,
    0xCB64,
    0xF9FF,
    0xE876,
    0x2102,
    0x308B,
    0x0210,
    0x1399,
    0x6726,
    0x76AF,
    0x4434,
    0x55BD,
    0xAD4A,
    0xBCC3,
    0x8E58,
    0x9FD1,
    0xEB6E,
    0xFAE7,
    0xC87C,
    0xD9F5,
    0x3183,
    0x200A,
    0x1291,
    0x0318,
    0x77A7,
    0x662E,
    0x54B5,
    0x453C,
    0xBDCB,
    0xAC42,
    0x9ED9,
    0x8F50,
    0xFBEF,
    0xEA66,
    0xD8FD,
    0xC974,
    0x4204,
    0x538D,
    0x6116,
    0x709F,
    0x0420,
    0x15A9,
    0x2732,
    0x36BB,
    0xCE4C,
    0xDFC5,
    0xED5E,
    0xFCD7,
    0x8868,
    0x99E1,
    0xAB7A,
    0xBAF3,
    0x5285,
    0x430C,
    0x7197,
    0x601E,
    0x14A1,
    0x0528,
    0x37B3,
    0x263A,
    0xDECD,
    0xCF44,
    0xFDDF,
    0xEC56,
    0x98E9,
    0x8960,
    0xBBFB,
    0xAA72,
    0x6306,
    0x728F,
    0x4014,
    0x519D,
    0x2522,
    0x34AB,
    0x0630,
    0x17B9,
    0xEF4E,
    0xFEC7,
    0xCC5C,
    0xDDD5,
    0xA96A,
    0xB8E3,
    0x8A78,
    0x9BF1,
    0x7387,
    0x620E,
    0x5095,
    0x411C,
    0x35A3,
    0x242A,
    0x16B1,
    0x0738,
    0xFFCF,
    0xEE46,
    0xDCDD,
    0xCD54,
    0xB9EB,
    0xA862,
    0x9AF9,
    0x8B70,
    0x8408,
    0x9581,
    0xA71A,
    0xB693,
    0xC22C,
    0xD3A5,
    0xE13E,
    0xF0B7,
    0x0840,
    0x19C9,
    0x2B52,
    0x3ADB,
    0x4E64,
    0x5FED,
    0x6D76,
    0x7CFF,
    0x9489,
    0x8500,
    0xB79B,
    0xA612,
    0xD2AD,
    0xC324,
    0xF1BF,
    0xE036,
    0x18C1,
    0x0948,
    0x3BD3,
    0x2A5A,
    0x5EE5,
    0x4F6C,
    0x7DF7,
    0x6C7E,
    0xA50A,
    0xB483,
    0x8618,
    0x9791,
    0xE32E,
    0xF2A7,
    0xC03C,
    0xD1B5,
    0x2942,
    0x38CB,
    0x0A50,
    0x1BD9,
    0x6F66,
    0x7EEF,
    0x4C74,
    0x5DFD,
    0xB58B,
    0xA402,
    0x9699,
    0x8710,
    0xF3AF,
    0xE226,
    0xD0BD,
    0xC134,
    0x39C3,
    0x284A,
    0x1AD1,
    0x0B58,
    0x7FE7,
    0x6E6E,
    0x5CF5,
    0x4D7C,
    0xC60C,
    0xD785,
    0xE51E,
    0xF497,
    0x8028,
    0x91A1,
    0xA33A,
    0xB2B3,
    0x4A44,
    0x5BCD,
    0x6956,
    0x78DF,
    0x0C60,
    0x1DE9,
    0x2F72,
    0x3EFB,
    0xD68D,
    0xC704,
    0xF59F,
    0xE416,
    0x90A9,
    0x8120,
    0xB3BB,
    0xA232,
    0x5AC5,
    0x4B4C,
    0x79D7,
    0x685E,
    0x1CE1,
    0x0D68,
    0x3FF3,
    0x2E7A,
    0xE70E,
    0xF687,
    0xC41C,
    0xD595,
    0xA12A,
    0xB0A3,
    0x8238,
    0x93B1,
    0x6B46,
    0x7ACF,
    0x4854,
    0x59DD,
    0x2D62,
    0x3CEB,
    0x0E70,
    0x1FF9,
    0xF78F,
    0xE606,
    0xD49D,
    0xC514,
    0xB1AB,
    0xA022,
    0x92B9,
    0x8330,
    0x7BC7,
    0x6A4E,
    0x58D5,
    0x495C,
    0x3DE3,
    0x2C6A,
    0x1EF1,
    0x0F78,
]


def calc_checksum(memory: bytes, start: int, plength: int) -> int:
    v = 0x3692  # P3

    for i in range(plength):
        byte_val = memory[start + i]
        v = CRC_TABLE[(byte_val ^ (v & 0xFF)) & 0xFF] ^ (v >> 8)
        v &= 0xFFFF  # keep 16-bit behavior

    return v


# Message type definitions with their struct formats and field mappings
MESSAGE_DEFINITIONS = {
    2096: {  # GPS data
        "name": "GPS",
        "payload_size": 66,
        "fields": [
            ("date", "I", 0),
            ("time", "I", 4),
            ("longitude", "i", 8, lambda x: x / 1.0e7),
            ("latitude", "i", 12, lambda x: x / 1.0e7),
            ("heightMSL", "i", 16, lambda x: x / 1000.0),
            ("velN", "f", 20, lambda x: x / 100.0),
            ("velE", "f", 24, lambda x: x / 100.0),
            ("velD", "f", 28, lambda x: x / 100.0),
            ("hdop", "f", 32),
            ("pdop", "f", 36),
            ("hacc", "f", 40),
            ("sacc", "f", 44),
            ("numGPS", "I", 56),
            ("numGLN", "I", 60),
            ("numSV", "H", 64),
        ],
    },
    53234: {  # RTK data
        "name": "RTK",
        "payload_size": 72,
        "fields": [
            ("date", "I", 0),
            ("time", "I", 4),
            ("lon_p", "d", 8),
            ("lat_p", "d", 16),
            ("hmsl_p", "f", 24),
            ("lon_s", "i", 28),
            ("lat_s", "i", 32),
            ("hmsl_s", "i", 36),
            ("vel_n", "f", 40),
            ("vel_e", "f", 44),
            ("vel_d", "f", 48),
            ("yaw", "h", 50),
            ("svn_s", "B", 52),
            ("svn_p", "B", 53),
            ("hdop", "f", 54),
            ("pitch", "f", 58),
            ("pos_flg_0", "B", 62),
            ("pos_flg_1", "B", 63),
            ("pos_flg_2", "B", 64),
            ("pos_flg_3", "B", 65),
            ("pos_flg_4", "B", 66),
            ("pos_flg_5", "B", 67),
            ("gps_state", "H", 68),
        ],
    },
}


class DJIDrone:
    """DJI Drone data loader supporting CSV and DAT binary formats.

    Handles loading and processing of DJI drone flight logs from both
    CSV exports and encrypted DAT binary files.

    Attributes:
        path: Path to drone data file
        data: Loaded drone data (DataFrame or dict of DataFrames)
        sync_params: Synchronization parameters (slope, intercept)
        source_format: Format of source data ('csv' or 'dat')
        aligned_df: Aligned DataFrame after GPS synchronization
    """

    def __init__(self, path: str | Path, source_format: str | None = None) -> None:
        """Initialize DJIDrone loader.

        Parameters
        ----------
        path : Union[str, Path]
            Path to DJI drone data file (CSV or DAT).
        source_format : Optional[str], optional
            Optional format specification ('csv' or 'dat').
        """
        self.path = path
        self.data: (
            dict[str, pl.DataFrame] | pl.DataFrame
        ) = {}  # Dictionary or DataFrame
        self.sync_params: tuple[float, float] | None = (
            None  # Store (slope, intercept) from Gaussian sync
        )
        self.source_format: str | None = None  # Track if data came from CSV or DAT
        self.aligned_df: pl.DataFrame | None = None  # Store aligned DataFrame

    def load_data(
        self,
        cols: list[str] | None = None,
        use_dat: bool = True,
        remove_duplicate: bool = False,
        correct_timestamp: bool = True,
        polars_interpolation: bool = True,
        align: bool = True,
    ) -> None:
        """ "Load and filter drone data from a CSV or DAT file.

        The function:
        - Loads only specified columns (for CSV).
        - Converts 'GPS:dateTimeStamp' to datetime.
        - Filters out rows with missing or zero values in critical columns.
        - Drops any columns that are fully NaN or zero using `drop_nan_and_zero_cols`.
        - Removes consecutive duplicate position samples.

        Parameters
        ----------
        cols : Optional[List[str]], optional
            List of columns to load (for CSV files). Defaults to key RTK and timestamp fields.
        use_dat : bool, optional
            If True, try to load from DAT file instead of CSV.
        remove_duplicate : bool, optional
            If True, remove consecutive duplicate position samples.
        correct_timestamp : bool, optional
            If True, correct timestamps.
        polars_interpolation : bool, optional
            If True, use polars for interpolation.
        align : bool, optional
            If True, align DAT file data with GPS.
        """
        # Auto-detect file format if not specified
        # if use_dat is None:
        #     file_extension = Path(self.path).suffix.lower()
        #     use_dat = file_extension in [".dat", ".bin"]

        if use_dat:
            self._load_from_dat()
            self.source_format = "dat"
        else:
            self._load_from_csv(cols)
            self.source_format = "csv"

        # Remove consecutive duplicate position samples
        if remove_duplicate:
            self._remove_consecutive_duplicates()

        if correct_timestamp:
            logger.info("Converting timestamps to milliseconds")
            if self.source_format == "csv":
                # For CSV format, self.data is a DataFrame
                assert isinstance(self.data, pl.DataFrame), (
                    "Expected DataFrame for CSV format"
                )
                # Calculate mean offset from actual data, not expressions
                timestamp_vals = self.data.get_column("timestamp").to_numpy()
                tags = np.where(np.diff(timestamp_vals) > 0.5)[0] + 1

                offset_vals = self.data.get_column("Clock:offsetTime").to_numpy()

                offset_vals = offset_vals[tags].astype(np.float64)
                timestamp_vals = timestamp_vals[tags].astype(np.float64)
                mean_offset = float(np.mean(timestamp_vals - offset_vals))

                self.data = self.data.with_columns(
                    ((pl.col("Clock:offsetTime") + mean_offset).cast(pl.Float64)).alias(
                        "correct_timestamp"
                    )
                )
            elif self.source_format == "dat":
                # For DAT format, align_datfile returns a DataFrame
                aligned = self.align_datfile(polars_interpolation=polars_interpolation)
                if aligned is not None:
                    self.data = aligned

        else:
            if self.source_format == "DAT" and align:
                aligned = self.align_datfile(correct_timestamp=False)

                if aligned is not None:
                    self.data = aligned

    def _load_from_csv(self, cols: list[str] | None) -> None:
        """Load drone data from CSV file.

        Parameters
        ----------
        cols : Optional[List[str]]
            List of columns to load, or None to load all columns.
        """
        # Load CSV file
        data = pl.read_csv(self.path, columns=cols) if cols else pl.read_csv(self.path)

        # Build filter conditions (only if specific columns were requested)
        filter_expr = pl.lit(True)
        apply_filters = cols is not None  # Only filter if specific columns requested

        if "GPS:dateTimeStamp" in data.columns:
            # Check if already datetime type (parsed by polars automatically)
            if data["GPS:dateTimeStamp"].dtype == pl.Datetime:
                # Already parsed, just use it
                data = data.with_columns(
                    [
                        pl.col("GPS:dateTimeStamp")
                        .dt.replace_time_zone(None)
                        .alias("datetime")
                    ]
                )
            elif (
                data["GPS:dateTimeStamp"].dtype == pl.String
                or data["GPS:dateTimeStamp"].dtype == pl.Utf8
            ):
                # Parse datetime with format string to handle timezone
                try:
                    data = data.with_columns(
                        [
                            pl.col("GPS:dateTimeStamp")
                            .str.to_datetime(
                                strict=False,
                                time_zone="UTC",  # Optional: Set to 'UTC' since your string has 'Z'
                            )
                            .alias("datetime")
                        ]
                    )
                except Exception as e:
                    # If parsing with timezone fails, try without timezone
                    logger.warning(f"Failed to parse datetime with timezone: {e}")
                    data = data.with_columns(
                        [
                            pl.col("GPS:dateTimeStamp")
                            .str.to_datetime(
                                format="%Y-%m-%d %H:%M:%S%.f", strict=False
                            )
                            .alias("GPS:dateTimeStamp")
                        ]
                    )

            data = data.with_columns(
                (pl.col("datetime").dt.timestamp("ms") / 1000).alias("timestamp")
            )

            if apply_filters:
                filter_expr = filter_expr & pl.col("GPS:dateTimeStamp").is_not_null()

        if apply_filters:
            if "RTKdata:GpsState" in data.columns:
                filter_expr = filter_expr & pl.col("RTKdata:GpsState").is_not_null()

            if "RTKdata:Lat_P" in data.columns:
                filter_expr = filter_expr & (pl.col("RTKdata:Lat_P") != 0)

            data = data.filter(filter_expr)
            data = drop_nan_and_zero_cols(data)

        # Store as 'CSV' dataset in dictionary
        self.data = data

    def _remove_consecutive_duplicates(self) -> None:
        """Remove consecutive duplicate position samples from all loaded data.

        Removes rows where ALL position columns are identical to the previous row,
        eliminating static position artifacts from the data.

        Position columns checked:
        - CSV: GPS:Lat[degrees], GPS:Long[degrees]
        - DAT: GPS:latitude, GPS:longitude
        - Both: RTKdata:Lat_P, RTKdata:Lon_P, RTKdata:Lat_S, RTKdata:Lon_S
        """
        # Handle both dict (DAT format) and DataFrame (CSV format)
        if isinstance(self.data, pl.DataFrame):
            items = [("CSV", self.data)]
        elif isinstance(self.data, dict):
            items = self.data.items()
        else:
            return

        for data_key, df in items:
            if df is None or len(df) < 2:
                continue

            # Determine which position columns are available
            # Check only GPS columns (not RTK) to identify consecutive duplicates
            gps_cols = []

            # CSV format columns
            if "GPS:Lat[degrees]" in df.columns:
                gps_cols.append("GPS:Lat[degrees]")
            if "GPS:Long[degrees]" in df.columns:
                gps_cols.append("GPS:Long[degrees]")

            # DAT format columns
            if "GPS:latitude" in df.columns:
                gps_cols.append("GPS:latitude")
            if "GPS:longitude" in df.columns:
                gps_cols.append("GPS:longitude")

            if not gps_cols:
                logger.debug(
                    f"No GPS position columns found in {data_key} data, skipping duplicate removal"
                )
                continue

            original_len = len(df)

            # Use numpy diff to find where ALL GPS position columns remain unchanged
            # Start by assuming all rows after first should be removed
            all_gps_unchanged = np.ones(original_len - 1, dtype=bool)

            for col in gps_cols:
                # Get column values as numpy array
                values = df[col].to_numpy()
                # Calculate differences
                diffs = np.diff(values)
                # Mark where this column changed (diff != 0)
                changed = diffs != 0
                # Update: row is unchanged only if ALL GPS columns are unchanged
                all_gps_unchanged = all_gps_unchanged & (~changed)

            # Indices to remove are where all_gps_unchanged is True (add 1 for diff offset)
            remove_indices = np.where(all_gps_unchanged)[0] + 1

            # Create mask: keep all except remove_indices
            keep_mask = np.ones(original_len, dtype=bool)
            keep_mask[remove_indices] = False

            # Get indices to keep
            keep_indices = np.where(keep_mask)[0]

            # Filter dataframe to keep only these indices
            filtered_df = df[keep_indices.tolist()]

            removed_count = original_len - len(filtered_df)
            if removed_count > 0:
                logger.info(
                    f"Removed {removed_count} consecutive duplicate position samples from {data_key} data ({original_len} -> {len(filtered_df)} samples, {removed_count / original_len * 100:.1f}% removed)"
                )
                if isinstance(self.data, pl.DataFrame):
                    self.data = filtered_df
                else:
                    self.data[data_key] = filtered_df
            else:
                logger.debug(f"No consecutive duplicates found in {data_key} data")

    def _load_from_dat(self) -> None:
        """Load and decode drone data from DJI DAT file.

        Raises
        ------
        FileNotFoundError
            If DAT file not found.
        Exception
            If DAT file cannot be parsed.
        """
        try:
            with open(self.path, "rb") as f:
                file_data = f.read()

            # Split messages using regex - much faster than manual parsing
            messages = re.split(b"(?=\\x55)", file_data)

            # Convert messages to records, organizing by message type
            records_by_type = {}  # {message_name: [records]}

            for msg_data in messages:
                if len(msg_data) < 12:
                    continue

                # Decode message and collect records
                decoded = self._parse_and_decode_message(msg_data)
                if len(decoded) == 0:
                    continue

                # Organize records by message type name
                for record in decoded:
                    msg_type_id = record.get("msg_type")
                    if msg_type_id in MESSAGE_DEFINITIONS:
                        msg_name = MESSAGE_DEFINITIONS[msg_type_id]["name"]
                        if msg_name not in records_by_type:
                            records_by_type[msg_name] = []
                        records_by_type[msg_name].append(record)

            # Convert to DataFrames and store in dictionary
            for msg_name, records in records_by_type.items():
                df = pl.DataFrame(records)
                # Unwrap tick values if they decrease significantly
                df = df.slice(0, df.height - 10)
                df = self._unwrap_tick(df)
                if msg_name == "GPS":
                    lon_mean = df["GPS:longitude"].mean()
                    lon_std = df["GPS:longitude"].std()
                    # Handle potential None values from mean/std
                    if lon_mean is not None and lon_std is not None:
                        lon_mean_f = float(lon_mean)
                        lon_std_f = float(lon_std)
                        df = df.filter(
                            pl.col("GPS:longitude").is_between(
                                lon_mean_f - 2 * lon_std_f,
                                lon_mean_f + 2 * lon_std_f,
                            )
                        )
                self.data[msg_name] = df
                logger.info(f"Loaded {len(records)} {msg_name} messages from DAT file")

            if not records_by_type:
                logger.warning("No messages could be decoded")

        except Exception as e:
            logger.error(f"Failed to load DAT file: {e}")
            raise

    def _parse_and_decode_message(self, msg_data: bytes) -> list[dict[str, Any]]:
        """Parse and decode a single message.

        Message structure:
        - Byte 0: 0x55 (marker)
        - Byte 1: Message length
        - Bytes 2-4: Reserved
        - Bytes 5-6: Message type (little-endian uint16)
        - Byte 6: XOR key
        - Bytes 8-11: Tick (timestamp in ticks, little-endian uint32)
        - Bytes 12+: Encrypted payload

        Parameters
        ----------
        msg_data : bytes
            Raw message bytes to parse.

        Returns
        -------
        List[Dict[str, Any]]
            List containing decoded message dictionary, or empty list if parsing fails.
        """
        if len(msg_data) < 12:
            return []
        # Handle case where message doesn't start with 0x55
        if msg_data[0] != 0x55:
            return []

        try:
            # Extract header fields
            msg_length = msg_data[1]
            msg_type = struct.unpack("<H", msg_data[4:6])[0]
            key = msg_data[6]
            tick_val = struct.unpack("<I", msg_data[6:10])[0]

            # Check if this message type is supported
            if msg_type not in MESSAGE_DEFINITIONS:
                return []

            # Check if we have enough data
            if len(msg_data) < msg_length:
                return []

            msg_def = MESSAGE_DEFINITIONS[msg_type]
            payload_size = msg_def["payload_size"]

            # Extract and decrypt payload (starts at byte 10)
            if len(msg_data) < 10 + payload_size:
                return []

            encrypted_payload = msg_data[10 : 10 + payload_size]
            decrypted = bytes(b ^ key for b in encrypted_payload)

            chk = calc_checksum(msg_data, 0, len(msg_data) - 2)

            if chk != struct.unpack("<H", msg_data[-2:])[0]:
                logger.info(
                    f"Checksum calculated {chk}, Checksum found: {struct.unpack('<H', msg_data[-2:])[0]}"
                )

                return []

            # Decode the message
            decoded = self._decode_message_data(decrypted, msg_type, tick_val, msg_def)

            return [decoded] if decoded else []

        except Exception as e:
            logger.debug(f"Failed to parse message: {e}")
            return []

    def _decode_message_data(
        self,
        decrypted_payload: bytes,
        msg_type: int,
        tick_val: int,
        msg_def: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Unified message decoder using message definition template.

        Decodes message by unpacking each field from its byte offset
        using the specified format code.

        Parameters
        ----------
        decrypted_payload : bytes
            The decrypted message payload.
        msg_type : int
            The message type identifier.
        tick_val : int
            The timestamp value in ticks.
        msg_def : Dict[str, Any]
            Message definition containing field mappings with offsets and format codes.

        Returns
        -------
        Optional[Dict[str, Any]]
            Decoded message as dictionary, or None if decoding fails.
        """
        try:
            result = {"msg_type": msg_type}

            # Unpack each field from its specified offset and format
            for field_info in msg_def["fields"]:
                field_name = msg_def["name"] + ":" + field_info[0]
                fmt_char = field_info[1]  # Format character (e.g., 'i', 'f', 'I')
                offset = field_info[2]  # Byte offset

                # Determine format string for struct.unpack_from
                fmt = f"<{fmt_char}"  # Little-endian

                # Unpack value from offset
                value = struct.unpack_from(fmt, decrypted_payload, offset)[0]

                # Apply conversion function if provided
                if len(field_info) > 3 and callable(field_info[3]):
                    result[field_name] = field_info[3](value)
                else:
                    result[field_name] = value

            # Add tick and timestamp - tick is the reliable time reference
            result["tick"] = tick_val
            # Convert to seconds

            # Try to format datetime from date/time fields (often empty for RTK)
            formatted_dt = self._format_date_time(
                result.get(msg_def["name"] + ":date", 0),
                result.get(msg_def["name"] + ":time", 0),
            )
            if formatted_dt:
                result["datetime"] = formatted_dt  # type: ignore

                fmt = "%Y-%m-%d %H:%M:%S"
                # Parse as UTC to avoid local timezone shifts
                dt = datetime.datetime.strptime(formatted_dt, fmt).replace(
                    tzinfo=datetime.UTC
                )
                result["timestamp"] = dt.timestamp()  # type: ignore

            return result

        except Exception as e:
            logger.error(f"Failed to decode message: {e}")
            return None

    @staticmethod
    def _format_date_time(date: int, time: int) -> str | None:
        """Convert date and time fields into a human-readable datetime string.

        Parameters
        ----------
        date : int
            Date as integer (YYYYMMDD format).
        time : int
            Time as integer (HHMMSS format).

        Returns
        -------
        Optional[str]
            Formatted datetime string, or None if date/time are zero or invalid.
        """
        try:
            # Return None if date/time are zero or invalid
            if date == 0 or time == 0:
                return None

            year = date // 10000
            month = (date % 10000) // 100
            day = date % 100
            hour = time // 10000
            minute = (time % 10000) // 100
            second = time % 100

            # Basic validation
            if year < 2000 or month < 1 or month > 12 or day < 1 or day > 31:
                return None

            return (
                f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
            )
        except Exception as e:
            logger.debug(f"Failed to format datetime: {e}")
            return None

    @staticmethod
    def _unwrap_tick(df: pl.DataFrame, wrap_threshold: float = 1e8) -> pl.DataFrame:
        """Unwrap tick values that wrap around due to uint32 overflow.

        Only unwraps when tick wraps from high value to near zero (< wrap_threshold).
        Ignores other negative jumps which are likely data corruption.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with 'tick' column.
        wrap_threshold : float, optional
            Maximum value after wraparound to be considered valid. Default: 1e8.

        Returns
        -------
        pl.DataFrame
            DataFrame with unwrapped tick values.
        """
        if "tick" not in df.columns or len(df) < 2:
            return df

        tick_values = df["tick"].to_list()
        unwrapped = [tick_values[0]]
        offset = 0

        for i in range(1, len(tick_values)):
            diff = tick_values[i] - tick_values[i - 1]

            # Detect uint32 wraparound: negative jump AND new value is near zero
            if diff < 0 and tick_values[i] < wrap_threshold:
                if 2**32 - wrap_threshold > tick_values[i - 1]:
                    offset += tick_values[i - 1]
                    logger.info(
                        f"Tick resetted at index {i}: {tick_values[i - 1]:,} -> {tick_values[i]:,} (adding offset {tick_values[i - 1]:,}, total offset: {offset:,})"
                    )
                else:
                    offset += 2**32  # Add uint32 max value
                    logger.info(f"{2**32 - 1e4 < tick_values[i - 1]}")
                    logger.info(
                        f"Tick unwrap at index {i}: {tick_values[i - 1]:,} -> {tick_values[i]:,} (adding offset 2^32, total offset: {offset:,})"
                    )

            unwrapped.append(tick_values[i] + offset)

        # Replace tick column with unwrapped values
        return df.with_columns(pl.Series("tick", unwrapped))

    def get_tick_offset(self) -> float:
        # Ensure self.data is a dict for DAT format
        assert isinstance(self.data, dict), "Expected dict for DAT format"

        if "GPS" not in self.data:
            logger.warning("No GPS data available for synchronization")
            return 0.0

        gps_data: pl.DataFrame = self.data["GPS"]

        # Check if we have the required fields
        if "tick" not in gps_data.columns:
            logger.error("GPS data missing 'tick' column")
            return 0.0

        if "timestamp" not in gps_data.columns:
            logger.error("GPS data missing 'timestamp' column")
            return 0.0

        # Calculate the linear regression parameters
        timestamp_arr = gps_data.get_column("timestamp").to_numpy()
        tick_arr = gps_data.get_column("tick").to_numpy()

        diff = np.diff(timestamp_arr)
        (idx,) = np.where(diff > 0.7)

        m, c = np.polyfit(
            tick_arr[idx + 1],
            timestamp_arr[idx + 1],
            1,
        )

        residuals = c + m * tick_arr[idx + 1] - timestamp_arr[idx + 1]
        off = 0

        (idx_fast,) = np.where(residuals[off:] > np.quantile(residuals[off:], 0.95))

        time_offset = float(
            np.average(
                +(c + m * tick_arr[idx + 1][idx_fast + off])
                - timestamp_arr[idx + 1][idx_fast + off]
            )
        )
        self.data["GPS"] = self.data["GPS"].with_columns(
            pl.Series(
                "correct_timestamp",
                (m * tick_arr + c - time_offset).astype(np.float64),
            )
        )

        self.sync_params = (float(m), time_offset)
        return time_offset

    def _parse_gps_datetime(self, payload: bytes) -> datetime.datetime | None:
        """Parse GPS datetime from message payload.

        Parameters
        ----------
        payload : bytes
            Binary payload containing GPS date and time fields.

        Returns
        -------
        Optional[datetime.datetime]
            Parsed datetime object, or None if parsing fails.
        """
        if len(payload) < 8:
            return None

        raw_date, raw_time = struct.unpack("<II", payload[:8])
        if raw_date == 0:
            return None

        try:
            y = raw_date // 10000
            m = (raw_date % 10000) // 100
            d = raw_date % 100
            H = raw_time // 10000
            M = (raw_time % 10000) // 100
            S = raw_time % 100
            return datetime.datetime(y, m, d, H, M, S, tzinfo=datetime.UTC)
        except ValueError:
            return None

    def align_datfile(
        self,
        correct_timestamp: bool = True,
        sampling_freq: float = 5.0,
        polars_interpolation: bool = True,
    ) -> pl.DataFrame | None:
        """Align DAT file data using GPS synchronization.

        Parameters
        ----------
        correct_timestamp : bool, optional
            If True, correct timestamps using GPS synchronization.
        sampling_freq : float, optional
            Target sampling frequency in Hz.
        polars_interpolation : bool, optional
            If True, use Polars for interpolation.

        Returns
        -------
        Optional[pl.DataFrame]
            Aligned DataFrame, or None if alignment fails.
        """
        # Ensure self.data is a dict for DAT format
        assert isinstance(self.data, dict), "Expected dict for DAT format"

        if correct_timestamp:
            _ = self.get_tick_offset()

            if polars_interpolation:
                tmp = pl.DataFrame(
                    {
                        "tick": pl.Series([], dtype=pl.Int64),
                        "msg_type": pl.Series([], dtype=pl.Int64),
                    }
                )

                for _i, key in enumerate(self.data):
                    tmp = tmp.join(
                        self.data[key],
                        on=["tick", "msg_type"],
                        how="full",
                        coalesce=True,
                    ).sort("tick")

                numeric_cols = [
                    col
                    for col in tmp.columns
                    if tmp[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
                ]
                exclude_cols = {"tick", "msg_type"}

                for col in numeric_cols:
                    if col not in exclude_cols:
                        tmp = tmp.with_columns(
                            pl.col(col).interpolate_by("tick").alias(col)
                        )

                aligned_df = tmp

            else:
                gps_df: pl.DataFrame = self.data["GPS"]
                rtk_df: pl.DataFrame = self.data["RTK"]

                # Determine the start and end ticks based on the overlap of the two datasets
                gps_min = gps_df.get_column("tick").min()
                gps_max = gps_df.get_column("tick").max()
                rtk_min = rtk_df.get_column("tick").min()
                rtk_max = rtk_df.get_column("tick").max()

                if (
                    gps_min is None
                    or gps_max is None
                    or rtk_min is None
                    or rtk_max is None
                ):
                    logger.warning("Could not determine tick range for alignment.")
                    return None

                # Cast to float to ensure numeric comparison
                gps_min_f, gps_max_f = float(gps_min), float(gps_max)  # type: ignore
                rtk_min_f, rtk_max_f = float(rtk_min), float(rtk_max)  # type: ignore

                start_tick = max(gps_min_f, rtk_min_f)
                end_tick = min(gps_max_f, rtk_max_f)

                # Logic limits for the ticks to ensure valid range
                if start_tick >= end_tick:
                    logger.warning("No overlapping data found between GPS and RTK.")
                    return None

                # Create the aligned tick grid based on sampling frequency
                tick_freq = 4_500_000.0
                tick_step = tick_freq / sampling_freq
                target_ticks = np.arange(start_tick, end_tick, tick_step)

                logger.info(
                    f"Target ticks: {len(target_ticks)}, {start_tick:.2f} to {end_tick:.2f}"
                )

                aligned_data: dict[str, np.ndarray] = {"corrected_tick": target_ticks}

                def interpolate_columns(df: pl.DataFrame, exclude_cols: set):
                    # Ensure unique and sorted by corrected_tick for reliable interpolation
                    x = df.get_column("tick").to_numpy()

                    for col in df.columns:
                        if col in exclude_cols:
                            continue

                        # Skip if column is not numeric
                        if df[col].dtype not in [
                            pl.Float32,
                            pl.Float64,
                            pl.Int32,
                            pl.Int64,
                            pl.UInt32,
                            pl.UInt64,
                        ]:
                            continue

                        y = df.get_column(col).to_numpy()
                        try:
                            f = interp1d(
                                x,
                                y,
                                kind="linear",
                                bounds_error=False,
                                fill_value=np.nan,
                            )
                            aligned_data[col] = f(target_ticks)
                        except Exception as e:
                            logger.warning(f"Failed to interpolate column {col}: {e}")

                # Columns to exclude from generic interpolation
                common_exclude = {
                    "tick",
                    "date",
                    "time",
                    "datetime",
                    "timestamp",
                }
                gps_exclude = common_exclude.union({"GPS:date", "GPS:time"})
                rtk_exclude = common_exclude.union({"RTK:date", "RTK:time"})

                interpolate_columns(gps_df, gps_exclude)
                interpolate_columns(rtk_df, rtk_exclude)

                aligned_df = pl.DataFrame(aligned_data)

                aligned_df = aligned_df.with_columns(
                    (pl.col("correct_timestamp") * 1000)
                    .cast(pl.Int64)
                    .cast(pl.Datetime("ms"))
                    .alias("datetime_converted")
                )

                min_val = aligned_data["correct_timestamp"].min()
                max_val = aligned_data["correct_timestamp"].max()

                logger.info(f"Timestamp corrected {min_val}, {max_val}")

                # Ensure correct_timestamp is present and maybe sort columns
                self.aligned_df = aligned_df

        else:
            logger.info("Alignment with no timestamp correction applied")

            base_tick = self.data["GPS"].get_column("tick")[0]

            tmp = pl.DataFrame(
                {
                    "tick": pl.Series([], dtype=pl.Int64),
                    "msg_type": pl.Series([], dtype=pl.Int64),
                }
            )

            for _, key in enumerate(self.data):
                tmp = tmp.join(
                    self.data[key], on=["tick", "msg_type"], how="full", coalesce=True
                ).sort("tick")

            numeric_cols = [
                col
                for col in tmp.columns
                if tmp[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
            ]
            exclude_cols = {"tick", "msg_type"}

            for col in numeric_cols:
                if col not in exclude_cols:
                    tmp = tmp.with_columns(
                        pl.col(col).interpolate_by("tick").alias(col)
                    )

            aligned_df = tmp

            aligned_df = aligned_df.with_columns(
                ((pl.col("tick") - base_tick) / 4_500_000.0).alias("offset")
            )

            timestamp_vals = aligned_df.get_column("timestamp").to_numpy()
            tags = np.where(np.diff(timestamp_vals) > 0.5)[0] + 1

            offset_vals = aligned_df.get_column("offset").to_numpy()

            offset_vals = offset_vals[tags].astype(np.float64)
            timestamp_vals = timestamp_vals[tags].astype(np.float64)
            mean_offset = float(np.mean(timestamp_vals - offset_vals))

            aligned_df = aligned_df.with_columns(
                ((pl.col("offset") + mean_offset).cast(pl.Float64)).alias(
                    "correct_timestamp"
                )
            )

        return aligned_df
