import polars as pl
import struct
import logging
import re
import datetime
import numpy as np
from pathlib import Path
from scipy.stats import norm
from scipy.interpolate import interp1d
from ..utils.tools import drop_nan_and_zero_cols

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

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
    def __init__(self, path):
        self.path = path
        self.data = {}  # Dictionary: {message_name: polars.DataFrame}
        self.sync_params = None  # Store (slope, intercept) from Gaussian sync
        self.source_format = None  # Track if data came from CSV or DAT

    def load_data(
        self,
        cols=[
            "Clock:offsetTime",
            "GPS:dateTimeStamp",
            "RTKdata:GpsState",
            "RTKdata:Lat_P",
            "RTKdata:Lon_P",
            "RTKdata:Hmsl_P",
        ],
        use_dat: bool = False,
    ):
        """
        Load and filter drone data from a CSV or DAT file.

        The function:
        - Loads only specified columns (for CSV).
        - Converts 'GPS:dateTimeStamp' to datetime.
        - Filters out rows with missing or zero values in critical columns.
        - Drops any columns that are fully NaN or zero using `drop_nan_and_zero_cols`.

        Parameters
        ----------
        cols : list of str, optional
            List of columns to load (for CSV files). Defaults to key RTK and timestamp fields.
        use_dat : bool, optional
            If True, try to load from DAT file instead of CSV. Default is False.

        Returns
        -------
        None
            Filtered data is stored in `self.data`.
        """
        if use_dat:
            self._load_from_dat()
            self.source_format = "DAT"
        else:
            self._load_from_csv(cols)
            self.source_format = "CSV"

    def _load_from_csv(self, cols):
        """Load drone data from CSV file."""
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
                except:
                    # If parsing with timezone fails, try without timezone
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
        self.data["CSV"] = data

    def _load_from_dat(self):
        """Load and decode drone data from DJI DAT file."""
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
                df = self._unwrap_tick(df)
                if msg_name == "GPS":
                    df = df.filter(
                        pl.col("GPS:longitude").is_between(
                            df["GPS:longitude"].mean() - 2 * df["GPS:longitude"].std(),
                            df["GPS:longitude"].mean() + 2 * df["GPS:longitude"].std(),
                        )
                    )
                self.data[msg_name] = df
                logger.info(f"Loaded {len(records)} {msg_name} messages from DAT file")

            if not records_by_type:
                logger.warning("No messages could be decoded")

        except Exception as e:
            logger.error(f"Failed to load DAT file: {e}")
            raise

    def _parse_and_decode_message(self, msg_data):
        """
        Parse and decode a single message.

        Message structure:
        - Byte 0: 0x55 (marker)
        - Byte 1: Message length
        - Bytes 2-4: Reserved
        - Bytes 5-6: Message type (little-endian uint16)
        - Byte 6: XOR key
        - Bytes 8-11: Tick (timestamp in ticks, little-endian uint32)
        - Bytes 12+: Encrypted payload
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
            expected_total_length = msg_length
            if len(msg_data) < msg_length:
                return []

            msg_def = MESSAGE_DEFINITIONS[msg_type]
            payload_size = msg_def["payload_size"]

            # Extract and decrypt payload (starts at byte 10)
            if len(msg_data) < 10 + payload_size:
                return []

            encrypted_payload = msg_data[10 : 10 + payload_size]
            decrypted = bytes(b ^ key for b in encrypted_payload)

            # Decode the message
            decoded = self._decode_message_data(decrypted, msg_type, tick_val, msg_def)

            return [decoded] if decoded else []

        except Exception as e:
            logger.debug(f"Failed to parse message: {e}")
            return []

    def _decode_message_data(self, decrypted_payload, msg_type, tick_val, msg_def):
        """
        Unified message decoder using message definition template.

        Decodes message by unpacking each field from its byte offset
        using the specified format code.

        Parameters
        ----------
        decrypted_payload : bytes
            The decrypted message payload
        msg_type : int
            The message type identifier
        tick_val : int
            The timestamp value in ticks
        msg_def : dict
            Message definition containing field mappings with offsets and format codes

        Returns
        -------
        dict or None
            Decoded message as dictionary, or None if decoding fails
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
                result["datetime"] = formatted_dt

                fmt = "%Y-%m-%d %H:%M:%S"
                # Parse as UTC to avoid local timezone shifts
                dt = datetime.datetime.strptime(formatted_dt, fmt).replace(
                    tzinfo=datetime.timezone.utc
                )
                result["timestamp"] = dt.timestamp()

            return result

        except Exception as e:
            print(f"Failed to decode message: {e}")
            logger.debug(f"Failed to decode message: {e}")
            return None

    @staticmethod
    def _format_date_time(date, time):
        """Convert date and time fields into a human-readable datetime string."""
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
    def _unwrap_tick(df, wrap_threshold=1e8):
        """
        Unwrap tick values that wrap around due to uint32 overflow.

        Only unwraps when tick wraps from high value to near zero (< wrap_threshold).
        Ignores other negative jumps which are likely data corruption.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with 'tick' column
        wrap_threshold : float, optional
            Maximum value after wraparound to be considered valid. Default: 1e8 (100 million)
            Real wraparounds go from ~4.3B to near 0, not to other large values.

        Returns
        -------
        pl.DataFrame
            DataFrame with unwrapped tick values
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
                offset += 2**32  # Add uint32 max value
                logger.info(
                    f"Tick unwrap at index {i}: {tick_values[i-1]:,} -> {tick_values[i]:,} (adding offset 2^32, total offset: {offset:,})"
                )

            unwrapped.append(tick_values[i] + offset)

        # Replace tick column with unwrapped values
        return df.with_columns(pl.Series("tick", unwrapped))

    def get_tick_offset(self):

        if "GPS" not in self.data:
            logger.warning("No GPS data available for synchronization")
            return 0.0

        gps_data = self.data["GPS"]

        # Check if we have the required fields
        if "tick" not in gps_data.columns:
            logger.error("GPS data missing 'tick' column")
            return 0.0

        if "timestamp" not in gps_data.columns:
            logger.error("GPS data missing 'timestamp' column")
            return 0.0

        # Calculate the linear regression parameters
        diff = np.diff(gps_data["timestamp"].to_numpy())
        (idx,) = np.where(diff > 0.7)

        m, c = np.polyfit(
            gps_data["timestamp"].to_numpy()[idx + 1],
            gps_data["tick"].to_numpy()[idx + 1],
            1,
        )

        residuals = (
            c
            + m * gps_data["timestamp"].to_numpy()[idx + 1]
            - gps_data["tick"].to_numpy()[idx + 1]
        )

        (idx_fast,) = np.where(residuals > np.quantile(residuals, 0.99))

        self.data["GPS"] = self.data["GPS"].with_columns(
            pl.Series(
                "corrected_tick",
                self.data["GPS"]["tick"]
                - np.average(self.data["GPS"]["tick"][idx + 1][idx_fast]),
            )
        )

        self.data["GPS"] = self.data["GPS"].with_columns(
            pl.Series(
                "correct_timestamp",
                (
                    self.data["GPS"]["timestamp"]
                    - (
                        self.data["GPS"]["corrected_tick"]
                        - self.data["GPS"]["corrected_tick"][0]
                    )
                    / m
                ).cast(pl.Float64),
            )
        )

        tick_offset = np.average(self.data["GPS"]["tick"][idx + 1][idx_fast])
        self.sync_params = (m, tick_offset)
        return tick_offset

    def compute_offset_time(self, message_type="GPS"):
        """
        Compute offset time for a message type using the formula:
        offsetTime = (tick - tickOffset) / 600.0

        where tickOffset is the intercept from the Gaussian synchronization.

        Parameters
        ----------
        message_type : str, optional
            Message type to compute offset time for (default: "GPS")

        Returns
        -------
        pl.Series or None
            Series with offset time values, or None if sync params not available
        """
        if self.sync_params is None:
            logger.warning(
                "No synchronization parameters available. Run get_tick_offset first."
            )
            return None

        if message_type not in self.data:
            logger.warning(f"Message type '{message_type}' not found in data")
            return None

        msg_data = self.data[message_type]
        if "tick" not in msg_data.columns:
            logger.error(f"{message_type} data missing 'tick' column")
            return None

        slope, tick_offset = self.sync_params

        # Compute offset time: (tick - tickOffset) / 600.0
        offset_time = msg_data.with_columns(
            [((pl.col("tick") - tick_offset) / 4500000.0).alias("offsetTime")]
        )["offsetTime"]

        logger.info(
            f"Computed offset time for {message_type}: {len(offset_time)} values"
        )
        logger.info(f"  Tick offset used: {tick_offset:.4f}")
        logger.info(
            f"  Offset time range: [{offset_time.min():.2f}, {offset_time.max():.2f}] s"
        )

        return offset_time

    def _parse_gps_datetime(self, payload):
        """Parse GPS datetime from message payload (deprecated - using decoded data now)."""
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
            return datetime.datetime(y, m, d, H, M, S, tzinfo=datetime.timezone.utc)
        except ValueError:
            return None

    def align_datfile(self, sampling_freq=5.0):

        tick_offset = self.get_tick_offset()

        self.data["RTK"] = self.data["RTK"].with_columns(
            (pl.col("tick") - tick_offset).alias("corrected_tick")
        )

        gps_df = self.data["GPS"]
        rtk_df = self.data["RTK"]

        # Determine the start and end ticks based on the overlap of the two datasets
        start_tick = max(gps_df["corrected_tick"].min(), rtk_df["corrected_tick"].min())
        end_tick = min(gps_df["corrected_tick"].max(), rtk_df["corrected_tick"].max())

        # Logic limits for the ticks to ensure valid range
        if start_tick >= end_tick:
            logger.warning("No overlapping data found between GPS and RTK.")
            return None

        # Create the aligned tick grid based on sampling frequency
        tick_freq = 4_500_000.0
        tick_step = tick_freq / sampling_freq
        target_ticks = np.arange(start_tick, end_tick, tick_step)

        aligned_data = {"corrected_tick": target_ticks}

        def interpolate_columns(df, exclude_cols):
            # Ensure unique and sorted by corrected_tick for reliable interpolation
            df_unique = df.unique(subset=["corrected_tick"]).sort("corrected_tick")
            x = df_unique["corrected_tick"].to_numpy()

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

                y = df_unique[col].to_numpy()
                try:
                    f = interp1d(
                        x, y, kind="linear", bounds_error=False, fill_value=np.nan
                    )
                    aligned_data[col] = f(target_ticks)
                except Exception as e:
                    logger.warning(f"Failed to interpolate column {col}: {e}")

        # Columns to exclude from generic interpolation
        common_exclude = {
            "tick",
            "corrected_tick",
            "date",
            "time",
            "datetime",
            "timestamp",
            "correct_timestamp",
        }
        gps_exclude = common_exclude.union({"GPS:date", "GPS:time"})
        rtk_exclude = common_exclude.union({"RTK:date", "RTK:time"})

        # Calculate a smooth, monotonic correct_timestamp for the target ticks
        # Anchor to the first GPS record
        base_time = gps_df["correct_timestamp"][0]
        base_tick = gps_df["corrected_tick"][0]

        aligned_data["correct_timestamp"] = (
            base_time + (target_ticks - base_tick) / tick_freq
        )

        interpolate_columns(gps_df, gps_exclude)
        interpolate_columns(rtk_df, rtk_exclude)

        self.aligned_df = pl.DataFrame(aligned_data)

        self.aligned_df = self.aligned_df.with_columns(
            (pl.col("correct_timestamp") * 1000)
            .cast(pl.Int64)
            .cast(pl.Datetime("ms"))
            .alias("datetime_converted")
        )

        # Ensure correct_timestamp is present and maybe sort columns
        return self.aligned_df
