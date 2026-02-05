from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from pyubx2 import UBX_PROTOCOL, UBXReader

from ..utils.tools import get_logpath_from_datapath, read_log_time


class GPS:
    """
    GPS sensor for reading UBX binary data.

    Attributes
    ----------
    data_path : Path
        Path to GPS binary file (*.gps.bin).
    logpath : Path
        Path to log file for timestamp extraction.
    tstart : Optional[datetime]
        Start timestamp from log file (set after load_data).
    data : Optional[pl.DataFrame]
        Polars DataFrame with GPS data (None until load_data is called).
    """

    def __init__(self, path: Path, logpath: Optional[Path] = None) -> None:
        """
        Initialize GPS sensor.

        Parameters
        ----------
        path : Path
            Directory containing GPS binary file.
        logpath : Optional[Path], default=None
            Optional path to log file. If None, will be inferred.
        """

        files = list(path.glob("*"))

        for f in files:
            if f.name.lower().endswith("gps.bin"):
                self.data_path = f
        if logpath is not None:
            self.logpath = logpath
        else:
            self.logpath = get_logpath_from_datapath(self.data_path)

        self.tstart: Optional[datetime] = None
        self.data: Optional[pl.DataFrame] = None

    def load_data(self, freq_interpolation: Optional[float] = None) -> None:
        """
        Load GPS data from UBX binary file.

        Reads GPS data in UBX protocol format and parses NAV messages.
        Merges different NAV message types (POSLLH, VELNED, STATUS, etc.)
        onto a common time grid with optional interpolation.

        Parameters
        ----------
        freq_interpolation : Optional[float], default=None
            Optional frequency for interpolation in Hz.
            If None, uses mean time difference from data.
            If provided, resamples data to this frequency.

        Notes
        -----
        Sets self.data: Polars DataFrame with:
        - unix_time_ms: Unix timestamp in milliseconds
        - datetime: UTC datetime
        - timestamp: Unix timestamp in seconds
        - NAV message columns (prefixed by message type)

        Requires log file with "Sensor ZED-F9P started" entry for date extraction.
        """

        # Dictionary to collect records from different NAV message types
        nav_records = {}

        with open(self.data_path, "rb") as stream:
            ubr = UBXReader(stream, protfilter=UBX_PROTOCOL, quitonerror=False)
            for raw_data, parsed_data in ubr:
                if parsed_data is None:
                    continue

                # Only process NAV messages
                if not parsed_data.identity.startswith("NAV-"):
                    continue

                msg_type = parsed_data.identity

                # Initialize list for this message type if not exists
                if msg_type not in nav_records:
                    nav_records[msg_type] = []

                # Extract all attributes from the parsed message
                record = {}
                for attr in dir(parsed_data):
                    if not attr.startswith("_") and attr not in [
                        "identity",
                        "payload",
                        "msg_cls",
                        "msg_id",
                        "length",
                    ]:
                        try:
                            value = getattr(parsed_data, attr)
                            # Only include serializable values (not methods)
                            if not callable(value):
                                record[attr] = value
                        except Exception:
                            pass

                nav_records[msg_type].append(record)

        # Convert each message type to a DataFrame with prefixed column names
        nav_dataframes = {}
        for msg_type, records in nav_records.items():
            if records:
                df = pl.DataFrame(records)
                # Prefix columns with message type (except iTOW which is used for joining)
                prefix = msg_type.replace("NAV-", "").lower()
                renamed_cols = {}
                for col in df.columns:
                    if col != "iTOW":
                        renamed_cols[col] = f"{prefix}_{col}"
                if renamed_cols:
                    df = df.rename(renamed_cols)
                nav_dataframes[msg_type] = df

        # Get the date from log file to compute GPS week
        time_start, date = read_log_time(keyphrase="Sensor ZED-F9P started", logfile=self.logpath)

        if date is None:
            self.data = pl.DataFrame()
            return

        # GPS epoch and leap seconds offset (GPS time is ahead of UTC by 18 seconds as of 2017)
        gps_epoch = datetime(1980, 1, 6).date()
        unix_epoch = datetime(1970, 1, 1).date()
        gps_leap_seconds = 18  # GPS-UTC offset in seconds

        duration = date - gps_epoch
        gps_week = duration.days // 7

        # Compute Unix timestamp in ms for each dataframe
        # GPS time = GPS epoch + (week * 7 days) + iTOW (in ms)
        # Unix time = GPS time - (GPS epoch - Unix epoch) - leap_seconds
        gps_to_unix_offset_ms = (gps_epoch - unix_epoch).days * 24 * 60 * 60 * 1000
        leap_seconds_ms = gps_leap_seconds * 1000
        week_ms = gps_week * 7 * 24 * 60 * 60 * 1000

        for msg_type, df in nav_dataframes.items():
            if "iTOW" in df.columns:
                first_itow = df["iTOW"][0]
                # Create Unix timestamp in ms
                # unix_time_ms = gps_week_ms + iTOW + gps_epoch_offset - leap_seconds
                df = df.with_columns(
                    [
                        (
                            pl.col("iTOW")
                            + pl.lit(week_ms + gps_to_unix_offset_ms - leap_seconds_ms)
                        ).alias("unix_time_ms"),
                        # Relative datetime: time_start + (iTOW - iTOW[0])
                        (
                            pl.lit(time_start)
                            + pl.duration(milliseconds=pl.col("iTOW") - first_itow)
                        ).alias("datetime_relative"),
                    ]
                )
                nav_dataframes[msg_type] = df

        # Combine dataframes based on a common time grid with interpolation
        gps_data = self._merge_nav_dataframes(nav_dataframes, freq=freq_interpolation)

        if gps_data is None or len(gps_data) == 0:
            gps_data = pl.DataFrame()
            self.data = gps_data
            return

        # Convert unix_time_ms to datetime (UTC)
        gps_data = gps_data.with_columns(
            (pl.from_epoch(pl.col("unix_time_ms"), time_unit="ms")).alias("datetime")
        )
        gps_data = gps_data.with_columns((pl.col("unix_time_ms") / 1000.0).alias("timestamp"))
        gps_data = gps_data.with_columns((pl.col("posllh_height") / 1000.0))

        self.data = gps_data

    def _merge_nav_dataframes(
        self, nav_dataframes: dict, freq: Optional[float] = None
    ) -> Optional[pl.DataFrame]:
        """
        Merge NAV dataframes onto a common time grid with interpolation.

        Parameters
        ----------
        nav_dataframes : dict
            Dictionary of NAV message type -> polars DataFrame.
            Each DataFrame must have unix_time_ms column.
        freq : Optional[float], default=None
            Time grid frequency in Hz. If None, uses mean time difference.

        Returns
        -------
        pl.DataFrame or None
            Merged DataFrame with interpolated values on common time grid,
            or None if no valid dataframes found.
        """
        if not nav_dataframes:
            return None

        # Find global time range across all dataframes
        min_time = None
        max_time = None
        first_valid_df = None
        for df in nav_dataframes.values():
            if "unix_time_ms" in df.columns and len(df) > 0:
                if first_valid_df is None:
                    first_valid_df = df
                df_min = df["unix_time_ms"].min()
                df_max = df["unix_time_ms"].max()
                if min_time is None or df_min < min_time:
                    min_time = df_min
                if max_time is None or df_max > max_time:
                    max_time = df_max

        if min_time is None or max_time is None or first_valid_df is None:
            return None

        if freq is None:
            freq_ms = int(np.mean(np.diff(first_valid_df["unix_time_ms"].to_numpy())))
        else:
            freq_ms = int(1000 / freq)

        # Create common time grid
        time_grid = pl.DataFrame(
            {"unix_time_ms": list(range(int(min_time), int(max_time) + 1, freq_ms))}
        )

        # Merge each dataframe onto the time grid with interpolation
        merged_df = time_grid
        for msg_type, df in nav_dataframes.items():
            if "unix_time_ms" not in df.columns or len(df) == 0:
                continue

            # Sort by time
            df = df.sort("unix_time_ms")

            # Get numeric columns for interpolation (exclude time columns)
            numeric_cols = [
                col
                for col in df.columns
                if col not in ["iTOW", "unix_time_ms", "datetime_relative"]
                and df[col].dtype
                in [
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
                ]
            ]

            # Include datetime_relative if present (take from first dataframe that has it)
            cols_to_join = ["unix_time_ms"] + numeric_cols
            if "datetime_relative" in df.columns and "datetime_relative" not in merged_df.columns:
                cols_to_join.append("datetime_relative")

            # Join with time grid using asof join (nearest previous value)
            df_to_join = df.select(cols_to_join)
            merged_df = merged_df.join_asof(df_to_join, on="unix_time_ms", strategy="nearest")

        return merged_df
