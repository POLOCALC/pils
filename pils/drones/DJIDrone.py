"""DJI drone logs: DatCon CSV exports and the binary .dat flight records."""

from pathlib import Path

import numpy as np
import polars as pl

from ..utils.logging_config import get_logger
from ..utils.tools import drop_nan_and_zero_cols
from .dji_alignment import DJIAlignmentMixin
from .dji_dat import (  # noqa: F401  (re-exported: the .dat format lived here)
    _CRC_TABLE_NP,
    _GPS_FIELDS,
    _HEADER_SIZE,
    _IMU_FIELDS,
    _MOTORS,
    _NUMPY_FORMATS,
    CRC_TABLE,
    DAT_RECORDS,
    MESSAGE_DEFINITIONS,
    add_euler_angles,
    calc_checksum,
    decode_records,
    scan_records,
    unwrap_ticks,
)
from .dji_messages import DJIMessageMixin

logger = get_logger(__name__)


class DJIDrone(DJIAlignmentMixin, DJIMessageMixin):
    """DJI Drone data loader supporting CSV and DAT binary formats.

    Handles loading and processing of DJI drone flight logs from both
    CSV exports and encrypted DAT binary files.

    Attributes
    ----------
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
        # Other .dat records (IMU attitude, motors, battery, ...), one table each
        self.records: dict[str, pl.DataFrame] = {}
        # timestamp = slope * tick + intercept, from the GPS records
        self.tick_to_time: tuple[float, float] | None = None

    def load_data(
        self,
        cols: list[str] | None = None,
        use_dat: bool = True,
        remove_duplicate: bool = False,
        correct_timestamp: bool = True,
        polars_interpolation: bool = True,
        align: bool = True,
    ) -> None:
        """Load and filter drone data from a CSV or DAT file.

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
            Not used for DAT files (see :meth:`align_datfile`).
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
                timestamp_vals = self.data.get_column("timestamp_old").to_numpy()
                tags = np.where(np.diff(timestamp_vals) > 0.5)[0] + 1

                offset_vals = self.data.get_column("Clock:offsetTime").to_numpy()

                offset_vals = offset_vals[tags].astype(np.float64)
                timestamp_vals = timestamp_vals[tags].astype(np.float64)
                mean_offset = float(np.mean(timestamp_vals - offset_vals))

                self.data = self.data.with_columns(
                    ((pl.col("Clock:offsetTime") + mean_offset).cast(pl.Float64)).alias(
                        "timestamp"
                    )
                )
            elif self.source_format == "dat":
                self._split_receivers()
                self._timestamp_records()

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
                (pl.col("datetime").dt.timestamp("ms") / 1000).alias("timestamp_old")
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

        GPS and RTK go to :attr:`data` (aligned into the drone table by
        :meth:`align_datfile`); the other decoded records (IMU attitude,
        motors, battery, controller, ...; see ``DAT_RECORDS``) go to
        :attr:`records`, one table per record.

        Raises
        ------
        FileNotFoundError
            If DAT file not found.
        Exception
            If DAT file cannot be parsed.
        """
        try:
            logger.info(f"PATH: {self.path}")
            with open(self.path, "rb") as f:
                file_data = f.read()

            arr, starts, lengths = scan_records(file_data)
            types = arr[starts + 4].astype(np.int64) | (
                arr[starts + 5].astype(np.int64) << 8
            )
            raw_ticks = (
                arr[starts + 6].astype(np.int64)
                | arr[starts + 7].astype(np.int64) << 8
                | arr[starts + 8].astype(np.int64) << 16
                | arr[starts + 9].astype(np.int64) << 24
            )
            ticks = unwrap_ticks(raw_ticks)
            logger.info(f"Found {starts.size} valid records in DAT file")

            for msg_type, msg_def in MESSAGE_DEFINITIONS.items():
                sel = types == msg_type
                if not sel.any():
                    continue
                df = self._gps_table(
                    decode_records(
                        arr,
                        starts[sel],
                        lengths[sel],
                        ticks[sel],
                        msg_def["name"],
                        msg_def["fields"],
                    ),
                    msg_type,
                    msg_def["name"],
                )
                # Unwrap tick values if they decrease significantly
                df = df.slice(0, df.height - 10)
                df = self._unwrap_tick(df)
                if msg_def["name"] == "GPS":
                    lon_mean = df["GPS:longitude"].mean()
                    lon_std = df["GPS:longitude"].std()
                    # Handle potential None values from mean/std
                    if lon_mean is not None and lon_std is not None:
                        lon_mean_f = float(lon_mean)  # type: ignore[arg-type]
                        lon_std_f = float(lon_std)  # type: ignore[arg-type]
                        df = df.filter(
                            pl.col("GPS:longitude").is_between(
                                lon_mean_f - 2 * lon_std_f,
                                lon_mean_f + 2 * lon_std_f,
                            )
                        )
                self.data[msg_def["name"]] = df  # type: ignore[index]
                logger.info(f"Loaded {df.height} {msg_def['name']} messages")

            for msg_type, record in DAT_RECORDS.items():
                sel = types == msg_type
                if not sel.any():
                    continue
                df = decode_records(
                    arr,
                    starts[sel],
                    lengths[sel],
                    ticks[sel],
                    record["name"],
                    record["fields"],
                )
                if record.get("euler"):
                    df = add_euler_angles(df, record["name"])
                self.records[record["name"]] = df.sort("tick")
                logger.info(f"Loaded {df.height} {record['name']} records")

            if not self.data:
                logger.warning("No messages could be decoded")

        except Exception as e:
            logger.error(f"Failed to load DAT file: {e}")
            raise

    @staticmethod
    def _gps_table(df: pl.DataFrame, msg_type: int, name: str) -> pl.DataFrame:
        """GPS/RTK table as the per-message decoder builds it.

        Adds ``msg_type`` and, where the record has a valid UTC date and
        time, ``datetime`` (UTC) and ``timestamp`` (s).
        """
        date, time = pl.col(f"{name}:date"), pl.col(f"{name}:time")
        year, month, day = date // 10000, (date % 10000) // 100, date % 100
        valid = (
            (date != 0)
            & (time != 0)
            & (year >= 2000)
            & month.is_between(1, 12)
            & day.is_between(1, 31)
            & (time // 10000 < 24)
            & ((time % 10000) // 100 < 60)
            & (time % 100 < 60)
        )
        # Placeholder components where invalid, so the datetime cannot fail
        moment = pl.datetime(
            pl.when(valid).then(year).otherwise(2000),
            pl.when(valid).then(month).otherwise(1),
            pl.when(valid).then(day).otherwise(1),
            pl.when(valid).then(time // 10000).otherwise(0),
            pl.when(valid).then((time % 10000) // 100).otherwise(0),
            pl.when(valid).then(time % 100).otherwise(0),
            time_zone="UTC",
        )
        fields = [c for c in df.columns if c != "tick"]
        df = df.with_columns(
            pl.lit(msg_type, dtype=pl.Int64).alias("msg_type"),
            pl.when(valid).then(moment.dt.cast_time_unit("us")).alias("datetime"),
            pl.when(valid)
            .then(moment.dt.epoch("s").cast(pl.Float64))
            .alias("timestamp"),
        ).select(["msg_type", *fields, "tick", "datetime", "timestamp"])
        # Without any valid date (RTK) the record has no datetime columns
        return df.drop(
            [c for c in ("datetime", "timestamp") if df[c].null_count() == df.height]
        )

    def _split_receivers(self) -> None:
        """Split the drone table and the main GNSS receiver, each with its own rows.

        :attr:`data` becomes the RTK solution (or the main GNSS receiver if
        the drone has no RTK) and the main receiver goes to
        ``records["GPS"]``. Both are timestamped with the GPS tick-time fit;
        no value is interpolated between the two receivers (as
        :meth:`align_datfile` does).
        """
        assert isinstance(self.data, dict), "Expected dict for DAT format"
        gps = self.data.get("GPS")
        rtk = self.data.get("RTK")
        if gps is None or gps.is_empty():
            logger.warning("No GPS records: no timestamps for the drone data")
            self.data = rtk if rtk is not None else pl.DataFrame()
            return
        self.get_tick_offset()
        gps = self.data["GPS"]
        self.records["GPS"] = gps
        if rtk is None or rtk.is_empty():
            self.data = gps
            return
        slope, intercept = self.tick_to_time  # type: ignore[misc]
        timestamp = pl.col("tick") * slope + intercept
        self.data = rtk.with_columns(
            timestamp.alias("timestamp"),
            pl.from_epoch((timestamp * 1e6).round().cast(pl.Int64), time_unit="us")
            .dt.replace_time_zone("UTC")
            .alias("datetime"),
        )

    def _timestamp_records(self) -> None:
        """UTC timestamps for :attr:`records`, from the GPS tick-time fit."""
        if self.tick_to_time is None:
            return
        slope, intercept = self.tick_to_time
        for name, df in self.records.items():
            self.records[name] = df.with_columns(
                (pl.col("tick") * slope + intercept).alias("timestamp")
            )
