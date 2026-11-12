"""Payload GPS: the u-blox receiver recorded as UBX.

The receiver is a ZED-F9P (ZED-F9R in 2023), recorded in ``*_GPS.bin`` or
``*_GPS.ubx`` in ``aux/sensors/``. The NAV messages are merged on one time
grid; positions are in degrees and metres (``lat_deg``, ``lon_deg``,
``h_ell_m``, ``h_msl_m``) and ``timestamp`` in UNIX seconds.

Notes
-----
- The messages are read one by one (:func:`read_ubx`); corrupted bytes
  (some April 2025 files) are skipped with a warning.
- The GPS week comes from the receiver (NAV-TIMEGPS week, or the UTC date
  of NAV-PVT, when valid); without them, from the log's ``Sensor ZED-F9...
  started`` line or the file name (``YYYYMMDD_hhmmss_GPS.bin``). The time
  of week rolling over on Sunday is handled.
- A folder without a GPS file raises ``FileNotFoundError``; a file with one
  epoch or none gives an empty table, and ``Flight.sync`` then uses the
  drone GPS as the reference.
- The 2023 ``*_GPS.pck`` files hold no data and are not read.
- The raw file is also the PPK rover (``pils.analyze.ppk``).
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from pyubx2 import UBX_PROTOCOL, UBXReader

from ..utils.gnss_time import (
    GPS_EPOCH,
    GPS_EPOCH_UNIX,
    SECONDS_PER_WEEK,
    leap_seconds,
)
from ..utils.logging_config import get_logger
from ..utils.tools import get_logpath_from_datapath, read_log_time

logger = get_logger(__name__)

GPS_SUFFIXES = ("gps.bin", "gps.ubx")


def read_ubx(path: Path, protocol: int = UBX_PROTOCOL):
    """Yield the parsed UBX messages of a file, skipping corrupted bytes.

    pyubx2 raises on some corrupted data (e.g. a ``$`` followed by binary
    bytes, read as an NMEA sentence); the reader then continues after it.
    """
    errors = 0
    with open(path, "rb") as stream:
        reader = UBXReader(stream, protfilter=protocol, quitonerror=0)
        while True:
            position = stream.tell()
            try:
                raw, parsed = reader.read()
            except Exception:
                # Broad: pyubx2 and pynmeagps raise many types on corrupted
                # bytes; they are counted and logged below
                if stream.tell() == position:
                    break  # not moving on: stop rather than loop
                errors += 1
                continue
            if raw is None:
                break
            if parsed is not None:
                yield parsed
    if errors:
        logger.warning(f"{path.name}: {errors} corrupted messages skipped")


def _receiver_gps_week(nav_dataframes: dict[str, pl.DataFrame]) -> int | None:
    """GPS week of the first epoch, from the receiver's own time, or None.

    From NAV-TIMEGPS (``week`` with ``weekValid``) or else NAV-PVT (UTC date
    and time with ``validDate`` and ``validTime``), at their first valid
    epoch. The tables have ``iTOW`` counted on from the first epoch (rolled
    over weeks added) and the original in ``_iTOW_raw``.
    """
    week_ms = SECONDS_PER_WEEK * 1000
    timegps = nav_dataframes.get("NAV-TIMEGPS")
    if timegps is not None and "timegps_week" in timegps.columns:
        valid = pl.col("timegps_week") > 0
        if "timegps_weekValid" in timegps.columns:
            valid = valid & (pl.col("timegps_weekValid") == 1)
        rows = timegps.filter(valid)
        if rows.height:
            row = rows.row(0, named=True)
            return int(row["timegps_week"]) - (
                (row["iTOW"] - row["_iTOW_raw"]) // week_ms
            )
    pvt = nav_dataframes.get("NAV-PVT")
    fields = ["year", "month", "day", "hour", "min", "second"]
    if pvt is None or not all(f"pvt_{f}" in pvt.columns for f in fields):
        return None
    valid = pl.col("pvt_year") >= 1980
    for flag in ("validDate", "validTime"):
        if f"pvt_{flag}" in pvt.columns:
            valid = valid & (pl.col(f"pvt_{flag}") == 1)
    rows = pvt.filter(valid)
    if not rows.height:
        return None
    row = rows.row(0, named=True)
    try:
        utc = datetime(*(int(row[f"pvt_{f}"]) for f in fields))
    except ValueError:
        return None
    gps_ms = ((utc - GPS_EPOCH).total_seconds() + leap_seconds(utc)) * 1000 - row[
        "iTOW"
    ]
    # The week of the first epoch: GPS time of this epoch minus its time of
    # week counted from there (to the nearest week: the UTC time has whole
    # seconds, iTOW milliseconds)
    return int(round(gps_ms / week_ms))


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

    def __init__(self, path: Path, logpath: Path | None = None) -> None:
        """
        Initialize GPS sensor.

        Parameters
        ----------
        path : Path
            Directory containing GPS binary file.
        logpath : Optional[Path], default=None
            Optional path to log file. If None, will be inferred.
        """
        path = Path(path)
        files = (
            [path]
            if path.is_file()
            else [
                f
                for f in sorted(path.glob("*"))
                if f.name.lower().endswith(GPS_SUFFIXES)
            ]
        )
        if not files:
            raise FileNotFoundError(
                f"No payload GPS file (*_GPS.bin, *_GPS.ubx) in {path}"
            )
        self.data_path = files[-1]
        self.logpath: Path | None = logpath
        if logpath is None:
            try:
                self.logpath = get_logpath_from_datapath(self.data_path)
            except FileNotFoundError:
                pass  # the date then comes from the file name

        self.tstart: datetime | None = None
        self.data: pl.DataFrame | None = None

    def load_data(self, freq_interpolation: float | None = None) -> None:
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

        for parsed_data in read_ubx(self.data_path):  # corrupted bytes skipped
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
                    except AttributeError:  # listed by dir() but not readable
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

        week = 7 * 24 * 60 * 60 * 1000
        for msg_type, df in nav_dataframes.items():
            if "iTOW" in df.columns:
                # The time of week restarts on Sunday 00:00 GPS time: counted
                # from the week of the first epoch
                rollover = (pl.col("iTOW").diff() < -week // 2).cum_sum().fill_null(0)
                nav_dataframes[msg_type] = df.with_columns(
                    pl.col("iTOW").alias("_iTOW_raw"),
                    pl.col("iTOW") + rollover * week,
                )

        # The GPS week of the first epoch: from the receiver (NAV-TIMEGPS
        # week, or the NAV-PVT UTC date) if it has it; else from the date of
        # the log ("Sensor ZED-F9P", in 2023 "ZED-F9R") or of the file name
        # (YYYYMMDD_hhmmss_GPS.bin), which can be a week off near Sunday
        # 00:00 GPS time or with local-time names
        gps_week = _receiver_gps_week(nav_dataframes)
        time_start, date = None, None
        if self.logpath is not None:
            time_start, date = read_log_time(
                keyphrase="Sensor ZED-F9", logfile=self.logpath
            )
        if date is None:
            try:
                time_start = datetime.strptime(
                    self.data_path.name[:15], "%Y%m%d_%H%M%S"
                )
                date = time_start.date()
            except ValueError:
                if gps_week is None:
                    logger.warning(
                        f"No date for {self.data_path}: no log, no date in the "
                        "name, none from the receiver"
                    )
                    self.data = pl.DataFrame()
                    return
        if gps_week is None:
            assert date is not None
            gps_week = (date - GPS_EPOCH.date()).days // 7
        elif date is not None and gps_week != (date - GPS_EPOCH.date()).days // 7:
            logger.info(
                f"GPS week {gps_week} from the receiver, "
                f"{(date - GPS_EPOCH.date()).days // 7} from the date {date}"
            )

        # Unix time = GPS epoch + week + iTOW - leap seconds (GPS - UTC)
        gps_to_unix_offset_ms = int(GPS_EPOCH_UNIX) * 1000
        epoch_utc = GPS_EPOCH + timedelta(weeks=gps_week)
        leap_seconds_ms = leap_seconds(epoch_utc) * 1000
        week_ms = gps_week * week

        for msg_type, df in nav_dataframes.items():
            if "iTOW" in df.columns:
                df = df.drop("_iTOW_raw")
                first_itow = df["iTOW"][0]
                start = (
                    time_start
                    if time_start is not None
                    else datetime(1970, 1, 1)
                    + timedelta(
                        milliseconds=week_ms
                        + gps_to_unix_offset_ms
                        - leap_seconds_ms
                        + first_itow
                    )
                )
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
                            pl.lit(start)
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
        gps_data = gps_data.with_columns(
            (pl.col("unix_time_ms") / 1000.0).alias("timestamp")
        )
        gps_data = gps_data.with_columns(pl.col("posllh_height") / 1000.0)

        self.data = gps_data

    def _merge_nav_dataframes(
        self, nav_dataframes: dict, freq: float | None = None
    ) -> pl.DataFrame | None:
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
            steps = [
                np.median(np.diff(np.sort(df["unix_time_ms"].to_numpy())))
                for df in nav_dataframes.values()
                if "unix_time_ms" in df.columns and len(df) > 1
            ]
            steps = [step for step in steps if step > 0]
            if not steps:
                return None  # one epoch: no track
            freq_ms = int(min(steps))
        else:
            freq_ms = int(1000 / freq)

        # Create common time grid
        time_grid = pl.DataFrame(
            {"unix_time_ms": list(range(int(min_time), int(max_time) + 1, freq_ms))}
        )

        # Merge each dataframe onto the time grid with interpolation
        merged_df = time_grid
        for _msg_type, df in nav_dataframes.items():
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
            if (
                "datetime_relative" in df.columns
                and "datetime_relative" not in merged_df.columns
            ):
                cols_to_join.append("datetime_relative")

            # Join with time grid using asof join (nearest previous value)
            df_to_join = df.select(cols_to_join)
            merged_df = merged_df.join_asof(
                df_to_join, on="unix_time_ms", strategy="nearest"
            )

        return merged_df
