"""DJI Matrice 400: the flight record the remote controller's app saves (``DJIFlightRecord_*.txt``).

The app (DJI Pilot 2) writes a frame about every 0.1 s. PILS reads the log
itself: :mod:`.txtlog` (container, decryption, keys), :mod:`.records` (what
the records hold) and here the table of the flight, one row per frame. This
is not the high-rate onboard ``.DAT`` log (see :mod:`pils.drones.dji.dat`) and
not the telemetry LAGER logs (see :mod:`.telemetry`).

The table
---------
A frame starts at each OSD record (type 1) and takes the records that follow
it until the next one. The state records (OSD, home, gimbal, sticks,
batteries, estimates) hold their values until they change; the RTK solution
(1 Hz), the app's record and messages are only in the frame they came in
(nulls elsewhere). Record 48 goes to the frame of its own time: it comes a
frame early or late at times.

The time of the frames: the aircraft's own time in record 48 (seconds and
nanoseconds, ``osd_aircraft_time``). The app's time (``start_time`` of the
log's details plus the OSD's ``fly_time``) runs behind it by
``details["aircraft_minus_app_s"]`` (0.24 to 0.49 s on the first Matrice 400
flights); a frame without a record 48 gets its time interpolated between its
neighbours, at the ends of the log the app's time moved by that offset. The
file's name is in the remote controller's time zone.

The keys
--------
The records of a log of format 13 or later are AES-encrypted; the keys come
from DJI's keychain API for a DJI Open API app key (the request carries the
log's encrypted key blocks and the app key, nothing of the flight), read
from the ``api_key`` argument or the ``DJI_API_KEY`` environment variable.
The answer is cached 30 days (``$XDG_CACHE_HOME/pils/dji-keychains``), so a
log can be read again offline in that time; keep the decoded table (parquet),
not the cache.
"""

from __future__ import annotations

import dataclasses
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ....utils.logging_config import get_logger
from . import records as rec
from .txtlog import FlightLog, FlightLogError, fetch_keychains

logger = get_logger(__name__)

API_KEY_VARIABLE = "DJI_API_KEY"


class DJIFlightRecordError(RuntimeError):
    """The flight record could not be decoded (no app key, no network, not a record...)."""


def resolve_api_key(api_key: str | None = None) -> str:
    """Return the DJI Open API app key: the argument, else ``$DJI_API_KEY``.

    Raises
    ------
    DJIFlightRecordError
        If neither is set.
    """
    key = api_key or os.environ.get(API_KEY_VARIABLE)
    if not key:
        raise DJIFlightRecordError(
            f"No DJI API key: pass api_key or set {API_KEY_VARIABLE} "
            "(needed to decrypt flight records of version 13 and later)."
        )
    return key


def frames_table(log: FlightLog, records: Any) -> pl.DataFrame:
    """Return the table of a log's frames (see the module's notes), in the file's order.

    Parameters
    ----------
    log : FlightLog
        The log (its format version and ``start_time``).
    records : iterable of txtlog.Record
        Its decoded records, in order.
    """
    state: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    row: dict[str, Any] | None = None
    positions: list[dict[str, Any]] = []  # record 48: placed by its own time
    for record in records:
        parsed = rec.parse(record.type, record.data, log.version)
        if parsed is None:
            continue
        values, holds = parsed
        if record.type == rec.OSD:
            if row is not None:
                rows.append(row)
            state.update(values)
            row = dict(state)
            continue
        if holds:
            state.update(values)
        if row is None:
            continue  # before the first frame: only its state is kept
        if holds:
            row.update(values)
        elif record.type == rec.POSITION:
            positions.append({**values, "_frame": len(rows)})
        else:
            for k, v in values.items():  # several messages in a frame: joined
                row[k] = f"{row[k]}; {v}" if isinstance(v, str) and row.get(k) else v
    if row is not None:
        rows.append(row)
    if not rows:
        return pl.DataFrame()
    table = pl.from_dicts(rows, infer_schema_length=None)
    start = log.details.start_time
    if start is None:
        raise DJIFlightRecordError(
            "the log has no start time: its frames cannot be timed"
        )
    if positions:
        table = _place_positions(table, pl.from_dicts(positions), start)
    app = _app_time(start)
    offset = aircraft_minus_app(table, start)
    if offset is None:
        time = app
    else:
        # The aircraft's own time (record 48); a frame without one gets it
        # interpolated between its neighbours (the frames are 0.1 s apart),
        # at the ends the app's time moved by the median difference
        time = pl.coalesce(pl.col("osd_aircraft_time").interpolate(), app + offset)
    table = table.with_columns(time.alias("timestamp")).with_columns(
        (pl.col("timestamp") * 1e6)
        .round()
        .cast(pl.Int64)
        .cast(pl.Datetime("us", "UTC"))
        .alias("datetime")
    )
    return table.select("timestamp", "datetime", pl.exclude("timestamp", "datetime"))


def _place_positions(
    table: pl.DataFrame, positions: pl.DataFrame, start: datetime
) -> pl.DataFrame:
    """Join the records 48 to the frames by their own time.

    A record 48 usually follows the OSD record of its frame, but at times
    arrives a frame early or late (a frame then has two, its neighbour none).
    Its time (``osd_aircraft_time``) minus the median offset to the app's
    time of the frames puts it in the frame whose app time is the nearest;
    of two records for one frame the nearer is kept.
    """
    app = table["osd_fly_time"].cast(pl.Float64).to_numpy() + start.timestamp()
    frame = positions["_frame"].to_numpy().clip(0, len(app) - 1)
    aircraft = positions["osd_aircraft_time"].to_numpy()
    offset = float(np.median(aircraft - app[frame]))
    at = aircraft - offset
    # the nearest frame in app time (the frames are in time order)
    i = np.searchsorted(app, at).clip(1, len(app) - 1)
    nearest = np.where(np.abs(app[i - 1] - at) <= np.abs(app[i] - at), i - 1, i)
    placed = (
        positions.drop("_frame")
        .with_columns(
            pl.Series("_row", nearest), pl.Series("_miss", np.abs(app[nearest] - at))
        )
        .sort("_miss")
        .unique("_row", keep="first")
        .drop("_miss")
    )
    return (
        table.drop([c for c in placed.columns if c in table.columns and c != "_row"])
        .with_row_index("_row")
        .with_columns(pl.col("_row").cast(pl.Int64))
        .join(placed.with_columns(pl.col("_row").cast(pl.Int64)), on="_row", how="left")
        .sort("_row")
        .drop("_row")
    )


def _app_time(start: datetime) -> pl.Expr:
    """Return the app's time of the frames: the log's start plus the fly time."""
    return pl.col("osd_fly_time").cast(pl.Float64) + start.timestamp()


def aircraft_minus_app(table: pl.DataFrame, start: datetime | None) -> float | None:
    """Return the median of the aircraft's time (record 48) minus the app's, in s.

    None when the frames have no aircraft time (no record 48) or no start.
    """
    if start is None or "osd_aircraft_time" not in table.columns:
        return None
    diff = table.select(pl.col("osd_aircraft_time") - _app_time(start)).to_series()
    diff = diff.drop_nulls()
    return float(diff.median()) if diff.len() else None  # type: ignore[arg-type]


class M400FlightRecord:
    """Loader of a flight record (``.txt``) saved by the app of the Matrice 400's remote controller.

    Attributes
    ----------
    path : the ``.txt`` file
    data : one row per frame, after :meth:`load_data`: ``timestamp``,
        ``datetime`` and the ``osd_*``, ``home_*``, ``gimbal_*``, ``rc_*``,
        ``battery<i>_*``, ``app_*`` and ``rtk_*`` columns of :mod:`.records`
    details : the log's details: start time and duration, aircraft and serial
        numbers, app platform and version (``version`` is the log format's)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data: pl.DataFrame | None = None
        self.details: dict[str, Any] = {}

    def load_data(self, api_key: str | None = None, use_cache: bool = True) -> None:
        """Decode the log into :attr:`data` and :attr:`details`.

        Parameters
        ----------
        api_key : str, optional
            DJI Open API app key; default ``$DJI_API_KEY``. Only logs of version 13
            and later need it.
        use_cache : bool
            Use the keys cached before (and cache new ones).

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        DJIFlightRecordError
            If there is no app key, the keychain request fails or the file is not a flight record.
        """
        raw = self.path.read_bytes()
        try:
            log = FlightLog.from_bytes(raw)
            keychains = (
                fetch_keychains(log, resolve_api_key(api_key), use_cache=use_cache)
                if log.version >= 13
                else None
            )
            records = list(log.records(keychains))
            self.data = frames_table(log, records)
        except FlightLogError as e:
            raise DJIFlightRecordError(f"{self.path.name}: {e}") from e
        self.details = {
            "version": log.version,
            **{
                k: (v.astimezone(UTC).isoformat() if isinstance(v, datetime) else v)
                for k, v in dataclasses.asdict(log.details).items()
            },
            **rec.metadata(records),
            # The app's clock against the aircraft's, which times the frames
            "aircraft_minus_app_s": aircraft_minus_app(
                self.data, log.details.start_time
            ),
        }
        logger.info(
            "DJI flight record %s: version %d, %d frames",
            self.path.name,
            log.version,
            self.data.height,
        )
