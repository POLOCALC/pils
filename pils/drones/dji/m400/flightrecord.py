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
batteries) hold their values until they change; the RTK solution (1 Hz), the
record 48 and the app's messages are only in the frame they came in (nulls
elsewhere).

The time of the frames: ``start_time`` of the log's details (UTC) plus the
OSD's ``fly_time`` (the aircraft's counter, 0.1 s). The app's own times (the
``custom`` record) are not usable in these logs, and the file's name is in the
remote controller's time zone. The record 48 has the UNIX second of each frame
(``osd_unix_time``): it agrees with this time to the second.

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
    t0 = start.timestamp()
    table = table.with_columns(
        (pl.col("osd_fly_time").cast(pl.Float64) + t0).alias("timestamp")
    ).with_columns(
        (pl.col("timestamp") * 1e6)
        .round()
        .cast(pl.Int64)
        .cast(pl.Datetime("us", "UTC"))
        .alias("datetime")
    )
    return table.select("timestamp", "datetime", pl.exclude("timestamp", "datetime"))


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
        }
        logger.info(
            "DJI flight record %s: version %d, %d frames",
            self.path.name,
            log.version,
            self.data.height,
        )
