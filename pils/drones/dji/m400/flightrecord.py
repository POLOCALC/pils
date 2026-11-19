"""DJI Matrice 400: the flight record the remote controller's app saves (``DJIFlightRecord_*.txt``).

The app writes at about 10 Hz. The records of a log of version 13 or later are
AES-encrypted; the keys are fetched from DJI's keychain API with a DJI Open API
app key (the request carries the log's encrypted key blocks and the app key,
nothing of the flight), and decoded with ``pydjirecord``. This is not the
high-rate onboard ``.DAT`` log (see :mod:`pils.drones.dji.dat`) and not the
telemetry LAGER logs (see :mod:`.telemetry`).

The app key is read from the ``api_key`` argument or the ``DJI_API_KEY``
environment variable. ``pydjirecord`` caches the keychains it fetched for 30
days (``$XDG_CACHE_HOME/pydjirecord``), so a log can be decoded again
offline within that time; keep the decoded table (parquet), not the cache.
"""

from __future__ import annotations

import dataclasses
import enum
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydjirecord import DJILog
from pydjirecord.error import DJILogError

from ....utils.logging_config import get_logger

logger = get_logger(__name__)

API_KEY_VARIABLE = "DJI_API_KEY"
MAX_VERSION = 14  # the newest log format pydjirecord reads

# The sub-records of a frame that vary in time, in the order of the table's columns
_SECTIONS = ("osd", "gimbal", "camera", "rc", "battery", "home", "app")
# Header fields not kept: the street address the app wrote for the take-off
_NOT_KEPT = {"sub_street", "street", "city", "area"}


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


def _value(v: Any) -> Any:
    """Return a column value: an enum by its name, a datetime as it is, anything else unchanged."""
    if isinstance(v, enum.Enum):
        return v.name
    return v


def frame_row(frame: Any) -> dict[str, Any]:
    """Flatten a ``pydjirecord`` frame into one row: ``osd_latitude``, ``gimbal_pitch``, ...

    ``timestamp`` (UNIX seconds, UTC) and ``datetime`` are the frame's time; the
    battery's cell voltages are ``battery_cell_voltage_1``, ``..._2``, ...
    """
    when: datetime = frame.custom.date_time
    when = when.replace(tzinfo=UTC) if when.tzinfo is None else when.astimezone(UTC)
    row: dict[str, Any] = {"timestamp": when.timestamp(), "datetime": when}
    for name in _SECTIONS:
        section = getattr(frame, name)
        for f in dataclasses.fields(section):
            v = getattr(section, f.name)
            if isinstance(v, list):
                for i, item in enumerate(v, start=1):
                    row[f"{name}_{f.name[:-1]}_{i}"] = item
            else:
                row[f"{name}_{f.name}"] = _value(v)
    return row


def frames_to_table(frames: list[Any]) -> pl.DataFrame:
    """Make a table from ``pydjirecord`` frames: one row per frame, sorted by time (stable)."""
    if not frames:
        return pl.DataFrame()
    table = pl.from_dicts(
        [frame_row(f) for f in frames], infer_schema_length=None, strict=False
    )
    return table.sort("timestamp", maintain_order=True)


class M400FlightRecord:
    """Loader of a flight record (``.txt``) saved by the app of the Matrice 400's remote controller.

    Attributes
    ----------
    path : the ``.txt`` file
    data : one row per frame, after :meth:`load_data`: ``timestamp``,
        ``datetime`` and the ``osd_*`` (position, attitude, speeds, flight
        state), ``gimbal_*``, ``camera_*``, ``rc_*``, ``battery_*``,
        ``home_*`` and ``app_*`` columns of ``pydjirecord``'s frame, in its
        units (degrees, metres, m/s); enumerations by name
    details : the log's header: aircraft name and serial numbers, app platform and
        version, start time, totals (``version`` is the log format's)
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
            Use the keychains ``pydjirecord`` cached before (and cache new ones).

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        DJIFlightRecordError
            If there is no app key, the keychain request fails or the file is not a flight record.
        """
        raw = self.path.read_bytes()
        try:
            log = DJILog.from_bytes(raw)
            if not 1 <= log.version <= MAX_VERSION:
                raise DJIFlightRecordError(
                    f"{self.path.name}: not a DJI flight record (format version {log.version})"
                )
            keychains = None
            if log.version >= 13:
                keychains = log.fetch_keychains(
                    resolve_api_key(api_key), cache=use_cache
                )
            frames = log.frames(keychains)
        except DJILogError as e:
            raise DJIFlightRecordError(
                f"{self.path.name}: {type(e).__name__}: {e}"
            ) from e
        except (
            ValueError,
            IndexError,
            EOFError,
        ) as e:  # not a flight record, or cut short
            raise DJIFlightRecordError(
                f"{self.path.name}: not a readable DJI flight record: {e}"
            ) from e

        self.details = {
            "version": log.version,
            **{
                k: _value(v)
                for k, v in dataclasses.asdict(log.details).items()
                if k not in _NOT_KEPT
            },
        }
        if frames:
            self.details.update(
                {
                    f"recover_{k}": _value(v)
                    for k, v in dataclasses.asdict(frames[0].recover).items()
                }
            )
        self.data = frames_to_table(frames)
        logger.info(
            "DJI flight record %s: version %d, %d frames",
            self.path.name,
            log.version,
            len(frames),
        )
