"""Gremsy T7 gimbal: the MAVLink messages of ``gimbal_telemetry.bin``.

Ported (copied and adapted, not imported) from LAGER, POLOCALC/lager at
825164a: ``lager/Gimbals/Gremsy_T7/DataLoader.py`` and ``keywords.py``.

LAGER appends to the file one pickled list of
``{"timestamp", "keyword", "data"}`` per write, where ``keyword`` is the
MAVLink message (``GIMBAL_DEVICE_ATTITUDE_STATUS``, ``MOUNT_ORIENTATION``,
``RAW_IMU``, ``MOUNT_STATUS``, ``SYS_STATUS``, ...) and ``data`` its fields.
The file is read with an unpickler that builds plain data only (numbers,
strings, lists, dicts), so a file that holds anything else is refused.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import polars as pl

from ..utils.logging_config import get_logger
from ..utils.telemetry_frames import resolve_file, rows_to_frame

logger = get_logger(__name__)

FILENAME = "gimbal_telemetry.bin"

# What a pickle may build besides the built-in types that need no import
_ALLOWED = {
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "bytearray"),
    ("_codecs", "encode"),
}


class _PlainUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"not plain data: {module}.{name}")


def read_entries(file: Path) -> tuple[list[dict], bool]:
    """Read the entries of the file, and whether it ended in a corrupted or partial one.

    A crash in the middle of a write leaves a partial pickle at the end: what
    precedes it is kept.
    """
    entries: list[dict] = []
    clean = True
    with file.open("rb") as f:
        unpickler = _PlainUnpickler(io.BufferedReader(f))
        while True:
            try:
                chunk = unpickler.load()
            except EOFError:
                break
            except Exception as e:  # corrupted or partial entry: keep what was read
                logger.warning("%s: corrupted entry, stopped reading: %s", file, e)
                clean = False
                break
            entries.extend(chunk if isinstance(chunk, list) else [chunk])
    return entries, clean


class GremsyT7Telemetry:
    """Loader of the telemetry of a Gremsy T7 gimbal.

    Attributes
    ----------
    path : the file or the folder holding ``gimbal_telemetry.bin``
    data : one row per message: ``timestamp`` (the logging computer's clock,
        UNIX seconds), ``keyword`` and the union of the messages' fields
        (null where a message has none), after :meth:`load_data`
    records : the same, one table per ``keyword`` and only with its fields
    complete : False if the file ended in a corrupted or partial entry
    """

    def __init__(self, path: str | Path) -> None:
        self.path = path
        self.data: pl.DataFrame | None = None
        self.records: dict[str, pl.DataFrame] = {}
        self.complete = True

    def load_data(self) -> None:
        """Read the file into :attr:`data` and :attr:`records`.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        file = resolve_file(self.path, FILENAME)
        entries, self.complete = read_entries(file)
        rows: list[dict] = []
        for e in entries:
            try:
                rows.append(
                    {
                        "timestamp": float(e["timestamp"]),
                        "keyword": str(e["keyword"]),
                        **e["data"],
                    }
                )
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    "%s: entry without timestamp, keyword or data skipped", file
                )
        logger.info(
            "Gimbal telemetry %s: %d entries, %d rows", file, len(entries), len(rows)
        )
        self.data = rows_to_frame(rows)
        self.records = {}
        if not self.data.is_empty():
            for keyword, part in self.data.partition_by(
                "keyword", as_dict=True, maintain_order=True
            ).items():
                part = part.select(
                    [c for c in part.columns if part[c].null_count() < part.height]
                )
                self.records[keyword[0]] = part
