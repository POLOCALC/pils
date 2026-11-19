"""DJI Matrice 600 telemetry as LAGER logs it (``uav_telemetry.bin``)."""

from __future__ import annotations

import struct
from pathlib import Path

import polars as pl

from ....utils.logging_config import get_logger
from ....utils.telemetry_frames import forward_fill, resolve_file, rows_to_frame
from . import protocol

logger = get_logger(__name__)

FILENAME = "uav_telemetry.bin"


class M600Telemetry:
    """Loader of the broadcast telemetry of a Matrice 600 (DJI OSDK over serial).

    Ported from LAGER's ``DJI_M600.DataLoader`` (see :mod:`.protocol`).
    ``data`` is one row per broadcast frame: ``timestamp`` (the logging
    computer's clock, UNIX seconds), ``t_ticks`` (the flight controller's
    400 Hz tick counter) and the fields of :func:`protocol.decode_broadcast`.
    The battery-only frames (flag 0x3000, at a lower rate) fill only their
    own columns, unless ``fill`` is set.

    Attributes
    ----------
    path : the file or the folder holding ``uav_telemetry.bin``
    data : the table, after :meth:`load_data`
    stats : frames read, and the ones left out (``bad_crc``, ``ack``,
        ``other_command``, ``short``, ``truncated``)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = path
        self.data: pl.DataFrame | None = None
        self.stats: dict[str, int] = {}

    def load_data(self, fill: bool = False) -> None:
        """Read the file into :attr:`data`.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        file = resolve_file(self.path, FILENAME)
        stats = {
            "frames": 0,
            "bad_crc": 0,
            "ack": 0,
            "other_command": 0,
            "short": 0,
            "truncated": 0,
        }
        rows: list[dict] = []
        with file.open("rb") as f:
            while True:
                head = f.read(12)  # float64 time + uint32 frame length
                if len(head) < 12:
                    if head:
                        stats["truncated"] += 1
                    break
                timestamp, length = struct.unpack("<dI", head)
                # A zero-length or impossibly short frame (the logger was killed
                # mid-write) cannot be resynchronised: stop reading
                if length < protocol.PKG_MIN:
                    stats["short"] += 1
                    break
                frame = f.read(length)
                if len(frame) < length:
                    stats["truncated"] += 1
                    break
                stats["frames"] += 1
                parsed = protocol.parse_frame(frame)
                if not parsed["crc32_ok"]:
                    stats["bad_crc"] += 1
                elif parsed["is_ack"]:
                    stats["ack"] += 1
                elif (parsed["cmd_set"], parsed["cmd_id"]) != (
                    protocol.CMD_SET_BROADCAST,
                    protocol.CMD_ID_BROADCAST,
                ):
                    stats["other_command"] += 1
                else:
                    decoded = protocol.decode_broadcast(parsed["payload"])
                    if decoded is not None:
                        rows.append({"timestamp": timestamp, **decoded})
        self.stats = stats
        logger.info(
            "M600 telemetry %s: %d frames, %d rows, left out: %s",
            file,
            stats["frames"],
            len(rows),
            {k: v for k, v in stats.items() if k != "frames" and v},
        )
        frame = rows_to_frame(rows)
        self.data = forward_fill(frame) if fill and not frame.is_empty() else frame
