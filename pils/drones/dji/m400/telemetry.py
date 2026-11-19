"""DJI Matrice 400 telemetry as LAGER logs it (``uav_telemetry.bin``, the ``psdk_telemetry`` stream)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ....utils.logging_config import get_logger
from ....utils.telemetry_frames import forward_fill, resolve_file, rows_to_frame
from . import decoders, stream
from . import parameters as params

logger = get_logger(__name__)

# M600 battery key -> field of the Matrice 400's single battery (bat1_*)
BATTERY_SUMMARY = {
    "bat_percentage": "bat1_percentage",
    "bat_voltage_mv": "bat1_voltage_mv",
    "bat_current_ma": "bat1_current_ma",
    "bat_capacity_mah": "bat1_remaining_mah",
}


class M400Telemetry:
    """Loader of the telemetry of a Matrice 400 (Payload SDK through the E-Port).

    Ported from LAGER's ``DJI_M400.DataLoader`` (see :mod:`.stream` and
    :mod:`.decoders`). ``data`` is one row per DATA record: ``timestamp`` (the
    logging computer's clock, UNIX seconds), ``t_aircraft_ms``, ``topic`` and
    the fields of that topic. The topics arrive at different rates, so with
    ``fill`` (the default, as in LAGER) each column carries its last value
    forward and every row is a full state; without it the other columns of a
    row are null.

    Attributes
    ----------
    path : the file or the folder holding ``uav_telemetry.bin``
    data : the table, after :meth:`load_data`
    status : the program's STATUS messages, ``(timestamp, state, text)``
    stats : records read and what was left out (``bad_crc`` of the stream,
        ``unknown_topic``, ``bad_size``, ``bad_value``)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = path
        self.data: pl.DataFrame | None = None
        self.status: list[tuple[float, str, str]] = []
        self.stats: dict[str, int] = {}

    def load_data(self, fill: bool = True) -> None:
        """Read the file into :attr:`data`.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        file = resolve_file(self.path, params.TELEMETRY_FILENAME)
        parser = stream.StreamParser()
        records = parser.feed(file.read_bytes())

        errors: dict[str, int] = {}
        topics: dict = {}
        rows: list[dict] = []
        self.status = []
        for rtype, topic, host_ns, ac_ms, _ac_us, payload in records:
            if rtype == params.RECORD_TOPIC_TABLE:
                topics = stream.parse_topic_table(
                    payload
                )  # a new table at every restart of the program
            elif rtype == params.RECORD_STATUS:
                state, text = stream.parse_status(payload)
                self.status.append(
                    (host_ns / 1e9, params.STATE_NAMES.get(state, str(state)), text)
                )
            elif rtype == params.RECORD_DATA and topic in topics:
                name = topics[topic]["name"]
                values = decoders.decode(name, payload, errors)
                if values:
                    values.pop(
                        "q", None
                    )  # the quaternion as a list: roll, pitch, yaw are enough
                    rows.append(
                        {
                            "timestamp": host_ns / 1e9,
                            "t_aircraft_ms": ac_ms,
                            "topic": name,
                            **values,
                        }
                    )
        self.stats = {"records": len(records), "bad_crc": parser.bad_crc, **errors}
        for t, state, text in self.status:
            logger.info("M400 psdk_telemetry %.3f %s: %s", t, state, text)
        logger.info(
            "M400 telemetry %s: %d records, %d rows, left out: %s",
            file,
            len(records),
            len(rows),
            {k: v for k, v in self.stats.items() if k != "records" and v},
        )

        frame = rows_to_frame(rows)
        if frame.is_empty():
            self.data = frame
            return
        if fill:
            frame = forward_fill(frame)
            # whole-aircraft battery keys, as the M600
            frame = frame.with_columns(
                [
                    pl.col(src).alias(col)
                    for col, src in BATTERY_SUMMARY.items()
                    if src in frame.columns
                ]
            )
        self.data = frame
