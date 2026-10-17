"""Payload files of the 2022 and 2023 campaigns (``*.pck``).

The payload computer appended one pickle per record (2022) or per block of
the ADC buffer (2023) to the file; the files can end with zeros where they
were preallocated. Only plain values are unpickled: anything else (classes,
functions) is refused.

- 2022: ``*_ADC_<n>.pck`` holds ``["ADC", [unix_time, volts]]``,
  ``*_INC_<n>.pck`` holds ``["INC", [unix_time, (x_deg, y_deg)]]``; ``<n>``
  numbers the files of one recording.
- 2023: ``*_ADC.pck`` holds 256-byte blocks of 8-byte samples: the time as a
  float32 (a 128 s resolution at 2023) and the ADS1115 reading (int32). The
  times of the samples are rebuilt from the sample count (see
  :func:`adc_blocks_to_frame`). ``*_GPS.pck`` and ``*_INC.pck`` hold only
  ``None``: those sensors recorded nothing.
"""

from __future__ import annotations

import io
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

PICKLE_START = b"\x80"  # PROTO opcode, any protocol >= 2
ADS1115_FULL_SCALE_V = 4.096  # gain 1, the ADS1115 default in the 2023 code


class _PlainUnpickler(pickle.Unpickler):
    """Unpickler for plain values only (numbers, strings, lists, ...)."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"{module}.{name} is not a plain value")


def read_pickles(path: str | Path) -> Iterator[Any]:
    """Every value pickled in a file, one after the other.

    Bytes that are not a pickle (preallocated zeros, a record cut at the
    end) are skipped.
    """
    data = Path(path).read_bytes()
    position, skipped = 0, 0
    while True:
        start = data.find(PICKLE_START, position)
        if start < 0:
            break
        stream = io.BytesIO(data)
        stream.seek(start)
        try:
            value = _PlainUnpickler(stream).load()
        except Exception:
            skipped += 1
            position = start + 1
            continue
        position = stream.tell()
        yield value
    if skipped:
        logger.debug(f"{Path(path).name}: {skipped} non-pickle bytes skipped")


def legacy_files(folder: str | Path, sensor: str) -> list[Path]:
    """List the ``*_<SENSOR>.pck`` or ``*_<SENSOR>_<n>.pck`` files, in order."""
    folder = Path(folder)
    files = sorted(folder.glob(f"*_{sensor}.pck")) + sorted(
        folder.glob(f"*_{sensor}_*.pck"),
        key=lambda p: (
            int(p.stem.rsplit("_", 1)[1]) if p.stem.rsplit("_", 1)[1].isdigit() else 0
        ),
    )
    return files


CLOCK_JUMP_S = 60.0


def undo_clock_jumps(times: np.ndarray) -> np.ndarray:
    """Remove the payload clock's jumps from the times of one recording.

    The 2022 payload computer started with the clock it had saved (days
    behind) and jumped forward when the network set it, during the
    recording. Samples before a forward jump of more than a minute are
    moved after it, as if the sampling went on at its median interval.
    """
    times = np.asarray(times, dtype=float).copy()
    if times.size < 2:
        return times
    steps = np.diff(times)
    interval = float(np.median(steps))
    for k in np.flatnonzero(steps > CLOCK_JUMP_S)[::-1]:
        times[: k + 1] += steps[k] - interval
    return times


def _records(files: list[Path], tag: str) -> list[list[Any]]:
    """Read the records of the files, each file's clock jumps removed."""
    records = []
    for path in files:
        found = [
            list(value[1])
            for value in read_pickles(path)
            if isinstance(value, list)
            and len(value) == 2
            and value[0] == tag
            and isinstance(value[1], list | tuple)
            and len(value[1]) == 2
        ]
        if not found:
            continue
        times = undo_clock_jumps(np.array([float(r[0]) for r in found]))
        for record, time in zip(found, times, strict=True):
            record[0] = float(time)
        records.extend(found)
    return records


def adc_blocks_to_frame(blocks: list[bytes]) -> pl.DataFrame:
    """2023 ADC: samples with times rebuilt from the coarse float32 times.

    A float32 time changes value (by 128 s) when the true time crosses the
    midpoint between two float32 values; those sample indices are exact
    times. A straight line through them (time against sample index) gives
    the time of every sample; with a single change the median sampling
    rate of the other 2023 files cannot be known, so the samples are
    spread over the recording at the rate between the first and last
    blocks' times.
    """
    raw = b"".join(blocks)
    raw = raw[: len(raw) // 8 * 8]
    samples = np.frombuffer(raw, dtype=np.dtype([("t", "<f4"), ("v", "<i4")]))
    if samples.size == 0:
        return pl.DataFrame()
    coarse = samples["t"].astype(np.float64)
    index = np.arange(coarse.size, dtype=float)
    changes = np.flatnonzero(np.diff(coarse) > 0) + 1
    if changes.size >= 2:
        # At a change the true time is the midpoint of the two float32 values
        crossing = (coarse[changes - 1] + coarse[changes]) / 2
        slope, intercept = np.polyfit(changes.astype(float), crossing, 1)
    elif changes.size == 1:
        step = coarse[changes[0]] - coarse[changes[0] - 1]
        crossing = (coarse[changes[0] - 1] + coarse[changes[0]]) / 2
        # Over one float32 step the sample count gives the rate
        slope = step / max(changes[0], coarse.size - changes[0])
        intercept = crossing - slope * changes[0]
    else:
        slope, intercept = 0.0, float(coarse[0])
    volts = samples["v"].astype(float) * ADS1115_FULL_SCALE_V / 32768
    return pl.DataFrame(
        {
            "timestamp": intercept + slope * index,
            "amplitude": volts * 1e3,  # mV
        }
    )


def read_legacy_adc(folder: str | Path) -> pl.DataFrame | None:
    """ADC data of the ``*_ADC*.pck`` files of a folder, None if there are none.

    Returns
    -------
    pl.DataFrame or None
        ``timestamp`` (s, payload clock) and ``amplitude`` (mV).
    """
    files = legacy_files(folder, "ADC")
    if not files:
        return None
    records = _records(files, "ADC")
    if records:
        frame = pl.DataFrame(
            {
                "timestamp": [float(r[0]) for r in records],
                "amplitude": [float(r[1]) * 1e3 for r in records],  # V to mV
            }
        )
    else:
        blocks = [
            value
            for path in files
            for value in read_pickles(path)
            if isinstance(value, bytes)
        ]
        frame = adc_blocks_to_frame(blocks)
    if frame.is_empty():
        return frame
    return frame.sort("timestamp").with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("datetime")
    )


def read_legacy_inclinometer(folder: str | Path) -> pl.DataFrame | None:
    """2022 inclinometer (``*_INC*.pck``), None if there are no such files.

    The two angles are stored as recorded, as ``angle_x_deg`` and
    ``angle_y_deg`` (the Kernel's ``Roll`` and ``Pitch`` outputs); which one
    is the payload pitch is found when synchronizing (see
    :meth:`pils.synchronizer.Synchronizer.add_inclinometer`).

    Returns
    -------
    pl.DataFrame or None
        ``timestamp`` (s, payload clock), ``roll`` and ``pitch`` (deg), empty
        if the files hold no records (2023).
    """
    files = legacy_files(folder, "INC")
    if not files:
        return None
    records = [
        r
        for r in _records(files, "INC")
        if isinstance(r[1], list | tuple) and len(r[1]) == 2
    ]
    if not records:
        return pl.DataFrame()
    frame = pl.DataFrame(
        {
            "timestamp": [float(r[0]) for r in records],
            # Kernel axes as in KernelInclinometer: Roll is the payload
            # pitch (sign flipped), Pitch the roll
            "pitch": [-float(r[1][0]) for r in records],
            "roll": [float(r[1][1]) for r in records],
        }
    )
    return frame.sort("timestamp").with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("datetime")
    )
