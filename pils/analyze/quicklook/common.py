"""Shared pieces of the quick look.

Thresholds, the report :class:`Section`, formatting helpers and the data
checks (pure functions).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from pils.utils.logging_config import get_logger

logger = get_logger(__name__)


QUICKLOOK_DIR = "quicklook"
REPORT_FILE = "report.md"

# The drone is airborne above this height over the take-off point (m)
AIRBORNE_HEIGHT_M = 1.0
# A sensor gap longer than this is reported (s)
MAX_GAP_S = 1.0
# A sensor covering less of the airborne time than this is flagged (%)
MIN_COVERAGE_PCT = 95.0
# A time offset correlation below this is flagged
MIN_CORRELATION = 0.9
# A clock offset larger than this is flagged (s)
MAX_CLOCK_OFFSET_S = 60.0
# A Litchi log overlapping the drone's airborne time less than this is
# flagged as the log of another flight (%)
MIN_LITCHI_OVERLAP_PCT = 80.0
# SNR levels (dB-Hz) at which the satellites are counted
SNR_LEVELS = (30, 35, 40)
# DJI RTK position flag (rtk_pos_flag_1): solution type
DJI_RTK_FLAGS = {50: "fixed", 34: "float", 16: "single", 0: "none"}
# Series colours, in this order (light theme)
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6")


# -- small helpers ----------------------------------------------------------


def _seconds(frame: pl.DataFrame) -> np.ndarray | None:
    """UNIX seconds of a table's ``timestamp`` (float or datetime), or None."""
    if "timestamp" not in frame.columns or frame.is_empty():
        return None
    column = frame["timestamp"]
    if column.dtype.is_temporal():
        values = column.dt.epoch("us").cast(pl.Float64) / 1e6
    elif column.dtype.is_numeric():
        values = column.cast(pl.Float64)
    else:
        return None
    array = values.drop_nulls().to_numpy()
    return array[np.isfinite(array)] if array.size else None


def _utc(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "-"
    moment = datetime.datetime.fromtimestamp(seconds, datetime.UTC)
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return "-" if not np.isfinite(value) else f"{value:.{digits}f}"
    return str(value)


def _size(megabytes: float | None) -> str:
    if megabytes is None:
        return "-"
    if megabytes >= 1000:
        return f"{megabytes / 1000:.1f} GB"
    if megabytes >= 1:
        return f"{megabytes:.1f} MB"
    return f"{megabytes * 1000:.0f} kB"


def md_table(rows: list[dict[str, Any]], digits: int = 1) -> str:
    """Markdown table of rows (dicts with the same keys)."""
    if not rows:
        return "_none_"
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_fmt(row.get(c), digits) for c in columns) + " |"
        )
    return "\n".join(lines)


def local_en(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """East and north (m) from the first point, small-area approximation."""
    radius = 6_371_000.0
    lat0, lon0 = np.radians(lat[0]), np.radians(lon[0])
    east = radius * np.cos(lat0) * (np.radians(lon) - lon0)
    north = radius * (np.radians(lat) - lat0)
    return east, north


# -- data checks (pure functions, tested on their own) ----------------------


def airborne_interval(
    times: np.ndarray, height: np.ndarray, threshold: float = AIRBORNE_HEIGHT_M
) -> tuple[float, float] | None:
    """Longest stretch with the height above ``threshold`` (UNIX s)."""
    above = np.asarray(height) > threshold
    if not above.any():
        return None
    edges = np.diff(np.concatenate([[0], above.astype(int), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    k = int(np.argmax(times[stops] - times[starts]))
    return float(times[starts[k]]), float(times[stops[k]])


def table_summary(
    times: np.ndarray, interval: tuple[float, float] | None
) -> dict[str, Any]:
    """Describe samples, span, rate, gaps and coverage of the airborne interval."""
    times = np.sort(times)
    steps = np.diff(times)
    # A gap: longer than 1 s and than 5 sampling steps (slow sensors)
    limit = max(MAX_GAP_S, 5 * float(np.median(steps))) if steps.size else MAX_GAP_S
    gaps = steps[steps > limit]
    summary: dict[str, Any] = {
        "samples": int(times.size),
        "start_utc": _utc(float(times[0])),
        "duration_s": float(times[-1] - times[0]),
        "rate_hz": float(1 / np.median(steps)) if steps.size else float("nan"),
        "gaps": int(gaps.size),
        "longest_gap_s": float(gaps.max()) if gaps.size else 0.0,
        "coverage_pct": float("nan"),
    }
    if interval is not None and interval[1] > interval[0]:
        start, end = interval
        inside = times[(times >= start) & (times <= end)]
        if inside.size:
            # Covered time: the span minus the gaps, inside the interval
            edges = np.concatenate([[start], inside, [end]])
            missing = np.diff(edges)
            covered = (end - start) - missing[missing > limit].sum()
            summary["coverage_pct"] = float(100 * max(covered, 0) / (end - start))
        else:
            summary["coverage_pct"] = 0.0
    return summary


def overlap_pct(span: tuple[float, float], interval: tuple[float, float]) -> float:
    """Share of ``interval`` (%) inside ``span``."""
    overlap = min(span[1], interval[1]) - max(span[0], interval[0])
    return float(100 * max(overlap, 0.0) / max(interval[1] - interval[0], 1e-9))


def rtk_flag_summary(flags: pl.Series) -> dict[str, float]:
    """Share (%) of each DJI RTK solution type."""
    values = flags.drop_nulls().to_numpy()
    if not values.size:
        return {}
    out = {
        name: float(100 * np.mean(values == code))
        for code, name in DJI_RTK_FLAGS.items()
    }
    out["other"] = float(100 - sum(out.values()))
    return out


def offset_rows(offsets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per synchronisation offset."""
    rows = []
    for source, result in offsets.items():
        rows.append(
            {
                "source": source,
                "offset_s": float(result.get("time_offset", float("nan"))),
                "correlation": result.get("correlation"),
                "method": result.get("method", "correlation"),
                "spatial_offset_m": result.get("spatial_offset_m"),
            }
        )
    return rows


def offset_warnings(offsets: dict[str, dict[str, Any]]) -> list[str]:
    """List what looks wrong in the synchronisation offsets."""
    warnings = []
    for source, result in offsets.items():
        corr = result.get("correlation")
        if corr is not None and corr < MIN_CORRELATION:
            warnings.append(
                f"Sync: {source} correlation {corr:.2f} (< {MIN_CORRELATION}): "
                "check the offset"
            )
        offset = result.get("time_offset")
        if (
            offset is not None
            and source in ("drone_gps", "litchi_gps")
            and abs(offset) > MAX_CLOCK_OFFSET_S
        ):
            warnings.append(f"Sync: {source} clock off by {offset:.0f} s")
    if offsets.get("payload_clock", {}).get("method") == "none":
        warnings.append(
            "Sync: no offset for the payload clock (ADC, LM76, IMU taken as GPS time)"
        )
    return warnings


@dataclass
class Section:
    """A report section: a title and Markdown lines."""

    title: str
    lines: list[str] = field(default_factory=list)

    def add(self, *lines: str) -> None:
        """Append lines of Markdown to the section."""
        self.lines.extend(lines)
