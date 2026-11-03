"""Resampling tables on a time grid: angles unwrapped, states held, clocks dropped."""

from __future__ import annotations

import logging
import re

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


# Angle columns, by name (the part after the last ":"), and their period.
# Pitch is included: it does not wrap, so unwrapping leaves it unchanged.
_ANGLE_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
    # Schema names: yaw_deg, gimbal_heading_deg, rtk_yaw_deg, roll_rad, ...
    (re.compile(r"(^|_)(yaw|roll|pitch|heading)_deg$"), 360.0),
    (re.compile(r"(^|_)(yaw|roll|pitch|heading)_rad$"), 2 * np.pi),
    # Source names: yaw, yaw(deg), gimbalYawAbs, yawRaw (0.1 deg)
    (re.compile(r"^(gimbal)?(yaw|roll|pitch|heading)(abs)?(\(deg\))?$"), 360.0),
    (re.compile(r"^(gimbal)?(yaw|roll|pitch)raw$"), 3600.0),
)


def angle_period(column: str) -> float | None:
    """Period of an angle column, or None if the column is not an angle.

    Parameters
    ----------
    column : str
        Column name: schema names such as ``"yaw_deg"``, ``"rtk_yaw_deg"``,
        ``"gimbal_heading_deg"`` or ``"roll_rad"``, or source names such as
        ``"yaw(deg)"``, ``"RTK:yaw"`` or ``"gimbalYawRaw"`` (0.1 deg).

    Returns
    -------
    float or None
        360 for degrees, 2*pi for radians, 3600 for 0.1 deg.
    """
    leaf = column.rsplit(":", 1)[-1].lower()
    for pattern, period in _ANGLE_PATTERNS:
        if pattern.match(leaf):
            return period
    return None


def interp_angle(
    x: np.ndarray, xp: np.ndarray, fp: np.ndarray, period: float = 360.0
) -> np.ndarray:
    """Linear interpolation of an angle, taking the short way across the wrap.

    The angle is unwrapped, interpolated and wrapped again to the range of
    the input: ``[0, period)`` if it has no negative values, otherwise
    ``[-period/2, period/2)``. Outside ``xp`` the result is NaN.

    Parameters
    ----------
    x : np.ndarray
        Times to interpolate at.
    xp : np.ndarray
        Sample times, increasing.
    fp : np.ndarray
        Angles at ``xp``; NaN samples are skipped by the unwrapping.
    period : float, default=360.0
        Angle period (360 for degrees, 2*pi for radians).

    Returns
    -------
    np.ndarray
        Interpolated angles.
    """
    fp = np.asarray(fp, dtype=float)
    unwrapped = fp.copy()
    ok = np.isfinite(fp)
    unwrapped[ok] = np.unwrap(fp[ok], period=period)
    out = np.interp(x, xp, unwrapped, left=np.nan, right=np.nan)
    low = 0.0 if np.nanmin(fp, initial=0.0) >= 0 else -period / 2
    return np.asarray((out - low) % period + low)


def interp_column(
    x: np.ndarray, xp: np.ndarray, fp: np.ndarray, column: str
) -> np.ndarray:
    """Interpolate a column at ``x``; angle columns are interpolated as angles.

    Outside ``xp`` the result is NaN. See :func:`angle_period` for the angle
    columns.
    """
    period = angle_period(column)
    if period is not None:
        return interp_angle(x, xp, fp, period)
    return np.asarray(np.interp(x, xp, fp, left=np.nan, right=np.nan))


# Columns that repeat the time base in another clock or unit: dropped by
# resampling, which keeps ``timestamp`` as the only time column
REDUNDANT_TIME_COLUMNS = frozenset(
    {
        "unix_time_ms",
        "timestamp_ms",
        "timestamp_us",
        "timestamp_ns",
        "monotonic_ns",
        "datetime_relative",
        "msg_type",
    }
)


# Columns that hold their last value instead of being interpolated: states,
# flags and counts (matched on any ":"-separated part of the name, any case)
_HELD_COLUMNS = re.compile(
    r"(status|flag|flg|state|mode|(^|_)num|sats|(^|_)svn|is_?flying|(^|_)week$)",
    re.IGNORECASE,
)


def is_held_column(column: str) -> bool:
    """Tell whether a column is a state, flag or count (not interpolated).

    Examples
    --------
    >>> is_held_column("rtk_pos_flag_0"), is_held_column("status_rfront")
    (True, True)
    >>> is_held_column("gps_num_sv"), is_held_column("ins_status")
    (True, True)
    >>> is_held_column("h_msl_m"), is_held_column("mag_x")
    (False, False)
    """
    return any(_HELD_COLUMNS.search(part) for part in column.split(":"))


# GNSS date and time fields (YYYYMMDD, hhmmss): other clocks, dropped too
_CLOCK_FIELDS = re.compile(r"(^|_)(date|time)$")


def resample_table(
    df: pl.DataFrame,
    source_time: np.ndarray,
    target_time: np.ndarray,
    time_col: str = "timestamp",
) -> pl.DataFrame:
    """Resample a table at ``target_time``.

    - ``timestamp`` (seconds) is the only time column: date/time and text
      columns, :data:`REDUNDANT_TIME_COLUMNS` and GNSS ``date``/``time``
      fields are dropped;
    - state, flag and count columns (:func:`is_held_column`) and booleans
      hold their last value and keep their type;
    - angle columns are interpolated across the wrap (:func:`interp_column`),
      the other columns linearly;
    - outside the source time range, or where the source has no value, the
      result is null.

    Parameters
    ----------
    df : pl.DataFrame
        Source table.
    source_time : np.ndarray
        Time of each row of ``df`` on the target clock, in seconds.
    target_time : np.ndarray
        Times to resample at, increasing.
    time_col : str, default="timestamp"
        Time column of ``df``, not resampled.

    Returns
    -------
    pl.DataFrame
        ``timestamp`` followed by the resampled columns.
    """
    source_time = np.asarray(source_time, dtype=float)
    keep = np.isfinite(source_time)
    order = np.argsort(source_time[keep], kind="stable")
    rows = np.flatnonzero(keep)[order]
    source_time = source_time[rows]
    df = df[rows]

    columns: list[pl.Series] = [pl.Series("timestamp", target_time)]
    if source_time.size == 0:
        return pl.DataFrame(columns)
    idx = np.clip(np.searchsorted(source_time, target_time, side="right") - 1, 0, None)
    outside = pl.Series(
        (target_time < source_time[0]) | (target_time > source_time[-1])
    )
    for col in df.columns:
        dtype = df.schema[col]
        if (
            col == time_col
            or col in REDUNDANT_TIME_COLUMNS
            or _CLOCK_FIELDS.search(col)
            or dtype.is_temporal()
            or not (dtype.is_numeric() or dtype == pl.Boolean)
        ):
            continue
        if dtype == pl.Boolean or is_held_column(col):
            held = df[col].gather(idx)
            columns.append(
                pl.select(pl.when(pl.lit(outside)).then(None).otherwise(pl.lit(held)))
                .to_series()
                .alias(col)
            )
        else:
            values = df[col].cast(pl.Float64).to_numpy()
            columns.append(
                pl.Series(
                    col,
                    interp_column(target_time, source_time, values, col),
                    nan_to_null=True,
                )
            )
    return pl.DataFrame(columns)
