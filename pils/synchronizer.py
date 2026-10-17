"""
Synchronizer - GPS-based correlation synchronization.

This module provides correlation-based TIME synchronization with GPS payload
as the single source of truth. Uses cross-correlation to detect time offsets
between data sources.

Key Features:
- GPS payload as mandatory reference timebase
- NED position correlation for GPS sources (3D signal)
- Pitch angle correlation for inclinometer (1D signal)
- Sub-sample precision using parabolic interpolation
- Rich offset metadata for all sources
- Support for non-correlation sensors (ADC, IMU, etc.) that are already time-aligned
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import polars as pl
import pymap3d as pm
from scipy import optimize, signal

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


class Synchronizer:
    """
    GPS-based correlation synchronizer with hierarchical reference.

    This synchronizer uses cross-correlation to detect TIME offsets between
    data sources, with GPS payload as the mandatory reference timebase.

    Correlation Methods:
    - GPS sources: NED position correlation (3D signal)
    - Inclinometer: Pitch angle correlation (1D signal)
    - Camera: Timestamp alignment with optional photogrammetry support

    All synchronization outputs are TIME OFFSETS (seconds) to align data
    to GPS payload timebase.

    Attributes
    ----------
    gps_payload : Optional[pl.DataFrame]
        GPS payload data (reference timebase)
    drone_gps : Optional[pl.DataFrame]
        Optional drone GPS data
    litchi_gps : Optional[pl.DataFrame]
        Optional litchi GPS data
    inclinometer : Optional[pl.DataFrame]
        Optional inclinometer data
    camera : Optional[pl.DataFrame]
        Optional camera data (Sony or Alvium camera, or photogrammetry results)
    other_payload : Dict[str, pl.DataFrame]
        Other payload sensors (adc, imu, etc.)
    drone_records : Dict[str, pl.DataFrame]
        Other drone records (DJI .dat IMU attitude, motors, ...), on the drone
        GPS clock
    offsets : Dict[str, Dict[str, Any]]
        Detected time offsets per source
    synchronized_data : Optional[pl.DataFrame]
        Final synchronized DataFrame

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> from pils.synchronizer import Synchronizer
    >>> # Create sample GPS payload data (reference)
    >>> t = np.linspace(0, 100, 1000)
    >>> gps_payload = pl.DataFrame({
    ...     'timestamp': t,
    ...     'latitude': 45.0 + 0.001 * np.sin(0.1 * t),
    ...     'longitude': 10.0 + 0.001 * np.cos(0.1 * t),
    ...     'altitude': 100.0 + 10.0 * np.sin(0.05 * t),
    ... })
    >>> # Drone GPS on a clock 2 s ahead: same positions, later timestamps
    >>> drone_gps = gps_payload.with_columns(pl.col('timestamp') + 2.0)
    >>> # Initialize synchronizer
    >>> sync = Synchronizer()
    >>> # Add GPS payload as reference (mandatory)
    >>> sync.add_gps_reference(gps_payload)
    >>> # Add drone GPS for correlation
    >>> sync.add_drone_gps(drone_gps)
    >>> # Execute synchronization
    >>> result = sync.synchronize(target_rate={'drone': 10.0}, common_time=True)
    >>> # Check detected offsets
    >>> print(sync.get_offset_summary())
    Correlation Synchronizer - Detected Time Offsets
    ============================================================

    DRONE_GPS
      Time Offset: -2.000 s
      Correlation: 1.000
      Spatial Offset: 0.00 m
        East: 0.00 m
        North: 0.00 m
        Up: 0.00 m
    >>> # Access synchronized data
    >>> print(list(result.keys()))
    ['drone', 'reference_gps']
    >>> # Verify time offset was applied
    >>> assert abs(sync.offsets['drone_gps']['time_offset'] + 2.0) < 0.01
    """

    def __init__(self):
        """Initialize empty Synchronizer."""
        self.gps_payload: pl.DataFrame | None = None
        self.drone_gps: pl.DataFrame | None = None
        self.litchi_gps: pl.DataFrame | None = None
        self.inclinometer: pl.DataFrame | None = None
        self.camera: pl.DataFrame | None = None
        self.other_payload: dict[str, pl.DataFrame] = {}
        self.drone_records: dict[str, pl.DataFrame] = {}
        self.inclinometer_clock = "payload"
        # The reference timebase: the payload GPS, or the drone GPS for
        # flights without a payload GPS (then the drone offset is zero)
        self.reference = "payload_gps"

        self.offsets: dict[str, dict[str, Any]] = {}
        self.synchronized_data: dict[str, Any] | None = None

    @staticmethod
    def _find_subsample_peak(correlation: np.ndarray) -> float:
        """
        Find sub-sample peak location using parabolic interpolation.

        Uses 3-point parabolic fit around the maximum correlation value
        to achieve sub-sample precision in peak detection.

        Parameters
        ----------
        correlation : np.ndarray
            1D correlation array

        Returns
        -------
        float
            Sub-sample peak index

        Notes
        -----
        - Fits parabola through 3 points: [peak-1, peak, peak+1]
        - Returns integer index if peak is at array boundary
        - Numerical stability: checks for zero denominator

        Examples
        --------
        >>> # Synthetic correlation with peak between samples
        >>> x = np.arange(100)
        >>> corr = 1.0 - 0.01 * (x - 50.3) ** 2
        >>> peak = Synchronizer._find_subsample_peak(corr)
        >>> # peak ≈ 50.3
        """
        # Find integer peak location
        max_idx = int(np.argmax(correlation))

        # Handle boundary cases
        if max_idx == 0 or max_idx == len(correlation) - 1:
            return float(max_idx)

        # Get 3 points around peak for parabolic fit
        y1 = correlation[max_idx - 1]
        y2 = correlation[max_idx]
        y3 = correlation[max_idx + 1]

        # Parabolic interpolation formula
        # Peak offset from max_idx: delta = (y1 - y3) / (2 * (y1 - 2*y2 + y3))
        denominator = 2.0 * (y1 - 2.0 * y2 + y3)

        # Check for numerical stability
        if abs(denominator) < 1e-10:
            return float(max_idx)

        delta = (y1 - y3) / denominator

        # Sub-sample peak location
        peak_idx = max_idx + delta

        return peak_idx

    @staticmethod
    def __clean_data(
        time: np.ndarray, east: np.ndarray, north: np.ndarray, up: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Remove outliers and NaN values from GPS position data using velocity thresholds.

        Identifies erroneous GPS measurements by detecting velocity spikes that exceed
        physical thresholds. Removes both the outlier sample and the subsequent sample
        to eliminate corrupted segments.

        Parameters
        ----------
        time : np.ndarray
            Timestamp array in seconds
        east : np.ndarray
            East position component in meters
        north : np.ndarray
            North position component in meters
        up : np.ndarray
            Up (altitude) position component in meters

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Cleaned (time, east, north, up) arrays with outliers and NaN removed

        Notes
        -----
        - Horizontal velocity threshold: 50 m/s
        - Vertical velocity threshold: 20 m/s
        - Removes sample immediately after outlier to avoid interpolation artifacts
        - Also removes any samples with NaN values in position

        Examples
        --------
        >>> time = np.array([0, 1, 2, 3, 4])
        >>> east = np.array([0, 1, 50, 3, 4])  # Jump at idx=2
        >>> north = np.array([0, 1, 2, 3, 4])
        >>> up = np.array([0, 1, 2, 3, 4])
        >>> t_clean, e_clean, n_clean, u_clean = Synchronizer.__clean_data(
        ...     time, east, north, up
        ... )
        >>> # Outliers at indices 2,3 (and next sample) are removed
        """
        dt = np.diff(time)
        de = np.diff(east)
        dn = np.diff(north)
        du = np.diff(up)

        velocity_horizontal = np.sqrt(de**2 + dn**2) / dt
        velocity_vertical = np.abs(du) / dt

        # Detect outliers based on velocity thresholds
        threshold_horizontal = 50.0  # m/s
        threshold_vertical = 20.0  # m/s

        outliers_h = np.where(velocity_horizontal > threshold_horizontal)[0]
        outliers_v = np.where(velocity_vertical > threshold_vertical)[0]
        outliers = np.unique(np.concatenate([outliers_h, outliers_v])).astype("int")

        good_mask = np.ones(len(time), dtype=bool)
        good_mask[outliers] = False
        good_mask[outliers + 1] = False  # Also remove the next sample

        # Also remove NaN values
        nan_mask = ~(np.isnan(east) | np.isnan(north) | np.isnan(up) | np.isnan(time))
        good_mask = good_mask & nan_mask

        return time[good_mask], east[good_mask], north[good_mask], up[good_mask]

    @staticmethod
    def _estimate_time_offset(
        time1: np.ndarray,
        values1: np.ndarray,
        time2: np.ndarray,
        values2: np.ndarray,
        rate_hz: float = 100.0,
    ) -> tuple[float, np.ndarray, np.ndarray, float] | None:
        """
        Time offset that aligns source 2 to source 1, by least squares.

        Both sources are resampled on a common grid over their overlap and
        their means are removed, so a constant offset between the signals
        (lever arm, pitch bias) does not matter. A coarse lag comes from the
        peak of the cross-correlation; the shrinking overlap at large lags
        pulls it towards zero, which avoids matching a repeat of the flight
        pattern. It is then refined by minimizing the mean-removed residual
        ``values1(t) - values2(t - offset)`` at the samples of source 1.

        Parameters
        ----------
        time1, time2 : np.ndarray
            Sample times in seconds, shape ``(n1,)`` and ``(n2,)``.
        values1, values2 : np.ndarray
            Signals, shape ``(k, n1)`` and ``(k, n2)`` (``k`` axes).
        rate_hz : float, default=100.0
            Rate of the common grid for the coarse lag.

        Returns
        -------
        tuple or None
            ``(offset, axis_offsets, axis_correlations, correlation)``:
            offset in seconds to add to ``time2``; the coarse offset of each
            axis; the absolute Pearson correlation of each axis and of all
            axes after alignment (source 2 flipped if anti-correlated). None
            if there are fewer than 10 s of overlap.
        """
        values1 = np.atleast_2d(np.asarray(values1, dtype=float))
        values2 = np.atleast_2d(np.asarray(values2, dtype=float))
        ok1 = np.isfinite(time1) & np.all(np.isfinite(values1), axis=0)
        ok2 = np.isfinite(time2) & np.all(np.isfinite(values2), axis=0)
        time1, values1 = time1[ok1], values1[:, ok1]
        time2, values2 = time2[ok2], values2[:, ok2]
        order2 = np.argsort(time2)
        time2, values2 = time2[order2], values2[:, order2]
        if time1.size < 2 or time2.size < 2:
            return None

        overlap_start = max(time1[0], time2[0])
        overlap_end = min(time1[-1], time2[-1])
        if overlap_end - overlap_start < 10.0:
            logger.warning(
                f"Insufficient overlap: {overlap_end - overlap_start:.1f}s < 10s minimum"
            )
            return None

        dt = 1.0 / rate_hz
        grid = np.arange(overlap_start, overlap_end, dt)
        n = grid.size
        a = np.array([np.interp(grid, time1, v) for v in values1])
        b = np.array([np.interp(grid, time2, v) for v in values2])
        a -= a.mean(axis=1, keepdims=True)
        b -= b.mean(axis=1, keepdims=True)

        # correlate(a, b)[k] = sum a[i + k] b[i]: b(t) ~ a(t + k dt) means
        # source 2 runs k dt behind, and the offset is k dt
        lags = signal.correlation_lags(n, n, mode="full")
        usable = np.abs(lags) <= n // 2
        # Not normalized by the overlap at each lag: the falling overlap
        # favours small lags over repeats of the flight pattern, and its pull
        # on the peak is removed by the least-squares step
        per_axis = np.array(
            [signal.correlate(x, y, mode="full") for x, y in zip(a, b, strict=True)]
        )

        def peak(corr: np.ndarray) -> float:
            corr = np.where(usable, np.abs(corr), -np.inf)
            k = int(np.argmax(corr))
            if 0 < k < corr.size - 1 and np.all(np.isfinite(corr[k - 1 : k + 2])):
                y1, y2, y3 = corr[k - 1 : k + 2]
                denominator = y1 - 2.0 * y2 + y3
                if abs(denominator) > 1e-12:
                    return float((lags[k] + 0.5 * (y1 - y3) / denominator) * dt)
            return float(lags[k] * dt)

        axis_offsets = np.array([peak(c) for c in per_axis])
        total = per_axis.sum(axis=0)
        coarse = peak(total)
        # Anti-correlated signals (e.g. pitch with the opposite sign) are
        # aligned after flipping source 2
        sign = np.sign(total[np.argmin(np.abs(lags * dt - coarse))]) or 1.0
        values2 = sign * values2

        def residual(offset: float) -> np.ndarray:
            inside = (time1 > time2[0] + offset) & (time1 < time2[-1] + offset)
            shifted = np.array(
                [np.interp(time1[inside] - offset, time2, v) for v in values2]
            )
            res = values1[:, inside] - shifted
            return np.asarray(res - res.mean(axis=1, keepdims=True))

        def cost(offset: float) -> float:
            res = residual(offset)
            return float(np.mean(np.sum(res**2, axis=0))) if res.size else np.inf

        # The coarse peak can be pulled by up to about its own size
        span = max(2.0, abs(coarse))
        steps = coarse + np.arange(-span, span + dt, dt)
        best = steps[int(np.argmin([cost(step) for step in steps]))]
        fit = optimize.minimize_scalar(
            cost, bounds=(best - dt, best + dt), method="bounded"
        )
        offset = float(fit.x) if fit.success else float(best)

        inside = (time1 > time2[0] + offset) & (time1 < time2[-1] + offset)
        x = values1[:, inside]
        y = np.array([np.interp(time1[inside] - offset, time2, v) for v in values2])
        x = x - x.mean(axis=1, keepdims=True)
        y = y - y.mean(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            axis_corr = np.sum(x * y, axis=1) / np.sqrt(
                np.sum(x**2, axis=1) * np.sum(y**2, axis=1)
            )
            corr = np.sum(x * y) / np.sqrt(np.sum(x**2) * np.sum(y**2))
        return (
            offset,
            axis_offsets,
            np.abs(np.nan_to_num(axis_corr)),
            float(np.nan_to_num(corr)),
        )

    @staticmethod
    def _find_gps_offset(
        time1: np.ndarray,
        lat1: np.ndarray,
        lon1: np.ndarray,
        alt1: np.ndarray,
        time2: np.ndarray,
        lat2: np.ndarray,
        lon2: np.ndarray,
        alt2: np.ndarray,
        target_rate_hz: float = 100.0,
    ) -> dict[str, Any] | None:
        """
        Find TIME offset between two GPS sources using NED position correlation.

        Uses cross-correlation of East-North-Up position signals to detect
        the time offset needed to align source2 with source1 (reference).

        Method:
        1. Convert both GPS sources to ENU coordinates (relative to the first
           reference sample) and drop outliers
        2. Coarse offset from the cross-correlation of the mean-removed E, N, U
           positions on a common timebase (default 100 Hz)
        3. Refine it by least squares (see :meth:`_estimate_time_offset`); a
           constant position offset (lever arm) does not affect it
        4. Compute spatial offsets after time alignment

        Parameters
        ----------
        time1 : np.ndarray
            Reference GPS timestamps (seconds)
        lat1 : np.ndarray
            Reference GPS latitude (degrees)
        lon1 : np.ndarray
            Reference GPS longitude (degrees)
        alt1 : np.ndarray
            Reference GPS altitude (meters)
        time2 : np.ndarray
            Target GPS timestamps (seconds)
        lat2 : np.ndarray
            Target GPS latitude (degrees)
        lon2 : np.ndarray
            Target GPS longitude (degrees)
        alt2 : np.ndarray
            Target GPS altitude (meters)
        target_rate_hz : float, default=100.0
            Interpolation rate for correlation

        Returns
        -------
        Dict[str, Any] or None
            Dictionary with:
            - time_offset: Time offset in seconds (add to time2 to align with time1)
            - correlation: Correlation of the aligned positions [0, 1]
            - east_offset_m: East spatial offset after time alignment (meters)
            - north_offset_m: North spatial offset after time alignment (meters)
            - up_offset_m: Up spatial offset after time alignment (meters)
            - spatial_offset_m: 3D spatial offset magnitude (meters)
            - offsets_enu: Individual axis offsets in seconds (coarse)
            - correlations_enu: Individual axis correlations after alignment
            Returns None if insufficient overlap or correlation too weak

        Notes
        -----
        - The offset is added to time2: if the clock of source2 is 2 s ahead,
          the offset is -2 s
        - Requires minimum 10 seconds overlap for reliable correlation

        Examples
        --------
        >>> # The clock of GPS source 2 is 2 seconds ahead of source 1
        >>> result = Synchronizer._find_gps_offset(
        ...     time1, lat1, lon1, alt1,
        ...     time1 + 2.0, lat1, lon1, alt1
        ... )
        >>> print(f"Time offset: {result['time_offset']:.3f} s")
        Time offset: -2.000 s
        >>> print(f"Correlation: {result['correlation']:.3f}")
        Correlation: 0.998
        """
        # Use midpoint of first GPS as ENU reference
        mid_idx = 0
        ref_lat = float(lat1[mid_idx])
        ref_lon = float(lon1[mid_idx])
        ref_alt = float(alt1[mid_idx])

        # Convert GPS1 to ENU (vectorized) using pymap3d
        e1, n1, u1 = pm.geodetic2enu(lat1, lon1, alt1, ref_lat, ref_lon, ref_alt)

        time1, e1, n1, u1 = Synchronizer.__clean_data(time1, e1, n1, u1)

        # Convert GPS2 to ENU (vectorized) using pymap3d
        e2, n2, u2 = pm.geodetic2enu(lat2, lon2, alt2, ref_lat, ref_lon, ref_alt)

        # Filter GPS data

        time2, e2, n2, u2 = Synchronizer.__clean_data(time2, e2, n2, u2)

        logger.debug(
            f"Correlation: {np.any(np.isnan(e2))} {np.any(np.isnan(n2))} {np.any(np.isnan(u2))}"
        )

        logger.debug(f"Correlation: {e2} {n2} {u2}")
        logger.debug(f"Correlation: {e1} {n1} {u1}")

        estimate = Synchronizer._estimate_time_offset(
            time1,
            np.array([e1, n1, u1]),
            time2,
            np.array([e2, n2, u2]),
            rate_hz=target_rate_hz,
        )
        if estimate is None:
            logger.warning("Insufficient GPS overlap for the offset")
            return None
        time_offset, axis_offsets, axis_corr, correlation = estimate
        offset_e, offset_n, offset_u = axis_offsets
        corr_e_strength, corr_n_strength, corr_u_strength = np.abs(axis_corr)

        if correlation < 0.1:
            logger.warning("GPS correlation too weak, cannot detect reliable offset")
            return None

        # Compute spatial offsets after time alignment
        # Interpolate GPS2 with time correction
        time2_corrected = time2 + time_offset
        e2_aligned = np.interp(time1, time2_corrected, e2, left=np.nan, right=np.nan)
        n2_aligned = np.interp(time1, time2_corrected, n2, left=np.nan, right=np.nan)
        u2_aligned = np.interp(time1, time2_corrected, u2, left=np.nan, right=np.nan)

        # Compute mean spatial offsets (after time alignment)
        valid_mask = ~(
            np.isnan(e2_aligned) | np.isnan(n2_aligned) | np.isnan(u2_aligned)
        )
        if valid_mask.sum() == 0:
            east_offset_m = 0.0
            north_offset_m = 0.0
            up_offset_m = 0.0
        else:
            east_offset_m = float(np.mean(e2_aligned[valid_mask] - e1[valid_mask]))
            north_offset_m = float(np.mean(n2_aligned[valid_mask] - n1[valid_mask]))
            up_offset_m = float(np.mean(u2_aligned[valid_mask] - u1[valid_mask]))

        spatial_offset_m = np.sqrt(
            east_offset_m**2 + north_offset_m**2 + up_offset_m**2
        )

        return {
            "time_offset": float(time_offset),
            "correlation": float(correlation),
            "east_offset_m": east_offset_m,
            "north_offset_m": north_offset_m,
            "up_offset_m": up_offset_m,
            "spatial_offset_m": float(spatial_offset_m),
            "offsets_enu": {
                "east": float(offset_e),
                "north": float(offset_n),
                "up": float(offset_u),
            },
            "correlations_enu": {
                "east": float(corr_e_strength),
                "north": float(corr_n_strength),
                "up": float(corr_u_strength),
            },
        }

    @staticmethod
    def _find_pitch_offset(
        time1: np.ndarray,
        pitch1: np.ndarray,
        time2: np.ndarray,
        pitch2: np.ndarray,
        target_rate_hz: float = 100.0,
    ) -> dict[str, Any] | None:
        """
        Find TIME offset between pitch signals using cross-correlation.

        Uses cross-correlation of pitch angle signals to detect the time
        offset needed to align source2 with source1 (reference).

        Method: see :meth:`_estimate_time_offset` (coarse cross-correlation,
        then least squares). A constant pitch bias does not affect it, and
        pitch signals with opposite signs are aligned too.

        Parameters
        ----------
        time1 : np.ndarray
            Reference timestamps (seconds) - typically Litchi
        pitch1 : np.ndarray
            Reference pitch angles (degrees) - typically Litchi gimbal
        time2 : np.ndarray
            Target timestamps (seconds) - typically Inclinometer
        pitch2 : np.ndarray
            Target pitch angles (degrees) - typically Inclinometer
        target_rate_hz : float, default=100.0
            Interpolation rate for correlation

        Returns
        -------
        Dict[str, Any] or None
            Dictionary with:
            - time_offset: Time offset in seconds (add to time2 to align with time1)
            - correlation: Correlation strength [0, 1]
            Returns None if insufficient overlap or correlation too weak

        Notes
        -----
        - This is TIME synchronization, not angular offset detection
        - The offset is added to time2: if the clock of source2 is ahead,
          the offset is negative
        - Requires minimum 10 seconds overlap for reliable correlation

        Examples
        --------
        >>> # The inclinometer clock is 1.5 seconds ahead of Litchi
        >>> result = Synchronizer._find_pitch_offset(
        ...     time1_litchi, pitch1_litchi,
        ...     time2_inclinometer, pitch2_inclinometer
        ... )
        >>> print(f"Time offset: {result['time_offset']:.3f} s")
        Time offset: -1.500 s
        """
        estimate = Synchronizer._estimate_time_offset(
            time1, np.asarray(pitch1), time2, np.asarray(pitch2), target_rate_hz
        )
        if estimate is None:
            logger.warning("Insufficient pitch overlap for the offset")
            return None
        time_offset, _, _, correlation = estimate

        # Check minimum correlation threshold
        if correlation < 0.1:
            logger.warning("Pitch correlation too weak, cannot detect reliable offset")
            return None

        return {
            "time_offset": float(time_offset),
            "correlation": float(correlation),
        }

    def _standardize_coordinate_columns(
        self, df: pl.DataFrame, column_mapping: dict[str, str]
    ) -> pl.DataFrame:
        """
        Rename GPS coordinate columns to standard names.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with GPS columns to standardize
        column_mapping : dict[str, str]
            Mapping from column type to actual column name in the DataFrame
            Expected keys: 'lat_col', 'lon_col', 'alt_col', 'timestamp'

        Returns
        -------
        pl.DataFrame
            DataFrame with standardized coordinate column names:
            - lat_deg, lon_deg (WGS84)
            - h_ell_m (height above the WGS84 ellipsoid)
            - timestamp

        Notes
        -----
        The other columns keep their names.
        """
        rename_map = {}
        for key, standard in (
            ("lat_col", "lat_deg"),
            ("lon_col", "lon_deg"),
            ("alt_col", "h_ell_m"),
            ("timestamp", "timestamp"),
        ):
            name = column_mapping.get(key)
            if (
                name is not None
                and name in df.columns
                and name != standard
                and standard not in df.columns
            ):
                rename_map[name] = standard

        if rename_map:
            return df.rename(rename_map)
        return df

    def add_gps_reference(
        self,
        gps_data: pl.DataFrame,
        timestamp_col: str = "timestamp",
        lat_col: str = "lat_deg",
        lon_col: str = "lon_deg",
        alt_col: str = "h_ell_m",
    ) -> None:
        """
        Set GPS payload as reference timebase (mandatory).

        Parameters
        ----------
        gps_data : pl.DataFrame
            Polars DataFrame with GPS data
        timestamp_col : str, default='timestamp'
            Name of timestamp column
        lat_col : str, default='lat_deg'
            Name of latitude column
        lon_col : str, default='lon_deg'
            Name of longitude column
        alt_col : str, default='h_ell_m'
            Name of the ellipsoidal height column

        Raises
        ------
        ValueError
            If required columns are missing or data is empty
        """
        required_cols = [timestamp_col, lat_col, lon_col, alt_col]
        missing_cols = [col for col in required_cols if col not in gps_data.columns]

        if missing_cols:
            raise ValueError(f"GPS payload missing columns: {missing_cols}")

        if len(gps_data) == 0:
            raise ValueError("GPS payload data is empty")

        self.__ref_height = gps_data[alt_col][0]
        self.__ref_names = {
            "timestamp": timestamp_col,
            "lat_col": lat_col,
            "lon_col": lon_col,
            "alt_col": alt_col,
        }

        self.gps_payload = gps_data
        logger.info(f"Set GPS payload reference with {len(gps_data)} samples")

    def add_drone_gps(
        self,
        gps_data: pl.DataFrame | dict[str, pl.DataFrame],
        timestamp_col: str = "timestamp",
        lat_col: str = "lat_deg",
        lon_col: str = "lon_deg",
        alt_col: str = "h_ell_m",
    ) -> None:
        """
        Add drone GPS data for correlation.

        Parameters
        ----------
        gps_data : pl.DataFrame | Dict[str, pl.DataFrame]
            Polars DataFrame with GPS data, or dict containing GPS data
                timestamp_col : str, default='timestamp'
            Name of timestamp column
        lat_col : str, default='lat_deg'
            Name of latitude column
        lon_col : str, default='lon_deg'
            Name of longitude column
        alt_col : str, default='h_ell_m'
            Name of the ellipsoidal height column

        Raises
        ------
        ValueError
            If required columns are missing or data is empty
        """
        # Extract DataFrame from dict if needed
        if isinstance(gps_data, dict):
            if "gps" in gps_data:
                df = gps_data["gps"]
            elif "GPS" in gps_data:
                df = gps_data["GPS"]
            else:
                df = next(iter(gps_data.values()))
        else:
            df = gps_data

        required_cols = [timestamp_col, lat_col, lon_col, alt_col]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Drone GPS missing columns: {missing_cols}")

        if len(df) == 0:
            raise ValueError("Drone GPS data is empty")

        self.drone_gps = df

        self.__drone_names = {
            "timestamp": timestamp_col,
            "lat_col": lat_col,
            "lon_col": lon_col,
            "alt_col": alt_col,
        }

        logger.info(f"Added drone GPS with {len(gps_data)} samples")

    def add_litchi_gps(
        self,
        gps_data: pl.DataFrame,
        timestamp_col: str = "timestamp",
        lat_col: str = "lat_deg",
        lon_col: str = "lon_deg",
        alt_col: str = "h_rel_m",
        pitch_col: str = "gimbal_pitch_deg",
    ) -> None:
        """
        Add Litchi GPS data for correlation.

        Parameters
        ----------
        gps_data : pl.DataFrame
            Polars DataFrame with GPS data
        timestamp_col : str, default='timestamp'
            Name of timestamp column
        lat_col : str, default='lat_deg'
            Name of latitude column
        lon_col : str, default='lon_deg'
            Name of longitude column
        alt_col : str, default='h_rel_m'
            Name of the height above take-off. ``h_ell_m`` is added as this
            height plus the first ellipsoidal height of the reference GPS.
        pitch_col : str, default='gimbal_pitch_deg'
            Gimbal pitch, for the inclinometer and camera offsets

        Raises
        ------
        ValueError
            If required columns are missing or data is empty
        """
        required_cols = [timestamp_col, lat_col, lon_col, alt_col]
        missing_cols = [col for col in required_cols if col not in gps_data.columns]

        if missing_cols:
            raise ValueError(f"Litchi GPS missing columns: {missing_cols}")

        if len(gps_data) == 0:
            raise ValueError("Litchi GPS data is empty")

        # Take-off at the height of the reference GPS when it starts
        self.litchi_gps = gps_data.with_columns(
            (pl.col(alt_col) + self.__ref_height).alias("h_ell_m")
        )

        self.__litchi_names = {
            "timestamp": timestamp_col,
            "lat_col": lat_col,
            "lon_col": lon_col,
            "alt_col": "h_ell_m",
            "pitch": pitch_col,
        }

        logger.info(f"Added litchi GPS with {len(gps_data)} samples")

    def add_inclinometer(
        self,
        inclinometer_data: pl.DataFrame,
        inclinometer_type: str | None,
        timestamp_col: str = "timestamp",
        pitch_col: str = "pitch_deg",
        clock: str = "payload",
    ) -> None:
        """
        Add inclinometer data for pitch-based correlation.

        Parameters
        ----------
        inclinometer_data : pl.DataFrame
            Polars DataFrame with inclinometer data
        inclinometer_type : str
            Type of inclinometer sensor (e.g., 'imx5')
        timestamp_col : str, default='timestamp'
            Name of timestamp column
        pitch_col : str, default='pitch_deg'
            Name of pitch column
        clock : {"payload", "own"}, default="payload"
            ``"payload"`` if its timestamps come from the payload computer's
            clock, as the other payload sensors (IMX-5), so that its offset
            applies to them too; ``"own"`` if it has its own clock (Kernel-100:
            log start time plus a sample counter). Then the payload clock is
            tied to it through the payload IMU gyroscope, if loaded.

        Raises
        ------
        ValueError
            If required columns are missing or data is empty
        """
        required_cols = [timestamp_col, pitch_col]
        missing_cols = [
            col for col in required_cols if col not in inclinometer_data.columns
        ]

        if missing_cols:
            raise ValueError(f"Inclinometer missing columns: {missing_cols}")

        if len(inclinometer_data) == 0:
            raise ValueError("Inclinometer data is empty")

        self.inclinometer = inclinometer_data
        self.__inclinometer_names = {"timestamp": timestamp_col, "pitch": pitch_col}
        self.inclinometer_clock = clock

        logger.info(f"Added inclinometer with {len(inclinometer_data)} samples")

    def add_camera(
        self,
        camera_data: pl.DataFrame,
        use_photogrammetry: bool = True,
        timestamp_col: str = "timestamp",
        pitch_col: str = "pitch_deg",
        camera_model: str | None = None,
    ):
        """Add camera data, aligned by its pitch (photogrammetry) or its clock."""
        if use_photogrammetry:
            required_cols = [timestamp_col, pitch_col]
            missing_cols = [
                col for col in required_cols if col not in camera_data.columns
            ]

            if missing_cols:
                raise ValueError(f"Photogrammetry data missing columns: {missing_cols}")

            if len(camera_data) == 0:
                raise ValueError("Photogrammetry data is empty")

            self.__camera_model = "photogrammetry"

            self.__camera_names = {"timestamp": timestamp_col, "pitch": pitch_col}
        else:
            self.__camera_model = camera_model

            if camera_model and camera_model.lower() == "sony":
                self.__camera_names = {"timestamp": timestamp_col, "pitch": pitch_col}

        self.camera = camera_data

    def add_drone_records(self, records: dict[str, pl.DataFrame]) -> None:
        """
        Add other drone records, aligned with the drone GPS offset.

        The records must have a ``timestamp`` column on the same clock as the
        drone GPS (for DJI .dat files, the GPS tick-time fit). Each record is
        synchronized as ``drone_<name>``. Records without timestamps are
        skipped.

        Parameters
        ----------
        records : dict[str, pl.DataFrame]
            Record tables by name, e.g. ``{"IMU_ATTI(0)": df}``.
        """
        for name, df in records.items():
            if "timestamp" not in df.columns or len(df) == 0:
                logger.info(f"Skipped drone record '{name}': no timestamps")
                continue
            self.drone_records[name] = df.filter(
                pl.col("timestamp").is_not_null()
            ).sort("timestamp")
            logger.info(f"Added drone record '{name}' with {len(df)} samples")

    def add_payload_sensor(
        self,
        sensor_name: str,
        sensor_data: pl.DataFrame,
    ) -> None:
        """
        Add other payload sensor data (no correlation, simple alignment).

        Parameters
        ----------
        sensor_name : str
            Name of sensor (e.g., 'adc', 'imu')
        sensor_data : pl.DataFrame
            Polars DataFrame with sensor data

        Raises
        ------
        ValueError
            If sensor data is empty
        """
        if len(sensor_data) == 0:
            raise ValueError(f"Sensor {sensor_name} data is empty")

        self.other_payload[sensor_name] = sensor_data
        logger.info(
            f"Added payload sensor '{sensor_name}' with {len(sensor_data)} samples"
        )

    def synchronize(
        self,
        target_rate: dict[str, float],
        common_time: bool = True,
    ) -> dict[str, Any]:
        """
        Execute correlation-based synchronization.

        Detects time offsets for all sources using correlation, then
        interpolates to GPS payload timebase at target rates.

        Parameters
        ----------
        target_rate : dict
            Target sample rates in Hz for each source
            Keys: "drone", "litchi", "inclinometer", "payload" and
            "drone_records" (defaults to the "drone" rate)
            Values: float sample rate in Hz
        common_time: bool
            Interpolate all the data at a common time, with a sampliing frequency
            determined by the target_rate. If False, the time is just shifted and the
            other columns are not touched

        Returns
        -------
        dict
            Synchronized data dictionary with interpolated values for each source.
            Keys: "drone", "litchi", "reference_gps", "inclinometer", "camera",
            one key per payload sensor and ``drone_<name>`` per drone record.

            Angle columns (yaw, roll, pitch, heading; see :func:`angle_period`)
            are interpolated the short way across the wrap; state, flag and
            count columns hold their last value; ``timestamp`` is the only
            time column; missing values are null (see :func:`resample_table`).

            GPS sources (drone, litchi, reference_gps) use standardized coordinate columns:
            - latitude: Latitude in degrees (WGS84)
            - longitude: Longitude in degrees (WGS84)
            - altitude: Altitude in meters (ellipsoidal height)
            - timestamp: Time in seconds

            Non-coordinate columns preserve their original names.

        Raises
        ------
        RuntimeError
            If GPS payload reference not set
        """
        if self.gps_payload is None:
            raise RuntimeError(
                "GPS payload reference not set. Call add_gps_reference() first."
            )

        # Get GPS payload timebase
        gps_time = self.gps_payload["timestamp"].to_numpy()
        t_start, t_end = float(gps_time[0]), float(gps_time[-1])

        # Detect offsets for each source
        self.offsets = {}

        # Drone GPS offset detection
        if self.drone_gps is not None and self.reference == "drone_gps":
            self.offsets["drone_gps"] = {
                "time_offset": 0.0,
                "correlation": 1.0,
                "method": "reference",
            }
        elif self.drone_gps is not None:
            logger.info("Detecting drone GPS offset via NED correlation...")
            result = self._find_gps_offset(
                time1=self.gps_payload[self.__ref_names["timestamp"]].to_numpy(),
                lat1=self.gps_payload[self.__ref_names["lat_col"]].to_numpy(),
                lon1=self.gps_payload[self.__ref_names["lon_col"]].to_numpy(),
                alt1=self.gps_payload[self.__ref_names["alt_col"]].to_numpy(),
                time2=self.drone_gps[self.__drone_names["timestamp"]].to_numpy(),
                lat2=self.drone_gps[self.__drone_names["lat_col"]].to_numpy(),
                lon2=self.drone_gps[self.__drone_names["lon_col"]].to_numpy(),
                alt2=self.drone_gps[self.__drone_names["alt_col"]].to_numpy(),
            )
            if result:
                self.offsets["drone_gps"] = result
                logger.info(
                    f"Drone GPS offset: {result['time_offset']:.3f}s (corr={result['correlation']:.3f})"
                )
            else:
                logger.warning("Failed to detect drone GPS offset")

        # Litchi GPS offset detection
        if self.litchi_gps is not None:
            logger.info("Detecting litchi GPS offset via NED correlation...")
            result = self._find_gps_offset(
                time1=self.gps_payload[self.__ref_names["timestamp"]].to_numpy(),
                lat1=self.gps_payload[self.__ref_names["lat_col"]].to_numpy(),
                lon1=self.gps_payload[self.__ref_names["lon_col"]].to_numpy(),
                alt1=self.gps_payload[self.__ref_names["alt_col"]].to_numpy(),
                time2=self.litchi_gps[self.__litchi_names["timestamp"]].to_numpy(),
                lat2=self.litchi_gps[self.__litchi_names["lat_col"]].to_numpy(),
                lon2=self.litchi_gps[self.__litchi_names["lon_col"]].to_numpy(),
                alt2=self.litchi_gps[self.__litchi_names["alt_col"]].to_numpy(),
            )
            if result:
                self.offsets["litchi_gps"] = result
                logger.info(
                    f"Litchi GPS offset: {result['time_offset']:.3f}s (corr={result['correlation']:.3f})"
                )
            else:
                logger.warning("Failed to detect litchi GPS offset")

        # Inclinometer offset detection (using litchi gimbal pitch if available)
        if self.inclinometer is not None and self.litchi_gps is not None:
            # Check if litchi has pitch data
            if self.__litchi_names["pitch"] in self.litchi_gps.columns:
                logger.info("Detecting inclinometer offset via pitch correlation...")
                result = self._find_pitch_offset(
                    time1=self.litchi_gps[self.__litchi_names["timestamp"]].to_numpy(),
                    pitch1=self.litchi_gps[self.__litchi_names["pitch"]].to_numpy(),
                    time2=self.inclinometer[
                        self.__inclinometer_names["timestamp"]
                    ].to_numpy(),
                    pitch2=self.inclinometer[
                        self.__inclinometer_names["pitch"]
                    ].to_numpy(),
                )
                if result:
                    self.offsets["inclinometer"] = result
                    logger.info(
                        f"Inclinometer offset (relative to Litchi): {result['time_offset']:.3f}s (corr={result['correlation']:.3f})"
                    )
                else:
                    logger.warning("Failed to detect inclinometer offset")

        if self.camera is not None:
            if self.__camera_model == "photogrammetry" or self.__camera_model == "sony":
                logger.info(
                    f"Camera Model {self.__camera_model}, this implies use of pitch correlation"
                )

                if self.litchi_gps is not None:
                    result = self._find_pitch_offset(
                        time1=self.litchi_gps[
                            self.__litchi_names["timestamp"]
                        ].to_numpy(),
                        pitch1=self.litchi_gps[self.__litchi_names["pitch"]].to_numpy(),
                        time2=self.camera[self.__camera_names["timestamp"]].to_numpy(),
                        pitch2=self.camera[self.__camera_names["pitch"]].to_numpy(),
                    )
                    if result:
                        self.offsets["camera"] = result
                        logger.info(
                            f"Camera offset (relative to Litchi): {result['time_offset']:.3f}s (corr={result['correlation']:.3f})"
                        )
                    else:
                        logger.warning("Failed to detect camera offset")
                else:
                    logger.warning(
                        "Litchi GPS data not available, skipping camera pitch correlation"
                    )

            else:
                if self.__camera_model == "alvium":
                    incl_offset = self.offsets.get("inclinometer", {}).get(
                        "time_offset", 0.0
                    )

                    self.offsets["camera"] = {"time_offset": incl_offset}

                else:
                    logger.info(
                        f"Camera Model {self.__camera_model}, skipping pitch correlation"
                    )
                    logger.info(
                        "Using data timestamp and inclinometer offset as camera offset"
                    )

        # An inclinometer on the payload clock without a Litchi pitch to
        # align it to: the payload clock is taken as the reference time
        if (
            self.inclinometer is not None
            and "inclinometer" not in self.offsets
            and self.inclinometer_clock == "payload"
        ):
            self.offsets["inclinometer"] = {
                "time_offset": -self.offsets.get("litchi_gps", {}).get(
                    "time_offset", 0.0
                ),
                "method": "payload_clock",
            }
            logger.warning(
                "No Litchi pitch for the inclinometer offset: the payload clock "
                "is taken as the reference time"
            )

        rates = {
            "drone": float(target_rate.get("drone", 10.0)),
            "inclinometer": float(target_rate.get("inclinometer", 100.0)),
            "payload": float(target_rate.get("payload", 100.0)),
        }
        rates["litchi"] = float(target_rate.get("litchi", rates["drone"]))
        rates["drone_records"] = float(target_rate.get("drone_records", rates["drone"]))
        rates["reference_gps"] = float(
            target_rate.get("reference_gps", rates["payload"])
        )
        self.rates = rates

        def grid(rate: float) -> np.ndarray:
            return np.linspace(t_start, t_end, int((t_end - t_start) * rate) + 1)

        def align(
            df: pl.DataFrame, time_col: str, offset: float, rate: float
        ) -> pl.DataFrame:
            """Resample on the common grid, or only shift the time."""
            source_time = df[time_col].to_numpy().astype(float) + offset
            if common_time:
                return resample_table(df, source_time, grid(rate), time_col)
            return df.with_columns(
                pl.Series(time_col, source_time).alias("timestamp")
            ).select(["timestamp", *(c for c in df.columns if c != time_col)])

        litchi_offset = self.offsets.get("litchi_gps", {}).get("time_offset", 0.0)
        incl_offset = self.offsets.get("inclinometer", {}).get("time_offset", 0.0)
        synced: dict[str, pl.DataFrame] = {}

        if self.drone_gps is not None and "drone_gps" in self.offsets:
            synced["drone"] = align(
                self.drone_gps,
                self.__drone_names["timestamp"],
                self.offsets["drone_gps"]["time_offset"],
                rates["drone"],
            )

        if self.litchi_gps is not None and "litchi_gps" in self.offsets:
            synced["litchi"] = align(
                self.litchi_gps,
                self.__litchi_names["timestamp"],
                litchi_offset,
                rates["litchi"],
            )

        if self.inclinometer is not None and "inclinometer" in self.offsets:
            # The inclinometer offset is relative to Litchi
            total_offset = incl_offset + litchi_offset
            logger.info(
                f"Applying inclinometer total offset: {total_offset:.3f}s "
                f"(incl→litchi: {incl_offset:.3f}s + litchi→gps: {litchi_offset:.3f}s)"
            )
            time_col = self.__inclinometer_names["timestamp"]
            if isinstance(self.inclinometer, dict):
                tables = [
                    align(table, time_col, total_offset, rates["inclinometer"])
                    for table in self.inclinometer.values()
                ]
                synced["inclinometer"] = pl.concat(
                    [
                        table.rename({c: f"{key}_{c}" for c in table.columns})
                        for key, table in zip(self.inclinometer, tables, strict=True)
                    ],
                    how="horizontal",
                )
            else:
                synced["inclinometer"] = align(
                    self.inclinometer, time_col, total_offset, rates["inclinometer"]
                )

        if self.camera is not None and "camera" in self.offsets:
            # The camera offset is relative to Litchi
            total_offset = self.offsets["camera"]["time_offset"] + litchi_offset
            logger.info(
                f"Applying camera total offset: {total_offset:.3f}s "
                f"(litchi→gps: {litchi_offset:.3f}s)"
            )
            camera_time = self.camera[self.__camera_names["timestamp"]].to_numpy()
            camera_rate = float(np.average(1 / np.diff(camera_time)))
            rates["camera"] = camera_rate
            synced["camera"] = align(
                self.camera, self.__camera_names["timestamp"], total_offset, camera_rate
            )

        # Other payload sensors: the payload computer's clock
        payload_offset = self._payload_clock_offset(incl_offset + litchi_offset)
        for sensor_name, sensor_df in self.other_payload.items():
            if "timestamp" not in sensor_df.columns:
                logger.warning(f"Payload sensor '{sensor_name}' has no timestamp")
                continue
            synced[sensor_name] = align(
                sensor_df, "timestamp", payload_offset, rates["payload"]
            )

        if self.drone_records and "drone_gps" in self.offsets:
            for name, df in self.drone_records.items():
                synced[f"drone_{name}"] = align(
                    df,
                    "timestamp",
                    self.offsets["drone_gps"]["time_offset"],
                    rates["drone_records"],
                )
        elif self.drone_records:
            logger.warning("No drone GPS offset: drone records not synchronized")

        if self.reference == "payload_gps":
            synced["reference_gps"] = align(
                self.gps_payload,
                self.__ref_names["timestamp"],
                0.0,
                rates["reference_gps"],
            )

        # Standardize coordinate column names for GPS sources
        for key, names in (
            ("drone", getattr(self, "_Synchronizer__drone_names", None)),
            ("litchi", getattr(self, "_Synchronizer__litchi_names", None)),
            ("reference_gps", self.__ref_names),
        ):
            if key in synced and names is not None:
                synced[key] = self._standardize_coordinate_columns(synced[key], names)

        self.synchronized_data = synced
        logger.info(f"({t_end - t_start:.2f}s duration)")

        return self.synchronized_data

    def _payload_clock_offset(self, inclinometer_offset: float) -> float:
        """Offset of the payload computer's clock (ADC, LM76, IMU, ...).

        Recorded in ``offsets["payload_clock"]`` with the method used:

        - ``"inclinometer"``: the inclinometer uses the payload clock, its
          offset (to the reference GPS, through Litchi) applies;
        - ``"imu_gyro"``: the inclinometer has its own clock; the payload
          IMU gyroscope is aligned to the inclinometer's, axis by axis, and
          the best-correlated axis gives the payload clock;
        - ``"none"``: no way to tie the payload clock (taken as the
          reference GPS time).
        """
        has_incl = (
            self.inclinometer is not None
            and "inclinometer" in self.offsets
            and self.offsets["inclinometer"].get("method") != "payload_clock"
        )
        if has_incl and self.inclinometer_clock == "payload":
            result = {"time_offset": inclinometer_offset, "method": "inclinometer"}
        elif has_incl and "imu_gyroscope" in self.other_payload:
            assert self.inclinometer is not None
            gyro = self.other_payload["imu_gyroscope"]
            time_col = self.__inclinometer_names["timestamp"]
            best = None
            for axis in "xyz":
                incl_col, imu_col = f"gyro_{axis}_deg_s", f"gyro_{axis}_rad_s"
                if incl_col not in self.inclinometer.columns or imu_col not in gyro:
                    continue
                estimate = self._estimate_time_offset(
                    self.inclinometer[time_col].to_numpy(),
                    self.inclinometer[incl_col].to_numpy(),
                    gyro["timestamp"].to_numpy(),
                    np.degrees(gyro[imu_col].to_numpy()),
                )
                if estimate is not None and (best is None or estimate[3] > best[3]):
                    best = (*estimate, axis)
            if best is not None and best[3] > 0.5:
                result = {
                    "time_offset": inclinometer_offset + best[0],
                    "method": "imu_gyro",
                    "axis": best[4],
                    "correlation": float(best[3]),
                }
            else:
                logger.warning("Payload IMU and inclinometer gyros do not match")
                result = {"time_offset": 0.0, "method": "none"}
        else:
            if self.other_payload:
                logger.warning(
                    "No offset for the payload clock: the inclinometer has its "
                    "own clock and there is no payload IMU gyroscope"
                )
            result = {"time_offset": 0.0, "method": "none"}
        self.offsets["payload_clock"] = result
        logger.info(f"Payload clock offset: {result}")
        return float(result["time_offset"])

    def get_offset_summary(self) -> str:
        """
        Get summary of detected time offsets.

        Returns
        -------
        str
            Formatted string with offset information
        """
        if not self.offsets:
            return "No offsets detected. Run synchronize() first."

        lines = ["Correlation Synchronizer - Detected Time Offsets", "=" * 60]

        for source_name, offset_data in self.offsets.items():
            lines.append(f"\n{source_name.upper()}")

            # For inclinometer, show both relative offset and total offset
            if source_name == "inclinometer":
                incl_offset = offset_data["time_offset"]
                litchi_offset = self.offsets.get("litchi_gps", {}).get(
                    "time_offset", 0.0
                )
                total_offset = incl_offset + litchi_offset
                lines.append(f"  Time Offset (relative to Litchi): {incl_offset:.3f} s")
                lines.append(
                    f"  Time Offset (total, relative to GPS): {total_offset:.3f} s"
                )
            else:
                lines.append(f"  Time Offset: {offset_data['time_offset']:.3f} s")

            if "method" in offset_data:
                lines.append(f"  Method: {offset_data['method']}")
            if "correlation" in offset_data:
                lines.append(f"  Correlation: {offset_data['correlation']:.3f}")

            if "spatial_offset_m" in offset_data:
                lines.append(
                    f"  Spatial Offset: {offset_data['spatial_offset_m']:.2f} m"
                )
                lines.append(f"    East: {offset_data['east_offset_m']:.2f} m")
                lines.append(f"    North: {offset_data['north_offset_m']:.2f} m")
                lines.append(f"    Up: {offset_data['up_offset_m']:.2f} m")

        return "\n".join(lines)
