"""Estimating the time offset between two sources by correlation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pymap3d as pm
from scipy import optimize, signal

logger = logging.getLogger(__name__)


class OffsetEstimationMixin:
    """Time offsets by correlation (GPS positions, pitch), for :class:`~pils.synchronizer.Synchronizer`."""

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
    def _clean_data(
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
        >>> t_clean, e_clean, n_clean, u_clean = Synchronizer._clean_data(
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

        time1, e1, n1, u1 = OffsetEstimationMixin._clean_data(time1, e1, n1, u1)

        # Convert GPS2 to ENU (vectorized) using pymap3d
        e2, n2, u2 = pm.geodetic2enu(lat2, lon2, alt2, ref_lat, ref_lon, ref_alt)

        # Filter GPS data

        time2, e2, n2, u2 = OffsetEstimationMixin._clean_data(time2, e2, n2, u2)

        logger.debug(
            f"Correlation: {np.any(np.isnan(e2))} {np.any(np.isnan(n2))} {np.any(np.isnan(u2))}"
        )

        logger.debug(f"Correlation: {e2} {n2} {u2}")
        logger.debug(f"Correlation: {e1} {n1} {u1}")

        estimate = OffsetEstimationMixin._estimate_time_offset(
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
        estimate = OffsetEstimationMixin._estimate_time_offset(
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
