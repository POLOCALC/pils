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
from typing import Any

import numpy as np
import polars as pl
from scipy import signal

logger = logging.getLogger(__name__)


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
    >>> # Create drone GPS data with 2-second time offset
    >>> drone_gps = pl.DataFrame({
    ...     'timestamp': t + 2.0,  # Drone data 2s ahead
    ...     'latitude': 45.0 + 0.001 * np.sin(0.1 * (t + 2.0)),
    ...     'longitude': 10.0 + 0.001 * np.cos(0.1 * (t + 2.0)),
    ...     'altitude': 100.0 + 10.0 * np.sin(0.05 * (t + 2.0)),
    ... })
    >>> # Initialize synchronizer
    >>> sync = Synchronizer()
    >>> # Add GPS payload as reference (mandatory)
    >>> sync.add_gps_reference(gps_payload)
    >>> # Add drone GPS for correlation
    >>> sync.add_drone_gps(drone_gps)
    >>> # Execute synchronization
    >>> result = sync.synchronize(target_rate_hz=10.0)
    >>> # Check detected offsets
    >>> print(sync.get_offset_summary())
    Correlation Synchronizer - Detected Time Offsets
    ============================================================

    DRONE_GPS
      Time Offset: 2.000 s
      Correlation: 0.998
      Spatial Offset: 15.32 m
        East: 10.23 m
        North: 11.45 m
        Up: 0.12 m
    >>> # Access synchronized data
    >>> print(result.columns)
    ['timestamp', 'gps_payload_latitude', 'gps_payload_longitude',
     'gps_payload_altitude', 'drone_gps_latitude', 'drone_gps_longitude',
     'drone_gps_altitude']
    >>> # Verify time offset was applied
    >>> assert abs(sync.offsets['drone_gps']['time_offset'] - 2.0) < 0.1
    """

    def __init__(self):
        """Initialize empty Synchronizer."""
        self.gps_payload: pl.DataFrame | None = None
        self.drone_gps: pl.DataFrame | None = None
        self.litchi_gps: pl.DataFrame | None = None
        self.inclinometer: pl.DataFrame | None = None
        self.camera: pl.DataFrame | None = None
        self.other_payload: dict[str, pl.DataFrame] = {}

        self.offsets: dict[str, dict[str, Any]] = {}
        self.synchronized_data: dict[str, Any] | None = None

    @staticmethod
    def _lla_to_enu(
        ref_lat: float,
        ref_lon: float,
        ref_alt: float,
        target_lat: float,
        target_lon: float,
        target_alt: float,
    ) -> tuple[float, float, float]:
        """
        Convert LLA (Latitude, Longitude, Altitude) to local ENU coordinates.

        Converts target LLA coordinates to East-North-Up (ENU) coordinates
        relative to a reference point. Uses spherical Earth approximation
        suitable for distances up to ~100 km.

        Parameters
        ----------
        ref_lat : float
            Reference latitude in degrees
        ref_lon : float
            Reference longitude in degrees
        ref_alt : float
            Reference altitude in meters
        target_lat : float
            Target latitude in degrees
        target_lon : float
            Target longitude in degrees
        target_alt : float
            Target altitude in meters

        Returns
        -------
        Tuple[float, float, float]
            (east, north, up) offsets in meters

        Notes
        -----
        - Uses mean Earth radius of 6371 km
        - Assumes flat Earth for small distances
        - Longitude correction for latitude (cos factor)

        Examples
        --------
        >>> # Point 1 degree north and 1 degree east at same altitude
        >>> e, n, u = Synchronizer._lla_to_enu(
        ...     45.0, 10.0, 100.0,
        ...     46.0, 11.0, 100.0
        ... )
        >>> # e ≈ 78 km (1° east at 45° lat), n ≈ 111 km (1° north), u ≈ 0
        """
        # Earth radius in meters
        R = 6371000.0

        # Convert degrees to radians
        ref_lat_rad = np.deg2rad(ref_lat)
        ref_lon_rad = np.deg2rad(ref_lon)
        target_lat_rad = np.deg2rad(target_lat)
        target_lon_rad = np.deg2rad(target_lon)

        # Differences in radians
        dlat = target_lat_rad - ref_lat_rad
        dlon = target_lon_rad - ref_lon_rad

        # ENU calculation (flat Earth approximation)
        # North: latitude difference
        north = R * dlat

        # East: longitude difference corrected for latitude
        east = R * dlon * np.cos(ref_lat_rad)

        # Up: altitude difference
        up = target_alt - ref_alt

        return east, north, up

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
        nan_mask = ~(np.isnan(east) | np.isnan(north) | np.isnan(up))
        good_mask = good_mask & nan_mask

        return time[good_mask], east[good_mask], north[good_mask], up[good_mask]

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
        1. Convert both GPS sources to ENU coordinates (relative to reference midpoint)
        2. Interpolate both to common high-rate timebase (default 100 Hz)
        3. Cross-correlate E, N, U axes independently
        4. Weighted average of offsets by correlation strength
        5. Compute spatial offsets after time alignment

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
            - correlation: Combined correlation strength [0, 1]
            - east_offset_m: East spatial offset after time alignment (meters)
            - north_offset_m: North spatial offset after time alignment (meters)
            - up_offset_m: Up spatial offset after time alignment (meters)
            - spatial_offset_m: 3D spatial offset magnitude (meters)
            - offsets_enu: Individual axis offsets in seconds
            - correlations_enu: Individual axis correlation strengths
            Returns None if insufficient overlap or correlation too weak

        Notes
        -----
        - Positive offset means source2 is ahead in time (add positive to align)
        - Uses sub-sample precision via parabolic interpolation
        - Requires minimum 10 seconds overlap for reliable correlation

        Examples
        --------
        >>> # GPS source 2 is 2 seconds ahead of source 1
        >>> result = Synchronizer._find_gps_offset(
        ...     time1, lat1, lon1, alt1,
        ...     time2, lat2, lon2, alt2
        ... )
        >>> print(f"Time offset: {result['time_offset']:.3f} s")
        Time offset: 2.000 s
        >>> print(f"Correlation: {result['correlation']:.3f}")
        Correlation: 0.998
        """
        # Check for minimum overlap
        t1_start, t1_end = time1[0], time1[-1]
        t2_start, t2_end = time2[0], time2[-1]

        overlap_start = max(t1_start, t2_start)
        overlap_end = min(t1_end, t2_end)
        overlap_duration = overlap_end - overlap_start

        if overlap_duration < 10.0:
            logger.warning(
                f"Insufficient GPS overlap: {overlap_duration:.1f}s < 10s minimum"
            )
            return None

        # Use midpoint of first GPS as ENU reference
        mid_idx = 0
        ref_lat = float(lat1[mid_idx])
        ref_lon = float(lon1[mid_idx])
        ref_alt = float(alt1[mid_idx])

        # Convert GPS1 to ENU
        e1 = np.zeros_like(lat1)
        n1 = np.zeros_like(lat1)
        u1 = np.zeros_like(lat1)
        for i in range(len(lat1)):
            e1[i], n1[i], u1[i] = Synchronizer._lla_to_enu(
                ref_lat, ref_lon, ref_alt, lat1[i], lon1[i], alt1[i]
            )

        # Convert GPS2 to ENU
        e2 = np.zeros_like(lat2)
        n2 = np.zeros_like(lat2)
        u2 = np.zeros_like(lat2)
        for i in range(len(lat2)):
            e2[i], n2[i], u2[i] = Synchronizer._lla_to_enu(
                ref_lat, ref_lon, ref_alt, lat2[i], lon2[i], alt2[i]
            )

        # Filter GPS data

        time2, e2, n2, u2 = Synchronizer.__clean_data(time2, e2, n2, u2)

        # Create common timebase for correlation (high rate for precision)
        dt = 1.0 / target_rate_hz
        common_time = np.arange(overlap_start, overlap_end, dt)

        # Interpolate GPS1 to common timebase
        e1_interp = np.interp(common_time, time1, e1)
        n1_interp = np.interp(common_time, time1, n1)
        u1_interp = np.interp(common_time, time1, u1)

        # Interpolate GPS2 to common timebase
        e2_interp = np.interp(common_time, time2, e2)
        n2_interp = np.interp(common_time, time2, n2)
        u2_interp = np.interp(common_time, time2, u2)

        # Cross-correlate each axis independently
        corr_e = signal.correlate(e1_interp, e2_interp, mode="same")
        corr_n = signal.correlate(n1_interp, n2_interp, mode="same")
        corr_u = signal.correlate(u1_interp, u2_interp, mode="same")

        # Normalize correlations properly
        # Divide by sqrt of auto-correlations at zero lag
        norm_e = np.sqrt(np.sum(e1_interp**2) * np.sum(e2_interp**2))
        norm_n = np.sqrt(np.sum(n1_interp**2) * np.sum(n2_interp**2))
        norm_u = np.sqrt(np.sum(u1_interp**2) * np.sum(u2_interp**2))

        corr_e_norm = corr_e / norm_e if norm_e > 0 else corr_e
        corr_n_norm = corr_n / norm_n if norm_n > 0 else corr_n
        corr_u_norm = corr_u / norm_u if norm_u > 0 else corr_u

        # Find sub-sample peaks for each axis
        peak_e = Synchronizer._find_subsample_peak(np.abs(corr_e_norm))
        peak_n = Synchronizer._find_subsample_peak(np.abs(corr_n_norm))
        peak_u = Synchronizer._find_subsample_peak(np.abs(corr_u_norm))

        # Convert peak indices to time offsets
        center_idx = len(corr_e_norm) // 2
        offset_e = (peak_e - center_idx) * dt
        offset_n = (peak_n - center_idx) * dt
        offset_u = (peak_u - center_idx) * dt

        # Get correlation strengths at peaks
        corr_e_strength = abs(corr_e_norm[int(peak_e)])
        corr_n_strength = abs(corr_n_norm[int(peak_n)])
        corr_u_strength = abs(corr_u_norm[int(peak_u)])

        # Weighted average of offsets by correlation strength
        total_weight = corr_e_strength + corr_n_strength + corr_u_strength
        if total_weight < 0.1:
            logger.warning("GPS correlation too weak, cannot detect reliable offset")
            return None

        time_offset = (
            offset_e * corr_e_strength
            + offset_n * corr_n_strength
            + offset_u * corr_u_strength
        ) / total_weight

        # Combined correlation (weighted average)
        correlation = total_weight / 3.0

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

        Method:
        1. Interpolate both pitch signals to common high-rate timebase
        2. Cross-correlate pitch signals
        3. Find sub-sample peak using parabolic interpolation
        4. Return time offset

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
        - Positive offset means source2 is ahead in time
        - Uses sub-sample precision via parabolic interpolation
        - Requires minimum 10 seconds overlap for reliable correlation

        Examples
        --------
        >>> # Inclinometer is 1.5 seconds behind litchi gimbal
        >>> result = Synchronizer._find_pitch_offset(
        ...     time1_litchi, pitch1_litchi,
        ...     time2_inclinometer, pitch2_inclinometer
        ... )
        >>> print(f"Time offset: {result['time_offset']:.3f} s")
        Time offset: -1.500 s
        """
        # Check for minimum overlap
        t1_start, t1_end = time1[0], time1[-1]
        t2_start, t2_end = time2[0], time2[-1]

        overlap_start = max(t1_start, t2_start)
        overlap_end = min(t1_end, t2_end)
        overlap_duration = overlap_end - overlap_start

        if overlap_duration < 10.0:
            logger.warning(
                f"Insufficient pitch overlap: {overlap_duration:.1f}s < 10s minimum"
            )
            return None

        # Create common timebase for correlation (high rate for precision)
        dt = 1.0 / target_rate_hz
        common_time = np.arange(overlap_start, overlap_end, dt)

        # Interpolate pitch1 to common timebase
        pitch1_interp = np.interp(common_time, time1, pitch1)

        # Interpolate pitch2 to common timebase
        pitch2_interp = np.interp(common_time, time2, pitch2)

        # Cross-correlate pitch signals
        corr = signal.correlate(pitch1_interp, pitch2_interp, mode="same")

        # Normalize correlation properly
        norm = np.sqrt(np.sum(pitch1_interp**2) * np.sum(pitch2_interp**2))
        corr_norm = corr / norm if norm > 0 else corr

        # Find sub-sample peak
        peak_idx = Synchronizer._find_subsample_peak(np.abs(corr_norm))

        # Convert peak index to time offset
        center_idx = len(corr_norm) // 2
        time_offset = (peak_idx - center_idx) * dt

        # Get correlation strength at peak
        correlation = abs(corr_norm[int(peak_idx)])

        # Check minimum correlation threshold
        if correlation < 0.1:
            logger.warning("Pitch correlation too weak, cannot detect reliable offset")
            return None

        return {
            "time_offset": float(time_offset),
            "correlation": float(correlation),
        }

    def add_gps_reference(
        self,
        gps_data: pl.DataFrame,
        timestamp_col: str = "timestamp",
        lat_col: str = "posllh_lat",
        lon_col: str = "posllh_lon",
        alt_col: str = "posllh_height",
    ) -> None:
        """
        Set GPS payload as reference timebase (mandatory).

        Parameters
        ----------
        gps_data : pl.DataFrame
            Polars DataFrame with GPS data
        timestamp_col : str, default='timestamp'
            Name of timestamp column
        lat_col : str, default='latitude'
            Name of latitude column
        lon_col : str, default='longitude'
            Name of longitude column
        alt_col : str, default='altitude'
            Name of altitude column

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
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        alt_col: str = "altitude",
    ) -> None:
        """
        Add drone GPS data for correlation.

        Parameters
        ----------
        gps_data : pl.DataFrame | Dict[str, pl.DataFrame]
            Polars DataFrame with GPS data, or dict containing GPS data
                timestamp_col : str, default='timestamp'
            Name of timestamp column
        lat_col : str, default='latitude'
            Name of latitude column
        lon_col : str, default='longitude'
            Name of longitude column
        alt_col : str, default='altitude'
            Name of altitude column

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
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        alt_col: str = "altitude(m)",
    ) -> None:
        """
        Add Litchi GPS data for correlation.

        Parameters
        ----------
        gps_data : pl.DataFrame
            Polars DataFrame with GPS data
        timestamp_col : str, default='timestamp'
            Name of timestamp column
        lat_col : str, default='latitude'
            Name of latitude column
        lon_col : str, default='longitude'
            Name of longitude column
        alt_col : str, default='altitude'
            Name of altitude column

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

        self.litchi_gps = gps_data
        self.litchi_gps = self.litchi_gps.with_columns(
            pl.col(alt_col) + self.__ref_height
        )

        self.__litchi_names = {
            "timestamp": timestamp_col,
            "lat_col": lat_col,
            "lon_col": lon_col,
            "alt_col": alt_col,
            "pitch": "gimbalPitch",
        }

        logger.info(f"Added litchi GPS with {len(gps_data)} samples")

    def add_inclinometer(
        self,
        inclinometer_data: pl.DataFrame,
        inclinometer_type: str,
        timestamp_col: str = "timestamp",
        pitch_col: str = "pitch",
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
        pitch_col : str, default='pitch'
            Name of pitch column

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

        if inclinometer_type == "imx5":
            self.__inclinometer_names = {"timestamp": "timestamp", "pitch": "pitch"}

        logger.info(f"Added inclinometer with {len(inclinometer_data)} samples")

    def add_camera(
        self,
        camera_data: pl.DataFrame,
        use_photogrammetry: bool = True,
        timestamp_col: str = "timestamp",
        pitch_col: str = "pitch",
        camera_model: str | None = None,
    ):

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
                self.__camera_names = {"timestamp": "timestamp", "pitch": "pitch"}

        self.camera = camera_data

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
        interpolate_camera: bool = False,
    ) -> dict[str, Any]:
        """
        Execute correlation-based synchronization.

        Detects time offsets for all sources using correlation, then
        interpolates to GPS payload timebase at target rates.

        Parameters
        ----------
        target_rate : dict
            Target sample rates in Hz for each source
            Keys: "drone", "litchi", "inclinometer" and "payload"
            Values: float sample rate in Hz

        Returns
        -------
        dict
            Synchronized data dictionary with interpolated values for each source
            Keys: "drone", "litchi", "inclinometer" and "payload"

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
        if self.drone_gps is not None:
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
            if "gimbalPitch" in self.litchi_gps.columns:
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

        sync_data = {}

        for key in self.offsets.keys():
            if key.lower() == "drone_gps":
                sync_data["drone"] = {}

                n_samples = int((t_end - t_start) * target_rate["drone"]) + 1
                target_time = np.linspace(t_start, t_end, n_samples)

                if self.drone_gps is not None and "drone_gps" in self.offsets:
                    offset = self.offsets["drone_gps"]["time_offset"]
                    drone_time = self.drone_gps["timestamp"].to_numpy() + offset

                    for col in self.drone_gps.columns:
                        try:
                            values = self.drone_gps[col].to_numpy().astype(float)
                            interpolated = np.interp(
                                target_time,
                                drone_time,
                                values,
                                left=np.nan,
                                right=np.nan,
                            )
                            sync_data["drone"][f"{col}"] = interpolated
                        except ValueError:
                            logger.info(f"Skipped drone column: {col}")

            elif key.lower() == "litchi_gps":
                sync_data["litchi"] = {}

                n_samples = int((t_end - t_start) * target_rate["drone"]) + 1
                target_time = np.linspace(t_start, t_end, n_samples)

                if self.litchi_gps is not None and "litchi_gps" in self.offsets:
                    offset = self.offsets["litchi_gps"]["time_offset"]
                    litchi_time = self.litchi_gps["timestamp"].to_numpy() + offset

                    for col in self.litchi_gps.columns:
                        values = self.litchi_gps[col].to_numpy().astype(float)
                        interpolated = np.interp(
                            target_time,
                            litchi_time,
                            values,
                            left=np.nan,
                            right=np.nan,
                        )
                        sync_data["litchi"][f"{col}"] = interpolated

            elif key.lower() == "inclinometer":
                sync_data["inclinometer"] = {}

                n_samples = int((t_end - t_start) * target_rate["inclinometer"]) + 1
                target_time = np.linspace(t_start, t_end, n_samples)

                if self.inclinometer is not None and "inclinometer" in self.offsets:
                    # Inclinometer offset is relative to Litchi, not GPS payload
                    # Need to add both: inclinometer-to-litchi + litchi-to-gps
                    incl_offset = self.offsets["inclinometer"]["time_offset"]
                    litchi_offset = self.offsets.get("litchi_gps", {}).get(
                        "time_offset", 0.0
                    )
                    total_offset = incl_offset + litchi_offset

                    logger.info(
                        f"Applying inclinometer total offset: {total_offset:.3f}s "
                        f"(incl→litchi: {incl_offset:.3f}s + litchi→gps: {litchi_offset:.3f}s)"
                    )

                    if isinstance(self.inclinometer, dict):
                        for key in self.inclinometer.keys():
                            inclinometer_time = (
                                self.inclinometer[key]["timestamp"].to_numpy()
                                + total_offset
                            )

                            for col in self.inclinometer[key].columns:
                                values = (
                                    self.inclinometer[key][col].to_numpy().astype(float)
                                )
                                interpolated = np.interp(
                                    target_time,
                                    inclinometer_time,
                                    values,
                                    left=np.nan,
                                    right=np.nan,
                                )
                                sync_data["inclinometer"][f"{key}_{col}"] = interpolated

                    else:
                        inclinometer_time = (
                            self.inclinometer["timestamp"].to_numpy() + total_offset
                        )

                        for col in self.inclinometer.columns:
                            values = self.inclinometer[col].to_numpy().astype(float)
                            interpolated = np.interp(
                                target_time,
                                inclinometer_time,
                                values,
                                left=np.nan,
                                right=np.nan,
                            )
                            sync_data["inclinometer"][f"{col}"] = interpolated

            elif key.lower() == "camera":
                if self.camera is None:
                    logger.warning("Camera data not available, skipping camera sync")
                    continue

                sync_data["camera"] = {}

                camera_rate = np.average(np.diff(self.camera["timestamp"]))

                n_samples = int((t_end - t_start) * camera_rate) + 1
                target_time = np.linspace(t_start, t_end, n_samples)

                if "camera" in self.offsets:
                    # Camera offset is relative to Litchi, not GPS payload
                    # Need to add both: camera-to-litchi + litchi-to-gps
                    camera_offset = self.offsets["camera"]["time_offset"]
                    litchi_offset = self.offsets.get("litchi_gps", {}).get(
                        "time_offset", 0.0
                    )
                    total_offset = camera_offset + litchi_offset

                    logger.info(
                        f"Applying camera total offset: {total_offset:.3f}s "
                        f"(camera→litchi: {camera_offset:.3f}s + litchi→gps: {litchi_offset:.3f}s)"
                    )

                    camera_time = self.camera["timestamp"].to_numpy() + total_offset

                    for col in self.camera.columns:
                        if interpolate_camera:
                            values = self.camera[col].to_numpy().astype(float)
                            interpolated = np.interp(
                                target_time,
                                camera_time,
                                values,
                                left=np.nan,
                                right=np.nan,
                            )
                            sync_data["camera"][f"{col}"] = interpolated
                        else:
                            if col == "timestamp":
                                sync_data["camera"][f"{col}"] = camera_time
                            else:
                                sync_data["camera"][f"{col}"] = self.camera[col]

        if "camera" not in sync_data and self.camera is not None:
            logger.info("Camera offset is applied ")

        if self.other_payload:
            sync_data["payload"] = {}

            n_samples = int((t_end - t_start) * target_rate["payload"]) + 1
            target_time = np.linspace(t_start, t_end, n_samples)

            for sensor_name, sensor_df in self.other_payload.items():
                if "timestamp" in sensor_df.columns:
                    sensor_time = sensor_df["timestamp"].to_numpy().copy()

                    incl_offset = self.offsets.get("inclinometer", {}).get(
                        "time_offset", 0.0
                    )
                    litchi_offset = self.offsets.get("litchi_gps", {}).get(
                        "time_offset", 0.0
                    )
                    total_offset = incl_offset + litchi_offset

                    sensor_time += total_offset

                    for col in sensor_df.columns:
                        values = sensor_df[col].to_numpy().astype(float)
                        interpolated = np.interp(
                            target_time,
                            sensor_time,
                            values,
                            left=np.nan,
                            right=np.nan,
                        )
                        sync_data["payload"][f"{sensor_name}_{col}"] = interpolated

        sync_data["reference_gps"] = self.gps_payload

        self.synchronized_data = {
            key: pl.DataFrame(value) for key, value in sync_data.items()
        }

        logger.info(f"({t_end - t_start:.2f}s duration)")

        return self.synchronized_data

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

            lines.append(f"  Correlation: {offset_data['correlation']:.3f}")

            if "spatial_offset_m" in offset_data:
                lines.append(
                    f"  Spatial Offset: {offset_data['spatial_offset_m']:.2f} m"
                )
                lines.append(f"    East: {offset_data['east_offset_m']:.2f} m")
                lines.append(f"    North: {offset_data['north_offset_m']:.2f} m")
                lines.append(f"    Up: {offset_data['up_offset_m']:.2f} m")

        return "\n".join(lines)
