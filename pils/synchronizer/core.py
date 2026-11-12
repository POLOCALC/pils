"""Synchronizer: the time offsets of every source and the synchronized tables."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl

from pils.synchronizer.offsets import OffsetEstimationMixin
from pils.synchronizer.resample import resample_table
from pils.synchronizer.sources import SourcesMixin

logger = logging.getLogger(__name__)


class Synchronizer(SourcesMixin, OffsetEstimationMixin):
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
                time1=self.gps_payload[self._ref_names["timestamp"]].to_numpy(),
                lat1=self.gps_payload[self._ref_names["lat_col"]].to_numpy(),
                lon1=self.gps_payload[self._ref_names["lon_col"]].to_numpy(),
                alt1=self.gps_payload[self._ref_names["alt_col"]].to_numpy(),
                time2=self.drone_gps[self._drone_names["timestamp"]].to_numpy(),
                lat2=self.drone_gps[self._drone_names["lat_col"]].to_numpy(),
                lon2=self.drone_gps[self._drone_names["lon_col"]].to_numpy(),
                alt2=self.drone_gps[self._drone_names["alt_col"]].to_numpy(),
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
                time1=self.gps_payload[self._ref_names["timestamp"]].to_numpy(),
                lat1=self.gps_payload[self._ref_names["lat_col"]].to_numpy(),
                lon1=self.gps_payload[self._ref_names["lon_col"]].to_numpy(),
                alt1=self.gps_payload[self._ref_names["alt_col"]].to_numpy(),
                time2=self.litchi_gps[self._litchi_names["timestamp"]].to_numpy(),
                lat2=self.litchi_gps[self._litchi_names["lat_col"]].to_numpy(),
                lon2=self.litchi_gps[self._litchi_names["lon_col"]].to_numpy(),
                alt2=self.litchi_gps[self._litchi_names["alt_col"]].to_numpy(),
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
            if self._litchi_names["pitch"] in self.litchi_gps.columns:
                logger.info("Detecting inclinometer offset via pitch correlation...")
                result = self._find_pitch_offset(
                    time1=self.litchi_gps[self._litchi_names["timestamp"]].to_numpy(),
                    pitch1=self.litchi_gps[self._litchi_names["pitch"]].to_numpy(),
                    time2=self.inclinometer[
                        self._inclinometer_names["timestamp"]
                    ].to_numpy(),
                    pitch2=self.inclinometer[
                        self._inclinometer_names["pitch"]
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
            if self._camera_model == "photogrammetry" or self._camera_model == "sony":
                logger.info(
                    f"Camera Model {self._camera_model}, this implies use of pitch correlation"
                )

                if self.litchi_gps is not None:
                    result = self._find_pitch_offset(
                        time1=self.litchi_gps[
                            self._litchi_names["timestamp"]
                        ].to_numpy(),
                        pitch1=self.litchi_gps[self._litchi_names["pitch"]].to_numpy(),
                        time2=self.camera[self._camera_names["timestamp"]].to_numpy(),
                        pitch2=self.camera[self._camera_names["pitch"]].to_numpy(),
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
                if self._camera_model == "alvium":
                    incl_offset = self.offsets.get("inclinometer", {}).get(
                        "time_offset", 0.0
                    )

                    self.offsets["camera"] = {"time_offset": incl_offset}

                else:
                    logger.info(
                        f"Camera Model {self._camera_model}, skipping pitch correlation"
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
        if self.inclinometer is not None and "inclinometer" not in self.offsets:
            # Its own clock (Kernel-100) and no Litchi pitch offset
            logger.warning(
                "Inclinometer NOT synchronized: it has its own clock and no "
                "offset to the Litchi pitch ("
                + ("no Litchi log" if self.litchi_gps is None else "no pitch match")
                + "); it is left out of the synchronized data"
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
            # Exactly 1/rate apart, from t_start to at most t_end
            n = int(np.floor((t_end - t_start) * rate + 1e-9)) + 1
            return t_start + np.arange(n) / rate

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
                self._drone_names["timestamp"],
                self.offsets["drone_gps"]["time_offset"],
                rates["drone"],
            )

        if self.litchi_gps is not None and "litchi_gps" in self.offsets:
            synced["litchi"] = align(
                self.litchi_gps,
                self._litchi_names["timestamp"],
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
            time_col = self._inclinometer_names["timestamp"]
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
            camera_time = self.camera[self._camera_names["timestamp"]].to_numpy()
            # From the median spacing (repeated timestamps left out)
            steps = np.diff(np.sort(camera_time.astype(float)))
            steps = steps[steps > 0]
            camera_rate = float(1.0 / np.median(steps)) if steps.size else 1.0
            rates["camera"] = camera_rate
            synced["camera"] = align(
                self.camera, self._camera_names["timestamp"], total_offset, camera_rate
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
                self._ref_names["timestamp"],
                0.0,
                rates["reference_gps"],
            )

        # Standardize coordinate column names for GPS sources
        for key, names in (
            ("drone", getattr(self, "_drone_names", None)),
            ("litchi", getattr(self, "_litchi_names", None)),
            ("reference_gps", self._ref_names),
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
            time_col = self._inclinometer_names["timestamp"]
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
                    allow_anticorrelation=True,
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
