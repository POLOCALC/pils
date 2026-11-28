"""Synchronization of the tables of a flight (``Flight.sync``)."""

import logging
from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl

from pils import schema
from pils.drones.sources import record_positions, record_track, telemetry_positions
from pils.synchronizer import Synchronizer

logger = logging.getLogger(__name__)


MIN_REFERENCE_COVERAGE = 0.5
"""Minimum fraction of the drone GPS time covered by the payload GPS.

The payload GPS is the sync reference when it covers at least this fraction;
otherwise (missing, stopped early, another session) the drone GPS is.
"""


MAX_GPS_H_ACC_M = 50.0
"""Largest horizontal accuracy (u-blox ``h_acc_m``) of a payload GPS epoch used to sync.

Epochs without a real fix carry positions thousands of km off, with an
accuracy estimate to match (April 2024: 4000 km, against 5 m with a fix).
"""


def _valid_positions(
    frame: pl.DataFrame, lat_col: str, lon_col: str, acc_col: str | None = None
) -> pl.DataFrame:
    """Keep rows with a position: not null, not at 0, 0, accurate enough.

    ``acc_col`` (if in the table) must be at most :data:`MAX_GPS_H_ACC_M`.
    """
    lat, lon = pl.col(lat_col), pl.col(lon_col)
    keep = lat.is_not_null() & lon.is_not_null() & ~((lat == 0) & (lon == 0))
    if acc_col is not None and acc_col in frame.columns:
        keep = keep & (pl.col(acc_col) <= MAX_GPS_H_ACC_M)
    return frame.filter(keep)


def _time_coverage(times: pl.Series, reference: pl.Series) -> float:
    """Fraction of the span of ``reference`` covered by the span of ``times``."""
    ref = reference.drop_nulls().to_numpy().astype(float)
    own = times.drop_nulls().to_numpy().astype(float)
    if ref.size == 0 or own.size == 0:
        return 0.0
    if ref.size < 2 or ref.max() <= ref.min():
        return 1.0
    overlap = min(ref.max(), own.max()) - max(ref.min(), own.min())
    return float(max(overlap, 0.0) / (ref.max() - ref.min()))


class FlightSyncMixin:
    """Synchronization of the drone and payload tables of a :class:`~pils.flight.Flight`."""

    def sync(
        self,
        target_rate: dict[str, float] | None = None,
        use_rtk_data: bool = True,
        common_time: bool = True,
        drone_records: Sequence[str] | Literal["all"] = ("imu_atti_0",),
        **kwargs,
    ) -> dict[str, pl.DataFrame]:
        """
        Synchronize flight data using GPS-based correlation.

        Creates a Synchronizer instance, adds available data sources,
        performs synchronization, and stores the result in sync_data attribute.

        Parameters
        ----------
        target_rate : dict, default=None
            Target sample rate in Hz of the different sensors; if None the following
            rates are applied:
            - 10 Hz for drone and litchi
            - 100 Hz for payload sensors (including inclinometer and ADC)
        use_rtk_data : bool, default=True
            For DJI drones: if True, use RTK data; if False, use standard GPS
        common_time: bool
            Interpolate all the data at a common time, with a sampliing frequency
            determined by the target_rate. If False, the time is just shifted and the
            other columns are not touched
        drone_records : sequence of str or "all", default=("imu_atti_0",)
            Other drone records (DJI .dat, see ``raw_data.drone_data.records``)
            to synchronize, each as ``drone_<name>``; they take the drone GPS
            offset. The default is the flight controller attitude (200 Hz).
            DJI names such as ``"IMU_ATTI(0)"`` are accepted too.
        **kwargs : dict
            Additional arguments passed to Synchronizer.synchronize()

        Returns
        -------
        pl.DataFrame
            Synchronized data

        Raises
        ------
        ValueError
            If no GPS payload data available (required as reference)

        Examples
        --------
        >>> # Basic synchronization with RTK data
        >>> flight.add_sensor_data(['gps', 'imu', 'adc'])
        >>> flight.add_drone_data()
        >>> sync_df = flight.sync(target_rate={'drone': 10.0, 'payload': 100.0}, use_rtk_data=True)
        >>> # Use standard GPS instead of RTK
        >>> sync_df = flight.sync(target_rate={'drone': 10.0}, use_rtk_data=False)
        >>> # Synchronization is stored in flight.sync_data as a dict of DataFrames
        >>> print(list(flight.sync_data.keys()))
        """
        # The payload GPS is the reference timebase; without it (not
        # recorded, or empty) the drone GPS is
        gps_data = None
        if self.raw_data.payload_data and "gps" in self.raw_data.payload_data:
            gps_sensor = self.raw_data.payload_data["gps"]
            gps_data = gps_sensor.data if hasattr(gps_sensor, "data") else gps_sensor
            required = {"timestamp", "lat_deg", "lon_deg", "h_ell_m"}
            if not isinstance(gps_data, pl.DataFrame) or gps_data.is_empty():
                logger.warning(
                    "Payload GPS has no data: the drone GPS is the reference"
                )
                gps_data = None
            elif not required <= set(gps_data.columns):
                logger.warning(
                    "Payload GPS has no position "
                    f"({sorted(required - set(gps_data.columns))} missing): "
                    "the drone GPS is the reference"
                )
                gps_data = None
            else:
                gps_data = _valid_positions(gps_data, "lat_deg", "lon_deg", "h_acc_m")
                if gps_data.is_empty():
                    logger.warning(
                        "Payload GPS has no position fix: the drone GPS is the reference"
                    )
                    gps_data = None

        # Create synchronizer
        sync = Synchronizer()
        position_columns: dict[str, str] | None = None

        # Add drone GPS if available
        drone_data = self.raw_data.drone_data.drone
        drone_has_data = (isinstance(drone_data, dict) and len(drone_data) > 0) or (
            isinstance(drone_data, pl.DataFrame) and len(drone_data) > 0
        )
        # A drone without a .DAT (the Matrice 400): the drone GPS is LAGER's
        # telemetry, if it has the RTK or GNSS receiver's positions
        telemetry = self.raw_data.drone_data.telemetry
        telemetry_is_drone = False
        if not drone_has_data and telemetry is not None:
            usable = telemetry_positions(telemetry, receivers_only=True)
            if usable is not None:
                drone_data, drone_has_data, telemetry_is_drone = telemetry, True, True
        # Neither: the Matrice 400's flight record, if it has the RTK
        # solution: its 10 Hz track (on the drone's GPS time) tied to the RTK
        # (1 Hz, late by its latency) is the drone GPS; the whole record is
        # then a table on the same clock
        drone_sources = self.raw_data.drone_data.sources
        flight_record = drone_sources.get("flight_record")
        record_is_drone = False
        self.sync_record_track: dict[str, float] | None = None
        if not drone_has_data and use_rtk_data and flight_record is not None:
            tied = record_track(flight_record)
            if tied is not None:
                drone_data, self.sync_record_track = tied
                drone_has_data, record_is_drone = True, True
                logger.info(
                    "No .DAT and no telemetry: the drone GPS is the flight "
                    "record's 10 Hz track tied to its RTK (%s)",
                    ", ".join(
                        f"{k}={v:.3f}" for k, v in self.sync_record_track.items()
                    ),
                )
        if gps_data is None and not drone_has_data:
            raise ValueError(
                "No reference timebase: no payload GPS data and no drone GPS. "
                "Call flight.add_sensor_data(['gps']) or flight.add_drone_data() first."
            )

        # A copy: the defaults below must not change the caller's dict
        target_rate = dict(target_rate or {})

        if drone_has_data:
            # Ensure target_rate dict has drone key with default 10 Hz
            if "drone" not in target_rate:
                target_rate["drone"] = 10.0
            drone_df = drone_data
            if isinstance(drone_df, dict):  # ArduPilot messages
                drone_df = drone_df.get("gps", next(iter(drone_df.values())))

            model = (
                "dji"
                if telemetry_is_drone or record_is_drone
                else (self._drone_model or "").lower()
            )
            records_gps = (self.raw_data.drone_data.records or {}).get("gps")
            rtk_df = None
            if record_is_drone:
                lat_col, lon_col = "rtk_track_lat_deg", "rtk_track_lon_deg"
                alt_col = "rtk_track_h_ell_m"
            elif "dji" in model and use_rtk_data and "rtk_lat_deg" in drone_df.columns:
                # Positions only where the RTK receiver has a solution (rows
                # without one are at 0, 0, 0)
                rtk_df = _valid_positions(drone_df, "rtk_lat_deg", "rtk_lon_deg")
                if rtk_df.is_empty():
                    logger.warning(
                        "Drone RTK has no solution: the drone GNSS is used instead"
                    )
                    rtk_df = None
            if record_is_drone:
                pass  # the track's columns, above
            elif rtk_df is not None:
                drone_df = rtk_df
                lat_col, lon_col, alt_col = "rtk_lat_deg", "rtk_lon_deg", "rtk_h_ell_m"
            else:
                # The main receiver: in the drone table (DatCon CSV,
                # BlackSquare) or, for DJI .dat files, its own record
                if "gps_lat_deg" in drone_df.columns:
                    prefix = "gps_"
                elif "dji" in model and records_gps is not None:
                    drone_df, prefix = records_gps, ""
                else:
                    prefix = ""
                lat_col, lon_col = f"{prefix}lat_deg", f"{prefix}lon_deg"
                alt_col = f"{prefix}h_ell_m"
                if alt_col not in drone_df.columns:
                    # Heights above mean sea level: to the ellipsoid with the
                    # geoid separation of the payload receiver (without it,
                    # heights stay above mean sea level)
                    separation = (
                        (gps_data["h_ell_m"] - gps_data["h_msl_m"]).median()
                        if gps_data is not None and "h_msl_m" in gps_data.columns
                        else 0.0
                    )
                    if gps_data is None:
                        logger.warning(
                            "No payload GPS: drone heights are above mean sea "
                            "level, not the ellipsoid"
                        )
                    drone_df = drone_df.with_columns(
                        (pl.col(f"{prefix}h_msl_m") + separation).alias(alt_col)
                    )
                # Positions only where the receiver has a solution
                drone_df = _valid_positions(drone_df, lat_col, lon_col)
            position_columns = {"lat": lat_col, "lon": lon_col, "alt": alt_col}
            if gps_data is not None:
                coverage = _time_coverage(gps_data["timestamp"], drone_df["timestamp"])
                if coverage < MIN_REFERENCE_COVERAGE:
                    logger.warning(
                        f"Payload GPS covers {100 * coverage:.0f} % of the drone "
                        "GPS time: the drone GPS is the reference"
                    )
                    gps_data = None
            if gps_data is None:
                sync.add_gps_reference(
                    drone_df,
                    timestamp_col="timestamp",
                    lat_col=lat_col,
                    lon_col=lon_col,
                    alt_col=alt_col,
                )
                sync.reference = "drone_gps"

            sync.add_drone_gps(
                drone_df,
                timestamp_col="timestamp",
                lat_col=lat_col,
                lon_col=lon_col,
                alt_col=alt_col,
            )

        if gps_data is not None:
            sync.add_gps_reference(
                gps_data,
                timestamp_col="timestamp",
                lat_col="lat_deg",
                lon_col="lon_deg",
                alt_col="h_ell_m",
            )

        # Add litchi GPS if available
        litchi_df = self.raw_data.drone_data.litchi
        if (
            isinstance(litchi_df, pl.DataFrame)
            and litchi_df.height > 0
            and {"lat_deg", "lon_deg"} <= set(litchi_df.columns)
        ):
            if "drone" not in target_rate:
                target_rate["drone"] = 10.0
            sync.add_litchi_gps(litchi_df)

        # Other sources of the drone's position: each its own offset to the
        # reference GPS (LAGER's telemetry when the .DAT is the drone GPS, the
        # Matrice 400's flight record); the gimbal log is on the telemetry's clock.
        # The flight record as the drone GPS (its RTK, 1 Hz): the whole record
        # is on the same clock
        for name, table in (
            ("telemetry", None if telemetry_is_drone else telemetry),
            ("flight_record", None if record_is_drone else flight_record),
        ):
            if table is None:
                continue
            if name == "flight_record":
                positions = record_positions(table)
            else:
                positions = telemetry_positions(table)
            if positions is None:
                continue
            frame, cols = positions
            sync.add_position_source(name, frame, cols["lat"], cols["lon"], cols["alt"])
        if record_is_drone and flight_record is not None:
            sync.add_attached_table("flight_record", flight_record, to="drone_gps")
        if telemetry is not None:
            clock = "drone_gps" if telemetry_is_drone else "telemetry_gps"
            for message, table in self.raw_data.drone_data.gimbal.items():
                sync.add_attached_table(f"gimbal_{message}", table, to=clock)

        # Add other drone records (DJI .dat attitude, motors, ...)
        records = getattr(self.raw_data.drone_data, "records", None) or {}
        if drone_has_data and records:
            if drone_records != "all":
                wanted = {schema.record_name(name) for name in drone_records}
                missing = sorted(wanted - set(records))
                if missing:
                    logger.warning(f"Drone records not found: {missing}")
                records = {k: v for k, v in records.items() if k in wanted}
            sync.add_drone_records(records)

        # Add inclinometer if available
        if "inclinometer" in self.raw_data.payload_data:
            if "inclinometer" not in target_rate:
                target_rate["inclinometer"] = 100.0

            incl_sensor = self.raw_data.payload_data["inclinometer"]
            incl_data = (
                incl_sensor.data if hasattr(incl_sensor, "data") else incl_sensor
            )
            if isinstance(incl_data, dict):  # IMX-5: the INS table
                incl_data = incl_data.get("ins")
            # The Kernel-100 has its own clock (log start + counter); the
            # IMX-5 and the 2022 payload are stamped with the payload
            # computer's clock
            if isinstance(incl_data, pl.DataFrame) and not incl_data.is_empty():
                sync.add_inclinometer(
                    incl_data,
                    clock="own" if self._inclinometer == "kernel" else "payload",
                )
            else:
                logger.warning("Inclinometer has no data: not synchronized")

        # Add Camera data if available

        if "camera" in self.raw_data.payload_data:
            sync.add_camera(
                self.raw_data.payload_data["camera"],
                use_photogrammetry=self._use_photogrammetry,
                camera_model=self._camera_model,
            )

        # Add other payload sensors
        payload = self.raw_data.payload_data

        # Add ADC if available
        if "adc" in payload:
            if "payload" not in target_rate:
                target_rate["payload"] = 100.0
            adc_sensor = payload["adc"]
            adc_data = adc_sensor.data if hasattr(adc_sensor, "data") else adc_sensor
            if isinstance(adc_data, pl.DataFrame) and not adc_data.is_empty():
                sync.add_payload_sensor("adc", adc_data)
            else:
                logger.warning("ADC has no data: not synchronized")

        # Add LM76 if available
        if "lm76" in payload:
            if "payload" not in target_rate:
                target_rate["payload"] = 100.0
            lm76_sensor = payload["lm76"]
            lm76_data = (
                lm76_sensor.data if hasattr(lm76_sensor, "data") else lm76_sensor
            )
            if isinstance(lm76_data, pl.DataFrame) and not lm76_data.is_empty():
                sync.add_payload_sensor("lm76", lm76_data)
            else:
                logger.warning("LM76 has no data: not synchronized")

        # Add IMU sensors if available ({"barometer": df, ...})
        if "imu" in payload and payload["imu"] is not None:
            if "payload" not in target_rate:
                target_rate["payload"] = 100.0
            imu_data = payload["imu"]
            if not isinstance(imu_data, dict):  # an IMU object
                imu_data = imu_data.data or {}
            for name, table in imu_data.items():
                if isinstance(table, pl.DataFrame) and not table.is_empty():
                    sync.add_payload_sensor(f"imu_{name}", table)

        # Perform synchronization
        self.sync_data = sync.synchronize(
            target_rate=target_rate,
            common_time=common_time,
            **kwargs,
        )
        # Detected time offsets in seconds, as in Synchronizer.offsets: the
        # inclinometer and camera offsets are relative to Litchi
        self.sync_offsets = {
            name: float(offset["time_offset"])
            for name, offset in sync.offsets.items()
            if "time_offset" in offset
        }
        self.sync_offset_details = {
            name: {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in offset.items()
            }
            for name, offset in sync.offsets.items()
        }
        self.sync_reference = sync.reference
        # Rates (Hz) of the synchronized tables, as used
        self.sync_rates = dict(sync.rates)
        self.sync_position_columns = (
            {"drone": position_columns} if position_columns is not None else {}
        )

        return self.sync_data
