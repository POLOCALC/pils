"""Adding the sources to synchronize (GPS, inclinometer, camera, ...)."""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)


class SourcesMixin:
    """The ``add_*`` methods of :class:`~pils.synchronizer.Synchronizer`.

    Each stores a source and the names of its columns, which
    :meth:`~pils.synchronizer.Synchronizer.synchronize` reads.
    """

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

        self._ref_height = gps_data[alt_col][0]
        self._ref_names = {
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

        self._drone_names = {
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
            (pl.col(alt_col) + self._ref_height).alias("h_ell_m")
        )

        self._litchi_names = {
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
        self._inclinometer_names = {"timestamp": timestamp_col, "pitch": pitch_col}
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

            self._camera_model = "photogrammetry"

            self._camera_names = {"timestamp": timestamp_col, "pitch": pitch_col}
        else:
            self._camera_model = camera_model
            # The timestamp for every model (alvium: its clock), the pitch
            # for the sony
            self._camera_names = {"timestamp": timestamp_col, "pitch": pitch_col}

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
