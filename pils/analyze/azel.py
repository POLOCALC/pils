"""AZEL (Azimuth-Elevation) analysis module for flight data."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import polars as pl
import pymap3d as pm

from pils.flight import Flight
from pils.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AZELVersion:
    """AZEL analysis version container.

    Args:
        version_name: Name identifier for this AZEL analysis version
        azel_data: Polars DataFrame with columns: timestamp, az, el, srange
        metadata: Dictionary containing observer position and other metadata

    Example:
        >>> azel_data = pl.DataFrame({
        ...     'timestamp': [1000.0, 2000.0],
        ...     'az': [45.0, 90.0],
        ...     'el': [30.0, 45.0],
        ...     'srange': [100.5, 150.3]
        ... })
        >>> metadata = {'observer_lat': 40.7128, 'observer_lon': -74.0060}
        >>> version = AZELVersion('v1', azel_data, metadata)
    """

    version_name: str
    azel_data: pl.DataFrame
    metadata: dict[str, Any]


class AZELAnalysis:
    """AZEL (Azimuth-Elevation) analysis for drone telescope tracking.

    Manages azimuth-elevation calculations for telescope tracking of drone
    flights. Completely separate from Flight class.

    File Structure
    --------------
    flight_dir/proc/azel/
    └── (AZEL analysis outputs will be stored here)

    Attributes
    ----------
    flight : Flight
        Flight object containing flight data
    flight_path : Path
        Root flight directory
    azel_dir : Path
        AZEL directory ({flight_path}/proc/azel/)

    Examples
    --------
    >>> from pils.flight import Flight
    >>> # Create Flight object
    >>> flight_info = {
    ...     "drone_data_folder_path": "/path/to/flight/drone",
    ... }
    >>> flight = Flight(flight_info)
    >>> # Create new analysis
    >>> azel = AZELAnalysis(flight)
    """

    def __init__(self, flight: Flight) -> None:
        """Initialize AZEL analysis with Flight object.

        Creates the proc/azel directory structure if it doesn't exist.

        Args:
            flight: Flight object with valid flight_path attribute

        Raises:
            TypeError: If flight is not a Flight object
            ValueError: If flight_path is None or doesn't exist

        Examples:
            >>> from pils.flight import Flight
            >>> flight_info = {
            ...     "drone_data_folder_path": "/path/to/flight/drone",
            ... }
            >>> flight = Flight(flight_info)
            >>> azel = AZELAnalysis(flight)
            >>> print(azel.azel_dir)
            /path/to/flight/proc/azel
        """
        # Validate flight object type
        if not isinstance(flight, Flight):
            raise TypeError(
                f"Expected Flight object, got {type(flight).__name__}. "
                "AZELAnalysis requires a Flight object instead of a path."
            )

        # Validate flight_path attribute exists
        if flight.flight_path is None:
            raise ValueError(
                "Flight object must have a valid flight_path attribute. "
                "Ensure the Flight was initialized with proper flight_info."
            )

        # Convert to Path and validate it's a directory
        flight_path = Path(flight.flight_path)
        if not flight_path.exists() or not flight_path.is_dir():
            raise ValueError(
                f"flight_path must be an existing directory. Got: {flight_path}"
            )

        # Store attributes
        self.flight = flight
        self.flight_path = flight_path

        # Create AZEL directory
        self.azel_dir = self.flight_path / "proc" / "azel"
        self.azel_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized AZEL analysis for flight: {self.flight_path}")
        logger.info(f"AZEL directory: {self.azel_dir}")

    @staticmethod
    def _compute_enu_positions(
        ref_lat: float,
        ref_lon: float,
        ref_alt: float,
        target_lat: float,
        target_lon: float,
        target_alt: float,
    ) -> tuple[float, float, float]:
        """Convert geodetic coordinates to local ENU (East-North-Up).

        Args:
            ref_lat: Reference latitude in degrees (WGS84)
            ref_lon: Reference longitude in degrees (WGS84)
            ref_alt: Reference altitude in meters (ellipsoidal height)
            target_lat: Target latitude in degrees (WGS84)
            target_lon: Target longitude in degrees (WGS84)
            target_alt: Target altitude in meters (ellipsoidal height)

        Returns:
            Tuple of (east, north, up) in meters

        Example:
            >>> e, n, u = AZELAnalysis._compute_enu_positions(
            ...     40.0, -105.0, 1000.0,
            ...     40.001, -105.0, 1000.0
            ... )
            >>> # Drone is ~111m north of reference
        """
        east, north, up = pm.geodetic2enu(
            target_lat, target_lon, target_alt, ref_lat, ref_lon, ref_alt
        )
        return east, north, up

    @staticmethod
    def _compute_azel(
        east: float, north: float, up: float
    ) -> tuple[float, float, float]:
        """Convert ENU coordinates to azimuth, elevation, slant range.

        Args:
            east: East position in meters
            north: North position in meters
            up: Up position in meters

        Returns:
            Tuple of (azimuth, elevation, slant_range)
            - azimuth: Angle in degrees (0° = North, 90° = East)
            - elevation: Angle in degrees above horizon (0° = horizon, 90° = zenith)
            - slant_range: Distance in meters

        Example:
            >>> az, el, sr = AZELAnalysis._compute_azel(0, 100, 0)
            >>> # Drone is 100m north: az≈0°, el≈0°, range≈100m
        """
        azimuth, elevation, slant_range = pm.enu2aer(east, north, up)
        return azimuth, elevation, slant_range

    def _load_emlid_reference_data(self,
        telescope_name: str, emlid_csv_path: str | Path = None,
    ) -> dict[str, dict[str, float]]:
        """Load telescope and DJI base positions from EMLID reference CSV.

        Args:
            emlid_csv_path: Path to EMLID CSV file with columns: Name, Longitude, Latitude, Ellipsoidal height
            telescope_name: Telescope identifier (e.g., 'SATP1') for filtering rows

        Returns:
            Dictionary with 'telescope' and 'base' keys, each containing:
            {'lat': float, 'lon': float, 'alt': float} in WGS84 degrees and meters

        Raises:
            FileNotFoundError: If EMLID CSV file not found
            ValueError: If telescope or base positions not found in CSV

        Example:
            >>> ref_data = AZELAnalysis._load_emlid_reference_data(
            ...     'emlid_ref.csv', 'SATP1'
            ... )
            >>> ref_data['telescope']['lat']  # Mean latitude of SATP1 positions
            40.0001
        """

        if not emlid_csv_path:
            emlid_path = self.flight_path.parents[1]

            emlid_path = emlid_path / "coordinates" / "202511_coordinates.csv"

        else:
            emlid_path = Path(emlid_csv_path)
            if not emlid_path.exists():
                raise FileNotFoundError(f"EMLID CSV file not found: {emlid_path}")

        # Load EMLID data with Polars
        df = pl.read_csv(
            emlid_path, columns=["Name", "Longitude", "Latitude", "Ellipsoidal height"]
        )

        # Filter and compute mean for telescope positions
        telescope_df = df.filter(pl.col("Name").str.starts_with(telescope_name.upper()))
        if telescope_df.height == 0:
            raise ValueError(
                f"No telescope positions found for '{telescope_name}' in EMLID CSV"
            )

        telescope_mean = telescope_df.select(
            [
                pl.col("Latitude").mean().alias("lat"),
                pl.col("Longitude").mean().alias("lon"),
                pl.col("Ellipsoidal height").mean().alias("alt"),
            ]
        ).row(0, named=True)

        # Filter and compute mean for DJI base positions
        base_df = df.filter(pl.col("Name").str.starts_with("DJI"))
        if base_df.height == 0:
            raise ValueError("No DJI base positions found in EMLID CSV")

        base_mean = base_df.select(
            [
                pl.col("Latitude").mean().alias("lat"),
                pl.col("Longitude").mean().alias("lon"),
                pl.col("Ellipsoidal height").mean().alias("alt"),
            ]
        ).row(0, named=True)

        return {"telescope": telescope_mean, "base": base_mean}

    @staticmethod
    def _compute_rtk_correction(
        dji_base_geod: dict[str, float], dji_broadcast_geod: dict[str, float]
    ) -> tuple[float, float, float]:
        """Compute RTK correction offset between actual and broadcast base positions.

        Args:
            dji_base_geod: Actual DJI base position {'lat': float, 'lon': float, 'alt': float}
            dji_broadcast_geod: Broadcast DJI base position {'lat': float, 'lon': float, 'alt': float}

        Returns:
            Tuple of (delta_east, delta_north, delta_up) offset in meters to subtract from drone ENU

        Example:
            >>> actual = {'lat': 40.0, 'lon': -105.0, 'alt': 1000.0}
            >>> broadcast = {'lat': 40.001, 'lon': -105.0, 'alt': 1000.0}
            >>> de, dn, du = AZELAnalysis._compute_rtk_correction(actual, broadcast)
            >>> # Offset is ~111m south (broadcast is north of actual)
        """
        delta_e, delta_n, delta_u = pm.geodetic2enu(
            dji_base_geod["lat"],
            dji_base_geod["lon"],
            dji_base_geod["alt"],
            dji_broadcast_geod["lat"],
            dji_broadcast_geod["lon"],
            dji_broadcast_geod["alt"],
        )
        return delta_e, delta_n, delta_u

    def run_analysis(
        self,
        telescope_name: str,
        dji_broadcast_geod: dict[str, float],
        drone_timezone_hours: float = 0.0,
        emlid_csv_path: str | Path = None,
    ) -> AZELVersion | None:
        """Run AZEL analysis for drone telescope tracking.

        Loads drone RTK data, applies corrections, and computes azimuth-elevation
        angles for telescope tracking.

        Args:
            emlid_csv_path: Path to EMLID reference CSV with telescope and base positions
            telescope_name: Telescope identifier (e.g., 'SATP1')
            dji_broadcast_geod: Broadcast DJI base position {'lat': float, 'lon': float, 'alt': float}
            drone_timezone_hours: Timezone offset for drone timestamps (default: 0.0 UTC)

        Returns:
            AZELVersion with computed azimuth, elevation, slant range data, or None if no valid data

        Raises:
            ValueError: If drone data not loaded or EMLID data invalid
            FileNotFoundError: If EMLID CSV file not found

        Example:
            >>> azel = AZELAnalysis(flight)
            >>> dji_broadcast = {'lat': -22.9597732, 'lon': -67.7866847, 'alt': 5173.020}
            >>> version = azel.run_analysis('emlid_ref.csv', 'SATP1', dji_broadcast)
            >>> print(version.azel_data)
        """
        import datetime

        logger.info("Starting AZEL analysis...")

        # 1. Determine data source (sync_data preferred, fallback to raw_data)
        drone_df = None
        litchi_df = None
        data_source = None

        # Check if sync_data is available
        if self.flight.sync_data is not None and "drone" in self.flight.sync_data:
            drone_df = self.flight.sync_data["drone"]
            data_source = "sync_data"
            logger.info("Using synchronized drone data from flight.sync_data")

            # Check for litchi in sync_data
            if "litchi" in self.flight.sync_data:
                litchi_df = self.flight.sync_data["litchi"]
                logger.info("Litchi data available in sync_data")
        else:
            # Fallback to raw_data
            if self.flight.raw_data.drone_data is None:
                raise ValueError(
                    "No drone data available. Call flight.add_drone_data() or flight.sync() first."
                )

            drone_df = self.flight.raw_data.drone_data.drone
            data_source = "raw_data"
            logger.info("Using raw drone data from flight.raw_data")

            # Check for litchi in raw_data
            if (
                hasattr(self.flight.raw_data.drone_data, "litchi")
                and self.flight.raw_data.drone_data.litchi is not None
            ):
                litchi_df = self.flight.raw_data.drone_data.litchi
                if isinstance(litchi_df, pl.DataFrame) and litchi_df.height > 0:
                    logger.info("Litchi data available in raw_data")

        # Validate we have data
        if drone_df is None or (
            isinstance(drone_df, pl.DataFrame) and drone_df.height == 0
        ):
            raise ValueError("Drone data is empty or unavailable")

        # Ensure drone_df is a DataFrame (not dict)
        if not isinstance(drone_df, pl.DataFrame):
            raise ValueError(
                "Expected drone data to be a DataFrame, got dict. "
                "This suggests DAT format - ensure data is aligned."
            )

        # 2. Detect drone format from flight's drone_model attribute
        drone_model = getattr(self.flight, "_Flight__drone_model", "dji").lower()
        logger.info(f"Detected drone model: {drone_model}")

        # Determine format type
        if "dji" in drone_model:
            format_type = "dji"
        elif "black" in drone_model or "blacksquare" in drone_model:
            format_type = "blacksquare"
        elif "litchi" in drone_model:
            format_type = "litchi"
        else:
            # Default to DJI for unknown models
            format_type = "dji"
            logger.warning(
                f"Unknown drone model '{drone_model}', defaulting to DJI format"
            )

        # Check if litchi should be used instead (placeholder for Phase 2)
        if (
            litchi_df is not None
            and isinstance(litchi_df, pl.DataFrame)
            and litchi_df.height > 0
        ):
            # Phase 2 will implement litchi-specific processing
            pass

        logger.info(f"Using drone format: {format_type}, data source: {data_source}")
        logger.info(f"Loaded drone data with {drone_df.height} rows")

        # 3. Set column names based on format and data source
        if data_source == "sync_data":
            # sync_data has standardized column names based on format
            if format_type == "dji":
                lat_col = "RTK:lat_p"
                lon_col = "RTK:lon_p"
                alt_col = "RTK:hmsl_p"
                timestamp_col = "correct_timestamp"
            elif format_type == "blacksquare":
                lat_col = "Latitude"
                lon_col = "Longitude"
                alt_col = "heightMSL"
                timestamp_col = "timestamp"
            elif format_type == "litchi":
                lat_col = "latitude"
                lon_col = "longitude"
                alt_col = "altitude(m)"
                timestamp_col = "timestamp"
            else:
                raise ValueError(f"Unsupported drone format: {format_type}")
        else:  # raw_data
            # raw_data uses original DJI column names
            if "RTKdata:Lat_P" in drone_df.columns:
                lat_col = "RTKdata:Lat_P"
                lon_col = "RTKdata:Lon_P"
                alt_col = "RTKdata:Hmsl_P"
            elif "RTK:lat_p" in drone_df.columns:
                lat_col = "RTK:lat_p"
                lon_col = "RTK:lon_p"
                alt_col = "RTK:hmsl_p"
            else:
                raise ValueError(
                    "No RTK position columns found. Expected RTKdata:Lat_P or RTK:lat_p"
                )
            timestamp_col = None  # Will use GPS Date/Time conversion for raw_data

        # Validate required columns exist
        required_cols = [lat_col, lon_col, alt_col]
        if data_source == "sync_data":
            required_cols.append(timestamp_col)

        missing_cols = [col for col in required_cols if col not in drone_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Filter valid position data (remove rows where lat=0 or NaN)
        valid_rtk = drone_df.filter(
            (pl.col(lat_col) != 0) & pl.col(lat_col).is_not_null()
        )

        if valid_rtk.height == 0:
            logger.warning("No valid position data found (all lat=0 or NaN)")
            return None

        logger.info(f"Filtered to {valid_rtk.height} valid position samples")

        # 4. Extract timestamps
        if data_source == "sync_data":
            # sync_data already has UTC timestamps in the correct column
            timestamps_ctime = valid_rtk[timestamp_col].to_list()
            logger.info(
                f"Extracted {len(timestamps_ctime)} UTC timestamps from sync_data"
            )
        else:
            # raw_data: Convert GPS Date/Time to Unix timestamp
            # Support both CSV format (GPS:Date) and DAT format (GPS:date or RTK:date)
            date_col = None
            time_col = None

            if "GPS:Date" in valid_rtk.columns:
                date_col = "GPS:Date"
                time_col = "GPS:Time"
            elif "GPS:date" in valid_rtk.columns:
                date_col = "GPS:date"
                time_col = "GPS:time"
            elif "RTK:date" in valid_rtk.columns:
                date_col = "RTK:date"
                time_col = "RTK:time"
            else:
                raise ValueError(
                    "No GPS date/time columns found. Expected GPS:Date or GPS:date or RTK:date"
                )

            # Convert GPS Date (YYYYMMDD int) and Time (HHMMSS int) to Unix timestamp
            timestamps_ctime = []
            timezone_correction = datetime.timedelta(hours=drone_timezone_hours)

            for row in valid_rtk.select([date_col, time_col]).iter_rows():
                date_int = int(row[0])
                time_int = int(row[1])

                # Format as strings with zero padding
                ymd_str = f"{date_int:08d}"  # Ensure 8 digits YYYYMMDD
                hms_str = f"{time_int:06d}"  # Ensure 6 digits HHMMSS

                timestamp_str = ymd_str + hms_str

                # Parse datetime and apply timezone correction
                dt = datetime.datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                dt_corrected = dt - timezone_correction

                # Convert to Unix timestamp
                timestamps_ctime.append(dt_corrected.timestamp())

            logger.info(
                f"Converted {len(timestamps_ctime)} timestamps with timezone offset {drone_timezone_hours}h"
            )

        # 4. Load EMLID reference data
        ref_data = self._load_emlid_reference_data(emlid_csv_path, telescope_name)
        telescope_geod = ref_data["telescope"]
        dji_base_geod = ref_data["base"]

        logger.info(
            f"Telescope position: lat={telescope_geod['lat']:.6f}, lon={telescope_geod['lon']:.6f}, alt={telescope_geod['alt']:.2f}"
        )
        logger.info(
            f"DJI base position: lat={dji_base_geod['lat']:.6f}, lon={dji_base_geod['lon']:.6f}, alt={dji_base_geod['alt']:.2f}"
        )

        # 5. Compute RTK correction offset (conditional based on drone format)
        # RTK correction only applies to DJI and Litchi (NOT BlackSquare)
        rtk_applied = format_type in ["dji", "litchi"]

        if rtk_applied:
            delta_e, delta_n, delta_u = self._compute_rtk_correction(
                dji_base_geod, dji_broadcast_geod
            )
            logger.info(
                f"RTK correction offset: dE={delta_e:.3f}m, dN={delta_n:.3f}m, dU={delta_u:.3f}m"
            )
        else:
            # BlackSquare: no RTK correction
            delta_e, delta_n, delta_u = 0.0, 0.0, 0.0
            logger.info(
                f"Skipping RTK correction for {format_type} format (not RTK-capable)"
            )

        # 6. Compute ENU positions for each drone position
        e_list = []
        n_list = []
        up_list = []

        for lat, lon, alt in valid_rtk.select([lat_col, lon_col, alt_col]).iter_rows():
            e, n, u = self._compute_enu_positions(
                telescope_geod["lat"],
                telescope_geod["lon"],
                telescope_geod["alt"],
                float(lat),
                float(lon),
                float(alt),
            )
            e_list.append(e)
            n_list.append(n)
            up_list.append(u)

        # 7. Apply RTK correction (subtract offset from ENU positions)
        e_corrected = [e - delta_e for e in e_list]
        n_corrected = [n - delta_n for n in n_list]
        u_corrected = [u - delta_u for u in up_list]

        logger.info("Applied RTK correction to ENU positions")

        # 8. Compute AZEL for each position
        az_list = []
        el_list = []
        srange_list = []

        for e, n, u in zip(e_corrected, n_corrected, u_corrected, strict=True):
            az, el, sr = self._compute_azel(e, n, u)
            az_list.append(az)
            el_list.append(el)
            srange_list.append(sr)

        logger.info("Computed azimuth, elevation, slant range")

        # 9. Create output DataFrame
        azel_data = pl.DataFrame(
            {
                "timestamp": timestamps_ctime,
                "az": az_list,
                "el": el_list,
                "srange": srange_list,
            }
        ).with_columns(
            [
                pl.col("timestamp").cast(pl.Float64),
                pl.col("az").cast(pl.Float64),
                pl.col("el").cast(pl.Float64),
                pl.col("srange").cast(pl.Float64),
            ]
        )

        # 10. Create metadata
        metadata = {
            "telescope_position": telescope_geod,
            "base_position": dji_base_geod,
            "dji_broadcast_position": dji_broadcast_geod,
            "rtk_correction": {
                "delta_e": delta_e,
                "delta_n": delta_n,
                "delta_u": delta_u,
            },
            "rtk_applied": rtk_applied,
            "drone_format": format_type,
            "timezone_offset_hours": drone_timezone_hours,
            "telescope_name": telescope_name,
            "num_samples": len(timestamps_ctime),
        }

        # 11. Generate version name (timestamp-based: rev_YYYYMMDD_HHMMSS)
        version_name = datetime.datetime.now().strftime("rev_%Y%m%d_%H%M%S")

        logger.info(
            f"AZEL analysis complete: {len(timestamps_ctime)} samples, version={version_name}"
        )

        # 12. Create AZELVersion
        version = AZELVersion(
            version_name=version_name, azel_data=azel_data, metadata=metadata
        )

        # 13. Save to HDF5
        self._save_to_hdf5(version)

        return version

    def _save_to_hdf5(self, version: AZELVersion) -> None:
        """Save AZEL version to HDF5 file.

        Args:
            version: AZELVersion to save

        Example:
            >>> version = azel.run_analysis(...)
            >>> azel._save_to_hdf5(version)
        """
        hdf5_path = self.azel_dir / "azel_solution.h5"

        with h5py.File(hdf5_path, "a") as f:
            # Create version group (overwrite if exists)
            if version.version_name in f:
                del f[version.version_name]

            version_group = f.create_group(version.version_name)

            # Save DataFrame columns
            df_group = version_group.create_group("azel_data")
            for col_name in version.azel_data.columns:
                col_data = version.azel_data[col_name].to_numpy()
                df_group.create_dataset(col_name, data=col_data)

            # Save DataFrame schema info
            df_group.attrs["columns"] = json.dumps(version.azel_data.columns)
            df_group.attrs["dtypes"] = json.dumps(
                [str(dtype) for dtype in version.azel_data.dtypes]
            )

            # Save metadata
            version_group.attrs["metadata"] = json.dumps(version.metadata)

        logger.info(f"Saved AZEL version '{version.version_name}' to HDF5")

    def _load_from_hdf5(self, version_name: str) -> AZELVersion:
        """Load AZEL version from HDF5 file.

        Args:
            version_name: Name of version to load

        Returns:
            Loaded AZELVersion

        Raises:
            FileNotFoundError: If HDF5 file doesn't exist
            KeyError: If version_name not found in HDF5

        Example:
            >>> version = azel._load_from_hdf5('rev_20260218_143022')
        """
        hdf5_path = self.azel_dir / "azel_solution.h5"

        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        with h5py.File(hdf5_path, "r") as f:
            if version_name not in f:
                raise KeyError(f"Version '{version_name}' not found in HDF5")

            version_group = f[version_name]
            df_group = version_group["azel_data"]

            # Reconstruct DataFrame from columns
            columns = json.loads(df_group.attrs["columns"])
            dtypes = json.loads(df_group.attrs["dtypes"])

            data_dict = {}
            for col_name in columns:
                data_dict[col_name] = df_group[col_name][:]

            azel_data = pl.DataFrame(data_dict)

            # Cast to original dtypes
            cast_exprs = []
            for col_name, dtype_str in zip(columns, dtypes, strict=True):
                if "Float64" in dtype_str:
                    cast_exprs.append(pl.col(col_name).cast(pl.Float64))
                elif "Int64" in dtype_str:
                    cast_exprs.append(pl.col(col_name).cast(pl.Int64))

            if cast_exprs:
                azel_data = azel_data.with_columns(cast_exprs)

            # Load metadata
            metadata = json.loads(version_group.attrs["metadata"])

        logger.info(f"Loaded AZEL version '{version_name}' from HDF5")

        return AZELVersion(
            version_name=version_name, azel_data=azel_data, metadata=metadata
        )

    def list_versions(self) -> list[str]:
        """List all AZEL versions in HDF5 file.

        Returns:
            List of version names, sorted chronologically

        Example:
            >>> azel.list_versions()
            ['rev_20260218_120000', 'rev_20260218_143022']
        """
        hdf5_path = self.azel_dir / "azel_solution.h5"

        if not hdf5_path.exists():
            return []

        with h5py.File(hdf5_path, "r") as f:
            versions = sorted(list(f.keys()))

        return versions

    def get_latest_version(self) -> AZELVersion | None:
        """Get the most recent AZEL version.

        Returns:
            Latest AZELVersion, or None if no versions exist

        Example:
            >>> latest = azel.get_latest_version()
            >>> if latest:
            ...     print(latest.version_name)
        """
        versions = self.list_versions()

        if not versions:
            return None

        latest_name = versions[-1]  # Sorted chronologically, last is latest
        return self._load_from_hdf5(latest_name)
