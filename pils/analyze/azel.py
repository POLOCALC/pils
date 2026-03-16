"""AZEL (Azimuth-Elevation) analysis module for flight data."""

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import polars as pl
import pymap3d as pm

from pils.flight import Flight
from pils.sensors.emlid import Emlid
from pils.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AZELVersion:
    """AZEL analysis version container.

    Parameters
    ----------
    version_name : str
        Name identifier for this AZEL analysis version.
    azel_data : pl.DataFrame
        Polars DataFrame with columns: timestamp, az, el, srange.
    metadata : dict[str, Any]
        Dictionary containing observer position and other metadata.

    Examples
    --------
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
    └── azel_solution.h5  # HDF5 with telescope-grouped structure

    HDF5 Structure
    --------------
    azel_solution.h5
    ├── SATP1/                     # Telescope group
    │   ├── rev_20260302_120000/   # Revision 1
    │   │   ├── azel_data/         # DataFrame columns
    │   │   └── metadata           # JSON metadata
    │   └── rev_20260302_130000/   # Revision 2
    └── SATP2/                     # Another telescope
        └── rev_20260302_120000/

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
    >>> # Run analysis with additional metadata
    >>> dji_broadcast = {'lat': -22.9597732, 'lon': -67.7866847, 'alt': 5173.020}
    >>> additional_meta = {'weather': 'clear', 'operator': 'Alice'}
    >>> version = azel.run_analysis('SATP1', dji_broadcast,
    ...                              additional_metadata=additional_meta,
    ...                              save_data=True)
    >>> # List versions for specific telescope
    >>> versions = azel.list_versions(telescope_name='SATP1')
    >>> # Get latest for specific telescope
    >>> latest = azel.get_latest_version(telescope_name='SATP1')
    """

    def __init__(self, flight: Flight) -> None:
        """Initialize AZEL analysis with Flight object.

        Creates the proc/azel directory structure if it doesn't exist.

        Parameters
        ----------
        flight : Flight
            Flight object with valid flight_path attribute.

        Raises
        ------
        TypeError
            If flight is not a Flight object.
        ValueError
            If flight_path is None or doesn't exist.

        Examples
        --------
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
    def _compute_rtk_correction(
        dji_base_geod: dict[str, float], dji_broadcast_geod: dict[str, float]
    ) -> tuple[float, float, float]:
        """Compute RTK correction offset between actual and broadcast base positions.

        Parameters
        ----------
        dji_base_geod : dict[str, float]
            Actual DJI base position with keys 'lat', 'lon', 'alt'.
            Values in WGS84 degrees (lat, lon) and meters (alt).
        dji_broadcast_geod : dict[str, float]
            Broadcast DJI base position with keys 'lat', 'lon', 'alt'.
            Values in WGS84 degrees (lat, lon) and meters (alt).

        Returns
        -------
        tuple[float, float, float]
            Tuple of (delta_east, delta_north, delta_up) offset in meters
            to subtract from drone ENU coordinates.

        Examples
        --------
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
        save_data: bool = False,
        additional_metadata: dict[str, Any] | None = None,
    ) -> AZELVersion | None:
        """Run AZEL analysis for drone telescope tracking.

        Loads drone RTK data, computes reference positions from EMLID survey data,
        applies RTK corrections, and computes azimuth-elevation angles for telescope
        tracking. Uses vectorized pymap3d operations for efficient coordinate transforms.

        Parameters
        ----------
        telescope_name : str
            Telescope identifier (e.g., 'SATP1') for filtering EMLID data.
            This will be used as the HDF5 group name when saving.
        dji_broadcast_geod : dict[str, float]
            Broadcast DJI base position dict with keys 'lat', 'lon', 'alt'.
            Values in WGS84 degrees (lat, lon) and meters (alt).
        drone_timezone_hours : float, optional
            Timezone offset for drone timestamps in hours.
            Only applied to raw_data; sync_data is assumed UTC.
            Default is 0.0 (UTC).
        save_data : bool, optional
            Whether to save the data and create new version.
            Default is False.
        additional_metadata : dict[str, Any] | None, optional
            Additional metadata to include in the version metadata.
            This can include experiment details, weather conditions, etc.
            Default is None.

        Returns
        -------
        AZELVersion | None
            AZELVersion with computed azimuth, elevation, slant range, ENU coordinates data 
            and metadata, or None if no valid RTK data available after filtering.

        Raises
        ------
        ValueError
            If drone data not loaded or empty, or telescope/base not found.
        FileNotFoundError
            If EMLID CSV file not found at expected location.

        Examples
        --------
        >>> from pils.flight import Flight
        >>> from pils.analyze.azel import AZELAnalysis
        >>> flight = Flight(flight_info)
        >>> flight.add_drone_data()
        >>> azel = AZELAnalysis(flight)
        >>> dji_broadcast = {'lat': -22.9597732, 'lon': -67.7866847, 'alt': 5173.020}
        >>> additional_meta = {'weather': 'clear', 'wind_speed_ms': 5.2}
        >>> version = azel.run_analysis('SATP1', dji_broadcast, additional_metadata=additional_meta)
        >>> print(version.azel_data.head())
        >>> # Shows: timestamp, az, el, srange columns
        >>> print(version.metadata['weather'])  # 'clear'
        """

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

        # 3. Set column names based on data source
        if data_source == "sync_data":
            # sync_data uses standardized column names (as of coordinate standardization feature)
            lat_col = "latitude"
            lon_col = "longitude"
            alt_col = "altitude"
            timestamp_col = "timestamp"
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
        emlid = Emlid(self.flight)

        ref_data = emlid.load_data(telescope_name=telescope_name)

        # Extract scalar values from DataFrames
        telescope_geod = ref_data["telescope"].row(0, named=True)

        if format_type == "dji":
            base_geod = ref_data["base"]["dji"].row(0, named=True)
        else:
            base_geod = ref_data["base"]["emlid"].row(0, named=True)

        logger.info(
            f"Telescope position: lat={telescope_geod['lat']:.6f}, lon={telescope_geod['lon']:.6f}, alt={telescope_geod['alt']:.2f}"
        )
        logger.info(
            f"DJI base position: lat={base_geod['lat']:.6f}, lon={base_geod['lon']:.6f}, alt={base_geod['alt']:.2f}"
        )

        # 5. Compute RTK correction offset (conditional based on drone format)
        # RTK correction only applies to DJI and Litchi (NOT BlackSquare)
        rtk_applied = format_type in ["dji", "litchi"]

        if rtk_applied:
            delta_e, delta_n, delta_u = self._compute_rtk_correction(
                base_geod, dji_broadcast_geod
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

        e, n, u = pm.geodetic2enu(
            valid_rtk[lat_col],
            valid_rtk[lon_col],
            valid_rtk[alt_col],
            telescope_geod["lat"],
            telescope_geod["lon"],
            telescope_geod["alt"],
        )

        # 7. Apply RTK correction (subtract offset from ENU positions)
        e_corrected = e - delta_e
        n_corrected = n - delta_n
        u_corrected = u - delta_u

        logger.info("Applied RTK correction to ENU positions")

        azimuth, elevation, slant_range = pm.enu2aer(
            e_corrected, n_corrected, u_corrected
        )

        logger.info("Computed azimuth, elevation, slant range")

        # 9. Create output DataFrame
        azel_data = pl.DataFrame(
            {
                "timestamp": timestamps_ctime,
                "az": azimuth,
                "el": elevation,
                "srange": slant_range,
                "E": e_corrected,
                "N": n_corrected,
                "U": u_corrected,
            }
        ).with_columns(
            [
                pl.col("timestamp").cast(pl.Float64),
                pl.col("az").cast(pl.Float64),
                pl.col("el").cast(pl.Float64),
                pl.col("srange").cast(pl.Float64),
                pl.col("E").cast(pl.Float64),
                pl.col("N").cast(pl.Float64),
                pl.col("U").cast(pl.Float64),
            ]
        )

        # 10. Create metadata
        metadata = {
            "telescope_position": telescope_geod,
            "base_position": base_geod,
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

        # 10b. Merge additional metadata if provided
        if additional_metadata is not None:
            metadata.update(additional_metadata)

        # 11. Generate version name (timestamp-based: rev_YYYYMMDD_HHMMSS)
        version_name = datetime.datetime.now().strftime("rev_%Y%m%d_%H%M%S")

        logger.info(
            f"AZEL analysis complete: {len(timestamps_ctime)} samples, version={version_name}"
        )

        # Create version object
        version = AZELVersion(
            version_name=version_name, azel_data=azel_data, metadata=metadata
        )

        # Save to HDF5 if requested
        if save_data:
            self._save_to_hdf5(version, telescope_name=telescope_name)

        return version

    def _save_to_hdf5(self, version: AZELVersion, telescope_name: str) -> None:
        """Save AZEL version to HDF5 file with telescope grouping.

        Creates hierarchical structure: telescope_name/revision_name/data

        Parameters
        ----------
        version : AZELVersion
            AZELVersion object to save to HDF5 file.
        telescope_name : str
            Telescope name to use as parent group.

        Examples
        --------
        >>> version = azel.run_analysis('SATP1', ...)
        >>> azel._save_to_hdf5(version, telescope_name='SATP1')
        """
        hdf5_path = self.azel_dir / "azel_solution.h5"

        with h5py.File(hdf5_path, "a") as f:
            # Create telescope group if doesn't exist
            if telescope_name not in f:
                telescope_group = f.create_group(telescope_name)
            else:
                telescope_group = f[telescope_name]

            # Create version group under telescope (overwrite if exists)
            if version.version_name in telescope_group:
                del telescope_group[version.version_name]

            version_group = telescope_group.create_group(version.version_name)

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

        logger.info(
            f"Saved AZEL version '{version.version_name}' for telescope '{telescope_name}' to HDF5"
        )

    def _load_from_hdf5(self, version_name: str, telescope_name: str) -> AZELVersion:
        """Load AZEL version from HDF5 file.

        Parameters
        ----------
        version_name : str
            Name of version to load (e.g., 'rev_20260218_143022').
        telescope_name : str
            Telescope group name containing the version.

        Returns
        -------
        AZELVersion
            Loaded AZELVersion object containing azel_data and metadata.

        Raises
        ------
        FileNotFoundError
            If HDF5 file doesn't exist.
        KeyError
            If telescope_name or version_name not found in HDF5 file.

        Examples
        --------
        >>> version = azel._load_from_hdf5('rev_20260218_143022', telescope_name='SATP1')
        """
        hdf5_path = self.azel_dir / "azel_solution.h5"

        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        with h5py.File(hdf5_path, "r") as f:
            if telescope_name not in f:
                raise KeyError(f"Telescope '{telescope_name}' not found in HDF5")

            telescope_group = f[telescope_name]

            if version_name not in telescope_group:
                raise KeyError(
                    f"Version '{version_name}' not found under telescope '{telescope_name}' in HDF5"
                )

            version_group = telescope_group[version_name]
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

        logger.info(
            f"Loaded AZEL version '{version_name}' for telescope '{telescope_name}' from HDF5"
        )

        return AZELVersion(
            version_name=version_name, azel_data=azel_data, metadata=metadata
        )

    def list_versions(
        self, telescope_name: str | None = None
    ) -> list[str] | dict[str, list[str]]:
        """List all AZEL versions in HDF5 file.

        Parameters
        ----------
        telescope_name : str | None, optional
            If provided, returns versions for that telescope only.
            If None, returns a dictionary with all telescopes and their versions.
            Default is None.

        Returns
        -------
        list[str] | dict[str, list[str]]
            If telescope_name provided: List of version names for that telescope,
            sorted chronologically. Empty list if telescope doesn't exist.
            If telescope_name is None: Dictionary mapping telescope names to
            their version lists. Empty dict if HDF5 file doesn't exist.

        Examples
        --------
        >>> # List all versions for specific telescope
        >>> azel.list_versions(telescope_name='SATP1')
        ['rev_20260218_120000', 'rev_20260218_143022']
        >>> # List all telescopes and their versions
        >>> azel.list_versions()
        {'SATP1': ['rev_20260218_120000'], 'SATP2': ['rev_20260218_130000']}
        """
        hdf5_path = self.azel_dir / "azel_solution.h5"

        if not hdf5_path.exists():
            return [] if telescope_name is not None else {}

        with h5py.File(hdf5_path, "r") as f:
            if telescope_name is not None:
                # Return versions for specific telescope
                if telescope_name not in f:
                    return []
                telescope_group = f[telescope_name]
                versions = sorted(list(telescope_group.keys()))
                return versions
            else:
                # Return all telescopes with their versions
                result = {}
                for tel_name in f.keys():
                    result[tel_name] = sorted(list(f[tel_name].keys()))
                return result

    def get_latest_version(
        self, telescope_name: str | None = None
    ) -> AZELVersion | None:
        """Get the most recent AZEL version.

        Parameters
        ----------
        telescope_name : str | None, optional
            If provided, returns the latest version for that telescope.
            If None, returns the latest version across all telescopes.
            Default is None.

        Returns
        -------
        AZELVersion | None
            Latest AZELVersion object, or None if no versions exist.

        Examples
        --------
        >>> # Get latest for specific telescope
        >>> latest = azel.get_latest_version(telescope_name='SATP1')
        >>> if latest:
        ...     print(latest.version_name)
        >>> # Get latest across all telescopes
        >>> latest = azel.get_latest_version()
        >>> if latest:
        ...     print(latest.metadata['telescope_name'])
        """
        if telescope_name is not None:
            # Get latest for specific telescope
            versions = self.list_versions(telescope_name=telescope_name)

            if not versions:
                return None

            latest_name = versions[-1]  # Sorted chronologically, last is latest
            return self._load_from_hdf5(latest_name, telescope_name=telescope_name)
        else:
            # Get latest across all telescopes
            all_versions = self.list_versions()

            if not all_versions:
                return None

            # Find the latest version across all telescopes
            latest_version_name = None
            latest_telescope = None

            for tel_name, versions in all_versions.items():
                if versions:
                    last_version = versions[-1]
                    if (
                        latest_version_name is None
                        or last_version > latest_version_name
                    ):
                        latest_version_name = last_version
                        latest_telescope = tel_name

            if latest_version_name is None:
                return None

            return self._load_from_hdf5(
                latest_version_name, telescope_name=latest_telescope
            )
