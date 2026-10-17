"""AZEL (Azimuth-Elevation) analysis module for flight data."""

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pymap3d as pm

from pils import store
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

    File Structure (see :mod:`pils.store`)
    --------------
    flight_dir/proc/azel/
    ├── SATP1/
    │   ├── rev_20260302_120000/
    │   │   ├── manifest.json      # metadata, code version, columns
    │   │   └── azel_data.parquet  # timestamp, az_deg, el_deg, srange_m, e/n/u_m
    │   └── rev_20260302_130000/
    └── SATP2/
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

    def __init__(self, flight: Flight, output_dir: str | Path | None = None) -> None:
        """Initialize AZEL analysis with Flight object.

        Creates the proc/azel directory structure if it doesn't exist.

        Parameters
        ----------
        flight : Flight
            Flight object with valid flight_path attribute.
        output_dir : str | Path, optional
            Base folder for the results instead of the flight folder
            (results go to ``<output_dir>/proc/azel``).

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
        base = Path(output_dir) if output_dir is not None else self.flight_path
        self.azel_dir = base / "proc" / "azel"
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
            Not used: the drone timestamps are UTC (kept for compatibility,
            and recorded in the metadata).
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
        drone_model = (
            getattr(self.flight, "_Flight__drone_model", None) or "dji"
        ).lower()
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

        # 3. Positions (schema names, see pils.schema): the synchronized
        # table has lat_deg/lon_deg/h_ell_m, the raw DJI table the RTK ones
        if data_source == "sync_data":
            lat_col, lon_col, alt_col = "lat_deg", "lon_deg", "h_ell_m"
        else:
            lat_col, lon_col, alt_col = "rtk_lat_deg", "rtk_lon_deg", "rtk_h_ell_m"
        timestamp_col = "timestamp"

        required_cols = [lat_col, lon_col, alt_col, timestamp_col]
        missing_cols = [col for col in required_cols if col not in drone_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Valid positions (lat=0 or missing are not)
        valid_rtk = drone_df.filter(
            (pl.col(lat_col) != 0)
            & pl.col(lat_col).is_not_null()
            & pl.col(timestamp_col).is_not_null()
        )

        if valid_rtk.height == 0:
            logger.warning("No valid position data found (all lat=0 or NaN)")
            return None

        logger.info(f"Filtered to {valid_rtk.height} valid position samples")
        timestamps_ctime = valid_rtk[timestamp_col].to_list()

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
                "timestamp": np.asarray(timestamps_ctime),
                "az_deg": azimuth,
                "el_deg": elevation,
                "srange_m": slant_range,
                "e_m": e_corrected,
                "n_m": n_corrected,
                "u_m": u_corrected,
            }
        ).with_columns(
            [
                pl.col("timestamp").cast(pl.Float64),
                pl.col("az_deg").cast(pl.Float64),
                pl.col("el_deg").cast(pl.Float64),
                pl.col("srange_m").cast(pl.Float64),
                pl.col("e_m").cast(pl.Float64),
                pl.col("n_m").cast(pl.Float64),
                pl.col("u_m").cast(pl.Float64),
            ]
        )

        mask_expr = (
            (~pl.col("timestamp").is_nan())
            & (~pl.col("az_deg").is_nan())
            & (~pl.col("el_deg").is_nan())
            & (~pl.col("srange_m").is_nan())
            & (~pl.col("e_m").is_nan())
            & (~pl.col("n_m").is_nan())
            & (~pl.col("u_m").is_nan())
        )
        azel_data = azel_data.filter(mask_expr)

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
            self._save_version(version, telescope_name=telescope_name)

        return version

    def _save_version(self, version: AZELVersion, telescope_name: str) -> Path:
        """Save a version: ``azel/<telescope>/<version>/`` (see :mod:`pils.store`).

        Parameters
        ----------
        version : AZELVersion
            Version to save.
        telescope_name : str
            Telescope it is for.

        Returns
        -------
        Path
            The version's directory.
        """
        path = store.save_version(
            self.azel_dir,
            version.version_name,
            {"azel_data": version.azel_data},
            version.metadata,
            parent=telescope_name,
            kind="azel",
        )
        logger.info(f"Saved AZEL version '{version.version_name}' to {path}")
        return path

    def version_dir(self, version_name: str, telescope_name: str) -> Path:
        """Directory of a version (its tables, manifest and CSV)."""
        return self.azel_dir / telescope_name / version_name

    def _load_version(self, version_name: str, telescope_name: str) -> AZELVersion:
        """Load a saved version.

        Raises
        ------
        KeyError
            If the version is not saved for this telescope.
        """
        tables, metadata = store.load_version(
            self.azel_dir, version_name, parent=telescope_name
        )
        return AZELVersion(
            version_name=version_name,
            azel_data=tables["azel_data"],  # type: ignore[arg-type]
            metadata=metadata,
        )

    def list_versions(
        self, telescope_name: str | None = None
    ) -> list[str] | dict[str, list[str]]:
        """List the saved AZEL versions.

        Parameters
        ----------
        telescope_name : str | None, optional
            If provided, returns versions for that telescope only.
            If None, returns a dictionary with all telescopes and their versions.

        Returns
        -------
        list[str] | dict[str, list[str]]
            Version names, oldest first; or, without ``telescope_name``, the
            versions of each telescope.

        Examples
        --------
        >>> azel.list_versions(telescope_name='SATP1')
        ['rev_20260218_120000', 'rev_20260218_143022']
        >>> azel.list_versions()
        {'SATP1': ['rev_20260218_120000'], 'SATP2': ['rev_20260218_130000']}
        """
        if telescope_name is not None:
            return store.list_versions(self.azel_dir, parent=telescope_name)
        if not self.azel_dir.is_dir():
            return {}
        result = {}
        for item in sorted(self.azel_dir.iterdir()):
            versions = store.list_versions(self.azel_dir, parent=item.name)
            if item.is_dir() and versions:
                result[item.name] = versions
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
            return self._load_version(latest_name, telescope_name=telescope_name)
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

            return self._load_version(
                latest_version_name, telescope_name=latest_telescope
            )
