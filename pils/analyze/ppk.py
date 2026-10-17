"""
PPK Analysis - Standalone Post-Processed Kinematic GPS Analysis.

This module provides standalone PPK analysis using RTKLIB with smart
execution logic, versioned storage, and HDF5 persistence.

Key Features
------------
- Smart re-run: Only executes if config changed or forced
- Auto-timestamp versioning (rev_YYYYMMDD_HHMMSS)
- Per-revision folder organization
- HDF5 storage for all versions
- Complete separation from Flight class

Examples
--------
>>> from pils.analyze.ppk import PPKAnalysis
>>> from pils.flight import Flight
>>> # Create Flight object
>>> flight_info = {
...     "drone_data_folder_path": "/path/to/flight/drone",
...     "aux_data_folder_path": "/path/to/flight/aux"
... }
>>> flight = Flight(flight_info)
>>> # Create new analysis with Flight object
>>> ppk = PPKAnalysis(flight)
>>> ppk.run_analysis('config.conf')  # Only runs if config changed
>>> # Load existing
>>> ppk = PPKAnalysis.from_hdf5(flight)
>>> latest = ppk.get_latest_version()
>>> print(latest.pos_data)  # Polars DataFrame
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import h5py
import polars as pl

from pils import store
from pils.analyze.ppk_tuning import rtklib_binary, tune
from pils.analyze.ppkdata.PPK.pos_analyzer import POSAnalyzer
from pils.analyze.ppkdata.PPK.report import RTKLIBReport
from pils.analyze.ppkdata.PPK.stat_analyzer import STATAnalyzer
from pils.analyze.ppkdata.RINEX.report import RINEXReport
from pils.flight import Flight
from pils.utils.hdf5 import read_table

logger = logging.getLogger(__name__)

# Position and statistics tables saved before the schema (pils.schema)
LEGACY_POSITION_NAMES = {
    "lat": "lat_deg",
    "lon": "lon_deg",
    "height": "h_ell_m",
    "Q": "quality",
    "ns": "num_sats",
    "sdn": "sd_n_m",
    "sde": "sd_e_m",
    "sdu": "sd_u_m",
    "age": "age_s",
    "east": "e_m",
    "north": "n_m",
    "up": "u_m",
}
LEGACY_STATISTICS_NAMES = {
    "tow": "tow_s",
    "azimuth": "az_deg",
    "elevation": "el_deg",
    "resid_code": "resid_code_m",
    "resid_phase": "resid_phase_m",
    "snr": "snr_dbhz",
}


def legacy_ppk_tables(
    position: pl.DataFrame, statistics: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Position and statistics tables saved before the schema, renamed to it.

    The old position ``time`` was GPS time (the parser only read GPST
    files) and becomes ``datetime`` (UTC) and ``timestamp``; heights were
    ellipsoidal (RTKLIB's default, and the POLOCALC configuration). The old
    statistics had no GPS week: it is taken from the first position epoch,
    and gives their ``timestamp``. Tables already in the schema are returned
    unchanged.
    """
    from pils.analyze.ppkdata.utils import GPS_EPOCH_UNIX, gps_utc_offset, to_utc

    position = position.rename(
        {k: v for k, v in LEGACY_POSITION_NAMES.items() if k in position.columns}
    )
    if "time" in position.columns:
        position = (
            position.with_columns(to_utc(pl.col("time"), "GPST").alias("datetime"))
            .with_columns((pl.col("datetime").dt.epoch("us") / 1e6).alias("timestamp"))
            .select(
                "datetime", "timestamp", pl.exclude("time", "datetime", "timestamp")
            )
        )
    legacy_statistics = "tow" in statistics.columns
    statistics = statistics.rename(
        {k: v for k, v in LEGACY_STATISTICS_NAMES.items() if k in statistics.columns}
    )
    if legacy_statistics:
        if "datetime" in position.columns and position.height and statistics.height:
            first = position["datetime"].min()
            gps_first = (
                first.timestamp()  # type: ignore[union-attr]
                - GPS_EPOCH_UNIX
                + 18  # leap seconds only matter at a week boundary
            )
            week0 = int(gps_first // 604800)
            tow0 = gps_first - week0 * 604800
            week = pl.lit(week0) + (pl.col("tow_s") < tow0 - 302400).cast(pl.Int64)
            gps_seconds = week * 604800 + pl.col("tow_s")
            gps_time = pl.from_epoch(
                ((gps_seconds + GPS_EPOCH_UNIX) * 1e6).round().cast(pl.Int64),
                time_unit="us",
            )
            statistics = statistics.with_columns(
                (gps_seconds + GPS_EPOCH_UNIX - gps_utc_offset(gps_time)).alias(
                    "timestamp"
                ),
                week.alias("gps_week"),
            ).select("timestamp", "gps_week", pl.exclude("timestamp", "gps_week"))
    return position, statistics


@dataclass
class PPKVersion:
    """
    Container for a single PPK analysis revision.

    Each revision represents one execution of RTKLIB with specific
    configuration parameters, storing position solutions, statistics,
    and metadata.

    Attributes
    ----------
    version_name : str
        Auto-timestamp version identifier (rev_YYYYMMDD_HHMMSS)
    pos_data : pl.DataFrame
        Position solution DataFrame from .pos file
    stat_data : pl.DataFrame
        Processing statistics DataFrame from .pos.stat file
    metadata : Dict[str, Any]
        Config hash, parsed parameters, and execution metadata
    revision_path : Path
        Path to revision folder containing .pos, .stat, .conf files

    Examples
    --------
    >>> version = PPKVersion(
    ...     version_name='rev_20260204_143022',
    ...     pos_data=pos_df,
    ...     stat_data=stat_df,
    ...     metadata={'config_hash': 'abc123'},
    ...     revision_path=Path('/flight/proc/ppk/rev_20260204_143022')
    ... )
    """

    version_name: str
    pos_data: pl.DataFrame
    stat_data: pl.DataFrame
    metadata: dict[str, Any]
    revision_path: Path
    # Candidate configurations tried by the tuning (run_analysis(optimize=True))
    tuning_data: pl.DataFrame | None = None


class PPKAnalysis:
    """
    Standalone PPK analysis manager with smart execution and versioning.

    Manages RTKLIB-based post-processing of GPS data with intelligent
    re-run logic, automatic versioning, and HDF5 persistence. Completely
    separate from Flight class.

    The analysis system:
    - Only runs RTKLIB when config changes (SHA256 hash comparison)
    - Stores each revision in its own folder with .pos, .stat, .conf files
    - Saves all versions to a single ppk_solution.h5 HDF5 file
    - Generates auto-timestamp version names

    File Structure
    --------------
    flight_dir/proc/ppk/
    ├── ppk_solution.h5              # HDF5 with all versions
    └── rev_20260204_143022/         # Per-revision folder
        ├── solution.pos              # RTKLIB position output
        ├── solution.pos.stat         # RTKLIB statistics output
        └── config.conf               # RTKLIB config used

    Attributes
    ----------
    flight_path : Path
        Root flight directory
    ppk_dir : Path
        PPK directory ({flight_path}/proc/ppk/)
    hdf5_path : Path
        Path to ppk_solution.h5 file
    versions : Dict[str, PPKVersion]
        Dictionary of loaded PPK versions

    Examples
    --------
    >>> from pils.flight import Flight
    >>> # Create Flight object
    >>> flight_info = {
    ...     "drone_data_folder_path": "/path/to/flight/drone",
    ...     "aux_data_folder_path": "/path/to/flight/aux"
    ... }
    >>> flight = Flight(flight_info)
    >>> # Create new analysis
    >>> ppk = PPKAnalysis(flight)
    >>> version = ppk.run_analysis('rtklib.conf')  # Smart execution
    >>> # Only runs if config changed
    >>> version2 = ppk.run_analysis('rtklib.conf')  # Skipped if same
    >>> # Force re-run
    >>> version3 = ppk.run_analysis('rtklib.conf', force=True)
    >>> # Access versions
    >>> latest = ppk.get_latest_version()
    >>> all_versions = ppk.list_versions()
    """

    def __init__(self, flight: Flight):
        """
        Initialize PPKAnalysis for a flight.

        Creates the proc/ppk directory structure if it doesn't exist.

        Parameters
        ----------
        flight : Flight
            Flight object with valid flight_path attribute

        Raises
        ------
        TypeError
            If flight is not a Flight object
        ValueError
            If flight.flight_path is None or not an existing directory

        Examples
        --------
        >>> from pils.flight import Flight
        >>> flight_info = {
        ...     "drone_data_folder_path": "/path/to/flight/drone",
        ...     "aux_data_folder_path": "/path/to/flight/aux"
        ... }
        >>> flight = Flight(flight_info)
        >>> ppk = PPKAnalysis(flight)
        >>> print(ppk.ppk_dir)
        /path/to/flight/proc/ppk
        """
        # Validate input type
        if not isinstance(flight, Flight):
            raise TypeError(
                f"Expected Flight object, got {type(flight).__name__}. "
                "PPKAnalysis now requires a Flight object instead of a path."
            )

        # Validate flight_path exists
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

        self.flight = flight
        self.flight_path = flight_path
        self.ppk_dir = self.flight_path / "proc" / "ppk"
        self.hdf5_path = self.ppk_dir / "ppk_solution.h5"
        self.versions: dict[str, PPKVersion] = {}

        # Create PPK directory structure
        self.ppk_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized PPKAnalysis for flight: {self.flight_path}")
        logger.info(f"PPK directory: {self.ppk_dir}")

    def _hash_config(self, config_path: Path) -> str:
        """
        Generate SHA256 hash of config file for change detection.

        Reads the entire config file and computes its SHA256 hash.
        Used to detect if configuration has changed since last run.

        Parameters
        ----------
        config_path : Path
            Path to RTKLIB config file

        Returns
        -------
        str
            64-character hexadecimal SHA256 hash

        Examples
        --------
        >>> ppk = PPKAnalysis('/path/to/flight')
        >>> hash1 = ppk._hash_config(Path('config1.conf'))
        >>> hash2 = ppk._hash_config(Path('config2.conf'))
        >>> print(hash1 == hash2)  # True if configs identical
        """
        content = config_path.read_bytes()
        return hashlib.sha256(content).hexdigest()

    def _parse_config_params(self, config_path: Path) -> dict[str, Any]:
        """
        Parse key RTKLIB configuration parameters.

        Extracts important configuration parameters from RTKLIB config file.
        Only parses key parameters (not full JSON dump), focusing on:
        - pos1-posmode: Processing mode (kinematic/static)
        - pos1-elmask: Elevation mask
        - pos2-armode: Ambiguity resolution mode
        - ant2-postype: Antenna position type

        Parameters
        ----------
        config_path : Path
            Path to RTKLIB config file

        Returns
        -------
        Dict[str, Any]
            Dictionary of key-value config parameters

        Examples
        --------
        >>> ppk = PPKAnalysis('/path/to/flight')
        >>> params = ppk._parse_config_params(Path('rtklib.conf'))
        >>> print(params['pos1-posmode'])
        'kinematic'
        """
        params = {}
        content = config_path.read_text()

        for line in content.splitlines():
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse key=value pairs
            if "=" in line:
                key, value = line.split("=", 1)
                params[key.strip()] = value.strip()

        return params

    def _generate_version_name(self) -> str:
        """
        Generate auto-timestamp version name.

        Creates version name in format: rev_YYYYMMDD_HHMMSS

        Returns
        -------
        str
            Version name string

        Examples
        --------
        >>> ppk = PPKAnalysis('/path/to/flight')
        >>> name = ppk._generate_version_name()
        >>> print(name)
        'rev_20260204_143022'
        """
        return datetime.now().strftime("rev_%Y%m%d_%H%M%S")

    def _create_revision_folder(self, version_name: str) -> Path:
        """
        Create per-revision folder for storing analysis files.

        Creates folder at {ppk_dir}/{version_name}/ to store:
        - solution.pos (RTKLIB position output)
        - solution.pos.stat (RTKLIB statistics)
        - config.conf (RTKLIB config used)

        Parameters
        ----------
        version_name : str
            Version name (rev_YYYYMMDD_HHMMSS)

        Returns
        -------
        Path
            Path to created revision folder

        Examples
        --------
        >>> ppk = PPKAnalysis('/path/to/flight')
        >>> folder = ppk._create_revision_folder('rev_20260204_143022')
        >>> print(folder)
        /path/to/flight/proc/ppk/rev_20260204_143022
        """
        revision_path = self.ppk_dir / version_name
        revision_path.mkdir(parents=True, exist_ok=True)
        return revision_path

    def _should_run_analysis(self, config_path: Path, optimize: bool = False) -> bool:
        """
        Determine if RTKLIB analysis should run.

        Smart execution logic:
        1. Run if no HDF5 file exists (first run)
        2. Run if no versions loaded (no previous runs)
        3. Run if config hash differs from latest version
        4. Skip if config unchanged

        Parameters
        ----------
        config_path : Path
            Path to RTKLIB config file

        Returns
        -------
        bool
            True if analysis should run, False to skip

        Examples
        --------
        >>> ppk = PPKAnalysis('/path/to/flight')
        >>> should_run = ppk._should_run_analysis(Path('config.conf'))
        >>> if should_run:
        ...     print("Config changed, running analysis")
        ... else:
        ...     print("Config unchanged, skipping")
        """
        # Run if no versions loaded
        if not self.versions:
            logger.info("No previous versions found - running analysis")
            return True

        # Get latest version and compare config hash
        latest = self.get_latest_version()
        if latest is None:
            return True

        current_hash = self._hash_config(config_path) + (
            ":optimize" if optimize else ""
        )
        previous_hash = latest.metadata.get("config_hash")

        if current_hash != previous_hash:
            logger.info("Config changed - running analysis")
            return True

        logger.info("Config unchanged - skipping analysis")
        return False

    def get_latest_version(self) -> PPKVersion | None:
        """
        Get the most recent PPK version.

        Returns the version with the latest timestamp based on
        version name sorting (rev_YYYYMMDD_HHMMSS format sorts chronologically).

        Returns
        -------
        Optional[PPKVersion]
            Latest PPKVersion or None if no versions exist

        Examples
        --------
        >>> ppk = PPKAnalysis.from_hdf5('/path/to/flight')
        >>> latest = ppk.get_latest_version()
        >>> if latest:
        ...     print(f"Latest: {latest.version_name}")
        ...     print(latest.pos_data)
        """
        if not self.versions:
            return None

        # Version names sort chronologically due to timestamp format
        latest_name = sorted(self.versions.keys())[-1]
        return self.versions[latest_name]

    def _parse_rinex_epoch_line(self, line):
        """Parse a RINEX 3 epoch line (> Y M D h m s)."""
        try:
            parts = line.strip().split()
            # Format: > 2026 01 21 14 00 00.0000000
            y, m, d, h, mn = map(int, parts[1:6])
            s = float(parts[6])
            return datetime(y, m, d, h, mn, int(s))
        except (ValueError, IndexError):
            return None

    def _get_rinex_bounds(self, rinex_file):
        """Read the first and last observation times of a RINEX file.

        Returns tuple (start_dt, end_dt).
        """
        start_dt = None
        last_dt = None

        # Read file efficiently
        with open(rinex_file) as f:
            # 1. Find Start Time
            for line in f:
                if line.startswith(">"):
                    start_dt = self._parse_rinex_epoch_line(line)
                    if start_dt:
                        last_dt = start_dt  # Initialize last_dt
                        break

            # 2. Find End Time
            # We continue reading line by line. For massive files,
            # this takes a moment but ensures we find the true last epoch.
            for line in f:
                if line.startswith(">"):
                    dt = self._parse_rinex_epoch_line(line)
                    if dt:
                        last_dt = dt

        return start_dt, last_dt

    def _covers(self, obs: Path, start: datetime, end: datetime) -> bool:
        """Check that a RINEX file spans [start, end] (to within 2 s)."""
        first, last = self._get_rinex_bounds(obs)
        if first is None or last is None:
            return False
        return (
            bool((first - start).total_seconds() <= 2)
            and bool(end - last).total_seconds() <= 2
        )

    def _base_candidates(self) -> list[Path]:
        """Raw u-blox base recordings: the day's, then the campaign's."""
        day = self.flight_path.parent / "base"
        campaign = self.flight_path.parent.parent / "base"
        files = sorted(p for p in day.glob("*") if p.suffix.lower() == ".ubx")
        if campaign.is_dir():
            files += sorted(
                p
                for p in campaign.rglob("*")
                if p.is_file()
                and (
                    p.suffix.lower() == ".ubx"
                    or p.name.startswith("rtk_data_")
                    or (p.suffix == ".bin" and "ZED-F9P" in p.name)
                )
            )
        return files

    def _convert_base(
        self, obs_path: Path, nav_path: Path, start_rover: datetime, end_rover: datetime
    ) -> None:
        """Convert the base recording covering the rover's span to RINEX.

        The span is the rover's +-10 min; the first base recording that
        covers it is used.

        Raises
        ------
        FileNotFoundError
            If no base recording covers the rover.
        """
        start = start_rover - timedelta(minutes=10)
        end = end_rover + timedelta(minutes=10)
        date_start, time_start = start.strftime("%Y/%m/%d %H:%M:%S").split(" ")
        date_end, time_end = end.strftime("%Y/%m/%d %H:%M:%S").split(" ")
        for raw in self._base_candidates():
            subprocess.run(
                [
                    rtklib_binary("convbin"),
                    "-od",
                    "-os",
                    "-oi",
                    "-ot",
                    "-ol",
                    "-r",
                    "ubx",
                    "-ts",
                    date_start,
                    time_start,
                    "-te",
                    date_end,
                    time_end,
                    "-o",
                    str(obs_path),
                    "-n",
                    str(nav_path),
                    str(raw),
                ],
                check=True,
                capture_output=True,
            )
            if obs_path.exists() and self._covers(obs_path, start_rover, end_rover):
                logger.info(f"Base from {raw}")
                return
        raise FileNotFoundError(
            f"No base recording covers {start_rover} - {end_rover} "
            f"(looked in {self.flight_path.parent / 'base'} and the campaign's base/)"
        )

    def _reference_track(self) -> pl.DataFrame | None:
        """Return the drone's RTK track (UTC), to validate tuned configurations."""
        try:
            drone = getattr(self.flight.raw_data.drone_data, "drone", None)
            if (
                not isinstance(drone, pl.DataFrame)
                or "rtk_lat_deg" not in drone.columns
            ):
                self.flight.add_drone_data()
                drone = self.flight.raw_data.drone_data.drone
            if not isinstance(drone, pl.DataFrame):
                return None
            track = drone.select(
                "timestamp",
                pl.col("rtk_lat_deg").alias("lat_deg"),
                pl.col("rtk_lon_deg").alias("lon_deg"),
                pl.col("rtk_h_ell_m").alias("h_ell_m"),
            ).filter((pl.col("lat_deg") != 0) & pl.col("timestamp").is_not_null())
            return track if track.height else None
        except Exception as e:
            logger.warning(f"No drone RTK track to validate the tuning: {e}")
            return None

    def check_overlap(self, rover_obs, base_obs):
        """Print the time overlap of the rover and base observations."""
        print("--- 1. Time Overlap Analysis ---")

        # Get Bounds
        r_start, r_end = self._get_rinex_bounds(rover_obs)
        b_start, b_end = self._get_rinex_bounds(base_obs)

        # Basic Check
        if not r_start or not r_end:
            print(f"  [Error] Failed to read timestamps from Rover: {rover_obs}")
            return False
        if not b_start or not b_end:
            print(f"  [Error] Failed to read timestamps from Base: {base_obs}")
            return False

        print(f"  Rover: {r_start}  -->  {r_end}")
        print(f"  Base:  {b_start}  -->  {b_end}")

        # Calculate Overlap
        overlap_start = max(r_start, b_start)
        overlap_end = min(r_end, b_end)

        duration = (overlap_end - overlap_start).total_seconds()

        if duration <= 0:
            print("  [CRITICAL] NO OVERLAP DETECTED!")
            print("  The Base data ends before the Rover starts (or vice versa).")
            print("  Gap: {abs(duration):.1f} seconds")
            return False

        print(f"  [OK] Common Window: {duration:.1f} seconds ({duration / 60:.1f} min)")

        if duration < 600:  # Less than 10 mins
            print(
                "  [Warning] Overlap is very short (<10 min). Solution may be unstable."
            )

        return True

    def run_analysis(
        self,
        config_path: str | Path,
        rover_obs: str | Path | None = None,
        base_obs: str | Path | None = None,
        nav_file: str | Path | None = None,
        force: bool = False,
        analyze_rinex: bool = False,
        analyze_ppk: bool = False,
        optimize: bool = True,
        reference: pl.DataFrame | str | None = "drone",
    ) -> PPKVersion | None:
        """
        Execute RTKLIB PPK analysis with smart re-run logic.

        Smart execution:
        - Runs only if config changed (hash comparison) or force=True
        - Generates auto-timestamp version name
        - Creates revision folder with .pos, .stat, .conf files
        - Executes RTKLIB rnx2rtkp subprocess
        - Parses results using POSAnalyzer and STATAnalyzer
        - Saves to HDF5

        Parameters
        ----------
        config_path : Union[str, Path]
            Path to RTKLIB configuration file
        rover_obs : Union[str, Path], optional
            Path to rover RINEX observation file. If None, first look for available
            files in the folder otherwise use convbin for conversion (default: None)
        base_obs : Union[str, Path], optional
            Path to base RINEX observation file. If None, first look for available
            files in the folder otherwise use convbin for conversion (default: None)
        nav_file : Union[str, Path], optional
            Path to navigation file. If None, first look for available
            files in the folder otherwise use convbin for conversion (default: None)
        force : bool, optional
            If True, force re-run even if config unchanged (default: False)
        analyze_rinex : bool, optional
            If True, analyze RINEX files (default: False)
        analyze_ppk : bool, optional
            If True, analyze PPK results (default: False)
        optimize : bool, default=True
            Tune the configuration to the data first (see
            :func:`pils.analyze.ppk_tuning.tune`): ``config_path`` is the
            template, the chosen configuration is saved in the revision as
            ``optimized.conf`` and every candidate in the ``tuning`` table.
            If no candidate passes validation, the template is used.
        reference : pl.DataFrame, "drone" or None, optional
            Track used to validate the candidates' fixes: ``"drone"`` (the
            drone's RTK track, the default), a table with ``timestamp``,
            ``lat_deg``, ``lon_deg``, ``h_ell_m``, or None to compare forward
            and backward solutions instead.

        Returns
        -------
        Optional[PPKVersion]
            PPKVersion object with results, or None if execution failed

        Raises
        ------
        FileNotFoundError
            If config file doesn't exist

        Examples
        --------
        >>> ppk = PPKAnalysis('/path/to/flight')
        >>> # Smart execution - only runs if needed
        >>> v1 = ppk.run_analysis(
        ...     'rtklib.conf',
        ...     'rover.obs',
        ...     'base.obs',
        ...     'nav.nav'
        ... )
        >>> v2 = ppk.run_analysis(...)  # Skipped if config same
        >>> # Force re-run
        >>> v3 = ppk.run_analysis(..., force=True)
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Check if should run
        if not force and not self._should_run_analysis(config_path, optimize):
            logger.info("Skipping analysis - config unchanged")
            latest = self.get_latest_version()
            if latest is not None:
                return latest

        # Generate version name and create folder
        version_name = self._generate_version_name()
        revision_path = self._create_revision_folder(version_name)

        # Initialize nav file variables
        rover_nav: Path | None = None
        base_nav: Path | None = None

        if rover_obs is not None:
            rover_obs = Path(rover_obs)
            # Infer rover nav file if exists
            potential_nav = rover_obs.with_suffix(".nav")
            if potential_nav.exists():
                rover_nav = potential_nav
        else:
            obs_path = self.ppk_dir / "rover.obs"
            nav_path = self.ppk_dir / "rover.nav"

            if obs_path.exists() and nav_path.exists():
                rover_obs = obs_path
                rover_nav = nav_path

            else:
                rover_path = self.flight_path / "aux" / "sensors"

                try:
                    rover_ubx = sorted(
                        [*rover_path.glob("*_GPS.bin"), *rover_path.glob("*_GPS.ubx")]
                    )[0]
                except IndexError:
                    logger.info("Missing rover .bin file")
                    return None

                cmd = [
                    rtklib_binary("convbin"),
                    "-od",
                    "-os",
                    "-oi",
                    "-ot",
                    "-ol",
                    "-r",
                    "ubx",
                    "-o",
                    str(obs_path),
                    "-n",
                    str(nav_path),
                    str(rover_ubx),
                ]

                subprocess.run(cmd, check=True)

                if not obs_path.exists():
                    logger.info("rover obs file not created")
                    raise FileNotFoundError("Conversion failed: .obs file not created")
                if not obs_path.exists():
                    logger.info("rover nav file not created")
                    raise FileNotFoundError("Conversion failed: .nav file not created")

                rover_obs = obs_path
                rover_nav = nav_path

        start_rover, end_rover = self._get_rinex_bounds(rover_obs)

        if base_obs is not None:
            base_obs = Path(base_obs)
            # Infer base nav file if exists
            potential_nav = base_obs.with_suffix(".nav")
            if potential_nav.exists():
                base_nav = potential_nav
        else:
            obs_path = self.ppk_dir / "base.obs"
            nav_path = self.ppk_dir / "base.nav"
            if start_rover is None or end_rover is None:
                raise ValueError("Could not determine rover observation time bounds")
            cached = obs_path.exists() and nav_path.exists()
            if cached and not self._covers(obs_path, start_rover, end_rover):
                logger.info(
                    "Cached base.obs does not cover the rover: converting again"
                )
                cached = False
            if not cached:
                self._convert_base(obs_path, nav_path, start_rover, end_rover)
            base_obs = obs_path
            base_nav = nav_path

        if nav_file is not None:
            nav_file = Path(nav_file)
        else:
            # Use the larger navigation file if both exist
            if base_nav and rover_nav:
                nav_file = max(base_nav, rover_nav, key=lambda p: p.stat().st_size)
            elif base_nav:
                nav_file = base_nav
            elif rover_nav:
                nav_file = rover_nav
            else:
                raise FileNotFoundError("No navigation file available (base or rover)")

        if not rover_obs.exists():
            raise FileNotFoundError(f"Rover observation file not found: {rover_obs}")
        if not base_obs.exists():
            raise FileNotFoundError(f"Base observation file not found: {base_obs}")
        if not nav_file.exists():
            raise FileNotFoundError(f"Navigation file not found: {nav_file}")

        if analyze_rinex:
            obs_files = [rover_obs, base_obs]
            nav_files = [rover_nav, base_nav]

            names = ["rover", "base"]

            for i, obs in enumerate(obs_files):
                report_md = self.ppk_dir / f"report_{names[i]}.md"

                if report_md.exists():
                    continue
                else:
                    report = RINEXReport(obs, nav_files[i])

                    plot_folder = self.ppk_dir / f"rinex_plots_{names[i]}"

                    report.generate(str(report_md), str(plot_folder.name))

        logger.info(f"Running PPK analysis: {version_name}")
        tuning = None
        if optimize:
            track = self._reference_track() if isinstance(reference, str) else reference
            tuning = tune(
                config_path,
                rover_obs,
                base_obs,
                nav_file,
                revision_path / "tuning",
                reference=track,
            )
            shutil.rmtree(revision_path / "tuning", ignore_errors=True)
        if tuning is not None and tuning.config_text is not None:
            config_copy = revision_path / "optimized.conf"
            config_copy.write_text(tuning.config_text)
            logger.info(f"Tuned config saved to {config_copy}")
        else:
            if optimize:
                logger.warning(
                    "Tuning found no valid configuration: using the template"
                )
            # Copy config to revision folder
            config_copy = revision_path / config_path.name
            shutil.copy2(config_path, config_copy)
            logger.info(f"Copied config to {config_copy}")

        # Define output file paths
        pos_file = revision_path / "solution.pos"
        stat_file = revision_path / "solution.pos.stat"

        cmd = [
            rtklib_binary("rnx2rtkp"),
            "-k",
            str(config_copy),
            "-o",
            str(pos_file),
            str(rover_obs),
            str(base_obs),
            str(nav_file),
        ]

        subprocess.run(cmd, check=True)

        if analyze_ppk:
            report_md = revision_path / "report_ppk.md"

            report = RTKLIBReport(pos_file, stat_file)

            plot_folder = revision_path / "plots"

            report.generate(str(report_md.parent), str(plot_folder.name))

        # Check if output files were created
        if not pos_file.exists():
            logger.error(f"Position file not created: {pos_file}")
            # Clean up revision folder on failure
            if revision_path.exists():
                shutil.rmtree(revision_path)
                logger.info(f"Cleaned up failed revision folder: {revision_path}")
            return None

        # Parse position file
        try:
            pos_analyzer = POSAnalyzer(str(pos_file))
            pos_data = pos_analyzer.parse()
            logger.info(f"Parsed position: {len(pos_data)} epochs")
        except Exception as e:
            logger.error(f"Failed to parse position file: {e}")
            # Clean up revision folder on failure
            if revision_path.exists():
                shutil.rmtree(revision_path)
                logger.info(f"Cleaned up failed revision folder: {revision_path}")
            return None

        # Parse statistics file (optional)
        if not stat_file.exists():
            logger.warning(f"Statistics file not created: {stat_file}")
            stat_data = None
        else:
            try:
                stat_analyzer = STATAnalyzer(str(stat_file))
                stat_data = stat_analyzer.parse()
                logger.info(f"Parsed statistics: {len(stat_data)} records")
            except Exception as e:
                logger.error(f"Failed to parse statistics file: {e}")
                stat_data = None

        # Create metadata with config hash and parsed parameters
        config_hash = self._hash_config(config_path) + (":optimize" if optimize else "")
        config_params = self._parse_config_params(config_copy)
        metadata = {
            "config_hash": config_hash,
            "timestamp": datetime.now().isoformat(),
            "config_params": config_params,
            "rover_obs": str(rover_obs),
            "base_obs": str(base_obs),
            "nav_file": str(nav_file),
            "pos_file": str(pos_file),
            "stat_file": str(stat_file) if stat_data is not None else None,
            "config_template": str(config_path),
            "config_used": str(config_copy),
            **(tuning.metadata() if tuning is not None else {}),
        }

        # Create version object (handle None stat_data)
        if stat_data is None:
            stat_data = pl.DataFrame()

        version = PPKVersion(
            version_name=version_name,
            pos_data=pos_data,
            stat_data=stat_data,
            metadata=metadata,
            revision_path=revision_path,
            tuning_data=tuning.table() if tuning is not None else None,
        )

        # Store in versions dict
        self.versions[version_name] = version

        # Save automatically
        self._save_version(version)

        logger.info(f"PPK analysis complete: {version_name}")
        return version

    def _save_version(self, version: PPKVersion) -> Path:
        """Save a version: ``ppk/<version>/`` (see :mod:`pils.store`).

        The tables (``position``, ``statistics``) and the manifest go next to
        the RTKLIB files of the revision.
        """
        path = store.save_version(
            self.ppk_dir,
            version.version_name,
            {
                "position": version.pos_data,
                "statistics": version.stat_data,
                **(
                    {"tuning": version.tuning_data}
                    if version.tuning_data is not None
                    else {}
                ),
            },
            {
                "config_hash": version.metadata.get("config_hash"),
                "timestamp": version.metadata.get("timestamp"),
                "config_params": version.metadata.get("config_params", {}),
                "revision_path": str(version.revision_path),
                **{
                    k: v
                    for k, v in version.metadata.items()
                    if k not in ("config_hash", "timestamp", "config_params")
                },
            },
            kind="ppk",
        )
        logger.info(f"Saved version {version.version_name} to {path}")
        return path

    def _load_version(self, version_name: str) -> PPKVersion:
        """Load a saved version and add it to :attr:`versions`."""
        tables, metadata = store.load_version(self.ppk_dir, version_name)
        revision_path = metadata.pop("revision_path", None)
        version = PPKVersion(
            version_name=version_name,
            pos_data=tables["position"],  # type: ignore[arg-type]
            stat_data=tables["statistics"],  # type: ignore[arg-type]
            metadata=metadata,
            tuning_data=tables.get("tuning"),  # type: ignore[arg-type]
            revision_path=Path(revision_path)
            if revision_path
            else self.ppk_dir / version_name,
        )
        self.versions[version_name] = version
        return version

    def _load_version_from_hdf5(self, version_name: str) -> PPKVersion:
        """
        Load a single PPKVersion from the legacy ppk_solution.h5 file.

        Reconstructs a PPKVersion object from HDF5 storage, loading position
        data, statistics data, and metadata. Adds loaded version to self.versions dict.

        Parameters
        ----------
        version_name : str
            Name of version to load (e.g., 'rev_20260204_143022')

        Returns
        -------
        PPKVersion
            Loaded PPKVersion object

        Raises
        ------
        KeyError
            If version_name not found in HDF5 file

        Examples
        --------
        >>> ppk = PPKAnalysis('/path/to/flight')
        >>> version = ppk._load_version_from_hdf5('rev_20260204_143022')
        >>> print(version.pos_data)
        """
        import h5py

        with h5py.File(self.hdf5_path, "r") as f:
            if version_name not in f:
                raise KeyError(f"Version {version_name} not found in HDF5")

            version_group = f[version_name]
            if not isinstance(version_group, h5py.Group):
                raise TypeError(f"{version_name} is not a group in HDF5 file")

            # Load position DataFrame
            pos_group = version_group["position"]
            if not isinstance(pos_group, h5py.Group):
                raise TypeError("position is not a group in HDF5 file")
            pos_data = self._load_dataframe_from_hdf5(pos_group)

            # Load statistics DataFrame
            stat_group = version_group["statistics"]
            if not isinstance(stat_group, h5py.Group):
                raise TypeError("statistics is not a group in HDF5 file")
            stat_data = self._load_dataframe_from_hdf5(stat_group)

            pos_data, stat_data = legacy_ppk_tables(pos_data, stat_data)

            # Load metadata from attrs
            from pils.flight import _deserialize_from_hdf5

            config_hash = _deserialize_from_hdf5(version_group.attrs.get("config_hash"))
            timestamp = _deserialize_from_hdf5(version_group.attrs.get("timestamp"))
            config_params = _deserialize_from_hdf5(
                version_group.attrs.get("config_params"), hint="dict"
            )
            revision_path_str = _deserialize_from_hdf5(
                version_group.attrs.get("revision_path")
            )

            metadata = {
                "config_hash": config_hash,
                "timestamp": timestamp,
                "config_params": config_params if config_params else {},
            }

            # Reconstruct PPKVersion
            version = PPKVersion(
                version_name=version_name,
                pos_data=pos_data,
                stat_data=stat_data,
                metadata=metadata,
                revision_path=(
                    Path(revision_path_str)
                    if revision_path_str
                    else self.ppk_dir / version_name
                ),
            )

            # Add to versions dict
            self.versions[version_name] = version

            return version

    def _load_dataframe_from_hdf5(self, dataset_group: h5py.Group) -> pl.DataFrame:
        """
        Load a Polars DataFrame from HDF5 (see :func:`pils.utils.hdf5.read_table`).

        Parameters
        ----------
        dataset_group : h5py.Group
            HDF5 group containing column datasets

        Returns
        -------
        pl.DataFrame
            Reconstructed Polars DataFrame
        """
        if "columns" not in dataset_group.attrs:
            return pl.DataFrame()
        # Older PPK files stored datetimes as microseconds whatever their unit
        return read_table(dataset_group, legacy_time_unit="us")

    @classmethod
    def load(cls, flight: Flight) -> PPKAnalysis:
        """
        PPKAnalysis with the saved versions of a flight.

        Loads the versions in ``proc/ppk/rev_*/`` (see :mod:`pils.store`) and,
        for results saved before, the ones in ``ppk_solution.h5``.

        Parameters
        ----------
        flight : Flight
            Flight object with valid flight_path attribute

        Returns
        -------
        PPKAnalysis
            PPKAnalysis instance with loaded versions

        Raises
        ------
        TypeError
            If flight is not a Flight object
        ValueError
            If flight.flight_path is None or not an existing directory

        Examples
        --------
        >>> ppk = PPKAnalysis.load(flight)
        >>> print(f"Loaded {len(ppk.versions)} versions")
        """
        ppk = cls(flight)
        for version_name in store.list_versions(ppk.ppk_dir):
            try:
                ppk._load_version(version_name)
            except Exception as e:
                logger.warning(f"Failed to load version {version_name}: {e}")
        if ppk.hdf5_path.exists():
            import h5py

            with h5py.File(ppk.hdf5_path, "r") as f:
                legacy = [name for name in f.keys() if name not in ppk.versions]
            for version_name in legacy:
                try:
                    ppk._load_version_from_hdf5(version_name)
                except Exception as e:
                    logger.warning(f"Failed to load version {version_name}: {e}")
        logger.info(f"Loaded {len(ppk.versions)} versions")
        return ppk

    @classmethod
    def from_hdf5(cls, flight: Flight) -> PPKAnalysis:
        """Load the PPK results, as :meth:`load` (kept for older code)."""
        return cls.load(flight)

    def list_versions(self) -> list[str]:
        """
        Return list of all version names in chronological order.

        Version names use timestamp format (rev_YYYYMMDD_HHMMSS) which ensures
        alphabetical sorting equals chronological sorting.

        Returns
        -------
        List[str]
            Sorted list of version names

        Examples
        --------
        >>> ppk = PPKAnalysis.from_hdf5('/path/to/flight')
        >>> versions = ppk.list_versions()
        >>> print(versions)
        ['rev_20260204_140000', 'rev_20260204_150000', 'rev_20260204_160000']
        """
        return sorted(list(self.versions.keys()))

    def get_version(self, version_name: str) -> PPKVersion | None:
        """
        Get specific version by name.

        Parameters
        ----------
        version_name : str
            Version identifier (e.g., 'rev_20260204_143022')

        Returns
        -------
        Optional[PPKVersion]
            PPKVersion if found, None otherwise

        Examples
        --------
        >>> ppk = PPKAnalysis.from_hdf5('/path/to/flight')
        >>> version = ppk.get_version('rev_20260204_143022')
        >>> if version:
        ...     print(version.pos_data)
        """
        return self.versions.get(version_name)

    def delete_version(self, version_name: str) -> None:
        """
        Delete a version from HDF5 and filesystem.

        Removes version from:
        1. self.versions dict
        2. HDF5 file (deletes group)
        3. Filesystem (deletes revision folder)

        Parameters
        ----------
        version_name : str
            Version identifier to delete

        Examples
        --------
        >>> ppk = PPKAnalysis.from_hdf5('/path/to/flight')
        >>> ppk.delete_version('rev_20260204_140000')
        >>> print(ppk.list_versions())  # Version gone
        """
        import shutil

        import h5py

        # Remove from versions dict
        if version_name in self.versions:
            version = self.versions.pop(version_name)

            # Delete revision folder from filesystem
            if version.revision_path.exists():
                shutil.rmtree(version.revision_path)
                logger.info(f"Deleted revision folder: {version.revision_path}")

        # Delete from the legacy HDF5 file
        if self.hdf5_path.exists():
            with h5py.File(self.hdf5_path, "a") as f:
                if version_name in f:
                    del f[version_name]
                    logger.info(f"Deleted version {version_name} from HDF5")

        logger.info(f"Version {version_name} deleted")
