"""
PPK Analysis - Standalone Post-Processed Kinematic GPS Analysis.

This module provides standalone PPK analysis using RTKLIB with smart
execution logic, versioned storage, and HDF5 persistence.

Key Features:
- Smart re-run: Only executes if config changed or forced
- Auto-timestamp versioning (rev_YYYYMMDD_HHMMSS)
- Per-revision folder organization
- HDF5 storage for all versions
- Complete separation from Flight class

Example:
    >>> from pils.analyze.ppk import PPKAnalysis
    >>>
    >>> # Create new analysis
    >>> ppk = PPKAnalysis('/path/to/flight')
    >>> ppk.run_analysis('config.conf')  # Only runs if config changed
    >>>
    >>> # Load existing
    >>> ppk = PPKAnalysis.from_hdf5('/path/to/flight')
    >>> latest = ppk.get_latest_version()
    >>> print(latest.pos_data)  # Polars DataFrame
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl

from pils.analyze.rtkdata.analyze_rtk import RTKLIBRunner
from pils.analyze.rtkdata.RTKLIB.pos_analyzer import POSAnalyzer
from pils.analyze.rtkdata.RTKLIB.stat_analyzer import STATAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class PPKVersion:
    """
    Container for a single PPK analysis revision.

    Each revision represents one execution of RTKLIB with specific
    configuration parameters, storing position solutions, statistics,
    and metadata.

    Attributes:
        version_name: Auto-timestamp version identifier (rev_YYYYMMDD_HHMMSS)
        pos_data: Position solution DataFrame from .pos file
        stat_data: Processing statistics DataFrame from .pos.stat file
        metadata: Config hash, parsed parameters, and execution metadata
        revision_path: Path to revision folder containing .pos, .stat, .conf files

    Example:
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
    metadata: Dict[str, Any]
    revision_path: Path


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

    File Structure:
        flight_dir/proc/ppk/
        ├── ppk_solution.h5              # HDF5 with all versions
        └── rev_20260204_143022/         # Per-revision folder
            ├── solution.pos              # RTKLIB position output
            ├── solution.pos.stat         # RTKLIB statistics output
            └── config.conf               # RTKLIB config used

    Attributes:
        flight_path: Root flight directory
        ppk_dir: PPK directory ({flight_path}/proc/ppk/)
        hdf5_path: Path to ppk_solution.h5 file
        versions: Dictionary of loaded PPK versions

    Example:
        >>> # Create new analysis
        >>> ppk = PPKAnalysis('/path/to/flight')
        >>> version = ppk.run_analysis('rtklib.conf')  # Smart execution
        >>>
        >>> # Only runs if config changed
        >>> version2 = ppk.run_analysis('rtklib.conf')  # Skipped if same
        >>>
        >>> # Force re-run
        >>> version3 = ppk.run_analysis('rtklib.conf', force=True)
        >>>
        >>> # Access versions
        >>> latest = ppk.get_latest_version()
        >>> all_versions = ppk.list_versions()
    """

    def __init__(self, flight_path: Union[str, Path]):
        """
        Initialize PPKAnalysis for a flight.

        Creates the proc/ppk directory structure if it doesn't exist.

        Args:
            flight_path: Path to flight root directory

        Example:
            >>> ppk = PPKAnalysis('/mnt/data/flight_001')
            >>> print(ppk.ppk_dir)
            /mnt/data/flight_001/proc/ppk
        """
        self.flight_path = Path(flight_path)
        self.ppk_dir = self.flight_path / "proc" / "ppk"
        self.hdf5_path = self.ppk_dir / "ppk_solution.h5"
        self.versions: Dict[str, PPKVersion] = {}

        # Create PPK directory structure
        self.ppk_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized PPKAnalysis for flight: {self.flight_path}")
        logger.info(f"PPK directory: {self.ppk_dir}")

    def _hash_config(self, config_path: Path) -> str:
        """
        Generate SHA256 hash of config file for change detection.

        Reads the entire config file and computes its SHA256 hash.
        Used to detect if configuration has changed since last run.

        Args:
            config_path: Path to RTKLIB config file

        Returns:
            64-character hexadecimal SHA256 hash

        Example:
            >>> ppk = PPKAnalysis('/path/to/flight')
            >>> hash1 = ppk._hash_config(Path('config1.conf'))
            >>> hash2 = ppk._hash_config(Path('config2.conf'))
            >>> hash1 == hash2  # True if configs identical
        """
        content = config_path.read_bytes()
        return hashlib.sha256(content).hexdigest()

    def _parse_config_params(self, config_path: Path) -> Dict[str, Any]:
        """
        Parse key RTKLIB configuration parameters.

        Extracts important configuration parameters from RTKLIB config file.
        Only parses key parameters (not full JSON dump), focusing on:
        - pos1-posmode: Processing mode (kinematic/static)
        - pos1-elmask: Elevation mask
        - pos2-armode: Ambiguity resolution mode
        - ant2-postype: Antenna position type

        Args:
            config_path: Path to RTKLIB config file

        Returns:
            Dictionary of key-value config parameters

        Example:
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

        Returns:
            Version name string

        Example:
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

        Args:
            version_name: Version name (rev_YYYYMMDD_HHMMSS)

        Returns:
            Path to created revision folder

        Example:
            >>> ppk = PPKAnalysis('/path/to/flight')
            >>> folder = ppk._create_revision_folder('rev_20260204_143022')
            >>> print(folder)
            /path/to/flight/proc/ppk/rev_20260204_143022
        """
        revision_path = self.ppk_dir / version_name
        revision_path.mkdir(parents=True, exist_ok=True)
        return revision_path

    def _should_run_analysis(self, config_path: Path) -> bool:
        """
        Determine if RTKLIB analysis should run.

        Smart execution logic:
        1. Run if no HDF5 file exists (first run)
        2. Run if no versions loaded (no previous runs)
        3. Run if config hash differs from latest version
        4. Skip if config unchanged

        Args:
            config_path: Path to RTKLIB config file

        Returns:
            True if analysis should run, False to skip

        Example:
            >>> ppk = PPKAnalysis('/path/to/flight')
            >>> should_run = ppk._should_run_analysis(Path('config.conf'))
            >>> if should_run:
            ...     print("Config changed, running analysis")
            ... else:
            ...     print("Config unchanged, skipping")
        """
        # Run if no HDF5 file exists (first run)
        if not self.hdf5_path.exists():
            logger.info("No HDF5 file found - running analysis")
            return True

        # Run if no versions loaded
        if not self.versions:
            logger.info("No previous versions found - running analysis")
            return True

        # Get latest version and compare config hash
        latest = self.get_latest_version()
        if latest is None:
            return True

        current_hash = self._hash_config(config_path)
        previous_hash = latest.metadata.get("config_hash")

        if current_hash != previous_hash:
            logger.info("Config changed - running analysis")
            return True

        logger.info("Config unchanged - skipping analysis")
        return False

    def get_latest_version(self) -> Optional[PPKVersion]:
        """
        Get the most recent PPK version.

        Returns the version with the latest timestamp based on
        version name sorting (rev_YYYYMMDD_HHMMSS format sorts chronologically).

        Returns:
            Latest PPKVersion or None if no versions exist

        Example:
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

    def run_analysis(
        self,
        config_path: Union[str, Path],
        rover_obs: Union[str, Path],
        base_obs: Union[str, Path],
        nav_file: Union[str, Path],
        force: bool = False,
        rnx2rtkp_path: str = "rnx2rtkp",
    ) -> Optional[PPKVersion]:
        """
        Execute RTKLIB PPK analysis with smart re-run logic.

        Smart execution:
        - Runs only if config changed (hash comparison) or force=True
        - Generates auto-timestamp version name
        - Creates revision folder with .pos, .stat, .conf files
        - Executes RTKLIB rnx2rtkp subprocess
        - Parses results using POSAnalyzer and STATAnalyzer
        - Saves to HDF5

        Args:
            config_path: Path to RTKLIB configuration file
            rover_obs: Path to rover RINEX observation file
            base_obs: Path to base RINEX observation file
            nav_file: Path to navigation file
            force: If True, force re-run even if config unchanged
            rnx2rtkp_path: Path to rnx2rtkp binary (default: "rnx2rtkp")

        Returns:
            PPKVersion object with results, or None if execution failed

        Raises:
            FileNotFoundError: If config file doesn't exist

        Example:
            >>> ppk = PPKAnalysis('/path/to/flight')
            >>>
            >>> # Smart execution - only runs if needed
            >>> v1 = ppk.run_analysis(
            ...     'rtklib.conf',
            ...     'rover.obs',
            ...     'base.obs',
            ...     'nav.nav'
            ... )
            >>> v2 = ppk.run_analysis(...)  # Skipped if config same
            >>>
            >>> # Force re-run
            >>> v3 = ppk.run_analysis(..., force=True)
        """
        config_path = Path(config_path)
        rover_obs = Path(rover_obs)
        base_obs = Path(base_obs)
        nav_file = Path(nav_file)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not rover_obs.exists():
            raise FileNotFoundError(f"Rover observation file not found: {rover_obs}")
        if not base_obs.exists():
            raise FileNotFoundError(f"Base observation file not found: {base_obs}")
        if not nav_file.exists():
            raise FileNotFoundError(f"Navigation file not found: {nav_file}")

        # Check if should run
        if not force and not self._should_run_analysis(config_path):
            logger.info("Skipping analysis - config unchanged")
            latest = self.get_latest_version()
            if latest is not None:
                return latest

        # Generate version name and create folder
        version_name = self._generate_version_name()
        revision_path = self._create_revision_folder(version_name)

        logger.info(f"Running PPK analysis: {version_name}")

        # Copy config to revision folder
        config_copy = revision_path / config_path.name
        shutil.copy2(config_path, config_copy)
        logger.info(f"Copied config to {config_copy}")

        # Define output file paths
        pos_file = revision_path / "solution.pos"
        stat_file = revision_path / "solution.pos.stat"

        # Initialize RTKLIB runner
        rtklib = RTKLIBRunner(rnx2rtkp_path=rnx2rtkp_path)

        # Check temporal overlap between rover and base
        if not rtklib.check_overlap(str(rover_obs), str(base_obs)):
            logger.error("No temporal overlap between rover and base observations")
            return None

        # Execute RTKLIB processing
        try:
            rtklib.run_ppk(
                rover=str(rover_obs),
                base=str(base_obs),
                nav=str(nav_file),
                config=str(config_copy),
                output=str(pos_file),
            )
            logger.info(f"RTKLIB execution completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"RTKLIB execution failed: {e}")
            return None
        except FileNotFoundError as e:
            logger.error(f"RTKLIB binary not found: {e}")
            return None

        # Check if output files were created
        if not pos_file.exists():
            logger.error(f"Position file not created: {pos_file}")
            return None

        # Parse position file
        try:
            pos_analyzer = POSAnalyzer(str(pos_file))
            pos_data = pos_analyzer.parse()
            logger.info(f"Parsed position: {len(pos_data)} epochs")
        except Exception as e:
            logger.error(f"Failed to parse position file: {e}")
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
        config_hash = self._hash_config(config_path)
        config_params = self._parse_config_params(config_path)
        metadata = {
            "config_hash": config_hash,
            "timestamp": datetime.now().isoformat(),
            "config_params": config_params,
            "rover_obs": str(rover_obs),
            "base_obs": str(base_obs),
            "nav_file": str(nav_file),
            "pos_file": str(pos_file),
            "stat_file": str(stat_file) if stat_data is not None else None,
        }

        # Create version object
        version = PPKVersion(
            version_name=version_name,
            pos_data=pos_data,
            stat_data=stat_data,
            metadata=metadata,
            revision_path=revision_path,
        )

        # Store in versions dict
        self.versions[version_name] = version

        # Save to HDF5 automatically (Phase 3)
        self._save_version_to_hdf5(version)

        logger.info(f"PPK analysis complete: {version_name}")
        return version

    def _save_version_to_hdf5(self, version: PPKVersion) -> None:
        """
        Save a single PPKVersion to HDF5 file.

        Creates or opens ppk_solution.h5 and saves the version's position data,
        statistics data, and metadata. Follows Flight.py pattern for DataFrame
        serialization with column-wise datasets.

        HDF5 Structure:
            /{version_name}/
            ├── position/           # POSAnalyzer DataFrame columns
            │   ├── timestamp (dataset)
            │   ├── lat (dataset)
            │   └── ...
            ├── statistics/         # STATAnalyzer DataFrame columns
            │   ├── timestamp (dataset)
            │   ├── num_sat (dataset)
            │   └── ...
            └── attrs:              # Group attributes
                ├── config_hash
                ├── timestamp
                ├── config_params (JSON string)
                └── revision_path

        Args:
            version: PPKVersion to save

        Example:
            >>> ppk = PPKAnalysis('/path/to/flight')
            >>> version = PPKVersion(...)
            >>> ppk._save_version_to_hdf5(version)
        """
        import h5py

        # Create/open HDF5 file
        with h5py.File(self.hdf5_path, "a") as f:
            # Create version group (remove if exists)
            if version.version_name in f:
                del f[version.version_name]

            version_group = f.create_group(version.version_name)

            # Save position DataFrame
            self._save_dataframe_to_hdf5(version_group, "position", version.pos_data)

            # Save statistics DataFrame
            self._save_dataframe_to_hdf5(version_group, "statistics", version.stat_data)

            # Save metadata as group attributes
            from pils.flight import _serialize_for_hdf5

            version_group.attrs["config_hash"] = _serialize_for_hdf5(
                version.metadata.get("config_hash")
            )
            version_group.attrs["timestamp"] = _serialize_for_hdf5(
                version.metadata.get("timestamp")
            )
            version_group.attrs["config_params"] = _serialize_for_hdf5(
                version.metadata.get("config_params", {})
            )
            version_group.attrs["revision_path"] = _serialize_for_hdf5(str(version.revision_path))

        logger.info(f"Saved version {version.version_name} to HDF5")

    def _save_dataframe_to_hdf5(
        self, parent_group: "h5py.Group", name: str, df: pl.DataFrame
    ) -> None:
        """
        Save a Polars DataFrame to HDF5 using column-wise storage.

        Follows Flight.py pattern exactly: each DataFrame column becomes a
        separate HDF5 dataset, with metadata (columns, dtypes, n_rows) stored
        as group attributes.

        Args:
            parent_group: Parent HDF5 group
            name: Name for the column group
            df: Polars DataFrame to save

        Example:
            >>> with h5py.File('test.h5', 'w') as f:
            ...     ppk._save_dataframe_to_hdf5(f, 'position', pos_df)
        """
        import json

        # Remove existing if present
        if name in parent_group:
            del parent_group[name]

        # Create group for columns
        column_group = parent_group.create_group(name)

        # Save each column as separate dataset
        for col_name in df.columns:
            col_data = df[col_name].to_numpy()
            if col_name in column_group:
                del column_group[col_name]
            column_group.create_dataset(col_name, data=col_data)

        # Save metadata as attrs
        column_group.attrs["columns"] = json.dumps(df.columns)
        column_group.attrs["dtypes"] = json.dumps([str(dtype) for dtype in df.dtypes])
        column_group.attrs["n_rows"] = len(df)

    def _load_version_from_hdf5(self, version_name: str) -> PPKVersion:
        """
        Load a single PPKVersion from HDF5 file.

        Reconstructs a PPKVersion object from HDF5 storage, loading position
        data, statistics data, and metadata. Adds loaded version to self.versions dict.

        Args:
            version_name: Name of version to load (e.g., 'rev_20260204_143022')

        Returns:
            Loaded PPKVersion object

        Raises:
            KeyError: If version_name not found in HDF5 file

        Example:
            >>> ppk = PPKAnalysis('/path/to/flight')
            >>> version = ppk._load_version_from_hdf5('rev_20260204_143022')
            >>> print(version.pos_data)
        """
        import h5py

        with h5py.File(self.hdf5_path, "r") as f:
            if version_name not in f:
                raise KeyError(f"Version {version_name} not found in HDF5")

            version_group = f[version_name]

            # Load position DataFrame
            pos_data = self._load_dataframe_from_hdf5(version_group["position"])

            # Load statistics DataFrame
            stat_data = self._load_dataframe_from_hdf5(version_group["statistics"])

            # Load metadata from attrs
            from pils.flight import _deserialize_from_hdf5

            config_hash = _deserialize_from_hdf5(version_group.attrs.get("config_hash"))
            timestamp = _deserialize_from_hdf5(version_group.attrs.get("timestamp"))
            config_params = _deserialize_from_hdf5(
                version_group.attrs.get("config_params"), hint="dict"
            )
            revision_path_str = _deserialize_from_hdf5(version_group.attrs.get("revision_path"))

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
                    Path(revision_path_str) if revision_path_str else self.ppk_dir / version_name
                ),
            )

            # Add to versions dict
            self.versions[version_name] = version

            return version

    def _load_dataframe_from_hdf5(self, dataset_group: "h5py.Group") -> pl.DataFrame:
        """
        Load a Polars DataFrame from HDF5 dataset group.

        Reconstructs DataFrame from column-wise HDF5 datasets. Follows Flight.py
        pattern for deserialization.

        Args:
            dataset_group: HDF5 group containing column datasets

        Returns:
            Reconstructed Polars DataFrame

        Example:
            >>> with h5py.File('test.h5', 'r') as f:
            ...     df = ppk._load_dataframe_from_hdf5(f['position'])
        """
        import json

        if "columns" not in dataset_group.attrs:
            return pl.DataFrame()

        columns_attr = dataset_group.attrs["columns"]
        # Handle different attr types (bytes or string)
        if isinstance(columns_attr, bytes):
            columns = json.loads(columns_attr.decode())
        else:
            columns = json.loads(str(columns_attr))

        data_dict = {}

        for col_name in columns:
            if col_name in dataset_group:
                col_dataset = dataset_group[col_name]
                data_dict[col_name] = col_dataset[:]

        if not data_dict:
            return pl.DataFrame()

        return pl.DataFrame(data_dict)

    @classmethod
    def from_hdf5(cls, flight_path: Union[str, Path]) -> "PPKAnalysis":
        """
        Load existing PPKAnalysis from HDF5 file.

        Creates a PPKAnalysis instance and loads all stored versions from
        ppk_solution.h5. If no HDF5 file exists, returns empty PPKAnalysis.

        Args:
            flight_path: Path to flight root directory

        Returns:
            PPKAnalysis instance with loaded versions

        Example:
            >>> ppk = PPKAnalysis.from_hdf5('/path/to/flight')
            >>> print(f"Loaded {len(ppk.versions)} versions")
            >>> for version_name in ppk.list_versions():
            ...     print(f"  - {version_name}")
        """
        import h5py

        ppk = cls(flight_path)

        # Check if HDF5 file exists
        if not ppk.hdf5_path.exists():
            logger.info("No HDF5 file found, returning empty PPKAnalysis")
            return ppk

        # Load all version groups
        with h5py.File(ppk.hdf5_path, "r") as f:
            for version_name in f.keys():
                try:
                    ppk._load_version_from_hdf5(version_name)
                    logger.info(f"Loaded version: {version_name}")
                except Exception as e:
                    logger.warning(f"Failed to load version {version_name}: {e}")

        logger.info(f"Loaded {len(ppk.versions)} versions from HDF5")
        return ppk

    def list_versions(self) -> List[str]:
        """
        Return list of all version names in chronological order.

        Version names use timestamp format (rev_YYYYMMDD_HHMMSS) which ensures
        alphabetical sorting equals chronological sorting.

        Returns:
            Sorted list of version names

        Example:
            >>> ppk = PPKAnalysis.from_hdf5('/path/to/flight')
            >>> versions = ppk.list_versions()
            >>> print(versions)
            ['rev_20260204_140000', 'rev_20260204_150000', 'rev_20260204_160000']
        """
        return sorted(list(self.versions.keys()))

    def get_version(self, version_name: str) -> Optional[PPKVersion]:
        """
        Get specific version by name.

        Args:
            version_name: Version identifier (e.g., 'rev_20260204_143022')

        Returns:
            PPKVersion if found, None otherwise

        Example:
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

        Args:
            version_name: Version identifier to delete

        Example:
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

        # Delete from HDF5 file
        if self.hdf5_path.exists():
            with h5py.File(self.hdf5_path, "a") as f:
                if version_name in f:
                    del f[version_name]
                    logger.info(f"Deleted version {version_name} from HDF5")

        logger.info(f"Version {version_name} deleted")
