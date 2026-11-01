"""The PPK revisions of a flight: config hashing, saving, loading, listing, deleting."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import polars as pl

from pils import store
from pils.analyze.ppk.versions import PPKVersion, legacy_ppk_tables
from pils.flight import Flight
from pils.utils.hdf5 import read_table

if TYPE_CHECKING:
    from pils.analyze.ppk.core import PPKAnalysis

logger = logging.getLogger(__name__)


class PPKStorageMixin:
    """Revision management of :class:`~pils.analyze.ppk.PPKAnalysis`."""

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
        return store.new_version_name()

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

    def _should_run_analysis(
        self, config_path: Path, optimize: bool = False, selection: str = "fix_rate"
    ) -> bool:
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
            f":optimize:{selection}" if optimize else ""
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
                # Broad: one unreadable revision does not hide the others
                logger.warning(f"Failed to load version {version_name}: {e}")
        if ppk.hdf5_path.exists():
            import h5py

            with h5py.File(ppk.hdf5_path, "r") as f:
                legacy = [name for name in f.keys() if name not in ppk.versions]
            for version_name in legacy:
                try:
                    ppk._load_version_from_hdf5(version_name)
                except Exception as e:  # Broad: as above, for the legacy HDF5 revisions
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
