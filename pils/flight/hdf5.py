"""Flight files in the single-HDF5 format (``Flight.to_hdf5`` / ``from_hdf5``).

The format written before :mod:`pils.store`; still read for old files.
"""

import json
import logging
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import h5py
import polars as pl

from pils import schema, store
from pils.flight.containers import DroneData, PayloadData
from pils.integrations.stout import layout
from pils.utils.hdf5 import read_table, write_table

if TYPE_CHECKING:
    from pils.flight.core import Flight

logger = logging.getLogger(__name__)


def _get_current_timestamp() -> str:
    """
    Get current timestamp in rev_YYYYMMDD_hhmmss format.

    Returns
    -------
    str
        Timestamp string with seconds precision
    """
    return store.new_version_name()


def _get_package_version() -> str:
    """
    Get PILS package version.

    Returns
    -------
    str
        Version string or 'unknown'
    """
    import pils

    return getattr(pils, "__version__", "unknown")


_LEGACY_SYNC_COORDINATES = {
    "latitude": "lat_deg",
    "longitude": "lon_deg",
    "altitude": "h_ell_m",
}
_SYNC_KINDS = {"drone": "dji", "reference_gps": "gps"}


def _legacy_sync_table(key: str, df: pl.DataFrame) -> tuple[str, pl.DataFrame]:
    """Rename a synchronized table from a file written before the schema."""
    df = df.rename(
        {k: v for k, v in _LEGACY_SYNC_COORDINATES.items() if k in df.columns}
    )
    if key.startswith("drone_"):
        record = key.removeprefix("drone_")
        return f"drone_{schema.record_name(record)}", schema.canonical(
            df, record=record
        )
    kind = _SYNC_KINDS.get(key, key.split("_")[0])  # imu_barometer -> imu
    return key, schema.canonical(df, kind)


def _parse_temporal_dtype(dtype: str) -> pl.DataType:
    """Polars temporal dtype from its string form, e.g. ``Datetime(time_unit='us', time_zone='UTC')``."""
    unit_match = re.search(r"time_unit='(\w+)'", dtype)
    unit = unit_match.group(1) if unit_match else "us"
    if dtype.startswith("Datetime"):
        zone_match = re.search(r"time_zone='([^']+)'", dtype)
        return pl.Datetime(unit, zone_match.group(1) if zone_match else None)  # type: ignore[arg-type]
    if dtype.startswith("Duration"):
        return pl.Duration(unit)  # type: ignore[arg-type]
    if dtype.startswith("Date"):
        return pl.Date()
    return pl.Time()


def _serialize_for_hdf5(obj: Any) -> Any:
    """
    Convert Python objects to HDF5-compatible types for attrs.

    Parameters
    ----------
    obj : Any
        Object to serialize

    Returns
    -------
    Any
        HDF5-compatible object
    """
    if obj is None:
        return "None"
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, dict):
        return json.dumps(obj)
    elif isinstance(obj, (list, tuple)):
        return json.dumps(list(obj))
    else:
        return str(obj)


def _deserialize_from_hdf5(value: Any, hint: str | None = None) -> Any:
    """
    Deserialize values from HDF5 attrs.

    Parameters
    ----------
    value : Any
        Value to deserialize
    hint : Optional[str], default=None
        Type hint (e.g., 'dict', 'list')

    Returns
    -------
    Any
        Deserialized object
    """
    if value == "None":
        return None
    if isinstance(value, (str, int, float, bool)):
        if hint == "dict" and isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:  # a plain string, not JSON
                return value
        elif hint == "list" and isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:  # a plain string, not JSON
                return value
        return value
    return value


class FlightHDF5Mixin:
    """Reading and writing a :class:`~pils.flight.Flight` as one HDF5 file."""

    @classmethod
    def from_hdf5(
        cls,
        filepath: str | Path,
        sync_version: str | None | bool = None,
        load_raw: bool = True,
    ) -> "Flight":
        """
        Load flight data from HDF5 file.

        Loads metadata and raw_data hierarchy. Optionally loads a specific
        synchronized data version or the latest available version.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to HDF5 file
        sync_version : Union[str, None, bool], default=None
            Specific sync version to load (e.g., 'rev_20260202_1430').
            If None and synchronized data exists, loads latest version.
            Set to False to skip loading synchronized data.
        load_raw : bool, default=True
            If True, loads raw_data. If False, only loads metadata and sync data.

        Returns
        -------
        Flight
            Returns new Flight instance

        Raises
        ------
        ImportError
            If h5py is not installed
        FileNotFoundError
            If HDF5 file doesn't exist
        ValueError
            If requested sync version not found

        Examples
        --------
        >>> # Load from file
        >>> flight = Flight.from_hdf5('flight_001.h5')
        >>> # Load specific sync version
        >>> flight = Flight.from_hdf5('flight_001.h5', sync_version='rev_20260202_1430')
        >>> # Load only metadata and raw data
        >>> flight = Flight.from_hdf5('flight_001.h5', sync_version=False)
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {filepath}")

        with h5py.File(str(filepath), "r") as f:
            # Load metadata
            metadata_dict = {}
            flight_info_dict = {}
            if "metadata" in f:
                metadata_group = f["metadata"]
                assert isinstance(metadata_group, h5py.Group)
                for key in metadata_group.attrs:
                    if key.startswith("flight_info_"):
                        # Strip prefix and add to flight_info
                        clean_key = key.replace("flight_info_", "", 1)
                        flight_info_dict[clean_key] = metadata_group.attrs[key]
                    else:
                        metadata_dict[key] = metadata_group.attrs[key]

            # Create new instance
            flight = cls(
                flight_info=flight_info_dict if flight_info_dict else metadata_dict
            )

            # Load metadata
            if "metadata" in f:
                metadata_group = f["metadata"]
                assert isinstance(metadata_group, h5py.Group)
                flight._load_metadata_from_hdf5(metadata_group, flight)

            # Load raw data
            if load_raw and "raw_data" in f:
                raw_data_group = f["raw_data"]
                assert isinstance(raw_data_group, h5py.Group)
                flight._load_raw_data_from_hdf5(raw_data_group, flight)

            # Load sync_data if available
            if sync_version is not False and "sync_data" in f:
                sync_data_group = f["sync_data"]
                assert isinstance(sync_data_group, h5py.Group)
                available_versions = sorted(
                    [
                        k
                        for k in sync_data_group.keys()
                        if k.startswith(layout.REVISION_PREFIX)
                    ]
                )
                if available_versions:
                    # Determine which version to load
                    if sync_version is None:
                        # Load latest version
                        sync_version = available_versions[-1]
                    elif sync_version not in available_versions:
                        raise ValueError(
                            f"Sync version '{sync_version}' not found. "
                            f"Available versions: {available_versions}"
                        )

                    revision_group = sync_data_group[sync_version]
                    assert isinstance(revision_group, h5py.Group)
                    # Load all datasets in the revision
                    sync_dict = {}
                    for key in revision_group.keys():
                        dataset_group = revision_group[key]
                        assert isinstance(dataset_group, h5py.Group)
                        df = flight._load_dataframe_from_hdf5(dataset_group)
                        if df is not None:
                            sync_dict[key] = df
                    if sync_dict:
                        flight.sync_data = dict(
                            _legacy_sync_table(key, df) for key, df in sync_dict.items()
                        )

        return flight

    @staticmethod
    def _load_metadata_from_hdf5(
        metadata_group: "h5py.Group", flight: "Flight"
    ) -> None:
        """
        Load metadata from HDF5 group.

        Parameters
        ----------
        metadata_group : h5py.Group
            Metadata group
        flight : Flight
            Flight object to populate
        """
        # Reconstruct flight_info from attrs
        flight_info = {}
        flight_metadata = {}

        for key, value in metadata_group.attrs.items():
            if key.startswith("flight_info_"):
                info_key = key.replace("flight_info_", "")
                flight_info[info_key] = _deserialize_from_hdf5(value)
            elif key.startswith("flight_metadata_"):
                meta_key = key.replace("flight_metadata_", "")
                flight_metadata[meta_key] = _deserialize_from_hdf5(value)

        if flight_info:
            flight.flight_info.update(flight_info)
        if flight_metadata:
            flight.metadata.update(flight_metadata)

    @staticmethod
    def _load_raw_data_from_hdf5(raw_group: "h5py.Group", flight: "Flight") -> None:
        """
        Load raw data hierarchy from HDF5 group.

        Parameters
        ----------
        raw_group : h5py.Group
            raw_data group
        flight : Flight
            Flight object to populate
        """
        # Load drone data
        if "drone_data" in raw_group:
            drone_group = raw_group["drone_data"]
            assert isinstance(drone_group, h5py.Group)
            drone_df = None
            litchi_df = None

            if "drone" in drone_group:
                drone_data = drone_group["drone"]
                assert isinstance(drone_data, h5py.Group)
                drone_df = FlightHDF5Mixin._load_dataframe_from_hdf5(drone_data)

            if "litchi" in drone_group:
                litchi_data = drone_group["litchi"]
                assert isinstance(litchi_data, h5py.Group)
                litchi_df = FlightHDF5Mixin._load_dataframe_from_hdf5(litchi_data)

            records = {}
            if "dat_records" in drone_group:
                for record_name, item in drone_group["dat_records"].items():
                    table = FlightHDF5Mixin._load_dataframe_from_hdf5(item)
                    if table is not None:
                        records[record_name] = table

            # Files written before the schema: rename to it (no-op otherwise)
            if drone_df is not None:
                drone_df = schema.canonical(drone_df, "dji")
            if litchi_df is not None:
                litchi_df = schema.canonical(litchi_df, "litchi")
            records = {
                schema.record_name(name): schema.canonical(table, record=name)
                for name, table in records.items()
            }
            if drone_df is not None or litchi_df is not None:
                flight.raw_data.drone_data = DroneData(drone_df, litchi_df, records)

        # Load payload data
        if "payload_data" in raw_group:
            payload_group = raw_group["payload_data"]
            assert isinstance(payload_group, h5py.Group)
            flight.raw_data.payload_data = PayloadData()

            for sensor_name in payload_group.keys():
                sensor_data = payload_group[sensor_name]
                assert isinstance(sensor_data, h5py.Group)
                sensor_df = FlightHDF5Mixin._load_dataframe_from_hdf5(sensor_data)
                if sensor_df is not None:
                    setattr(
                        flight.raw_data.payload_data,
                        sensor_name,
                        schema.canonical(sensor_df, sensor_name),
                    )
                    continue
                # A group of tables (e.g. the inclinometer message types)
                tables = {
                    name: FlightHDF5Mixin._load_dataframe_from_hdf5(item)
                    for name, item in sensor_data.items()
                    if isinstance(item, h5py.Group)
                }
                tables = {
                    schema.snake_case(k): schema.canonical(v, sensor_name)
                    for k, v in tables.items()
                    if v is not None
                }
                if tables:
                    setattr(flight.raw_data.payload_data, sensor_name, tables)

    @staticmethod
    def _load_dataframe_from_hdf5(
        dataset_group: "h5py.Group",
    ) -> Optional["pl.DataFrame"]:
        """
        Load a Polars DataFrame from HDF5 dataset group.

        Parameters
        ----------
        dataset_group : h5py.Group
            Group containing column datasets (see
            :func:`pils.utils.hdf5.write_table`)

        Returns
        -------
        pl.DataFrame or None
            Polars DataFrame if data found, None otherwise
        """
        if "columns" not in dataset_group.attrs:
            return None
        df = read_table(dataset_group)
        return df if df.width else None

    def to_hdf5(
        self,
        filepath: str | Path | None = None,
        sync_metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save flight data to HDF5 file.

        .. deprecated::
            Use :meth:`save` (Parquet tables, see :mod:`pils.store`);
            :meth:`from_hdf5` still reads HDF5 files.

        Saves metadata and raw_data hierarchy.

        Parameters
        ----------
        filepath : Union[str, Path], optional
            Path to output HDF5 file
        sync_metadata : Dict[str, Any], optional
            Additional metadata to store with synchronized data revision.
            Will be saved as attributes on the revision group.
            Example: {'comment': 'Initial sync', 'target_rate': 10.0}

        Returns
        -------
        str
            Timestamp string for the save operation

        Raises
        ------
        ImportError
            If h5py is not installed
        ValueError
            If no data to save

        Examples
        --------
        >>> # Save raw data
        >>> flight.to_hdf5('flight_001.h5')
        >>> # Save with sync metadata
        >>> flight.to_hdf5('flight_001.h5', sync_metadata={'comment': 'High rate sync', 'rate': 100.0})
        >>> # For synchronization, use Synchronizer separately:
        >>> from pils.synchronizer import Synchronizer
        >>> sync = Synchronizer()
        >>> sync.add_gps_reference(flight.raw_data.payload_data.gps)
        >>> # ... add other sources ...
        >>> result = sync.synchronize(target_rate={'drone': 10.0, 'payload': 100.0})
        """
        warnings.warn(
            "Flight.to_hdf5 is deprecated: use Flight.save (Parquet, pils.store)",
            DeprecationWarning,
            stacklevel=2,
        )

        if filepath:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

        else:
            filepath = Path(self.flight_info["proc_data_folder_path"])

        with h5py.File(str(filepath), "a") as f:
            # Save metadata
            self._save_metadata_to_hdf5(f)

            # Save raw data
            self._save_raw_data_to_hdf5(f)

            # Save sync_data if available
            if self.sync_data is not None and len(self.sync_data) > 0:
                self._save_sync_data_to_hdf5(f, sync_metadata)

        return _get_current_timestamp()

    def _save_metadata_to_hdf5(self, h5file: "h5py.File") -> None:
        """
        Save metadata to HDF5 file.

        Parameters
        ----------
        h5file : h5py.File
            Open HDF5 file handle
        """
        if "metadata" not in h5file:
            metadata_group = h5file.create_group("metadata")
        else:
            metadata_group = h5file["metadata"]

        # Save flight_info as attrs
        if self.flight_info:
            for key, value in self.flight_info.items():
                try:
                    metadata_group.attrs[f"flight_info_{key}"] = _serialize_for_hdf5(
                        value
                    )
                except Exception as e:  # h5py: TypeError, ValueError, OSError, ...
                    logger.warning(f"Could not save flight_info[{key}]: {e}")

        # Save flight_metadata as attrs
        if self.metadata:
            for key, value in self.metadata.items():
                try:
                    metadata_group.attrs[f"flight_metadata_{key}"] = (
                        _serialize_for_hdf5(value)
                    )
                except Exception as e:  # h5py: TypeError, ValueError, OSError, ...
                    logger.warning(f"Could not save metadata[{key}]: {e}")

    def _save_raw_data_to_hdf5(self, h5file: "h5py.File") -> None:
        """
        Save raw data hierarchy to HDF5 file.

        Parameters
        ----------
        h5file : h5py.File
            Open HDF5 file handle
        """
        if "raw_data" not in h5file:
            raw_data_group = h5file.create_group("raw_data")
        else:
            raw_data_group = h5file["raw_data"]
            assert isinstance(raw_data_group, h5py.Group)
        # Raw data is replaced at each save: the sync revisions record which
        # raw data (saved_at) they were made from
        raw_data_group.attrs["saved_at"] = _get_current_timestamp()
        raw_data_group.attrs["pils_version"] = _get_package_version()

        # Save drone data
        drone_has_data = (
            (
                isinstance(self.raw_data.drone_data.drone, dict)
                and len(self.raw_data.drone_data.drone) > 0
            )
            or (
                isinstance(self.raw_data.drone_data.drone, pl.DataFrame)
                and len(self.raw_data.drone_data.drone) > 0
            )
            or len(self.raw_data.drone_data.litchi) > 0
        )
        if drone_has_data:
            if "drone_data" not in raw_data_group:
                drone_group = raw_data_group.create_group("drone_data")
            else:
                drone_group = raw_data_group["drone_data"]
                assert isinstance(drone_group, h5py.Group)

            drone_data = self.raw_data.drone_data.drone
            if isinstance(drone_data, dict) and len(drone_data) > 0:
                pass  # Handle dict case if needed
            elif isinstance(drone_data, pl.DataFrame) and len(drone_data) > 0:
                self._save_dataframe_to_hdf5(drone_group, "drone", drone_data)

            if len(self.raw_data.drone_data.litchi) > 0:
                self._save_dataframe_to_hdf5(
                    drone_group, "litchi", self.raw_data.drone_data.litchi
                )

            records = getattr(self.raw_data.drone_data, "records", None) or {}
            if records:
                if "dat_records" in drone_group:
                    del drone_group["dat_records"]
                records_group = drone_group.create_group("dat_records")
                for record_name, table in records.items():
                    self._save_dataframe_to_hdf5(records_group, record_name, table)

        # Save payload data
        if len(self.raw_data.payload_data.list_loaded_sensors()) > 0:
            if "payload_data" not in raw_data_group:
                payload_group = raw_data_group.create_group("payload_data")
            else:
                payload_group = raw_data_group["payload_data"]
                assert isinstance(payload_group, h5py.Group)

            for sensor_name in self.raw_data.payload_data.list_loaded_sensors():
                sensor_data = getattr(self.raw_data.payload_data, sensor_name)
                if isinstance(sensor_data, dict):
                    # e.g. the inclinometer: one table per message type
                    if sensor_name in payload_group:
                        del payload_group[sensor_name]
                    sensor_group = payload_group.create_group(sensor_name)
                    for table_name, table in sensor_data.items():
                        if isinstance(table, pl.DataFrame):
                            self._save_dataframe_to_hdf5(
                                sensor_group, table_name, table
                            )
                elif sensor_data is not None:
                    self._save_dataframe_to_hdf5(
                        payload_group, sensor_name, sensor_data
                    )

    def _save_sync_data_to_hdf5(
        self, h5file: "h5py.File", sync_metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Save synchronized data to HDF5 file.

        Parameters
        ----------
        h5file : h5py.File
            Open HDF5 file handle
        sync_metadata : Dict[str, Any], optional
            Additional metadata to store as attributes on revision group
        """
        if self.sync_data is None or len(self.sync_data) == 0:
            return

        if "sync_data" not in h5file:
            sync_group = h5file.create_group("sync_data")
        else:
            sync_group = h5file["sync_data"]
            assert isinstance(sync_group, h5py.Group)

        # Create revision group
        revision_name = _get_current_timestamp()
        if revision_name in sync_group:
            del sync_group[revision_name]
        revision_group = sync_group.create_group(revision_name)

        # Save each key's DataFrame as a dataset
        if self.sync_data is not None:
            for key, df in self.sync_data.items():
                if isinstance(df, pl.DataFrame) and len(df) > 0:
                    self._save_dataframe_to_hdf5(revision_group, key, df)

            # Save metadata on revision group
            revision_group.attrs["created_at"] = revision_name
            revision_group.attrs["n_keys"] = len(self.sync_data)
        revision_group.attrs["pils_version"] = _get_package_version()
        if "raw_data" in h5file:
            revision_group.attrs["raw_data_saved_at"] = h5file["raw_data"].attrs.get(
                "saved_at", ""
            )

        # Save user-provided metadata
        if sync_metadata:
            for key, value in sync_metadata.items():
                try:
                    revision_group.attrs[f"user_{key}"] = _serialize_for_hdf5(value)
                except Exception as e:  # h5py: TypeError, ValueError, OSError, ...
                    logger.warning(f"Could not save sync_metadata[{key}]: {e}")

    def _save_dataframe_to_hdf5(
        self, parent_group: "h5py.Group", name: str, df: "pl.DataFrame"
    ) -> None:
        """
        Save a Polars DataFrame to HDF5 (see :func:`pils.utils.hdf5.write_table`).

        Parameters
        ----------
        parent_group : h5py.Group
            Parent HDF5 group
        name : str
            Dataset name
        df : pl.DataFrame
            DataFrame to save
        """
        write_table(parent_group, name, df)
