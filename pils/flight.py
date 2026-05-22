import sys
sys.path.append("/home/fastori/Desktop/ARS/pils/pils")

import glob
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, overload

import h5py
import polars as pl

from pils.drones.BlackSquareDrone import BlackSquareDrone
from pils.drones.DJIDrone import DJIDrone
from pils.drones.litchi import Litchi
from pils.sensors.camera import Camera
from pils.sensors.sensors import sensor_config
from pils.synchronizer import Synchronizer
from pils.utils.tools import get_path_from_keyword

logger = logging.getLogger(__name__)


def _get_current_timestamp() -> str:
    return datetime.now().strftime("rev_%Y%m%d_%H%M%S")


def _get_package_version() -> str:
    try:
        import pils
        return getattr(pils, "__version__", "unknown")
    except Exception:
        return "unknown"


def _serialize_for_hdf5(obj: Any) -> Any:
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
    if value == "None":
        return None
    if isinstance(value, (str, int, float, bool)):
        if hint == "dict" and isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        elif hint == "list" and isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value
    return value


class Flight:
    """
    This class provides a hierarchical structure to store and access drone flight data
    and sensor payloads. Data is stored in RAM for fast access using both attribute
    and dictionary-style notation.

    Attributes
    ----------
    flight_info : Dict
        Dictionary containing flight configuration paths
    flight_path : Path
        Path to the flight directory
    metadata : Dict
        Flight metadata (duration, date, conditions, etc.)
    raw_data : RawData
        Container for drone and payload sensor data
    sync_data : Optional[dict[str, pl.DataFrame]]
        Synchronized flight data (populated after calling sync())
    adc_gain_config : Optional
        Configuration for ADC gain settings

    Examples
    --------
    >>> flight_info = {
    ...     "drone_data_folder_path": "/data/flight_001/drone",
    ...     "aux_data_folder_path": "/data/flight_001/aux"
    ... }
    >>> flight = Flight(flight_info)
    >>> flight.add_drone_data()
    >>> flight.add_sensor_data(['gps', 'imu', 'adc'])
    >>> flight.add_camera_data(use_photogrammetry=False)
    >>> drone_df = flight.raw_data.drone_data.drone
    >>> camera_df = flight.raw_data.payload_data.camera
    >>> camera_obj = flight.raw_data.payload_data.camera_obj
    """

    def __init__(self, flight_info: dict[str, Any]):
        self.flight_info = flight_info
        self.flight_path = Path(flight_info["drone_data_folder_path"]).parent
        self.metadata = {}
        self.set_metadata()
        self.raw_data = RawData()
        self.sync_data: dict[str, pl.DataFrame] | None = None
        self.adc_gain_config = None

    @classmethod
    def from_hdf5(
        cls,
        filepath: str | Path,
        sync_version: str | None | bool = None,
        load_raw: bool = True,
    ) -> "Flight":
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {filepath}")

        with h5py.File(str(filepath), "r") as f:
            metadata_dict = {}
            flight_info_dict = {}
            if "metadata" in f:
                metadata_group = f["metadata"]
                assert isinstance(metadata_group, h5py.Group)
                for key in metadata_group.attrs:
                    if key.startswith("flight_info_"):
                        clean_key = key.replace("flight_info_", "", 1)
                        flight_info_dict[clean_key] = metadata_group.attrs[key]
                    else:
                        metadata_dict[key] = metadata_group.attrs[key]

            flight = cls(
                flight_info=flight_info_dict if flight_info_dict else metadata_dict
            )

            if "metadata" in f:
                metadata_group = f["metadata"]
                assert isinstance(metadata_group, h5py.Group)
                flight._load_metadata_from_hdf5(metadata_group, flight)

            if load_raw and "raw_data" in f:
                raw_data_group = f["raw_data"]
                assert isinstance(raw_data_group, h5py.Group)
                flight._load_raw_data_from_hdf5(raw_data_group, flight)

            if sync_version is not False and "sync_data" in f:
                sync_data_group = f["sync_data"]
                assert isinstance(sync_data_group, h5py.Group)
                available_versions = sorted(
                    [k for k in sync_data_group.keys() if k.startswith("rev_")]
                )
                if available_versions:
                    if sync_version is None:
                        sync_version = available_versions[-1]
                    elif sync_version not in available_versions:
                        raise ValueError(
                            f"Sync version '{sync_version}' not found. "
                            f"Available versions: {available_versions}"
                        )
                    revision_group = sync_data_group[sync_version]
                    assert isinstance(revision_group, h5py.Group)
                    sync_dict = {}
                    for key in revision_group.keys():
                        dataset_group = revision_group[key]
                        assert isinstance(dataset_group, h5py.Group)
                        df = flight._load_dataframe_from_hdf5(dataset_group)
                        if df is not None:
                            sync_dict[key] = df
                    if sync_dict:
                        flight.sync_data = sync_dict

        return flight

    @staticmethod
    def _load_metadata_from_hdf5(metadata_group: "h5py.Group", flight: "Flight") -> None:
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
        if "drone_data" in raw_group:
            drone_group = raw_group["drone_data"]
            assert isinstance(drone_group, h5py.Group)
            drone_df = None
            litchi_df = None
            if "drone" in drone_group:
                drone_data = drone_group["drone"]
                assert isinstance(drone_data, h5py.Group)
                drone_df = Flight._load_dataframe_from_hdf5(drone_data)
            if "litchi" in drone_group:
                litchi_data = drone_group["litchi"]
                assert isinstance(litchi_data, h5py.Group)
                litchi_df = Flight._load_dataframe_from_hdf5(litchi_data)
            if drone_df is not None or litchi_df is not None:
                flight.raw_data.drone_data = DroneData(drone_df, litchi_df)

        if "payload_data" in raw_group:
            payload_group = raw_group["payload_data"]
            assert isinstance(payload_group, h5py.Group)
            flight.raw_data.payload_data = PayloadData()
            for sensor_name in payload_group.keys():
                sensor_data = payload_group[sensor_name]
                assert isinstance(sensor_data, h5py.Group)
                sensor_df = Flight._load_dataframe_from_hdf5(sensor_data)
                if sensor_df is not None:
                    setattr(flight.raw_data.payload_data, sensor_name, sensor_df)

    @staticmethod
    def _load_dataframe_from_hdf5(dataset_group: "h5py.Group") -> Optional["pl.DataFrame"]:
        if "columns" not in dataset_group.attrs:
            return None
        columns_attr = dataset_group.attrs["columns"]
        if isinstance(columns_attr, bytes):
            columns = json.loads(columns_attr.decode())
        else:
            columns = json.loads(str(columns_attr))
        data_dict = {}
        for col_name in columns:
            if col_name in dataset_group:
                col_dataset = dataset_group[col_name]
                assert isinstance(col_dataset, h5py.Dataset)
                data_dict[col_name] = col_dataset[:]
        if not data_dict:
            return None
        return pl.DataFrame(data_dict)

    def set_metadata(self, metadata: dict[str, Any] | None = None) -> None:
        if isinstance(metadata, dict):
            self.metadata.update(metadata)
        info_source: dict[str, Any] = {}
        if isinstance(metadata, dict):
            info_source = metadata
        elif isinstance(self.flight_info, dict):
            info_source = self.flight_info
        takeoff = info_source.get("takeoff_time") or info_source.get("takeoff_datetime")
        landing = info_source.get("landing_time") or info_source.get("landing_datetime")
        if takeoff is not None and landing is not None:
            try:
                self.metadata["takeoff_time"] = takeoff
                self.metadata["flight_time"] = landing - takeoff
            except Exception:
                self.metadata["takeoff_time"] = takeoff
                self.metadata["landing_time"] = landing
        if "flight_name" in info_source and "flight_name" not in self.metadata:
            self.metadata["flight_name"] = info_source.get("flight_name")

    def _detect_drone_model(self, drone_folder: str) -> str:
        try:
            drone_id = (
                self.flight_info.get("drone_id")
                if isinstance(self.flight_info, dict)
                else None
            )
            if drone_id:
                try:
                    from stout.services.inventory.service import InventoryService
                    inventory = InventoryService()
                    item = inventory.get_item_by_id(drone_id)
                    if item:
                        specs_obj = (
                            item.get("specifications") if isinstance(item, dict) else None
                        )
                        specs = specs_obj if isinstance(specs_obj, dict) else {}
                        model_val = (
                            specs.get("model") or item.get("name") or item.get("category")
                        )
                        if isinstance(model_val, str):
                            m = model_val.lower()
                            if "matrice" in m:
                                return "dji"
                            if "black" in m or "blacksquare" in m:
                                return "blacksquare"
                            return m
                except Exception:
                    pass

            dji_pattern = get_path_from_keyword(str(drone_folder), "DJI")
            if dji_pattern:
                return "dji"
            blacksquare_pattern = get_path_from_keyword(str(drone_folder), "blacksquare")
            if blacksquare_pattern:
                return "blacksquare"
            return "dji"
        except Exception:
            return "dji"

    def add_drone_data(
        self,
        dji_dat_loader: bool = True,
        drone_model: str | None = None,
    ):
        if not isinstance(self.flight_info, dict):
            raise ValueError("flight_info must be a dict containing 'drone_data_folder_path'")

        drone_folder = self.flight_info.get("drone_data_folder_path")
        if not drone_folder:
            raise ValueError("drone_data_folder_path not found in flight_info")

        if not drone_model:
            drone_model = self._detect_drone_model(str(drone_folder))

        self.__drone_model = drone_model
        available_files = glob.glob(str(drone_folder) + "/*")

        if isinstance(self.__drone_model, str) and "dji" in self.__drone_model.lower():
            drone_data_path = None
            litchi_data_path = None

            for file in available_files:
                fname = file.lower()
                if dji_dat_loader and fname.endswith("drone.dat"):
                    drone_data_path = file
                else:
                    if fname.endswith("drone.csv") and drone_data_path is None:
                        drone_data_path = file
                if fname.endswith("litchi.csv") and "dji" in self.__drone_model.lower():
                    litchi_data_path = file

            logger.info(f"Drone : {drone_data_path}")
            drone = DJIDrone(drone_data_path)
            drone.load_data(use_dat=dji_dat_loader)
            drone_data = drone.data
            litchi_data = None

            if litchi_data_path is not None:
                litchi_loader = Litchi(litchi_data_path)
                litchi_loader.load_data()
                litchi_data = litchi_loader.data

        elif isinstance(self.__drone_model, str) and (
            "black" in self.__drone_model.lower()
            or "blacksquare" in self.__drone_model.lower()
        ):
            drone = BlackSquareDrone(drone_folder)
            drone.load_data()
            drone_data = drone.data
            litchi_data = None

        else:
            try:
                drone = DJIDrone(drone_data_path or str(drone_folder))
                drone.load_data(use_dat=dji_dat_loader)
                drone_data = drone.data
                litchi_loader = Litchi(litchi_data_path or str(drone_folder))
                litchi_loader.load_data()
                litchi_data = litchi_loader.data
            except Exception:
                drone = BlackSquareDrone(str(drone_folder))
                drone.load_data()
                drone_data = drone.data
                litchi_data = None

        self.raw_data.drone_data = DroneData(drone_data, litchi_data)

    def _read_sensor_data(self, sensor_name: str, sensor_folder: Path) -> Any | None:
        result = None
        config = sensor_config.get(sensor_name.lower())
        if config:
            sensor = config["class"](sensor_folder)
            if sensor_name == "inclinometer":
                self.__inclinometer = sensor.sensor_type
            getattr(sensor, config["load_method"])()
            result = sensor.data
        return result

    def add_sensor_data(self, sensor_name: str | list[str]) -> None:
        sensor_path = Path(self.flight_info["aux_data_folder_path"]) / "sensors"
        if sensor_path.exists():
            if isinstance(sensor_name, str):
                sensor_name = [sensor_name]
            for sensor in sensor_name:
                sensor_data = self._read_sensor_data(sensor, sensor_path)
                if isinstance(sensor_data, dict) and (sensor != "inclinometer"):
                    setattr(self.raw_data.payload_data, sensor, sensor_data["data"])
                    self.flight_info["flight_info"].update(
                        {f"{sensor}_metadata": sensor_data["metadata"]}
                    )
                else:
                    setattr(self.raw_data.payload_data, sensor, sensor_data)
        else:
            logger.info("Sensor datasets are not available")

    # -------------------------------------------------------------------------
    # CHANGED: now also stores camera_obj for run_photogrammetry()
    # -------------------------------------------------------------------------
    def add_camera_data(
        self, use_photogrammetry: bool = False, get_sony_angles: bool = True
    ) -> None:
        """
        Load camera data from the payload.

        Supports both video cameras (Sony RX0 MarkII with telemetry, Alvium industrial)
        and photogrammetry-processed data. For video cameras, can compute Euler angles
        (roll, pitch, yaw) and quaternions from inertial measurement data.

        Always stores two attributes on ``flight.raw_data.payload_data``:

        - ``camera``     — the raw ``pl.DataFrame`` (timestamps / IMU / angles),
                           used by ``sync()`` and other downstream consumers.
                           Behaviour is identical to before.
        - ``camera_obj`` — the ``Camera`` instance, needed to call
                           ``camera_obj.run_photogrammetry(...)``.

        Parameters
        ----------
        use_photogrammetry : bool, default=False
            If True, loads pre-processed photogrammetry results from proc_data folder.
            If False, loads camera data from aux_data/camera folder (video or logs).
        get_sony_angles : bool, default=True
            For Sony cameras, whether to compute Euler angles and quaternions from
            telemetry gyro/accel data using AHRS (Madgwick) filter.

        Raises
        ------
        FileNotFoundError
            If camera data folder or photogrammetry folder not found

        Examples
        --------
        >>> # Load Alvium / Sony camera ready for the photogrammetry pipeline
        >>> flight.add_camera_data(use_photogrammetry=False)
        >>> camera_obj = flight.raw_data.payload_data.camera_obj
        >>> result = camera_obj.run_photogrammetry(
        ...     csv_file="targets.csv",
        ...     config=cfg,
        ...     flight=flight,
        ... )

        >>> # Load pre-processed photogrammetry results (read-only)
        >>> flight.add_camera_data(use_photogrammetry=True)
        >>> df = flight.raw_data.payload_data.camera   # pl.DataFrame
        """

        self.__use_photogrammetry = use_photogrammetry

        if use_photogrammetry:
            self.__camera_data_type = "photogrammetry"
            path = Path(self.flight_info["proc_data_folder_path"]) / "photogrammetry"
        else:
            self.__camera_data_type = "camera"
            path = Path(self.flight_info["aux_data_folder_path"]) / "camera"

        if path.exists():
            camera = Camera(path, use_photogrammetry=use_photogrammetry)
            camera.load_data()

            # DataFrame — unchanged, used by sync() and other consumers
            self.raw_data.payload_data.camera = camera.data[0]

            # Camera object — exposes run_photogrammetry()           ← NEW
            self.raw_data.payload_data.camera_obj = camera           # ← NEW

            self.__camera_model = camera.data[1]

        else:
            logger.info("Camera path does not exist: %s", path)

    def sync(
        self,
        target_rate: dict[str, float] | None = None,
        use_rtk_data: bool = True,
        common_time: bool = True,
        **kwargs,
    ) -> dict[str, pl.DataFrame]:
        if not self.raw_data.payload_data or "gps" not in self.raw_data.payload_data:
            raise ValueError(
                "GPS payload data is required as reference timebase. "
                "Call flight.add_sensor_data(['gps']) first."
            )

        sync = Synchronizer()

        gps_sensor = self.raw_data.payload_data["gps"]
        gps_data = gps_sensor.data if hasattr(gps_sensor, "data") else gps_sensor
        sync.add_gps_reference(
            gps_data,
            timestamp_col="timestamp",
            alt_col="posllh_height",
            lat_col="posllh_lat",
            lon_col="posllh_lon",
        )

        drone_data = self.raw_data.drone_data.drone
        drone_has_data = (isinstance(drone_data, dict) and len(drone_data) > 0) or (
            isinstance(drone_data, pl.DataFrame) and len(drone_data) > 0
        )

        if target_rate is None:
            target_rate = {}

        if drone_has_data:
            if "drone" not in target_rate:
                target_rate["drone"] = 10.0
            drone_df = drone_data

            if "dji" in self.__drone_model.lower():
                timestamp_col = "timestamp"
                if use_rtk_data:
                    lat_col = "RTK:lat_p"
                    lon_col = "RTK:lon_p"
                    alt_col = "RTK:hmsl_p"
                else:
                    lat_col = "GPS:Latitude"
                    lon_col = "GPS:Longitude"
                    alt_col = "GPS:heightMSL"
            else:
                timestamp_col = "timestamp"
                lat_col = "Latitude"
                lon_col = "Longitude"
                alt_col = "heightMSL"

            sync.add_drone_gps(
                drone_df,
                timestamp_col=timestamp_col,
                lat_col=lat_col,
                lon_col=lon_col,
                alt_col=alt_col,
            )

        if len(self.raw_data.drone_data.litchi) > 0:
            litchi_df = self.raw_data.drone_data.litchi
            if (
                isinstance(litchi_df, pl.DataFrame)
                and "latitude" in litchi_df.columns
                and "longitude" in litchi_df.columns
            ):
                if "drone" not in target_rate:
                    target_rate["drone"] = 10.0
                sync.add_litchi_gps(litchi_df)

        if "inclinometer" in self.raw_data.payload_data:
            if "inclinometer" not in target_rate:
                target_rate["inclinometer"] = 100.0
            incl_sensor = self.raw_data.payload_data["inclinometer"]
            incl_data = incl_sensor.data if hasattr(incl_sensor, "data") else incl_sensor
            if self.__inclinometer == "imx5":
                incl_data = incl_data["INS"]
            sync.add_inclinometer(incl_data, self.__inclinometer)

        if "camera" in self.raw_data.payload_data:
            sync.add_camera(
                self.raw_data.payload_data["camera"],
                use_photogrammetry=self.__use_photogrammetry,
                camera_model=self.__camera_model,
            )

        payload = self.raw_data.payload_data

        if "adc" in payload:
            if "payload" not in target_rate:
                target_rate["payload"] = 100.0
            adc_sensor = payload["adc"]
            adc_data = adc_sensor.data if hasattr(adc_sensor, "data") else adc_sensor
            sync.add_payload_sensor("adc", adc_data)

        if "lm76" in payload:
            if "payload" not in target_rate:
                target_rate["payload"] = 100.0
            lm76_sensor = payload["lm76"]
            lm76_data = lm76_sensor.data if hasattr(lm76_sensor, "data") else lm76_sensor
            sync.add_payload_sensor("lm76", lm76_data)

        if "imu" in payload:
            if "payload" not in target_rate:
                target_rate["payload"] = 100.0
            imu_sensor = payload["imu"]
            if hasattr(imu_sensor, "barometer") and imu_sensor.barometer is not None:
                sync.add_payload_sensor("imu_barometer", imu_sensor.barometer)
            if hasattr(imu_sensor, "accelerometer") and imu_sensor.accelerometer is not None:
                sync.add_payload_sensor("imu_accelerometer", imu_sensor.accelerometer)
            if hasattr(imu_sensor, "gyroscope") and imu_sensor.gyroscope is not None:
                sync.add_payload_sensor("imu_gyroscope", imu_sensor.gyroscope)
            if hasattr(imu_sensor, "magnetometer") and imu_sensor.magnetometer is not None:
                sync.add_payload_sensor("imu_magnetometer", imu_sensor.magnetometer)

        self.sync_data = sync.synchronize(
            target_rate=target_rate,
            common_time=common_time,
            **kwargs,
        )

        return self.sync_data

    def to_hdf5(
        self,
        filepath: str | Path | None = None,
        sync_metadata: dict[str, Any] | None = None,
    ) -> str:
        if filepath:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
        else:
            filepath = Path(self.flight_info["proc_data_folder_path"])

        with h5py.File(str(filepath), "a") as f:
            self._save_metadata_to_hdf5(f)
            self._save_raw_data_to_hdf5(f)
            if self.sync_data is not None and len(self.sync_data) > 0:
                self._save_sync_data_to_hdf5(f, sync_metadata)

        return _get_current_timestamp()

    def _save_metadata_to_hdf5(self, h5file: "h5py.File") -> None:
        if "metadata" not in h5file:
            metadata_group = h5file.create_group("metadata")
        else:
            metadata_group = h5file["metadata"]

        if self.flight_info:
            for key, value in self.flight_info.items():
                try:
                    metadata_group.attrs[f"flight_info_{key}"] = _serialize_for_hdf5(value)
                except Exception as e:
                    logger.info(f"Warning: Could not save flight_info[{key}]: {e}")

        if self.metadata:
            for key, value in self.metadata.items():
                try:
                    metadata_group.attrs[f"flight_metadata_{key}"] = _serialize_for_hdf5(value)
                except Exception as e:
                    logger.info(f"Warning: Could not save metadata[{key}]: {e}")

    def _save_raw_data_to_hdf5(self, h5file: "h5py.File") -> None:
        if "raw_data" not in h5file:
            raw_data_group = h5file.create_group("raw_data")
        else:
            raw_data_group = h5file["raw_data"]
            assert isinstance(raw_data_group, h5py.Group)

        drone_has_data = (
            (isinstance(self.raw_data.drone_data.drone, dict) and len(self.raw_data.drone_data.drone) > 0)
            or (isinstance(self.raw_data.drone_data.drone, pl.DataFrame) and len(self.raw_data.drone_data.drone) > 0)
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
                pass
            elif isinstance(drone_data, pl.DataFrame) and len(drone_data) > 0:
                self._save_dataframe_to_hdf5(drone_group, "drone", drone_data)

            if len(self.raw_data.drone_data.litchi) > 0:
                self._save_dataframe_to_hdf5(drone_group, "litchi", self.raw_data.drone_data.litchi)

        if len(self.raw_data.payload_data.list_loaded_sensors()) > 0:
            if "payload_data" not in raw_data_group:
                payload_group = raw_data_group.create_group("payload_data")
            else:
                payload_group = raw_data_group["payload_data"]
                assert isinstance(payload_group, h5py.Group)

            for sensor_name in self.raw_data.payload_data.list_loaded_sensors():
                # Skip camera_obj — it's a Camera instance, not a DataFrame
                if sensor_name == "camera_obj":
                    continue
                sensor_data = getattr(self.raw_data.payload_data, sensor_name)
                if sensor_data is not None:
                    self._save_dataframe_to_hdf5(payload_group, sensor_name, sensor_data)

    def _save_sync_data_to_hdf5(
        self, h5file: "h5py.File", sync_metadata: dict[str, Any] | None = None
    ) -> None:
        if self.sync_data is None or len(self.sync_data) == 0:
            return

        if "sync_data" not in h5file:
            sync_group = h5file.create_group("sync_data")
        else:
            sync_group = h5file["sync_data"]
            assert isinstance(sync_group, h5py.Group)

        revision_name = _get_current_timestamp()
        if revision_name in sync_group:
            del sync_group[revision_name]
        revision_group = sync_group.create_group(revision_name)

        if self.sync_data is not None:
            for key, df in self.sync_data.items():
                if isinstance(df, pl.DataFrame) and len(df) > 0:
                    self._save_dataframe_to_hdf5(revision_group, key, df)
            revision_group.attrs["created_at"] = revision_name
            revision_group.attrs["n_keys"] = len(self.sync_data)
        revision_group.attrs["pils_version"] = _get_package_version()

        if sync_metadata:
            for key, value in sync_metadata.items():
                try:
                    revision_group.attrs[f"user_{key}"] = _serialize_for_hdf5(value)
                except Exception as e:
                    logger.info(f"Warning: Could not save sync_metadata[{key}]: {e}")

    def _save_dataframe_to_hdf5(
        self, parent_group: "h5py.Group", name: str, df: "pl.DataFrame"
    ) -> None:
        if name in parent_group:
            del parent_group[name]
        column_group = parent_group.create_group(name)
        for col_name in df.columns:
            col_data = df[col_name].to_numpy()
            if col_name in column_group:
                del column_group[col_name]
            column_group.create_dataset(col_name, data=col_data)
        column_group.attrs["columns"] = json.dumps(df.columns)
        column_group.attrs["dtypes"] = json.dumps([str(dtype) for dtype in df.dtypes])
        column_group.attrs["n_rows"] = len(df)

    def __getitem__(self, key):
        if key == "raw_data":
            return self.raw_data
        elif key == "metadata":
            return self.metadata
        else:
            raise KeyError(f"Key '{key}' not found")


class RawData:
    def __init__(self):
        self.drone_data: DroneData = DroneData(None, None)
        self.payload_data: PayloadData = PayloadData()

    def __getitem__(self, key):
        if key == "drone_data":
            return self.drone_data
        elif key == "payload_data" or key == "payload":
            return self.payload_data
        else:
            raise KeyError(f"Key '{key}' not found")

    def __repr__(self):
        output = []
        if len(self.drone_data.drone) > 0 or len(self.drone_data.litchi) > 0:
            output.append("=== DRONE DATA ===")
            output.append(str(self.drone_data))
        if len(self.payload_data.list_loaded_sensors()) > 0:
            output.append("\n=== PAYLOAD DATA ===")
            output.append(str(self.payload_data))
        return "\n".join(output) if output else "No data loaded"


class DroneData:
    def __init__(
        self,
        drone_df: Union[dict[str, "pl.DataFrame"], "pl.DataFrame", None] = None,
        litchi_df: Optional["pl.DataFrame"] = None,
    ) -> None:
        self.drone: dict[str, pl.DataFrame] | pl.DataFrame = (
            drone_df if drone_df is not None else pl.DataFrame()
        )
        self.litchi: pl.DataFrame = (
            litchi_df if litchi_df is not None else pl.DataFrame()
        )

    def __getitem__(self, key: str) -> Union["pl.DataFrame", dict[str, "pl.DataFrame"]]:
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Key '{key}' not found")

    def __repr__(self):
        output = []
        if self.drone is not None:
            output.append(f"Drone:\n{self.drone}")
        if self.litchi is not None:
            output.append(f"\nLitchi:\n{self.litchi}")
        return "\n".join(output)


class PayloadData:
    def __init__(self):
        pass

    def __getattr__(self, name: str) -> Any:
        available = self.list_loaded_sensors()
        if available:
            raise AttributeError(
                f"Sensor '{name}' not loaded. Available sensors: {available}"
            )
        else:
            raise AttributeError(
                f"Sensor '{name}' not loaded. No sensors currently loaded."
            )

    @overload
    def __setattr__(self, name: str, value: pl.DataFrame) -> None: ...

    @overload
    def __setattr__(self, name: str, value: Any) -> None: ...

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Sensor '{key}' not found")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def list_loaded_sensors(self) -> list[str]:
        return [
            attr
            for attr in dir(self)
            if not attr.startswith("_")
            and attr != "list_loaded_sensors"
            and not callable(getattr(self, attr))
        ]

    def __repr__(self):
        output = []
        for sensor_name in self.list_loaded_sensors():
            output.append(f"{sensor_name}:\n{getattr(self, sensor_name)}\n")
        return "\n".join(output) if output else "No sensors loaded"