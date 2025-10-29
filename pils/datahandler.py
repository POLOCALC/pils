import os
import sys

from pils.pathhandler import PathHandler
from pils.sensors.kernel import Inclinometer
from pils.sensors.gps import GPS
from pils.sensors.adc import ADC
from pils.sensors.IMU import IMUSensor
from pils.drones.litchi import Litchi
from pils.drones.DJIDrone import DJIDrone
from pils.drones.BlackSquareDrone import BlackSquareDrone
from pils.sensors.camera import Camera


class Drone:
    """
    Factory class for initializing drone objects based on drone model.
    Also checks for Litchi flight plans when using DJI or BlackSquare drones.
    """

    # Mapping between drone model names and their classes
    DRONE_MAP = {
        "dji": {"class": DJIDrone, "pattern": "FLY*.csv"},
        "blacksquare": {"class": BlackSquareDrone, "pattern": "*.csv"},
    }

    def __init__(self, pathhandler, drone_model="dji"):
        """
        Initialize drone based on model.

        Parameters
        ----------
        pathhandler : PathHandler
            PathHandler instance to get drone files
        drone_model : str
            Drone model ('dji', 'blacksquare', or 'litchi')
        """
        drone_model = drone_model.lower()

        if drone_model not in self.DRONE_MAP:
            raise ValueError(
                f"[Drone] Drone model '{drone_model}' is unknown. "
                f"Available models: {list(self.DRONE_MAP.keys())}"
            )

        drone_info = self.DRONE_MAP[drone_model]
        drone_class = drone_info["class"]
        pattern = drone_info["pattern"]

        # Get drone files from PathHandler
        drone_files = pathhandler.get_drone_files(pattern=pattern)
        if not drone_files:
            raise FileNotFoundError(
                f"[Drone] No drone files found for pattern '{pattern}'"
            )

        # Initialize the drone instance
        self.drone = drone_class(drone_files[0])
        self.model = drone_model
        self.path = drone_files[0]

        # Check for Litchi flight plan if using DJI or BlackSquare
        self.litchi = None
        if drone_model in ["dji"]:
            try:
                all_csv_files = pathhandler.get_drone_files(pattern="*.csv")

                # Filter out the main drone file patterns
                litchi_files = []
                for f in all_csv_files:
                    basename = os.path.basename(f)
                    # Exclude DJI FLY*.csv files
                    if drone_model == "dji" and basename.startswith("FLY"):
                        continue
                    # Exclude the main drone file itself
                    if f == self.path:
                        continue
                    litchi_files.append(f)

                if litchi_files:
                    self.litchi = Litchi(litchi_files[0])
                    print(
                        f"[Drone] Found Litchi flight plan: {os.path.basename(litchi_files[0])}"
                    )
            except Exception as e:
                # Silently ignore if no Litchi file found
                pass

    def __getattr__(self, name):
        """Delegate attribute access to the drone instance."""
        return getattr(self._drone_instance, name)

    def load_data(self):
        """Load drone data."""
        if hasattr(self.drone, "load_data"):
            return self.drone.load_data()


class Payload:
    # Mapping between sensor names, their classes, and file patterns
    SENSOR_MAP = {
        "gps": {"class": GPS, "pattern": "**/ZED-F9P*"},
        "adc": {"class": ADC, "pattern": "**/ADS1015*"},
        "inclino": {"class": Inclinometer, "pattern": "**/Kernel-100*"},
        "baro": {"class": IMUSensor, "pattern": "**/barometer*", "args": ["baro"]},
        "accelero": {
            "class": IMUSensor,
            "pattern": "**/accelerometer*",
            "args": ["accelero"],
        },
        "magneto": {
            "class": IMUSensor,
            "pattern": "**/magnetometer*",
            "args": ["magneto"],
        },
        "gyro": {"class": IMUSensor, "pattern": "**/gyroscope*", "args": ["gyro"]},
        "camera": {"class": Camera, "pattern": "*.MP4*"},
    }

    def __init__(self, pathhandler):
        # Dynamically initialize all sensors based on SENSOR_MAP
        for sensor_name, sensor_info in self.SENSOR_MAP.items():
            sensor_class = sensor_info["class"]
            pattern = sensor_info["pattern"]
            extra_args = sensor_info.get("args", [])

            # Get sensor files from PathHandler using pattern-based search
            sensor_files = pathhandler.get_aux_files(pattern=pattern)
            sensor_path = sensor_files[0] if sensor_files else None

            # Initialize sensor with class, path, and any extra arguments
            if extra_args:
                sensor_instance = sensor_class(sensor_path, *extra_args)
            else:
                sensor_instance = sensor_class(sensor_path)

            # Set as attribute on self
            setattr(self, sensor_name, sensor_instance)

    def load_data(self):
        for attr_name in vars(self):
            if attr_name.startswith("__"):
                continue

            attr = getattr(self, attr_name)
            # Call load_data() if the attribute has it
            if (
                hasattr(attr, "load_data")
                and callable(getattr(attr, "load_data"))
                and getattr(attr, "path") is not None
            ):
                attr.load_data()


class DataHandler:
    """
    Coordinates data loading from drone logs, payload sensors, and Litchi flight logs.
    Supports DJI and BlackSquare drone models.

    Automatically detects whether to load from stout database or filesystem:
    - If flight_name_or_path exists as a path on disk, loads directly from filesystem
    - Otherwise, treats it as a flight name and queries the database
    """

    def __init__(self, flight_name_or_path: str, drone_model="DJI", db_manager=None):
        """
        Initialize DataHandler with a flight name or direct path.

        Automatically detects whether to load from database or filesystem based on
        whether the provided string is a valid existing path.

        Parameters
        ----------
        flight_name_or_path : str
            Either a flight name (e.g., 'flight_20240715_1430') for database loading,
            or a direct path to the flight data folder for filesystem loading
        drone_model : str
            Drone model ('DJI' or 'BlackSquare')
        db_manager : optional
            Database manager instance (used for database loading)
        """
        # PathHandler will automatically detect if this is a path or flight name
        self.paths = PathHandler(flight_name_or_path, db_manager=db_manager)

        flag = self.paths.load_flight_data()

        if not flag:
            print(
                f"[DataHandler] Could not load flight data for '{flight_name_or_path}'. Stopped the process."
            )
            return

        # Initialize the drone object using the Drone factory class
        self.drone = Drone(self.paths, drone_model=drone_model)

        # Initialize the payload object
        self.payload = Payload(self.paths)

        # Dictionary to store all loaded data
        self.data = {}

    def load_data(self):
        """
        Load all data from drone and payload sensors.

        After loading, creates a dictionary `self.data` with the following structure:
        {
            'drone': drone_dataframe,
            'litchi': litchi_dataframe (if available),
            'gps': gps_dataframe,
            'adc': adc_dataframe,
            'inclino': inclino_dataframe,
            'baro': baro_dataframe,
            'accelero': accelero_dataframe,
            'magneto': magneto_dataframe,
            'gyro': gyro_dataframe,
            'camera': camera_object
        }

        Access data via: d.data['adc'] instead of d.payload.adc.data
        """
        for attr_name in vars(self):
            if attr_name.startswith("__") or attr_name in ["paths", "data"]:
                continue

            attr = getattr(self, attr_name)

            # Call load_data() if the attribute has it
            if hasattr(attr, "load_data") and callable(getattr(attr, "load_data")):
                if hasattr(attr, "path") and getattr(attr, "path") is not None:
                    attr.load_data()
                elif not hasattr(attr, "path"):
                    attr.load_data()
                else:
                    print(f"[load_data] Could not load data for '{attr_name}'")
                    continue

        # Build the data dictionary
        self._build_data_dict()

    def _build_data_dict(self):
        """
        Build a dictionary representation of all loaded data.
        """
        self.data = {}

        # Add drone data
        if hasattr(self.drone, "drone") and hasattr(self.drone.drone, "data"):
            self.data["drone"] = self.drone.drone.data

        # Add litchi data if available
        if hasattr(self.drone, "litchi") and self.drone.litchi is not None:
            if hasattr(self.drone.litchi, "data"):
                self.data["litchi"] = self.drone.litchi.data

        # Add all payload sensor data
        for sensor_name in self.payload.SENSOR_MAP.keys():
            if hasattr(self.payload, sensor_name):
                sensor = getattr(self.payload, sensor_name)
                if hasattr(sensor, "data") and sensor.data is not None:
                    self.data[sensor_name] = sensor.data
                elif sensor_name == "camera":
                    # Camera might not have a 'data' attribute, store the object
                    self.data[sensor_name] = sensor

        print(
            f"[DataHandler] Data dictionary created with {len(self.data)} items: {list(self.data.keys())}"
        )

    def __getitem__(self, key):
        """
        Allow dictionary-style access: d['adc'] instead of d.data['adc']

        Parameters
        ----------
        key : str
            Sensor name (e.g., 'adc', 'gps', 'drone')

        Returns
        -------
        DataFrame or object
            The data for the requested sensor

        Examples
        --------
        >>> d = DataHandler('/path/to/flight')
        >>> d.load_data()
        >>> adc_data = d['adc']  # Instead of d.data['adc']
        >>> gps_data = d['gps']  # Instead of d.data['gps']
        """
        if not self.data:
            raise RuntimeError("No data loaded. Call load_data() first.")
        return self.data[key]

    def __contains__(self, key):
        """
        Allow 'in' operator: 'adc' in d

        Examples
        --------
        >>> if 'adc' in d:
        ...     adc_data = d['adc']
        """
        return key in self.data

    def keys(self):
        """
        Return available data keys.

        Examples
        --------
        >>> d.load_data()
        >>> print(list(d.keys()))
        ['drone', 'gps', 'adc', 'inclino', ...]
        """
        return self.data.keys()
