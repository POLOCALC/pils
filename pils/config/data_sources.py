"""
Data source configuration for PILS loader.

This module contains the mapping dictionaries for sensors and drones.
Each entry defines how to load a specific data type.

To add a new sensor or drone:
1. Add an entry to the appropriate dictionary
2. Ensure the class has a compatible interface:
   - __init__(path) or __init__(dirpath) for directory-based sources
   - load_data() method
   - data attribute containing the loaded DataFrame

Note: Folder paths (aux_data_folder_path, drone_data_folder_path) are
automatically retrieved from flight_info - no need to specify them here.
"""

# Sensor mapping: sensor_type -> configuration
# Each sensor config contains:
#   - module: relative import path from pils package
#   - class: class name to instantiate
#   - patterns: file patterns to search for (glob patterns)
#   - is_directory: True if the sensor loads from a directory (optional, default False)
#
# All sensors are loaded from flight_info['aux_data_folder_path']
SENSOR_MAP = {
    "gps": {
        "module": ".sensors.gps",
        "class": "GPS",
        "patterns": [
            "sensors/*_GPS.bin",
            "sensors/*.ubx",
            "sensors/*.bin",
            "*_GPS.bin",
            "*.ubx",
            "*.bin",
        ],
    },
    "imu": {
        "module": ".sensors.IMU",
        "class": "IMU",
        "patterns": ["sensors/imu/", "sensors/IMU/", "**/imu/", "**/IMU/"],
        "is_directory": True,
    },
    "adc": {
        "module": ".sensors.adc",
        "class": "ADC",
        "patterns": [
            "sensors/*_ADC.bin",
            "sensors/*.adc",
            "sensors/*.txt",
            "*_ADC.bin",
            "*.adc",
            "*.txt",
        ],
    },
    "camera": {
        "module": ".sensors.camera",
        "class": "Camera",
        "patterns": [
            "sensors/camera/*.mp4",
            "sensors/camera/*.avi",
            "sensors/camera/*.mov",
            "*.mp4",
            "*.avi",
            "*.mov",
        ],
    },
    "inclinometer": {
        "module": ".sensors.inclinometer",
        "class": "Inclinometer",
        "patterns": ["sensors/*.csv", "sensors/*.kernel", "*.csv", "*.kernel"],
    },
}

# Drone mapping: drone_type -> configuration
# Each drone config contains:
#   - module: relative import path from pils package
#   - class: class name to instantiate
#   - patterns: file patterns to search for (glob patterns)
#
# All drones are loaded from flight_info['drone_data_folder_path']
DRONE_MAP = {
    "dji": {
        "module": ".drones.DJIDrone",
        "class": "DJIDrone",
        "patterns": ["*.dat", "*_drone.dat", "*.csv", "*_drone.csv", "*.DAT"],
    },
    "litchi": {
        "module": ".drones.litchi",
        "class": "Litchi",
        "patterns": ["*litchi*.csv", "*Litchi*.csv"],
    },
    "blacksquare": {
        "module": ".drones.BlackSquareDrone",
        "class": "BlackSquareDrone",
        "patterns": ["*.log"],
    },
}
