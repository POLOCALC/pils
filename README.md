# POLOCALC Data Loader

This repository provides tools to load and visualize data from drone missions, including drone logs, payload sensor data, Litchi flight logs, and (optionally) photogrammetry data.

## Requirements

Ensure you have the following Python packages installed:

* `matplotlib`
* `pandas`
* Your local `datahandler` module and its dependencies (e.g., `DJIDrone`, `IMUSensor`, etc.)

## Examples

```python
from datahandler import DataHandler
import matplotlib.pyplot as plt

# Define the base directory for the campaign and the flight number
dirpath = "/data/POLOCALC/campaigns/2025_04"
num = 522

# Initialize and load all available data
DH = DataHandler(num=num, dirpath=dirpath)
DH.load_data()

# Plot pitch from the inclinometer sensor
plt.figure()
plt.plot(DH.payload.inclino.data["datetime"], DH.payload.inclino.data["pitch"])
plt.xlabel("Time")
plt.ylabel("Pitch (degrees)")
plt.title("Inclinometer Pitch Over Time")
plt.show()
```

Otherwise, any sensor can be read independantly using the corresponding script and following a similar structure

```python
from inclinometer import Inclinometer
import matplotlib.pyplot as plt

datapath = "/path/to/data"
logpath = "/path/to/logfile"

inclino = Inclinometer(datapath=datapath, logpath=logpath)
inclino.load_data()

fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(inclino.data["datetime"], inclino.data["yaw"], '.')
axs[1].plot(inclino.data["datetime"], inclino.data["pitch"], '.')
axs[2].plot(inclino.data["datetime"], inclino.data["roll"], '.')
axs[0].set_ylabel("Yaw (degrees)")
axs[1].set_ylabel("Pitch (degrees)")
axs[2].set_ylabel("Roll (degrees)")
axs[-1].set_xlabel("Time")

plt.show()
```

## 📦 Object Structure Overview

The `DataHandler` class organizes and loads drone and sensor data from logs using a modular design. It integrates drone-specific data (DJI or BlackSquare), payload sensor data, and Litchi flight plans.

---

### 🔧 `DataHandler`

Main class to coordinate data loading.

| Attribute        | Type                          | Description                                                                 |
|------------------|-------------------------------|-----------------------------------------------------------------------------|
| `paths`          | `PathHandler`                 | Handles path discovery and filename organization for all logs.             |
| `drone_model`    | `str`                         | Drone model name (e.g., `"dji"`, `"blacksquare"`).                         |
| `drone`          | `DJIDrone` or `BlackSquareDrone` | Instantiated based on `drone_model`, contains drone log data.         |
| `payload`        | `Payload`                     | Contains data from onboard sensors like GPS, barometer, and IMUs.          |
| `litchi`         | `Litchi`                      | Parses and stores Litchi waypoint and telemetry data.                      |
| `photogrammetry` | `None`                        | Placeholder for future photogrammetry integration.                         |

---

### 🎛️ `Payload`

Wraps and loads all payload sensor components.

| Attribute    | Type           | Description                         |
|--------------|----------------|-------------------------------------|
| `gps`        | `GPS`          | Global Positioning System sensor.   |
| `adc`        | `ADC`          | Analog-to-digital converter data.   |
| `inclino`    | `Inclinometer` | Measures tilt or inclination.       |
| `baro`       | `IMUSensor`    | Barometric pressure sensor.         |
| `accelero`   | `IMUSensor`    | Accelerometer for motion detection. |
| `magneto`    | `IMUSensor`    | Magnetometer for orientation.       |
| `gyro`       | `IMUSensor`    | Gyroscope for rotation sensing.     |

---

### 🚁 Drone Classes

Instantiated via `drone_init(drone_model, path)`:

| Class               | Source                      | Description                            |
|---------------------|-----------------------------|----------------------------------------|
| `DJIDrone`          | `drones.DJIDrone`           | Handles DJI drone log parsing.         |
| `BlackSquareDrone`  | `drones.BlackSquareDrone`   | Handles BlackSquare drone log parsing. |

---

### 🗺️ Litchi

| Class    | Source          | Description                                |
|----------|------------------|--------------------------------------------|
| `Litchi` | `drones.litchi` | Parses Litchi CSV flight plans and logs.   |

---

### 📁 PathHandler

| Class         | Source            | Description                                                  |
|---------------|-------------------|--------------------------------------------------------------|
| `PathHandler` | `pathhandler.py`  | Finds and stores file paths to sensor and drone data files. |

---

### 🛠️ Notes

- All components support a `load_data()` method if applicable.
- `PathHandler` must be initialized first to provide paths to all modules.
- The `Payload` class automatically initializes and links all relevant sensors.


---

## Utilities

This repository also contains two Python scripts designed for quick visualization and structured export of sensor log data:

* Generates plots and a descriptive PDF summary report.
* Processes raw sensor logs and converts them into clean, time-aligned CSV files.

### quick\_look.py – Quick Visualization

This script is used for exploratory analysis and test validation. It creates a multi-page PDF containing:

* Line plots (e.g., GPS position, accelerometer, barometer).
* Sampling time diagnostics.

### process\_data.py – CSV Export

Converts raw logs into aligned CSV files suitable for downstream analysis or archiving.

### ▶️ Example Usage

```bash
python utils/quick_look.py /path/to/logs
python utils/process_data.py /path/to/logs
```
