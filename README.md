# PILS - POLOCALC Inertial & drone Loader System

This repository provides tools to load and visualize data from drone missions, including drone logs, payload sensor data, Litchi flight logs, and (optionally) photogrammetry data.

## Installation

### Option 1: Install with pip (recommended for development)

```bash
cd /path/to/data_loader
pip install -e .
```

### Option 2: Install with conda environment

If you're using conda and need opencv:

```bash
# Create/activate your conda environment first
conda install -c conda-forge numpy pandas matplotlib astropy

# Install opencv via conda (avoids dependency conflicts)
conda install -c conda-forge opencv

# Then install pils without opencv dependency
pip install -e . --no-deps
pip install pyubx2  # Install remaining pip-only dependencies
```

### Option 3: Install with opencv support via pip

```bash
pip install -e ".[cv]"  # Installs opencv-python from PyPI
```

### Option 4: Install with PDF report generation support

```bash
pip install -e ".[report]"  # Installs jinja2, weasyprint, markdown
# OR install everything:
pip install -e ".[report,dev]"
```

## Usage

### Basic Example

```python
import pils

# Automatic detection: if path exists, loads from filesystem; otherwise from database
handler = pils.PathHandler("flight_20240715_1430")  # Loads from database
# OR
handler = pils.PathHandler("/path/to/flight/data")  # Loads from filesystem

handler.load_flight_data()

# Access drone and sensor files
drone_files = handler.get_drone_files("*.csv")
gps_files = handler.get_aux_files("**/ZED-F9P*")
```

### DataHandler with Dictionary Access

The `DataHandler` provides convenient dictionary access to all loaded data using `__getitem__`:

```python
from pils import DataHandler

# Initialize (auto-detects path vs flight name)
d = DataHandler('flight_20240715_1430')  # Database mode
# OR
d = DataHandler('/path/to/flight/data')  # Filesystem mode

# Load all data
d.load_data()

# Direct dictionary access - CLEANEST!
adc_data = d['adc']
gps_data = d['gps']
drone_data = d['drone']

# Check if sensor data is available
if 'adc' in d:
    print(d['adc'].head())

# See what's available
print(list(d.keys()))  # ['drone', 'litchi', 'gps', 'adc', 'inclino', ...]

# Other ways that still work:
# adc_data = d.data['adc']           # Via .data attribute
# adc_data = d.payload.adc.data      # OLD WAY (still works)
```

### PDF Report Generation

Generate comprehensive PDF reports with plots and analysis:

```python
from pils import DataHandler, FlightReport, quick_report

# Load data
d = DataHandler('/path/to/flight/data')
d.load_data()

# Option 1: Quick automated report
pdf_path = quick_report(d, '~/flight_report.pdf')

# Option 2: Custom report
report = FlightReport(d, title="My Flight Analysis")
report.add_data_summary()
report.add_section("GPS Analysis", "Analyzing GPS coordinates...")
report.add_plot_from_data('gps', 'datetime', 'latitude', 
                         title='GPS Latitude', ylabel='Latitude (deg)')
report.add_plot_from_data('inclino', 'datetime', ['pitch', 'roll', 'yaw'],
                         title='Inclinometer Angles')
report.add_statistics_table('gps', columns=['latitude', 'longitude', 'altitude'])
pdf_path = report.generate('~/custom_report.pdf')

# Option 3: Add custom matplotlib plots
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(d['gps']['longitude'], d['gps']['latitude'])
ax.set_title('Flight Path')
report.add_plot(fig, caption="GPS Track")
```

**Note:** Report generation requires additional dependencies:
```bash
pip install -e ".[report]"  # Installs jinja2, weasyprint, markdown
```

### Enable Logging (Optional)

By default, pils runs in silent mode to avoid cluttering console output. If you want to see internal logging messages from the stout database integration:

```python
import pils

# Enable logging to see debug information
pils.enable_stout_logging(verbose=True)

# Now you'll see logging messages
handler = pils.PathHandler("flight_20240715_1430")
```

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
- All components have a `data` attribute

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
