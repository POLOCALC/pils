# PILS - POLOCALC Inertial & Drone Loading System

A comprehensive Python package for loading, decoding, and analyzing flight data from drone missions integrated with the **STOUT** campaign management system database.

## 🎯 Key Features

- **STOUT Database Integration**: Seamlessly query campaigns, flights, and metadata from the STOUT database
- **Automatic Data Decoding**: Automatically detect and decode drone logs, sensor data, and camera files
- **Multi-Drone Support**: Parse data from DJI, BlackSquare, and Litchi drones
- **Comprehensive Sensors**: Support for GPS, IMU (accelerometer, gyroscope, barometer, magnetometer), ADC, Inclinometer, and camera data
- **Payload Synchronization**: Merge all sensors to a common time base with configurable framerate and interpolation
- **Flexible API**: Use the high-level handler or lower-level loaders and decoders as needed
- **Pip Installable**: Standard Python package installation with optional STOUT support

---

## 📦 Installation

### Basic Installation
```bash
pip install pils
```

### With STOUT Support
If you have STOUT installed and want database integration:
```bash
pip install "pils[stout]"
```

### Development Installation
```bash
git clone https://github.com/POLOCALC/pils.git
cd pils
pip install -e ".[dev,stout]"
```

---

## 🚀 Quick Start

### Loading a Single Flight (Complete Example)

```python
from pils.datahandler import Payload

# =============================================================================
# Load all payload sensor data from a flight's aux folder
# =============================================================================
aux_path = '/mnt/data/POLOCALC/campaigns/202511/20251201/flight_20251201_1515/aux'

# Initialize and load all sensors
payload = Payload(dirpath=aux_path)
payload.load_all()

# =============================================================================
# Access individual sensor data
# =============================================================================

# GPS data
if payload.gps and payload.gps.data is not None:
    gps_df = payload.gps.data
    print(f"GPS: {gps_df.shape[0]} samples")
    print(f"  Columns: {gps_df.columns}")
    print(f"  Time range: {gps_df['timestamp'].min():.1f}s - {gps_df['timestamp'].max():.1f}s")

# ADC data (analog sensors)
if payload.adc and payload.adc.data is not None:
    adc_df = payload.adc.data
    print(f"ADC: {adc_df.shape[0]} samples")
    print(f"  Columns: {adc_df.columns}")

# Inclinometer data (pitch, roll, yaw)
if payload.inclinometer and payload.inclinometer.data is not None:
    inc_df = payload.inclinometer.data
    print(f"Inclinometer: {inc_df.shape[0]} samples")
    print(f"  Columns: {inc_df.columns}")

# IMU data (barometer, accelerometer, gyroscope, magnetometer)
if payload.imu:
    if payload.imu.barometer and payload.imu.barometer.data is not None:
        baro_df = payload.imu.barometer.data
        print(f"Barometer: {baro_df.shape[0]} samples")
    
    if payload.imu.accelerometer and payload.imu.accelerometer.data is not None:
        accel_df = payload.imu.accelerometer.data
        print(f"Accelerometer: {accel_df.shape[0]} samples")
    
    if payload.imu.gyroscope and payload.imu.gyroscope.data is not None:
        gyro_df = payload.imu.gyroscope.data
        print(f"Gyroscope: {gyro_df.shape[0]} samples")
    
    if payload.imu.magnetometer and payload.imu.magnetometer.data is not None:
        mag_df = payload.imu.magnetometer.data
        print(f"Magnetometer: {mag_df.shape[0]} samples")

# =============================================================================
# Get sensor summary
# =============================================================================
print(payload.summary())

# =============================================================================
# Synchronize all sensors to a common time base
# =============================================================================
info = payload.get_sensor_info()
print(f"\nMax sample rate: {payload.get_max_sample_rate():.1f} Hz")
print(f"Common time range: {payload.get_common_time_range()}")

# Synchronize to 10 Hz (must be <= max sensor rate)
sync_df = payload.synchronize(target_rate_hz=10.0)
print(f"\nSynchronized data: {sync_df.shape[0]} samples x {sync_df.shape[1]} columns")
print(f"Columns: {sync_df.columns}")

# Save to file
payload.save_synchronized('/path/to/output/synchronized_payload.parquet', format='parquet')
```

### Using FlightDataHandler (with STOUT)

```python
from pils import FlightDataHandler

# Initialize with STOUT database
handler = FlightDataHandler(use_stout=True)

# Load and automatically decode a single flight
flight = handler.load_flight(flight_id='flight-123')

# Access decoded data
print(f"Flight: {flight.flight_name}")
print(f"Takeoff: {flight.takeoff_datetime}")

# Access payload sensors
if flight.payload and flight.payload.gps:
    gps_df = flight.payload.gps.data
    print(gps_df.head())

# Access drone telemetry
if flight.drone:
    drone_df = flight.drone.data
    print(drone_df[['latitude', 'longitude', 'altitude']])
```

### Loading Multiple Flights

```python
# Load all flights from a campaign
flights = handler.load_campaign_flights(
    campaign_id='campaign-123'
)

# Load flights by date range
flights = handler.load_flights_by_date(
    start_date='2025-01-01',
    end_date='2025-01-31',
    campaign_id='campaign-123'
)

# Process all flights
for flight in flights:
    print(f"Processing {flight.flight_name}...")
    if flight.drone:
        print(f"  Drone: {flight.drone.data.shape[0]} records")
    if flight.payload and flight.payload.imu:
        print(f"  IMU data available")
```

---

## 🏗️ Architecture & API Layers

### Layer 1: High-Level Interface (User-Friendly)

**`FlightDataHandler`** (`handler.py`)
- Recommended entry point for most users
- Automatically combines database queries and data decoding
- Provides convenient methods for common tasks
- Returns `Flight` objects with decoded data

```python
from pils import FlightDataHandler
handler = FlightDataHandler(use_stout=True)
flight = handler.load_flight(flight_id='...')
```

### Layer 2: Mid-Level Components (Flexible)

**`StoutDataLoader`** (`loader.py`)
- Queries STOUT database or filesystem
- Returns flight metadata and data paths
- No automatic decoding

```python
from pils import StoutDataLoader
loader = StoutDataLoader()
flights = loader.load_all_campaign_flights()
```

**`DataDecoder`** (`decoder.py`)
- Automatic detection and decoding of drone/sensor data
- Works with flight directory paths
- Flexible sensor loading

```python
from pils import DataDecoder
decoder = DataDecoder('/path/to/flight')
decoder.load_all()
```

### Layer 3: Low-Level Modules (Advanced)

**Sensor Decoders** (`sensors/`)
- Individual classes for each sensor type
- Direct access to raw parsing logic
- Useful for custom workflows

**Drone Parsers** (`drones/`)
- Drone-specific telemetry parsing
- Factory function for auto-detection

**Utilities** (`utils/`)
- Log file parsing
- Path handling
- Data cleaning functions

---

## 📚 Detailed Usage Examples

### Example 1: Load Flight with Custom Decoding

```python
from pils import StoutDataLoader, DataDecoder

# Step 1: Query database for flight
loader = StoutDataLoader(use_stout=True)
flight_meta = loader.load_single_flight(flight_id='flight-123')

# Step 2: Get flight directory path
flight_path = flight_meta['drone_data_folder_path'].rsplit('/drone', 1)[0]

# Step 3: Manually decode with specific drone model
decoder = DataDecoder(flight_path, drone_model='dji')
decoder.load_drone_data()
decoder.load_payload_data()

# Step 4: Access raw decoded objects
drone_data = decoder.drone.data
imu_data = decoder.payload.imu
```

### Example 2: Access Specific Sensors Only

```python
from pils import FlightDataHandler

handler = FlightDataHandler(use_stout=True)
flight = handler.load_flight(flight_id='flight-123')

# Access specific sensors
if flight.payload:
    if flight.payload.gps:
        gps_timestamps = flight.payload.gps.data['datetime']
        print(f"GPS data from {gps_timestamps.min()} to {gps_timestamps.max()}")
    
    if flight.payload.imu:
        # IMU contains barometer, accelerometer, gyroscope, magnetometer
        accel_data = flight.payload.imu.accelerometer.data
        gyro_data = flight.payload.imu.gyroscope.data
```

### Example 3: Without STOUT Database (Filesystem Only)

```python
from pils import FlightDataHandler

# Use filesystem directly without STOUT
handler = FlightDataHandler(
    use_stout=False,
    base_data_path='/mnt/data/POLOCALC'
)

# Load flights by name
flight = handler.load_flight(flight_name='flight_20250115_1430')
```

### Example 4: Payload Synchronization

```python
from pils.datahandler import Payload

# Load payload data
payload = Payload(aux_path='/path/to/flight/aux')
payload.load_all()

# Get sensor information
info = payload.get_sensor_info()
for sensor, details in info.items():
    print(f"{sensor}: {details['samples']} samples @ {details['sample_rate_hz']:.1f} Hz")

# Synchronize all sensors to a common time base
# Target rate cannot exceed the maximum sensor rate
sync_data = payload.synchronize(
    target_rate_hz=10.0,  # 10 Hz output
    sensors=['gps', 'adc', 'inclinometer', 'imu_barometer', 'imu_accelerometer'],
    method='linear'
)

# Save synchronized data
payload.save_synchronized('/path/to/output.parquet', format='parquet')
# Also supports: 'csv', 'json'

# Get summary
print(payload.summary())
```

### Example 5: Direct Sensor Access

```python
from pils.sensors.gps import GPS
from pils.sensors.IMU import IMU
from pils.sensors.inclinometer import Inclinometer
import matplotlib.pyplot as plt

# Load GPS data directly
gps = GPS(datapath='/path/to/sensors', logpath='/path/to/log')
gps.load_data()
print(gps.data.head())

# Load inclinometer data
inclino = Inclinometer(datapath='/path/to/sensors', logpath='/path/to/log')
inclino.load_data()

# Plot inclinometer angles
fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(inclino.data["timestamp"], inclino.data["yaw"], '.')
axs[1].plot(inclino.data["timestamp"], inclino.data["pitch"], '.')
axs[2].plot(inclino.data["timestamp"], inclino.data["roll"], '.')
axs[0].set_ylabel("Yaw (degrees)")
axs[1].set_ylabel("Pitch (degrees)")
axs[2].set_ylabel("Roll (degrees)")
axs[-1].set_xlabel("Time")
plt.show()
```

---

## 📦 Package Layout

```
pils/
├── __init__.py                 # Package initialization & exports
├── loader.py                   # StoutDataLoader - Database queries
├── decoder.py                  # DataDecoder - Auto data parsing
├── handler.py                  # FlightDataHandler - High-level API
├── datahandler.py              # Payload class - Sensor grouping & synchronization
├── sensors/                    # Sensor-specific decoders
│   ├── __init__.py
│   ├── camera.py              # Video/image loading
│   ├── gps.py                 # GPS (UBX binary)
│   ├── IMU.py                 # IMU sensors (baro, accel, gyro, mag)
│   ├── adc.py                 # ADC data
│   └── inclinometer.py        # Inclinometer data (Kernel-100, IMX-5)
├── drones/                     # Drone-specific parsers
│   ├── __init__.py
│   ├── DJIDrone.py            # DJI telemetry
│   ├── BlackSquareDrone.py    # BlackSquare telemetry
│   └── litchi.py              # Litchi flight plans
├── decoders/                   # Data type decoders (extensible)
│   └── __init__.py
└── utils/                      # Utility functions
    ├── __init__.py
    └── tools.py               # Log parsing, file handling
```

---

## 🔧 Object Structure Overview

### `Payload`

Wraps and loads all payload sensor components with synchronization support.

| Attribute       | Type           | Description                              |
|-----------------|----------------|------------------------------------------|
| `gps`           | `GPS`          | Global Positioning System sensor         |
| `adc`           | `ADC`          | Analog-to-digital converter data         |
| `inclinometer`  | `Inclinometer` | Measures tilt/inclination                |
| `imu`           | `IMU`          | IMU container (baro, accel, gyro, mag)   |
| `camera`        | `Camera`       | Video/image data                         |

**Methods:**
| Method | Description |
|--------|-------------|
| `load_all()` | Load all available sensors |
| `get_sensor_info()` | Get sample rates, time ranges for all sensors |
| `get_max_sample_rate()` | Returns highest sensor sample rate |
| `get_common_time_range()` | Returns overlapping time range |
| `synchronize(target_rate_hz, sensors, method)` | Create unified DataFrame |
| `save_synchronized(output_path, format)` | Export to parquet/csv/json |
| `summary()` | Text summary of loaded sensors |

### Drone Classes

| Class               | Source                      | Description                            |
|---------------------|-----------------------------|----------------------------------------|
| `DJIDrone`          | `drones.DJIDrone`           | Handles DJI drone log parsing          |
| `BlackSquareDrone`  | `drones.BlackSquareDrone`   | Handles BlackSquare drone log parsing  |
| `Litchi`            | `drones.litchi`             | Parses Litchi CSV flight plans/logs    |

### `IMU`

Container for IMU sub-sensors.

| Attribute       | Type         | Description                    |
|-----------------|--------------|--------------------------------|
| `barometer`     | `IMUSensor`  | Barometric pressure sensor     |
| `accelerometer` | `IMUSensor`  | Accelerometer for motion       |
| `gyroscope`     | `IMUSensor`  | Gyroscope for rotation         |
| `magnetometer`  | `IMUSensor`  | Magnetometer for orientation   |

---

## 📋 Supported Data Types

### Drones
- ✅ **DJI** - CSV telemetry logs
- ✅ **BlackSquare** - Custom drone telemetry  
- ✅ **Litchi** - Flight plans and waypoint data

### Sensors
- ✅ **GPS** - UBX binary format (u-blox)
- ✅ **IMU**
  - Barometer (pressure, temperature)
  - Accelerometer (X, Y, Z)
  - Gyroscope (X, Y, Z)
  - Magnetometer (X, Y, Z)
- ✅ **ADC** - Analog-to-digital converter (auto-detect gain from config.yml)
- ✅ **Inclinometer** - Kernel-100 (binary) and IMX-5 (CSV) formats
- ✅ **Camera** - Video files and image sequences

---

## 📁 Data Structure

The package expects data organized as:

```
base_data_path/
└── campaigns/
    └── Campaign_Name/
        └── 20250115/                    # YYYYMMDD
            └── flight_20250115_1430/    # flight_YYYYMMDD_HHMM
                ├── drone/
                │   ├── 20250115_143000_drone.csv  # Drone telemetry
                │   └── ...other files...
                │
                ├── aux/
                │   ├── sensors/
                │   │   ├── gps.bin                 # GPS data (UBX format)
                │   │   ├── barometer.bin           # Barometric sensor
                │   │   ├── accelerometer.bin       # Accelerometer
                │   │   ├── gyroscope.bin           # Gyroscope
                │   │   ├── magnetometer.bin        # Magnetometer
                │   │   ├── adc.bin                 # ADC data
                │   │   └── inclinometer.log        # Inclinometer data
                │   ├── config.yml                  # Flight configuration
                │   └── *.log                       # Log files
                │
                └── proc/
                    └── ...processed data files...
```

---

## 🔄 Integration with STOUT

When `use_stout=True`, the loader queries the STOUT database to:
- Get campaign and flight metadata
- Access automatically managed file paths
- Query by date, campaign, or flight ID
- Track flight relationships and payloads

Without STOUT, the loader scans the filesystem structure directly.

---

## 📖 API Reference

### StoutDataLoader
```python
loader = StoutDataLoader(use_stout=True, base_data_path=None)
loader.load_all_campaign_flights() -> List[Dict]
loader.load_single_flight(flight_id=None, flight_name=None) -> Dict
loader.load_flights_by_date(start_date, end_date, campaign_id=None) -> List[Dict]
loader.get_campaign_list() -> List[Dict]
```

### DataDecoder
```python
decoder = DataDecoder(flight_path, drone_model=None)
decoder.load_all() -> Dict
decoder.load_drone_data() -> None
decoder.load_payload_data() -> None
decoder.load_specific_sensors(sensor_types) -> Dict
decoder.get_data_summary() -> Dict
```

### FlightDataHandler
```python
handler = FlightDataHandler(use_stout=True, base_data_path=None, auto_decode=True)
handler.load_flight(flight_id=None, flight_name=None, decode=None) -> Flight
handler.load_campaign_flights(campaign_id, date_range=None, decode=None) -> List[Flight]
handler.load_flights_by_date(start_date, end_date, campaign_id=None) -> List[Flight]
handler.get_campaigns() -> List[Dict]
```

### Flight Object
```python
flight.flight_id -> str
flight.flight_name -> str
flight.flight_path -> str
flight.campaign_id -> str
flight.takeoff_datetime -> datetime
flight.drone -> Drone data object
flight.payload -> Payload object with sensors
flight.litchi -> Litchi flight plan
flight.raw_files -> Dict of file catalogs
```

### Payload
```python
payload = Payload(aux_path, config_path=None)
payload.load_all() -> None
payload.get_sensor_info() -> Dict[str, Dict]
payload.get_max_sample_rate() -> float
payload.get_common_time_range() -> Tuple[float, float]
payload.synchronize(target_rate_hz, sensors=None, method='linear') -> pl.DataFrame
payload.save_synchronized(output_path, format='parquet') -> None
payload.summary() -> str
```

---

## 🔌 Requirements

### Core Dependencies
- polars >= 0.19.0
- numpy >= 1.23.0
- opencv-python >= 4.6.0
- matplotlib >= 3.5.0
- pyubx2 >= 1.2.0
- pyyaml >= 6.0

### Optional Dependencies
- stout >= 2.0.0 (for STOUT database integration)

### Development Dependencies
- pytest
- pytest-cov
- black
- flake8
- mypy

---

## 🛠️ Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Coverage
```bash
pytest tests/ --cov=pils --cov-report=html
```

### Linting
```bash
flake8 pils/
black pils/
mypy pils/
```

---

## 🚀 Extension Points

### Adding New Drone Types
1. Create new class in `drones/`
2. Inherit parsing logic
3. Update `drone_init()` factory function
4. Document in sensor files

### Adding New Sensor Types
1. Create new class in `sensors/`
2. Implement `load_data()` method returning polars DataFrame
3. Update Payload class to include new sensor
4. Update DataDecoder sensor detection

### Custom Decoders
Use `decoders/` folder to add custom data type parsers:
1. Create decoder module
2. Register with DataDecoder
3. Use `load_specific_sensors()` for custom types

---

## 🔀 Data Synchronization

### Overview

The **`Synchronizer`** class provides unified time-series data synchronization across heterogeneous data sources. It enables you to merge payload sensor data with drone data (or any other time-series data) to a common time base.

### Key Features

- **Flexible Data Sources**: Add any data source with a timestamp column
- **Automatic Interpolation**: Linear interpolation to target sample rate
- **Multi-Format Output**: Save as Parquet, CSV, or JSON
- **Source Prefixing**: Automatic column naming with configurable prefixes
- **Time Range Analysis**: Automatic detection of overlapping time windows

### Usage Example

```python
from pils.datahandler import Payload
from pils.synchronizer import Synchronizer
import polars as pl

# Load payload data
aux_path = '/path/to/flight/aux'
payload = Payload(dirpath=aux_path)
payload.load_all()

# Create synchronizer
sync = Synchronizer()

# Add payload GPS data as source
if payload.gps and payload.gps.data:
    sync.add_source(
        'payload',
        payload.gps.data.select(['timestamp', 'lon', 'lat', 'height']),
        timestamp_col='timestamp',
        prefix='gps_'
    )

# Add drone data
drone_data = pl.read_parquet('/path/to/drone/data.parquet')
sync.add_source(
    'drone',
    drone_data,
    timestamp_col='timestamp',
    prefix='drone_'
)

# Synchronize to 2 Hz
synchronized = sync.synchronize(target_rate_hz=2.0)

# Save results
sync.save_synchronized('synchronized_data.parquet', format='parquet')
```

### API Reference

#### `Synchronizer` Class

**Methods:**

- `add_source(name, data, timestamp_col='timestamp', prefix=None)` - Add a data source
  - `name`: Unique identifier for the source
  - `data`: Polars DataFrame with time-series data
  - `timestamp_col`: Name of the timestamp column
  - `prefix`: Optional prefix for output columns (defaults to `name_`)

- `remove_source(name)` - Remove a data source

- `get_source_info()` - Get metadata about all sources
  - Returns dict with sample count, duration, sample rate, time range

- `get_max_sample_rate()` - Get maximum sample rate across sources

- `get_common_time_range()` - Get overlapping time window across all sources

- `synchronize(target_rate_hz=None, sources=None, source_columns=None, method='linear')` - Perform synchronization
  - `target_rate_hz`: Target sample rate (defaults to max source rate)
  - `sources`: List of source names to include (defaults to all)
  - `source_columns`: Dict mapping source names to column lists to include
  - `method`: Interpolation method (currently 'linear')
  - Returns: Polars DataFrame with synchronized data

- `save_synchronized(output_path, format='parquet')` - Save synchronized data
  - `format`: 'parquet', 'csv', or 'json'

### Synchronization Workflow

1. **Create Synchronizer**
   ```python
   sync = Synchronizer()
   ```

2. **Add Data Sources**
   ```python
   sync.add_source('source_name', dataframe, timestamp_col='timestamp')
   ```

3. **Analyze Sources** (optional)
   ```python
   info = sync.get_source_info()
   t_start, t_end = sync.get_common_time_range()
   ```

4. **Synchronize**
   ```python
   synced = sync.synchronize(target_rate_hz=10.0)
   ```

5. **Export Results**
   ```python
   sync.save_synchronized('output.parquet')
   ```

### Time Interpolation

The synchronizer uses numpy linear interpolation to resample data:
- Points outside the common time range are filled with NaN
- Missing values between source samples are interpolated linearly
- Interpolation preserves the temporal relationships in the data

---

## 🔄 Version History

- **0.1.0** - Initial release with core functionality
  - STOUT database integration
  - Multi-drone support (DJI, BlackSquare, Litchi)
  - Comprehensive sensor support
  - High-level and low-level APIs
  - Polars-based data processing
  - Payload synchronization with interpolation
  - **NEW**: Separate `Synchronizer` class for flexible data source merging

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📧 Contact

For issues, questions, or suggestions, please open an issue on GitHub or contact the POLOCALC team.

---

## 🔗 Related Projects

- [STOUT](https://github.com/POLOCALC/stout) - Campaign management system
- [POLOCALC](https://github.com/POLOCALC) - Organization repository
