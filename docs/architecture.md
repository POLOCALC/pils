# Architecture Overview

PILS is designed with modularity, extensibility, and data consistency in mind.

## System Design

```
┌─────────────────────────────────────────────────────────────┐
│                        User Code                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flight Container                         │
│  - Unified interface for all flight data                    │
│  - HDF5 persistence                                         │
│  - Metadata management                                      │
└────────┬─────────────────────────────────────┬──────────────┘
         │                                     │
         ▼                                     ▼
┌─────────────────────┐            ┌─────────────────────────┐
│      Loaders        │            │    Synchronizer         │
│  - PathLoader       │            │  - Time alignment       │
│  - StoutLoader      │            │  - Correlation sync     │
└──────────┬──────────┘            │  - GPS/IMU sync         │
           │                       └─────────────────────────┘
           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources                             │
├────────────────┬────────────────┬────────────────┬──────────┤
│   Drones       │   Sensors      │   Decoders     │ Analysis │
│  - DJIDrone    │  - GPS         │  - KERNEL      │  - PPK   │
│  - BlackSquare │  - IMU         │  - Binary      │  - RTK   │
│  - Litchi      │  - Camera      │  - NMEA        │          │
│                │  - ADC         │                │          │
│                │  - Inclinometer│                │          │
└────────────────┴────────────────┴────────────────┴──────────┘
```

## Core Components

### 1. Flight Container (`pils/flight.py`)

Central data structure that aggregates all flight information.

**Responsibilities:**
- Load drone and sensor data
- Manage metadata (date, time, location)
- Provide unified data access
- Handle HDF5 serialization
- Coordinate synchronization

**Key Methods:**
```python
flight = Flight(flight_info)
flight.add_drone_data()              # Load drone platform data
flight.add_sensor_data(['gps', 'imu'])  # Load specific sensors
flight.to_hdf5('output.h5')          # Save to HDF5
Flight.from_hdf5('output.h5')        # Load from HDF5
```

### 2. Loaders (`pils/loader/`)

Load flight metadata and file paths from various sources.

#### PathLoader (`path.py`)
- Filesystem-based loading
- Scans campaign directories
- Builds flight dictionaries from folder structure
- Filters by date, name, campaign

#### StoutLoader (`stout.py`)
- Database-based loading
- Queries PostgreSQL/MySQL
- Legacy support for existing infrastructure
- Additional metadata from database

**Usage:**
```python
loader = PathLoader('/campaigns')
flights = loader.load_all_flights()
flight = loader.load_single_flight(flight_name='...')
```

### 3. Drone Platforms (`pils/drones/`)

Platform-specific data loaders for different drone types.

#### DJIDrone (`DJIDrone.py`)
- DJI drone CSV and DAT formats
- Encrypted DAT file decryption
- GPS/RTK data extraction
- Tick-based synchronization

#### BlackSquareDrone (`BlackSquareDrone.py`)
- ArduPilot log file parsing
- Multiple message types (GPS, IMU, ATT, etc.)
- GPS time to UTC conversion
- Leap second handling

#### Litchi (`litchi.py`)
- Litchi CSV format
- Simple waypoint missions
- Timestamp conversion
- Minimal processing

### 4. Sensors (`pils/sensors/`)

Sensor-specific decoders for various instruments.

#### GPS (`gps.py`)
- Binary GPS data decoding
- NMEA message parsing
- Multiple frame types (NAV, TP, etc.)
- Frequency interpolation

#### IMU (`IMU.py`)
- Accelerometer, gyroscope, magnetometer, barometer
- Binary format decoding
- Timestamp conversion
- Multi-sensor coordination

#### Camera (`camera.py`)
- Video file reading (MP4, AVI, etc.)
- Image sequence loading
- Frame extraction
- Timestamp indexing

#### ADC (`adc.py`)
- Analog-to-digital converter data
- Binary and ASCII formats
- Gain configuration
- Voltage conversion

#### Inclinometer (`inclinometer.py`)
- Kernel-100 format
- IMX-5 format
- Auto-detection
- INS/IMU data

### 5. Decoders (`pils/decoders/`)

Low-level binary format decoders.

#### KERNEL_utils (`KERNEL_utils.py`)
- Kernel inclinometer protocol
- Message checksum validation
- Multi-message decoding
- Binary struct parsing

### 6. Synchronizer (`pils/synchronizer.py`)

Time alignment and sensor correlation.

**Features:**
- GPS offset detection (cross-correlation)
- Pitch offset detection
- Common timebase resampling
- Metadata tracking

**Usage:**
```python
sync = Synchronizer()
sync.add_gps_reference(ref_gps)
sync.add_drone_gps(drone_gps)
offsets = sync.synchronize()
```

### 7. Analysis Tools (`pils/analyze/`)

Post-processing and analysis capabilities.

#### PPK Analysis (`ppk.py`)
- RTKLIB integration
- Configuration management
- Version control
- HDF5 persistence
- Smart re-execution

**Usage:**
```python
ppk = PPKAnalysis(flight_path)
version = ppk.run_analysis(base_station='REFERENCE')
pos_df = version.position
stats_df = version.statistics
```

## Data Flow

### Loading Workflow

```
1. Loader (PathLoader/StoutLoader)
   └─> Scans filesystem/database
   └─> Returns flight_info dict

2. Flight Container
   └─> Creates Flight object
   └─> Stores metadata

3. Drone Data
   └─> Auto-detects platform (DJI/BlackSquare/Litchi)
   └─> Loads drone-specific data
   └─> Stores in flight.drone_data

4. Sensor Data
   └─> For each requested sensor:
       └─> Instantiate sensor class
       └─> Decode binary/ASCII data
       └─> Store in flight.raw_data

5. Synchronization (optional)
   └─> Align timestamps
   └─> Resample to common timebase
   └─> Store synchronized data

6. Persistence
   └─> Save to HDF5 file
   └─> Store all data + metadata
```

### Data Structures

All sensor data is stored as **Polars DataFrames** for:
- High performance
- Memory efficiency
- Type safety
- Lazy evaluation support
- Arrow compatibility

Example structure:
```python
gps_df = pl.DataFrame({
    'timestamp': [1000, 2000, 3000],  # int64 (milliseconds)
    'latitude': [40.7, 40.8, 40.9],   # float64 (degrees)
    'longitude': [-74.0, -74.1, -74.2],  # float64 (degrees)
    'altitude': [10.0, 11.0, 12.0],   # float64 (meters)
    'fix_quality': [4, 4, 4],         # int32 (RTK fixed)
})
```

## File Organization

### Campaign Structure

```
campaigns/
└── 202511/                    # Year-Month
    └── 20251208/              # Year-Month-Day
        └── flight_20251208_1506/  # flight_YYYYMMDD_HHMM
            ├── aux/
            │   ├── sensors/
            │   │   ├── gps.bin
            │   │   ├── imu/
            │   │   ├── adc.dat
            │   │   └── inclino.bin
            │   ├── camera/
            │   │   └── video.mp4
            │   └── drone_data/
            │       └── flight.csv
            └── ppk/               # PPK analysis results
                ├── ppk_data.h5    # HDF5 with all versions
                └── v_20250102_143022/  # Version folder
                    ├── config.conf
                    ├── rover.pos
                    └── rover.stat
```

### HDF5 Structure

```
flight.h5
├── metadata/                  # Flight metadata (attributes)
│   ├── flight_name
│   ├── date
│   ├── campaign
│   └── ...
├── drone_data/               # Drone platform data
│   └── data (DataFrame)
└── raw_data/                 # Sensor data
    ├── gps (DataFrame)
    ├── imu/
    │   ├── accelerometer
    │   ├── gyroscope
    │   ├── magnetometer
    │   └── barometer
    ├── camera (metadata only)
    └── adc (DataFrame)
```

## Design Principles

### 1. Polars-First
- All DataFrames are Polars (not Pandas)
- Leverages lazy evaluation
- Type-safe operations
- High performance

### 2. Pathlib over os.path
- Modern path handling
- Cross-platform compatibility
- Cleaner API

### 3. Type Hints
- All functions have type hints
- Modern `str | Path` syntax (Python 3.10+)
- Improves IDE support and catches errors

### 4. Centralized Logging
- No `print()` statements
- Use `pils.utils.logging_config.get_logger(__name__)`
- Consistent log levels (DEBUG, INFO, WARNING, ERROR)

### 5. Test-Driven Development
- Tests written first (RED)
- Minimal implementation (GREEN)
- Refactor and clean (REFACTOR)
- All tests pass (VERIFY)

### 6. Modular Design
- Each sensor is independent
- Loaders are pluggable
- Easy to add new platforms
- Clear separation of concerns

## Extension Points

### Adding a New Sensor

1. Create sensor class in `pils/sensors/`:
```python
class NewSensor:
    def __init__(self, path: str | Path, **kwargs) -> None:
        self.path = Path(path)
        self.data = None
    
    def load_data(self) -> None:
        # Decode data from files
        self.data = pl.DataFrame(...)
```

2. Register in `pils/sensors/sensors.py`:
```python
sensor_config["newsensor"] = {
    "class": NewSensor,
    "load_method": "load_data"
}
```

3. Write tests in `tests/test_sensors_newsensor.py`

### Adding a New Drone Platform

1. Create drone class in `pils/drones/`:
```python
class NewDrone:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data = {}
    
    def load_data(self) -> None:
        # Parse drone-specific format
        self.data = {...}
```

2. Update platform detection in `pils/drones/__init__.py`

3. Write tests in `tests/test_drones_newdrone.py`

## Performance Considerations

- **Lazy Loading**: Data loaded only when accessed
- **Polars DataFrames**: 10-100x faster than Pandas
- **Binary Formats**: Use struct for efficient parsing
- **HDF5 Storage**: Fast read/write, compression
- **Chunked Processing**: For large datasets, process in chunks

## Next Steps

- [Loaders Documentation](loaders.md)
- [Sensors Documentation](sensors.md)
- [API Reference](api_reference.md)
