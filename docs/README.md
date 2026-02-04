# PILS Documentation

**PILS** (POLOCALC Inertial & Drone Loading System) is a Python library for loading, processing, and analyzing drone flight data from various platforms and sensors.

## Table of Contents

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [Architecture](architecture.md)
- [Loaders](loaders.md)
- [Sensors](sensors.md)
- [Drones](drones.md)
- [Flight Container](flight.md)
- [Synchronization](synchronization.md)
- [PPK Analysis](ppk_analysis.md)
- [API Reference](api_reference.md)
- [Testing](testing.md)
- [Contributing](contributing.md)

## Overview

PILS provides a unified interface for working with drone flight data from multiple sources:

- **Multiple Drone Platforms**: DJI, ArduPilot (BlackSquare), Litchi
- **Various Sensors**: GPS, IMU, Camera, ADC, Inclinometer
- **Data Loaders**: Filesystem-based (PathLoader) and Database-based (StoutLoader)
- **Synchronization**: Cross-platform sensor alignment and correlation
- **PPK Analysis**: Post-processed kinematic analysis with RTKLIB integration
- **HDF5 Storage**: Efficient binary storage for processed flight data

## Key Features

### 🚁 Multi-Platform Support
Load flight data from various drone platforms with unified APIs.

### 📊 Rich Sensor Support
Access GPS, IMU, camera, ADC, and inclinometer data with consistent interfaces.

### 🔄 Data Synchronization
Align sensor data across different sampling rates and time bases.

### 📈 PPK Analysis
Perform high-precision positioning with post-processed kinematic workflows.

### 💾 Efficient Storage
Store and retrieve flight data in HDF5 format for fast access.

### ✅ Fully Tested
Comprehensive test suite with 63% code coverage and growing.

## Quick Example

```python
from pils.loader import PathLoader
from pils.flight import Flight

# Load flight data
loader = PathLoader('/path/to/campaigns')
flight_info = loader.load_single_flight(flight_name='flight_20251208_1506')

# Create flight container
flight = Flight(flight_info)
flight.add_drone_data()  # Auto-detect platform
flight.add_sensor_data(['gps', 'imu', 'camera'])

# Access sensor data
gps_df = flight['gps'].data  # Returns Polars DataFrame
print(f"GPS samples: {gps_df.shape[0]}")

# Save to HDF5
flight.to_hdf5('flight_data.h5')
```

## Project Structure

```
pils/
├── loader/          # Data loading from filesystem/database
├── sensors/         # Sensor decoders (GPS, IMU, Camera, etc.)
├── drones/          # Drone platform loaders (DJI, BlackSquare, Litchi)
├── decoders/        # Low-level binary decoders
├── analyze/         # Analysis tools (PPK, RTK)
├── config/          # Configuration management
├── utils/           # Utility functions
├── flight.py        # Main Flight container class
└── synchronizer.py  # Sensor synchronization
```

## Documentation

See the individual documentation files for detailed information:

- **[Installation Guide](installation.md)** - Setup and dependencies
- **[Quick Start](quickstart.md)** - Get started in 5 minutes
- **[Architecture](architecture.md)** - System design and data flow
- **[Loaders](loaders.md)** - PathLoader and StoutLoader usage
- **[Sensors](sensors.md)** - Sensor decoder documentation
- **[Drones](drones.md)** - Drone platform support
- **[Flight Container](flight.md)** - Flight class API
- **[Synchronization](synchronization.md)** - Time alignment strategies
- **[PPK Analysis](ppk_analysis.md)** - Post-processed kinematic workflows
- **[API Reference](api_reference.md)** - Complete API documentation
- **[Testing](testing.md)** - Running and writing tests
- **[Contributing](contributing.md)** - Development guidelines

## Requirements

- Python 3.10+
- Polars >= 0.20.0 (primary DataFrame library)
- NumPy, SciPy (numerical computing)
- h5py (HDF5 storage)
- OpenCV (camera/video processing)
- pytest (testing)

## Support

For bugs, feature requests, or questions:
- Open an issue on GitHub
- Contact the development team

## License

See LICENSE file for details.
