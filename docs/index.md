# PILS Documentation

**POLOCALC Inertial & Drone Loading System**

Welcome to PILS, a Python library for loading, processing, and analyzing drone flight data with integrated sensor fusion capabilities.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Getting Started__

    ---

    Install PILS and run your first flight analysis in under 5 minutes

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Complete documentation for all classes, methods, and data types

    [:octicons-arrow-right-24: API Docs](api/index.md)

-   :material-database:{ .lg .middle } __Data Formats__

    ---

    Detailed schemas for all DataFrame types and file formats

    [:octicons-arrow-right-24: Data Formats](data-formats/index.md)

-   :material-cog:{ .lg .middle } __Development__

    ---

    Architecture, testing, and contribution guidelines

    [:octicons-arrow-right-24: Development](development/index.md)

</div>

---

## What is PILS?

PILS is designed for drone-based geophysical surveys, providing:

- **Multi-platform support**: DJI, ArduPilot (BlackSquare), and Litchi
- **Sensor integration**: GPS, IMU, ADC, Camera, Inclinometer
- **Time synchronization**: Align data from multiple sources
- **PPK processing**: Post-Processed Kinematic analysis with version control
- **Polars-native**: High-performance DataFrames throughout

## Quick Example

```python
from pils.loader.path import PathLoader
from pils.flight import Flight

# Load a flight
loader = PathLoader(base_path="/path/to/campaigns")
flight_info = loader.load_single_flight(flight_id="flight_20251208_1506")

# Create flight container
flight = Flight(flight_info)
flight.add_drone_data()
flight.add_sensor_data(['gps', 'imu'])

# Access data
gps_df = flight['gps'].data
print(gps_df.head())
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Polars DataFrames** | All data returned as high-performance Polars DataFrames |
| **Type Safety** | Full type hints on all public APIs |
| **HDF5 Export** | Persistent storage with compression |
| **PPK Versioning** | Track and compare multiple PPK solutions |
| **Lazy Loading** | Load only what you need |
| **TDD Tested** | 283+ tests with 63% coverage |

## Supported Platforms

### Drone Platforms

- **DJI**: Phantom, Mavic, Matrice series (CSV + DAT formats)
- **ArduPilot**: BlackSquare and compatible (BIN logs)
- **Litchi**: Mission planner CSV exports

### Sensors

- **GPS**: u-blox, NMEA compatible
- **IMU**: MPU9250, accelerometer/gyroscope/magnetometer/barometer
- **ADC**: ADS1256, 4-channel voltage measurement
- **Camera**: Video and image sequence
- **Inclinometer**: Kernel-100, IMX-5

## Project Status

!!! success "Production Ready"
    PILS v1.0 is stable and production-ready with comprehensive test coverage.

| Metric | Value |
|--------|-------|
| Tests Passing | 283/284 (99.6%) |
| Code Coverage | 63% |
| Python Version | 3.10+ |
| License | MIT |
