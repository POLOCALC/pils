# Changelog

All notable changes to PILS are documented here.

## [Unreleased]

### Added
- `pils package`: sync a flight and save its data products in `proc/`
  (`sync_data.h5`, AZEL per telescope with the telescope AZEL files,
  optional power analysis, `package_manifest.json`)
- `output_dir` for `AZELAnalysis`
- DJI .dat: IMU attitude (three IMUs, with roll/pitch/yaw), motors, motor
  PWM, battery, controller, wind and the other GNSS receivers, in
  `flight.raw_data.drone_data.records` and the flight HDF5
- Litchi: `gimbalRoll` and `gimbalYaw` in degrees (as `gimbalPitch`), and
  `gimbalYawAbs`, the gimbal heading from north
- Sync: DJI .dat records on the payload time base, with the drone GPS
  offset (`drone_IMU_ATTI(0)` by default, `Flight.sync(drone_records=...)`);
  `flight.sync_offsets`, saved by `pils package`
- MkDocs documentation with Material theme
- Comprehensive API reference
- Data format specifications
- Development guides

### Fixed
- DJI .dat: about one GPS/RTK record in five was dropped (records with a
  `0x55` byte in the encrypted payload); records are now found by length and
  CRC, and decoded with numpy
- DJI .dat RTK layout after `vel_d` (yaw, satellites, hdop, pitch, flags,
  state were read 2 bytes early) and missing scale factors (RTK secondary
  position and velocities, GPS DOPs and accuracies)
- `Flight.to_hdf5` failed on text columns with missing values, on date and
  time columns and on the inclinometer (a dict of tables); these now round-trip,
  and the columns are gzip-compressed
- `PowerAnalysis(output_dir=...)` wrote its AZEL results into the flight folder
- Sync: angles (yaw, roll, pitch, heading) are interpolated across the
  ±180° wrap instead of through 0°
- Sync: the GPS and pitch time offsets were biased towards zero (a 2 s clock
  difference came out as 1.27 s; on flight_20251208_1927 the Litchi offset
  moves by 54 ms, the drone one by 26 ms). Offsets are now fitted by least
  squares on mean-removed signals, after a coarse cross-correlation

---

## [1.0.0] - 2024-01-XX

### Added

#### Core
- `Flight` class for hierarchical flight data storage
- HDF5 export/import with compression
- Versioned synchronized data storage
- Dictionary and attribute access patterns

#### Loaders
- `PathLoader` for local filesystem data
- `StoutLoader` for POLOCALC STOUT remote storage
- Automatic sensor detection
- Recursive file search

#### Sensors
- `GPS` - GPS/GNSS data with NMEA and binary support
- `IMU` - Inertial measurement unit data
- `ADC` - Analog-to-digital converter readings
- `Camera` - Camera trigger events
- `Inclinometer` - KERNEL inclinometer binary decoder

#### Drones
- `DJIDrone` - DJI SRT and CSV telemetry
- `Litchi` - Litchi flight logs
- `BlackSquareDrone` - BlackSquare platform data

#### Synchronization
- `CorrelationSynchronizer` - Cross-correlation time alignment
- Multi-sensor temporal synchronization
- Automatic timestamp offset detection

#### Analysis
- `PPKAnalyzer` - Post-processed kinematic analysis
- RTKLIB integration
- Multi-version position storage
- Quality statistics computation

### Changed
- Migrated from Pandas to Polars for all DataFrames
- Unified sensor interface across all types
- Standardized timestamp format (Unix microseconds)

### Fixed
- HDF5 string column encoding
- Timezone handling in datetime parsing
- Memory efficiency for large files

---

## [0.9.0] - 2023-XX-XX

### Added
- Initial beta release
- Basic flight loading functionality
- GPS and IMU sensor support
- DJI drone data parsing

---

## Version Format

PILS follows [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH

MAJOR - Breaking API changes
MINOR - New features (backward compatible)
PATCH - Bug fixes (backward compatible)
```

---

## Migration Guides

### 0.9.x → 1.0.0

#### DataFrame Migration (Pandas → Polars)

```python
# Old (Pandas)
import pandas as pd
df = flight['gps'].data
filtered = df[df['lat'] > 0]

# New (Polars)
import polars as pl
df = flight['gps'].data
filtered = df.filter(pl.col('lat') > 0)
```

#### Common Operations

| Operation | Pandas | Polars |
|-----------|--------|--------|
| Filter | `df[df['col'] > 0]` | `df.filter(pl.col('col') > 0)` |
| Select | `df[['col1', 'col2']]` | `df.select(['col1', 'col2'])` |
| Rename | `df.rename(columns={...})` | `df.rename({...})` |
| Sort | `df.sort_values('col')` | `df.sort('col')` |
| Group | `df.groupby('col').mean()` | `df.groupby('col').mean()` |

---

## See Also

- [Getting Started](getting-started/index.md) - Installation and quickstart
- [User Guide](user-guide/index.md) - Usage documentation
- [API Reference](api/index.md) - Complete API docs
