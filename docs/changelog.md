# Changelog

All notable changes to PILS are documented here.

## [Unreleased]

### Added
- MkDocs documentation with Material theme
- Comprehensive API reference
- Data format specifications
- Development guides

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
