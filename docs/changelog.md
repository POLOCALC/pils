# Changelog

All notable changes to PILS are documented here.

## [Unreleased]

### Changed (breaking)
- Column names follow one schema (`pils.schema`, version 1): `snake_case`,
  the unit as a suffix (`_deg`, `_m`, `_m_s`, `_rad_s`, `_g`, `_hpa`, `_c`,
  `_mv`, ...), heights with their reference (`h_ell_m`, `h_msl_m`,
  `h_rel_m`, `h_m`), `timestamp` as the time base. `Flight` renames the
  columns of every loaded table and converts units where the source used
  others (u-blox mm/cm to m); tables are `snake_case` too (inclinometer
  `ins`/`imu`/`inl2`, DJI records `imu_atti_0`, `gps_1`, `osd`, ...).
  Synchronized positions are `lat_deg`/`lon_deg`/`h_ell_m` (were
  `latitude`/`longitude`/`altitude`); AZEL tables `az_deg`, `el_deg`,
  `srange_m`, `e_m`, `n_m`, `u_m` (the telescope CSV keeps its header). See
  [Data Schema](user-guide/data-schema.md)
- DJI .dat: the drone table is the RTK solution only, with its own rows and
  timestamps from the tick fit; the main GNSS receiver is the record `gps`.
  The two receivers were merged and every column interpolated between their
  rows (e.g. RTK yaw across 359° → 0° gave 293°)
- Results are saved as write-once revision directories of Parquet tables
  with a `manifest.json` (`pils.store`) instead of HDF5: the flight in
  `proc/flight/{raw,sync/rev_*}`, AZEL in `proc/azel/<TELESCOPE>/rev_*`,
  power in `proc/power/<TELESCOPE>_<wafer slot>/rev_*`, EKF and PPK in
  `proc/<analysis>/rev_*`

### Added
- `pils package`: sync a flight and save its data products in `proc/`
  (raw and synchronized data, AZEL per telescope with the telescope AZEL
  files, optional power analysis, `package_manifest.json`)
- `Flight.save`/`Flight.load`, `pils.store` (revisions with the schema
  version, the code's git commit and the units of every column), and
  `pils migrate <proc>` to convert HDF5 results (the HDF5 files are kept)
- `output_dir` for `AZELAnalysis`
- DJI .dat: IMU attitude (three IMUs, with roll/pitch/yaw), motors, motor
  PWM, battery, controller, wind and the other GNSS receivers, in
  `flight.raw_data.drone_data.records`
- Litchi: gimbal roll and yaw in degrees (as the pitch), and
  `gimbal_heading_deg`, the gimbal heading from north
- `pils.schema`: names and units of the flight table columns
- Each sync revision records the raw data it was made from (`raw_saved_at`)
- Sync: DJI .dat records on the payload time base, with the drone GPS
  offset (`drone_imu_atti_0` by default, `Flight.sync(drone_records=...)`);
  `flight.sync_offsets`, saved with the revision
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
- Sync tables follow one set of rules: `timestamp` is the only time column
  (other clocks, dates and text are dropped), states/flags/counts hold their
  last value and keep their type instead of being averaged, missing values
  are null, and `reference_gps` is resampled like the other tables. The
  rates actually used are in `flight.sync_rates` and saved by `pils package`
  (the saved rates listed keys the synchronizer ignored)
- Raw datetime columns are all `Datetime("us", "UTC")`; the DJI `datetime`
  was a string
- HDF5 tables: one writer/reader (`pils.utils.hdf5.write_table`/`read_table`);
  nulls are kept (they became NaN or ""). Used to read results saved before
  `pils.store`
- DJI .dat: when all tick-fit residuals were equal, `get_tick_offset` gave
  NaN timestamps
- `Flight.sync` and the power analysis changed the caller's rate dict
- Payload IMU: never loaded (it looked for `barometer.bin` and the other
  old names, and `Flight` expected one table). It now reads the payload's
  `*_BAR.bin`, `*_ACC.bin`, `*_GYR.bin` and `*_MAG.bin`, drops the damaged
  tail of a recording that stopped uncleanly, and is a dict of tables
  (`flight.raw_data.payload_data.imu['accelerometer']`); timestamps are in
  seconds, with `timestamp_us` as logged
- Sync: the GPS and pitch time offsets were biased towards zero (a 2 s clock
  difference came out as 1.27 s; on flight_20251208_1927 the Litchi offset
  moves by 54 ms, the drone one by 26 ms). Offsets are now fitted by least
  squares on mean-removed signals, after a coarse cross-correlation

### Deprecated
- `Flight.to_hdf5`: use `Flight.save`. `Flight.from_hdf5` still reads HDF5
  flight files and renames their columns to the schema

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
