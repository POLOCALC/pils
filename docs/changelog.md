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
  `proc/<analysis>/rev_*`, lab results in `proc/<analysis>/<stem>/rev_*`
  (older HDF5 lab runs and `ppk_solution.h5` are still read)
- PPK tables: `lat_deg`, `lon_deg`, `h_ell_m` (or `h_msl_m`, from the `.pos`
  header), `quality`, `num_sats`, `sd_n_m`..., and UTC `datetime` and
  `timestamp` (the RTKLIB time was GPST, without a time zone); statistics
  with `timestamp`, `az_deg`, `el_deg`, `resid_*_m`, `snr_dbhz`. EKF output
  `timestamp`, `monotonic_ns`; EKF input accepts the schema names
- Payload sensors (ADC, LM76, IMU) take the payload computer's clock offset.
  With a Kernel-100 inclinometer (own clock: log start + counter) they were
  given the inclinometer's offset: 1.05 s wrong on flight_20250428_1625. The
  payload clock is now tied to the inclinometer through the payload IMU
  gyroscope, or not corrected without one (`offsets["payload_clock"]`)

### Added
- `pils quicklook FLIGHT...` (`pils.analyze.quicklook.QuickLook`): one
  Markdown report per flight, with plots, in `proc/quicklook/rev_*`: files,
  drone, payload sensors (rate, gaps, coverage), synchronisation offsets,
  rover and base GNSS data per band (satellites above 30/35/40 dB-Hz, SNR,
  loss of lock, tracking), AZEL, and with `--ppk` the tuned RTKLIB
  configuration; every problem found is listed first
- PPK: `run_analysis(template)` (tuning on by default) chooses the RTKLIB settings
  from the data (frequencies, systems, SNR and elevation masks, ambiguity
  resolution) and validates each candidate's fixes against the drone's RTK
  track, and says when the rover data limits fixing; the base is found in the
  day's or the campaign's `base/`; RTKLIB from `PILS_RTKLIB_BIN`
- `pils prune <proc> --keep N`: delete old revisions, keeping the newest N of
  each analysis and the synchronized revisions the kept ones were made from
- `pils migrate` also converts lab results (`proc/<analysis>/*_solution.h5`)
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

### Changed
- Docstrings follow PEP 257 and numpydoc everywhere in `pils/`, checked by
  ruff (`D` rules, numpy convention). File formats, packaging steps and
  settings are documented in the code (module docstrings, attribute
  docstrings for constants) and the pages include them; the docs render the
  Sphinx roles of the docstrings as links

### Added
- PPK tuning: 1000 random configurations over 18 RTKLIB settings besides
  the grid (`n_random`), every epoch scored against the drone RTK track
  (good and wrong fixes), the best is the one with the most good fixes and
  at most 0.5 % wrong; and how each setting acted on the flight
  (`setting_effects`, `tuning_effects`, shown by the quick look). On the 119
  November 2025 flights: median 9 % good fixes, against 5 % with the grid
  and 1.6 % with the template (17 % wrong fixes)
- AZEL telescope positions for campaigns without a survey: the newest
  survey of another campaign (the telescopes do not move), or, with
  `--telescope-source so3g` (`pils package`, `pils quicklook`), the so3g
  site used by sotodlib (2-3 m from the survey). The DJI base correction
  needs the campaign's own survey; without it the drone track is not
  corrected (in 202511 the correction is 1.3 m east, 9.2 m up). The source
  of each position is in the AZEL metadata (`telescope_position_source`,
  `base_position_source`, `position_notes`)

### Fixed
- Payload files of every campaign load: `pils package --dry-run` packs 274
  of the 275 flights of 2022-2025 (was 156; flight_20251208_1404 has no
  data):
  - 2022 and 2023 pickle files (`*_ADC*.pck`, `*_INC*.pck`,
    `pils.sensors.legacy`): only plain values are unpickled; the 2022
    payload clock jump (saved clock days behind, set by the network during
    the recording) is undone; the 2023 ADC times (float32, 128 s steps)
    are rebuilt from the sample count
  - April 2024 ADC records (time, volts: 12 bytes) besides the 20-byte
    records; records cut at the end and preallocated zeros are dropped
  - payload log times without milliseconds (2024) and in the 2023 format;
    the 2023 log in `sensors/`
  - payload GPS: `*_GPS.ubx` files; corrupted bytes that stopped the u-blox
    reader (April 2025); files with one epoch; a clear error when there is
    no GPS file; the GPS week from the file name without a log; the time of
    week rolling over on Sunday
  - IMX-5 CSV files cut by a power loss (an incomplete last line, NUL
    bytes); Kernel files with no messages give an error
- Sync: positions at latitude and longitude 0 (DJI RTK rows without a
  solution, payload GPS epochs without a fix) are dropped; they stopped the
  drone offset on 9 flights
- Sync: payload GPS epochs without a real fix (u-blox horizontal accuracy
  worse than 50 m, `MAX_GPS_H_ACC_M`; positions thousands of km off in April
  2024) are dropped; they stopped the drone offset on 3 April 2024 flights
- A clear error for an empty drone folder
- Sync: without a payload GPS, or with one covering less than half of the
  drone GPS time, the drone GPS is the reference; empty payload tables are
  left out instead of stopping the sync; `pils package` reports the drone,
  Litchi or inclinometer data left out of the sync and every offset with
  its method
- PPK: a cached `base.obs` was reused even when it did not cover the rover
  (flight_20251206_1833 had 2 minutes of overlap); rover `*_GPS.ubx` files
  were not found
- Kernel-100 inclinometer (April-May 2025 flights) did not load: the decoder
  was given the sensors folder; `*_INC_<time>.bin` files are found too
- LM76 without a file and AZEL without campaign coordinates are reported
  as skipped, with the reason
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
