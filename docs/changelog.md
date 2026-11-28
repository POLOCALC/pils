# Changelog

All notable changes to PILS are documented here.

## [Unreleased]

### Changed
- The Matrice 400 flight record (`DJIFlightRecord_*.txt`) is decoded by PILS
  itself (`pils.drones.dji.m400.txtlog`, `.records`): `pydjirecord` and
  `protobuf` are no longer dependencies (`cryptography` is, for the AES). The
  output was checked record by record against `pydjirecord` 1.3.0 on four
  Matrice 400 logs (format 14). The table's columns: `osd_*` codes are DJI's
  integers, the batteries are `battery<i>_*`, the sticks come from the
  record 62 too; `osd_altitude` (height + the app's barometric home altitude)
  is gone.
- Its frames are timed with the log's start plus the fly time (the app's times
  in these logs were garbage, and the table was sorted by them).
- Flight record: records 57, 253 and 254 are decrypted with their key groups
  (AirLink, DJIFlyCustom), found on Matrice 400 logs. Left encrypted, they
  broke their groups' IV chain, so the app's record (5) and the link signal
  (49) came out as noise, in `pydjirecord` too.

### Added
- The RTK solution (record 45) and the positions of record 48 of the flight
  record. A Matrice 400 flight with only its flight record is synchronized to
  the record's 10 Hz track tied to that RTK (`record_track`: RTK latency and
  offset found, reported by `pils package`), and AZEL takes the base position
  it broadcast from the record.
- Flight record time: the frames are timed by the aircraft's own clock
  (record 48: seconds and nanoseconds), no longer by the app's (the log's
  start plus the fly time), which ran 0.24-0.49 s behind on the first
  Matrice 400 flights: their AZEL moves later by that much. Records 48 that
  come a frame early or late are placed by their own time.
  `details["aircraft_minus_app_s"]` gives the app's offset.
- Flight record, after `dji-log-parser` 0.6.0 and checked on Matrice 400
  logs: the RTK's standard deviations and satellite counts by receiver and
  constellation (45, `rtk_std_*`, `rtk_<receiver>_<constellation>`), the
  yaw, home distance and geoid height of record 48, and the flight
  controller's estimates (63, `fc_*`: flight time left, time and battery to
  return home and to land). `osd_unix_time` is now `osd_aircraft_time`.
- Flight record columns: the app's clock and distance flown (record 5,
  `app_time`, `app_distance`), the controller's GNSS position (14,
  `app_gps_*`), the flight controller's settings (19, `mc_*`), the radio link
  (49, `link_*`), the position from the home point (OSD bytes 74-81,
  `osd_north`, `osd_east`); firmware versions and component serial numbers
  (15, 40) in `details`.
- Docs: the drone pages are split by drone (Matrice 600, Matrice 400, LAGER
  and the gimbal for both); the `.DAT` format moved from the user guide to the
  Matrice 600 page. "Matrice 400 flight record" documents the decoding in
  full: container, XOR and AES, keys, every record type of a Matrice 400 log
  with its layout (decoded or not, with what was found), the frame table and
  its time, and the record track.
- `--telescope-source campaign` (and `telescope_source="campaign"`): the
  telescope position from the campaign's own survey, else the so3g site
  (never another campaign's survey). `$PILS_TELESCOPE_SOURCE` sets the default
  for `pils package`, `pils quicklook` and AZEL (e.g. for tripel's packaging);
  without it the default stays `survey`.
- LAGER telemetry decoders, ported from POLOCALC/lager (no import): the
  Matrice 600 OSDK frames (`pils.drones.M600Telemetry`), the Matrice 400
  `psdk_telemetry` stream (`M400Telemetry`) and the Gremsy T7 gimbal
  (`pils.gimbals.GremsyT7Telemetry`), as polars tables. `load_telemetry(folder)`
  picks them from the YAML in the data folder. The gimbal's pickle log is read
  with a restricted unpickler. `M400FlightRecord` decodes the `.txt` flight
  record of the Matrice 400's remote controller with `pydjirecord` (now a
  required dependency, with `protobuf>=6.31.1`), the app key from
  `$DJI_API_KEY`. Tests write the byte layouts independently and
  were checked against lager's own loaders.

- `Flight.add_drone_data` loads every source of the flight it finds: the
  .DAT/CSV and Litchi as before, and the LAGER telemetry, gimbal and the
  Matrice 400 flight record (`dji_api_key` or `$DJI_API_KEY`) in the drone or
  aux folder. They are in `raw_data.drone_data.sources` (`.telemetry`,
  `.gimbal`, `.flight_record`), saved by `Flight.save` as parquet
  (`drone/uav_telemetry.parquet`, `drone/gimbal/<message>.parquet`,
  `drone/flight_record.parquet`) and loaded back by `Flight.load`. A Matrice
  400 flight needs no `.DAT`. Not written to HDF5 (a warning says so).
- `Flight.sync` synchronizes them with the payload GPS as the reference: a
  Matrice 400's drone GPS is its telemetry (RTK or GNSS epochs); with a `.DAT`,
  the telemetry and the flight record get their own offsets
  (`telemetry_gps`, `flight_record_gps`) and tables, and the gimbal log
  (`gimbal_<message>`) follows the telemetry's clock. New
  `Synchronizer.add_position_source` and `add_attached_table`.

### Removed (breaking)
- Code nothing used (in pils, gose, tripel or STOUT): the `PILS` multi-flight
  class; `StoutLoader`'s filesystem fallback, `load_flight_data`,
  `load_specific_data`, `get_campaign_list` and `get_available_*` (with
  `pils.config.SENSOR_MAP`/`DRONE_MAP` and `integrations.stout.all_campaigns`);
  `drones.drone_init` and `find_first_drone_file`; the per-message DJI .dat
  decoder (`dat_messages.py`) and `DJIDat`'s ignored `source_format` and
  `polars_interpolation` arguments; `Flight.add_camera_data`'s ignored
  `get_sony_angles`; `Synchronizer.add_inclinometer`'s ignored
  `inclinometer_type` (callers drop the second argument); BlackSquare's
  `compute_datetime`, `get_leapseconds` (whose astropy 8 fix was listed here)
  and `generate_log_file`; the Kernel decoder's `decode_multi`;
  `PPKAnalysis.check_overlap` and `delete_version`, `store.delete_version`;
  the numdifftools interpolation of `utils.uncertainty` (and the numdifftools
  dependency); the re-exports kept for the former single-file modules
  (`pils.flight`, `pils.synchronizer`, `pils.analyze.ppk`, `ppk_tuning`,
  `quicklook`, `drones.dji.dat`) and the `_Flight__*` aliases; the broken
  `utils/` scripts and the `sensors/LM76.py` prototype at the top of the
  repository.

### Changed (breaking)
- No HDF5 in PILS: every product is parquet (`pils.store`). Removed
  `Flight.to_hdf5` / `Flight.from_hdf5` (`pils.flight.hdf5`), the PPK
  `ppk_solution.h5` (`PPKAnalysis.hdf5_path`, `from_hdf5`: use `load`), the
  lab analyses' `<analysis>_solution.h5` runs and the HDF5 writers of
  `pils.utils.hdf5` (now only reads the lab's `.h5py` inputs). Old
  `*_solution.h5` results are converted once with `pils migrate` (its
  read-only reader is `pils.legacy_hdf5`); `sync_data.h5` flight files are not
  converted any more.
- Drone code is organised by make and source: `pils.drones.dji.dat`
  (`DJIDat`, was `pils.drones.DJIDrone.DJIDrone`; the .DAT format is in
  `dji/dat_format.py`), `pils.drones.dji.litchi` (`Litchi`) and
  `pils.drones.blacksquare` (`BlackSquare`, was `BlackSquareDrone`).
  `DRONE_MAP` and `drone_init` call the DJI .DAT source `dji_dat` (was
  `dji`); `load_flight_data(drones=...)` too. No old names are kept.
  This is the first step towards one folder per DJI model (M400, M600).

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
  the grid (`n_random`); the best has the most epochs fixed by RTKLIB (Q=1,
  `selection="fix_rate"`, the default), or, with `selection="reference"`,
  the most fixes agreeing with the drone RTK track; and how each setting
  acted on the flight (`setting_effects`, `tuning_effects`, shown by the
  quick look). A PPK run that fails no longer leaves an empty revision
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
- `DJIDat` - DJI SRT and CSV telemetry
- `Litchi` - Litchi flight logs
- `BlackSquare` - BlackSquare platform data

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
