# Directory Structure

The general data for POLOCALC campaigns is organized following the scheme

```
campaigns
└── YYYYMM
    ├── metadata
    │   └── YYYYMM_coordinates.csv   # telescopes and base (needed for AZEL)
    ├── base                         # campaign base recordings (optional)
    └── YYYYMMDD
        ├── flight_YYYYMMDD_hhmm
        ├── base                     # the day's base recording (PPK)
        ├── calibration
        └── flightplans
```

Without `metadata/*_coordinates.csv` the AZEL step is skipped (in 2022-2025
only 202511 has one).

where each flight is is organized as follows:

```
flight_YYYYMMDD_hhmm/
├── aux
│   ├── YYYYMMDD_hhmmss_config.yml
│   ├── YYYYMMDD_hhmmss_file.log
│   ├── camera
│   │   ├── YYYYMMDD_hhmmss_video.mp4
│   │   └── YYYYMMDD_hhmmss_video.xml
│   └── sensors
│       ├── YYYYMMDD_hhmmss_ACC.bin
│       ├── YYYYMMDD_hhmmss_ADC.bin
│       ├── YYYYMMDD_hhmmss_BAR.bin
│       ├── YYYYMMDD_hhmmss_GPS.bin
│       ├── YYYYMMDD_hhmmss_GYR.bin
│       ├── YYYYMMDD_hhmmss_INC_imu.csv
│       ├── YYYYMMDD_hhmmss_INC_inl2.csv
│       ├── YYYYMMDD_hhmmss_INC_ins.csv
│       ├── YYYYMMDD_hhmmss_INC_log.log
│       ├── YYYYMMDD_hhmmss_MAG.bin
│       ├── YYYYMMDD_hhmmss_TMP.csv
│       └── YYYYMMDD_hhmmss_TMP.log
├── drone
│   ├── YYYYMMDD_hhmmss_drone.csv
│   ├── YYYYMMDD_hhmmss_drone.dat
│   └── YYYYMMDD_hhmmss_litchi.csv
└── proc
    ├── flight/                 # raw/ and sync/rev_*/
    ├── azel/, power/, ekf/, ppk/, quicklook/
    └── package_manifest.json


```

The tree above is the current payload (November 2025). Earlier payloads
wrote other files; see [Raw files by campaign](#raw-files-by-campaign).

This structure is designed to work with both available loaders in PILS, [PathLoader](../api/loaders/path-loader.md) and [StoutLoader](../api/loaders/stout-loader.md). The specific structure for the the ```proc\``` folder is given in the following section. The whole ```proc/``` folder of a flight is produced by ```pils package <flight>``` (see [Flight Packaging](../api/analysis/package.md)).


## Raw files by campaign

The payload changed between campaigns. Every format below is read by the
same loaders (`flight.add_sensor_data(...)`), which find the files by name
and detect the format from the data.

| Campaign | GPS | ADC | Inclinometer | Other | Log |
|----------|-----|-----|--------------|-------|-----|
| 202204 | none | `*_ADC_<n>.pck` | Kernel, `*_INC_<n>.pck` | | `aux/*_file.log` |
| 202302 | `*_GPS.pck` (empty) | `*_ADC.pck` | `*_INC.pck` (empty) | | `sensors/*_log.log` |
| 202404 | `*_GPS.bin` | `*_ADC.bin`, 12-byte records | Kernel, `*_INC.bin` | | `aux/*_file.log` |
| 202412 | `*_GPS.bin` | `*_ADC.bin`, 20-byte records | Kernel, `*_INC.bin` | | `aux/*_file.log` |
| 202504 | `*_GPS.bin` | `*_ADC.bin` (ASCII) | Kernel, `*_INC.bin`, `*_INC_<hhmmss>.bin` | IMU `ACC`, `GYR`, `BAR`, `MAG` | `aux/*_file.log`, `*_config.yml` |
| 202511 | `*_GPS.bin`, `*_GPS.ubx` | `*_ADC.bin` (ASCII) | IMX-5, `*_INC_ins/imu/inl2.csv` | IMU, LM76 `*_TMP.csv` | `aux/*_file.log`, `*_config.yml` |

- `*.pck` (2022, 2023): pickled records appended by the payload computer,
  read by `pils.sensors.legacy` (plain values only). `<n>` numbers the files
  of one recording. The 2022 payload computer started with a clock days
  behind and jumped to the right time during the recording; the jump is
  undone. The 2023 ADC times were stored as float32 (128 s steps); the
  sample times are rebuilt from the sample count. The 2023 GPS and
  inclinometer files hold no data.
- ADC binary records: time (s), reading time (µs), amplitude (mV) in 20
  bytes; in April 2024 time (s) and amplitude (V) in 12 bytes. ASCII files:
  time (µs) and counts per line. Amplitudes are in mV after loading
  (`amplitude_mv`).
- GPS: u-blox UBX, read message by message; corrupted bytes are skipped. The
  GPS week comes from the log (`Sensor ZED-F9P started`, 2023 `ZED-F9R`), or
  from the file name without a log.
- Log time stamps: `2025/12/08 19:25:43.590`, `2024/04/29 18:59:31` and, in
  2023, `02/21/2023 19:07:09`.
- Files cut by a power loss (an incomplete last record or line, trailing
  zeros) are read up to the last complete record.
- Drone: DJI `*_drone.dat` (and the DatCon `*_drone.csv` export) every year;
  `*_litchi.csv` where Litchi flew the mission. The 2022 Litchi files are
  logs of April 2025 and several February 2023 flights share one file: they
  do not match the drone track and are left out of the sync.

## Output Structure

Every product in ```proc/``` is stored the same way (`pils.store`): one
directory per revision, written once, with Parquet tables and a
`manifest.json`. Column names, units and the storage format are described in
[Data Schema and Storage](data-schema.md).

```
rev_YYYYMMDD_hhmmss/
├── manifest.json        # schema version, code (pils version, git commit,
│                        # uncommitted changes), tables (rows, dtypes, units),
│                        # the run's metadata
├── <table>.parquet      # one file per table; a "/" in a name is a folder
└── ...                  # other files of the run (plots, CSV)
```

The manifest is written last: a revision with a manifest is complete and is
never changed. A new run adds a new revision; older ones are kept.

### Flight: raw and synchronized data

`Flight.save(directory)` writes the loaded data and, if the flight is
synchronized, a new synchronized revision; `Flight.load(directory)` reads them
back (the latest revision by default).

```
proc/flight/
├── raw/                                # replaced at each save
│   ├── manifest.json                   # flight info, loaders, code version
│   ├── drone/
│   │   ├── drone.parquet               # DJI RTK solution (or BlackSquare)
│   │   ├── litchi.parquet              # Litchi log (if present)
│   │   └── records/                    # DJI .dat records
│   │       ├── gps.parquet             # main GNSS receiver
│   │       ├── imu_atti_0.parquet
│   │       └── ...
│   └── payload/
│       ├── gps.parquet, adc.parquet, lm76.parquet
│       ├── inclinometer/               # ins, imu, inl2
│       └── imu/                        # barometer, accelerometer, ...
└── sync/
    └── rev_YYYYMMDD_hhmmss/            # one per synchronization
        ├── manifest.json               # reference, time offsets (with
        │                               #   method and correlation), rates,
        │                               #   raw_saved_at, the run
        ├── drone.parquet, litchi.parquet, inclinometer.parquet
        ├── adc.parquet, lm76.parquet, imu_<sensor>.parquet
        ├── drone_imu_atti_0.parquet
        └── reference_gps.parquet       # only with the payload GPS as reference
```

The synchronized revision's manifest (`metadata`):

| Key | Content |
|-----|---------|
| `reference` | `"payload_gps"`, or `"drone_gps"` when there is no payload GPS covering the flight |
| `time_offsets` | seconds added to each source's time |
| `time_offset_details` | the same with `method` and `correlation` |
| `rates` | Hz of each synchronized table |
| `raw_saved_at` | the raw data it was made from |

See [Time Synchronization](synchronization.md#without-a-payload-gps).

- Every table has `timestamp` in UNIX seconds (UTC). In the raw data the
  datetime columns are `Datetime("us", "UTC")`; the synchronized tables have
  `timestamp` as their only time column (see
  [Synchronization](synchronization.md#resampling)).
- `raw/` is replaced at each save; `raw_saved_at` in a synchronized revision's
  manifest tells which raw data it was made from.
- A synchronized revision of a packaged flight is about 45 MB.

### Analyses

Each analysis has its folder, with one revision directory per run, possibly
grouped by telescope or target:

```
proc/
├── azel/<TELESCOPE>/rev_YYYYMMDD_hhmmss/
│   ├── manifest.json
│   ├── azel_data.parquet               # timestamp, az_deg, el_deg, srange_m, e/n/u_m
│   └── rev_..._azel_<telescope>_<YYYYMMDD_hhmm>.csv
├── power/<TELESCOPE>_<wafer slot>/rev_*/   # power_data.parquet, plots
├── ekf/rev_*/                          # ekf_data.parquet
├── ppk/rev_*/                          # position, statistics + RTKLIB files
└── quicklook/rev_*/                    # report.md, plots/, its tables
```

`pils quicklook` writes the report there (see
[Quick look](../api/analysis/quicklook.md)).

Results saved in HDF5 by earlier versions (`sync_data.h5`,
`*_solution.h5`) are converted with `pils migrate <flight>/proc`; the HDF5
files are kept.

---

## See Also

- [PathLoader API](../api/loaders/path-loader.md)
- [StoutLoader API](../api/loaders/stout-loader.md)
- [Synchronizer API](../api/core/synchronizer.md)
