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

In 2022-2025 only 202511 has `metadata/*_coordinates.csv`. The other
campaigns take the telescope positions from it (the telescopes do not move)
and have no surveyed DJI base (see
[AZEL](../api/analysis/azel.md#telescope-and-base-positions)).

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

::: pils.sensors
    options:
      show_root_heading: false
      members: false

Drone: DJI `*_drone.dat` (and the DatCon `*_drone.csv` export) every year,
and `*_litchi.csv` where Litchi flew the mission (see
[Litchi](../api/drones/litchi.md)).

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
