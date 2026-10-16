# Directory Structure

The general data for POLOCALC campaigns is organized following the scheme

```
campaigns
└── YYYYMM
    └── YYYYMMDD
        ├── flight_YYYYMMDD_hhmm
        ├── base
        ├── calibration
        └── flightplans
```

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
    ├── sync_data.h5
    ├── package_manifest.json
    └── analyzed_data (azel/, power/, ppk/, ...)


```

This structure is designed to work with both available loaders in PILS, [PathLoader](../api/loaders/path-loader.md) and [StoutLoader](../api/loaders/stout-loader.md). The specific structure for the the ```proc\``` folder is given in the following section. The whole ```proc/``` folder of a flight is produced by ```pils package <flight>``` (see [Flight Packaging](../api/analysis/package.md)).


## Output Structure


### HDF5 RAw and Synchronyzed Output

The ```Flight```container should be considered _immutable_ at least for the ```raw_data``` and ```metadata```. Indeed, those are read from the multiple files and stored for conveninece in a single ```.h5``` file. This file should contain only a single set of _processed_ data. This is the synchronized data which should save all the data in a single dataframe following the recipe given in the [Synchronizer](../api/core/synchronizer.md) section. The resulting dataset should have associated metadata, such as _columnnames_ or _synchronizationmethod_ (or any useful, such as the timing reference).

```
sync_data.h5
├── metadata/                           # [attrs] flight_info_*, flight_metadata_*
├── raw_data/                           # [attrs] saved_at, pils_version
│   ├── drone_data/
│   │   ├── drone                       # DJI GPS/RTK table (or BlackSquare)
│   │   ├── litchi                      # Litchi log (if present)
│   │   └── dat_records/                # DJI .dat records
│   │       ├── IMU_ATTI(0)
│   │       └── ...
│   └── payload_data/
│       ├── gps
│       ├── adc
│       ├── lm76
│       ├── inclinometer/               # one table per message
│       │   ├── INS
│       │   ├── IMU
│       │   └── INL2
│       └── imu/                        # one table per sensor
│           ├── barometer
│           └── ...
└── sync_data/                          # one revision per synchronization
    └── rev_YYYYMMDD_hhmmss/            # [attrs] created_at, pils_version,
        │                               #   raw_data_saved_at, user_* (run,
        │                               #   rates, offsets, ...)
        ├── drone
        ├── litchi
        ├── inclinometer
        ├── adc, lm76, imu_<sensor>
        ├── drone_IMU_ATTI(0)
        └── reference_gps
```

Every table is a group (see `pils.utils.hdf5.write_table`):

```
<table>/                                # [attrs] columns, dtypes, units (JSON),
├── <column>                            #   n_rows, format
├── ...                                 # one gzip-compressed dataset per column
└── _nulls/<column>                     # null mask, for columns with nulls
```

- Numbers and booleans are stored as they are, text as UTF-8, dates and times
  as integers converted back to their type on load (`dtypes`).
- Nulls come back as nulls (the `_nulls` masks); a NaN stays a NaN.
- `units` gives the unit of each column whose unit is known (see
  `pils.utils.column_units`); e.g. `{"timestamp": "s", "RTK:lat_p": "deg"}`.
- Every table has `timestamp` in UNIX seconds (UTC). In the raw data the
  datetime columns are `Datetime("us", "UTC")`; the synchronized tables have
  `timestamp` as their only time column (see
  [Synchronization](synchronization.md#resampling)).

`raw_data` is replaced at each save, while synchronized revisions are added;
`raw_data_saved_at` on a revision tells which raw data it was made from. Each
revision of a packaged flight adds about 40 MB; older revisions are not
removed.

### HDF5 Analysis Output

Each product of the analysis should be stored in a ```.h5``` file. Each analysis will have its dedicated folder. Inside the folder, there will be a single ```.h5```file and a series of folders (one per revision) that include the auxilliary results of the data analysis (plots, CSV files etc.).

```
proc/
    └──analysis_name
        ├── analysis_result.h5
        └── rev_YYYYMMDD_hhmmss
            ├── rev_YYYYMMDD_hhmmss_file.png
            └── rev_YYYYMMDD_hhmmss_file.csv
```

The structure of the ```.h5``` file is the following:

```
analysis_result.h5                       
    ├── rev_YYYYMMDD_hhmmss/
    │   ├── metadata
    │   └── dataset
    └── rev_20231101_140000/
        ├── metadata
        └── Group
            ├── group_metadata
            ├── data_1
            |   ├── metadata
            |   └── dataset
            └── data_2
                ├── metadata
                └── dataset
```

**metadata** are assigned using ```.attrs``` and are mandatory for each dataset. They should at least include the column names. 

The analysis files (AZEL, EKF, PPK) store their tables in the same format.

---

## See Also

- [PathLoader API](../api/loaders/path-loader.md)
- [StoutLoader API](../api/loaders/stout-loader.md)
- [Synchronizer API](../api/core/synchronizer.md)
