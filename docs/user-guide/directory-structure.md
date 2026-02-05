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
    └── analyzed_data


```

This structure is designed to work with both available loaders in PILS, [PathLoader](../api/loaders/path-loader.md) and [StoutLoader](../api/loaders/stout-loader.md). The specific structure for the the ```proc\``` folder is given in the following section. 


## Output Structure


### HDF5 RAw and Synchronyzed Output

The ```Flight```container should be considered _immutable_ at least for the ```raw_data``` and ```metadata```. Indeed, those are read from the multiple files and stored for conveninece in a single ```.h5``` file. This file should contain only a single set of _processed_ data. This is the synchronized data which should save all the data in a single dataframe following the recipe given in the [Synchronizer](../api/core/synchronizer.md) section. The resulting dataset should have associated metadata, such as _columnnames_ or _synchronizationmethod_ (or any useful, such as the timing reference).

```
flight.h5
├── metadata/
│   ├── flight_info/                    # Flight identification
│   │   ├── flight_id: "flight-001"
│   │   ├── date: "2023-11-01"
│   │   └── ...
│   └── flight_metadata/                # Processing metadata
│       ├── pils_version: "1.0.0"
│       ├── created_at: "2023-11-01T12:00:00"
│       └── ...
├── raw_data/
│   ├── drone/                          # Drone telemetry
│   │   └── [DataFrame columns]
│   ├── litchi/                         # Litchi data (if present)
│   │   └── [DataFrame columns]
│   └── sensors/
│       ├── gps/
│       │   └── [DataFrame columns]
│       ├── imu/
│       │   └── [DataFrame columns]
│       └── ...
└── synchronized/                        # Versioned sync data
    ├── rev_20231101_120000/
    │   ├── metadata
    │   │   ├── sync_method: "correlation"
    │   │   └── column_names: []
    │   └── [Merged DataFrame columns]
    └── rev_20231101_140000/
        └── ...
```


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

---

## See Also

- [PathLoader API](../api/loaders/path-loader.md)
- [StoutLoader API](../api/loaders/stout-loader.md)
- [Synchronizer API](../api/core/synchronizer.md)
