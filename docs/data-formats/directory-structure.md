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
    └── analysis_name
        ├── analysis_result.h5
        └── analysis_auxiliary
            ├── rev_YYYYMMDD_hhmmss
                └── rev_YYYYMMDD_hhmmss_file.csv    

```


This structure is designed to work with both available loaders in PILS, [PathLoader](../api/loaders/path-loader.md) and [StoutLoader](../api/loaders/stout-loader.md).

## PathLoader Structure

Standard directory layout for local flight data.

```
flight_data/


### Key Paths

| Path | Description |
|------|-------------|
| `aux/sensors/` | Sensor CSV/binary files |
| `aux/camera/` | Camera event logs |
| `aux/*.log` | System logs |
| `drone/` | Drone telemetry files |
| `media/` | Video/photo files |

---

## StoutLoader Structure

Remote storage (POLOCALC STOUT) layout.

```
stout:/
├── campaigns/
│   └── campaign-2023/
│       └── flights/
│           ├── flight-001/
│           │   ├── raw/
│           │   │   ├── sensors/
│           │   │   └── drone/
│           │   └── processed/
│           │       ├── ppk/
│           │       └── sync/
│           └── flight-002/
└── metadata/
    └── flight_index.json
```

---

## Sensor File Naming

### Convention

```
{sensor_type}_{descriptor}[_{version}].{ext}
```

### Examples

| Filename | Sensor | Description |
|----------|--------|-------------|
| `gps_data.csv` | GPS | Main GPS data |
| `gps_raw.bin` | GPS | Raw binary |
| `imu_data.csv` | IMU | IMU readings |
| `adc_channels.csv` | ADC | ADC voltages |
| `inclinometer.bin` | Inclinometer | KERNEL binary |
| `camera_events.csv` | Camera | Trigger events |

---

## Log File Structure

### System Log

```
aux/system.log
```

Contains system events with timestamps:

```
2023/11/01 12:00:00.000 [INFO] Recording started
2023/11/01 12:00:01.500 [INFO] GPS fix acquired (4/12)
2023/11/01 12:30:00.000 [INFO] Recording stopped
```

### Sensor Log

```
aux/*_file.log
```

Sensor-specific logging:

```
2023/11/01 12:00:00.000 [GPS] Init complete
2023/11/01 12:00:00.100 [GPS] Fix type: 4
```

---

## PPK Data Structure

```
ppk/
├── base/
│   ├── base.obs         # Base RINEX observation
│   └── base.nav         # Navigation file
├── rover/
│   └── rover.obs        # Rover RINEX observation
├── solutions/
│   ├── v1/
│   │   ├── solution.pos # Position output
│   │   └── config.conf  # RTKLIB config
│   └── v2/
│       └── ...
└── reports/
    └── ppk_report.html
```

---

## Output Structure

### HDF5 Output

```
output/
├── flight-001.h5
├── flight-002.h5
└── batch_results/
    └── combined.h5
```

### CSV Export

```
export/
├── flight-001/
│   ├── gps_export.csv
│   ├── imu_export.csv
│   └── synchronized.csv
└── flight-002/
    └── ...
```

---

## Loading Examples

### PathLoader

```python
from pils.loader import PathLoader
from pathlib import Path

loader = PathLoader()

# Load single flight
flight = loader.load_single_flight(
    data_path=Path("/data/flight_data/flight-001")
)

# Auto-detect sensors
flight.add_sensor_data(['gps', 'imu', 'adc'])
```

### StoutLoader

```python
from pils.loader import StoutLoader

loader = StoutLoader()

# Load by flight ID
flight = loader.load_single_flight(flight_id="flight-001")
```

### Custom Paths

```python
from pils.flight import Flight
from pils.sensors.gps import GPS
from pathlib import Path

# Create empty flight
flight = Flight(flight_info={'flight_id': 'custom'})

# Add sensor with custom path
gps = GPS(file=Path("/custom/path/gps.csv"))
flight.raw_data['gps'] = gps
```

---

## File Discovery

### Automatic Detection

PILS auto-detects files using keyword matching:

```python
from pils.utils.tools import get_path_from_keyword
from pathlib import Path

# Find GPS files
gps_path = get_path_from_keyword(
    dirpath=Path("/data/flight"),
    keyword="gps"
)
```

### Supported Keywords

| Sensor | Keywords |
|--------|----------|
| GPS | `gps`, `gnss`, `position` |
| IMU | `imu`, `inertial`, `motion` |
| ADC | `adc`, `analog`, `voltage` |
| Camera | `camera`, `trigger`, `event` |
| Inclinometer | `inclinometer`, `kernel`, `tilt` |

---

## Best Practices

!!! tip "Organization Tips"

    1. **Consistent naming**: Use lowercase with underscores
    2. **Descriptive folders**: Group by type (sensors, drone, media)
    3. **Date prefixes**: For multiple flights: `20231101_flight-001`
    4. **Version tracking**: Keep processing versions in separate folders
    5. **Metadata**: Include a `README.txt` with flight details

### Recommended Layout

```
project/
├── flights/
│   ├── 20231101_flight-001/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── README.txt
│   └── 20231102_flight-002/
├── base_stations/
│   └── BASE_01/
├── exports/
└── reports/
```

---

## See Also

- [PathLoader API](../api/loaders/path-loader.md)
- [StoutLoader API](../api/loaders/stout-loader.md)
- [HDF5 Format](files/hdf5.md)
