# GPS Schema

Complete GPS data schema reference.

## DataFrame Schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | `Int64` | μs | Unix microseconds |
| `lat` | `Float64` | ° | Latitude (WGS84) |
| `lon` | `Float64` | ° | Longitude (WGS84) |
| `altitude` | `Float64` | m | Altitude above ellipsoid |
| `height` | `Float64` | m | Height above geoid (MSL) |
| `speed` | `Float64` | m/s | Ground speed |
| `heading` | `Float64` | ° | True heading (0-360) |
| `accuracy` | `Float64` | m | Horizontal accuracy estimate |
| `vertical_accuracy` | `Float64` | m | Vertical accuracy estimate |
| `fix_type` | `Int64` | - | Fix type code |
| `satellite_count` | `Int64` | - | Visible satellites |
| `pdop` | `Float64` | - | Position DOP |
| `hdop` | `Float64` | - | Horizontal DOP |
| `vdop` | `Float64` | - | Vertical DOP |

---

## Fix Type Codes

| Code | Name | Description |
|------|------|-------------|
| 0 | No Fix | No GPS fix |
| 1 | GPS Fix | Standard single-point |
| 2 | DGPS | Differential GPS |
| 3 | PPS | Precise Positioning Service |
| 4 | RTK Fixed | Real-time kinematic fixed |
| 5 | RTK Float | Real-time kinematic float |
| 6 | Estimated | Dead reckoning |

---

## PPK Position Schema

Post-processed kinematic positions (RTKLIB format).

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `datetime` | `Datetime` | - | UTC timestamp |
| `lat` | `Float64` | ° | Latitude |
| `lon` | `Float64` | ° | Longitude |
| `height` | `Float64` | m | Ellipsoidal height |
| `Q` | `Int64` | - | Quality flag (1=fix, 2=float, 5=single) |
| `ns` | `Int64` | - | Number of satellites |
| `sdn` | `Float64` | m | North std dev |
| `sde` | `Float64` | m | East std dev |
| `sdu` | `Float64` | m | Up std dev |
| `sdne` | `Float64` | m | N-E covariance |
| `sdeu` | `Float64` | m | E-U covariance |
| `sdun` | `Float64` | m | U-N covariance |
| `age` | `Float64` | s | Age of differential |
| `ratio` | `Float64` | - | Ambiguity ratio |

### Quality Flags

| Q | Name | Accuracy |
|---|------|----------|
| 1 | Fixed | cm-level |
| 2 | Float | dm-level |
| 5 | Single | m-level |

---

## NMEA Sentence Types

| Sentence | Description | Key Fields |
|----------|-------------|------------|
| `$GPGGA` | Fix data | lat, lon, alt, fix, sats |
| `$GPRMC` | Recommended minimum | lat, lon, speed, heading |
| `$GPGSA` | DOP and satellites | pdop, hdop, vdop |
| `$GPGSV` | Satellites in view | PRN, elevation, azimuth, SNR |
| `$GPVTG` | Velocity | speed, heading |

---

## File Formats

### CSV Format

```csv
timestamp,lat,lon,altitude,speed,heading,fix_type,satellite_count
1700000000000000,40.7128,-74.0060,10.5,0.0,0.0,4,12
1700000001000000,40.7128,-74.0060,10.6,0.1,45.2,4,12
```

### Binary Format (u-blox)

| Offset | Size | Field | Type |
|--------|------|-------|------|
| 0 | 4 | iTOW | U4 (ms) |
| 4 | 4 | lon | I4 (1e-7 deg) |
| 8 | 4 | lat | I4 (1e-7 deg) |
| 12 | 4 | height | I4 (mm) |
| 16 | 4 | hMSL | I4 (mm) |
| 20 | 4 | hAcc | U4 (mm) |
| 24 | 4 | vAcc | U4 (mm) |

---

## Example

```python
import polars as pl
from pils.sensors.gps import GPS
from pathlib import Path

# Load GPS data
gps = GPS(file=Path("/data/gps.csv"))
df = gps.data

# Check schema
print(df.schema)
# {'timestamp': Int64, 'lat': Float64, 'lon': Float64, ...}

# Filter by fix type
fixed = df.filter(pl.col('fix_type') >= 4)

# Calculate statistics
stats = df.select([
    pl.col('lat').mean().alias('mean_lat'),
    pl.col('lon').mean().alias('mean_lon'),
    pl.col('satellite_count').mean().alias('mean_sats'),
])
```

---

## See Also

- [GPS Sensor API](../api/sensors/gps.md)
- [IMU Schema](imu.md)
- [PPK Schema](ppk.md)
