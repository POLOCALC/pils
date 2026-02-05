# Drone Data Schema

Drone telemetry data schema reference for all supported platforms.

## DJI Drone Schema

### SRT Subtitle Format

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `frame_number` | `Int64` | - | Video frame index |
| `timestamp` | `Utf8` | - | Formatted timestamp |
| `datetime` | `Datetime` | - | Parsed datetime |
| `lat` | `Float64` | ° | Latitude |
| `lon` | `Float64` | ° | Longitude |
| `rel_alt` | `Float64` | m | Relative altitude (takeoff = 0) |
| `abs_alt` | `Float64` | m | Absolute altitude (MSL) |
| `distance` | `Float64` | m | Distance from home |
| `h_speed` | `Float64` | m/s | Horizontal speed |
| `v_speed` | `Float64` | m/s | Vertical speed |
| `iso` | `Int64` | - | Camera ISO |
| `shutter` | `Utf8` | - | Shutter speed string |
| `fnum` | `Float64` | - | F-number (aperture) |
| `ev` | `Float64` | EV | Exposure compensation |
| `ct` | `Int64` | K | Color temperature |
| `focal_len` | `Float64` | mm | Focal length |
| `dzoom_ratio` | `Float64` | - | Digital zoom ratio |

### CSV Telemetry Format

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `time` | `Float64` | s | Elapsed time |
| `latitude` | `Float64` | ° | Latitude |
| `longitude` | `Float64` | ° | Longitude |
| `altitude` | `Float64` | m | Altitude |
| `height` | `Float64` | m | Height AGL |
| `speed` | `Float64` | m/s | Ground speed |
| `pitch` | `Float64` | ° | Aircraft pitch |
| `roll` | `Float64` | ° | Aircraft roll |
| `yaw` | `Float64` | ° | Aircraft yaw |
| `gimbal_pitch` | `Float64` | ° | Gimbal pitch |
| `gimbal_roll` | `Float64` | ° | Gimbal roll |
| `gimbal_yaw` | `Float64` | ° | Gimbal yaw |
| `battery` | `Int64` | % | Battery level |

---

## Litchi Schema

Litchi flight log format.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `time` | `Float64` | s | Elapsed time |
| `datetime` | `Utf8` | - | ISO datetime |
| `latitude` | `Float64` | ° | Latitude |
| `longitude` | `Float64` | ° | Longitude |
| `altitude` | `Float64` | m | Altitude (MSL) |
| `height` | `Float64` | m | Height (takeoff) |
| `speed` | `Float64` | m/s | Ground speed |
| `heading` | `Float64` | ° | Aircraft heading |
| `gimbal_heading` | `Float64` | ° | Gimbal heading |
| `gimbal_pitch` | `Float64` | ° | Gimbal pitch |
| `flightmode` | `Int64` | - | Flight mode code |
| `message` | `Utf8` | - | Status message |
| `datetime_utc` | `Datetime` | - | Parsed UTC datetime |

### Litchi Flight Modes

| Code | Mode | Description |
|------|------|-------------|
| 0 | Manual | Manual control |
| 1 | P-GPS | Position GPS mode |
| 2 | Follow | Follow mode |
| 3 | Waypoint | Waypoint mission |
| 6 | Sport | Sport mode |
| 9 | Tripod | Tripod mode |

### Gimbal Modes

| Code | Mode |
|------|------|
| 0 | Free |
| 1 | FPV |
| 2 | Follow |

---

## BlackSquare Schema

BlackSquare drone platform format.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | `Int64` | μs | Unix microseconds |
| `lat` | `Float64` | ° | Latitude |
| `lon` | `Float64` | ° | Longitude |
| `altitude` | `Float64` | m | Altitude |
| `vn` | `Float64` | m/s | North velocity |
| `ve` | `Float64` | m/s | East velocity |
| `vd` | `Float64` | m/s | Down velocity |
| `roll` | `Float64` | ° | Roll angle |
| `pitch` | `Float64` | ° | Pitch angle |
| `yaw` | `Float64` | ° | Yaw angle |
| `motor_rpm` | `List[Int64]` | RPM | Motor RPM array |
| `battery_v` | `Float64` | V | Battery voltage |
| `battery_pct` | `Float64` | % | Battery percentage |
| `flight_mode` | `Int64` | - | Flight mode code |

### BlackSquare Flight Modes

| Code | Mode |
|------|------|
| 0 | Disarmed |
| 1 | Manual |
| 2 | Stabilize |
| 3 | AltHold |
| 4 | Loiter |
| 5 | Auto |
| 6 | RTL |
| 7 | Land |

---

## Common Conversions

### Speed Units

| From | To | Factor |
|------|-----|--------|
| m/s | km/h | × 3.6 |
| m/s | mph | × 2.237 |
| m/s | knots | × 1.944 |

### Altitude Reference

| Type | Description |
|------|-------------|
| MSL | Mean Sea Level (absolute) |
| AGL | Above Ground Level |
| HAE | Height Above Ellipsoid |
| Relative | From takeoff point |

---

## Example

```python
import polars as pl
from pils.drones import DJIDrone
from pathlib import Path

# Load DJI data
dji = DJIDrone(file=Path("/data/DJI.SRT"))
df = dji.data

# Check schema
print(df.schema)

# Convert speeds
df = df.with_columns([
    (pl.col('h_speed') * 3.6).alias('h_speed_kmh'),
    (pl.col('v_speed') * 3.6).alias('v_speed_kmh'),
])

# Filter by altitude
high_alt = df.filter(pl.col('rel_alt') > 100)
print(f"High altitude samples: {high_alt.height}")

# Camera settings analysis
iso_stats = df.groupby('iso').agg(
    pl.count().alias('count')
).sort('count', descending=True)
print(iso_stats)
```

---

## See Also

- [DJI Drone API](../api/drones/dji.md)
- [Litchi API](../api/drones/litchi.md)
- [BlackSquare API](../api/drones/blacksquare.md)
- [GPS Schema](gps.md)
