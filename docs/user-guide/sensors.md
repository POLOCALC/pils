# Working with Sensors

Access and analyze sensor data from GPS, IMU, ADC, Camera, and Inclinometer.

## Loading Sensors

```python
from pils.flight import Flight

flight = Flight(flight_info)

# Load specific sensors
flight.add_sensor_data(['gps', 'imu', 'adc'])

# Or load all available
flight.add_sensor_data()

# Access via dictionary-style
gps = flight['gps']
imu = flight['imu']
```

---

## GPS Sensor

GPS load exclusively UBX-NAV file at the moment, raw data are skipped. 

### Loading

```python
from pils.sensors import GPS

# Via flight container
gps = flight['gps']

# Or directly
gps = GPS(path="/path/to/sensors/gps.bin")
gps.load_data()
```

### Data Access

```python
# Main DataFrame
gps_df = gps.data

print(gps_df.columns)
# ['timestamp', 'latitude', 'longitude', 'altitude', 'fix_quality', 
#  'num_satellites', 'hdop', 'velocity_north', 'velocity_east', ...]
```

### Schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | `Int64` | ms | Milliseconds since epoch (UTC) |
| `latitude` | `Float64` | degrees | WGS84, -90 to 90 |
| `longitude` | `Float64` | degrees | WGS84, -180 to 180 |
| `altitude` | `Float64` | meters | MSL or HAE |
| `fix_quality` | `Int32` | - | 0=None, 1=GPS, 2=DGPS, 4=RTK Fix, 5=RTK Float |
| `num_satellites` | `Int32` | - | Satellites used, 0-32 |
| `hdop` | `Float64` | - | Horizontal DOP, lower is better |
| `velocity_north` | `Float64` | m/s | North velocity (ENU) |
| `velocity_east` | `Float64` | m/s | East velocity (ENU) |
| `velocity_down` | `Float64` | m/s | Down velocity (ENU) |
| `course` | `Float64` | degrees | Ground track, 0-360 |

### Analysis Examples

```python
import polars as pl

# Filter RTK fixed positions
rtk_fixed = gps_df.filter(pl.col('fix_quality') == 4)
print(f"RTK Fix rate: {rtk_fixed.height / gps_df.height:.1%}")

# Calculate flight duration
duration_ms = gps_df['timestamp'].max() - gps_df['timestamp'].min()
print(f"Duration: {duration_ms / 60000:.1f} minutes")

# Calculate total distance
gps_df = gps_df.with_columns([
    pl.col('latitude').diff().alias('lat_diff'),
    pl.col('longitude').diff().alias('lon_diff'),
])
# Note: Use proper geodetic distance for accuracy

# Statistics
stats = gps_df.select([
    pl.col('hdop').mean().alias('mean_hdop'),
    pl.col('num_satellites').mean().alias('mean_sats'),
    pl.col('altitude').min().alias('min_alt'),
    pl.col('altitude').max().alias('max_alt'),
])
print(stats)
```

---

## IMU Sensor

The IMU contains four sub-sensors: accelerometer, gyroscope, magnetometer, and barometer.

### Loading

```python
from pils.sensors import IMU

# Via flight container
imu = flight['imu']

# Or directly
imu = IMU(path="/path/to/sensors/imu/")
imu.load_data()
```

### Data Access

```python
# Sub-sensor DataFrames
accel_df = imu.accelerometer
gyro_df = imu.gyroscope
mag_df = imu.magnetometer
baro_df = imu.barometer
```

### Accelerometer Schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | `Int64` | ms | Milliseconds since epoch |
| `ax` | `Float64` | m/s² | X-axis (forward) |
| `ay` | `Float64` | m/s² | Y-axis (right) |
| `az` | `Float64` | m/s² | Z-axis (down) |
| `temperature` | `Float64` | °C | Sensor temperature (optional) |

**Coordinate Frame:** Body-fixed, right-handed (FRD: Forward-Right-Down)

### Gyroscope Schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | `Int64` | ms | Milliseconds since epoch |
| `gx` | `Float64` | rad/s | Roll rate |
| `gy` | `Float64` | rad/s | Pitch rate |
| `gz` | `Float64` | rad/s | Yaw rate |
| `temperature` | `Float64` | °C | Sensor temperature (optional) |

### Magnetometer Schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | `Int64` | ms | Milliseconds since epoch |
| `mx` | `Float64` | µT | X-axis magnetic field |
| `my` | `Float64` | µT | Y-axis magnetic field |
| `mz` | `Float64` | µT | Z-axis magnetic field |

### Barometer Schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | `Int64` | ms | Milliseconds since epoch |
| `pressure` | `Float64` | hPa | Atmospheric pressure |
| `temperature` | `Float64` | °C | Sensor temperature |
| `altitude` | `Float64` | meters | Derived from pressure (optional) |

### Analysis Examples

```python
import polars as pl
import numpy as np

# Accelerometer magnitude
accel_df = accel_df.with_columns([
    (pl.col('ax')**2 + pl.col('ay')**2 + pl.col('az')**2).sqrt().alias('accel_mag')
])

# Detect static periods (magnitude ≈ 9.81 m/s²)
static = accel_df.filter(
    (pl.col('accel_mag') > 9.5) & (pl.col('accel_mag') < 10.1)
)

# Gyroscope bias estimation (during static)
gyro_static = gyro_df.filter(
    pl.col('timestamp').is_in(static['timestamp'])
)
bias = gyro_static.select([
    pl.col('gx').mean().alias('bias_x'),
    pl.col('gy').mean().alias('bias_y'),
    pl.col('gz').mean().alias('bias_z'),
])
print(f"Gyro bias: {bias}")

# Barometric altitude
baro_df = baro_df.with_columns([
    (44330 * (1 - (pl.col('pressure') / 1013.25) ** 0.1903)).alias('baro_altitude')
])
```

---

## ADC Sensor

Analog-to-Digital Converter for voltage measurements.

### Loading

```python
from pils.sensors import ADC

# Via flight container
adc = flight['adc']

# Or directly
adc = ADC(path="/path/to/sensors/adc.dat")
adc.load_data()
```

### Data Access

```python
adc_df = adc.data
print(adc_df.columns)
# ['timestamp', 'datetime', 'channel_0', 'channel_1', 'channel_2', 'channel_3']
```

### Schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | `Int64` | ms | Milliseconds since epoch |
| `datetime` | `Datetime` | - | UTC datetime (optional) |
| `channel_0` | `Float64` | V | Voltage channel 0 |
| `channel_1` | `Float64` | V | Voltage channel 1 |
| `channel_2` | `Float64` | V | Voltage channel 2 |
| `channel_3` | `Float64` | V | Voltage channel 3 |

### Analysis Examples

```python
import polars as pl

# Statistics per channel
for i in range(4):
    col = f'channel_{i}'
    stats = adc_df.select([
        pl.col(col).mean().alias('mean'),
        pl.col(col).std().alias('std'),
        pl.col(col).min().alias('min'),
        pl.col(col).max().alias('max'),
    ])
    print(f"Channel {i}: {stats}")

# Convert to physical units (example: load cell)
SENSITIVITY = 0.002  # V/kg
adc_df = adc_df.with_columns([
    (pl.col('channel_0') / SENSITIVITY).alias('load_kg')
])
```

<!-- ---

## Camera Sensor

Video or image sequence with timestamps.

### Loading

```python
from pils.sensors import Camera

# Via flight container
camera = flight['camera']

# Or directly
camera = Camera(path="/path/to/camera/video.mp4")
```

### Properties

```python
# Check if video or images
print(f"Is video: {camera.is_video}")
print(f"Frame count: {camera.frame_count}")
print(f"Resolution: {camera.resolution}")  # (width, height)
print(f"FPS: {camera.fps}")
```

### Access Frames

```python
import numpy as np

# Get single frame (0-indexed)
frame: np.ndarray = camera.get_frame(100)
print(f"Frame shape: {frame.shape}")  # (height, width, 3) BGR

# Get frame timestamp
ts = camera.get_timestamp(100)
print(f"Timestamp: {ts} ms")

# Get time index DataFrame
time_df = camera.get_time_index()
print(time_df.columns)  # ['frame_index', 'timestamp']
```

### Analysis Examples

```python
import cv2
import numpy as np

# Extract frame at specific timestamp
target_ts = 1702034500000  # ms
time_df = camera.get_time_index()

# Find nearest frame
nearest = time_df.filter(
    pl.col('timestamp') >= target_ts
).head(1)
frame_idx = nearest['frame_index'][0]

frame = camera.get_frame(frame_idx)

# Process frame
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
``` -->

---

## Inclinometer Sensor

Attitude measurement from Kernel-100 or IMX-5.

### Loading

```python
from pils.sensors import Inclinometer

# Via flight container (auto-detects type)
inclino = flight['inclinometer']

# Or directly with type
inclino = Inclinometer(
    path="/path/to/inclino.bin",
    sensor_type='kernel'  # or 'imx5'
)
inclino.load_data()
```

### Schema (Kernel-100)



### Schema (IMX-5)

IMX-5 returns a dictionary 

```python
{
    "INS": pl.Dataframe,
    "INL2" pl.Dataframe
}
```

### Analysis Examples

```python
import polars as pl

inclino_df = inclino.data

# Attitude statistics
stats = inclino_df.select([
    pl.col('roll').mean().alias('roll_mean'),
    pl.col('roll').std().alias('roll_std'),
    pl.col('pitch').mean().alias('pitch_mean'),
    pl.col('pitch').std().alias('pitch_std'),
    pl.col('heading').std().alias('heading_std'),
])
print(stats)

# Detect turns (heading change)
inclino_df = inclino_df.with_columns([
    pl.col('heading').diff().alias('heading_rate')
])
turns = inclino_df.filter(pl.col('heading_rate').abs() > 10)
```

---

## Sensor Registry

Available sensors are defined in `pils/sensors/sensors.py`:

```python
sensor_config = {
    'gps': {'class': GPS, 'load_method': 'load_data'},
    'imu': {'class': IMU, 'load_method': 'load_data'},
    'camera': {'class': Camera, 'load_method': None},
    'adc': {'class': ADC, 'load_method': 'load_data'},
    'inclinometer': {'class': Inclinometer, 'load_method': 'load_data'},
}
```

To add a new sensor, see [Adding Sensors](../development/adding-sensors.md).

---

## Next Steps

- [Drone Platforms](drones.md) - Platform-specific data
- [Time Synchronization](synchronization.md) - Align sensor data
