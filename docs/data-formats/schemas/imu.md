# IMU Schema

Inertial Measurement Unit data schema reference.

## DataFrame Schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | `Int64` | μs | Unix microseconds |
| `acc_x` | `Float64` | g | X-axis acceleration |
| `acc_y` | `Float64` | g | Y-axis acceleration |
| `acc_z` | `Float64` | g | Z-axis acceleration |
| `gyro_x` | `Float64` | °/s | X-axis angular rate |
| `gyro_y` | `Float64` | °/s | Y-axis angular rate |
| `gyro_z` | `Float64` | °/s | Z-axis angular rate |
| `mag_x` | `Float64` | μT | X-axis magnetic field |
| `mag_y` | `Float64` | μT | Y-axis magnetic field |
| `mag_z` | `Float64` | μT | Z-axis magnetic field |
| `temperature` | `Float64` | °C | Sensor temperature |

---

## Coordinate Frame

```
      Z (Up)
       ^
       |
       |
       +-----> Y (Right)
      /
     /
    v
   X (Forward)
```

### Axis Conventions

| Axis | Positive Direction | Rotation |
|------|-------------------|----------|
| X | Forward | Roll (right wing down = +) |
| Y | Right | Pitch (nose up = +) |
| Z | Down | Yaw (clockwise = +) |

!!! note "NED Convention"
    PILS uses North-East-Down (NED) convention internally. Some sensors output different frames and are converted automatically.

---

## Derived Quantities

### Euler Angles

| Column | Type | Unit | Range | Description |
|--------|------|------|-------|-------------|
| `roll` | `Float64` | ° | ±180 | Rotation about X |
| `pitch` | `Float64` | ° | ±90 | Rotation about Y |
| `yaw` | `Float64` | ° | 0-360 | Rotation about Z |

### Quaternion

| Column | Type | Description |
|--------|------|-------------|
| `q_w` | `Float64` | Scalar component |
| `q_x` | `Float64` | X vector component |
| `q_y` | `Float64` | Y vector component |
| `q_z` | `Float64` | Z vector component |

---

## Inclinometer Extended Schema

KERNEL inclinometer provides additional fields:

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `type` | `Utf8` | - | Message type |
| `usw` | `Int64` | - | Unit status word |
| `acc_x` | `Float64` | g | X acceleration |
| `acc_y` | `Float64` | g | Y acceleration |
| `acc_z` | `Float64` | g | Z acceleration |
| `gyro_x` | `Float64` | °/s | X rotation rate |
| `gyro_y` | `Float64` | °/s | Y rotation rate |
| `gyro_z` | `Float64` | °/s | Z rotation rate |
| `roll` | `Float64` | ° | Computed roll |
| `pitch` | `Float64` | ° | Computed pitch |
| `temperature` | `Float64` | °C | Sensor temperature |

### Unit Status Word (USW)

| Bit | Name | Description |
|-----|------|-------------|
| 0 | Acc_X_Err | X accelerometer error |
| 1 | Acc_Y_Err | Y accelerometer error |
| 2 | Acc_Z_Err | Z accelerometer error |
| 3 | Gyro_X_Err | X gyro error |
| 4 | Gyro_Y_Err | Y gyro error |
| 5 | Gyro_Z_Err | Z gyro error |
| 6 | Temp_Err | Temperature error |
| 7 | Cal_Err | Calibration error |

---

## File Formats

### CSV Format

```csv
timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,temperature
1700000000000000,0.01,-0.02,1.01,0.5,-0.3,0.1,25.5
1700000000010000,0.02,-0.01,1.00,0.4,-0.2,0.2,25.5
```

### Binary Format (KERNEL)

| Offset | Size | Field | Type | Scale |
|--------|------|-------|------|-------|
| 0 | 2 | Header | - | 0xAA55 |
| 2 | 1 | Length | U1 | - |
| 3 | 1 | Type | U1 | - |
| 4 | 2 | Acc_X | I2 | 1/8192 g |
| 6 | 2 | Acc_Y | I2 | 1/8192 g |
| 8 | 2 | Acc_Z | I2 | 1/8192 g |
| 10 | 2 | Gyro_X | I2 | 1/100 °/s |
| 12 | 2 | Gyro_Y | I2 | 1/100 °/s |
| 14 | 2 | Gyro_Z | I2 | 1/100 °/s |
| 16 | 2 | Roll | I2 | 1/100 ° |
| 18 | 2 | Pitch | I2 | 1/100 ° |
| 20 | 2 | Temp | I2 | 1/100 °C |
| 22 | 2 | Checksum | U2 | - |

---

## Example

```python
import polars as pl
from pils.sensors.IMU import IMU
from pathlib import Path

# Load IMU data
imu = IMU(file=Path("/data/imu.csv"))
df = imu.data

# Check schema
print(df.schema)

# Calculate acceleration magnitude
df = df.with_columns([
    (pl.col('acc_x')**2 + pl.col('acc_y')**2 + pl.col('acc_z')**2)
    .sqrt()
    .alias('acc_mag')
])

# Detect high-g events
high_g = df.filter(pl.col('acc_mag') > 2.0)
print(f"High-g events: {high_g.height}")

# Angular rate magnitude
df = df.with_columns([
    (pl.col('gyro_x')**2 + pl.col('gyro_y')**2 + pl.col('gyro_z')**2)
    .sqrt()
    .alias('gyro_mag')
])
```

---

## See Also

- [IMU Sensor API](../api/sensors/imu.md)
- [Inclinometer API](../api/sensors/inclinometer.md)
- [GPS Schema](gps.md)
