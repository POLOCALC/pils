# Sensors

Sensor data access modules.

## Overview

PILS supports five sensor types:

| Sensor | Module | Data Type | Description |
|--------|--------|-----------|-------------|
| [GPS](gps.md) | `pils.sensors.GPS` | Position | GNSS receiver |
| [IMU](imu.md) | `pils.sensors.IMU` | Motion | Inertial measurement |
| [Camera](camera.md) | `pils.sensors.Camera` | Video/Image | Visual data |
| [ADC](adc.md) | `pils.sensors.ADC` | Voltage | Analog signals |
| [Inclinometer](inclinometer.md) | `pils.sensors.Inclinometer` | Attitude | Tilt sensor |

## Import

```python
# Individual
from pils.sensors import GPS
from pils.sensors import IMU
from pils.sensors import Camera
from pils.sensors import ADC
from pils.sensors import Inclinometer

# All
from pils.sensors import GPS, IMU, Camera, ADC, Inclinometer
```

## Common Pattern

All sensors follow the same pattern:

```python
# Via Flight container (recommended): schema column names
flight.add_sensor_data(['gps', 'imu'])
gps_df = flight.raw_data.payload_data.gps

# Direct instantiation: the names of the file format
from pils.sensors import GPS
gps = GPS(path="/path/to/sensors/gps.bin")
gps.load_data()
gps_df = gps.data
```

## Sensor Registry

Sensors are registered in `pils/sensors/sensors.py`:

```python
sensor_config = {
    'gps': {'class': GPS, 'load_method': 'load_data'},
    'imu': {'class': IMU, 'load_method': 'load_all'},
    'camera': {'class': Camera, 'load_method': None},
    'adc': {'class': ADC, 'load_method': 'load_data'},
    'inclinometer': {'class': Inclinometer, 'load_method': 'load_data'},
}
```

## Data Access

All sensors provide data as Polars DataFrames, with the column names of the
[Data Schema](../../user-guide/data-schema.md):

```python
import polars as pl

# GPS (columns as in the schema: lat_deg, lon_deg, h_ell_m, h_msl_m, ...)
payload = flight.raw_data.payload_data
gps_df = payload.gps
good = gps_df.filter(pl.col('h_acc_m') < 2.0)

# IMU (multiple sub-sensors)
accel_df = payload.imu['accelerometer']  # acc_x_g, acc_y_g, acc_z_g
gyro_df = payload.imu['gyroscope']  # gyro_x_rad_s, ...

# ADC
adc_df = payload.adc  # amplitude_mv

# Inclinometer (IMX-5: ins, imu, inl2 tables)
ins_df = payload.inclinometer['ins']  # roll_deg, pitch_deg, yaw_deg, ...
```

## Detailed Documentation

- [GPS](gps.md) - Position and velocity
- [IMU](imu.md) - Accelerometer, gyroscope, magnetometer, barometer
- [Camera](camera.md) - Video and images
- [ADC](adc.md) - Voltage channels
- [Inclinometer](inclinometer.md) - Attitude (Kernel-100, IMX-5)
