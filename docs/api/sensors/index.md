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
# Via Flight container (recommended)
flight.add_sensor_data(['gps', 'imu'])
gps = flight['gps']
gps_df = gps.data

# Direct instantiation
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
    'imu': {'class': IMU, 'load_method': 'load_data'},
    'camera': {'class': Camera, 'load_method': None},
    'adc': {'class': ADC, 'load_method': 'load_data'},
    'inclinometer': {'class': Inclinometer, 'load_method': 'load_data'},
}
```

## Data Access

All sensors provide data as Polars DataFrames:

```python
import polars as pl

# GPS
gps_df = flight['gps'].data
rtk_fixes = gps_df.filter(pl.col('fix_quality') == 4)

# IMU (multiple sub-sensors)
accel_df = flight['imu'].accelerometer
gyro_df = flight['imu'].gyroscope

# ADC
adc_df = flight['adc'].data

# Inclinometer
inclino_df = flight['inclinometer'].data
```

## Detailed Documentation

- [GPS](gps.md) - Position and velocity
- [IMU](imu.md) - Accelerometer, gyroscope, magnetometer, barometer
- [Camera](camera.md) - Video and images
- [ADC](adc.md) - Voltage channels
- [Inclinometer](inclinometer.md) - Attitude (Kernel-100, IMX-5)
