# IMU

The payload IMU writes four text files in `aux/sensors/`, one sample per
line, at about 150 Hz: the timestamp in microseconds since the epoch, then
the values.

| File | Table | Columns | Units |
|------|-------|---------|-------|
| `*_ACC.bin` | `accelerometer` | `acc_x`, `acc_y`, `acc_z` | g |
| `*_GYR.bin` | `gyroscope` | `gyro_x`, `gyro_y`, `gyro_z` | as logged (rad/s, from the turns of flight_20250428_1625) |
| `*_MAG.bin` | `magnetometer` | `mag_x`, `mag_y`, `mag_z` | raw counts |
| `*_BAR.bin` | `barometer` | `pressure`, `temperature` | hPa, deg C |

Every table also has `timestamp` (seconds), `timestamp_us` (as logged) and
`datetime` (UTC).

```python
flight.add_sensor_data("imu")
imu = flight.raw_data.payload_data.imu  # {"barometer": df, ...}
imu["accelerometer"]
```

`flight.sync()` adds them as `imu_barometer`, `imu_accelerometer`,
`imu_gyroscope` and `imu_magnetometer`, with the other payload sensors.

## Availability

- Campaigns up to 2024 have no IMU files.
- In the April-May 2025 campaign (202504) the four files cover the whole
  flight. A few end with zero bytes or a cut line, from an unclean stop; the
  loader drops them.
- In the November-December 2025 campaign (202511) `ACC` and `GYR` are empty
  and `BAR` and `MAG` stop about 2 s after the start, in every flight. The
  loader keeps those 2 s and logs a warning.

Lines that cannot be parsed and barometer samples with zero pressure (the
first one of each file) are dropped.

::: pils.sensors.IMU.IMU

::: pils.sensors.IMU.IMUSensor
