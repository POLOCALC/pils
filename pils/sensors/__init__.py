"""Payload sensors: readers for every payload file format of 2022-2025.

The payload changed between campaigns. The same readers load all of them
(``flight.add_sensor_data(...)``): they find the files by name in
``aux/sensors/`` and detect the format from the data.

Notes
-----
Files of each campaign:

| Campaign | GPS | ADC | Inclinometer | Other | Log |
|----------|-----|-----|--------------|-------|-----|
| 202204 | none | `*_ADC_<n>.pck` | Kernel, `*_INC_<n>.pck` | | `aux/*_file.log` |
| 202302 | `*_GPS.pck` (empty) | `*_ADC.pck` | `*_INC.pck` (empty) | | `sensors/*_log.log` |
| 202404 | `*_GPS.bin` | `*_ADC.bin`, 12-byte records | Kernel, `*_INC.bin` | | `aux/*_file.log` |
| 202412 | `*_GPS.bin` | `*_ADC.bin`, 20-byte records | Kernel, `*_INC.bin` | | `aux/*_file.log` |
| 202504 | `*_GPS.bin` | `*_ADC.bin` (ASCII) | Kernel, `*_INC.bin`, `*_INC_<hhmmss>.bin` | IMU `ACC`, `GYR`, `BAR`, `MAG` | `aux/*_file.log`, `*_config.yml` |
| 202511 | `*_GPS.bin`, `*_GPS.ubx` | `*_ADC.bin` (ASCII) | IMX-5, `*_INC_ins/imu/inl2.csv` | IMU, LM76 `*_TMP.csv` | `aux/*_file.log`, `*_config.yml` |

- The formats of each sensor are described in its module:
  :mod:`pils.sensors.gps`, :mod:`pils.sensors.adc`,
  :mod:`pils.sensors.inclinometer`, and :mod:`pils.sensors.legacy` for the
  2022 and 2023 pickle files.
- Log time stamps (:func:`pils.utils.tools.parse_log_datetime`):
  ``2025/12/08 19:25:43.590``, ``2024/04/29 18:59:31`` and, in 2023,
  ``02/21/2023 19:07:09``.
- Files cut by a power loss (an incomplete last record or line, trailing
  zeros) are read up to the last complete record.
"""

from pils.sensors.adc import ADC
from pils.sensors.camera import Camera
from pils.sensors.emlid import Emlid
from pils.sensors.gps import GPS
from pils.sensors.IMU import IMU
from pils.sensors.inclinometer import Inclinometer
from pils.sensors.LM76 import LM76
from pils.sensors.sensors import sensor_config

__all__ = [
    "Camera",
    "GPS",
    "IMU",
    "ADC",
    "Inclinometer",
    "LM76",
    "Emlid",
    "sensor_config",
]
