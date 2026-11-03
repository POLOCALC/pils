# AZEL Analysis

Azimuth, elevation and range of the drone from each telescope.

```bash
pils package flight_20250428_1625                           # survey positions
pils package flight_20250428_1625 --telescope-source so3g   # so3g sites
```

## Telescope and base positions

::: pils.sensors.emlid
    options:
      show_root_heading: false
      members: false

::: pils.sensors.emlid.reference_positions

::: pils.sensors.emlid.ReferencePositions

::: pils.sensors.emlid.coordinate_files

::: pils.sensors.emlid.so3g_position

## Analysis

::: pils.analyze.azel.AZELAnalysis

::: pils.analyze.azel.AZELVersion
