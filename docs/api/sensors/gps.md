# GPS

The payload u-blox receiver (ZED-F9P; ZED-F9R in 2023), recorded as UBX:
`*_GPS.bin` or `*_GPS.ubx` in `aux/sensors/`. The NAV messages are merged on
one time grid; positions in degrees and metres (`lat_deg`, `lon_deg`,
`h_ell_m`, `h_msl_m`), `timestamp` in UNIX seconds.

- The messages are read one by one; corrupted bytes (some April 2025 files)
  are skipped with a warning instead of stopping the reader.
- The GPS week comes from the log's `Sensor ZED-F9... started` line or,
  without a log, from the file name (`YYYYMMDD_hhmmss_GPS.bin`); the time of
  week rolling over on Sunday is handled.
- A folder without a GPS file raises `FileNotFoundError`; a file with one
  epoch or none gives an empty table, and the flight is synchronized on the
  drone GPS.
- The 2023 `*_GPS.pck` files hold no data and are not read.

The raw u-blox file is also the PPK rover (see [PPK](../analysis/ppk.md)).

::: pils.sensors.gps.GPS

::: pils.sensors.gps.read_ubx
