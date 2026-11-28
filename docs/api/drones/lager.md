# LAGER telemetry (Matrice 400 / 600 and the gimbal)

The telemetry that LAGER logs next to the flight: the drone's own state
(`uav_telemetry.bin`) and the gimbal's (`gimbal_telemetry.bin`), with the
YAML of the run in the same folder. The decoders are ported from
[POLOCALC/lager](https://github.com/POLOCALC/lager) (commit `825164a`); PILS
does not import lager.

```python
from pils.drones import load_telemetry

t = load_telemetry("/path/to/flight/drone")  # the YAML picks the decoders
t.uav_model          # "DJI M400"
t.uav.data           # polars table, one row per record
t.gimbal.data        # one row per MAVLink message
t.gimbal.records     # the same, one table per message (MOUNT_ORIENTATION, RAW_IMU, ...)
t.missing            # configured devices whose log is not in the folder
```

The YAML has the `uav` and `gimbal` blocks (at the top or under
`pointing_controller`); the `name` of each picks the decoder:

| `name` | Decoder | File |
|--------|---------|------|
| `DJI M600` | `pils.drones.M600Telemetry` | `uav_telemetry.bin` (OSDK frames) |
| `DJI M400` | `pils.drones.M400Telemetry` | `uav_telemetry.bin` (`psdk_telemetry` stream) |
| `Gremsy T7` | `pils.gimbals.GremsyT7Telemetry` | `gimbal_telemetry.bin` |

Each decoder can also be used alone with a file or a folder:
`M400Telemetry(path).load_data()`; `.data` is the table, `.stats` counts what
was left out (CRC failures, unknown topics, truncated frames).

- **Time:** `timestamp` is the logging computer's clock (UNIX seconds); the
  aircraft's own counter is `t_ticks` (M600, 2.5 ms ticks) or `t_aircraft_ms`
  (M400). Align to the GNSS time as for the other drone sources.
- **M400 rows** carry their `topic`; the topics arrive at different rates, so
  by default each column holds its last value (`load_data(fill=False)` leaves
  the other columns null). The M600's battery-only frames are not filled
  unless `fill=True`.
- **Safety:** the gimbal log is a pickle stream, read with an unpickler that
  builds plain data only; a file with anything else is refused.

# Matrice 400 flight record (`.txt`)

The app of the Matrice 400's remote controller saves `DJIFlightRecord_*.txt`
(about 10 Hz). PILS decodes it itself (`pils.drones.dji.m400`: `txtlog` for
the container and its decryption, `records` for what the records hold,
`flightrecord` for the table), log formats 6 to 14.
The records of a log of version 13 or later are encrypted: the keys come from
DJI's keychain API with a DJI Open API app key, read from the `api_key`
argument or the `DJI_API_KEY` environment variable (never stored by PILS).

```python
from pils.drones import M400FlightRecord

r = M400FlightRecord("DJIFlightRecord_2026-11-19_[10-15-00].txt")
r.load_data()            # uses $DJI_API_KEY
r.data                   # one row per frame: timestamp, datetime, osd_*, home_*, rc_*, battery<i>_*, rtk_*, ...
r.details                # aircraft and serial numbers, app version, start time, totals
```

- Only the keychain request leaves the machine (the log's encrypted key
  blocks and the app key); PILS caches the answer for 30 days in
  `$XDG_CACHE_HOME/pils/dji-keychains`. Keep the decoded table, not the cache.
- The frames are timed with the log's start (UTC, in its details) plus the
  aircraft's fly time: the app's own times are not usable, and the file's name
  is in the remote controller's time zone (not necessarily the site's).
- Two records no public parser reads are decoded: the RTK solution (type 45,
  1 Hz, `rtk_*`: position, fix flag and the base's broadcast position) and a
  position with a height of known reference (type 48, `osd_abs_*`). Without a
  payload GPS, a .DAT or LAGER telemetry, the RTK solution is the flight's
  time and position reference (`pils package` says "synchronized to the flight
  record's RTK"), and AZEL takes the broadcast base position from the record.
- The street address the app writes in the header is not kept.
- This is not the onboard high-rate `.DAT` log and not LAGER's telemetry.
