# LAGER telemetry (both drones, and the gimbal)

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
- **Per drone:** the OSDK frames of the Matrice 600 are described in
  [Matrice 600](matrice-600.md#lager-telemetry-osdk), the `psdk_telemetry`
  stream of the Matrice 400 in [Matrice 400](matrice-400.md#lager-telemetry-psdk).
- **Safety:** the gimbal log is a pickle stream, read with an unpickler that
  builds plain data only; a file with anything else is refused.
