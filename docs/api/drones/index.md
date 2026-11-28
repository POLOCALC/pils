# Drones

The drone data of a flight, by drone. Each drone has its own sources; the
payload (GPS, inclinometer, IMU, ADC...) is the same on both.

| Drone | Sources | Page |
|---|---|---|
| **DJI Matrice 600** | the onboard `.DAT` (or DatCon's CSV), the Litchi log, LAGER's OSDK telemetry | [Matrice 600](matrice-600.md) |
| **DJI Matrice 400** | the app's flight record (`DJIFlightRecord_*.txt`), LAGER's `psdk_telemetry` | [Matrice 400](matrice-400.md), [its flight record](matrice-400-flight-record.md) |
| Both | LAGER's YAML and the Gremsy T7 gimbal log | [LAGER telemetry](lager.md) |
| BlackSquare (ArduPilot) | `.BIN` / `.log` | [BlackSquare](blacksquare.md) |

## Loading

`Flight.add_drone_data()` loads every source of the flight it finds in the
`drone` and `aux` folders (and their direct subfolders):

```python
from pils.flight import Flight

flight = Flight(flight_info)
flight.add_drone_data()               # DJI_API_KEY for a Matrice 400 flight record

flight.raw_data.drone_data.drone      # the .DAT's RTK (M600), or BlackSquare / ArduPilot
flight.raw_data.drone_data.records    # the .DAT's other records (M600)
flight.raw_data.drone_data.litchi     # the Litchi log (M600)
flight.raw_data.drone_data.telemetry  # LAGER's telemetry (M600 or M400)
flight.raw_data.drone_data.gimbal     # LAGER's gimbal log
flight.raw_data.drone_data.flight_record  # the app's flight record (M400)
```

Each loader can also be used alone:

```python
from pils.drones import DJIDat, Litchi, M600Telemetry, M400Telemetry, M400FlightRecord

dat = DJIDat("flight_drone.dat"); dat.load_data()
record = M400FlightRecord("DJIFlightRecord_2026-11-28_[12-38-29].txt"); record.load_data()
```

## The time reference

`Flight.sync` puts every table on one time base. The reference is the
payload GPS; without one, or when it covers less than half of the drone GPS's
time, the drone's GPS:

| Drone | Drone GPS, in order |
|---|---|
| Matrice 600 | the `.DAT`'s RTK solution, else its main GNSS receiver |
| Matrice 400 | LAGER's telemetry (RTK or GNSS receiver), else the flight record's 10 Hz track tied to its RTK |

Every other source of the drone's position (Litchi, the telemetry next to a
`.DAT`, the flight record next to a payload GPS) gets its own time offset,
found by matching its track to the reference. See
[Synchronization](../../user-guide/synchronization.md).
