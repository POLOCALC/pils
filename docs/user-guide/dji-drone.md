# DJI Drone Data

PILS reads the data of two DJI drones. They write different files, and PILS
decodes each itself:

| | Matrice 600 | Matrice 400 |
|---|---|---|
| Drone's own log | onboard `.DAT` (`*_drone.dat`), or DatCon's CSV | the app's flight record (`DJIFlightRecord_*.txt`), encrypted: needs `DJI_API_KEY` |
| Drone position | `.DAT` RTK (5 Hz) and GNSS | flight record: 10 Hz track, RTK (1 Hz) |
| Gimbal attitude | Litchi log (`*_litchi.csv`) | LAGER's gimbal log (Gremsy T7) |
| Payload computer's log | LAGER OSDK telemetry | LAGER `psdk_telemetry` |
| Time | `.DAT` ticks fitted to GPS time | log start (UTC) + the aircraft's fly time |
| Details | [Matrice 600](../api/drones/matrice-600.md) | [Matrice 400](../api/drones/matrice-400.md), [flight record](../api/drones/matrice-400-flight-record.md) |

LAGER's YAML and the gimbal log, common to both, are in
[LAGER telemetry](../api/drones/lager.md).

## Matrice 600

The `.DAT` of the flight controller is decrypted and split into records (RTK,
GNSS, three IMUs, motors, battery, sticks, ...), checked against DatCon's CSV
of the same flight. The RTK solution is the drone table; the other records
are in `flight.raw_data.drone_data.records`. The record layouts are in
[Matrice 600](../api/drones/matrice-600.md#the-dat-flight-controller-log).

## Matrice 400

The flight record is decrypted with keys from DJI's keychain API (set
`DJI_API_KEY`; the answer is cached for 30 days). PILS reads its frames
(state, position with a known height, RTK solution, batteries, sticks, the
app's messages and clock, the controller's GNSS) and lists every record type
of the log, those decoded and those not, in
[Matrice 400 flight record](../api/drones/matrice-400-flight-record.md).

Without a payload GPS or LAGER telemetry, the flight is synchronized on the
record's 10 Hz track tied to its RTK; `pils package` reports the offset and
the RTK's latency ("record track").

!!! warning "The record's file name is not the take-off time in UTC"
    It is in the remote controller's time zone (UTC+1 on the first flights).
    The take-off in UTC is in the record's details:
    `pils.drones.dji.m400.txtlog.FlightLog.read(path).details.start_time`
    (no key needed).
