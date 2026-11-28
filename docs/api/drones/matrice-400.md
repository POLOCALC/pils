# Matrice 400

The DJI Matrice 400 flights give two drone sources, next to the payload (PILS
reads no onboard `.DAT` of the Matrice 400).

| Source | File | Module | What it is |
|---|---|---|---|
| App flight record | `DJIFlightRecord_*.txt` | `pils.drones.dji.m400.flightrecord` (`M400FlightRecord`) | the remote controller's log (DJI Pilot 2), 10 Hz: state, position with known height, RTK (1 Hz), batteries, sticks, app messages; encrypted, keys from DJI's API |
| LAGER telemetry | `uav_telemetry.bin` + the run's YAML | `pils.drones.M400Telemetry` | the PSDK `psdk_telemetry` stream logged by the payload computer, see [LAGER telemetry](lager.md) |

In a flight folder the record stays under its own name in `drone/` (PILS finds
it as `DJIFlightRecord_*.txt` in `drone/`, `aux/` or a subfolder); LAGER's
files are in `drone/` or `aux/`.

The full decoding of the flight record (container, decryption, every record
type of the log and its layout, the table and its time) is in
[Matrice 400 flight record](matrice-400-flight-record.md).

## Which source is the reference

`Flight.sync` takes the time reference in this order:

1. the **payload GPS**;
2. else the **LAGER telemetry**, if it has the RTK or GNSS receiver's
   positions;
3. else the **flight record's 10 Hz track tied to its RTK**
   (`pils.drones.sources.record_track`): the record-48 positions of every
   frame, moved into the RTK's frame after the RTK's latency is found; see
   [the record track](matrice-400-flight-record.md#synchronization-the-records-track-tied-to-its-rtk).

When it is not the reference, the flight record is a position source of its
own (its 10 Hz track, with its own time offset to the reference), and the
LAGER gimbal log goes on the telemetry's clock.

For AZEL, the position the RTK base broadcast is taken from the flight
record (`rtk_base_*`) when it has one, instead of the configuration's.

## Time

- **Flight record:** the aircraft's own time, in record 48 (seconds and
  nanoseconds). The app's clock (the log's start time, its own `app_time`)
  runs behind it by 0.24-0.49 s, a different amount each flight
  (`details["aircraft_minus_app_s"]`); see
  [Time](matrice-400-flight-record.md#time). The file's name is in the remote
  controller's time zone (UTC+1 on the first flights), not the site's: tools
  that name the flight folder should read the start time from the details
  (`pils.drones.dji.m400.txtlog.FlightLog.read(path).details.start_time`,
  no key needed; the app's, so within half a second).
- **LAGER telemetry:** `timestamp` is the logging computer's clock;
  `t_aircraft_ms` the aircraft's.

## LAGER telemetry (PSDK)

`M400Telemetry` decodes the `psdk_telemetry` stream LAGER logs in
`uav_telemetry.bin`. Its rows carry their `topic`; the topics arrive at
different rates, so by default each column holds its last value
(`load_data(fill=False)` leaves the other columns null). The single battery
is `bat1_*`. `.stats` counts what was left out (CRC failures, unknown topics,
truncated frames). The YAML and the gimbal log are described in
[LAGER telemetry](lager.md).

## API

::: pils.drones.dji.m400.telemetry
