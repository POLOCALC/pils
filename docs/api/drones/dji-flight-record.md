# DJI flight record (`DJIFlightRecord_*.txt`)

The app on the remote controller (DJI Pilot 2 for the Matrice 400) writes a
flight record for every flight: `DJIFlightRecord_<date>_[<time>].txt`. PILS
decodes it itself, with no third-party parser:

| Module | Role |
|---|---|
| `pils.drones.dji.m400.txtlog` | container: header, details, record stream, decryption, keys |
| `pils.drones.dji.m400.records` | what each record type holds |
| `pils.drones.dji.m400.flightrecord` | the table of the flight (`M400FlightRecord`) |
| `pils.drones.sources.record_track` | the 10 Hz track tied to the RTK, for synchronization |

```python
from pils.drones import M400FlightRecord

r = M400FlightRecord("DJIFlightRecord_2026-11-28_[12-38-29].txt")
r.load_data()      # DJI Open API app key from $DJI_API_KEY (formats 13+)
r.data             # one row per frame
r.details          # the log's details: start time, duration, aircraft, app
```

Formats 6 to 14 are read; the Matrice 400 writes format 14.

!!! note "Where the layouts come from"
    The container, the decryption and the standard records (OSD, home,
    gimbal, sticks, batteries, messages) follow DJI's FlightRecordParsingLib
    (C++) and the MIT-licensed `dji-log-parser` (Rust) and `pydjirecord`
    (Python), which documented them. PILS's decoder was checked record by
    record against `pydjirecord` 1.3.0 on four Matrice 400 logs: the same
    records in the same order, byte-identical decryption, the same details,
    identical OSD values.

    Records **45** and **48** are documented nowhere public. Their layouts
    were found by comparing them with the OSD record of real Matrice 400
    flights (November 2026); see [Records 45 and 48](#records-45-and-48).

All integers and floats are little-endian.

## The container

### Prefix (bytes 0-99)

| Offset | Type | Field |
|---|---|---|
| 0 | u64 | detail offset |
| 8 | u16 | detail length |
| 10 | u8 | format version |
| 11-99 | | not used |

### Where the details and the records are

| Format | Details | Records |
|---|---|---|
| 6-11 | at the detail offset | from byte 100 to the detail offset |
| 12 | at byte 100 | from byte 536 to the end of the file |
| 13-14 | in the first auxiliary block, at byte 100 | after the auxiliary blocks (detail offset 0), else from the detail offset, to the end |

**Auxiliary blocks** (format 13+): `u8 kind, u16 size, size bytes`.

- Kind 0 (*info*), XOR-decoded as a record of type 0: `u8, u16 n, n bytes of
  details, u16 m, m bytes of signature`.
- Kind 1 (*version*): `u16 keychain version, u8 department` (the app family:
  1 SDK, 2 DJI GO, 3 DJI Fly, ..., 7 DJI Pilot; unknown values are taken as 3).
  Both go into the key request.

### Details

From format 6. Strings are NUL-terminated UTF-8 in fixed-size fields.

| Offset | Type | Field |
|---|---|---|
| 0 | 4 × char[20] | sub-street, street, city, area (**not kept**) |
| 80 | 3 × u8 | favourite, new, needs upload |
| 83 | i32 | record line count |
| 87 | i32 | checksum |
| 91 | i64 | **start time**, ms since 1970 (UTC) |
| 99 | f64, f64 | longitude, latitude (deg) |
| 115 | f32 | total distance (km) |
| 119 | i32 | **total time** (ms) |
| 123 | 3 × f32 | max height, max horizontal speed, max vertical speed |
| 135 | i32, i64 | photos, video time |
| 147 | 96 bytes | "moment" pictures: sizes and positions |
| 243 | i64, 16 bytes | analysis offset, user id |
| 267 | f32 | take-off altitude × 10 (barometric on the Matrice 400) |
| 271 | u8 | product type (181: Matrice 400) |
| 272 | i64 | activation time |
| 280 | char[32] | aircraft name |
| 312 | 4 × char[16] | aircraft, camera, remote controller, battery serial numbers |
| 376 | u8 | app platform (1 iOS, 2 Android) |
| 377 | 3 × u8 | app version |

### Record stream

Each record is:

```
u8 type | u16 n | n bytes of data | 0xFF
```

The stream ends at the first record not followed by `0xFF`, or one that is cut
short. A log that ends in the middle of a record therefore loses only that
record.

## Decryption

### XOR (format 7+)

Every record's data is XOR-encoded. Its first byte `b` gives the key and is
dropped:

```
seed  = (b + type) mod 256
input = (0x123456789ABCDEF0 × b mod 2^64) as 8 bytes, little-endian
key   = CRC-64(seed, input) as 8 bytes, little-endian
plain[i] = data[i + 1] XOR key[i mod 8]      (i = 0 .. n - 2)
```

So the decoded data is one byte shorter than the stored data.

CRC-64 is the Jones polynomial in reflected form (`0x95AC9329AC4BC9B5`), with
the seed as its initial value and no final XOR.

### AES (format 13+)

After the XOR, most records are also encrypted with AES-256-CBC. Each record
type belongs to a *feature point*, a group with its own key and IV:

| Feature point | Record types |
|---|---|
| 1 Base | 1 OSD, 2 home, 6, 13, 14, 15, 40, 58, 59, 63 (and 3, 4, 7, 8, 11, 25, 29, 33 in format 13) |
| 2 Vision | 17, 18 |
| 3 Waypoint | 31, 32, 34, 35, 36, 38, 39 |
| 4 Agriculture | 21, 41, 43, 44, **45**, 46, 47, **48** |
| 5 AirLink | 49 |
| 6 AfterSales | 12, 16, 19, 23, 26, 27, 93, 102, 103, 113 (and 22, 28, 51, 52 in format 13) |
| 7 DJIFlyCustom | 5, 9, 10, 20, 24, 30, 54 |
| 8 Plaintext | 50, 56, and any type not listed |
| 9 FlightHub | 53 |
| 10 Gimbal (14) | 3 |
| 11 RC | 62; and 4, 11, 29, 33 in format 14 |
| 12 Camera (14) | 25 |
| 13 Battery (14) | 7, 8, 22 |
| 14 FlySafe (14) | 28, 51, 52 |
| 15 Security | 55 |

The XOR-decoded data minus its last byte is the ciphertext, which must be a
whole number of 16-byte blocks. It is decrypted with the feature point's key
and current IV, and the PKCS#7 padding is removed when it is valid. The IV is
chained: the next record of the same feature point uses the last 16 bytes of
this record's ciphertext.

### Keys

The keys are in the log, encrypted by DJI:

- **Type 56** (key storage, XOR only): `u16 feature point, u16 n, n bytes`,
  the encrypted key of that feature point.
- **Type 50** (key-storage recover): starts the next set of keys. The records
  after it use the next set.

DJI's keychain API returns them in clear. The request is

```
POST https://dev.dji.com/openapi/v1/flight-records/keychains
Api-Key: <DJI Open API app key>

{"version": <keychain version>, "department": <department>,
 "keychainsArray": [[{"featurePoint": "FR_Standardization_Feature_<Name>_<n>",
                      "aesCiphertext": "<base64 of the type 56 bytes>"}, ...], ...]}
```

The answer's `data` has the same shape, with `aesKey` (32 bytes) and `aesIv`
(16 bytes) in base64. Only the log's encrypted key blocks and the app key
leave the machine, nothing of the flight. The answer is cached for 30 days in
`$XDG_CACHE_HOME/pils/dji-keychains`, under the SHA-256 of the request, so a
log can be read again offline in that time. Keep the decoded table, not the
cache. The app key comes from the `api_key` argument or `$DJI_API_KEY`, and
PILS never stores it.

## Records

The records PILS reads, after decryption. Angles are in 0.1° and speeds in
0.1 m/s unless noted; the table gives them in degrees, m/s, metres and
seconds. Codes such as flight mode and flight action are kept as DJI's
integers.

### 1: OSD (the aircraft's state, starts a frame)

| Offset | Type | Column |
|---|---|---|
| 0, 8 | f64 (rad) | `osd_longitude`, `osd_latitude` |
| 16 | i16 / 10 | `osd_height`: above the take-off point (m) |
| 18, 20, 22 | i16 / 10 | `osd_x_speed`, `osd_y_speed`, `osd_z_speed` (`osd_h_speed` = their horizontal norm) |
| 24, 26, 28 | i16 / 10 | `osd_pitch`, `osd_roll`, `osd_yaw` |
| 30 | bits | `osd_flight_mode` (0x7F), `osd_rc_out_of_control` (0x80) |
| 31 | u8 | `osd_app_command` |
| 32 | bits | `osd_ground_or_sky` (0x06), `osd_is_motor_on` (0x08), `osd_go_home_status` (0xE0) |
| 33 | bits | `osd_is_vision_used` (0x01), `osd_voltage_warning` (0x06), `osd_is_gps_valid` (0x80) |
| 34 | bits | `osd_is_compass_error` (0x01), `osd_gps_level` (0x3C) |
| 35 | bits | `osd_is_out_of_limit` (0x01), `osd_is_motor_blocked` (0x08), `osd_is_not_enough_force` (0x10), `osd_is_vibrating` (0x40) |
| 36 | u8 | `osd_gps_num` (satellites) |
| 37 | u8 | `osd_flight_action` |
| 38 | u8 | `osd_motor_start_failed_cause` |
| 39 | bits | `osd_non_gps_cause` (0x0F) |
| 40 | u8 | `osd_battery_percent` |
| 41 | u8 / 10 | `osd_vps_height` (downward sensor, m) |
| 42 | u16 / 10 | **`osd_fly_time`** (s): the aircraft's counter, 0.1 s steps |
| 48 | u8 | `osd_drone_type` (format 2+) |

### 2: home point

| Offset | Type | Column |
|---|---|---|
| 0, 8 | f64 (rad) | `home_longitude`, `home_latitude` (NaN before a fix: the app writes a sentinel of about 800000 rad) |
| 16 | f32 / 10 | `home_altitude`: the app's, barometric on the Matrice 400 (varied by 400 m between flights from the same spot) |
| 20 | bits | `home_is_home_record` (0x01) |
| 22 | u16 | `home_go_home_height` (m) |
| 30 | u16 | `home_flight_record_index` |
| 37 | f32 | `home_max_allowed_height` (format 8+) |

### 3: gimbal

| Offset | Type | Column |
|---|---|---|
| 0, 2, 4 | i16 / 10 | `gimbal_pitch`, `gimbal_roll`, `gimbal_yaw` |
| 6 | bits | `gimbal_mode` (0xC0) |
| 10 | bits | `gimbal_is_pitch_at_limit` (0x01), `_roll_` (0x02), `_yaw_` (0x04), `gimbal_is_stuck` (0x40) |

The Matrice 400 logs read so far have no gimbal record.

### 4 and 62: sticks

Type 4 is `4 × u16`: `rc_aileron`, `rc_elevator`, `rc_throttle`, `rc_rudder`,
raw (1024 is the centre). Type 62 (the sticks as the remote controller shows
them) has the same four values after 7 bytes; the Matrice 400 writes this one.

### 9 and 10: the app's messages

NUL-terminated UTF-8: `app_tip` (9), `app_warn` (10). Several in one frame are
joined with `; `.

### 22: smart batteries

The first byte is the sub-type; the second is the battery's index `i`.

| Sub-type | Layout from byte 2 | Columns |
|---|---|---|
| 2 (dynamic) | i32 mV, i32 mA, u32 full, u32 remaining (mAh), i16 temperature × 10 (°C), u8 cells, u8 percent | `battery<i>_voltage`, `_current` (negative when discharging), `_full_capacity`, `_remaining_capacity`, `_temperature`, `_percent` |
| 3 (cells) | u8 n, n × u16 mV | `battery<i>_cell_voltage_1` ... `_n` |
| 1 (static) | serial number, design | not kept |

### Records 45 and 48

**48** comes with every frame (10 Hz). It holds the aircraft's position with a
height of known reference, and the frame's UNIX second:

| Offset | Type | Column |
|---|---|---|
| 0, 8 | f64 (rad) | `osd_abs_longitude`, `osd_abs_latitude` |
| 24 | u32 | `osd_unix_time`: UNIX seconds, UTC |
| 50 | f32 | `home_abs_altitude`: height of the take-off point (m) |
| 58 | f32 | `osd_abs_altitude`: height of the aircraft (m) |

On the flights checked: the position equals the OSD's; `osd_abs_altitude` =
`home_abs_altitude` + `osd_height` to within 1.5 m; `osd_abs_altitude`
agrees with the RTK height to 0.1 m; `osd_unix_time` agrees with the frame's
time to the second. The take-off height was the same on all flights from the
same spot (5180.2 m), while the app's barometric `home_altitude` varied
between 4980 and 5378 m. Bytes 16-23, 28-49, 54-57 and 62-70 are not read
(62 holds a second height 1.5-3 m lower, and 66 a constant of 31.46 that looks
like a geoid separation; both unconfirmed).

**45** comes once a second. It is the RTK solution:

| Offset | Type | Column |
|---|---|---|
| 1 | u8 | `rtk_pos_flag`: 50 when fixed (the code of the `.DAT`) |
| 2-10 | 9 × u8 | not read (they look like satellite counts) |
| 11, 19 | f64 (deg) | `rtk_latitude`, `rtk_longitude` |
| 27 | f32 | `rtk_height` (m) |
| 31, 39 | f64 (deg) | `rtk_base_latitude`, `rtk_base_longitude`: the position the base broadcasts |
| 47 | f32 | `rtk_base_height` (m) |

The flag was 50 in every epoch of the flights checked, so its other values are
not confirmed. The RTK heights are taken as ellipsoidal, as those of the
`.DAT` (the base broadcasts an ellipsoidal height); to check against a payload
GPS. With a survey, AZEL's base correction absorbs a constant datum offset.

In the schema (`pils.schema`, table `dji_record`) these become `abs_lat_deg`,
`abs_lon_deg`, `abs_h_m`, `home_abs_h_m`, `unix_time_s`, `rtk_pos_flag_1`,
`rtk_lat_deg`, `rtk_lon_deg`, `rtk_h_ell_m` and `rtk_base_lat_deg`,
`rtk_base_lon_deg`, `rtk_base_h_ell_m`.

## The table

`M400FlightRecord.data` has one row per frame.

- **Frames.** A frame starts at each OSD record (type 1) and takes every
  record after it up to the next OSD record.
- **States hold.** The values of OSD, home, gimbal, sticks and batteries hold
  until the next record of the same type. Records before the first frame give
  its starting state.
- **Epochs do not.** The RTK solution (45), record 48 and the app's messages
  are only in the frame they came in; the other frames have nulls.
- **Order.** Rows stay in the order of the file.

### Time

```
timestamp = details.start_time + osd_fly_time
```

`datetime` is the same time as a UTC datetime. Other time sources are not
used:

- The app's *custom* record (type 5) has no usable time in these logs: zeros,
  or values from 2010 to 2099. Sorting by it used to scramble the rows.
- The file's name is in the remote controller's time zone. On the first
  flights that was UTC+1, so `12-38-29` in the name was a take-off at
  11:38:29 UTC (08:38 in Chile).

`osd_unix_time` (record 48) checks the result: on four flights,
`timestamp − osd_unix_time` stayed between −0.6 and +0.7 s, the truncation to
the second, with no drift.

## Synchronization: the record's track tied to its RTK

When a flight has no payload GPS, no `.DAT` and no LAGER telemetry, its
reference is the flight record's 10 Hz track, tied to the RTK
(`pils.drones.sources.record_track`):

1. **Track and RTK.** The track is the frames with a record-48 position
   (`abs_*`, the drone's GPS time); the RTK is the fixed epochs (flag 50).
   At least 10 are needed.
2. **Latency.** The RTK solution is logged late. The shift of the RTK epochs
   that best matches the track (smallest horizontal scatter, between −1 and
   +1 s, in 0.02 s steps) is its latency.
3. **Offset.** With that shift, the median of RTK − track (east, north, up,
   about the RTK's median position) is the offset. The track is moved by it
   into the RTK's frame: `rtk_track_lat_deg`, `rtk_track_lon_deg`,
   `rtk_track_h_ell_m`.
4. **Fix flag.** Each frame gets the flag of the last RTK epoch within 2 s
   (`rtk_pos_flag_1`), so AZEL keeps only the frames near a fixed solution.
   Because the columns are `rtk_*`, AZEL applies the base correction once a
   survey exists.

The whole record is attached on the same clock (`flight_record` in the
synchronized data). `pils package` reports the result in a "record track"
step:

```
record track   done   10 Hz track tied to the RTK: offset E +0.20, N -0.37, U +0.08 m
                      (scatter 0.51 m horizontal, 0.39 m vertical), RTK latency 0.22 s, 576 epochs
```

| Flight (2026-11-28) | Offset E, N, U (m) | Scatter horizontal / vertical (m) | RTK latency (s) |
|---|---|---|---|
| 11:38 | −0.24, −0.03, +0.03 | 0.50 / 0.63 | 0.34 |
| 12:02 | +0.20, −0.37, +0.08 | 0.51 / 0.39 | 0.22 |
| 12:18 | +0.20, −0.40, +0.05 | 0.41 / 0.34 | 0.12 |
| 12:32 | +0.21, −0.40, +0.04 | 0.41 / 0.37 | 0.28 |

Syncing on the 1 Hz RTK alone, interpolated to the 100 Hz grid, would miss
the motion between epochs by up to 2.5-3.8 m in the turns.

AZEL also takes the base position it uses for the correction from the record
(`rtk_base_*`, median), not from `TELESCOPES_CONFIG["dji_broadcast_geod"]`,
which is an earlier campaign's.
