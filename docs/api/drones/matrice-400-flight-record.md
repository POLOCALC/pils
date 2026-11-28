# Matrice 400: flight record (`DJIFlightRecord_*.txt`)

The app on the Matrice 400's remote controller (DJI Pilot 2) writes a flight
record for every flight: `DJIFlightRecord_<date>_[<time>].txt`. PILS decodes
it itself, with no third-party parser:

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
r.details          # start time, duration, aircraft, app, firmware, serial numbers
```

Formats 6 to 14 are read; the Matrice 400 writes format 14. All integers and
floats are little-endian.

!!! note "Where the layouts come from"
    The container, the decryption and the standard records (OSD, home,
    gimbal, sticks, batteries, messages, custom, app GPS, flight controller,
    link, firmware, components) follow DJI's FlightRecordParsingLib (C++) and
    the MIT-licensed `dji-log-parser` (Rust) and `pydjirecord` (Python), which
    documented them. PILS's decoder was checked record by record against
    `pydjirecord` 1.3.0 on four Matrice 400 logs (November 2026): the same
    records in the same order, the same details, identical OSD values.

    Where PILS differs from them, the Matrice 400 logs showed why:

    - **Key groups of types 57, 253 and 254** (see [Decryption](#aes-format-13)).
      The reference parsers leave them encrypted, which breaks the IV chain of
      their group: the app's time (record 5) and the link signal (49) came out
      as noise.
    - **The app GPS (14)** is latitude, longitude (degrees) and altitude, in
      that order.

    Records **45** (RTK), **48** (position and time) and **63** (battery
    estimates) follow `dji-log-parser` **0.6.0** (November 2026), which added
    them along with the key groups above; every field PILS reads was checked
    on the Matrice 400 logs (see [Records 45, 48 and 63](#records-45-48-and-63)).
    Byte 58 of record 48 and OSD bytes 74-81 were found on those logs.

## Record types of a Matrice 400 log

Every record type in four Matrice 400 logs (format 14; 2358 s of flight),
with its rate, its key group and what PILS does with it. Sizes are after
decryption.

| Type | Name | Rate | Size (bytes) | Key group | In PILS |
|---|---|---|---|---|---|
| 1 | OSD | 10 Hz | 85 | Base | **read** (`osd_*`), starts a frame |
| 2 | home point | 0.7 Hz | 97 | Base | **read** (`home_*`) |
| 5 | custom (the app's) | 10 Hz | 18 | DJIFlyCustom | **read** (`app_time`, `app_distance`, ...) |
| 9 | app tip | events | 28-52 | DJIFlyCustom | **read** (`app_tip`) |
| 10 | app warning | events | 41-194 | DJIFlyCustom | **read** (`app_warn`) |
| 13 | recover | 5 per flight, first 50 s | 110 | Base | not read: repeats the details (aircraft, serial numbers, app) |
| 14 | app GPS | 0.5 Hz | 20 | Base | **read** (`app_gps_*`) |
| 15 | firmware | at start | 18 | Base | **read** (`details["firmware"]`) |
| 17 | vision | rare | 9 | Vision | not decoded (constant flags) |
| 19 | flight controller settings | 0.1 Hz | 33 | AfterSales | **read** (`mc_*`) |
| 22 | smart battery | 2 Hz | 29-46 | Battery | **read** in part (`battery<i>_*`) |
| 26 | ? | once | 2 | AfterSales | not decoded (`41 00`) |
| 40 | component serial | once | 24 | Base | **read** (`details["component_serials"]`) |
| 45 | RTK solution | 1 Hz | 83 | Agriculture | **read** (`rtk_*`: position, solution, sigmas, satellites, base) |
| 48 | position and time | 10 Hz | 71 | Agriculture | **read** (`osd_aircraft_time`, `osd_abs_*`, home distance, geoid): **times the frames** |
| 49 | radio link | 4 Hz | 1 | AirLink | **read** (`link_*`) |
| 50 | key-storage recover | | | Plaintext | container: starts the next set of keys |
| 55 | security | 1 Hz | 257 | Security | **cannot be decrypted**: DJI's API does not return this group's key |
| 56 | key storage | at start | 261 | Plaintext | container: the encrypted keys |
| 57 | link configuration | 6 Hz | 3, 35 | AirLink | not decoded (two constant blocks) |
| 62 | sticks (display) | 2 Hz | 19 | RC | **read** (`rc_*`) |
| 63 | flight controller estimates | 1 Hz | 10 | Base | **read** (`fc_*`: flight time left, return home and landing) |
| 253 | aircraft information | at start | 25, 32 | DJIFlyCustom | not read: a header and a time, then protobuf (total flight time, serial numbers; read by `dji-log-parser`) |
| 254 | key-value log | 35 Hz | 8-16 | DJIFlyCustom | not decoded (protobuf values), see below |

The Matrice 400 logs have no gimbal record (3) and no stick record (4); both
are read when present.

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
short. A log that ends in the middle of a record loses only that record.

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

So the decoded data is one byte shorter than the stored data. CRC-64 is the
Jones polynomial in reflected form (`0x95AC9329AC4BC9B5`), with the seed as
its initial value and no final XOR.

### AES (format 13+)

After the XOR, most records are also encrypted with AES-256-CBC. Each record
type belongs to a *key group* (DJI's "feature point"), with its own key and IV:

| Key group | Record types |
|---|---|
| 1 Base | 1 OSD, 2 home, 6, 13, 14, 15, 40, 58, 59, 63 (and 3, 4, 7, 8, 11, 25, 29, 33 in format 13) |
| 2 Vision | 17, 18 |
| 3 Waypoint | 31, 32, 34, 35, 36, 38, 39 |
| 4 Agriculture | 21, 41, 43, 44, **45**, 46, 47, **48** |
| 5 AirLink | 49, **57** |
| 6 AfterSales | 12, 16, 19, 23, 26, 27, 93, 102, 103, 113 (and 22, 28, 51, 52 in format 13) |
| 7 DJIFlyCustom | 5, 9, 10, 20, 24, 30, 54, **253**, **254** |
| 8 Plaintext | 50, 56, and any type not listed |
| 9 FlightHub | 53 |
| 10 Gimbal (14) | 3 |
| 11 RC | 62; and 4, 11, 29, 33 in format 14 |
| 12 Camera (14) | 25 |
| 13 Battery (14) | 7, 8, 22 |
| 14 FlySafe (14) | 28, 51, 52 |
| 15 Security | 55 |

The XOR-decoded data minus its last byte is the ciphertext, which must be a
whole number of 16-byte blocks. It is decrypted with the group's key and
current IV, and the PKCS#7 padding is removed when it is valid. The IV is
chained: the next record of the same group uses the last 16 bytes of this
record's ciphertext.

**Types 57, 253 and 254** were found on the Matrice 400 logs. For records of
two blocks or more, only the right key gives a valid padding (the last block
is decrypted with the block before it, whatever the IV): on the four logs,
every multi-block record of 254 and 253 had a valid padding with the
DJIFlyCustom key, every one of 57 with the AirLink key, and essentially never
with another. A record of a group left undecrypted breaks that group's IV
chain, so the first 16 bytes of the group's next record come out wrong.

### Keys

The keys are in the log, encrypted by DJI:

- **Type 56** (key storage, XOR only): `u16 key group, u16 n, n bytes`, the
  encrypted key of that group.
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
(16 bytes) in base64. For the Matrice 400 logs it returned groups 1, 2, 4, 5,
6, 7, 11 and 13: not 15 (Security, record 55). Only the log's encrypted key
blocks and the app key leave the machine, nothing of the flight. The answer
is cached for 30 days in `$XDG_CACHE_HOME/pils/dji-keychains`, under the
SHA-256 of the request, so a log can be read again offline in that time. Keep
the decoded table, not the cache. The app key comes from the `api_key`
argument or `$DJI_API_KEY`, and PILS never stores it.

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
| 74, 78 | f32 | **`osd_north`, `osd_east`**: from the home point (m); Matrice 400, see [below](#osd-bytes-74-81-position-from-home) |

Bytes 44-47, 49-73 and 82-84 of the Matrice 400's OSD are not decoded; see
[below](#present-not-decoded).

### 2: home point

| Offset | Type | Column |
|---|---|---|
| 0, 8 | f64 (rad) | `home_longitude`, `home_latitude` (NaN before a fix: the app writes a sentinel of about 800000 rad) |
| 16 | f32 / 10 | `home_altitude`: the app's, barometric on the Matrice 400 (varied by 400 m between flights from the same spot) |
| 20 | bits | `home_is_home_record` (0x01) |
| 22 | u16 | `home_go_home_height` (m) |
| 30 | u16 | `home_flight_record_index` |
| 37 | f32 | `home_max_allowed_height` (format 8+; 600 m on the Matrice 400) |

### 3: gimbal

| Offset | Type | Column |
|---|---|---|
| 0, 2, 4 | i16 / 10 | `gimbal_pitch`, `gimbal_roll`, `gimbal_yaw` |
| 6 | bits | `gimbal_mode` (0xC0) |
| 10 | bits | `gimbal_is_pitch_at_limit` (0x01), `_roll_` (0x02), `_yaw_` (0x04), `gimbal_is_stuck` (0x40) |

### 4 and 62: sticks

Type 4 is `4 × u16`: `rc_aileron`, `rc_elevator`, `rc_throttle`, `rc_rudder`,
raw (1024 is the centre). Type 62 (the sticks as the remote controller shows
them) has the same four values after 7 bytes; the Matrice 400 writes this one.

### 5: the app's record

| Offset | Type | Column |
|---|---|---|
| 0 | u8 | `app_is_photo` |
| 1 | u8 | `app_is_video` |
| 2 | f32 | horizontal speed (m/s): the OSD's, not kept |
| 6 | f32 (km) | `app_distance` (m): distance flown since take-off |
| 10 | i64 (ms) | `app_time` (s): the app's clock, UNIX time (UTC) |

On the four logs, `app_time` agreed with the app's time of the frames
(`start_time + osd_fly_time`) within 0.1 s (median offset −0.01 to +0.04 s,
drift about 0.01 s per 10 minutes): both are the app's clock, which runs
0.24-0.49 s behind the aircraft's (see [Time](#time)).

### 9 and 10: the app's messages

NUL-terminated UTF-8: `app_tip` (9), `app_warn` (10). Several in one frame are
joined with `; `.

### 14: app GPS

The position of the app's device, here the remote controller: `f64 latitude,
f64 longitude` (degrees), `f32 altitude` (m): `app_gps_latitude`,
`app_gps_longitude`, `app_gps_altitude`. On the flights checked, it stayed
about 5 m from the home point, within 1 m.

### 19: flight controller settings

| Offset | Type | Column |
|---|---|---|
| 0 | u8 | `mc_fail_safe`: the action on a lost link (0 hover, 1 land, 2 go home) |
| 1 | bits | `mc_is_vision_positioning_on` (0x01), `mc_is_obstacle_avoidance_on` (0x02), `mc_is_user_avoidance_on` (0x04) |

Bytes 2-32 were zero on the Matrice 400.

### 22: smart batteries

The first byte is the sub-type; the second is the battery's index `i`.

| Sub-type | Layout from byte 2 | Columns |
|---|---|---|
| 2 (dynamic) | i32 mV, i32 mA, u32 full, u32 remaining (mAh), i16 temperature × 10 (°C), u8 cells, u8 percent | `battery<i>_voltage`, `_current` (negative when discharging), `_full_capacity`, `_remaining_capacity`, `_temperature`, `_percent` |
| 3 (cells) | u8 n, n × u16 mV | `battery<i>_cell_voltage_1` ... `_n` |
| 1 (static) | serial number, design, model strings | not kept |

The Matrice 400 logs showed one battery entry (index 0, 13 cells). Its
dynamic record is 43-45 bytes: bytes 22-29 are a state word (u64) and bytes
30-44 are not decoded.

### 49: radio link

One byte: the signal in % (0x7F), and whether it is the uplink (0x80 set) or
the downlink: `link_uplink_signal`, `link_downlink_signal`. The two alternate,
at 2 Hz each; both stayed at 90 on the flights checked.

### 15 and 40: firmware and components (details)

Not in the table; in `details`:

- **15** (firmware): `u8 sender` (1 camera, 3 flight controller, 4 gimbal,
  6 remote controller, 11 battery), `u8 sub-sender`, 4 bytes of version:
  `details["firmware"]`, e.g. `{"flight controller": "9.5.0.32"}`.
- **40** (component serial): `u16 component` (1 camera, 2 aircraft,
  3 remote controller, 4 battery), `u8 n`, n bytes:
  `details["component_serials"]`.

## Records 45, 48 and 63

Layouts as `dji-log-parser` 0.6.0 (`RTKDifference`, `GroundStationPushData`,
`FlightControllerCommonOSDField`), checked on the four Matrice 400 logs.

### 48: position and time

Every frame (10 Hz):

| Offset | Type | Column | Check on the Matrice 400 logs |
|---|---|---|---|
| 0, 8 | f64 (rad) | `osd_abs_longitude`, `osd_abs_latitude` | the OSD's position, within 0.35 m |
| 16 | f32 | height above the take-off point (not kept: `osd_height`) | |
| 20 | i16 / 10 | `osd_abs_yaw` (deg) | the OSD's yaw within 0.1° |
| 24, 28 | u32, u32 | **`osd_aircraft_time`** = seconds + nanoseconds × 10⁻⁹: the aircraft's time, UNIX (UTC on these logs) | see [Time](#time) |
| 33 | u8 | `osd_abs_gps_num` | 19-25, 2-5 below the OSD's `osd_gps_num`: another count, not confirmed |
| 34, 42 | f64 (rad) | home position (not kept: `home_*`) | |
| 50 | f32 | `home_abs_altitude`: ellipsoidal height of the take-off point (m) | 5180.2 m on every flight from the same spot |
| 54 | f32 | `osd_home_distance`: horizontal distance to the home point (m) | √(north² + east²) within 0.4 m |
| 58 | f32 | `osd_abs_altitude`: the aircraft's height (m), found on these logs | `home_abs_altitude` + `osd_height` within 1.5 m; the RTK height within 0.1 m |
| 66 | f32 | `osd_geoid_height` (m, when the record has 70 bytes): the geoid above the ellipsoid | 31.46 m |

The app's barometric `home_altitude` (record 2) varied between 4980 and
5378 m over the same flights. Bytes 22-23, 32 and 62-65 are not read.

A record 48 usually follows the OSD record of its frame, but at times comes a
frame early or late (on one flight 86 frames had two and 87 none). PILS
places each in the frame whose time is nearest its own (see [Time](#time)).

### 45: RTK solution

Once a second:

| Offset | Type | Column |
|---|---|---|
| 1 | u8 | `rtk_pos_flag`: 0-15 none, 16 single, 17-49 float, **50 fixed** |
| 2-10 | 9 × u8 | satellites: antenna 1, antenna 2, base, each GPS, BeiDou, GLONASS: `rtk_antenna1_gps`, ..., `rtk_base_glonass` |
| 11, 19 | f64 (deg) | `rtk_latitude`, `rtk_longitude` |
| 27 | f32 | `rtk_height`: above the WGS84 ellipsoid (m) |
| 31, 39 | f64 (deg) | `rtk_base_latitude`, `rtk_base_longitude`: the position the base broadcasts (null when 0, 0: no base) |
| 47 | f32 | `rtk_base_height` (m, null when 0) |
| 65, 69, 73 | 3 × f32 | `rtk_std_latitude`, `rtk_std_longitude`, `rtk_std_height`: standard deviations (m) |
| 77-79 | 3 × u8 | Galileo satellites of antenna 1, antenna 2, base: `rtk_*_galileo` |
| 80-82 | 3 × u8 | QZSS satellites: `rtk_*_qzss` |

A satellite count is stored shifted by one bit, the low bit set when it is
known (null otherwise: QZSS on these logs). Bytes 0 and 51-64 are not read.

On the flights checked: fixed in every epoch; standard deviations of 2.1-2.4 mm
horizontally and 5-7 mm vertically; 9-10 GPS, 6-8 BeiDou, 6-8 GLONASS and 6-7
Galileo satellites per antenna. The solution is logged late: 0.1-0.3 s on
the flights checked (see
[the record track](#synchronization-the-records-track-tied-to-its-rtk)).
With a survey, AZEL's base correction absorbs a constant datum offset between
the base's broadcast position and the survey.

### 63: flight controller estimates

`u16 subtype`, then for subtype 0 (once a second):

| Offset | Type | Column |
|---|---|---|
| 2 | u16 | `fc_flight_time_left` (s, 0 on the ground) |
| 4 | u16 | `fc_go_home_time` (s; null when 0xFFFF, before take-off) |
| 6 | u16 | `fc_land_time` (s) |
| 8 | u8 | `fc_go_home_battery`: battery needed to return home (%) |
| 9 | u8 | `fc_land_battery`: battery needed to land (%) |

On the flights checked: up to about 1500 s of flight time left, up to 129 s
to land (it grows with the height), 27-32 % of battery to return home and
5-10 % to land. Subtype 1 (a countdown from 9, meaning unknown) is not read.

### OSD bytes 74-81: position from home

`f32 north, f32 east` from the home point, in metres (`osd_north`,
`osd_east`). They equal the OSD latitude and longitude minus the home
point's, in metres, within 0.1 m.

In the schema (`pils.schema`, table `dji_record`) these become, among others,
`aircraft_time_s`, `abs_lat_deg`, `abs_lon_deg`, `abs_h_m`, `home_abs_h_m`,
`home_distance_m`, `geoid_height_m`, `rtk_pos_flag_1`, `rtk_lat_deg`,
`rtk_lon_deg`, `rtk_h_ell_m`, `rtk_std_lat_m`, `rtk_base_lat_deg`,
`fc_flight_time_left_s`, `north_m` and `east_m`.

### Present, not decoded

What the investigation found, without a meaning that the logs could confirm:

| Where | What was seen |
|---|---|
| OSD u16 @ 63 | 0 on the ground, 1900-2100 in flight: possibly the motors' speed (rpm); nothing in the log to check it against |
| OSD u8 @ 68 | a code that follows `osd_flight_action` (r ≈ 0.9) |
| OSD i16 @ 53, u8 @ 66, 71, 82, 83 | vary, with no relation to a known signal |
| OSD 44-52, 55-62, 65, 67, 69-70, 72-73, 84 | constant |
| Home 32-96 | byte 33 repeats the OSD's ground/air state; 52-67 a serial number (`…2YUW`); 68-72 vary; the rest is constant |
| 254 (35 Hz) | a key-value log with the header of 253: `u8 5`, 3 bytes (an id at byte 2), `u32 n` (high bit: a time follows), then a protobuf value (`08 xx`: field 1, varint). Id 2 follows height and distance (r ≈ 0.85); id 3 takes a few values (1-8); ids 6 and 7 hold a millisecond time, constant per flight. DJI's library calls it "frame data"; no parser reads its values |
| 57 (6 Hz) | two constant blocks (3 and 35 bytes): link configuration |
| 17 | 9 bytes of flags, nearly constant |
| 26 | once: `41 00` |
| 55 | encrypted with the Security key, which DJI does not return |

## The table

`M400FlightRecord.data` has one row per frame.

- **Frames.** A frame starts at each OSD record (type 1) and takes every
  record after it up to the next OSD record.
- **States hold.** The values of OSD, home, gimbal, sticks, batteries, app
  GPS, flight controller settings and estimates (19, 63) and link hold until
  the next record of the same type. Records before the first frame give its starting state.
- **Epochs do not.** The RTK solution (45), the app's record (5) and the
  app's messages are only in the frame they came in; the other frames have
  nulls. Record 48 goes to the frame of its own time (below).
- **Order.** Rows stay in the order of the file.

### Time

```
timestamp = osd_aircraft_time        (record 48: the aircraft's seconds + nanoseconds)
```

Each frame is timed by the aircraft's own clock, read in record 48.

1. **The app's time** of each frame is the log's `start_time` plus the OSD's
   `fly_time` (0.1 s steps).
2. **Placing record 48.** The median of aircraft time − app time over the
   records gives the offset; each record 48 goes to the frame whose app
   time + offset is nearest its own time (of two for one frame, the nearer is
   kept). This puts back the records that came a frame early or late.
3. **Frames without one** get their time interpolated between their
   neighbours; at the ends of the log, the app's time + the offset.
4. **Without any record 48** (other aircraft), the frames have the app's time.

`details["aircraft_minus_app_s"]` is that offset. On the first Matrice 400
flights it was **+0.44, +0.47, +0.49 and +0.24 s**, stable within a few
milliseconds over each flight: the app's clock (start time and its own
`app_time`, record 5) runs behind the aircraft's by a different amount each
flight. At 9 m/s, half a second is 4 m along the track. Before this, the
frames were timed by the app (start + fly time), and AZEL was that much early.

The file's name is not used either: it is in the remote controller's time
zone. On the first flights that was UTC+1, so `12-38-29` in the name was a
take-off at 11:38:29 UTC (08:38 in Chile).

## Synchronization: the record's track tied to its RTK

When a flight has no payload GPS, no `.DAT` and no LAGER telemetry, its
reference is the flight record's 10 Hz track, tied to the RTK
(`pils.drones.sources.record_track`):

1. **Track and RTK.** The track is the frames with a record-48 position
   (`abs_*`, on the aircraft's time); the RTK is the fixed epochs (flag 50).
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
| 11:38 | −0.23, −0.03, +0.03 | 0.50 / 0.63 | 0.34 |
| 12:02 | +0.20, −0.37, +0.08 | 0.51 / 0.39 | 0.22 |
| 12:18 | +0.21, −0.38, +0.05 | 0.41 / 0.39 | 0.14 |
| 12:32 | +0.21, −0.40, +0.04 | 0.40 / 0.39 | 0.28 |

Syncing on the 1 Hz RTK alone, interpolated to the 100 Hz grid, would miss
the motion between epochs by up to 2.5-3.8 m in the turns.

AZEL also takes the base position it uses for the correction from the record
(`rtk_base_*`, median), not from `TELESCOPES_CONFIG["dji_broadcast_geod"]`,
which is an earlier campaign's.

## API

::: pils.drones.dji.m400.flightrecord

::: pils.drones.dji.m400.txtlog

::: pils.drones.dji.m400.records
