# DJI Drone Data

PILS can perform automatic decoding of the ```.DAT``` file created by the DJI Matrice 600. The decoding covers the GPS, RTK, IMU attitude, motor, battery and controller records, checked against DatCon's CSV of the same flight. The details for DJI drone data is presented here. 


## Supported Formats

=== "CSV Export"

    DATCON CSV exports:
    
    ```
    datetime,latitude,longitude,altitude,height,velocity_x,velocity_y,
    velocity_z,pitch,roll,yaw,gimbal_pitch,gimbal_roll,gimbal_yaw,...
    ```

=== "DAT Files"

    Encrypted binary flight logs:
    
    - `.DAT` files from DJI Matrice 600
    - Internal decryption tool available
    - XOR-encrypted messages
    - Organized by message type (GPS, RTK, IMU attitude, motors, battery, ...)

??? info "DAT File Format & Decoding"

    DAT files contain encrypted binary messages that are parsed and decrypted by PILS.

    #### Message Structure

    Each message in a DAT file follows this structure:

    ```
    Byte 0:       0x55 (marker)
    Byte 1:       Record length (header, payload and CRC)
    Bytes 2-3:    Reserved
    Bytes 4-5:    Message type (little-endian uint16)
    Bytes 6-9:    Tick (little-endian uint32); byte 6 is the XOR key
    Bytes 10..-3: Encrypted payload
    Last 2 bytes: CRC-16 of the record (seed 0x3692)
    ```

    #### Decoding Process

    1. **Find records**: every `0x55` byte is a candidate record start; a
       candidate is kept when its CRC-16 (last two bytes) matches, and dropped
       when it falls inside an already accepted record (`0x55` also occurs
       inside encrypted payloads). Done with numpy for all records at once
       (~4 million records in about 2 s).
    2. **Unwrap ticks**: the tick (4.5 MHz flight-controller clock) is a
       uint32 and wraps during long logs; ticks are unwrapped over the whole
       file, so all record types stay on one time axis.
    3. **Decrypt payload**: XOR each payload byte with the tick's low byte
    4. **Unpack fields**: each record type is decoded in one step from fixed
       byte offsets, with unit conversions
    5. **Timestamps**: a linear tick → UTC fit on the GPS records gives the
       `timestamp` of every record

    Before October 2026 the file was split on `0x55` bytes, which dropped
    about one GPS/RTK record in five (those with a `0x55` in their payload).

    #### Decoded records

    GPS and RTK make the drone table (`flight.raw_data.drone_data.drone`). The
    other records are in `flight.raw_data.drone_data.records`, one table per
    record with `tick` and UTC `timestamp`, and are saved in the flight HDF5
    under `raw_data/drone_data/dat_records/`. Names follow DatCon's CSV
    columns. Every field was compared with DatCon 4.3.0's CSV of the same
    flight, tick by tick, unless marked otherwise.

    | Record | Type | Rate | Content |
    |--------|------|------|---------|
    | GPS | 2096 | 5 Hz | Main GNSS receiver (drone table) |
    | RTK | 53234 | 5 Hz | RTK solution (drone table) |
    | `IMU_ATTI(0..2)` | 2048-2050 | 200 Hz | Three IMUs: quaternion and DatCon's roll/pitch/yaw, accelerometer (g), gyroscope (deg/s), velocities, magnetometer, barometric and fused altitude, IMU temperature, fused position |
    | `GPS(1)`, `GPS(2)` | 2097, 2098 | 5 Hz | Other GNSS receivers, GPS layout (DatCon does not export them) |
    | `Motor` | 10090 | 25 Hz | Per motor (RFront, LFront, LSide, LBack, RBack, RSide): speed (rpm), ESC voltage and temperature, current (not in DatCon's CSV), PPM in/out (%), status |
    | `MotorCtrl` | 1307 | 260 Hz | PWM command per motor (%) |
    | `BatteryInfo` | 1710 | 50 Hz | Voltages (V), currents (A), capacity, temperature (°C) |
    | `Controller` | 1000 | 50 Hz | Stick inputs, flight mode, motor state, limits |
    | `AirSpeed` | 10100 | 5 Hz | Wind speed estimate (`windX`/`windY` not in DatCon's CSV) |
    | `osd_data` | 12 | 20 Hz | Relative height, VPS height, flight time |

    The camera gimbal attitude is not in the flight controller log.

    `BatteryInfo` is in physical units; DatCon's `:D` columns have the raw
    counts (mV, mA, 0.1 °C). Motor PPM is the pulse width in µs / 19.2.

    DatCon's euler convention (kept so the columns match its CSV):
    `roll = asin(2(wx + yz))`, `pitch = atan2(2(wy − xz), 1 − 2(x² + y²))`,
    `yaw = atan2(2(wz − xy), 1 − 2(z² + x²))`.

    The layouts come from DatCon's record classes
    ([source](https://github.com/BudWalkerJava/DatCon), `src/DatConRecs`) and
    are in `pils/drones/DJIDrone.py` (`MESSAGE_DEFINITIONS`, `DAT_RECORDS`).
    About 100 record types are in the file; the others are not decoded yet.

    #### GPS Message (Type 2096)

    | Field | Format | Offset | Unit | Conversion | Description |
    |-------|--------|--------|------|------------|-------------|
    | `date` | `I` | 0 | - | - | Date (YYYYMMDD) |
    | `time` | `I` | 4 | - | - | Time (HHMMSS) |
    | `longitude` | `i` | 8 | degrees | ÷ 1e7 | GPS longitude |
    | `latitude` | `i` | 12 | degrees | ÷ 1e7 | GPS latitude |
    | `heightMSL` | `i` | 16 | m | ÷ 1000 | Height above MSL |
    | `velN` | `f` | 20 | m/s | ÷ 100 | North velocity |
    | `velE` | `f` | 24 | m/s | ÷ 100 | East velocity |
    | `velD` | `f` | 28 | m/s | ÷ 100 | Down velocity |
    | `hdop` | `f` | 32 | - | ÷ 100 | Horizontal DOP |
    | `pdop` | `f` | 36 | - | ÷ 100 | Position DOP |
    | `hacc` | `f` | 40 | m | ÷ 100 | Horizontal accuracy (not checked) |
    | `sacc` | `f` | 44 | m/s | ÷ 100 | Speed accuracy |
    | `numGPS` | `I` | 56 | - | - | GPS satellites |
    | `numGLN` | `I` | 60 | - | - | GLONASS satellites |
    | `numSV` | `H` | 64 | - | - | Total satellites |

    #### RTK Message (Type 53234)

    | Field | Format | Offset | Unit | Conversion | Description |
    |-------|--------|--------|------|------------|-------------|
    | `date` | `I` | 0 | - | - | Date (YYYYMMDD, 0 in these logs) |
    | `time` | `I` | 4 | - | - | Time (HHMMSS, 0 in these logs) |
    | `lon_p` | `d` | 8 | degrees | - | RTK longitude (primary) |
    | `lat_p` | `d` | 16 | degrees | - | RTK latitude (primary) |
    | `hmsl_p` | `f` | 24 | m | - | Height MSL (primary) |
    | `lon_s` | `i` | 28 | degrees | ÷ 1e7 | Longitude (secondary) |
    | `lat_s` | `i` | 32 | degrees | ÷ 1e7 | Latitude (secondary) |
    | `hmsl_s` | `i` | 36 | m | ÷ 1000 | Height MSL (secondary) |
    | `vel_n` | `f` | 40 | m/s | ÷ 100 | North velocity |
    | `vel_e` | `f` | 44 | m/s | ÷ 100 | East velocity |
    | `vel_d` | `f` | 48 | m/s | ÷ 100 | Down velocity |
    | `yaw` | `h` | 52 | degrees | - | Heading from the two RTK antennas |
    | `svn_s` | `B` | 54 | - | - | Satellites (secondary) |
    | `svn_p` | `B` | 55 | - | - | Satellites (primary) |
    | `hdop` | `f` | 56 | - | ÷ 100 | Horizontal DOP |
    | `pitch` | `f` | 60 | degrees | - | Pitch angle |
    | `pos_flg_0-5` | `B` | 64-69 | - | - | Position flags |
    | `gps_state` | `H` | 70 | - | - | GPS state code |

    !!! note "Struct Format Codes"
        - `I` = unsigned int (4 bytes)
        - `i` = signed int (4 bytes)
        - `f` = float (4 bytes)
        - `d` = double (8 bytes)
        - `H` = unsigned short (2 bytes)
        - `h` = signed short (2 bytes)
        - `B` = unsigned byte (1 byte)