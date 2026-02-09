# DJI Drone Data

PILS can perform automatic decoding of the ```.DAT``` file created by the DJI Matrice 600. The resulting conversion does not return a detailed file such as DATCON but has all the required fields for the project analysis. The details for DJI drone data is presented here. 


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
    - Organized by message type (GPS, RTK)

??? info "DAT File Format & Decoding"

    DAT files contain encrypted binary messages that are parsed and decrypted by PILS.

    #### Message Structure

    Each message in a DAT file follows this structure:

    ```
    Byte 0:       0x55 (marker)
    Byte 1:       Message length
    Bytes 2-4:    Reserved
    Bytes 4-5:    Message type (little-endian uint16)
    Byte 6:       XOR encryption key
    Bytes 6-9:    Tick (timestamp, little-endian uint32)
    Bytes 10+:    Encrypted payload
    ```

    #### Decoding Process

    1. **Split messages**: File is split on `0x55` marker bytes
    2. **Extract header**: Parse message type, encryption key, and tick
    3. **Decrypt payload**: XOR each payload byte with the key byte
    4. **Unpack fields**: Use struct format codes to extract fields from specific byte offsets
    5. **Convert units**: Apply conversion functions (e.g., lat/lon scaling)
    6. **Organize by type**: Group messages into separate DataFrames by type

    #### Supported Message Types

    | Message Type | ID | Description | Payload Size |
    |--------------|-----|-------------|--------------|
    | GPS | 2096 | Standard GPS data | 66 bytes |
    | RTK | 53234 | RTK positioning data | 72 bytes |

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
    | `hdop` | `f` | 32 | - | - | Horizontal DOP |
    | `pdop` | `f` | 36 | - | - | Position DOP |
    | `hacc` | `f` | 40 | m | - | Horizontal accuracy |
    | `sacc` | `f` | 44 | m | - | Speed accuracy |
    | `numGPS` | `I` | 56 | - | - | GPS satellites |
    | `numGLN` | `I` | 60 | - | - | GLONASS satellites |
    | `numSV` | `H` | 64 | - | - | Total satellites |

    #### RTK Message (Type 53234)

    | Field | Format | Offset | Unit | Conversion | Description |
    |-------|--------|--------|------|------------|-------------|
    | `date` | `I` | 0 | - | - | Date (YYYYMMDD) |
    | `time` | `I` | 4 | - | - | Time (HHMMSS) |
    | `lon_p` | `d` | 8 | degrees | - | RTK longitude (primary) |
    | `lat_p` | `d` | 16 | degrees | - | RTK latitude (primary) |
    | `hmsl_p` | `f` | 24 | m | - | Height MSL (primary) |
    | `lon_s` | `i` | 28 | degrees | ÷ 1e7 | Longitude (secondary) |
    | `lat_s` | `i` | 32 | degrees | ÷ 1e7 | Latitude (secondary) |
    | `hmsl_s` | `i` | 36 | m | ÷ 1000 | Height MSL (secondary) |
    | `vel_n` | `f` | 40 | m/s | - | North velocity |
    | `vel_e` | `f` | 44 | m/s | - | East velocity |
    | `vel_d` | `f` | 48 | m/s | - | Down velocity |
    | `yaw` | `h` | 50 | degrees | - | Yaw angle |
    | `svn_s` | `B` | 52 | - | - | Satellites (secondary) |
    | `svn_p` | `B` | 53 | - | - | Satellites (primary) |
    | `hdop` | `f` | 54 | - | - | Horizontal DOP |
    | `pitch` | `f` | 58 | degrees | - | Pitch angle |
    | `pos_flg_0-5` | `B` | 62-67 | - | - | Position flags |
    | `gps_state` | `H` | 68 | - | - | GPS state code |

    !!! note "Struct Format Codes"
        - `I` = unsigned int (4 bytes)
        - `i` = signed int (4 bytes)
        - `f` = float (4 bytes)
        - `d` = double (8 bytes)
        - `H` = unsigned short (2 bytes)
        - `h` = signed short (2 bytes)
        - `B` = unsigned byte (1 byte)