# Binary Formats

Binary file format specifications for PILS sensors.

## KERNEL Inclinometer

### Message Structure

```
┌────────┬────────┬────────┬─────────┬──────────┬──────────┐
│ Header │ Length │  Type  │ Payload │ Checksum │
├────────┼────────┼────────┼─────────┼──────────┤
│ 2 bytes│ 1 byte │ 1 byte │ N bytes │ 2 bytes  │
│ AA 55  │        │        │         │ LSB MSB  │
└────────┴────────┴────────┴─────────┴──────────┘
```

### Header

| Bytes | Value | Description |
|-------|-------|-------------|
| 0-1 | `0xAA 0x55` | Sync pattern |

### Message Types

| Type | Address | Description | Payload Size |
|------|---------|-------------|--------------|
| Standard | `0x01` | IMU + angles | 20 bytes |
| Extended | `0x02` | With temperature | 24 bytes |
| Config | `0x03` | Configuration | variable |

### Standard Payload (Type 0x01)

| Offset | Size | Field | Type | Scale | Unit |
|--------|------|-------|------|-------|------|
| 0 | 2 | Acc_X | I2 | 1/8192 | g |
| 2 | 2 | Acc_Y | I2 | 1/8192 | g |
| 4 | 2 | Acc_Z | I2 | 1/8192 | g |
| 6 | 2 | Gyro_X | I2 | 1/100 | °/s |
| 8 | 2 | Gyro_Y | I2 | 1/100 | °/s |
| 10 | 2 | Gyro_Z | I2 | 1/100 | °/s |
| 12 | 2 | Roll | I2 | 1/100 | ° |
| 14 | 2 | Pitch | I2 | 1/100 | ° |
| 16 | 2 | USW | U2 | - | - |
| 18 | 2 | Checksum | U2 | - | - |

### Checksum Calculation

```python
def checksum(msg: bytes) -> bytes:
    """Calculate KERNEL checksum."""
    # Skip header (0xAA 0x55)
    if msg.startswith(b"\xaa\x55"):
        msg = msg[2:]
    
    # Sum all bytes, return as little-endian U16
    return sum(msg).to_bytes(2, byteorder="little", signed=False)
```

### Decoding Example

```python
import struct
from pils.decoders.KERNEL_utils import KernelMsg

# Raw message
msg = bytes([0xAA, 0x55, 0x14, 0x01, ...])

# Using decoder
decoder = KernelMsg()
data = decoder.decode_single(msg)

# Manual decode
if msg[0:2] == b'\xAA\x55':
    acc_x_raw = struct.unpack('<h', msg[4:6])[0]
    acc_x = acc_x_raw / 8192.0  # Convert to g
```

---

## u-blox GNSS

### UBX Protocol

```
┌──────┬──────┬───────┬────────┬─────────┬────────┬────────┐
│ Sync │ Sync │ Class │ ID     │ Length  │Payload │ CK     │
│ 0xB5 │ 0x62 │       │        │(2 bytes)│        │ A+B    │
└──────┴──────┴───────┴────────┴─────────┴────────┴────────┘
```

### NAV-PVT (Position, Velocity, Time)

Class: `0x01`, ID: `0x07`

| Offset | Size | Field | Type | Scale | Unit |
|--------|------|-------|------|-------|------|
| 0 | 4 | iTOW | U4 | 1 | ms |
| 4 | 2 | year | U2 | - | - |
| 6 | 1 | month | U1 | - | - |
| 7 | 1 | day | U1 | - | - |
| 8 | 1 | hour | U1 | - | - |
| 9 | 1 | min | U1 | - | - |
| 10 | 1 | sec | U1 | - | - |
| 11 | 1 | valid | U1 | - | flags |
| 12 | 4 | tAcc | U4 | 1 | ns |
| 16 | 4 | nano | I4 | 1 | ns |
| 20 | 1 | fixType | U1 | - | - |
| 21 | 1 | flags | U1 | - | - |
| 22 | 1 | flags2 | U1 | - | - |
| 23 | 1 | numSV | U1 | - | - |
| 24 | 4 | lon | I4 | 1e-7 | deg |
| 28 | 4 | lat | I4 | 1e-7 | deg |
| 32 | 4 | height | I4 | 1 | mm |
| 36 | 4 | hMSL | I4 | 1 | mm |
| 40 | 4 | hAcc | U4 | 1 | mm |
| 44 | 4 | vAcc | U4 | 1 | mm |

### Checksum (Fletcher-8)

```python
def ubx_checksum(payload: bytes) -> tuple[int, int]:
    """Calculate UBX checksum."""
    ck_a = 0
    ck_b = 0
    for byte in payload:
        ck_a = (ck_a + byte) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return ck_a, ck_b
```

---

## Generic Binary Patterns

### Integer Types

| Type | Bytes | Range | Python struct |
|------|-------|-------|---------------|
| U1 | 1 | 0-255 | `B` |
| I1 | 1 | -128 to 127 | `b` |
| U2 | 2 | 0-65535 | `H` |
| I2 | 2 | ±32767 | `h` |
| U4 | 4 | 0-4.3B | `I` |
| I4 | 4 | ±2.1B | `i` |

### Float Types

| Type | Bytes | Python struct |
|------|-------|---------------|
| F4 | 4 | `f` |
| F8 | 8 | `d` |

### Byte Order

| Order | Python | Description |
|-------|--------|-------------|
| Little-endian | `<` | LSB first (Intel) |
| Big-endian | `>` | MSB first (Network) |

---

## File Reading Example

```python
import struct
from pathlib import Path

def read_binary_records(filepath: Path, record_size: int) -> list[bytes]:
    """Read fixed-size binary records."""
    data = filepath.read_bytes()
    records = []
    
    for i in range(0, len(data), record_size):
        record = data[i:i + record_size]
        if len(record) == record_size:
            records.append(record)
    
    return records

def parse_record(record: bytes) -> dict:
    """Parse a binary record."""
    return {
        'timestamp': struct.unpack('<I', record[0:4])[0],
        'value': struct.unpack('<f', record[4:8])[0],
    }

# Usage
records = read_binary_records(Path("data.bin"), record_size=8)
parsed = [parse_record(r) for r in records]
```

---

## See Also

- [Text Formats](text.md) - CSV and ASCII formats
- [HDF5 Format](hdf5.md) - Hierarchical data format
- [Decoders API](../api/utilities/decoders.md)
