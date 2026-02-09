# Drones

Drone platform data access modules.

## Overview

PILS supports three drone platforms:

| Platform | Module | File Formats | Detection |
|----------|--------|--------------|-----------|
| [DJI](dji.md) | `pils.drones.DJIDrone` | `.csv`, `.DAT` | CSV headers |
| [BlackSquare](blacksquare.md) | `pils.drones.BlackSquareDrone` | `.BIN`, `.log` | Binary header |
| [Litchi](litchi.md) | `pils.drones.Litchi` | `.csv` | Column names |

## Import

```python
# Individual
from pils.drones import DJIDrone
from pils.drones import BlackSquareDrone
from pils.drones import Litchi

# All
from pils.drones import DJIDrone, BlackSquareDrone, Litchi
```

## Auto-Detection

When using `flight.add_drone_data()`, the platform is auto-detected:

```python
from pils.flight import Flight

flight = Flight(flight_info)
flight.add_drone_data()  # Auto-detects platform

# Check what was loaded
print(list(flight.drone_data.keys()))
# DJI: ['data']
# BlackSquare: ['GPS', 'IMU', 'BARO', 'ATT', ...]
# Litchi: ['data']
```

## Detection Logic

1. **DJI**: Look for `.csv` with DJI column headers or `.DAT` files
2. **BlackSquare**: Look for `.BIN` or `.log` with ArduPilot header (0xA395)
3. **Litchi**: Look for `.csv` with Litchi-specific columns

## Direct Instantiation

```python
# DJI
from pils.drones import DJIDrone
dji = DJIDrone(path="/data/drone_data/flight.csv")
dji.load_data()
df = dji.data

# BlackSquare
from pils.drones import BlackSquareDrone
bs = BlackSquareDrone(path="/data/drone_data/flight.BIN")
bs.load_data()
gps_df = bs.data['GPS']
imu_df = bs.data['IMU']

# Litchi
from pils.drones import Litchi
litchi = Litchi(path="/data/drone_data/mission.csv")
litchi.load_data()
df = litchi.data
```

## Detailed Documentation

- [DJI](dji.md) - Phantom, Mavic, Matrice series
- [BlackSquare](blacksquare.md) - ArduPilot-based drones
- [Litchi](litchi.md) - Mission planner data
