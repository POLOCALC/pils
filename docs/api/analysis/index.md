# Analysis

Positioning analysis modules.

## Overview

| Module | Description |
|--------|-------------|
| [PPK](ppk.md) | Post-Processed Kinematic analysis |
| [RTK](rtk.md) | Real-Time Kinematic analysis |

## Import

```python
from pils.analyze.ppk import PPKAnalyzer
from pils.analyze.rtkdata.RTKLIB import RTKAnalyzer
```

## Quick Start

### PPK Analysis

```python
from pils.analyze.ppk import PPKAnalyzer

# Initialize
ppk = PPKAnalyzer(flight_path=flight.flight_path)

# Process
version = ppk.process(
    rover_obs="rover.obs",
    base_obs="base.obs",
    nav="rover.nav"
)

# Results
print(f"Fix rate: {version.fix_rate:.1%}")
position_df = version.position
```

### RTK Analysis

```python
from pils.analyze.rtkdata.RTKLIB import RTKAnalyzer

# Analyze existing solution
rtk = RTKAnalyzer(pos_file="solution.pos")
stats = rtk.get_statistics()
```

## Detailed Documentation

- [PPK](ppk.md) - Post-processed positioning with version control
- [RTK](rtk.md) - Real-time positioning analysis
