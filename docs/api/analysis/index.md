# Analysis

Positioning analysis modules.

## Overview

| Module | Description |
|--------|-------------|
| [PPK](ppk.md) | Post-Processed Kinematic analysis |

## Import

```python
from pils.analyze.ppk import PPKAnalysis
```

## Quick Start

### PPK Analysis

```python
from pils.analyze.ppk import PPKAnalysis

# Initialize
ppk = PPKAnalysis(flight_path=flight.flight_path)

# Process
version = ppk.run_analysis(
    config_path="rtklib.conf",
    rover_obs="rover.obs",
    base_obs="base.obs",
    nav_file="rover.nav"
)

# Results
if version:
    print(f"Fix rate: {version.fix_rate:.1%}")
    position_df = version.position
```

## Detailed Documentation

- [PPK](ppk.md) - Post-processed positioning with version control
- [RTK](rtk.md) - Real-time positioning analysis
