# Analysis

Flight analysis modules. Lab measurements have their own [lab analysis](../lab/index.md) modules.

## Overview

| Module | Description |
|--------|-------------|
| [Packaging](package.md) | Sync a flight and save all its data products in `proc/` |
| [Quick look](quicklook.md) | Markdown report of a flight's data and their quality |
| [AZEL](azel.md) | Azimuth, elevation and range of the drone from each telescope |
| [PPK](ppk.md) | Post-Processed Kinematic analysis |
| [Power](power.md) | Expected source power on telescope detectors |

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
