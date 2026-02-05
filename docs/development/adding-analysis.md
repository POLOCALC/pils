# Adding Analysis Modules

Extend PILS with new analysis capabilities.

## Overview

PILS analysis modules are standalone components that process flight data to extract insights. Analysis modules should:

1. Be independent from the Flight class
2. Support versioned execution (track when analysis was run)
3. Store results in HDF5 format
4. Use Polars DataFrames for data processing
5. Implement smart re-run logic (don't reprocess unchanged data)

---

## Analysis Module Structure

### Core Principles

**Standalone Design:**
```python
# ✓ Good - Independent analysis
analysis = MyAnalysis(flight_path)
result = analysis.run()

# ✗ Bad - Coupled to Flight
flight.run_my_analysis()
```

**Versioned Results:**
```python
# Each run creates a timestamped version
analysis.run(config)  # → rev_20260204_143022
analysis.run(config)  # → rev_20260204_150315 (if config changed)
```

**HDF5 Persistence:**
```python
# Save all versions to HDF5
analysis.to_hdf5('results.h5')

# Load from HDF5
analysis = MyAnalysis.from_hdf5('results.h5')
latest = analysis.get_latest_version()
```

---

## File Organization & HDF5 Structure

### Primary Storage: HDF5 File

**All analysis results MUST be stored in a single HDF5 file** with versioned structure:

```
flight_dir/proc/my_analysis/
└── my_analysis.h5          # Primary storage (HDF5 format)
```

**HDF5 Structure:**
```
my_analysis.h5
├── [attrs]                          # File-level metadata
│   ├── analysis_type: "my_analysis"
│   ├── flight_path: "/path/to/flight"
│   └── latest_version: "rev_20260204_143022"
│
├── rev_20260204_143022/            # Group = Version 1
│   ├── [attrs]                      # Version metadata
│   │   ├── config_hash: "a1b2c3..."
│   │   ├── execution_time: "2026-02-04T14:30:22"
│   │   └── revision_path: "proc/my_analysis/rev_20260204_143022"
│   │
│   └── result_data/                 # Nested group with datasets
│       ├── timestamp [dataset]      # Column data
│       ├── latitude [dataset]
│       ├── longitude [dataset]
│       └── [attrs]
│           └── columns: ["timestamp", "latitude", "longitude"]
│
└── rev_20260204_150315/            # Group = Version 2
    ├── [attrs]
    └── result_data/
```

**Key Points:**

- Each **revision is a separate HDF5 group**
- **Metadata stored in group attributes** (`attrs`)
- **Results stored as HDF5 datasets** (columnar format for DataFrames)
- **Single file contains all versions** - easy to share and backup

### Auxiliary Storage: Revision Folders

**Revision folders are for auxiliary outputs ONLY** (plots, logs, intermediate files):

```
flight_dir/proc/my_analysis/
├── my_analysis.h5                    # ← PRIMARY: All analysis results
├── rev_20260204_143022/              # ← AUXILIARY: Plots and extras
│   ├── velocity_plot.png             # Visualization
│   ├── statistics.json               # Summary report
│   ├── debug.log                     # Execution log
│   └── intermediate_data.csv         # Optional intermediate output
└── rev_20260204_150315/
    ├── velocity_plot.png
    └── statistics.json
```

**What Goes Where:**

| Data Type | Storage Location | Format | Purpose |
|-----------|------------------|--------|---------|
| **Analysis results** (DataFrames) | `my_analysis.h5` (HDF5 groups) | HDF5 datasets | Primary data, fast loading |
| **Metadata** (config, timestamps) | `my_analysis.h5` (group attrs) | HDF5 attributes | Version tracking |
| **Plots** (figures, visualizations) | `rev_*/` folders | PNG, PDF, SVG | Human review |
| **Reports** (summaries, statistics) | `rev_*/` folders | JSON, TXT, MD | Documentation |
| **Logs** (execution details) | `rev_*/` folders | LOG, TXT | Debugging |
| **Intermediate data** (optional) | `rev_*/` folders | CSV, Parquet | Inspection |

!!! warning "Do Not Store Results in Folders"
    Revision folders are **NOT** for primary data storage. Always save results to HDF5.
    
    ```python
    # ✓ Correct - Results to HDF5
    result_df.to_hdf5(hdf5_group)
    
    # ✗ Wrong - Don't use folders for primary data
    result_df.write_csv(revision_folder / "results.csv")
    ```

!!! tip "Why HDF5 for Primary Data?"
    - **Fast**: Binary format, no parsing overhead
    - **Compact**: Efficient compression
    - **Versioned**: All revisions in one file
    - **Portable**: Single file to share/backup
    - **Self-describing**: Metadata embedded as attributes

### Implementation Example

```python
def run_analysis(self, config: Dict[str, Any]) -> MyAnalysisVersion:
    """Run analysis with proper storage."""
    version_name = self._generate_version_name()
    revision_path = self.proc_path / version_name
    revision_path.mkdir(parents=True, exist_ok=True)
    
    # Execute analysis
    result_df = self._execute_analysis(config)
    
    # ✓ PRIMARY: Save to HDF5 (handled by to_hdf5())
    # This happens when we call self.to_hdf5() at the end
    
    # ✓ AUXILIARY: Save plots/reports to revision folder
    self._save_plot(result_df, revision_path / "velocity_plot.png")
    self._save_report(statistics, revision_path / "report.json")
    
    # Create version object
    version = MyAnalysisVersion(
        version_name=version_name,
        result_data=result_df,  # This will go to HDF5
        metadata={'config_hash': config_hash},
        revision_path=revision_path
    )
    
    self.versions[version_name] = version
    
    # Save all versions to HDF5
    self.to_hdf5(self.proc_path / "my_analysis.h5")
    
    return version
```

---

## Analysis Template

### Basic Structure

```python
"""
My analysis module.

Brief description of what this analysis does and why it's useful.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class MyAnalysisVersion:
    """
    Container for a single analysis revision.
    
    Attributes:
        version_name: Auto-timestamp version (rev_YYYYMMDD_HHMMSS)
        result_data: Analysis results as Polars DataFrame
        metadata: Config hash and execution parameters
        revision_path: Path to revision folder
    """
    version_name: str
    result_data: pl.DataFrame
    metadata: Dict[str, Any]
    revision_path: Path


class MyAnalysis:
    """
    My analysis module with versioning and HDF5 storage.
    
    This analysis processes flight data to compute [describe output].
    Supports smart re-run logic based on configuration changes.
    
    File Structure:
        flight_dir/proc/my_analysis/
        ├── my_analysis.h5               # PRIMARY: All results (HDF5)
        └── rev_20260204_143022/         # AUXILIARY: Plots, logs, reports
            ├── velocity_plot.png         # Visualization
            ├── statistics.json           # Summary report
            └── debug.log                 # Execution log
    
    Attributes:
        flight_path: Path to flight directory
        proc_path: Path to processing directory (flight_path/proc/my_analysis)
        versions: Dictionary of all analysis versions
        latest_version: Name of most recent version
    
    Example:
        >>> # Create new analysis
        >>> analysis = MyAnalysis('/path/to/flight')
        >>> 
        >>> # Run with configuration
        >>> config = {'param1': 10, 'param2': 'value'}
        >>> version = analysis.run_analysis(config)
        >>> print(version.result_data.head())
        >>> 
        >>> # Smart re-run (only if config changed)
        >>> version2 = analysis.run_analysis(config)  # Skips execution
        >>> 
        >>> # Force re-run
        >>> version3 = analysis.run_analysis(config, force=True)
        >>> 
        >>> # Load existing analysis
        >>> analysis = MyAnalysis.from_hdf5('/path/to/flight/proc/my_analysis/my_analysis.h5')
        >>> latest = analysis.get_latest_version()
    """
    
    def __init__(self, flight_path: str | Path) -> None:
        """
        Initialize analysis for a flight directory.
        
        Args:
            flight_path: Path to flight directory
        """
        self.flight_path = Path(flight_path)
        self.proc_path = self.flight_path / "proc" / "my_analysis"
        self.proc_path.mkdir(parents=True, exist_ok=True)
        
        self.versions: Dict[str, MyAnalysisVersion] = {}
        self.latest_version: Optional[str] = None
        
        # Load existing versions if HDF5 exists
        h5_path = self.proc_path / "my_analysis.h5"
        if h5_path.exists():
            self._load_from_hdf5(h5_path)
    
    def run_analysis(
        self,
        config: Dict[str, Any],
        force: bool = False
    ) -> MyAnalysisVersion:
        """
        Execute analysis with smart re-run logic.
        
        Only runs if configuration changed or force=True.
        Creates a new timestamped version for each execution.
        
        Args:
            config: Analysis configuration parameters
            force: If True, run even if config unchanged
        
        Returns:
            Analysis version with results
        
        Example:
            >>> config = {'threshold': 0.5, 'window_size': 10}
            >>> version = analysis.run_analysis(config)
            >>> print(f"Version: {version.version_name}")
            >>> print(version.result_data.head())
        """
        # Compute config hash
        config_hash = self._compute_config_hash(config)
        
        # Check if we need to run
        if not force and self._config_exists(config_hash):
            logger.info(f"Config unchanged (hash: {config_hash[:8]}...), skipping execution")
            return self.get_latest_version()
        
        # Generate version name
        version_name = self._generate_version_name()
        revision_path = self.proc_path / version_name
        revision_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Running analysis: {version_name}")
        
        # Execute analysis
        result_data = self._execute_analysis(config, revision_path)
        
        # Save auxiliary data to revision folder (plots, reports, etc.)
        self._save_auxiliary_outputs(result_data, config, revision_path)
        
        # Create version object (result_data will be saved to HDF5)
        version = MyAnalysisVersion(
            version_name=version_name,
            result_data=result_data,
            metadata={
                'config_hash': config_hash,
                'config': config,
                'execution_time': datetime.now().isoformat(),
            },
            revision_path=revision_path
        )
        
        # Store version
        self.versions[version_name] = version
        self.latest_version = version_name
        
        # Save to HDF5
        self.to_hdf5(self.proc_path / "my_analysis.h5")
        
        return version
    
    def _execute_analysis(
        self,
        config: Dict[str, Any],
        output_path: Path
    ) -> pl.DataFrame:
        """
        Execute the actual analysis computation.
        
        This is where you implement your analysis logic.
        
        Args:
            config: Analysis parameters
            output_path: Path to save intermediate results
        
        Returns:
            Analysis results as Polars DataFrame
        """
        # Load input data (example: GPS from flight)
        gps_path = self.flight_path / "aux" / "gps" / "gps.csv"
        if not gps_path.exists():
            raise FileNotFoundError(f"GPS data not found: {gps_path}")
        
        df = pl.read_csv(gps_path)
        
        # Apply analysis logic
        threshold = config.get('threshold', 0.5)
        window_size = config.get('window_size', 10)
        
        # Example: compute rolling statistics
        result = df.with_columns([
            pl.col('latitude').rolling_mean(window_size).alias('lat_smooth'),
            pl.col('longitude').rolling_mean(window_size).alias('lon_smooth'),
            (pl.col('latitude').abs() > threshold).alias('exceeds_threshold')
        ])
        
        # Note: Primary results returned (will be saved to HDF5)
        # Auxiliary data saved separately in _save_auxiliary_outputs()
        
        return result
    
    def _save_auxiliary_outputs(
        self,
        result_data: pl.DataFrame,
        config: Dict[str, Any],
        output_path: Path
    ) -> None:
        """
        Save auxiliary outputs to revision folder.
        
        This is for plots, reports, logs - NOT primary data.
        Primary data (result_data) is saved to HDF5 automatically.
        
        Args:
            result_data: Analysis results (already in memory)
            config: Configuration used
            output_path: Revision folder path
        """
        import json
        
        # Save configuration as JSON
        with open(output_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Optional: Save summary statistics
        stats = {
            'num_samples': result_data.height,
            'num_exceeded': int(result_data['exceeds_threshold'].sum()),
            'max_lat': float(result_data['lat_smooth'].max()),
        }
        with open(output_path / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Optional: Create a plot (example)
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.plot(result_data['timestamp'], result_data['lat_smooth'])
        # fig.savefig(output_path / "latitude_plot.png")
        # plt.close(fig)
    
    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute SHA256 hash of configuration."""
        import json
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def _config_exists(self, config_hash: str) -> bool:
        """Check if this configuration was already run."""
        for version in self.versions.values():
            if version.metadata.get('config_hash') == config_hash:
                return True
        return False
    
    def _generate_version_name(self) -> str:
        """Generate timestamped version name."""
        return datetime.now().strftime("rev_%Y%m%d_%H%M%S")
    
    def get_latest_version(self) -> MyAnalysisVersion:
        """Get most recent analysis version."""
        if not self.latest_version:
            raise ValueError("No analysis versions available")
        return self.versions[self.latest_version]
    
    def get_version(self, version_name: str) -> MyAnalysisVersion:
        """Get specific analysis version by name."""
        if version_name not in self.versions:
            raise KeyError(f"Version not found: {version_name}")
        return self.versions[version_name]
    
    def list_versions(self) -> List[str]:
        """List all available version names in chronological order."""
        return sorted(self.versions.keys())
    
    def to_hdf5(self, filepath: str | Path) -> None:
        """
        Save all analysis versions to HDF5 file.
        
        Args:
            filepath: Path to HDF5 file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            # Store metadata
            f.attrs['analysis_type'] = 'my_analysis'
            f.attrs['flight_path'] = str(self.flight_path)
            f.attrs['latest_version'] = self.latest_version or ''
            
            # Store each version
            for version_name, version in self.versions.items():
                version_group = f.create_group(version_name)
                
                # Save DataFrame as dataset
                self._save_dataframe_to_hdf5(version_group, 'result_data', version.result_data)
                
                # Save metadata as attributes
                for key, value in version.metadata.items():
                    version_group.attrs[key] = str(value)
                
                version_group.attrs['revision_path'] = str(version.revision_path)
        
        logger.info(f"Saved {len(self.versions)} versions to {filepath}")
    
    @classmethod
    def from_hdf5(cls, filepath: str | Path) -> MyAnalysis:
        """
        Load analysis from HDF5 file.
        
        Args:
            filepath: Path to HDF5 file
        
        Returns:
            MyAnalysis instance with loaded versions
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {filepath}")
        
        with h5py.File(filepath, 'r') as f:
            flight_path = Path(f.attrs['flight_path'])
            analysis = cls(flight_path)
            
            # Load versions
            for version_name in f.keys():
                version_group = f[version_name]
                
                # Load DataFrame
                result_data = cls._load_dataframe_from_hdf5(version_group['result_data'])
                
                # Load metadata
                metadata = dict(version_group.attrs)
                revision_path = Path(metadata.pop('revision_path'))
                
                # Create version object
                version = MyAnalysisVersion(
                    version_name=version_name,
                    result_data=result_data,
                    metadata=metadata,
                    revision_path=revision_path
                )
                
                analysis.versions[version_name] = version
            
            analysis.latest_version = f.attrs.get('latest_version', None)
        
        logger.info(f"Loaded {len(analysis.versions)} versions from {filepath}")
        return analysis
    
    @staticmethod
    def _save_dataframe_to_hdf5(group: h5py.Group, name: str, df: pl.DataFrame) -> None:
        """Save Polars DataFrame to HDF5 group."""
        df_group = group.create_group(name)
        for col in df.columns:
            df_group.create_dataset(col, data=df[col].to_numpy())
        df_group.attrs['columns'] = df.columns
    
    @staticmethod
    def _load_dataframe_from_hdf5(group: h5py.Group) -> pl.DataFrame:
        """Load Polars DataFrame from HDF5 group."""
        columns = list(group.attrs['columns'])
        data = {col: group[col][:] for col in columns}
        return pl.DataFrame(data)
    
    def _load_from_hdf5(self, filepath: Path) -> None:
        """Load existing versions from HDF5 (internal use)."""
        loaded = self.from_hdf5(filepath)
        self.versions = loaded.versions
        self.latest_version = loaded.latest_version
```

---

## Registration and Integration

### Step 1: Create Module File

Create `pils/analyze/my_analysis.py` with the template above.

### Step 2: Export in __init__.py

Update `pils/analyze/__init__.py`:

```python
from pils.analyze.ppk import PPKAnalysis
from pils.analyze.my_analysis import MyAnalysis

__all__ = ['PPKAnalysis', 'MyAnalysis']
```

### Step 3: Usage Example

```python
from pils.analyze.my_analysis import MyAnalysis

# Initialize
analysis = MyAnalysis('/path/to/flight')

# Run with config
config = {
    'threshold': 0.5,
    'window_size': 10
}
version = analysis.run_analysis(config)

# Access results
print(version.result_data.head())

# Load later
analysis = MyAnalysis.from_hdf5('/path/to/flight/proc/my_analysis/my_analysis.h5')
latest = analysis.get_latest_version()
```

---

## Documentation Requirements

When adding a new analysis module or sensor, you **MUST** create documentation in **two places**:

### 1. API Documentation

Create a new file in `docs/api/analysis/` (or `docs/api/sensors/` for sensors) that **automatically extracts docstrings** from your Python code using mkdocstrings.

!!! info "Docstrings are Automatic"
    You **do NOT manually copy docstrings** into the API documentation. The `:::` syntax automatically extracts them from your Python source code. Just reference the module path.

**Example: `docs/api/analysis/my-analysis.md`**

````markdown
# MyAnalysis API

::: pils.analyze.my_analysis.MyAnalysis
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
      
::: pils.analyze.my_analysis.MyAnalysisVersion
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
````

The `:::` directive tells mkdocstrings to:
1. Find the Python class/function
2. Extract its docstring
3. Render it as formatted documentation
4. Optionally show source code (`show_source: true`)

**For Sensors: `docs/api/sensors/barometer.md`**

````markdown
# Barometer Sensor API

::: pils.sensors.barometer.Barometer
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
````

**Or for entire module:**

````markdown
# Barometer API

::: pils.sensors.barometer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
````

### 2. User Guide Documentation

Create a comprehensive user guide in `docs/user-guide/` explaining:

- What the analysis/sensor does
- When to use it
- How to use it with examples
- Input/output data formats
- Common use cases

**Example: `docs/user-guide/my-analysis.md`**

```markdown
# My Analysis

Description of what this analysis computes and why it's useful.

## When to Use

- Use case 1
- Use case 2

## Quick Start

\```python
from pils.analyze.my_analysis import MyAnalysis

analysis = MyAnalysis('/path/to/flight')
config = {'threshold': 0.5}
version = analysis.run_analysis(config)
print(version.result_data.head())
\```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.5` | Detection threshold |
| `window_size` | `int` | `10` | Rolling window size |

## Output Schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | `Int64` | ms | Time |
| `lat_smooth` | `Float64` | degrees | Smoothed latitude |

## Examples

### Basic Usage

\```python
# Example code
\```

## Next Steps

- [Related Guide](other-page.md)
```

### 3. Update Navigation

Add your new documentation to `mkdocs.yml`:

```yaml
nav:
  - API Reference:
    - Analysis:
      - api/analysis/index.md
      - PPK: api/analysis/ppk.md
      - My Analysis: api/analysis/my-analysis.md  # Add here
  - User Guide:
    - user-guide/index.md
    - My Analysis: user-guide/my-analysis.md  # Add here
```

---

## Complete Example: Velocity Analysis

### File: `pils/analyze/velocity.py`

```python
"""
Velocity Analysis - Compute velocity statistics from GPS data.

Analyzes GPS trajectories to compute instantaneous velocities,
accelerations, and motion statistics.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class VelocityVersion:
    """Container for velocity analysis results."""
    version_name: str
    velocity_data: pl.DataFrame
    statistics: Dict[str, float]
    metadata: Dict[str, Any]
    revision_path: Path


class VelocityAnalysis:
    """
    Velocity analysis with smart versioning.
    
    Computes velocities and accelerations from GPS position data.
    
    Example:
        >>> vel = VelocityAnalysis('/path/to/flight')
        >>> version = vel.run_analysis({'smoothing_window': 5})
        >>> print(version.statistics)
        {'max_speed': 15.3, 'max_accel': 2.1, ...}
    """
    
    def __init__(self, flight_path: str | Path) -> None:
        self.flight_path = Path(flight_path)
        self.proc_path = self.flight_path / "proc" / "velocity"
        self.proc_path.mkdir(parents=True, exist_ok=True)
        self.versions: Dict[str, VelocityVersion] = {}
        self.latest_version: Optional[str] = None
    
    def run_analysis(
        self,
        config: Dict[str, Any],
        force: bool = False
    ) -> VelocityVersion:
        """Run velocity analysis."""
        config_hash = self._compute_config_hash(config)
        
        if not force and self._config_exists(config_hash):
            return self.get_latest_version()
        
        version_name = datetime.now().strftime("rev_%Y%m%d_%H%M%S")
        revision_path = self.proc_path / version_name
        revision_path.mkdir(parents=True, exist_ok=True)
        
        # Load GPS data
        gps_path = self.flight_path / "aux" / "gps" / "gps.csv"
        gps = pl.read_csv(gps_path)
        
        # Compute velocities
        window = config.get('smoothing_window', 5)
        velocity_data = self._compute_velocities(gps, window)
        
        # Compute statistics
        statistics = {
            'max_speed': float(velocity_data['speed'].max()),
            'avg_speed': float(velocity_data['speed'].mean()),
            'max_accel': float(velocity_data['accel'].abs().max()),
        }
        
        # Save results
        velocity_data.write_parquet(revision_path / "velocity.parquet")
        
        version = VelocityVersion(
            version_name=version_name,
            velocity_data=velocity_data,
            statistics=statistics,
            metadata={'config_hash': config_hash, 'config': config},
            revision_path=revision_path
        )
        
        self.versions[version_name] = version
        self.latest_version = version_name
        self.to_hdf5(self.proc_path / "velocity.h5")
        
        return version
    
    def _compute_velocities(
        self,
        gps: pl.DataFrame,
        window: int
    ) -> pl.DataFrame:
        """Compute velocities from GPS positions."""
        # Compute differences
        result = gps.with_columns([
            pl.col('timestamp').diff().alias('dt'),
            pl.col('latitude').diff().alias('dlat'),
            pl.col('longitude').diff().alias('dlon'),
        ])
        
        # Convert to meters (approximate)
        R = 6371000  # Earth radius
        result = result.with_columns([
            (pl.col('dlat') * np.pi / 180 * R).alias('dy'),
            (pl.col('dlon') * np.pi / 180 * R * pl.col('latitude').cos()).alias('dx'),
        ])
        
        # Compute speed
        result = result.with_columns([
            ((pl.col('dx')**2 + pl.col('dy')**2).sqrt() / pl.col('dt')).alias('speed')
        ])
        
        # Smooth with rolling window
        result = result.with_columns([
            pl.col('speed').rolling_mean(window).alias('speed_smooth')
        ])
        
        # Compute acceleration
        result = result.with_columns([
            pl.col('speed_smooth').diff().alias('accel')
        ])
        
        return result
    
    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        import json
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    
    def _config_exists(self, config_hash: str) -> bool:
        return any(v.metadata.get('config_hash') == config_hash for v in self.versions.values())
    
    def get_latest_version(self) -> VelocityVersion:
        if not self.latest_version:
            raise ValueError("No versions available")
        return self.versions[self.latest_version]
    
    def to_hdf5(self, filepath: str | Path) -> None:
        """Save to HDF5."""
        # Implementation similar to template
        pass
    
    @classmethod
    def from_hdf5(cls, filepath: str | Path) -> VelocityAnalysis:
        """Load from HDF5."""
        # Implementation similar to template
        pass
```

---

## Best Practices

!!! tip "Keep Analysis Standalone"
    Analysis modules should not depend on Flight class. Pass paths, not Flight objects.

!!! tip "Version Everything"
    Always use auto-timestamp versioning (rev_YYYYMMDD_HHMMSS) to track when analysis was run.

!!! tip "Smart Re-run Logic"
    Hash configurations and skip execution if config unchanged (unless force=True).

!!! tip "Use Polars for Performance"
    Polars is 10-100x faster than Pandas for large datasets.

!!! tip "HDF5 for Storage"
    Store all versions in a single HDF5 file for efficient loading and organization.

!!! warning "Document Thoroughly"
    Create both API docs (docstrings) and user guide (examples) for every new module.

---

## Testing

Create tests in `tests/test_my_analysis.py`:

```python
import pytest
import polars as pl
from pathlib import Path
from pils.analyze.my_analysis import MyAnalysis

def test_analysis_execution(tmp_path):
    """Test analysis runs successfully."""
    # Create mock GPS data
    gps_dir = tmp_path / "aux" / "gps"
    gps_dir.mkdir(parents=True)
    
    gps_data = pl.DataFrame({
        'timestamp': range(100),
        'latitude': [40.0 + i * 0.0001 for i in range(100)],
        'longitude': [-74.0 + i * 0.0001 for i in range(100)],
    })
    gps_data.write_csv(gps_dir / "gps.csv")
    
    # Run analysis
    analysis = MyAnalysis(tmp_path)
    config = {'threshold': 0.5, 'window_size': 10}
    version = analysis.run_analysis(config)
    
    assert version.result_data.height == 100
    assert 'lat_smooth' in version.result_data.columns

def test_smart_rerun(tmp_path):
    """Test analysis skips unchanged config."""
    # Setup mock data (same as above)
    
    analysis = MyAnalysis(tmp_path)
    config = {'threshold': 0.5}
    
    # First run
    v1 = analysis.run_analysis(config)
    
    # Second run with same config - should skip
    v2 = analysis.run_analysis(config)
    
    assert v1.version_name == v2.version_name

def test_hdf5_persistence(tmp_path):
    """Test save/load to HDF5."""
    # Setup and run analysis
    
    analysis = MyAnalysis(tmp_path)
    config = {'threshold': 0.5}
    v1 = analysis.run_analysis(config)
    
    # Save to HDF5
    h5_path = tmp_path / "proc" / "my_analysis" / "my_analysis.h5"
    analysis.to_hdf5(h5_path)
    
    # Load from HDF5
    analysis2 = MyAnalysis.from_hdf5(h5_path)
    v2 = analysis2.get_latest_version()
    
    assert v1.version_name == v2.version_name
    assert v1.result_data.equals(v2.result_data)
```

---

## Next Steps

- [Testing Guide](testing.md) - Write comprehensive tests
- [Code Style](code-style.md) - Follow coding standards
- [Adding Sensors](adding-sensors.md) - Add new sensor types
- [Contributing](contributing.md) - Submit your module
