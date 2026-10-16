# HDF5 Tables and Units

The flight file (`sync_data.h5`) and the analysis results (AZEL, EKF, PPK)
store every table with the same writer; see
[Directory Structure](../../user-guide/directory-structure.md) for the layout.

```python
import h5py
from pils.utils.hdf5 import read_table, read_units

with h5py.File("proc/sync_data.h5") as f:
    group = f["raw_data/drone_data/drone"]
    drone = read_table(group)
    units = read_units(group)  # {"timestamp": "s", "RTK:lat_p": "deg", ...}
```

::: pils.utils.hdf5.write_table

::: pils.utils.hdf5.read_table

::: pils.utils.hdf5.read_units

::: pils.utils.column_units.column_unit

::: pils.utils.column_units.table_units
