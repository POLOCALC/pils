# Storage

Every product of a flight (raw and synchronized data, AZEL, power, EKF, PPK)
is a write-once revision directory with Parquet tables and a
`manifest.json` (`pils.store`); the column names and units follow
`pils.schema`. See [Data Schema and Storage](../../user-guide/data-schema.md)
and [Directory Structure](../../user-guide/directory-structure.md).

```python
from pils import store

store.list_versions("proc/azel", parent="SATP1")
tables, metadata = store.load_version("proc/azel", "rev_20261016_152031", parent="SATP1")
store.read_units("proc/azel/SATP1/rev_20261016_152031/azel_data.parquet")
# {"timestamp": "s", "az_deg": "deg", ...}
```

The flight itself is saved and loaded with `Flight.save(directory)` and
`Flight.load(directory)`.

## Revisions

::: pils.store.save_version

::: pils.store.load_version

::: pils.store.list_versions

::: pils.store.read_manifest

::: pils.store.write_tables

::: pils.store.read_tables

::: pils.store.read_units

## Schema

::: pils.schema.unit_of

::: pils.schema.canonical

::: pils.schema.snake_case

::: pils.schema.record_name

## Results saved in HDF5

PILS reads and writes no HDF5 results: every product is parquet. Earlier
versions saved the analyses in `*_solution.h5`; `pils migrate <proc>`
converts them, once, to revision directories (for a flight's `proc/` or a
lab item's `proc/`) and keeps the HDF5 files. It is the only code that reads
them (`pils.legacy_hdf5`), and goes once the old files are converted.
Flights in the single-HDF5 format (`sync_data.h5`) are not converted any
more. HDF5 remains only as an input: the lab's raw `.h5py` measurements.

::: pils.migrate.migrate_proc

::: pils.migrate.migrate_lab

## Old revisions

Revisions are never deleted by a run. `pils prune <proc> --keep N` deletes
all but the newest N of each folder of revisions; a synchronized revision
that a kept analysis revision was made from is kept.

::: pils.store.prune_versions

::: pils.legacy_hdf5.read_table
