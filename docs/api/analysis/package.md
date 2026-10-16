# Flight Packaging

One command takes a flight from its raw files to the products in `proc/`
described in [Directory Structure](../../user-guide/directory-structure.md):

1. load the payload sensors (GPS, ADC, inclinometer, LM76, IMU; the ones not
   on the payload are skipped) and the drone data;
2. synchronize on the payload GPS (common time base, RTK drone data, the same
   rates as the power analysis);
3. save the raw and synchronized data in `proc/flight/` (`Flight.save`);
4. compute AZEL for each telescope and write the telescope's AZEL file;
5. optionally, the expected power on telescope detectors;
6. write `proc/package_manifest.json` with what was produced.

```bash
# Flights already in STOUT
pils package flight_20251208_1927 flight_20251208_2010

# A flight folder that is not in STOUT yet
pils package /mnt/data/POLOCALC/campaigns/202511/20251208/flight_20251208_1927

# Only some telescopes, plus the power analysis for a wafer slot
pils package flight_20251208_1927 -t SATP1 -t SATP3 --power satp1:ws0

# Check a flight without writing anything
pils package flight_20251208_1927 --dry-run

# Write somewhere else: <output-dir>/<flight>/proc/
pils package flight_20251208_1927 --output-dir ~/packages

# Convert results saved in HDF5 by earlier versions (the HDF5 files are kept)
pils migrate /mnt/data/POLOCALC/campaigns/202511/20251206/flight_20251206_1833/proc --dry-run
```

```python
from pils.analyze.package import FlightPackager

packager = FlightPackager.from_flight("flight_20251208_1927")
result = packager.run(telescopes=["SATP1", "SATP3"])
print(result.summary())
result.ok          # False if a step failed
packager.flight    # the loaded and synchronized Flight
```

## Output

```
flight_YYYYMMDD_hhmm/proc/
├── flight/
│   ├── raw/                        # loaded data, replaced at each run
│   └── sync/rev_YYYYMMDD_hhmmss/   # synchronized data
├── azel/<TELESCOPE>/rev_YYYYMMDD_hhmmss/
│   ├── manifest.json
│   ├── azel_data.parquet           # timestamp, az_deg, el_deg, srange_m, e/n/u_m
│   └── rev_YYYYMMDD_hhmmss_azel_<telescope>_<YYYYMMDD_hhmm>.csv
├── power/<TELESCOPE>_<wafer slot>/rev_*/   # only with --power
└── package_manifest.json
```

Every revision is a write-once directory with Parquet tables and a
`manifest.json` (see [Data Schema and Storage](../../user-guide/data-schema.md)).
Each run adds new revisions and keeps the earlier ones.

The AZEL file for the telescopes keeps its own header, `timestamp, az, el,
srange` (UNIX seconds, degrees, metres). The synchronized revision's manifest
records the packaging run (`package_version`), the sync settings, the time
offsets and the rates; each AZEL revision records the synchronized revision
it used (`sync_revision`). `package_manifest.json` records the code that ran
(pils version and git commit).

## Steps that are skipped or fail

Every step is reported as `done`, `skipped` or `failed`, and the command exits
with 1 if any flight has a failed step.

- A payload sensor without files is skipped. Without the payload GPS nothing
  else can run, because it is the synchronization reference.
- A telescope that is not in the campaign coordinates file
  (`campaign/metadata/*_coordinates.csv`) is skipped.
- A failed AZEL or power step does not stop the others.

## Inputs

- The DJI RTK base position comes from the telescopes config
  (`dji_broadcast_geod`, see [Power Analysis](power.md#telescopes-config)).
- The telescope and base positions come from the campaign coordinates file.
- The power analysis also needs the payload in STOUT and its lab results.

::: pils.analyze.package.FlightPackager

::: pils.analyze.package.PackageResult

::: pils.analyze.package.StepResult

::: pils.analyze.package.flight_info_from_path

::: pils.analyze.package.package_flight
