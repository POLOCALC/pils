# Flight Packaging

One command takes a flight from its raw files to the products in `proc/`
described in [Directory Structure](../../user-guide/directory-structure.md):

1. load the payload sensors (GPS, ADC, inclinometer, LM76, IMU; the ones not
   on the payload are skipped) and the drone data, in any of the file
   formats of 2022-2025 (see
   [Raw files by campaign](../../user-guide/directory-structure.md#raw-files-by-campaign));
2. synchronize on the payload GPS, or on the drone GPS when there is no
   payload GPS covering the flight (common time base, RTK drone data, the
   same rates as the power analysis), and report every time offset;
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

# Delete old revisions, keeping the newest 3 of each analysis (and the
# synchronized revisions the kept ones were made from)
pils prune /mnt/data/POLOCALC/campaigns/202511/20251208/flight_20251208_1927/proc --keep 3 --dry-run
```

```python
from pils.analyze.package import FlightPackager

packager = FlightPackager.from_flight("flight_20251208_1927")
result = packager.run(telescopes=["SATP1", "SATP3"])
print(result.summary())
result.ok          # False if a step failed
packager.flight    # the loaded and synchronized Flight
```

A run prints one line per step:

```
flight_20251208_2005 (rev_20261017_181502)
  load gps           done     2037x21
  load adc           done     ...
  load drone         done     3851x27
  sync               done     drone 77221x23, litchi ..., reference_gps ...
  offsets            done     reference payload_gps: drone_gps +0.141 s (r=1.00),
                              litchi_gps -0.181 s (r=1.00), inclinometer +0.325 s (r=1.00),
                              payload_clock +0.144 s [inclinometer]
  save flight        done     ...
  azel SATP1         done     ...
```

`--dry-run` loads and synchronizes without writing; it is the quickest
check of a flight or of a whole campaign (a flight takes 5-60 s). On the
archive (275 flights, 2022-2025) it packs 274: flight_20251208_1404 has an
empty `drone/` folder and no payload data.

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

`pils quicklook` adds `proc/quicklook/rev_*/` (a Markdown report), PPK and
EKF their own folders; they are run separately.

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

| Step | Status | Meaning |
|------|--------|---------|
| `load <sensor>` | skipped | no file for the sensor, or a file without data (the reason is given) |
| `load drone` | failed | no drone log (e.g. an empty `drone/` folder) |
| `load` | done | with "synchronized to the drone GPS": no payload GPS |
| `load` | failed | no payload GPS and no drone GPS: nothing to synchronize on |
| `sync` | failed | the synchronization raised (the error is given) |
| `offsets` | done | the reference and every offset, with correlation `r` and method |
| `sync drone` | failed | no time offset between the drone and the payload GPS |
| `sync litchi` | skipped | the Litchi log does not match the drone track (another flight's log) |
| `sync inclinometer` | skipped | an inclinometer with its own clock (Kernel) and no Litchi pitch to align it |
| `azel <TELESCOPE>` | skipped | no campaign coordinates file, or the telescope is not in it |

- The payload GPS is the reference if it covers at least half of the drone
  GPS time; otherwise (missing, empty, stopped early, recorded at another
  time) the drone GPS is. The payload sensors then keep the payload
  computer's time. See
  [Time Synchronization](../../user-guide/synchronization.md#without-a-payload-gps).
- Payload sensors with empty files are left out of the sync instead of
  stopping it.
- Positions at latitude and longitude 0 (no solution) are not used.
- Offset methods: `reference` (the reference itself), `payload_clock` (an
  inclinometer on the payload clock without a Litchi pitch: its time is
  kept), `inclinometer`, `imu_gyro` and `none` for the payload clock (see
  [Payload clock](../../user-guide/synchronization.md#step-4-payload-clock)).
  A correlation below about 0.9 deserves a look (e.g.
  flight_20240430_2148, `r=0.57`: a payload GPS with many epochs without a
  fix).
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
