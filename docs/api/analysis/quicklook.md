# Quick Look

`pils quicklook` checks a flight's data and writes one Markdown report with
plots, without changing anything else in the flight folder:

```bash
# Flights in STOUT, or flight folders
pils quicklook flight_20251208_1927
pils quicklook /mnt/data/POLOCALC/campaigns/202504/20250428/flight_20250428_1625

# Also tune an RTKLIB configuration to the data (a few minutes)
pils quicklook flight_20251206_1833 --ppk optimized.conf

# Write <output-dir>/<flight>/proc/quicklook/ instead
pils quicklook flight_20251208_1927 --output-dir ~/quicklook
```

The report goes in a new revision (see [Storage](../utilities/storage.md)):

```
flight_YYYYMMDD_hhmm/proc/quicklook/rev_YYYYMMDD_hhmmss/
├── report.md            # the report
├── plots/*.png          # drone track and height, payload sensors, rover satellites
├── inventory.parquet    # the report's tables
├── sensors.parquet
├── sync_offsets.parquet
├── gnss_bands.parquet
├── ppk_tuning.parquet   # with --ppk
├── optimized.conf       # with --ppk, when a configuration passes validation
└── manifest.json
```

## What the report says

The report starts with a list of every warning, then one section per topic.
A section that cannot be done says why (a missing file, no RTKLIB, no
telescope coordinates).

| Section | Contents | Flagged |
|---|---|---|
| Flight | STOUT entry: campaign, take-off/landing, type, frequency, payload, notes | not in STOUT |
| Files | Files in `drone/`, `aux/` and the day's and campaign's `base/`, with sizes; loader messages | empty files; sensor recordings that stopped early; a Litchi log that does not cover the drone's airborne time (the log of another flight) |
| Drone | Airborne time (longest stretch above 1 m), height, extent, RTK solution types, battery | take-off more than 10 min from the STOUT time; RTK fixed below 95 % |
| Payload sensors | Per table: samples, rate, gaps, coverage of the airborne time (raw timestamps); plots | coverage below 95 %; no GPS or ADC |
| Synchronisation | Offsets of the drone GPS, Litchi, inclinometer and payload clock, with the method and correlation | correlation below 0.9; drone or Litchi clock off by more than 60 s; no payload clock offset |
| GNSS data for PPK | Rover and base (RTKLIB `convbin`), per system and band: satellites tracked and above 30/35/40 dB-Hz, SNR, loss of lock, continuous tracking; satellites in common, baseline; with `--ppk`, the tuned configuration | no base covering the flight; rover data that limits fixing whatever the configuration; no validated configuration |
| AZEL | Per telescope, elevation, azimuth and range while airborne (not saved) | |

The GNSS section needs RTKLIB `convbin` (and `rnx2rtkp` for `--ppk`) on the
`PATH` or in `$PILS_RTKLIB_BIN`. The base is looked for as in
[PPK](ppk.md#inputs): the day's `base/*.ubx`, then the campaign's `base/`.
With `--ppk`, the conversion and the tuning run in a temporary folder; the
PPK revisions in `proc/ppk/` are not touched.

```python
from pils.analyze.quicklook import QuickLook

quicklook = QuickLook.from_flight("flight_20251208_1927", output_dir="/tmp/ql")
report = quicklook.run(ppk_template="optimized.conf")
print(quicklook.warnings)
```

::: pils.analyze.quicklook.QuickLook

::: pils.analyze.quicklook.band_table

::: pils.analyze.quicklook.common_satellites

::: pils.analyze.quicklook.table_summary

::: pils.analyze.quicklook.airborne_interval
