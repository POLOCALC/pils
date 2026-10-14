# Lab Analysis

Versioned analyses of component measurements made in the lab, and access to
the lab results of inventory items (used by the [power analysis](../analysis/power.md)).

| Module | Test types | Raw data |
|--------|------------|----------|
| [VNA](vna.md) | `attenuation`, `transmission`, `coupling_performance`, `directivity` | `<stem>_raw.s2p` |
| [Beam](beam.md) | `radiation_pattern` | `<stem>_raw.h5py` (H/E/D cuts) |
| [Multiplier / diode](multiplier-diode.md) | `output_power`, `responsivity` | `<stem>_raw.h5py` (climate chamber) |
| [Store](store.md) | any | reads results |

## Lab tree

```
<lab root>/<band>/<category>/<item name>/<test_type>/
├── 20251101_140000.md            # notes: YAML front matter (config) + description
├── 20251101_140000_raw.h5py      # raw data
└── proc/<analysis>/
    ├── <analysis>_solution.h5    # <stem>/rev_YYYYMMDD_HHMMSS/<tables> + metadata
    └── <stem>/rev_YYYYMMDD_HHMMSS/*.png
```

The lab root is `PILS_LAB_DATA_PATH`, else STOUT's data path + `lab`, else
`/mnt/data/POLOCALC/lab`. The notes front matter names the STOUT items used:

```yaml
---
config:
  components:
    multiplier: {item_id: ...}
    detector: {item_id: ...}
  auxiliary_components:
    directional_coupler: {item_id: ...}
    waveguides:
      STR: {item_id: ...}
---
```

Nothing is written until a run is saved, and `output_dir` sends the results
elsewhere (useful to try an analysis without touching the data share).

## Command line

```bash
pils lab vna /mnt/data/POLOCALC/lab/WR10/attenuator/ATT_002/attenuation
pils lab beam .../WR10/antenna/ANT_001/radiation_pattern --nominal-gain 15
pils lab mult-diode .../WR10/detector/DET_001/responsivity
# --dry-run: compute only; --output-dir DIR: save elsewhere; --stem: older measurement
```

## Python

```python
from pils.lab_analysis import BeamAnalysis, LabStore, VNAAnalysis

vna = VNAAnalysis.from_folder(".../ATT_002/attenuation")
version = vna.run_analysis(save_data=True, plot=True)
version["summary"]                 # frequency_ghz, s21_avg_db, s21_std_db, ...
vna.list_versions()                # ['rev_20261014_184801', ...]

store = LabStore()                 # items from STOUT
coupler = store.item("9a4ac3a3-...")
store.transmission(coupler, "coupling_performance", 93.0)  # (dB, 1σ)
```

## Notes

- Touchstone columns are S11, S21, S12, S22. CSVs written before PILS had
  lab analyses call the S21 column "S12"; the store relabels them, so the
  transmission values are unchanged.
- Antenna insertion loss and impedance matching are read from their
  processed files; there is no PILS analysis for them yet.
