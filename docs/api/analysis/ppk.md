# PPK Analysis

Each run is saved in `proc/ppk/rev_YYYYMMDD_hhmmss/` (see
[Storage](../utilities/storage.md)), next to the RTKLIB files: `position.parquet`
from the `.pos` solution and `statistics.parquet` from the `.pos.stat` file,
with the names of the [schema](../../user-guide/data-schema.md).

| Table | Columns |
|-------|---------|
| `position` | `datetime` (UTC), `timestamp` (UNIX s, UTC), `lat_deg`, `lon_deg`, `h_ell_m` (`h_msl_m` if RTKLIB wrote geodetic heights), `quality` (1 fix, 2 float, 3 sbas, 4 dgps, 5 single, 6 ppp), `num_sats`, `sd_n_m`, `sd_e_m`, `sd_u_m`, `age_s`, `ratio`, `e_m`, `n_m`, `u_m` (from the mean position) |
| `statistics` | `timestamp` (UNIX s, UTC), `gps_week`, `tow_s`, `satellite`, `frequency`, `az_deg`, `el_deg`, `resid_code_m`, `resid_phase_m`, `snr_dbhz`, `slip`, `lock`, `reject`, `constellation` |

RTKLIB writes GPS time by default: the time system of the `.pos` header (GPST,
UTC or JST) is converted to UTC with the leap seconds (18 s since 2017).
Versions in `ppk_solution.h5` from earlier versions are still loaded, with
their columns renamed (`pils.analyze.ppk.legacy_ppk_tables`); `pils migrate`
converts them.

## Inputs

- Rover: the payload GPS (`aux/sensors/*_GPS.bin` or `*_GPS.ubx`), converted
  with `convbin`.
- Base: the day's `base/*.ubx`, or a recording in the campaign's `base/`
  (`rtk_data_*`, `*/sensors_data/ZED-F9P_*.bin`), whichever covers the rover
  (10 minutes before and after). A cached `proc/ppk/base.obs` is converted
  again if it does not cover the rover.
- RTKLIB: `rnx2rtkp` and `convbin` from `PATH`, or from the folder in
  `PILS_RTKLIB_BIN`.

## Tuning the configuration

`run_analysis(template)` chooses the RTKLIB settings from the data (on by
default; `optimize=False` runs the template as it is) (`pils.analyze.ppk_tuning.tune`):

1. **Profile** the rover and base RINEX: interval, time span, satellites per
   system, SNR per system and band, base receiver position.
2. **Candidates** around the template:
   - a grid from the data profile: frequency (L1, plus L1+L2 only if the
     rover tracks L2 well), GPS with every combination of the systems both
     receivers see, SNR mask (template, from the rover's L1 SNR, off),
     elevation mask 10/15/20 deg, continuous or fix-and-hold, ratio 2 or 3
     (`candidates`);
   - 1000 random configurations over 18 settings (`random_candidates`,
     `N_RANDOM`; `n_random=0` for the grid only).
3. **Base position**: if the template's is more than 50 m from the base
   receiver's own, the average of the base's single solutions is used (with
   a warning: relative positions stay at centimetres, the absolute position
   is good to a few metres).
4. **Run** every candidate with `rnx2rtkp` in parallel.
5. **Choose** the candidate with the most epochs fixed by RTKLIB (Q=1), the
   default (`selection="fix_rate"`). Optional checks:
   `selection="reference"` keeps the candidates whose fixes agree with the
   drone's RTK track (good fixes within 0.10/0.15 m, at most 0.5 % of epochs
   off by more than 0.5 m; `score_against_reference`), and
   `selection="directions"` needs the forward and backward solutions to agree.
6. If no candidate fixes any epoch, the template is used.
7. **Explain**: from the random runs, how each setting acted on this flight
   (`setting_effects`, `tuning_effects` in the manifest; the quick look
   shows it).

The fix rate alone is not a safe criterion. On flight_20251206_1833:

| Configuration | Fixed | vs drone RTK (h / v RMS) | Fixes off > 1 m |
|---|---|---|---|
| Template (L1+L2, SNR mask 35) | 0.8 % | | |
| L1, GPS+Galileo, fix-and-hold, ratio 2 | 72 % | 0.18 / 0.10 m | 1.5 % |
| L1, GPS+Galileo+BeiDou, no SNR mask, 15 deg, continuous (chosen) | 67 % | 0.04 / 0.05 m | 0 % |

The search adapts to each flight: the grid follows the data profile, and the
random configurations explore the settings around it.

!!! warning "Fix rate and wrong fixes"
    The most Q=1 epochs is not always the most correct positions. On the 119
    November 2025 flights, scored against the drone RTK track, the template
    fixed 17 % of the epochs on average more than 0.5 m off. Choosing by fix
    rate keeps such configurations; `selection="reference"` excludes them
    when a reference track exists. With the reference selection the median
    of correct fixes was 9 % with the random search, 5 % with the grid alone
    and 1.6 % with the template.

The random search costs about a minute on 32 cores (`n_random` sets it).
A model of the settings' effects, trained on other days and tested on days
it had not seen, chose no better than random sampling, so the effects are
measured on each flight.

When the rover data itself limits fixing (L1 phase loss of lock on more than
10 % of observations, or continuous tracking shorter than 150 s), the run
says so (`tuning_data_quality` in the manifest): no configuration fixes such
data reliably. On the November 2025 campaign the best configuration differs
from flight to flight, and neither a single configuration nor a model trained
on other days finds it; the search on the flight's own data does.

The revision keeps the chosen configuration as `optimized.conf`, every
candidate in the `tuning` table (settings, fix rate; good and wrong fixes with
`selection="reference"`), and the data profiles, the choice and the effects in
its manifest.

```python
ppk = PPKAnalysis(flight)
version = ppk.run_analysis("template.conf", optimize=True)
version.tuning_data.sort("fix_pct", descending=True)
```

::: pils.analyze.ppk_tuning.tune

::: pils.analyze.ppk_tuning.candidates

::: pils.analyze.ppk_tuning.random_candidates

::: pils.analyze.ppk_tuning.N_RANDOM

::: pils.analyze.ppk_tuning.score_against_reference

::: pils.analyze.ppk_tuning.setting_effects

::: pils.analyze.ppk_tuning.profile_obs

::: pils.analyze.ppk_tuning.base_position_check

::: pils.analyze.ppk.legacy_ppk_tables

::: pils.analyze.ppk.PPKAnalysis

::: pils.analyze.ppk.PPKVersion

## RINEX Analysis

::: pils.analyze.ppkdata.RINEX.analyzer.RINEXAnalyzer
::: pils.analyze.ppkdata.RINEX.plotter.RINEXPlotter
::: pils.analyze.ppkdata.RINEX.report.RINEXReport

## PPK Analysis

::: pils.analyze.ppkdata.PPK.pos_analyzer.POSAnalyzer
::: pils.analyze.ppkdata.PPK.stat_analyzer.STATAnalyzer
::: pils.analyze.ppkdata.PPK.report.RTKLIBReport
::: pils.analyze.ppkdata.PPK.plotter.PPKPlotter
