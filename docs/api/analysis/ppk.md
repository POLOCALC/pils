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
