# Power Analysis

Expected source power on telescope detectors during a calibration flight:

1. payload diode reading (ADC envelope of the chopped signal, binned to 20 Hz);
2. diode input power from the lab responsivity at the payload temperature
   (LM76, else inclinometer, or a constant);
3. radiated power through the lab RF chain (coupler, attenuators, antenna
   gain, insertion and mismatch losses);
4. pointing loss of the source antenna towards the telescope (gimbal and
   inclinometer attitude, AZEL positions, measured antenna FWHMs);
5. free-space path and the detector beam (telescopes config).

The drone's thermal emission in the beam is computed alongside.

```python
from pils.analyze.power import PowerAnalysis

flight.add_sensor_data(["lm76", "adc", "inclinometer", "gps"])
flight.add_drone_data()
power = PowerAnalysis(flight)
version = power.run_analysis("satp1", "ws0", save_data=True, plot=True)
version.power_data   # timestamp, expected_det_power_pw, drone_power_pw, ...
```

```bash
pils power flight_20251201_1515 -t satp1 -w ws0
```

Results: one revision per run in `flight_dir/proc/power/SATP1_ws0/rev_*/`
(telescope and wafer slot), with `manifest.json`, `power_data.parquet` and
the plots (see [Data Schema and Storage](../../user-guide/data-schema.md)).

## Telescopes config

Detector beams, bandpasses, the DJI broadcast base and the scan altitude
threshold are in `pils/config/telescopes.py`. A YAML file with the same
structure replaces it (`PILS_TELESCOPES_CONFIG`), and the beam profile folder
can be set with `PILS_TELESCOPE_BEAMS_PATH`. `solid_angle_sr=` bypasses the
config.

::: pils.analyze.power.PowerAnalysis

::: pils.analyze.power.PowerVersion

::: pils.analyze.power.RFChain

::: pils.analyze.power.DiodeCalibration

::: pils.analyze.power.telescope.telescope_beam
