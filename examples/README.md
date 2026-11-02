# PILS Examples

- **`flight_data.ipynb`** — everything recorded for one flight and the results stored with it:
  drone (DJI `.DAT` + Litchi, or Matrice 400 `uav_telemetry.bin`), camera, payload GPS / IMU /
  IMX5 inclinometer, `flight.sync()`, and the stored photogrammetry (`PhotogrammetryAnalysis`)
  and EKF (`EKFAnalysis`) versions. Nothing is re-processed.

Running the photogrammetry pipeline itself is in IPA_flight (`IPA_flight/examples/`):
`photogrammetry_supervised.ipynb` for one flight step by step, `photogrammetry.py` for many flights.
