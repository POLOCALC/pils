# Photogrammetry Pipeline

This page documents how to run the photogrammetry pipeline end to end from
[`examples/photogrammetry_walkthrough.ipynb`](https://github.com/POLOCALC/pils/blob/main/examples/photogrammetry_walkthrough.ipynb),
the geodetic targets CSV it needs as input, the data folder layout it reads
from and writes to, and the column-naming/suffix conventions in its output.

For environment setup and import-path troubleshooting (conda env, making
`IPA_flight` importable, the `pils`/`IPA_flight` sibling-repo layout), see
`IPA_flight/documnetation/PHOTOGRAMMETRY_SETUP.md`. For the underlying math
(reference frames, pose conventions), see
`IPA_flight/documnetation/REFERENCE_FRAMES_AND_MATH.md`. This page focuses on
the practical side: what to put where, and what comes out.

## 1 — Running the pipeline

1. Activate the `photo` conda environment and launch the notebook with that
   kernel selected (`jupyter lab` / `jupyter notebook`, or VS Code with the
   `photo` interpreter picked for the notebook).
2. **Before doing anything else**, run the Step 0.1 "verify the interactive
   backend" cell near the top of the notebook. Steps 5 and 6 need to open a
   *clickable* matplotlib canvas (to mark telescopes/targets on the first
   frame). The notebook's import cell sets `%matplotlib qt`, which opens each
   such figure as a **real OS window** via PyQt5 -- independent of Jupyter's
   own rendering entirely, so none of the usual notebook-widget failure modes
   (blank ipywidgets output, VS Code CDN settings, browser vs. webview
   quirks) apply. It only needs:
   - a Qt binding in the kernel's environment (`pip install pyqt5` if
     missing -- `photo` already has it);
   - a display to draw to: this just works locally, and over SSH needs
     `ssh -X`/`-Y` (`echo $DISPLAY` should be non-empty) or a remote
     desktop/VNC session.

   On a fully headless machine with no display at all, switch that cell to
   `%matplotlib widget` instead -- that needs `ipympl` installed in the same
   environment as the kernel (`pip install ipympl`, then restart the kernel)
   *and* a frontend that renders ipywidgets (in VS Code: reload the window
   after installing, and check `Jupyter: Widget Script Sources` includes
   `jsdelivr.com`; `jupyter lab` in an actual browser tab is the most robust
   fallback if that stays unreliable). Prefer `%matplotlib qt` whenever a
   display is available -- it has one less moving part.

   If the smoke-test cell doesn't let you place clicks, fix that first --
   otherwise Step 5's target-picker will silently run with nothing recorded.
3. Fill in the **Step 0** cell: `DATA_ROOT`, `FLIGHT_NAME`, `CONFIG_PATH`,
   `TARGETS_CSV` (see §3 below), `OUTPUT_DIR`, `CAMERA_MODEL`.
4. Run top to bottom. Step 5 (`camera_obj.run_photogrammetry(...)`) is the
   full run and is interactive at step p2 (click telescopes, then targets,
   on the first frame). Step 6 lets you resume from any `dictionary_pN.ecsv`
   checkpoint instead of re-clicking / re-solving from scratch.
5. Section 7 reloads the merged result (`load_photogrammetry_result`) and
   plots the standard diagnostics: projection error, attitude over time,
   ENU ground track, and line-of-sight attitude.

For non-interactive, multi-flight batch runs (click once, reuse across the
rest of a campaign day via ORB alignment), see
`Camera.run_photogrammetry_multi_flights(...)`.

## 2 — Data folder layout

Raw flight data follows the general PILS campaign layout described in
[Directory Structure](directory-structure.md):

```
campaigns/YYYYMM/YYYYMMDD/flight_YYYYMMDD_hhmm/
├── aux/camera/YYYYMMDD_hhmmss_video.mp4   # or an image sequence for Alvium
├── aux/sensors/...                        # IMU/GPS/barometer raw logs
├── drone/YYYYMMDD_hhmmss_drone.csv        # RTK drone telemetry (lat/lon/alt)
└── proc/                                  # sync_data.h5, analyzed_data
```

`PathLoader`/`StoutLoader` resolve this layout into a `Flight` object; that
`Flight`, plus a `Camera` and a `PhotogrammetryConfig`, are the three inputs
`run_photogrammetry()` needs (see the notebook's Steps 1-3).

Pipeline **outputs** are separate from the raw campaign tree, under
whatever `OUTPUT_DIR` you pass in:

```
OUTPUT_DIR/
└── <flight_name>/
    ├── dictionary_p2.ecsv        # target pixel detections (p2)
    ├── dictionary_p3.ecsv        # PnP/MCMC camera attitude (p3)
    ├── dictionary_p4.ecsv        # drone GPS correlated to frames (p4)
    ├── dictionary_p5.ecsv        # GPS-corrected attitude (p5)
    ├── dictionary_p6.ecsv        # telescope-frame az/el/attitude (p6)
    ├── dictionary_p7.ecsv        # line-of-sight-frame attitude (p7)
    ├── attitude_reconstruction.parquet   # final merged result (§4)
    └── plots/                    # diagnostic PNGs (one per step, if check_results is set)
```

`<flight_name>` matches `flight.metadata["flight_name"]` (falls back to the
raw drone-data folder's parent directory name). Every `dictionary_pN.ecsv`
is a self-describing Astropy ECSV checkpoint -- reloadable on its own via
`load_dictionary()` (notebook §4.3) without re-running earlier steps.

## 3 — The geodetic targets CSV

`TARGETS_CSV` is an Emlid/RTK survey export, one row per surveyed point,
parsed by `IPA_flight.IPA_flight.genParamFile.GenParamFile.GeoPlot`. It must
have at least these columns (names are matched exactly, case-sensitive):

| Column | Meaning |
|---|---|
| `Name` | Point label -- matched case-*insensitively* against the patterns below |
| `Latitude` | WGS84 latitude, decimal degrees |
| `Longitude` | WGS84 longitude, decimal degrees |
| `Ellipsoidal height` | Height above the WGS84 ellipsoid, metres |

`Name` values are matched against these patterns (whitespace/case-insensitive;
repeated rows for the same point, e.g. several stick measurements, are
averaged):

| `Name` pattern | Role |
|---|---|
| `Dji rtk base (antenna head)` | Surveyed position of the DJI RTK base station |
| `Emlid base position measured with ...` | Self-measured base position (diagnostic / GPS-offset source) |
| `Target N`, `New_Target N` (e.g. `Target 1`, `Target 12`) | Photogrammetry ground targets |
| `SATp1 N`, `SATp2 N`, `SATp3 N` | SAT telescope survey points |
| `Lat N` | LAT telescope survey point |
| `Class1 N`, `Class2 N` | CLASS telescope survey points |

Example (abridged):

```csv
Name,Latitude,Longitude,Ellipsoidal height
Dji rtk base (antenna head),-22.9597732,-67.7866847,5175.412
Emlid base position measured with stick 1,-22.9597735,-67.7866850,5175.480
Target 1,-22.9601001,-67.7869992,5171.220
Target 2,-22.9601004,-67.7870210,5171.190
SATp1 1,-22.9598850,-67.7868001,5182.310
SATp2 1,-22.9598920,-67.7868150,5182.290
Lat 1,-22.9599010,-67.7868300,5182.350
Class1 1,-22.9599100,-67.7868420,5182.410
Class2 1,-22.9599150,-67.7868500,5182.400
```

`reference_point` (the campaign's fixed ENU origin) and `dji_base_logged`
(the RTK base position as logged by the *drone's own telemetry*, as opposed
to the value surveyed in this CSV) are **not** read from this CSV -- they
are campaign-level constants set once in
`pils/pils/config/photogrammetryConfig.yaml` under `pipeline:`.

> **Known caveat:** `photogrammetryConfig.yaml` explicitly notes
> `reference_point`'s altitude is "geodetic not ellipsoidal", while this
> CSV's `Ellipsoidal height` column is (as the name says) an ellipsoidal
> height -- these are different vertical datums, offset by the local geoid
> undulation. `GenParamFile.GeoPlot`'s current `gps_offset` computation
> differences `dji_base` (ellipsoidal, from this CSV) directly against
> `dji_base_logged` (geodetic, from the YAML) without correcting for that
> gap. An earlier version of the same function (still present, commented
> out, right above the active one) *did* carry an explicit correction term
> for this ("needed to handle ellipsoid 2 hmsl height"). If your GPS-vs-photogrammetry
> position comparison (§4, `drone_E/N/U` vs `tvec_E/N/U`) shows a
> suspiciously large and consistent vertical offset, this is the first place
> to check -- on one campaign, an offset computed by hand came out to
> ~0.5 m, while the pipeline's automatic `gps_offset` gave ~7 m for the same
> flight, consistent with a several-metre geoid-undulation-sized datum bug
> rather than a real position error.
>
> **Status: open.** The old dead code's correction can't be ported over
> mechanically -- it derives the local ellipsoidal↔geodetic gap by assuming
> `dji_base` (the CSV's "Dji rtk base (antenna head)" row) and
> `reference_point` refer to the same physical point, which may or may not
> hold for a given campaign's survey. Fixing this for real needs whoever set
> `reference_point`'s "geodetic" altitude by hand to say what they subtracted
> and from where, so the same conversion can be applied to `dji_base`'s
> altitude before it's differenced against `dji_base_logged`.

## 4 — Output columns and suffixes

`load_photogrammetry_result()` (notebook §4.2) reloads
`attitude_reconstruction.parquet`, the outer-join of all six
`dictionary_pN.ecsv` checkpoints on `frame` (and, for the telescope-frame
steps, `tel_name`). Each physical quantity appears under **exactly one**
column name -- a step that only carries an earlier column through unchanged
(e.g. p4 forwarding p3's `rvec_*`) does not duplicate it.

| Prefix / suffix | Meaning | Introduced by |
|---|---|---|
| `target_E/N/U` | Surveyed target position, ENU metres relative to `reference_point` | p2 |
| `rvec_x/y/z` | World→camera rotation (Rodrigues vector), raw PnP/MCMC solution | p3 |
| `tvec_x/y/z` | World→camera translation, **camera-frame** (`cv2.solvePnP` convention, not a position) | p3 |
| `quat_x/y/z/w` | Same rotation as `rvec_*`, quaternion form | p3 |
| `projection_error` | RMS reprojection error (pixels) for that frame's solution | p3 (p5 overwrites with the GPS-corrected value) |
| `tvec_E/N/U` | Camera position in ENU, converted from `tvec_*`/`rvec_*` -- diagnostic only, not propagated past p4 | p4 |
| `drone_E/N/U` | Drone GPS position in ENU (after `gps_offset` correction, §3) | p4 |
| `time` | Frame timestamp, aligned to the drone GPS clock | p4 |
| `*_corr` (`rvec_*_corr`, `quat_*_corr`) | Rotation re-solved with `tvec` fixed to the GPS position (`drone_E/N/U`) instead of the raw PnP translation | p5 |
| `tel_name` | Telescope identifier (`SATp1`, `SATp2`, `SATp3`, `CLASS1`, `CLASS2`, ...) -- one row per telescope per frame from here on | p6 |
| `az`, `el` | Drone azimuth/elevation as seen from that telescope | p6 |
| `yaw`, `pitch`, `roll` | Drone attitude expressed in the telescope frame | p6 |
| `*_LOS` (`yaw_LOS`, `pitch_LOS`, `roll_LOS`) | Drone attitude re-expressed in the telescope's line-of-sight frame -- the polarization-angle-ready output | p7 |

Two cardinality notes when working with the merged table:

- p2 has **one row per target per frame** (it's joined in first), so every
  later column is repeated across all targets of a frame.
- p6/p7 have **one row per telescope per frame**, joined on
  `["frame", "tel_name"]` -- so the final table has one row per
  `(frame, target_id, tel_name)` combination. Use `result.select([...]).unique()`
  or filter to a single `target_id`/`tel_name` when you only care about the
  per-frame trajectory (as the walkthrough's §7.3-7.5 plots do implicitly,
  since they select columns that don't vary by target/telescope and then
  `drop_nulls()`).

If you see a column with a `_right` suffix, that means a new pipeline step
was added (or an existing one changed) without accounting for a column it
shares with an earlier step -- see the "watch for duplicate columns" note in
`IPA_flight/documnetation/PHOTOGRAMMETRY_SETUP.md` §4, and extend the
`.drop([...])` list in the final join in
`Camera.run_photogrammetry()` (`pils/pils/sensors/camera.py`) accordingly,
rather than renaming the column on one side -- only give two columns
different names (e.g. a `_drone`/`_cam` suffix) when they represent
genuinely different quantities that happen to collide; if they're the same
quantity carried through unchanged, drop the duplicate instead.
