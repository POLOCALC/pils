"""
Telescope and site parameters for the expected-power analysis.

Per telescope and source band, the detector beam is one of:

- ``fwhm_arcmin``: a Gaussian beam of that FWHM;
- ``profile``: a measured radial beam profile (angle, power) integrated
  over the sky, one file per frequency and wafer slot; the frequencies in
  ``band_average_frequencies_ghz`` have no profile and use a Gaussian with
  the mean FWHM of ``fwhm_table`` instead.

``wafer_slots`` overrides the band entry for one wafer slot. File names are
relative to the beam folder (``PILS_TELESCOPE_BEAMS_PATH`` or ``beam_dir``)
and can use ``{telescope}``, ``{wafer_slot}`` and ``{freq}`` (integer GHz).

A YAML file with the same structure can replace these values
(``PILS_TELESCOPES_CONFIG`` or the ``config`` argument of the analysis).
"""

TELESCOPES_CONFIG = {
    "beam_dir": "/home/jupyter-ludovico/optical efficiency/beams",
    # Telescope bandpass per detector setting (GHz)
    "bandpass_ghz": {"gab": 30.0, "SO": 26.9},
    # Site: DJI RTK base position broadcast during the 2025 campaigns
    "dji_broadcast_geod": {"lat": -22.9597732, "lon": -67.7866847, "alt": 5173.020},
    # The drone scans above this altitude (m, ellipsoidal)
    "scan_min_altitude_m": 5400.0,
    "telescopes": {
        "satp1": {
            "WR10": {
                "profile": "satp1/coadded_invw_fitted_full_unsmoothed_prof_T_satp1_{freq}_{wafer_slot}_mid.txt",
                "profile_angle_unit": "arcmin",
                "fwhm_table": "satp1/fwhms_satp1_{wafer_slot}.csv",
                "band_average_frequencies_ghz": [90, 96],
            },
            "WR06": {"fwhm_arcmin": 12.1},
        },
        "satp3": {
            "WR10": {
                "profile": "satp3/coadded_unsmoothed_prof_T_satp3_{freq}_{wafer_slot}_mid.txt",
                "profile_angle_unit": "rad",
                "fwhm_table": "satp3/fwhms_satp3_{wafer_slot}.csv",
                "band_average_frequencies_ghz": [90, 96],
                # Average beam from the SO MF SAT commissioning paper
                "wafer_slots": {"ws5": {"fwhm_arcmin": 28.6}},
            },
            "WR06": {"fwhm_arcmin": 12.7},
        },
    },
}
