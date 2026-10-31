"""Tests for the expected-power (telescope) analysis."""

import numpy as np
import polars as pl
import pytest

from pils.analyze.power import DiodeCalibration, PowerAnalysis, RFChain
from pils.analyze.power.rf_chain import (
    detector_power,
    drone_emission_mw,
    free_space_loss_db,
    payload_item,
    payload_items,
)
from pils.analyze.power.signal import adc_envelope, bin_mean, scanning_range
from pils.analyze.power.telescope import (
    detector_gain_dbi,
    gaussian_solid_angle,
    load_config,
    profile_solid_angle,
    telescope_beam,
)
from pils.flight import Flight
from pils.lab_analysis.store import LabStore

FREQ = 93.0


class FakeStore(LabStore):
    """Lab results of a test payload, without files."""

    VNA = {
        ("COU_001", "transmission"): (-0.7, 0.01),
        ("COU_001", "coupling_performance"): (-20.0, 0.02),
        ("ATT_001", "attenuation"): (-10.0, 0.01),
    }

    def __init__(self):
        super().__init__(root="/nonexistent", get_item=lambda _: {})

    def transmission(self, item, test_type, frequency_ghz):
        return self.VNA[(item["name"], test_type)]

    def antenna_gain(self, item):
        return pl.DataFrame(
            {"frequency_ghz": [FREQ], "gain_dbi": [15.0], "gain_err_db": [0.01]}
        )

    def antenna_fwhm(self, item):
        return pl.DataFrame(
            {
                "plane": ["Hplane", "Eplane"],
                "frequency_ghz": [FREQ, FREQ],
                "fwhm_deg": [17.0, 15.0],
                "fwhm_err_deg": [0.1, 0.1],
            }
        )

    def insertion_loss(self, item, frequency_ghz):
        return (-0.1, 0.01)

    def reflection(self, item, frequency_ghz, dataset="", test_type=""):
        return (-20.0, 0.1)

    def responsivity(self, item):
        cv = [0.6, 0.7, 0.8, 0.9]
        rows = []
        for temp in [-10.0, 10.0]:
            for i, v in enumerate(cv):
                p_in = 10.0 ** (i - 4)  # 1e-4 .. 1e-1 mW
                rows.append(
                    {
                        "frequency_ghz": FREQ,
                        "temperature_c": temp,
                        "control_voltage_v": v,
                        "diode_output_mv": 1000 * p_in,  # 1 mV per µW
                        "diode_output_err_mv": 0.0,
                        "diode_input_mw": p_in,
                        "diode_input_err_mw": 0.0,
                    }
                )
        return pl.DataFrame(rows)


PAYLOAD = {
    "name": "PAY_TEST",
    "items": [
        {
            "category": "multiplier",
            "name": "MUL_001",
            "specifications": {"output_band": "WR10"},
        },
        {"category": "coupler", "name": "COU_001", "specifications": {}},
        {"category": "attenuator", "name": "ATT_001", "specifications": {}},
        {"category": "antenna", "name": "ANT_001", "specifications": {}},
        {"category": "detector", "name": "DET_001", "specifications": {}},
    ],
}


class TestTelescope:
    def test_gaussian_solid_angle(self):
        fwhm = np.radians(1.0)
        assert gaussian_solid_angle(fwhm) == pytest.approx(1.1331 * fwhm**2, rel=1e-4)
        assert detector_gain_dbi(4 * np.pi / 100) == pytest.approx(20.0)

    def test_profile_solid_angle_of_gaussian(self):
        sigma = np.radians(0.2)
        theta = np.linspace(0, 10 * sigma, 5001)
        power = np.exp(-(theta**2) / (2 * sigma**2))
        assert profile_solid_angle(theta, power) == pytest.approx(
            2 * np.pi * sigma**2, rel=1e-3
        )

    def test_config_resolution(self, tmp_path, monkeypatch):
        config = load_config()
        wr06 = telescope_beam("SATP1", "ws0", "WR06", 150.0, config)
        assert wr06.method == "gaussian" and wr06.fwhm_arcmin == 12.1
        ws5 = telescope_beam("satp3", "ws5", "WR10", 93.0, config)
        assert ws5.fwhm_arcmin == 28.6

        monkeypatch.setenv("PILS_TELESCOPE_BEAMS_PATH", str(tmp_path))
        (tmp_path / "satp1").mkdir()
        (tmp_path / "satp1" / "fwhms_satp1_ws0.csv").write_text(
            "freq,fwhm\n90,30\n93,28\n96,26\n"
        )
        averaged = telescope_beam("satp1", "ws0", "WR10", 90.0, config)
        assert averaged.fwhm_arcmin == pytest.approx(28.0)
        assert "band-average" in averaged.method

        theta_arcmin = np.linspace(0, 120, 2001)
        np.savetxt(
            tmp_path
            / "satp1"
            / "coadded_invw_fitted_full_unsmoothed_prof_T_satp1_93_ws0_mid.txt",
            np.column_stack([theta_arcmin, np.exp(-(theta_arcmin**2) / (2 * 12.0**2))]),
        )
        profile = telescope_beam("satp1", "ws0", "WR10", 93.0, config)
        assert profile.method == "measured profile"
        assert profile.fwhm_arcmin == 28.0
        assert profile.solid_angle_sr == pytest.approx(
            2 * np.pi * np.radians(12.0 / 60) ** 2, rel=1e-3
        )
        with pytest.raises(FileNotFoundError, match="PILS_TELESCOPE_BEAMS_PATH"):
            telescope_beam("satp1", "ws9", "WR10", 93.0, config)
        with pytest.raises(ValueError, match="Unknown telescope"):
            telescope_beam("lat", "ws0", "WR10", 93.0, config)


class TestRFChain:
    def test_payload_items(self):
        assert payload_item(PAYLOAD, "coupler")["name"] == "COU_001"
        with pytest.raises(ValueError, match="No grid"):
            payload_items(PAYLOAD, "grid")

    def test_source_power(self):
        rf = RFChain.from_payload(PAYLOAD, FREQ, FakeStore())
        mismatch = 10 * np.log10(1 - 10 ** (-20.0 / 10))
        assert rf.coupled_to_output_db == pytest.approx(19.3)
        assert rf.coupler_to_radiated_db == pytest.approx(-10.0 + 15.0 - 0.1 + mismatch)
        dbm, err = rf.source_power(1e-3, 1e-4)
        assert dbm == pytest.approx(-30.0 + 19.3 + rf.coupler_to_radiated_db)
        assert err == pytest.approx(
            1e-4 * 10 ** ((19.3 + rf.coupler_to_radiated_db) / 10)
        )

    def test_path_and_drone(self):
        wavelength = 299792458.0 / 93e9
        expected = 20 * np.log10(wavelength / (4 * np.pi * 500))
        assert free_space_loss_db(93.0, 500.0) == pytest.approx(expected)
        dbm, _ = detector_power(0.0, 0.0, 40.0, 93.0, 500.0, duty_cycle=1.0)
        assert dbm == pytest.approx(40.0 + expected)
        mw = drone_emission_mw(500.0, 1e-4, 30.0)
        dilution = np.pi * (0.5 / 500 / 2) ** 2 / 1e-4
        assert mw == pytest.approx(1.380649e-20 * 300 * 30e9 * dilution)


class TestSignal:
    def test_scanning_range(self):
        t = np.arange(0, 100, 0.01)
        alt = 100 + 5 * np.sin(2 * np.pi * (t - 0) / 20)
        start, end = scanning_range(t, alt, min_altitude_m=0)
        assert start == pytest.approx(5 - 10, abs=0.05)
        assert end == pytest.approx(85, abs=0.05)
        with pytest.raises(ValueError, match="no scan"):
            scanning_range(t, alt, min_altitude_m=1000)

    def test_envelope_and_binning(self):
        square = np.tile(np.r_[np.zeros(200), np.full(200, 50.0)], 4)
        _, envelope, zero = adc_envelope(square, window=400)
        assert np.all(envelope == 50.0) and np.all(zero == 0.0)
        t, mean, err = bin_mean(np.arange(10.0), np.arange(10.0), 4)
        assert list(t) == [1.5, 5.5] and list(mean) == [1.5, 5.5]
        assert err[0] == pytest.approx(np.std([0, 1, 2, 3], ddof=1) / 2)

    def test_diode_calibration(self):
        calibration = DiodeCalibration(FakeStore().responsivity(None), FREQ)
        assert list(calibration.temperature_c) == [-10.0, 10.0]
        power, err = calibration.input_power([50.0, 5.0], [0.0, 1.0], [0.0, 25.0])
        assert power[0] == pytest.approx(0.05)
        assert power[1] == pytest.approx(0.005)
        assert err[1] == pytest.approx(1e-3)
        with pytest.raises(ValueError, match="No diode responsivity"):
            DiodeCalibration(FakeStore().responsivity(None), 84.0)


def synthetic_flight(tmp_path, voltage_mv=50.0):
    """Drone hovering 500 m from the telescope, pointing at it."""
    folder = tmp_path / "flight_test"
    (folder / "drone").mkdir(parents=True)
    flight = Flight(
        {
            "drone_data_folder_path": str(folder / "drone"),
            "aux_data_folder_path": str(folder / "aux"),
            "flight_name": "flight_test",
            "flight_info": {"frequency": FREQ},
        }
    )
    t = np.arange(0, 100, 0.01)
    n = len(t)
    flight.sync_data = {
        "drone": pl.DataFrame(
            {
                "timestamp": t,
                "lat_deg": np.full(n, -22.96),
                "lon_deg": np.full(n, -67.79),
                "h_ell_m": 100 + 5 * np.sin(2 * np.pi * t / 20),
            }
        ),
        "litchi": pl.DataFrame(
            {
                "timestamp": t,
                "yaw_deg": np.full(n, 180.0),
                "gimbal_yaw_deg": np.zeros(n),
                "gimbal_pitch_deg": np.full(n, -np.degrees(np.arctan2(300, 400))),
            }
        ),
        "inclinometer": pl.DataFrame({"timestamp": t, "roll_deg": np.zeros(n)}),
        "lm76": pl.DataFrame({"timestamp": t, "temperature_c": np.zeros(n)}),
    }
    t_adc = np.arange(0, 100, 1 / 1600)
    # 37 Hz chopping: ~22 samples per half period, many per envelope window
    square = np.where((np.arange(len(t_adc)) // 22) % 2, voltage_mv, 0.0)
    flight.raw_data.payload_data.adc = pl.DataFrame(
        {"timestamp": t_adc, "amplitude_mv": square}
    )
    return flight


def test_power_analysis_end_to_end(tmp_path, monkeypatch):
    flight = synthetic_flight(tmp_path)
    azel = pl.DataFrame(
        {
            "timestamp": np.arange(0, 100, 0.01),
            "e_m": np.zeros(10000),
            "n_m": np.full(10000, 400.0),
            "u_m": np.full(10000, 300.0),
        }
    )
    monkeypatch.setattr(
        PowerAnalysis, "_azel", lambda self, *a, **k: (azel, "rev_test")
    )
    config = load_config() | {"scan_min_altitude_m": 0.0}
    analysis = PowerAnalysis(flight, output_dir=tmp_path / "out")
    version = analysis.run_analysis(
        "satp1",
        "ws0",
        payload=PAYLOAD,
        store=FakeStore(),
        solid_angle_sr=1e-4,
        config=config,
        save_data=True,
        plot=True,
    )
    data = version.power_data
    assert data["distance_m"].to_numpy() == pytest.approx(500.0)
    assert np.abs(data["pointing_loss_db"].to_numpy()).max() < 1e-6
    assert data["diode_input_mw"].to_numpy() == pytest.approx(0.05)

    rf = RFChain.from_payload(PAYLOAD, FREQ, FakeStore())
    source = 10 * np.log10(0.05) + rf.coupled_to_output_db + rf.coupler_to_radiated_db
    expected_dbm = (
        source
        + detector_gain_dbi(1e-4)
        + free_space_loss_db(FREQ, 500.0)
        + 10 * np.log10(2 / np.pi)
    )
    assert data["expected_det_power_pw"].to_numpy() == pytest.approx(
        10 ** (expected_dbm / 10) * 1e9
    )
    assert version.metadata["temperature_source"] == "lm76"
    assert version.metadata["temperature_outside_calibration_fraction"] == 0.0
    # The lab results as of the take-off (here from the folder name), listed
    assert "lab_as_of" in version.metadata
    assert version.metadata["lab_inputs"] == []

    assert analysis.list_versions("satp1", "ws0") == [version.version_name]
    loaded = analysis.get_latest_version("SATP1", "ws0")
    assert loaded.power_data.equals(data)
    assert loaded.metadata["rf_chain"]["coupler"] == "COU_001"
    # One folder per run: the saved tables (pils.store) and the plots
    folder = tmp_path / "out" / "proc" / "power" / "SATP1_ws0" / version.version_name
    assert sorted(p.name for p in folder.iterdir()) == [
        "adc.png",
        "manifest.json",
        "pointing.png",
        "power.png",
        "power_data.parquet",
    ]
    from pils import store

    assert store.read_manifest(folder)["kind"] == "power"
    assert not (tmp_path / "flight_test" / "proc").exists()


def test_power_analysis_needs_frequency(tmp_path):
    flight = synthetic_flight(tmp_path)
    flight.flight_info["flight_info"] = {"frequency": 0}
    with pytest.raises(ValueError, match="no source frequency"):
        PowerAnalysis(flight).run_analysis(
            "satp1", "ws0", payload=PAYLOAD, store=FakeStore()
        )
    with pytest.raises(TypeError):
        PowerAnalysis("not a flight")
