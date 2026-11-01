"""Failures that used to be swallowed silently are now logged (or narrowed)."""

import logging
from pathlib import Path

import pytest

from pils.analyze.ppkdata.RINEX.analyzer import RINEXAnalyzer
from pils.flight import Flight, _deserialize_from_hdf5
from pils.sensors.adc import ADC
from pils.sensors.LM76 import LM76


def flight_in(tmp_path: Path, **info) -> Flight:
    drone = tmp_path / "drone"
    drone.mkdir()
    return Flight(
        {
            "drone_data_folder_path": str(drone),
            "aux_data_folder_path": str(tmp_path / "aux"),
            **info,
        }
    )


def test_drone_model_lookup_failure_is_logged(tmp_path, monkeypatch, caplog):
    import pils.integrations.stout as stout_db

    def unavailable(drone_id):
        raise RuntimeError("database down")

    monkeypatch.setattr(stout_db, "drone_model", unavailable)
    flight = flight_in(tmp_path, drone_id="drone-1")
    (tmp_path / "drone" / "x_blacksquare.log").write_text("")
    with caplog.at_level(logging.WARNING, logger="pils"):
        model = flight._detect_drone_model(str(tmp_path / "drone"))
    assert model == "blacksquare"  # the file names are still the fallback
    assert "database down" in caplog.text


def test_deserialize_keeps_plain_strings():
    assert _deserialize_from_hdf5("not json", hint="dict") == "not json"
    assert _deserialize_from_hdf5('{"a": 1}', hint="dict") == {"a": 1}


def test_unreadable_adc_config_logs_the_default_gain(tmp_path, caplog):
    (tmp_path / "payload_config.yml").write_text("sensors: [unclosed")
    adc = ADC.__new__(ADC)
    adc.data_path = str(tmp_path / "x_ADC.bin")
    with caplog.at_level(logging.WARNING, logger="pils"):
        assert adc._read_gain_from_config() == 16
    assert "default gain 16" in caplog.text


def test_unreadable_lm76_config_is_logged(tmp_path, caplog):
    (tmp_path / "payload_config.yml").write_text("")  # empty: None, not a mapping
    lm76 = LM76.__new__(LM76)
    lm76.path = str(tmp_path / "x_TMP.csv")
    with caplog.at_level(logging.WARNING, logger="pils"):
        assert lm76._read_settings_from_config() is None
    assert "LM76 settings" in caplog.text


def test_malformed_nav_records_are_counted(tmp_path, caplog):
    nav = tmp_path / "base.nav"
    nav.write_text(
        "     3.04           N: GNSS NAV DATA    M: Mixed            RINEX VERSION / TYPE\n"
        "                                                            END OF HEADER\n"
        "G01 2025 12 08 19 00 00 garbage\n" + "     not a number\n" * 7
    )
    analyzer = RINEXAnalyzer(tmp_path / "base.obs", navpath=nav)
    with caplog.at_level(logging.WARNING, logger="pils"):
        analyzer.parse_nav_file()
    assert analyzer.nav_data == {}
    assert "1 malformed NAV records skipped" in caplog.text


def test_detect_drone_model_unexpected_error_propagates(tmp_path, monkeypatch):
    import pils.flight.core as core

    def broken(dirpath, keyword):
        raise KeyError(keyword)  # a bug, not an unreadable folder

    monkeypatch.setattr(core, "get_path_from_keyword", broken)
    flight = flight_in(tmp_path)
    with pytest.raises(KeyError):
        flight._detect_drone_model(str(tmp_path / "drone"))
