"""The telemetry of a data folder, decoded as its YAML says."""

import pytest

from pils.drones import load_telemetry
from pils.drones.dji.m400.telemetry import M400Telemetry
from pils.drones.dji.m600.telemetry import M600Telemetry
from pils.drones.telemetry import read_config
from pils.gimbals import GremsyT7Telemetry

from .synthetic_lager import (
    broadcast_payload,
    gimbal_entry,
    gimbal_file,
    m400_file,
    m600_file,
    osdk_frame,
)

M600_YAML = "pointing_controller:\n  name: lager\n  gimbal:\n    name: Gremsy T7\n  uav:\n    name: DJI M600\n"
M400_YAML = "pointing_controller:\n  name: lager\n  gimbal:\n    name: Gremsy T7\n  uav:\n    name: DJI M400\n"


def folder(tmp_path, yaml_text, uav=None, gimbal=True):
    (tmp_path / "config.yaml").write_text(yaml_text)
    if uav is not None:
        (tmp_path / "uav_telemetry.bin").write_bytes(uav)
    if gimbal:
        (tmp_path / "gimbal_telemetry.bin").write_bytes(
            gimbal_file([[gimbal_entry(1.0, "RAW_IMU", xgyro=1)]])
        )
    return tmp_path


def test_a_matrice_600_folder_is_read_with_the_m600_decoder(tmp_path):
    folder(tmp_path, M600_YAML, uav=m600_file([(1.0, osdk_frame(broadcast_payload()))]))
    t = load_telemetry(tmp_path)
    assert isinstance(t.uav, M600Telemetry) and isinstance(t.gimbal, GremsyT7Telemetry)
    assert (
        t.uav_model == "DJI M600"
        and t.uav.data.height == 1
        and t.gimbal.data.height == 1
    )
    assert t.missing == []


def test_a_matrice_400_folder_is_read_with_the_m400_decoder(tmp_path):
    folder(tmp_path, M400_YAML, uav=m400_file())
    t = load_telemetry(tmp_path)
    assert isinstance(t.uav, M400Telemetry) and t.uav.data.height == 5


def test_blocks_at_the_top_level_and_other_name_spellings(tmp_path):
    folder(tmp_path, "uav:\n  name: dji-m400\n", uav=m400_file(), gimbal=False)
    t = load_telemetry(tmp_path)
    assert (
        isinstance(t.uav, M400Telemetry) and t.gimbal is None and t.gimbal_model is None
    )


def test_a_device_without_its_log_is_named_missing(tmp_path):
    folder(tmp_path, M600_YAML, uav=None)
    t = load_telemetry(tmp_path)
    assert (
        t.uav is None
        and t.missing == ["uav"]
        and t.gimbal is not None
        and t.uav_model == "DJI M600"
    )


def test_no_yaml_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="YAML"):
        load_telemetry(tmp_path)


def test_yaml_files_without_device_blocks_are_passed_over(tmp_path):
    (tmp_path / "a_other.yaml").write_text("something: else\n")
    (tmp_path / "b_broken.yml").write_text("a: [unclosed\n")
    folder(tmp_path, M600_YAML, uav=m600_file([(1.0, osdk_frame(broadcast_payload()))]))
    block, file = read_config(tmp_path)
    assert file.name == "config.yaml" and block["uav"]["name"] == "DJI M600"


def test_unknown_device_names_are_errors(tmp_path):
    folder(tmp_path, "uav:\n  name: DJI Mavic\n", gimbal=False)
    with pytest.raises(ValueError, match="No decoder for uav 'DJI Mavic'"):
        load_telemetry(tmp_path)
    (tmp_path / "config.yaml").write_text("uav:\n  simulator: false\n")
    with pytest.raises(ValueError, match="no 'name'"):
        load_telemetry(tmp_path)
