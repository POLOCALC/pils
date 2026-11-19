"""Flights with LAGER telemetry, the gimbal log and the Matrice 400 flight record."""

import polars as pl
import pytest

from pils.drones import sources
from pils.flight import Flight

from .synthetic_lager import (
    broadcast_payload,
    gimbal_entry,
    gimbal_file,
    m400_file,
    m600_file,
    osdk_frame,
)

M400_YAML = "pointing_controller:\n  name: lager\n  gimbal:\n    name: Gremsy T7\n  uav:\n    name: DJI M400\n"
M600_YAML = M400_YAML.replace("M400", "M600")


def lager_folder(folder, yaml_text=M400_YAML, uav=None):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.yaml").write_text(yaml_text)
    (folder / "uav_telemetry.bin").write_bytes(uav if uav is not None else m400_file())
    (folder / "gimbal_telemetry.bin").write_bytes(
        gimbal_file(
            [
                [
                    gimbal_entry(
                        10.0,
                        "MOUNT_ORIENTATION",
                        roll=0.5,
                        pitch=-30.0,
                        yaw=120.0,
                        yaw_absolute=121.0,
                    ),
                    gimbal_entry(10.1, "RAW_IMU", xgyro=1, ygyro=2, zgyro=3),
                ]
            ]
        )
    )


def flight_of(tmp_path, **folders):
    info = {
        "drone_data_folder_path": str(tmp_path / "drone"),
        "aux_data_folder_path": str(tmp_path / "aux"),
    }
    for name in ("drone", "aux"):
        (tmp_path / name).mkdir(exist_ok=True)
    return Flight(info)


class FakeRecord:
    """M400FlightRecord without DJI's API."""

    keys = []

    def __init__(self, path):
        self.path = path
        self.details = {"version": 13, "aircraft_name": "M400"}
        self.data = None

    def load_data(self, api_key=None, use_cache=True):
        FakeRecord.keys.append(api_key)
        self.data = pl.DataFrame(
            {
                "timestamp": [1.0, 1.1],
                "osd_latitude": [45.5, 45.5001],
                "osd_longitude": [9.2, 9.2],
                "osd_height": [10.0, 11.0],
                "osd_altitude": [250.0, 251.0],
                "osd_yaw": [90.0, 91.0],
            }
        )


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    monkeypatch.delenv("DJI_API_KEY", raising=False)
    FakeRecord.keys = []
    monkeypatch.setattr(sources, "M400FlightRecord", FakeRecord)


def test_a_matrice_400_flight_has_telemetry_and_gimbal_without_a_dat(tmp_path):
    flight = flight_of(tmp_path)
    lager_folder(tmp_path / "aux")  # the payload computer's folder
    flight.add_drone_data()
    d = flight.raw_data.drone_data
    assert flight.raw_data.drone_data.has_data()
    assert not isinstance(d.drone, pl.DataFrame) or d.drone.is_empty()
    t = d.telemetry
    assert t is not None and t.height == 5
    # the schema's names
    assert {
        "timestamp",
        "fused_lat_deg",
        "fused_lon_deg",
        "fused_h_m",
        "gps_lat_deg",
        "gps_h_msl_m",
        "yaw_deg",
        "roll_deg",
    } <= set(t.columns)
    assert set(d.gimbal) == {"mount_orientation", "raw_imu"}
    assert d.gimbal["mount_orientation"]["pitch_deg"].to_list() == [-30.0]
    assert d.info["uav_model"] == "DJI M400" and d.info["gimbal_model"] == "Gremsy T7"
    assert d.info["uav_stats"]["unknown_topic"] == 1


def test_lager_in_the_drone_folder_or_a_subfolder_is_found(tmp_path):
    flight = flight_of(tmp_path)
    lager_folder(tmp_path / "drone" / "lager")
    flight.add_drone_data()
    assert flight.raw_data.drone_data.telemetry is not None


def test_a_matrice_600_flight_keeps_its_dat_litchi_and_the_telemetry(
    tmp_path, monkeypatch
):
    import pils.flight.core as core

    class Fake:
        def __init__(self, path):
            self.data = pl.DataFrame({"timestamp": [0.0, 1.0]})
            self.records = {}

        def load_data(self, **kwargs):
            pass

    monkeypatch.setattr(core, "DJIDat", type("DJIDat", (Fake,), {}))
    monkeypatch.setattr(core, "Litchi", type("Litchi", (Fake,), {}))
    flight = flight_of(tmp_path)
    (tmp_path / "drone" / "x_drone.csv").write_text("")
    (tmp_path / "drone" / "x_litchi.csv").write_text("")
    lager_folder(
        tmp_path / "aux",
        M600_YAML,
        uav=m600_file([(1.0, osdk_frame(broadcast_payload()))]),
    )
    flight.add_drone_data()
    d = flight.raw_data.drone_data
    assert d.drone.height == 2 and d.litchi.height == 2
    assert d.telemetry["fused_lat_deg"].to_list() == pytest.approx([45.5])
    assert d.telemetry["rtk_h_ell_m"].to_list() == pytest.approx([211.0])
    assert d.info["uav_model"] == "DJI M600"


def test_the_flight_record_needs_a_key_and_is_left_out_without_one(tmp_path):
    flight = flight_of(tmp_path)
    lager_folder(tmp_path / "aux")
    (tmp_path / "drone" / "DJIFlightRecord_2026-11-19_[10-15-00].txt").write_bytes(b"x")
    flight.add_drone_data()
    d = flight.raw_data.drone_data
    assert (
        d.flight_record is None and "DJI_API_KEY" in d.info["errors"]["flight_record"]
    )
    assert d.telemetry is not None  # the rest of the flight is still there


def test_the_flight_record_is_decoded_with_the_key(tmp_path):
    flight = flight_of(tmp_path)
    (tmp_path / "drone" / "DJIFlightRecord_2026-11-19_[10-15-00].txt").write_bytes(b"x")
    (tmp_path / "drone" / "DJIFlightRecord_2026-11-19_[11-00-00].txt").write_bytes(b"x")
    flight.add_drone_data(dji_api_key="KEY")
    d = flight.raw_data.drone_data
    assert FakeRecord.keys == ["KEY", "KEY"]
    assert d.flight_record["lat_deg"].to_list() == [45.5, 45.5001]
    assert d.flight_record["h_rel_m"].to_list() == [10.0, 11.0]
    assert "flight_record_2" in d.sources
    assert d.info["flight_record_details"]["flight_record"]["aircraft_name"] == "M400"


def test_a_config_that_cannot_be_decoded_does_not_stop_the_flight(tmp_path):
    flight = flight_of(tmp_path)
    (tmp_path / "aux" / "run.yaml").write_text("uav:\n  name: DJI Mavic\n")
    (tmp_path / "drone" / "DJIFlightRecord_x.txt").write_bytes(b"x")
    flight.add_drone_data(dji_api_key="KEY")
    d = flight.raw_data.drone_data
    assert d.telemetry is None and "lager" in d.info["errors"]
    assert d.flight_record is not None


def test_nothing_to_load_is_an_error(tmp_path):
    flight = flight_of(tmp_path)
    with pytest.raises(FileNotFoundError):
        flight.add_drone_data()


def test_the_sources_are_saved_as_parquet_and_loaded_back(tmp_path):
    flight = flight_of(tmp_path)
    lager_folder(tmp_path / "aux")
    (tmp_path / "drone" / "DJIFlightRecord_x.txt").write_bytes(b"x")
    flight.add_drone_data(dji_api_key="KEY")
    out = tmp_path / "proc" / "flight"
    flight.save(out)

    raw = out / "raw" / "drone"
    assert (raw / "uav_telemetry.parquet").is_file() and (
        raw / "flight_record.parquet"
    ).is_file()
    assert (raw / "gimbal" / "mount_orientation.parquet").is_file()

    back = Flight.load(out).raw_data.drone_data
    original = flight.raw_data.drone_data
    assert back.telemetry.equals(original.telemetry)
    assert back.flight_record.equals(original.flight_record)
    assert back.gimbal["raw_imu"].equals(original.gimbal["raw_imu"])
    assert back.info["uav_model"] == "DJI M400"

    # saving again without the sources (a re-package) keeps the saved ones
    flight.raw_data.drone_data.sources = {}
    flight.raw_data.drone_data.info = {}
    flight.save(out)
    assert (raw / "uav_telemetry.parquet").is_file()
