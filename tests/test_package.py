"""Tests for flight packaging (pils.analyze.package and ``pils package``)."""

import json
from datetime import UTC, datetime

import polars as pl
import pytest

from pils import store
from pils.analyze import package as pkg
from pils.analyze.azel import AZELAnalysis, AZELVersion
from pils.analyze.package import (
    FlightPackager,
    PackageResult,
    flight_info_from_path,
    resolve_flight,
)
from pils.cli import main
from pils.flight import DroneData, Flight

GPS = pl.DataFrame({"timestamp": [0.0, 1.0], "lat_deg": [1.0, 1.0]})
DRONE = pl.DataFrame({"timestamp": [0.0, 1.0], "lat_deg": [1.0, 1.0]})
SYNCED = pl.DataFrame({"timestamp": [0.0, 0.01], "lat_deg": [1.0, 1.0]})
AZEL = pl.DataFrame(
    {
        "timestamp": [0.0, 0.01],
        "az_deg": [10.0, 11.0],
        "el_deg": [40.0, 41.0],
        "srange_m": [100.0, 101.0],
        "e_m": [1.0, 2.0],
        "n_m": [1.0, 2.0],
        "u_m": [1.0, 2.0],
    }
)


@pytest.fixture
def flight_dir(tmp_path):
    folder = tmp_path / "campaigns" / "202512" / "20251208" / "flight_20251208_1927"
    (folder / "drone").mkdir(parents=True)
    (folder / "aux" / "sensors").mkdir(parents=True)
    return folder


@pytest.fixture
def fake_flight(monkeypatch):
    """Flight whose loaders and sync return small tables; GPS only on the payload."""

    def add_sensor_data(self, sensor):
        if sensor == "gps":
            self.raw_data.payload_data.gps = GPS
        else:
            raise FileNotFoundError(f"no {sensor} file\nsecond line")

    def add_drone_data(self, *args, **kwargs):
        self.raw_data.drone_data = DroneData(drone_df=DRONE, litchi_df=None)

    def sync(self, **kwargs):
        self.sync_data = {"drone": SYNCED}
        return self.sync_data

    monkeypatch.setattr(Flight, "add_sensor_data", add_sensor_data)
    monkeypatch.setattr(Flight, "add_drone_data", add_drone_data)
    monkeypatch.setattr(Flight, "sync", sync)


@pytest.fixture
def fake_azel(monkeypatch):
    """AZEL that knows SATP1 only and records what it was asked to save."""
    calls = []

    def run_analysis(self, telescope, geod, save_data=False, **kwargs):
        calls.append((telescope, save_data, kwargs.get("additional_metadata")))
        if telescope != "SATP1":
            raise ValueError(f"No telescope positions found for '{telescope}'")
        version = AZELVersion("rev_20260101_000000", AZEL, {})
        if save_data:
            self._save_version(version, telescope_name=telescope)
        return version

    monkeypatch.setattr(AZELAnalysis, "run_analysis", run_analysis)
    return calls


def steps(result: PackageResult) -> dict[str, str]:
    return {step.name: step.status for step in result.steps}


class TestFlightInfo:
    def test_from_path(self, flight_dir):
        info = flight_info_from_path(flight_dir)
        assert info["flight_name"] == "flight_20251208_1927"
        assert info["aux_data_folder_path"] == str(flight_dir / "aux")
        assert info["processed_data_folder_path"] == str(flight_dir / "proc")
        assert info["takeoff_datetime"] == datetime(2025, 12, 8, 19, 27, tzinfo=UTC)
        assert info["flight_info"] == {}

    def test_from_path_needs_drone_folder(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            flight_info_from_path(tmp_path)

    def test_resolve_dict_and_folder(self, flight_dir):
        info = {"flight_name": "x", "drone_data_folder_path": "/a/drone"}
        assert resolve_flight(info) is info
        assert resolve_flight(flight_dir)["flight_name"] == flight_dir.name

    def test_resolve_unknown_name(self, monkeypatch):
        from pils.loader.stout import StoutLoader

        monkeypatch.setattr(StoutLoader, "__init__", lambda self: None)
        monkeypatch.setattr(StoutLoader, "load_single_flight", lambda self, **kw: None)
        with pytest.raises(LookupError):
            resolve_flight("flight_20990101_0000")

    def test_packager_does_not_modify_the_input(self, flight_dir):
        info = flight_info_from_path(flight_dir)
        FlightPackager(info).flight_info["flight_info"]["x"] = 1
        assert info["flight_info"] == {}


class TestRun:
    def test_package(self, flight_dir, fake_flight, fake_azel):
        result = FlightPackager.from_flight(flight_dir).run(
            telescopes=["SATP1", "SATP9"]
        )
        proc = flight_dir / "proc"

        assert result.ok
        assert steps(result) == {
            "load gps": "done",
            "load adc": "skipped",
            "load inclinometer": "skipped",
            "load lm76": "skipped",
            "load imu": "skipped",
            "load drone": "done",
            "sync": "done",
            "save flight": "done",
            "azel SATP1": "done",
            "azel SATP9": "skipped",
            "manifest": "done",
        }
        # Errors are reported on one line
        assert "second line" not in result.summary()

        # Flight.save: raw data and one synchronized revision
        flight_out = proc / pkg.FLIGHT_DIR
        assert (flight_out / "raw" / "payload" / "gps.parquet").is_file()
        (revision,) = store.list_versions(flight_out / "sync")
        sync_manifest = store.read_manifest(flight_out / "sync" / revision)
        assert sync_manifest["metadata"]["package_version"] == result.version_name
        assert "drone" in sync_manifest["tables"]

        # AZEL saved, with the sync revision it used, and its telescope file
        telescope, saved, metadata = fake_azel[0]
        assert (telescope, saved) == ("SATP1", True)
        assert metadata["sync_revision"] == revision
        azel_dir = proc / "azel" / "SATP1" / "rev_20260101_000000"
        assert (azel_dir / "manifest.json").is_file()
        assert (azel_dir / "azel_data.parquet").is_file()
        csv = azel_dir / "rev_20260101_000000_azel_satp1_20251208_1927.csv"
        assert pl.read_csv(csv).columns == ["timestamp", "az", "el", "srange"]

        manifest = json.loads((proc / "package_manifest.json").read_text())
        assert manifest["ok"] is True
        assert manifest["version_name"] == result.version_name
        assert manifest["inputs"]["telescopes"] == ["SATP1", "SATP9"]

    def test_dry_run_writes_nothing(self, flight_dir, fake_flight, fake_azel):
        result = FlightPackager.from_flight(flight_dir).run(
            telescopes=["SATP1"], save=False
        )
        assert result.ok
        assert steps(result)["save flight"] == "skipped"
        assert steps(result)["azel SATP1"] == "done"
        assert fake_azel[0][1] is False
        assert not (flight_dir / "proc").exists()

    def test_output_dir_per_flight(self, flight_dir, tmp_path, fake_flight, fake_azel):
        out = tmp_path / "out"
        result = FlightPackager.from_flight(flight_dir, output_dir=out).run(
            telescopes=["SATP1"]
        )
        proc = out / flight_dir.name / "proc"
        assert result.proc_dir == str(proc)
        assert store.list_versions(proc / pkg.FLIGHT_DIR / "sync")
        assert store.list_versions(proc / "azel" / "SATP1")
        assert not (flight_dir / "proc").exists()

    def test_no_gps_stops_before_sync(self, flight_dir, fake_flight, monkeypatch):
        monkeypatch.setattr(
            Flight, "add_sensor_data", lambda self, sensor: None
        )  # nothing on the payload
        result = FlightPackager.from_flight(flight_dir).run()
        assert not result.ok
        assert steps(result)["load"] == "failed"
        assert "sync" not in steps(result)
        assert not (flight_dir / "proc" / pkg.FLIGHT_DIR).exists()

    def test_sync_failure_is_reported(self, flight_dir, fake_flight, monkeypatch):
        def sync(self, **kwargs):
            raise ValueError("no overlap")

        monkeypatch.setattr(Flight, "sync", sync)
        result = FlightPackager.from_flight(flight_dir).run()
        assert not result.ok
        assert steps(result)["sync"] == "failed"
        assert "save flight" not in steps(result)

    def test_power_targets(self, flight_dir, fake_flight, fake_azel, monkeypatch):
        from pils.analyze.power import PowerAnalysis

        runs = []

        def run_analysis(self, telescope, wafer_slot, **kwargs):
            runs.append((telescope, wafer_slot, kwargs["save_data"]))
            if wafer_slot == "ws9":
                raise ValueError("no beam for ws9")
            return type("Version", (), {"version_name": "rev_20260101_000001"})()

        monkeypatch.setattr(PowerAnalysis, "run_analysis", run_analysis)
        result = FlightPackager.from_flight(flight_dir).run(
            telescopes=["SATP1"], power_targets=[("satp1", "ws0"), ("satp1", "ws9")]
        )
        assert runs == [("satp1", "ws0", True), ("satp1", "ws9", True)]
        assert steps(result)["power satp1/ws0"] == "done"
        assert steps(result)["power satp1/ws9"] == "failed"
        assert not result.ok


class TestCli:
    def test_package(self, flight_dir, fake_flight, fake_azel, capsys):
        assert main(["package", str(flight_dir), "-t", "SATP1"]) == 0
        out = capsys.readouterr().out
        assert "azel SATP1" in out
        assert store.list_versions(flight_dir / "proc" / pkg.FLIGHT_DIR / "sync")

    def test_exit_code_on_failure(self, flight_dir, fake_flight, monkeypatch, capsys):
        monkeypatch.setattr(Flight, "add_sensor_data", lambda self, sensor: None)
        assert main(["package", str(flight_dir), "--dry-run"]) == 1
        assert "Nothing saved" in capsys.readouterr().out

    def test_unknown_flight(self, tmp_path, monkeypatch, capsys):
        def raise_lookup(flight, output_dir=None):
            raise LookupError(f"Flight {flight} is not a folder and is not in STOUT")

        monkeypatch.setattr(FlightPackager, "from_flight", staticmethod(raise_lookup))
        assert main(["package", "flight_20990101_0000"]) == 1
        assert "not in STOUT" in capsys.readouterr().err

    def test_bad_power_target(self, flight_dir, capsys):
        assert main(["package", str(flight_dir), "--power", "satp1"]) == 2
        assert "TELESCOPE:WAFER_SLOT" in capsys.readouterr().err


def test_describe_and_error():
    assert pkg._describe(GPS) == "2x2"
    assert pkg._describe({"INS": GPS}) == "{INS: 2x2}"
    long = ValueError("x" * 300 + "\nmore")
    message = pkg._error(long)
    assert message.startswith("ValueError: xxx") and message.endswith("...")
    assert "more" not in message
