"""Tests for the flight quick look (pils.analyze.quicklook, ``pils quicklook``)."""

import numpy as np
import polars as pl
import pytest

from pils import store
from pils.analyze import quicklook as Q
from pils.analyze.azel import AZELAnalysis, AZELVersion
from pils.analyze.package import flight_info_from_path
from pils.cli import main
from pils.flight import DroneData, Flight

T0 = 1_765_221_000.0
T = T0 + np.arange(0.0, 600.0, 0.5)
HEIGHT = np.where((T > T0 + 60) & (T < T0 + 540), 100.0, 0.0)
DRONE = pl.DataFrame(
    {
        "timestamp": T,
        "rtk_lat_deg": -22.96 + 1e-6 * np.arange(T.size),
        "rtk_lon_deg": np.full(T.size, -67.78),
        "rtk_h_ell_m": 5200 + HEIGHT,
        "rtk_pos_flag_1": np.where(np.arange(T.size) % 10 == 0, 34, 50),
    }
)
OSD = pl.DataFrame({"timestamp": T, "h_rel_m": HEIGHT})
BATTERY = pl.DataFrame({"charge_pct": [0, 98, 40], "pack_voltage_v": [0.0, 52.0, 45.0]})
LITCHI = pl.DataFrame(  # the log of another flight, an hour later
    {
        "timestamp": T + 3600,
        "lat_deg": np.full(T.size, -22.96),
        "lon_deg": np.full(T.size, -67.78),
        "is_flying": (HEIGHT > 0).astype(int),
    }
)
GPS = pl.DataFrame(
    {"timestamp": T, "lat_deg": np.full(T.size, -22.96), "h_ell_m": 5200 + HEIGHT}
)
ADC = pl.DataFrame(  # stops before landing
    {"timestamp": T[: T.size // 2], "amplitude_mv": np.ones(T.size // 2)}
)
AZEL = pl.DataFrame(
    {
        "timestamp": T,
        "az_deg": np.linspace(10, 20, T.size),
        "el_deg": np.linspace(30, 60, T.size),
        "srange_m": np.linspace(100, 400, T.size),
    }
)


@pytest.fixture
def flight_dir(tmp_path):
    folder = tmp_path / "campaigns" / "202512" / "20251208" / "flight_20251208_1927"
    (folder / "drone").mkdir(parents=True)
    (folder / "drone" / "x_drone.dat").write_bytes(b"1")
    (folder / "aux" / "sensors").mkdir(parents=True)
    (folder / "aux" / "sensors" / "x_ACC.bin").write_bytes(b"")
    return folder


@pytest.fixture
def fake_flight(monkeypatch):
    def add_sensor_data(self, sensor):
        if sensor == "gps":
            self.raw_data.payload_data.gps = GPS
        elif sensor == "adc":
            self.raw_data.payload_data.adc = ADC
        else:
            raise FileNotFoundError(f"no {sensor}")

    def add_drone_data(self, *args, **kwargs):
        self.raw_data.drone_data = DroneData(
            drone_df=DRONE,
            litchi_df=LITCHI,
            records={"osd": OSD, "battery_info": BATTERY},
        )

    def sync(self, **kwargs):
        from pils.synchronizer import Synchronizer

        sync = Synchronizer()
        sync.offsets = {
            "drone_gps": {"time_offset": 150.0, "correlation": 0.99},
            "litchi_gps": {"time_offset": 0.2, "correlation": 0.5},
            "payload_clock": {"time_offset": 0.0, "method": "none"},
        }
        # What Flight.sync does: the Synchronizer is called
        Synchronizer.synchronize(sync)
        self.sync_data = {"drone": DRONE}
        return self.sync_data

    def synchronize(self, *args, **kwargs):  # keep the offsets set above
        return {}

    monkeypatch.setattr(Flight, "add_sensor_data", add_sensor_data)
    monkeypatch.setattr(Flight, "add_drone_data", add_drone_data)
    monkeypatch.setattr(Flight, "sync", sync)
    from pils.synchronizer import Synchronizer

    monkeypatch.setattr(Synchronizer, "synchronize", synchronize)


@pytest.fixture
def fake_azel(monkeypatch):
    def run_analysis(self, telescope, geod, save_data=False, **kwargs):
        if telescope != "SATP1":
            raise ValueError(f"No telescope positions found for '{telescope}'")
        return AZELVersion("rev_20260101_000000", AZEL, {})

    monkeypatch.setattr(AZELAnalysis, "run_analysis", run_analysis)


# -- pure functions -----------------------------------------------------------


def test_airborne_interval():
    assert Q.airborne_interval(T, HEIGHT) == (T[121], T[1079])
    # the longest of two hops
    two = np.where((T > T0 + 10) & (T < T0 + 20), 5.0, HEIGHT)
    assert Q.airborne_interval(T, two) == (T[121], T[1079])
    assert Q.airborne_interval(T, np.zeros(T.size)) is None


def test_table_summary_coverage_and_gaps():
    times = np.concatenate([np.arange(0, 50, 0.1), np.arange(60, 100, 0.1)])
    summary = Q.table_summary(times, (0.0, 100.0))
    assert summary["gaps"] == 1
    assert summary["longest_gap_s"] == pytest.approx(10.1)
    assert summary["rate_hz"] == pytest.approx(10)
    assert summary["coverage_pct"] == pytest.approx(89.9, abs=0.2)
    # a slow sensor (0.1 Hz) has no gaps
    slow = Q.table_summary(np.arange(0, 100, 10.0), (0.0, 90.0))
    assert slow["gaps"] == 0 and slow["coverage_pct"] == pytest.approx(100)
    assert Q.table_summary(times, (200.0, 300.0))["coverage_pct"] == 0


def test_overlap_and_flags():
    assert Q.overlap_pct((0, 50), (0, 100)) == pytest.approx(50)
    assert Q.overlap_pct((200, 300), (0, 100)) == 0
    flags = Q.rtk_flag_summary(pl.Series([50, 50, 34, 16]))
    assert flags["fixed"] == 50 and flags["float"] == 25 and flags["other"] == 0


def test_offsets():
    offsets = {
        "drone_gps": {"time_offset": 150.0, "correlation": 0.99},
        "inclinometer": {"time_offset": 0.3, "correlation": 0.4},
        "payload_clock": {"time_offset": 0.0, "method": "none"},
    }
    rows = Q.offset_rows(offsets)
    assert rows[0]["method"] == "correlation" and rows[2]["method"] == "none"
    text = " ".join(Q.offset_warnings(offsets))
    assert "drone_gps clock off by 150 s" in text
    assert "inclinometer correlation 0.40" in text
    assert "payload clock" in text


HEADER = """\
     3.04           OBSERVATION DATA    M: Mixed            RINEX VERSION / TYPE
G    4 L1C S1C L2X S2X                                      SYS / # / OBS TYPES
E    2 L1X S1X                                              SYS / # / OBS TYPES
                                                            END OF HEADER
"""


def field(value, lli=""):
    return f"{value:14.3f}{lli:1s} " if value is not None else " " * 16


def rinex(path, epochs=10, drop=(), lli_at=()):
    lines = [HEADER.rstrip("\n")]
    for k in range(epochs):
        lines.append(f"> 2025 12 06 18 33 {k:10.7f}  0  3")
        for prn, snr in (("G01", 45.0), ("G02", 33.0)):
            if (prn, k) in drop:
                continue
            flag = "1" if (prn, k) in lli_at else ""
            lines.append(prn + field(1e8, flag) + field(snr) + field(1e8) + field(28.0))
        lines.append("E05" + field(1e8) + field(38.0))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_band_table(tmp_path):
    times, epochs = Q.read_rinex(
        rinex(tmp_path / "r.obs", drop={("G02", 5)}, lli_at={("G01", 3)})
    )
    assert len(times) == 10
    rows = {(r["system"], r["band"]): r for r in Q.band_table(epochs, 1.0)}
    g1 = rows[("G", "1")]
    assert g1["sats"] == 2 and g1["sats_35dbhz"] == 1 and g1["sats_40dbhz"] == 1
    assert g1["snr_dbhz"] == pytest.approx(45.0)
    assert g1["lli_pct"] == pytest.approx(100 / 19)
    # G01 10 epochs, G02 5 + 4: (100 + 25 + 16) / 19
    assert g1["arc_s"] == pytest.approx(141 / 19)
    assert rows[("G", "2")]["sats_30dbhz"] == 0
    assert rows[("E", "1")]["sats"] == 1


def test_common_satellites(tmp_path):
    rover = Q.read_rinex(
        rinex(tmp_path / "r.obs", drop={("G02", k) for k in range(10)})
    )
    base = Q.read_rinex(rinex(tmp_path / "b.obs", epochs=6))
    assert Q.common_satellites(rover, base) == {"E": 1.0, "G": 1.0}


def test_md_table():
    text = Q.md_table([{"a": 1.234, "b": None}, {"a": float("nan"), "b": "x"}])
    assert text.splitlines() == ["| a | b |", "|---|---|", "| 1.2 | - |", "| - | x |"]
    assert Q.md_table([]) == "_none_"


# -- the report ----------------------------------------------------------------


def test_report(flight_dir, tmp_path, fake_flight, fake_azel):
    out = tmp_path / "out"
    quicklook = Q.QuickLook(flight_info_from_path(flight_dir), output_dir=out)
    report = quicklook.run(rinex=True)
    revision = report.parent
    assert revision.parent == out / flight_dir.name / "proc" / "quicklook"
    assert not (flight_dir / "proc").exists()  # nothing in the data folder
    text = report.read_text()
    for title in ("Warnings", "Flight", "Files", "Drone", "Payload sensors"):
        assert f"## {title}" in text
    warnings = " ".join(quicklook.warnings)
    assert "x_ACC.bin is empty" in warnings
    assert "Litchi log covers 0 %" in warnings
    assert "adc covers" in warnings
    assert "drone_gps clock off by 150 s" in warnings
    assert "litchi_gps correlation 0.50" in warnings
    assert "RTK fixed only 90 %" in warnings
    assert "No rover recording" in text  # no GPS.bin in the folder
    assert "| SATP1 | 33.0 | 57.0 |" in text  # airborne part only
    assert "98 -> 40 %" in text
    for plot in ("drone_track", "drone_height", "payload_adc"):
        assert (revision / "plots" / f"{plot}.png").exists()
        assert f"plots/{plot}.png" in text
    tables, manifest = store.read_tables(revision)
    assert set(tables) >= {"inventory", "sensors", "sync_offsets"}
    assert manifest["kind"] == "quicklook"
    assert manifest["metadata"]["warnings"] == quicklook.warnings


def test_failed_section_does_not_stop_the_report(
    flight_dir, tmp_path, fake_flight, fake_azel, monkeypatch
):
    def boom(self, section):
        raise RuntimeError("broken")

    monkeypatch.setattr(Q.QuickLook, "_drone_section", boom)
    quicklook = Q.QuickLook(flight_info_from_path(flight_dir), output_dir=tmp_path)
    text = quicklook.run(rinex=False).read_text()
    assert "Could not be done: `RuntimeError: broken`" in text
    assert "## AZEL" in text
    assert "Drone: failed" in " ".join(quicklook.warnings)


def test_cli(flight_dir, tmp_path, fake_flight, fake_azel, capsys):
    code = main(
        ["quicklook", str(flight_dir), "--output-dir", str(tmp_path), "--no-rinex"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "flight_20251208_1927:" in out and "report.md" in out


def test_cli_unknown_flight(tmp_path, monkeypatch, capsys):
    def raise_lookup(flight, output_dir=None):
        raise LookupError(f"Flight {flight} is not a folder and is not in STOUT")

    monkeypatch.setattr(Q.QuickLook, "from_flight", raise_lookup)
    assert main(["quicklook", "flight_x", "--output-dir", str(tmp_path)]) == 1
    assert "not in STOUT" in capsys.readouterr().err
