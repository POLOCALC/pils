"""Matrice 400 flight record (.txt of the remote controller's app).

No real record can be decoded in the tests (the keys come from DJI's API), so
the frames ``pydjirecord`` returns are built here, and ``DJILog`` is replaced
where the call to the API matters.
"""

from datetime import UTC, datetime, timedelta, timezone

import polars as pl
import pytest
from pydjirecord import Frame
from pydjirecord.layout.details import Details
from pydjirecord.record.osd import FlightAction

from pils.drones.dji.m400 import flightrecord
from pils.drones.dji.m400.flightrecord import (
    DJIFlightRecordError,
    M400FlightRecord,
    frame_row,
    frames_to_table,
    resolve_api_key,
)

T0 = datetime(2026, 11, 19, 10, 15, 0, tzinfo=UTC)


def make_frame(i: int, **osd) -> Frame:
    f = Frame()
    f.custom.date_time = T0 + timedelta(milliseconds=100 * i)
    f.osd.latitude, f.osd.longitude = 45.5 + i * 1e-5, 9.2
    f.osd.height, f.osd.altitude, f.osd.yaw = 30.0 + i, 240.0 + i, 90.0
    f.osd.flight_action = FlightAction.NONE
    f.gimbal.pitch = -30.0
    f.battery.charge_level = 80 - i
    f.battery.cell_voltages = [4.0, 4.1, 4.2]
    f.recover.aircraft_name = "M400"
    for k, v in osd.items():
        setattr(f.osd, k, v)
    return f


class FakeLog:
    """What DJILog.from_bytes returns, recording how it was asked for its keychains."""

    calls: list = []

    def __init__(self, version=13, frames=None):
        self.version = version
        self.details = Details(aircraft_name="M400", street="Via Roma 1", city="Milano")
        self._frames = (
            frames if frames is not None else [make_frame(i) for i in range(3)]
        )

    def fetch_keychains(self, api_key, *, cache=True, verify=True):
        FakeLog.calls.append((api_key, cache))
        return [["kc"]]

    def frames(self, keychains=None):
        self.seen_keychains = keychains
        return self._frames


@pytest.fixture
def txt(tmp_path, monkeypatch):
    FakeLog.calls = []
    f = tmp_path / "DJIFlightRecord_2026-11-19_[10-15-00].txt"
    f.write_bytes(b"not parsed: DJILog is replaced")
    monkeypatch.setattr(
        flightrecord.DJILog,
        "from_bytes",
        classmethod(lambda cls, data: FakeLog()),
        raising=False,
    )
    monkeypatch.delenv(flightrecord.API_KEY_VARIABLE, raising=False)
    return f


def test_a_frame_becomes_a_row_with_its_time_and_sections():
    row = frame_row(make_frame(2))
    assert row["timestamp"] == pytest.approx(T0.timestamp() + 0.2)
    assert row["datetime"] == T0 + timedelta(milliseconds=200)
    assert row["osd_latitude"] == pytest.approx(45.50002) and row["osd_height"] == 32.0
    assert row["osd_flight_action"] == "NONE"  # an enumeration by its name
    assert row["gimbal_pitch"] == -30.0 and row["battery_charge_level"] == 78
    assert (row["battery_cell_voltage_1"], row["battery_cell_voltage_3"]) == (4.0, 4.2)
    assert "battery_cell_voltages" not in row


def test_a_time_without_a_zone_is_taken_as_utc_and_other_zones_converted():
    naive, east = make_frame(0), make_frame(0)
    naive.custom.date_time = datetime(2026, 11, 19, 10, 15, 0)
    east.custom.date_time = datetime(
        2026, 11, 19, 11, 15, 0, tzinfo=timezone(timedelta(hours=1))
    )
    assert (
        frame_row(naive)["timestamp"] == frame_row(east)["timestamp"] == T0.timestamp()
    )


def test_frames_make_a_sorted_table_with_utc_datetimes():
    table = frames_to_table([make_frame(2), make_frame(0), make_frame(1)])
    assert table["timestamp"].is_sorted() and table.height == 3
    assert table["datetime"].dtype == pl.Datetime("us", "UTC")
    assert table["osd_height"].to_list() == [30.0, 31.0, 32.0]
    assert frames_to_table([]).is_empty()


def test_a_log_of_version_13_needs_the_key_and_asks_the_api(txt):
    r = M400FlightRecord(txt)
    with pytest.raises(DJIFlightRecordError, match="DJI_API_KEY"):
        r.load_data()
    r.load_data(api_key="KEY")
    assert FakeLog.calls == [("KEY", True)] and r.data.height == 3


def test_the_key_comes_from_the_environment_and_the_cache_can_be_off(txt, monkeypatch):
    monkeypatch.setenv("DJI_API_KEY", "ENVKEY")
    r = M400FlightRecord(txt)
    r.load_data(use_cache=False)
    assert FakeLog.calls == [("ENVKEY", False)]
    assert resolve_api_key("ARG") == "ARG"


def test_an_older_log_needs_no_key_and_no_api_call(tmp_path, monkeypatch):
    f = tmp_path / "old.txt"
    f.write_bytes(b"x")
    monkeypatch.delenv("DJI_API_KEY", raising=False)
    FakeLog.calls = []
    monkeypatch.setattr(
        flightrecord.DJILog,
        "from_bytes",
        classmethod(lambda cls, d: FakeLog(version=12)),
    )
    r = M400FlightRecord(f)
    r.load_data()
    assert FakeLog.calls == [] and r.data.height == 3


def test_the_header_is_kept_without_the_street_address(txt):
    r = M400FlightRecord(txt)
    r.load_data(api_key="KEY")
    assert r.details["version"] == 13 and r.details["aircraft_name"] == "M400"
    assert r.details["recover_aircraft_name"] == "M400"
    assert not {"street", "city", "area", "sub_street"} & set(r.details)


def test_a_log_without_frames_gives_an_empty_table(tmp_path, monkeypatch):
    f = tmp_path / "e.txt"
    f.write_bytes(b"x")
    monkeypatch.setattr(
        flightrecord.DJILog,
        "from_bytes",
        classmethod(lambda cls, d: FakeLog(12, frames=[])),
    )
    r = M400FlightRecord(f)
    r.load_data()
    assert r.data.is_empty() and r.details["version"] == 12


def test_errors_of_pydjirecord_are_one_error(txt, monkeypatch):
    from pydjirecord.error import ApiKeyError

    class Denied(FakeLog):
        def fetch_keychains(self, *a, **k):
            raise ApiKeyError("Invalid DJI API key (403)")

    monkeypatch.setattr(
        flightrecord.DJILog, "from_bytes", classmethod(lambda cls, d: Denied())
    )
    with pytest.raises(DJIFlightRecordError, match="ApiKeyError.*403"):
        M400FlightRecord(txt).load_data(api_key="BAD")


def test_files_that_are_not_flight_records(tmp_path):
    for name, content in (
        ("empty.txt", b""),
        ("junk.txt", b"junk" * 100),
        ("zeros.txt", b"\x00" * 500),
    ):
        f = tmp_path / name
        f.write_bytes(content)
        with pytest.raises(
            DJIFlightRecordError, match="not a (readable )?DJI flight record"
        ):
            M400FlightRecord(f).load_data(api_key="KEY")


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        M400FlightRecord(tmp_path / "none.txt").load_data(api_key="KEY")
