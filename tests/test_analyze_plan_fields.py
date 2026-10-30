"""The fields a planned campaign flies (pils.analyze.plan.fields)."""

import polars as pl
import pytest

from pils.analyze.plan import fields as F

SCHEDULE = pl.DataFrame({
    "date": ["2026-11-29", "2026-11-29", "2026-11-30", "2026-11-30", "2026-11-30"],
    "start": ["09:00", "09:30", "10:00", "10:30", "08:00"],
    "end": ["09:30", "10:00", "10:30", "11:00", "08:30"],
    "telescope": ["satp1", "satp3", "satp1", "LAT", None],
    "project": ["POLOCALC", "HOVERCAL", "POLOCALC", "HOVERCAL", "COMMISSIONING"],
    "field": ["S45", "S45", "E40", "LATSCAN", None],
})
FIELDS = pl.DataFrame({
    "field": ["S45", "S45", "E40", "LATSCAN"],
    "telescope": ["satp1", "satp3", "satp1", "LAT"],
    "reference": ["satp1", "satp1", "satp1", None],
    "distance_m": [480.0, 480.0, 500.0, None],
    "az_deg": [180.0, 175.0, 90.0, 120.0],
    "el_deg": [45.0, 46.0, 40.0, 40.0],
    "E_m": [10.0, 10.0, 500.0, 300.0],
    "N_m": [-480.0, -480.0, 0.0, 200.0],
    "n_flights": [1, 1, 1, 1],
})


def test_one_row_per_field_flown():
    s = F.field_summary(SCHEDULE, FIELDS)
    assert s["field"].to_list() == ["S45", "E40", "LATSCAN"]
    first = s.row(0, named=True)
    assert first["set from"] == "satp1" and first["az / el"] == "180° / 45°" and first["distance [m]"] == 480.0
    assert first["telescopes"] == "satp1 1, satp3 1" and first["dates"] == "29 Nov" and first["hours"] == "09:00–10:00"
    assert s.row(2, named=True)["set from"] == "its own scan path"


def test_the_fields_on_the_ground_and_the_telescopes():
    pts = F.ground_points(SCHEDULE, FIELDS, F.field_summary(SCHEDULE, FIELDS))
    assert pts.rows() == [("S45", 10.0, -480.0, "satp1", 2), ("E40", 500.0, 0.0, "satp1", 1), ("LATSCAN", 300.0, 200.0, "", 1)]
    pytest.importorskip("so3g")
    sites = F.telescope_sites({"telescopes": {"LAT": {"site": [0, 0, 0]}, "satp1": {"site": [100.0, -50.0, 2.0]}},
                               "observing_telescopes": {"LAT": {"site": [0, 0, 0]}}})
    assert sites.rows() == [("LAT", 0.0, 0.0), ("satp1", 100.0, -50.0)]


def test_flights_per_field_per_day():
    d = F.field_days(SCHEDULE)
    assert d == {"fields": ["E40", "LATSCAN", "S45"], "dates": ["2026-11-29", "2026-11-30"],
                 "flights": [[0, 1], [0, 1], [2, 0]]}
    assert F.sky_positions(FIELDS, ["S45"]).height == 2


def test_a_telescope_with_its_own_location_is_placed():
    pytest.importorskip("so3g")
    from so3g.proj import coords

    from pils.geometry.site import register_site, site_enu

    lat = coords.SITES["so_lat"]
    sites = F.telescope_sites({"telescopes": {"CLASS": {"site": "test_class", "location": {
        "lon": lat.lon, "lat": lat.lat + 0.001, "elev": lat.elev + 10}}}})
    (name, e, n), = sites.rows()
    assert name == "CLASS" and abs(e) < 1e-6 and n == pytest.approx(111.2, abs=0.1)
    assert site_enu("test_class")[2] == pytest.approx(10)
    with pytest.raises(ValueError):
        register_site("so_lat", 0.0, 0.0, 0.0)
