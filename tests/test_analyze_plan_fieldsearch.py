"""What a field search found (pils.analyze.plan.fieldsearch)."""

import numpy as np
import polars as pl

from pils.analyze.plan import fieldsearch as FS


def test_the_searches_of_a_run(tmp_path):
    (tmp_path / "optimized_fields" / "south").mkdir(parents=True)
    (tmp_path / "optimized_fields" / "north").mkdir()
    assert [s["label"] for s in FS.searches(tmp_path)] == ["optimized_fields / north", "optimized_fields / south"]
    (tmp_path / "optimized_fields" / "run.json").write_text("{}")
    assert FS.searches(tmp_path)[0] == {"path": "optimized_fields", "label": "optimized_fields"}


def test_when_a_field_can_be_flown():
    ok = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], bool)
    bl = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], bool)
    win = {"dates": np.array(["2026-11-26", "2026-11-27"]), "times": np.array(["08:00", "08:30", "09:00", "09:30"]),
           "S45|satp1|ws0|flyable": ok, "S45|satp1|ws0|blocked": bl, "S45|satp1|own": np.zeros_like(bl),
           "days": np.array([True, True])}
    out = FS.field_windows(win, "S45")
    [lane] = out["lanes"]
    assert lane["name"] == "satp1" and lane["rows"][0]["ok"] == [[8.0, 9.0]] and lane["rows"][1]["other"] == [[9.5, 10.0]]
    assert lane["hours_per_day"] == 1.0 and out["telescopes"] == ["satp1"]
    one = FS.field_windows(win, "S45", "satp1", day="2026-11-27")
    assert one["dates"] == ["2026-11-27"] and one["lanes"][0]["name"] == "ws0"


def test_the_sun_and_the_landscape():
    sep = np.array([[50, 40, 50, 50], [50, 50, 50, 30]], float)
    times = ["08:00", "08:30", "09:00", "09:30"]
    assert FS.sun_safe(times, sep, 45, day=0) == [[8.0, 8.5], [9.0, 10.0]]
    assert FS.sun_safe(times, sep, 45) == [[8.0, 8.5], [9.0, 9.5]]
    cand = pl.DataFrame({"az_from_satp1": [180.0, 180.0, 185.0], "el_from_satp1": [45.0, 47.0, 45.0],
                         "usable_h_per_day": [2.0, 3.0, 1.0]})
    land = FS.landscape(cand)
    assert land["az"] == [180.0, 185.0] and land["grid"] == [[2.0, 1.0], [3.0, None]] and land["best"]["el"] == 47.0
    conf = pl.DataFrame({"date": ["d1", "d1", "d2"], "time": ["08:00", "08:30", "08:00"], "field": ["S45", "S45", "S45"],
                         "separation_deg": [20.0, 19.0, 18.0], "exclusion_deg": [10.0] * 3})
    s = FS.separation(conf, ["S45"], "d1")
    assert s["days"] == ["d1", "d2"] and s["series"][0]["separation_deg"] == [20.0, 19.0] and s["exclusion_deg"] == 10.0


def test_a_field_over_the_campaign_for_every_telescope():
    sun = {"telescope": "satp1", "dates": ["d1"], "times": ["08:00", "08:30", "09:00", "09:30"], "threshold_deg": 45.0,
           "min_el_deg": 40.0, "scans": [{"label": "ws0 bs 0", "wafer": "ws0", "boresight": 0, "usable": True, "el": 50.0}],
           "separation": {"ws0 bs 0": [[50, 50, 40, 50]]}}
    ov = FS.field_overview({"satp1": sun, "CLASS": None}, {"satp1": {"d1": [[9.5, 10.0]]}, "CLASS": {}})
    assert ov["lanes"][0]["rows"] == [[[8.0, 9.0]]] and ov["lanes"][1]["note"] == "no Sun avoidance"
    assert ov["lanes"][1]["rows"] == [[[8.0, 10.0]]]
    assert FS.field_overview({"satp1": sun}, {}, calibratable=set())["lanes"][0]["note"].startswith("no wafer")
    c = FS.field_campaign(sun, "ws0", {"d1": [[9.5, 10.0]]})
    assert c["lines"][0]["rows"] == [[[8.0, 9.0], [9.5, 10.0]]] and c["protected"] == [[[9.5, 10.0]]] and c["note"] is None
    assert FS.field_campaign(sun, "ws0", calibratable=set())["note"].startswith("ws0 cannot be calibrated")
