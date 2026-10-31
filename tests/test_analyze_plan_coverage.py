"""The coverage a planned campaign gives (pils.analyze.plan.coverage)."""

import polars as pl
import pytest

from pils.analyze.plan import coverage as C

SCHEDULE = pl.DataFrame({
    "date": ["2026-11-29", "2026-11-29", "2026-11-30"],
    "telescope": ["satp1", "satp1", "satp3"], "wafer": ["ws0", "ws1", "ws0"],
    "project": ["POLOCALC", "HOVERCAL", "POLOCALC"], "field": ["S45", "S45", "E40"],
    "section": [0, 1, 0], "boresight": [0, -30, 0],
})
SECONDARY = pl.DataFrame({"telescope": ["satp3", "satp2"], "wafer": ["ws1", "ws0"], "project": ["POLOCALC"] * 2,
                          "status": ["joined", "skipped"]})


def test_flights_per_wafer_own_and_joined():
    out = C.wafer_flights(SCHEDULE, SECONDARY)
    assert out["telescopes"] == ["satp1", "satp3"] and out["wafers"] == ["ws0", "ws1"] and out["has_joins"]
    [panel] = out["panels"]
    assert panel["project"] == C.ALL and panel["own"] == [[1, 1], [1, 0]] and panel["joined"] == [[0, 0], [0, 1]]
    assert [p["project"] for p in C.wafer_flights(SCHEDULE, SECONDARY, "POLOCALC")["panels"]] == ["POLOCALC"]
    assert C.telescope_field_flights(SCHEDULE)["flights"] == [[0, 2], [1, 0]]


def test_rotations_and_sections():
    rot = pl.DataFrame({"telescope": ["satp1"] * 4, "project": ["POLOCALC"] * 4, "source_ghz": [90] * 4,
                        "wafer": ["ws0"] * 4, "pixel": [1, 1, 1, 2], "boresight": [0, 30, -30, 0],
                        "kind": ["own", "own", "joined", "own"], "n_flights": [1] * 4})
    per = C.rotations_per_detector(rot)
    assert per.rows() == [("satp1", "ws0", 1, 3, "-30 +0 +30"), ("satp1", "ws0", 2, 1, "+0")]
    assert C.rotations_per_detector(rot, own_only=True).row(0)[3] == 2
    pixels = pl.DataFrame({"telescope": ["satp1"] * 3, "wafer": ["ws0"] * 3, "pixel": [1, 2, 3],
                           "xi_deg": [0.0, 1.0, 2.0], "eta_deg": [0.0] * 3})
    cell = C.rotations_summary(per, pixels)["cells"]["satp1"]["ws0"]
    assert cell == {"seen": 2, "two": 1, "three": 1, "total": 3, "share": pytest.approx(1 / 3)}
    assert C.rotations_map(per, pixels, "satp1")["n_rotations"].to_list() == [3, 1, 0]
    assert C.section_flights(SCHEDULE).rows()[0] == ("satp1", "ws0", "0", "bs +0", 1)


def test_detectors_with_their_secondary_crossings():
    counts = pl.DataFrame({"telescope": ["satp1"] * 2, "project": ["P"] * 2, "source_ghz": [90, 90], "band": ["b"] * 2,
                           "beam_arcmin": [30.0] * 2, "wafer": ["ws0"] * 2, "pixel": [1, 2], "n_flights": [2, 1],
                           "n_crossings": [5, 0]})
    sec = pl.DataFrame({"telescope": ["satp1", "satp2"], "project": ["P"] * 2, "source_ghz": ["90.0", "90"],
                        "band": ["b"] * 2, "beam_arcmin": [30.0] * 2, "wafer": ["ws0"] * 2, "pixel": [1, 7], "n_crossings": [3, 4]})
    both = C.with_secondaries(counts, sec).sort("telescope", "pixel")
    assert both.select("telescope", "pixel", "n_crossings_primary", "n_crossings_secondary", "n_crossings").rows() == [
        ("satp1", 1, 5, 3, 8), ("satp1", 2, 0, 0, 0), ("satp2", 7, 0, 4, 4)]
    pixels = pl.DataFrame({"telescope": ["satp1"] * 3, "wafer": ["ws0"] * 3, "pixel": [1, 2, 3],
                           "xi_deg": [0.0, 1.0, 2.0], "eta_deg": [0.0] * 3})
    m = C.detector_map(pixels, both, "satp1", "P", "90", "n_crossings")
    assert m["reached"] == 1 and m["vmax"] == 8 and m["beam_arcmin"] == 30.0
    assert m["detectors"]["n_crossings"].to_list() == [8, 0, 0]
    h = C.wafer_histogram(both, "satp1", "P", "90", "ws0", "n_flights")
    assert h["counts"] == [0, 1, 1] and h["reached"] == 2 and h["median"] == 1.5


def test_how_long_the_crossings_last():
    times = pl.DataFrame({"telescope": ["satp1"] * 3, "project": ["P"] * 3, "source_ghz": [90] * 3, "band": ["b"] * 3,
                          "beam_arcmin": [30.0] * 3, "wafer": ["ws0"] * 3, "pixel": [1, 1, 2],
                          "crossing_s": [0.2, 1.0, 0.6], "n_flights": [2, 1, 1]})
    out = C.crossing_times(times, "satp1", C.ALL, "90", 2.0)
    assert out["n_crossings"] == 4 and out["n_detectors"] == 2 and sum(out["crossings"]) == 4
    assert sum(out["detectors"]) == 2 and 0 < out["limit_s"] < 1 and out["share_long_enough"] == 1.0
    assert C.crossing_times(times, "satp1", C.ALL, "150", 2.0) == {"empty": True}
