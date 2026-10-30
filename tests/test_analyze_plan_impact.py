"""The impact a planned campaign has on the other telescopes (pils.analyze.plan.impact)."""

import polars as pl

from pils.analyze.plan import impact as I

IMPACT = pl.DataFrame({
    "trim_deg": [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
    "exclusion_deg": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 5.0],
    "target": ["TOTAL", "ALL High", "ALL Low", "CMB_A", "CMB_B", "TOTAL", "TOTAL"],
    "priority": [None, None, None, "High", "Low", None, None],
    "affected_min": [60.0, 40.0, 30.0, 40.0, 50.0, 20.0, 10.0],
    "observed_min": [1000.0, 500.0, 20.0, 500.0, 20.0, 1000.0, 1000.0],
    "observed_in_campaign_min": [1000.0] * 7,
    "affected_frac_of_target": [0.06, 0.08, 1.5, 0.08, 2.5, 0.02, 0.01],
    "n_flights_affected": [6, 4, 3, 4, 3, 2, 1],
})
DAYS = pl.DataFrame({"date": ["2026-11-29", "2026-11-30", "2026-12-01"], "target": ["CMB_A", "CMB_A", "CMB_B"],
                     "priority": ["High", "High", "Low"], "affected_min": [20.0, 20.0, 50.0],
                     "fields": ["S45", "E40", "S45"], "min_separation_deg": [8.0, 6.0, 9.0]})


def test_the_planned_angle_and_the_whole_telescope():
    # the trim scenarios are run at the planned angle only
    assert I.planned_exclusion(IMPACT.to_dicts()) == 10.0 and I.exclusion_of(IMPACT) == 10.0
    assert I.total(IMPACT, 10.0)["affected_min"] == 60.0


def test_by_priority_and_by_scan():
    p = I.by_priority(IMPACT, 10.0)
    assert p["class"].to_list() == ["High", "Low"] and p["unaffected_min"].to_list() == [460.0, 0.0]
    s = I.by_scan(IMPACT, 10.0)
    assert s["target"].to_list() == ["CMB_A", "CMB_B"]
    sc = I.scenarios(IMPACT, 10.0)
    assert sc["trim"]["trim_deg"].to_list() == [0.0, 5.0] and sc["exclusion"]["exclusion_deg"].to_list() == [5.0, 10.0]


def test_how_close_each_scan_came():
    assert I.closest_by_day(DAYS)["CMB_A"]["date"] == "2026-11-30"
    rows = I.scan_rows(IMPACT, 10.0, DAYS, None, recorded=True)
    assert [(r["scan"], r["closest_deg"], r["when"]) for r in rows] == [
        ("CMB_A", 6.0, "2026-11-30, E40"), ("CMB_B", 9.0, "2026-12-01, S45")]
    approach = {"CMB_A": {"closest_deg": 12.0, "date": "2026-11-29", "time": "10:00", "field": "S45", "flown_for": "satp1"}}
    assert I.scan_rows(IMPACT, 10.0, DAYS, approach, True)[0]["when"].endswith("outside the 10.0° angle")
    assert I.scan_rows(IMPACT, 10.0, DAYS.clear(), None, False)[0]["when"] == "not recorded — older run"


def test_the_rule_in_words():
    assert "own `exclusion_deg`, 12°" in I.rule_text({"telescopes": {"LAT": {"exclusion_deg": 12}}}, "LAT")
    assert I.rule_text({"drone_source": {"exclusion_rule": "fixed", "fixed_deg": 8}}, "LAT").startswith("the `fixed` rule")
    assert "source_beam" in I.rule_text({}, "LAT")
