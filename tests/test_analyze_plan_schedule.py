"""A planned campaign's schedule shaped for its plots (pils.analyze.plan)."""

import json

import polars as pl
import pytest

from pils.analyze.plan import PlanRun, project_name
from pils.analyze.plan import schedule as S

SCHEDULE = """day,date,start,end,telescope,project,field,wafer,section,boresight,flight,band_crossed,drone_az
0,2026-11-29,08:00,08:30,,COMMISSIONING,,,,,,,
0,2026-11-29,09:00,09:30,satp1,POLOCALC,S45,ws3,1,0,1,0.62,10
0,2026-11-29,09:30,10:00,satp3,HOVERCAL,S45,ws0,0,45,2,0.5,200
1,2026-11-30,09:00,09:30,satp1,POLOCALC_ENG_PRE,S45,ws0,2,0,3,0.7,350
1,2026-11-30,09:30,10:00,satp1,HOVERCAL,S45,ws1,0,0,4,,
"""


@pytest.fixture
def run(tmp_path):
    plan = tmp_path / "dec26" / "plan"
    plan.mkdir(parents=True)
    (plan / "campaign_schedule.csv").write_text(SCHEDULE)
    (plan / "campaign_summary.csv").write_text(
        "project,telescope,n_flights,hours,block_hours\n"
        "POLOCALC_ENG_PRE,satp1,2,1.0,1.0\nPOLOCALC_ENG_POST,satp1,3,1.5,1.5\nALL,,5,2.5,2.5\n")
    (plan / "gaps.csv").write_text(
        "date,day,reason,start,end,start_block,n_blocks,minutes\n"
        "2026-11-29,0,free,10:00,11:00,4,2,60\n2026-11-29,0,Sun,11:00,12:00,6,2,60\n"
        "2026-11-30,1,Sun,10:00,10:30,2,1,30\n")
    (plan / "availability.csv").write_text(
        "date,time,day,block,telescope,field,n_safe_sections\n"
        "2026-11-29,09:00,0,0,satp1,S45,3\n2026-11-29,09:30,0,1,satp1,S45,0\n2026-11-29,10:00,0,2,satp1,S45,0\n"
        "2026-11-29,10:30,0,3,satp1,S45,2\n2026-11-30,09:00,1,0,satp1,S45,1\n2026-11-30,09:30,1,1,satp1,S45,0\n")
    (plan / "run.json").write_text(json.dumps({"config": "configs/dec26.yaml"}))
    (plan / "dec26.yaml").write_text(
        "projects:\n  HOVERCAL:\n    window: [2026-11-29, 2026-11-30]\n"
        "    share: {mode: min, fraction: 0.5, daily: {min: 0.4, max: 0.9}}\n"
        "  POLOCALC_ENG_PRE:\n    share_with: HOVERCAL\n  POLOCALC: {}\n")
    return PlanRun(tmp_path / "dec26")


def test_the_engineering_windows_are_one_project(run):
    assert project_name("POLOCALC_ENG_POST") == "POLOCALC_ENG" and project_name(None) is None
    summary = run.table("campaign_summary.csv")
    assert summary.rows() == [("POLOCALC_ENG", "satp1", 5, 2.5, 2.5), ("ALL", None, 5, 2.5, 2.5)]
    assert PlanRun(run.path, merge=False).table("campaign_summary.csv").height == 3
    assert run.name == "dec26" and run.config["projects"]["HOVERCAL"]["share"]["fraction"] == 0.5
    assert run.table("nothing.csv").is_empty()


def test_the_gantt_boxes_and_their_colours(run):
    sched = run.table("campaign_schedule.csv")
    boxes = S.gantt_flights(sched, "project")
    assert boxes["label"].to_list() == ["", "3.1", "0.0", "0.2", "1.0"]
    assert S.colour_keys(boxes) == ["COMMISSIONING/—", "POLOCALC/satp1", "HOVERCAL/satp3",
                                    "POLOCALC_ENG/satp1", "HOVERCAL/satp1"]
    assert S.colour_keys(S.gantt_flights(sched, "project_total")) == [
        "COMMISSIONING", "POLOCALC", "HOVERCAL", "POLOCALC_ENG"]
    assert S.colour_keys(S.gantt_flights(sched, "field")) == ["—", "S45"]
    assert S.days(sched) == ["2026-11-29", "2026-11-30"]
    with pytest.raises(ValueError):
        S.gantt_flights(sched, "wafer")


def test_the_empty_blocks(run):
    sched, gaps = run.table("campaign_schedule.csv"), run.table("gaps.csv")
    assert S.gap_bands(gaps)["reason"].to_list() == ["Sun", "Sun", "free"]
    assert S.gap_summary(gaps).rows() == [("Sun", 90, 2), ("free", 60, 1)]
    # the Sun closing the field flown that day; a stretch open at the day's end lasts its block
    assert S.sun_blocked(sched, run.table("availability.csv")).rows() == [
        ("2026-11-29", "09:30", "10:30"), ("2026-11-30", "09:30", "10:00")]


def test_a_shared_window_day_by_day(run):
    [w] = S.share_windows(run.config)
    assert (w["project"], w["from"], w["to"]) == ("HOVERCAL", "2026-11-29", "2026-11-30")
    assert w["members"] == ["HOVERCAL", "POLOCALC_ENG"]
    sched = run.table("campaign_schedule.csv")
    assert S.share_days(sched, w).rows() == [
        ("2026-11-29", 1, 1, 2, 0.5, "yes"), ("2026-11-30", 2, 0, 2, 1.0, "NO")]
    assert S.in_window(sched, {"from": "2026-11-30", "to": "2026-12-01"}).height == 2


def test_the_flights_table(run):
    table = S.flights_table(run.table("campaign_schedule.csv"))
    assert table.columns[:3] == ["flight", "date", "start"] and "drone_az" not in table.columns
    assert table["drone_heading_deg"].to_list()[1:4] == [190.0, 20.0, 170.0]
    assert table["band_crossed"].to_list()[1] == pytest.approx(62.0)
    assert isinstance(table, pl.DataFrame)
