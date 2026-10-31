"""A plan read where it is kept: in a campaign's folder (plans/<run>/, a pils revision), as its run reads."""

import json

import polars as pl

from pils import store
from pils.analyze.plan import PlanRun


def _published(tmp_path):
    plan = tmp_path / "plans" / "dec26"
    (plan / "config").mkdir(parents=True)
    (plan / "config" / "dec26.yaml").write_text("name: dec26\ncampaign: {start_date: 2026-11-26}\n")
    (plan / "config" / "fields").mkdir()
    (plan / "config" / "fields" / "dec26.yaml").write_text("fields: {}\n")
    tables = {
        "schedule": pl.DataFrame({"flight": [1, 2], "project": ["POLOCALC_ENG_PRE", "POLOCALC"],
                                  "field": ["S45", ""], "independent": ["True", "False"]}),
        "campaign_summary": pl.DataFrame({"project": ["POLOCALC_ENG_PRE", "POLOCALC_ENG_POST"],
                                          "telescope": ["satp1", "satp1"], "n_flights": [2, 3]}),
        "impact_lat": pl.DataFrame({"target": ["RISING_SCAN_50"]}),
    }
    meta = {"run_meta": {"run": "dec26", "config": "dec26.yaml"},
            "files": {"campaign_schedule.csv": "schedule", "campaign_summary.csv": "campaign_summary",
                      "impact_LAT.csv": "impact_lat"}}
    store.save_version(plan / "proc", "rev_20261030_120000", tables, meta, parent="plan", kind="campaign_plan")
    return plan


def test_a_plan_in_a_campaign_s_folder_reads_as_its_run(tmp_path):
    run = PlanRun(_published(tmp_path))
    assert run.published and run.name == "dec26"
    assert run.names() == ["campaign_schedule.csv", "campaign_summary.csv", "impact_LAT.csv"]
    assert run.has("campaign_schedule.csv") and run.has("impact_LAT.csv") and not run.has("gaps.csv")
    sched = run.table("campaign_schedule.csv")
    # as its CSV reads: no value for an empty text, booleans for True / False, the projects merged
    assert sched["field"].to_list() == ["S45", None] and sched["independent"].dtype == pl.Boolean
    assert sched["project"].to_list() == ["POLOCALC_ENG", "POLOCALC"]
    assert run.table("campaign_summary.csv")["n_flights"].to_list() == [5]
    assert run.table("gaps.csv").is_empty()
    assert run.meta["config"] == "dec26.yaml" and run.config["name"] == "dec26"


def test_a_revision_is_read_by_its_folder(tmp_path):
    plan = _published(tmp_path)
    rev = plan / "proc" / "plan" / "rev_20261030_120000"
    assert PlanRun(rev).name == "dec26" and PlanRun(rev).has("campaign_schedule.csv")
    assert json.loads((rev / "manifest.json").read_text())["kind"] == "campaign_plan"
