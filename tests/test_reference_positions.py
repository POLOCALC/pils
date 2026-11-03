"""Telescope and DJI base positions: campaign survey, other surveys, so3g."""

import sys
import types

import pytest

from pils.sensors import emlid

HEADER = "Name,Longitude,Latitude,Ellipsoidal height\n"
BASE = "Emlid base,-67.78685015,-22.95982209,5182.212\n"
DJI = "DJI rtk base (antenna base),-67.786697,-22.959775,5182.214\n"


def survey(campaign, rows):
    folder = campaign / "metadata"
    folder.mkdir(parents=True)
    (folder / f"{campaign.name}_coordinates.csv").write_text(HEADER + "".join(rows))


def sat(name, lat, lon=-67.7883810, h=5184.89):
    return f"{name} 1,{lon},{lat},{h}\n{name} 2,{lon},{lat + 2e-6},{h}\n"


@pytest.fixture
def campaigns(tmp_path):
    root = tmp_path / "campaigns"
    survey(root / "202511", [BASE, DJI, sat("SATP1", -22.96009)])
    survey(root / "202504", [BASE, sat("SATP1", -22.96000)])  # older survey
    (root / "202412").mkdir()  # no survey
    return root


def test_campaign_survey(campaigns):
    found = emlid.reference_positions(campaigns / "202511", "SATP1")
    assert found.telescope_source == "campaign survey"
    assert found.telescope["lat"] == pytest.approx(-22.960089, abs=1e-6)
    assert found.dji_base["alt"] == pytest.approx(5182.214)
    assert found.notes == []


def test_newest_other_survey_and_no_base(campaigns):
    found = emlid.reference_positions(campaigns / "202412", "SATP1")
    assert found.telescope_source == "survey 202511"  # newest first
    assert found.dji_base is None
    assert any("202511 survey" in n for n in found.notes)
    assert any("not corrected" in n for n in found.notes)
    meta = found.as_metadata()
    assert meta["telescope_position_source"] == "survey 202511"
    assert meta["base_position_source"] is None


def test_campaign_without_the_telescope(campaigns):
    # 202504 has no DJI base and SATP1 only: SATP2 is in no survey
    with pytest.raises(ValueError, match="No telescope positions for 'SATP2'"):
        emlid.reference_positions(campaigns / "202504", "SATP2")


def test_so3g(campaigns, monkeypatch):
    site = types.SimpleNamespace(lon=-67.78836, lat=-22.96011, elev=5188.0)
    proj = types.ModuleType("so3g.proj")
    proj.SITES = {"so_sat1": site}
    monkeypatch.setitem(sys.modules, "so3g", types.ModuleType("so3g"))
    monkeypatch.setitem(sys.modules, "so3g.proj", proj)
    found = emlid.reference_positions(
        campaigns / "202511", "SATP1", telescope_source="so3g"
    )
    assert found.telescope_source == "so3g so_sat1"
    assert found.telescope["alt"] == 5188.0
    assert found.dji_base is not None  # the campaign's base is still used
    with pytest.raises(ValueError, match="No so3g site"):
        emlid.reference_positions(campaigns / "202511", "XYZ", telescope_source="so3g")


def test_unknown_source(campaigns):
    with pytest.raises(ValueError, match="telescope_source"):
        emlid.reference_positions(campaigns / "202511", "SATP1", "sotodlib")
