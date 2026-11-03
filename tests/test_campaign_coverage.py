"""A campaign's crossings added up per detector (pils.analyze.campaign_coverage)."""

import numpy as np
import polars as pl
import pytest

from pils import store
from pils.analyze.campaign_coverage import ENOUGH, campaign_coverage, flight_telescopes
from pils.analyze.coverage import describe_telescope
from pils.geometry.focal_plane import DEFAULT_WAFERS

pytest.importorskip("so3g")

REV = "rev_20260101_000000"


def _flight(folder, boresight=0.0):
    """The drone crossing SATP3's boresight along the azimuth, in 60 s."""
    t = 1_700_000_000 + np.arange(0, 60, 0.05)
    azel = {
        "timestamp": t,
        "az_deg": np.linspace(178, 182, t.size),
        "el_deg": np.full(t.size, 50.0),
    }
    store.save_version(
        folder / "proc" / "azel" / "SATP3", REV, {"azel_data": pl.DataFrame(azel)}, {}
    )
    pointing = {
        "timestamp": t,
        "az_enc_deg": np.full(t.size, 180.0),
        "el_enc_deg": np.full(t.size, 50.0),
        "boresight_enc_deg": np.full(t.size, boresight),
        "flagged": np.zeros(t.size, bool),
    }
    store.save_version(
        folder / "proc" / "telescope" / "SATP3",
        REV,
        {"pointing": pl.DataFrame(pointing)},
        {},
    )
    return folder


def test_crossings_add_up_per_detector(tmp_path):
    satp3 = describe_telescope(
        "satp3", wafers=DEFAULT_WAFERS, wafer_type="SAT_MF", own_wafers=False
    )
    flights = [
        {"name": "a", "path": _flight(tmp_path / "a"), "source_ghz": 93.0},
        {"name": "b", "path": _flight(tmp_path / "b"), "source_ghz": 90.0},
        {
            "name": "c",
            "path": _flight(tmp_path / "c", boresight=45.0),
            "source_ghz": 93.0,
        },
        {
            "name": "d",
            "path": _flight(tmp_path / "d"),
            "source_ghz": 280.0,
        },  # not a band of SATP3
        {"name": "e", "path": _flight(tmp_path / "e"), "source_ghz": None},
        {
            "name": "f",
            "path": tmp_path / "f",
            "source_ghz": 90.0,
        },  # no data: not counted
    ]
    assert (
        flight_telescopes(tmp_path / "a") == ["SATP3"]
        and flight_telescopes(tmp_path / "f") == []
    )
    seen = []
    out = campaign_coverage(
        flights,
        {"SATP3": satp3},
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen[-1] == (5, 5)
    status = {r["flight"]: r["status"] for r in out["flights"]}
    assert status == {
        "a": "done",
        "b": "done",
        "c": "done",
        "d": "no band",
        "e": "skipped",
    }
    target = out["targets"]["SATP3 SAT_f090"]
    assert target["flights"] == ["a", "b", "c"]
    assert target["rotations"] == {"0": 2, "45": 1}
    px = target["pixels"]
    crossings, flights_crossing = np.array(px["crossings"]), np.array(px["flights"])
    assert len(px["wafer"]) == len(px["xi"]) == 7 * 430
    # The same detectors twice at 0 deg; at 45 deg the path crosses others too
    assert flights_crossing.max() == 3 and (crossings >= flights_crossing).all()
    ws0 = target["wafers"]["ws0"]
    assert ws0["detectors"] == 430 and 0 < ws0["enough"] <= ws0["crossed"] <= 430
    assert out["enough"] == ENOUGH
