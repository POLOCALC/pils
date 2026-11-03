"""A flown flight against its planned one (pils.analyze.plan_compare)."""

import numpy as np
import polars as pl
import pytest

from pils import store
from pils.analyze.coverage import describe_telescope
from pils.analyze.plan_compare import compare
from pils.geometry.focal_plane import DEFAULT_WAFERS

T0 = 1_795_781_400.0  # the planned take-off
REV = "rev_20261101_000000"


def _azel(folder, t, az, el, e, n, u):
    store.save_version(
        folder / "proc" / "azel" / "SATP3",
        REV,
        {
            "azel_data": pl.DataFrame(
                {
                    "timestamp": t,
                    "az_deg": az,
                    "el_deg": el,
                    "e_m": e,
                    "n_m": n,
                    "u_m": u,
                }
            )
        },
        {},
    )


def _scan(t, up_m=0.0):
    """A scan at 480 m due south-ish, 330 to 350 m up, sideways along the azimuth."""
    s = (t - t[0]) / (t[-1] - t[0])
    bearing = np.radians(183.0 + 4 * s)
    e = 480 * np.sin(bearing)
    n = 480 * np.cos(bearing)
    u = 330 + 20 * s + up_m
    az = np.degrees(np.arctan2(e, n)) % 360
    el = np.degrees(np.arctan2(u, np.hypot(e, n)))
    return az, el, e, n, u


@pytest.fixture
def flights(tmp_path):
    planned = tmp_path / "plans" / "p" / "flights" / "plan_20261127_0910"
    t = T0 + np.arange(0, 420, 0.1)
    az, el, e, n, u = _scan(t)
    _azel(planned, t, az, el, e, n, u)
    store.save_version(
        planned / "proc" / "plan",
        REV,
        {
            "telescope": pl.DataFrame(
                {
                    "timestamp": t,
                    "az_encoder_deg": np.full(t.size, 185.0),
                    "el_deg": np.full(t.size, 35.0),
                    "boresight_deg": np.zeros(t.size),
                }
            )
        },
        {
            "takeoff": T0,
            "row": {
                "telescope": "satp3",
                "field": "S45",
                "wafer": "ws0",
                "source": "90",
            },
        },
        kind="flight_plan",
    )
    # Flown 10 minutes late, 3 m above the plan: a climb, the scan, a descent
    flown = tmp_path / "flight_20261127_0920"
    late = 600.0
    climb = T0 + late + np.arange(0, 60, 0.1)
    scan = climb[-1] + 0.1 + np.arange(0, 420, 0.1)
    down = scan[-1] + 0.1 + np.arange(0, 60, 0.1)
    az, el, e, n, u = _scan(scan, up_m=3.0)
    t = np.concatenate([climb, scan, down])
    u_all = np.concatenate(
        [np.linspace(0, 330, climb.size), u, np.linspace(350, 0, down.size)]
    )

    def pad(first, x, last):
        return np.concatenate([np.full(climb.size, first), x, np.full(down.size, last)])

    _azel(
        flown,
        t,
        pad(az[0], az, az[-1]),
        pad(0.0, el, 0.0),
        pad(e[0], e, e[-1]),
        pad(n[0], n, n[-1]),
        u_all,
    )
    store.save_version(
        flown / "proc" / "flight" / "sync",
        REV,
        {"drone": pl.DataFrame({"timestamp": t, "h_ell_m": 4000 + u_all})},
        {},
        kind="flight_sync",
    )
    return flown, planned


def test_when_and_where(flights):
    flown, planned = flights
    out = compare(flown, planned)
    assert out["telescope"] == "SATP3" and out["plan"]["field"] == "S45"
    when = out["when"]
    assert when["minutes_late"] == pytest.approx(10.0, abs=0.2)
    assert when["planned_scan_minutes"] == pytest.approx(7.0, abs=0.01)
    # At the scan's height: the scan and the top of the climb and the descent
    assert 7.0 <= when["flown_scan_minutes"] < 7.2
    assert when["flown_minutes"] == pytest.approx(9.0, abs=0.2)
    path = out["path"]
    # The nearest planned point: 3 m up, less along the climbing path (20 m up in 33.5 m)
    along = 480 * np.radians(4)
    assert path["deviation_m"]["median"] == pytest.approx(
        3 * along / np.hypot(along, 20), abs=0.05
    )
    # In angle: 3 m up, 35 deg up at 588 m, and as much less along the path
    angle = 3 * np.cos(np.radians(35)) / np.hypot(480, 340) * 180 / np.pi * 60
    assert path["deviation_arcmin"]["median"] == pytest.approx(
        angle * along / np.hypot(along, 20), rel=0.1
    )
    assert path["planned_reached_pct"] == pytest.approx(100.0)
    assert path["flown_off_pct"] < 5
    assert out["sky"]["planned_az_deg"] == pytest.approx([183.0, 187.0])
    assert out["coverage"] is None and out["problems"] == []


def test_detectors_planned_and_crossed(flights):
    pytest.importorskip("so3g")
    flown, planned = flights
    description = describe_telescope(
        "satp3", wafers=DEFAULT_WAFERS, wafer_type="SAT_MF", own_wafers=False
    )
    # The flown flight's telescope pointed as planned
    store.save_version(
        flown / "proc" / "telescope" / "SATP3",
        REV,
        {
            "pointing": pl.DataFrame(
                {
                    "timestamp": np.arange(T0, T0 + 1500, 0.1),
                    "az_enc_deg": np.full(15000, 185.0),
                    "el_enc_deg": np.full(15000, 35.0),
                    "boresight_enc_deg": np.zeros(15000),
                    "flagged": np.zeros(15000, bool),
                }
            )
        },
        {},
    )
    coverage = compare(flown, planned, description)["coverage"]
    assert coverage["planned"] > 0 and coverage["band"] == "SAT_f090"
    # 3 m at 480 m is 21 arcmin: most detectors planned are crossed still
    assert coverage["both"] + coverage["missed"] == coverage["planned"]
    assert coverage["planned_crossed_pct"] > 50


def test_what_cannot_be_compared(flights, tmp_path):
    flown, planned = flights
    with pytest.raises(ValueError, match="not a planned flight"):
        compare(flown, flown)
    out = compare(tmp_path / "nothing", planned)
    assert out["problems"] == ["The flight has no azel/SATP3 (pils package)."]
