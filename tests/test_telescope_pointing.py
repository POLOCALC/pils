"""The SO SATs' own pointing during a flight (pils.telescope_pointing)."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pils import store
from pils.telescope_pointing import flight_window, observations, run, thin

REAL_FLIGHT = Path("/mnt/data/POLOCALC/campaigns/202511/20251213/flight_20251213_1511")


def books(root: Path, telescope: str, starts: list[int]) -> None:
    for t0 in starts:
        (
            root
            / telescope
            / "obs"
            / str(t0 // 100000)
            / f"obs_{t0}_{telescope}_1111111"
        ).mkdir(parents=True)


def test_the_observations_of_a_flight_are_those_that_overlap_it(tmp_path):
    books(
        tmp_path, "satp3", [1765600000, 1765630000, 1765636445, 1765640000, 1765700000]
    )
    books(tmp_path, "satp2", [1765636398])
    names = lambda obs: [o.name.split("_")[1] for o in obs]  # noqa: E731
    # a flight inside one observation: that one (it started before the flight)
    assert names(observations(tmp_path, "satp3", 1765637000, 1765637400)) == [
        "1765636445"
    ]
    # across two
    assert names(observations(tmp_path, "satp3", 1765639000, 1765641000)) == [
        "1765636445",
        "1765640000",
    ]
    # an observation of its previous folder (17656 / 17657)
    assert names(observations(tmp_path, "satp3", 1765700100, 1765700200)) == [
        "1765700000"
    ]
    assert observations(tmp_path, "satp1", 1765637000, 1765637400) == []
    assert observations(tmp_path / "nope", "satp3", 1765637000, 1765637400) == []


def test_the_pointing_is_thinned_in_time_order():
    t = np.arange(0, 2, 1 / 200.0)[::-1]  # 200 Hz, out of order
    samples = {
        "timestamp": t,
        "az": t * 10,
        "el": np.full(len(t), 60.0),
        "boresight": np.zeros(len(t)),
        "flagged": t > 1.9,
    }
    frame = thin(samples, rate_hz=20)
    assert frame.columns == [
        "timestamp",
        "az_enc_deg",
        "el_enc_deg",
        "boresight_enc_deg",
        "flagged",
    ]
    assert len(frame) == 40 and frame["timestamp"].is_sorted()
    assert np.allclose(frame["az_enc_deg"], frame["timestamp"] * 10)
    assert frame["flagged"].any()


def test_the_flight_s_window_is_its_synchronized_data(tmp_path):
    drone = pl.DataFrame(
        {"timestamp": [100.0, 150.0, 420.0], "lat_deg": [0.0, 0.0, 0.0]}
    )
    store.save_version(
        tmp_path / "proc" / "flight" / "sync",
        "rev_20260101_000000",
        {"drone": drone},
        {},
        kind="flight_sync",
    )
    assert flight_window(tmp_path) == (100.0, 420.0)
    assert flight_window(tmp_path / "nope") is None


def test_without_synchronized_data_nothing_is_read(tmp_path):
    assert [r.status for r in run(tmp_path, ["SATP1"])] == ["failed"]


@pytest.mark.skipif(not REAL_FLIGHT.is_dir(), reason="the campaign's data are not here")
def test_a_real_flight_s_pointing():
    pytest.importorskip("so3g")
    found = {r.telescope: r for r in run(REAL_FLIGHT, ["SATP3"], save=False)}
    r = found["SATP3"]
    assert r.status == "done" and r.observations == ["obs_1765636445_satp3_1111111"]
    assert r.rows > 6 * 60 * 20  # the 6-minute flight at 20 Hz, and its margins
