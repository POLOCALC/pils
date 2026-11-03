"""The flight's clocks (pils.analyze.timing), on a synthetic packaged flight."""

import numpy as np
import polars as pl
import pytest

from pils import store
from pils.analyze import timing

T0 = 1_764_512_000.0
RATE = 100.0


def track(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A drone going back and forth at 5 m/s, 30 m up, with a climb and a descent."""
    x = 150 * np.sin(2 * np.pi * (t - T0) / 120)  # m east
    y = 60 * np.sin(2 * np.pi * (t - T0) / 77)  # m north
    h = 4000 + np.clip(t - T0 - 10, 0, 30) - np.clip(t - T0 - 560, 0, 30)
    lat = -22.96 + y / 6_371_000 * 180 / np.pi
    lon = -67.79 + x / (6_371_000 * np.cos(np.radians(-22.96))) * 180 / np.pi
    return lat, lon, h


def packaged(tmp_path, lag_s=0.0, height_error=False, telescope=True):
    """A flight folder with its sync (drone ``lag_s`` behind the reference) and telescope data."""
    flight = tmp_path / "flight_20251130_1400"
    t = T0 + np.arange(0, 600, 1 / RATE)
    lat, lon, h = track(t)
    dlat, dlon, dh = track(t - lag_s)
    if (
        height_error
    ):  # a height wrong with the motion: what pulls an offset from positions
        dh = dh + 3 * np.cos(2 * np.pi * (t - T0) / 120)
    drone = pl.DataFrame(
        {"timestamp": t, "lat_deg": dlat, "lon_deg": dlon, "h_ell_m": dh}
    )
    reference = pl.DataFrame(
        {
            "timestamp": t,
            "lat_deg": lat,
            "lon_deg": lon,
            "h_ell_m": h,
            "time_acc_ns": np.full(t.size, 20.0),
        }
    )
    meta = {
        "reference": "payload_gps",
        "time_offset_details": {
            "drone_gps": {"time_offset": 0.5, "correlation": 0.9999},
            "payload_clock": {"time_offset": 0.0, "method": "none"},
        },
    }
    store.save_version(
        flight / "proc" / "flight" / "sync",
        "rev_20261101_000000",
        {"drone": drone, "reference_gps": reference},
        meta,
        kind="flight_sync",
    )
    if telescope:
        tt = np.concatenate(
            [np.arange(T0, T0 + 300, 0.05), np.arange(T0 + 305, T0 + 600, 0.05)]
        )
        store.save_version(
            flight / "proc" / "telescope",
            "rev_20261101_000000",
            {
                "pointing": pl.DataFrame(
                    {"timestamp": tt, "az_enc_deg": np.zeros(tt.size)}
                )
            },
            {},
            parent="SATP1",
            kind="telescope_pointing",
        )
        x, y, _ = 150 * np.sin(2 * np.pi * (t - T0) / 120), 60.0 + 0 * t, None
        az = np.degrees(np.arctan2(x, y + 500)) % 360
        store.save_version(
            flight / "proc" / "azel",
            "rev_20261101_000000",
            {
                "azel_data": pl.DataFrame(
                    {"timestamp": t, "az_deg": az, "el_deg": np.full(t.size, 10.0)}
                )
            },
            {},
            parent="SATP1",
        )
    return flight


def by_name(result):
    return {c["name"]: c for c in result["checks"]}


def test_not_packaged(tmp_path):
    out = timing.check(tmp_path / "flight_x")
    assert out["status"] == "none" and out["checks"][0]["name"] == "Synchronised data"


def test_a_flight_in_time(tmp_path):
    out = timing.check(packaged(tmp_path))
    checks = by_name(out)
    assert (
        checks["Reference clock"]["status"] == "ok"
        and "20 ns" in checks["Reference clock"]["detail"]
    )
    assert checks["Drone time"]["status"] == "ok"
    assert abs(out["agreement"]["offset_s"]) < 0.01
    assert checks["Drift"]["status"] == "ok" and len(out["drift"]) == timing.SEGMENTS
    assert checks["Offset of payload_clock"]["status"] == "warn"  # none found
    # The telescope's 5 s gap, in the airborne time
    assert (
        checks["SATP1 time"]["status"] == "warn"
        and "1 gaps" in checks["SATP1 time"]["detail"]
    )
    assert out["telescopes"][0]["longest_gap_s"] == pytest.approx(5.05, abs=0.01)
    sensitivity = out["sensitivity"][0]
    assert sensitivity["telescope"] == "SATP1" and sensitivity["arcsec_per_ms"] > 0
    assert out["status"] == "warn"


def test_a_drone_left_late_by_a_height_error(tmp_path):
    # The drone 0.4 s behind and a height wrong with the motion: positions mix the two, the
    # horizontal velocities find the 0.4 s
    out = timing.check(
        packaged(tmp_path, lag_s=0.4, height_error=True, telescope=False)
    )
    checks = by_name(out)
    assert out["agreement"]["offset_s"] == pytest.approx(0.4, abs=0.02)
    assert checks["Drone time"]["status"] == "bad"
    assert (
        "+400 ms" in checks["Drone time"]["detail"]
        or "+399 ms" in checks["Drone time"]["detail"]
    )
    assert checks["Telescope time"]["status"] == "none"
    assert out["status"] == "bad"


def test_no_reference(tmp_path):
    flight = packaged(tmp_path, telescope=False)
    sync = flight / "proc" / "flight" / "sync" / "rev_20261101_000000"
    (sync / "reference_gps.parquet").unlink()
    manifest = store.read_manifest(sync)
    del manifest["tables"]["reference_gps"]
    import json

    (sync / "manifest.json").write_text(json.dumps(manifest))
    checks = by_name(timing.check(flight))
    assert checks["Reference clock"]["status"] == "bad"
    assert checks["Drone time"]["status"] == "none"
