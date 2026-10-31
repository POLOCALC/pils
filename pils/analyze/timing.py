"""A flight's clocks: how well its data are tied to GPS time, and how much a timing error costs.

The drone's position is put on the payload GPS's time (pils sync: each source's offset from the
correlation of its positions or rates with the reference); the telescopes' pointing is on their own
GPS-disciplined clocks. Checked here, from what a packaged flight has:

- the reference: a payload GPS (its own time accuracy), or none -- then nothing ties the drone's
  clock to GPS time;
- each offset found by the sync: how well the sources correlated (a poor one is a doubtful offset),
  the payload clock's method;
- agreement: the drone's offset from the reference again, after the sync, from the horizontal
  velocities -- a fixed position difference (lever arm, a standalone receiver's metres) cancels in
  them, and a height that is wrong with the motion does not enter: what is left should be zero;
- drift: that residual offset in successive stretches of the flight -- a clock that runs at another
  rate shows as a change across them;
- each telescope's pointing (``proc/telescope/<T>``): time going forward, its rate, gaps and how much
  of the airborne time it covers;
- what a timing error costs: the drone's angular speed as each telescope saw it
  (``proc/azel/<T>``), so the arcseconds per millisecond, and the drift's in arcminutes.

The drone's clock against the telescope's directly needs the detectors' data: the time the source
crossed a detector in its data against the time predicted from the drone's position.

    check(flight_dir) -> dict      (``status``: ok, warn or bad, and ``checks``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pils import store

MIN_CORRELATION = 0.9
# Change of the drone's residual offset across the flight above which its clock looks to drift
MAX_DRIFT_S = 0.02
# Residual offset (s) after the sync above which the drone's time is doubtful, and wrong
MAX_RESIDUAL_S = 0.05
BAD_RESIDUAL_S = 0.2
# The payload GPS's own time accuracy above which its time is doubtful
MAX_TIME_ACC_NS = 1_000.0
MIN_COVERAGE_PCT = 95.0
AIRBORNE_HEIGHT_M = 1.0
SEGMENTS = 3
MIN_SEGMENT_S = 30.0
ORDER = {"none": 0, "ok": 1, "warn": 2, "bad": 3}


def _latest(root: Path, parent: str | None = None) -> Path | None:
    base = root / parent if parent else root
    name = store.latest_version(base) if base.is_dir() else None
    return base / name if name else None


def _manifest(directory: Path) -> dict[str, Any]:
    try:
        return store.read_manifest(directory)
    except (OSError, ValueError):
        return {}


def airborne(times: np.ndarray, height: np.ndarray) -> tuple[float, float] | None:
    """Longest stretch more than 1 m above the take-off height (the first samples' median)."""
    ok = np.isfinite(times) & np.isfinite(height)
    times, height = times[ok], height[ok]
    if times.size < 10:
        return None
    ground = float(np.median(height[: max(10, times.size // 100)]))
    above = (height - ground) > AIRBORNE_HEIGHT_M
    if not above.any():
        return None
    edges = np.diff(np.concatenate([[0], above.astype(int), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    k = int(np.argmax(times[stops] - times[starts]))
    return float(times[starts[k]]), float(times[stops[k]])


def _enu(frame: pl.DataFrame, ref: tuple[float, float, float]) -> np.ndarray:
    """East, north, up (m) of a table's lat_deg/lon_deg/h_ell_m from a point, small-area."""
    radius = 6_371_000.0
    lat = np.radians(frame["lat_deg"].to_numpy().astype(float))
    lon = np.radians(frame["lon_deg"].to_numpy().astype(float))
    lat0, lon0 = np.radians(ref[0]), np.radians(ref[1])
    return np.array(
        [
            radius * np.cos(lat0) * (lon - lon0),
            radius * (lat - lat0),
            frame["h_ell_m"].to_numpy().astype(float) - ref[2],
        ]
    )


def _usable(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(
        pl.col("lat_deg").is_not_null()
        & pl.col("lat_deg").is_not_nan()
        & (pl.col("lat_deg") != 0)
        & pl.col("h_ell_m").is_not_null()
    )


def _velocity(
    frame: pl.DataFrame, ref: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return times and east/north/up velocities (m/s), positions smoothed over 1 s."""
    t = frame["timestamp"].to_numpy().astype(float)
    enu = _enu(frame, ref)
    k = max(1, int(round(1.0 / float(np.median(np.diff(t)))))) if t.size > 1 else 1
    kernel = np.ones(k) / k
    smooth = np.array([np.convolve(x, kernel, mode="same") for x in enu])
    velocity = np.array([np.gradient(x, t) for x in smooth])
    return t[k:-k], velocity[:, k:-k]


def residual(
    drone: pl.DataFrame,
    reference: pl.DataFrame,
    window: tuple[float, float],
    ref: tuple[float, float, float],
) -> dict[str, Any] | None:
    """Find the drone's offset from the reference left after the sync (s), from velocities.

    ``offset_s`` from the horizontal velocities, ``correlation``, and per axis (east, north, up)
    the offsets from the velocities and from the positions, for comparison.
    """
    from pils.synchronizer import Synchronizer

    a = drone.filter(pl.col("timestamp").is_between(*window))
    b = reference.filter(pl.col("timestamp").is_between(*window))
    if a.height < 300 or b.height < 300:
        return None
    ta, va = _velocity(a, ref)
    tb, vb = _velocity(b, ref)
    found = Synchronizer._estimate_time_offset(ta, va[:2], tb, vb[:2])
    if found is None:
        return None
    out: dict[str, Any] = {"offset_s": float(found[0]), "correlation": float(found[3])}
    pa, pb = _enu(a, ref), _enu(b, ref)
    ta_all, tb_all = a["timestamp"].to_numpy(), b["timestamp"].to_numpy()
    for i, axis in enumerate("enu"):
        by_velocity = Synchronizer._estimate_time_offset(
            ta, va[i : i + 1], tb, vb[i : i + 1]
        )
        by_position = Synchronizer._estimate_time_offset(
            ta_all, pa[i : i + 1], tb_all, pb[i : i + 1]
        )
        out[f"velocity_{axis}_s"] = float(by_velocity[0]) if by_velocity else None
        out[f"position_{axis}_s"] = float(by_position[0]) if by_position else None
    return out


def drift(
    drone: pl.DataFrame, reference: pl.DataFrame, window: tuple[float, float]
) -> tuple[dict[str, Any] | None, list[dict[str, float]]]:
    """Find the residual offset over the airborne time, and in successive stretches of it.

    Only stretches where the velocities correlate (the drone moving) give one.
    """
    drone, reference = _usable(drone), _usable(reference)
    if drone.height < 300 or reference.height < 300:
        return None, []
    ref = (
        float(reference["lat_deg"][0]),
        float(reference["lon_deg"][0]),
        float(reference["h_ell_m"][0]),
    )
    whole = residual(drone, reference, window, ref)
    start, end = window
    step = (end - start) / SEGMENTS
    stretches = []
    if step >= MIN_SEGMENT_S:
        for k in range(SEGMENTS):
            lo, hi = start + k * step, start + (k + 1) * step
            found = residual(drone, reference, (lo, hi), ref)
            if found is None or found["correlation"] < MIN_CORRELATION:
                continue
            stretches.append(
                {
                    "start": lo,
                    "end": hi,
                    "residual_s": found["offset_s"],
                    "correlation": found["correlation"],
                }
            )
    return whole, stretches


def angular_speed(azel: pl.DataFrame, window: tuple[float, float] | None) -> np.ndarray:
    """Return the drone's angular speed (deg/s) as a telescope saw it, over 1 s steps."""
    frame = azel.filter(pl.col("az_deg").is_not_nan() & pl.col("el_deg").is_not_nan())
    if window is not None:
        frame = frame.filter(pl.col("timestamp").is_between(*window))
    if frame.height < 3:
        return np.array([])
    t = frame["timestamp"].to_numpy()
    grid = np.arange(t[0], t[-1], 1.0)
    if grid.size < 3:
        return np.array([])
    az = np.radians(
        np.interp(
            grid,
            t,
            np.unwrap(np.radians(frame["az_deg"].to_numpy()), period=2 * np.pi)
            * 180
            / np.pi,
        )
    )
    el = np.radians(np.interp(grid, t, frame["el_deg"].to_numpy()))
    v = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    cos = np.clip(np.sum(v[:, 1:] * v[:, :-1], axis=0), -1, 1)
    return np.degrees(np.arccos(cos))  # per 1 s step


def _pointing(directory: Path, window: tuple[float, float] | None) -> dict[str, Any]:
    from pils.analyze.quicklook import table_summary

    times = pl.read_parquet(directory / "pointing.parquet", columns=["timestamp"])[
        "timestamp"
    ]
    times = times.to_numpy().astype(float)
    backwards = int(np.sum(np.diff(times) <= 0))
    summary = table_summary(times, window)
    # Gaps inside the airborne time only
    inside = (
        np.sort(times[(times >= window[0]) & (times <= window[1])]) if window else times
    )
    steps = np.diff(inside)
    limit = (
        max(1.0, 5 * float(np.median(np.diff(np.sort(times)))))
        if times.size > 1
        else 1.0
    )
    gaps = steps[steps > limit]
    return {
        "samples": summary["samples"],
        "rate_hz": summary["rate_hz"],
        "coverage_pct": summary["coverage_pct"],
        "gaps": int(gaps.size),
        "longest_gap_s": float(gaps.max()) if gaps.size else 0.0,
        "backwards": backwards,
    }


def check(flight_dir: str | Path) -> dict[str, Any]:
    """Check the flight's timing (see the module).

    Returns ``status`` (the worst of the checks: ``ok``, ``warn``, ``bad``; ``none`` when the
    flight is not packaged), ``checks`` (``name``, ``status``, ``detail``), ``airborne``,
    ``reference``, ``offsets``, ``agreement``, ``drift``, ``telescopes`` and ``sensitivity``.
    """
    flight_dir = Path(flight_dir)
    checks: list[dict[str, str]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    sync = _latest(flight_dir / "proc" / "flight" / "sync")
    out: dict[str, Any] = {
        "revision": sync.name if sync else None,
        "checks": checks,
        "airborne": None,
        "reference": None,
        "offsets": [],
        "agreement": None,
        "drift": [],
        "telescopes": [],
        "sensitivity": [],
    }
    if sync is None:
        add("Synchronised data", "none", "The flight is not packaged (pils package).")
        out["status"] = "none"
        return out
    meta = _manifest(sync).get("metadata") or {}
    tables = (_manifest(sync).get("tables") or {}).keys()
    drone = pl.read_parquet(sync / "drone.parquet") if "drone" in tables else None
    window = (
        airborne(
            drone["timestamp"].to_numpy(), drone["h_ell_m"].to_numpy().astype(float)
        )
        if drone is not None and "h_ell_m" in drone.columns
        else None
    )
    out["airborne"] = list(window) if window else None

    # The reference clock
    reference = meta.get("reference")
    out["reference"] = reference
    ref_table = (
        pl.read_parquet(sync / "reference_gps.parquet")
        if "reference_gps" in tables
        else None
    )
    if ref_table is None:
        add(
            "Reference clock",
            "bad",
            "No payload GPS: nothing ties the drone's clock to GPS time.",
        )
    else:
        accuracy = (
            ref_table["time_acc_ns"].drop_nulls().drop_nans()
            if "time_acc_ns" in ref_table.columns
            else pl.Series([], dtype=pl.Float64)
        )
        if accuracy.len():
            median = float(accuracy.median())
            add(
                "Reference clock",
                "ok" if median <= MAX_TIME_ACC_NS else "warn",
                f"Payload GPS, its time to {median:.0f} ns (median of its own estimate).",
            )
        else:
            add(
                "Reference clock",
                "ok",
                "Payload GPS (no time accuracy in its records).",
            )

    # The offsets the sync found
    details = meta.get("time_offset_details") or {}
    for source, found in details.items():
        corr = found.get("correlation")
        method = found.get("method", "correlation")
        out["offsets"].append(
            {
                "source": source,
                "offset_s": found.get("time_offset"),
                "correlation": corr,
                "method": method,
            }
        )
        if method == "none":
            add(
                f"Offset of {source}",
                "warn",
                "None found: its clock is taken as GPS time.",
            )
        elif corr is not None and corr < MIN_CORRELATION:
            add(
                f"Offset of {source}",
                "warn",
                f"{found.get('time_offset', float('nan')):+.3f} s, from a poor correlation ({corr:.2f}).",
            )
        else:
            add(
                f"Offset of {source}",
                "ok",
                f"{found.get('time_offset', float('nan')):+.3f} s"
                + (f", correlation {corr:.3f}" if corr is not None else "")
                + (f" ({method})" if method not in ("correlation",) else "")
                + ".",
            )

    # What is left of the drone's offset after the sync, and its drift across the flight
    spread = left = None
    if drone is not None and ref_table is not None and window is not None:
        whole, stretches = drift(drone, ref_table, window)
        out["agreement"], out["drift"] = whole, stretches
        if whole is not None and whole["correlation"] >= MIN_CORRELATION:
            left = whole["offset_s"]
            add(
                "Drone time",
                "ok"
                if abs(left) <= MAX_RESIDUAL_S
                else "warn"
                if abs(left) <= BAD_RESIDUAL_S
                else "bad",
                f"After the sync the drone's velocities still put it {left * 1e3:+.0f} ms off the payload "
                "GPS"
                + (
                    "."
                    if abs(left) <= MAX_RESIDUAL_S
                    else " (east, north, up: "
                    + ", ".join(
                        f"{whole[f'velocity_{a}_s'] * 1e3:+.0f}"
                        for a in "enu"
                        if whole.get(f"velocity_{a}_s") is not None
                    )
                    + " ms; from the positions: "
                    + ", ".join(
                        f"{whole[f'position_{a}_s'] * 1e3:+.0f}"
                        for a in "enu"
                        if whole.get(f"position_{a}_s") is not None
                    )
                    + " ms)."
                ),
            )
        else:
            add(
                "Drone time", "none", "Too little movement to check the drone's offset."
            )
        if len(stretches) >= 2:
            residuals = [s["residual_s"] for s in stretches]
            spread = max(residuals) - min(residuals)
            add(
                "Drift",
                "ok" if spread <= MAX_DRIFT_S else "warn",
                f"The drone's offset changes by {spread * 1e3:.0f} ms across the flight "
                f"({len(stretches)} stretches).",
            )
        else:
            add("Drift", "none", "Too little movement to follow the drone's offset.")
    else:
        add(
            "Drone time",
            "none",
            "No drone track, payload GPS or airborne time to check it.",
        )

    # The telescopes' pointing
    telescope_root = flight_dir / "proc" / "telescope"
    names = (
        sorted(p.name for p in telescope_root.iterdir() if p.is_dir())
        if telescope_root.is_dir()
        else []
    )
    for name in names:
        directory = _latest(telescope_root, name)
        if directory is None or not (directory / "pointing.parquet").is_file():
            continue
        found = {
            "telescope": name,
            "revision": directory.name,
            **_pointing(directory, window),
        }
        out["telescopes"].append(found)
        if found["backwards"]:
            add(
                f"{name} time",
                "bad",
                f"Its timestamps go backwards {found['backwards']} times.",
            )
            continue
        coverage = found["coverage_pct"]
        problems = []
        if np.isfinite(coverage) and coverage < MIN_COVERAGE_PCT:
            problems.append(f"covers {coverage:.0f}% of the airborne time")
        if found["gaps"]:
            problems.append(
                f"{found['gaps']} gaps in the airborne time, the longest {found['longest_gap_s']:.1f} s"
            )
        add(
            f"{name} time",
            "warn" if problems else "ok",
            ("Its pointing " + "; ".join(problems) + ".")
            if problems
            else f"Its pointing at {found['rate_hz']:.0f} Hz covers the airborne time, no gaps.",
        )
    if not names:
        add(
            "Telescope time",
            "none",
            "No telescope pointing for this flight (pils telescope).",
        )

    # What a timing error costs, per telescope
    azel_root = flight_dir / "proc" / "azel"
    for name in (
        sorted(p.name for p in azel_root.iterdir() if p.is_dir())
        if azel_root.is_dir()
        else []
    ):
        directory = _latest(azel_root, name)
        if directory is None or not (directory / "azel_data.parquet").is_file():
            continue
        speed = angular_speed(pl.read_parquet(directory / "azel_data.parquet"), window)
        if not speed.size:
            continue
        p95, top = float(np.percentile(speed, 95)), float(speed.max())
        out["sensitivity"].append(
            {
                "telescope": name,
                "speed_p95_deg_s": p95,
                "speed_max_deg_s": top,
                # 1 ms at the 95th percentile speed
                "arcsec_per_ms": p95 * 3600 / 1000,
                "drift_arcmin": spread * p95 * 60 if spread is not None else None,
                "residual_arcmin": abs(left) * p95 * 60 if left is not None else None,
            }
        )

    worst = max((c["status"] for c in checks), key=lambda s: ORDER[s], default="none")
    out["status"] = worst
    return out
