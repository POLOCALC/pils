"""A flown flight against the flight planned for it (GOSE), in numbers.

The planned flight (``plans/<plan>/flights/plan_*``, as pils data: ``proc/plan`` and
``proc/azel/<T>``) and the flown one (``proc/flight/sync``, ``proc/azel/<T>``), compared as the
planned telescope sees the drone:

- when: the take-off planned and flown, the planned scan's length and the time the drone spent at
  the scan's height (no lower than :data:`SCAN_MARGIN_M` below the planned path: a planned flight
  is its scan; the climb and the landing are not compared), and its whole time in the air;
- where: for each flown sample, the nearest point of the planned path, in metres (the positions
  from the telescope) and as the telescope sees it (arcmin); and the share of the planned path the
  drone came near (within :data:`NEAR_M`);
- the drone's elevation and azimuth as the telescope saw it, planned and flown;
- what the telescope saw: the detectors the planned path crosses and the flown one did
  (:mod:`pils.analyze.coverage`, given the telescope's description).

Paths are compared where they are, not when: a flight flown late but along its path deviates
little. The telescope's position may differ between the two (the plan's nominal sites, the
flight's survey): its difference shifts the metres, not the angles much.

    compare(flown_dir, planned_dir, telescope_description=None, source_ghz=None) -> dict
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pils import store

# A planned point is reached when the drone came this near it (m)
NEAR_M = 5.0
# Above this, a flown sample is off the planned path (m)
OFF_M = 10.0
# The flown samples compared: at the scan's height, at most this below the planned path's lowest
SCAN_MARGIN_M = 20.0


def _latest(root: Path) -> Path | None:
    name = store.latest_version(root) if root.is_dir() else None
    return root / name if name else None


def planned_row(planned_dir: str | Path) -> dict[str, Any]:
    """Return the planned flight's metadata: its schedule ``row``, ``takeoff``, block."""
    plan = _latest(Path(planned_dir) / "proc" / "plan")
    if plan is None:
        raise ValueError(f"{planned_dir} is not a planned flight (no proc/plan)")
    return store.read_manifest(plan).get("metadata") or {}


def _azel(flight_dir: Path, telescope: str) -> pl.DataFrame | None:
    found = _latest(flight_dir / "proc" / "azel" / telescope)
    if found is None:
        return None
    return store.read_columns(
        found, "azel_data", ["timestamp", "az_deg", "el_deg", "e_m", "n_m", "u_m"]
    ).drop_nulls()


def _unit(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.column_stack(
        [np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)]
    )


def _flown_airborne(flown_dir: Path) -> tuple[float, float] | None:
    from pils.analyze.timing import airborne

    sync = _latest(flown_dir / "proc" / "flight" / "sync")
    if sync is None:
        return None
    drone = store.read_columns(sync, "drone", ["timestamp", "h_ell_m"]).drop_nulls()
    return airborne(
        drone["timestamp"].to_numpy(), drone["h_ell_m"].to_numpy().astype(float)
    )


def _range(values: np.ndarray) -> list[float] | None:
    values = values[np.isfinite(values)]
    return [float(values.min()), float(values.max())] if values.size else None


def _stats(values: np.ndarray) -> dict[str, float] | None:
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def compare(
    flown_dir: str | Path,
    planned_dir: str | Path,
    telescope_description: dict[str, Any] | None = None,
    source_ghz: float | None = None,
) -> dict[str, Any]:
    """Compare a flown flight with its planned one (see the module).

    Parameters
    ----------
    flown_dir, planned_dir : str or Path
        The flight's folder and the planned flight's.
    telescope_description : dict, optional
        The planned telescope (:func:`pils.analyze.coverage.describe_telescope`): with it, the
        detectors crossed, planned and flown.
    source_ghz : float, optional
        The source flown (else the plan's).

    Returns
    -------
    dict
        ``telescope``, ``plan`` (the schedule row's field, wafer, section...), ``when``
        (take-offs, minutes late, times in the air), ``path`` (deviation in m and arcmin, the
        share of the planned path reached, of the flight off it), ``sky`` (elevation and
        azimuth ranges), ``coverage`` (detectors planned, crossed, both, missed) and
        ``problems`` (what could not be compared, and why).
    """
    from scipy.spatial import cKDTree

    flown_dir, planned_dir = Path(flown_dir), Path(planned_dir)
    meta = planned_row(planned_dir)
    row = meta.get("row") or {}
    telescope = str(row.get("telescope", "")).upper()
    out: dict[str, Any] = {
        "telescope": telescope or None,
        "plan": {
            k: row.get(k)
            for k in (
                "flight_type",
                "field",
                "wafer",
                "boresight",
                "section",
                "n_sections",
                "source",
                "drone_az",
                "drone_el_lo",
                "drone_el_hi",
                "drone_range_m",
            )
        },
        "when": None,
        "path": None,
        "sky": None,
        "coverage": None,
        "problems": [],
    }
    if not telescope:
        out["problems"].append("The planned flight names no telescope.")
        return out

    planned = _azel(planned_dir, telescope)
    flown = _azel(flown_dir, telescope)
    flown_air = _flown_airborne(flown_dir)
    takeoff = meta.get("takeoff")
    out["when"] = {
        "planned_takeoff": takeoff,
        "flown_takeoff": flown_air[0] if flown_air else None,
        "minutes_late": (flown_air[0] - takeoff) / 60
        if flown_air and takeoff
        else None,
        "planned_scan_minutes": None,
        "flown_scan_minutes": None,
        "flown_minutes": (flown_air[1] - flown_air[0]) / 60 if flown_air else None,
    }
    if planned is None:
        out["problems"].append(f"The planned flight has no azel/{telescope}.")
        return out
    if flown is None:
        out["problems"].append(f"The flight has no azel/{telescope} (pils package).")
        return out
    planned_span = (
        float(planned["timestamp"].min()),
        float(planned["timestamp"].max()),
    )
    out["when"]["planned_scan_minutes"] = (planned_span[1] - planned_span[0]) / 60
    if flown_air is not None:
        flown = flown.filter(pl.col("timestamp").is_between(*flown_air))
    floor = float(planned["u_m"].min()) - SCAN_MARGIN_M
    flown = flown.filter(pl.col("u_m") >= floor)
    if flown.height < 2:
        out["problems"].append(
            f"The drone never reached the scan's height ({floor:.0f} m above the telescope)."
        )
        return out
    flown_span = (float(flown["timestamp"].min()), float(flown["timestamp"].max()))
    steps = np.diff(flown["timestamp"].to_numpy())
    out["when"]["flown_scan_minutes"] = (
        float(steps[steps < 5 * np.median(steps)].sum()) / 60 if steps.size else 0.0
    )

    # Where: nearest planned point to each flown sample, in metres and in angle
    p_xyz = planned.select("e_m", "n_m", "u_m").to_numpy().astype(float)
    f_xyz = flown.select("e_m", "n_m", "u_m").to_numpy().astype(float)
    metres, _ = cKDTree(p_xyz).query(f_xyz)
    p_unit = _unit(planned["az_deg"].to_numpy(), planned["el_deg"].to_numpy())
    f_unit = _unit(flown["az_deg"].to_numpy(), flown["el_deg"].to_numpy())
    chord, _ = cKDTree(p_unit).query(f_unit)
    arcmin = np.degrees(2 * np.arcsin(np.clip(chord / 2, 0, 1))) * 60
    reached, _ = cKDTree(f_xyz).query(p_xyz)
    out["path"] = {
        "deviation_m": _stats(metres),
        "deviation_arcmin": _stats(arcmin),
        "planned_reached_pct": float(100 * np.mean(reached <= NEAR_M)),
        "flown_off_pct": float(100 * np.mean(metres > OFF_M)),
        "near_m": NEAR_M,
        "off_m": OFF_M,
    }
    out["sky"] = {
        "planned_el_deg": _range(planned["el_deg"].to_numpy()),
        "flown_el_deg": _range(flown["el_deg"].to_numpy()),
        "planned_az_deg": _range(planned["az_deg"].to_numpy()),
        "flown_az_deg": _range(flown["az_deg"].to_numpy()),
    }

    # What the telescope saw: detectors crossed, planned and flown
    if telescope_description is not None:
        from pils.analyze.coverage import telescope_coverage

        source = source_ghz if source_ghz is not None else float(row.get("source") or 0)
        try:
            planned_cov = telescope_coverage(
                planned_dir, telescope_description, source, *planned_span
            )
            flown_cov = telescope_coverage(
                flown_dir, telescope_description, source, *flown_span
            )
        except ValueError as e:
            out["problems"].append(f"No coverage: {e}")
        else:

            def crossed(cov: dict[str, Any]) -> set[tuple[str, float, float]]:
                px = cov["pixels"]
                return {
                    (w, x, y)
                    for w, x, y, n in zip(
                        px["wafer"], px["xi"], px["eta"], px["crossings"], strict=True
                    )
                    if n
                }

            want, got = crossed(planned_cov), crossed(flown_cov)
            missed = want - got
            by_wafer: dict[str, dict[str, int]] = {}
            for wafer, *_ in want | got:
                by_wafer.setdefault(wafer, {"planned": 0, "crossed": 0, "missed": 0})
            for wafer, *_ in want:
                by_wafer[wafer]["planned"] += 1
            for wafer, *_ in got:
                by_wafer[wafer]["crossed"] += 1
            for wafer, *_ in missed:
                by_wafer[wafer]["missed"] += 1
            out["coverage"] = {
                "source_ghz": source,
                "band": flown_cov.get("band"),
                "planned": len(want),
                "crossed": len(got),
                "both": len(want & got),
                "missed": len(missed),
                "extra": len(got - want),
                "planned_crossed_pct": float(100 * len(want & got) / len(want))
                if want
                else None,
                "wafers": by_wafer,
                "boresight_deg": {
                    "planned": planned_cov.get("boresight_deg"),
                    "flown": flown_cov.get("boresight_deg"),
                },
            }
    return out
