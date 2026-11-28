"""The sources of a flight's drone data, found in its folders and decoded.

A flight can have, for the same drone, the data the drone itself saved and
the telemetry the payload logged:

==============  =========================================================
Matrice 600     ``*drone.dat`` (or DatCon ``*drone.csv``) and ``*litchi.csv``
                on the drone; LAGER's ``uav_telemetry.bin`` and
                ``gimbal_telemetry.bin`` with the run's YAML
Matrice 400     ``DJIFlightRecord_*.txt`` from the remote controller's app;
                LAGER's ``uav_telemetry.bin`` and ``gimbal_telemetry.bin``
                with the run's YAML
==============  =========================================================

:func:`find_lager_folder` and :func:`find_flight_records` locate the new
sources; :func:`load_extra_sources` decodes them. The .DAT, CSV and Litchi
files are read by :class:`pils.flight.Flight` as before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from ..utils.logging_config import get_logger
from .dji.m400.flightrecord import (
    DJIFlightRecordError,
    M400FlightRecord,
    resolve_api_key,
)
from .telemetry import load_telemetry, read_config

logger = get_logger(__name__)

FLIGHT_RECORD_GLOB = "DJIFlightRecord_*.txt"


def _folders(roots: list[str | Path | None]) -> list[Path]:
    """List the folders and their direct subfolders, in order, without repeats."""
    out: list[Path] = []
    for root in roots:
        if not root or not Path(root).is_dir():
            continue
        root = Path(root)
        for folder in [root, *sorted(p for p in root.iterdir() if p.is_dir())]:
            if folder not in out:
                out.append(folder)
    return out


def find_lager_folder(roots: list[str | Path | None]) -> Path | None:
    """Find the folder with LAGER's configuration (YAML with a ``uav`` or ``gimbal`` block).

    Looks in each root and its direct subfolders, in order, and returns the
    first that has one.
    """
    for folder in _folders(roots):
        try:
            read_config(folder)
        except FileNotFoundError:
            continue
        return folder
    return None


def find_flight_records(roots: list[str | Path | None]) -> list[Path]:
    """Find the app's flight records (``DJIFlightRecord_*.txt``) in the roots and their subfolders."""
    found: list[Path] = []
    for folder in _folders(roots):
        found.extend(sorted(folder.glob(FLIGHT_RECORD_GLOB)))
    return found


def load_extra_sources(
    roots: list[str | Path | None], api_key: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode the LAGER telemetry and the flight records found in the roots.

    A source that cannot be decoded is left out with a warning (and named in
    ``info["errors"]``): the flight is still loaded with what it has.

    Parameters
    ----------
    roots : list of folders
        The flight's drone folder and the payload's (aux) folder.
    api_key : str, optional
        DJI Open API app key for the flight records; default ``$DJI_API_KEY``.

    Returns
    -------
    sources, info
        ``sources``: ``uav_telemetry`` (table), ``gimbal`` (table per MAVLink
        message) and ``flight_record`` (table of the first record; the others
        are ``flight_record_2``, ...). ``info``: the models, files, decoder
        statistics, the records' headers and ``errors``.
    """
    sources: dict[str, Any] = {}
    info: dict[str, Any] = {"files": {}, "errors": {}}

    lager = find_lager_folder(roots)
    if lager is not None:
        try:
            t = load_telemetry(lager)
        except (ValueError, OSError) as e:
            logger.warning("LAGER telemetry in %s not decoded: %s", lager, e)
            info["errors"]["lager"] = str(e)
        else:
            info["files"]["lager"] = str(lager)
            info["config_file"] = t.config_file.name
            info["uav_model"], info["gimbal_model"] = t.uav_model, t.gimbal_model
            info["missing"] = t.missing
            if (
                t.uav is not None
                and t.uav.data is not None
                and not t.uav.data.is_empty()
            ):
                sources["uav_telemetry"] = t.uav.data
                info["uav_stats"] = t.uav.stats
            if t.gimbal is not None and t.gimbal.records:
                sources["gimbal"] = t.gimbal.records
                info["gimbal_complete"] = t.gimbal.complete

    records = find_flight_records(roots)
    if records:
        try:
            key = resolve_api_key(api_key)
        except DJIFlightRecordError as e:
            logger.warning("%d flight record(s) not decoded: %s", len(records), e)
            info["errors"]["flight_record"] = str(e)
            key = None
        for n, path in enumerate(records if key else [], start=1):
            name = "flight_record" if n == 1 else f"flight_record_{n}"
            loader = M400FlightRecord(path)
            try:
                loader.load_data(api_key=key)
            except (DJIFlightRecordError, OSError) as e:
                logger.warning("%s", e)
                info["errors"][name] = str(e)
                continue
            if loader.data is not None and not loader.data.is_empty():
                sources[name] = loader.data
                info["files"][name] = str(path)
                info.setdefault("flight_record_details", {})[name] = loader.details
    return sources, info


def telemetry_positions(
    table: pl.DataFrame, receivers_only: bool = False
) -> tuple[pl.DataFrame, dict[str, str]] | None:
    """Pick the positions of a telemetry or flight-record table for the synchronization.

    The RTK solution where the receiver has one, else the GNSS receiver, else
    the fused position (telemetry), else the app's position (flight record).
    Rows without a fix (0, 0) are left out, and rows that repeat the position of the one
    before (a value carried forward).

    Parameters
    ----------
    table : pl.DataFrame
        A telemetry or flight-record table (in the schema's names).
    receivers_only : bool, default=False
        Only the RTK and GNSS receivers' positions, whose heights the
        synchronization can refer to the ellipsoid (``rtk_h_ell_m``;
        ``gps_h_msl_m`` with the geoid separation). The fused position and
        the app's have a height of unknown reference.

    Returns
    -------
    (table, columns) or None
        The table of the valid rows and the names of its ``lat``, ``lon`` and
        ``alt`` columns; None if it has no position.
    """
    candidates = [
        ("rtk_lat_deg", "rtk_lon_deg", "rtk_h_ell_m"),
        ("gps_lat_deg", "gps_lon_deg", "gps_h_msl_m"),
        ("fused_lat_deg", "fused_lon_deg", "fused_h_m"),
        ("lat_deg", "lon_deg", "h_m"),
    ]
    for lat, lon, alt in candidates[: 2 if receivers_only else None]:
        if not {"timestamp", lat, lon, alt} <= set(table.columns):
            continue
        valid = table.filter(
            pl.col(lat).is_finite()
            & pl.col(lon).is_finite()
            & pl.col(alt).is_finite()
            & ((pl.col(lat) != 0) | (pl.col(lon) != 0))
            & (pl.col(lat).abs() <= 90)
            & (pl.col(lon).abs() <= 180)
        )
        # Telemetry carries each value forward until the receiver's next epoch:
        # keep the row where a position first appears (a held value would lag
        # the true time by up to one epoch)
        valid = valid.filter(
            (
                (pl.col(lat) != pl.col(lat).shift())
                | (pl.col(lon) != pl.col(lon).shift())
            ).fill_null(True)
        )
        if valid.height:
            return valid, {"lat": lat, "lon": lon, "alt": alt}
    return None


def record_positions(table: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, str]] | None:
    """Pick the positions of a flight record as a position source of its own.

    The positions of every frame (10 Hz): those of the record 48, whose
    heights are the take-off point's plus the height above it, else the app's
    (``lat_deg``, ``h_m``). Not the RTK solution, which comes once a second:
    the record keeps its rate. See :func:`telemetry_positions` for the rows kept.
    """
    if {"abs_lat_deg", "abs_lon_deg", "abs_h_m"} <= set(table.columns):
        valid = table.filter(
            pl.col("abs_lat_deg").is_finite()
            & pl.col("abs_lon_deg").is_finite()
            & pl.col("abs_h_m").is_finite()
            & ((pl.col("abs_lat_deg") != 0) | (pl.col("abs_lon_deg") != 0))
        )
        if valid.height:
            return valid, {"lat": "abs_lat_deg", "lon": "abs_lon_deg", "alt": "abs_h_m"}
    return telemetry_positions(table)


def record_rtk_base(table: pl.DataFrame | None) -> dict[str, float] | None:
    """Return the position the DJI RTK base broadcast, as the flight record has it.

    The median of the rows that have it (``rtk_base_*``), as ``lat``, ``lon``
    and ``alt`` (degrees, metres), the form of the ``dji_broadcast_geod`` of
    AZEL; None without it.
    """
    cols = ["rtk_base_lat_deg", "rtk_base_lon_deg", "rtk_base_h_ell_m"]
    if table is None or not set(cols) <= set(table.columns):
        return None
    rows = table.select(cols).drop_nulls()
    rows = rows.filter((pl.col(cols[0]) != 0) | (pl.col(cols[1]) != 0))
    if rows.is_empty():
        return None
    return {
        key: float(rows[col].median())  # type: ignore[arg-type]
        for key, col in zip(("lat", "lon", "alt"), cols, strict=True)
    }


RTK_FIXED_FLAG = 50  # rtk_pos_flag_1 of a fixed solution (as the .DAT's)


def record_track(
    table: pl.DataFrame,
    min_epochs: int = 10,
    max_lag_s: float = 1.0,
    hold_s: float = 2.0,
) -> tuple[pl.DataFrame, dict[str, float]] | None:
    """Return the 10 Hz track of a flight record, tied to its RTK solution.

    The positions of every frame (record 48, ``abs_*``) are the drone's at
    the frame's time (the drone's GPS time); the RTK solution (record 45,
    1 Hz) comes with a latency. The latency is the shift of the RTK epochs
    that best matches the track (horizontal scatter, within ``max_lag_s``);
    the track is then moved by the median of RTK minus track at the RTK
    epochs (east, north, up), so that it is in the RTK's frame (that of the
    base, which AZEL corrects to the survey). Each frame has the fix flag of
    the last RTK epoch within ``hold_s`` (``rtk_pos_flag_1``, null without).

    Parameters
    ----------
    table : pl.DataFrame
        A flight record (schema names): ``abs_lat_deg``, ``abs_lon_deg``,
        ``abs_h_m`` and ``rtk_lat_deg``, ``rtk_lon_deg``, ``rtk_h_ell_m``.
    min_epochs : int
        Fixed RTK epochs needed (else None).

    Returns
    -------
    (track, info) or None
        ``track``: the frames with positions, the record's other columns and
        ``rtk_track_lat_deg``, ``rtk_track_lon_deg``, ``rtk_track_h_ell_m``
        (the raw ``rtk_*`` columns of the 1 Hz epochs left out). ``info``:
        ``offset_e_m``, ``offset_n_m``, ``offset_u_m`` (RTK minus track),
        ``rms_h_m``, ``rms_u_m`` (scatter about them), ``rtk_latency_s``
        and ``epochs``.
    """
    import numpy as np
    import pymap3d as pm

    track_cols = ["abs_lat_deg", "abs_lon_deg", "abs_h_m"]
    rtk_cols = ["rtk_lat_deg", "rtk_lon_deg", "rtk_h_ell_m"]
    if not {"timestamp", *track_cols, *rtk_cols} <= set(table.columns):
        return None
    track = table.filter(
        pl.all_horizontal(pl.col(c).is_finite() for c in track_cols)
        & ((pl.col("abs_lat_deg") != 0) | (pl.col("abs_lon_deg") != 0))
    ).sort("timestamp")
    fixed = pl.all_horizontal(pl.col(c).is_not_null() for c in rtk_cols) & (
        (pl.col("rtk_lat_deg") != 0) | (pl.col("rtk_lon_deg") != 0)
    )
    if "rtk_pos_flag_1" in table.columns:
        fixed = fixed & (pl.col("rtk_pos_flag_1") == RTK_FIXED_FLAG)
    rtk = table.filter(fixed).sort("timestamp")
    if rtk.height < min_epochs or track.height < 2:
        return None

    lat0, lon0, h0 = (float(rtk[c].median()) for c in rtk_cols)  # type: ignore[arg-type]
    t = track["timestamp"].to_numpy()
    e, n, u = pm.geodetic2enu(
        *(track[c].to_numpy() for c in track_cols), lat0, lon0, h0
    )
    tr = rtk["timestamp"].to_numpy()
    er, nr, ur = pm.geodetic2enu(*(rtk[c].to_numpy() for c in rtk_cols), lat0, lon0, h0)

    def residuals(lag: float):
        at = tr + lag
        return (
            er - np.interp(at, t, e),
            nr - np.interp(at, t, n),
            ur - np.interp(at, t, u),
        )

    def scatter(lag: float) -> float:
        de, dn, _ = residuals(lag)
        return float(np.mean((de - np.median(de)) ** 2 + (dn - np.median(dn)) ** 2))

    lags = np.arange(-max_lag_s, max_lag_s + 1e-9, 0.02)
    lag = float(lags[int(np.argmin([scatter(x) for x in lags]))])
    de, dn, du = residuals(lag)
    off = (float(np.median(de)), float(np.median(dn)), float(np.median(du)))
    lat, lon, h = pm.enu2geodetic(e + off[0], n + off[1], u + off[2], lat0, lon0, h0)

    flags = (
        rtk.select("timestamp", pl.col("rtk_pos_flag_1").alias("_flag"))
        if "rtk_pos_flag_1" in rtk.columns
        else rtk.select("timestamp", pl.lit(RTK_FIXED_FLAG).alias("_flag"))
    )
    out = (
        track.drop([c for c in track.columns if c.startswith("rtk_")])
        .with_columns(
            rtk_track_lat_deg=pl.Series(np.asarray(lat, dtype=float)),
            rtk_track_lon_deg=pl.Series(np.asarray(lon, dtype=float)),
            rtk_track_h_ell_m=pl.Series(np.asarray(h, dtype=float)),
        )
        .join_asof(flags, on="timestamp", strategy="backward", tolerance=hold_s)
        .rename({"_flag": "rtk_pos_flag_1"})
    )
    info = {
        "offset_e_m": off[0],
        "offset_n_m": off[1],
        "offset_u_m": off[2],
        "rms_h_m": float(np.sqrt(np.mean((de - off[0]) ** 2 + (dn - off[1]) ** 2))),
        "rms_u_m": float(np.sqrt(np.mean((du - off[2]) ** 2))),
        "rtk_latency_s": -lag,
        "epochs": float(rtk.height),
    }
    return out, info
