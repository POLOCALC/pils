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
    table: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, str]] | None:
    """Pick the positions of a telemetry or flight-record table for the synchronization.

    The RTK solution where the receiver has one, else the GNSS receiver, else
    the fused position (telemetry), else the app's position (flight record).
    Rows without a fix (0, 0) are left out.

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
    for lat, lon, alt in candidates:
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
        if valid.height:
            return valid, {"lat": lat, "lon": lon, "alt": alt}
    return None
