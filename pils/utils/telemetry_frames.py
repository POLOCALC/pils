"""Tables of telemetry rows: the same few steps for every LAGER log."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def resolve_file(path: str | Path, filename: str) -> Path:
    """Return the log file: ``path`` itself, or ``filename`` in it if ``path`` is a folder."""
    p = Path(path)
    if p.is_dir():
        p = p / filename
    if not p.is_file():
        raise FileNotFoundError(f"Telemetry file not found: {p}")
    return p


def rows_to_frame(rows: list[dict], sort_by: str = "timestamp") -> pl.DataFrame:
    """Make a table with one row per record, sorted by time (stable).

    The columns are the union of the records' keys (null where a record has none);
    integers and floats of one column are promoted to float.
    """
    if not rows:
        return pl.DataFrame()
    frame = pl.from_dicts(rows, infer_schema_length=None, strict=False)
    return frame.sort(sort_by, maintain_order=True)


def forward_fill(frame: pl.DataFrame) -> pl.DataFrame:
    """Each column carries its last value forward (the topics arrive at different rates)."""
    return frame.with_columns(pl.all().forward_fill())
