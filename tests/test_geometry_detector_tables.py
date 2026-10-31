"""A focal plane read from a detector table: CLASS's, the LAT's (pils.geometry.detector_tables)."""

import numpy as np
import pytest

from pils.geometry.detector_tables import (
    BUNDLED_TABLES,
    bundled_table,
    read_detector_table,
    wafers_from_table,
)

TABLE = """# offsets relative to boresight, degrees
 det_uid wafer feed type  az_off  el_off
       0    WA    1    V  -1.000  +0.000
       1    WA    1    H  -1.010  +0.010
       2    WA    2    V  +1.000  +0.000
       3    WA    3    V  +0.000  +2.000
       4    WA    4    V  +0.000  -2.000
       5    WA    0   DD  +9.000  +9.000
       6   nan    9    V  +5.000  +5.000
       7    WB    1    V     nan     nan
       8    WB    2    V  +10.00  +0.000
"""


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "focal_plane.txt"
    path.write_text(TABLE)
    return path


def test_the_rows_are_read_with_numbers_for_the_offsets(table):
    rows = read_detector_table(str(table))
    assert len(rows) == 9 and rows[0]["wafer"] == "WA" and rows[0]["az_off"] == -1.0
    assert np.isnan(rows[7]["az_off"])


def test_a_wafer_per_named_module_of_its_optical_feeds(table):
    # WB has one feed with a pointing solution: too few for a wafer
    with pytest.raises(ValueError):
        wafers_from_table(str(table))
    rows = TABLE.splitlines()
    path = table.with_name("wa_only.txt")
    path.write_text("\n".join(r for r in rows if " WB " not in r) + "\n")
    (wa,) = wafers_from_table(str(path), outline="hull")
    assert wa.name == "WA"
    # 4 feeds (the two polarisations of feed 1 averaged; the dark and the unnamed left out)
    assert wa.pixels.shape == (4, 2)
    assert np.allclose(
        sorted(wa.pixels[:, 0] * 180 / np.pi), [-1.005, 0.0, 0.0, 1.0], atol=1e-6
    )


def test_the_shipped_tables_are_found_by_name():
    assert len(BUNDLED_TABLES) == 3
    for name in BUNDLED_TABLES:
        path = bundled_table(name)
        assert path.is_file() and path.name == name
        assert bundled_table(f"pils:{name}") == path
        assert bundled_table(name.removesuffix(".txt")) == path
    with pytest.raises(KeyError):
        bundled_table("pils:no_such_table")


@pytest.mark.parametrize(
    "name, wafers",
    [
        (
            "class_focal_plane_w1_band.txt",
            {
                "W2-1": 28,
                "W2-2": 31,
                "W2-3": 32,
                "W2-4": 26,
                "W3": 34,
                "W6": 37,
                "W7": 27,
            },
        ),
        ("class_focal_plane_w2_band.txt", {"W1": 37, "W2": 37, "W3": 37, "W5": 37}),
        ("lat_mf_inner_wafer.txt", {"LAT-MF": 430}),
    ],
)
def test_the_shipped_tables_read_as_their_wafers(name, wafers):
    read = wafers_from_table(str(bundled_table(name)))
    assert {w.name: w.pixels.shape[0] for w in read} == wafers
    for w in read:
        assert np.isfinite(w.pixels).all() and w.outline
        # radians, within 20 degrees of the boresight; the centre among its feeds
        assert np.abs(w.pixels).max() < np.radians(20)
        lo, hi = w.pixels.min(axis=0), w.pixels.max(axis=0)
        assert lo[0] <= w.xi <= hi[0] and lo[1] <= w.eta <= hi[1]
