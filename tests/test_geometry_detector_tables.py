"""A focal plane read from a detector table: CLASS's, the LAT's (pils.geometry.detector_tables)."""

from pathlib import Path

import numpy as np
import pytest

from pils.geometry.detector_tables import read_detector_table, wafers_from_table

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
# The planner's real tables, when its checkout is here
GOSE_CONFIGS = Path("/home/polocalc_software/gose/configs")


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


@pytest.mark.skipif(not GOSE_CONFIGS.is_dir(), reason="GOSE's tables are not here")
def test_the_same_wafers_as_gose_s_planner():
    gose = pytest.importorskip("gose.focal_plane_file")
    for path in sorted(GOSE_CONFIGS.glob("*/*.txt")):
        mine, theirs = wafers_from_table(str(path)), gose.wafers_from_table(str(path))
        assert [(w.name, w.xi, w.eta, w.outline) for w in mine] == [
            (w.name, w.xi, w.eta, w.outline) for w in theirs
        ]
        assert all(
            np.array_equal(a.pixels, b.pixels)
            for a, b in zip(mine, theirs, strict=True)
        )
