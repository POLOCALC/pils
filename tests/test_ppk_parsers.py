"""RTKLIB .pos and .stat parsers: schema names and UTC timestamps."""

from datetime import UTC, datetime

import pytest

from pils.analyze.ppkdata.PPK.pos_analyzer import POSAnalyzer
from pils.analyze.ppkdata.PPK.stat_analyzer import STATAnalyzer

POS_HEADER = """% program   : RTKLIB ver.EX 2.5.0
% (lat/lon/height=WGS84/{datum},Q=1:fix,2:float,3:sbas,4:dgps,5:single,6:ppp,ns=# of satellites)
%  {system}                  latitude(deg) longitude(deg)  height(m)   Q  ns   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m) age(s)  ratio
"""
POS_LINES = """2025/12/06 18:33:43.000  -22.959738310  -67.786904970  5146.8740   1  16   0.0100   0.0090   0.0200   0.0000   0.0000   0.0000   1.00    3.5
2025/12/06 18:33:44.000  -22.959738300  -67.786904960  5146.8750   2  15   0.0200   0.0190   0.0400   0.0000   0.0000   0.0000   1.00    0.0
"""


@pytest.mark.parametrize(
    "system, datum, height, utc_second",
    [
        ("GPST", "ellipsoidal", "h_ell_m", 25),  # 18 leap seconds
        ("UTC", "geodetic", "h_msl_m", 43),
    ],
)
def test_pos(tmp_path, system, datum, height, utc_second):
    path = tmp_path / "solution.pos"
    path.write_text(POS_HEADER.format(system=system, datum=datum) + POS_LINES)
    analyzer = POSAnalyzer(path)
    df = analyzer.parse()
    assert df.columns == [
        "datetime",
        "timestamp",
        "lat_deg",
        "lon_deg",
        height,
        "quality",
        "num_sats",
        "sd_n_m",
        "sd_e_m",
        "sd_u_m",
        "age_s",
        "ratio",
        "e_m",
        "n_m",
        "u_m",
    ]
    expected = datetime(2025, 12, 6, 18, 33, utc_second, tzinfo=UTC)
    assert df["datetime"][0] == expected
    assert df["timestamp"][0] == expected.timestamp()
    stats = analyzer.get_statistics()
    assert stats["fix_epochs"] == 1 and stats["float_epochs"] == 1
    assert stats["avg_ns"] == 15.5


def test_stat(tmp_path):
    path = tmp_path / "solution.pos.stat"
    path.write_text(
        "$POS,2395,585223.000,0,1,2,3,0,0,0\n"
        "$SAT,2395,585223.000,G01,1,120.5,45.2,1.2500,0.0030,1,42.0,1,0,5,0,0,0\n"
    )
    analyzer = STATAnalyzer(path)
    df = analyzer.parse()
    assert df.columns == [
        "timestamp",
        "gps_week",
        "tow_s",
        "satellite",
        "frequency",
        "az_deg",
        "el_deg",
        "resid_code_m",
        "resid_phase_m",
        "snr_dbhz",
        "slip",
        "lock",
        "reject",
        "constellation",
    ]
    # GPS week 2395, 585223 s, minus 18 leap seconds
    assert (
        df["timestamp"][0] == datetime(2025, 12, 6, 18, 33, 25, tzinfo=UTC).timestamp()
    )
    assert analyzer.get_satellite_stats()["avg_snr_dbhz"][0] == 42.0
