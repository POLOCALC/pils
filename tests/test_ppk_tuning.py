"""Tests for the RTKLIB configuration tuning (pils.analyze.ppk_tuning)."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pils.analyze import ppk_tuning as T

HEADER = """\
     3.04           OBSERVATION DATA    M: Mixed            RINEX VERSION / TYPE
  2223185.6344 -5444177.7559 -2474640.8671                  APPROX POSITION XYZ
G    8 C1C L1C D1C S1C C2X L2X D2X S2X                      SYS / # / OBS TYPES
E    8 C1X L1X D1X S1X C7X L7X D7X S7X                      SYS / # / OBS TYPES
                                                            END OF HEADER
"""


def obs_field(value=None):
    return f"{value:14.3f}  " if value is not None else " " * 16


def rinex(path: Path, epochs=5, l2_snr=22.0, gps=6, gal=5) -> Path:
    lines = [HEADER.rstrip("\n")]
    for k in range(epochs):
        lines.append(f"> 2025 12 06 18 33 {k:10.7f}  0 {gps + gal:2d}")
        for i in range(gps):
            vals = [2e7, 1e8, 1.0, 42.0, 2e7, 1e8, 1.0, l2_snr]
            lines.append(f"G{i + 1:02d}" + "".join(obs_field(v) for v in vals))
        for i in range(gal):
            vals = [2e7, 1e8, 1.0, 37.0, 2e7, 1e8, 1.0, 30.0]
            lines.append(f"E{i + 1:02d}" + "".join(obs_field(v) for v in vals))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_profile(tmp_path):
    p = T.profile_obs(rinex(tmp_path / "r.obs"))
    assert p.epochs == 5 and p.interval_s == pytest.approx(1.0)
    assert p.sats == {"G": 6.0, "E": 5.0}
    assert p.snr["G"] == {"1": 42.0, "2": 22.0}
    assert p.band_fraction["G"]["2"] == 1.0
    assert p.approx_xyz == pytest.approx((2223185.6344, -5444177.7559, -2474640.8671))


def test_candidates_follow_the_data(tmp_path):
    base = T.profile_obs(rinex(tmp_path / "b.obs", l2_snr=40))
    weak = T.profile_obs(rinex(tmp_path / "r.obs", l2_snr=22))
    strong = T.profile_obs(rinex(tmp_path / "s.obs", l2_snr=40))
    names = [c.name for c in T.candidates(weak, base)]
    assert all(n.startswith("l1_") for n in names)  # weak L2: L1 only
    assert any(
        n.startswith("l1+l2_") for n in (c.name for c in T.candidates(strong, base))
    )
    assert {n.split("_")[1] for n in names} == {"G", "GE"}  # no GLONASS/BeiDou data
    assert not any(
        "fix-and-hold" in c.name for c in T.candidates(weak, base, hold=False)
    )


def test_snr_masks_from_template_and_data(tmp_path):
    rover = T.profile_obs(rinex(tmp_path / "r.obs"))
    template = "pos1-snrmask_r =on\npos1-snrmask_L1 =35,30,30\n"
    # template 35, then from the data: median(42, 37) = 39.5 -> 40 - 5 = 35,
    # already there
    masks = T.snr_masks(rover, template)
    assert masks[0] == ("snr35", 35) and masks[-1] == ("snroff", None)


def test_apply_overrides_and_config_value():
    text = "pos1-elmask       =10   # (deg)\npos2-armode =fix-and-hold\n"
    out = T.apply_overrides(text, {"pos1-elmask": "15", "pos1-navsys": "9"})
    assert T.config_value(out, "pos1-elmask") == "15"
    assert "# (deg)" in out
    assert T.config_value(out, "pos1-navsys") == "9"
    assert T.config_value(out, "pos2-armode") == "fix-and-hold"


def test_base_position_check(tmp_path):
    base = T.profile_obs(rinex(tmp_path / "b.obs"))
    lat, lon, h = -22.9598220899992, -67.78685015, 5182.212  # 3 m from APPROX
    good = f"ant2-postype =llh\nant2-pos1 ={lat}\nant2-pos2 ={lon}\nant2-pos3 ={h}\n"
    assert T.base_position_check(good, base)[0] == {}
    far = good.replace(str(lat), "-22.9618")
    overrides, note = T.base_position_check(far, base)
    assert overrides == {"ant2-postype": "single"} and "m from" in note


def track(t, east=0.0):
    lat0, lon0 = -22.96, -67.78
    return pl.DataFrame(
        {
            "timestamp": t,
            "lat_deg": lat0 + 1e-6 * t,
            "lon_deg": lon0 + east / 102_000.0,
            "h_ell_m": 5200.0 + 0.1 * t,
        }
    )


def test_compare_to_reference():
    t = np.arange(0.0, 100.0)
    ref = track(t)
    sol = track(t, east=1.5).with_columns(pl.lit(1).alias("quality"))  # lever arm
    h, v, bad = T.compare_to_reference(sol, ref)
    assert h < 0.01 and v < 0.01 and bad == 0
    jump = sol.with_columns(
        pl.when(pl.col("timestamp") > 90)
        .then(pl.col("lon_deg") + 3 / 102_000.0)
        .otherwise(pl.col("lon_deg"))
        .alias("lon_deg")
    )
    # 8 of the 98 epochs inside the reference span (ends excluded)
    assert T.compare_to_reference(jump, ref)[2] == pytest.approx(100 * 8 / 98)


def test_tune_picks_most_fixes_among_valid(tmp_path, monkeypatch):
    rover = rinex(tmp_path / "r.obs")
    base = rinex(tmp_path / "b.obs")
    t = np.arange(0.0, 100.0)

    def fake_run(config, rover_obs, base_obs, nav, pos):
        text = config.read_text()
        hold = T.config_value(text, "pos2-armode") == "fix-and-hold"
        el = int(T.config_value(text, "pos1-elmask"))
        quality = np.where(t < 50 + el + (30 if hold else 0), 1, 2)
        sol = track(t).with_columns(pl.Series("quality", quality))
        if hold:  # more fixes, but wrong ones
            sol = sol.with_columns(
                pl.when(pl.col("timestamp") > 60)
                .then(pl.col("lon_deg") + 5 / 102_000.0)
                .otherwise(pl.col("lon_deg"))
                .alias("lon_deg")
            )
        sol.write_parquet(pos)
        return pos

    monkeypatch.setattr(T, "run_rnx2rtkp", fake_run)
    monkeypatch.setattr(T, "read_pos", lambda pos: pl.read_parquet(pos))
    result = T.tune(
        "pos2-armode =continuous\npos1-elmask =10\n",
        rover,
        base,
        tmp_path / "n.nav",
        tmp_path / "work",
        reference=track(t),
        n_random=0,
        selection="reference",
    )
    assert result.best is not None
    assert result.best.overrides["pos2-armode"] == "continuous"
    assert result.best.overrides["pos1-elmask"] == "20"
    assert T.config_value(result.config_text, "pos1-elmask") == "20"
    assert result.table().filter(pl.col("valid")).height > 0
    assert not result.table().filter(pl.col("name").str.contains("hold"))["valid"].any()


def test_data_quality_note():
    good = T.ObsProfile(1.0, None, None, 700, lli_fraction=0.04, arc_s=388.0)
    bad = T.ObsProfile(1.0, None, None, 880, lli_fraction=0.15, arc_s=95.0)
    assert T.data_quality_note(good) == ""
    note = T.data_quality_note(bad)
    assert "15 %" in note and "95 s" in note


def test_score_against_reference():
    t = np.arange(0.0, 100.0)
    ref = track(t)
    sol = track(t, east=1.5).with_columns(
        pl.Series("quality", np.where(t < 50, 1, 2))  # half fixed
    )
    jump = sol.with_columns(
        pl.when(pl.col("timestamp").is_between(40, 49))
        .then(pl.col("lon_deg") + 3 / 102_000.0)
        .otherwise(pl.col("lon_deg"))
        .alias("lon_deg")
    )
    score = T.score_against_reference(jump, ref)
    assert score["fix_pct"] == pytest.approx(49, abs=1)  # epochs inside the span
    assert score["wrong_pct"] == pytest.approx(10, abs=1)
    assert score["good_pct"] == pytest.approx(39, abs=1)


def test_random_candidates_follow_the_data(tmp_path):
    base = T.profile_obs(rinex(tmp_path / "b.obs"))
    rover = T.profile_obs(rinex(tmp_path / "r.obs"))
    a = T.random_candidates(rover, base, n=50, seed=1)
    assert [c.overrides for c in a] == [
        c.overrides for c in T.random_candidates(rover, base, n=50, seed=1)
    ]
    navsys = {int(c.overrides["pos1-navsys"]) for c in a}
    assert navsys <= {1, 9}  # GPS, Galileo: the only systems in the data
    assert not any(
        c.overrides["pos2-armode"] == "fix-and-hold"
        for c in T.random_candidates(rover, base, n=50, hold=False)
    )


def test_setting_effects():
    trials = []
    for i in range(200):
        hold = i % 2 == 0
        c = T.Candidate(
            name=f"random_{i:04d}",
            overrides={
                "pos1-frequency": "l1",
                "pos1-navsys": "1",
                "pos2-armode": "fix-and-hold" if hold else "continuous",
                "pos1-snrmask_r": "off",
                "pos1-snrmask_L1": "30",
                "pos1-elmask": "15",
                "pos2-arthres": "3",
                "pos2-minfixsats": "5",
            },
        )
        c.epochs = 100
        c.good_pct, c.wrong_pct = (40.0 if hold else 10.0), (2.0 if hold else 0.0)
        trials.append(c)
    effects = {e["choice"]: e for e in T.setting_effects(trials)}
    hold = effects["fix-and-hold (vs continuous)"]
    assert hold["gain"] == pytest.approx(30) and hold["risk"] == pytest.approx(2)
    assert "L1+L2 (vs L1)" not in effects  # never tried


def test_base_cache_covers_the_rover(tmp_path, monkeypatch):
    from datetime import datetime

    from pils.analyze.ppk import PPKAnalysis

    ppk = PPKAnalysis.__new__(PPKAnalysis)
    span = (datetime(2025, 12, 6, 18, 30), datetime(2025, 12, 6, 18, 50))
    monkeypatch.setattr(ppk, "_get_rinex_bounds", lambda obs: span)
    obs = tmp_path / "base.obs"
    assert ppk._covers(
        obs, datetime(2025, 12, 6, 18, 35), datetime(2025, 12, 6, 18, 45)
    )
    assert not ppk._covers(
        obs, datetime(2025, 12, 6, 18, 35), datetime(2025, 12, 6, 19, 0)
    )
    assert not ppk._covers(
        obs, datetime(2025, 12, 6, 18, 0), datetime(2025, 12, 6, 18, 45)
    )


def test_tune_by_fix_rate(tmp_path, monkeypatch):
    """The default: the most fixed epochs (Q=1), no reference needed."""
    rover = rinex(tmp_path / "r.obs")
    base = rinex(tmp_path / "b.obs")
    t = np.arange(0.0, 100.0)

    def fake_run(config, rover_obs, base_obs, nav, pos):
        text = config.read_text()
        el = int(T.config_value(text, "pos1-elmask"))
        hold = T.config_value(text, "pos2-armode") == "fix-and-hold"
        quality = np.where(t < 30 + el + (20 if hold else 0), 1, 2)
        track(t).with_columns(pl.Series("quality", quality)).write_parquet(pos)
        return pos

    monkeypatch.setattr(T, "run_rnx2rtkp", fake_run)
    monkeypatch.setattr(T, "read_pos", lambda pos: pl.read_parquet(pos))
    result = T.tune(
        "pos2-armode =continuous\npos1-elmask =10\n",
        rover,
        base,
        tmp_path / "n.nav",
        tmp_path / "work",
        n_random=60,
    )
    assert result.validation == "fix rate"
    best = result.best
    assert best is not None and best.fix_pct == max(c.fix_pct for c in result.trials)
    assert best.overrides["pos2-armode"] == "fix-and-hold"
    effects = {e["choice"]: e for e in result.effects}
    assert effects["fix-and-hold (vs continuous)"]["gain"] > 10
    assert effects["fix-and-hold (vs continuous)"]["risk"] is None
    with pytest.raises(ValueError, match="needs a reference"):
        T.tune(
            "x", rover, base, tmp_path / "n.nav", tmp_path / "w", selection="reference"
        )


def test_failed_ppk_run_leaves_no_revision(tmp_path, monkeypatch):
    from pils.analyze.ppk import PPKAnalysis

    ppk = PPKAnalysis.__new__(PPKAnalysis)
    ppk.ppk_dir = tmp_path / "ppk"
    ppk.ppk_dir.mkdir()
    (ppk.ppk_dir / "rev_20250101_000000").mkdir()  # an earlier, kept

    def fail(*args, **kwargs):
        (ppk.ppk_dir / "rev_20261018_093715").mkdir()
        raise FileNotFoundError("No base recording covers ...")

    monkeypatch.setattr(ppk, "_run_analysis", fail)
    with pytest.raises(FileNotFoundError):
        ppk.run_analysis("template.conf")
    assert [p.name for p in ppk.ppk_dir.iterdir()] == ["rev_20250101_000000"]
