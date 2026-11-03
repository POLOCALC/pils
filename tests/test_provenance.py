"""Where a revision stands (pils.provenance)."""

import subprocess

import numpy as np
import polars as pl
import pytest

from pils import provenance, store


def test_compare_revisions(tmp_path):
    t = np.arange(10.0)
    old = {
        "drone": pl.DataFrame({"timestamp": t, "h": t * 2, "gone": t}),
        "extra": pl.DataFrame({"x": [1]}),
    }
    new = {
        "drone": pl.DataFrame(
            {"timestamp": t, "h": t * 2 + np.r_[0.0, 0.5, [0.0] * 8], "more": t}
        )
    }
    a = store.save_version(tmp_path, "rev_20260102_000000", new, {})
    b = store.save_version(tmp_path, "rev_20260101_000000", old, {})
    out = provenance.compare_revisions(a, b)
    tables = {x["name"]: x for x in out["tables"]}
    assert tables["extra"]["status"] == "removed"
    drone = tables["drone"]
    assert (
        drone["status"] == "changed"
        and drone["rows"] == [10, 10]
        and drone["same_times"]
    )
    assert drone["added"] == ["more"] and drone["removed"] == ["gone"]
    assert drone["columns"]["h"]["max_diff"] == pytest.approx(0.5)
    assert drone["columns"]["timestamp"]["max_diff"] == 0
    assert drone["changed"] == ["gone", "h", "more"]
    assert out["pils"][0]["version"]


def test_code_since(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / "pils").mkdir()
    (tmp_path / "pils" / "a.py").write_text("1")
    (tmp_path / "README").write_text("1")
    git("add", "-A")
    git("commit", "-qm", "one")
    first = git("rev-parse", "HEAD")
    (tmp_path / "pils" / "a.py").write_text("2")
    (tmp_path / "README").write_text("2")
    git("commit", "-qam", "two")
    head = git("rev-parse", "HEAD")
    monkeypatch.setattr(provenance, "SOURCE", tmp_path)
    monkeypatch.setattr(
        store,
        "code_version",
        lambda: {"version": "x", "git_commit": head, "git_dirty": False},
    )
    provenance._since.cache_clear()
    out = provenance.code_since(first)
    assert out["known"] and out["ancestor"] and out["behind"] == 1
    assert out["files"] == ["pils/a.py"]  # only the code
    assert (
        provenance.code_since(head)["same"]
        and provenance.code_since(head)["behind"] == 0
    )
    assert provenance.code_since("0" * 40)["known"] is False
    assert provenance.code_since(None)["known"] is False
