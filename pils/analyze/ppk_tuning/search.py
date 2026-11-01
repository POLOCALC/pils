"""The search: :func:`tune` runs every candidate and keeps the best."""

from __future__ import annotations

import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from pils.analyze.ppk_tuning.configs import (
    N_RANDOM,
    Candidate,
    apply_overrides,
    base_position_check,
    candidates,
    random_candidates,
)
from pils.analyze.ppk_tuning.profile import (
    ObsProfile,
    data_quality_note,
    profile_obs,
)
from pils.analyze.ppk_tuning.runs import (
    compare_directions,
    read_pos,
    run_rnx2rtkp,
    score_against_reference,
    setting_effects,
)

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """Outcome of :func:`tune`."""

    best: Candidate | None
    trials: list[Candidate]
    rover: ObsProfile
    base: ObsProfile
    config_text: str | None
    validation: str
    base_note: str = ""
    data_note: str = ""

    @property
    def effects(self) -> list[dict[str, Any]]:
        """Effect of each setting on this data (see :func:`setting_effects`)."""
        return setting_effects(self.trials)

    def table(self) -> pl.DataFrame:
        """Return the trials as a table, one row per candidate."""
        return pl.DataFrame([c.row() for c in self.trials])

    def metadata(self) -> dict[str, Any]:
        """Return the tuning summary for the PPK revision's manifest."""
        return {
            "tuning_validation": self.validation,
            "tuning_base_position": self.base_note,
            "tuning_data_quality": self.data_note,
            "tuning_best": self.best.row() if self.best else None,
            "tuning_trials": len(self.trials),
            "tuning_effects": self.effects,
            "rover_profile": self.rover.as_dict(),
            "base_profile": self.base.as_dict(),
        }


def tune(
    template: str | Path,
    rover_obs: str | Path,
    base_obs: str | Path,
    nav: str | Path,
    workdir: str | Path,
    reference: pl.DataFrame | None = None,
    max_h_rms: float = 0.10,
    max_v_rms: float = 0.15,
    max_bad_pct: float = 0.5,
    workers: int | None = None,
    n_random: int = N_RANDOM,
    seed: int = 0,
    max_wrong_pct: float = 0.5,
    selection: str = "fix_rate",
) -> TuningResult:
    """Choose the RTKLIB configuration for this data.

    Parameters
    ----------
    template : str or Path
        RTKLIB configuration the candidates start from (base position,
        statistics, output format, ... come from it).
    rover_obs, base_obs, nav : str or Path
        RINEX observation files and navigation file; the base must cover
        the rover's time span.
    workdir : str or Path
        Folder for the candidates' configs and solutions.
    reference : pl.DataFrame, optional
        Independent track of the rover (``timestamp`` UNIX s UTC,
        ``lat_deg``, ``lon_deg``, ``h_ell_m``), e.g. the drone RTK track;
        used by ``selection="reference"``.
    max_h_rms, max_v_rms : float
        Largest RMS (m) of the fixed epochs against the reference.
    max_bad_pct : float
        Largest % of fixed epochs off by more than 1 m (0.2 m between
        directions without a reference).
    workers : int, optional
        Parallel runs (default: CPU count).
    n_random : int, default=N_RANDOM
        Random configurations tried besides the grid
        (:func:`random_candidates`); 0 for the grid only.
    seed : int, default=0
        Seed of the random configurations.
    max_wrong_pct : float, default=0.5
        With a reference: largest % of epochs with a wrong fix (off by more
        than ``WRONG_M``) for a valid candidate.
    selection : {"fix_rate", "reference", "directions"}, default="fix_rate"
        How the best candidate is chosen (see Notes).

    Returns
    -------
    TuningResult
        The best candidate (None if none is valid), every trial, the data
        profiles, the chosen config and the effect of each setting on this
        data (:func:`setting_effects`).

    Notes
    -----
    - ``"fix_rate"``: the candidate with the most epochs fixed by RTKLIB
      (Q=1, of all the solution's epochs); no other check. On the November
      2025 flights the highest fix rates often included wrong fixes (the
      template: 17 % of epochs fixed off by more than 0.5 m, on average).
    - ``"reference"``: every epoch is scored against ``reference``
      (:func:`score_against_reference`): valid with at most
      ``max_wrong_pct`` wrong fixes, the best has the most good fixes.
    - ``"directions"``: forward and backward solutions must agree (two runs
      per candidate, no fix-and-hold); the most fixes.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    text = Path(template).read_text() if Path(str(template)).exists() else str(template)
    rover, base = profile_obs(rover_obs), profile_obs(base_obs)
    base_overrides, base_note = base_position_check(text, base)
    data_note = data_quality_note(rover)
    if data_note:
        logger.warning(data_note)
    if base_overrides:
        logger.warning(base_note)
    if selection not in ("fix_rate", "reference", "directions"):
        raise ValueError(f"Unknown selection {selection!r}")
    if selection == "reference" and reference is None:
        raise ValueError('selection="reference" needs a reference track')
    directions = selection == "directions"
    by_reference = selection == "reference"
    # Forward and backward fix-and-hold solutions can hold the same wrong
    # ambiguities: with "directions", only continuous resolution is tried
    trials = candidates(rover, base, hold=not directions, template=text)
    if n_random:
        trials += random_candidates(
            rover, base, n=n_random, seed=seed, hold=not directions
        )
    method = {
        "fix_rate": "fix rate",
        "reference": "reference track",
        "directions": "forward/backward",
    }[selection]
    logger.info(f"Tuning RTKLIB: {len(trials)} candidates, selection by {method}")

    def run(candidate: Candidate) -> Candidate:
        candidate.overrides.update(base_overrides)
        overrides = dict(candidate.overrides)
        try:
            if directions:
                results = []
                for soltype in ("forward", "backward"):
                    overrides["pos1-soltype"] = soltype
                    conf = workdir / f"{candidate.name}_{soltype}.conf"
                    conf.write_text(apply_overrides(text, overrides))
                    results.append(
                        read_pos(
                            run_rnx2rtkp(
                                conf,
                                Path(rover_obs),
                                Path(base_obs),
                                Path(nav),
                                workdir / f"{candidate.name}_{soltype}.pos",
                            )
                        )
                    )
                solution = results[0]
                check = (
                    compare_directions(*results)
                    if all(r.height for r in results)
                    else None
                )
            else:
                conf = workdir / f"{candidate.name}.conf"
                conf.write_text(apply_overrides(text, overrides))
                solution = read_pos(
                    run_rnx2rtkp(
                        conf,
                        Path(rover_obs),
                        Path(base_obs),
                        Path(nav),
                        workdir / f"{candidate.name}.pos",
                    )
                )
                if not by_reference:
                    candidate.epochs = solution.height
                    if solution.height:
                        candidate.fix_pct = float(
                            100 * (solution["quality"] == 1).sum() / solution.height
                        )
                    candidate.valid = candidate.fix_pct > 0
                    if not candidate.valid:
                        candidate.note = "no fixed epochs"
                    return candidate
                assert reference is not None
                score = (
                    score_against_reference(solution, reference)
                    if solution.height
                    else None
                )
                candidate.epochs = solution.height
                if score is None:
                    candidate.note = "no epoch inside the reference track"
                    return candidate
                candidate.fix_pct = score["fix_pct"]
                candidate.good_pct = score["good_pct"]
                candidate.wrong_pct = score["wrong_pct"]
                candidate.h_rms, candidate.v_rms = score["h_rms"], score["v_rms"]
                candidate.valid = (
                    candidate.good_pct > 0 and candidate.wrong_pct <= max_wrong_pct
                )
                if not candidate.valid:
                    candidate.note = (
                        "no good fixes"
                        if candidate.good_pct == 0
                        else f"{candidate.wrong_pct:.1f} % wrong fixes"
                    )
                return candidate
        except subprocess.CalledProcessError as e:
            candidate.note = f"rnx2rtkp failed: {e.stderr[-200:]!r}"
            return candidate
        candidate.epochs = solution.height
        if solution.height:
            candidate.fix_pct = float(
                100 * (solution["quality"] == 1).sum() / solution.height
            )
        if check is not None:
            candidate.h_rms, candidate.v_rms, candidate.bad_pct = check
            candidate.valid = (
                candidate.h_rms <= max_h_rms
                and candidate.v_rms <= max_v_rms
                and candidate.bad_pct <= max_bad_pct
            )
            if not candidate.valid:
                candidate.note = "fixes do not pass validation"
        else:
            candidate.note = "too few fixed epochs to validate"
        return candidate

    with ThreadPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        trials = list(pool.map(run, trials))

    valid = [c for c in trials if c.valid]

    def merit(c: Candidate) -> tuple[float, float]:
        if directions:
            return round(c.fix_pct, 1), -(c.h_rms or 0)
        if by_reference:
            return round(c.good_pct or 0.0, 1), -(c.wrong_pct or 0.0)
        return round(c.fix_pct, 1), 0.0

    best = max(valid, key=merit) if valid else None
    config_text = None
    if best is not None:
        config_text = apply_overrides(text, best.overrides)
        logger.info(
            f"Best RTKLIB config {best.name}: {best.fix_pct:.1f} % fixed"
            + (
                f", {best.good_pct:.1f} % good, {best.wrong_pct:.2f} % wrong"
                if by_reference
                else ""
            )
        )
    else:
        logger.warning("No candidate configuration passed validation")
    return TuningResult(
        best=best,
        trials=sorted(trials, key=lambda c: (not c.valid, *(-x for x in merit(c)))),
        rover=rover,
        base=base,
        config_text=config_text,
        validation=method,
        base_note=base_note,
        data_note=data_note,
    )
