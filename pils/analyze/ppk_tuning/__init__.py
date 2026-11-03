"""Tune the RTKLIB configuration to the data of a flight.

The best RTKLIB settings depend on the data: which signals the rover tracks
well (a payload antenna may have a weak L2), which constellations both
receivers see, how fast the data are. :func:`tune` profiles the rover and base
RINEX files, builds candidate configurations around a template (a grid from
the profile and :data:`N_RANDOM` random configurations), runs them all with
``rnx2rtkp`` in parallel, and keeps the best one that passes validation:

- with a reference track (the drone's RTK solution, flown about a metre from
  the payload antenna), every fixed epoch is scored against it after
  removing the median offset (lever arm): the most good fixes among the
  candidates with at most ``max_wrong_pct`` % wrong fixes;
- without one, the forward and backward solutions must agree on the epochs
  that both fix; this check is weaker (both directions can hold the same
  wrong ambiguities with fix-and-hold, which is then not tried).

The fix rate alone is not a safe criterion: lower ratio thresholds and
fix-and-hold raise it with wrong fixes (on flight_20251206_1833, 72 % fixed
but 1.5 % of the fixes off by more than a metre, against 65 % with none).

The random configurations also show how each setting acted on the flight
(:func:`setting_effects`).

Examples
--------
>>> result = tune(template, rover_obs, base_obs, nav, workdir,
...               reference=drone_track)  # doctest: +SKIP
>>> result.best.overrides, result.best.good_pct  # doctest: +SKIP
>>> result.effects  # doctest: +SKIP

The package:

- :mod:`~pils.analyze.ppk_tuning.profile`: profiles of the RINEX data
- :mod:`~pils.analyze.ppk_tuning.configs`: candidate configurations
- :mod:`~pils.analyze.ppk_tuning.runs`: running ``rnx2rtkp`` and scoring
- :mod:`~pils.analyze.ppk_tuning.search`: :func:`tune`

Every name of the former ``pils/analyze/ppk_tuning.py`` module is re-exported
here.
"""

from pils.analyze.ppk_tuning.configs import (  # noqa: F401
    MAX_BASE_OFFSET_M,
    N_RANDOM,
    Candidate,
    apply_overrides,
    base_position_check,
    candidates,
    config_value,
    random_candidates,
    snr_masks,
    usable_systems,
)
from pils.analyze.ppk_tuning.profile import (  # noqa: F401
    MAX_ROVER_LLI,
    MIN_COMMON_SATS,
    MIN_L2_SNR,
    MIN_ROVER_ARC_S,
    NAVSYS_BITS,
    ObsProfile,
    _epoch_time,
    data_quality_note,
    profile_obs,
)
from pils.analyze.ppk_tuning.runs import (  # noqa: F401
    GOOD_H_M,
    GOOD_V_M,
    SETTING_CHOICES,
    WRONG_M,
    compare_directions,
    compare_to_reference,
    read_pos,
    rtklib_binary,
    run_rnx2rtkp,
    score_against_reference,
    setting_effects,
)
from pils.analyze.ppk_tuning.search import (  # noqa: F401
    TuningResult,
    tune,
)

__all__ = [
    "Candidate",
    "GOOD_H_M",
    "GOOD_V_M",
    "MAX_BASE_OFFSET_M",
    "MAX_ROVER_LLI",
    "MIN_COMMON_SATS",
    "MIN_L2_SNR",
    "MIN_ROVER_ARC_S",
    "NAVSYS_BITS",
    "N_RANDOM",
    "ObsProfile",
    "SETTING_CHOICES",
    "TuningResult",
    "WRONG_M",
    "apply_overrides",
    "base_position_check",
    "candidates",
    "compare_directions",
    "compare_to_reference",
    "config_value",
    "data_quality_note",
    "profile_obs",
    "random_candidates",
    "read_pos",
    "rtklib_binary",
    "run_rnx2rtkp",
    "score_against_reference",
    "setting_effects",
    "snr_masks",
    "tune",
    "usable_systems",
]
