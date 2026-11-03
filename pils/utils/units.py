"""Power and gain unit conversions (dB, dBm, mW)."""

import numpy as np
from numpy.typing import ArrayLike


def db_to_linear(value_db: ArrayLike) -> np.ndarray:
    """Power ratio in dB -> linear ratio.

    Examples
    --------
    >>> float(db_to_linear(10.0))
    10.0
    """
    return np.power(10.0, np.asarray(value_db, dtype=float) / 10.0)


def linear_to_db(value: ArrayLike) -> np.ndarray:
    """Linear power ratio -> dB.

    Examples
    --------
    >>> float(linear_to_db(100.0))
    20.0
    """
    return 10.0 * np.log10(np.asarray(value, dtype=float))


def mw_to_dbm(power_mw: ArrayLike) -> np.ndarray:
    """Power in mW -> dBm."""
    return linear_to_db(power_mw)


def dbm_to_mw(power_dbm: ArrayLike) -> np.ndarray:
    """Power in dBm -> mW."""
    return db_to_linear(power_dbm)
