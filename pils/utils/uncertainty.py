"""Interpolation of calibration curves with uncertainty propagation."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d

from pils.utils.logging_config import get_logger

logger = get_logger(__name__)


def _sorted_calibration(
    x_cal: ArrayLike, y_cal: ArrayLike, dx_cal: ArrayLike, dy_cal: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x_cal, dtype=float)
    order = np.argsort(x)
    return (
        x[order],
        np.asarray(y_cal, dtype=float)[order],
        np.asarray(dx_cal, dtype=float)[order],
        np.asarray(dy_cal, dtype=float)[order],
    )


def _interpolate(
    x: np.ndarray, y: np.ndarray, x_new: ArrayLike, method: str
) -> np.ndarray:
    order = np.argsort(x)
    f = interp1d(x[order], y[order], kind=method, fill_value="extrapolate")
    return np.asarray(f(x_new), dtype=float)


def _jacobian(func, params: np.ndarray, n_out: int) -> np.ndarray:
    """(n_out, n_params) numerical Jacobian (numdifftools)."""
    import numdifftools as nd

    if n_out == 1:  # nd.Jacobian mishandles a single output
        gradient = nd.Gradient(lambda p: float(func(p)[0]))(params)
        return np.atleast_2d(gradient)
    return np.asarray(nd.Jacobian(func)(params))


def interpolate_with_errors(
    x_cal: ArrayLike,
    y_cal: ArrayLike,
    dx_cal: ArrayLike,
    dy_cal: ArrayLike,
    x_new: float,
    dx_new: float,
    method: str = "linear",
) -> tuple[float, float]:
    """Interpolate y at ``x_new`` and propagate the uncertainties.

    The uncertainty combines in quadrature the contributions of ``x_new``
    and of every calibration point (x and y), using numerical derivatives
    of the interpolation (numdifftools).

    Parameters
    ----------
    x_cal, y_cal : array_like
        Calibration points.
    dx_cal, dy_cal : array_like
        1σ uncertainties of the calibration points.
    x_new : float
        Where to interpolate.
    dx_new : float
        1σ uncertainty of ``x_new``.
    method : str
        ``interp1d`` kind: 'linear', 'quadratic' or 'cubic'.

    Returns
    -------
    tuple[float, float]
        Interpolated value and its 1σ uncertainty.

    Examples
    --------
    >>> y, dy = interpolate_with_errors([1, 2], [2, 4], [0, 0], [0, 0], 1.5, 0.1)
    >>> round(y, 6), round(dy, 6)
    (3.0, 0.2)
    """
    y, dy = interpolate_with_errors_jacobian(
        x_cal, y_cal, dx_cal, dy_cal, [x_new], [dx_new], method=method
    )
    return float(y[0]), float(dy[0])


def interpolate_with_errors_jacobian(
    x_cal: ArrayLike,
    y_cal: ArrayLike,
    dx_cal: ArrayLike,
    dy_cal: ArrayLike,
    x_new: ArrayLike,
    dx_new: ArrayLike,
    method: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised :func:`interpolate_with_errors` for many ``x_new`` values.

    Correlations between the new points through the shared calibration are
    not returned; each point gets its own 1σ uncertainty.
    """
    x, y, dx, dy = _sorted_calibration(x_cal, y_cal, dx_cal, dy_cal)
    x_new = np.atleast_1d(np.asarray(x_new, dtype=float))
    dx_new = np.atleast_1d(np.asarray(dx_new, dtype=float))
    n_new, n_cal = len(x_new), len(x)

    outside = (x_new < x[0]) | (x_new > x[-1])
    if np.any(outside):
        logger.warning(
            f"{int(outside.sum())} point(s) outside the calibration range "
            f"[{x[0]}, {x[-1]}]: extrapolated"
        )

    def func(params: np.ndarray) -> np.ndarray:
        return _interpolate(
            params[n_new : n_new + n_cal],
            params[n_new + n_cal :],
            params[:n_new],
            method,
        )

    params = np.concatenate([x_new, x, y])
    jac = _jacobian(func, params, n_new)
    var = (np.diagonal(jac[:, :n_new]) * dx_new) ** 2
    var += np.sum((jac[:, n_new : n_new + n_cal] * dx) ** 2, axis=1)
    var += np.sum((jac[:, n_new + n_cal :] * dy) ** 2, axis=1)
    return _interpolate(x, y, x_new, method), np.sqrt(var)


def mc_interp_sigma(
    x_cal: ArrayLike,
    y_cal: ArrayLike,
    dx_cal: ArrayLike,
    dy_cal: ArrayLike,
    x_new: float,
    dx_new: float,
    method: str = "linear",
    n: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Monte Carlo cross-check of :func:`interpolate_with_errors`.

    Returns the mean and standard deviation of ``n`` interpolations with
    Gaussian-perturbed inputs.
    """
    rng = rng or np.random.default_rng()
    x, y, dx, dy = _sorted_calibration(x_cal, y_cal, dx_cal, dy_cal)
    samples = np.empty(n)
    for i in range(n):
        samples[i] = _interpolate(
            x + rng.normal(0, dx),
            y + rng.normal(0, dy),
            x_new + rng.normal(0, dx_new),
            method,
        )
    return float(samples.mean()), float(samples.std(ddof=1))
