"""Interpolation of calibration curves with uncertainty propagation."""

import numpy as np
from numpy.typing import ArrayLike


def linear_interpolate_rows(
    x_cal: ArrayLike,
    y_cal: ArrayLike,
    dx_cal: ArrayLike,
    dy_cal: ArrayLike,
    x_new: ArrayLike,
    dx_new: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation with errors, one calibration curve per point.

    Row ``i`` of the (n, m) calibration arrays is the curve for ``x_new[i]``;
    outside the curve the end segments are extended. The uncertainty combines
    in quadrature the contributions of ``x_new`` and of the two calibration
    points (x and y) of its segment, with the analytic derivatives.
    """
    x = np.atleast_2d(np.asarray(x_cal, dtype=float))
    order = np.argsort(x, axis=1)
    x = np.take_along_axis(x, order, axis=1)
    y, dx, dy = (
        np.take_along_axis(np.atleast_2d(np.asarray(a, dtype=float)), order, axis=1)
        for a in (y_cal, dx_cal, dy_cal)
    )
    x_new = np.asarray(x_new, dtype=float)
    dx_new = np.asarray(dx_new, dtype=float)
    rows = np.arange(len(x_new))

    # Segment of each point: interior, or the end segment when outside
    upper = np.array([np.searchsorted(r, v) for r, v in zip(x, x_new, strict=True)])
    i = np.clip(upper - 1, 0, x.shape[1] - 2)
    x0, x1 = x[rows, i], x[rows, i + 1]
    y0, y1 = y[rows, i], y[rows, i + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (y1 - y0) / (x1 - x0)
        t = (x_new - x0) / (x1 - x0)
    value = y0 + (y1 - y0) * t
    var = (
        (slope * dx_new) ** 2
        + (slope * (1 - t) * dx[rows, i]) ** 2
        + (slope * t * dx[rows, i + 1]) ** 2
        + ((1 - t) * dy[rows, i]) ** 2
        + (t * dy[rows, i + 1]) ** 2
    )
    return value, np.sqrt(var)
