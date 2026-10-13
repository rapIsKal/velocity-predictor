"""Estimate velocity Vx(t) from noisy scatter samples of position x(t)."""
import numpy as np
from scipy.interpolate import make_smoothing_spline

MIN_POINTS_FOR_SPLINE = 5


def _prepare(t, x):
    """Sort by t and average x over duplicate t values."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    order = np.argsort(t)
    t, x = t[order], x[order]
    t_unique, inverse = np.unique(t, return_inverse=True)
    x_mean = np.bincount(inverse, weights=x) / np.bincount(inverse)
    return t_unique, x_mean


def predict_velocity(t, x, t_eval=None, smoothing=None):
    """Fit x(t) and return (t_eval, x_fit, vx) evaluated at t_eval.

    Uses a cubic smoothing spline; the smoothing strength is chosen
    automatically by generalized cross-validation unless `smoothing`
    (the spline's lambda, >= 0; 0 = interpolate exactly) is given.
    With too few points, falls back to finite differences.
    """
    t_u, x_u = _prepare(t, x)
    if len(t_u) < 2:
        raise ValueError("need at least 2 distinct t values")

    t_eval = t_u if t_eval is None else np.asarray(t_eval, dtype=float)

    if len(t_u) < MIN_POINTS_FOR_SPLINE:
        vx_u = np.gradient(x_u, t_u)
        return t_eval, np.interp(t_eval, t_u, x_u), np.interp(t_eval, t_u, vx_u)

    spline = make_smoothing_spline(t_u, x_u, lam=smoothing)
    return t_eval, spline(t_eval), spline.derivative()(t_eval)
