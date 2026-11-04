"""Helpers for sampling functions on a rectangular grid."""

from __future__ import annotations

import numpy as np

from ovis.core.functions import Array, Function


def grid(domain: tuple[float, float, float, float], n: int = 200) -> tuple[Array, Array]:
    """Return meshgrid arrays ``X, Y`` covering ``domain`` with ``n`` points per axis."""
    xmin, xmax, ymin, ymax = domain
    x = np.linspace(xmin, xmax, n)
    y = np.linspace(ymin, ymax, n)
    return np.meshgrid(x, y)


def evaluate_on_grid(func: Function, X: Array, Y: Array) -> Array:
    """Evaluate ``func.f`` at every point of the grid ``X, Y``."""
    return func.f(np.stack([X, Y]))
