"""Smooth test functions of two variables, each with its gradient."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class Function:
    """A smooth scalar function f(x, y) with its gradient and plotting domain.

    ``f`` and ``grad`` take a length-2 array ``[x, y]``. ``f`` must also accept
    arrays of shape ``(2, ...)`` so it can be evaluated on a grid.
    """

    name: str
    f: Callable[[Array], Array]
    grad: Callable[[Array], Array]
    domain: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    minimizer: Array | None = None
    start: tuple[float, float] = (0.0, 0.0)  # default starting point for optimizers


def quadratic(a: float = 1.0, b: float = 10.0) -> Function:
    """Elliptical bowl a*x^2 + b*y^2, minimized at the origin."""
    return Function(
        name=f"quadratic(a={a:g}, b={b:g})",
        f=lambda x: a * x[0] ** 2 + b * x[1] ** 2,
        grad=lambda x: np.array([2 * a * x[0], 2 * b * x[1]]),
        domain=(-3.0, 3.0, -3.0, 3.0),
        minimizer=np.zeros(2),
        start=(-2.5, 2.5),
    )


def rosenbrock() -> Function:
    """Rosenbrock's banana-shaped valley, minimized at (1, 1)."""
    return Function(
        name="rosenbrock",
        f=lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2,
        grad=lambda x: np.array(
            [
                -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
                200 * (x[1] - x[0] ** 2),
            ]
        ),
        domain=(-2.0, 2.0, -1.0, 3.0),
        minimizer=np.ones(2),
        start=(-1.5, 2.5),
    )


REGISTRY: dict[str, Callable[[], Function]] = {
    "quadratic": quadratic,
    "rosenbrock": rosenbrock,
}
