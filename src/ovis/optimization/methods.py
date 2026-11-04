"""Iterative optimization methods that return their full trajectory."""

from __future__ import annotations

import numpy as np

from ovis.core.functions import Array, Function


def gradient_descent(
    func: Function,
    x0: Array | tuple[float, float],
    lr: float = 0.1,
    steps: int = 50,
) -> Array:
    """Run fixed-step gradient descent from ``x0``.

    Returns the iterates ``x_0, ..., x_steps`` as an array of shape ``(steps + 1, 2)``.
    """
    if lr <= 0:
        raise ValueError(f"learning rate must be positive, got {lr}")
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")

    x = np.asarray(x0, dtype=float)
    path = [x]
    for _ in range(steps):
        x = x - lr * func.grad(x)
        path.append(x)
    return np.array(path)
