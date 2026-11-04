"""Plots of optimization trajectories over a function's level sets."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ovis.core.functions import Array, Function
from ovis.core.grids import evaluate_on_grid, grid


def draw_landscape(ax: Axes, func: Function, n: int = 200, levels: int = 30) -> None:
    """Draw filled contours of ``log(1 + f)`` over the function's domain.

    The log transform keeps steep valleys such as Rosenbrock's readable.
    """
    X, Y = grid(func.domain, n)
    Z = np.log1p(np.maximum(evaluate_on_grid(func, X, Y), 0.0))
    filled = ax.contourf(X, Y, Z, levels=levels, cmap="viridis")
    ax.figure.colorbar(filled, ax=ax, label="log(1 + f)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(func.name)
    ax.set_aspect("equal", adjustable="box")


def _mark_endpoints(ax: Axes, func: Function, path: Array) -> None:
    ax.plot(*path[0], "o", color="tab:green", label="start", zorder=5)
    ax.plot(*path[-1], "o", color="tab:red", label="end", zorder=5)
    if func.minimizer is not None:
        ax.plot(*func.minimizer, "*", color="white", markersize=14, label="minimizer", zorder=5)


def plot_descent(func: Function, path: Array) -> Figure:
    """Static figure showing a full optimization path over the landscape."""
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_landscape(ax, func)
    ax.plot(path[:, 0], path[:, 1], "-o", color="white", markersize=3, linewidth=1, zorder=4)
    _mark_endpoints(ax, func, path)
    ax.legend(loc="best", framealpha=0.8)
    fig.tight_layout()
    return fig


def animate_descent(func: Function, path: Array, interval: int = 100) -> FuncAnimation:
    """Animate the optimization path, one iterate per frame."""
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_landscape(ax, func)
    (trail,) = ax.plot([], [], "-", color="white", linewidth=1.5, zorder=4)
    (head,) = ax.plot([], [], "o", color="tab:orange", zorder=5)
    _mark_endpoints(ax, func, path)
    ax.legend(loc="best", framealpha=0.8)
    fig.tight_layout()

    def update(i: int):
        trail.set_data(path[: i + 1, 0], path[: i + 1, 1])
        head.set_data([path[i, 0]], [path[i, 1]])
        return trail, head

    return FuncAnimation(fig, update, frames=len(path), interval=interval, blit=True)
