import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from ovis.cli import main
from ovis.core.functions import quadratic, rosenbrock
from ovis.optimization.methods import gradient_descent
from ovis.optimization.plots import animate_descent, plot_descent


def test_path_shape_includes_start():
    path = gradient_descent(quadratic(), [1.0, 1.0], lr=0.01, steps=20)
    assert path.shape == (21, 2)
    np.testing.assert_allclose(path[0], [1.0, 1.0])


def test_quadratic_converges_to_minimizer():
    func = quadratic(a=1.0, b=10.0)
    path = gradient_descent(func, [2.0, -2.0], lr=0.05, steps=300)
    np.testing.assert_allclose(path[-1], func.minimizer, atol=1e-6)


def test_objective_decreases_monotonically_for_small_lr():
    func = quadratic(a=1.0, b=10.0)  # Lipschitz constant of the gradient is 20
    path = gradient_descent(func, [2.0, -2.0], lr=0.01, steps=100)
    values = np.array([func.f(x) for x in path])
    assert np.all(np.diff(values) <= 0)


def test_rosenbrock_objective_decreases():
    func = rosenbrock()
    path = gradient_descent(func, [-1.5, 2.5], lr=1e-4, steps=200)
    assert func.f(path[-1]) < func.f(path[0])


@pytest.mark.parametrize("lr, steps", [(0.0, 10), (-0.1, 10), (0.1, -1)])
def test_rejects_invalid_arguments(lr, steps):
    with pytest.raises(ValueError):
        gradient_descent(quadratic(), [1.0, 1.0], lr=lr, steps=steps)


def test_plot_descent_returns_figure():
    func = quadratic()
    path = gradient_descent(func, [2.0, -2.0], lr=0.05, steps=30)
    fig = plot_descent(func, path)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_animate_descent_has_one_frame_per_iterate():
    func = quadratic()
    path = gradient_descent(func, [2.0, -2.0], lr=0.05, steps=10)
    anim = animate_descent(func, path)
    assert anim._save_count == len(path)
    plt.close(anim._fig)


def test_cli_writes_png(tmp_path):
    out = tmp_path / "descent.png"
    args = ["gradient-descent", "--function", "rosenbrock", "--lr", "0.001", "--out", str(out)]
    assert main(args) == 0
    assert out.stat().st_size > 0
    plt.close("all")
