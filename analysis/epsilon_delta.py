"""Interactive visualizer for the epsilon-delta definition of a limit.

lim_{p -> a} f(p) = L  means: for every eps > 0 there is a delta > 0 such that
0 < |p - a| < delta  implies  |f(p) - L| < eps.

Type a function of x (one input) or of x and y (two inputs) and a point a, then
press Enter. With one input the plot is 2D: delta is an interval around a and
eps is a band around L. With two inputs the plot is 3D: delta is a disk around
a = (a1, a2) and eps is a slab L - eps < z < L + eps; a top view shows where
|f - L| < eps. Drag the sliders to pick eps and delta.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, TextBox

F_INIT = "x"
A_INIT = "2"
X_HALF_WIDTH = 3.0 # plotted range is a +- X_HALF_WIDTH in each input
EPS_INIT = 1.0
DELTA_INIT = 0.5
EPS_MAX = 2.0
DELTA_MAX = 2.0
SAMPLES = 2001     # points in a 1D delta-window
DISK_RADII = 60    # polar grid for a 2D delta-disk; radii are log-spaced so
DISK_ANGLES = 720  # thin features near a (e.g. along y = x^2) are not missed
GRID = 120         # grid size for the 3D surface and top view
# Default opacity of the f, eps and delta layers in each view.
ALPHAS_1D = {"f": 1.0, "eps": 0.15, "delta": 0.15}
ALPHAS_2D = {"f": 0.3, "eps": 0.2, "delta": 0.15}

X = sp.Symbol("x", real=True)
Y = sp.Symbol("y", real=True)


@dataclass
class Problem:
    dim: int           # number of inputs, 1 or 2
    f: Callable        # vectorized: f(x) or f(x, y)
    a: tuple           # point the limit is taken at, as floats
    L: float           # limit used for the eps-band
    expr: sp.Expr
    a_exprs: tuple     # a as SymPy numbers, for the title
    note: str          # extra status message, e.g. when no limit exists


def _as_float(value, what):
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} is not a real number: {value}")
    if not np.isfinite(out):
        raise ValueError(f"{what} is not finite: {value}")
    return out


def _limit_1d(expr, a_expr, a_text):
    """Return (L, note) for lim_{x -> a} expr."""
    try:
        return _as_float(sp.limit(expr, X, a_expr, dir="+-"), "The limit"), ""
    except ValueError as err:
        if "limit" not in str(err).lower():
            raise
    except NotImplementedError:
        raise ValueError(f"SymPy could not compute the limit of f at a = {a_text}")

    # Two-sided limit failed: report one-sided limits if they are finite.
    try:
        left = sp.limit(expr, X, a_expr, dir="-")
        right = sp.limit(expr, X, a_expr, dir="+")
    except Exception:
        raise ValueError(f"Could not compute the limit of f at a = {a_text}")
    if left == right:
        if isinstance(right, sp.AccumBounds):
            raise ValueError(f"The limit at a = {a_text} does not exist "
                             f"(f oscillates between {right.min} and {right.max})")
        raise ValueError(f"The limit at a = {a_text} is {right} (not a finite real number)")
    for side in (right, left):
        try:
            L = _as_float(side, "The limit")
        except ValueError:
            continue
        return L, f"No limit at a (left {left}, right {right}); testing L = {L:.4g}"
    raise ValueError(f"The limit at a = {a_text} does not exist (left: {left}, right: {right})")


def _limit_2d(expr, a1, a2, a_text):
    """Return (L, note) for lim_{(x, y) -> a} expr, via polar coordinates around a.

    If the polar limit depends on the angle, no limit exists. If it does not, it
    is only a candidate: the numeric disk check in delta_works catches functions
    like x^2*y/(x^4 + y^2) that agree along every line but not along curves.
    """
    r = sp.Symbol("r", positive=True)
    t = sp.Symbol("t", real=True)
    polar = expr.subs({X: a1 + r * sp.cos(t), Y: a2 + r * sp.sin(t)}, simultaneous=True)
    try:
        res = sp.simplify(sp.limit(polar, r, 0, "+"))
    except Exception:
        res = None

    if res is None:
        # SymPy failed: fall back to f(a) if f is defined there.
        try:
            L = _as_float(expr.subs({X: a1, Y: a2}, simultaneous=True), "f(a)")
        except ValueError:
            raise ValueError(f"SymPy could not compute the limit of f at a = {a_text}")
        return L, f"SymPy could not find the limit; testing L = f(a) = {L:.4g}"

    if isinstance(res, sp.AccumBounds):
        raise ValueError(f"The limit at a = {a_text} does not exist "
                         f"(f oscillates between {res.min} and {res.max})")
    if not res.has(t):
        return _as_float(res, f"The limit at a = {a_text}"), ""
    for angle in (0, sp.pi / 2, sp.pi / 4, sp.pi):
        try:
            L = _as_float(res.subs(t, angle), "The limit")
        except ValueError:
            continue
        return L, f"No limit at a: it depends on the direction of approach; testing L = {L:.4g}"
    raise ValueError(f"The limit at a = {a_text} does not exist (it depends on direction)")


def _vectorize(fn):
    def f(*coords):
        coords = np.broadcast_arrays(*(np.asarray(c, dtype=float) for c in coords))
        with np.errstate(all="ignore"):
            out = np.asarray(fn(*coords))
        if np.iscomplexobj(out):
            out = np.where(np.abs(out.imag) < 1e-12, out.real, np.nan)
        return np.broadcast_to(out.astype(float), coords[0].shape)
    return f


def parse(f_text, a_text):
    """Parse user input into a Problem. Raises ValueError with a readable message."""
    try:
        expr = sp.sympify(f_text, locals={"x": X, "y": Y}, convert_xor=True)
    except (sp.SympifyError, SyntaxError, TypeError):
        raise ValueError(f"Could not parse f = {f_text!r}")
    if not isinstance(expr, sp.Expr):
        raise ValueError(f"f = {f_text!r} is not an expression")
    extra = expr.free_symbols - {X, Y}
    if extra:
        raise ValueError(f"Unknown variable(s) in f: {', '.join(sorted(map(str, extra)))} "
                         "(use x, or x and y)")
    unknown = expr.atoms(sp.core.function.AppliedUndef)
    if unknown:
        raise ValueError(f"Unknown function(s) in f: {', '.join(sorted(map(str, unknown)))}")
    dim = 2 if Y in expr.free_symbols else 1

    try:
        a_parsed = sp.sympify(a_text, convert_xor=True)
    except (sp.SympifyError, SyntaxError, TypeError):
        raise ValueError(f"Could not parse a = {a_text!r}")
    a_exprs = tuple(a_parsed) if isinstance(a_parsed, (sp.Tuple, tuple)) else (a_parsed,)
    for coord in a_exprs:
        if not isinstance(coord, sp.Expr) or coord.free_symbols:
            raise ValueError(f"a must be a number or a pair of numbers, got {a_text!r}")
    if len(a_exprs) != dim:
        if dim == 2:
            raise ValueError("f uses x and y, so a needs two coordinates, e.g. 0, 0")
        raise ValueError("f uses only x, so a should be a single number "
                         "(use y in f for a function of two inputs)")
    a = tuple(_as_float(c, "a") for c in a_exprs)

    if dim == 1:
        L, note = _limit_1d(expr, a_exprs[0], a_text)
        fn = sp.lambdify(X, expr, "numpy")
    else:
        L, note = _limit_2d(expr, *a_exprs, a_text)
        fn = sp.lambdify((X, Y), expr, "numpy")
    return Problem(dim, _vectorize(fn), a, L, expr, a_exprs, note)


def _ball_samples(a, delta):
    """Sample points p with 0 < |p - a| <= delta, as a tuple of coordinate arrays."""
    a = np.atleast_1d(np.asarray(a, dtype=float))
    if a.size == 1:
        x = np.linspace(a[0] - delta, a[0] + delta, SAMPLES)
        return (x[np.abs(x - a[0]) > 1e-12 * max(1.0, abs(a[0]))],)
    r = np.geomspace(delta * 1e-4, delta, DISK_RADII)
    t = np.linspace(0, 2 * np.pi, DISK_ANGLES, endpoint=False)
    R, T = np.meshgrid(r, t)
    return (a[0] + R * np.cos(T)).ravel(), (a[1] + R * np.sin(T)).ravel()


def delta_works(f, a, L, eps, delta):
    """Return True if |f(p) - L| <= eps for all sampled p with 0 < |p - a| < delta.

    Sampling the closed ball and using <= matches the strict definition for
    continuous f, and lets delta == eps count as working for f(x) = x.
    Points outside the domain of f (NaN) are ignored.
    """
    dist = np.abs(f(*_ball_samples(a, delta)) - L)
    dist = dist[~np.isnan(dist)]
    return bool(np.all(dist <= eps + 1e-12))


def largest_delta(f, a, L, eps, upper=X_HALF_WIDTH, iters=40):
    """Bisect for the largest delta in (0, upper] that works for this eps."""
    if delta_works(f, a, L, eps, upper):
        return upper
    lo, hi = 0.0, upper
    for _ in range(iters):
        mid = (lo + hi) / 2
        if delta_works(f, a, L, eps, mid):
            lo = mid
        else:
            hi = mid
    return lo


def _value_limits(values, L):
    """Display range for f: robust to asymptotes, always showing L +- EPS_MAX."""
    finite = values[np.isfinite(values)]
    lo, hi = np.percentile(finite, [2, 98]) if finite.size else (L, L)
    lo, hi = min(lo, L - EPS_MAX * 1.1), max(hi, L + EPS_MAX * 1.1)
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def _set_opacity(artist, alpha):
    """Set an artist's alpha, hiding it entirely at 0 so no edges are left behind."""
    artist.set_alpha(alpha)
    artist.set_visible(alpha > 0)
    return artist


class View1D:
    """f: R -> R. Graph with an eps-band around L and a delta-window around a."""

    ALPHAS = ALPHAS_1D

    def __init__(self, fig, prob):
        self.prob = prob
        self.ax = fig.add_axes([0.08, 0.38, 0.86, 0.52])
        self.axes = [self.ax]
        self.dynamic = []

        ax, f, (a,), L = self.ax, prob.f, prob.a, prob.L
        xs = np.linspace(a - X_HALF_WIDTH, a + X_HALF_WIDTH, SAMPLES)
        ys = f(xs)
        lo, hi = _value_limits(ys, L)
        # Drop far off-screen values so asymptotes don't draw vertical lines.
        span = hi - lo
        ys = np.where((ys > lo - span) & (ys < hi + span), ys, np.nan)

        self.f_artist = ax.plot(xs, ys, color="black", lw=1.5)[0]
        ax.axvline(a, color="gray", ls="--", lw=1)
        ax.axhline(L, color="gray", ls="--", lw=1)
        ax.plot([a], [L], "o", mfc="white", mec="black", zorder=5)
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(lo, hi)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")

    def update(self, eps, delta, alphas):
        for artist in self.dynamic:
            artist.remove()
        self.dynamic.clear()
        ax, f, (a,), L = self.ax, self.prob.f, self.prob.a, self.prob.L

        _set_opacity(self.f_artist, alphas["f"])
        self.dynamic.append(_set_opacity(
            ax.axhspan(L - eps, L + eps, color="tab:blue"), alphas["eps"]))
        self.dynamic.append(_set_opacity(
            ax.axvspan(a - delta, a + delta, color="tab:orange"), alphas["delta"]))

        # Graph over the delta-window: green where inside the eps-band, red where not.
        (xw,) = _ball_samples(a, delta)
        yw = f(xw)
        inside = np.abs(yw - L) <= eps + 1e-12
        self.dynamic.append(ax.plot(xw, np.where(inside, yw, np.nan), color="tab:green", lw=4)[0])
        self.dynamic.append(ax.plot(xw, np.where(inside, np.nan, yw), color="tab:red", lw=4)[0])

        # Image of the delta-window projected onto the y-axis.
        finite = yw[np.isfinite(yw)]
        if finite.size:
            self.dynamic.append(ax.vlines(ax.get_xlim()[0], finite.min(), finite.max(),
                                          color="tab:purple", lw=6))

    def remove(self):
        for ax in self.axes:
            ax.remove()


class View2D:
    """f: R^2 -> R. 3D surface with an eps-slab and delta-cylinder, plus a top view."""

    ALPHAS = ALPHAS_2D

    def __init__(self, fig, prob, view_angle=None):
        self.prob = prob
        self.ax3 = fig.add_axes([0.0, 0.33, 0.52, 0.62], projection="3d")
        self.ax2 = fig.add_axes([0.58, 0.38, 0.38, 0.52])
        self.axes = [self.ax3, self.ax2]
        self.dynamic = []

        (a1, a2), L = prob.a, prob.L
        g = np.linspace(-X_HALF_WIDTH, X_HALF_WIDTH, GRID)
        self.X, self.Y = np.meshgrid(a1 + g, a2 + g)
        self.Z = prob.f(self.X, self.Y)
        self.lo, self.hi = _value_limits(self.Z, L)

        # 3D axes don't clip, so hide surface parts outside the z-range.
        shown = np.where((self.Z >= self.lo) & (self.Z <= self.hi), self.Z, np.nan)
        ax3 = self.ax3
        self.f_artist = ax3.plot_surface(self.X, self.Y, shown, color="gray", linewidth=0)
        ax3.scatter([a1], [a2], [L], facecolors="white", edgecolors="black", s=40,
                    depthshade=False, zorder=10)
        ax3.set_xlim(a1 - X_HALF_WIDTH, a1 + X_HALF_WIDTH)
        ax3.set_ylim(a2 - X_HALF_WIDTH, a2 + X_HALF_WIDTH)
        ax3.set_zlim(self.lo, self.hi)
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("f(x, y)")
        if view_angle:
            ax3.view_init(*view_angle)

        ax2 = self.ax2
        ax2.plot([a1], [a2], "o", mfc="white", mec="black", zorder=10)
        ax2.set_xlim(a1 - X_HALF_WIDTH, a1 + X_HALF_WIDTH)
        ax2.set_ylim(a2 - X_HALF_WIDTH, a2 + X_HALF_WIDTH)
        ax2.set_aspect("equal")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title(r"Top view: green where $|f(x,y) - L| < \varepsilon$", fontsize=10)

    def view_angle(self):
        return self.ax3.elev, self.ax3.azim

    def update(self, eps, delta, alphas):
        for artist in self.dynamic:
            artist.remove()
        self.dynamic.clear()
        ax3, ax2 = self.ax3, self.ax2
        f, (a1, a2), L = self.prob.f, self.prob.a, self.prob.L

        _set_opacity(self.f_artist, alphas["f"])

        # eps-slab: the planes z = L +- eps.
        px, py = np.meshgrid([a1 - X_HALF_WIDTH, a1 + X_HALF_WIDTH],
                             [a2 - X_HALF_WIDTH, a2 + X_HALF_WIDTH])
        for z0 in (L - eps, L + eps):
            self.dynamic.append(_set_opacity(ax3.plot_surface(
                px, py, np.full_like(px, z0), color="tab:blue", linewidth=0), alphas["eps"]))

        # delta-cylinder over the disk.
        t, z = np.meshgrid(np.linspace(0, 2 * np.pi, 60), [self.lo, self.hi])
        self.dynamic.append(_set_opacity(ax3.plot_surface(
            a1 + delta * np.cos(t), a2 + delta * np.sin(t), z, color="tab:orange", linewidth=0),
            alphas["delta"]))

        # Surface over the delta-disk: green inside the eps-slab, red outside.
        r, t = np.meshgrid(np.linspace(delta / 25, delta, 25),
                           np.linspace(0, 2 * np.pi, 90, endpoint=False))
        xd, yd = (a1 + r * np.cos(t)).ravel(), (a2 + r * np.sin(t)).ravel()
        zd = f(xd, yd)
        inside = np.abs(zd - L) <= eps + 1e-12
        shown = np.isfinite(zd) & (zd >= self.lo) & (zd <= self.hi)
        for mask, color in ((inside & shown, "tab:green"), (~inside & shown, "tab:red")):
            self.dynamic.append(ax3.scatter(xd[mask], yd[mask], zd[mask], color=color, s=4,
                                            depthshade=False))

        # Top view: the preimage of the eps-ball, the delta-circle, and failing points.
        dist = np.abs(self.Z - L)
        region = np.where(np.isnan(dist), np.nan, (dist <= eps + 1e-12).astype(float))
        self.dynamic.append(ax2.contourf(self.X, self.Y, region, levels=[-0.5, 0.5, 1.5],
                                         colors=["#e6e6e6", "#a8dca8"]))
        circle = Circle((a1, a2), delta, fill=False, color="tab:orange", lw=2, zorder=5)
        ax2.add_patch(circle)
        self.dynamic.append(circle)
        bad = ~inside & np.isfinite(zd)
        self.dynamic.append(ax2.scatter(xd[bad], yd[bad], color="tab:red", s=2, zorder=6))

    def remove(self):
        # Axes3D connects mouse handlers to the canvas that remove() leaves behind;
        # its release handler then fails on every click once the axes has no figure.
        registry = self.ax3.get_figure(root=True).canvas.callbacks
        for signal in ("motion_notify_event", "button_press_event", "button_release_event"):
            for cid, ref in list(registry.callbacks.get(signal, {}).items()):
                if getattr(ref(), "__self__", None) is self.ax3:
                    registry.disconnect(cid)
        for ax in self.axes:
            ax.remove()


def make_view(fig, prob, old=None):
    if prob.dim == 1:
        return View1D(fig, prob)
    return View2D(fig, prob, old.view_angle() if isinstance(old, View2D) else None)


def title(prob):
    if prob.dim == 1:
        a = sp.latex(prob.a_exprs[0])
        return rf"$f(x) = {sp.latex(prob.expr)}$,   $a = {a}$,   $L = {prob.L:.4g}$"
    a = ", ".join(sp.latex(c) for c in prob.a_exprs)
    return rf"$f(x, y) = {sp.latex(prob.expr)}$,   $a = ({a})$,   $L = {prob.L:.4g}$"


def build():
    """Create the visualizer window and return its figure (call plt.show() to run it)."""
    fig = plt.figure(figsize=(11, 8))
    fig.canvas.manager.set_window_title("ε-δ limits")
    prob = parse(F_INIT, A_INIT)
    state = {"prob": prob, "view": make_view(fig, prob)}

    status = fig.text(0.08, 0.315, "", va="top", fontsize=10, fontweight="bold")
    error = fig.text(0.1, 0.165, "", color="tab:red", fontsize=9)

    ax_f = fig.add_axes([0.1, 0.19, 0.5, 0.045])
    ax_a = fig.add_axes([0.7, 0.19, 0.2, 0.045])
    ax_eps = fig.add_axes([0.1, 0.11, 0.8, 0.03])
    ax_delta = fig.add_axes([0.1, 0.06, 0.8, 0.03])
    tb_f = TextBox(ax_f, "f = ", initial=F_INIT)
    tb_a = TextBox(ax_a, "a = ", initial=A_INIT)
    s_eps = Slider(ax_eps, r"$\varepsilon$", 0.05, EPS_MAX, valinit=EPS_INIT, color="tab:blue")
    s_delta = Slider(ax_delta, r"$\delta$", 0.05, DELTA_MAX, valinit=DELTA_INIT, color="tab:orange")

    # One opacity slider per layer, starting at the current view's defaults.
    alpha_sliders = {}
    for i, (key, label, color) in enumerate((("f", "f opacity", "gray"),
                                             ("eps", r"$\varepsilon$ opacity", "tab:blue"),
                                             ("delta", r"$\delta$ opacity", "tab:orange"))):
        ax_alpha = fig.add_axes([0.12 + 0.33 * i, 0.015, 0.15, 0.025])
        alpha_sliders[key] = Slider(ax_alpha, label, 0.0, 1.0,
                                    valinit=state["view"].ALPHAS[key], color=color)

    def update(_=None):
        if state.get("resetting"):
            return
        prob = state["prob"]
        eps, delta = s_eps.val, s_delta.val
        state["view"].update(eps, delta, {k: s.val for k, s in alpha_sliders.items()})

        best = largest_delta(prob.f, prob.a, prob.L, eps)
        best_text = f"≥ {best:.3f}" if best >= X_HALF_WIDTH else f"≈ {best:.3f}"
        if delta_works(prob.f, prob.a, prob.L, eps, delta):
            text, color = f"δ works for this ε   (largest working δ {best_text})", "tab:green"
        else:
            text, color = f"δ too large for this ε   (largest working δ {best_text})", "tab:red"
        if prob.note:
            text += "\n" + prob.note
        status.set_text(text)
        status.set_color(color)

        fig.canvas.draw_idle()

    def submit(_=None):
        try:
            prob = parse(tb_f.text, tb_a.text)
        except ValueError as err:
            error.set_text(str(err))
            fig.canvas.draw_idle()
            return
        old = state["view"]
        old.remove()
        state.update(prob=prob, view=make_view(fig, prob, old))
        if prob.dim != old.prob.dim:
            # New kind of view: reset opacities to its defaults, redrawing only once.
            state["resetting"] = True
            for key, slider in alpha_sliders.items():
                slider.set_val(state["view"].ALPHAS[key])
            state["resetting"] = False
        error.set_text("")
        fig.suptitle(title(prob))
        update()

    tb_f.on_submit(submit)
    tb_a.on_submit(submit)
    s_eps.on_changed(update)
    s_delta.on_changed(update)
    for slider in alpha_sliders.values():
        slider.on_changed(update)
    fig.suptitle(title(prob))
    update()
    # The canvas holds only weak references to widgets, so keep them alive with the figure.
    fig._keep = [tb_f, tb_a, s_eps, s_delta, alpha_sliders, state]
    return fig


def main():
    build()
    plt.show()


if __name__ == "__main__":
    main()
