"""Helpers shared by the optimization visualizers (lagrange.py, kkt.py).

Parsing user input into SymPy expressions, solving small polynomial or numeric
systems in (x, y), sampling level sets, and a few plotting utilities, including
a secondary text window with headed sections for test results.
"""

import threading

import numpy as np
import sympy as sp
import contourpy
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from scipy.optimize import brentq, least_squares

SOLVE_TIMEOUT = 8.0     # seconds for SymPy before falling back to numeric root finding
NUMERIC_STARTS = 12     # numeric root finding starts from a 12 x 12 grid over the window
MAX_POINTS = 60         # more critical points than this means a curve of them

X = sp.Symbol("x", real=True)
Y = sp.Symbol("y", real=True)
XY = (X, Y)


def as_float(value, what):
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} is not a real number: {value}")
    if not np.isfinite(out):
        raise ValueError(f"{what} is not finite: {value}")
    return out


def vectorize(fn):
    def f(*coords):
        coords = np.broadcast_arrays(*(np.asarray(c, dtype=float) for c in coords))
        with np.errstate(all="ignore"):
            out = np.asarray(fn(*coords))
        if np.iscomplexobj(out):
            out = np.where(np.abs(out.imag) < 1e-12, out.real, np.nan)
        return np.broadcast_to(out.astype(float), coords[0].shape)
    return f


def at(fn, p):
    """Evaluate a lambdified (nested) list of expressions at the point p."""
    with np.errstate(all="ignore"):
        out = np.array(fn(*p), dtype=complex)
    return np.where(np.abs(out.imag) < 1e-12, out.real, np.nan)


def fmt(v):
    v = float(v)
    return "0" if abs(v) < 5e-13 else f"{v:.4g}"


def vec(v):
    return "(" + ", ".join(fmt(c) for c in v) + ")"


def with_timeout(fn, timeout):
    """Run fn() in a daemon thread; raise TimeoutError if it takes too long."""
    box = {}

    def run():
        try:
            box["out"] = fn()
        except Exception as err:
            box["err"] = err

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError
    if "err" in box:
        raise box["err"]
    return box["out"]


def parse_expr(text, what):
    try:
        expr = sp.sympify(text, locals={"x": X, "y": Y}, convert_xor=True)
    except (sp.SympifyError, SyntaxError, TypeError):
        raise ValueError(f"Could not parse {what} = {text.strip()!r}")
    if not isinstance(expr, sp.Expr):
        raise ValueError(f"{what} = {text.strip()!r} is not an expression")
    extra = expr.free_symbols - {X, Y}
    if extra:
        raise ValueError(f"Unknown variable(s) in {what}: {', '.join(sorted(map(str, extra)))} "
                         "(use x and y)")
    unknown = expr.atoms(sp.core.function.AppliedUndef)
    if unknown:
        raise ValueError(f"Unknown function(s) in {what}: {', '.join(sorted(map(str, unknown)))}")
    return expr


def parse_window(text):
    try:
        parsed = sp.sympify(text, convert_xor=True)
    except (sp.SympifyError, SyntaxError, TypeError):
        raise ValueError(f"Could not parse window = {text!r}")
    vals = tuple(parsed) if isinstance(parsed, (sp.Tuple, tuple)) else (parsed,)
    if len(vals) not in (1, 3):
        raise ValueError("window should be a half-width (e.g. 3) or cx, cy, half-width")
    vals = [as_float(v, "window") for v in vals]
    cx, cy, hw = (0.0, 0.0, vals[0]) if len(vals) == 1 else vals
    if hw <= 0:
        raise ValueError("The window half-width must be positive")
    return cx, cy, hw


def singular(expr):
    """True if expr may be discontinuous (a symbolic denominator, log, tan, ...)."""
    _, den = sp.fraction(sp.together(expr))
    return bool(den.free_symbols) or expr.has(sp.log, sp.tan, sp.cot, sp.sec, sp.csc,
                                              sp.sign, sp.floor, sp.ceiling)


def dedup(pts, scale):
    out = []
    for p in pts:
        if all(np.hypot(*(p - q)) > 1e-6 * max(1.0, scale) for q in out):
            out.append(np.asarray(p, dtype=float))
    return out


def numeric_solutions(eqs, window):
    """Real roots of eqs = 0 (at least two equations) by multistart least squares."""
    cx, cy, hw = window
    F = sp.lambdify(XY, list(eqs), "numpy")
    jac = sp.lambdify(XY, sp.Matrix(eqs).jacobian(XY).tolist(), "numpy")
    res = lambda z: at(F, z)
    jac_at = lambda z: at(jac, z)
    g = np.linspace(-hw, hw, NUMERIC_STARTS)
    pts = []
    for sx in g:
        for sy in g:
            try:
                sol = least_squares(res, [cx + sx, cy + sy], jac=jac_at, method="lm",
                                    xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=400)
            except (ValueError, np.linalg.LinAlgError):
                continue  # started outside the domain of an equation
            if np.all(np.isfinite(sol.fun)) and np.max(np.abs(sol.fun)) < 1e-9:
                pts.append(sol.x)
    return dedup(pts, hw)


def solve(eqs, window):
    """Real (x, y) solutions of eqs = 0.

    Returns (points, exact, continuum): exact is False when the numeric fallback
    (which only searches near the window) was used; continuum is True when part of
    the solution set is a curve rather than isolated points.
    """
    nums, dens = zip(*(sp.fraction(sp.together(e)) for e in eqs))
    if all(e.is_polynomial(X, Y) for e in nums + dens):
        try:
            sols = with_timeout(lambda: sp.solve(list(nums), XY, dict=True), SOLVE_TIMEOUT)
        except Exception:
            sols = None
        if sols is not None:
            den_fns = [vectorize(sp.lambdify(XY, d, "numpy")) for d in dens]
            pts, continuum = [], False
            for sol in sols:
                if any(s not in sol or sol[s].free_symbols for s in XY):
                    continuum = True
                    continue
                try:
                    z = [complex(sp.N(sol[s], 30)) for s in XY]
                except TypeError:
                    continue
                if max(abs(c.imag) for c in z) > 1e-9:
                    continue
                p = np.array([c.real for c in z])
                if all(abs(float(d(*p))) > 1e-12 for d in den_fns):
                    pts.append(p)
            return dedup(pts, window[2]), True, continuum
    pts = numeric_solutions(eqs, window)
    if len(pts) > MAX_POINTS:
        return [], False, True
    return pts, False, False


def circle_crossings(hs, c, r, n):
    """Points of S = {h_i = 0} on the circle of radius r around c.

    Finds sign changes of h_1 around the circle, refines each with brentq, and keeps
    those where the other constraints also hold.
    """
    t = np.linspace(0, 2 * np.pi, n + 1)
    v = hs[0](c[0] + r * np.cos(t), c[1] + r * np.sin(t))
    on_circle = lambda a: (c[0] + r * np.cos(a), c[1] + r * np.sin(a))
    h0 = lambda a: float(hs[0](*on_circle(a)))
    angles = list(t[:-1][v[:-1] == 0])
    for i in np.nonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0]:
        try:
            angles.append(brentq(h0, t[i], t[i + 1], xtol=1e-14))
        except ValueError:
            continue
    pts = np.array([on_circle(a) for a in angles]).reshape(-1, 2)
    for h in hs[1:]:
        pts = pts[np.abs(h(pts[:, 0], pts[:, 1])) < 1e-6 * (1 + r)]
    return pts


def value_limits(values, extra=()):
    """Display range for f: robust to asymptotes, always showing the critical values."""
    finite = values[np.isfinite(values)]
    lo, hi = np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0)
    lo, hi = min([lo, *extra]), max([hi, *extra])
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def level_curves(fn, X, Y, Z, level):
    """Pieces of the curve fn = level, from a grid Z = fn(X, Y), as N x 2 arrays.

    Contouring connects +inf to -inf across a pole (e.g. 1/x), so points where fn is
    not actually close to the level are replaced by NaN to break the line there.
    """
    gen = contourpy.contour_generator(X, Y, np.ma.masked_invalid(Z))
    scale = np.nanmedian(np.abs(Z - level)) if np.isfinite(Z).any() else 1.0
    out = []
    for seg in gen.lines(level):
        seg = np.array(seg, dtype=float)
        bad = ~(np.abs(fn(seg[:, 0], seg[:, 1]) - level) <= 0.1 * scale)
        seg[bad] = np.nan
        out.append(seg)
    return out


def set_opacity(artist, alpha):
    """Set an artist's alpha, hiding it entirely at 0 so no edges are left behind."""
    artist.set_alpha(alpha)
    artist.set_visible(alpha > 0)
    return artist


def disconnect_axes3d(ax3):
    """Disconnect the mouse handlers a 3D axes left on its canvas, before removing it.

    Axes3D connects mouse handlers to the canvas that remove() leaves behind; its
    release handler then fails on every click once the axes has no figure.
    """
    registry = ax3.get_figure(root=True).canvas.callbacks
    for signal in ("motion_notify_event", "button_press_event", "button_release_event"):
        for cid, ref in list(registry.callbacks.get(signal, {}).items()):
            if getattr(ref(), "__self__", None) is ax3:
                registry.disconnect(cid)


class SecondaryWindow:
    """Separate window of text sections, each with a small heading and a rule.

    Subclasses set SECTIONS to (heading, top) pairs in figure coordinates and
    VERDICT to the index of the section drawn bold in a tinted box.
    """

    TITLE = ""
    SIZE = (11, 7)
    SECTIONS = ()
    VERDICT = None

    def __init__(self, on_close):
        self.fig = plt.figure(figsize=self.SIZE)
        self.fig.canvas.manager.set_window_title(self.TITLE)
        self.bodies = []
        for heading, y in self.SECTIONS:
            self.fig.text(0.03, y, heading.upper(), va="bottom", fontsize=9, fontweight="bold",
                          color="dimgray")
            self.fig.add_artist(Line2D([0.03, 0.97], [y - 0.008] * 2, color="lightgray", lw=1,
                                       transform=self.fig.transFigure))
            self.bodies.append(self.fig.text(0.03, y - 0.03, "", va="top", fontsize=11,
                                             linespacing=1.4))
        if self.VERDICT is not None:
            self.bodies[self.VERDICT].set_fontweight("bold")
        self.fig.canvas.mpl_connect("close_event", lambda _: on_close())
        self.fig.show()

    def set_texts(self, title, texts, color="black"):
        self.fig.suptitle(title, fontsize=13)
        for body, text in zip(self.bodies, texts):
            body.set_text(text)
        if self.VERDICT is not None:
            # Verdict in a tinted box of its own color.
            box = self.bodies[self.VERDICT]
            box.set_color(color)
            box.set_bbox({"boxstyle": "round,pad=0.5", "facecolor": to_rgba(color, 0.08),
                          "edgecolor": color} if texts[self.VERDICT] else None)
        self.fig.canvas.draw_idle()

    def raise_(self):
        self.fig.canvas.manager.show()
