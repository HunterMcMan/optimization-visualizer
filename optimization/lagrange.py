"""Interactive visualizer for Lagrange multipliers.

Find the extrema of f(x, y) subject to equality constraints g_i(x, y) = c_i. At a
constrained extremum p where the constraint gradients are linearly independent
(the constraint qualification, CQ),

    grad f(p) = lambda_1 grad g_1(p) + ... + lambda_m grad g_m(p),

so with one constraint the level curve of f through p is tangent to the
constraint curve. Points where the CQ fails (e.g. a cusp) can be extrema without
satisfying this, so they are found and reported separately.

Type f and the constraints (separated by ;) and press Enter. The left panel shows
contours of f, the constraint curves, every critical point, and at the selected
point the vectors grad f and lambda_i grad g_i drawn tip to tail. Drag k to sweep
the level curve f = k until it just touches the constraint. The right panel shows
the graph of f with the constraints lifted onto it and the plane z = k.

Each critical point is checked with the second-order sufficient condition (SOSC)
in two equivalent forms: the Hessian of the Lagrangian L = f - sum lambda_i g_i on
the tangent space of the constraint, and the bordered Hessian. When the SOSC is
inconclusive, f is compared with nearby points of the feasible set S. Finally the
shape of S (closed? bounded?) decides whether each point is a global extremum.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider, TextBox

if __package__:  # imported from the optimization package (app.py, the exe)
    from .common import (X, Y, XY, SecondaryWindow, at, circle_crossings,
                         disconnect_axes3d, fmt, level_curves, parse_expr, parse_window,
                         set_opacity, singular, solve, value_limits, vec, vectorize)
else:  # run as a script: python optimization/lagrange.py
    from common import (X, Y, XY, SecondaryWindow, at, circle_crossings,
                        disconnect_axes3d, fmt, level_curves, parse_expr, parse_window,
                        set_opacity, singular, solve, value_limits, vec, vectorize)

F_INIT = "x + y"
G_INIT = "x^2 + y^2 = 1"
W_INIT = "2"            # plot window: half-width, or "cx, cy, half-width"
GRID = 200              # grid size for the contour view
GRID_3D = 70            # grid size for the 3D surface
RANK_TOL = 1e-8         # relative singular value below which grad g_i are dependent
SOSC_TOL = 1e-9         # relative size below which q or a bordered minor counts as 0
LOCAL_RADII = (1e-3, 3e-3, 1e-2)  # circles (times half-width) for the numeric local check
FAR_RADIUS = 1e4        # S is sampled on circles out to this radius (times half-width)
FAR_CIRCLES = 40
CIRCLE_ANGLES = 2048
ARROW_LEN = 0.35        # longest gradient arrow, as a fraction of the half-width
LIST_ROWS = 5           # critical points listed in the summary
ALPHAS = {"f": 0.5, "constraint": 1.0, "level": 0.3}
CONSTRAINT_COLORS = ["tab:purple", "tab:brown", "tab:cyan", "tab:olive"]
KIND_COLORS = {"max": "tab:red", "min": "tab:blue", "neither": "dimgray"}
OTHER_COLOR = "tab:green"  # isolated points, f constant nearby, unknown


@dataclass
class Constraint:
    h: sp.Expr         # lhs - rhs, so the constraint is h = 0 and grad h = grad g
    latex: str         # the constraint as typed, e.g. "x^{2} + y^{2} = 1"


@dataclass
class CritPoint:
    p: np.ndarray                 # (x, y)
    f: float
    grad_f: np.ndarray
    J: np.ndarray                 # m x 2 Jacobian of the constraints at p
    rank: int                     # rank of J
    lam: np.ndarray               # multipliers (least squares when the CQ fails)
    lam_exists: bool              # grad f lies in the span of the grad g_i
    d: Optional[np.ndarray] = None   # unit tangent to S (one effective constraint)
    q: Optional[float] = None        # d^T (Hessian of L) d
    minors: list = field(default_factory=list)  # bordered Hessian minors checked
    tangent: str = "n/a"          # verdict of each test: min, max, inconclusive,
    bordered: str = "n/a"         # isolated, or n/a when the CQ fails
    local: str = "unknown"        # min, max, neither, isolated, flat or unknown
    local_numeric: bool = False   # local verdict comes from sampling S near p
    glob: str = ""                # "max", "min", "both" or ""

    @property
    def cq(self):
        return self.rank == len(self.J)


@dataclass
class Problem:
    expr: sp.Expr
    cons: list           # Constraint
    window: tuple        # (cx, cy, half-width)
    f: Callable          # vectorized f(x, y)
    hs: list             # vectorized h_i(x, y)
    points: list         # CritPoint, largest f first
    compact: bool        # S is known to be closed and bounded
    S_text: str          # description of the feasible set
    has_max: bool        # f attains a global max on S (at a critical point)
    has_min: bool
    notes: list          # extra messages


def _parse_constraints(text):
    pieces = [piece.strip() for piece in text.split(";") if piece.strip()]
    if not pieces:
        raise ValueError("Enter at least one constraint, e.g. x^2 + y^2 = 1")
    cons = []
    for i, piece in enumerate(pieces, 1):
        if any(op in piece for op in ("<", ">", "==", "!=")):
            raise ValueError(f"Constraint {i} must be an equation g = c "
                             "(inequality constraints are not supported)")
        sides = piece.split("=")
        if len(sides) > 2:
            raise ValueError(f"Constraint {i} ({piece}) has more than one '='")
        lhs = parse_expr(sides[0], f"constraint {i}")
        rhs = parse_expr(sides[1], f"constraint {i}") if len(sides) == 2 else sp.Integer(0)
        h = lhs - rhs
        if not h.free_symbols or sp.simplify(h) == 0:
            always = sp.simplify(h) == 0
            raise ValueError(f"Constraint {i} ({piece}) is {'always' if always else 'never'} "
                             "satisfied")
        cons.append(Constraint(h, f"{sp.latex(lhs)} = {sp.latex(rhs)}"))
    return cons


def _local_numeric(prob_f, hs, p, f0, hw):
    """Compare f(p) with f on S near p: min, max, neither, flat, or isolated."""
    for r in np.array(LOCAL_RADII) * hw:
        near = circle_crossings(hs, p, r, 720)
        if len(near):
            break
    else:
        return "isolated"
    diffs = prob_f(near[:, 0], near[:, 1]) - f0
    tol = 1e-12 * (1 + abs(f0))
    up, down = np.any(diffs > tol), np.any(diffs < -tol)
    if up and down:
        return "neither"
    return "min" if up else "max" if down else "flat"


def _sign_text(v, tol):
    return "= 0" if abs(v) <= tol else "> 0" if v > 0 else "< 0"


def _analyze(p, f_fn, hs, derivs, hw):
    """Lagrange multipliers and second-order tests at the critical point p."""
    grad_f_fn, hess_f_fn, jac_fn, hess_h_fns = derivs
    m = len(hs)
    grad_f = at(grad_f_fn, p)
    J = at(jac_fn, p).reshape(m, 2)
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int(np.sum(sv > RANK_TOL * max(1.0, sv.max())))
    lam = np.linalg.lstsq(J.T, grad_f, rcond=None)[0]
    lam_exists = np.linalg.norm(J.T @ lam - grad_f) <= 1e-7 * max(1.0, np.linalg.norm(grad_f))
    cp = CritPoint(p, float(f_fn(*p)), grad_f, J, rank, lam, bool(lam_exists))

    if cp.cq:
        H_L = at(hess_f_fn, p) - sum(l * at(fn, p) for l, fn in zip(lam, hess_h_fns))
        B = np.block([[np.zeros((m, m)), J], [J.T, H_L]])
        # Last n - m leading principal minors (n = 2): sign (-1)^m for a min,
        # sign (-1)^(k - m) for the k x k minor for a max.
        ks = range(2 * m + 1, m + 3)
        cp.minors = [np.linalg.det(B[:k, :k]) for k in ks]
        tols = [SOSC_TOL * (1 + np.linalg.norm(B[:k, :k])) ** k for k in ks]
        if not cp.minors:
            cp.bordered = "isolated"
        elif any(abs(d) <= t for d, t in zip(cp.minors, tols)):
            cp.bordered = "inconclusive"
        elif all(np.sign(d) == (-1) ** m for d in cp.minors):
            cp.bordered = "min"
        elif all(np.sign(d) == (-1) ** (k - m) for d, k in zip(cp.minors, ks)):
            cp.bordered = "max"
        else:
            cp.bordered = "inconclusive"

        if m == 1:
            d = np.array([-J[0, 1], J[0, 0]])
            cp.d = d / np.linalg.norm(d)
            cp.q = float(cp.d @ H_L @ cp.d)
            sign = _sign_text(cp.q, SOSC_TOL * (1 + np.linalg.norm(H_L)))
            cp.tangent = {"> 0": "min", "< 0": "max", "= 0": "inconclusive"}[sign]
        else:
            cp.tangent = "isolated"  # two independent constraints: T = {0}

    if cp.tangent in ("min", "max", "isolated"):
        cp.local = cp.tangent
    else:
        cp.local = _local_numeric(f_fn, hs, p, cp.f, hw)
        cp.local_numeric = True
    return cp


def parse(f_text, g_text, w_text):
    """Parse user input, find and classify the critical points. Raises ValueError."""
    expr = parse_expr(f_text, "f")
    cons = _parse_constraints(g_text)
    window = parse_window(w_text)
    cx, cy, hw = window
    m = len(cons)
    hexprs = [c.h for c in cons]
    f = vectorize(sp.lambdify(XY, expr, "numpy"))
    hs = [vectorize(sp.lambdify(XY, h, "numpy")) for h in hexprs]
    derivs = (sp.lambdify(XY, [sp.diff(expr, s) for s in XY], "numpy"),
              sp.lambdify(XY, sp.hessian(expr, XY).tolist(), "numpy"),
              sp.lambdify(XY, [[sp.diff(h, s) for s in XY] for h in hexprs], "numpy"),
              [sp.lambdify(XY, sp.hessian(h, XY).tolist(), "numpy") for h in hexprs])
    notes = []

    if m == 2:
        # Constraints whose gradients are parallel everywhere are redundant
        # (x + y = 1; 2x + 2y = 2) or inconsistent (x = 0; x = 1).
        rng = np.random.default_rng(0)
        ranks = []
        for p in rng.uniform(-hw, hw, (20, 2)) + (cx, cy):
            J = at(derivs[2], p)
            if np.all(np.isfinite(J)):
                sv = np.linalg.svd(J, compute_uv=False)
                ranks.append(np.sum(sv > RANK_TOL * max(1.0, sv.max())))
        if ranks and max(ranks) < 2:
            raise ValueError("The two constraint gradients are parallel everywhere, so the "
                             "constraints are redundant or inconsistent; remove one")
    if m > 2:
        notes.append(f"{m} constraints in 2 variables: the CQ fails at every feasible point")

    # Candidates. One constraint: h = 0 and grad f || grad h, i.e. the 2x2
    # determinant f_x h_y - f_y h_x = 0 (this eliminates lambda and also catches the
    # points where grad h = 0). Two or more: every feasible point is a candidate.
    if m == 1:
        h = hexprs[0]
        det = sp.diff(expr, X) * sp.diff(h, Y) - sp.diff(expr, Y) * sp.diff(h, X)
        if sp.simplify(det) == 0:
            raise ValueError("grad f is parallel to grad g everywhere, so f is constant on "
                             "each piece of S: every feasible point is critical")
        pts, exact, continuum = solve([h, det], window)
        if continuum:
            notes.append("f is constant along part of S (a curve of critical points, not shown)")
    else:
        pts, exact, continuum = solve(hexprs, window)
        if continuum:
            notes.append("The constraints share a whole curve of solutions (not shown)")
    if not exact:
        notes.append("SymPy could not solve exactly: critical points were found numerically "
                     "near the window")
    points = sorted((_analyze(p, f, hs, derivs, hw) for p in pts), key=lambda cp: -cp.f)
    if any(not cp.cq for cp in points) and m <= 2:
        notes.append("◇ marks a point where the CQ fails: Lagrange's condition can miss these")

    # Feasible set: closed unless a constraint has a singularity; bounded if it never
    # meets a far-away circle. With two or more constraints S is the finite set of
    # feasible points found above.
    closed = not any(singular(h) for h in hexprs)
    samples = np.empty((0, 2))
    if m == 1:
        radii = np.geomspace(1e-2 * hw, FAR_RADIUS * max(1.0, hw), FAR_CIRCLES)
        rings = [circle_crossings(hs, (cx, cy), R, CIRCLE_ANGLES) for R in radii]
        samples = np.vstack(rings)
        bounded = len(rings[-1]) == 0
        if not points and not len(samples):
            raise ValueError("No feasible points found: the constraint seems to have no "
                             "real solutions (S is empty)")
        size = "bounded" if bounded else "unbounded"
    else:
        if not points and not continuum:
            raise ValueError("No feasible points: the constraints have no common solution"
                             + ("" if exact else " near the window"))
        bounded = not continuum
        size = f"finite ({len(points)} point{'s' * (len(points) != 1)})" if bounded else "unbounded?"
    compact = closed and bounded
    S_text = f"S is {'closed' if closed else 'maybe not closed (a constraint is singular)'}, {size}"
    if compact:
        S_text += " ⇒ compact: f has a global max and min on S (EVT)"
    elif closed and not bounded:
        S_text += " ⇒ not compact: global extrema may not exist"
    if singular(expr):
        notes.append("f may be discontinuous, so the extreme value theorem may not apply")

    # A global max on S (if any) is at a candidate. It fails to exist exactly when some
    # point of S has a larger f than every candidate; S is sampled to check this.
    has_max = has_min = False
    if points:
        fs = np.array([cp.f for cp in points])
        tol = 1e-9 * (1 + np.abs(fs).max())
        sf = f(samples[:, 0], samples[:, 1])
        sf = sf[np.isfinite(sf)]
        has_max = not (sf.size and sf.max() > fs.max() + tol)
        has_min = not (sf.size and sf.min() < fs.min() - tol)
        for cp in points:
            is_max = has_max and cp.f >= fs.max() - tol and cp.local not in ("min", "neither")
            is_min = has_min and cp.f <= fs.min() + tol and cp.local not in ("max", "neither")
            cp.glob = "both" if is_max and is_min else "max" if is_max else "min" if is_min else ""
        has_max = any(cp.glob in ("max", "both") for cp in points)
        has_min = any(cp.glob in ("min", "both") for cp in points)
    return Problem(expr, cons, window, f, hs, points, compact, S_text, has_max, has_min, notes)


def _kind(cp):
    """Which way the point is an extremum, for its marker and color."""
    if cp.glob in ("max", "min"):
        return cp.glob
    if cp.glob == "both":
        return "other"
    return cp.local if cp.local in KIND_COLORS else "other"


def label(cp):
    if cp.glob == "both":
        return "global min & max"
    if cp.glob:
        return f"global {cp.glob}"
    return {"min": "local min", "max": "local max", "neither": "neither",
            "isolated": "local min & max (isolated)", "flat": "f constant nearby",
            "unknown": "unknown"}[cp.local]


class View:
    """Contours of f with the constraints (left) and the graph of f (right)."""

    def __init__(self, fig, prob, view_angle=None):
        self.prob = prob
        self.ax2 = fig.add_axes([0.05, 0.22, 0.40, 0.71])
        self.ax3 = fig.add_axes([0.47, 0.18, 0.52, 0.80], projection="3d")
        self.axes = [self.ax2, self.ax3]
        self.dynamic = []

        cx, cy, hw = prob.window
        g = np.linspace(-hw, hw, GRID)
        self.X, self.Y = np.meshgrid(cx + g, cy + g)
        self.Z = prob.f(self.X, self.Y)
        self.shown = [cp for cp in prob.points
                      if abs(cp.p[0] - cx) <= hw and abs(cp.p[1] - cy) <= hw]
        self.lo, self.hi = value_limits(self.Z, [cp.f for cp in self.shown])
        Zc = np.ma.masked_invalid(np.clip(self.Z, self.lo, self.hi))

        ax = self.ax2
        levels = np.linspace(self.lo, self.hi, 16)
        self.fill = ax.contourf(self.X, self.Y, Zc, levels=levels, cmap="Greys")
        self.lines = ax.contour(self.X, self.Y, Zc, levels=levels, colors="gray", linewidths=0.5)
        self.con_artists, handles, self.con_segs = [], [], []
        for i, (con, h) in enumerate(zip(prob.cons, prob.hs)):
            color = CONSTRAINT_COLORS[i % len(CONSTRAINT_COLORS)]
            segs = level_curves(h, self.X, self.Y, h(self.X, self.Y), 0.0)
            self.con_segs.append(segs)
            for seg in segs:
                self.con_artists.append(ax.plot(seg[:, 0], seg[:, 1], color=color, lw=2)[0])
            handles.append(Line2D([], [], color=color, lw=2, label=f"${con.latex}$"))
        ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.85)
        for cp in prob.points:
            ax.plot(*cp.p, **self._marker_style(cp), ms=9, zorder=6)
        ax.set_xlim(cx - hw, cx + hw)
        ax.set_ylim(cy - hw, cy + hw)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(r"Contours of $f$.  Orange: $f = k$.  Dashed: $f = f(p)$", fontsize=9)

        ax3 = self.ax3
        g3 = np.linspace(-hw, hw, GRID_3D)
        X3, Y3 = np.meshgrid(cx + g3, cy + g3)
        Z3 = prob.f(X3, Y3)
        # 3D axes don't clip, so hide parts outside the z-range.
        Z3 = np.where((Z3 >= self.lo) & (Z3 <= self.hi), Z3, np.nan)
        self.surface = ax3.plot_surface(X3, Y3, Z3, color="gray", linewidth=0)
        for i, segs in enumerate(self.con_segs):
            color = CONSTRAINT_COLORS[i % len(CONSTRAINT_COLORS)]
            for seg in segs:
                z = prob.f(seg[:, 0], seg[:, 1])
                z = np.where((z >= self.lo) & (z <= self.hi), z, np.nan)
                self.con_artists.append(ax3.plot(seg[:, 0], seg[:, 1], z, color=color, lw=2)[0])
                self.con_artists.append(ax3.plot(seg[:, 0], seg[:, 1], np.full(len(seg), self.lo),
                                                 color=color, lw=1, ls="--")[0])
        for cp in self.shown:
            style = self._marker_style(cp)
            ax3.scatter([cp.p[0]], [cp.p[1]], [cp.f], marker=style["marker"], s=60,
                        facecolors=style["mfc"], edgecolors=style["mec"], depthshade=False,
                        zorder=10)
        ax3.set_xlim(cx - hw, cx + hw)
        ax3.set_ylim(cy - hw, cy + hw)
        ax3.set_zlim(self.lo, self.hi)
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("f(x, y)")
        if view_angle:
            ax3.view_init(*view_angle)

    @staticmethod
    def _marker_style(cp):
        """Shape by kind (diamond where the CQ fails); filled if global, hollow if local."""
        kind = _kind(cp)
        color = KIND_COLORS.get(kind, OTHER_COLOR)
        marker = "D" if not cp.cq else {"max": "^", "min": "v", "neither": "X"}.get(kind, "o")
        return {"marker": marker, "mec": color, "mfc": color if cp.glob else "white",
                "ls": "none"}

    def view_angle(self):
        return self.ax3.elev, self.ax3.azim

    def update(self, sel, k, alphas):
        for artist in self.dynamic:
            artist.remove()
        self.dynamic.clear()
        ax, ax3, prob = self.ax2, self.ax3, self.prob
        cx, cy, hw = prob.window

        for artist in (self.fill, self.lines, self.surface):
            set_opacity(artist, alphas["f"])
        for artist in self.con_artists:
            set_opacity(artist, alphas["constraint"])

        # Level curve f = k and the plane z = k.
        for seg in level_curves(prob.f, self.X, self.Y, self.Z, k):
            self.dynamic.append(ax.plot(seg[:, 0], seg[:, 1], color="tab:orange", lw=2)[0])
        px, py = np.meshgrid([cx - hw, cx + hw], [cy - hw, cy + hw])
        self.dynamic.append(set_opacity(ax3.plot_surface(
            px, py, np.full_like(px, k), color="tab:orange", linewidth=0), alphas["level"]))

        if sel is None:
            return
        cp = prob.points[sel]
        self.dynamic.append(ax.plot(*cp.p, "o", ms=17, mfc="none", mec="black", mew=1.5,
                                    zorder=7)[0])
        if self.lo < cp.f < self.hi:
            for seg in level_curves(prob.f, self.X, self.Y, self.Z, cp.f):
                self.dynamic.append(ax.plot(seg[:, 0], seg[:, 1], color="black", ls="--",
                                            lw=1.2, zorder=5)[0])
            self.dynamic.append(ax3.scatter([cp.p[0]], [cp.p[1]], [cp.f], s=250,
                                            facecolors="none", edgecolors="black",
                                            depthshade=False, zorder=11))
        if cp.d is not None:
            ends = np.array([cp.p - 0.4 * hw * cp.d, cp.p + 0.4 * hw * cp.d])
            self.dynamic.append(ax.plot(ends[:, 0], ends[:, 1], ":", color="black", lw=1.5)[0])

        # grad f, and lambda_i grad g_i tip to tail ending at the tip of grad f. Where
        # no multipliers exist (the CQ fails), show grad g_i itself instead.
        m = len(prob.cons)
        if cp.lam_exists:
            parts = [(cp.lam[i] * cp.J[i], r"\lambda" + (f"_{i + 1}" if m > 1 else "")
                      + r"\nabla g" + (f"_{i + 1}" if m > 1 else "")) for i in range(m)]
        else:
            parts = [(cp.J[i], r"\nabla g" + (f"_{i + 1}" if m > 1 else "")) for i in range(m)]
        size = max([np.linalg.norm(cp.grad_f)] + [np.linalg.norm(v) for v, _ in parts])
        if size == 0:
            return
        s = ARROW_LEN * hw / size
        arrows = [(cp.p, cp.grad_f, "black", r"\nabla f")]
        start = cp.p.copy()
        for i, (v, name) in enumerate(parts):
            color = CONSTRAINT_COLORS[i % len(CONSTRAINT_COLORS)]
            arrows.append((start, v, color, name))
            if cp.lam_exists:
                start = start + s * v
        # Label grad f at its tip and the others at their middles, since with one
        # constraint lambda grad g lies exactly on top of grad f.
        for j, ((x0, y0), (u, v), color, name) in enumerate(arrows):
            if np.hypot(u, v) == 0:
                continue
            self.dynamic.append(ax.quiver(x0, y0, s * u, s * v, angles="xy", scale_units="xy",
                                          scale=1, color=color, width=0.009 if j == 0 else 0.005,
                                          zorder=8))
            at, offset = (1.0, (5, 5)) if j == 0 else (0.5, (6, -14))
            self.dynamic.append(ax.annotate(f"${name}$", (x0 + at * s * u, y0 + at * s * v),
                                            color=color, fontsize=10, xytext=offset,
                                            textcoords="offset points", zorder=9))

    def remove(self):
        disconnect_axes3d(self.ax3)
        for ax in self.axes:
            ax.remove()


def title(prob):
    cons = ",   ".join(f"${c.latex}$" for c in prob.cons)
    return rf"$f(x, y) = {sp.latex(prob.expr)}$     subject to     {cons}"


VERDICT = {"min": "strict local min", "max": "strict local max", "inconclusive": "inconclusive"}


def describe(prob, sel):
    """Text about the selected point: (point, second-order tests, verdict, verdict color)."""
    if sel is None:
        return ("No critical points: f has no extremum on S", "", "", "black")
    cp, m, n = prob.points[sel], len(prob.cons), len(prob.points)
    tol = SOSC_TOL * (1 + np.abs(cp.minors).max()) if cp.minors else 0
    lines = [f"Point {sel + 1} of {n}:   p = {vec(cp.p)},   f(p) = {fmt(cp.f)}"]
    lam = fmt(cp.lam[0]) if m == 1 else vec(cp.lam)
    grads = ",  ".join(rf"$\nabla g{'_' + str(i + 1) if m > 1 else ''}$ = {vec(row)}"
                       for i, row in enumerate(cp.J))
    if cp.cq:
        lines.append(rf"$\nabla f$ = {vec(cp.grad_f)},  {grads},  $\lambda$ = {lam}")
    else:
        why = "λ is not unique" if cp.lam_exists else "no λ exists"
        lines.append(rf"CQ fails (rank J = {cp.rank} < {m}): $\nabla f$ = {vec(cp.grad_f)}, "
                     f"{grads}; {why}")

    if cp.tangent == "isolated":
        lines.append(r"Tangent test: $T = \{0\}$ ($\nabla g_1, \nabla g_2$ independent) "
                     "⇒ p is an isolated point of S")
        lines.append(r"Bordered Hessian: $\bar H$ is 4×4 and n − m = 0, so no minors to check")
    elif cp.cq:
        qt = SOSC_TOL * (1 + abs(cp.q))
        lines.append(rf"Tangent test: $d$ = {vec(cp.d)},  $q = d^T\nabla^2 L\,d$ = {fmt(cp.q)} "
                     f"{_sign_text(cp.q, qt)} ⇒ {VERDICT[cp.tangent]}")
        det = cp.minors[0]
        check = -float(cp.J[0] @ cp.J[0]) * cp.q
        lines.append(rf"Bordered Hessian: $\det\bar H$ = {fmt(det)} {_sign_text(det, tol)} "
                     rf"⇒ {VERDICT[cp.bordered]}   (agrees: $-|\nabla g|^2 q$ = {fmt(check)})")
    else:
        lines.append("Tangent test and bordered Hessian: not applicable where the CQ fails")

    how = {"min": "f is larger at nearby points of S", "max": "f is smaller at nearby points of S",
           "neither": "f is larger on one side of p along S and smaller on the other",
           "flat": "f is constant along S near p", "isolated": "no other points of S nearby",
           "unknown": ""}
    if cp.local_numeric:
        why = "SOSC inconclusive" if cp.cq else "SOSC does not apply"
        local = {"min": "local min", "max": "local max", "neither": "neither a min nor a max",
                 "flat": "non-strict local min and max", "isolated": "strict local min and max",
                 "unknown": "unknown"}[cp.local]
        local = f"Local: {local}: {how[cp.local]} ({why}; checked numerically)"
    elif cp.local == "isolated":
        local = "Local: strict local min and max on S (isolated feasible point)"
    else:
        local = f"Local: strict local {cp.local} (second-order sufficient condition holds)"

    fs = [c.f for c in prob.points]
    why = "S is compact (EVT)" if prob.compact else "no sampled point of S does better"
    if cp.glob == "both":
        glob = f"Global: global max and min: f takes the same value at every candidate; {why}"
    elif cp.glob:
        word = "largest" if cp.glob == "max" else "smallest"
        glob = f"Global: global {cp.glob}: the {word} value of f among all candidates; {why}"
    elif cp.local == "neither":
        glob = "Global: neither a global max nor a global min"
    else:
        parts = []
        if cp.local in ("max", "isolated", "flat"):
            parts.append("not a global max: " + (f"f = {fmt(max(fs))} at another candidate"
                                                 if prob.has_max else
                                                 "f is larger elsewhere on S (no global max)"))
        if cp.local in ("min", "isolated", "flat"):
            parts.append("not a global min: " + (f"f = {fmt(min(fs))} at another candidate"
                                                 if prob.has_min else
                                                 "f is smaller elsewhere on S (no global min)"))
        glob = "Global: " + ";  ".join(parts or ["unknown"])
    color = ("tab:green" if cp.glob else "tab:red" if cp.local == "neither" else "tab:orange")
    return "\n".join(lines[:2]), "\n".join(lines[2:]), local + "\n" + glob, color


def summary(prob, sel):
    gmax = [cp for cp in prob.points if cp.glob in ("max", "both")]
    gmin = [cp for cp in prob.points if cp.glob in ("min", "both")]
    extreme = lambda cps, word: (f"global {word} {fmt(cps[0].f)} at {len(cps)} point"
                                 + "s" * (len(cps) > 1) if cps else f"no global {word}")
    lines = ["Feasible set: " + prob.S_text,
             f"Result: {extreme(gmax, 'max')},  {extreme(gmin, 'min')}",
             "Critical points (click one, or use ◀ ▶):"]
    start = 0 if sel is None else min(max(0, sel - LIST_ROWS // 2),
                                      max(0, len(prob.points) - LIST_ROWS))
    for i in range(start, min(start + LIST_ROWS, len(prob.points))):
        cp = prob.points[i]
        mark = "▸" if i == sel else "   "
        lines.append(f"{mark} {i + 1}. {vec(cp.p)}   f = {fmt(cp.f)}   {label(cp)}"
                     + ("   ◇ CQ fails" if not cp.cq else ""))
    if len(prob.points) > LIST_ROWS:
        lines.append(f"     ({len(prob.points)} points in all)")
    return "\n".join(lines + prob.notes)


class TestsWindow(SecondaryWindow):
    """Separate window with the second-order and global tests for the selected point."""

    TITLE = "Lagrange multipliers: tests"
    # (heading, top of the section) in figure coordinates; each body starts just below.
    SECTIONS = (("Selected point", 0.89), ("Second-order sufficient condition", 0.73),
                ("Verdict", 0.57), ("Feasible set and all critical points", 0.39))
    VERDICT = 2

    def update(self, prob, sel):
        point, tests, verdict, color = describe(prob, sel)
        self.set_texts(title(prob), (point, tests, verdict, summary(prob, sel)), color)


def build():
    """Create the visualizer window and return its figure (call plt.show() to run it)."""
    fig = plt.figure(figsize=(13, 8))
    fig.canvas.manager.set_window_title("Lagrange multipliers")
    prob = parse(F_INIT, G_INIT, W_INIT)
    state = {"prob": prob, "view": View(fig, prob), "sel": 0 if prob.points else None,
             "tests": None}

    error = fig.text(0.1, 0.083, "", color="tab:red", fontsize=9)

    ax_f = fig.add_axes([0.08, 0.1, 0.3, 0.035])
    ax_g = fig.add_axes([0.5, 0.1, 0.28, 0.035])
    ax_w = fig.add_axes([0.87, 0.1, 0.08, 0.035])
    tb_f = TextBox(ax_f, "f = ", initial=F_INIT)
    tb_g = TextBox(ax_g, "constraints ", initial=G_INIT)
    tb_w = TextBox(ax_w, "window ", initial=W_INIT)
    b_prev = Button(fig.add_axes([0.66, 0.047, 0.08, 0.03]), "◀ prev")
    b_next = Button(fig.add_axes([0.75, 0.047, 0.08, 0.03]), "next ▶")
    b_tests = Button(fig.add_axes([0.85, 0.047, 0.1, 0.03]), "show tests")

    def small(slider):
        slider.label.set_fontsize(8)
        slider.valtext.set_fontsize(8)
        return slider

    # One opacity slider per layer.
    alpha_sliders = {}
    for i, (key, text, color) in enumerate((("f", "f opacity", "gray"),
                                            ("constraint", "constraint opacity", "tab:purple"),
                                            ("level", "plane z = k opacity", "tab:orange"))):
        ax_alpha = fig.add_axes([0.14 + 0.33 * i, 0.018, 0.15, 0.016])
        alpha_sliders[key] = small(Slider(ax_alpha, text, 0.0, 1.0, valinit=ALPHAS[key],
                                          color=color))

    def make_k_slider():
        # The range of k follows the range of f in the window, so rebuild it.
        old = state.get("k")
        if old is not None:
            old.disconnect_events()
            old.ax.remove()
        view = state["view"]
        slider = small(Slider(fig.add_axes([0.1, 0.053, 0.47, 0.018]), "level k", view.lo,
                              view.hi, valinit=(view.lo + view.hi) / 2, color="tab:orange"))
        slider.on_changed(update)
        state["k"] = slider

    def update(_=None):
        prob, view, sel = state["prob"], state["view"], state["sel"]
        view.update(sel, state["k"].val, {k: s.val for k, s in alpha_sliders.items()})
        if state["tests"] is not None:
            state["tests"].update(prob, sel)
        fig.canvas.draw_idle()

    def show_tests(_=None):
        if state["tests"] is None:
            state["tests"] = TestsWindow(on_close=lambda: state.update(tests=None))
            state["tests"].update(state["prob"], state["sel"])
        else:
            state["tests"].raise_()

    def submit(_=None):
        try:
            prob = parse(tb_f.text, tb_g.text, tb_w.text)
        except ValueError as err:
            error.set_text(str(err))
            fig.canvas.draw_idle()
            return
        old = state["view"]
        old.remove()
        state.update(prob=prob, view=View(fig, prob, old.view_angle()),
                     sel=0 if prob.points else None)
        make_k_slider()
        error.set_text("")
        fig.suptitle(title(prob))
        update()

    def step(delta):
        n = len(state["prob"].points)
        if n:
            state["sel"] = (state["sel"] + delta) % n
            update()

    def on_click(event):
        view, points = state["view"], state["prob"].points
        if event.inaxes is not view.ax2 or not points:
            return
        disp = view.ax2.transData.transform(np.array([cp.p for cp in points]))
        dist = np.hypot(disp[:, 0] - event.x, disp[:, 1] - event.y)
        if dist.min() < 15:
            state["sel"] = int(dist.argmin())
            update()

    tb_f.on_submit(submit)
    tb_g.on_submit(submit)
    tb_w.on_submit(submit)
    b_prev.on_clicked(lambda _: step(-1))
    b_next.on_clicked(lambda _: step(1))
    b_tests.on_clicked(show_tests)
    fig.canvas.mpl_connect("button_press_event", on_click)
    for slider in alpha_sliders.values():
        slider.on_changed(update)
    make_k_slider()
    fig.suptitle(title(prob))
    update()
    # The canvas holds only weak references to widgets, so keep them alive with the figure.
    fig._keep = [tb_f, tb_g, tb_w, b_prev, b_next, b_tests, alpha_sliders, state]
    return fig


def main():
    build()
    plt.show()


if __name__ == "__main__":
    main()
