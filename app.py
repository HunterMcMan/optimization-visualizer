"""Landing page for the Optimization Visualizer: pick a tool to open it.

Run with `python app.py`, or build a single Windows executable with
`python build_exe.py`. Each card opens one visualizer in its own window; several
can be open at once, and the app exits when the last window is closed.
"""

import base64
import io
from pathlib import Path
import sys
import traceback

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch, Wedge
from matplotlib.widgets import Button

# Colors shared with the tools: blue min / feasible region, red max, orange level or
# delta, purple constraint, green "works".
BLUE, RED, ORANGE, PURPLE, GREEN = "tab:blue", "tab:red", "tab:orange", "tab:purple", "tab:green"
PANEL = "#f5f5f5"
RULE = "lightgray"
HEADING = "dimgray"


def _open_epsilon_delta():
    from analysis import epsilon_delta
    return epsilon_delta.build()


def _open_lagrange():
    from optimization import lagrange
    return lagrange.build()


def _open_kkt():
    from optimization import kkt
    return kkt.build()


# (section, name, description, illustration, opener); opener None means coming soon.
TOOLS = [
    ("Analysis", "ε-δ limits",
     "Drag ε and δ to test the definition\nof a limit for f(x) or f(x, y).",
     "epsilon_delta", _open_epsilon_delta),
    ("Optimization", "Lagrange multipliers",
     "Equality constraints: ∇f = λ∇g,\nsecond-order tests, global extrema.",
     "lagrange", _open_lagrange),
    ("Optimization", "KKT conditions",
     "Inequality constraints, the cone of\nactive gradients, and CQs.",
     "kkt", _open_kkt),
    ("Topology", "Coming soon",
     "Open sets, continuity and\ncompactness, visualized.",
     "topology", None),
]


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_logo(ax):
    """Contours of a quadratic, a feasible disk, and the KKT point on its boundary."""
    g = np.linspace(-1.3, 2.3, 200)
    X, Y = np.meshgrid(g, g)
    c = np.array([1.7, 1.275])
    Z = (X - c[0]) ** 2 + (Y - c[1]) ** 2
    ax.contourf(X, Y, Z, levels=14, cmap="Greys", alpha=0.55)
    ax.contour(X, Y, Z, levels=14, colors="gray", linewidths=0.4)
    ax.add_patch(Circle((0, 0), 1, color=BLUE, alpha=0.25, lw=0))
    ax.add_patch(Circle((0, 0), 1, fill=False, color=PURPLE, lw=2))
    p = c / np.linalg.norm(c)
    ax.contour(X, Y, Z, levels=[float(np.sum((p - c) ** 2))], colors="black",
               linestyles="--", linewidths=1)
    for length, color, width in ((1.0, PURPLE, 2.6), (0.65, "black", 3.0)):
        ax.annotate("", p + length * p, p,
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": width,
                                "mutation_scale": 16})
    ax.plot(*p, "v", color=BLUE, ms=13, mec="white", mew=1, zorder=5)
    ax.set_xlim(-1.3, 2.3)
    ax.set_ylim(-1.3, 2.3)
    ax.set_aspect("equal")
    _bare(ax)


def _draw_epsilon_delta(ax):
    a, L, eps, delta = 1.0, 1.0, 0.45, 0.35
    x = np.linspace(-0.2, 2.2, 300)
    y = 0.5 * x ** 2 + 0.5
    ax.axhspan(L - eps, L + eps, color=BLUE, alpha=0.15, lw=0)
    ax.axvspan(a - delta, a + delta, color=ORANGE, alpha=0.18, lw=0)
    ax.plot(x, y, color="black", lw=1.5)
    w = np.abs(x - a) <= delta
    ax.plot(x[w], y[w], color=GREEN, lw=4)
    ax.plot([a], [L], "o", mfc="white", mec="black", zorder=5)
    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(0.2, 2.4)


def _draw_lagrange(ax):
    g = np.linspace(-1.6, 1.6, 120)
    X, Y = np.meshgrid(g, g)
    ax.contour(X, Y, X + Y, levels=10, colors="gray", linewidths=0.5)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color=PURPLE, lw=2)
    p = np.array([1, 1]) / np.sqrt(2)
    ax.plot(g, np.sqrt(2) - g, color="black", ls="--", lw=1)
    for length, color, width in ((0.7, PURPLE, 2.2), (0.45, "black", 2.6)):
        ax.annotate("", p + length * p, p,
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": width})
    ax.plot(*p, "^", color=RED, ms=10, mec="white", zorder=5)
    ax.plot(*-p, "v", color=BLUE, ms=10, mec="white", zorder=5)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.15, 1.44)


def _draw_kkt(ax):
    g = np.linspace(-1.2, 1.6, 120)
    X, Y = np.meshgrid(g, g)
    ax.contour(X, Y, (X + 1) ** 2 + (Y + 0.6) ** 2, levels=10, colors="gray", linewidths=0.5)
    ax.fill_between([0, 1.6], 0, 1.6, color=BLUE, alpha=0.25, lw=0)
    ax.plot([0, 0], [0, 1.6], color=PURPLE, lw=2)
    ax.plot([0, 1.6], [0, 0], color="tab:brown", lw=2)
    ax.add_patch(Wedge((0, 0), 0.9, 180, 270, color=BLUE, alpha=0.18, lw=0))
    for v, color in (((-0.8, 0), PURPLE), ((0, -0.8), "tab:brown")):
        ax.annotate("", v, (0, 0), arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2})
    d = np.array([-1, -0.6]) / np.hypot(1, 0.6) * 0.75
    ax.annotate("", d, (0, 0), arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 2.6})
    ax.plot(0, 0, "v", color=BLUE, ms=10, mec="white", zorder=5)
    ax.set_xlim(-1.2, 1.6)
    ax.set_ylim(-0.85, 1.42)


def _draw_topology(ax):
    t = np.linspace(0, 2 * np.pi, 200)
    blob = 1 + 0.25 * np.sin(3 * t) + 0.1 * np.cos(5 * t)
    ax.fill(blob * np.cos(t), blob * np.sin(t), color="lightgray", alpha=0.6, lw=0)
    ax.plot(blob * np.cos(t), blob * np.sin(t), color="gray", lw=1.5, ls="--")
    ax.add_patch(Circle((0.25, 0.1), 0.35, fill=False, color="gray", lw=1.2))
    ax.plot(0.25, 0.1, "o", color="gray", ms=4)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.3, 1.3)


ILLUSTRATIONS = {"epsilon_delta": _draw_epsilon_delta, "lagrange": _draw_lagrange,
                 "kkt": _draw_kkt, "topology": _draw_topology}


def logo_png(size=256):
    """The logo as PNG bytes (used for the window and exe icons)."""
    fig = Figure(figsize=(size / 100, size / 100), dpi=100)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    draw_logo(ax)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    return buf.getvalue()


def set_icon(fig):
    """Use the logo as the window icon (Tk backend only; ignored elsewhere)."""
    try:
        import tkinter
        window = fig.canvas.manager.window
        photo = tkinter.PhotoImage(master=window, data=base64.b64encode(logo_png(64)))
        window.iconphoto(False, photo)
        window._logo_icon = photo  # Tk drops images that Python no longer references
    except Exception:
        pass


def build():
    """Create the landing page and return its figure."""
    fig = plt.figure(figsize=(12, 7.8), facecolor="white")
    fig.canvas.manager.set_window_title("Optimization Visualizer")

    # Header: logo, title, subtitle, rule.
    logo = fig.add_axes([0.05, 0.69, 0.167, 0.256])
    draw_logo(logo)
    fig.add_artist(FancyBboxPatch((0.05, 0.69), 0.167, 0.256, boxstyle="round,pad=0.006",
                                  transform=fig.transFigure, fc="none", ec=RULE, lw=1))
    fig.text(0.25, 0.855, "Optimization Visualizer", fontsize=28, fontweight="bold",
             va="center")
    fig.text(0.25, 0.785, "Interactive visuals for analysis, optimization and topology",
             fontsize=13, color=HEADING, va="center")
    fig.text(0.25, 0.735, "Choose a visualizer below; each one opens in its own window.",
             fontsize=10, color="gray", va="center")
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.655, 0.655], color=RULE, lw=1,
                              transform=fig.transFigure))

    # Cards, grouped under section headings like the tests windows.
    left, width, gap = 0.05, 0.205, 0.0283
    xs = [left + i * (width + gap) for i in range(len(TOOLS))]
    sections = {}
    for x, (section, *_rest) in zip(xs, TOOLS):
        lo, hi = sections.get(section, (x, x + width))
        sections[section] = (min(lo, x), max(hi, x + width))
    for section, (lo, hi) in sections.items():
        fig.text(lo, 0.605, section.upper(), fontsize=9, fontweight="bold", color=HEADING)
        fig.add_artist(plt.Line2D([lo, hi], [0.597, 0.597], color=RULE, lw=1,
                                  transform=fig.transFigure))

    status = fig.text(0.05, 0.025, "", fontsize=9, color="gray")
    buttons = []
    for x, (section, name, text, art, opener) in zip(xs, TOOLS):
        soon = opener is None
        fig.add_artist(FancyBboxPatch((x, 0.06), width, 0.51, boxstyle="round,pad=0.004",
                                      transform=fig.transFigure, fc=PANEL, ec=RULE, lw=1, zorder=-1))
        ax = fig.add_axes([x + 0.012, 0.325, width - 0.024, 0.225], facecolor="white")
        ILLUSTRATIONS[art](ax)
        _bare(ax)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(RULE)
        fig.text(x + 0.012, 0.285, name, fontsize=13, fontweight="bold",
                 color="gray" if soon else "black", va="center")
        fig.text(x + 0.012, 0.255, text, fontsize=9, va="top", linespacing=1.4,
                 color="gray" if soon else "#333333")
        if soon:
            fig.text(x + width / 2, 0.11, "coming soon", fontsize=10, style="italic",
                     color="gray", ha="center", va="center")
            continue
        button = Button(fig.add_axes([x + 0.012, 0.085, width - 0.024, 0.05]), "Open  ▸")
        button.label.set_fontsize(11)
        button.on_clicked(_opener(fig, status, name, opener))
        buttons.append(button)

    set_icon(fig)
    fig._keep = buttons  # the canvas holds only weak references to widgets
    fig._status = status
    return fig


def _opener(fig, status, name, opener):
    def open_tool(_):
        status.set_text(f"Opening {name}…")
        status.set_color("gray")
        fig.canvas.draw()
        fig.canvas.flush_events()
        try:
            tool = opener()
            set_icon(tool)
            tool.show()
            status.set_text(f"Opened {name}.")
        except Exception as err:
            traceback.print_exc()
            status.set_text(f"Could not open {name}: {err}")
            status.set_color(RED)
        fig.canvas.draw_idle()
    return open_tool


def self_test(report=None):
    """Press every Open button on the landing page, as a user would, then close all
    windows. Exit code 0 if every tool opened; the status lines go to `report`."""
    landing = build()
    lines = []
    for button in landing._keep:
        button._observers.process("clicked", None)
        lines.append(landing._status.get_text())
    plt.close("all")
    if report:
        Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(0 if all(line.startswith("Opened") for line in lines) else 1)


def main():
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("TkAgg")
    if "--self-test" in sys.argv:
        i = sys.argv.index("--self-test")
        self_test(sys.argv[i + 1] if i + 1 < len(sys.argv) else None)
    build()
    plt.show()


if __name__ == "__main__":
    main()
