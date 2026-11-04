"""Command-line entry point: ``ovis <command> [options]``."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ovis.core.functions import REGISTRY
from ovis.optimization.methods import gradient_descent
from ovis.optimization.plots import animate_descent, plot_descent


def _gradient_descent(args: argparse.Namespace) -> None:
    func = REGISTRY[args.function]()
    start = args.start if args.start is not None else func.start
    path = gradient_descent(func, start, lr=args.lr, steps=args.steps)

    if args.animate:
        anim = animate_descent(func, path, interval=args.interval)
        if args.out:
            anim.save(args.out, writer="pillow", fps=1000 // args.interval)
        else:
            plt.show()
    else:
        fig = plot_descent(func, path)
        if args.out:
            fig.savefig(args.out, dpi=150)
        else:
            plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ovis", description="Visualize optimization, analysis, and topology concepts."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gd = sub.add_parser("gradient-descent", help="visualize fixed-step gradient descent")
    gd.add_argument("--function", choices=sorted(REGISTRY), default="quadratic")
    gd.add_argument(
        "--start", type=float, nargs=2, metavar=("X", "Y"),
        help="starting point (default: the function's own start)",
    )
    gd.add_argument("--lr", type=float, default=0.1, help="learning rate")
    gd.add_argument("--steps", type=int, default=50, help="number of iterations")
    gd.add_argument("--animate", action="store_true", help="animate the path")
    gd.add_argument("--interval", type=int, default=100, help="ms between animation frames")
    gd.add_argument(
        "--out", type=Path, help="save to this file (.png for static, .gif for animated)"
    )
    gd.set_defaults(handler=_gradient_descent)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0
