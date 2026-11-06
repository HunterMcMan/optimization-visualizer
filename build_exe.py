"""Build dist/OptimizationVisualizer.exe, a single-file app with every visualizer.

    pip install -r requirements.txt -r requirements-dev.txt
    python build_exe.py

The logo from app.py becomes the exe icon, and the finished exe is checked by
opening every tool through its landing page (--self-test). Swap --onefile for
--onedir below to get a folder that starts faster (a one-file exe unpacks itself
on every launch).
"""

import io
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
from PIL import Image
import PyInstaller.__main__

from app import logo_png

HERE = Path(__file__).resolve().parent
ICON = HERE / "build" / "icon.ico"


def make_icon():
    ICON.parent.mkdir(exist_ok=True)
    image = Image.open(io.BytesIO(logo_png(256)))
    image.save(ICON, sizes=[(s, s) for s in (16, 32, 48, 64, 128, 256)])


def main():
    make_icon()
    PyInstaller.__main__.run([
        str(HERE / "app.py"),
        "--name", "OptimizationVisualizer",
        "--onefile",
        "--windowed",
        "--noconfirm",
        "--icon", str(ICON),
        "--exclude-module", "PyQt5",
        "--exclude-module", "PyQt6",
        "--exclude-module", "PySide2",
        "--exclude-module", "PySide6",
        "--exclude-module", "IPython",
        "--distpath", str(HERE / "dist"),
        "--workpath", str(HERE / "build"),
        "--specpath", str(HERE / "build"),
    ])
    check()


def check():
    """Open every tool in the new exe through the landing page's buttons."""
    exe = HERE / "dist" / "OptimizationVisualizer.exe"
    report = HERE / "build" / "self_test.txt"
    code = subprocess.run([str(exe), "--self-test", str(report)], timeout=300).returncode
    sys.stdout.reconfigure(errors="replace")  # e.g. "ε" on a cp1252 Windows console
    print(report.read_text(encoding="utf-8") if report.exists() else "(no self-test report)")
    if code:
        sys.exit(f"Self-test failed: {exe} could not open every tool")
    print(f"Self-test passed: {exe}")


if __name__ == "__main__":
    main()
