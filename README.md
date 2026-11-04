# optimization-visualizer
Visualize advanced concepts in mathematical optimization, analysis, and topology.

Built on NumPy, SciPy, and Matplotlib.

## Setup

```sh
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install -e ".[dev]"         # add ",topology" for persistent homology support
```

## Usage

```sh
ovis gradient-descent --function rosenbrock --start -1.5 2.5 --lr 0.001 --steps 300
ovis gradient-descent --animate --out descent.gif
ovis gradient-descent --out descent.png
```

Run the tests with `pytest`.

## Layout

- `ovis.core` — test functions with gradients, grids
- `ovis.optimization` — optimization methods and their plots
- `ovis.analysis`, `ovis.topology` — planned
