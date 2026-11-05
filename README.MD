# Optimization Visualizer
This tool generates visuals pertaining to the topics of mathematical optimization, analysis, and topology. I created this to better help students like myself grasp difficult concepts where visualization can mean the difference between understanding and confusion. More coming soon...

## Setup
```
pip install -r requirements.txt
```

## Visualizers
### Analysis
- **Epsilon-delta limit** (`python analysis/epsilon_delta.py`): adjust ε and δ with sliders to see whether every point within δ of a maps within ε of the limit L. Type a function and a point a into the boxes and press Enter; the plot switches automatically based on the inputs f uses.
  - **f(x)** gives a 2D plot: δ is an interval around a, ε is a band around L. Try `x^2` at `1`, `sin(x)/x` at `0` (the limit is computed even where f is undefined), or `sign(x)` at `0` (no limit).
  - **f(x, y)** gives a 3D plot plus a top view: δ is a disk around a = (a₁, a₂), ε is a slab L − ε < z < L + ε. Enter a as `0, 0`. Try `x^2+y^2`, `sin(x^2+y^2)/(x^2+y^2)`, `x*y/(x^2+y^2)` (limit depends on direction), or `x^2*y/(x^4+y^2)` with ε = 0.25 (approaches 0 along every line, but not along y = x², so there is no limit).
  - The opacity sliders along the bottom set how solid the graph of f, the ε-band and the δ-window are, from 0 (hidden) to 1 (solid).

### Optimization
- **Lagrange multipliers** (`python optimization/lagrange.py`): find the extrema of f(x, y) subject to equality constraints g(x, y) = c. Type f, the constraints (separated by `;`, e.g. `x^2 + y^2 = 1; y = x`) and the plot window (a half-width like `3`, or `cx, cy, half-width`), then press Enter.
  - The left panel shows the contours of f, each constraint curve, and every critical point: ▲ max, ▼ min, ✕ neither. Filled markers are global and hollow markers are local; ◇ marks a point where the constraint gradients are dependent (the constraint qualification fails), where Lagrange's condition can miss an extremum. At the selected point (click it, or use ◀ ▶) the arrows ∇f and λᵢ∇gᵢ are drawn tip to tail, so you can see that ∇f = Σ λᵢ∇gᵢ. Drag **k** to sweep the level curve f = k until it just touches the constraint.
  - The right panel shows the graph of f, with the constraints lifted onto it and the plane z = k.
  - Press **show tests** to open a second window with the results for the selected point. It updates as you select points or change the inputs. Each point is checked with the second-order sufficient condition in two equivalent forms: the Hessian of the Lagrangian on the tangent line, q = dᵀ∇²L d, and the bordered Hessian, det H̄ = −|∇g|² q. When the test is inconclusive, f is compared with nearby points of the constraint set.
  - The feasible set S is checked for being closed and bounded. If S is compact, the extreme value theorem guarantees a global max and min. Otherwise S is sampled far out to decide whether a global extremum exists.
  - Try `x + y` with `x^2 + y^2 = 1` (compact, so there is a global max and min); `x^2 + y^2` with `x*y = 1` (two global minima, no max); `x*y` with `x + y = 2` (global max only); `y` with `y = x^3` (the SOSC is inconclusive and the origin is neither); `x` with `y^2 = x^3` (a cusp, where the constraint qualification fails at the minimum); or `x + y^2` with `x^2 + y^2/4 = 1; y = x` (two constraints, so the feasible points are isolated).
