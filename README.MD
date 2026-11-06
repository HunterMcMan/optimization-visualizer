# Optimization Visualizer
This tool generates visuals pertaining to the topics of mathematical optimization, analysis, and topology. I created this to better help students like myself grasp difficult concepts where visualization can mean the difference between understanding and confusion. More coming soon...

## Setup
```
pip install -r requirements.txt
```

## Run the app
```
python app.py
```
This opens a landing page with a card for each visualizer. Press **Open** on a card to launch that tool in its own window; you can have several open at once. Each visualizer can also still be run on its own, as shown below.

## Build a Windows executable
```
pip install -r requirements-dev.txt
python build_exe.py
```
This creates `dist/OptimizationVisualizer.exe` (about 80 MB), a single file that runs without Python installed. Because the exe unpacks itself on every launch, it takes a few seconds to start. To check a build without opening any windows, run `OptimizationVisualizer.exe --self-test`; it opens every tool in the background and exits with code 0 when all of them work.

## Showcase video
```
python scripts/make_video.py
```
This renders `media/showcase.mp4`, a narrated vertical (9:16) tour of the app. It drives the real tools headlessly, so after any UI change you can run it again to get an up-to-date video. The voice is synthesized with `edge-tts`, which needs an internet connection; without one, the script falls back to a built-in Windows voice. Add `--preview` to get quick contact sheets of sampled frames in `build/video_preview/` instead of a video.

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
- **KKT conditions** (`python optimization/kkt.py`): minimize or maximize f(x, y) subject to inequality and equality constraints, written with `<=`, `>=` or `=` and separated by `;` (e.g. `x^2 + y^2 <= 4; x + y >= 1; y = x^2`). **mode** switches between minimizing and maximizing, and **examples ▸** cycles through ten preset problems.
  - The left panel shows the contours of f, the feasible region in blue, and every candidate point: ▼ min (▲ max) is filled if global and hollow if local, ✕ marks a point where a multiplier has the wrong sign, and ◇ marks a point where LICQ fails. Click a point or use ◀ ▶ to select it.
  - The right panel is a close-up of the selected point. It shows the active constraint gradients and the cone they generate. The KKT conditions say −∇f lies in that cone; the arrow turns red when it doesn't.
  - **tests** opens a window with the four KKT conditions and the multipliers. It also shows the second-order test on the critical cone (with the bordered Hessian when it applies) and whether each point is a global or local extremum. Convex problems get the shortcut "every KKT point is a global min".
  - **CQs** opens a window that checks LICQ, MFCQ, Abadie, linear CQ and Slater at the selected point, shows how they imply one another, and explains what the result means there. While it is open, the close-up also draws the tangent cone (green) and the linearized cone (dashed); Abadie's CQ holds exactly when they match.
  - Good examples to try: the cusp `x` with `y^2 <= x^3`, where no CQ holds and the minimizer is not a KKT point; the Kuhn–Tucker example `-x` with `y <= (1 - x)^3; y >= 0`, where Abadie's CQ fails; and `x + 2*y` with `x >= 0; y >= 0; x + y >= 0`, where LICQ fails but MFCQ holds.
