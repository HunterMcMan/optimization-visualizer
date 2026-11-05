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
