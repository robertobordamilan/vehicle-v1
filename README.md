# Vehicle v1.0 — Reproducible Experiments (Diffusion + Emergent Internal Time)

Author: Roberto Borda Milan (2026)

This package reproduces the computational experiments conducted on **Vehicle v1.0** as a discrete graph medium:

- **Experiment 1**: Diffusion propagation; center readout signatures per source.
- **Experiment 1B**: Extended readout (center + 1-hop neighbors) to break symmetries.
- **Experiment 2**: Emergent internal time from monotonic state functionals (entropy & Dirichlet energy).

## Quick start

### 1) Create environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2) Run experiments
```bash
python experiment1_diffusion.py
python experiment1B_center_plus_neighbors.py
python experiment2_emergent_time.py
```

Outputs are written to `./outputs/` as CSV and PNG.

## Vehicle definition used here (computational topology)

Vehicle v1.0 is represented as a graph with:
- Cube-center nodes: ("CUBE", cube_index)
- For each axis in {X,Y,Z}:
  - Face-center nodes: (axis, "CENTER", cube_index)
  - Face-corner nodes: (axis, "CORNER", cube_index, corner_id in {0..3})

Edges:
- Internal: each face-center and face-corner connects to its cube-center.
- Lattice: corresponding nodes connect between adjacent cubes in the 3D grid.

Note: This code focuses on **topology** for diffusion experiments (graph Laplacian). Spatial embedding is optional.

## Parameters (defaults)
- Grid: 2×2×4 cubes (16 cube-centers)
- Diffusion: ψ(k+1) = ψ(k) − α L ψ(k), with α = 0.95 / max_degree for stability
- Steps: Experiment 1/1B uses 60 steps; Experiment 2 uses 120 steps

## Outputs
See `./outputs/` after running.

## Reproducibility notes
- The experiments are deterministic (no randomness).
- Floating point rounding may vary slightly across platforms; group comparisons use rounding tolerance.
