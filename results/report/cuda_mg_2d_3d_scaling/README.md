# CUDA MG 2D vs 3D Scaling

This folder contains the report generator for the G experiment:

- compare the same CUDA MG solver in 2D and 3D
- track runtime growth against total cells
- inspect runtime per cell
- inspect throughput in cells / second

## Default setup

- backend: `cuda`
- solver: `mg`
- cycle: `v`
- coarse solve: `exact`
- `nu = 3`
- `omega = 1.25`
- cases: `sine`, `cosine`
- grid sizes:
  - 2D: `127, 255, 511, 1023, 2047, 4095`
  - 3D: `31, 63, 127, 255`

## Outputs

- `results_all.csv`
- `results_2d.csv`
- `results_3d.csv`
- `plots_{case}.png`

## Run

```bash
python results/report/result/cuda_mg_2d_3d_scaling/run_and_plot.py
```
