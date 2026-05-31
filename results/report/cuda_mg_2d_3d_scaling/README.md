# CUDA MG 2D vs 3D Scaling

This folder contains the report generator for the G experiment:

- compare the same CUDA MG solver in 2D and 3D
- track runtime growth against total cells for MG V/W SOR
- inspect runtime per cell
- inspect throughput in cells / second

## Default setup

- backend: `cuda`
- solver: `mg`
- cycles: `v`, `w`
- coarse solve: `sor`
- `nu = 3`
- `omega = 1.25`
- cases: `sine`, `cosine`
- grid sizes: taken from `results/report/cuda_mg_convergence/results_all.csv` for the same backend / case / cycle / tolerance / iteration settings

## Outputs

- `results_all.csv`
- `results_2d.csv`
- `results_3d.csv`
- `plots_{case}.png`

## Run

```bash
python results/report/cuda_mg_2d_3d_scaling/run_and_plot.py
```
