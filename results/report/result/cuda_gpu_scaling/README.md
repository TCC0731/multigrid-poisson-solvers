# CUDA GPU Scaling

This folder contains the report generator for the I experiment:

- compare CPU baseline, OpenMP, and CUDA runtime
- compute CUDA speedup against CPU and OpenMP
- inspect throughput as the grid grows

## Default setup

- backend: `cuda` and `omp`
- solver: `mg`
- cycle: `v`
- coarse solve: `exact`
- `nu = 3`
- `omega = 1.25`
- OpenMP thread sweep: `1, 2, 4, 6, 8, 12, 16`
- cases: `sine`, `cosine`
- grid sizes:
  - 2D: `127, 255, 511, 1023, 2047, 4095`
  - 3D: `31, 63, 127, 255`

## Outputs

- `wrapper_cache.csv`
- `results_raw_all.csv`
- `results_all.csv`
- `results_raw_2d.csv`
- `results_raw_3d.csv`
- `results_2d.csv`
- `results_3d.csv`
- `plots_{dim}d_{case}.png`

## Run

```bash
python results/report/result/cuda_gpu_scaling/run_and_plot.py
```
