# OMP 3D v2 best thread table

Source: `results/benchmark/omp_3d_v2/*_all.csv`

Cell format: `best thread / speedup vs 1 thread`

## solver_comparison

| grid | Jacobi 3D | RB GS 3D | RB SOR 3D |
|---|---|---|---|
| 15 | 7t / 1.56x | 4t / 1.52x | - |
| 31 | 16t / 4.12x | 14t / 3.92x | 14t / 3.88x |
| 47 | 28t / 6.62x | 28t / 6.13x | - |
| 63 | 32t / 14.52x | 28t / 10.01x | 28t / 9.84x |
| all grids | 32t / 10.84x | 28t / 8.13x | 56t / 24.29x |

## mg_compare

| grid | MG 3D(v,w=1.25,coarse=exact) | MG 3D(v,w=1.25,coarse=sor) | MG 3D(w,w=1.25,coarse=exact) | MG 3D(w,w=1.25,coarse=sor) |
|---|---|---|---|---|
| 15 | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x |
| 31 | 7t / 1.84x | 7t / 1.57x | 4t / 1.40x | 1t / 1.00x |
| 63 | 16t / 3.92x | 16t / 3.58x | 8t / 2.49x | 7t / 1.72x |
| 127 | 28t / 8.11x | 28t / 7.79x | 16t / 5.23x | 14t / 3.64x |
| 255 | 112t / 10.42x | 112t / 9.88x | 56t / 7.78x | 32t / 6.62x |
| 383 | 28t / 8.00x | 28t / 7.98x | 28t / 7.35x | 28t / 6.73x |
| all grids | 64t / 8.19x | 32t / 8.13x | 28t / 7.24x | 28t / 6.35x |

Notes:

- This version exposes more thread-count options, and the larger grids now prefer much higher thread counts, especially for `MG` and `RB SOR 3D`.
- Compared with v1, the absolute 1-thread times are higher on the overlapping grids, but the best-thread speedups are often larger on the bigger problems.
- The smallest grids still favor low thread counts because OpenMP overhead dominates the useful work.
