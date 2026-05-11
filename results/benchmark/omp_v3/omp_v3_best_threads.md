# OMP v3 best thread table

Source: `results/benchmark/omp_v3/*_all.csv`

Cell format: `best thread / speedup vs 1 thread`

## solver_comparison

| grid | Jacobi | RB GS | RB SOR |
|---|---|---|---|
| 15 | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x |
| 31 | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x |
| 63 | 4t / 1.45x | 2t / 1.39x | 2t / 1.40x |
| 127 | 8t / 3.09x | 8t / 2.88x | 8t / 2.91x |
| 256 | - | - | 16t / 5.89x |
| 512 | - | - | 28t / 12.37x |
| all grids | 8t / 2.73x | 8t / 2.53x | 16t / 9.93x |

## mg_compare

| grid | MG(v,w=1.25,coarse=exact) | MG(v,w=1.25,coarse=sor) | MG(w,w=1.25,coarse=exact) | MG(w,w=1.25,coarse=sor) |
|---|---|---|---|---|
| 15 | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x |
| 31 | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x |
| 63 | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x |
| 127 | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x |
| 255 | 8t / 1.61x | 4t / 1.44x | 1t / 1.00x | 1t / 1.00x |
| 511 | 8t / 3.34x | 8t / 3.29x | 2t / 1.11x | 1t / 1.00x |
| 1023 | 16t / 5.81x | 16t / 5.50x | 4t / 1.51x | 2t / 1.00x |
| 2047 | 32t / 6.31x | 32t / 6.12x | 8t / 2.24x | 4t / 1.35x |
| 4095 | 64t / 7.04x | 64t / 6.99x | 16t / 3.24x | 8t / 2.25x |
| all grids | 56t / 6.31x | 32t / 6.11x | 8t / 2.75x | 7t / 1.77x |

Notes:

- `-` means that solver was not benchmarked on that grid in `solver_comparison`.
- The best thread is chosen by the minimum `mean_time_ms` among the available thread counts.
- The `all grids` row sums `mean_time_ms` over every grid available for that solver, then compares against the 1-thread total.
