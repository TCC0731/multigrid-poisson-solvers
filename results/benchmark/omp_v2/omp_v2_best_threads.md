# OMP v2 best thread table

Source: `results/benchmark/omp_v2/*_all.csv`

Cell format: `best thread / speedup vs 1 thread`

## solver_comparison

| grid | Jacobi | RB GS | RB SOR |
|---|---|---|---|
| 15 | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x |
| 31 | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x |
| 63 | 4t / 1.87x | 4t / 1.62x | 4t / 1.76x |
| 127 | 6t / 3.22x | 8t / 3.08x | 6t / 3.33x |
| 256 | - | - | 8t / 5.48x |
| 512 | - | - | 8t / 5.41x |
| all grids | 6t / 2.98x | 6t / 2.75x | 8t / 5.32x |

## mg_compare

| grid | MG(v,w=1.25,coarse=exact) | MG(v,w=1.25,coarse=sor) | MG(w,w=1.25,coarse=exact) | MG(w,w=1.25,coarse=sor) |
|---|---|---|---|---|
| 15 | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x |
| 31 | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x |
| 63 | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x |
| 127 | 4t / 1.10x | 4t / 1.06x | 1t / 1.00x | 1t / 1.00x |
| 255 | 4t / 2.08x | 4t / 1.95x | 2t / 1.02x | 1t / 1.00x |
| 511 | 8t / 3.47x | 8t / 3.06x | 4t / 1.25x | 1t / 1.00x |
| 1023 | 8t / 5.25x | 12t / 4.44x | 4t / 2.10x | 4t / 1.24x |
| 2047 | 6t / 2.31x | 12t / 2.24x | 6t / 1.89x | 4t / 1.64x |
| 4095 | 6t / 1.89x | 6t / 1.89x | 6t / 1.91x | 6t / 1.72x |
| all grids | 6t / 1.98x | 6t / 1.96x | 6t / 1.89x | 6t / 1.63x |

Notes:

- `-` means that solver was not benchmarked on that grid in `solver_comparison`.
- The best thread is chosen by the minimum `mean_time_ms` among the available thread counts.
- The `all grids` row sums `mean_time_ms` over every grid available for that solver, then compares against the 1-thread total.
