# OMP v3 best thread table

Source: `results/benchmark/omp_v3/*_all.csv`

Cell format: `best thread / speedup vs 1 thread`

## solver_comparison

| grid      | Jacobi     | RB GS      | RB SOR       |
| --------- | ---------- | ---------- | ------------ |
| 15        | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x   |
| 31        | 1t / 1.00x | 1t / 1.00x | 1t / 1.00x   |
| 63        | 7t / 1.44x | 2t / 1.39x | 2t / 1.40x   |
| 127       | 7t / 3.29x | 8t / 2.94x | 8t / 2.94x   |
| 256       | -          | -          | 14t / 5.63x  |
| 512       | -          | -          | 28t / 12.56x |
| all grids | 7t / 2.95x | 8t / 2.60x | 16t / 9.67x  |

## mg_compare

| grid      | MG(v,w=1.25,coarse=exact) | MG(v,w=1.25,coarse=sor) | MG(w,w=1.25,coarse=exact) | MG(w,w=1.25,coarse=sor) |
| --------- | ------------------------- | ----------------------- | ------------------------- | ----------------------- |
| 15        | 1t / 1.00x                | 1t / 1.00x              | 1t / 1.00x                | 1t / 1.00x              |
| 31        | 1t / 1.00x                | 1t / 1.00x              | 1t / 1.00x                | 1t / 1.00x              |
| 63        | 1t / 1.00x                | 1t / 1.00x              | 1t / 1.00x                | 1t / 1.00x              |
| 127       | 1t / 1.00x                | 1t / 1.00x              | 1t / 1.00x                | 1t / 1.00x              |
| 255       | 7t / 1.70x                | 7t / 1.52x              | 1t / 1.00x                | 1t / 1.00x              |
| 511       | 8t / 3.16x                | 8t / 3.27x              | 2t / 1.08x                | 1t / 1.00x              |
| 1023      | 16t / 5.70x               | 16t / 5.34x             | 7t / 1.59x                | 1t / 1.00x              |
| 2047      | 32t / 6.22x               | 32t / 6.14x             | 8t / 2.34x                | 7t / 1.48x              |
| 4095      | 64t / 7.32x               | 64t / 7.29x             | 16t / 3.46x               | 8t / 2.49x              |
| all grids | 56t / 6.38x               | 32t / 6.20x             | 8t / 2.94x                | 7t / 2.00x              |

Notes:

- `-` means that solver was not benchmarked on that grid in `solver_comparison`.
- The best thread is chosen by the minimum `mean_time_ms` among the available thread counts.
- The `all grids` row sums `mean_time_ms` over every grid available for that solver, then compares against the 1-thread total.
