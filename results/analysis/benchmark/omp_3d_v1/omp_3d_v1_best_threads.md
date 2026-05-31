# OMP 3D v1 best thread table

Source: `results/benchmark/omp_3d_v1/*_all.csv`

Cell format: `best thread / speedup vs 1 thread`

## solver_comparison

| grid | Jacobi 3D | RB GS 3D | RB SOR 3D |
|---|---|---|---|
| 15 | 4t / 2.15x | 6t / 1.95x | - |
| 31 | 6t / 3.92x | 8t / 3.99x | 8t / 4.25x |
| 47 | 6t / 4.49x | 6t / 4.73x | - |
| 63 | 8t / 5.22x | 6t / 4.63x | 8t / 5.84x |
| all grids | 6t / 5.01x | 6t / 4.61x | 12t / 4.42x |

## mg_compare

| grid | MG 3D(v,w=1.25,coarse=exact) | MG 3D(v,w=1.25,coarse=sor) | MG 3D(w,w=1.25,coarse=exact) | MG 3D(w,w=1.25,coarse=sor) |
|---|---|---|---|---|
| 15 | 2t / 1.23x | 1t / 1.00x | 2t / 1.05x | 1t / 1.00x |
| 31 | 6t / 2.48x | 4t / 2.00x | 6t / 1.76x | 4t / 1.15x |
| 63 | 6t / 3.87x | 12t / 3.74x | 6t / 2.80x | 4t / 2.17x |
| 127 | 16t / 4.41x | 12t / 4.29x | 12t / 4.22x | 12t / 3.59x |
| 255 | 12t / 2.22x | 12t / 2.11x | 12t / 2.13x | 12t / 2.21x |
| 383 | 4t / 2.02x | 4t / 2.01x | 4t / 2.09x | 6t / 1.99x |
| all grids | 8t / 2.06x | 4t / 2.02x | 4t / 2.10x | 6t / 2.01x |

Notes:

- Small grids are mostly latency-bound, so the best thread count stays low and `15` already saturates at `2t` for the MG variants.
- Among the 3D solver comparisons, `RB SOR 3D` scales the most strongly and reaches its best at `12t` on the larger grids, while `Jacobi 3D` and `RB GS 3D` peak earlier.
- For the multigrid cases, the `coarse=exact` variants generally outscale the `coarse=sor` variants on the same grid sizes.
