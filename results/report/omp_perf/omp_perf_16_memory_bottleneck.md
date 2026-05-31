> Quick note: this write-up uses the new 3-run `perf stat` results. The solver is backend/memory bound, and the 3D case is more severe than the 2D case.

# OMP Perf Memory Bottleneck Evidence

## Conclusion

The new 3-run `perf stat` data, together with `perf report` and `perf annotate`, supports that `poisson_cpp_omp` is primarily limited by backend/memory behavior rather than arithmetic throughput.

The strongest signals are:

- The hottest kernel is `smooth_red_black` / `smooth_red_black_3d`.
- The inner loop is dominated by load instructions (`movupd`, `movhpd`, `movsd`) that fetch neighboring grid values and `rhs`.
- `perf stat` now shows backend-bound behavior directly: `cpu_core/topdown-be-bound` is 57.7% for 2D and 58.2% for 3D.
- `cpu_core/topdown-mem-bound` rises from 43.7% in 2D to 49.3% in 3D, which is consistent with a memory bottleneck.
- The 3D case also has a much higher LLC load miss rate than the 2D case (89.365% vs 26.436%).

## 1. Summary Numbers

| Case | Grid | Runs | Avg time (ms) | Core cycles | Core instructions | IPC | L1D miss rate | LLC miss rate | Mem bound | Backend bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2D | 2047 | 3 | 536.518 | 5.74B | 9.29B | 1.619 | 1.513% | 26.436% | 43.7% | 57.7% |
| 3D | 383 | 3 | 2442.025 | 146.56B | 182.80B | 1.247 | 3.242% | 89.365% | 49.3% | 58.2% |

Notes:

- The average time is the mean of the three per-run timings in `stat_run.log`.
- The 3D problem is larger, so raw counts are not directly comparable.
- Even after normalization, the 3D run has a lower IPC and much higher L1D/LLC miss rates, which is consistent with stronger memory pressure.

## 2. `perf stat` Evidence

### 2D: `omp_perf_16_2D_2047_stat.txt`

- `cpu_core/cycles/u`: `5,740,499,804`
- `cpu_core/instructions/u`: `9,294,112,249`
- `cpu_core/topdown-be-bound/u`: `57.7%`
- `cpu_core/topdown-mem-bound/u`: `43.7%`
- `cpu_core/L1-dcache-loads:u/`: `3,132,958,740`
- `cpu_core/L1-dcache-load-misses:u/`: `47,408,495`
- `cpu_core/LLC-loads:u/`: `16,350,351`
- `cpu_core/LLC-load-misses:u/`: `4,322,429`

Source:

- `results/report/result/omp_perf/omp_perf_16_2D_2047_stat.txt` lines 4-52

Interpretation:

- IPC is about `1.619`, which is still modest for a stencil-like kernel.
- L1D miss rate is about `1.513%`, and LLC load miss rate is about `26.436%`.
- The topdown breakdown says the core spends a large fraction of time backend-bound and memory-bound, which fits a load-heavy stencil.

### 3D: `omp_perf_16_3D_383_stat.txt`

- `cpu_core/cycles/u`: `146,559,577,332`
- `cpu_core/instructions/u`: `182,795,792,927`
- `cpu_core/topdown-be-bound/u`: `58.2%`
- `cpu_core/topdown-mem-bound/u`: `49.3%`
- `cpu_core/L1-dcache-loads:u/`: `56,169,658,030`
- `cpu_core/L1-dcache-load-misses:u/`: `1,821,009,739`
- `cpu_core/LLC-loads:u/`: `478,054,428`
- `cpu_core/LLC-load-misses:u/`: `427,215,572`

Source:

- `results/report/result/omp_perf/omp_perf_16_3D_383_stat.txt` lines 4-52

Interpretation:

- IPC drops to about `1.247`, lower than the 2D case.
- L1D miss rate is about `3.242%`, and LLC load miss rate jumps to about `89.365%`.
- The topdown breakdown shows both backend-bound and memory-bound behavior, which is strong evidence of a memory bottleneck.

## 3. `perf report` Evidence

### 2D: hot spot concentration

`smooth_red_black<double>` is the dominant kernel in the cycle profile:

- `gomp_thread_start` accounts for `72.01%` self/children in the top entry.
- Under that, `smooth_red_black<double>` contributes `29.37%`.
- Other phases such as `make_problem_impl`, `residual_full`, `restrict_full_weighting`, `prolong_add`, and `finite_grid` are each much smaller.

Source:

- `results/report/result/omp_perf/omp_perf_16_2D_2047_report.txt` lines 12-84

Interpretation:

- The solver spends most of its time inside the smoothing kernel, not in complex arithmetic or large control-flow overhead.
- There is some synchronization cost (`gomp_simple_barrier_wait`, `gomp_team_barrier_wait_end`, `gomp_team_barrier_wait_final`), but it is secondary compared with the kernel itself.

### 3D: hot spot is even more concentrated

`smooth_red_black_3d<double>` dominates the profile:

- `smooth_red_black_3d<double>` contributes `55.30%` of the thread-start profile.
- The same symbol also appears with `62.74%` self time in the main object.
- The remaining work is split across `make_problem_3d_impl` (`9.07%`), `residual_full_3d` (`5.70%`), `relative_physical_residual_l2` (`3.18%`), `prolong_add_3d` (`2.85%`), `restrict_full_weighting_3d` (`2.48%`), and `finite_grid` (`2.33%`).

Source:

- `results/report/result/omp_perf/omp_perf_16_3D_383_report.txt` lines 12-75
- `results/report/result/omp_perf/omp_perf_16_3D_383_report.txt` lines 107-120

Interpretation:

- The kernel itself dominates the profile, which is exactly where a memory-bound stencil would concentrate time.
- The non-kernel phases are comparatively small, so the bottleneck is not mainly in setup or validation.

## 4. `perf annotate` Evidence

### 2D inner loop

In `smooth_red_black<double>`, the inner loop is dominated by memory loads:

- `movupd (%rsi,%rax,1),%xmm10` at `30.39%`
- `movupd (%r10,%rax,1),%xmm10` at `19.08%`
- `movhpd 0x10(%r10,%rax,1),%xmm10` at `10.61%`
- `movupd (%r9,%rax,1),%xmm0` / `movhpd 0x10(%r9,%rax,1),%xmm0` also consume visible samples
- arithmetic instructions such as `addpd` and `mulpd` are present, but each is much smaller than the load instructions
- the store path is tiny compared with the loads

Source:

- `results/report/result/omp_perf/omp_perf_16_2D_2047_annotate.txt` lines 149-184

Interpretation:

- Each grid point update requires multiple grid reads plus one write.
- The load-heavy instruction mix is a classic sign of low arithmetic intensity.
- The CPU spends far more sampled cycles waiting on data movement than on floating-point math.

### 3D inner loop

In `smooth_red_black_3d<double>`, the same pattern is even clearer:

- `movupd 0x0(%rbp,%rax,1),%xmm10` at `25.79%`
- `movupd (%rdi,%rax,1),%xmm10` at `21.15%`
- `movupd (%r11,%rax,1),%xmm10` at `10.53%`
- `movhpd 0x10(%rbp,%rax,1),%xmm10` at `9.83%`
- `movupd (%rbx,%rax,1),%xmm0` at `7.77%`
- `movhpd 0x10(%rbx,%rax,1),%xmm0` at `2.44%`
- `addpd` / `mulpd` are much smaller than the load instructions
- the write-back instruction is again tiny

Source:

- `results/report/result/omp_perf/omp_perf_16_3D_383_annotate.txt` lines 187-233

Interpretation:

- A 3D stencil has more neighbors per point, so it naturally increases memory traffic.
- The annotate output shows that the extra cost is paid mostly in loads, not in additional floating-point operations.
- This matches the higher cache miss rates seen in `perf stat`.

## 5. Bottom Line

The data point to a memory/backend bottleneck, specifically a stencil-style kernel that is limited by repeated grid fetches and cache behavior:

- the hottest code path is the smoothing kernel,
- that kernel is load-dominated in `perf annotate`,
- and the new topdown metrics show the core spending most of its time backend-bound and memory-bound.
- the 3D case has a dramatically higher LLC load miss rate than the 2D case.

If we want to go one step further, the next useful checks would be:

1. `perf stat -d` or `perf stat -d -d` for richer cache and branch metrics.
2. `perf c2c` to see whether the threads are fighting over shared cache lines.
3. A blocking / tiling experiment to reduce reuse distance in the smoothing kernel.
