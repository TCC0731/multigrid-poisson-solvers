# OMP Perf Memory Bottleneck Evidence

## Conclusion

The evidence from `perf stat`, `perf report`, and `perf annotate` supports that `poisson_cpp_omp` is primarily limited by memory access behavior rather than arithmetic throughput.

The strongest signals are:

- The hottest kernel is `smooth_red_black` / `smooth_red_black_3d`.
- The inner loop is dominated by load instructions (`movupd`, `movhpd`, `movsd`) that fetch neighboring grid values and `rhs`.
- `perf stat` shows non-trivial cache miss and LLC miss rates, with the 3D case worse than the 2D case even after normalizing by instruction count.
- `stalled-cycles-backend` is not supported in these runs, so the conclusion is inferred from the cache/memory profile and the instruction mix.

## 1. Summary Numbers

| Case | Grid | Iterations | Time (ms) | Cycles | Instructions | IPC | Cache misses | LLC load misses | Cache miss rate | LLC miss rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2D | 2047 | 2 | 436.236 | 2.00B | 2.95B | 1.473 | 11.67M | 2.06M | 0.396% | 0.070% |
| 3D | 383 | 3 | 2492.221 | 50.52B | 61.52B | 1.218 | 567.49M | 130.26M | 0.922% | 0.212% |

Notes:

- The 3D problem is larger, so raw counts are not directly comparable.
- Even after normalization, the 3D run has a lower IPC and higher cache miss rates, which is consistent with stronger memory pressure.
- The raw run logs and the `perf stat` runs report the same benchmark configuration; the 3D runtime is essentially identical across `run.log` (`2492.221 ms`) and `stat_run.log` (`2491.847 ms`), while the 2D runtime is close enough (`436.236 ms` vs `455.055 ms`) that the gap is plausibly profiling overhead.

## 2. `perf stat` Evidence

### 2D: `omp_perf_16_2D_2047_stat.txt`

- `cpu_core/cycles:u/`: `2,000,314,990`
- `cpu_core/instructions:u/`: `2,946,478,493`
- `cpu_core/cache-misses:u/`: `11,671,259`
- `cpu_core/LLC-load-misses:u/`: `2,056,402`
- `stalled-cycles-backend:u/`: not supported

Source:

- `results/report/result/omp_perf/omp_perf_16_2D_2047_stat.txt` lines 6-17

Interpretation:

- IPC is about `1.473`, which is not high for a stencil-like kernel.
- Cache miss rate is about `0.396%`, and LLC load miss rate is about `0.070%`.
- These misses are consistent with repeated streaming access to grid data rather than compute-heavy work.

### 3D: `omp_perf_16_3D_383_stat.txt`

- `cpu_core/cycles:u/`: `50,524,224,620`
- `cpu_core/instructions:u/`: `61,523,572,934`
- `cpu_core/cache-misses:u/`: `567,492,016`
- `cpu_core/LLC-load-misses:u/`: `130,257,288`
- `stalled-cycles-backend:u/`: not supported

Source:

- `results/report/result/omp_perf/omp_perf_16_3D_383_stat.txt` lines 6-17

Interpretation:

- IPC drops to about `1.218`, lower than the 2D case.
- Cache miss rate rises to about `0.922%`, and LLC load miss rate to about `0.212%`.
- The 3D case shows substantially more memory pressure, which matches the larger stencil footprint.

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

The data point to a memory bottleneck, specifically a stencil-style kernel that is limited by repeated grid fetches and cache behavior:

- the hottest code path is the smoothing kernel,
- that kernel is load-dominated in `perf annotate`,
- and the normalized cache/LLC miss rates are high enough to explain the relatively low IPC.

If we want to go one step further, the next useful checks would be:

1. `perf stat -d` or `perf stat -d -d` for richer cache and branch metrics.
2. `perf c2c` to see whether the threads are fighting over shared cache lines.
3. A blocking / tiling experiment to reduce reuse distance in the smoothing kernel.
