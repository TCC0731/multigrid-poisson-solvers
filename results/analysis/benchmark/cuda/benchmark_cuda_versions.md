# CUDA Benchmark Change Log (v1 ~ v6)

This document is based on the git history under the `cuda/` path. Its goal is to connect the changes made to the CUDA Poisson solver across the different benchmark versions.

Version mapping:

- `v1` is the original CUDA solver baseline.
- `v2` through `v6` correspond to later commits on the CUDA branch.
- The `v1` benchmark CSV files were added later, but their contents match the original implementation.

## Version Overview

| Version | Commit | Main Changes |
| --- | --- | --- |
| v1 | `f1148e9` | Initial CUDA Poisson solver implementation, including Jacobi, Gauss-Seidel, SOR, Multigrid, problem setup, validation, metrics, and basic residual computation. |
| v2 | `ceb0207` | Switched `check_kernel` to `cudaPeekAtLastError`, added `cudaDeviceSynchronize` only in debug builds, documented the CUDA execution path, and added benchmark CSV files. |
| v3 | `feef0cc` | Reworked relative residual computation into a reusable workspace model, introduced `ResidualPair` and reduction kernels, and reduced memory allocation overhead during residual checks. |
| v4 | `a167248` | Fixed RB SOR kernel interior and checkerboard indexing so the updated grid range matches the actual active cells more accurately. |
| v5 | `f9dff81` | Updated the multigrid coarsest exact solve to use homogeneous Dirichlet boundary conditions for the error equation. |
| v6 | `091ad7c` | Added cached rhs norm handling and new residual kernels; changed non-MG solvers to check residuals only at intervals to reduce per-iteration overhead. |

## Details by Version

### v1

The first CUDA mainline version established the full Poisson solver skeleton:

- `cuda/src/main.cu` serves as the unified entry point and handles argument parsing and solver dispatch.
- All four methods are already available: `jacobi`, `gs`, `sor`, and `mg`.
- The `problem`, `validation`, and `metrics` pipeline is also in place.
- Residual computation uses the original reduction approach, which is functional but still a baseline implementation.

The main goal of this version was simply to get the CUDA solver running.

### v2

This version focused on stability and readability improvements:

- `check_kernel` was changed from `cudaGetLastError()` to `cudaPeekAtLastError()` so the error check matches the usual post-launch pattern.
- Debug builds now call `cudaDeviceSynchronize()` as well, which makes it easier to catch issues earlier during debugging.
- `note/cuda_execution_path.md` was added to document the CUDA execution flow, solver branches, and benchmark path.
- Benchmark CSV files were also filled in for `jacobi`, `RB GS`, `RB SOR`, and `MG`.

### v3

This version introduced a major change to residual computation:

- Added `RelativeResidualWorkspace` to manage the device buffers used by residual checks.
- `ResidualPair` was introduced into the reduction pipeline so residual and rhs norm can be reduced together.
- `compute_relative_residual` now performs a partial reduction on the GPU first and then finishes the reduction through the workspace.
- The result is fewer temporary allocations and less host/device back-and-forth during residual computation.

The core goal of this version was to make residual checks cheaper.

### v4

This version fixed the RB SOR grid indexing and interior handling:

- `cuda/kernels/sor_kernels.cuh` rewrote the checkerboard color coordinate mapping.
- `cuda/include/poisson/cuda_utils.hpp` also adjusted the grid setup and used `active_columns` to represent the actual number of x-direction work items.

This change is mainly about correctness, making the SOR update range match the checkerboard coloring logic more closely.

### v5

This version changed how the multigrid coarsest level is solved:

- `cuda/src/mg.cu` updated the coarse solve to use homogeneous Dirichlet boundary conditions for the error equation.
- In other words, coarse correction no longer injects the original boundary values and instead treats the boundary as fixed at 0.
- `note/cuda_execution_path.md` was updated to reflect this interpretation as well.

This is an important correctness fix for MG, because the coarse-grid solve is solving for a correction, not the original solution itself.

### v6

This version further reduced the cost of residual checks:

- `cuda/include/poisson/cuda_utils.hpp` added caching for the rhs norm.
- New kernels were introduced: `rhs_norm_partial_kernel`, `residual_partial_kernel`, and updated finalize/store kernels.
- `Gauss-Seidel`, `Jacobi`, and `SOR` now use `should_check_non_mg_residual()` to control how often residuals are checked.
- Non-MG solvers no longer compute residuals every iteration. Instead, they check:
  - iteration 1
  - every 50 iterations
  - the final iteration
- `MG` still uses the uncached residual path because its rhs and coarse-correction flow is not the same as the standard iterative solvers.

The main point of this version was to reduce residual-check overhead and make the benchmark results closer to solver runtime itself.

## Related Benchmark Files

The following files correspond to the benchmark results for each version:

- `benchmark_cuda_v1.csv`
- `benchmark_cuda_v1_all.csv`
- `benchmark_cuda_v2.csv`
- `benchmark_cuda_v2_all.csv`
- `benchmark_cuda_v3.csv`
- `benchmark_cuda_v3_all.csv`
- `benchmark_cuda_v4.csv`
- `benchmark_cuda_v4_all.csv`
- `benchmark_cuda_v5.csv`
- `benchmark_cuda_v5_all.csv`
- `benchmark_cuda_v6.csv`
- `benchmark_cuda_v6_all.csv`

If you want to compare versions, start with this document and then inspect the corresponding CSV data.
