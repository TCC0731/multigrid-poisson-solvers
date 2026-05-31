# Poisson Multigrid Solvers

This repository collects reference implementations and benchmarks for solving the Poisson equation on uniform grids. The focus is geometric multigrid, with matching solver sets in Python, C++, OpenMP, and CUDA so the same problem setup can be compared across backends.

The code is intended for experimentation and side-by-side comparison rather than as a packaged library.

## Overview

- Shared Poisson convention across every backend
- 2D and 3D support
- Jacobi, Gauss-Seidel, SOR, and multigrid solvers
- Manufactured-solution validation for accuracy checks
- CSV-based benchmarking for runtime and convergence comparisons

## Equation Convention

All solvers in this repository use the same sign convention:

$$
-\nabla^2 \phi = f
$$

For the 2D case on a uniform grid, the finite-difference discretization is

$$
\frac{4\phi_{i,j}-\phi_{i+1,j}-\phi_{i-1,j}-\phi_{i,j+1}-\phi_{i,j-1}}{h^2}=f_{i,j}
$$

For the 3D case on a uniform grid, the finite-difference discretization is

$$
\frac{6\phi_{i,j,k}
-\phi_{i+1,j,k}-\phi_{i-1,j,k}
-\phi_{i,j+1,k}-\phi_{i,j-1,k}
-\phi_{i,j,k+1}-\phi_{i,j,k-1}}{h^2}=f_{i,j,k}
$$

The residual is defined as

$$
r = f - A\phi
$$

where

$$
A\phi = -\nabla^2 \phi
$$

For multigrid methods, the error equation is

$$
Ae = r
$$

and the solution is corrected by

$$
\phi \leftarrow \phi + e
$$

## Solver Coverage

All backends expose the same solver family in both 2D and 3D: Jacobi, Gauss-Seidel, SOR, and multigrid.

| Backend | 2D | 3D | Notes |
| ------- | --: | --: | ----- |
| Python + Numba | Yes | Yes | Reference implementation |
| C++ | Yes | Yes | Native baseline |
| C++ + OpenMP | Yes | Yes | Parallelized loops with the same numerical logic |
| CUDA | Yes | Yes | GPU implementation with the same CLI style |

## Validation Problems

Manufactured solutions are used to validate correctness, boundary handling, and convergence. For the 2D cases, the right-hand side is defined by

$$
-\nabla^2 \phi = f
$$

| Case               | Exact Solution $$\phi(x,y)$$ | Right-Hand Side $$f(x,y)$$          | Boundary          |
| ------------------ | ---------------------------- | ----------------------------------- | ----------------- |
| Sine mode          | $$\sin(\pi x)\sin(\pi y)$$   | $$2\pi^2\sin(\pi x)\sin(\pi y)$$    | Zero Dirichlet    |
| Mixed sine mode    | $$\sin(2\pi x)\sin(3\pi y)$$ | $$13\pi^2\sin(2\pi x)\sin(3\pi y)$$ | Zero Dirichlet    |
| Polynomial bubble  | $$x(1-x)y(1-y)$$             | $$2x(1-x)+2y(1-y)$$                 | Zero Dirichlet    |
| Smooth exponential | $$e^{x+y}$$                  | $$-2e^{x+y}$$                       | Nonzero Dirichlet |
| Cosine mode        | $$\cos(\pi x)\cos(\pi y)$$   | $$2\pi^2\cos(\pi x)\cos(\pi y)$$    | Nonzero Dirichlet |

These cases provide simple tests for residual convergence, boundary-condition handling, and second-order accuracy.

The 3D implementations use matching manufactured solutions:

| Case               | Exact Solution $$\phi(x,y,z)$$           | Right-Hand Side $$f(x,y,z)$$                    | Boundary          |
| ------------------ | ---------------------------------------- | ----------------------------------------------- | ----------------- |
| Sine mode          | $$\sin(\pi x)\sin(\pi y)\sin(\pi z)$$    | $$3\pi^2\sin(\pi x)\sin(\pi y)\sin(\pi z)$$     | Zero Dirichlet    |
| Mixed sine mode    | $$\sin(2\pi x)\sin(3\pi y)\sin(4\pi z)$$ | $$29\pi^2\sin(2\pi x)\sin(3\pi y)\sin(4\pi z)$$ | Zero Dirichlet    |
| Polynomial bubble  | $$x(1-x)y(1-y)z(1-z)$$                   | $$2[y(1-y)z(1-z)+x(1-x)z(1-z)+x(1-x)y(1-y)]$$   | Zero Dirichlet    |
| Smooth exponential | $$e^{x+y+z}$$                            | $$-3e^{x+y+z}$$                                 | Nonzero Dirichlet |
| Cosine mode        | $$\cos(\pi x)\cos(\pi y)\cos(\pi z)$$    | $$3\pi^2\cos(\pi x)\cos(\pi y)\cos(\pi z)$$     | Nonzero Dirichlet |

## Build and Test

The project is typically built inside the `mg` conda environment described in [installation notes](note/install.md).

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

OpenMP and CUDA targets are enabled automatically when the corresponding compiler is available.

## Running the Solvers

The native runners share the same core CLI. Python uses `--dtype float32|float64`, while C++, OpenMP, and CUDA use `--dtype float|double`.

Supported options include `--dim 2|3`, `--solver jacobi|gs|sor|mg`, `--case sine|mixed_sine|bubble|exp|cosine`, `-n/--grid-size`, `--tol`, `--max-iter`, `--cycle v|w`, `--nu`, `--omega auto|VALUE`, and `--mg-coarse exact|sor`.

### Python

```bash
python python/run_poisson.py --solver mg --case sine -n 31 --tol 1e-10 --max-iter 100
python python/run_poisson.py --dim 3 --solver mg --case sine -n 15 --tol 1e-8 --max-iter 100
```

### C++

```bash
./build/poisson_cpp --solver mg --case sine -n 31 --tol 1e-10 --max-iter 100
./build/poisson_cpp --dim 3 --solver sor --case exp --dtype float -n 15 --tol 1e-3
```

### OpenMP

```bash
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores

./build/poisson_cpp_omp --dim 3 --solver mg --case sine -n 31 --cycle v --nu 2 --mg-coarse exact
```

### CUDA

```bash
./build/poisson_cuda --dim 3 --solver mg --case sine -n 31 --tol 1e-8 --max-iter 100
./build/poisson_cuda --dim 3 --solver mg --case sine -n 31 --tol 1e-8 --max-iter 100 --dump-phi
```

### Multigrid Parameters

The `--cycle`, `--nu`, `--omega`, and `--mg-coarse` flags only affect `--solver mg`.

| Flag | Meaning | Notes |
| ---- | ------- | ----- |
| `--cycle v|w` | Select a V-cycle or W-cycle. | Default `v`. `w` does extra coarse-grid work per cycle. |
| `--nu N` | Number of red-black smoothing steps used in each smoothing pass. | Default `2`. The value is applied before restriction and after prolongation. |
| `--omega auto|VALUE` | Relaxation factor for MG smoothing and MG coarse-grid SOR. | Default `1.0` unless `auto` is requested. `auto` uses the classical grid-dependent estimate `2 / (1 + sin(pi / (n + 1)))`. Manual values should stay in the open interval `(0, 2]`. |
| `--mg-coarse exact|sor` | Coarsest-level solve strategy. | Default `exact`. `exact` solves the coarsest grid directly; `sor` uses red-black SOR on the coarsest grid. |
| `--tol T` | Residual stopping threshold. | Default `1e-10` for double precision and `1e-6` for float. Smaller values increase work and usually improve accuracy. |
| `--max-iter N` | Maximum number of solver cycles. | Default `20000`. Prevents long runs when the requested tolerance is hard to reach. |

Note that standalone `--solver sor` does not read `--omega` from the CLI. The SOR solvers choose a grid-dependent relaxation factor internally, while `--omega` is used by the multigrid path.

### OpenMP Runtime Knobs

The OpenMP binary uses the same solver flags as the pure C++ binary, plus `--repeat-runs N` to average the timed solver section and `--residual-history [PATH]` to write the iteration-by-iteration relative residual to a CSV file for `--solver sor` and `--solver mg`. The CSV has two columns: `iteration` and `residual_l2`. If `PATH` is omitted, the binary writes `residual_history_<dim>d_<solver>_<case>_n<grid>_<dtype>.csv` in the current directory. The repeat count defaults to `1`, so a single run behaves exactly like the old binary.

The important OpenMP-specific knobs are environment variables:

- `OMP_NUM_THREADS` controls how many worker threads are launched.
- `OMP_PROC_BIND=close` keeps threads close together, which is usually a good default for repeatable timings.
- `OMP_PLACES=cores` binds threads to cores instead of letting them float freely.

If you are comparing runs, keep those values fixed and avoid changing the CPU affinity between measurements.

### CUDA Runtime Knobs

The CUDA binary also uses the same solver CLI as the C++ binary, plus `--repeat-runs N`, `--dump-phi [PATH]`, and `--coarse-steps N`, but the work happens on the GPU.

- `--dim 2|3` switches between the 2D and 3D CUDA code paths.
- `--dtype float|double` selects the precision. `float` is usually faster; `double` is the conservative choice when you want tighter numerical agreement.
- `--solver jacobi|gs|sor|mg` selects the kernel family.
- `--cycle v|w`, `--nu`, `--omega`, and `--mg-coarse` only matter for `--solver mg`.
- `--solver sor` still uses the internal grid-based relaxation factor and does not take a CLI override for omega.
- `--repeat-runs N` repeats only the timed solver section `N` times and reports the average `time_ms`.
- `--dump-phi [PATH]` writes the final solution grid to a binary file after the solve completes. If `PATH` is omitted, the binary is named `phi_cuda_<dim>d_<solver>_<case>_n<grid>_<dtype>.bin` in the current directory.
- `--coarse-steps N` sets the number of red-black SOR steps used on the coarsest grid when `--mg-coarse sor` is selected. The default is `16`.
- The executable checks that a CUDA device is available before solving, so a missing GPU or driver will fail early.

The runtime solvers print CSV rows in the format

```text
solver,backend,dtype,grid_size,iterations,residual_l2,error_l2,error_linf,time_ms
```

Here `time_ms` is the average time across the repeated timed solver runs. For CUDA multigrid,
the reported time uses the solver's compute-only timing so graph capture and instantiation
are not counted in `time_ms`.

When `--dump-phi` is enabled, the binary file starts with a small metadata header and then
stores the grid values in row-major order as `double`s.

## Benchmarking

CMake also builds a shared benchmark driver for C++, OpenMP, and CUDA when the relevant compiler is available:

- `poisson_benchmark_cpp`
- `poisson_benchmark_omp`
- `poisson_benchmark_cuda`

The benchmark suites are:

- `all`: runs both benchmark groups
- `solver_comparison`: Jacobi, RB GS, and RB SOR
- `mg_compare`: multigrid V/W cycles with exact or SOR coarse-grid solves

Example usage:

```bash
cmake --build build --target poisson_benchmark_cpp poisson_benchmark_omp

./build/poisson_benchmark_cpp --suite all --output results/cpp/benchmark
./build/poisson_benchmark_cpp --dim 3 --suite mg_compare --output results/cpp_3d/benchmark
OMP_NUM_THREADS=8 ./build/poisson_benchmark_omp --suite solver_comparison --output results/omp/benchmark
./build/poisson_benchmark_cuda --dim 3 --suite mg_compare --output results/cuda_3d/benchmark

python python/run_benchmark.py --output results/python/benchmark
python python/run_benchmark_3d.py --output results/python_3d/benchmark
```

When `--output BASE` is provided, the benchmark writers create `BASE.csv` and `BASE_all.csv`.

OpenMP timings are sensitive to `OMP_NUM_THREADS`, `OMP_PROC_BIND`, `OMP_PLACES`, and CPU topology, so set those values explicitly for reproducible comparisons.

The benchmark drivers accept the same `--dim` flag as the solver binaries, plus:

- `--suite all|solver_comparison|mg_compare`
- `--max-grid-size N` to cap the grid sizes included in a smoke run
- `--output BASE` to write `BASE.csv` and `BASE_all.csv`

The benchmark suites use the sine manufactured solution and fixed comparison settings so the same cases can be compared across backends.

## Notes

- The Python runners and benchmark scripts use the same manufactured problems as the C++ implementations.
- The CUDA runner uses the same solver names and manufactured cases as the native C++ path.
- If you only want one benchmark group, pass `--suite solver_comparison` or `--suite mg_compare`.
