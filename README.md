# Poisson Multigrid Solver

This repository is for developing and benchmarking Poisson equation solvers, with a focus on geometric multigrid methods and a Python reference implementation.

The project is organized as a research and coursework repository, not as a packaged library. The goal is to keep different solver implementations easy to compare, replace, and optimize.

## Equation Convention

All solvers in this repository use the same Poisson equation convention:

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

## Current Scope

The Python reference implementation supports both the 2D and 3D Poisson
equation. The pure C++ implementation supports both 2D and 3D. The OpenMP and
CUDA implementations currently focus on the existing accelerated 2D paths.

## Current Solver Support

| Backend        | Jacobi 2D | Gauss-Seidel 2D | SOR 2D | Multigrid 2D | Jacobi 3D | Gauss-Seidel 3D | SOR 3D | Multigrid 3D |
| -------------- | --------: | --------------: | -----: | -----------: | --------: | --------------: | -----: | -----------: |
| Python + Numba |       Yes |             Yes |    Yes |          Yes |       Yes |             Yes |    Yes |          Yes |
| C++ (pure)     |       Yes |             Yes |    Yes |          Yes |       Yes |             Yes |    Yes |          Yes |
| C++ + OpenMP   |        No |              No |     No |           No |        No |              No |     No |           No |
| CUDA           |        No |              No |     No |           No |        No |              No |     No |           No |

## Repository Structure

Current files are listed first; the remaining entries are still planned.

```text
multigrid-poisson-solvers/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── note/
│   ├── Architecture.md
│   ├── cpp_code_analysis.md
│   ├── install.md
│   └── python_code_analysis.md
│
├── python/
│   ├── run_poisson.py
│   ├── problems.py
│   ├── operators.py
│   ├── metrics.py
│   └── solvers/
│       ├── jacobi_2d.py
│       ├── jacobi_3d.py
│       ├── gs_2d.py
│       ├── gs_3d.py
│       ├── sor_2d.py
│       ├── sor_3d.py
│       ├── mg_2d.py
│       └── mg_3d.py
│
├── CMakeLists.txt
├── configs/
│   └── poisson2d_sin.json
│
├── cpp/
│   ├── include/
│   │   └── poisson/
│   │       ├── grid2d.hpp
│   │       ├── grid3d.hpp
│   │       ├── gs.hpp
│   │       ├── jacobi.hpp
│   │       ├── metrics.hpp
│   │       ├── mg.hpp
│   │       ├── operators.hpp
│   │       ├── problem.hpp
│   │       ├── red_black.hpp
│   │       ├── solver.hpp
│   │       ├── sor.hpp
│   │       └── validation.hpp
│   ├── src/
│   │   ├── gs.cpp
│   │   ├── jacobi.cpp
│   │   ├── main.cpp
│   │   ├── metrics.cpp
│   │   ├── mg.cpp
│   │   ├── operators.cpp
│   │   ├── problem.cpp
│   │   ├── red_black.cpp
│   │   ├── sor.cpp
│   │   └── validation.cpp
│   └── omp/
│       ├── main_jacobi_2d.cpp
│       ├── main_rbgs_2d.cpp
│       ├── main_sor_2d.cpp
│       └── main_mg_2d.cpp
│
├── cuda/
│   ├── include/
│   ├── kernels/
│   └── src/
│       ├── main_sor_2d.cu
│       └── main_mg_2d.cu
│
├── scripts/
│   ├── build_cpp.sh
│   ├── build_cuda.sh
│   ├── run_benchmark.py
│   └── plot_results.py
│
├── tests/
│   ├── conftest.py
│   ├── test_python.py
│   ├── test_python_3d.py
│   ├── cpp/
│   │   └── test_poisson.cpp
│   └── reference/
│
└── results/
    ├── cpp/
    ├── cpp_3d/
    ├── python/
    ├── python_3d/
    └── benchmark/
```

## Validation Problems

Manufactured solutions are used to validate correctness and convergence. For each case, the right-hand side is defined by

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

| Case               | Exact Solution $$\phi(x,y,z)$$                         | Right-Hand Side $$f(x,y,z)$$                              | Boundary          |
| ------------------ | ------------------------------------------------------ | --------------------------------------------------------- | ----------------- |
| Sine mode          | $$\sin(\pi x)\sin(\pi y)\sin(\pi z)$$                  | $$3\pi^2\sin(\pi x)\sin(\pi y)\sin(\pi z)$$               | Zero Dirichlet    |
| Mixed sine mode    | $$\sin(2\pi x)\sin(3\pi y)\sin(4\pi z)$$               | $$29\pi^2\sin(2\pi x)\sin(3\pi y)\sin(4\pi z)$$           | Zero Dirichlet    |
| Polynomial bubble  | $$x(1-x)y(1-y)z(1-z)$$                                 | $$2[y(1-y)z(1-z)+x(1-x)z(1-z)+x(1-x)y(1-y)]$$             | Zero Dirichlet    |
| Smooth exponential | $$e^{x+y+z}$$                                          | $$-3e^{x+y+z}$$                                           | Nonzero Dirichlet |
| Cosine mode        | $$\cos(\pi x)\cos(\pi y)\cos(\pi z)$$                  | $$3\pi^2\cos(\pi x)\cos(\pi y)\cos(\pi z)$$               | Nonzero Dirichlet |

## Development Plan

1. The Python + Numba 2D and 3D solvers are implemented.
2. The Python version serves as the reference implementation.
3. Implement the C++ + OpenMP 2D solvers.
4. Implement the CUDA 2D SOR and multigrid solvers.
5. Compare correctness, convergence, and performance across implementations.

## Output and Benchmarking

The current Python runner in `python/run_poisson.py` and the C++ baseline in `cpp/src/main.cpp` report the following CSV columns:

```text
solver,backend,dtype,grid_size,iterations,residual_l2,error_l2,error_linf,time_ms
```

The same output format should remain consistent for the future C++ OpenMP and CUDA implementations.

The C++ baseline is covered by a small GTest suite built as `poisson_tests` and executed through CTest.
The pure C++ 3D extension is covered by `poisson_tests_3d`, using the same GTest
and CTest flow.

### C++ CLI

The pure C++ runner defaults to the existing 2D behavior. Pass `--dim 3` to use
the 3D problem setup, stencil, metrics, and solvers:

```bash
./build/poisson_cpp --solver mg --case sine -n 31 --tol 1e-10 --max-iter 100
./build/poisson_cpp --dim 3 --solver mg --case sine -n 15 --tol 1e-8 --max-iter 100
./build/poisson_cpp --dim 3 --solver sor --case exp --dtype float -n 15 --tol 1e-3
```

The available 3D cases are the same as the Python 3D reference: `sine`,
`mixed_sine`, `bubble`, `exp`, and `cosine`. Dirichlet boundary values are copied
from the manufactured exact solution on all boundary faces, and the interior
initial guess remains zero, matching the C++ 2D convention.

### Benchmark Entry Points

Two standalone C++ benchmark executables are available after building:

- `poisson_benchmark_cpp`
- `poisson_benchmark_omp`

The Python reference implementation exposes a matching benchmark runner:

- `python/run_benchmark.py`
- `python/run_benchmark_3d.py`

Both benchmarks run the sine manufactured-solution case with one warmup run and five timed runs per measurement, then report the mean and standard deviation of the measured solver time.
The warmup pass uses the same solver setup but caps `max_iter` at `10` to keep the warmup cheap.
Running either executable with no `--suite` argument is equivalent to `--suite all`, so one invocation covers both benchmark groups.
When `--output BASE` is provided, the benchmark writes two files:

- `BASE.csv` for the simplified view
- `BASE_all.csv` for the full output

The simplified CSV keeps only `solver,grid_size,iterations,mean_time_ms,std_time_ms`.

The benchmark suites mirror the repository's existing comparison groups:

- `solver_comparison`: Jacobi, RB GS, and RB SOR with the requested `max_iter` and grid sizes.
- `mg_compare`: MG(V, `omega=1.25`) and MG(W, `omega=1.25`), each with both exact and SOR coarse-grid solves.

The multigrid benchmark keeps the current comparison defaults of `nu=3`, `omega=1.25`, `coarse_steps=16`, and reports both `coarse=exact` and `coarse=sor`.

The pure C++ benchmark also accepts `--dim 3`. The default remains `--dim 2`, so
existing benchmark commands keep their previous behavior. A small smoke run can
be bounded with `--max-grid-size`:

```bash
./build/poisson_benchmark_cpp --dim 3 --suite solver_comparison --max-grid-size 31
./build/poisson_benchmark_cpp --dim 3 --suite mg_compare --max-grid-size 31
```

The default C++ 3D benchmark sizes are:

- Jacobi 3D and RB GS 3D: `15, 31, 47, 63`
- RB SOR 3D: `31, 63, 95, 127, 159`
- MG 3D: `15, 31, 63, 127, 255, 383`

The 3D Python benchmark is calibrated for the `mg` conda environment so each
timed solve stays below 10 seconds on the development workstation. It records
`max_time_ms` in both CSV files to make that limit explicit. The default 3D
grid sizes are:

- Jacobi 3D and RB GS 3D: `15, 31, 47, 63`
- RB SOR 3D: `31, 63, 95, 127, 159`
- MG 3D: `15, 31, 63, 127, 255, 383`

Example usage:

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target poisson_benchmark_cpp poisson_benchmark_omp

./build/poisson_benchmark_cpp --output results/cpp/benchmark
./build/poisson_benchmark_cpp --dim 3 --output results/cpp_3d/benchmark
./build/poisson_benchmark_omp --output results/omp/benchmark
python python/run_benchmark.py --output results/benchmark/python/benchmark_python_v1
python python/run_benchmark_3d.py --output results/benchmark/python/benchmark_python_3d_v1
```

If you only want one group, you can still pass `--suite solver_comparison` or `--suite mg_compare`.

Run the Python CLI in 3D with `--dim 3`:

```bash
python python/run_poisson.py --dim 3 --solver mg --case sine -n 15 --tol 1e-8 --max-iter 100
```

## C++ Build and Test

Configure and build the C++ targets with CMake:

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Run the C++ test suite with CTest:

```bash
ctest --test-dir build --output-on-failure
```

The C++ analysis scripts follow the existing results layout. The 3D pure C++
analysis entry points are:

- `results/cpp_3d/solver_comparison/run_and_plot.py`
- `results/cpp_3d/solver_comparison_float32/run_and_plot_float32.py`
- `results/cpp_3d/mg_compare/run_and_plot_mg_compare.py`
- `results/cpp_3d/mg_compare_float32/run_and_plot_mg_compare_float32.py`

The float32 runs use looser tolerances than float64, matching the Python 3D
workflow and the existing C++ 2D float32 analysis scripts.
