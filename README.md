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

The initial development focuses only on the 2D Poisson equation.

## Current Solver Support

| Backend        | Jacobi 2D | Gauss-Seidel 2D | SOR 2D | Multigrid 2D |
| -------------- | --------: | --------------: | -----: | -----------: |
| Python + Numba |       Yes |             Yes |    Yes |          Yes |
| C++ (pure)     |       Yes |             Yes |    Yes |          Yes |
| C++ + OpenMP   |        No |              No |     No |           No |
| CUDA           |        No |              No |     No |           No |

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
│       ├── gs_2d.py
│       ├── sor_2d.py
│       └── mg_2d.py
│
├── CMakeLists.txt
├── configs/
│   └── poisson2d_sin.json
│
├── cpp/
│   ├── include/
│   │   └── poisson/
│   │       ├── grid2d.hpp
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
│   ├── cpp/
│   │   └── test_poisson.cpp
│   └── reference/
│
└── results/
    ├── raw/
    ├── plots/
    └── tables/
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

## Development Plan

1. The Python + Numba 2D solvers are implemented.
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

### C++ Benchmark Executables

Two standalone benchmark executables are available after building:

- `poisson_benchmark_cpp`
- `poisson_benchmark_omp`

Both benchmarks run the sine manufactured-solution case with one warmup run and five timed runs per measurement, then report the mean and standard deviation of the measured solver time.
The warmup pass uses the same solver setup but caps `max_iter` at `10` to keep the warmup cheap.
Running either executable with no `--suite` argument is equivalent to `--suite all`, so one invocation covers both benchmark groups.
When `--output BASE` is provided, the benchmark writes two files:

- `BASE.csv` for the simplified view
- `BASE_all.csv` for the full output

The simplified CSV keeps only `solver,grid_size,iterations,mean_time_ms,std_time_ms`.

The benchmark suites mirror the repository's existing comparison groups:

- `solver_comparison`: Jacobi, RB GS, and RB SOR with the requested `max_iter` and grid sizes.
- `mg_compare`: MG(V, `omega=1.25`) and MG(W, `omega=1.25`).

The multigrid benchmark keeps the current comparison defaults of `nu=3` and exact coarse-grid solve.

Example usage:

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target poisson_benchmark_cpp poisson_benchmark_omp

./build/poisson_benchmark_cpp --output results/cpp/benchmark
./build/poisson_benchmark_omp --output results/omp/benchmark
```

If you only want one group, you can still pass `--suite solver_comparison` or `--suite mg_compare`.

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
