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
│   └── install.md
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

The current Python runner in `python/run_poisson.py` reports the following CSV columns:

```text
solver,backend,dtype,grid_size,iterations,residual_l2,error_l2,error_linf,time_ms
```

The output format should remain consistent across the Python, C++ OpenMP, and CUDA implementations as they are added.
