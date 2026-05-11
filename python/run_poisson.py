"""CLI runner for the Python Poisson examples.

This file stays thin on purpose. It only wires together the future modular
pieces that match the README structure:

* ``problems.make_problem(case, grid_size)`` for 2D
* ``problems.make_problem_3d(case, grid_size)`` for 3D
* ``solvers.<name>_<dim>d.solve(problem, *, tol, max_iter)``
* ``metrics.error_metrics(problem, phi)`` or ``metrics.error_metrics_3d``

The runner prints the CSV row described in ``README.md``.
"""

from __future__ import annotations

import argparse
from importlib import import_module
from time import perf_counter

import numpy as np


CASES = ("sine", "mixed_sine", "bubble", "exp", "cosine")
DTYPES = ("float32", "float64")
# Public solver names stay short; GS and SOR are expected to use red-black
# updates in their implementations.
SOLVER_MODULES = {
    "jacobi": "solvers.jacobi_2d",
    "gs": "solvers.gs_2d",
    "sor": "solvers.sor_2d",
    "mg": "solvers.mg_2d",
}
SOLVER_MODULES_3D = {
    "jacobi": "solvers.jacobi_3d",
    "gs": "solvers.gs_3d",
    "sor": "solvers.sor_3d",
    "mg": "solvers.mg_3d",
}


def _require_attr(module_name: str, attr_name: str):
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"Missing module {module_name!r}. "
            "This runner expects the modular Python files from README.md."
        ) from exc

    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise SystemExit(
            f"Module {module_name!r} must define {attr_name!r}."
        ) from exc


def load_problem(case: str, grid_size: int, dtype, dim: int = 2):
    attr_name = "make_problem_3d" if dim == 3 else "make_problem"
    make_problem = _require_attr("problems", attr_name)
    return make_problem(case=case, grid_size=grid_size, dtype=dtype)


def load_solver(name: str, dim: int = 2):
    modules = SOLVER_MODULES_3D if dim == 3 else SOLVER_MODULES
    solve = _require_attr(modules[name], "solve")
    return solve


def load_error_metrics(dim: int = 2):
    attr_name = "error_metrics_3d" if dim == 3 else "error_metrics"
    return _require_attr("metrics", attr_name)


def warmup_solver(solve, problem):
    # Numba compiles on first call, so do a tiny run before timing the real solve.
    solve(problem, tol=1e-2, max_iter=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a small Python Poisson benchmark from the command line."
    )
    parser.add_argument("--dim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--solver", choices=tuple(SOLVER_MODULES), default="jacobi")
    parser.add_argument("--case", choices=CASES, default="sine")
    parser.add_argument("--dtype", choices=DTYPES, default="float64")
    parser.add_argument("-n", "--grid-size", type=int, default=31)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--max-iter", type=int, default=20000)
    args = parser.parse_args()
    if args.grid_size < 1:
        parser.error("grid-size must be positive")
    return args


def main():
    args = parse_args()
    dim = getattr(args, "dim", 2)
    dtype = np.dtype(args.dtype).type
    if dim == 3:
        problem = load_problem(args.case, args.grid_size, dtype, dim=3)
        solve = load_solver(args.solver, dim=3)
        error_metrics = load_error_metrics(dim=3)
    else:
        problem = load_problem(args.case, args.grid_size, dtype)
        solve = load_solver(args.solver)
        error_metrics = load_error_metrics()

    warmup_solver(solve, problem)

    start = perf_counter()
    phi, iterations, residual_l2 = solve(
        problem, tol=args.tol, max_iter=args.max_iter
    )
    time_ms = (perf_counter() - start) * 1000.0

    error_l2, error_linf = error_metrics(problem, phi)
    print(
        "solver,backend,dtype,grid_size,iterations,"
        "residual_l2,error_l2,error_linf,time_ms"
    )
    print(
        f"{args.solver},python,{args.dtype},{args.grid_size},{iterations},"
        f"{residual_l2:.6e},{error_l2:.6e},{error_linf:.6e},{time_ms:.3f}"
    )


if __name__ == "__main__":
    main()
