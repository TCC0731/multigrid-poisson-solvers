from __future__ import annotations

import math

from solvers.gs_2d import _solve_rb


def _omega(n: int) -> float:
    return 2.0 / (1.0 + math.sin(math.pi / (n + 1)))


def solve(problem, tol=1e-10, max_iter=20000, omega=None):
    if omega is None:
        omega = _omega(problem.grid_size)
    phi, iterations, residual = _solve_rb(
        problem.phi0.copy(), problem.rhs, problem.h, tol, max_iter, omega
    )
    return phi, int(iterations), float(residual)
