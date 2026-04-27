from __future__ import annotations

import numpy as np
from numba import njit

from solvers.utils import _relative_residual_l2, _residual_l2, _residual

@njit(cache=True)
def _solve_jacobi(phi, rhs, h, tol, max_iter):
    h2 = h * h
    work = phi.copy()
    n = phi.shape[0] - 2
    for iteration in range(1, max_iter + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                work[i, j] = 0.25 * (
                    phi[i + 1, j]
                    + phi[i - 1, j]
                    + phi[i, j + 1]
                    + phi[i, j - 1]
                    + h2 * rhs[i, j]
                )
        res = _residual(work, rhs, h)
        if res <= tol:
            return work, iteration, res
        phi, work = work, phi
    return phi, max_iter, _residual(phi, rhs, h)


def solve(problem, tol=1e-10, max_iter=20000):
    phi, iterations, residual = _solve_jacobi(
        problem.phi0.copy(), problem.rhs, problem.h, tol, max_iter
    )
    return phi, int(iterations), float(residual)
