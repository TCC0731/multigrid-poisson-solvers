from __future__ import annotations

import numpy as np
from numba import njit

from solvers.utils import _relative_residual_l2, _residual_l2


@njit(cache=True)
def _solve_rb(phi, rhs, h, tol, max_iter, omega):
    h2 = h * h
    n = phi.shape[0] - 2
    for iteration in range(1, max_iter + 1):
        for color in range(2):
            for i in range(1, n + 1):
                j0 = 1 + ((i + color) & 1)
                for j in range(j0, n + 1, 2):
                    phi[i, j] = (1.0 - omega) * phi[i, j] + omega * 0.25 * (
                        phi[i + 1, j]
                        + phi[i - 1, j]
                        + phi[i, j + 1]
                        + phi[i, j - 1]
                        + h2 * rhs[i, j]
                    )
        res = _residual_l2(phi, rhs, h)
        if res <= tol:
            return phi, iteration, res
    return phi, max_iter, _residual_l2(phi, rhs, h)


def solve(problem, tol=1e-10, max_iter=20000):
    phi, iterations, residual = _solve_rb(
        problem.phi0.copy(), problem.rhs, problem.h, tol, max_iter, 1.0
    )
    return phi, int(iterations), float(residual)
