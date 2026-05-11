from __future__ import annotations

from numba import njit

from solvers.utils import _relative_physical_residual_l2_3d


@njit(cache=True)
def _solve_rb_3d(phi, rhs, h, tol, max_iter, omega):
    h2 = h * h
    n = phi.shape[0] - 2
    for iteration in range(1, max_iter + 1):
        for color in range(2):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    k0 = 1 + ((i + j + color) & 1)
                    for k in range(k0, n + 1, 2):
                        phi[i, j, k] = (1.0 - omega) * phi[i, j, k] + omega * (
                            phi[i + 1, j, k]
                            + phi[i - 1, j, k]
                            + phi[i, j + 1, k]
                            + phi[i, j - 1, k]
                            + phi[i, j, k + 1]
                            + phi[i, j, k - 1]
                            + h2 * rhs[i, j, k]
                        ) / 6.0
        res = _relative_physical_residual_l2_3d(phi, rhs, h)
        if res <= tol:
            return phi, iteration, res
    return phi, max_iter, _relative_physical_residual_l2_3d(phi, rhs, h)


def solve(problem, tol=1e-10, max_iter=20000):
    phi, iterations, residual = _solve_rb_3d(
        problem.phi0.copy(), problem.rhs, problem.h, tol, max_iter, 1.0
    )
    return phi, int(iterations), float(residual)
