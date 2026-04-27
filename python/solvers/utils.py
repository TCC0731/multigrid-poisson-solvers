from __future__ import annotations

import numpy as np
from numba import njit

@njit(cache=True)
def _residual_l2(phi, rhs, h):
    h2 = h * h
    total = 0.0
    n = phi.shape[0] - 2
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            r = h2 * rhs[i, j] - (
                4.0 * phi[i, j]
                - phi[i + 1, j]
                - phi[i - 1, j]
                - phi[i, j + 1]
                - phi[i, j - 1]
            )
            total += r * r
    return np.sqrt(total) / h

@njit(cache=True)
def _relative_physical_residual_l2(phi, rhs, h):
    h2 = h * h

    res_total = 0.0
    rhs_total = 0.0

    n = phi.shape[0] - 2

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            Lphi = (
                4.0 * phi[i, j]
                - phi[i + 1, j]
                - phi[i - 1, j]
                - phi[i, j + 1]
                - phi[i, j - 1]
            )

            b = h2 * rhs[i, j]
            d = b - Lphi

            res_total += d * d
            rhs_total += b * b

    if rhs_total > 0.0:
        return np.sqrt(res_total / rhs_total)
    else:
        return np.sqrt(res_total)

@njit(cache=True)
def _relative_backward_error_l2(phi: np.ndarray, rhs: np.ndarray, h: float) -> float:
    """
    Calculate the relative backward-error L2 norm for the 2D Poisson equation
        (4*phi[i,j] - phi[i+1,j] - phi[i-1,j] - phi[i,j+1] - phi[i,j-1]) / h^2 = rhs[i,j]
    The scaled residual is
        d_ij = 4*phi[i,j] - phi[i+1,j] - phi[i-1,j] - phi[i,j+1] - phi[i,j-1] - h^2 * rhs[i,j]
    The scaling factor is
        s_ij = 4*|phi[i,j]| + |phi[i+1,j]| + |phi[i-1,j]| + |phi[i,j+1]| + |phi[i,j-1]| + h^2 * |rhs[i,j]|
    The global relative backward error is
        eta_2 = ||d||_2 / ||s||_2
    """
    h2 = h * h
    n = phi.shape[0] - 2
    sum_d2 = 0.0
    sum_s2 = 0.0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            d_ij = (
                4.0 * phi[i, j]
                - phi[i + 1, j]
                - phi[i - 1, j]
                - phi[i, j + 1]
                - phi[i, j - 1]
                - h2 * rhs[i, j]
            )
            s_ij = (
                abs(phi[i + 1, j])
                + abs(phi[i - 1, j])
                + abs(phi[i, j + 1])
                + abs(phi[i, j - 1])
                + 4.0 * abs(phi[i, j])
                + h2 * abs(rhs[i, j])
            )
            sum_d2 += d_ij * d_ij
            sum_s2 += s_ij * s_ij

    if sum_s2 == 0:
        return 0.0
    return np.sqrt(sum_d2) / np.sqrt(sum_s2)