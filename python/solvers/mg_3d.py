from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from numba import njit

from solvers.utils import _relative_physical_residual_l2_3d


@njit(cache=True)
def _residual_full_3d(phi, rhs, h):
    inv_h2 = 1.0 / (h * h)
    res = np.zeros_like(phi)
    n = phi.shape[0] - 2
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            for k in range(1, n + 1):
                res[i, j, k] = rhs[i, j, k] - (
                    6.0 * phi[i, j, k]
                    - phi[i + 1, j, k]
                    - phi[i - 1, j, k]
                    - phi[i, j + 1, k]
                    - phi[i, j - 1, k]
                    - phi[i, j, k + 1]
                    - phi[i, j, k - 1]
                ) * inv_h2
    return res


@njit(cache=True)
def _smooth_rb_3d(phi, rhs, h, omega, steps):
    h2 = h * h
    n = phi.shape[0] - 2
    for _ in range(steps):
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
    return phi


@njit(cache=True)
def _weight_3(offset):
    if offset == 0:
        return 2.0
    return 1.0


@njit(cache=True)
def _restrict_full_weighting_3d(fine):
    n = fine.shape[0] - 2
    nc = (n - 1) // 2
    coarse = np.zeros((nc + 2, nc + 2, nc + 2), dtype=fine.dtype)
    for i in range(1, nc + 1):
        fi = 2 * i
        for j in range(1, nc + 1):
            fj = 2 * j
            for k in range(1, nc + 1):
                fk = 2 * k
                total = 0.0
                for di in range(-1, 2):
                    wi = _weight_3(di)
                    for dj in range(-1, 2):
                        wij = wi * _weight_3(dj)
                        for dk in range(-1, 2):
                            total += (
                                wij
                                * _weight_3(dk)
                                * fine[fi + di, fj + dj, fk + dk]
                            )
                coarse[i, j, k] = total / 64.0
    return coarse


@njit(cache=True)
def _prolong_add_3d(coarse, fine):
    nc = coarse.shape[0] - 2
    nf = 2 * nc + 1
    for fi in range(1, nf + 1):
        if (fi & 1) == 0:
            i0 = fi // 2
            i1 = i0
            wi0 = 1.0
            wi1 = 0.0
        else:
            i0 = fi // 2
            i1 = i0 + 1
            wi0 = 0.5
            wi1 = 0.5
        for fj in range(1, nf + 1):
            if (fj & 1) == 0:
                j0 = fj // 2
                j1 = j0
                wj0 = 1.0
                wj1 = 0.0
            else:
                j0 = fj // 2
                j1 = j0 + 1
                wj0 = 0.5
                wj1 = 0.5
            for fk in range(1, nf + 1):
                if (fk & 1) == 0:
                    k0 = fk // 2
                    k1 = k0
                    wk0 = 1.0
                    wk1 = 0.0
                else:
                    k0 = fk // 2
                    k1 = k0 + 1
                    wk0 = 0.5
                    wk1 = 0.5
                fine[fi, fj, fk] += (
                    wi0 * wj0 * wk0 * coarse[i0, j0, k0]
                    + wi1 * wj0 * wk0 * coarse[i1, j0, k0]
                    + wi0 * wj1 * wk0 * coarse[i0, j1, k0]
                    + wi1 * wj1 * wk0 * coarse[i1, j1, k0]
                    + wi0 * wj0 * wk1 * coarse[i0, j0, k1]
                    + wi1 * wj0 * wk1 * coarse[i1, j0, k1]
                    + wi0 * wj1 * wk1 * coarse[i0, j1, k1]
                    + wi1 * wj1 * wk1 * coarse[i1, j1, k1]
                )
    return fine


@lru_cache(maxsize=None)
def _coarse_inverse_3d(n, h):
    m = n * n * n
    inv_h2 = 1.0 / (h * h)
    a = np.zeros((m, m), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                row = (i * n + j) * n + k
                a[row, row] = 6.0 * inv_h2
                if i > 0:
                    a[row, row - n * n] = -inv_h2
                if i + 1 < n:
                    a[row, row + n * n] = -inv_h2
                if j > 0:
                    a[row, row - n] = -inv_h2
                if j + 1 < n:
                    a[row, row + n] = -inv_h2
                if k > 0:
                    a[row, row - 1] = -inv_h2
                if k + 1 < n:
                    a[row, row + 1] = -inv_h2
    return np.linalg.inv(a)


def _solve_coarsest_3d(phi, rhs, h):
    n = phi.shape[0] - 2
    inv_a = _coarse_inverse_3d(n, float(h))
    b = np.empty(n * n * n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                row = (i * n + j) * n + k
                b[row] = rhs[i + 1, j + 1, k + 1]
    phi[1:-1, 1:-1, 1:-1] = (inv_a @ b).reshape(n, n, n).astype(
        phi.dtype,
        copy=False,
    )
    return phi


def _solve_coarsest_sor_3d(phi, rhs, h, omega, steps):
    _smooth_rb_3d(phi, rhs, h, omega, steps)
    return phi


def _cycle_3d(phi, rhs, h, nu, omega, is_w, coarse_mode, coarse_steps):
    n = phi.shape[0] - 2
    if n <= 4:
        if coarse_mode == "exact":
            _solve_coarsest_3d(phi, rhs, h)
        else:
            _solve_coarsest_sor_3d(phi, rhs, h, omega, coarse_steps)
        return

    _smooth_rb_3d(phi, rhs, h, omega, nu)
    coarse_rhs = _restrict_full_weighting_3d(_residual_full_3d(phi, rhs, h))
    coarse_err = np.zeros_like(coarse_rhs)
    coarse_n = coarse_rhs.shape[0] - 2
    _cycle_3d(
        coarse_err,
        coarse_rhs,
        2.0 * h,
        nu,
        omega,
        is_w,
        coarse_mode,
        coarse_steps,
    )
    if is_w and coarse_n > 4:
        # W-cycle: repeat the same coarse problem, but only when the next
        # level is not already the terminal coarse grid.
        _cycle_3d(
            coarse_err,
            coarse_rhs,
            2.0 * h,
            nu,
            omega,
            is_w,
            coarse_mode,
            coarse_steps,
        )
    _prolong_add_3d(coarse_err, phi)
    _smooth_rb_3d(phi, rhs, h, omega, nu)


def _v_cycle_3d(phi, rhs, h, nu, omega, coarse_mode, coarse_steps):
    _cycle_3d(phi, rhs, h, nu, omega, False, coarse_mode, coarse_steps)
    return phi


def _w_cycle_3d(phi, rhs, h, nu, omega, coarse_mode, coarse_steps):
    _cycle_3d(phi, rhs, h, nu, omega, True, coarse_mode, coarse_steps)
    return phi


def solve(
    problem,
    tol=1e-10,
    max_iter=20000,
    cycle="v",
    nu=2,
    omega=1,
    coarse_mode="exact",
    coarse_steps=16,
):
    if omega is None:
        omega = 2.0 / (1.0 + math.sin(math.pi / (problem.grid_size + 1)))
    cycle = cycle.lower()
    coarse_mode = coarse_mode.lower()
    if cycle == "v":
        step = _v_cycle_3d
    elif cycle == "w":
        step = _w_cycle_3d
    else:
        raise ValueError("cycle must be 'v' or 'w'")
    if coarse_mode not in {"exact", "sor"}:
        raise ValueError("coarse_mode must be 'exact' or 'sor'")
    if coarse_steps < 1:
        raise ValueError("coarse_steps must be positive")

    phi = problem.phi0.copy()
    residual = _relative_physical_residual_l2_3d(phi, problem.rhs, problem.h)
    iterations = 0
    while iterations < max_iter and residual > tol:
        phi = step(phi, problem.rhs, problem.h, nu, omega, coarse_mode, coarse_steps)
        iterations += 1
        residual = _relative_physical_residual_l2_3d(phi, problem.rhs, problem.h)
    return phi, iterations, float(residual)
