from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from numba import njit

from solvers.utils import _relative_residual_l2, _residual_l2, _residual

@njit(cache=True)
def _residual_full(phi, rhs, h):
    inv_h2 = 1.0 / (h * h)
    res = np.zeros_like(phi)
    n = phi.shape[0] - 2
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            res[i, j] = rhs[i, j] - (
                4.0 * phi[i, j]
                - phi[i + 1, j]
                - phi[i - 1, j]
                - phi[i, j + 1]
                - phi[i, j - 1]
            ) * inv_h2
    return res


@njit(cache=True)
def _smooth_rb(phi, rhs, h, omega, steps):
    h2 = h * h
    n = phi.shape[0] - 2
    for _ in range(steps):
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
    return phi


@njit(cache=True)
def _restrict_full_weighting(fine):
    n = fine.shape[0] - 2
    nc = (n - 1) // 2
    coarse = np.zeros((nc + 2, nc + 2), dtype=fine.dtype)
    for i in range(1, nc + 1):
        fi = 2 * i
        for j in range(1, nc + 1):
            fj = 2 * j
            coarse[i, j] = (
                4.0 * fine[fi, fj]
                + 2.0
                * (
                    fine[fi - 1, fj]
                    + fine[fi + 1, fj]
                    + fine[fi, fj - 1]
                    + fine[fi, fj + 1]
                )
                + fine[fi - 1, fj - 1]
                + fine[fi - 1, fj + 1]
                + fine[fi + 1, fj - 1]
                + fine[fi + 1, fj + 1]
            ) / 16.0
    return coarse


@njit(cache=True)
def _prolong_add(coarse, fine):
    nc = coarse.shape[0] - 2
    for i in range(1, nc + 1):
        fi = 2 * i
        for j in range(1, nc + 1):
            fj = 2 * j
            fine[fi, fj] += coarse[i, j]
    for i in range(1, nc):
        fi = 2 * i + 1
        for j in range(1, nc + 1):
            fj = 2 * j
            fine[fi, fj] += 0.5 * (coarse[i, j] + coarse[i + 1, j])
    for i in range(1, nc + 1):
        fi = 2 * i
        for j in range(1, nc):
            fj = 2 * j + 1
            fine[fi, fj] += 0.5 * (coarse[i, j] + coarse[i, j + 1])
    for i in range(1, nc):
        fi = 2 * i + 1
        for j in range(1, nc):
            fj = 2 * j + 1
            fine[fi, fj] += 0.25 * (
                coarse[i, j]
                + coarse[i + 1, j]
                + coarse[i, j + 1]
                + coarse[i + 1, j + 1]
    )
    return fine


@lru_cache(maxsize=None)
def _coarse_inverse(n, h):
    m = n * n
    inv_h2 = 1.0 / (h * h)
    a = np.zeros((m, m), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            k = i * n + j
            a[k, k] = 4.0 * inv_h2
            if i > 0:
                a[k, k - n] = -inv_h2
            if i + 1 < n:
                a[k, k + n] = -inv_h2
            if j > 0:
                a[k, k - 1] = -inv_h2
            if j + 1 < n:
                a[k, k + 1] = -inv_h2
    return np.linalg.inv(a)


def _solve_coarsest(phi, rhs, h):
    n = phi.shape[0] - 2
    inv_a = _coarse_inverse(n, float(h))
    b = np.empty(n * n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            k = i * n + j
            b[k] = rhs[i + 1, j + 1]
    phi[1:-1, 1:-1] = (inv_a @ b).reshape(n, n).astype(phi.dtype, copy=False)
    return phi


def _cycle(phi, rhs, h, nu, omega, is_w):
    n = phi.shape[0] - 2
    if n <= 4:
        _solve_coarsest(phi, rhs, h)
        return

    _smooth_rb(phi, rhs, h, omega, nu)
    coarse_rhs = _restrict_full_weighting(_residual_full(phi, rhs, h))
    coarse_err = np.zeros_like(coarse_rhs)
    _cycle(coarse_err, coarse_rhs, 2.0 * h, nu, omega, is_w)
    if is_w:
        # W-cycle: run the same coarse problem a second time, using the first
        # correction as the initial guess for the second pass.
        _cycle(coarse_err, coarse_rhs, 2.0 * h, nu, omega, is_w)
    _prolong_add(coarse_err, phi)
    _smooth_rb(phi, rhs, h, omega, nu)


def _v_cycle(phi, rhs, h, nu, omega):
    _cycle(phi, rhs, h, nu, omega, False)
    return phi


def _w_cycle(phi, rhs, h, nu, omega):
    _cycle(phi, rhs, h, nu, omega, True)
    return phi


def solve(problem, tol=1e-10, max_iter=20000, cycle="v", nu=2, omega=1):
    if omega is None:
        omega = 2.0 / (1.0 + math.sin(math.pi / (problem.grid_size + 1)))
    cycle = cycle.lower()
    if cycle == "v":
        step = _v_cycle
    elif cycle == "w":
        step = _w_cycle
    else:
        raise ValueError("cycle must be 'v' or 'w'")

    phi = problem.phi0.copy()
    residual = _residual(phi, problem.rhs, problem.h)
    iterations = 0
    while iterations < max_iter and residual > tol:
        phi = step(phi, problem.rhs, problem.h, nu, omega)
        iterations += 1
        residual = _residual(phi, problem.rhs, problem.h)
    return phi, iterations, float(residual)
