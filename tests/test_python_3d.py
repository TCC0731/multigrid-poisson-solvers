from __future__ import annotations

import math

import numpy as np
import pytest

from metrics import error_metrics_3d, residual_l2_3d
from operators import apply_A_3d, residual_3d
from problems import CASES_3D, make_problem_3d
from solvers.gs_3d import solve as solve_gs_3d
from solvers.jacobi_3d import solve as solve_jacobi_3d
from solvers.mg_3d import solve as solve_mg_3d
from solvers.sor_3d import solve as solve_sor_3d


SOLVERS_3D = {
    "jacobi": solve_jacobi_3d,
    "gs": solve_gs_3d,
    "sor": solve_sor_3d,
    "mg": solve_mg_3d,
}

DTYPES = (np.float32, np.float64)
ZERO_BC_CASES = {"sine", "mixed_sine", "bubble"}
SOLVER_TEST_GRID_SIZE = 7
MG_TEST_GRID_SIZE = 15
ALL_SOLVERS_MAX_ITER = 12000
MG_CYCLES_MAX_ITER = 60
DEFAULT_TOL_F32 = 1e-2
DEFAULT_TOL_F64 = 1e-8
MAX_ERROR_L2 = 2e-1
MAX_ERROR_LINF = 2e-1


def _dtype_tolerances(dtype):
    if np.dtype(dtype) == np.float32:
        return 1e-2, 1e-2
    return 1e-8, 1e-8


def _benchmark_tol(dtype):
    if np.dtype(dtype) == np.float32:
        return DEFAULT_TOL_F32
    return DEFAULT_TOL_F64


def _check_boundaries(problem, dtype):
    np.testing.assert_allclose(problem.phi0[1:-1, 1:-1, 1:-1], 0.0)
    np.testing.assert_allclose(problem.phi0[0, :, :], problem.exact[0, :, :])
    np.testing.assert_allclose(problem.phi0[-1, :, :], problem.exact[-1, :, :])
    np.testing.assert_allclose(problem.phi0[:, 0, :], problem.exact[:, 0, :])
    np.testing.assert_allclose(problem.phi0[:, -1, :], problem.exact[:, -1, :])
    np.testing.assert_allclose(problem.phi0[:, :, 0], problem.exact[:, :, 0])
    np.testing.assert_allclose(problem.phi0[:, :, -1], problem.exact[:, :, -1])
    assert problem.exact.dtype == np.dtype(dtype)
    assert problem.rhs.dtype == np.dtype(dtype)
    assert problem.phi0.dtype == np.dtype(dtype)
    assert np.array(problem.h).dtype == np.dtype(dtype)


def _assert_solution(
    problem,
    phi,
    iterations,
    res,
    dtype,
    max_iter,
    residual_tol,
):
    assert phi.shape == problem.phi0.shape
    assert phi.dtype == problem.phi0.dtype
    assert 1 <= iterations <= max_iter, (
        f"iterations {iterations} outside [1, {max_iter}]"
    )
    assert math.isfinite(res), f"solver residual is not finite: {res}"
    assert res >= 0.0, f"solver residual is negative: {res}"

    np.testing.assert_allclose(phi[0, :, :], problem.phi0[0, :, :])
    np.testing.assert_allclose(phi[-1, :, :], problem.phi0[-1, :, :])
    np.testing.assert_allclose(phi[:, 0, :], problem.phi0[:, 0, :])
    np.testing.assert_allclose(phi[:, -1, :], problem.phi0[:, -1, :])
    np.testing.assert_allclose(phi[:, :, 0], problem.phi0[:, :, 0])
    np.testing.assert_allclose(phi[:, :, -1], problem.phi0[:, :, -1])

    rtol, atol = _dtype_tolerances(dtype)
    actual_res = residual_l2_3d(problem, phi)
    np.testing.assert_allclose(actual_res, res, rtol=rtol, atol=atol)
    assert actual_res <= residual_tol, (
        f"residual {actual_res} exceeded tolerance {residual_tol}"
    )

    err_l2, err_linf = error_metrics_3d(problem, phi)
    assert math.isfinite(err_l2), f"error_l2 is not finite: {err_l2}"
    assert math.isfinite(err_linf), f"error_linf is not finite: {err_linf}"
    assert err_l2 >= 0.0, f"error_l2 is negative: {err_l2}"
    assert err_linf >= 0.0, f"error_linf is negative: {err_linf}"
    assert err_l2 <= MAX_ERROR_L2, (
        f"error_l2 {err_l2} exceeded {MAX_ERROR_L2}"
    )
    assert err_linf <= MAX_ERROR_LINF, (
        f"error_linf {err_linf} exceeded {MAX_ERROR_LINF}"
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=[d.__name__ for d in DTYPES])
@pytest.mark.parametrize("case", CASES_3D)
def test_make_problem_3d_cases_and_dtypes(case, dtype):
    problem = make_problem_3d(case, 7, dtype=dtype)

    assert problem.case == case
    assert problem.grid_size == 7
    assert problem.exact.shape == (9, 9, 9)
    assert problem.rhs.shape == (9, 9, 9)
    assert problem.phi0.shape == (9, 9, 9)

    _check_boundaries(problem, dtype)

    if case in ZERO_BC_CASES:
        np.testing.assert_allclose(problem.phi0, 0.0, atol=1e-6)
    else:
        assert np.any(np.abs(problem.phi0[[0, -1], :, :]) > 0.0)
        assert np.any(np.abs(problem.phi0[:, [0, -1], :]) > 0.0)
        assert np.any(np.abs(problem.phi0[:, :, [0, -1]]) > 0.0)


def test_apply_a_3d_and_residual_stencil():
    phi = np.zeros((5, 5, 5), dtype=np.float64)
    phi[2, 2, 2] = 1.0
    rhs = np.zeros_like(phi)
    h = 1.0

    expected = np.zeros((3, 3, 3), dtype=np.float64)
    expected[1, 1, 1] = 6.0
    expected[0, 1, 1] = -1.0
    expected[2, 1, 1] = -1.0
    expected[1, 0, 1] = -1.0
    expected[1, 2, 1] = -1.0
    expected[1, 1, 0] = -1.0
    expected[1, 1, 2] = -1.0

    np.testing.assert_allclose(apply_A_3d(phi, h), expected)
    np.testing.assert_allclose(residual_3d(phi, rhs, h), -expected)


@pytest.mark.parametrize("dtype", DTYPES, ids=[d.__name__ for d in DTYPES])
@pytest.mark.parametrize("case", CASES_3D)
@pytest.mark.parametrize("solver_name", tuple(SOLVERS_3D))
def test_all_3d_solvers_run(case, dtype, solver_name):
    problem = make_problem_3d(case, SOLVER_TEST_GRID_SIZE, dtype=dtype)
    solve = SOLVERS_3D[solver_name]
    tol = _benchmark_tol(dtype)

    phi, iterations, res = solve(problem, tol=tol, max_iter=ALL_SOLVERS_MAX_ITER)

    _assert_solution(
        problem,
        phi,
        iterations,
        res,
        dtype,
        ALL_SOLVERS_MAX_ITER,
        tol,
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=[d.__name__ for d in DTYPES])
@pytest.mark.parametrize("cycle", ("v", "w"))
@pytest.mark.parametrize("coarse_mode", ("exact", "sor"))
def test_3d_multigrid_v_and_w_cycles(dtype, cycle, coarse_mode):
    problem = make_problem_3d("sine", MG_TEST_GRID_SIZE, dtype=dtype)
    tol = _benchmark_tol(dtype)

    solve_jacobi_3d(problem, tol=tol, max_iter=1)
    solve_gs_3d(problem, tol=tol, max_iter=1)
    solve_sor_3d(problem, tol=tol, max_iter=1)

    phi, iterations, res = solve_mg_3d(
        problem,
        tol=tol,
        max_iter=MG_CYCLES_MAX_ITER,
        cycle=cycle,
        coarse_mode=coarse_mode,
    )

    _assert_solution(
        problem,
        phi,
        iterations,
        res,
        dtype,
        MG_CYCLES_MAX_ITER,
        tol,
    )


def test_invalid_3d_inputs_raise():
    with pytest.raises(ValueError):
        make_problem_3d("unknown", 7)

    with pytest.raises(ValueError):
        make_problem_3d("sine", 0)

    with pytest.raises(ValueError):
        make_problem_3d("sine", 7, dtype=np.int32)

    problem = make_problem_3d("sine", 7)
    with pytest.raises(ValueError):
        solve_mg_3d(problem, cycle="bad")
    with pytest.raises(ValueError):
        solve_mg_3d(problem, coarse_mode="bad")
    with pytest.raises(ValueError):
        solve_mg_3d(problem, coarse_steps=0)
