from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import run_poisson


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_poisson.py"])

    args = run_poisson.parse_args()

    assert args.solver == "jacobi"
    assert args.case == "sine"
    assert args.dtype == "float64"
    assert args.grid_size == 31
    assert args.tol == 1e-10
    assert args.max_iter == 20000


def test_parse_args_rejects_non_positive_grid_size(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_poisson.py", "--grid-size", "0"])

    with pytest.raises(SystemExit) as exc_info:
        run_poisson.parse_args()

    assert exc_info.value.code == 2


def test_load_helpers_use_expected_modules(monkeypatch):
    problem_result = object()
    solver_result = object()
    error_metrics_result = object()

    problem_module = SimpleNamespace(
        make_problem=Mock(return_value=problem_result),
    )
    solver_module = SimpleNamespace(
        solve=Mock(return_value=solver_result),
    )
    metrics_module = SimpleNamespace(
        error_metrics=Mock(return_value=error_metrics_result),
    )
    import_module = Mock(
        side_effect=lambda name: {
            "problems": problem_module,
            "solvers.sor_2d": solver_module,
            "metrics": metrics_module,
        }[name]
    )
    monkeypatch.setattr(run_poisson, "import_module", import_module)

    problem = run_poisson.load_problem("cosine", 13, np.float32)
    solve = run_poisson.load_solver("sor")
    error_metrics = run_poisson.load_error_metrics()

    assert problem is problem_result
    assert solve is solver_module.solve
    assert error_metrics is metrics_module.error_metrics
    problem_module.make_problem.assert_called_once_with(
        case="cosine", grid_size=13, dtype=np.float32
    )
    assert import_module.call_args_list == [
        (( "problems",), {}),
        (("solvers.sor_2d",), {}),
        (("metrics",), {}),
    ]


def test_require_attr_raises_for_missing_module_and_attribute(monkeypatch):
    def fake_import_module(name):
        if name == "metrics":
            return SimpleNamespace()
        raise ImportError(name)

    monkeypatch.setattr(run_poisson, "import_module", Mock(side_effect=fake_import_module))

    with pytest.raises(SystemExit, match="Missing module 'missing.module'"):
        run_poisson._require_attr("missing.module", "thing")

    with pytest.raises(SystemExit, match="Module 'metrics' must define 'error_metrics'"):
        run_poisson.load_error_metrics()


def test_warmup_solver_uses_fixed_short_run():
    solve = Mock()
    problem = object()

    run_poisson.warmup_solver(solve, problem)

    solve.assert_called_once_with(problem, tol=1e-2, max_iter=2)


def test_main_prints_expected_csv_row(monkeypatch, capsys):
    problem = SimpleNamespace(
        phi0=np.zeros((5, 5), dtype=np.float32),
        rhs=np.zeros((5, 5), dtype=np.float32),
        h=np.float32(0.25),
        exact=np.zeros((5, 5), dtype=np.float32),
    )
    phi = np.full_like(problem.phi0, 3.0)
    solve = Mock(return_value=(phi, 7, 0.125))
    error_metrics = Mock(return_value=(0.25, 0.5))

    parse_args_result = SimpleNamespace(
        solver="mg",
        case="bubble",
        dtype="float32",
        grid_size=5,
        tol=1e-6,
        max_iter=99,
    )
    load_problem = Mock(return_value=problem)
    load_solver = Mock(return_value=solve)
    load_error_metrics = Mock(return_value=error_metrics)
    warmup_solver = Mock()
    perf_counter = Mock(side_effect=[100.0, 100.123456])

    monkeypatch.setattr(run_poisson, "parse_args", Mock(return_value=parse_args_result))
    monkeypatch.setattr(run_poisson, "load_problem", load_problem)
    monkeypatch.setattr(run_poisson, "load_solver", load_solver)
    monkeypatch.setattr(run_poisson, "load_error_metrics", load_error_metrics)
    monkeypatch.setattr(run_poisson, "warmup_solver", warmup_solver)
    monkeypatch.setattr(run_poisson, "perf_counter", perf_counter)

    run_poisson.main()

    out = capsys.readouterr().out.strip().splitlines()

    assert out == [
        "solver,backend,dtype,grid_size,iterations,residual_l2,error_l2,error_linf,time_ms",
        "mg,python,float32,5,7,1.250000e-01,2.500000e-01,5.000000e-01,123.456",
    ]
    load_problem.assert_called_once_with("bubble", 5, np.float32)
    load_solver.assert_called_once_with("mg")
    load_error_metrics.assert_called_once_with()
    warmup_solver.assert_called_once_with(solve, problem)
    solve.assert_called_once_with(problem, tol=1e-6, max_iter=99)
    error_metrics.assert_called_once_with(problem, phi)
    assert perf_counter.call_count == 2
