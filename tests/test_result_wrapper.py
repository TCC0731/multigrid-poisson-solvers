from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "report" / "result"
if str(RESULT_DIR) not in sys.path:
    sys.path.insert(0, str(RESULT_DIR))

import _poisson_wrapper as wrapper


CSV_OUTPUT = (
    "solver,backend,dtype,grid_size,iterations,residual_l2,error_l2,error_linf,time_ms\n"
    "mg_sor,cuda,double,31,7,1.250000e-01,2.500000e-01,5.000000e-01,12.345678\n"
)

LEGACY_CACHE_OUTPUT = """\
backend,dim,dtype,solver,case,grid_size,tol,max_iter,repeat_runs,cycle,nu,omega,mg_coarse,omp_num_threads,iterations,residual_l2,error_l2,error_linf,time_ms
cuda,2,double,mg,sine,31,1e-10,20000,25,v,2,1.0,sor,,7,1.250000e-01,2.500000e-01,5.000000e-01,12.345678
"""


def _touch(path: Path) -> Path:
    path.write_text("", encoding="utf-8")
    return path


def _read_cache_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


def test_build_request_sets_default_and_custom_coarse_steps():
    default_request = wrapper.build_request(
        backend="cuda",
        solver="mg",
        mg_coarse="sor",
    )
    custom_request = wrapper.build_request(
        backend="cuda",
        solver="mg",
        mg_coarse="sor",
        coarse_steps=32,
    )

    assert default_request.coarse_steps == 16
    assert default_request.to_row_dict()["coarse_steps"] == "16"
    assert custom_request.coarse_steps == 32
    assert custom_request.to_row_dict()["coarse_steps"] == "32"


def test_run_or_load_builds_cuda_command_with_coarse_steps(tmp_path, monkeypatch):
    cache_csv = tmp_path / "solver_results.csv"
    executable = _touch(tmp_path / "poisson_cuda")
    calls: list[dict[str, object]] = []

    def fake_run(cmd, check, capture_output, text, env=None):
        calls.append({"cmd": list(cmd), "env": env})
        return SimpleNamespace(stdout=CSV_OUTPUT, stderr="")

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)

    result = wrapper.run_or_load(
        backend="cuda",
        dim=3,
        dtype="double",
        solver="mg",
        case="bubble",
        grid_size=63,
        tol=None,
        max_iter=1234,
        cycle="w",
        nu=3,
        omega="auto",
        mg_coarse="sor",
        coarse_steps=32,
        cache_csv=cache_csv,
        executable=executable,
    )

    assert result.backend == "cuda"
    assert result.solver == "mg"
    assert result.coarse_steps == 32
    assert result.time_s == pytest.approx(0.012345678)
    assert len(calls) == 1

    cmd = calls[0]["cmd"]
    assert cmd == [
        str(executable),
        "--dim",
        "3",
        "--solver",
        "mg",
        "--case",
        "bubble",
        "--grid-size",
        "63",
        "--dtype",
        "double",
        "--tol",
        "1e-10",
        "--max-iter",
        "1234",
        "--repeat-runs",
        "25",
        "--cycle",
        "w",
        "--nu",
        "3",
        "--omega",
        "auto",
        "--mg-coarse",
        "sor",
        "--coarse-steps",
        "32",
    ]
    assert calls[0]["env"] is None

    rows = _read_cache_rows(cache_csv)
    assert len(rows) == 1
    row = rows[0]
    assert row["mg_coarse"] == "sor"
    assert row["coarse_steps"] == "32"


def test_run_or_load_ignores_coarse_steps_for_non_mg(tmp_path, monkeypatch):
    cache_csv = tmp_path / "solver_results.csv"
    executable = _touch(tmp_path / "poisson_cuda")
    calls: list[dict[str, object]] = []

    def fake_run(cmd, check, capture_output, text, env=None):
        calls.append({"cmd": list(cmd), "env": env})
        return SimpleNamespace(stdout=CSV_OUTPUT.replace("mg_sor", "jacobi"), stderr="")

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)

    result = wrapper.run_or_load(
        backend="cuda",
        solver="jacobi",
        case="sine",
        grid_size=31,
        cache_csv=cache_csv,
        executable=executable,
        cycle="w",
        nu=7,
        omega="auto",
        mg_coarse="sor",
        coarse_steps=99,
    )

    assert len(calls) == 1
    assert result.backend == "cuda"
    assert result.solver == "jacobi"
    assert result.coarse_steps is None
    assert all(part != "--coarse-steps" for part in calls[0]["cmd"])

    rows = _read_cache_rows(cache_csv)
    assert len(rows) == 1
    row = rows[0]
    assert row["solver"] == "jacobi"
    assert row["mg_coarse"] == ""
    assert row["coarse_steps"] == ""


def test_run_or_load_omits_coarse_steps_for_omp(tmp_path, monkeypatch):
    cache_csv = tmp_path / "solver_results.csv"
    executable = _touch(tmp_path / "poisson_cpp_omp")
    calls: list[dict[str, object]] = []

    def fake_run(cmd, check, capture_output, text, env=None):
        calls.append({"cmd": list(cmd), "env": env})
        return SimpleNamespace(stdout=CSV_OUTPUT.replace("mg_sor", "mg_sor_omp"), stderr="")

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)

    result = wrapper.run_or_load(
        backend="omp",
        dim=2,
        dtype="double",
        solver="mg",
        case="sine",
        grid_size=31,
        tol=None,
        max_iter=1234,
        cycle="v",
        nu=3,
        omega=1.25,
        mg_coarse="sor",
        coarse_steps=32,
        omp_num_threads=4,
        cache_csv=cache_csv,
        executable=executable,
    )

    assert len(calls) == 1
    assert result.backend == "omp"
    assert result.mg_coarse == "sor"
    assert result.coarse_steps is None
    assert result.omp_num_threads == 4
    assert all(part != "--coarse-steps" for part in calls[0]["cmd"])

    rows = _read_cache_rows(cache_csv)
    assert len(rows) == 1
    row = rows[0]
    assert row["backend"] == "omp"
    assert row["mg_coarse"] == "sor"
    assert row["coarse_steps"] == ""
    assert row["omp_num_threads"] == "4"


def test_load_cache_accepts_legacy_header_and_defaults_coarse_steps(tmp_path):
    cache_csv = tmp_path / "legacy_solver_results.csv"
    cache_csv.write_text(LEGACY_CACHE_OUTPUT, encoding="utf-8")

    records = wrapper.load_cache(cache_csv)
    request = wrapper.build_request(
        backend="cuda",
        solver="mg",
        mg_coarse="sor",
    )
    result = records[request.cache_key()]

    assert result.coarse_steps == 16
    assert result.mg_coarse == "sor"
