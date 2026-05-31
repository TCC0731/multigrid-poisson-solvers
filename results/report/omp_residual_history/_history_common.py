from __future__ import annotations

import csv
import io
import os
import pickle
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from _poisson_wrapper import DEFAULT_OMP_ENV, build_request, resolve_executable

DEFAULT_BACKEND = "omp"
DEFAULT_CASE = "sine"
DEFAULT_DIMS = (2, 3)
DEFAULT_DTYPE = "double"
DEFAULT_GRID_SIZES = {
    2: {"sor": 512, "mg": 511},
    3: {"sor": 127, "mg": 127},
}
DEFAULT_OMEGA = 1.25
DEFAULT_NU = 3
DEFAULT_OMP_NUM_THREADS = 1
DEFAULT_REPEAT_RUNS = 1
DEFAULT_TOL = 1e-9
DEFAULT_SOR_MAX_ITER = 10_000
DEFAULT_MG_MAX_ITER = 150
DEFAULT_COARSE_STEPS = 16


@dataclass(frozen=True)
class ModeSpec:
    label: str
    mode_key: str
    solver: str
    color: str
    marker: str
    linestyle: str
    cycle: str | None = None
    mg_coarse: str | None = None
    omega: float | None = None
    nu: int | None = None
    max_iter: int = 0


MODE_SPECS = (
    ModeSpec(
        label="SOR",
        mode_key="sor",
        solver="sor",
        color="#1f77b4",
        marker="o",
        linestyle="-",
        max_iter=DEFAULT_SOR_MAX_ITER,
    ),
    ModeSpec(
        label="MG V-SOR",
        mode_key="mg_v_sor",
        solver="mg",
        color="#ff7f0e",
        marker="s",
        linestyle="--",
        cycle="v",
        mg_coarse="sor",
        omega=DEFAULT_OMEGA,
        nu=DEFAULT_NU,
        max_iter=DEFAULT_MG_MAX_ITER,
    ),
    ModeSpec(
        label="MG W-SOR",
        mode_key="mg_w_sor",
        solver="mg",
        color="#2ca02c",
        marker="^",
        linestyle="-.",
        cycle="w",
        mg_coarse="sor",
        omega=DEFAULT_OMEGA,
        nu=DEFAULT_NU,
        max_iter=DEFAULT_MG_MAX_ITER,
    ),
)
MODE_SPECS_BY_KEY = {spec.mode_key: spec for spec in MODE_SPECS}


def default_grid_size(dim: int, solver: str) -> int:
    try:
        return int(DEFAULT_GRID_SIZES[dim][solver])
    except KeyError as exc:
        raise ValueError(f"unsupported grid size request for dim={dim}, solver={solver!r}") from exc


@dataclass(frozen=True)
class HistoryRecord:
    dimension: int
    case: str
    grid_size: int
    mode_label: str
    mode_key: str
    solver: str
    backend: str
    dtype: str
    iterations: int
    residual_l2: float
    error_l2: float
    error_linf: float
    time_ms: float
    tol: float
    max_iter: int
    repeat_runs: int
    omp_num_threads: int
    cycle: str | None
    mg_coarse: str | None
    omega: float | None
    nu: int | None
    converged: bool
    history: list[float]


def _validate_odd_grid_size(dim: int, grid_size: int) -> None:
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    if grid_size % 2 == 0:
        raise ValueError(
            f"{dim}D multigrid residual history requires an odd interior grid size; got {grid_size}"
        )


def _build_omp_env(omp_num_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(DEFAULT_OMP_ENV)
    env["OMP_NUM_THREADS"] = str(int(omp_num_threads))
    return env


def _build_command(request, residual_history_path: Path) -> list[str]:
    executable = resolve_executable("omp")
    cmd = [
        str(executable),
        "--dim",
        str(request.dim),
        "--solver",
        request.solver,
        "--case",
        request.case,
        "--grid-size",
        str(request.grid_size),
        "--dtype",
        request.dtype,
        "--tol",
        repr(float(request.tol)),
        "--max-iter",
        str(request.max_iter),
        "--repeat-runs",
        str(request.repeat_runs),
    ]

    if request.solver == "mg":
        cmd.extend(
            [
                "--cycle",
                request.cycle or "v",
                "--nu",
                str(request.nu if request.nu is not None else DEFAULT_NU),
                "--omega",
                repr(float(request.omega if request.omega is not None else DEFAULT_OMEGA)),
                "--mg-coarse",
                request.mg_coarse or "sor",
            ]
        )

    cmd.extend(["--residual-history", str(residual_history_path)])
    return cmd


def _read_solver_stdout(stdout: str, *, cmd: list[str]) -> dict[str, object]:
    output = stdout.strip()
    if not output:
        raise RuntimeError(f"solver did not produce CSV output.\nCommand: {' '.join(cmd)}")

    rows = list(csv.DictReader(io.StringIO(output)))
    if len(rows) != 1:
        raise RuntimeError(
            "expected exactly one CSV row from the solver.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{stdout}"
        )

    row = rows[0]
    return {
        "solver": row["solver"],
        "backend": row["backend"],
        "dtype": row["dtype"],
        "iterations": int(row["iterations"]),
        "residual_l2": float(row["residual_l2"]),
        "error_l2": float(row["error_l2"]),
        "error_linf": float(row["error_linf"]),
        "time_ms": float(row["time_ms"]),
    }


def _read_history_csv(path: Path) -> list[float]:
    if not path.is_file():
        raise RuntimeError(f"residual history file was not created: {path}")

    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    if not rows:
        raise RuntimeError(f"residual history file is empty: {path}")

    history: list[float] = []
    for row in rows:
        history.append(float(row["residual_l2"]))
    return history


def collect_history_record(
    *,
    dim: int,
    case: str,
    grid_size: int,
    mode: ModeSpec,
    tol: float = DEFAULT_TOL,
    max_iter: int | None = None,
    repeat_runs: int = DEFAULT_REPEAT_RUNS,
    omp_num_threads: int = DEFAULT_OMP_NUM_THREADS,
    dtype: str = DEFAULT_DTYPE,
) -> HistoryRecord:
    if mode.solver == "mg":
        _validate_odd_grid_size(dim, grid_size)
    mode_max_iter = mode.max_iter if max_iter is None else int(max_iter)

    request = build_request(
        backend=DEFAULT_BACKEND,
        dim=dim,
        dtype=dtype,
        solver=mode.solver,
        case=case,
        grid_size=grid_size,
        tol=tol,
        max_iter=mode_max_iter,
        repeat_runs=repeat_runs,
        cycle=mode.cycle or "v",
        nu=mode.nu or DEFAULT_NU,
        omega=mode.omega if mode.omega is not None else DEFAULT_OMEGA,
        mg_coarse=mode.mg_coarse or "exact",
        omp_num_threads=omp_num_threads,
    )

    with tempfile.TemporaryDirectory(prefix="omp_residual_history_") as tmpdir:
        residual_history_path = Path(tmpdir) / "residual_history.csv"
        cmd = _build_command(request, residual_history_path)
        env = _build_omp_env(omp_num_threads)

        try:
            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "OpenMP solver failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{exc.stdout}\n"
                f"stderr:\n{exc.stderr}"
            ) from exc

        result_row = _read_solver_stdout(completed.stdout, cmd=cmd)
        history = _read_history_csv(residual_history_path)
        expected_length = result_row["iterations"] + 1
        if len(history) != expected_length:
            raise RuntimeError(
                "residual history length does not match the reported iteration count.\n"
                f"Command: {' '.join(cmd)}\n"
                f"expected {expected_length} rows but saw {len(history)}"
            )

    return HistoryRecord(
        dimension=dim,
        case=case,
        grid_size=grid_size,
        mode_label=mode.label,
        mode_key=mode.mode_key,
        solver=str(result_row["solver"]),
        backend=str(result_row["backend"]),
        dtype=str(result_row["dtype"]),
        iterations=int(result_row["iterations"]),
        residual_l2=float(result_row["residual_l2"]),
        error_l2=float(result_row["error_l2"]),
        error_linf=float(result_row["error_linf"]),
        time_ms=float(result_row["time_ms"]),
        tol=float(tol),
        max_iter=mode_max_iter,
        repeat_runs=int(repeat_runs),
        omp_num_threads=int(omp_num_threads),
        cycle=mode.cycle,
        mg_coarse=mode.mg_coarse,
        omega=mode.omega,
        nu=mode.nu,
        converged=float(result_row["residual_l2"]) <= float(tol),
        history=history,
    )


def load_payload(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        return pickle.load(stream)


def write_payload(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
