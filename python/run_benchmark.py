from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from metrics import error_metrics
from problems import make_problem
from solvers.gs_2d import solve as solve_gs
from solvers.jacobi_2d import solve as solve_jacobi
from solvers.mg_2d import solve as solve_mg
from solvers.sor_2d import solve as solve_sor


CASE_NAME = "sine"
BACKEND_NAME = "python"
DTYPE_NAME = "double"
SUITE_SOLVER_COMPARISON = "solver_comparison"
SUITE_MG_COMPARE = "mg_compare"
WARMUP_RUNS = 1
TIMED_RUNS = 5
WARMUP_MAX_ITER = 10
TOL = 1e-9
MG_OMEGA = 1.25
MG_NU = 3
MG_COARSE_STEPS = 16
MG_MAX_ITER = 150
SOLVER_SIZES = (15, 31, 63, 127)
RB_SOR_SIZES = (15, 31, 63, 127, 256, 512)
MG_SIZES = (15, 31, 63, 127, 255, 511, 1023, 2047, 4095)


@dataclass(frozen=True)
class SolveResult:
    phi: np.ndarray
    iterations: int
    residual_l2: float


@dataclass(frozen=True)
class TimingStats:
    mean_ms: float
    std_ms: float


@dataclass(frozen=True)
class BenchmarkRow:
    suite: str
    backend: str
    dtype: str
    case_name: str
    solver: str
    grid_size: int
    max_iter: int
    tol: float
    warmup_runs: int
    timed_runs: int
    cycle: str
    omega: str
    nu: str
    iterations: int
    residual_l2: float
    error_l2: float
    error_linf: float
    mean_time_ms: float
    std_time_ms: float


def _normalize_token(value: str) -> str:
    return value.lower().replace("-", "_")


def parse_suite(value: str) -> str:
    normalized = _normalize_token(value)
    if normalized == "all":
        return "all"
    if normalized in {"solver", SUITE_SOLVER_COMPARISON}:
        return SUITE_SOLVER_COMPARISON
    if normalized in {"mg", SUITE_MG_COMPARE}:
        return SUITE_MG_COMPARE
    raise argparse.ArgumentTypeError(f"unknown benchmark suite: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Python benchmark suite that mirrors the C++ benchmark CSV output."
        )
    )
    parser.add_argument(
        "--suite",
        type=parse_suite,
        default="all",
        help="Benchmark suite: all, solver_comparison, or mg_compare.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output base path. Writes BASE.csv and BASE_all.csv when provided.",
    )
    return parser.parse_args()


def _as_result(raw_result) -> SolveResult:
    phi, iterations, residual_l2 = raw_result
    return SolveResult(phi=phi, iterations=int(iterations), residual_l2=float(residual_l2))


def time_solver(warmup_solver, timed_solver, warmup_runs: int, timed_runs: int):
    if timed_runs < 1:
        raise ValueError("timed_runs must be positive")

    for _ in range(warmup_runs):
        warmup_solver()

    samples: list[float] = []
    last_result: SolveResult | None = None

    for _ in range(timed_runs):
        start = perf_counter()
        last_result = _as_result(timed_solver())
        end = perf_counter()
        samples.append((end - start) * 1000.0)

    mean_ms = sum(samples) / len(samples)
    if len(samples) == 1:
        std_ms = 0.0
    else:
        variance = sum((sample - mean_ms) ** 2 for sample in samples) / (len(samples) - 1)
        std_ms = float(np.sqrt(variance))

    if last_result is None:
        raise RuntimeError("timed solver did not produce any result")

    return last_result, TimingStats(mean_ms=mean_ms, std_ms=std_ms)


def make_row(
    suite: str,
    solver_name: str,
    grid_size: int,
    max_iter: int,
    tol: float,
    cycle: str,
    omega: str,
    nu: str,
    warmup_solver_fn,
    timed_solver_fn,
) -> BenchmarkRow:
    problem = make_problem(CASE_NAME, grid_size, dtype=np.float64)

    def warmup_solver():
        return warmup_solver_fn(problem)

    def timed_solver():
        return timed_solver_fn(problem)

    result, stats = time_solver(warmup_solver, timed_solver, WARMUP_RUNS, TIMED_RUNS)
    metrics_problem = make_problem(CASE_NAME, grid_size, dtype=np.float64)
    error_l2, error_linf = error_metrics(metrics_problem, result.phi)

    return BenchmarkRow(
        suite=suite,
        backend=BACKEND_NAME,
        dtype=DTYPE_NAME,
        case_name=CASE_NAME,
        solver=solver_name,
        grid_size=grid_size,
        max_iter=max_iter,
        tol=tol,
        warmup_runs=WARMUP_RUNS,
        timed_runs=TIMED_RUNS,
        cycle=cycle,
        omega=omega,
        nu=nu,
        iterations=result.iterations,
        residual_l2=result.residual_l2,
        error_l2=float(error_l2),
        error_linf=float(error_linf),
        mean_time_ms=stats.mean_ms,
        std_time_ms=stats.std_ms,
    )


def append_mg_rows(
    rows: list[BenchmarkRow],
    solver_name: str,
    cycle: str,
    omega: str,
    nu: str,
    max_iter: int,
    tol: float,
    warmup_kwargs: dict[str, object],
    timed_kwargs: dict[str, object],
) -> None:
    for grid_size in MG_SIZES:
        rows.append(
            make_row(
                SUITE_MG_COMPARE,
                solver_name,
                grid_size,
                max_iter,
                tol,
                cycle,
                omega,
                nu,
                lambda problem: solve_mg(problem, **warmup_kwargs),
                lambda problem: solve_mg(problem, **timed_kwargs),
            )
        )


def make_solver_comparison_rows() -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []

    for grid_size in SOLVER_SIZES:
        rows.append(
            make_row(
                SUITE_SOLVER_COMPARISON,
                "Jacobi",
                grid_size,
                100000,
                TOL,
                "",
                "",
                "",
                lambda problem: solve_jacobi(problem, tol=TOL, max_iter=WARMUP_MAX_ITER),
                lambda problem: solve_jacobi(problem, tol=TOL, max_iter=100000),
            )
        )

    for grid_size in SOLVER_SIZES:
        rows.append(
            make_row(
                SUITE_SOLVER_COMPARISON,
                "RB GS",
                grid_size,
                100000,
                TOL,
                "",
                "",
                "",
                lambda problem: solve_gs(problem, tol=TOL, max_iter=WARMUP_MAX_ITER),
                lambda problem: solve_gs(problem, tol=TOL, max_iter=100000),
            )
        )

    for grid_size in RB_SOR_SIZES:
        rows.append(
            make_row(
                SUITE_SOLVER_COMPARISON,
                "RB SOR",
                grid_size,
                10000,
                TOL,
                "",
                "",
                "",
                lambda problem: solve_sor(problem, tol=TOL, max_iter=WARMUP_MAX_ITER),
                lambda problem: solve_sor(problem, tol=TOL, max_iter=10000),
            )
        )

    return rows


def make_mg_compare_rows() -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []

    mg_v_warmup_kwargs = {
        "tol": TOL,
        "max_iter": WARMUP_MAX_ITER,
        "nu": MG_NU,
        "cycle": "v",
        "omega": MG_OMEGA,
        "coarse_mode": "exact",
        "coarse_steps": MG_COARSE_STEPS,
    }
    mg_v_kwargs = {
        "tol": TOL,
        "max_iter": MG_MAX_ITER,
        "nu": MG_NU,
        "cycle": "v",
        "omega": MG_OMEGA,
        "coarse_mode": "exact",
        "coarse_steps": MG_COARSE_STEPS,
    }
    append_mg_rows(
        rows,
        "MG(v,w=1.25,coarse=exact)",
        "v",
        "1.25",
        "3",
        MG_MAX_ITER,
        TOL,
        mg_v_warmup_kwargs,
        mg_v_kwargs,
    )

    mg_v_warmup_kwargs_sor = dict(mg_v_warmup_kwargs, coarse_mode="sor")
    mg_v_kwargs_sor = dict(mg_v_kwargs, coarse_mode="sor")
    append_mg_rows(
        rows,
        "MG(v,w=1.25,coarse=sor)",
        "v",
        "1.25",
        "3",
        MG_MAX_ITER,
        TOL,
        mg_v_warmup_kwargs_sor,
        mg_v_kwargs_sor,
    )

    mg_w_warmup_kwargs = {
        "tol": TOL,
        "max_iter": WARMUP_MAX_ITER,
        "nu": MG_NU,
        "cycle": "w",
        "omega": MG_OMEGA,
        "coarse_mode": "exact",
        "coarse_steps": MG_COARSE_STEPS,
    }
    mg_w_kwargs = {
        "tol": TOL,
        "max_iter": MG_MAX_ITER,
        "nu": MG_NU,
        "cycle": "w",
        "omega": MG_OMEGA,
        "coarse_mode": "exact",
        "coarse_steps": MG_COARSE_STEPS,
    }
    append_mg_rows(
        rows,
        "MG(w,w=1.25,coarse=exact)",
        "w",
        "1.25",
        "3",
        MG_MAX_ITER,
        TOL,
        mg_w_warmup_kwargs,
        mg_w_kwargs,
    )

    mg_w_warmup_kwargs_sor = dict(mg_w_warmup_kwargs, coarse_mode="sor")
    mg_w_kwargs_sor = dict(mg_w_kwargs, coarse_mode="sor")
    append_mg_rows(
        rows,
        "MG(w,w=1.25,coarse=sor)",
        "w",
        "1.25",
        "3",
        MG_MAX_ITER,
        TOL,
        mg_w_warmup_kwargs_sor,
        mg_w_kwargs_sor,
    )

    return rows


def run_suite(suite: str) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []

    if suite in {SUITE_SOLVER_COMPARISON, "all"}:
        rows.extend(make_solver_comparison_rows())

    if suite in {SUITE_MG_COMPARE, "all"}:
        rows.extend(make_mg_compare_rows())

    return rows


def _resolve_output_paths(output_base: Path) -> tuple[Path, Path]:
    base = output_base
    if base.suffix == ".csv":
        base = base.with_suffix("")
    return base.with_suffix(".csv"), Path(f"{base}_all.csv")


def write_csv_header(writer: csv.writer) -> None:
    writer.writerow(
        [
            "suite",
            "backend",
            "dtype",
            "case",
            "solver",
            "grid_size",
            "max_iter",
            "tol",
            "warmup_runs",
            "timed_runs",
            "cycle",
            "omega",
            "nu",
            "iterations",
            "residual_l2",
            "error_l2",
            "error_linf",
            "mean_time_ms",
            "std_time_ms",
        ]
    )


def write_csv_row(writer: csv.writer, row: BenchmarkRow) -> None:
    writer.writerow(
        [
            row.suite,
            row.backend,
            row.dtype,
            row.case_name,
            row.solver,
            row.grid_size,
            row.max_iter,
            f"{row.tol:.6e}",
            row.warmup_runs,
            row.timed_runs,
            row.cycle,
            row.omega,
            row.nu,
            row.iterations,
            f"{row.residual_l2:.6e}",
            f"{row.error_l2:.6e}",
            f"{row.error_linf:.6e}",
            f"{row.mean_time_ms:.6f}",
            f"{row.std_time_ms:.6f}",
        ]
    )


def write_csv(stream, rows: list[BenchmarkRow]) -> None:
    writer = csv.writer(stream)
    write_csv_header(writer)
    for row in rows:
        write_csv_row(writer, row)


def write_simple_csv_header(writer: csv.writer) -> None:
    writer.writerow(["solver", "grid_size", "iterations", "mean_time_ms", "std_time_ms"])


def write_simple_csv_row(writer: csv.writer, row: BenchmarkRow) -> None:
    writer.writerow(
        [
            row.solver,
            row.grid_size,
            row.iterations,
            f"{row.mean_time_ms:.6f}",
            f"{row.std_time_ms:.6f}",
        ]
    )


def write_simple_csv(stream, rows: list[BenchmarkRow]) -> None:
    writer = csv.writer(stream)
    write_simple_csv_header(writer)
    for row in rows:
        write_simple_csv_row(writer, row)


def main() -> int:
    args = parse_args()
    print(
        f"Running benchmark suite '{args.suite}' on backend '{BACKEND_NAME}'",
        file=sys.stderr,
    )
    rows = run_suite(args.suite)

    if args.output is None:
        write_csv(sys.stdout, rows)
        return 0

    summary_path, full_path = _resolve_output_paths(args.output)
    full_path.parent.mkdir(parents=True, exist_ok=True)

    with full_path.open("w", newline="") as full_file:
        write_csv(full_file, rows)
    with summary_path.open("w", newline="") as summary_file:
        write_simple_csv(summary_file, rows)

    print(f"Wrote full CSV to {full_path}", file=sys.stderr)
    print(f"Wrote summary CSV to {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
