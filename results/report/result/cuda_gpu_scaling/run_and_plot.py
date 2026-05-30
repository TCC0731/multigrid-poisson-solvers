from __future__ import annotations

"""Generate CUDA GPU scaling tables and plots.

This report keeps the execution path through
``results/report/result/_poisson_wrapper.py`` so that all solver runs are cached
and reproducible. The report compares three timing views for a single
representative MG configuration:

* CPU baseline: OpenMP with 1 thread
* OpenMP: the best thread count found in the sweep
* CUDA: the GPU backend

That makes it easy to answer the report questions:

* Does CUDA scale well as the grid size grows?
* When does GPU acceleration become worthwhile?
* How much faster is CUDA than the CPU and the best OpenMP result?
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR.parent
if str(RESULT_ROOT) not in sys.path:
    sys.path.insert(0, str(RESULT_ROOT))

from _poisson_wrapper import run_or_load
from _report_common import csv_float, load_pyplot, normalize_choices, positive_float, positive_int, write_rows_csv
from _scaling_common import DEFAULT_GRID_SIZES_BY_DIM, cells_for_grid_size, group_rows_by_dimension, unique_positive_ints


DEFAULT_CASES = ("sine", "cosine")
DEFAULT_DIMS = (2, 3)
DEFAULT_THREADS = (1, 2, 4, 6, 8, 12, 16)
DEFAULT_BACKEND_CUDA = "cuda"
DEFAULT_BACKEND_OMP = "omp"
DEFAULT_DTYPE = "double"
DEFAULT_OMEGA = 1.25
DEFAULT_NU = 3
DEFAULT_TOL = 1e-9
DEFAULT_MAX_ITER = 150
DEFAULT_REPEAT_RUNS = 5
DEFAULT_CYCLE = "v"
DEFAULT_MG_COARSE = "exact"
CSV_COLUMNS_RAW = (
    "dimension",
    "case",
    "mode",
    "mode_key",
    "cycle",
    "mg_coarse",
    "omega",
    "nu",
    "solver",
    "backend",
    "dtype",
    "omp_num_threads",
    "grid_size",
    "cells",
    "iterations",
    "residual_l2",
    "error_l2",
    "error_linf",
    "time_ms",
    "time_s",
    "time_per_cell_us",
    "throughput_cells_per_s",
    "tol",
    "max_iter",
    "repeat_runs",
    "converged",
    "source",
)
CSV_COLUMNS_SUMMARY = (
    "dimension",
    "case",
    "mode",
    "mode_key",
    "cycle",
    "mg_coarse",
    "omega",
    "nu",
    "solver",
    "grid_size",
    "cells",
    "cpu_backend",
    "cpu_omp_num_threads",
    "cpu_iterations",
    "cpu_residual_l2",
    "cpu_error_l2",
    "cpu_error_linf",
    "cpu_time_ms",
    "cpu_time_s",
    "cpu_time_per_cell_us",
    "cpu_throughput_cells_per_s",
    "cpu_converged",
    "omp_best_backend",
    "omp_best_omp_num_threads",
    "omp_best_iterations",
    "omp_best_residual_l2",
    "omp_best_error_l2",
    "omp_best_error_linf",
    "omp_best_time_ms",
    "omp_best_time_s",
    "omp_best_time_per_cell_us",
    "omp_best_throughput_cells_per_s",
    "omp_best_converged",
    "cuda_backend",
    "cuda_iterations",
    "cuda_residual_l2",
    "cuda_error_l2",
    "cuda_error_linf",
    "cuda_time_ms",
    "cuda_time_s",
    "cuda_time_per_cell_us",
    "cuda_throughput_cells_per_s",
    "cuda_converged",
    "omp_speedup_vs_cpu",
    "cuda_speedup_vs_cpu",
    "cuda_speedup_vs_omp_best",
    "tol",
    "max_iter",
    "repeat_runs",
    "source",
)


def _grid_sizes_for_dim(dim: int, grid_sizes_2d: Sequence[int], grid_sizes_3d: Sequence[int]) -> tuple[int, ...]:
    if dim == 2:
        return tuple(grid_sizes_2d)
    if dim == 3:
        return tuple(grid_sizes_3d)
    raise ValueError("dim must be 2 or 3")


def _raw_row_key(row: Mapping[str, object]) -> tuple[object, ...]:
    thread_key = int(row["omp_num_threads"]) if row["omp_num_threads"] is not None else -1
    return (
        int(row["dimension"]),
        str(row["case"]),
        int(row["grid_size"]),
        str(row["backend"]),
        thread_key,
    )


def _summary_row_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        int(row["dimension"]),
        str(row["case"]),
        int(row["grid_size"]),
    )


def _build_raw_row(
    *,
    dim: int,
    case: str,
    grid_size: int,
    backend: str,
    omp_num_threads: int | None,
    tol: float,
    max_iter: int,
    repeat_runs: int,
    output_dir: Path,
) -> dict[str, object]:
    kwargs = {
        "backend": backend,
        "dim": dim,
        "dtype": DEFAULT_DTYPE,
        "solver": "mg",
        "case": case,
        "grid_size": grid_size,
        "tol": tol,
        "max_iter": max_iter,
        "repeat_runs": repeat_runs,
        "cycle": DEFAULT_CYCLE,
        "nu": DEFAULT_NU,
        "omega": DEFAULT_OMEGA,
        "mg_coarse": DEFAULT_MG_COARSE,
        "cache_csv": output_dir / "wrapper_cache.csv",
    }
    if backend == DEFAULT_BACKEND_OMP:
        kwargs["omp_num_threads"] = omp_num_threads

    result = run_or_load(**kwargs)

    cells = cells_for_grid_size(dim, grid_size)
    time_s = float(result.time_s)
    time_per_cell_us = (time_s / cells) * 1e6
    throughput_cells_per_s = cells / time_s if time_s > 0.0 else 0.0

    return {
        "dimension": dim,
        "case": case,
        "mode": "MG V exact",
        "mode_key": "mg_v_exact",
        "cycle": DEFAULT_CYCLE,
        "mg_coarse": DEFAULT_MG_COARSE,
        "omega": DEFAULT_OMEGA,
        "nu": DEFAULT_NU,
        "solver": result.solver,
        "backend": result.backend,
        "dtype": result.dtype,
        "omp_num_threads": omp_num_threads,
        "grid_size": result.grid_size,
        "cells": cells,
        "iterations": result.iterations,
        "residual_l2": result.residual_l2,
        "error_l2": result.error_l2,
        "error_linf": result.error_linf,
        "time_ms": result.time_ms,
        "time_s": time_s,
        "time_per_cell_us": time_per_cell_us,
        "throughput_cells_per_s": throughput_cells_per_s,
        "tol": tol,
        "max_iter": max_iter,
        "repeat_runs": repeat_runs,
        "converged": bool(result.residual_l2 <= tol),
        "source": "wrapper",
    }


def collect_raw_rows_from_wrapper(
    *,
    dims: Iterable[int],
    cases: Iterable[str],
    grid_sizes_2d: Sequence[int],
    grid_sizes_3d: Sequence[int],
    threads: Iterable[int],
    tol: float,
    max_iter: int,
    repeat_runs: int,
    output_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dim in dims:
        grid_sizes = _grid_sizes_for_dim(dim, grid_sizes_2d, grid_sizes_3d)
        for case in cases:
            for grid_size in grid_sizes:
                for thread_count in threads:
                    rows.append(
                        _build_raw_row(
                            dim=dim,
                            case=case,
                            grid_size=grid_size,
                            backend=DEFAULT_BACKEND_OMP,
                            omp_num_threads=thread_count,
                            tol=tol,
                            max_iter=max_iter,
                            repeat_runs=repeat_runs,
                            output_dir=output_dir,
                        )
                    )
                rows.append(
                    _build_raw_row(
                        dim=dim,
                        case=case,
                        grid_size=grid_size,
                        backend=DEFAULT_BACKEND_CUDA,
                        omp_num_threads=None,
                        tol=tol,
                        max_iter=max_iter,
                        repeat_runs=repeat_runs,
                        output_dir=output_dir,
                    )
                )
    return rows


def summarize_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["dimension"]), str(row["case"]), int(row["grid_size"]))
        grouped.setdefault(key, []).append(dict(row))

    summary_rows: list[dict[str, object]] = []
    for (dim, case, grid_size), group_rows in sorted(grouped.items()):
        omp_rows = [row for row in group_rows if str(row["backend"]) == DEFAULT_BACKEND_OMP]
        cuda_rows = [row for row in group_rows if str(row["backend"]) == DEFAULT_BACKEND_CUDA]
        if not omp_rows or not cuda_rows:
            raise RuntimeError(f"Missing OMP or CUDA rows for {dim}D {case}, grid={grid_size}")

        cpu_row = next((row for row in omp_rows if int(row["omp_num_threads"]) == 1), None)
        if cpu_row is None:
            raise RuntimeError(f"Missing 1-thread OMP baseline for {dim}D {case}, grid={grid_size}")

        omp_best_row = min(omp_rows, key=lambda row: (float(row["time_s"]), int(row["omp_num_threads"])))
        cuda_row = cuda_rows[0]

        cells = int(cpu_row["cells"])
        cpu_time_s = float(cpu_row["time_s"])
        omp_best_time_s = float(omp_best_row["time_s"])
        cuda_time_s = float(cuda_row["time_s"])

        summary_rows.append(
            {
                "dimension": dim,
                "case": case,
                "mode": "MG V exact",
                "mode_key": "mg_v_exact",
                "cycle": DEFAULT_CYCLE,
                "mg_coarse": DEFAULT_MG_COARSE,
                "omega": DEFAULT_OMEGA,
                "nu": DEFAULT_NU,
                "solver": cpu_row["solver"],
                "grid_size": grid_size,
                "cells": cells,
                "cpu_backend": cpu_row["backend"],
                "cpu_omp_num_threads": int(cpu_row["omp_num_threads"]),
                "cpu_iterations": int(cpu_row["iterations"]),
                "cpu_residual_l2": float(cpu_row["residual_l2"]),
                "cpu_error_l2": float(cpu_row["error_l2"]),
                "cpu_error_linf": float(cpu_row["error_linf"]),
                "cpu_time_ms": float(cpu_row["time_ms"]),
                "cpu_time_s": cpu_time_s,
                "cpu_time_per_cell_us": float(cpu_row["time_per_cell_us"]),
                "cpu_throughput_cells_per_s": float(cpu_row["throughput_cells_per_s"]),
                "cpu_converged": bool(cpu_row["converged"]),
                "omp_best_backend": omp_best_row["backend"],
                "omp_best_omp_num_threads": int(omp_best_row["omp_num_threads"]),
                "omp_best_iterations": int(omp_best_row["iterations"]),
                "omp_best_residual_l2": float(omp_best_row["residual_l2"]),
                "omp_best_error_l2": float(omp_best_row["error_l2"]),
                "omp_best_error_linf": float(omp_best_row["error_linf"]),
                "omp_best_time_ms": float(omp_best_row["time_ms"]),
                "omp_best_time_s": omp_best_time_s,
                "omp_best_time_per_cell_us": float(omp_best_row["time_per_cell_us"]),
                "omp_best_throughput_cells_per_s": float(omp_best_row["throughput_cells_per_s"]),
                "omp_best_converged": bool(omp_best_row["converged"]),
                "cuda_backend": cuda_row["backend"],
                "cuda_iterations": int(cuda_row["iterations"]),
                "cuda_residual_l2": float(cuda_row["residual_l2"]),
                "cuda_error_l2": float(cuda_row["error_l2"]),
                "cuda_error_linf": float(cuda_row["error_linf"]),
                "cuda_time_ms": float(cuda_row["time_ms"]),
                "cuda_time_s": cuda_time_s,
                "cuda_time_per_cell_us": float(cuda_row["time_per_cell_us"]),
                "cuda_throughput_cells_per_s": float(cuda_row["throughput_cells_per_s"]),
                "cuda_converged": bool(cuda_row["converged"]),
                "omp_speedup_vs_cpu": cpu_time_s / omp_best_time_s if omp_best_time_s > 0.0 else 0.0,
                "cuda_speedup_vs_cpu": cpu_time_s / cuda_time_s if cuda_time_s > 0.0 else 0.0,
                "cuda_speedup_vs_omp_best": omp_best_time_s / cuda_time_s if cuda_time_s > 0.0 else 0.0,
                "tol": float(cpu_row["tol"]),
                "max_iter": int(cpu_row["max_iter"]),
                "repeat_runs": int(cpu_row["repeat_runs"]),
                "source": "derived",
            }
        )

    return summary_rows


def _format_raw_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    ordered_rows = sorted(rows, key=_raw_row_key)
    formatted: list[dict[str, object]] = []
    for row in ordered_rows:
        formatted.append(
            {
                "dimension": int(row["dimension"]),
                "case": row["case"],
                "mode": row["mode"],
                "mode_key": row["mode_key"],
                "cycle": row["cycle"],
                "mg_coarse": row["mg_coarse"],
                "omega": f"{float(row['omega']):.2f}",
                "nu": int(row["nu"]),
                "solver": row["solver"],
                "backend": row["backend"],
                "dtype": row["dtype"],
                "omp_num_threads": "" if row["omp_num_threads"] is None else int(row["omp_num_threads"]),
                "grid_size": int(row["grid_size"]),
                "cells": int(row["cells"]),
                "iterations": int(row["iterations"]),
                "residual_l2": csv_float(float(row["residual_l2"])),
                "error_l2": csv_float(float(row["error_l2"])),
                "error_linf": csv_float(float(row["error_linf"])),
                "time_ms": csv_float(float(row["time_ms"])),
                "time_s": csv_float(float(row["time_s"])),
                "time_per_cell_us": csv_float(float(row["time_per_cell_us"])),
                "throughput_cells_per_s": csv_float(float(row["throughput_cells_per_s"])),
                "tol": csv_float(float(row["tol"])),
                "max_iter": int(row["max_iter"]),
                "repeat_runs": int(row["repeat_runs"]),
                "converged": str(bool(row["converged"])).lower(),
                "source": row["source"],
            }
        )
    return formatted


def _format_summary_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    ordered_rows = sorted(rows, key=_summary_row_key)
    formatted: list[dict[str, object]] = []
    for row in ordered_rows:
        formatted.append(
            {
                "dimension": int(row["dimension"]),
                "case": row["case"],
                "mode": row["mode"],
                "mode_key": row["mode_key"],
                "cycle": row["cycle"],
                "mg_coarse": row["mg_coarse"],
                "omega": f"{float(row['omega']):.2f}",
                "nu": int(row["nu"]),
                "solver": row["solver"],
                "grid_size": int(row["grid_size"]),
                "cells": int(row["cells"]),
                "cpu_backend": row["cpu_backend"],
                "cpu_omp_num_threads": int(row["cpu_omp_num_threads"]),
                "cpu_iterations": int(row["cpu_iterations"]),
                "cpu_residual_l2": csv_float(float(row["cpu_residual_l2"])),
                "cpu_error_l2": csv_float(float(row["cpu_error_l2"])),
                "cpu_error_linf": csv_float(float(row["cpu_error_linf"])),
                "cpu_time_ms": csv_float(float(row["cpu_time_ms"])),
                "cpu_time_s": csv_float(float(row["cpu_time_s"])),
                "cpu_time_per_cell_us": csv_float(float(row["cpu_time_per_cell_us"])),
                "cpu_throughput_cells_per_s": csv_float(float(row["cpu_throughput_cells_per_s"])),
                "cpu_converged": str(bool(row["cpu_converged"])).lower(),
                "omp_best_backend": row["omp_best_backend"],
                "omp_best_omp_num_threads": int(row["omp_best_omp_num_threads"]),
                "omp_best_iterations": int(row["omp_best_iterations"]),
                "omp_best_residual_l2": csv_float(float(row["omp_best_residual_l2"])),
                "omp_best_error_l2": csv_float(float(row["omp_best_error_l2"])),
                "omp_best_error_linf": csv_float(float(row["omp_best_error_linf"])),
                "omp_best_time_ms": csv_float(float(row["omp_best_time_ms"])),
                "omp_best_time_s": csv_float(float(row["omp_best_time_s"])),
                "omp_best_time_per_cell_us": csv_float(float(row["omp_best_time_per_cell_us"])),
                "omp_best_throughput_cells_per_s": csv_float(float(row["omp_best_throughput_cells_per_s"])),
                "omp_best_converged": str(bool(row["omp_best_converged"])).lower(),
                "cuda_backend": row["cuda_backend"],
                "cuda_iterations": int(row["cuda_iterations"]),
                "cuda_residual_l2": csv_float(float(row["cuda_residual_l2"])),
                "cuda_error_l2": csv_float(float(row["cuda_error_l2"])),
                "cuda_error_linf": csv_float(float(row["cuda_error_linf"])),
                "cuda_time_ms": csv_float(float(row["cuda_time_ms"])),
                "cuda_time_s": csv_float(float(row["cuda_time_s"])),
                "cuda_time_per_cell_us": csv_float(float(row["cuda_time_per_cell_us"])),
                "cuda_throughput_cells_per_s": csv_float(float(row["cuda_throughput_cells_per_s"])),
                "cuda_converged": str(bool(row["cuda_converged"])).lower(),
                "omp_speedup_vs_cpu": csv_float(float(row["omp_speedup_vs_cpu"])),
                "cuda_speedup_vs_cpu": csv_float(float(row["cuda_speedup_vs_cpu"])),
                "cuda_speedup_vs_omp_best": csv_float(float(row["cuda_speedup_vs_omp_best"])),
                "tol": csv_float(float(row["tol"])),
                "max_iter": int(row["max_iter"]),
                "repeat_runs": int(row["repeat_runs"]),
                "source": row["source"],
            }
        )
    return formatted


def plot_case(
    *,
    dim: int,
    case: str,
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    plt = load_pyplot()

    ordered_rows = sorted(rows, key=lambda row: int(row["cells"]))
    cells = [int(row["cells"]) for row in ordered_rows]
    speedup_cpu = [float(row["cuda_speedup_vs_cpu"]) for row in ordered_rows]
    speedup_omp = [float(row["cuda_speedup_vs_omp_best"]) for row in ordered_rows]
    speedup_omp_vs_cpu = [float(row["omp_speedup_vs_cpu"]) for row in ordered_rows]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    panel_specs = (
        (
            "Runtime (s)",
            "runtime_s",
            {
                "cpu": ("CPU (OMP 1t)", "#7f7f7f", "o", "-"),
                "omp": ("OpenMP best", "#1f77b4", "s", "--"),
                "cuda": ("CUDA", "#d62728", "^", "-"),
            },
        ),
        (
            "Speedup",
            "speedup",
            {
                "cpu": ("OpenMP best / CPU", "#1f77b4", "s", "--"),
                "omp": ("CUDA / OMP best", "#d62728", "^", "-"),
                "cuda": ("CUDA / CPU", "#2ca02c", "D", ":"),
            },
        ),
        (
            "Throughput (cells/s)",
            "throughput",
            {
                "cpu": ("CPU (OMP 1t)", "#7f7f7f", "o", "-"),
                "omp": ("OpenMP best", "#1f77b4", "s", "--"),
                "cuda": ("CUDA", "#d62728", "^", "-"),
            },
        ),
    )

    for ax, (ylabel, panel_key, style_map) in zip(axes, panel_specs):
        if panel_key == "runtime_s":
            ax.loglog(
                cells,
                [float(row["cpu_time_s"]) for row in ordered_rows],
                color=style_map["cpu"][1],
                marker=style_map["cpu"][2],
                linestyle=style_map["cpu"][3],
                linewidth=1.8,
                label=style_map["cpu"][0],
            )
            ax.loglog(
                cells,
                [float(row["omp_best_time_s"]) for row in ordered_rows],
                color=style_map["omp"][1],
                marker=style_map["omp"][2],
                linestyle=style_map["omp"][3],
                linewidth=1.8,
                label=style_map["omp"][0],
            )
            ax.loglog(
                cells,
                [float(row["cuda_time_s"]) for row in ordered_rows],
                color=style_map["cuda"][1],
                marker=style_map["cuda"][2],
                linestyle=style_map["cuda"][3],
                linewidth=1.8,
                label=style_map["cuda"][0],
            )
            ax.set_yscale("log")
        elif panel_key == "speedup":
            ax.semilogx(
                cells,
                speedup_omp_vs_cpu,
                color=style_map["cpu"][1],
                marker=style_map["cpu"][2],
                linestyle=style_map["cpu"][3],
                linewidth=1.8,
                label=style_map["cpu"][0],
            )
            ax.semilogx(
                cells,
                speedup_omp,
                color=style_map["omp"][1],
                marker=style_map["omp"][2],
                linestyle=style_map["omp"][3],
                linewidth=1.8,
                label=style_map["omp"][0],
            )
            ax.semilogx(
                cells,
                speedup_cpu,
                color=style_map["cuda"][1],
                marker=style_map["cuda"][2],
                linestyle=style_map["cuda"][3],
                linewidth=1.8,
                label=style_map["cuda"][0],
            )
        else:
            ax.loglog(
                cells,
                [float(row["cpu_throughput_cells_per_s"]) for row in ordered_rows],
                color=style_map["cpu"][1],
                marker=style_map["cpu"][2],
                linestyle=style_map["cpu"][3],
                linewidth=1.8,
                label=style_map["cpu"][0],
            )
            ax.loglog(
                cells,
                [float(row["omp_best_throughput_cells_per_s"]) for row in ordered_rows],
                color=style_map["omp"][1],
                marker=style_map["omp"][2],
                linestyle=style_map["omp"][3],
                linewidth=1.8,
                label=style_map["omp"][0],
            )
            ax.loglog(
                cells,
                [float(row["cuda_throughput_cells_per_s"]) for row in ordered_rows],
                color=style_map["cuda"][1],
                marker=style_map["cuda"][2],
                linestyle=style_map["cuda"][3],
                linewidth=1.8,
                label=style_map["cuda"][0],
            )
            ax.set_yscale("log")

        ax.set_xlabel("cells")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", frameon=True, framealpha=0.88, facecolor="white", edgecolor="0.85", fontsize=9)
        ax.grid(True, which="both", linestyle="--", alpha=0.45)

    fig.suptitle(
        (
            f"CUDA GPU scaling - {dim}D {case}\n"
            f"MG V exact, CPU=OMP 1 thread, OpenMP=best thread, CUDA"
        ),
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build CUDA GPU scaling tables and plots.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR, help="Directory for CSV and plot outputs.")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES), help="Cases to include.")
    parser.add_argument("--dims", nargs="+", type=int, default=list(DEFAULT_DIMS), help="Dimensions to include.")
    parser.add_argument("--threads", nargs="+", type=int, default=list(DEFAULT_THREADS), help="OMP thread counts.")
    parser.add_argument(
        "--grid-2d",
        nargs="+",
        type=int,
        default=list(DEFAULT_GRID_SIZES_BY_DIM[2]),
        help="2D grid sizes to sweep.",
    )
    parser.add_argument(
        "--grid-3d",
        nargs="+",
        type=int,
        default=list(DEFAULT_GRID_SIZES_BY_DIM[3]),
        help="3D grid sizes to sweep.",
    )
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL, help="Convergence tolerance.")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help="Maximum iterations.")
    parser.add_argument(
        "--repeat-runs",
        type=int,
        default=DEFAULT_REPEAT_RUNS,
        help="Repeat runs passed to the native solver.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = tuple(normalize_choices(args.cases, valid=DEFAULT_CASES, name="case"))
    dims = unique_positive_ints(args.dims, name="dimension")
    dims = tuple(normalize_choices(dims, valid=DEFAULT_DIMS, name="dimension"))
    threads = unique_positive_ints((1, *args.threads), name="thread count")
    grid_sizes_2d = unique_positive_ints(args.grid_2d, name="grid_2d")
    grid_sizes_3d = unique_positive_ints(args.grid_3d, name="grid_3d")
    tol = positive_float(args.tol, "tol")
    max_iter = positive_int(args.max_iter, "max_iter")
    repeat_runs = positive_int(args.repeat_runs, "repeat_runs")

    raw_rows = collect_raw_rows_from_wrapper(
        dims=dims,
        cases=cases,
        grid_sizes_2d=grid_sizes_2d,
        grid_sizes_3d=grid_sizes_3d,
        threads=threads,
        tol=tol,
        max_iter=max_iter,
        repeat_runs=repeat_runs,
        output_dir=output_dir,
    )

    if not raw_rows:
        raise RuntimeError("No benchmark rows were collected.")

    raw_rows = sorted(raw_rows, key=_raw_row_key)
    summary_rows = summarize_rows(raw_rows)
    summary_rows = sorted(summary_rows, key=_summary_row_key)

    write_rows_csv(output_dir / "results_raw_all.csv", _format_raw_rows(raw_rows), columns=CSV_COLUMNS_RAW)
    write_rows_csv(output_dir / "results_all.csv", _format_summary_rows(summary_rows), columns=CSV_COLUMNS_SUMMARY)

    raw_by_dim = group_rows_by_dimension(raw_rows)
    for dim in sorted(raw_by_dim):
        dim_rows = sorted(raw_by_dim[dim], key=_raw_row_key)
        write_rows_csv(output_dir / f"results_raw_{dim}d.csv", _format_raw_rows(dim_rows), columns=CSV_COLUMNS_RAW)

    summary_by_dim = group_rows_by_dimension(summary_rows)
    for dim in sorted(summary_by_dim):
        dim_rows = sorted(summary_by_dim[dim], key=_summary_row_key)
        write_rows_csv(output_dir / f"results_{dim}d.csv", _format_summary_rows(dim_rows), columns=CSV_COLUMNS_SUMMARY)

    for dim in dims:
        for case in cases:
            case_rows = [row for row in summary_rows if int(row["dimension"]) == dim and str(row["case"]) == case]
            if not case_rows:
                continue
            plot_case(
                dim=dim,
                case=case,
                rows=case_rows,
                output_path=output_dir / f"plots_{dim}d_{case}.png",
            )

    print(f"Wrote {output_dir / 'results_all.csv'}")
    print(f"Wrote {output_dir / 'results_raw_all.csv'}")
    print(f"Cases: {', '.join(cases)}")
    print(f"Dimensions: {', '.join(str(dim) for dim in dims)}")
    print(f"Threads: {', '.join(str(thread) for thread in threads)}")
    print(f"Rows: {len(summary_rows)} summary / {len(raw_rows)} raw")
    print(f"Non-converged raw rows: {sum(1 for row in raw_rows if not row['converged'])}")


if __name__ == "__main__":
    main()
