from __future__ import annotations

"""Generate CUDA MG 2D vs 3D scaling tables and plots.

This report keeps the execution path through
``results/report/result/_poisson_wrapper.py`` so that all solver runs are cached
and reproducible. The default configuration benchmarks a single representative
MG setup:

* CUDA backend
* MG V-cycle
* exact coarse solve
* ``nu = 3``
* ``omega = 1.25``

The output focuses on the question asked in the report notes:

* how runtime grows with total cells
* how runtime per cell changes
* how throughput changes as the grid gets larger
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
from _report_common import csv_float, finalize_figure_header, load_pyplot, normalize_choices, positive_float, positive_int, power_law_fit_curve, write_rows_csv
from _scaling_common import DEFAULT_GRID_SIZES_BY_DIM, cells_for_grid_size, group_rows_by_dimension, unique_positive_ints


DEFAULT_CASES = ("sine", "cosine")
DEFAULT_DIMS = (2, 3)
DEFAULT_BACKEND = "cuda"
DEFAULT_DTYPE = "double"
DEFAULT_OMEGA = 1.25
DEFAULT_NU = 3
DEFAULT_TOL = 1e-9
DEFAULT_MAX_ITER = 150
DEFAULT_REPEAT_RUNS = 5
DEFAULT_CYCLE = "v"
DEFAULT_MG_COARSE = "exact"
CSV_COLUMNS = (
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


def _grid_sizes_for_dim(dim: int, grid_sizes_2d: Sequence[int], grid_sizes_3d: Sequence[int]) -> tuple[int, ...]:
    if dim == 2:
        return tuple(grid_sizes_2d)
    if dim == 3:
        return tuple(grid_sizes_3d)
    raise ValueError("dim must be 2 or 3")


def _row_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        int(row["dimension"]),
        str(row["case"]),
        int(row["grid_size"]),
    )


def _row_from_wrapper(
    *,
    dim: int,
    case: str,
    grid_size: int,
    tol: float,
    max_iter: int,
    repeat_runs: int,
    output_dir: Path,
) -> dict[str, object]:
    result = run_or_load(
        backend=DEFAULT_BACKEND,
        dim=dim,
        dtype=DEFAULT_DTYPE,
        solver="mg",
        case=case,
        grid_size=grid_size,
        tol=tol,
        max_iter=max_iter,
        repeat_runs=repeat_runs,
        cycle=DEFAULT_CYCLE,
        nu=DEFAULT_NU,
        omega=DEFAULT_OMEGA,
        mg_coarse=DEFAULT_MG_COARSE,
    )

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


def collect_rows_from_wrapper(
    *,
    dims: Iterable[int],
    cases: Iterable[str],
    grid_sizes_2d: Sequence[int],
    grid_sizes_3d: Sequence[int],
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
                rows.append(
                    _row_from_wrapper(
                        dim=dim,
                        case=case,
                        grid_size=grid_size,
                        tol=tol,
                        max_iter=max_iter,
                        repeat_runs=repeat_runs,
                        output_dir=output_dir,
                    )
                )
    return rows


def _format_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    ordered_rows = sorted(rows, key=_row_key)
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


def plot_case(
    *,
    case: str,
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    plt = load_pyplot()

    series_by_dim: dict[int, list[dict[str, object]]] = {2: [], 3: []}
    for row in rows:
        series_by_dim[int(row["dimension"])].append(row)

    legend_labels: dict[int, str] = {}
    for dim, series in series_by_dim.items():
        ordered = sorted(series, key=lambda row: int(row["cells"]))
        if not ordered:
            continue
        fit = power_law_fit_curve(
            [float(row["cells"]) for row in ordered],
            [float(row["time_s"]) for row in ordered],
        )
        if fit is None:
            legend_labels[dim] = f"{dim}D"
        else:
            _, _, slope = fit
            legend_labels[dim] = f"{dim}D (time ~ cells^{slope:.2f})"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = (
        ("time_s", "Runtime (s)"),
        ("time_per_cell_us", "Runtime per cell (us)"),
        ("throughput_cells_per_s", "Throughput (cells/s)"),
    )

    for ax, (metric_key, ylabel) in zip(axes, panels):
        for dim in DEFAULT_DIMS:
            series = sorted(series_by_dim[dim], key=lambda row: int(row["cells"]))
            if not series:
                continue
            x = [int(row["cells"]) for row in series]
            y = [float(row[metric_key]) for row in series]
            ax.loglog(
                x,
                y,
                marker="o" if dim == 2 else "s",
                linestyle="-" if dim == 2 else "--",
                linewidth=1.9,
                label=legend_labels.get(dim, f"{dim}D"),
            )
        ax.set_title(ylabel)
        ax.set_xlabel("cells")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle="--", alpha=0.45)

    handles, labels = axes[0].get_legend_handles_labels()
    finalize_figure_header(
        fig,
        title=(
            f"CUDA MG 2D vs 3D scaling - {case}\n"
            f"V-cycle, exact coarse solve, nu={DEFAULT_NU}, omega={DEFAULT_OMEGA:.2f}"
        ),
        handles=handles,
        labels=labels,
        ncol=2,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build CUDA MG 2D vs 3D scaling tables and plots.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR, help="Directory for CSV and plot outputs.")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES), help="Cases to include.")
    parser.add_argument("--dims", nargs="+", type=int, default=list(DEFAULT_DIMS), help="Dimensions to include.")
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
    grid_sizes_2d = unique_positive_ints(args.grid_2d, name="grid_2d")
    grid_sizes_3d = unique_positive_ints(args.grid_3d, name="grid_3d")
    tol = positive_float(args.tol, "tol")
    max_iter = positive_int(args.max_iter, "max_iter")
    repeat_runs = positive_int(args.repeat_runs, "repeat_runs")

    rows = collect_rows_from_wrapper(
        dims=dims,
        cases=cases,
        grid_sizes_2d=grid_sizes_2d,
        grid_sizes_3d=grid_sizes_3d,
        tol=tol,
        max_iter=max_iter,
        repeat_runs=repeat_runs,
        output_dir=output_dir,
    )

    if not rows:
        raise RuntimeError("No benchmark rows were collected.")

    rows = sorted(rows, key=_row_key)

    write_rows_csv(output_dir / "results_all.csv", _format_rows(rows), columns=CSV_COLUMNS)

    grouped_by_dim = group_rows_by_dimension(rows)
    for dim in sorted(grouped_by_dim):
        dim_rows = sorted(grouped_by_dim[dim], key=_row_key)
        write_rows_csv(output_dir / f"results_{dim}d.csv", _format_rows(dim_rows), columns=CSV_COLUMNS)

    for case in cases:
        case_rows = [row for row in rows if str(row["case"]) == case]
        case_rows = sorted(case_rows, key=_row_key)
        plot_case(
            case=case,
            rows=case_rows,
            output_path=output_dir / f"plots_{case}.png",
        )

    print(f"Wrote {output_dir / 'results_all.csv'}")
    print(f"Cases: {', '.join(cases)}")
    print(f"Dimensions: {', '.join(str(dim) for dim in dims)}")
    print(f"Rows: {len(rows)}")
    print(f"Non-converged rows: {sum(1 for row in rows if not row['converged'])}")


if __name__ == "__main__":
    main()
