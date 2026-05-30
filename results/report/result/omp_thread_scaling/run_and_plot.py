from __future__ import annotations

"""Generate OpenMP thread-scaling tables and plots for the Poisson solvers.

The script uses ``results/report/result/_poisson_wrapper.py`` as the execution
path so the solver results are cached and reproducible. By default it benchmarks
the following modes on the requested grid sizes:

* SOR
* MG V with exact coarse solve
* MG V with SOR coarse solve
* MG W with exact coarse solve
* MG W with SOR coarse solve

The default grid sizes match the request in the task:

* 2D: 4095
* 3D: 383

Output files are written next to this script:

* ``results_all.csv`` for the combined table
* ``results_2d.csv`` / ``results_3d.csv`` for per-dimension tables
* ``plots_2d.png`` / ``plots_3d.png`` for the thread-scaling figures
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR.parent
if str(RESULT_ROOT) not in sys.path:
    sys.path.insert(0, str(RESULT_ROOT))

from _poisson_wrapper import run_or_load
from _report_common import csv_float, load_pyplot, positive_float, positive_int, write_rows_csv


DEFAULT_CASE = "sine"
DEFAULT_DIMS = (2, 3)
DEFAULT_THREADS = (1, 2, 4, 6, 8, 12, 16)
DEFAULT_GRID_SIZES = {2: 4095, 3: 383}
DEFAULT_BACKEND = "omp"
DEFAULT_DTYPE = "double"
DEFAULT_TOL = 1e-9
DEFAULT_REPEAT_RUNS = 10
DEFAULT_SOR_MAX_ITER = 10_000
DEFAULT_MG_MAX_ITER = 150
DEFAULT_OMEGA = 1.25
DEFAULT_NU = 3
DEFAULT_COARSE_STEPS = 16
CSV_COLUMNS = (
    "dimension",
    "case",
    "grid_size",
    "mode",
    "mode_key",
    "solver",
    "cycle",
    "mg_coarse",
    "omega",
    "nu",
    "backend",
    "dtype",
    "omp_num_threads",
    "iterations",
    "residual_l2",
    "error_l2",
    "error_linf",
    "time_ms",
    "time_s",
    "speedup",
    "efficiency",
    "tol",
    "max_iter",
    "repeat_runs",
    "converged",
    "source",
)


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
        label="MG V exact",
        mode_key="mg_v_exact",
        solver="mg",
        color="#ff7f0e",
        marker="s",
        linestyle="-",
        cycle="v",
        mg_coarse="exact",
        omega=DEFAULT_OMEGA,
        nu=DEFAULT_NU,
        max_iter=DEFAULT_MG_MAX_ITER,
    ),
    ModeSpec(
        label="MG V SOR",
        mode_key="mg_v_sor",
        solver="mg",
        color="#2ca02c",
        marker="^",
        linestyle="--",
        cycle="v",
        mg_coarse="sor",
        omega=DEFAULT_OMEGA,
        nu=DEFAULT_NU,
        max_iter=DEFAULT_MG_MAX_ITER,
    ),
    ModeSpec(
        label="MG W exact",
        mode_key="mg_w_exact",
        solver="mg",
        color="#d62728",
        marker="D",
        linestyle="-",
        cycle="w",
        mg_coarse="exact",
        omega=DEFAULT_OMEGA,
        nu=DEFAULT_NU,
        max_iter=DEFAULT_MG_MAX_ITER,
    ),
    ModeSpec(
        label="MG W SOR",
        mode_key="mg_w_sor",
        solver="mg",
        color="#9467bd",
        marker="P",
        linestyle="--",
        cycle="w",
        mg_coarse="sor",
        omega=DEFAULT_OMEGA,
        nu=DEFAULT_NU,
        max_iter=DEFAULT_MG_MAX_ITER,
    ),
)
MODE_SPECS_BY_KEY = {spec.mode_key: spec for spec in MODE_SPECS}


def _unique_positive_ints(values: Iterable[object], *, name: str) -> tuple[int, ...]:
    items: list[int] = []
    seen: set[int] = set()
    for value in values:
        item = positive_int(value, name)
        if item in seen:
            continue
        seen.add(item)
        items.append(item)
    if not items:
        raise ValueError(f"at least one {name} is required")
    return tuple(items)


def _grid_size_for_dim(dim: int, grid_size_2d: int, grid_size_3d: int) -> int:
    if dim == 2:
        return grid_size_2d
    if dim == 3:
        return grid_size_3d
    raise ValueError("dim must be 2 or 3")


def _row_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        int(row["dimension"]),
        str(row["case"]),
        int(row["grid_size"]),
        str(row["mode_key"]),
        int(row["omp_num_threads"]),
    )


def _format_optional_int(value: int | None) -> str:
    return "" if value is None else str(int(value))


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else f"{float(value):.6f}"


def _collect_rows_from_wrapper(
    *,
    dims: Iterable[int],
    case: str,
    grid_size_2d: int,
    grid_size_3d: int,
    threads: Iterable[int],
    tol: float,
    repeat_runs: int,
    dtype: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for dim in dims:
        grid_size = _grid_size_for_dim(dim, grid_size_2d, grid_size_3d)
        for thread_count in threads:
            for mode in MODE_SPECS:
                result = run_or_load(
                    backend=DEFAULT_BACKEND,
                    dim=dim,
                    dtype=dtype,
                    solver=mode.solver,
                    case=case,
                    grid_size=grid_size,
                    tol=tol,
                    max_iter=mode.max_iter,
                    repeat_runs=repeat_runs,
                    cycle=mode.cycle or "v",
                    nu=mode.nu or 2,
                    omega=mode.omega if mode.omega is not None else 1.0,
                    mg_coarse=mode.mg_coarse or "exact",
                    coarse_steps=DEFAULT_COARSE_STEPS,
                    omp_num_threads=thread_count,
                )

                rows.append(
                    {
                        "dimension": dim,
                        "case": case,
                        "grid_size": grid_size,
                        "mode": mode.label,
                        "mode_key": mode.mode_key,
                        "solver": result.solver,
                        "cycle": result.cycle or "",
                        "mg_coarse": result.mg_coarse or "",
                        "omega": result.omega if isinstance(result.omega, float) else None,
                        "nu": result.nu,
                        "backend": result.backend,
                        "dtype": result.dtype,
                        "omp_num_threads": result.omp_num_threads,
                        "iterations": result.iterations,
                        "residual_l2": result.residual_l2,
                        "error_l2": result.error_l2,
                        "error_linf": result.error_linf,
                        "time_ms": result.time_ms,
                        "time_s": result.time_s,
                        "tol": tol,
                        "max_iter": mode.max_iter,
                        "repeat_runs": repeat_runs,
                        "converged": bool(result.residual_l2 <= tol),
                        "source": "wrapper",
                    }
                )

    return rows


def _enrich_speedup(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["dimension"]), str(row["mode_key"]))].append(row)

    for group_rows in grouped.values():
        baseline = next((row for row in group_rows if int(row["omp_num_threads"]) == 1), None)
        if baseline is None:
            raise RuntimeError("Every mode needs a 1-thread baseline to compute speedup.")
        baseline_time = float(baseline["time_ms"])
        for row in group_rows:
            current_time = float(row["time_ms"])
            thread_count = int(row["omp_num_threads"])
            row["speedup"] = baseline_time / current_time
            row["efficiency"] = row["speedup"] / thread_count


def _format_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            int(row["dimension"]),
            str(row["mode_key"]),
            int(row["omp_num_threads"]),
        ),
    )

    formatted: list[dict[str, object]] = []
    for row in ordered_rows:
        formatted.append(
            {
                "dimension": int(row["dimension"]),
                "case": row["case"],
                "grid_size": int(row["grid_size"]),
                "mode": row["mode"],
                "mode_key": row["mode_key"],
                "solver": row["solver"],
                "cycle": row["cycle"],
                "mg_coarse": row["mg_coarse"],
                "omega": _format_optional_float(row.get("omega")),
                "nu": _format_optional_int(row.get("nu")),
                "backend": row["backend"],
                "dtype": row["dtype"],
                "omp_num_threads": int(row["omp_num_threads"]),
                "iterations": int(row["iterations"]),
                "residual_l2": csv_float(float(row["residual_l2"])),
                "error_l2": csv_float(float(row["error_l2"])),
                "error_linf": csv_float(float(row["error_linf"])),
                "time_ms": csv_float(float(row["time_ms"])),
                "time_s": csv_float(float(row["time_s"])),
                "speedup": csv_float(float(row["speedup"])),
                "efficiency": csv_float(float(row["efficiency"])),
                "tol": csv_float(float(row["tol"])),
                "max_iter": int(row["max_iter"]),
                "repeat_runs": int(row["repeat_runs"]),
                "converged": str(bool(row["converged"])).lower(),
                "source": row["source"],
            }
        )
    return formatted


def _group_rows(rows: Iterable[Mapping[str, object]]) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["dimension"])].append(dict(row))
    return grouped


def _plot_dimension(
    *,
    dim: int,
    case: str,
    grid_size: int,
    rows: list[dict[str, object]],
    tol: float,
    repeat_runs: int,
    output_path: Path,
) -> None:
    plt = load_pyplot()

    series_by_mode: dict[str, list[dict[str, object]]] = {spec.mode_key: [] for spec in MODE_SPECS}
    for row in rows:
        series_by_mode[str(row["mode_key"])].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    panels = (
        ("time_ms", "Runtime (ms)", "log"),
        ("speedup", "Speedup vs 1 thread", "linear"),
    )

    thread_values = sorted({int(row["omp_num_threads"]) for row in rows})

    for ax, (metric_key, ylabel, scale) in zip(axes, panels):
        for spec in MODE_SPECS:
            series = sorted(series_by_mode[spec.mode_key], key=lambda row: int(row["omp_num_threads"]))
            if not series:
                continue
            x = [int(row["omp_num_threads"]) for row in series]
            y = [float(row[metric_key]) for row in series]
            ax.plot(
                x,
                y,
                marker=spec.marker,
                linestyle=spec.linestyle,
                color=spec.color,
                linewidth=1.9,
                label=spec.label,
            )

        if metric_key == "speedup":
            ideal_x = thread_values
            ideal_y = ideal_x
            ax.plot(
                ideal_x,
                ideal_y,
                color="#666666",
                linestyle=":",
                linewidth=1.5,
                label="Ideal",
            )

        ax.set_xlabel("OMP threads")
        ax.set_ylabel(ylabel)
        ax.set_xticks(thread_values)
        ax.grid(True, linestyle="--", alpha=0.45)
        if scale == "log":
            ax.set_yscale("log")
        if metric_key == "speedup":
            upper = max(max(thread_values), max(float(row["speedup"]) for row in rows))
            ax.set_ylim(0.0, upper * 1.08)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=False,
        borderaxespad=0.0,
    )
    fig.suptitle(
        f"OMP thread scaling - {dim}D {case}, grid={grid_size}\n"
        f"tol={tol:.0e}, repeat_runs={repeat_runs}",
        y=0.92,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build OMP thread-scaling tables and plots.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR, help="Directory for CSV and plot outputs.")
    parser.add_argument("--case", default=DEFAULT_CASE, help="Poisson case to benchmark.")
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        choices=DEFAULT_DIMS,
        default=list(DEFAULT_DIMS),
        help="Dimensions to include.",
    )
    parser.add_argument("--threads", nargs="+", type=int, default=list(DEFAULT_THREADS), help="OMP thread counts.")
    parser.add_argument("--grid-size-2d", type=int, default=DEFAULT_GRID_SIZES[2], help="Grid size for 2D runs.")
    parser.add_argument("--grid-size-3d", type=int, default=DEFAULT_GRID_SIZES[3], help="Grid size for 3D runs.")
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, help="Floating-point dtype.")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL, help="Convergence tolerance.")
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

    case = str(args.case)
    dims = _unique_positive_ints(args.dims, name="dimension")

    threads = _unique_positive_ints(args.threads, name="thread count")
    tol = positive_float(args.tol, "tol")
    repeat_runs = positive_int(args.repeat_runs, "repeat_runs")
    grid_size_2d = positive_int(args.grid_size_2d, "grid_size_2d")
    grid_size_3d = positive_int(args.grid_size_3d, "grid_size_3d")
    dtype = str(args.dtype).strip().lower()
    if dtype != DEFAULT_DTYPE:
        raise ValueError(f"Only dtype={DEFAULT_DTYPE!r} is supported in this report.")

    rows = _collect_rows_from_wrapper(
        dims=dims,
        case=case,
        grid_size_2d=grid_size_2d,
        grid_size_3d=grid_size_3d,
        threads=threads,
        tol=tol,
        repeat_runs=repeat_runs,
        dtype=dtype,
    )

    if not rows:
        raise RuntimeError("No benchmark rows were collected.")

    _enrich_speedup(rows)
    rows = sorted(rows, key=_row_key)

    write_rows_csv(output_dir / "results_all.csv", _format_rows(rows), columns=CSV_COLUMNS)

    grouped = _group_rows(rows)
    for dim in sorted(grouped):
        dim_rows = sorted(grouped[dim], key=_row_key)
        write_rows_csv(output_dir / f"results_{dim}d.csv", _format_rows(dim_rows), columns=CSV_COLUMNS)
        _plot_dimension(
            dim=dim,
            case=case,
            grid_size=_grid_size_for_dim(dim, grid_size_2d, grid_size_3d),
            rows=dim_rows,
            tol=tol,
            repeat_runs=repeat_runs,
            output_path=output_dir / f"plots_{dim}d.png",
        )

    print(f"Wrote {output_dir / 'results_all.csv'}")
    print(f"Cases: {case}")
    print(f"Dimensions: {', '.join(str(dim) for dim in dims)}")
    print(f"Threads: {', '.join(str(thread) for thread in threads)}")
    print(f"Rows: {len(rows)}")
    print(f"Non-converged rows: {sum(1 for row in rows if not row['converged'])}")


if __name__ == "__main__":
    main()
