from __future__ import annotations

"""Sweep CUDA MG V/W SOR coarse_steps and write CSV + plots.

This report uses ``results/report/result/_poisson_wrapper.py`` to execute the
native CUDA solver, caches raw wrapper results locally, and then writes
report-friendly CSV summaries and plots for each case.
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

from _poisson_wrapper import resolve_executable, run_or_load
from _report_common import csv_float, load_pyplot, normalize_choices, positive_float, positive_int, write_rows_csv


DEFAULT_BACKEND = "cuda"
DEFAULT_DTYPE = "double"
DEFAULT_CASES = ("sine", "cosine")
DEFAULT_DIMS = (2, 3)
DEFAULT_COARSE_STEPS = (1, 2, 4, 6, 8, 12, 16)
DEFAULT_GRID_SIZES = {2: 4095, 3: 383}
DEFAULT_NU = 3
DEFAULT_OMEGA = 1.25
DEFAULT_TOL = 1e-9
DEFAULT_MAX_ITER = 15
DEFAULT_REPEAT_RUNS = 5

CSV_COLUMNS = (
    "dimension",
    "case",
    "mode",
    "mode_key",
    "cycle",
    "coarse",
    "coarse_steps",
    "omega",
    "solver",
    "backend",
    "dtype",
    "grid_size",
    "iterations",
    "residual_l2",
    "error_l2",
    "error_linf",
    "time_ms",
    "time_s",
    "tol",
    "max_iter",
    "nu",
    "converged",
    "source",
)


@dataclass(frozen=True)
class ModeSpec:
    label: str
    mode_key: str
    cycle: str
    color: str
    marker: str
    linestyle: str


MODE_SPECS = (
    ModeSpec("V-SOR", "v_sor", "v", "#1f77b4", "o", "-"),
    ModeSpec("W-SOR", "w_sor", "w", "#ff7f0e", "s", "--"),
)
MODE_SPECS_BY_KEY = {spec.mode_key: spec for spec in MODE_SPECS}


def _row_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        int(row["dimension"]),
        str(row["case"]),
        str(row["mode_key"]),
        int(row["coarse_steps"]),
    )


def _row_from_wrapper(
    *,
    dim: int,
    case: str,
    coarse_steps: int,
    mode: ModeSpec,
    grid_size: int,
    nu: int,
    omega: float,
    tol: float,
    max_iter: int,
    repeat_runs: int,
) -> dict[str, object]:
    print(
        f"  {dim}D {case} | {mode.label} | coarse_steps={coarse_steps} | grid={grid_size}",
        flush=True,
    )

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
        cycle=mode.cycle,
        nu=nu,
        omega=omega,
        mg_coarse="sor",
        coarse_steps=coarse_steps,
    )

    return {
        "dimension": dim,
        "case": case,
        "mode": mode.label,
        "mode_key": mode.mode_key,
        "cycle": mode.cycle,
        "coarse": "sor",
        "coarse_steps": coarse_steps,
        "omega": omega,
        "solver": result.solver,
        "backend": result.backend,
        "dtype": result.dtype,
        "grid_size": result.grid_size,
        "iterations": result.iterations,
        "residual_l2": result.residual_l2,
        "error_l2": result.error_l2,
        "error_linf": result.error_linf,
        "time_ms": result.time_ms,
        "time_s": result.time_s,
        "tol": tol,
        "max_iter": max_iter,
        "nu": nu,
        "converged": bool(result.residual_l2 <= tol),
        "source": "wrapper",
    }


def collect_rows_from_wrapper(
    *,
    dims: Iterable[int],
    cases: Iterable[str],
    coarse_steps_list: Iterable[int],
    grid_sizes: Mapping[int, int],
    nu: int,
    omega: float,
    tol: float,
    max_iter: int,
    repeat_runs: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dim in dims:
        grid_size = grid_sizes[dim]
        for case in cases:
            for coarse_steps in coarse_steps_list:
                for mode in MODE_SPECS:
                    rows.append(
                        _row_from_wrapper(
                            dim=dim,
                            case=case,
                            coarse_steps=coarse_steps,
                            mode=mode,
                            grid_size=grid_size,
                            nu=nu,
                            omega=omega,
                            tol=tol,
                            max_iter=max_iter,
                            repeat_runs=repeat_runs,
                        )
                    )
    return rows


def _format_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            int(row["dimension"]),
            str(row["case"]),
            str(row["mode_key"]),
            int(row["coarse_steps"]),
        ),
    )

    formatted: list[dict[str, object]] = []
    for row in ordered_rows:
        formatted.append(
            {
                "dimension": int(row["dimension"]),
                "case": row["case"],
                "mode": row["mode"],
                "mode_key": row["mode_key"],
                "cycle": row["cycle"],
                "coarse": row["coarse"],
                "coarse_steps": int(row["coarse_steps"]),
                "omega": f"{float(row['omega']):.2f}",
                "solver": row["solver"],
                "backend": row["backend"],
                "dtype": row["dtype"],
                "grid_size": int(row["grid_size"]),
                "iterations": int(row["iterations"]),
                "residual_l2": csv_float(float(row["residual_l2"])),
                "error_l2": csv_float(float(row["error_l2"])),
                "error_linf": csv_float(float(row["error_linf"])),
                "time_ms": csv_float(float(row["time_ms"])),
                "time_s": csv_float(float(row["time_s"])),
                "tol": csv_float(float(row["tol"])),
                "max_iter": int(row["max_iter"]),
                "nu": int(row["nu"]),
                "converged": str(bool(row["converged"])).lower(),
                "source": row["source"],
            }
        )
    return formatted


def group_rows(rows: Iterable[Mapping[str, object]]) -> dict[tuple[int, str], list[dict[str, object]]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["dimension"]), str(row["case"]))].append(dict(row))
    return grouped


def plot_case(
    *,
    dim: int,
    case: str,
    grid_size: int,
    rows: list[dict[str, object]],
    nu: int,
    omega: float,
    tol: float,
    max_iter: int,
    output_path: Path,
) -> None:
    plt = load_pyplot()

    series_by_label: dict[str, list[dict[str, object]]] = {spec.label: [] for spec in MODE_SPECS}
    for row in rows:
        series_by_label[MODE_SPECS_BY_KEY[str(row["mode_key"])].label].append(row)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = (
        ("error_l2", "L2 error"),
        ("iterations", "Iterations"),
        ("time_s", "Time (s)"),
    )
    ylabels = {
        "error_l2": "L2 error",
        "iterations": "Iterations",
        "time_s": "Time (s)",
    }

    for ax, (metric_key, title) in zip(axes, panels):
        for spec in MODE_SPECS:
            series = sorted(series_by_label[spec.label], key=lambda row: int(row["coarse_steps"]))
            if not series:
                continue
            x = [int(row["coarse_steps"]) for row in series]
            y = [float(row[metric_key]) for row in series]
            ax.semilogy(
                x,
                y,
                marker=spec.marker,
                linestyle=spec.linestyle,
                color=spec.color,
                linewidth=1.9,
                label=spec.label,
            )
        ax.set_title(title)
        ax.set_xlabel("coarse steps")
        ax.set_ylabel(ylabels[metric_key])
        ax.set_xticks(list(DEFAULT_COARSE_STEPS))
        ax.grid(True, which="both", linestyle="--", alpha=0.45)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        frameon=False,
        borderaxespad=0.0,
    )
    fig.suptitle(
        f"CUDA MG coarse_steps - {dim}D {case}\n"
        f"grid={grid_size}, nu={nu}, omega={omega:.2f}, tol={tol:.0e}, max_iter={max_iter}",
        y=0.92,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build CUDA MG coarse_steps tables and plots.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR, help="Directory for CSV and plot outputs.")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES), help="Cases to include.")
    parser.add_argument("--dims", nargs="+", type=int, default=list(DEFAULT_DIMS), help="Dimensions to include.")
    parser.add_argument(
        "--coarse-steps",
        nargs="+",
        type=int,
        default=list(DEFAULT_COARSE_STEPS),
        help="Coarse SOR iteration counts to sweep.",
    )
    parser.add_argument("--grid-2d", type=int, default=DEFAULT_GRID_SIZES[2], help="2D grid size.")
    parser.add_argument("--grid-3d", type=int, default=DEFAULT_GRID_SIZES[3], help="3D grid size.")
    parser.add_argument("--nu", type=int, default=DEFAULT_NU, help="MG nu value.")
    parser.add_argument("--omega", type=float, default=DEFAULT_OMEGA, help="MG omega value.")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL, help="Convergence tolerance.")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help="Maximum iterations.")
    parser.add_argument(
        "--repeat-runs",
        type=int,
        default=DEFAULT_REPEAT_RUNS,
        help="Repeat runs per solver invocation when regenerating from CUDA.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = tuple(normalize_choices(args.cases, valid=DEFAULT_CASES, name="case"))
    dims = tuple(positive_int(dim, "dim") for dim in args.dims)
    dims = tuple(normalize_choices(dims, valid=DEFAULT_DIMS, name="dimension"))
    coarse_steps = tuple(sorted({positive_int(value, "coarse_steps") for value in args.coarse_steps}))
    grid_sizes = {
        2: positive_int(args.grid_2d, "grid_2d"),
        3: positive_int(args.grid_3d, "grid_3d"),
    }
    nu = positive_int(args.nu, "nu")
    omega = positive_float(args.omega, "omega")
    tol = positive_float(args.tol, "tol")
    max_iter = positive_int(args.max_iter, "max_iter")
    repeat_runs = positive_int(args.repeat_runs, "repeat_runs")

    print(f"Using CUDA executable: {resolve_executable(DEFAULT_BACKEND)}")
    print(f"Output directory: {output_dir}")
    print(f"Cases: {', '.join(cases)}")
    print(f"Dimensions: {', '.join(str(dim) for dim in dims)}")
    print(f"Grid sizes: 2D={grid_sizes[2]}, 3D={grid_sizes[3]}")
    print(f"coarse_steps: {', '.join(str(value) for value in coarse_steps)}")
    print(f"nu={nu}, omega={omega:.2f}, tol={tol:.0e}, max_iter={max_iter}, repeat_runs={repeat_runs}")

    rows = collect_rows_from_wrapper(
        dims=dims,
        cases=cases,
        coarse_steps_list=coarse_steps,
        grid_sizes=grid_sizes,
        nu=nu,
        omega=omega,
        tol=tol,
        max_iter=max_iter,
        repeat_runs=repeat_runs,
    )

    if not rows:
        raise RuntimeError("No benchmark rows were produced.")

    rows = sorted(
        rows,
        key=lambda row: (
            int(row["dimension"]),
            str(row["case"]),
            str(row["mode_key"]),
            int(row["coarse_steps"]),
        ),
    )

    write_rows_csv(output_dir / "results_all.csv", _format_rows(rows), columns=CSV_COLUMNS)

    grouped = group_rows(rows)
    for dim, case in sorted(grouped):
        case_rows = sorted(grouped[(dim, case)], key=lambda row: (str(row["mode_key"]), int(row["coarse_steps"])))
        grid_size = grid_sizes[dim]
        write_rows_csv(
            output_dir / f"results_{dim}d_{case}.csv",
            _format_rows(case_rows),
            columns=CSV_COLUMNS,
        )
        plot_case(
            dim=dim,
            case=case,
            grid_size=grid_size,
            rows=case_rows,
            nu=nu,
            omega=omega,
            tol=tol,
            max_iter=max_iter,
            output_path=output_dir / f"plots_{dim}d_{case}.png",
        )

    print(f"Wrote {output_dir / 'results_all.csv'}")
    print(f"Rows: {len(rows)}")
    print(f"Non-converged rows: {sum(1 for row in rows if not row['converged'])}")


if __name__ == "__main__":
    main()
