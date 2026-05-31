from __future__ import annotations

"""Generate CUDA RB SOR tables and plots.

Run this inside the `conda activate mg` environment so `matplotlib` is
available. The script keeps the CUDA regeneration path via
`results/report/result/_poisson_wrapper.py`, but it can also fall back to the
archived example CSVs in this repository.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR.parent
if str(RESULT_ROOT) not in sys.path:
    sys.path.insert(0, str(RESULT_ROOT))

from _poisson_wrapper import REPO_ROOT, run_or_load
from _report_common import (
    csv_float,
    load_pyplot,
    normalize_choices,
    power_law_fit_curve,
    positive_float,
    positive_int,
    write_rows_csv,
)


DEFAULT_CASES = ("sine", "cosine")
DEFAULT_DIMS = (2, 3)
DEFAULT_BACKEND = "cuda"
DEFAULT_DTYPE = "double"
DEFAULT_TOL = 1e-9
DEFAULT_MAX_ITER = 10_000
DEFAULT_REPEAT_RUNS = 25
DEFAULT_SOURCE_ROOT = REPO_ROOT / "results" / "example"
TARGET_GRID_SIZES = {
    2: (63, 95, 127, 159, 191, 255, 319, 383, 511, 639, 767, 1023),
    3: (31, 47, 63, 79, 95, 111, 127, 143, 159, 191, 255),
}
CSV_COLUMNS = (
    "dimension",
    "case",
    "solver",
    "backend",
    "dtype",
    "grid_size",
    "iterations",
    "error_l2",
    "error_linf",
    "time_ms",
    "time_s",
    "converged",
    "source",
)

SERIES_COLOR = "#1f77b4"
SERIES_MARKER = "o"
SERIES_LINESTYLE = "-"


def _row_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        int(row["dimension"]),
        str(row["case"]),
        str(row["solver"]),
        int(row["grid_size"]),
    )


def _solver_label(dim: int) -> str:
    return "RB SOR 3D" if dim == 3 else "RB SOR"


def _source_file(source_root: Path, dim: int, case: str) -> Path:
    folder = "cuda_3d" if dim == 3 else "cuda"
    return source_root / folder / "solver_comparison" / f"results_{case}.csv"


def _parse_source_row(row: Mapping[str, str], source: Path, *, dim: int, case: str) -> dict[str, object]:
    time_s = float(row["Time_s"])
    return {
        "dimension": dim,
        "case": case,
        "solver": row["Solver"].strip(),
        "backend": DEFAULT_BACKEND,
        "dtype": DEFAULT_DTYPE,
        "grid_size": positive_int(row["N"], "N"),
        "iterations": positive_int(row["Iterations"], "Iterations"),
        "error_l2": float(row["L2_Error"]),
        "error_linf": float(row["Linf_Error"]),
        "time_ms": time_s * 1000.0,
        "time_s": time_s,
        "converged": True,
        "source": str(source),
    }


def _row_from_wrapper(
    *,
    dim: int,
    case: str,
    grid_size: int,
    tol: float,
    max_iter: int,
    repeat_runs: int,
) -> dict[str, object]:
    result = run_or_load(
        backend=DEFAULT_BACKEND,
        dim=dim,
        dtype=DEFAULT_DTYPE,
        solver="sor",
        case=case,
        grid_size=grid_size,
        tol=tol,
        max_iter=max_iter,
        repeat_runs=repeat_runs,
    )
    return {
        "dimension": dim,
        "case": case,
        "solver": _solver_label(dim),
        "backend": result.backend,
        "dtype": result.dtype,
        "grid_size": result.grid_size,
        "iterations": result.iterations,
        "error_l2": result.error_l2,
        "error_linf": result.error_linf,
        "time_ms": result.time_ms,
        "time_s": result.time_s,
        "converged": bool(result.residual_l2 <= tol),
        "source": "wrapper",
    }


def collect_rows_from_wrapper(
    *,
    dims: Iterable[int],
    cases: Iterable[str],
    tol: float,
    max_iter: int,
    repeat_runs: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dim in dims:
        for case in cases:
            for grid_size in TARGET_GRID_SIZES[dim]:
                rows.append(
                    _row_from_wrapper(
                        dim=dim,
                        case=case,
                        grid_size=grid_size,
                        tol=tol,
                        max_iter=max_iter,
                        repeat_runs=repeat_runs,
                    )
                )
    return rows


def collect_rows_from_source(
    *,
    source_root: Path,
    dims: Iterable[int],
    cases: Iterable[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for dim in dims:
        for case in cases:
            csv_path = _source_file(source_root, dim, case)
            if not csv_path.is_file():
                continue
            with csv_path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                for raw in reader:
                    solver = raw["Solver"].strip()
                    if solver != _solver_label(dim):
                        continue
                    grid_size = positive_int(raw["N"], "N")
                    if grid_size not in TARGET_GRID_SIZES[dim]:
                        continue
                    row = _parse_source_row(raw, csv_path, dim=dim, case=case)
                    key = _row_key(row)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(row)
    return rows


def _format_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            int(row["dimension"]),
            str(row["case"]),
            str(row["solver"]),
            int(row["grid_size"]),
        ),
    )
    formatted: list[dict[str, object]] = []
    for row in ordered_rows:
        formatted.append(
            {
                "dimension": int(row["dimension"]),
                "case": row["case"],
                "solver": row["solver"],
                "backend": row["backend"],
                "dtype": row["dtype"],
                "grid_size": int(row["grid_size"]),
                "iterations": int(row["iterations"]),
                "error_l2": csv_float(float(row["error_l2"])),
                "error_linf": csv_float(float(row["error_linf"])),
                "time_ms": csv_float(float(row["time_ms"])),
                "time_s": csv_float(float(row["time_s"])),
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
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    plt = load_pyplot()

    ordered_rows = sorted(rows, key=lambda row: int(row["grid_size"]))
    x = [int(row["grid_size"]) for row in ordered_rows]
    error_l2 = [float(row["error_l2"]) for row in ordered_rows]
    iterations = [float(row["iterations"]) for row in ordered_rows]
    time_s = [float(row["time_s"]) for row in ordered_rows]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    panels = (
        (error_l2, "error_l2", "L2 error"),
        (iterations, "iter", "Iterations"),
        (time_s, "time_s", "Time (s)"),
    )

    for ax, (y, ylabel, title) in zip(axes, panels):
        ax.loglog(
            x,
            y,
            marker=SERIES_MARKER,
            linestyle=SERIES_LINESTYLE,
            color=SERIES_COLOR,
            linewidth=1.9,
            label=_solver_label(dim),
        )
        fit = power_law_fit_curve(x, y)
        if fit is not None:
            fit_x, fit_y, slope = fit
            ax.loglog(
                fit_x,
                fit_y,
                color=SERIES_COLOR,
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
                label=f"{_solver_label(dim)} fit" if ylabel == "error_l2" else None,
            )
            if ylabel != "error_l2":
                ax.text(
                    0.04,
                    0.96,
                    rf"$O(N^{{{slope:.2f}}})$",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
                )
        if ylabel == "error_l2" and len(x) >= 2:
            ref_x = [x[0], x[-1]]
            ref_y = [y[0], y[0] * (ref_x[1] / ref_x[0]) ** -2.0]
            ax.loglog(ref_x, ref_y, color="#5f6675", linestyle=":", linewidth=1.5, label=r"$O(N^{-2})$")
        ax.set_title(title)
        ax.set_xlabel("grid size")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle="--", alpha=0.45)

    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle(f"CUDA SOR convergence - {dim}D {case}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build CUDA SOR tables and plots.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR, help="Directory for CSV and plot outputs.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help="Archived example CSV root.")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES), help="Cases to include.")
    parser.add_argument("--dims", nargs="+", type=int, default=list(DEFAULT_DIMS), help="Dimensions to include.")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL, help="Convergence tolerance for wrapper runs.")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help="Maximum iterations for wrapper runs.")
    parser.add_argument(
        "--repeat-runs",
        type=int,
        default=DEFAULT_REPEAT_RUNS,
        help="Repeat runs per solver invocation when regenerating from CUDA.",
    )
    parser.add_argument("--source-only", action="store_true", help="Skip CUDA regeneration and read archived CSVs only.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = tuple(normalize_choices(args.cases, valid=DEFAULT_CASES, name="case"))
    dims = tuple(positive_int(dim, "dim") for dim in args.dims)
    dims = tuple(normalize_choices(dims, valid=DEFAULT_DIMS, name="dimension"))
    tol = positive_float(args.tol, "tol")
    max_iter = positive_int(args.max_iter, "max_iter")
    repeat_runs = positive_int(args.repeat_runs, "repeat_runs")

    rows: list[dict[str, object]]
    source_name: str

    if not args.source_only:
        try:
            rows = collect_rows_from_wrapper(
                dims=dims,
                cases=cases,
                tol=tol,
                max_iter=max_iter,
                repeat_runs=repeat_runs,
            )
            source_name = "wrapper"
        except Exception as exc:
            print(
                "CUDA regeneration failed, falling back to archived example CSVs:\n"
                f"  {exc}",
                file=sys.stderr,
            )
            rows = collect_rows_from_source(
                source_root=args.source_root,
                dims=dims,
                cases=cases,
            )
            source_name = str(args.source_root)
    else:
        rows = collect_rows_from_source(
            source_root=args.source_root,
            dims=dims,
            cases=cases,
        )
        source_name = str(args.source_root)

    if not rows:
        raise RuntimeError("No benchmark rows matched the requested filters.")

    rows = sorted(
        rows,
        key=lambda row: (
            int(row["dimension"]),
            str(row["case"]),
            str(row["solver"]),
            int(row["grid_size"]),
        ),
    )

    write_rows_csv(output_dir / "results_all.csv", _format_rows(rows), columns=CSV_COLUMNS)

    grouped = group_rows(rows)
    for dim, case in sorted(grouped):
        case_rows = grouped[(dim, case)]
        case_rows = sorted(case_rows, key=lambda row: int(row["grid_size"]))
        write_rows_csv(
            output_dir / f"results_{dim}d_{case}.csv",
            _format_rows(case_rows),
            columns=CSV_COLUMNS,
        )
        plot_case(
            dim=dim,
            case=case,
            rows=case_rows,
            output_path=output_dir / f"plots_{dim}d_{case}.png",
        )

    print(f"Wrote {output_dir / 'results_all.csv'}")
    print(f"Source used: {source_name}")
    print(f"Cases: {', '.join(cases)}")
    print(f"Dimensions: {', '.join(str(dim) for dim in dims)}")
    print(f"Rows: {len(rows)}")
    print(f"Non-converged rows: {sum(1 for row in rows if not row['converged'])}")


if __name__ == "__main__":
    main()
