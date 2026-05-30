from __future__ import annotations

"""Generate CUDA MG convergence tables and plots.

Run this script inside the `conda activate mg` environment so `matplotlib`
is available. The script keeps the CUDA regeneration path through
`results/report/result/_poisson_wrapper.py`, but also falls back to the
archived sweep CSVs when CUDA is unavailable.
"""

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
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
DEFAULT_OMEGA = 1.25
DEFAULT_NU = 3
DEFAULT_TOL = 1e-9
DEFAULT_MAX_ITER = 15
DEFAULT_REPEAT_RUNS = 50
DEFAULT_SOURCE_ROOT = REPO_ROOT / "results" / "example" / "cuda_mg_omega_sweep"
TARGET_GRID_RANGES = {2: (127, 4095), 3: (32, 383)}
CSV_COLUMNS = (
    "dimension",
    "case",
    "mode",
    "mode_key",
    "cycle",
    "coarse",
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
    coarse: str
    color: str
    marker: str
    linestyle: str


MODE_SPECS = (
    ModeSpec("V-exact", "v_exact", "v", "exact", "#1f77b4", "o", "-"),
    ModeSpec("V-SOR", "v_sor", "v", "sor", "#1f77b4", "s", "--"),
    ModeSpec("W-exact", "w_exact", "w", "exact", "#d62728", "o", "-"),
    ModeSpec("W-SOR", "w_sor", "w", "sor", "#d62728", "s", "--"),
)
MODE_SPECS_BY_KEY = {spec.mode_key: spec for spec in MODE_SPECS}


def generate_valid_grid_sizes(min_n: int, max_n: int) -> tuple[int, ...]:
    values: list[int] = []
    seen: set[int] = set()
    power = 0
    while True:
        scale = 1 << power
        added = False
        for base in (1, 3, 5):
            n = base * scale - 1
            if min_n <= n <= max_n and n not in seen:
                seen.add(n)
                values.append(n)
                added = True
        if scale - 1 > max_n and not added:
            break
        power += 1
    return tuple(sorted(values))


def _row_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        int(row["dimension"]),
        str(row["case"]),
        str(row["mode_key"]),
        int(row["grid_size"]),
    )


def _parse_source_row(row: Mapping[str, str], source: Path) -> dict[str, object]:
    time_s = float(row["time_s"]) if row.get("time_s") else float(row["time_ms"]) / 1000.0
    return {
        "dimension": positive_int(row["dimension"], "dimension"),
        "case": row["case"],
        "mode": row["mode"],
        "mode_key": row["mode_key"],
        "cycle": row["cycle"],
        "coarse": row["coarse"],
        "omega": float(row["omega"]),
        "solver": row["solver"],
        "backend": row["backend"],
        "dtype": row["dtype"],
        "grid_size": positive_int(row["grid_size"], "grid_size"),
        "iterations": positive_int(row["iterations"], "iterations"),
        "residual_l2": float(row["residual_l2"]),
        "error_l2": float(row["error_l2"]),
        "error_linf": float(row["error_linf"]),
        "time_ms": float(row["time_ms"]),
        "time_s": time_s,
        "tol": float(row["tol"]),
        "max_iter": positive_int(row["max_iter"], "max_iter"),
        "nu": positive_int(row["nu"], "nu"),
        "converged": str(row["converged"]).strip().lower() == "true",
        "source": str(source),
    }


def _row_from_wrapper(
    *,
    dim: int,
    case: str,
    grid_size: int,
    mode: ModeSpec,
    omega: float,
    nu: int,
    tol: float,
    max_iter: int,
    repeat_runs: int,
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
        cycle=mode.cycle,
        nu=nu,
        omega=omega,
        mg_coarse=mode.coarse,
    )
    return {
        "dimension": dim,
        "case": case,
        "mode": mode.label,
        "mode_key": mode.mode_key,
        "cycle": mode.cycle,
        "coarse": mode.coarse,
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
    omega: float,
    nu: int,
    tol: float,
    max_iter: int,
    repeat_runs: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dim in dims:
        min_n, max_n = TARGET_GRID_RANGES[dim]
        grid_sizes = generate_valid_grid_sizes(min_n, max_n)
        for case in cases:
            for grid_size in grid_sizes:
                for mode in MODE_SPECS:
                    rows.append(
                        _row_from_wrapper(
                            dim=dim,
                            case=case,
                            grid_size=grid_size,
                            mode=mode,
                            omega=omega,
                            nu=nu,
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
    omega: float,
    nu: int,
    tol: float,
    max_iter: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()

    for csv_path in sorted(source_root.rglob("results_all.csv")):
        with csv_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            for raw in reader:
                try:
                    dim = positive_int(raw["dimension"], "dimension")
                except Exception:
                    continue
                case = raw["case"]
                mode_key = raw["mode_key"]
                if dim not in dims or case not in cases or mode_key not in MODE_SPECS_BY_KEY:
                    continue
                if raw["backend"] != DEFAULT_BACKEND or raw["dtype"] != DEFAULT_DTYPE:
                    continue
                if abs(float(raw["omega"]) - omega) > 1e-12:
                    continue
                if positive_int(raw["nu"], "nu") != nu:
                    continue
                if abs(float(raw["tol"]) - tol) > 1e-15:
                    continue
                if positive_int(raw["max_iter"], "max_iter") != max_iter:
                    continue
                min_n, max_n = TARGET_GRID_RANGES[dim]
                grid_size = positive_int(raw["grid_size"], "grid_size")
                if not (min_n <= grid_size <= max_n):
                    continue
                row = _parse_source_row(raw, csv_path)
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
            str(row["mode_key"]),
            int(row["grid_size"]),
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
    rows: list[dict[str, object]],
    omega: float,
    nu: int,
    tol: float,
    max_iter: int,
    output_path: Path,
) -> None:
    plt = load_pyplot()

    series_by_label: dict[str, list[dict[str, object]]] = {spec.label: [] for spec in MODE_SPECS}
    for row in rows:
        series_by_label[MODE_SPECS_BY_KEY[str(row["mode_key"])].label].append(row)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4))
    panels = (
        ("error_l2", "error_l2", "L2 error"),
        ("iterations", "iter", "Iterations"),
        ("time_s", "time_s", "Time (s)"),
    )
    legend_handles = []
    legend_labels: list[str] = []

    for ax_idx, (ax, (metric_key, ylabel, title)) in enumerate(zip(axes, panels)):
        use_fit = metric_key != "iterations"
        fit_orders: list[tuple[ModeSpec, float]] = []
        for spec in MODE_SPECS:
            series = sorted(series_by_label[spec.label], key=lambda row: int(row["grid_size"]))
            if not series:
                continue
            x = [int(row["grid_size"]) for row in series]
            y = [float(row[metric_key]) for row in series]
            plot_fn = ax.loglog if metric_key != "iterations" else ax.semilogx
            (main_line,) = plot_fn(
                x,
                y,
                marker=spec.marker,
                linestyle=spec.linestyle,
                color=spec.color,
                linewidth=1.8,
                label=spec.label,
            )
            if spec.label not in legend_labels:
                legend_handles.append(main_line)
                legend_labels.append(spec.label)
            if use_fit:
                fit = power_law_fit_curve(x, y)
                if fit is not None:
                    fit_x, fit_y, slope = fit
                    ax.loglog(
                        fit_x,
                        fit_y,
                        color=spec.color,
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.85,
                        label="_nolegend_",
                    )
                    fit_orders.append((spec, slope))
        ax.set_title(title)
        ax.set_xlabel("grid size")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle="--", alpha=0.45)

        if metric_key == "iterations":
            ax.set_ylim(0, 10)

        if fit_orders:
            fit_text = "\n".join(
                rf"{spec.label}: $O(N^{{{slope:.2f}}})$"
                for spec, slope in fit_orders
            )
            text_x = 0.03
            text_y = 0.97
            text_ha = "left"
            text_va = "top"
            if metric_key == "error_l2":
                text_y = 0.12
                text_va = "bottom"
            ax.text(
                text_x,
                text_y,
                fit_text,
                transform=ax.transAxes,
                ha=text_ha,
                va=text_va,
                fontsize=8.8,
                linespacing=1.15,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.80),
            )

        if metric_key == "error_l2":
            series = sorted(series_by_label[MODE_SPECS[0].label], key=lambda row: int(row["grid_size"]))
            if len(series) >= 2:
                x = [int(row["grid_size"]) for row in series]
                y = [float(row[metric_key]) for row in series]
                ref_x = [x[0], x[-1]]
                ref_y = [y[0], y[0] * (ref_x[1] / ref_x[0]) ** -2.0]
                ax.loglog(ref_x, ref_y, color="#5f6675", linestyle=":", linewidth=1.5, label="_nolegend_")

    axes[0].legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        frameon=False,
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    axes[0].text(
        0.98,
        0.12,
        r"$O(N^{-2})$",
        transform=axes[0].transAxes,
        ha="right",
        va="center",
        fontsize=9,
        color="#5f6675",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.72),
    )
    fig.suptitle(
        f"CUDA MG convergence - {dim}D {case}\n",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 1.05))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build CUDA MG convergence tables and plots.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR, help="Directory for CSV and plot outputs.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help="Archived sweep CSV root.")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES), help="Cases to include.")
    parser.add_argument("--dims", nargs="+", type=int, default=list(DEFAULT_DIMS), help="Dimensions to include.")
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
    parser.add_argument("--source-only", action="store_true", help="Skip CUDA regeneration and read archived CSVs only.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = tuple(normalize_choices(args.cases, valid=DEFAULT_CASES, name="case"))
    dims = tuple(positive_int(dim, "dim") for dim in args.dims)
    dims = tuple(normalize_choices(dims, valid=DEFAULT_DIMS, name="dimension"))
    omega = positive_float(args.omega, "omega")
    nu = positive_int(args.nu, "nu")
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
                omega=omega,
                nu=nu,
                tol=tol,
                max_iter=max_iter,
                repeat_runs=repeat_runs,
            )
            source_name = "wrapper"
        except Exception as exc:
            print(
                "CUDA regeneration failed, falling back to archived sweep CSVs:\n"
                f"  {exc}",
                file=sys.stderr,
            )
            rows = collect_rows_from_source(
                source_root=args.source_root,
                dims=dims,
                cases=cases,
                omega=omega,
                nu=nu,
                tol=tol,
                max_iter=max_iter,
            )
            source_name = str(args.source_root)
    else:
        rows = collect_rows_from_source(
            source_root=args.source_root,
            dims=dims,
            cases=cases,
            omega=omega,
            nu=nu,
            tol=tol,
            max_iter=max_iter,
        )
        source_name = str(args.source_root)

    if not rows:
        raise RuntimeError("No benchmark rows matched the requested filters.")

    rows = sorted(
        rows,
        key=lambda row: (
            int(row["dimension"]),
            str(row["case"]),
            str(row["mode_key"]),
            int(row["grid_size"]),
        ),
    )

    write_rows_csv(output_dir / "results_all.csv", _format_rows(rows), columns=CSV_COLUMNS)

    grouped = group_rows(rows)
    for dim, case in sorted(grouped):
        case_rows = grouped[(dim, case)]
        case_rows = sorted(case_rows, key=lambda row: (str(row["mode_key"]), int(row["grid_size"])))
        write_rows_csv(
            output_dir / f"results_{dim}d_{case}.csv",
            _format_rows(case_rows),
            columns=CSV_COLUMNS,
        )
        plot_case(
            dim=dim,
            case=case,
            rows=case_rows,
            omega=omega,
            nu=nu,
            tol=tol,
            max_iter=max_iter,
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
