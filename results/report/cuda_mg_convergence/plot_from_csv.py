from __future__ import annotations

"""Plot CUDA MG convergence figures from an existing ``results_all.csv``.

This script is a plotting-only companion to ``run_and_plot.py``. It does not
regenerate solver data; it simply reads the archived CSV and writes a few
comparison plots that are easier to use in the report:

1. L2 error comparison for 2D and 3D, with the same visual style as the left
   panel of the original convergence figure, but without the bottom-right
   ``O(N^-2)`` text label and with fit-order text in the upper right.
2. A 2x2 iteration/time figure with 2D and 3D on separate rows and
   iterations/time on separate columns, showing V/W exact and SOR.
3. The same 2x2 layout, but with only V/W SOR and a power-law fit on the time
   panels from the last ``N`` points of each series, drawn across the full
   grid range.
"""

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR.parent
if str(RESULT_ROOT) not in sys.path:
    sys.path.insert(0, str(RESULT_ROOT))

from _figure_sizes import (
    FIGSIZE_1X2_CUDA_MG_CONVERGENCE,
    FIGSIZE_2X2_2_ROWS,
    figsize_for_columns,
    figsize_for_rows,
)
from _report_common import finalize_figure_header, load_pyplot, normalize_choices, positive_int, power_law_fit_curve


DEFAULT_INPUT_CSV = SCRIPT_DIR / "results_all.csv"
DEFAULT_CASES = ("sine",)
DEFAULT_DIMS = (2, 3)
DEFAULT_FIT_LAST_N = 10

MODE_LABELS = {
    "v_exact": "V-exact",
    "v_sor": "V-SOR",
    "w_exact": "W-exact",
    "w_sor": "W-SOR",
}
MODE_ORDER = ("v_exact", "v_sor", "w_exact", "w_sor")
SOR_MODE_ORDER = ("v_sor", "w_sor")
MODE_MARKERS = {
    "v_exact": "o",
    "v_sor": "s",
    "w_exact": "o",
    "w_sor": "s",
}
MODE_LINESTYLES = {
    "v_exact": "-",
    "v_sor": "--",
    "w_exact": "-",
    "w_sor": "--",
}
FIG1_MODE_COLORS = {
    "v_exact": "#1f77b4",
    "v_sor": "#1f77b4",
    "w_exact": "#d62728",
    "w_sor": "#d62728",
}


@dataclass(frozen=True)
class PlotRow:
    case: str
    dimension: int
    mode_key: str
    grid_size: int
    iterations: int
    error_l2: float
    time_s: float


def _parse_time_s(row: Mapping[str, str]) -> float:
    if row.get("time_s"):
        return float(row["time_s"])
    if row.get("time_ms"):
        return float(row["time_ms"]) / 1000.0
    raise ValueError("row is missing both time_s and time_ms")


def _load_rows(
    csv_path: Path,
    *,
    cases: Sequence[str],
    dims: Sequence[int],
) -> list[PlotRow]:
    rows: list[PlotRow] = []
    seen: set[tuple[str, int, str, int]] = set()

    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for raw in reader:
            case = str(raw.get("case", ""))
            if case not in cases:
                continue

            try:
                dimension = positive_int(raw["dimension"], "dimension")
            except Exception:
                continue
            if dimension not in dims:
                continue

            mode_key = str(raw.get("mode_key", ""))
            if mode_key not in MODE_LABELS:
                continue

            try:
                grid_size = positive_int(raw["grid_size"], "grid_size")
                iterations = positive_int(raw["iterations"], "iterations")
                error_l2 = float(raw["error_l2"])
                time_s = _parse_time_s(raw)
            except Exception:
                continue

            key = (case, dimension, mode_key, grid_size)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                PlotRow(
                    case=case,
                    dimension=dimension,
                    mode_key=mode_key,
                    grid_size=grid_size,
                    iterations=iterations,
                    error_l2=error_l2,
                    time_s=time_s,
                )
            )

    rows.sort(key=lambda row: (row.case, row.dimension, row.mode_key, row.grid_size))
    return rows


def _series(rows: Iterable[PlotRow], *, dimension: int, mode_key: str) -> list[PlotRow]:
    ordered = [row for row in rows if row.dimension == dimension and row.mode_key == mode_key]
    ordered.sort(key=lambda row: row.grid_size)
    return ordered


def _unique_sorted_ints(values: Iterable[int]) -> list[int]:
    return sorted({int(value) for value in values})


def _apply_grid_ticks(ax, x_values: Iterable[int]) -> None:
    x_ticks = _unique_sorted_ints(x_values)
    if not x_ticks:
        return

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(value) for value in x_ticks], rotation=90, fontsize=7)


def _ref_line(x_values: Sequence[int], y_values: Sequence[float]) -> tuple[list[float], list[float]] | None:
    if len(x_values) < 2:
        return None
    x0 = float(x_values[0])
    x1 = float(x_values[-1])
    y0 = float(y_values[0])
    if x0 <= 0.0 or x1 <= 0.0 or y0 <= 0.0:
        return None
    return [x0, x1], [y0, y0 * (x1 / x0) ** -2.0]


def _fit_last_n_curve(
    x_values: Sequence[int],
    y_values: Sequence[float],
    *,
    last_n: int,
) -> tuple[list[float], list[float], float] | None:
    points = [(float(x), float(y)) for x, y in zip(x_values, y_values) if float(x) > 0.0 and float(y) > 0.0]
    if len(points) < 2 or last_n < 2:
        return None

    fit_points = points[-last_n:] if len(points) > last_n else points
    if len(fit_points) < 2:
        return None

    log_x = [math.log10(x) for x, _ in fit_points]
    log_y = [math.log10(y) for _, y in fit_points]
    n = len(fit_points)
    sum_x = sum(log_x)
    sum_y = sum(log_y)
    sum_xx = sum(value * value for value in log_x)
    sum_xy = sum(x * y for x, y in zip(log_x, log_y))
    denom = n * sum_xx - sum_x * sum_x
    slope = 0.0 if abs(denom) < 1e-12 else (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    x_min = min(x for x, _ in points)
    x_max = max(x for x, _ in points)
    if x_min == x_max:
        fit_x = [x_min, x_max]
    else:
        steps = 200
        log_min = math.log10(x_min)
        log_max = math.log10(x_max)
        fit_x = [
            10 ** (log_min + (log_max - log_min) * index / (steps - 1))
            for index in range(steps)
        ]

    fit_y = [10 ** (intercept + slope * math.log10(x)) for x in fit_x]
    return fit_x, fit_y, slope


def plot_l2_error_comparison(
    *,
    rows: Sequence[PlotRow],
    case: str,
    dims: Sequence[int],
    output_path: Path,
) -> None:
    plt = load_pyplot()

    fig, axes = plt.subplots(
        1,
        len(dims),
        figsize=figsize_for_columns(FIGSIZE_1X2_CUDA_MG_CONVERGENCE, len(dims)),
    )
    if len(dims) == 1:
        axes = [axes]  # type: ignore[list-item]
    for ax, dimension in zip(axes, dims):
        fit_orders: list[tuple[str, float]] = []
        for mode_key in SOR_MODE_ORDER:
            ordered = _series(rows, dimension=dimension, mode_key=mode_key)
            if not ordered:
                continue
            x = [row.grid_size for row in ordered]
            y = [row.error_l2 for row in ordered]
            ax.loglog(
                x,
                y,
                marker=MODE_MARKERS[mode_key],
                linestyle=MODE_LINESTYLES[mode_key],
                color=FIG1_MODE_COLORS[mode_key],
                linewidth=1.8,
                label=MODE_LABELS[mode_key],
            )
            fit = power_law_fit_curve(x, y)
            if fit is not None:
                fit_x, fit_y, slope = fit
                ax.loglog(
                    fit_x,
                    fit_y,
                    color=FIG1_MODE_COLORS[mode_key],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.85,
                    label="_nolegend_",
                )
                fit_orders.append((MODE_LABELS[mode_key], slope))

        ref = _ref_line(
            [row.grid_size for row in _series(rows, dimension=dimension, mode_key="v_sor")],
            [row.error_l2 for row in _series(rows, dimension=dimension, mode_key="v_sor")],
        )
        if ref is not None:
            ref_x, ref_y = ref
            ax.loglog(
                ref_x,
                ref_y,
                color="#5f6675",
                linestyle=":",
                linewidth=1.5,
                label="_nolegend_",
            )

        ax.set_title(f"{dimension}D L2 error")
        ax.set_xlabel("grid size")
        ax.set_ylabel("L2 Error")
        ax.grid(True, which="both", linestyle="--", alpha=0.45)

        if fit_orders:
            fit_text = "\n".join(
                rf"{label}: $O(N^{{{slope:.2f}}})$"
                for label, slope in fit_orders
            )
            ax.text(
                0.97,
                0.97,
                fit_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.8,
                linespacing=1.15,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.80),
            )

        _apply_grid_ticks(ax, (row.grid_size for row in rows if row.dimension == dimension))

    handles, labels = axes[0].get_legend_handles_labels()
    finalize_figure_header(
        fig,
        title=f"CUDA MG convergence - {case} (2D/3D L2 error)",
        handles=handles,
        labels=labels,
        ncol=2,
        legend_y=0.955,
        title_y=0.995,
        tight_top=0.96,
        legend_kwargs={"fontsize": 9},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_iter_time_panel(
    ax,
    *,
    rows: Sequence[PlotRow],
    dimension: int,
    metric_key: str,
    title: str,
    include_fit: bool,
    fit_last_n: int,
    mode_order: Sequence[str],
) -> list[tuple[str, float]]:
    fit_orders: list[tuple[str, float]] = []
    plot_fn = ax.semilogx if metric_key == "iterations" else ax.loglog
    for mode_key in mode_order:
        ordered = _series(rows, dimension=dimension, mode_key=mode_key)
        if not ordered:
            continue

        x = [row.grid_size for row in ordered]
        y = [getattr(row, metric_key) for row in ordered]
        label = MODE_LABELS[mode_key]
        plot_fn(
            x,
            y,
            marker=MODE_MARKERS[mode_key],
            linestyle=MODE_LINESTYLES[mode_key],
            color=FIG1_MODE_COLORS[mode_key],
            linewidth=1.8,
            label=label,
        )

        if include_fit and metric_key == "time_s":
            fit = _fit_last_n_curve(x, y, last_n=fit_last_n)
            if fit is not None:
                fit_x, fit_y, slope = fit
                plot_fn(
                    fit_x,
                    fit_y,
                    color=FIG1_MODE_COLORS[mode_key],
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.9,
                    label="_nolegend_",
                )
                fit_orders.append((label, slope))

    ax.set_title(title)
    ax.set_xlabel("grid size")
    ax.set_ylabel("Iterations" if metric_key == "iterations" else "Time (s)")
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    _apply_grid_ticks(ax, (row.grid_size for row in rows if row.dimension == dimension))
    if metric_key == "iterations":
        ax.set_ylim(0, 10)
    return fit_orders


def plot_iter_time_comparison(
    *,
    rows: Sequence[PlotRow],
    case: str,
    dims: Sequence[int],
    output_path: Path,
    include_fit: bool,
    fit_last_n: int,
    sor_only: bool = False,
) -> None:
    """Render the 2x2 iteration/time figure.

    Each row corresponds to a dimension and each column corresponds to a
    metric, so the raw curves remain separated while the overall figure still
    compares 2D and 3D side-by-side.
    """

    plt = load_pyplot()

    mode_order = SOR_MODE_ORDER if sor_only else MODE_ORDER
    nrows = len(dims)
    fig, axes = plt.subplots(nrows, 2, figsize=figsize_for_rows(FIGSIZE_2X2_2_ROWS, nrows))
    if nrows == 1:
        axes = [axes]  # type: ignore[list-item]

    for row_idx, dimension in enumerate(dims):
        fit_orders_by_metric: dict[str, list[tuple[str, float]]] = {}
        for col_idx, (metric_key, title, ylabel) in enumerate((
            ("iterations", "Iterations", "Iterations"),
            ("time_s", "Time (s)", "Time (s)"),
        )):
            ax = axes[row_idx][col_idx]
            fit_orders_by_metric[metric_key] = _plot_iter_time_panel(
                ax,
                rows=rows,
                dimension=dimension,
                metric_key=metric_key,
                title=f"{dimension}D {title}",
                include_fit=include_fit,
                fit_last_n=fit_last_n,
                mode_order=mode_order,
            )
            ax.set_ylabel(ylabel)
            if row_idx == nrows - 1:
                ax.set_xlabel("grid size")
            else:
                ax.set_xlabel("")

        if include_fit:
            time_ax = axes[row_idx][1]
            fit_orders = fit_orders_by_metric["time_s"]
            if fit_orders:
                fit_text = "\n".join(
                    rf"{label}: $O(N^{{{slope:.2f}}})$"
                    for label, slope in fit_orders
                )
                time_ax.text(
                    0.03,
                    0.97,
                    fit_text,
                    transform=time_ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.6,
                    linespacing=1.12,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.80),
                )

    handles, labels = axes[0][0].get_legend_handles_labels()
    title_suffix = "SOR only" if sor_only else "all modes"
    fit_suffix = f", time fit last {fit_last_n} points" if include_fit else ""
    finalize_figure_header(
        fig,
        title=f"CUDA MG convergence - {case} ({title_suffix})",
        handles=handles,
        labels=labels,
        ncol=4,
        legend_y=0.955,
        title_y=0.995,
        tight_top=0.96,
        legend_kwargs={"fontsize": 8},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot CUDA MG convergence figures from an existing results_all.csv.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Source results_all.csv file.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR, help="Directory for generated plots.")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES), help="Cases to include.")
    parser.add_argument("--dims", nargs="+", type=int, default=list(DEFAULT_DIMS), help="Dimensions to include.")
    parser.add_argument(
        "--fit-last-n",
        type=int,
        default=DEFAULT_FIT_LAST_N,
        help="Number of trailing points used for the fit in the third figure.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = tuple(normalize_choices(args.cases, valid=DEFAULT_CASES, name="case"))
    dims = tuple(positive_int(dim, "dim") for dim in args.dims)
    dims = tuple(normalize_choices(dims, valid=DEFAULT_DIMS, name="dimension"))
    fit_last_n = positive_int(args.fit_last_n, "fit_last_n")

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    rows = _load_rows(args.input_csv, cases=cases, dims=dims)
    if not rows:
        raise RuntimeError("No rows matched the requested filters.")

    for case in cases:
        case_rows = [row for row in rows if row.case == case]
        if not case_rows:
            continue

        plot_l2_error_comparison(
            rows=case_rows,
            case=case,
            dims=dims,
            output_path=output_dir / f"plots_2d_3d_l2_error_{case}.png",
        )
        plot_iter_time_comparison(
            rows=case_rows,
            case=case,
            dims=dims,
            output_path=output_dir / f"plots_2d_3d_iter_time_{case}.png",
            include_fit=False,
            fit_last_n=fit_last_n,
            sor_only=False,
        )
        plot_iter_time_comparison(
            rows=case_rows,
            case=case,
            dims=dims,
            output_path=output_dir / f"plots_2d_3d_iter_time_fit_last{fit_last_n}_{case}.png",
            include_fit=True,
            fit_last_n=fit_last_n,
            sor_only=True,
        )

        print(f"Wrote {output_dir / f'plots_2d_3d_l2_error_{case}.png'}")
        print(f"Wrote {output_dir / f'plots_2d_3d_iter_time_{case}.png'}")
        print(f"Wrote {output_dir / f'plots_2d_3d_iter_time_fit_last{fit_last_n}_{case}.png'}")

    print(f"Input CSV: {args.input_csv}")
    print(f"Cases: {', '.join(cases)}")
    print(f"Dimensions: {', '.join(str(dim) for dim in dims)}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
