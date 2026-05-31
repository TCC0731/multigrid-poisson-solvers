from __future__ import annotations

"""Plot CUDA MG convergence and CUDA SOR benchmark timings on one figure.

This companion script reads the archived ``results_all.csv`` files from
``cuda_mg_convergence`` and ``cuda_sor_benchmark`` and combines them into a
single 1x2 figure:

* 2D on the left
* 3D on the right

Within each panel, the MG SOR modes and the RB SOR benchmark are drawn on the
same y-axis. The visual encoding follows the same marker / line-style palette
used by ``omp_thread_scaling/plot_from_csv.py``. Only time is shown, each
series gets a power-law fit in log-log space, and all available grid sizes are
shown on the x-axis.
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR.parent
if str(RESULT_ROOT) not in sys.path:
    sys.path.insert(0, str(RESULT_ROOT))

MPLCONFIGDIR = Path("/tmp/codex-matplotlib")
XDG_CACHE_HOME = Path("/tmp/codex-cache")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

from _figure_sizes import FIGSIZE_1X2_CUDA_MG_SOR_TIME_FIT, figsize_for_columns
from _report_common import finalize_figure_header, load_pyplot, normalize_choices, positive_int


DEFAULT_CASE = "sine"
DEFAULT_DIMS = (2, 3)
DEFAULT_MG_INPUT_CSV = SCRIPT_DIR.parent / "cuda_mg_convergence" / "results_all.csv"
DEFAULT_SOR_INPUT_CSV = SCRIPT_DIR.parent / "cuda_sor_benchmark" / "results_all.csv"
DEFAULT_OUTPUT_PNG = SCRIPT_DIR / "plots_2d_3d_mg_sor_time_fit_sine.png"
DEFAULT_BACKEND = "cuda"
DEFAULT_DTYPE = "double"
DEFAULT_FIT_LAST_N = 10


@dataclass(frozen=True)
class SeriesSpec:
    label: str
    series_key: str
    color: str
    marker: str
    linestyle: str


MG_SERIES_SPECS = (
    SeriesSpec("V-SOR", "v_sor", "#1f77b4", "s", "--"),
    SeriesSpec("W-SOR", "w_sor", "#d62728", "s", "--"),
)
SOR_SERIES_SPEC = SeriesSpec("SOR", "sor", "#2ca02c", "^", ":")
SERIES_SPECS = MG_SERIES_SPECS + (SOR_SERIES_SPEC,)
SERIES_SPECS_BY_KEY = {spec.series_key: spec for spec in SERIES_SPECS}


@dataclass(frozen=True)
class PlotRow:
    dimension: int
    case: str
    series_key: str
    grid_size: int
    time_s: float


def _parse_time_s(row: Mapping[str, str]) -> float:
    time_s = row.get("time_s")
    if time_s not in (None, ""):
        return float(time_s)
    time_ms = row.get("time_ms")
    if time_ms not in (None, ""):
        return float(time_ms) / 1000.0
    raise ValueError("row is missing both time_s and time_ms")


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


def _log_limits(values: Sequence[float], *, pad_low: float = 0.85, pad_high: float = 1.15) -> tuple[float, float]:
    positive = sorted(float(value) for value in values if float(value) > 0.0)
    if not positive:
        raise ValueError("at least one positive value is required")
    lo = positive[0]
    hi = positive[-1]
    if lo == hi:
        return lo * pad_low, hi * pad_high
    return lo * pad_low, hi * pad_high


def _unique_sorted_ints(values: Iterable[int]) -> list[int]:
    return sorted({int(value) for value in values})


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


def _mg_series_key(row: Mapping[str, str]) -> str:
    series_key = str(row.get("mode_key", "")).strip()
    if series_key not in SERIES_SPECS_BY_KEY:
        raise ValueError(f"unsupported MG mode_key: {series_key!r}")
    return series_key


def _sor_series_key() -> str:
    return SOR_SERIES_SPEC.series_key


def _sor_solver_label(dimension: int) -> str:
    return "RB SOR 3D" if dimension == 3 else "RB SOR"


def _load_mg_rows(
    csv_path: Path,
    *,
    case: str,
    dims: Sequence[int],
) -> list[PlotRow]:
    rows: list[PlotRow] = []
    seen: set[tuple[int, str, int]] = set()

    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for raw in reader:
            if str(raw.get("case", "")).strip() != case:
                continue
            if str(raw.get("backend", "")).strip() != DEFAULT_BACKEND:
                continue
            if str(raw.get("dtype", "")).strip() != DEFAULT_DTYPE:
                continue
            if str(raw.get("solver", "")).strip() != "mg":
                continue

            try:
                dimension = positive_int(raw["dimension"], "dimension")
            except Exception:
                continue
            if dimension not in dims:
                continue

            try:
                series_key = _mg_series_key(raw)
                grid_size = positive_int(raw["grid_size"], "grid_size")
                time_s = _parse_time_s(raw)
            except Exception:
                continue

            key = (dimension, series_key, grid_size)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                PlotRow(
                    dimension=dimension,
                    case=case,
                    series_key=series_key,
                    grid_size=grid_size,
                    time_s=time_s,
                )
            )

    rows.sort(key=lambda row: (row.dimension, row.series_key, row.grid_size))
    return rows


def _load_sor_rows(
    csv_path: Path,
    *,
    case: str,
    dims: Sequence[int],
) -> list[PlotRow]:
    rows: list[PlotRow] = []
    seen: set[tuple[int, str, int]] = set()

    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for raw in reader:
            if str(raw.get("case", "")).strip() != case:
                continue
            if str(raw.get("backend", "")).strip() != DEFAULT_BACKEND:
                continue
            if str(raw.get("dtype", "")).strip() != DEFAULT_DTYPE:
                continue

            try:
                dimension = positive_int(raw["dimension"], "dimension")
            except Exception:
                continue
            if dimension not in dims:
                continue

            if str(raw.get("solver", "")).strip() != _sor_solver_label(dimension):
                continue

            try:
                grid_size = positive_int(raw["grid_size"], "grid_size")
                time_s = _parse_time_s(raw)
            except Exception:
                continue

            series_key = _sor_series_key()
            key = (dimension, series_key, grid_size)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                PlotRow(
                    dimension=dimension,
                    case=case,
                    series_key=series_key,
                    grid_size=grid_size,
                    time_s=time_s,
                )
            )

    rows.sort(key=lambda row: (row.dimension, row.series_key, row.grid_size))
    return rows


def _series(rows: Iterable[PlotRow], *, dimension: int, series_key: str) -> list[PlotRow]:
    ordered = [row for row in rows if row.dimension == dimension and row.series_key == series_key]
    ordered.sort(key=lambda row: row.grid_size)
    return ordered


def _plot_series(
    ax,
    *,
    rows: Sequence[PlotRow],
    dimension: int,
    spec: SeriesSpec,
    fit_last_n: int,
) -> float | None:
    series = _series(rows, dimension=dimension, series_key=spec.series_key)
    if not series:
        return None

    x = [row.grid_size for row in series]
    y = [row.time_s for row in series]
    ax.plot(
        x,
        y,
        marker=spec.marker,
        linestyle=spec.linestyle,
        color=spec.color,
        linewidth=1.8,
        markersize=4.5,
        label=spec.label,
    )

    fit = _fit_last_n_curve(x, y, last_n=fit_last_n)
    if fit is not None:
        fit_x, fit_y, slope = fit
        ax.plot(
            fit_x,
            fit_y,
            color=spec.color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.55,
            label="_nolegend_",
        )
        return slope
    return None


def _panel_fit_text(fit_orders: Sequence[tuple[str, float]]) -> str:
    return "\n".join(rf"{label}: $O(N^{{{slope:.2f}}})$" for label, slope in fit_orders)


def _collect_legend_handles(primary_ax) -> tuple[list[object], list[str]]:
    handles_by_label: dict[str, object] = {}
    handles, labels = primary_ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label == "_nolegend_":
            continue
        handles_by_label.setdefault(label, handle)

    ordered_labels = [spec.label for spec in SERIES_SPECS if spec.label in handles_by_label]
    ordered_handles = [handles_by_label[label] for label in ordered_labels]
    return ordered_handles, ordered_labels


def _plot_combined_time_figure(
    *,
    mg_rows: Sequence[PlotRow],
    sor_rows: Sequence[PlotRow],
    case: str,
    dims: Sequence[int],
    output_path: Path,
    fit_last_n: int,
) -> None:
    plt = load_pyplot()

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(dims),
        figsize=figsize_for_columns(FIGSIZE_1X2_CUDA_MG_SOR_TIME_FIT, len(dims)),
        sharey=False,
    )
    if len(dims) == 1:
        axes = [axes]  # type: ignore[list-item]

    legend_handles: list[object] = []
    legend_labels: list[str] = []

    for col_idx, dimension in enumerate(dims):
        ax = axes[col_idx]

        fit_orders: list[tuple[str, float]] = []
        for spec in MG_SERIES_SPECS:
            slope = _plot_series(ax, rows=mg_rows, dimension=dimension, spec=spec, fit_last_n=fit_last_n)
            if slope is not None:
                fit_orders.append((spec.label, slope))

        slope = _plot_series(ax, rows=sor_rows, dimension=dimension, spec=SOR_SERIES_SPEC, fit_last_n=fit_last_n)
        if slope is not None:
            fit_orders.append((SOR_SERIES_SPEC.label, slope))

        all_rows = [row for row in mg_rows if row.dimension == dimension] + [row for row in sor_rows if row.dimension == dimension]
        x_values = [row.grid_size for row in all_rows]
        time_values = [row.time_s for row in all_rows]
        x_ticks = _unique_sorted_ints(x_values)

        x_min, x_max = _log_limits(x_values)
        time_min, time_max = _log_limits(time_values)

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(time_min, time_max)

        ax.set_title(f"{dimension}D time")
        ax.set_ylabel("Time (s)")
        ax.grid(True, which="both", linestyle="--", alpha=0.45)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(value) for value in x_ticks], rotation=90, fontsize=7)
        ax.set_xlabel("grid size")

        if fit_orders:
            ax.text(
                0.03,
                0.97,
                _panel_fit_text(fit_orders),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.4,
                linespacing=1.12,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.80,
                ),
            )

        if col_idx == 0:
            legend_handles, legend_labels = _collect_legend_handles(ax)

    finalize_figure_header(
        fig,
        title=f"CUDA MG convergence vs CUDA SOR benchmark - {case}",
        handles=legend_handles,
        labels=legend_labels,
        ncol=3,
        legend_y=0.965,
        title_y=0.996,
        tight_top=0.99,
        legend_kwargs={
            "fontsize": 8.8,
            "columnspacing": 1.2,
            "handletextpad": 0.6,
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a combined CUDA MG convergence and CUDA SOR benchmark time comparison."
    )
    parser.add_argument(
        "--mg-input-csv",
        type=Path,
        default=DEFAULT_MG_INPUT_CSV,
        help="Source cuda_mg_convergence results_all.csv file.",
    )
    parser.add_argument(
        "--sor-input-csv",
        type=Path,
        default=DEFAULT_SOR_INPUT_CSV,
        help="Source cuda_sor_benchmark results_all.csv file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PNG,
        help="Destination PNG path.",
    )
    parser.add_argument(
        "--case",
        default=DEFAULT_CASE,
        help="Poisson case to plot.",
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=list(DEFAULT_DIMS),
        help="Dimensions to include.",
    )
    parser.add_argument(
        "--fit-last-n",
        type=int,
        default=DEFAULT_FIT_LAST_N,
        help="Number of trailing points used for the power-law fit.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    case = str(args.case).strip()
    case = tuple(normalize_choices((case,), valid=("sine", "cosine"), name="case"))[0]
    dims = _unique_positive_ints(args.dims, name="dimension")
    dims = tuple(normalize_choices(dims, valid=DEFAULT_DIMS, name="dimension"))
    fit_last_n = positive_int(args.fit_last_n, "fit_last_n")

    if not args.mg_input_csv.exists():
        raise FileNotFoundError(f"MG input CSV not found: {args.mg_input_csv}")
    if not args.sor_input_csv.exists():
        raise FileNotFoundError(f"SOR input CSV not found: {args.sor_input_csv}")

    mg_rows = _load_mg_rows(args.mg_input_csv, case=case, dims=dims)
    sor_rows = _load_sor_rows(args.sor_input_csv, case=case, dims=dims)
    if not mg_rows:
        raise RuntimeError("No MG rows matched the requested filters.")
    if not sor_rows:
        raise RuntimeError("No SOR rows matched the requested filters.")

    _plot_combined_time_figure(
        mg_rows=mg_rows,
        sor_rows=sor_rows,
        case=case,
        dims=dims,
        output_path=args.output_path,
        fit_last_n=fit_last_n,
    )

    print(f"Wrote {args.output_path}")
    print(f"MG input CSV: {args.mg_input_csv}")
    print(f"SOR input CSV: {args.sor_input_csv}")
    print(f"Case: {case}")
    print(f"Dimensions: {', '.join(str(dim) for dim in dims)}")
    print(f"Fit last N: {fit_last_n}")
    print(f"MG rows: {len(mg_rows)}")
    print(f"SOR rows: {len(sor_rows)}")


if __name__ == "__main__":
    main()
