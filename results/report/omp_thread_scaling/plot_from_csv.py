from __future__ import annotations

"""Plot OpenMP thread-scaling figures from an existing ``results_all.csv``.

This script is the plotting-only companion to ``run_and_plot.py``. It does not
rerun the solver. Instead, it reads the cached CSV and draws a single 2x3
figure with 2D and 3D stacked by row and runtime / speedup / efficiency shown
by column.
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR.parent
if str(RESULT_ROOT) not in sys.path:
    sys.path.insert(0, str(RESULT_ROOT))

from _report_common import finalize_figure_header, load_pyplot, normalize_choices, positive_int


DEFAULT_INPUT_CSV = SCRIPT_DIR / "results_all.csv"
DEFAULT_OUTPUT_PNG = SCRIPT_DIR / "plots_2d_3d.png"
DEFAULT_CASE = "sine"
DEFAULT_DIMS = (2, 3)
DEFAULT_BACKEND = "omp"
DEFAULT_DTYPE = "double"
IDEAL_SPEEDUP_THREAD_LIMIT = 4


@dataclass(frozen=True)
class ModeSpec:
    label: str
    mode_key: str
    color: str
    marker: str
    linestyle: str


MODE_SPECS = (
    ModeSpec(
        label="SOR",
        mode_key="sor",
        color="#2ca02c",
        marker="^",
        linestyle=":",
    ),
    ModeSpec(
        label="MG V exact",
        mode_key="mg_v_exact",
        color="#1f77b4",
        marker="o",
        linestyle="-",
    ),
    ModeSpec(
        label="MG V SOR",
        mode_key="mg_v_sor",
        color="#1f77b4",
        marker="s",
        linestyle="--",
    ),
    ModeSpec(
        label="MG W exact",
        mode_key="mg_w_exact",
        color="#d62728",
        marker="o",
        linestyle="-",
    ),
    ModeSpec(
        label="MG W SOR",
        mode_key="mg_w_sor",
        color="#d62728",
        marker="s",
        linestyle="--",
    ),
)
MODE_ORDER = {spec.mode_key: index for index, spec in enumerate(MODE_SPECS)}
MODE_LABELS = {spec.mode_key: spec.label for spec in MODE_SPECS}


@dataclass(frozen=True)
class PlotRow:
    dimension: int
    case: str
    grid_size: int
    mode_key: str
    omp_num_threads: int
    time_ms: float
    speedup: float
    efficiency: float


def _parse_float(row: Mapping[str, str], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"row is missing {key}")
    return float(value)


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


def _load_rows(
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
            if str(raw.get("case", "")) != case:
                continue
            if str(raw.get("backend", "")) != DEFAULT_BACKEND:
                continue
            if str(raw.get("dtype", "")) != DEFAULT_DTYPE:
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
                thread_count = positive_int(raw["omp_num_threads"], "omp_num_threads")
                time_ms = _parse_float(raw, "time_ms")
                speedup = _parse_float(raw, "speedup")
                efficiency = _parse_float(raw, "efficiency")
            except Exception:
                continue

            key = (dimension, mode_key, thread_count)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                PlotRow(
                    dimension=dimension,
                    case=case,
                    grid_size=grid_size,
                    mode_key=mode_key,
                    omp_num_threads=thread_count,
                    time_ms=time_ms,
                    speedup=speedup,
                    efficiency=efficiency,
                )
            )

    rows.sort(
        key=lambda row: (
            row.dimension,
            MODE_ORDER[row.mode_key],
            row.omp_num_threads,
        )
    )
    return rows


def _series(rows: Iterable[PlotRow], *, dimension: int, mode_key: str) -> list[PlotRow]:
    ordered = [row for row in rows if row.dimension == dimension and row.mode_key == mode_key]
    ordered.sort(key=lambda row: row.omp_num_threads)
    return ordered


def _plot_mode_series(
    ax,
    *,
    rows: Sequence[PlotRow],
    dimension: int,
    metric_key: str,
) -> None:
    for spec in MODE_SPECS:
        series = _series(rows, dimension=dimension, mode_key=spec.mode_key)
        if not series:
            continue
        x = [row.omp_num_threads for row in series]
        y = [float(getattr(row, metric_key)) for row in series]
        ax.plot(
            x,
            y,
            marker=spec.marker,
            linestyle=spec.linestyle,
            color=spec.color,
            linewidth=1.9,
            label=spec.label,
        )


def _grid_desc(rows: Sequence[PlotRow], dims: Sequence[int]) -> str:
    parts: list[str] = []
    for dim in dims:
        dim_rows = [row for row in rows if row.dimension == dim]
        if not dim_rows:
            continue
        grid_size = dim_rows[0].grid_size
        parts.append(f"{dim}D grid={grid_size}")
    return ", ".join(parts)


def _expanded_limits(values: Sequence[int]) -> tuple[float, float]:
    ordered = sorted({float(value) for value in values})
    if not ordered:
        raise ValueError("at least one value is required")
    if len(ordered) == 1:
        center = ordered[0]
        pad = max(abs(center) * 0.05, 0.5)
        return center - pad, center + pad

    gaps = [right - left for left, right in zip(ordered, ordered[1:]) if right > left]
    if gaps:
        pad = min(gaps) * 0.5
    else:
        span = ordered[-1] - ordered[0]
        pad = max(span * 0.05, 0.5)
    return ordered[0] - pad, ordered[-1] + pad


def _plot_combined_thread_scaling(
    *,
    rows: Sequence[PlotRow],
    case: str,
    dims: Sequence[int],
    output_path: Path,
) -> None:
    plt = load_pyplot()
    thread_values = sorted({row.omp_num_threads for row in rows})
    if not thread_values:
        raise RuntimeError("No thread counts found in the filtered rows.")
    x_min, x_max = _expanded_limits(thread_values)

    runtime_min = min(row.time_ms for row in rows)
    runtime_max = max(row.time_ms for row in rows)
    speedup_max = max(IDEAL_SPEEDUP_THREAD_LIMIT, max(row.speedup for row in rows))
    efficiency_max = max(1.0, max(row.efficiency for row in rows))

    panels = (
        ("time_ms", "Runtime (ms)", "log"),
        ("speedup", "Speedup vs 1 thread", "linear"),
        ("efficiency", "Efficiency", "linear"),
    )

    fig, axes = plt.subplots(
        nrows=len(dims),
        ncols=3,
        figsize=(15.8, 4.25 * len(dims)),
        sharex=True,
    )
    if len(dims) == 1:
        axes = [axes]  # type: ignore[list-item]

    for row_idx, dimension in enumerate(dims):
        for col_idx, (metric_key, ylabel, scale) in enumerate(panels):
            ax = axes[row_idx][col_idx]
            _plot_mode_series(ax, rows=rows, dimension=dimension, metric_key=metric_key)

            if metric_key == "speedup":
                ideal_end = min(IDEAL_SPEEDUP_THREAD_LIMIT, thread_values[-1])
                ax.plot(
                    [1, ideal_end],
                    [1, ideal_end],
                    color="#666666",
                    linestyle=":",
                    linewidth=1.5,
                    label="_nolegend_",
                )
                ax.set_ylim(0.0, speedup_max * 1.08)
            elif metric_key == "efficiency":
                ax.axhline(1.0, color="#666666", linestyle=":", linewidth=1.5, label="_nolegend_")
                ax.set_ylim(0.0, efficiency_max * 1.08)
            elif metric_key == "time_ms":
                ax.set_yscale("log")
                ax.set_ylim(runtime_min * 0.90, runtime_max * 1.10)

            ax.set_title(f"{dimension}D {ylabel}")
            ax.set_xlabel("OMP threads" if row_idx == len(dims) - 1 else "")
            ax.set_ylabel(ylabel)
            ax.set_xticks(thread_values)
            ax.set_xlim(x_min, x_max)
            ax.grid(True, linestyle="--", alpha=0.45)

    handles, labels = axes[0][0].get_legend_handles_labels()
    finalize_figure_header(
        fig,
        title=f"OMP thread scaling - {case} ({_grid_desc(rows, dims)})",
        handles=handles,
        labels=labels,
        ncol=len(MODE_SPECS),
        legend_y=0.965,
        title_y=0.995,
        tight_top=0.975,
        legend_kwargs={
            "fontsize": 9.0,
            "columnspacing": 1.2,
            "handletextpad": 0.6,
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot OMP thread-scaling figures from an existing results_all.csv.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Source results_all.csv file.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PNG, help="Destination PNG path.")
    parser.add_argument("--case", default=DEFAULT_CASE, help="Poisson case to plot.")
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=list(DEFAULT_DIMS),
        help="Dimensions to include.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    case = str(args.case).strip()
    dims = _unique_positive_ints(args.dims, name="dimension")
    dims = tuple(normalize_choices(dims, valid=DEFAULT_DIMS, name="dimension"))
    rows = _load_rows(args.input_csv, case=case, dims=dims)
    if not rows:
        raise RuntimeError("No rows matched the requested case/dimension filters.")

    _plot_combined_thread_scaling(
        rows=rows,
        case=case,
        dims=dims,
        output_path=args.output_path,
    )

    print(f"Wrote {args.output_path}")
    print(f"Input CSV: {args.input_csv}")
    print(f"Case: {case}")
    print(f"Dimensions: {', '.join(str(dim) for dim in dims)}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
