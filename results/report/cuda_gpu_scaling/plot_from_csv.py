from __future__ import annotations

"""Plot CUDA GPU scaling figures from an existing ``results_all.csv``.

This plotting-only companion mirrors the visual style of
``run_and_plot.py``, but it does not rerun the solver. Instead, it reads the
cached summary CSV and draws a single 2x3 figure per case with 2D and 3D
stacked by row and runtime / speedup / throughput shown by column.
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

from _figure_sizes import FIGSIZE_2X3_2_ROWS, figsize_for_rows
from _report_common import load_pyplot, normalize_choices, positive_int
from _scaling_common import unique_positive_ints


DEFAULT_INPUT_CSV = SCRIPT_DIR / "results_all.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
DEFAULT_CASES = ("sine", "cosine")
DEFAULT_DIMS = (2, 3)
DEFAULT_MODE_KEY = "mg_v_exact"


@dataclass(frozen=True)
class PlotRow:
    dimension: int
    case: str
    cells: int
    cpu_time_s: float
    omp_best_time_s: float
    cuda_time_s: float
    omp_speedup_vs_cpu: float
    cuda_speedup_vs_cpu: float
    cuda_speedup_vs_omp_best: float
    cpu_throughput_cells_per_s: float
    omp_best_throughput_cells_per_s: float
    cuda_throughput_cells_per_s: float


def _parse_float(row: Mapping[str, str], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"row is missing {key}")
    return float(value)


def _load_rows(
    csv_path: Path,
    *,
    case: str,
    dims: Sequence[int],
) -> list[PlotRow]:
    rows: list[PlotRow] = []
    seen: set[tuple[int, int]] = set()

    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for raw in reader:
            if str(raw.get("case", "")) != case:
                continue

            try:
                dimension = positive_int(raw["dimension"], "dimension")
            except Exception:
                continue
            if dimension not in dims:
                continue

            mode_key = str(raw.get("mode_key", ""))
            if mode_key and mode_key != DEFAULT_MODE_KEY:
                continue
            mode = str(raw.get("mode", ""))
            if mode and mode != "MG V exact":
                continue

            try:
                cells = positive_int(raw["cells"], "cells")
                cpu_time_s = _parse_float(raw, "cpu_time_s")
                omp_best_time_s = _parse_float(raw, "omp_best_time_s")
                cuda_time_s = _parse_float(raw, "cuda_time_s")
                omp_speedup_vs_cpu = _parse_float(raw, "omp_speedup_vs_cpu")
                cuda_speedup_vs_cpu = _parse_float(raw, "cuda_speedup_vs_cpu")
                cuda_speedup_vs_omp_best = _parse_float(raw, "cuda_speedup_vs_omp_best")
                cpu_throughput = _parse_float(raw, "cpu_throughput_cells_per_s")
                omp_best_throughput = _parse_float(raw, "omp_best_throughput_cells_per_s")
                cuda_throughput = _parse_float(raw, "cuda_throughput_cells_per_s")
            except Exception:
                continue

            key = (dimension, cells)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                PlotRow(
                    dimension=dimension,
                    case=case,
                    cells=cells,
                    cpu_time_s=cpu_time_s,
                    omp_best_time_s=omp_best_time_s,
                    cuda_time_s=cuda_time_s,
                    omp_speedup_vs_cpu=omp_speedup_vs_cpu,
                    cuda_speedup_vs_cpu=cuda_speedup_vs_cpu,
                    cuda_speedup_vs_omp_best=cuda_speedup_vs_omp_best,
                    cpu_throughput_cells_per_s=cpu_throughput,
                    omp_best_throughput_cells_per_s=omp_best_throughput,
                    cuda_throughput_cells_per_s=cuda_throughput,
                )
            )

    rows.sort(key=lambda row: (row.dimension, row.cells))
    return rows


def _series(rows: Iterable[PlotRow], *, dimension: int) -> list[PlotRow]:
    ordered = [row for row in rows if row.dimension == dimension]
    ordered.sort(key=lambda row: row.cells)
    return ordered


def _plot_case(
    *,
    rows: Sequence[PlotRow],
    case: str,
    dims: Sequence[int],
    output_path: Path,
) -> None:
    plt = load_pyplot()

    runtime_style = {
        "cpu": ("CPU (OMP 1t)", "#7f7f7f", "o", "-"),
        "omp": ("OpenMP best", "#1f77b4", "s", "--"),
        "cuda": ("CUDA", "#d62728", "^", "-"),
    }
    speedup_style = {
        "cpu": ("OpenMP best / CPU", "#1f77b4", "s", "--"),
        "omp": ("CUDA / OMP best", "#d62728", "^", "-"),
        "cuda": ("CUDA / CPU", "#2ca02c", "D", ":"),
    }

    fig, axes = plt.subplots(
        nrows=len(dims),
        ncols=3,
        figsize=figsize_for_rows(FIGSIZE_2X3_2_ROWS, len(dims)),
        sharex=True,
    )
    if len(dims) == 1:
        axes = [axes]  # type: ignore[list-item]

    for row_idx, dimension in enumerate(dims):
        ordered_rows = _series(rows, dimension=dimension)
        if not ordered_rows:
            continue

        cells = [row.cells for row in ordered_rows]
        runtime_series = {
            "cpu": [row.cpu_time_s for row in ordered_rows],
            "omp": [row.omp_best_time_s for row in ordered_rows],
            "cuda": [row.cuda_time_s for row in ordered_rows],
        }
        speedup_series = {
            "cpu": [row.omp_speedup_vs_cpu for row in ordered_rows],
            "omp": [row.cuda_speedup_vs_omp_best for row in ordered_rows],
            "cuda": [row.cuda_speedup_vs_cpu for row in ordered_rows],
        }
        throughput_series = {
            "cpu": [row.cpu_throughput_cells_per_s for row in ordered_rows],
            "omp": [row.omp_best_throughput_cells_per_s for row in ordered_rows],
            "cuda": [row.cuda_throughput_cells_per_s for row in ordered_rows],
        }

        panel_specs = (
            ("Runtime (s)", "runtime"),
            ("Speedup", "speedup"),
            ("Throughput (cells/s)", "throughput"),
        )

        for col_idx, (ylabel, panel_key) in enumerate(panel_specs):
            ax = axes[row_idx][col_idx]

            if panel_key == "runtime":
                series_map = runtime_series
                style_map = runtime_style
                for key in ("cpu", "omp", "cuda"):
                    ax.loglog(
                        cells,
                        series_map[key],
                        color=style_map[key][1],
                        marker=style_map[key][2],
                        linestyle=style_map[key][3],
                        linewidth=1.8,
                        label=style_map[key][0],
                    )
            elif panel_key == "speedup":
                series_map = speedup_series
                style_map = speedup_style
                for key in ("cpu", "omp", "cuda"):
                    ax.semilogx(
                        cells,
                        series_map[key],
                        color=style_map[key][1],
                        marker=style_map[key][2],
                        linestyle=style_map[key][3],
                        linewidth=1.8,
                        label=style_map[key][0],
                    )
            else:
                series_map = throughput_series
                style_map = runtime_style
                for key in ("cpu", "omp", "cuda"):
                    ax.loglog(
                        cells,
                        series_map[key],
                        color=style_map[key][1],
                        marker=style_map[key][2],
                        linestyle=style_map[key][3],
                        linewidth=1.8,
                        label=style_map[key][0],
                    )

            ax.set_title(f"{dimension}D {ylabel}")
            ax.set_ylabel(ylabel)
            ax.set_xlabel("cells" if row_idx == len(dims) - 1 else "")
            ax.tick_params(labelbottom=row_idx == len(dims) - 1)
            ax.legend(loc="best", frameon=True, framealpha=0.88, facecolor="white", edgecolor="0.85", fontsize=9)
            ax.grid(True, which="both", linestyle="--", alpha=0.45)

    fig.suptitle(
        (
            f"CUDA GPU scaling - {case}\n"
            f"2D and 3D rows, MG V exact, CPU=OMP 1 thread, OpenMP=best thread, CUDA"
        ),
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot CUDA GPU scaling figures from an existing results_all.csv.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Source results_all.csv file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for plot outputs.")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES), help="Cases to include.")
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

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = tuple(normalize_choices(args.cases, valid=DEFAULT_CASES, name="case"))
    dims = unique_positive_ints(args.dims, name="dimension")
    dims = tuple(normalize_choices(dims, valid=DEFAULT_DIMS, name="dimension"))

    for case in cases:
        rows = _load_rows(args.input_csv, case=case, dims=dims)
        if not rows:
            raise RuntimeError("No rows matched the requested case/dimension filters.")

        output_path = output_dir / f"plots_2d_3d_{case}.png"
        _plot_case(rows=rows, case=case, dims=dims, output_path=output_path)
        print(f"Wrote {output_path}")

    print(f"Input CSV: {args.input_csv}")
    print(f"Cases: {', '.join(cases)}")
    print(f"Dimensions: {', '.join(str(dim) for dim in dims)}")


if __name__ == "__main__":
    main()
