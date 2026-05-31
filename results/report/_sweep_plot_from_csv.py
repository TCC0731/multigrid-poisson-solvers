from __future__ import annotations

"""Shared plotting helpers for CUDA MG sweep ``results_all.csv`` files."""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR
if str(RESULT_ROOT) not in sys.path:
    sys.path.insert(0, str(RESULT_ROOT))

from _figure_sizes import FIGSIZE_2X2_2_ROWS, figsize_for_rows
from _report_common import finalize_figure_header, load_pyplot, normalize_choices, positive_float, positive_int
from _sweep_common import DEFAULT_DIMS, _format_sweep_value


DEFAULT_BACKEND = "cuda"
DEFAULT_DTYPE = "double"


@dataclass(frozen=True)
class ModeSpec:
    label: str
    mode_key: str
    color: str
    marker: str
    linestyle: str


MODE_SPECS = (
    ModeSpec(
        label="V-exact",
        mode_key="v_exact",
        color="#1f77b4",
        marker="o",
        linestyle="-",
    ),
    ModeSpec(
        label="V-SOR",
        mode_key="v_sor",
        color="#1f77b4",
        marker="s",
        linestyle="--",
    ),
    ModeSpec(
        label="W-exact",
        mode_key="w_exact",
        color="#d62728",
        marker="o",
        linestyle="-",
    ),
    ModeSpec(
        label="W-SOR",
        mode_key="w_sor",
        color="#d62728",
        marker="s",
        linestyle="--",
    ),
)
MODE_ORDER = {spec.mode_key: index for index, spec in enumerate(MODE_SPECS)}
MODE_SPECS_BY_KEY = {spec.mode_key: spec for spec in MODE_SPECS}


@dataclass(frozen=True)
class PlotRow:
    dimension: int
    case: str
    grid_size: int
    mode_key: str
    sweep_value: float | int
    iterations: int
    time_s: float


def _sweep_name_to_label(sweep_name: str) -> str:
    if sweep_name == "omega":
        return "omega"
    if sweep_name == "nu":
        return "nu"
    raise ValueError(f"unsupported sweep name: {sweep_name!r}")


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


def _parse_time_s(row: Mapping[str, str]) -> float:
    if row.get("time_s"):
        return float(row["time_s"])
    if row.get("time_ms"):
        return float(row["time_ms"]) / 1000.0
    raise ValueError("row is missing both time_s and time_ms")


def _load_rows(
    csv_path: Path,
    *,
    case: str,
    dims: Sequence[int],
    sweep_name: str,
) -> list[PlotRow]:
    rows: list[PlotRow] = []
    seen: set[tuple[int, str, float | int]] = set()

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
            if mode_key not in MODE_SPECS_BY_KEY:
                continue

            try:
                grid_size = positive_int(raw["grid_size"], "grid_size")
                iterations = positive_int(raw["iterations"], "iterations")
                time_s = _parse_time_s(raw)
            except Exception:
                continue

            if sweep_name == "omega":
                try:
                    sweep_value: float | int = positive_float(raw["omega"], "omega")
                except Exception:
                    continue
            elif sweep_name == "nu":
                try:
                    sweep_value = positive_int(raw["nu"], "nu")
                except Exception:
                    continue
            else:
                raise ValueError(f"unsupported sweep name: {sweep_name!r}")

            key = (dimension, mode_key, sweep_value)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                PlotRow(
                    dimension=dimension,
                    case=case,
                    grid_size=grid_size,
                    mode_key=mode_key,
                    sweep_value=sweep_value,
                    iterations=iterations,
                    time_s=time_s,
                )
            )

    rows.sort(
        key=lambda row: (
            row.dimension,
            MODE_ORDER[row.mode_key],
            row.sweep_value,
        )
    )
    return rows


def _series(rows: Iterable[PlotRow], *, dimension: int, mode_key: str) -> list[PlotRow]:
    ordered = [row for row in rows if row.dimension == dimension and row.mode_key == mode_key]
    ordered.sort(key=lambda row: row.sweep_value)
    return ordered


def _grid_description(rows: Sequence[PlotRow], dims: Sequence[int]) -> str:
    parts: list[str] = []
    for dim in dims:
        dim_rows = [row for row in rows if row.dimension == dim]
        if not dim_rows:
            continue
        parts.append(f"{dim}D grid={dim_rows[0].grid_size}")
    return ", ".join(parts)


def _expanded_limits(values: Sequence[float | int]) -> tuple[float, float]:
    ordered = sorted({float(value) for value in values})
    if not ordered:
        raise ValueError("at least one value is required")
    if len(ordered) == 1:
        center = ordered[0]
        pad = max(abs(center) * 0.05, 0.1)
        return center - pad, center + pad

    gaps = [right - left for left, right in zip(ordered, ordered[1:]) if right > left]
    if gaps:
        pad = min(gaps) * 0.5
    else:
        span = ordered[-1] - ordered[0]
        pad = max(span * 0.05, 0.5)
    return ordered[0] - pad, ordered[-1] + pad


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
        x = [row.sweep_value for row in series]
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


def plot_iter_time_2x2_from_csv(
    *,
    input_csv: Path,
    output_path: Path,
    case: str = "sine",
    dims: Sequence[int] = DEFAULT_DIMS,
    sweep_name: str,
) -> None:
    case = str(case).strip()
    dims = _unique_positive_ints(dims, name="dimension")
    dims = tuple(normalize_choices(dims, valid=DEFAULT_DIMS, name="dimension"))
    sweep_name = _sweep_name_to_label(sweep_name)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    rows = _load_rows(input_csv, case=case, dims=dims, sweep_name=sweep_name)
    if not rows:
        raise RuntimeError("No rows matched the requested filters.")

    plt = load_pyplot()
    sweep_values = sorted({row.sweep_value for row in rows})
    sweep_labels = [_format_sweep_value(value, sweep_name) for value in sweep_values]
    x_rotation = 45 if sweep_name == "omega" else 0
    x_min, x_max = _expanded_limits(sweep_values)

    fig, axes = plt.subplots(
        nrows=len(dims),
        ncols=2,
        figsize=figsize_for_rows(FIGSIZE_2X2_2_ROWS, len(dims)),
        sharex=True,
    )
    if len(dims) == 1:
        axes = [axes]  # type: ignore[list-item]

    panels = (
        ("iterations", "Iterations"),
        ("time_s", "Time (s)"),
    )

    for row_idx, dimension in enumerate(dims):
        for col_idx, (metric_key, title) in enumerate(panels):
            ax = axes[row_idx][col_idx]
            _plot_mode_series(ax, rows=rows, dimension=dimension, metric_key=metric_key)

            ax.set_title(f"{dimension}D {title}")
            ax.set_ylabel(title)
            if row_idx == len(dims) - 1:
                ax.set_xlabel(_sweep_name_to_label(sweep_name))
                ax.set_xticks(sweep_values)
                ax.set_xticklabels(
                    sweep_labels,
                    rotation=x_rotation,
                    ha="right" if x_rotation else "center",
                )
            else:
                ax.set_xlabel("")
                ax.set_xticks(sweep_values)
            ax.grid(True, linestyle="--", alpha=0.45)
            ax.set_xlim(x_min, x_max)

    handles, labels = axes[0][0].get_legend_handles_labels()
    finalize_figure_header(
        fig,
        title=f"CUDA MG {sweep_name} sweep - {case} {_grid_description(rows, dims)}",
        handles=handles,
        labels=labels,
        ncol=4,
        legend_y=0.965,
        title_y=0.995,
        tight_top=0.99,
        legend_kwargs={
            "fontsize": 9.0,
            "columnspacing": 1.2,
            "handletextpad": 0.6,
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_iter_time_2x2(
    *,
    rows: Sequence[PlotRow],
    case: str,
    dims: Sequence[int] = DEFAULT_DIMS,
    sweep_name: str,
    output_path: Path,
) -> None:
    """Plot helper for callers that already loaded rows from CSV."""

    case = str(case).strip()
    dims = _unique_positive_ints(dims, name="dimension")
    dims = tuple(normalize_choices(dims, valid=DEFAULT_DIMS, name="dimension"))
    sweep_name = _sweep_name_to_label(sweep_name)
    rows = [row for row in rows if row.case == case and row.dimension in dims]
    if not rows:
        raise RuntimeError("No rows matched the requested filters.")

    plt = load_pyplot()
    sweep_values = sorted({row.sweep_value for row in rows})
    sweep_labels = [_format_sweep_value(value, sweep_name) for value in sweep_values]
    x_rotation = 45 if sweep_name == "omega" else 0
    x_min, x_max = _expanded_limits(sweep_values)

    fig, axes = plt.subplots(
        nrows=len(dims),
        ncols=2,
        figsize=figsize_for_rows(FIGSIZE_2X2_2_ROWS, len(dims)),
        sharex=True,
    )
    if len(dims) == 1:
        axes = [axes]  # type: ignore[list-item]

    panels = (
        ("iterations", "Iterations"),
        ("time_s", "Time (s)"),
    )

    for row_idx, dimension in enumerate(dims):
        for col_idx, (metric_key, title) in enumerate(panels):
            ax = axes[row_idx][col_idx]
            _plot_mode_series(ax, rows=rows, dimension=dimension, metric_key=metric_key)

            ax.set_title(f"{dimension}D {title}")
            ax.set_ylabel(title)
            if row_idx == len(dims) - 1:
                ax.set_xlabel(_sweep_name_to_label(sweep_name))
                ax.set_xticks(sweep_values)
                ax.set_xticklabels(
                    sweep_labels,
                    rotation=x_rotation,
                    ha="right" if x_rotation else "center",
                )
            else:
                ax.set_xlabel("")
                ax.set_xticks(sweep_values)
            ax.grid(True, linestyle="--", alpha=0.45)
            ax.set_xlim(x_min, x_max)

    handles, labels = axes[0][0].get_legend_handles_labels()
    finalize_figure_header(
        fig,
        title=f"CUDA MG {sweep_name} sweep - {case} (2D/3D iter/time)\n{_grid_description(rows, dims)}",
        handles=handles,
        labels=labels,
        ncol=4,
        legend_y=0.965,
        title_y=0.995,
        tight_top=0.90,
        legend_kwargs={
            "fontsize": 9.0,
            "columnspacing": 1.2,
            "handletextpad": 0.6,
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "PlotRow",
    "plot_iter_time_2x2",
    "plot_iter_time_2x2_from_csv",
]
