from __future__ import annotations

"""Read a cached residual-history PKL and render the comparison plot."""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR.parent
if str(RESULT_ROOT) not in sys.path:
    sys.path.insert(0, str(RESULT_ROOT))

from _history_common import MODE_SPECS, MODE_SPECS_BY_KEY, load_payload
from _report_common import finalize_figure_header, load_pyplot


DEFAULT_INPUT_NAME = "residual_history.pkl"
DEFAULT_OUTPUT_NAME = "residual_history.png"


def _extract_records(payload: object) -> tuple[dict[str, object], list[dict[str, object]]]:
    if isinstance(payload, dict):
        metadata = dict(payload.get("metadata", {}))
        records = payload.get("records", [])
        if not isinstance(records, list):
            raise TypeError("PKL payload field 'records' must be a list.")
        return metadata, [dict(record) for record in records]

    if isinstance(payload, list):
        return {}, [dict(record) for record in payload]

    raise TypeError("Unsupported PKL payload format.")


def _grid_summary(records: list[dict[str, object]]) -> str:
    grid_sizes = {str(record["mode_key"]): int(record["grid_size"]) for record in records}
    unique_sizes = sorted(set(grid_sizes.values()))
    if len(unique_sizes) == 1:
        return f"n={unique_sizes[0]}"

    parts: list[str] = []
    for spec in MODE_SPECS:
        if spec.mode_key not in grid_sizes:
            continue
        parts.append(f"{spec.label} n={grid_sizes[spec.mode_key]}")
    return ", ".join(parts)


def _mode_sort_key(mode_key: str) -> int:
    for index, spec in enumerate(MODE_SPECS):
        if spec.mode_key == mode_key:
            return index
    return len(MODE_SPECS)


def _plot_solver_panel(
    *,
    ax,
    dim: int,
    case: str,
    solver: str,
    records: list[dict[str, object]],
    y_min: float,
    y_max: float,
    legend_handles_by_label: dict[str, object],
) -> None:
    panel_records = [
        record
        for record in records
        if (spec := MODE_SPECS_BY_KEY.get(str(record["mode_key"]))) is not None
        and spec.solver == solver
    ]
    panel_records = sorted(panel_records, key=lambda row: _mode_sort_key(str(row["mode_key"])))
    if not panel_records:
        ax.set_visible(False)
        return

    for row in panel_records:
        mode_key = str(row["mode_key"])
        spec = MODE_SPECS_BY_KEY.get(mode_key)
        if spec is None:
            continue
        history = [float(value) for value in row["history"]]
        if not history:
            continue
        x_values = list(range(len(history)))
        markevery = max(1, len(history) // 12)
        line = ax.plot(
            x_values,
            history,
            color=spec.color,
            linestyle=spec.linestyle,
            marker=spec.marker,
            markevery=markevery,
            markersize=4.0,
            linewidth=1.8,
            label=spec.label,
        )[0]
        if spec.label not in legend_handles_by_label:
            legend_handles_by_label[spec.label] = line

    ax.set_yscale("log")
    ax.set_ylim(y_min * 0.8, y_max * 1.2)
    panel_x_max = max(len(record["history"]) - 1 for record in panel_records)
    ax.set_xlim(0, max(1, panel_x_max))
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    ax.set_xlabel("Iteration")
    ax.set_title(f"{dim}D {case}, {'SOR' if solver == 'sor' else 'MG'}, {_grid_summary(panel_records)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render residual-history plots from a PKL archive.")
    parser.add_argument(
        "--input-pkl",
        type=Path,
        default=SCRIPT_DIR / DEFAULT_INPUT_NAME,
        help="Path to the residual_history.pkl file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=SCRIPT_DIR / DEFAULT_OUTPUT_NAME,
        help="Destination PNG path.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = load_payload(args.input_pkl)
    metadata, records = _extract_records(payload)
    if not records:
        raise RuntimeError(f"No records were found in {args.input_pkl}")

    dims = sorted({int(record["dimension"]) for record in records})
    case = str(metadata.get("case") or records[0].get("case", "sine"))
    omp_num_threads = int(metadata.get("omp_num_threads", 1))
    tol = float(metadata.get("tol", 1e-9))
    repeat_runs = int(metadata.get("repeat_runs", 1))

    all_positive = [float(value) for record in records for value in record["history"] if float(value) > 0.0]
    if not all_positive:
        raise RuntimeError("Residual histories do not contain any positive values.")
    y_min = min(all_positive)
    y_max = max(all_positive)
    plt = load_pyplot()
    fig, axes = plt.subplots(
        len(dims),
        2,
        figsize=(16.8, 4.4 * len(dims)),
        sharey=True,
        squeeze=False,
    )
    legend_handles_by_label: dict[str, object] = {}

    for row_idx, dim in enumerate(dims):
        dim_records = [record for record in records if int(record["dimension"]) == dim]
        if not dim_records:
            for ax in axes[row_idx]:
                ax.set_visible(False)
            continue

        for col_idx, solver in enumerate(("sor", "mg")):
            ax = axes[row_idx][col_idx]
            _plot_solver_panel(
                ax=ax,
                dim=dim,
                case=case,
                solver=solver,
                records=dim_records,
                y_min=y_min,
                y_max=y_max,
                legend_handles_by_label=legend_handles_by_label,
            )
            if col_idx == 0:
                ax.set_ylabel("Residual L2")
            else:
                ax.tick_params(labelleft=False)

    legend_labels = [spec.label for spec in MODE_SPECS if spec.label in legend_handles_by_label]
    handles = [legend_handles_by_label[label] for label in legend_labels]
    finalize_figure_header(
        fig,
        title=(
            f"OMP residual history - case={case}, tol={tol:.0e}, "
            f"OMP_NUM_THREADS={omp_num_threads}, repeat_runs={repeat_runs}"
        ),
        handles=handles,
        labels=legend_labels,
        ncol=max(1, len(MODE_SPECS)),
        tight_top=0.86,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()
