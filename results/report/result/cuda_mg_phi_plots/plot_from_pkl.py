from __future__ import annotations

"""Re-render CUDA MG phi plots from a cached PKL archive.

This script only reads ``phi_results.pkl`` and redraws the stored exact/MG/
difference images with a tighter layout than the original report generator.
It is useful when you want to tweak spacing, title placement, or colorbar
geometry without rerunning the CUDA solver.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR.parent
if str(RESULT_ROOT) not in sys.path:
    sys.path.insert(0, str(RESULT_ROOT))

from _report_common import load_pyplot


DEFAULT_INPUT_NAME = "phi_results.pkl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "plots"
DEFAULT_DPI = 300

# Compact layout tuned for report figures and slide inserts.
FIGSIZE = (11, 3.75)
FIG_LEFT = 0.035
FIG_RIGHT = 0.985
FIG_BOTTOM = 0.11
FIG_TOP = 0.86
OUTER_GROUP_WSPACE = 0.12
LEFT_GROUP_WSPACE = 0.02
RIGHT_GROUP_WSPACE = 0.14
OUTER_WIDTH_RATIOS = (2.1, 1.05)
LEFT_WIDTH_RATIOS = (1.0, 1.0, 0.045)
RIGHT_WIDTH_RATIOS = (1.0, 0.045)
AX_TITLE_PAD = 4
AX_TITLE_SIZE = 11
AX_LABEL_SIZE = 8
TICK_LABEL_SIZE = 9
SAVE_PAD_INCHES = 0.03


def _extract_payload(payload: object) -> tuple[dict[str, object], list[dict[str, object]]]:
    if isinstance(payload, dict):
        metadata = {key: value for key, value in payload.items() if key != "records"}
        records = payload.get("records")
        if not isinstance(records, list):
            raise TypeError("PKL payload field 'records' must be a list.")
        return metadata, records

    if isinstance(payload, list):
        return {}, payload

    raise TypeError("Unsupported PKL payload format.")


def _omega_tag(value: object) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def _plot_name(record: dict[str, object]) -> str:
    plot_path = record.get("plot_path")
    if plot_path:
        return Path(str(plot_path)).name

    return (
        f"phi_{int(record['dimension'])}d_{str(record['cycle'])}_sor_{str(record['case'])}_n{int(record['grid_size'])}"
        f"_nu{int(record['nu'])}_w{_omega_tag(record['omega'])}_cs{int(record['coarse_steps'])}.png"
    )


def _format_main_title(record: dict[str, object]) -> str:
    cycle = str(record["cycle"]).upper()
    dim = int(record["dimension"])
    case = str(record["case"])
    grid_size = int(record["grid_size"])
    title = f"CUDA MG {cycle}-SOR - {dim}D {case}, n={grid_size}"

    slice_index = record.get("slice_index")
    slice_z = record.get("slice_z")
    if slice_index is not None and slice_z is not None:
        title += f", z={float(slice_z):.3f} (idx {int(slice_index)})"

    return title


def _plot_record(record: dict[str, object], output_dir: Path, dpi: int) -> Path:
    plt = load_pyplot()

    exact_image = np.asarray(record["exact_image"])
    mg_image = np.asarray(record["mg_image"])
    diff_image = mg_image - exact_image

    field_vmin = float(min(np.min(exact_image), np.min(mg_image)))
    field_vmax = float(max(np.max(exact_image), np.max(mg_image)))
    if field_vmin == field_vmax:
        field_vmin -= 1.0
        field_vmax += 1.0

    diff_absmax = float(np.max(np.abs(diff_image)))
    if diff_absmax <= 0.0:
        diff_absmax = float(np.finfo(np.float64).eps)

    fig = plt.figure(figsize=FIGSIZE)
    fig.subplots_adjust(left=FIG_LEFT, right=FIG_RIGHT, bottom=FIG_BOTTOM, top=FIG_TOP)
    # Split the layout into a left group and a right group so we can tune the
    # spacing within each group independently.
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=OUTER_WIDTH_RATIOS,
        wspace=OUTER_GROUP_WSPACE,
    )
    left_grid = outer[0, 0].subgridspec(
        1,
        3,
        width_ratios=LEFT_WIDTH_RATIOS,
        wspace=LEFT_GROUP_WSPACE,
    )
    right_grid = outer[0, 1].subgridspec(
        1,
        2,
        width_ratios=RIGHT_WIDTH_RATIOS,
        wspace=RIGHT_GROUP_WSPACE,
    )

    ax_exact = fig.add_subplot(left_grid[0, 0])
    ax_mg = fig.add_subplot(left_grid[0, 1], sharey=ax_exact)
    cax_shared = fig.add_subplot(left_grid[0, 2])
    ax_diff = fig.add_subplot(right_grid[0, 0], sharey=ax_exact)
    cax_diff = fig.add_subplot(right_grid[0, 1])

    image_titles = ("Exact", f"CUDA MG {str(record['cycle']).upper()}-SOR", "Difference (MG - exact)")
    image_data = (
        (exact_image, "viridis", field_vmin, field_vmax),
        (mg_image, "viridis", field_vmin, field_vmax),
        (diff_image, "coolwarm", -diff_absmax, diff_absmax),
    )

    axes = (ax_exact, ax_mg, ax_diff)
    mappables: list[object] = []
    for index, (ax, title, (data, cmap, vmin, vmax)) in enumerate(zip(axes, image_titles, image_data)):
        im = ax.imshow(
            data,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=AX_TITLE_SIZE, pad=AX_TITLE_PAD)
        ax.set_xlabel("x", fontsize=AX_LABEL_SIZE, labelpad=2)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)
        ax.set_aspect("equal")
        if index == 0:
            ax.set_ylabel("y", fontsize=AX_LABEL_SIZE, labelpad=2)
        else:
            # Share the y-axis but only print one set of tick labels.
            ax.tick_params(labelleft=False)
        mappables.append(im)

    cbar_shared = fig.colorbar(
        mappables[0],
        cax=cax_shared,
    )
    cbar_shared.ax.tick_params(labelsize=TICK_LABEL_SIZE)
    cbar_diff = fig.colorbar(
        mappables[2],
        cax=cax_diff,
    )
    cbar_diff.ax.tick_params(labelsize=TICK_LABEL_SIZE)

    fig.suptitle(_format_main_title(record), y=0.98, fontsize=12)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / _plot_name(record)
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)

    return plot_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-render CUDA MG phi plots from phi_results.pkl.")
    parser.add_argument(
        "--input-pkl",
        type=Path,
        default=SCRIPT_DIR / DEFAULT_INPUT_NAME,
        help="Path to the cached phi_results.pkl file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the regenerated plots will be written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Output image DPI.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    with args.input_pkl.open("rb") as stream:
        payload = pickle.load(stream)

    metadata, records = _extract_payload(payload)
    if not records:
        raise RuntimeError(f"No records were found in {args.input_pkl}")

    print(
        f"Loaded {len(records)} records from {args.input_pkl} "
        f"(generated_at_utc={metadata.get('generated_at_utc', 'unknown')})",
        flush=True,
    )

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise TypeError("Each PKL record must be a dictionary.")

        plot_path = _plot_record(record, args.output_dir, args.dpi)
        print(
            f"[{index}/{len(records)}] Wrote {plot_path}",
            flush=True,
        )

        # Release the large cached arrays as soon as they are no longer needed.
        record.pop("exact_image", None)
        record.pop("mg_image", None)


if __name__ == "__main__":
    main()
