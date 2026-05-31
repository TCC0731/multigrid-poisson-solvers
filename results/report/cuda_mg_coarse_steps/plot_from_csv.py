from __future__ import annotations

"""Plot CUDA MG coarse_steps figures from an existing ``results_all.csv``."""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR.parent
if str(RESULT_ROOT) not in sys.path:
    sys.path.insert(0, str(RESULT_ROOT))

from _report_common import normalize_choices, positive_int
from _sweep_common import DEFAULT_CASES, DEFAULT_DIMS
from _coarse_steps_plot_from_csv import plot_iter_time_2x2_from_csv


DEFAULT_INPUT_CSV = SCRIPT_DIR / "results_all.csv"
DEFAULT_OUTPUT_PNG = SCRIPT_DIR / "plots_2d_3d_iter_time_sine.png"
DEFAULT_CASE = "sine"


def _unique_positive_ints(values: list[object], *, name: str) -> tuple[int, ...]:
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot CUDA MG coarse_steps figures from an existing results_all.csv.")
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
    case = str(args.case).strip()
    case = tuple(normalize_choices((case,), valid=DEFAULT_CASES, name="case"))[0]
    dims = _unique_positive_ints(args.dims, name="dimension")
    dims = tuple(normalize_choices(dims, valid=DEFAULT_DIMS, name="dimension"))

    plot_iter_time_2x2_from_csv(
        input_csv=args.input_csv,
        output_path=args.output_path,
        case=case,
        dims=dims,
    )

    print(f"Wrote {args.output_path}")
    print(f"Input CSV: {args.input_csv}")
    print(f"Case: {case}")
    print(f"Dimensions: {', '.join(str(dim) for dim in dims)}")


if __name__ == "__main__":
    main()
