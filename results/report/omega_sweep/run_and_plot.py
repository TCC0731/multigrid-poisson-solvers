from __future__ import annotations

"""Generate CUDA MG omega-sweep tables and plots."""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR.parent
if str(RESULT_ROOT) not in sys.path:
    sys.path.insert(0, str(RESULT_ROOT))

from _sweep_common import (
    DEFAULT_CASES,
    DEFAULT_DIMS,
    DEFAULT_GRID_SIZES,
    DEFAULT_MAX_ITER,
    DEFAULT_NU,
    DEFAULT_OMEGA,
    DEFAULT_OMEGA_VALUES,
    DEFAULT_REPEAT_RUNS,
    DEFAULT_TOL,
    normalize_choices,
    positive_float,
    positive_int,
    run_sweep_report,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build CUDA MG omega sweep tables and plots.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR, help="Directory for CSV and plot outputs.")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES), help="Cases to include.")
    parser.add_argument("--dims", nargs="+", type=int, default=list(DEFAULT_DIMS), help="Dimensions to include.")
    parser.add_argument(
        "--omega",
        nargs="+",
        type=float,
        default=list(DEFAULT_OMEGA_VALUES),
        help="Omega values to sweep.",
    )
    parser.add_argument("--nu", type=int, default=DEFAULT_NU, help="Fixed MG nu value.")
    parser.add_argument("--grid-2d", type=int, default=DEFAULT_GRID_SIZES[2], help="2D grid size.")
    parser.add_argument("--grid-3d", type=int, default=DEFAULT_GRID_SIZES[3], help="3D grid size.")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL, help="Convergence tolerance.")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help="Maximum iterations.")
    parser.add_argument(
        "--repeat-runs",
        type=int,
        default=DEFAULT_REPEAT_RUNS,
        help="Repeat runs per solver invocation.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cases = tuple(normalize_choices(args.cases, valid=DEFAULT_CASES, name="case"))
    dims = tuple(normalize_choices(tuple(positive_int(dim, "dim") for dim in args.dims), valid=DEFAULT_DIMS, name="dimension"))
    omega_values = tuple(sorted({positive_float(value, "omega") for value in args.omega}))
    grid_sizes = {
        2: positive_int(args.grid_2d, "grid_2d"),
        3: positive_int(args.grid_3d, "grid_3d"),
    }
    nu = positive_int(args.nu, "nu")
    tol = positive_float(args.tol, "tol")
    max_iter = positive_int(args.max_iter, "max_iter")
    repeat_runs = positive_int(args.repeat_runs, "repeat_runs")

    run_sweep_report(
        sweep_name="omega",
        output_dir=args.output_dir,
        cases=cases,
        dims=dims,
        sweep_values=omega_values,
        grid_sizes=grid_sizes,
        fixed_omega=DEFAULT_OMEGA,
        fixed_nu=nu,
        tol=tol,
        max_iter=max_iter,
        repeat_runs=repeat_runs,
    )


if __name__ == "__main__":
    main()

