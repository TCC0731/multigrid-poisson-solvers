from __future__ import annotations

import argparse
from pathlib import Path

from _analysis_common import (
    DEFAULT_DTYPE_CLI,
    DEFAULT_GRID_SIZES_BY_DIM,
    DEFAULT_MAX_ITER,
    DEFAULT_NU_VALUES,
    DEFAULT_TOL,
    run_analysis,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CUDA MG omega sweeps.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory that will receive the CSV files and plots.",
    )
    parser.add_argument(
        "--nu",
        nargs="+",
        type=int,
        default=list(DEFAULT_NU_VALUES),
        help="One or more nu values to sweep.",
    )
    parser.add_argument(
        "--grid-2d",
        nargs="+",
        type=int,
        default=list(DEFAULT_GRID_SIZES_BY_DIM[2]),
        help="One or more 2D grid sizes to sweep.",
    )
    parser.add_argument(
        "--grid-3d",
        nargs="+",
        type=int,
        default=list(DEFAULT_GRID_SIZES_BY_DIM[3]),
        help="One or more 3D grid sizes to sweep.",
    )
    parser.add_argument(
        "--dtype",
        default=DEFAULT_DTYPE_CLI,
        help="CUDA dtype CLI value passed to the benchmark executable.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        help="Convergence tolerance.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help="Maximum number of iterations.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_analysis(
        output_dir=args.output_dir,
        dtype_cli=args.dtype,
        tol=args.tol,
        max_iter=args.max_iter,
        nu_values=tuple(args.nu),
        grid_sizes_by_dim={2: tuple(args.grid_2d), 3: tuple(args.grid_3d)},
    )


if __name__ == "__main__":
    main()
