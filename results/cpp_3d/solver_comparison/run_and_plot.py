from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _analysis_common import (
    SOLVER_COMPARISON_GRID_SIZES as BASE_SOLVER_COMPARISON_GRID_SIZES,
    SOLVER_COMPARISON_MODULES,
    run_analysis,
)


SOLVER_COMPARISON_GRID_SIZES = {
    name: grid.copy() for name, grid in BASE_SOLVER_COMPARISON_GRID_SIZES.items()
}
SOLVER_COMPARISON_GRID_SIZES["RB SOR 3D"] = np.array(
    [15, 23, 31, 47, 63, 79, 95, 111, 127, 143, 159]
)
SOLVER_COMPARISON_GRID_SIZES["MG V-cycle 3D"] = np.array(
    [15, 31, 47, 63, 95, 127, 191, 255]
)
SOLVER_COMPARISON_GRID_SIZES["MG W-cycle 3D"] = np.array(
    [15, 31, 47, 63, 95, 127, 191, 255]
)


def main() -> None:
    run_analysis(
        output_dir=Path(__file__).resolve().parent,
        solver_modules=SOLVER_COMPARISON_MODULES,
        grid_sizes=SOLVER_COMPARISON_GRID_SIZES,
        dtype_cli="double",
        tol=1e-10,
        max_iter=50000,
        plot_suffix="",
        csv_suffix="",
        reference_solver="MG V-cycle 3D",
        sor_max_iter=1000,
        mg_max_iter=100,
    )


if __name__ == "__main__":
    main()
