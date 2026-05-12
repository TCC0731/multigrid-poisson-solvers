from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _analysis_common import (
    MG_COMPARE_GRID_SIZES as BASE_MG_COMPARE_GRID_SIZES,
    MG_COMPARE_MODULES,
    run_analysis,
)


MG_COMPARE_GRID_SIZES = {
    name: grid.copy() for name, grid in BASE_MG_COMPARE_GRID_SIZES.items()
}
MG_COMPARE_GRID_SIZES["RB SOR 3D"] = np.array([15, 31, 47, 63])


def main() -> None:
    run_analysis(
        output_dir=Path(__file__).resolve().parent,
        solver_modules=MG_COMPARE_MODULES,
        grid_sizes=MG_COMPARE_GRID_SIZES,
        dtype_cli="float",
        tol=1e-3,
        max_iter=50000,
        plot_suffix="_mg_compare",
        csv_suffix="_mg_compare",
        reference_solver="RB SOR 3D",
        sor_max_iter=1000,
        mg_max_iter=100,
    )


if __name__ == "__main__":
    main()
