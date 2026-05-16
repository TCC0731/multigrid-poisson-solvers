from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _analysis_common import (
    MG_COMPARE_MODULES,
    MG_GRID_SIZES,
    RB_SOR_GRID_SIZES,
    run_analysis,
)


MG_COMPARE_GRID_SIZES = {
    name: MG_GRID_SIZES[:8].copy() for name in MG_COMPARE_MODULES
}
MG_COMPARE_GRID_SIZES["RB SOR 3D"] = np.insert(RB_SOR_GRID_SIZES, 0, 15)


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
        mg_max_iter=20,
    )


if __name__ == "__main__":
    main()
