from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _analysis_common import (
    MG_COMPARE_GRID_SIZES,
    MG_COMPARE_MODULES,
    run_analysis,
)


def main() -> None:
    run_analysis(
        output_dir=Path(__file__).resolve().parent,
        solver_modules=MG_COMPARE_MODULES,
        grid_sizes=MG_COMPARE_GRID_SIZES,
        dtype=np.float64,
        tol=1e-10,
        max_iter=50000,
        plot_suffix="_mg_compare",
        csv_suffix="_mg_compare",
        reference_solver="RB SOR 3D",
    )


if __name__ == "__main__":
    main()
