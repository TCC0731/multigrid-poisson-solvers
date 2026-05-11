from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _analysis_common import (
    SOLVER_COMPARISON_MODULES,
    SOLVER_COMPARISON_GRID_SIZES,
    run_analysis,
)


def main() -> None:
    run_analysis(
        output_dir=Path(__file__).resolve().parent,
        solver_modules=SOLVER_COMPARISON_MODULES,
        grid_sizes=SOLVER_COMPARISON_GRID_SIZES,
        dtype=np.float64,
        tol=1e-10,
        max_iter=50000,
        plot_suffix="",
        csv_suffix="",
        reference_solver="Jacobi 3D",
    )


if __name__ == "__main__":
    main()
