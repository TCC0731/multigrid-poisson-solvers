from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _analysis_common import MG_COMPARE_GRID_SIZES, MG_COMPARE_MODULES, run_analysis


def main() -> None:
    run_analysis(
        output_dir=Path(__file__).resolve().parent,
        solver_modules=MG_COMPARE_MODULES,
        grid_sizes=MG_COMPARE_GRID_SIZES,
        dtype_cli="float",
        tol=1e-3,
        max_iter=20000,
        plot_suffix="_mg_compare",
        csv_suffix="_mg_compare",
        reference_solver="MG 3D(v, nu=3, w=1.25)",
        sor_max_iter=1000,
        mg_max_iter=20,
    )


if __name__ == "__main__":
    main()
