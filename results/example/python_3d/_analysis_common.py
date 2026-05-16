from __future__ import annotations

import csv
import itertools
import sys
from importlib import import_module
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULT_ROOT = Path(__file__).resolve().parent
PYTHON_DIR = RESULT_ROOT.parents[1] / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from metrics import error_metrics_3d
from problems import CASES_3D, make_problem_3d


DEFAULT_MARKERS = ("o", "s", "^", "D", "v", "p", "*", "h", "H", "<", ">", "P", "X", "d")

# Keep the same endpoints as python/run_benchmark_3d.py, but add intermediate
# odd grid sizes so analysis plots have smoother trends than the timing suite.
JACOBI_GS_GRID_SIZES = np.array([15, 23, 31, 39, 47, 55, 63])
RB_SOR_GRID_SIZES = np.array([31, 47, 63, 79, 95, 111, 127, 143, 159])
MG_GRID_SIZES = np.array([15, 31, 47, 63, 95, 127, 191, 255, 319, 383])

SOLVER_COMPARISON_MODULES = {
    "Jacobi 3D": ("solvers.jacobi_3d", {}),
    "RB GS 3D": ("solvers.gs_3d", {}),
    "RB SOR 3D": ("solvers.sor_3d", {}),
    "MG V-cycle 3D": ("solvers.mg_3d", {"cycle": "v"}),
    "MG W-cycle 3D": ("solvers.mg_3d", {"cycle": "w"}),
}

MG_COMPARE_MODULES = {
    "RB SOR 3D": ("solvers.sor_3d", {}),
}
for cycle in ("v", "w"):
    for omega, omega_name in ((1.25, "1.25"), (None, "auto")):
        name = f"MG 3D({cycle}, nu=3, w={omega_name})"
        MG_COMPARE_MODULES[name] = (
            "solvers.mg_3d",
            {"cycle": cycle, "nu": 3, "omega": omega},
        )


SOLVER_COMPARISON_GRID_SIZES = {
    "Jacobi 3D": JACOBI_GS_GRID_SIZES,
    "RB GS 3D": JACOBI_GS_GRID_SIZES,
    "RB SOR 3D": RB_SOR_GRID_SIZES,
    "MG V-cycle 3D": MG_GRID_SIZES,
    "MG W-cycle 3D": MG_GRID_SIZES,
}

MG_COMPARE_GRID_SIZES = {
    name: MG_GRID_SIZES for name in MG_COMPARE_MODULES
}
MG_COMPARE_GRID_SIZES["RB SOR 3D"] = RB_SOR_GRID_SIZES


def load_solver(module_name: str):
    module = import_module(module_name)
    return getattr(module, "solve")


def build_markers(names):
    markers = itertools.cycle(DEFAULT_MARKERS)
    return {name: next(markers) for name in names}


def make_empty_results(solver_names):
    return {
        name: {
            "N": [],
            "iterations": [],
            "time_s": [],
            "error_l2": [],
            "error_linf": [],
        }
        for name in solver_names
    }


def compute_error(case: str, grid_size: int, dtype, problem, phi):
    if np.dtype(dtype) == np.dtype(np.float32):
        problem_f64 = make_problem_3d(case=case, grid_size=grid_size, dtype=np.float64)
        return error_metrics_3d(problem_f64, phi.astype(np.float64))
    return error_metrics_3d(problem, phi)


def run_case(
    *,
    case: str,
    solver_modules: dict[str, tuple[str, dict[str, object]]],
    grid_sizes,
    dtype,
    tol: float,
    max_iter: int,
    sor_max_iter: int = 1000,
    mg_max_iter: int = 100,
):
    results = make_empty_results(solver_modules)

    for name, (module_name, kwargs) in solver_modules.items():
        solve = load_solver(module_name)
        print(f"Running {name}...")
        solver_grid_sizes = grid_sizes[name] if isinstance(grid_sizes, dict) else grid_sizes

        for grid_size in solver_grid_sizes:
            print(f"  N={grid_size}")
            problem = make_problem_3d(case=case, grid_size=int(grid_size), dtype=dtype)

            solve(problem, tol=1e-2, max_iter=2, **kwargs)

            start = perf_counter()
            if "MG" in name:
                current_max_iter = mg_max_iter
            elif "SOR" in name:
                current_max_iter = sor_max_iter
            else:
                current_max_iter = max_iter
            phi, iterations, _res_l2 = solve(
                problem,
                tol=tol,
                max_iter=current_max_iter,
                **kwargs,
            )
            time_s = perf_counter() - start

            error_l2, error_linf = compute_error(case, int(grid_size), dtype, problem, phi)

            results[name]["N"].append(int(grid_size))
            results[name]["iterations"].append(int(iterations))
            results[name]["time_s"].append(float(time_s))
            results[name]["error_l2"].append(float(error_l2))
            results[name]["error_linf"].append(float(error_linf))

    return results


def plot_results(
    *,
    case: str,
    results,
    markers,
    output_path: Path,
    reference_solver: str,
):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    ax = axs[0, 0]
    for name, res in results.items():
        ax.loglog(
            res["N"],
            res["error_l2"],
            marker=markers[name],
            linestyle="",
            alpha=0.7,
            label=name,
        )
    n_fit = np.array(
        [
            min(results[reference_solver]["N"]),
            max(results[reference_solver]["N"]),
        ]
    )
    h_fit = 1.0 / (n_fit + 1)
    err_fit = results[reference_solver]["error_l2"][0] * (h_fit / h_fit[0]) ** 2
    ax.loglog(n_fit, err_fit, "--", color="gray", label="Fit (Order=2.00)")
    ax.set_title(f"$L_2$ Error vs N ({case}, 3D)")
    ax.set_xlabel("N")
    ax.set_ylabel("$L_2$ Error Norm")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize="small")

    ax = axs[0, 1]
    for name, res in results.items():
        ax.loglog(
            res["N"],
            res["error_linf"],
            marker=markers[name],
            linestyle="",
            alpha=0.7,
            label=name,
        )
    err_fit_inf = results[reference_solver]["error_linf"][0] * (h_fit / h_fit[0]) ** 2
    ax.loglog(n_fit, err_fit_inf, "--", color="gray", label="Fit (Order=2.00)")
    ax.set_title(fr"$L_\infty$ Error vs N ({case}, 3D)")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$L_\infty$ Error Norm")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize="small")

    ax = axs[1, 0]
    for name, res in results.items():
        n_arr = np.array(res["N"])
        it_arr = np.array(res["iterations"])
        if len(n_arr) > 1 and np.all(it_arr > 0):
            coeffs = np.polyfit(np.log(n_arr), np.log(it_arr), 1)
            label = f"{name} (~$N^{{{coeffs[0]:.2f}}}$)"
        else:
            label = name
        ax.loglog(
            res["N"],
            res["iterations"],
            marker=markers[name],
            linestyle="--",
            alpha=0.7,
            label=label,
        )
    ax.set_title(f"Iterations vs N ({case}, 3D)")
    ax.set_xlabel("N")
    ax.set_ylabel("Iterations")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize="small", ncol=2)

    ax = axs[1, 1]
    for name, res in results.items():
        ax.loglog(
            res["N"],
            res["time_s"],
            marker=markers[name],
            linestyle="-",
            alpha=0.7,
            label=name,
        )
    ax.set_title(f"Time vs N ({case}, 3D)")
    ax.set_xlabel("N")
    ax.set_ylabel("Execution Time (s)")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize="small", ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_results_csv(path: Path, results) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Solver", "N", "Iterations", "Time_s", "L2_Error", "Linf_Error"])
        for name, res in results.items():
            for i in range(len(res["N"])):
                writer.writerow(
                    [
                        name,
                        res["N"][i],
                        res["iterations"][i],
                        f"{res['time_s'][i]:.6e}",
                        f"{res['error_l2'][i]:.6e}",
                        f"{res['error_linf'][i]:.6e}",
                    ]
                )


def run_analysis(
    *,
    output_dir: Path,
    solver_modules: dict[str, tuple[str, dict[str, object]]],
    grid_sizes,
    dtype,
    tol: float,
    max_iter: int,
    plot_suffix: str,
    csv_suffix: str,
    reference_solver: str,
    sor_max_iter: int = 1000,
    mg_max_iter: int = 100,
) -> None:
    markers = build_markers(solver_modules)
    output_dir.mkdir(parents=True, exist_ok=True)

    for case in CASES_3D:
        print("\n=======================")
        print(f"Running 3D case: {case}")
        print("=======================\n")

        results = run_case(
            case=case,
            solver_modules=solver_modules,
            grid_sizes=grid_sizes,
            dtype=dtype,
            tol=tol,
            max_iter=max_iter,
            sor_max_iter=sor_max_iter,
            mg_max_iter=mg_max_iter,
        )

        plot_path = output_dir / f"plots_{case}{plot_suffix}.png"
        plot_results(
            case=case,
            results=results,
            markers=markers,
            output_path=plot_path,
            reference_solver=reference_solver,
        )
        print(f"Plot saved to {plot_path}")

        csv_path = output_dir / f"results_{case}{csv_suffix}.csv"
        write_results_csv(csv_path, results)
        print(f"Data saved to {csv_path}")
