from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULT_ROOT = Path(__file__).resolve().parent
CPP_RESULTS_DIR = RESULT_ROOT.parent / "cpp"

if str(CPP_RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(CPP_RESULTS_DIR))

from _cpp_benchmark import CASES, build_markers, resolve_executable, run_solver


DEFAULT_MARKERS = ("o", "s", "^", "D", "v", "p", "*", "h", "H", "<", ">", "P", "X", "d")

JACOBI_GS_GRID_SIZES = np.array([7, 15, 23, 31])
RB_SOR_GRID_SIZES = np.array([15, 31, 47, 63])
MG_GRID_SIZES = np.array([7, 15, 31, 63, 127])

SOLVER_COMPARISON_MODULES = {
    "Jacobi 3D": ("jacobi", ()),
    "RB GS 3D": ("gs", ()),
    "RB SOR 3D": ("sor", ()),
    "MG V-cycle 3D": ("mg", ("--cycle", "v", "--nu", "2", "--mg-coarse", "exact")),
    "MG W-cycle 3D": ("mg", ("--cycle", "w", "--nu", "2", "--mg-coarse", "exact")),
}

MG_COMPARE_MODULES = {
    "RB SOR 3D": ("sor", ()),
}
for cycle in ("v", "w"):
    for omega, omega_name in (("1.25", "1.25"), ("auto", "auto")):
        MG_COMPARE_MODULES[f"MG 3D({cycle}, nu=3, w={omega_name})"] = (
            "mg",
            (
                "--cycle",
                cycle,
                "--nu",
                "3",
                "--mg-coarse",
                "exact",
                "--omega",
                omega,
            ),
        )

SOLVER_COMPARISON_GRID_SIZES = {
    "Jacobi 3D": JACOBI_GS_GRID_SIZES,
    "RB GS 3D": JACOBI_GS_GRID_SIZES,
    "RB SOR 3D": RB_SOR_GRID_SIZES,
    "MG V-cycle 3D": MG_GRID_SIZES,
    "MG W-cycle 3D": MG_GRID_SIZES,
}

MG_COMPARE_GRID_SIZES = {name: MG_GRID_SIZES for name in MG_COMPARE_MODULES}
MG_COMPARE_GRID_SIZES["RB SOR 3D"] = RB_SOR_GRID_SIZES


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


def run_case(
    *,
    case: str,
    solver_modules: dict[str, tuple[str, tuple[str, ...]]],
    grid_sizes,
    dtype_cli: str,
    tol: float,
    max_iter: int,
    sor_max_iter: int = 1000,
    mg_max_iter: int = 100,
):
    results = make_empty_results(solver_modules)

    for name, (solver_name, extra_args) in solver_modules.items():
        print(f"Running {name}...")
        solver_grid_sizes = grid_sizes[name] if isinstance(grid_sizes, dict) else grid_sizes

        for grid_size in solver_grid_sizes:
            print(f"  N={grid_size}")

            run_solver(
                solver=solver_name,
                case=case,
                grid_size=int(grid_size),
                dtype_cli=dtype_cli,
                tol=1e-2,
                max_iter=2,
                dim=3,
                extra_args=extra_args,
            )

            if "MG" in name:
                current_max_iter = mg_max_iter
            elif "SOR" in name:
                current_max_iter = sor_max_iter
            else:
                current_max_iter = max_iter

            start = perf_counter()
            row = run_solver(
                solver=solver_name,
                case=case,
                grid_size=int(grid_size),
                dtype_cli=dtype_cli,
                tol=tol,
                max_iter=current_max_iter,
                dim=3,
                extra_args=extra_args,
            )
            fallback_time_s = perf_counter() - start

            results[name]["N"].append(int(grid_size))
            results[name]["iterations"].append(row["iterations"])
            results[name]["time_s"].append(float(row.get("time_s", fallback_time_s)))
            results[name]["error_l2"].append(row["error_l2"])
            results[name]["error_linf"].append(row["error_linf"])

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
        ax.loglog(res["N"], res["error_l2"], marker=markers[name], linestyle="", alpha=0.7, label=name)
    n_fit = np.array([min(results[reference_solver]["N"]), max(results[reference_solver]["N"])])
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
        ax.loglog(res["N"], res["error_linf"], marker=markers[name], linestyle="", alpha=0.7, label=name)
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
        ax.loglog(res["N"], res["iterations"], marker=markers[name], linestyle="--", alpha=0.7, label=label)
    ax.set_title(f"Iterations vs N ({case}, 3D)")
    ax.set_xlabel("N")
    ax.set_ylabel("Iterations")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize="small", ncol=2)

    ax = axs[1, 1]
    for name, res in results.items():
        ax.loglog(res["N"], res["time_s"], marker=markers[name], linestyle="-", alpha=0.7, label=name)
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
    solver_modules: dict[str, tuple[str, tuple[str, ...]]],
    grid_sizes,
    dtype_cli: str,
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

    print(f"Using C++ executable: {resolve_executable()}")

    for case in CASES:
        print("\n=======================")
        print(f"Running C++ 3D case: {case}")
        print("=======================\n")

        results = run_case(
            case=case,
            solver_modules=solver_modules,
            grid_sizes=grid_sizes,
            dtype_cli=dtype_cli,
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
