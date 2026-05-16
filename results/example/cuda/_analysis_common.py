from __future__ import annotations

import csv
import sys
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULT_ROOT = Path(__file__).resolve().parent
CUDA_RESULTS_DIR = RESULT_ROOT

if str(CUDA_RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(CUDA_RESULTS_DIR))

from _cuda_benchmark import CASES, build_markers, resolve_executable, run_solver


SOLVER_COMPARISON_MODULES = {
    "Jacobi": ("jacobi", ()),
    "RB GS": ("gs", ()),
    "RB SOR": ("sor", ()),
    "MG V-cycle": ("mg", ("--cycle", "v", "--nu", "2", "--mg-coarse", "exact")),
    "MG W-cycle": ("mg", ("--cycle", "w", "--nu", "2", "--mg-coarse", "exact")),
}

MG_COMPARE_MODULES = {
    "RB SOR": ("sor", ()),
}
for cycle in ("v", "w"):
    for omega, omega_name in (("1.25", "1.25"), ("auto", "auto")):
        MG_COMPARE_MODULES[f"MG({cycle}, nu=3, w={omega_name})"] = (
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


SOLVER_COMPARISON_GRID_SIZES = np.array([16, 24, 32, 48, 64, 96, 128], dtype=int) - 1
MG_COMPARE_GRID_SIZES = np.array([16, 32, 64, 128, 256, 512], dtype=int) - 1


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
):
    results = make_empty_results(solver_modules)

    for name, (solver_name, extra_args) in solver_modules.items():
        print(f"Running {name}...")

        for grid_size in grid_sizes:
            print(f"  N={grid_size}")

            run_solver(
                solver=solver_name,
                case=case,
                grid_size=int(grid_size),
                dtype_cli=dtype_cli,
                tol=1e-2,
                max_iter=2,
                extra_args=extra_args,
            )

            current_max_iter = 100 if "MG" in name else max_iter
            start = perf_counter()
            row = run_solver(
                solver=solver_name,
                case=case,
                grid_size=int(grid_size),
                dtype_cli=dtype_cli,
                tol=tol,
                max_iter=current_max_iter,
                extra_args=extra_args,
            )
            fallback_time_s = perf_counter() - start

            results[name]["N"].append(int(grid_size))
            results[name]["iterations"].append(row["iterations"])
            results[name]["time_s"].append(float(row.get("time_s", fallback_time_s)))
            results[name]["error_l2"].append(row["error_l2"])
            results[name]["error_linf"].append(row["error_linf"])

    return results


def plot_results(*, case: str, results, markers, output_path: Path, reference_solver: str):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    ax = axs[0, 0]
    for name, res in results.items():
        ax.loglog(res["N"], res["error_l2"], marker=markers[name], linestyle="", alpha=0.7, label=name)
    n_fit = np.array([min(results[reference_solver]["N"]), max(results[reference_solver]["N"])])
    h_fit = 1.0 / (n_fit + 1)
    err_fit = results[reference_solver]["error_l2"][0] * (h_fit / h_fit[0]) ** 2
    ax.loglog(n_fit, err_fit, "--", color="gray", label="Fit (Order=2.00)")
    ax.set_title(f"$L_2$ Error vs N ({case})")
    ax.set_xlabel("N")
    ax.set_ylabel("$L_2$ Error Norm")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="upper right")

    ax = axs[0, 1]
    for name, res in results.items():
        ax.loglog(res["N"], res["error_linf"], marker=markers[name], linestyle="", alpha=0.7, label=name)
    err_fit_inf = results[reference_solver]["error_linf"][0] * (h_fit / h_fit[0]) ** 2
    ax.loglog(n_fit, err_fit_inf, "--", color="gray", label="Fit (Order=2.00)")
    ax.set_title(fr"$L_\infty$ Error vs N ({case})")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$L_\infty$ Error Norm")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="upper right")

    ax = axs[1, 0]
    for name, res in results.items():
        n_arr = np.array(res["N"])
        it_arr = np.array(res["iterations"])
        if len(n_arr) > 1:
            coeffs = np.polyfit(np.log(n_arr), np.log(it_arr), 1)
            power = coeffs[0]
            label = f"{name} (~$N^{{{power:.2f}}}$)"
        else:
            label = name
        ax.loglog(res["N"], res["iterations"], marker=markers[name], linestyle="--", alpha=0.7, label=label)
    ax.set_title(f"Iterations vs N ({case})")
    ax.set_xlabel("N")
    ax.set_ylabel("Iterations")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="upper left")

    ax = axs[1, 1]
    for name, res in results.items():
        ax.loglog(res["N"], res["time_s"], marker=markers[name], linestyle="-", label=name)
    ax.set_title(f"Time vs N ({case})")
    ax.set_xlabel("N")
    ax.set_ylabel("Execution Time (s)")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_results_csv(path: Path, results) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
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
) -> None:
    markers = build_markers(solver_modules)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using CUDA executable: {resolve_executable()}")

    for case in CASES:
        print("\n=======================")
        print(f"Running case: {case}")
        print("=======================\n")

        results = run_case(
            case=case,
            solver_modules=solver_modules,
            grid_sizes=grid_sizes,
            dtype_cli=dtype_cli,
            tol=tol,
            max_iter=max_iter,
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
