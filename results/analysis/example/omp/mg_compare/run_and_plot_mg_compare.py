from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _omp_benchmark import CASES, build_markers, resolve_executable, run_solver

SOLVER_MODULES = {
    "RB SOR": ("sor", ()),
}
for cycle in ("v", "w"):
    for omega, o_name in ((1.25, "1.25"), (None, "auto")):
        name = f"MG({cycle}, nu=3, w={o_name})"
        SOLVER_MODULES[name] = (
            "mg",
            (
                "--cycle",
                cycle,
                "--nu",
                "3",
                "--mg-coarse",
                "exact",
                "--omega",
                "auto" if omega is None else str(omega),
            ),
        )

MARKERS = build_markers(SOLVER_MODULES.keys())


def main():
    grid_sizes = np.array([16, 32, 64, 128, 256, 512], dtype=int) - 1
    dtype_cli = "double"
    tol = 1e-10
    max_iter = 50000

    print(f"Using OpenMP executable: {resolve_executable()}")

    for case in CASES:
        print(f"\n=======================")
        print(f"Running case: {case}")
        print(f"=======================\n")

        results = {
            name: {"N": [], "iterations": [], "time_s": [], "error_l2": [], "error_linf": []}
            for name in SOLVER_MODULES
        }

        for name, config in SOLVER_MODULES.items():
            solver_name, extra_args = config
            print(f"Running {name}...")

            for N in grid_sizes:
                print(f"  N={N}")

                # Warm up
                run_solver(
                    solver=solver_name,
                    case=case,
                    grid_size=int(N),
                    dtype_cli=dtype_cli,
                    tol=1e-2,
                    max_iter=2,
                    extra_args=extra_args,
                )

                # Solve
                current_max_iter = 100 if "MG" in name else max_iter
                row = run_solver(
                    solver=solver_name,
                    case=case,
                    grid_size=int(N),
                    dtype_cli=dtype_cli,
                    tol=tol,
                    max_iter=current_max_iter,
                    extra_args=extra_args,
                )

                results[name]["N"].append(N)
                results[name]["iterations"].append(row["iterations"])
                results[name]["time_s"].append(row["time_s"])
                results[name]["error_l2"].append(row["error_l2"])
                results[name]["error_linf"].append(row["error_linf"])

        # Plotting
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))

        # 1. L2 Error vs N
        ax = axs[0, 0]
        for name, res in results.items():
            ax.loglog(res["N"], res["error_l2"], marker=MARKERS[name], linestyle="", alpha=0.7, label=name)

        n_fit = np.array([grid_sizes[0], grid_sizes[-1]])
        h_fit = 1.0 / (n_fit + 1)
        err_fit = results["RB SOR"]["error_l2"][0] * (h_fit / h_fit[0]) ** 2
        ax.loglog(n_fit, err_fit, "--", color="gray", label="Fit (Order=2.00)")
        ax.set_title(f"$L_2$ Error vs N ({case})")
        ax.set_xlabel("N")
        ax.set_ylabel("$L_2$ Error Norm")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(loc="upper right")

        # 2. Linf Error vs N
        ax = axs[0, 1]
        for name, res in results.items():
            ax.loglog(res["N"], res["error_linf"], marker=MARKERS[name], linestyle="", alpha=0.7, label=name)
        err_fit_inf = results["RB SOR"]["error_linf"][0] * (h_fit / h_fit[0]) ** 2
        ax.loglog(n_fit, err_fit_inf, "--", color="gray", label="Fit (Order=2.00)")
        ax.set_title(fr"$L_\infty$ Error vs N ({case})")
        ax.set_xlabel("N")
        ax.set_ylabel(r"$L_\infty$ Error Norm")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(loc="upper right")

        # 3. Iterations vs N
        ax = axs[1, 0]
        for name, res in results.items():
            N_arr = np.array(res["N"])
            it_arr = np.array(res["iterations"])
            if len(N_arr) > 1:
                coeffs = np.polyfit(np.log(N_arr), np.log(it_arr), 1)
                power = coeffs[0]
                label = f"{name} (~$N^{{{power:.2f}}}$)"
            else:
                label = name
            ax.loglog(res["N"], res["iterations"], marker=MARKERS[name], linestyle="--", alpha=0.7, label=label)
        ax.set_title(f"Iterations vs N ({case})")
        ax.set_xlabel("N")
        ax.set_ylabel("Iterations")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(loc="upper left", fontsize="small", ncol=2)

        # 4. Time vs N
        ax = axs[1, 1]
        for name, res in results.items():
            ax.loglog(res["N"], res["time_s"], marker=MARKERS[name], linestyle="-", alpha=0.7, label=name)
        ax.set_title(f"Time vs N ({case})")
        ax.set_xlabel("N")
        ax.set_ylabel("Execution Time (s)")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(loc="upper left", fontsize="small", ncol=2)

        plt.tight_layout()
        out_path = Path(__file__).resolve().parent / f"plots_{case}_mg_compare.png"
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Plot saved to {out_path}")

        csv_path = Path(__file__).resolve().parent / f"results_{case}_mg_compare.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Solver", "N", "Iterations", "Time_s", "L2_Error", "Linf_Error"])
            for name, res in results.items():
                for i in range(len(res["N"])):
                    writer.writerow([
                        name,
                        res["N"][i],
                        res["iterations"][i],
                        f"{res['time_s'][i]:.6e}",
                        f"{res['error_l2'][i]:.6e}",
                        f"{res['error_linf'][i]:.6e}",
                    ])
        print(f"Data saved to {csv_path}")


if __name__ == "__main__":
    main()
