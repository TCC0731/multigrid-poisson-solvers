from __future__ import annotations

import csv
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

RESULT_ROOT = Path(__file__).resolve().parent
if str(RESULT_ROOT) not in sys.path:
    sys.path.insert(0, str(RESULT_ROOT))

from _cuda_benchmark import CASES, build_markers, resolve_executable, run_solver


OMEGA_VALUES = tuple(round(1.0 + 0.05 * i, 2) for i in range(11))
GRID_SIZES = {2: 4095, 3: 383}
DEFAULT_DTYPE_CLI = "double"
DEFAULT_TOL = 1e-9
DEFAULT_MAX_ITER = 15
DEFAULT_NU = 2
DEFAULT_GRID_SIZES_BY_DIM = {dim: (grid_size,) for dim, grid_size in GRID_SIZES.items()}
DEFAULT_NU_VALUES = (DEFAULT_NU,)

# The user asked for the four multigrid variants below:
# V/W cycle combined with exact/SOR coarse-grid solves.
MG_MODE_SPECS = (
    ("v_exact", "V-exact", "v", "exact"),
    ("v_sor", "V-SOR", "v", "sor"),
    ("w_exact", "W-exact", "w", "exact"),
    ("w_sor", "W-SOR", "w", "sor"),
)

CSV_FIELDS = (
    "dimension",
    "case",
    "mode",
    "mode_key",
    "cycle",
    "coarse",
    "omega",
    "solver",
    "backend",
    "dtype",
    "grid_size",
    "iterations",
    "residual_l2",
    "error_l2",
    "error_linf",
    "time_ms",
    "time_s",
    "tol",
    "max_iter",
    "nu",
    "converged",
)


def _case_title(dim: int, case: str) -> str:
    return f"{dim}D {case}"


def _csv_float(value: float) -> str:
    return f"{float(value):.6e}"


def _csv_omega(value: float) -> str:
    return f"{float(value):.2f}"


def _normalize_grid_sizes_by_dim(
    grid_sizes_by_dim: Mapping[int, Sequence[int]] | None,
) -> dict[int, tuple[int, ...]]:
    source = DEFAULT_GRID_SIZES_BY_DIM if grid_sizes_by_dim is None else grid_sizes_by_dim
    normalized: dict[int, tuple[int, ...]] = {}
    for dim in (2, 3):
        try:
            values = source[dim]
        except KeyError as exc:
            raise ValueError(f"Missing grid sizes for dimension {dim}") from exc
        grid_sizes = tuple(int(grid_size) for grid_size in values)
        if not grid_sizes:
            raise ValueError(f"At least one grid size is required for dimension {dim}")
        normalized[dim] = grid_sizes
    return normalized


def _normalize_nu_values(nu_values: Sequence[int] | None) -> tuple[int, ...]:
    source = DEFAULT_NU_VALUES if nu_values is None else nu_values
    normalized = tuple(int(nu) for nu in source)
    if not normalized:
        raise ValueError("At least one nu value is required")
    return normalized


def run_case(
    *,
    dim: int,
    case: str,
    grid_size: int,
    dtype_cli: str,
    tol: float,
    max_iter: int,
    nu: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    print(f"Running {_case_title(dim, case)} (grid={grid_size}, nu={nu})...")
    for mode_key, mode_label, cycle, coarse in MG_MODE_SPECS:
        print(f"  Mode: {mode_label}")

        for omega in OMEGA_VALUES:
            print(f"    omega={omega:.2f}")
            try:
                solver_row = run_solver(
                    solver="mg",
                    case=case,
                    grid_size=grid_size,
                    dtype_cli=dtype_cli,
                    tol=tol,
                    max_iter=max_iter,
                    dim=dim,
                    extra_args=(
                        "--cycle",
                        cycle,
                        "--nu",
                        str(int(nu)),
                        "--mg-coarse",
                        coarse,
                        "--omega",
                        f"{omega:.2f}",
                    ),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to run {_case_title(dim, case)} "
                    f"for mode={mode_label}, omega={omega:.2f}"
                ) from exc

            rows.append(
                {
                    "dimension": dim,
                    "case": case,
                    "mode": mode_label,
                    "mode_key": mode_key,
                    "cycle": cycle,
                    "coarse": coarse,
                    "omega": float(omega),
                    "solver": solver_row["solver"],
                    "backend": solver_row["backend"],
                    "dtype": solver_row["dtype"],
                    "grid_size": int(solver_row["grid_size"]),
                    "iterations": int(solver_row["iterations"]),
                    "residual_l2": float(solver_row["residual_l2"]),
                    "error_l2": float(solver_row["error_l2"]),
                    "error_linf": float(solver_row["error_linf"]),
                    "time_ms": float(solver_row["time_ms"]),
                    "time_s": float(solver_row["time_s"]),
                    "tol": float(tol),
                    "max_iter": int(max_iter),
                    "nu": int(nu),
                    "converged": bool(float(solver_row["residual_l2"]) <= tol),
                }
            )

    return rows


def plot_results(
    *,
    dim: int,
    case: str,
    grid_size: int,
    nu: int,
    tol: float,
    max_iter: int,
    rows: list[dict[str, object]],
    markers,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: dict[str, dict[str, list[float]]] = {
        mode_label: {
            "omega": [],
            "iterations": [],
            "error_l2": [],
            "time_ms": [],
        }
        for _, mode_label, _, _ in MG_MODE_SPECS
    }
    for row in rows:
        mode_label = str(row["mode"])
        series[mode_label]["omega"].append(float(row["omega"]))
        series[mode_label]["iterations"].append(float(row["iterations"]))
        series[mode_label]["error_l2"].append(float(row["error_l2"]))
        series[mode_label]["time_ms"].append(float(row["time_ms"]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for _, mode_label, _, _ in MG_MODE_SPECS:
        ax.plot(
            series[mode_label]["omega"],
            series[mode_label]["iterations"],
            marker=markers[mode_label],
            linewidth=1.8,
            label=mode_label,
        )
    ax.set_title(
        f"CUDA MG omega sweep - {_case_title(dim, case)}\n"
        f"tol={tol:.0e}, max_iter={max_iter}, grid={grid_size}, nu={nu}"
    )
    ax.set_xlabel("omega")
    ax.set_ylabel("Iterations")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize="small", ncol=2)

    ax = axes[1]
    for _, mode_label, _, _ in MG_MODE_SPECS:
        ax.semilogy(
            series[mode_label]["omega"],
            series[mode_label]["error_l2"],
            marker=markers[mode_label],
            linewidth=1.8,
            label=mode_label,
        )
    ax.set_xlabel("omega")
    ax.set_ylabel("L2 error")
    ax.set_title("L2 error vs omega")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize="small", ncol=2)

    ax = axes[2]
    for _, mode_label, _, _ in MG_MODE_SPECS:
        ax.plot(
            series[mode_label]["omega"],
            series[mode_label]["time_ms"],
            marker=markers[mode_label],
            linewidth=1.8,
            label=mode_label,
        )
    ax.set_xlabel("omega")
    ax.set_ylabel("Average time (ms)")
    ax.set_title("Average solver time vs omega")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize="small", ncol=2)

    for ax in axes:
        ax.set_xticks(list(OMEGA_VALUES))
        ax.set_xticklabels([f"{omega:.2f}" for omega in OMEGA_VALUES], rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_results_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dimension": row["dimension"],
                    "case": row["case"],
                    "mode": row["mode"],
                    "mode_key": row["mode_key"],
                    "cycle": row["cycle"],
                    "coarse": row["coarse"],
                    "omega": _csv_omega(float(row["omega"])),
                    "solver": row["solver"],
                    "backend": row["backend"],
                    "dtype": row["dtype"],
                    "grid_size": int(row["grid_size"]),
                    "iterations": int(row["iterations"]),
                    "residual_l2": _csv_float(float(row["residual_l2"])),
                    "error_l2": _csv_float(float(row["error_l2"])),
                    "error_linf": _csv_float(float(row["error_linf"])),
                    "time_ms": _csv_float(float(row["time_ms"])),
                    "time_s": _csv_float(float(row["time_s"])),
                    "tol": _csv_float(float(row["tol"])),
                    "max_iter": int(row["max_iter"]),
                    "nu": int(row["nu"]),
                    "converged": str(bool(row["converged"])).lower(),
                }
            )


def run_analysis(
    *,
    output_dir: Path,
    dtype_cli: str = DEFAULT_DTYPE_CLI,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
    nu: int = DEFAULT_NU,
    nu_values: Sequence[int] | None = None,
    grid_sizes_by_dim: Mapping[int, Sequence[int]] | None = None,
) -> None:
    grid_sizes_by_dim = _normalize_grid_sizes_by_dim(grid_sizes_by_dim)
    nu_values = _normalize_nu_values(nu_values if nu_values is not None else (nu,))

    markers = build_markers([mode_label for _, mode_label, _, _ in MG_MODE_SPECS])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using CUDA executable: {resolve_executable()}")

    all_rows: list[dict[str, object]] = []
    rows_by_dim: dict[int, list[dict[str, object]]] = {2: [], 3: []}

    for dim in (2, 3):
        for grid_size in grid_sizes_by_dim[dim]:
            for nu_value in nu_values:
                combo_dir = output_dir / f"{dim}D_{grid_size}_{nu_value}"
                combo_dir.mkdir(parents=True, exist_ok=True)
                combo_rows: list[dict[str, object]] = []
                for case in CASES:
                    case_rows = run_case(
                        dim=dim,
                        case=case,
                        grid_size=grid_size,
                        dtype_cli=dtype_cli,
                        tol=tol,
                        max_iter=max_iter,
                        nu=nu_value,
                    )
                    combo_rows.extend(case_rows)
                    rows_by_dim[dim].extend(case_rows)
                    all_rows.extend(case_rows)

                    csv_path = combo_dir / f"results_{case}.csv"
                    write_results_csv(csv_path, case_rows)
                    print(f"  Wrote {csv_path}")

                    plot_path = combo_dir / f"plots_{case}.png"
                    plot_results(
                        dim=dim,
                        case=case,
                        grid_size=grid_size,
                        nu=nu_value,
                        tol=tol,
                        max_iter=max_iter,
                        rows=case_rows,
                        markers=markers,
                        output_path=plot_path,
                    )
                    print(f"  Wrote {plot_path}")

                combo_csv_path = combo_dir / "results_all.csv"
                write_results_csv(combo_csv_path, combo_rows)
                print(f"Wrote {combo_csv_path}")

    for dim in (2, 3):
        dim_csv_path = output_dir / f"{dim}D_results_all.csv"
        write_results_csv(dim_csv_path, rows_by_dim[dim])
        print(f"Wrote {dim_csv_path}")

    all_csv_path = output_dir / "results_all.csv"
    write_results_csv(all_csv_path, all_rows)
    print(f"Wrote {all_csv_path}")
