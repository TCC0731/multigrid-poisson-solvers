from __future__ import annotations

"""Shared helpers for CUDA MG parameter-sweep reports."""

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR
if str(RESULT_ROOT) not in sys.path:
    sys.path.insert(0, str(RESULT_ROOT))

from _poisson_wrapper import resolve_executable, run_or_load
from _report_common import csv_float, load_pyplot, normalize_choices, positive_float, positive_int, write_rows_csv


DEFAULT_BACKEND = "cuda"
DEFAULT_DTYPE = "double"
DEFAULT_CASES = ("sine", "cosine")
DEFAULT_DIMS = (2, 3)
DEFAULT_GRID_SIZES = {2: 4095, 3: 383}
DEFAULT_OMEGA = 1.25
DEFAULT_NU = 3
DEFAULT_OMEGA_VALUES = tuple(round(1.0 + 0.05 * index, 2) for index in range(11))
DEFAULT_NU_VALUES = (1, 2, 3, 4, 5)
DEFAULT_TOL = 1e-9
DEFAULT_MAX_ITER = 15
DEFAULT_REPEAT_RUNS = 5

CSV_COLUMNS = (
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
    "source",
)


@dataclass(frozen=True)
class ModeSpec:
    label: str
    mode_key: str
    cycle: str
    coarse: str
    color: str
    marker: str
    linestyle: str


MODE_SPECS = (
    ModeSpec("V-exact", "v_exact", "v", "exact", "#1f77b4", "o", "-"),
    ModeSpec("V-SOR", "v_sor", "v", "sor", "#ff7f0e", "s", "--"),
    ModeSpec("W-exact", "w_exact", "w", "exact", "#2ca02c", "^", "-"),
    ModeSpec("W-SOR", "w_sor", "w", "sor", "#d62728", "D", "--"),
)
MODE_SPECS_BY_KEY = {spec.mode_key: spec for spec in MODE_SPECS}


def _normalize_grid_sizes(grid_sizes: Mapping[int, object]) -> dict[int, int]:
    return {
        2: positive_int(grid_sizes[2], "grid_2d"),
        3: positive_int(grid_sizes[3], "grid_3d"),
    }


def _sweep_value(row: Mapping[str, object], sweep_name: str) -> float | int:
    if sweep_name == "omega":
        return float(row["omega"])
    return int(row["nu"])


def _sorted_sweep_values(values: Iterable[object], sweep_name: str) -> tuple[float | int, ...]:
    if sweep_name == "omega":
        return tuple(sorted({positive_float(value, "omega") for value in values}))
    if sweep_name == "nu":
        return tuple(sorted({positive_int(value, "nu") for value in values}))
    raise ValueError(f"unsupported sweep name: {sweep_name!r}")


def _format_sweep_value(value: float | int, sweep_name: str) -> str:
    if sweep_name == "omega":
        return f"{float(value):.2f}"
    return str(int(value))


def _row_key(row: Mapping[str, object], sweep_name: str) -> tuple[object, ...]:
    return (
        int(row["dimension"]),
        str(row["case"]),
        str(row["mode_key"]),
        _sweep_value(row, sweep_name),
    )


def _row_from_wrapper(
    *,
    dim: int,
    case: str,
    grid_size: int,
    mode: ModeSpec,
    omega: float,
    nu: int,
    tol: float,
    max_iter: int,
    repeat_runs: int,
) -> dict[str, object]:
    result = run_or_load(
        backend=DEFAULT_BACKEND,
        dim=dim,
        dtype=DEFAULT_DTYPE,
        solver="mg",
        case=case,
        grid_size=grid_size,
        tol=tol,
        max_iter=max_iter,
        repeat_runs=repeat_runs,
        cycle=mode.cycle,
        nu=nu,
        omega=omega,
        mg_coarse=mode.coarse,
    )
    return {
        "dimension": dim,
        "case": case,
        "mode": mode.label,
        "mode_key": mode.mode_key,
        "cycle": mode.cycle,
        "coarse": mode.coarse,
        "omega": omega,
        "solver": result.solver,
        "backend": result.backend,
        "dtype": result.dtype,
        "grid_size": result.grid_size,
        "iterations": result.iterations,
        "residual_l2": result.residual_l2,
        "error_l2": result.error_l2,
        "error_linf": result.error_linf,
        "time_ms": result.time_ms,
        "time_s": result.time_s,
        "tol": tol,
        "max_iter": max_iter,
        "nu": nu,
        "converged": bool(result.residual_l2 <= tol),
        "source": "wrapper",
    }


def collect_rows_from_wrapper(
    *,
    dims: Iterable[int],
    cases: Iterable[str],
    sweep_name: str,
    sweep_values: Sequence[object],
    grid_sizes: Mapping[int, object],
    fixed_omega: float,
    fixed_nu: int,
    tol: float,
    max_iter: int,
    repeat_runs: int,
) -> list[dict[str, object]]:
    dims = normalize_choices(tuple(positive_int(dim, "dim") for dim in dims), valid=DEFAULT_DIMS, name="dimension")
    cases = normalize_choices(cases, valid=DEFAULT_CASES, name="case")
    grid_sizes = _normalize_grid_sizes(grid_sizes)
    sweep_values = _sorted_sweep_values(sweep_values, sweep_name)
    fixed_omega = positive_float(fixed_omega, "omega")
    fixed_nu = positive_int(fixed_nu, "nu")
    tol = positive_float(tol, "tol")
    max_iter = positive_int(max_iter, "max_iter")
    repeat_runs = positive_int(repeat_runs, "repeat_runs")

    rows: list[dict[str, object]] = []
    for dim in dims:
        grid_size = grid_sizes[dim]
        for case in cases:
            for sweep_value in sweep_values:
                omega = float(sweep_value) if sweep_name == "omega" else fixed_omega
                nu = int(sweep_value) if sweep_name == "nu" else fixed_nu
                for mode in MODE_SPECS:
                    print(
                        f"  {dim}D {case} | {mode.label} | omega={omega:.2f} | nu={nu} | grid={grid_size}",
                        flush=True,
                    )
                    rows.append(
                        _row_from_wrapper(
                            dim=dim,
                            case=case,
                            grid_size=grid_size,
                            mode=mode,
                            omega=omega,
                            nu=nu,
                            tol=tol,
                            max_iter=max_iter,
                            repeat_runs=repeat_runs,
                        )
                    )
    return rows


def _format_rows(rows: Iterable[Mapping[str, object]], sweep_name: str) -> list[dict[str, object]]:
    ordered_rows = sorted(rows, key=lambda row: _row_key(row, sweep_name))
    formatted: list[dict[str, object]] = []
    for row in ordered_rows:
        formatted.append(
            {
                "dimension": int(row["dimension"]),
                "case": row["case"],
                "mode": row["mode"],
                "mode_key": row["mode_key"],
                "cycle": row["cycle"],
                "coarse": row["coarse"],
                "omega": f"{float(row['omega']):.2f}",
                "solver": row["solver"],
                "backend": row["backend"],
                "dtype": row["dtype"],
                "grid_size": int(row["grid_size"]),
                "iterations": int(row["iterations"]),
                "residual_l2": csv_float(float(row["residual_l2"])),
                "error_l2": csv_float(float(row["error_l2"])),
                "error_linf": csv_float(float(row["error_linf"])),
                "time_ms": csv_float(float(row["time_ms"])),
                "time_s": csv_float(float(row["time_s"])),
                "tol": csv_float(float(row["tol"])),
                "max_iter": int(row["max_iter"]),
                "nu": int(row["nu"]),
                "converged": str(bool(row["converged"])).lower(),
                "source": row["source"],
            }
        )
    return formatted


def group_rows(rows: Iterable[Mapping[str, object]]) -> dict[tuple[int, str], list[dict[str, object]]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["dimension"]), str(row["case"]))].append(dict(row))
    return grouped


def plot_case(
    *,
    dim: int,
    case: str,
    grid_size: int,
    rows: list[dict[str, object]],
    sweep_name: str,
    fixed_omega: float,
    fixed_nu: int,
    tol: float,
    max_iter: int,
    output_path: Path,
) -> None:
    plt = load_pyplot()

    series_by_label: dict[str, list[dict[str, object]]] = {spec.label: [] for spec in MODE_SPECS}
    for row in rows:
        series_by_label[MODE_SPECS_BY_KEY[str(row["mode_key"])].label].append(row)

    sweep_values = sorted({_sweep_value(row, sweep_name) for row in rows})
    sweep_labels = [_format_sweep_value(value, sweep_name) for value in sweep_values]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = (
        ("error_l2", "L2 error"),
        ("iterations", "Iterations"),
        ("time_s", "Time (s)"),
    )

    for ax, (metric_key, title) in zip(axes, panels):
        for spec in MODE_SPECS:
            series = sorted(series_by_label[spec.label], key=lambda row: _sweep_value(row, sweep_name))
            if not series:
                continue
            x = [float(row["omega"]) if sweep_name == "omega" else int(row["nu"]) for row in series]
            y = [float(row[metric_key]) for row in series]
            plot_fn = ax.semilogy if metric_key == "error_l2" else ax.plot
            plot_fn(
                x,
                y,
                marker=spec.marker,
                linestyle=spec.linestyle,
                color=spec.color,
                linewidth=1.9,
                label=spec.label,
            )

        ax.set_title(title)
        ax.set_xlabel(sweep_name)
        ax.set_ylabel(title)
        ax.set_xticks(sweep_values)
        ax.set_xticklabels(sweep_labels, rotation=45 if sweep_name == "omega" else 0, ha="right" if sweep_name == "omega" else "center")
        ax.grid(True, which="both", linestyle="--", alpha=0.45)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    if sweep_name == "omega":
        subtitle = f"grid={grid_size}, nu={fixed_nu}, tol={tol:.0e}, max_iter={max_iter}"
    else:
        subtitle = f"grid={grid_size}, omega={fixed_omega:.2f}, tol={tol:.0e}, max_iter={max_iter}"
    fig.suptitle(f"CUDA MG {sweep_name} sweep - {dim}D {case}\n{subtitle}", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_sweep_report(
    *,
    sweep_name: str,
    output_dir: Path,
    cases: Iterable[str],
    dims: Iterable[int],
    sweep_values: Sequence[object],
    grid_sizes: Mapping[int, object] = DEFAULT_GRID_SIZES,
    fixed_omega: float = DEFAULT_OMEGA,
    fixed_nu: int = DEFAULT_NU,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
    repeat_runs: int = DEFAULT_REPEAT_RUNS,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = normalize_choices(cases, valid=DEFAULT_CASES, name="case")
    dims = normalize_choices(tuple(positive_int(dim, "dim") for dim in dims), valid=DEFAULT_DIMS, name="dimension")
    grid_sizes = _normalize_grid_sizes(grid_sizes)
    sweep_values = _sorted_sweep_values(sweep_values, sweep_name)
    fixed_omega = positive_float(fixed_omega, "omega")
    fixed_nu = positive_int(fixed_nu, "nu")
    tol = positive_float(tol, "tol")
    max_iter = positive_int(max_iter, "max_iter")
    repeat_runs = positive_int(repeat_runs, "repeat_runs")

    print(f"Using CUDA executable: {resolve_executable(DEFAULT_BACKEND)}")
    print(f"Output directory: {output_dir}")
    print(f"Cases: {', '.join(cases)}")
    print(f"Dimensions: {', '.join(str(dim) for dim in dims)}")
    print(f"Grid sizes: 2D={grid_sizes[2]}, 3D={grid_sizes[3]}")
    print(f"{sweep_name} values: {', '.join(_format_sweep_value(value, sweep_name) for value in sweep_values)}")
    if sweep_name == "omega":
        print(f"Fixed nu: {fixed_nu}")
    else:
        print(f"Fixed omega: {fixed_omega:.2f}")
    print(f"tol={tol:.0e}, max_iter={max_iter}, repeat_runs={repeat_runs}")

    rows = collect_rows_from_wrapper(
        dims=dims,
        cases=cases,
        sweep_name=sweep_name,
        sweep_values=sweep_values,
        grid_sizes=grid_sizes,
        fixed_omega=fixed_omega,
        fixed_nu=fixed_nu,
        tol=tol,
        max_iter=max_iter,
        repeat_runs=repeat_runs,
    )
    if not rows:
        raise RuntimeError("No benchmark rows were produced.")

    rows = sorted(rows, key=lambda row: _row_key(row, sweep_name))
    write_rows_csv(output_dir / "results_all.csv", _format_rows(rows, sweep_name), columns=CSV_COLUMNS)

    grouped = group_rows(rows)
    for dim, case in sorted(grouped):
        case_rows = sorted(grouped[(dim, case)], key=lambda row: _sweep_value(row, sweep_name))
        grid_size = grid_sizes[dim]
        write_rows_csv(
            output_dir / f"results_{dim}d_{case}.csv",
            _format_rows(case_rows, sweep_name),
            columns=CSV_COLUMNS,
        )
        plot_case(
            dim=dim,
            case=case,
            grid_size=grid_size,
            rows=case_rows,
            sweep_name=sweep_name,
            fixed_omega=fixed_omega,
            fixed_nu=fixed_nu,
            tol=tol,
            max_iter=max_iter,
            output_path=output_dir / f"plots_{dim}d_{case}.png",
        )

    print(f"Wrote {output_dir / 'results_all.csv'}")
    print(f"Rows: {len(rows)}")
    print(f"Non-converged rows: {sum(1 for row in rows if not row['converged'])}")
