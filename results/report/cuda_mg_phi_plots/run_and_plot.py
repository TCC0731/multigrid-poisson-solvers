from __future__ import annotations

"""Generate CUDA MG V/W SOR phi plots and cache the plotted data in a PKL.

The script runs the CUDA solver directly with ``--dump-phi`` so we can reuse the
native binary output instead of reconstructing the solution in Python.

Default settings follow the report-style multigrid configuration used elsewhere
in the repository:

* 2D grid size: 4095
* 3D grid size: 383
* MG cycle: V and W
* coarse solve: SOR
* ``omega = 1.25``
* ``nu = 3``
* ``coarse_steps = 16``

For each selected case/dimension/cycle combination the script writes:

* a CUDA ``--dump-phi`` binary file
* a 3-panel figure with the exact solution, MG solution, and difference
* a compact PKL archive containing the plotted arrays and run metadata

For 3D, only the ``z=0.5`` slice is plotted and stored in the PKL.
"""

import argparse
import csv
import gc
import io
import pickle
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR.parent
if str(RESULT_ROOT) not in sys.path:
    sys.path.insert(0, str(RESULT_ROOT))

from _poisson_wrapper import REPO_ROOT, build_request, resolve_executable
from _report_common import load_pyplot, normalize_choices, positive_float, positive_int

PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import problems


DEFAULT_BACKEND = "cuda"
DEFAULT_CASES = ("sine", "mixed_sine")
DEFAULT_DIMS = (2, 3)
DEFAULT_CYCLES = ("v", "w")
DEFAULT_DTYPE = "double"
DEFAULT_GRID_SIZES = {2: 4095, 3: 383}
DEFAULT_OMEGA = 1.25
DEFAULT_NU = 3
DEFAULT_COARSE_STEPS = 16
DEFAULT_TOL = 1e-9
DEFAULT_MAX_ITER = 150
DEFAULT_REPEAT_RUNS = 1
PHI_DUMP_MAGIC = b"PHIDUMP1"
PHI_DUMP_HEADER = struct.Struct("<8sIIQQQQ")
PHI_DUMP_VERSION = 1


@dataclass(frozen=True)
class RunSpec:
    dim: int
    cycle: str
    case: str
    grid_size: int
    omega: float
    nu: int
    coarse_steps: int
    tol: float
    max_iter: int
    repeat_runs: int


@dataclass(frozen=True)
class PhiDump:
    dimension: int
    interior_n: int
    array_n: int
    values: np.ndarray


def _unique_positive_ints(values: Iterable[object], *, name: str) -> tuple[int, ...]:
    items: list[int] = []
    seen: set[int] = set()
    for value in values:
        item = positive_int(value, name)
        if item in seen:
            continue
        seen.add(item)
        items.append(item)
    if not items:
        raise ValueError(f"at least one {name} is required")
    return tuple(items)


def _omega_cli_value(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return repr(float(value))


def _omega_tag(value: float | str) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = f"{float(value):.4f}".rstrip("0").rstrip(".")
        if not text:
            text = "0"
    return text.replace("-", "m").replace(".", "p")


def _case_exact_fn(dim: int, case: str):
    if dim == 2:
        return problems.CASE_FUNCS[case][0]
    return problems.CASE_FUNCS_3D[case][0]


def _build_request(spec: RunSpec):
    return build_request(
        backend=DEFAULT_BACKEND,
        dim=spec.dim,
        dtype=DEFAULT_DTYPE,
        solver="mg",
        case=spec.case,
        grid_size=spec.grid_size,
        tol=spec.tol,
        max_iter=spec.max_iter,
        repeat_runs=spec.repeat_runs,
        cycle=spec.cycle,
        nu=spec.nu,
        omega=spec.omega,
        mg_coarse="sor",
        coarse_steps=spec.coarse_steps,
    )


def _build_command(request, executable: Path, dump_path: Path) -> list[str]:
    cmd = [
        str(executable),
        "--dim",
        str(request.dim),
        "--solver",
        request.solver,
        "--case",
        request.case,
        "--grid-size",
        str(request.grid_size),
        "--dtype",
        request.dtype,
        "--tol",
        repr(float(request.tol)),
        "--max-iter",
        str(request.max_iter),
        "--repeat-runs",
        str(request.repeat_runs),
    ]

    if request.solver == "mg":
        cmd.extend(
            [
                "--cycle",
                request.cycle or "v",
                "--nu",
                str(request.nu if request.nu is not None else DEFAULT_NU),
                "--omega",
                _omega_cli_value(request.omega if request.omega is not None else DEFAULT_OMEGA),
                "--mg-coarse",
                request.mg_coarse or "exact",
            ]
        )
        if request.mg_coarse == "sor":
            cmd.extend(
                [
                    "--coarse-steps",
                    str(request.coarse_steps if request.coarse_steps is not None else DEFAULT_COARSE_STEPS),
                ]
            )

    cmd.extend(["--dump-phi", str(dump_path)])
    return cmd


def _run_cuda_solver(request, dump_path: Path) -> dict[str, object]:
    executable = resolve_executable(DEFAULT_BACKEND)
    cmd = _build_command(request, executable, dump_path)

    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "CUDA solver failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc

    output = completed.stdout.strip()
    if not output:
        raise RuntimeError(
            "CUDA solver did not produce CSV output.\n"
            f"Command: {' '.join(cmd)}"
        )

    rows = list(csv.DictReader(io.StringIO(output)))
    if len(rows) != 1:
        raise RuntimeError(
            "Expected exactly one CSV row from the CUDA solver.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    row = rows[0]
    return {
        "solver": row["solver"],
        "backend": row["backend"],
        "dtype": row["dtype"],
        "grid_size": positive_int(row["grid_size"], "grid_size"),
        "iterations": positive_int(row["iterations"], "iterations"),
        "residual_l2": float(row["residual_l2"]),
        "error_l2": float(row["error_l2"]),
        "error_linf": float(row["error_linf"]),
        "time_ms": float(row["time_ms"]),
    }


def _read_phi_dump(path: Path) -> PhiDump:
    with path.open("rb") as stream:
        header = stream.read(PHI_DUMP_HEADER.size)
        if len(header) != PHI_DUMP_HEADER.size:
            raise RuntimeError(f"phi dump header is truncated: {path}")

        magic, version, dimension, interior_n, array_n, elements, value_size = PHI_DUMP_HEADER.unpack(header)

        if magic != PHI_DUMP_MAGIC:
            raise RuntimeError(f"unexpected phi dump magic in {path}")
        if version != PHI_DUMP_VERSION:
            raise RuntimeError(f"unsupported phi dump version {version} in {path}")
        if value_size != 8:
            raise RuntimeError(f"expected double precision dump data in {path}")

    grid = np.memmap(
        path,
        mode="r",
        dtype=np.dtype("<f8"),
        offset=PHI_DUMP_HEADER.size,
        shape=(int(elements),),
    )
    values = grid.reshape((int(array_n),) * int(dimension))
    return PhiDump(
        dimension=int(dimension),
        interior_n=int(interior_n),
        array_n=int(array_n),
        values=values,
    )


def _build_field_images(dim: int, case: str, dump: PhiDump) -> tuple[np.ndarray, np.ndarray, int | None, float | None]:
    exact_fn = _case_exact_fn(dim, case)
    coords = np.linspace(0.0, 1.0, dump.array_n, dtype=np.float64)

    if dim == 2:
        exact_field = np.asarray(exact_fn(coords[:, None], coords[None, :]), dtype=np.float64)
        mg_field = dump.values
        slice_index = None
        slice_z = None
    else:
        slice_index = int(np.argmin(np.abs(coords - 0.375)))
        slice_z = float(coords[slice_index])
        exact_field = np.asarray(exact_fn(coords[:, None], coords[None, :], slice_z), dtype=np.float64)
        mg_field = dump.values[:, :, slice_index]

    return exact_field, mg_field, slice_index, slice_z


def _figure_paths(output_dir: Path, spec: RunSpec) -> tuple[Path, Path]:
    tag = (
        f"phi_{spec.dim}d_{spec.cycle}_sor_{spec.case}_n{spec.grid_size}"
        f"_nu{spec.nu}_w{_omega_tag(spec.omega)}_cs{spec.coarse_steps}"
    )
    dump_path = output_dir / "dumps" / f"{tag}.bin"
    plot_path = output_dir / "plots" / f"{tag}.png"
    return dump_path, plot_path


def _plot_case(
    *,
    spec: RunSpec,
    solver_row: dict[str, object],
    exact_field: np.ndarray,
    mg_field: np.ndarray,
    slice_index: int | None,
    slice_z: float | None,
    dump_path: Path,
    plot_path: Path,
) -> dict[str, object]:
    plt = load_pyplot()

    exact_plot = exact_field.T
    mg_plot = mg_field.T
    diff_plot = mg_plot - exact_plot

    field_vmin = float(min(np.min(exact_plot), np.min(mg_plot)))
    field_vmax = float(max(np.max(exact_plot), np.max(mg_plot)))
    diff_absmax = float(np.max(np.abs(diff_plot)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    image_titles = ("Exact", f"CUDA MG {spec.cycle.upper()}-SOR", "Difference (MG - exact)")
    image_data = (
        (exact_plot, "viridis", field_vmin, field_vmax),
        (mg_plot, "viridis", field_vmin, field_vmax),
        (diff_plot, "coolwarm", -diff_absmax, diff_absmax),
    )

    for ax, title, (data, cmap, vmin, vmax) in zip(axes, image_titles, image_data):
        im = ax.imshow(
            data,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    slice_note = ""
    if slice_index is not None and slice_z is not None:
        slice_note = f", z={slice_z:.3f} (index {slice_index})"

    fig.suptitle(
        f"CUDA MG {spec.cycle.upper()}-SOR - {spec.dim}D {spec.case}, n={spec.grid_size}{slice_note}\n"
        f"nu={spec.nu}, omega={spec.omega:g}, coarse_steps={spec.coarse_steps}, tol={spec.tol:.0e}, max_iter={spec.max_iter}",
        y=1.03,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    exact_image = np.array(exact_plot, dtype=np.float32, copy=True)
    mg_image = np.array(mg_plot, dtype=np.float32, copy=True)

    return {
        "dimension": spec.dim,
        "cycle": spec.cycle,
        "case": spec.case,
        "grid_size": spec.grid_size,
        "nu": spec.nu,
        "omega": spec.omega,
        "coarse_steps": spec.coarse_steps,
        "tol": spec.tol,
        "max_iter": spec.max_iter,
        "repeat_runs": spec.repeat_runs,
        "backend": solver_row["backend"],
        "dtype": solver_row["dtype"],
        "solver": solver_row["solver"],
        "iterations": solver_row["iterations"],
        "residual_l2": solver_row["residual_l2"],
        "error_l2": solver_row["error_l2"],
        "error_linf": solver_row["error_linf"],
        "time_ms": solver_row["time_ms"],
        "dump_path": str(dump_path.resolve()),
        "plot_path": str(plot_path.resolve()),
        "slice_index": slice_index,
        "slice_z": slice_z,
        "field_vmin": field_vmin,
        "field_vmax": field_vmax,
        "diff_absmax": diff_absmax,
        "exact_image": exact_image,
        "mg_image": mg_image,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CUDA MG V/W SOR and plot phi comparisons.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR, help="Directory for plots, dumps, and PKL.")
    parser.add_argument(
        "--case",
        "--cases",
        dest="cases",
        nargs="+",
        default=list(DEFAULT_CASES),
        choices=sorted(problems.CASES),
        help="Poisson cases to run.",
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        choices=DEFAULT_DIMS,
        default=list(DEFAULT_DIMS),
        help="Dimensions to include.",
    )
    parser.add_argument(
        "--cycles",
        nargs="+",
        choices=DEFAULT_CYCLES,
        default=list(DEFAULT_CYCLES),
        help="MG cycles to include.",
    )
    parser.add_argument("--grid-size-2d", type=int, default=DEFAULT_GRID_SIZES[2], help="2D interior grid size.")
    parser.add_argument("--grid-size-3d", type=int, default=DEFAULT_GRID_SIZES[3], help="3D interior grid size.")
    parser.add_argument("--omega", type=float, default=DEFAULT_OMEGA, help="MG relaxation factor.")
    parser.add_argument("--nu", type=int, default=DEFAULT_NU, help="MG smoothing steps per pass.")
    parser.add_argument(
        "--coarse-steps",
        type=int,
        default=DEFAULT_COARSE_STEPS,
        help="Red-black SOR steps on the coarsest grid when using MG coarse SOR.",
    )
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL, help="Solver tolerance.")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help="Maximum MG iterations.")
    parser.add_argument("--repeat-runs", type=int, default=DEFAULT_REPEAT_RUNS, help="Repeat runs passed to CUDA.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = tuple(normalize_choices(args.cases, valid=sorted(problems.CASES), name="case"))
    dims = _unique_positive_ints(args.dims, name="dimension")
    cycles = tuple(dict.fromkeys(args.cycles))
    omega = positive_float(args.omega, "omega")
    nu = positive_int(args.nu, "nu")
    coarse_steps = positive_int(args.coarse_steps, "coarse_steps")
    tol = positive_float(args.tol, "tol")
    max_iter = positive_int(args.max_iter, "max_iter")
    repeat_runs = positive_int(args.repeat_runs, "repeat_runs")

    records: list[dict[str, object]] = []

    for case in cases:
        print(f"Running case={case}", flush=True)
        for dim in dims:
            grid_size = args.grid_size_2d if dim == 2 else args.grid_size_3d
            for cycle in cycles:
                spec = RunSpec(
                    dim=dim,
                    cycle=cycle,
                    case=case,
                    grid_size=grid_size,
                    omega=omega,
                    nu=nu,
                    coarse_steps=coarse_steps,
                    tol=tol,
                    max_iter=max_iter,
                    repeat_runs=repeat_runs,
                )
                dump_path, plot_path = _figure_paths(output_dir, spec)
                dump_path.parent.mkdir(parents=True, exist_ok=True)

                request = _build_request(spec)

                print(f"Running {dim}D {cycle.upper()}-SOR for case={case}, n={grid_size}", flush=True)
                solver_row = _run_cuda_solver(request, dump_path)
                dump = _read_phi_dump(dump_path)

                exact_field, mg_field, slice_index, slice_z = _build_field_images(dim, case, dump)
                record = _plot_case(
                    spec=spec,
                    solver_row=solver_row,
                    exact_field=exact_field,
                    mg_field=mg_field,
                    slice_index=slice_index,
                    slice_z=slice_z,
                    dump_path=dump_path,
                    plot_path=plot_path,
                )
                records.append(record)

                print(
                    f"  iterations={solver_row['iterations']}, residual={solver_row['residual_l2']:.3e}, "
                    f"error_l2={solver_row['error_l2']:.3e}"
                )
                print(f"  dump -> {dump_path}")
                print(f"  plot -> {plot_path}")

                del dump, exact_field, mg_field
                gc.collect()

    pkl_path = output_dir / "phi_results.pkl"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "case": cases[0] if len(cases) == 1 else ", ".join(cases),
        "cases": cases,
        "dims": dims,
        "cycles": cycles,
        "grid_sizes": {2: args.grid_size_2d, 3: args.grid_size_3d},
        "omega": omega,
        "nu": nu,
        "coarse_steps": coarse_steps,
        "tol": tol,
        "max_iter": max_iter,
        "repeat_runs": repeat_runs,
        "records": records,
    }

    with pkl_path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {pkl_path}")


if __name__ == "__main__":
    main()
