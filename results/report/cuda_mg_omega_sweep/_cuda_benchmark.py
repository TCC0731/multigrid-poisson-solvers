from __future__ import annotations

import csv
import itertools
import os
import subprocess
from pathlib import Path
from typing import Sequence

CASES = ("sine", "mixed_sine", "bubble", "exp", "cosine")
DEFAULT_MARKERS = ("o", "s", "^", "D", "v", "p", "*", "h", "H", "<", ">", "P", "X", "d")


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "CMakeLists.txt").is_file():
            return parent
    raise RuntimeError("Unable to locate repository root from CUDA benchmark helper.")


REPO_ROOT = find_repo_root()
DEFAULT_EXECUTABLE = REPO_ROOT / "build" / "poisson_cuda"


def build_markers(names: Sequence[str]) -> dict[str, str]:
    marker_cycle = itertools.cycle(DEFAULT_MARKERS)
    return {name: next(marker_cycle) for name in names}


def resolve_executable() -> Path:
    override = os.environ.get("POISSON_CUDA_BIN")
    if override:
        return Path(override).expanduser()
    return DEFAULT_EXECUTABLE


def run_solver(
    *,
    solver: str,
    case: str,
    grid_size: int,
    dtype_cli: str,
    tol: float,
    max_iter: int,
    dim: int = 2,
    extra_args: Sequence[str] = (),
    executable: Path | None = None,
) -> dict[str, object]:
    exe = Path(executable).expanduser() if executable is not None else resolve_executable()
    if not exe.is_file():
        raise FileNotFoundError(
            f"CUDA executable not found at {exe}. "
            "Build the project first or set POISSON_CUDA_BIN."
        )

    cmd = [
        str(exe),
        "--dim",
        str(int(dim)),
        "--solver",
        solver,
        "--case",
        case,
        "--grid-size",
        str(int(grid_size)),
        "--dtype",
        dtype_cli,
        "--tol",
        f"{float(tol):.17g}",
        "--max-iter",
        str(int(max_iter)),
        *extra_args,
    ]

    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "CUDA solver executable failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc

    rows = list(csv.DictReader(completed.stdout.strip().splitlines()))
    if len(rows) != 1:
        raise RuntimeError(
            "Expected exactly one CSV row from the CUDA executable.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    row = rows[0]
    time_ms = float(row["time_ms"])
    time_ms_including_graph = float(
        row.get("time_ms_including_graph", row.get("time_ms_including_graph_ms", row["time_ms"]))
    )
    return {
        "solver": row["solver"],
        "backend": row["backend"],
        "dtype": row["dtype"],
        "dim": dim,
        "grid_size": int(row["grid_size"]),
        "iterations": int(row["iterations"]),
        "residual_l2": float(row["residual_l2"]),
        "error_l2": float(row["error_l2"]),
        "error_linf": float(row["error_linf"]),
        "time_ms": time_ms,
        "time_s": time_ms / 1000.0,
        "time_ms_including_graph": time_ms_including_graph,
        "time_s_including_graph": time_ms_including_graph / 1000.0,
    }
