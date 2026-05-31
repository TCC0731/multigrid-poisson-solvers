"""Cache-backed wrapper for the CUDA and OpenMP Poisson solvers.

This module keeps the public surface small:

* build a normalized request from Python-friendly arguments
* resolve the native CUDA/OpenMP executable
* run the solver only on cache miss
* persist the full result table in the shared cache, and optionally mirror it
  to a caller-provided CSV

The cache key is parameter-based only. If the native binaries change, clear the
CSV manually.
"""

from __future__ import annotations

import csv
import io
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

CASES = ("sine", "mixed_sine", "bubble", "exp", "cosine")
DTYPES = ("float", "double")
SOLVERS = ("jacobi", "gs", "sor", "mg")
BACKENDS = ("cuda", "omp")
MG_CYCLES = ("v", "w")
MG_COARSE_MODES = ("exact", "sor")
AUTO_OMEGA_ALIASES = {"auto", "default", "none"}
DEFAULT_REPEAT_RUNS = 25
DEFAULT_OMP_ENV = {
    "OMP_PROC_BIND": "close",
    "OMP_PLACES": "cores",
    "OMP_DYNAMIC": "FALSE",
}
CSV_COLUMNS = (
    "backend",
    "dim",
    "dtype",
    "solver",
    "case",
    "grid_size",
    "tol",
    "max_iter",
    "repeat_runs",
    "cycle",
    "nu",
    "omega",
    "mg_coarse",
    "coarse_steps",
    "omp_num_threads",
    "iterations",
    "residual_l2",
    "error_l2",
    "error_linf",
    "time_ms",
)
LEGACY_CSV_COLUMNS = tuple(column for column in CSV_COLUMNS if column != "coarse_steps")


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "CMakeLists.txt").is_file():
            return parent
    raise RuntimeError("Unable to locate repository root from _poisson_wrapper.py")


REPO_ROOT = _find_repo_root()
RESULT_DIR = Path(__file__).resolve().parent
SHARED_CACHE_CSV = RESULT_DIR / "solver_results.csv"
DEFAULT_CACHE_CSV = SHARED_CACHE_CSV
DEFAULT_CUDA_EXECUTABLE = REPO_ROOT / "build" / "poisson_cuda"
DEFAULT_OMP_EXECUTABLE = REPO_ROOT / "build" / "poisson_cpp_omp"


def _normalize_token(value: str) -> str:
    return value.strip().lower()


def _positive_int(value: object, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _positive_float(value: object, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if parsed <= 0.0:
        raise ValueError(f"{name} must be a positive number")
    return parsed


def _normalize_backend(value: str) -> str:
    backend = _normalize_token(value)
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS!r}")
    return backend


def _normalize_dtype(value: str) -> str:
    dtype = _normalize_token(value)
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES!r}")
    return dtype


def _normalize_solver(value: str) -> str:
    solver = _normalize_token(value)
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS!r}")
    return solver


def _normalize_case(value: str) -> str:
    case = _normalize_token(value)
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES!r}")
    return case


def _normalize_cycle(value: str) -> str:
    cycle = _normalize_token(value)
    if cycle not in MG_CYCLES:
        raise ValueError(f"cycle must be one of {MG_CYCLES!r}")
    return cycle


def _normalize_mg_coarse(value: str) -> str:
    mg_coarse = _normalize_token(value)
    if mg_coarse not in MG_COARSE_MODES:
        raise ValueError(f"mg_coarse must be one of {MG_COARSE_MODES!r}")
    return mg_coarse


def _resolve_tol(dtype: str, tol: object | None) -> float:
    if tol is None:
        return 1e-6 if dtype == "float" else 1e-10
    return _positive_float(tol, "tol")


def _normalize_omega(value: object | None) -> float | str:
    if value is None:
        return "auto"
    if isinstance(value, str):
        normalized = _normalize_token(value)
        if normalized in AUTO_OMEGA_ALIASES:
            return "auto"
        try:
            parsed = float(normalized)
        except ValueError as exc:
            raise ValueError("omega must be a positive number or 'auto'") from exc
    else:
        parsed = float(value)

    if not (0.0 < parsed <= 2.0):
        raise ValueError("omega must be in the interval (0, 2]")
    return parsed


def _resolve_coarse_steps(value: object | None) -> int:
    if value is None:
        return 16
    return _positive_int(value, "coarse_steps")


def _resolve_omp_num_threads(value: object | None) -> int:
    if value is not None:
        return _positive_int(value, "omp_num_threads")

    env_value = os.environ.get("OMP_NUM_THREADS")
    if env_value:
        return _positive_int(env_value, "OMP_NUM_THREADS")
    return 1


def _format_float(value: float) -> str:
    return repr(float(value))


def _format_optional_int(value: int | None) -> str:
    return "" if value is None else str(int(value))


def _format_optional_omega(value: float | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return _format_float(float(value))


def _parse_optional_int(value: str | None, field_name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"invalid integer for {field_name!r}: {value!r}") from exc


def _parse_optional_positive_int(value: str | None, field_name: str) -> int | None:
    parsed = _parse_optional_int(value, field_name)
    if parsed is not None and parsed < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _parse_required_int(value: str, field_name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"invalid integer for {field_name!r}: {value!r}") from exc


def _parse_required_float(value: str, field_name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"invalid float for {field_name!r}: {value!r}") from exc


def _parse_optional_omega(value: str | None) -> float | str | None:
    if value is None or value == "":
        return None
    if value == "auto":
        return "auto"
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"invalid omega value: {value!r}") from exc


def _omega_sort_key(value: float | str | None) -> tuple[int, float]:
    if value is None:
        return (0, -1.0)
    if isinstance(value, str):
        return (2, 0.0)
    return (1, float(value))


def _choose_default_executable(backend: str) -> Path:
    if backend == "cuda":
        return DEFAULT_CUDA_EXECUTABLE
    if backend == "omp":
        return DEFAULT_OMP_EXECUTABLE
    raise ValueError(f"unknown backend: {backend}")


def _env_override_name(backend: str) -> str:
    if backend == "cuda":
        return "POISSON_CUDA_BIN"
    if backend == "omp":
        return "POISSON_OMP_BIN"
    raise ValueError(f"unknown backend: {backend}")


@dataclass(frozen=True)
class PoissonRequest:
    backend: str
    dim: int = 2
    dtype: str = "double"
    solver: str = "jacobi"
    case: str = "sine"
    grid_size: int = 31
    tol: float = 1e-10
    max_iter: int = 20_000
    repeat_runs: int = DEFAULT_REPEAT_RUNS
    cycle: str | None = None
    nu: int | None = None
    omega: float | str | None = None
    mg_coarse: str | None = None
    coarse_steps: int | None = None
    omp_num_threads: int | None = None

    def cache_key(self) -> tuple[object, ...]:
        return (
            self.backend,
            self.dim,
            self.dtype,
            self.solver,
            self.case,
            self.grid_size,
            self.tol,
            self.max_iter,
            self.repeat_runs,
            self.cycle or "",
            self.nu if self.nu is not None else "",
            self.omega if self.omega is not None else "",
            self.mg_coarse or "",
            self.coarse_steps if self.coarse_steps is not None else "",
            self.omp_num_threads if self.omp_num_threads is not None else "",
        )

    def sort_key(self) -> tuple[object, ...]:
        return (
            self.backend,
            self.dim,
            self.dtype,
            self.solver,
            self.case,
            self.grid_size,
            self.tol,
            self.max_iter,
            self.repeat_runs,
            self.cycle or "",
            self.nu if self.nu is not None else -1,
            _omega_sort_key(self.omega),
            self.mg_coarse or "",
            self.coarse_steps if self.coarse_steps is not None else -1,
            self.omp_num_threads if self.omp_num_threads is not None else -1,
        )

    def to_row_dict(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "dim": str(self.dim),
            "dtype": self.dtype,
            "solver": self.solver,
            "case": self.case,
            "grid_size": str(self.grid_size),
            "tol": _format_float(self.tol),
            "max_iter": str(self.max_iter),
            "repeat_runs": str(self.repeat_runs),
            "cycle": self.cycle or "",
            "nu": _format_optional_int(self.nu),
            "omega": _format_optional_omega(self.omega),
            "mg_coarse": self.mg_coarse or "",
            "coarse_steps": _format_optional_int(self.coarse_steps),
            "omp_num_threads": _format_optional_int(self.omp_num_threads),
        }


@dataclass(frozen=True)
class PoissonResult(PoissonRequest):
    iterations: int = 0
    residual_l2: float = 0.0
    error_l2: float = 0.0
    error_linf: float = 0.0
    time_ms: float = 0.0

    @property
    def time_s(self) -> float:
        return self.time_ms / 1000.0

    def to_row_dict(self) -> dict[str, str]:
        row = super().to_row_dict()
        row.update(
            {
                "iterations": str(self.iterations),
                "residual_l2": _format_float(self.residual_l2),
                "error_l2": _format_float(self.error_l2),
                "error_linf": _format_float(self.error_linf),
                "time_ms": _format_float(self.time_ms),
            }
        )
        return row

    @classmethod
    def from_row_dict(cls, row: Mapping[str, str]) -> "PoissonResult":
        solver = _normalize_solver(row["solver"])
        backend = _normalize_backend(row["backend"])
        mg_coarse = _normalize_token(row["mg_coarse"]) or None
        coarse_steps = _parse_optional_positive_int(row.get("coarse_steps"), "coarse_steps")
        if solver == "mg" and mg_coarse == "sor" and backend == "cuda":
            coarse_steps_value = 16 if coarse_steps is None else coarse_steps
        else:
            coarse_steps_value = None

        request = PoissonRequest(
            backend=backend,
            dim=_parse_required_int(row["dim"], "dim"),
            dtype=_normalize_dtype(row["dtype"]),
            solver=solver,
            case=_normalize_case(row["case"]),
            grid_size=_parse_required_int(row["grid_size"], "grid_size"),
            tol=_parse_required_float(row["tol"], "tol"),
            max_iter=_parse_required_int(row["max_iter"], "max_iter"),
            repeat_runs=_parse_required_int(row["repeat_runs"], "repeat_runs"),
            cycle=_normalize_token(row["cycle"]) or None,
            nu=_parse_optional_int(row["nu"], "nu"),
            omega=_parse_optional_omega(row["omega"]),
            mg_coarse=mg_coarse,
            coarse_steps=coarse_steps_value,
            omp_num_threads=_parse_optional_int(row["omp_num_threads"], "omp_num_threads"),
        )
        return cls(
            **vars(request),
            iterations=_parse_required_int(row["iterations"], "iterations"),
            residual_l2=_parse_required_float(row["residual_l2"], "residual_l2"),
            error_l2=_parse_required_float(row["error_l2"], "error_l2"),
            error_linf=_parse_required_float(row["error_linf"], "error_linf"),
            time_ms=_parse_required_float(row["time_ms"], "time_ms"),
        )


def build_request(
    *,
    backend: str,
    dim: int = 2,
    dtype: str = "double",
    solver: str = "jacobi",
    case: str = "sine",
    grid_size: int = 31,
    tol: object | None = None,
    max_iter: int = 20_000,
    repeat_runs: int = DEFAULT_REPEAT_RUNS,
    cycle: str = "v",
    nu: int = 2,
    omega: object | None = 1.0,
    mg_coarse: str = "exact",
    coarse_steps: object | None = 16,
    omp_num_threads: object | None = None,
) -> PoissonRequest:
    backend = _normalize_backend(backend)
    dim = _positive_int(dim, "dim")
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    dtype = _normalize_dtype(dtype)
    solver = _normalize_solver(solver)
    case = _normalize_case(case)
    grid_size = _positive_int(grid_size, "grid_size")
    max_iter = _positive_int(max_iter, "max_iter")
    repeat_runs = _positive_int(repeat_runs, "repeat_runs")
    tol_value = _resolve_tol(dtype, tol)

    if solver == "mg":
        cycle_value = _normalize_cycle(cycle)
        nu_value = _positive_int(nu, "nu")
        omega_value = _normalize_omega(omega)
        mg_coarse_value = _normalize_mg_coarse(mg_coarse)
        coarse_steps_value = (
            _resolve_coarse_steps(coarse_steps)
            if mg_coarse_value == "sor" and backend == "cuda"
            else None
        )
    else:
        cycle_value = None
        nu_value = None
        omega_value = None
        mg_coarse_value = None
        coarse_steps_value = None

    if backend == "omp":
        omp_threads_value = _resolve_omp_num_threads(omp_num_threads)
    else:
        omp_threads_value = None

    return PoissonRequest(
        backend=backend,
        dim=dim,
        dtype=dtype,
        solver=solver,
        case=case,
        grid_size=grid_size,
        tol=tol_value,
        max_iter=max_iter,
        repeat_runs=repeat_runs,
        cycle=cycle_value,
        nu=nu_value,
        omega=omega_value,
        mg_coarse=mg_coarse_value,
        coarse_steps=coarse_steps_value,
        omp_num_threads=omp_threads_value,
    )


def resolve_executable(
    backend: str,
    executable: Path | str | None = None,
) -> Path:
    backend = _normalize_backend(backend)
    if executable is not None:
        path = Path(executable).expanduser()
    else:
        override = os.environ.get(_env_override_name(backend))
        path = Path(override).expanduser() if override else _choose_default_executable(backend)

    if not path.is_file():
        raise FileNotFoundError(
            f"{backend.upper()} executable not found at {path}. "
            f"Build the project first or set {_env_override_name(backend)}."
        )
    return path


def _build_command(request: PoissonRequest, executable: Path) -> list[str]:
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
        _format_float(request.tol),
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
                str(request.nu if request.nu is not None else 2),
                "--omega",
                _format_optional_omega(request.omega),
                "--mg-coarse",
                request.mg_coarse or "exact",
            ]
        )
        if request.mg_coarse == "sor" and request.backend == "cuda":
            cmd.extend(
                [
                    "--coarse-steps",
                    str(request.coarse_steps if request.coarse_steps is not None else 16),
                ]
            )
    return cmd


def _build_env(request: PoissonRequest) -> dict[str, str] | None:
    if request.backend != "omp":
        return None

    env = os.environ.copy()
    env.update(DEFAULT_OMP_ENV)
    env["OMP_NUM_THREADS"] = str(request.omp_num_threads if request.omp_num_threads is not None else 1)
    return env


def _run_solver(request: PoissonRequest, executable: Path) -> PoissonResult:
    cmd = _build_command(request, executable)
    env = _build_env(request)

    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Poisson solver failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc

    output = completed.stdout.strip()
    if not output:
        raise RuntimeError(
            "Solver did not produce CSV output.\n"
            f"Command: {' '.join(cmd)}"
        )

    rows = list(csv.DictReader(io.StringIO(output)))
    if len(rows) != 1:
        raise RuntimeError(
            "Expected exactly one CSV row from the solver.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    row = rows[0]
    return PoissonResult(
        backend=request.backend,
        dim=request.dim,
        dtype=request.dtype,
        solver=request.solver,
        case=request.case,
        grid_size=request.grid_size,
        tol=request.tol,
        max_iter=request.max_iter,
        repeat_runs=request.repeat_runs,
        cycle=request.cycle,
        nu=request.nu,
        omega=request.omega,
        mg_coarse=request.mg_coarse,
        coarse_steps=request.coarse_steps,
        omp_num_threads=request.omp_num_threads,
        iterations=_parse_required_int(row["iterations"], "iterations"),
        residual_l2=_parse_required_float(row["residual_l2"], "residual_l2"),
        error_l2=_parse_required_float(row["error_l2"], "error_l2"),
        error_linf=_parse_required_float(row["error_linf"], "error_linf"),
        time_ms=_parse_required_float(row["time_ms"], "time_ms"),
    )


def load_cache(cache_csv: Path | str) -> dict[tuple[object, ...], PoissonResult]:
    path = Path(cache_csv).expanduser()
    if not path.exists() or path.stat().st_size == 0:
        return {}

    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return {}

    reader = csv.DictReader(io.StringIO(content))
    fieldnames = list(reader.fieldnames or [])
    if fieldnames not in (list(CSV_COLUMNS), list(LEGACY_CSV_COLUMNS)):
        raise ValueError(
            f"Unexpected cache CSV header in {path}. "
            f"Expected {list(CSV_COLUMNS)!r} or {list(LEGACY_CSV_COLUMNS)!r}, "
            f"got {reader.fieldnames!r}"
        )

    records: dict[tuple[object, ...], PoissonResult] = {}
    for row in reader:
        result = PoissonResult.from_row_dict(row)
        records[result.cache_key()] = result
    return records


def _write_cache(cache_csv: Path, records: dict[tuple[object, ...], PoissonResult]) -> None:
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    ordered_results = sorted(records.values(), key=lambda item: item.sort_key())

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=str(cache_csv.parent),
            prefix=f".{cache_csv.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            temp_path = Path(tmp_file.name)
            writer = csv.DictWriter(tmp_file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for result in ordered_results:
                writer.writerow(result.to_row_dict())
        temp_path.replace(cache_csv)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def _resolve_cache_paths(cache_csv: Path | str | None) -> tuple[Path, ...]:
    paths = [SHARED_CACHE_CSV]
    if cache_csv is not None:
        extra_path = Path(cache_csv).expanduser().resolve()
        if extra_path != SHARED_CACHE_CSV:
            paths.append(extra_path)
    return tuple(paths)


def _load_cache_bundle(
    cache_paths: tuple[Path, ...],
) -> tuple[
    dict[tuple[object, ...], PoissonResult],
    dict[Path, dict[tuple[object, ...], PoissonResult]],
]:
    merged_records: dict[tuple[object, ...], PoissonResult] = {}
    loaded_records: dict[Path, dict[tuple[object, ...], PoissonResult]] = {}
    for path in cache_paths:
        path_records = load_cache(path)
        loaded_records[path] = path_records
        for key, result in path_records.items():
            merged_records.setdefault(key, result)
    return merged_records, loaded_records


def _cache_bundle_needs_sync(
    merged_records: dict[tuple[object, ...], PoissonResult],
    loaded_records: Mapping[Path, dict[tuple[object, ...], PoissonResult]],
) -> bool:
    return any(path_records != merged_records for path_records in loaded_records.values())


def _sync_cache_bundle(
    cache_paths: tuple[Path, ...],
    records: dict[tuple[object, ...], PoissonResult],
) -> None:
    for path in cache_paths:
        _write_cache(path, records)


def run_or_load(
    *,
    request: PoissonRequest | None = None,
    backend: str | None = None,
    dim: int = 2,
    dtype: str = "double",
    solver: str = "jacobi",
    case: str = "sine",
    grid_size: int = 31,
    tol: object | None = None,
    max_iter: int = 20_000,
    repeat_runs: int = DEFAULT_REPEAT_RUNS,
    cycle: str = "v",
    nu: int = 2,
    omega: object | None = 1.0,
    mg_coarse: str = "exact",
    coarse_steps: object | None = 16,
    omp_num_threads: object | None = None,
    cache_csv: Path | str | None = None,
    executable: Path | str | None = None,
) -> PoissonResult:
    if request is None:
        if backend is None:
            raise TypeError("backend is required when request is not provided")
        request = build_request(
            backend=backend,
            dim=dim,
            dtype=dtype,
            solver=solver,
            case=case,
            grid_size=grid_size,
            tol=tol,
            max_iter=max_iter,
            repeat_runs=repeat_runs,
            cycle=cycle,
            nu=nu,
            omega=omega,
            mg_coarse=mg_coarse,
            coarse_steps=coarse_steps,
            omp_num_threads=omp_num_threads,
        )
    elif backend is not None:
        raise TypeError("pass either request or keyword parameters, not both")

    # Keep the shared cache authoritative, and mirror any caller-specific cache.
    cache_paths = _resolve_cache_paths(cache_csv)
    records, loaded_records = _load_cache_bundle(cache_paths)

    cached = records.get(request.cache_key())
    if cached is not None:
        if _cache_bundle_needs_sync(records, loaded_records):
            _sync_cache_bundle(cache_paths, records)
        return cached

    executable_path = resolve_executable(request.backend, executable)
    result = _run_solver(request, executable_path)
    records[result.cache_key()] = result
    _sync_cache_bundle(cache_paths, records)
    return result


__all__ = [
    "AUTO_OMEGA_ALIASES",
    "BACKENDS",
    "CASES",
    "CSV_COLUMNS",
    "DEFAULT_CACHE_CSV",
    "DEFAULT_CUDA_EXECUTABLE",
    "DEFAULT_OMP_ENV",
    "DEFAULT_OMP_EXECUTABLE",
    "DEFAULT_REPEAT_RUNS",
    "DTYPES",
    "MG_CYCLES",
    "MG_COARSE_MODES",
    "PoissonRequest",
    "PoissonResult",
    "REPO_ROOT",
    "RESULT_DIR",
    "SHARED_CACHE_CSV",
    "SOLVERS",
    "build_request",
    "load_cache",
    "resolve_executable",
    "run_or_load",
]
