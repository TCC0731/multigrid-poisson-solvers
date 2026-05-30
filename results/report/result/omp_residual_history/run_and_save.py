from __future__ import annotations

"""Run OMP residual-history benchmarks and cache the raw traces in a PKL."""

import argparse
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR.parent
if str(RESULT_ROOT) not in sys.path:
    sys.path.insert(0, str(RESULT_ROOT))

from _history_common import (
    DEFAULT_BACKEND,
    DEFAULT_CASE,
    DEFAULT_DTYPE,
    DEFAULT_DIMS,
    DEFAULT_OMP_NUM_THREADS,
    DEFAULT_REPEAT_RUNS,
    DEFAULT_TOL,
    HistoryRecord,
    MODE_SPECS,
    collect_history_record,
    default_grid_size,
    write_payload,
)
from _report_common import positive_float, positive_int


DEFAULT_OUTPUT_NAME = "residual_history.pkl"


MODE_INDEX = {spec.mode_key: index for index, spec in enumerate(MODE_SPECS)}


def _record_sort_key(record: HistoryRecord) -> tuple[int, int]:
    return int(record.dimension), MODE_INDEX[str(record.mode_key)]


def _collect_records(
    *,
    case: str,
    dtype: str,
    tol: float,
    repeat_runs: int,
    omp_num_threads: int,
) -> list[HistoryRecord]:
    records: list[HistoryRecord] = []
    for dim in DEFAULT_DIMS:
        for mode in MODE_SPECS:
            grid_size = default_grid_size(dim, mode.solver)
            print(
                f"Running {dim}D {mode.label} | grid={grid_size} | "
                f"tol={tol:.0e} | threads={omp_num_threads}"
            )
            records.append(
                collect_history_record(
                    dim=dim,
                    case=case,
                    grid_size=grid_size,
                    mode=mode,
                    tol=tol,
                    repeat_runs=repeat_runs,
                    omp_num_threads=omp_num_threads,
                    dtype=dtype,
                )
            )
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OMP SOR / MG V-SOR / MG W-SOR and store the residual histories in a PKL."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory for the generated residual_history.pkl file.",
    )
    parser.add_argument("--case", default=DEFAULT_CASE, help="Poisson case to run.")
    parser.add_argument(
        "--dtype",
        default=DEFAULT_DTYPE,
        help="Floating-point dtype. This report is intended for double precision.",
    )
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL, help="Convergence tolerance.")
    parser.add_argument(
        "--omp-num-threads",
        type=int,
        default=DEFAULT_OMP_NUM_THREADS,
        help="OMP_NUM_THREADS passed to the native solver.",
    )
    parser.add_argument(
        "--repeat-runs",
        type=int,
        default=DEFAULT_REPEAT_RUNS,
        help="Repeat runs passed to the native solver.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = str(args.dtype).strip().lower()
    if dtype != DEFAULT_DTYPE:
        raise ValueError(f"Only dtype={DEFAULT_DTYPE!r} is supported in this residual-history report.")

    case = str(args.case).strip().lower()
    tol = positive_float(args.tol, "tol")
    omp_num_threads = positive_int(args.omp_num_threads, "omp_num_threads")
    repeat_runs = positive_int(args.repeat_runs, "repeat_runs")

    records = _collect_records(
        case=case,
        dtype=dtype,
        tol=tol,
        repeat_runs=repeat_runs,
        omp_num_threads=omp_num_threads,
    )
    if not records:
        raise RuntimeError("No residual-history records were collected.")

    records = sorted(records, key=_record_sort_key)
    grid_sizes: dict[int, dict[str, int]] = {}
    for record in records:
        grid_sizes.setdefault(int(record.dimension), {})[str(record.mode_key)] = int(record.grid_size)

    payload = {
        "schema_version": 1,
        "metadata": {
            "backend": DEFAULT_BACKEND,
            "case": case,
            "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "dims": list(DEFAULT_DIMS),
            "dtype": dtype,
            "grid_sizes": grid_sizes,
            "omp_num_threads": omp_num_threads,
            "repeat_runs": repeat_runs,
            "tol": tol,
            "modes": [asdict(spec) for spec in MODE_SPECS],
            "script": str(Path(__file__).resolve()),
        },
        "records": [asdict(record) for record in records],
    }

    output_path = output_dir / DEFAULT_OUTPUT_NAME
    write_payload(output_path, payload)
    print(f"Wrote {output_path}")
    print(f"Records: {len(records)}")


if __name__ == "__main__":
    main()
