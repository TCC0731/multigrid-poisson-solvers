#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


THREAD_FILE_RE = re.compile(r"^benchmark_omp_(\d+)(?:_all)?\.csv$")

GROUP_FIELDS = (
    "suite",
    "backend",
    "dtype",
    "case",
    "solver",
    "grid_size",
    "max_iter",
    "tol",
    "warmup_runs",
    "timed_runs",
    "cycle",
    "omega",
    "nu",
)

OUTPUT_FIELDS = (
    "solver",
    "grid_size",
    "iterations",
    "mean_time_ms",
    "std_time_ms",
    "best_omp_num_threads",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the fastest OpenMP benchmark row for each solver/grid "
            "combination and write a compact CSV per dimension."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the 2d/ and 3d/ benchmark folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where the summary CSV files will be written.",
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        choices=(2, 3),
        default=(2, 3),
        help="Dimensions to summarize. Default: 2 3.",
    )
    return parser.parse_args()


def discover_thread_files(dim_dir: Path) -> list[tuple[int, Path]]:
    """Return the preferred CSV for each thread count.

    When both `benchmark_omp_<n>.csv` and `benchmark_omp_<n>_all.csv` exist,
    the `_all.csv` file wins because it carries the richer metadata.
    """

    preferred: dict[int, tuple[Path, bool]] = {}
    for path in dim_dir.glob("benchmark_omp_*.csv"):
        match = THREAD_FILE_RE.match(path.name)
        if match is None:
            continue

        threads = int(match.group(1))
        is_all = path.name.endswith("_all.csv")
        current = preferred.get(threads)
        if current is None or (is_all and not current[1]):
            preferred[threads] = (path, is_all)

    return sorted(((threads, path) for threads, (path, _) in preferred.items()), key=lambda item: item[0])


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def normalize_float(value: str, field_name: str, source: Path) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"invalid float in {source} for {field_name!r}: {value!r}") from exc


def build_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in GROUP_FIELDS)


def summarize_dimension(dim: int, input_root: Path, output_root: Path) -> Path:
    dim_dir = input_root / f"{dim}d"
    if not dim_dir.is_dir():
        raise FileNotFoundError(f"benchmark directory not found: {dim_dir}")

    thread_files = discover_thread_files(dim_dir)
    if not thread_files:
        raise FileNotFoundError(f"no benchmark CSV files found in {dim_dir}")

    best_rows: dict[tuple[str, ...], dict[str, str]] = {}
    best_times: dict[tuple[str, ...], float] = {}
    best_threads: dict[tuple[str, ...], int] = {}
    order: list[tuple[str, ...]] = []

    for threads, path in thread_files:
        for row in read_rows(path):
            key = build_key(row)
            candidate_time = normalize_float(row["mean_time_ms"], "mean_time_ms", path)

            current_time = best_times.get(key)
            if current_time is None:
                best_rows[key] = row
                best_times[key] = candidate_time
                best_threads[key] = threads
                order.append(key)
                continue

            current_threads = best_threads[key]

            if candidate_time < current_time or (candidate_time == current_time and threads < current_threads):
                best_rows[key] = row
                best_times[key] = candidate_time
                best_threads[key] = threads

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"benchmark_omp_best_threads_{dim}d.csv"

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for key in order:
            row = best_rows[key]
            writer.writerow(
                {
                    "solver": row["solver"],
                    "grid_size": row["grid_size"],
                    "iterations": row["iterations"],
                    "mean_time_ms": row["mean_time_ms"],
                    "std_time_ms": row["std_time_ms"],
                    "best_omp_num_threads": best_threads[key],
                }
            )

    return output_path


def main() -> int:
    args = parse_args()
    for dim in args.dims:
        output_path = summarize_dimension(dim, args.input_root, args.output_root)
        print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
