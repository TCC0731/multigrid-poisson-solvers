#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
RESULT_DIR="$SCRIPT_DIR"

# Default to the 16-thread 3D MG run used in the report, but keep the script
# configurable via environment variables so it can be reused for other cases.
THREADS="${OMP_PERF_THREADS:-16}"
DIM="${OMP_PERF_DIM:-3}"
GRID_SIZE="${OMP_PERF_GRID_SIZE:-383}"
SOLVER="${OMP_PERF_SOLVER:-mg}"
CASE_NAME="${OMP_PERF_CASE:-sine}"
CYCLE="${OMP_PERF_CYCLE:-w}"
NU="${OMP_PERF_NU:-3}"
MG_COARSE="${OMP_PERF_MG_COARSE:-sor}"
REPEAT_RUNS="${OMP_PERF_REPEAT_RUNS:-1}"

BENCHMARK_BIN="${OMP_PERF_BIN:-$REPO_ROOT/build/poisson_cpp_omp}"
DATA_FILE="$RESULT_DIR/omp_perf_${THREADS}.data"
REPORT_FILE="$RESULT_DIR/omp_perf_${THREADS}_report.txt"
ANNOTATE_FILE="$RESULT_DIR/omp_perf_${THREADS}_annotate.txt"
RUN_LOG_FILE="$RESULT_DIR/omp_perf_${THREADS}_run.log"

mkdir -p "$RESULT_DIR"

export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"

if [[ ! -x "$BENCHMARK_BIN" ]]; then
  printf 'error: benchmark binary not found or not executable: %s\n' "$BENCHMARK_BIN" >&2
  exit 1
fi

printf 'Benchmark binary: %s\n' "$BENCHMARK_BIN" >&2
printf 'Output directory: %s\n' "$RESULT_DIR" >&2
printf 'OMP_NUM_THREADS=%s OMP_PROC_BIND=%s OMP_PLACES=%s OMP_DYNAMIC=%s\n' \
  "$OMP_NUM_THREADS" "$OMP_PROC_BIND" "$OMP_PLACES" "$OMP_DYNAMIC" >&2
printf 'Problem: %sD grid=%s solver=%s case=%s cycle=%s nu=%s coarse=%s repeat-runs=%s\n' \
  "$DIM" "$GRID_SIZE" "$SOLVER" "$CASE_NAME" "$CYCLE" "$NU" "$MG_COARSE" "$REPEAT_RUNS" >&2
printf 'perf.data: %s\n' "$DATA_FILE" >&2
printf 'perf report text: %s\n' "$REPORT_FILE" >&2
printf 'perf annotate text: %s\n' "$ANNOTATE_FILE" >&2
printf 'run log: %s\n' "$RUN_LOG_FILE" >&2

perf record \
  -o "$DATA_FILE" \
  -g \
  --call-graph dwarf \
  -- \
  "$BENCHMARK_BIN" \
  --dim "$DIM" \
  --solver "$SOLVER" \
  --case "$CASE_NAME" \
  -n "$GRID_SIZE" \
  --cycle "$CYCLE" \
  --nu "$NU" \
  --mg-coarse "$MG_COARSE" \
  --repeat-runs "$REPEAT_RUNS" \
  >"$RUN_LOG_FILE" 2>&1

perf report \
  -i "$DATA_FILE" \
  --stdio \
  >"$REPORT_FILE"

# perf annotate needs debug info and good symbolization. Keep the report even if
# annotation is unavailable on the current build or host.
if ! perf annotate -i "$DATA_FILE" --stdio >"$ANNOTATE_FILE"; then
  printf 'warning: perf annotate failed; see %s for the main report\n' "$REPORT_FILE" >&2
fi
