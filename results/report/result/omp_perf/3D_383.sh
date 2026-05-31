#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
RESULT_DIR="$SCRIPT_DIR"

# Keep the run configurable, but default to the same 16-thread 3D MG case used
# by the existing perf-record profile.
THREADS="${OMP_PERF_THREADS:-16}"
DIM="${OMP_PERF_DIM:-3}"
GRID_SIZE="${OMP_PERF_GRID_SIZE:-383}"
SOLVER="${OMP_PERF_SOLVER:-mg}"
CASE_NAME="${OMP_PERF_CASE:-sine}"
CYCLE="${OMP_PERF_CYCLE:-w}"
NU="${OMP_PERF_NU:-3}"
MG_COARSE="${OMP_PERF_MG_COARSE:-sor}"
REPEAT_RUNS="${OMP_PERF_REPEAT_RUNS:-1}"

# perf stat is the counter-based pass. Keep the events configurable so the same
# script can be reused if the host exposes a slightly different event name set.
STAT_EVENTS="${OMP_PERF_STAT_EVENTS:-}"
STAT_REPEATS="${OMP_PERF_STAT_REPEATS:-3}"
STAT_ONLY="${OMP_PERF_STAT_ONLY:-0}"

BENCHMARK_BIN="${OMP_PERF_BIN:-$REPO_ROOT/build/poisson_cpp_omp}"
PREFIX="omp_perf_${THREADS}_3D_383"
DATA_FILE="$RESULT_DIR/${PREFIX}.data"
REPORT_FILE="$RESULT_DIR/${PREFIX}_report.txt"
ANNOTATE_FILE="$RESULT_DIR/${PREFIX}_annotate.txt"
STAT_FILE="$RESULT_DIR/${PREFIX}_stat.txt"
RUN_LOG_FILE="$RESULT_DIR/${PREFIX}_run.log"
STAT_RUN_LOG_FILE="$RESULT_DIR/${PREFIX}_stat_run.log"

mkdir -p "$RESULT_DIR"

export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"

if [[ ! -x "$BENCHMARK_BIN" ]]; then
  printf 'error: benchmark binary not found or not executable: %s\n' "$BENCHMARK_BIN" >&2
  exit 1
fi

RUN_ARGS=(
  --dim "$DIM"
  --solver "$SOLVER"
  --case "$CASE_NAME"
  -n "$GRID_SIZE"
  --cycle "$CYCLE"
  --nu "$NU"
  --mg-coarse "$MG_COARSE"
  --repeat-runs "$REPEAT_RUNS"
)

printf 'Benchmark binary: %s\n' "$BENCHMARK_BIN" >&2
printf 'Output directory: %s\n' "$RESULT_DIR" >&2
printf 'OMP_NUM_THREADS=%s OMP_PROC_BIND=%s OMP_PLACES=%s OMP_DYNAMIC=%s\n' \
  "$OMP_NUM_THREADS" "$OMP_PROC_BIND" "$OMP_PLACES" "$OMP_DYNAMIC" >&2
printf 'Problem: %sD grid=%s solver=%s case=%s cycle=%s nu=%s coarse=%s repeat-runs=%s\n' \
  "$DIM" "$GRID_SIZE" "$SOLVER" "$CASE_NAME" "$CYCLE" "$NU" "$MG_COARSE" "$REPEAT_RUNS" >&2
printf 'perf.data: %s\n' "$DATA_FILE" >&2
printf 'perf report text: %s\n' "$REPORT_FILE" >&2
printf 'perf annotate text: %s\n' "$ANNOTATE_FILE" >&2
printf 'perf stat text: %s\n' "$STAT_FILE" >&2
printf 'record run log: %s\n' "$RUN_LOG_FILE" >&2
printf 'stat run log: %s\n' "$STAT_RUN_LOG_FILE" >&2
if [[ -n "$STAT_EVENTS" ]]; then
  printf 'perf stat events: %s\n' "$STAT_EVENTS" >&2
else
  printf 'perf stat events: perf stat -d -d -d default detailed counters\n' >&2
fi
printf 'stat only: %s\n' "$STAT_ONLY" >&2
printf 'perf stat repeats: %s\n' "$STAT_REPEATS" >&2

if [[ "$STAT_ONLY" != "1" ]]; then
  perf record \
    -o "$DATA_FILE" \
    -g \
    --call-graph dwarf \
    -- \
    "$BENCHMARK_BIN" \
    "${RUN_ARGS[@]}" \
    >"$RUN_LOG_FILE" 2>&1

  perf report \
    -i "$DATA_FILE" \
    --stdio \
    >"$REPORT_FILE"

  # perf annotate needs debug info and a symbolized build. Keep the main report
  # even if annotation is unavailable on the current host.
  if ! perf annotate -i "$DATA_FILE" --stdio >"$ANNOTATE_FILE"; then
    printf 'warning: perf annotate failed; see %s for the main report\n' "$REPORT_FILE" >&2
  fi
fi

# Counter-based pass for memory-bottleneck evidence.
STAT_CMD=(
  perf stat
  -d -d -d
  -o "$STAT_FILE"
  -r "$STAT_REPEATS"
)

if [[ -n "$STAT_EVENTS" ]]; then
  STAT_CMD+=(-e "$STAT_EVENTS")
fi

STAT_CMD+=(
  --
  "$BENCHMARK_BIN"
  "${RUN_ARGS[@]}"
)

"${STAT_CMD[@]}" >"$STAT_RUN_LOG_FILE" 2>&1