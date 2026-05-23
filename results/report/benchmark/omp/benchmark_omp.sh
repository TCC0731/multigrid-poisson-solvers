#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This script lives under results/report/benchmark/omp, so go back to the repo root.
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

BENCHMARK_BIN="${OMP_BENCHMARK_BIN:-$REPO_ROOT/build/poisson_benchmark_omp}"
OUTPUT_ROOT="${OMP_BENCHMARK_OUTPUT_ROOT:-$SCRIPT_DIR}"
THREADS_SPEC="${THREADS_LIST:-1 2 4 7 8 14 16 28 32 56 64 112}"
DIMS_SPEC="${DIMS_LIST:-2 3}"

if [[ ${1-} == "-h" || ${1-} == "--help" ]]; then
  cat <<'EOF'
Usage: benchmark_omp.sh [extra benchmark args...]

This driver runs the OpenMP benchmark for every thread count in THREADS_LIST
and every dimension in DIMS_LIST.

Environment:
  OMP_BENCHMARK_BIN           Benchmark binary to execute.
  OMP_BENCHMARK_OUTPUT_ROOT   Output directory root for CSV files.
  THREADS_LIST                Space- or comma-separated thread counts.
  DIMS_LIST                   Space- or comma-separated dimensions (2 or 3).
  OMP_PROC_BIND               OpenMP binding policy (default: close).
  OMP_PLACES                  OpenMP place list (default: cores).
  OMP_DYNAMIC                 OpenMP dynamic teams setting (default: FALSE).
EOF
  exit 0
fi

# Keep the runtime knobs explicit so benchmark runs are repeatable.
: "${OMP_PROC_BIND:=close}"
: "${OMP_PLACES:=cores}"
: "${OMP_DYNAMIC:=FALSE}"

export OMP_PROC_BIND
export OMP_PLACES
export OMP_DYNAMIC

THREADS_SPEC="${THREADS_SPEC//,/ }"
DIMS_SPEC="${DIMS_SPEC//,/ }"

read -r -a THREADS <<< "$THREADS_SPEC"
read -r -a DIMS <<< "$DIMS_SPEC"
EXTRA_ARGS=("$@")

if [[ ! -x "$BENCHMARK_BIN" ]]; then
  printf 'error: benchmark binary not found or not executable: %s\n' "$BENCHMARK_BIN" >&2
  exit 1
fi

if (( ${#THREADS[@]} == 0 )); then
  printf 'error: THREADS_LIST cannot be empty\n' >&2
  exit 1
fi

if (( ${#DIMS[@]} == 0 )); then
  printf 'error: DIMS_LIST cannot be empty\n' >&2
  exit 1
fi

for threads in "${THREADS[@]}"; do
  if [[ ! "$threads" =~ ^[1-9][0-9]*$ ]]; then
    printf 'error: invalid thread count: %s\n' "$threads" >&2
    exit 1
  fi
done

for dim in "${DIMS[@]}"; do
  case "$dim" in
    2|3) ;;
    *)
      printf 'error: invalid dimension: %s (expected 2 or 3)\n' "$dim" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$OUTPUT_ROOT"

printf 'OMP_PROC_BIND=%s OMP_PLACES=%s OMP_DYNAMIC=%s\n' \
  "$OMP_PROC_BIND" "$OMP_PLACES" "$OMP_DYNAMIC" >&2
printf 'Benchmark binary: %s\n' "$BENCHMARK_BIN" >&2
printf 'Output root: %s\n' "$OUTPUT_ROOT" >&2
printf 'Threads: %s\n' "${THREADS[*]}" >&2
printf 'Dimensions: %s\n' "${DIMS[*]}" >&2

run_one() {
  local dim="$1"
  local threads="$2"
  local output_base="$OUTPUT_ROOT/${dim}d/benchmark_omp_${threads}"

  mkdir -p "$(dirname "$output_base")"

  printf '\n=== dim=%s OMP_NUM_THREADS=%s ===\n' "$dim" "$threads" >&2
  printf 'Output base: %s\n' "$output_base" >&2

  OMP_NUM_THREADS="$threads" "$BENCHMARK_BIN" \
    "${EXTRA_ARGS[@]}" \
    --dim "$dim" \
    --output "$output_base"
}

for dim in "${DIMS[@]}"; do
  for threads in "${THREADS[@]}"; do
    run_one "$dim" "$threads"
  done
done
