#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Allow overrides from the environment, but provide deterministic defaults.
: "${OMP_NUM_THREADS:=1}"
: "${OMP_PROC_BIND:=close}"
: "${OMP_PLACES:=cores}"

export OMP_NUM_THREADS
export OMP_PROC_BIND
export OMP_PLACES

printf 'OMP_NUM_THREADS=%s OMP_PROC_BIND=%s OMP_PLACES=%s\n' \
  "$OMP_NUM_THREADS" "$OMP_PROC_BIND" "$OMP_PLACES" >&2

exec "$REPO_ROOT/build/poisson_benchmark_omp" \
  --output "$REPO_ROOT/results/benchmark/benchmark_omp_v2/benchmark_omp_v2_$OMP_NUM_THREADS" \
  "$@"


#./build/poisson_benchmark_cuda --output ./results/benchmark/cuda/benchmark_cuda_v8