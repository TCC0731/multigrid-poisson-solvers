export OMP_NUM_THREADS=16
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=FALSE

perf record -g --call-graph dwarf -- \
  ./build/poisson_cpp_omp \
  --dim 3 --solver mg --case sine -n 383 \
  --cycle w --nu 3 --mg-coarse sor \
  --repeat-runs 1

perf report
perf annotate