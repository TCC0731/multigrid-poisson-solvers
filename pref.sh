: "${OMP_NUM_THREADS:=16}"
: "${OMP_PROC_BIND:=close}"
: "${OMP_PLACES:=cores}"

export OMP_NUM_THREADS
export OMP_PROC_BIND
export OMP_PLACES

echo $OMP_NUM_THREADS $OMP_PROC_BIND $OMP_PLACES

perf stat -r 1 -d -o results/benchmark/omp_v4/perf_mg_state_2.txt \
  ./build/poisson_cpp_omp --solver mg --case sine -n 4095 --omega 1.25 --cycle w --nu 3 --mg-coarse exact --max-iter 10 --tol 1e-9

perf record -g -o results/benchmark/omp_v4/perf_mg_2.data -- \
  ./build/poisson_cpp_omp --solver mg --case sine -n 4095 --omega 1.25 --cycle w --nu 3 --mg-coarse exact --max-iter 10 --tol 1e-9

perf report   -i results/benchmark/omp_v4/perf_mg_2.data   --stdio  > results/benchmark/omp_v4/perf_report_mg_2.txt