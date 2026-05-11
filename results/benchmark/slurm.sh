#!/bin/bash
#SBATCH --account=MST113394
#SBATCH --job-name=OMP
#SBATCH --partition=hm112
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=32G
#SBATCH --time=96:00:00

set -euo pipefail

cd /home/u3518384/multigrid-poisson-solvers

mkdir -p results/benchmark/omp_v4

module load tools/miniconda3
conda activate RL

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=FALSE

THREADS_LIST=(1 2 4 7 8 14 16 28 32 56 64 112)

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"
echo "OMP_PLACES=$OMP_PLACES"
echo

lscpu | grep -E "CPU\\(s\\)|Thread\\(s\\)|Core\\(s\\)|Socket\\(s\\)|NUMA"

for t in "${THREADS_LIST[@]}"; do
    echo
    echo "========================================"
    echo "Running OMP_NUM_THREADS=$t"
    echo "========================================"

    export OMP_NUM_THREADS="$t"

    ./results/benchmark/benchmark_omp.sh
done