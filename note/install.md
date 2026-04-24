# Installation Notes

This workflow assumes you are operating inside the `mg` conda environment. The rough order is:

1. Create the Python environment
2. Configure conda channels and the solver
3. Install Python scientific packages
4. Install the build toolchain
5. Install MPI
6. Install the CUDA Toolkit
7. Set environment variables and formatting tools

## 1. Create the Environment

Create a conda environment named `mg` and pin Python to 3.10. This version is usually stable for scientific packages and tends to work well with the rest of the toolchain.

```bash
conda create -n mg python=3.10 -y
conda activate mg
```

## 2. Configure conda Channels

This step resets the current environment's channels to `conda-forge` and `nvidia`, then sets channel priority to `strict` so packages from different channels do not conflict as easily.

```bash
conda config --env --remove-key channels 2>/dev/null || true
conda config --env --add channels conda-forge
conda config --env --add channels nvidia
conda config --env --set channel_priority strict
conda config --show channels
```

`conda config --env --remove-key channels 2>/dev/null || true` is used so that the command will not fail if the environment does not already have a `channels` setting.

## 3. Enable the libmamba Solver

`libmamba` is usually much faster than the default solver, especially when there are many package dependencies to resolve.

```bash
conda install -n base -c conda-forge conda-libmamba-solver -y
conda config --set solver libmamba
```

## 4. Install Python Packages

These are the common Python packages used for numerical computing, plotting, testing, and notebooks:

- `numpy`, `scipy`: numerical computing
- `numba`: JIT acceleration
- `matplotlib`: plotting
- `pandas`: data handling
- `tqdm`: progress bars
- `pytest`: testing
- `ipykernel`: Jupyter kernel support

```bash
conda install -y numpy scipy numba matplotlib pandas tqdm pytest ipykernel
```

Verify that Python and the packages load correctly:

```bash
python --version
python -c "import numpy, scipy, numba, matplotlib, pandas; print('Python packages OK')"
```

## 5. Install the Build Toolchain

This step ensures that C/C++, CMake, Ninja, and related dependencies are built with the same conda toolchain. Pinning `gcc_linux-64=12` and `gxx_linux-64=12` helps keep the GCC/G++ version stable, which is especially useful for CUDA and MPI builds.

```bash
conda install -y gcc_linux-64=12 gxx_linux-64=12 cmake ninja make pkg-config git nlohmann_json
```

Check that the compiler and build tools are installed correctly:

```bash
which gcc
which g++
which x86_64-conda-linux-gnu-gcc
which x86_64-conda-linux-gnu-g++
gcc --version
g++ --version
cmake --version
ninja --version
```

## 6. Install MPI

`mpich` and `mpi4py` are the packages you will use for parallel execution and Python MPI tests in this project.

```bash
conda install -c conda-forge -y mpich mpi4py
```

Verify the MPI runtime:

```bash
which mpicc
which mpicxx
mpirun --version
python -c "from mpi4py import MPI; print('rank', MPI.COMM_WORLD.Get_rank(), 'size', MPI.COMM_WORLD.Get_size())"
```

## 7. Install the CUDA Toolkit

The `12.8.*` version pin is usually there to keep the CUDA toolchain aligned with the version you want to use. `--override-channels` tells conda to resolve this install using only the specified channels.

```bash
conda install -y --override-channels -c nvidia -c conda-forge "cuda-toolkit=12.8.*" "cuda-nvcc_linux-64=12.8.*"
```

Check that the CUDA toolchain is available:

```bash
which nvcc
nvcc --version
nvidia-smi
```

## 8. Set Environment Variables

This block writes a script that will be loaded automatically after `conda activate mg`. That way, every time you enter the `mg` environment, the CUDA, compiler, and MPI variables are restored automatically.

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

cat > $CONDA_PREFIX/etc/conda/activate.d/mg_env.sh << 'EOF'
# CUDA headers and libraries live inside the conda prefix.
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX

# Put the conda-provided binaries and libraries at the front of the search path.
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH

# Use the conda toolchain for C/C++ compilation.
export CC=x86_64-conda-linux-gnu-gcc
export CXX=x86_64-conda-linux-gnu-g++
export OMPI_CC=$CC
export OMPI_CXX=$CXX
EOF
```

## 9. Install Formatting Tools

`black` and `isort` are common Python formatting tools, while `clang-format` is used for C/C++ code.

```bash
pip install black isort
conda install -y clang-format
```
