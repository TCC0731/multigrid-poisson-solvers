下面是一個適合你這個 **Multigrid Poisson Solver GitHub repo** 的建議架構。重點是：

1. **所有 solver 統一解同一個形式：**

[
-\nabla^2 \phi = f
]

2. **所有版本共享同一套問題設定、誤差定義、輸出格式。**

3. **不是套件開發，而是研究 / 作業 / benchmark 型 repo。**

4. **Python、C++ OMP、C++ MPI、CUDA 都可以互相比較與替換。**

---

# 一、整體 repo 架構

建議 repo 名稱可以是：

```text
poisson-multigrid-solvers
```

整體結構：

```text
poisson-multigrid-solvers/
│
├── README.md
├── CMakeLists.txt
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── poisson2d_sin.json
│   ├── poisson3d_sin.json
│   ├── mg2d_omp.json
│   ├── mg3d_cuda.json
│   └── benchmark_suite.json
│
├── docs/
│   ├── math_convention.md
│   ├── discretization.md
│   ├── multigrid.md
│   ├── cuda_design.md
│   └── mpi_design.md
│
├── python/
│   ├── run_poisson.py
│   ├── problems.py
│   ├── operators.py
│   ├── metrics.py
│   ├── io_utils.py
│   │
│   └── solvers/
│       ├── jacobi_2d.py
│       ├── jacobi_3d.py
│       ├── rbgs_2d.py
│       ├── rbgs_3d.py
│       ├── sor_2d.py
│       ├── sor_3d.py
│       ├── mg_2d.py
│       └── mg_3d.py
│
├── cpp/
│   ├── include/
│   │   └── poisson/
│   │       ├── grid2d.hpp
│   │       ├── grid3d.hpp
│   │       ├── operators2d.hpp
│   │       ├── operators3d.hpp
│   │       ├── problems.hpp
│   │       ├── metrics.hpp
│   │       ├── io.hpp
│   │       ├── mg_level2d.hpp
│   │       └── mg_level3d.hpp
│   │
│   ├── omp/
│   │   ├── main_jacobi_2d.cpp
│   │   ├── main_jacobi_3d.cpp
│   │   ├── main_rbgs_2d.cpp
│   │   ├── main_rbgs_3d.cpp
│   │   ├── main_sor_2d.cpp
│   │   ├── main_sor_3d.cpp
│   │   ├── main_mg_2d.cpp
│   │   └── main_mg_3d.cpp
│   │
│   ├── mpi/
│   │   ├── main_mg_mpi_2d.cpp
│   │   ├── main_mg_mpi_3d.cpp
│   │   ├── domain_decomp2d.hpp
│   │   ├── domain_decomp3d.hpp
│   │   ├── halo_exchange2d.hpp
│   │   └── halo_exchange3d.hpp
│   │
│   └── common/
│       ├── timer.hpp
│       ├── config.hpp
│       └── index.hpp
│
├── cuda/
│   ├── include/
│   │   ├── cuda_utils.cuh
│   │   ├── kernels2d.cuh
│   │   ├── kernels3d.cuh
│   │   ├── mg_level2d.cuh
│   │   └── mg_level3d.cuh
│   │
│   ├── src/
│   │   ├── main_sor_2d.cu
│   │   ├── main_sor_3d.cu
│   │   ├── main_mg_2d.cu
│   │   └── main_mg_3d.cu
│   │
│   └── kernels/
│       ├── poisson2d_kernels.cu
│       ├── poisson3d_kernels.cu
│       ├── restriction2d.cu
│       ├── restriction3d.cu
│       ├── prolongation2d.cu
│       └── prolongation3d.cu
│
├── scripts/
│   ├── build_cpp.sh
│   ├── build_cuda.sh
│   ├── run_python_tests.sh
│   ├── run_cpp_tests.sh
│   ├── run_cuda_tests.sh
│   ├── run_benchmark.py
│   └── plot_results.py
│
├── tests/
│   ├── test_discretization.py
│   ├── test_python_solvers.py
│   ├── test_cpp_outputs.py
│   └── reference/
│       ├── poisson2d_sin_ref.npy
│       └── poisson3d_sin_ref.npy
│
├── results/
│   ├── raw/
│   ├── plots/
│   └── tables/
│
└── notes/
    ├── development_log.md
    └── todo.md
```

---

# 二、最重要的設計原則

## 1. 統一數學 convention

整個 repo 都只使用：

[
-\nabla^2 \phi = f
]

離散後寫成：

[
A\phi = f
]

其中在 2D：

[
A\phi\_{i,j}
===========

\frac{
4\phi*{i,j}
-\phi*{i+1,j}
-\phi*{i-1,j}
-\phi*{i,j+1}
-\phi\_{i,j-1}
}{h^2}
]

在 3D：

[
A\phi\_{i,j,k}
=============

\frac{
6\phi*{i,j,k}
-\phi*{i+1,j,k}
-\phi*{i-1,j,k}
-\phi*{i,j+1,k}
-\phi*{i,j-1,k}
-\phi*{i,j,k+1}
-\phi\_{i,j,k-1}
}{h^2}
]

residual 統一定義為：

[
r = f - A\phi
]

Multigrid 裡面解的是 error equation：

[
Ae = r
]

然後修正：

[
\phi \leftarrow \phi + e
]

這個 convention 一定要寫在：

```text
docs/math_convention.md
```

否則之後 Python、C++、CUDA 很容易正負號搞混。

---

## 2. 統一 grid convention

建議全部使用：

```text
interior grid size = N
array size = N + 2
```

例如 2D：

```text
phi.shape = (N + 2, N + 2)
```

3D：

```text
phi.shape = (N + 2, N + 2, N + 2)
```

其中：

```text
i = 0 和 i = N+1 是 boundary
j = 0 和 j = N+1 是 boundary
k = 0 和 k = N+1 是 boundary
```

interior 是：

```text
1 <= i <= N
1 <= j <= N
1 <= k <= N
```

對於 ([0,1]) domain：

[
h = \frac{1}{N+1}
]

interior 座標：

[
x_i = ih
]

---

# 三、版本矩陣

你最後的 repo 可以規劃成這樣：

| Solver          | Python + Numba | C++ + OMP | C++ + MPI |     CUDA |
| --------------- | -------------: | --------: | --------: | -------: |
| Jacobi 2D       |            yes |       yes |  optional | optional |
| Jacobi 3D       |            yes |       yes |  optional | optional |
| Gauss-Seidel 2D |            yes |       yes |        no |       no |
| Gauss-Seidel 3D |            yes |       yes |        no |       no |
| SOR 2D          |            yes |       yes |  optional |      yes |
| SOR 3D          |            yes |       yes |  optional |      yes |
| Multigrid 2D    |            yes |       yes |       yes |      yes |
| Multigrid 3D    |            yes |       yes |       yes |      yes |

對於 OMP / CUDA 的 Gauss-Seidel 和 SOR，建議實作 **red-black Gauss-Seidel / red-black SOR**，因為普通 Gauss-Seidel 有資料相依性，不適合直接平行化。

所以命名上建議明確寫：

```text
rbgs
rbsor
```

不要只寫 `gs`，避免之後混淆。

---

# 四、Python 部分架構

Python 版本主要用來：

1. 快速驗證數學正確性。
2. 當 C++ / CUDA 的 reference。
3. 畫圖、分析誤差、做收斂率測試。
4. 用 Numba 加速，但不追求最終極效能。

建議：

```text
python/
├── run_poisson.py
├── problems.py
├── operators.py
├── metrics.py
├── io_utils.py
└── solvers/
    ├── jacobi_2d.py
    ├── jacobi_3d.py
    ├── rbgs_2d.py
    ├── rbgs_3d.py
    ├── sor_2d.py
    ├── sor_3d.py
    ├── mg_2d.py
    └── mg_3d.py
```

其中：

## `problems.py`

負責定義解析解和右手邊 (f)。

例如：

[
\phi(x,y)=\sin(\pi x)\sin(\pi y)
]

則：

[
-\nabla^2 \phi = 2\pi^2 \sin(\pi x)\sin(\pi y)
]

3D 可以用：

[
\phi(x,y,z)=\sin(\pi x)\sin(\pi y)\sin(\pi z)
]

則：

[
-\nabla^2 \phi = 3\pi^2 \sin(\pi x)\sin(\pi y)\sin(\pi z)
]

---

## `operators.py`

放：

```python
apply_A_2d(phi, h)
apply_A_3d(phi, h)
compute_residual_2d(phi, f, h)
compute_residual_3d(phi, f, h)
```

---

## `metrics.py`

放：

```python
l2_error(phi, phi_exact)
linf_error(phi, phi_exact)
residual_norm(phi, f, h)
```

---

## `solvers/mg_2d.py`

放 multigrid 的核心：

```python
v_cycle_2d(phi, f, h, level)
restrict_2d(r)
prolongate_2d(e_c)
smooth_2d(phi, f, h)
```

---

# 五、C++ OMP 部分架構

C++ OMP 版本是 CPU 高效能版本。

建議不要做得太像 library，但可以有一些共用 header，避免 2D / 3D / MG 重複太多。

```text
cpp/
├── include/
│   └── poisson/
│       ├── grid2d.hpp
│       ├── grid3d.hpp
│       ├── operators2d.hpp
│       ├── operators3d.hpp
│       ├── problems.hpp
│       ├── metrics.hpp
│       ├── io.hpp
│       ├── mg_level2d.hpp
│       └── mg_level3d.hpp
│
├── omp/
│   ├── main_jacobi_2d.cpp
│   ├── main_jacobi_3d.cpp
│   ├── main_rbgs_2d.cpp
│   ├── main_rbgs_3d.cpp
│   ├── main_sor_2d.cpp
│   ├── main_sor_3d.cpp
│   ├── main_mg_2d.cpp
│   └── main_mg_3d.cpp
```

每一個 `main_xxx.cpp` 都可以獨立執行，例如：

```bash
./build/main_mg_2d configs/mg2d_omp.json
./build/main_sor_3d configs/poisson3d_sin.json
```

這樣你不是在做套件，而是在做一組可執行的 solver。

---

# 六、C++ MPI Multigrid 架構

MPI 版本建議只先做 Multigrid，不要一開始把 Jacobi / GS / SOR 都做成 MPI。

MPI MG 的重點是：

1. domain decomposition
2. halo exchange
3. 每一層 coarse grid 都要有對應的分割
4. residual restriction 後仍然需要分布在 MPI ranks 上
5. prolongation 後再回到 finer distributed grid

架構：

```text
cpp/mpi/
├── main_mg_mpi_2d.cpp
├── main_mg_mpi_3d.cpp
├── domain_decomp2d.hpp
├── domain_decomp3d.hpp
├── halo_exchange2d.hpp
└── halo_exchange3d.hpp
```

建議實作順序：

```text
1. 先做 2D MPI Jacobi smoothing + halo exchange
2. 再接 2D MPI residual
3. 再接 2D MPI restriction/prolongation
4. 完成 2D MPI MG
5. 最後再擴展到 3D
```

不要一開始直接寫 3D MPI MG，會太容易爆炸。

---

# 七、CUDA 部分架構

CUDA 版本建議分成：

1. Red-black SOR
2. Multigrid
3. 2D kernels
4. 3D kernels

```text
cuda/
├── include/
│   ├── cuda_utils.cuh
│   ├── kernels2d.cuh
│   ├── kernels3d.cuh
│   ├── mg_level2d.cuh
│   └── mg_level3d.cuh
│
├── src/
│   ├── main_sor_2d.cu
│   ├── main_sor_3d.cu
│   ├── main_mg_2d.cu
│   └── main_mg_3d.cu
│
└── kernels/
    ├── poisson2d_kernels.cu
    ├── poisson3d_kernels.cu
    ├── restriction2d.cu
    ├── restriction3d.cu
    ├── prolongation2d.cu
    └── prolongation3d.cu
```

CUDA SOR 一定建議用 red-black SOR：

```text
update red cells
update black cells
```

不要寫普通 SOR，因為普通 SOR 是 sequential dependency，不適合 GPU。

---

# 八、Multigrid 模組設計

每一個 MG 版本都應該有相同結構：

```text
MGLevel
├── phi
├── rhs
├── residual
├── error
├── nx, ny, nz
├── h
```

2D：

```cpp
struct MGLevel2D {
    int nx, ny;
    double h;
    std::vector<double> phi;
    std::vector<double> rhs;
    std::vector<double> residual;
};
```

3D：

```cpp
struct MGLevel3D {
    int nx, ny, nz;
    double h;
    std::vector<double> phi;
    std::vector<double> rhs;
    std::vector<double> residual;
};
```

CUDA 則是：

```cpp
struct MGLevel2DDevice {
    int nx, ny;
    double h;
    double* phi;
    double* rhs;
    double* residual;
};
```

核心函數統一命名：

```text
smooth()
compute_residual()
restrict_residual()
prolongate_and_correct()
v_cycle()
solve()
```

---

# 九、config 設計

你可以用 JSON，不一定要做複雜 CLI。

例如：

```json
{
    "dimension": 2,
    "nx": 128,
    "ny": 128,
    "nz": 1,

    "equation": "-laplacian_phi_equals_f",
    "problem": "sin_pi",
    "boundary": "dirichlet_zero",

    "solver": "multigrid",
    "backend": "omp",

    "smoother": "rb_sor",
    "omega": 1.8,

    "pre_smooth": 2,
    "post_smooth": 2,
    "coarse_iterations": 50,
    "max_cycles": 100,
    "tolerance": 1e-10,

    "output_solution": true,
    "output_prefix": "results/raw/mg2d_omp_128"
}
```

這樣你要替換 solver 只需要改：

```json
"solver": "jacobi"
```

或：

```json
"backend": "cuda"
```

---

# 十、統一輸出格式

所有版本都輸出同樣的 CSV row，方便比較。

例如：

```text
results/raw/benchmark.csv
```

欄位：

```text
backend,solver,dimension,nx,ny,nz,iterations,cycles,residual_l2,error_l2,error_linf,time_ms
```

例如：

```csv
backend,solver,dimension,nx,ny,nz,iterations,cycles,residual_l2,error_l2,error_linf,time_ms
python_numba,mg,2,128,128,1,0,12,1.2e-10,3.1e-5,8.0e-5,532.1
omp,mg,2,128,128,1,0,12,1.1e-10,3.0e-5,7.9e-5,42.5
cuda,mg,2,128,128,1,0,12,1.3e-10,3.1e-5,8.2e-5,3.8
```

這對最後寫報告非常重要。

---

# 十一、建議開發順序

我建議不要一開始就全部開發。比較穩的順序是：

| 階段     | 目標                                |
| -------- | ----------------------------------- |
| Stage 1  | Python Numba 2D Jacobi / RBGS / SOR |
| Stage 2  | Python Numba 2D MG                  |
| Stage 3  | 加入 3D Python MG                   |
| Stage 4  | C++ OMP 2D Jacobi / RBGS / SOR      |
| Stage 5  | C++ OMP 2D MG                       |
| Stage 6  | C++ OMP 3D MG                       |
| Stage 7  | CUDA 2D red-black SOR               |
| Stage 8  | CUDA 2D MG                          |
| Stage 9  | CUDA 3D MG                          |
| Stage 10 | MPI 2D MG                           |
| Stage 11 | MPI 3D MG                           |

最推薦先完成：

```text
Python 2D MG → C++ OMP 2D MG → CUDA 2D MG
```

因為這三個最容易互相比對。

---

# 十二、README 建議內容

README 可以這樣安排：

```text
# Poisson Multigrid Solvers

## Equation Convention

This repository solves:

    -∇²ϕ = f

## Implementations

- Python + Numba
- C++ + OpenMP
- C++ + MPI
- CUDA

## Supported Dimensions

- 2D
- 3D

## Solvers

- Jacobi
- Red-black Gauss-Seidel
- Red-black SOR
- Geometric Multigrid

## Build

### Python

pip install -r requirements.txt

### C++ / OpenMP

./scripts/build_cpp.sh

### CUDA

./scripts/build_cuda.sh

## Run Examples

python python/run_poisson.py configs/poisson2d_sin.json

./build/main_mg_2d configs/mg2d_omp.json

./build/main_mg_cuda_2d configs/mg3d_cuda.json

## Validation

Manufactured solution:

    ϕ(x,y) = sin(πx) sin(πy)

and

    -∇²ϕ = 2π² sin(πx) sin(πy)

## Benchmark

python scripts/run_benchmark.py configs/benchmark_suite.json
```

---

# 十三、最小可行版本 MVP

如果你想先快速把 repo 架起來，我建議 MVP 先只做這些：

```text
poisson-multigrid-solvers/
├── README.md
├── configs/
│   ├── poisson2d_sin.json
│   └── poisson3d_sin.json
├── python/
│   ├── run_poisson.py
│   ├── problems.py
│   ├── operators.py
│   ├── metrics.py
│   └── solvers/
│       ├── sor_2d.py
│       ├── sor_3d.py
│       ├── mg_2d.py
│       └── mg_3d.py
├── cpp/
│   ├── include/
│   │   └── poisson/
│   └── omp/
│       ├── main_sor_2d.cpp
│       ├── main_sor_3d.cpp
│       ├── main_mg_2d.cpp
│       └── main_mg_3d.cpp
├── cuda/
│   └── src/
│       ├── main_sor_2d.cu
│       └── main_mg_2d.cu
├── scripts/
└── results/
```

先不要碰 MPI。
等 Python、OMP、CUDA 的 2D / 3D 都穩定後，再加 MPI MG。

---

# 十四、我建議你的核心架構命名

為了讓後續很好維護，我會建議統一這些命名：

```text
jacobi_2d
jacobi_3d
rbgs_2d
rbgs_3d
rbsor_2d
rbsor_3d
mg_2d
mg_3d
```

不要混用：

```text
gs
gauss_seidel
red_black_gs
odd_even_gs
```

統一叫：

```text
rbgs
rbsor
```

因為這是你後續平行版本的核心。

---

# 十五、最推薦的實作策略

你的專案可以分成三層：

```text
problem layer
    ↓
operator layer
    ↓
solver layer
```

也就是：

```text
problem:
    exact solution, RHS f, boundary condition

operator:
    apply_A, residual, norm

solver:
    Jacobi, RBGS, RBSOR, MG
```

這樣未來要換問題，只改：

```text
problem layer
```

要換 solver，只改：

```text
solver layer
```

要換 backend，只換：

```text
python / cpp / cuda
```

整體比較不會亂掉。
