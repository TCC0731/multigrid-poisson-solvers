# CUDA 執行路徑說明

這份文件對應目前 repo 的 CUDA 實作，重點是把「實際執行順序」和「每一層分支怎麼跳」講清楚。

主要對應檔案：

- `cuda/src/main.cu`
- `cuda/include/poisson/cuda_utils.hpp`
- `cuda/src/jacobi.cu`
- `cuda/src/gs.cu`
- `cuda/src/sor.cu`
- `cuda/src/mg.cu`
- `cuda/kernels/*.cuh`
- `benchmark/main.cpp`
- `benchmark/poisson_benchmark.hpp`

---

## 1. CUDA 入口總覽

目前 CUDA 不是拆成很多個 `main_*.cu`，而是統一由 `cuda/src/main.cu` 當入口。

整體路徑可以先記成這條：

```text
poisson_cuda
  -> main()
  -> parse_args()
  -> run<float>() / run<double>()
  -> ensure_device_available()
  -> make_problem()
  -> validate_problem()
  -> solver dispatch
  -> solve_jacobi() / solve_gs() / solve_sor() / solve_mg_exact() / solve_mg_sor()
  -> cuda kernel launches
  -> residual check
  -> download result
  -> metrics + CSV output
```

如果把它拆成兩層看，會更好理解：

1. `main.cu` 負責「命令列解析、dtype 分支、solver 分支、最後輸出」。
2. `cuda_utils.hpp` 負責「device 記憶體、kernel 包裝、residual、restriction、prolongation」。

---

## 2. 命令列解析順序

`parse_args()` 的參數解析順序，就是程式裡 `if / else if` 的順序：

1. `-h` / `--help`
2. `--dtype`
3. `--case`
4. `--solver`
5. `--cycle`
6. `--nu`
7. `--omega`
8. `--mg-coarse`
9. `-n` / `--grid-size`
10. `--tol`
11. `--max-iter`

解析完之後，還會做一次固定檢查：

1. `dtype` 只能是 `float` 或 `double`
2. `tol` 若有提供，必須大於 0
3. `cycle` 只能是 `v` 或 `w`
4. `mg-coarse` 只能是 `exact` 或 `sor`

這裡有一個很重要的分支點：

- `--omega auto|default|none` 會把 `omega_is_auto` 打開
- `--omega VALUE` 會把 `omega` 當成數值存進去
- 如果根本沒給 `--omega`，那就是保留預設值

注意：這個 `--omega` 在目前程式裡主要是給 MG 路徑使用，不是給一般 `solve_sor()` 用。

---

## 3. `run<Real>()` 的執行順序

`main()` 只做兩件事：

1. 先呼叫 `parse_args()`
2. 再依 `dtype` 分支到 `run<float>()` 或 `run<double>()`

進到 `run<Real>()` 之後，實際順序是：

1. `poisson::cuda::ensure_device_available()`
2. `make_problem<Real>(options.case_name, options.grid_size)`
3. `validate_problem(problem)`
4. 決定 `tol`
5. 組出 `SolveOptions` / `MGOptions`
6. 依 `solver_name` 分支呼叫對應 solver
7. 計時結束
8. 用 `make_problem<double>()` 再做一次 metrics 比對
9. 輸出 CSV

這裡有幾個細節要注意：

1. `grid_size` 是 interior size，不是整張 array size。
2. 真正的 array size 會是 `interior_n + 2`。
3. `tol` 沒提供時，`float` 預設 `1e-6`，`double` 預設 `1e-10`。
4. solver 的計時只包住 solver 本體，不包含最後 metrics 計算。

---

## 4. Problem 建立順序

`make_problem()` 在 `cuda/src/problem.cu` 內完成，流程如下：

1. 根據 case 名稱選出 manufactured solution
2. 計算 `h = 1 / (interior_n + 1)`
3. 建立 `exact`、`rhs`、`phi0`
4. `phi0` 的 interior 先清成 0
5. `phi0` 的 boundary 直接抄 `exact`

所以 CUDA solver 一開始的狀態其實是：

- 邊界值已經是正確解析解
- interior 是 0
- `rhs` 和 `exact` 都已經準備好

接著 `validate_problem(problem)` 會再檢查：

1. grid 大小是否一致
2. `h` 是否正確
3. 數值是否 finite
4. boundary 是否真的等於 exact
5. interior 是否真的接近 0

只要驗證沒過，solver 就直接停掉。

---

## 5. solver 分支順序

`run<Real>()` 裡的 solver 分支是照這個順序檢查：

1. `jacobi`
2. `gs`
3. `sor`
4. `mg`

如果選到 `mg`，還會再多一層分支：

1. `mg-coarse=exact` -> `solve_mg_exact()`
2. `mg-coarse=sor` -> `solve_mg_sor()`

也就是說，MG 是唯一一個有「第二層分支」的 solver。

---

## 6. Jacobi 的路徑

`solve_jacobi()` 的執行順序是：

1. 驗證輸入
2. 確認 device 存在
3. 把 `phi0` 和 `rhs` upload 到 GPU
4. 另外準備一個 `work` grid
5. 反覆做 Jacobi step
6. 每次 step 後都算 residual
7. residual 達標就 download `work` 並回傳
8. 沒達標就 `swap(phi, work)` 繼續下一輪

它對應到 `cuda/src/jacobi.cu` 和 `cuda/kernels/jacobi_kernels.cuh`。

kernel 層做的事情很單純：

- 每個 thread 對應一個 interior cell
- 用舊的 `phi` 算新的 `work`
- 邊界不更新

Jacobi 的核心特色是：

- 舊解和新解是分開的
- 所以需要 `phi` 和 `work` 兩個 grid

---

## 7. GS 的路徑

`solve_gs()` 表面上叫 GS，但 CUDA 實作其實是 red-black Gauss-Seidel。

執行順序是：

1. 驗證輸入
2. 確認 device 存在
3. upload `phi0` 和 `rhs`
4. 每一輪只跑 `run_rb_sor_steps(phi, rhs, h, 1, 1)`
5. 每輪後算 residual
6. 收斂就回傳

這裡的關鍵是：

- `omega = 1`
- 所以它本質上就是 red-black GS

對應到：

- `cuda/src/gs.cu`
- `cuda/include/poisson/cuda_utils.hpp`
- `cuda/kernels/sor_kernels.cuh`

---

## 8. SOR 的路徑

`solve_sor()` 的流程跟 GS 很像，但差別在 relaxation factor：

1. 驗證輸入
2. 確認 device 存在
3. 算 `default_sor_omega(problem.interior_n)`
4. upload `phi0` 和 `rhs`
5. 每一輪做一次 red-black SOR
6. 每輪後算 residual
7. 收斂就回傳

這個 solver 也是 red-black SOR，不是傳統 sequential SOR。

重點：

- `solve_sor()` 會自己算最佳 `omega`
- `main.cu` 裡的 `--omega` 不會改變這個 direct SOR solver 的 `omega`

---

## 9. `run_rb_sor_steps()` 的 kernel 分枝

這是 CUDA 路徑裡很重要的一層，因為它決定每個更新週期的實際順序。

`run_rb_sor_steps()` 的步驟是：

1. 檢查 `phi` 和 `rhs` size 是否相同
2. 建立 2D grid / block
3. 計算 `h2 = h * h`
4. 每個 step 先跑 color 0
5. 再跑 color 1

也就是說，red-black 的更新順序永遠是：

```text
red cells -> black cells
```

而且 `check_kernel()` 會在每次 launch 後做：

1. `cudaGetLastError()`
2. `cudaDeviceSynchronize()`

所以這條路徑在程式上是「明確同步」的，不會把後面的 kernel 偷偷排在前面。

---

## 10. residual 是怎麼算的

`compute_relative_residual()` 的流程比較特別，因為它不是單一 kernel 就結束。

順序是：

1. 檢查 grid size
2. 算 interior points 數量
3. 根據 `kReductionThreads = 256` 決定 blocks
4. 配兩個暫存 device buffer
5. 跑 `relative_residual_partial_kernel`
6. 把 partial sums download 回 host
7. host 再把每個 block 的結果加總
8. 回傳 relative residual

它回傳的是：

- `sqrt(sum(diff^2) / sum((h^2 rhs)^2))`

如果 RHS 為 0，則退化成：

- `sqrt(sum(diff^2))`

這個 residual 會用在：

1. Jacobi / GS / SOR / MG 的收斂判斷
2. 最後 CSV 的 `residual_l2` 欄位

---

## 11. Multigrid 的路徑

MG 的執行順序是目前最複雜的部分。

外層入口是：

- `solve_mg_exact()`
- `solve_mg_sor()`

這兩個最後都會進到 `solve_mg_impl()`，差別只在 coarse solve mode。

### 11.1 先做輸入檢查

`solve_mg_impl()` 先做：

1. `validate_mg_inputs()`
2. `ensure_device_available()`
3. 決定 effective `omega`
4. upload `phi0`、`rhs`

### 11.2 外層 iteration

外層是：

1. 跑一次 `mg_cycle()`
2. 算 residual
3. 若 residual <= tol，就回傳
4. 否則進下一次 iteration

也就是說，`mg_cycle()` 本身不是「整個 solver」，它只是一次 V-cycle 或 W-cycle。

---

## 12. `mg_cycle()` 的遞迴順序

`mg_cycle()` 內部順序如下：

1. 先看目前這層的 interior size `n`
2. 如果 `n <= 4`，進 base case
3. 如果不是 base case，先做 `nu` 次 red-black SOR 當 pre-smoothing
4. 計算 fine residual
5. 把 residual restrict 到 coarse RHS
6. 建立 coarse error grid
7. 對 coarse error 做遞迴呼叫
8. 如果 cycle 是 `W`，再做第二次遞迴呼叫
9. prolongate coarse error 回 fine grid
10. 做 `nu` 次 red-black SOR 當 post-smoothing

### 12.1 Base case

當 `n <= 4` 時，會直接走 coarse solve：

1. `coarse_mode == Exact` -> `solve_coarsest_exact()`
2. `coarse_mode == Sor` -> `run_rb_sor_steps(..., coarse_steps)`

這裡的 `solve_coarsest_exact()` 不是 kernel，而是：

1. 把 coarse grid download 到 host
2. 在 host 上組 dense linear system
3. Gaussian elimination
4. 再 upload 回 device

所以 MG 的最深層會「回到 CPU 做 exact solve」。

### 12.2 V-cycle

如果 `cycle == V`，遞迴只做一次。

### 12.3 W-cycle

如果 `cycle == W`，遞迴會再做第二次。

這段對應到程式裡很明確的分支：

1. 先呼叫一次 `mg_cycle(...)`
2. 如果是 `W`，再呼叫一次 `mg_cycle(...)`

---

## 13. restriction / prolongation 的路徑

MG 的 grid transfer 都在 `cuda/include/poisson/cuda_utils.hpp` 和 `cuda/kernels/mg_kernels.cuh`。

### 13.1 residual 計算

`compute_residual_full()` 會先把 `residual_out` 清零，然後用 kernel 算完整 residual grid。

### 13.2 restriction

`restrict_full_weighting()` 會把 fine residual 用 full-weighting 壓到 coarse grid。

### 13.3 prolongation

`prolong_add()` 會把 coarse correction 插值回 fine grid，並且直接加到 fine solution 上。

這三個步驟合起來，就是 MG 的核心資料流：

```text
fine phi
  -> residual
  -> restrict
  -> coarse error solve
  -> prolong
  -> fine correction
```

---

## 14. 結果輸出順序

solver 跑完之後，`run<Real>()` 會做最後輸出：

1. 取得 `result.phi`
2. 用 `make_problem<double>()` 建 metrics 用的參考問題
3. 計算 `error_l2` 和 `error_linf`
4. 印 CSV header
5. 印一行結果

輸出欄位是：

```text
solver,backend,dtype,grid_size,iterations,residual_l2,error_l2,error_linf,time_ms
```

如果是 MG，solver 名稱會再被改成：

- `mg_exact`
- `mg_sor`

---

## 15. benchmark 路徑

如果不是直接跑 `poisson_cuda`，而是跑 benchmark executable，那入口會換成 `benchmark/main.cpp`。

benchmark 的順序是：

1. `main()`
2. `parse_args()`
3. `run_suite<double>(suite, kBackendLabel)`
4. `make_solver_comparison_rows()`
5. `make_mg_compare_rows()`
6. 用同一批 solver API 跑 warmup + timed runs
7. 輸出 CSV

對 CUDA 來說，backend label 是在編譯時塞進去的：

- `POISSON_BENCHMARK_BACKEND_LABEL=cuda`

所以 `benchmark_cuda.csv` 本質上是：

1. 用同一套 CUDA solver API
2. 只是被 benchmark 包裝起來
3. 多了 warmup / timed runs / CSV 檔輸出

補充一下：目前 benchmark 的 MG compare 走的是 `solve_mg_exact()`，不是 `solve_mg_sor()`。

---

## 16. 一句話總結

這個 CUDA 路徑可以濃縮成一句話：

1. 先在 host 解析參數與建 problem
2. 再依 solver 分支進入 Jacobi / RBGS / RB-SOR / MG
3. 每個 solver 都透過 `cuda_utils.hpp` 觸發 kernel
4. MG 會在 V/W-cycle、exact/sor coarse solve、pre/post smoothing 之間再做第二層分支
5. 最後把結果下載回 host，算 error metrics，輸出 CSV

