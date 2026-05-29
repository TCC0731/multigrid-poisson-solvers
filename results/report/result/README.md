# Poisson Solver Wrapper 說明

這個資料夾放的是給 CUDA / OpenMP Poisson solver 使用的 Python wrapper。

主要目的很單純：

1. 接收 Python 端的參數
2. 自動選擇 CUDA 或 OpenMP 執行檔
3. 先查快取 CSV
4. 沒有相同參數的結果時才重新執行 solver
5. 把新結果寫回 CSV，並保持排序

## 主要檔案

- [`_poisson_wrapper.py`](./_poisson_wrapper.py)
  - 核心 wrapper 實作
  - 提供 request 建立、執行、快取讀寫、結果回傳
- [`omp_thread_scaling/run_and_plot.py`](./omp_thread_scaling/run_and_plot.py)
  - OpenMP thread-scaling 報表腳本
  - 預設用 `OMP_NUM_THREADS=1,2,4,6,8,12,16` 收集 SOR 與 MG V/W exact / SOR 的結果
  - 輸出 combined CSV、每個 dimension 的 CSV，以及對應的 thread-scaling 圖
- [`cuda_mg_phi_plots/run_and_plot.py`](./cuda_mg_phi_plots/run_and_plot.py)
  - CUDA MG V/W SOR 的 phi 視覺化腳本
  - 預設跑 `2D=4095`、`3D=383`，並輸出 `exact / MG / difference` 三聯圖
  - 3D 只取 `z=0.5` 截面，結果會分別寫到 `dumps/`、`plots/`，並打包成 `phi_results.pkl`
- `solver_results.csv`
  - wrapper 自動產生的共用快取檔，供所有報告腳本共用
  - 如果呼叫 `run_or_load(..., cache_csv=...)`，同一份結果也會同步寫到你指定的路徑
  - 若 native binary 重新編譯或 solver 邏輯有變，建議手動清掉再重跑

## 公開 API

### `build_request(...)`

把 Python 參數整理成標準化的 request 物件。

常用參數如下：

- `backend`: `cuda` 或 `omp`
- `dim`: `2` 或 `3`
- `dtype`: `float` 或 `double`
- `solver`: `jacobi`、`gs`、`sor`、`mg`
- `case`: `sine`、`mixed_sine`、`bubble`、`exp`、`cosine`
- `grid_size`: 正整數
- `tol`: 容許誤差，不給時會依 dtype 自動選預設值
- `max_iter`: 正整數
- `repeat_runs`: 重複執行次數，預設 `25`
- `cycle` / `nu` / `omega` / `mg_coarse`: 只在 `solver="mg"` 時使用
- `coarse_steps`: 只在 `solver="mg"` 且 `mg_coarse="sor"` 時使用，預設 `16`
- `omp_num_threads`: 只在 `backend="omp"` 時使用

### `run_or_load(...)`

這是主要入口。

如果快取裡已經有完全相同的參數組合，就直接回傳快取結果；如果沒有，就呼叫 native solver，讀取 stdout 輸出的 CSV，然後把新結果寫進快取檔。

預設會先讀寫 `results/report/result/solver_results.csv` 這份共用快取；如果你另外傳入 `cache_csv`，wrapper 也會把同一份結果鏡像到那個路徑，方便各個 benchmark script 保留自己的本地 cache。

範例：

```python
from _poisson_wrapper import run_or_load

result = run_or_load(
    backend="cuda",
    solver="mg",
    case="sine",
    grid_size=63,
)

print(result.time_ms)
print(result.time_s)
print(result.iterations)
```

## 回傳值

`run_or_load(...)` 會回傳 [`PoissonResult`](./_poisson_wrapper.py) 物件，內容包含：

- 請求參數
  - `backend`
  - `dim`
  - `dtype`
  - `solver`
  - `case`
  - `grid_size`
  - `tol`
  - `max_iter`
  - `repeat_runs`
  - `cycle`
  - `nu`
  - `omega`
  - `mg_coarse`
  - `coarse_steps`
  - `omp_num_threads`
- 計算結果
  - `iterations`
  - `residual_l2`
  - `error_l2`
  - `error_linf`
  - `time_ms`

另外還有一個方便使用的屬性：

- `time_s = time_ms / 1000.0`

## 快取規則

快取檔預設是：

```text
results/report/result/solver_results.csv
```

快取比對的 key 會包含完整參數，因此以下情況會被視為不同結果：

- CUDA 和 OpenMP
- 2D 和 3D
- 不同 grid size
- 不同 `repeat_runs`
- MG 的 `cycle`、`nu`、`omega`、`mg_coarse`
- MG 的 `coarse_steps` 也會納入 cache key，但只在 `mg_coarse="sor"` 時生效
- OpenMP 的 `omp_num_threads`

如果參數完全相同，wrapper 會直接讀 CSV，不會重跑 solver。

每次新增或更新一筆資料後，CSV 會重新排序。排序順序以參數為主，方便後續比對與人工檢查。

## 執行檔與環境變數

wrapper 預設會找這兩個執行檔：

- CUDA: `build/poisson_cuda`
- OpenMP: `build/poisson_cpp_omp`

也可以用環境變數覆蓋：

- `POISSON_CUDA_BIN`
- `POISSON_OMP_BIN`

OpenMP 執行時，wrapper 會固定這些環境變數，讓結果更穩定：

- `OMP_PROC_BIND=close`
- `OMP_PLACES=cores`
- `OMP_DYNAMIC=FALSE`

如果你想改 thread 數，可以直接傳 `omp_num_threads`。

## 備註

- 這個 wrapper 是 library-style 設計，主要供其他報告腳本或分析腳本 `import` 使用。
- 若 solver binary 有更新，建議刪掉 `solver_results.csv` 後重新收集結果，避免舊快取混入新資料。
- 目前 wrapper 不另外提供 CLI，重點是簡單、可擴充、方便在報告流程中重用。
