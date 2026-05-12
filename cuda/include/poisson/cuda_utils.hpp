#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "poisson/grid2d.hpp"
#include "poisson/grid3d.hpp"

#if defined(__has_include)
#  if __has_include(<nvtx3/nvtx3.hpp>)
#    include <nvtx3/nvtx3.hpp>
#    define POISSON_CUDA_HAS_NVTX3 1
#  else
#    define POISSON_CUDA_HAS_NVTX3 0
#  endif
#else
#  define POISSON_CUDA_HAS_NVTX3 0
#endif

#include "../../kernels/common.cuh"
#include "../../kernels/jacobi_kernels.cuh"
#include "../../kernels/mg_kernels.cuh"
#include "../../kernels/reduction_kernels.cuh"
#include "../../kernels/sor_kernels.cuh"

namespace poisson::cuda {

// Keep NVTX instrumentation optional so the CUDA build still works when the
// profiler headers are not installed in the current environment.
namespace detail {

#if POISSON_CUDA_HAS_NVTX3
class ScopedNvtxRange {
public:
    explicit ScopedNvtxRange(const char* message)
        : range_{message} {}

    explicit ScopedNvtxRange(const std::string& message)
        : range_{message.c_str()} {}

private:
    nvtx3::scoped_range range_;
};
#else
class ScopedNvtxRange {
public:
    explicit ScopedNvtxRange(const char*) {}
    explicit ScopedNvtxRange(const std::string&) {}
};
#endif

} // namespace detail

#undef POISSON_CUDA_HAS_NVTX3

inline void check(cudaError_t status, const char* expr, const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }

    std::ostringstream oss;
    oss << "CUDA error at " << file << ':' << line << " for " << expr << ": "
        << cudaGetErrorString(status);
    throw std::runtime_error(oss.str());
}

inline void check_kernel(const char* kernel_name) {
    check(cudaPeekAtLastError(), kernel_name, __FILE__, __LINE__);
#ifndef NDEBUG
    check(cudaDeviceSynchronize(), kernel_name, __FILE__, __LINE__);
#endif
}

inline void ensure_device_available() {
    int device_count = 0;
    check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", __FILE__, __LINE__);
    if (device_count < 1) {
        throw std::runtime_error("no CUDA device detected");
    }
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count)
        : count_{count} {
        if (count_ > 0) {
            check(
                cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)),
                "cudaMalloc",
                __FILE__,
                __LINE__
            );
        }
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            (void)cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : count_{std::exchange(other.count_, 0)},
          ptr_{std::exchange(other.ptr_, nullptr)} {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                (void)cudaFree(ptr_);
            }
            count_ = std::exchange(other.count_, 0);
            ptr_ = std::exchange(other.ptr_, nullptr);
        }
        return *this;
    }

    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] T* data() noexcept { return ptr_; }
    [[nodiscard]] const T* data() const noexcept { return ptr_; }

    void zero() {
        if (count_ > 0) {
            check(cudaMemset(ptr_, 0, count_ * sizeof(T)), "cudaMemset", __FILE__, __LINE__);
        }
    }

    void upload(const T* host_ptr, std::size_t count) {
        if (count != count_) {
            throw std::invalid_argument("device buffer upload size mismatch");
        }
        if (count_ > 0) {
            check(
                cudaMemcpy(ptr_, host_ptr, count_ * sizeof(T), cudaMemcpyHostToDevice),
                "cudaMemcpyHostToDevice",
                __FILE__,
                __LINE__
            );
        }
    }

    void download(T* host_ptr, std::size_t count) const {
        if (count != count_) {
            throw std::invalid_argument("device buffer download size mismatch");
        }
        if (count_ > 0) {
            check(
                cudaMemcpy(host_ptr, ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost),
                "cudaMemcpyDeviceToHost",
                __FILE__,
                __LINE__
            );
        }
    }

private:
    std::size_t count_{0};
    T* ptr_{nullptr};
};

template <typename Real>
class DeviceGrid2D {
public:
    DeviceGrid2D() = default;

    explicit DeviceGrid2D(std::size_t size)
        : size_{size}, buffer_{size * size} {}

    explicit DeviceGrid2D(const Grid2D<Real>& host_grid)
        : DeviceGrid2D(host_grid.size()) {
        upload(host_grid);
    }

    DeviceGrid2D(const DeviceGrid2D&) = delete;
    DeviceGrid2D& operator=(const DeviceGrid2D&) = delete;
    DeviceGrid2D(DeviceGrid2D&&) noexcept = default;
    DeviceGrid2D& operator=(DeviceGrid2D&&) noexcept = default;

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t elements() const noexcept { return size_ * size_; }
    [[nodiscard]] Real* data() noexcept { return buffer_.data(); }
    [[nodiscard]] const Real* data() const noexcept { return buffer_.data(); }

    void zero() {
        buffer_.zero();
    }

    void upload(const Grid2D<Real>& host_grid) {
        if (host_grid.size() != size_) {
            throw std::invalid_argument("device grid upload size mismatch");
        }
        buffer_.upload(host_grid.data().data(), host_grid.elements());
    }

    [[nodiscard]] Grid2D<Real> download() const {
        Grid2D<Real> host_grid{size_};
        buffer_.download(host_grid.data().data(), host_grid.elements());
        return host_grid;
    }

private:
    std::size_t size_{0};
    DeviceBuffer<Real> buffer_{};
};

template <typename Real>
class DeviceGrid3D {
public:
    DeviceGrid3D() = default;

    explicit DeviceGrid3D(std::size_t size)
        : size_{size}, buffer_{size * size * size} {}

    explicit DeviceGrid3D(const Grid3D<Real>& host_grid)
        : DeviceGrid3D(host_grid.size()) {
        upload(host_grid);
    }

    DeviceGrid3D(const DeviceGrid3D&) = delete;
    DeviceGrid3D& operator=(const DeviceGrid3D&) = delete;
    DeviceGrid3D(DeviceGrid3D&&) noexcept = default;
    DeviceGrid3D& operator=(DeviceGrid3D&&) noexcept = default;

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t elements() const noexcept { return size_ * size_ * size_; }
    [[nodiscard]] Real* data() noexcept { return buffer_.data(); }
    [[nodiscard]] const Real* data() const noexcept { return buffer_.data(); }

    void zero() {
        buffer_.zero();
    }

    void upload(const Grid3D<Real>& host_grid) {
        if (host_grid.size() != size_) {
            throw std::invalid_argument("device grid upload size mismatch");
        }
        buffer_.upload(host_grid.data().data(), host_grid.elements());
    }

    [[nodiscard]] Grid3D<Real> download() const {
        Grid3D<Real> host_grid{size_};
        buffer_.download(host_grid.data().data(), host_grid.elements());
        return host_grid;
    }

private:
    std::size_t size_{0};
    DeviceBuffer<Real> buffer_{};
};

template <typename Real>
class DeviceGridView2D {
public:
    DeviceGridView2D() = default;

    DeviceGridView2D(Real* data, std::size_t size)
        : size_{size}, data_{data} {
        if (size_ > 0 && data_ == nullptr) {
            throw std::invalid_argument("device grid view cannot own a null data pointer");
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t elements() const noexcept { return size_ * size_; }
    [[nodiscard]] Real* data() noexcept { return data_; }
    [[nodiscard]] const Real* data() const noexcept { return data_; }

    void zero() {
        if (elements() > 0) {
            check(cudaMemset(data_, 0, elements() * sizeof(Real)), "cudaMemset", __FILE__, __LINE__);
        }
    }

    void upload(const Grid2D<Real>& host_grid) {
        if (host_grid.size() != size_) {
            throw std::invalid_argument("device grid view upload size mismatch");
        }
        if (elements() > 0) {
            check(
                cudaMemcpy(
                    data_,
                    host_grid.data().data(),
                    elements() * sizeof(Real),
                    cudaMemcpyHostToDevice
                ),
                "cudaMemcpyHostToDevice",
                __FILE__,
                __LINE__
            );
        }
    }

    [[nodiscard]] Grid2D<Real> download() const {
        Grid2D<Real> host_grid{size_};
        if (elements() > 0) {
            check(
                cudaMemcpy(
                    host_grid.data().data(),
                    data_,
                    elements() * sizeof(Real),
                    cudaMemcpyDeviceToHost
                ),
                "cudaMemcpyDeviceToHost",
                __FILE__,
                __LINE__
            );
        }
        return host_grid;
    }

private:
    std::size_t size_{0};
    Real* data_{nullptr};
};

template <typename Real>
class DeviceGridView3D {
public:
    DeviceGridView3D() = default;

    DeviceGridView3D(Real* data, std::size_t size)
        : size_{size}, data_{data} {
        if (size_ > 0 && data_ == nullptr) {
            throw std::invalid_argument("device grid view cannot own a null data pointer");
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t elements() const noexcept { return size_ * size_ * size_; }
    [[nodiscard]] Real* data() noexcept { return data_; }
    [[nodiscard]] const Real* data() const noexcept { return data_; }

    void zero() {
        if (elements() > 0) {
            check(cudaMemset(data_, 0, elements() * sizeof(Real)), "cudaMemset", __FILE__, __LINE__);
        }
    }

    void upload(const Grid3D<Real>& host_grid) {
        if (host_grid.size() != size_) {
            throw std::invalid_argument("device grid view upload size mismatch");
        }
        if (elements() > 0) {
            check(
                cudaMemcpy(
                    data_,
                    host_grid.data().data(),
                    elements() * sizeof(Real),
                    cudaMemcpyHostToDevice
                ),
                "cudaMemcpyHostToDevice",
                __FILE__,
                __LINE__
            );
        }
    }

    [[nodiscard]] Grid3D<Real> download() const {
        Grid3D<Real> host_grid{size_};
        if (elements() > 0) {
            check(
                cudaMemcpy(
                    host_grid.data().data(),
                    data_,
                    elements() * sizeof(Real),
                    cudaMemcpyDeviceToHost
                ),
                "cudaMemcpyDeviceToHost",
                __FILE__,
                __LINE__
            );
        }
        return host_grid;
    }

private:
    std::size_t size_{0};
    Real* data_{nullptr};
};

class RelativeResidualWorkspace {
public:
    RelativeResidualWorkspace() = default;

    explicit RelativeResidualWorkspace(std::size_t array_n) {
        reserve_for(array_n);
    }

    void reserve_for(std::size_t array_n) {
        const std::size_t total_points = (array_n > 2) ? (array_n - 2) * (array_n - 2) : 0;
        const int blocks = total_points == 0
            ? 0
            : static_cast<int>(
                  std::min<std::size_t>(
                      static_cast<std::size_t>(65'535),
                      (total_points + static_cast<std::size_t>(cuda_kernels::kReductionThreads) - 1) /
                          static_cast<std::size_t>(cuda_kernels::kReductionThreads)
                  )
              );

        blocks_ = blocks;
        if (static_cast<std::size_t>(blocks_) <= capacity_) {
            return;
        }

        partial_sums_ =
            DeviceBuffer<cuda_kernels::ResidualPair>{static_cast<std::size_t>(blocks_)};
        scratch_sums_ =
            DeviceBuffer<cuda_kernels::ResidualPair>{static_cast<std::size_t>(blocks_)};
        capacity_ = static_cast<std::size_t>(blocks_);
    }

    [[nodiscard]] int blocks() const noexcept { return blocks_; }
    [[nodiscard]] cuda_kernels::ResidualPair* partial_sums() noexcept {
        return partial_sums_.data();
    }
    [[nodiscard]] cuda_kernels::ResidualPair* scratch_sums() noexcept {
        return scratch_sums_.data();
    }
    [[nodiscard]] double* final_residual() noexcept {
        return final_residual_.data();
    }
    [[nodiscard]] double* rhs_norm2() noexcept {
        return rhs_norm2_.data();
    }
    [[nodiscard]] bool rhs_norm_cached(std::size_t array_n, double h2) const noexcept {
        return rhs_norm_cached_ && rhs_norm_array_n_ == array_n && rhs_norm_h2_ == h2;
    }
    void mark_rhs_norm_cached(std::size_t array_n, double h2) noexcept {
        rhs_norm_cached_ = true;
        rhs_norm_array_n_ = array_n;
        rhs_norm_h2_ = h2;
    }

private:
    std::size_t capacity_{0};
    int blocks_{0};
    bool rhs_norm_cached_{false};
    std::size_t rhs_norm_array_n_{0};
    double rhs_norm_h2_{0.0};
    DeviceBuffer<cuda_kernels::ResidualPair> partial_sums_{};
    DeviceBuffer<cuda_kernels::ResidualPair> scratch_sums_{};
    DeviceBuffer<double> final_residual_{1};
    DeviceBuffer<double> rhs_norm2_{1};
};

class RelativeResidualWorkspace3D {
public:
    RelativeResidualWorkspace3D() = default;

    explicit RelativeResidualWorkspace3D(std::size_t array_n) {
        reserve_for(array_n);
    }

    void reserve_for(std::size_t array_n) {
        const std::size_t interior_n = (array_n > 2) ? (array_n - 2) : 0;
        const std::size_t total_points = interior_n * interior_n * interior_n;
        const int blocks = total_points == 0
            ? 0
            : static_cast<int>(
                  std::min<std::size_t>(
                      static_cast<std::size_t>(65'535),
                      (total_points + static_cast<std::size_t>(cuda_kernels::kReductionThreads) - 1) /
                          static_cast<std::size_t>(cuda_kernels::kReductionThreads)
                  )
              );

        blocks_ = blocks;
        if (static_cast<std::size_t>(blocks_) <= capacity_) {
            return;
        }

        partial_sums_ =
            DeviceBuffer<cuda_kernels::ResidualPair>{static_cast<std::size_t>(blocks_)};
        scratch_sums_ =
            DeviceBuffer<cuda_kernels::ResidualPair>{static_cast<std::size_t>(blocks_)};
        capacity_ = static_cast<std::size_t>(blocks_);
    }

    [[nodiscard]] int blocks() const noexcept { return blocks_; }
    [[nodiscard]] cuda_kernels::ResidualPair* partial_sums() noexcept {
        return partial_sums_.data();
    }
    [[nodiscard]] cuda_kernels::ResidualPair* scratch_sums() noexcept {
        return scratch_sums_.data();
    }
    [[nodiscard]] double* final_residual() noexcept {
        return final_residual_.data();
    }
    [[nodiscard]] double* rhs_norm2() noexcept {
        return rhs_norm2_.data();
    }
    [[nodiscard]] bool rhs_norm_cached(std::size_t array_n, double h2) const noexcept {
        return rhs_norm_cached_ && rhs_norm_array_n_ == array_n && rhs_norm_h2_ == h2;
    }
    void mark_rhs_norm_cached(std::size_t array_n, double h2) noexcept {
        rhs_norm_cached_ = true;
        rhs_norm_array_n_ = array_n;
        rhs_norm_h2_ = h2;
    }

private:
    std::size_t capacity_{0};
    int blocks_{0};
    bool rhs_norm_cached_{false};
    std::size_t rhs_norm_array_n_{0};
    double rhs_norm_h2_{0.0};
    DeviceBuffer<cuda_kernels::ResidualPair> partial_sums_{};
    DeviceBuffer<cuda_kernels::ResidualPair> scratch_sums_{};
    DeviceBuffer<double> final_residual_{1};
    DeviceBuffer<double> rhs_norm2_{1};
};

[[nodiscard]] inline std::size_t reduction_output_count(std::size_t input_count) {
    const std::size_t elements_per_block = static_cast<std::size_t>(cuda_kernels::kReductionThreads) * 2;
    return (input_count + elements_per_block - 1) / elements_per_block;
}

inline constexpr std::size_t kNonMgResidualCheckInterval = 50;

[[nodiscard]] inline bool should_check_non_mg_residual(
    std::size_t iteration, std::size_t max_iter
) noexcept {
    return iteration == 1 || iteration == max_iter ||
        (iteration % kNonMgResidualCheckInterval) == 0;
}

inline cuda_kernels::ResidualPair* reduce_residual_pairs(
    std::size_t active_count,
    cuda_kernels::ResidualPair* input,
    cuda_kernels::ResidualPair* output,
    int threads,
    std::size_t shared_bytes
) {
    const detail::ScopedNvtxRange range{"cuda::reduce_residual_pairs"};
    while (active_count > 1) {
        const std::size_t next_count = reduction_output_count(active_count);
        cuda_kernels::reduce_residual_pairs_kernel<>
            <<<static_cast<int>(next_count), threads, shared_bytes>>>(input, output, active_count);
        check_kernel("reduce_residual_pairs_kernel");
        active_count = next_count;
        std::swap(input, output);
    }
    return input;
}

template <typename Real>
void ensure_rhs_norm_cached(
    const DeviceGrid2D<Real>& rhs,
    Real h,
    RelativeResidualWorkspace& workspace
) {
    const detail::ScopedNvtxRange range{"cuda::ensure_rhs_norm_cached"};
    const Real h2 = h * h;
    if (workspace.rhs_norm_cached(rhs.size(), static_cast<double>(h2))) {
        return;
    }

    workspace.reserve_for(rhs.size());

    const int threads = cuda_kernels::kReductionThreads;
    const int blocks = workspace.blocks();
    const std::size_t shared_bytes =
        static_cast<std::size_t>(threads) * sizeof(cuda_kernels::ResidualPair);

    cuda_kernels::rhs_norm_partial_kernel<Real>
        <<<blocks, threads, shared_bytes>>>(rhs.data(), rhs.size(), h2, workspace.partial_sums());
    check_kernel("rhs_norm_partial_kernel");

    auto* totals = reduce_residual_pairs(
        static_cast<std::size_t>(blocks),
        workspace.partial_sums(),
        workspace.scratch_sums(),
        threads,
        shared_bytes
    );

    cuda_kernels::store_rhs_norm_kernel<>
        <<<1, 1>>>(totals, workspace.rhs_norm2());
    check_kernel("store_rhs_norm_kernel");
    workspace.mark_rhs_norm_cached(rhs.size(), static_cast<double>(h2));
}

template <typename Real>
void ensure_rhs_norm_cached(
    const DeviceGrid3D<Real>& rhs,
    Real h,
    RelativeResidualWorkspace3D& workspace
) {
    const detail::ScopedNvtxRange range{"cuda::ensure_rhs_norm_cached_3d"};
    const Real h2 = h * h;
    if (workspace.rhs_norm_cached(rhs.size(), static_cast<double>(h2))) {
        return;
    }

    workspace.reserve_for(rhs.size());

    const int threads = cuda_kernels::kReductionThreads;
    const int blocks = workspace.blocks();
    const std::size_t shared_bytes =
        static_cast<std::size_t>(threads) * sizeof(cuda_kernels::ResidualPair);

    cuda_kernels::rhs_norm_partial_kernel_3d<Real>
        <<<blocks, threads, shared_bytes>>>(rhs.data(), rhs.size(), h2, workspace.partial_sums());
    check_kernel("rhs_norm_partial_kernel_3d");

    auto* totals = reduce_residual_pairs(
        static_cast<std::size_t>(blocks),
        workspace.partial_sums(),
        workspace.scratch_sums(),
        threads,
        shared_bytes
    );

    cuda_kernels::store_rhs_norm_kernel<>
        <<<1, 1>>>(totals, workspace.rhs_norm2());
    check_kernel("store_rhs_norm_kernel");
    workspace.mark_rhs_norm_cached(rhs.size(), static_cast<double>(h2));
}

template <typename Real, typename PhiGrid, typename RhsGrid>
void run_rb_sor_steps(
    PhiGrid& phi,
    const RhsGrid& rhs,
    Real h,
    Real omega,
    std::size_t steps
) {
    if (phi.size() != rhs.size()) {
        throw std::invalid_argument("phi and rhs device grid sizes do not match");
    }

    const detail::ScopedNvtxRange range{"cuda::run_rb_sor_steps"};
    const std::size_t interior_n = phi.size() - 2;
    const dim3 block = cuda_kernels::make_block_2d();
    const std::size_t active_columns = (interior_n + 1) / 2;
    const dim3 grid = cuda_kernels::make_grid_2d(active_columns, interior_n, block);
    const Real h2 = h * h;

    for (std::size_t step = 0; step < steps; ++step) {
        for (int color = 0; color < 2; ++color) {
            cuda_kernels::rb_sor_color_kernel<Real>
                <<<grid, block>>>(phi.data(), rhs.data(), phi.size(), h2, omega, color);
            check_kernel("rb_sor_color_kernel");
        }
    }
}

template <typename Real>
void run_rb_sor_steps(
    DeviceGrid3D<Real>& phi,
    const DeviceGrid3D<Real>& rhs,
    Real h,
    Real omega,
    std::size_t steps
) {
    if (phi.size() != rhs.size()) {
        throw std::invalid_argument("phi and rhs device grid sizes do not match");
    }

    const detail::ScopedNvtxRange range{"cuda::run_rb_sor_steps_3d"};
    const std::size_t interior_n = phi.size() - 2;
    const dim3 block = cuda_kernels::make_block_3d();
    const std::size_t active_k = (interior_n + 1) / 2;
    const dim3 grid = cuda_kernels::make_grid_3d(active_k, interior_n, interior_n, block);
    const Real h2 = h * h;

    for (std::size_t step = 0; step < steps; ++step) {
        for (int color = 0; color < 2; ++color) {
            cuda_kernels::rb_sor_color_kernel_3d<Real>
                <<<grid, block>>>(phi.data(), rhs.data(), phi.size(), h2, omega, color);
            check_kernel("rb_sor_color_kernel_3d");
        }
    }
}

template <typename Real>
void run_rb_sor_steps(
    DeviceGridView3D<Real>& phi,
    const DeviceGridView3D<Real>& rhs,
    Real h,
    Real omega,
    std::size_t steps
) {
    if (phi.size() != rhs.size()) {
        throw std::invalid_argument("phi and rhs device grid sizes do not match");
    }

    const detail::ScopedNvtxRange range{"cuda::run_rb_sor_steps_3d"};
    const std::size_t interior_n = phi.size() - 2;
    const dim3 block = cuda_kernels::make_block_3d();
    const std::size_t active_k = (interior_n + 1) / 2;
    const dim3 grid = cuda_kernels::make_grid_3d(active_k, interior_n, interior_n, block);
    const Real h2 = h * h;

    for (std::size_t step = 0; step < steps; ++step) {
        for (int color = 0; color < 2; ++color) {
            cuda_kernels::rb_sor_color_kernel_3d<Real>
                <<<grid, block>>>(phi.data(), rhs.data(), phi.size(), h2, omega, color);
            check_kernel("rb_sor_color_kernel_3d");
        }
    }
}

template <typename Real, typename PhiGrid, typename RhsGrid>
void run_fused_rb_sor_steps(
    PhiGrid& phi,
    const RhsGrid& rhs,
    Real h,
    Real omega,
    std::size_t steps
) {
    if (phi.size() != rhs.size()) {
        throw std::invalid_argument("phi and rhs device grid sizes do not match");
    }
    if (phi.size() < 3 || phi.size() > cuda_kernels::kFusedCoarseSorMaxArrayN) {
        throw std::invalid_argument("fused coarse SOR only supports 3x3 through 6x6 grids");
    }

    const detail::ScopedNvtxRange range{"cuda::run_fused_rb_sor_steps"};
    const Real h2 = h * h;
    const dim3 block{
        static_cast<unsigned int>(phi.size()),
        static_cast<unsigned int>(phi.size()),
        1U,
    };

    cuda_kernels::rb_sor_fused_coarse_kernel<Real>
        <<<1, block>>>(phi.data(), rhs.data(), phi.size(), h2, omega, steps);
    check_kernel("rb_sor_fused_coarse_kernel");
}

template <typename Real>
void run_fused_rb_sor_steps(
    DeviceGrid3D<Real>& phi,
    const DeviceGrid3D<Real>& rhs,
    Real h,
    Real omega,
    std::size_t steps
) {
    if (phi.size() != rhs.size()) {
        throw std::invalid_argument("phi and rhs device grid sizes do not match");
    }
    if (phi.size() < 3 || phi.size() > cuda_kernels::kFusedCoarseSorMaxArrayN) {
        throw std::invalid_argument("fused coarse SOR only supports 3x3x3 through 6x6x6 grids");
    }

    const detail::ScopedNvtxRange range{"cuda::run_fused_rb_sor_steps_3d"};
    const Real h2 = h * h;
    const dim3 block{
        static_cast<unsigned int>(phi.size()),
        static_cast<unsigned int>(phi.size()),
        static_cast<unsigned int>(phi.size()),
    };

    cuda_kernels::rb_sor_fused_coarse_kernel_3d<Real>
        <<<1, block>>>(phi.data(), rhs.data(), phi.size(), h2, omega, steps);
    check_kernel("rb_sor_fused_coarse_kernel_3d");
}

template <typename Real>
void run_fused_rb_sor_steps(
    DeviceGridView3D<Real>& phi,
    const DeviceGridView3D<Real>& rhs,
    Real h,
    Real omega,
    std::size_t steps
) {
    if (phi.size() != rhs.size()) {
        throw std::invalid_argument("phi and rhs device grid sizes do not match");
    }
    if (phi.size() < 3 || phi.size() > cuda_kernels::kFusedCoarseSorMaxArrayN) {
        throw std::invalid_argument("fused coarse SOR only supports 3x3x3 through 6x6x6 grids");
    }

    const detail::ScopedNvtxRange range{"cuda::run_fused_rb_sor_steps_3d"};
    const Real h2 = h * h;
    const dim3 block{
        static_cast<unsigned int>(phi.size()),
        static_cast<unsigned int>(phi.size()),
        static_cast<unsigned int>(phi.size()),
    };

    cuda_kernels::rb_sor_fused_coarse_kernel_3d<Real>
        <<<1, block>>>(phi.data(), rhs.data(), phi.size(), h2, omega, steps);
    check_kernel("rb_sor_fused_coarse_kernel_3d");
}

template <typename Real>
void run_jacobi_step(
    const DeviceGrid2D<Real>& phi,
    const DeviceGrid2D<Real>& rhs,
    Real h,
    DeviceGrid2D<Real>& work
) {
    if (phi.size() != rhs.size() || phi.size() != work.size()) {
        throw std::invalid_argument("jacobi device grid sizes do not match");
    }

    const detail::ScopedNvtxRange range{"cuda::run_jacobi_step"};
    const dim3 block = cuda_kernels::make_block_2d();
    const dim3 grid = cuda_kernels::make_grid_2d(phi.size() - 2, phi.size() - 2, block);
    const Real h2 = h * h;

    cuda_kernels::jacobi_update_kernel<Real>
        <<<grid, block>>>(phi.data(), rhs.data(), work.data(), phi.size(), h2);
    check_kernel("jacobi_update_kernel");
}

template <typename Real>
void run_jacobi_step(
    const DeviceGrid3D<Real>& phi,
    const DeviceGrid3D<Real>& rhs,
    Real h,
    DeviceGrid3D<Real>& work
) {
    if (phi.size() != rhs.size() || phi.size() != work.size()) {
        throw std::invalid_argument("jacobi device grid sizes do not match");
    }

    const detail::ScopedNvtxRange range{"cuda::run_jacobi_step_3d"};
    const dim3 block = cuda_kernels::make_block_3d();
    const dim3 grid = cuda_kernels::make_grid_3d(
        phi.size() - 2,
        phi.size() - 2,
        phi.size() - 2,
        block
    );
    const Real h2 = h * h;

    cuda_kernels::jacobi_update_kernel_3d<Real>
        <<<grid, block>>>(phi.data(), rhs.data(), work.data(), phi.size(), h2);
    check_kernel("jacobi_update_kernel_3d");
}

template <typename Real>
[[nodiscard]] double compute_relative_residual(
    const DeviceGrid2D<Real>& phi,
    const DeviceGrid2D<Real>& rhs,
    Real h,
    RelativeResidualWorkspace& workspace
) {
    if (phi.size() != rhs.size()) {
        throw std::invalid_argument("phi and rhs device grid sizes do not match");
    }

    const detail::ScopedNvtxRange range{"cuda::compute_relative_residual"};
    const std::size_t interior_n = phi.size() - 2;
    const std::size_t total_points = interior_n * interior_n;
    if (total_points == 0) {
        return 0.0;
    }

    workspace.reserve_for(phi.size());
    ensure_rhs_norm_cached(rhs, h, workspace);

    const int threads = cuda_kernels::kReductionThreads;
    const int blocks = workspace.blocks();
    const std::size_t shared_bytes =
        static_cast<std::size_t>(threads) * sizeof(cuda_kernels::ResidualPair);

    cuda_kernels::residual_partial_kernel<Real>
        <<<blocks, threads, shared_bytes>>>(
            phi.data(), rhs.data(), phi.size(), h * h, workspace.partial_sums()
        );
    check_kernel("residual_partial_kernel");

    auto* totals = reduce_residual_pairs(
        static_cast<std::size_t>(blocks),
        workspace.partial_sums(),
        workspace.scratch_sums(),
        threads,
        shared_bytes
    );

    cuda_kernels::finalize_relative_residual_kernel<>
        <<<1, 1>>>(totals, workspace.rhs_norm2(), workspace.final_residual());
    check_kernel("finalize_relative_residual_kernel");

    std::array<double, 1> host_residual{};
    check(
        cudaMemcpy(
            host_residual.data(),
            workspace.final_residual(),
            sizeof(double),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpyDeviceToHost",
        __FILE__,
        __LINE__
    );
    return host_residual[0];
}

template <typename Real>
[[nodiscard]] double compute_relative_residual(
    const DeviceGrid3D<Real>& phi,
    const DeviceGrid3D<Real>& rhs,
    Real h,
    RelativeResidualWorkspace3D& workspace
) {
    if (phi.size() != rhs.size()) {
        throw std::invalid_argument("phi and rhs device grid sizes do not match");
    }

    const detail::ScopedNvtxRange range{"cuda::compute_relative_residual_3d"};
    const std::size_t interior_n = phi.size() - 2;
    const std::size_t total_points = interior_n * interior_n * interior_n;
    if (total_points == 0) {
        return 0.0;
    }

    workspace.reserve_for(phi.size());
    ensure_rhs_norm_cached(rhs, h, workspace);

    const int threads = cuda_kernels::kReductionThreads;
    const int blocks = workspace.blocks();
    const std::size_t shared_bytes =
        static_cast<std::size_t>(threads) * sizeof(cuda_kernels::ResidualPair);

    cuda_kernels::residual_partial_kernel_3d<Real>
        <<<blocks, threads, shared_bytes>>>(
            phi.data(), rhs.data(), phi.size(), h * h, workspace.partial_sums()
        );
    check_kernel("residual_partial_kernel_3d");

    auto* totals = reduce_residual_pairs(
        static_cast<std::size_t>(blocks),
        workspace.partial_sums(),
        workspace.scratch_sums(),
        threads,
        shared_bytes
    );

    cuda_kernels::finalize_relative_residual_kernel<>
        <<<1, 1>>>(totals, workspace.rhs_norm2(), workspace.final_residual());
    check_kernel("finalize_relative_residual_kernel");

    std::array<double, 1> host_residual{};
    check(
        cudaMemcpy(
            host_residual.data(),
            workspace.final_residual(),
            sizeof(double),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpyDeviceToHost",
        __FILE__,
        __LINE__
    );
    return host_residual[0];
}

template <typename Real>
[[nodiscard]] double compute_relative_residual_uncached(
    const DeviceGrid2D<Real>& phi,
    const DeviceGrid2D<Real>& rhs,
    Real h,
    RelativeResidualWorkspace& workspace
) {
    if (phi.size() != rhs.size()) {
        throw std::invalid_argument("phi and rhs device grid sizes do not match");
    }

    const detail::ScopedNvtxRange range{"cuda::compute_relative_residual_uncached"};
    const std::size_t interior_n = phi.size() - 2;
    const std::size_t total_points = interior_n * interior_n;
    if (total_points == 0) {
        return 0.0;
    }

    workspace.reserve_for(phi.size());

    const int threads = cuda_kernels::kReductionThreads;
    const int blocks = workspace.blocks();
    const std::size_t shared_bytes =
        static_cast<std::size_t>(threads) * sizeof(cuda_kernels::ResidualPair);

    cuda_kernels::relative_residual_partial_kernel<Real>
        <<<blocks, threads, shared_bytes>>>(
            phi.data(), rhs.data(), phi.size(), h * h, workspace.partial_sums()
        );
    check_kernel("relative_residual_partial_kernel");

    auto* totals = reduce_residual_pairs(
        static_cast<std::size_t>(blocks),
        workspace.partial_sums(),
        workspace.scratch_sums(),
        threads,
        shared_bytes
    );

    cuda_kernels::finalize_relative_residual_pair_kernel<>
        <<<1, 1>>>(totals, workspace.final_residual());
    check_kernel("finalize_relative_residual_pair_kernel");

    std::array<double, 1> host_residual{};
    check(
        cudaMemcpy(
            host_residual.data(),
            workspace.final_residual(),
            sizeof(double),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpyDeviceToHost",
        __FILE__,
        __LINE__
    );
    return host_residual[0];
}

template <typename Real, typename PhiGrid, typename RhsGrid>
[[nodiscard]] double compute_relative_residual_uncached_3d(
    const PhiGrid& phi,
    const RhsGrid& rhs,
    Real h,
    RelativeResidualWorkspace3D& workspace
) {
    if (phi.size() != rhs.size()) {
        throw std::invalid_argument("phi and rhs device grid sizes do not match");
    }

    const detail::ScopedNvtxRange range{"cuda::compute_relative_residual_uncached_3d"};
    const std::size_t interior_n = phi.size() - 2;
    const std::size_t total_points = interior_n * interior_n * interior_n;
    if (total_points == 0) {
        return 0.0;
    }

    workspace.reserve_for(phi.size());

    const int threads = cuda_kernels::kReductionThreads;
    const int blocks = workspace.blocks();
    const std::size_t shared_bytes =
        static_cast<std::size_t>(threads) * sizeof(cuda_kernels::ResidualPair);

    cuda_kernels::relative_residual_partial_kernel_3d<Real>
        <<<blocks, threads, shared_bytes>>>(
            phi.data(), rhs.data(), phi.size(), h * h, workspace.partial_sums()
        );
    check_kernel("relative_residual_partial_kernel_3d");

    auto* totals = reduce_residual_pairs(
        static_cast<std::size_t>(blocks),
        workspace.partial_sums(),
        workspace.scratch_sums(),
        threads,
        shared_bytes
    );

    cuda_kernels::finalize_relative_residual_pair_kernel<>
        <<<1, 1>>>(totals, workspace.final_residual());
    check_kernel("finalize_relative_residual_pair_kernel");

    std::array<double, 1> host_residual{};
    check(
        cudaMemcpy(
            host_residual.data(),
            workspace.final_residual(),
            sizeof(double),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpyDeviceToHost",
        __FILE__,
        __LINE__
    );
    return host_residual[0];
}

template <typename Real>
[[nodiscard]] double compute_relative_residual(
    const DeviceGrid2D<Real>& phi, const DeviceGrid2D<Real>& rhs, Real h
) {
    RelativeResidualWorkspace workspace{phi.size()};
    return compute_relative_residual(phi, rhs, h, workspace);
}

template <typename Real>
[[nodiscard]] double compute_relative_residual(
    const DeviceGrid3D<Real>& phi, const DeviceGrid3D<Real>& rhs, Real h
) {
    RelativeResidualWorkspace3D workspace{phi.size()};
    return compute_relative_residual(phi, rhs, h, workspace);
}

template <typename Real, typename PhiGrid, typename RhsGrid, typename ResidualGrid>
void compute_residual_full(
    const PhiGrid& phi,
    const RhsGrid& rhs,
    Real h,
    ResidualGrid& residual_out
) {
    if (phi.size() != rhs.size() || phi.size() != residual_out.size()) {
        throw std::invalid_argument("residual_full device grid sizes do not match");
    }

    const detail::ScopedNvtxRange range{"cuda::compute_residual_full"};
    residual_out.zero();
    const dim3 block = cuda_kernels::make_block_2d();
    const dim3 grid = cuda_kernels::make_grid_2d(phi.size() - 2, phi.size() - 2, block);

    cuda_kernels::residual_full_kernel<Real>
        <<<grid, block>>>(phi.data(), rhs.data(), residual_out.data(), phi.size(), Real{1} / (h * h));
    check_kernel("residual_full_kernel");
}

template <typename Real>
void compute_residual_full(
    const DeviceGrid3D<Real>& phi,
    const DeviceGrid3D<Real>& rhs,
    Real h,
    DeviceGridView3D<Real>& residual_out
) {
    if (phi.size() != rhs.size() || phi.size() != residual_out.size()) {
        throw std::invalid_argument("residual_full device grid sizes do not match");
    }

    const detail::ScopedNvtxRange range{"cuda::compute_residual_full_3d"};
    residual_out.zero();
    const dim3 block = cuda_kernels::make_block_3d();
    const dim3 grid = cuda_kernels::make_grid_3d(
        phi.size() - 2,
        phi.size() - 2,
        phi.size() - 2,
        block
    );

    cuda_kernels::residual_full_kernel_3d<Real>
        <<<grid, block>>>(phi.data(), rhs.data(), residual_out.data(), phi.size(), Real{1} / (h * h));
    check_kernel("residual_full_kernel_3d");
}

template <typename Real>
void compute_residual_full(
    const DeviceGridView3D<Real>& phi,
    const DeviceGridView3D<Real>& rhs,
    Real h,
    DeviceGridView3D<Real>& residual_out
) {
    if (phi.size() != rhs.size() || phi.size() != residual_out.size()) {
        throw std::invalid_argument("residual_full device grid sizes do not match");
    }

    const detail::ScopedNvtxRange range{"cuda::compute_residual_full_3d"};
    residual_out.zero();
    const dim3 block = cuda_kernels::make_block_3d();
    const dim3 grid = cuda_kernels::make_grid_3d(
        phi.size() - 2,
        phi.size() - 2,
        phi.size() - 2,
        block
    );

    cuda_kernels::residual_full_kernel_3d<Real>
        <<<grid, block>>>(phi.data(), rhs.data(), residual_out.data(), phi.size(), Real{1} / (h * h));
    check_kernel("residual_full_kernel_3d");
}

template <typename Real, typename FineGrid, typename CoarseGrid>
void restrict_full_weighting(const FineGrid& fine, CoarseGrid& coarse) {
    const detail::ScopedNvtxRange range{"cuda::restrict_full_weighting"};
    const std::size_t coarse_interior_n = coarse.size() - 2;
    coarse.zero();

    const dim3 block = cuda_kernels::make_block_2d();
    const dim3 grid = cuda_kernels::make_grid_2d(
        coarse_interior_n,
        coarse_interior_n,
        block
    );

    cuda_kernels::restrict_full_weighting_kernel<Real>
        <<<grid, block>>>(fine.data(), coarse.data(), coarse.size());
    check_kernel("restrict_full_weighting_kernel");
}

template <typename Real>
void restrict_full_weighting(const DeviceGridView3D<Real>& fine, DeviceGridView3D<Real>& coarse) {
    const detail::ScopedNvtxRange range{"cuda::restrict_full_weighting_3d"};
    const std::size_t coarse_interior_n = coarse.size() - 2;
    coarse.zero();

    const dim3 block = cuda_kernels::make_block_3d();
    const dim3 grid = cuda_kernels::make_grid_3d(
        coarse_interior_n,
        coarse_interior_n,
        coarse_interior_n,
        block
    );

    cuda_kernels::restrict_full_weighting_kernel_3d<Real>
        <<<grid, block>>>(fine.data(), coarse.data(), coarse.size());
    check_kernel("restrict_full_weighting_kernel_3d");
}

template <typename Real, typename CoarseGrid, typename FineGrid>
void prolong_add(const CoarseGrid& coarse, FineGrid& fine) {
    const detail::ScopedNvtxRange range{"cuda::prolong_add"};
    const std::size_t fine_interior_n = fine.size() - 2;
    const dim3 block = cuda_kernels::make_block_2d();
    const dim3 grid = cuda_kernels::make_grid_2d(
        fine_interior_n,
        fine_interior_n,
        block
    );

    cuda_kernels::prolong_add_kernel<Real>
        <<<grid, block>>>(coarse.data(), fine.data(), fine.size());
    check_kernel("prolong_add_kernel");
}

template <typename Real>
void prolong_add(const DeviceGridView3D<Real>& coarse, DeviceGrid3D<Real>& fine) {
    const detail::ScopedNvtxRange range{"cuda::prolong_add_3d"};
    const std::size_t fine_interior_n = fine.size() - 2;
    const dim3 block = cuda_kernels::make_block_3d();
    const dim3 grid = cuda_kernels::make_grid_3d(
        fine_interior_n,
        fine_interior_n,
        fine_interior_n,
        block
    );

    cuda_kernels::prolong_add_kernel_3d<Real>
        <<<grid, block>>>(coarse.data(), fine.data(), fine.size());
    check_kernel("prolong_add_kernel_3d");
}

template <typename Real>
void prolong_add(const DeviceGridView3D<Real>& coarse, DeviceGridView3D<Real>& fine) {
    const detail::ScopedNvtxRange range{"cuda::prolong_add_3d"};
    const std::size_t fine_interior_n = fine.size() - 2;
    const dim3 block = cuda_kernels::make_block_3d();
    const dim3 grid = cuda_kernels::make_grid_3d(
        fine_interior_n,
        fine_interior_n,
        fine_interior_n,
        block
    );

    cuda_kernels::prolong_add_kernel_3d<Real>
        <<<grid, block>>>(coarse.data(), fine.data(), fine.size());
    check_kernel("prolong_add_kernel_3d");
}

} // namespace poisson::cuda

#define POISSON_CUDA_CHECK(expr) ::poisson::cuda::check((expr), #expr, __FILE__, __LINE__)
