#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "poisson/grid2d.hpp"

#include "../../kernels/common.cuh"
#include "../../kernels/jacobi_kernels.cuh"
#include "../../kernels/mg_kernels.cuh"
#include "../../kernels/reduction_kernels.cuh"
#include "../../kernels/sor_kernels.cuh"

namespace poisson::cuda {

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
void run_rb_sor_steps(
    DeviceGrid2D<Real>& phi,
    const DeviceGrid2D<Real>& rhs,
    Real h,
    Real omega,
    std::size_t steps
) {
    if (phi.size() != rhs.size()) {
        throw std::invalid_argument("phi and rhs device grid sizes do not match");
    }

    const dim3 block = cuda_kernels::make_block_2d();
    const dim3 grid = cuda_kernels::make_grid_2d(phi.size() - 2, phi.size() - 2, block);
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
void run_jacobi_step(
    const DeviceGrid2D<Real>& phi,
    const DeviceGrid2D<Real>& rhs,
    Real h,
    DeviceGrid2D<Real>& work
) {
    if (phi.size() != rhs.size() || phi.size() != work.size()) {
        throw std::invalid_argument("jacobi device grid sizes do not match");
    }

    const dim3 block = cuda_kernels::make_block_2d();
    const dim3 grid = cuda_kernels::make_grid_2d(phi.size() - 2, phi.size() - 2, block);
    const Real h2 = h * h;

    cuda_kernels::jacobi_update_kernel<Real>
        <<<grid, block>>>(phi.data(), rhs.data(), work.data(), phi.size(), h2);
    check_kernel("jacobi_update_kernel");
}

template <typename Real>
[[nodiscard]] double compute_relative_residual(
    const DeviceGrid2D<Real>& phi, const DeviceGrid2D<Real>& rhs, Real h
) {
    if (phi.size() != rhs.size()) {
        throw std::invalid_argument("phi and rhs device grid sizes do not match");
    }

    const std::size_t interior_n = phi.size() - 2;
    const std::size_t total_points = interior_n * interior_n;
    if (total_points == 0) {
        return 0.0;
    }

    const int threads = cuda_kernels::kReductionThreads;
    const int blocks = static_cast<int>(
        std::min<std::size_t>(
            static_cast<std::size_t>(65'535),
            (total_points + static_cast<std::size_t>(threads) - 1) /
                static_cast<std::size_t>(threads)
        )
    );

    DeviceBuffer<double> partial_residuals{static_cast<std::size_t>(blocks)};
    DeviceBuffer<double> partial_rhs{static_cast<std::size_t>(blocks)};

    cuda_kernels::relative_residual_partial_kernel<Real>
        <<<blocks, threads, static_cast<std::size_t>(threads) * 2 * sizeof(double)>>>(
            phi.data(),
            rhs.data(),
            phi.size(),
            h * h,
            partial_residuals.data(),
            partial_rhs.data()
        );
    check_kernel("relative_residual_partial_kernel");

    std::vector<double> host_residuals(static_cast<std::size_t>(blocks));
    std::vector<double> host_rhs(static_cast<std::size_t>(blocks));
    partial_residuals.download(host_residuals.data(), host_residuals.size());
    partial_rhs.download(host_rhs.data(), host_rhs.size());

    double residual_total = 0.0;
    double rhs_total = 0.0;
    for (int block = 0; block < blocks; ++block) {
        residual_total += host_residuals[static_cast<std::size_t>(block)];
        rhs_total += host_rhs[static_cast<std::size_t>(block)];
    }

    if (rhs_total == 0.0) {
        return std::sqrt(residual_total);
    }
    return std::sqrt(residual_total / rhs_total);
}

template <typename Real>
void compute_residual_full(
    const DeviceGrid2D<Real>& phi,
    const DeviceGrid2D<Real>& rhs,
    Real h,
    DeviceGrid2D<Real>& residual_out
) {
    if (phi.size() != rhs.size() || phi.size() != residual_out.size()) {
        throw std::invalid_argument("residual_full device grid sizes do not match");
    }

    residual_out.zero();
    const dim3 block = cuda_kernels::make_block_2d();
    const dim3 grid = cuda_kernels::make_grid_2d(phi.size() - 2, phi.size() - 2, block);

    cuda_kernels::residual_full_kernel<Real>
        <<<grid, block>>>(phi.data(), rhs.data(), residual_out.data(), phi.size(), Real{1} / (h * h));
    check_kernel("residual_full_kernel");
}

template <typename Real>
void restrict_full_weighting(const DeviceGrid2D<Real>& fine, DeviceGrid2D<Real>& coarse) {
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
void prolong_add(const DeviceGrid2D<Real>& coarse, DeviceGrid2D<Real>& fine) {
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

} // namespace poisson::cuda

#define POISSON_CUDA_CHECK(expr) ::poisson::cuda::check((expr), #expr, __FILE__, __LINE__)
