#pragma once

#include <cmath>

#include "common.cuh"

namespace poisson::cuda_kernels {

struct ResidualPair {
    double residual{0.0};
    double rhs{0.0};
};

template <typename Real>
__global__ void rhs_norm_partial_kernel(
    const Real* rhs,
    std::size_t array_n,
    Real h2,
    ResidualPair* partial_sums
) {
    extern __shared__ unsigned char shared_bytes[];
    auto* shared = reinterpret_cast<ResidualPair*>(shared_bytes);

    const std::size_t interior_n = array_n - 2;
    const std::size_t total_points = interior_n * interior_n;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    double rhs_sum = 0.0;

    for (std::size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total_points;
         linear += stride) {
        const std::size_t i = linear / interior_n + 1;
        const std::size_t j = linear % interior_n + 1;
        const Real b = h2 * rhs[offset(array_n, i, j)];
        rhs_sum += static_cast<double>(b) * static_cast<double>(b);
    }

    shared[threadIdx.x] = ResidualPair{0.0, rhs_sum};
    __syncthreads();

    for (unsigned int offset_value = blockDim.x / 2; offset_value > 0; offset_value >>= 1U) {
        if (threadIdx.x < offset_value) {
            shared[threadIdx.x].residual += shared[threadIdx.x + offset_value].residual;
            shared[threadIdx.x].rhs += shared[threadIdx.x + offset_value].rhs;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

template <typename Real>
__global__ void rhs_norm_partial_kernel_3d(
    const Real* rhs,
    std::size_t array_n,
    Real h2,
    ResidualPair* partial_sums
) {
    extern __shared__ unsigned char shared_bytes[];
    auto* shared = reinterpret_cast<ResidualPair*>(shared_bytes);

    const std::size_t interior_n = array_n - 2;
    const std::size_t total_points = interior_n * interior_n * interior_n;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    double rhs_sum = 0.0;

    for (std::size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total_points;
         linear += stride) {
        const std::size_t plane = interior_n * interior_n;
        const std::size_t i = linear / plane + 1;
        const std::size_t rem = linear % plane;
        const std::size_t j = rem / interior_n + 1;
        const std::size_t k = rem % interior_n + 1;
        const Real b = h2 * rhs[offset(array_n, i, j, k)];
        rhs_sum += static_cast<double>(b) * static_cast<double>(b);
    }

    shared[threadIdx.x] = ResidualPair{0.0, rhs_sum};
    __syncthreads();

    for (unsigned int offset_value = blockDim.x / 2; offset_value > 0; offset_value >>= 1U) {
        if (threadIdx.x < offset_value) {
            shared[threadIdx.x].residual += shared[threadIdx.x + offset_value].residual;
            shared[threadIdx.x].rhs += shared[threadIdx.x + offset_value].rhs;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

template <typename Real>
__global__ void relative_residual_partial_kernel(
    const Real* phi,
    const Real* rhs,
    std::size_t array_n,
    Real h2,
    ResidualPair* partial_sums
) {
    extern __shared__ unsigned char shared_bytes[];
    auto* shared = reinterpret_cast<ResidualPair*>(shared_bytes);

    const std::size_t interior_n = array_n - 2;
    const std::size_t total_points = interior_n * interior_n;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    double residual_sum = 0.0;
    double rhs_sum = 0.0;

    for (std::size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total_points;
         linear += stride) {
        const std::size_t i = linear / interior_n + 1;
        const std::size_t j = linear % interior_n + 1;
        const std::size_t idx = offset(array_n, i, j);
        const Real lap = (
            Real{4} * phi[idx] -
            phi[offset(array_n, i + 1, j)] -
            phi[offset(array_n, i - 1, j)] -
            phi[offset(array_n, i, j + 1)] -
            phi[offset(array_n, i, j - 1)]
        );
        const Real b = h2 * rhs[idx];
        const Real diff = b - lap;

        residual_sum += static_cast<double>(diff) * static_cast<double>(diff);
        rhs_sum += static_cast<double>(b) * static_cast<double>(b);
    }

    shared[threadIdx.x] = ResidualPair{residual_sum, rhs_sum};
    __syncthreads();

    for (unsigned int offset_value = blockDim.x / 2; offset_value > 0; offset_value >>= 1U) {
        if (threadIdx.x < offset_value) {
            shared[threadIdx.x].residual += shared[threadIdx.x + offset_value].residual;
            shared[threadIdx.x].rhs += shared[threadIdx.x + offset_value].rhs;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

template <typename Real>
__global__ void relative_residual_partial_kernel_3d(
    const Real* phi,
    const Real* rhs,
    std::size_t array_n,
    Real h2,
    ResidualPair* partial_sums
) {
    extern __shared__ unsigned char shared_bytes[];
    auto* shared = reinterpret_cast<ResidualPair*>(shared_bytes);

    const std::size_t interior_n = array_n - 2;
    const std::size_t total_points = interior_n * interior_n * interior_n;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    double residual_sum = 0.0;
    double rhs_sum = 0.0;

    for (std::size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total_points;
         linear += stride) {
        const std::size_t plane = interior_n * interior_n;
        const std::size_t i = linear / plane + 1;
        const std::size_t rem = linear % plane;
        const std::size_t j = rem / interior_n + 1;
        const std::size_t k = rem % interior_n + 1;
        const std::size_t idx = offset(array_n, i, j, k);
        const Real lap = (
            Real{6} * phi[idx] -
            phi[offset(array_n, i + 1, j, k)] -
            phi[offset(array_n, i - 1, j, k)] -
            phi[offset(array_n, i, j + 1, k)] -
            phi[offset(array_n, i, j - 1, k)] -
            phi[offset(array_n, i, j, k + 1)] -
            phi[offset(array_n, i, j, k - 1)]
        );
        const Real b = h2 * rhs[idx];
        const Real diff = b - lap;

        residual_sum += static_cast<double>(diff) * static_cast<double>(diff);
        rhs_sum += static_cast<double>(b) * static_cast<double>(b);
    }

    shared[threadIdx.x] = ResidualPair{residual_sum, rhs_sum};
    __syncthreads();

    for (unsigned int offset_value = blockDim.x / 2; offset_value > 0; offset_value >>= 1U) {
        if (threadIdx.x < offset_value) {
            shared[threadIdx.x].residual += shared[threadIdx.x + offset_value].residual;
            shared[threadIdx.x].rhs += shared[threadIdx.x + offset_value].rhs;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

template <typename Real>
__global__ void residual_partial_kernel(
    const Real* phi,
    const Real* rhs,
    std::size_t array_n,
    Real h2,
    ResidualPair* partial_sums
) {
    extern __shared__ unsigned char shared_bytes[];
    auto* shared = reinterpret_cast<ResidualPair*>(shared_bytes);

    const std::size_t interior_n = array_n - 2;
    const std::size_t total_points = interior_n * interior_n;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    double residual_sum = 0.0;

    for (std::size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total_points;
         linear += stride) {
        const std::size_t i = linear / interior_n + 1;
        const std::size_t j = linear % interior_n + 1;
        const std::size_t idx = offset(array_n, i, j);
        const Real lap = (
            Real{4} * phi[idx] -
            phi[offset(array_n, i + 1, j)] -
            phi[offset(array_n, i - 1, j)] -
            phi[offset(array_n, i, j + 1)] -
            phi[offset(array_n, i, j - 1)]
        );
        const Real b = h2 * rhs[idx];
        const Real diff = b - lap;

        residual_sum += static_cast<double>(diff) * static_cast<double>(diff);
    }

    shared[threadIdx.x] = ResidualPair{residual_sum, 0.0};
    __syncthreads();

    for (unsigned int offset_value = blockDim.x / 2; offset_value > 0; offset_value >>= 1U) {
        if (threadIdx.x < offset_value) {
            shared[threadIdx.x].residual += shared[threadIdx.x + offset_value].residual;
            shared[threadIdx.x].rhs += shared[threadIdx.x + offset_value].rhs;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

template <typename Real>
__global__ void residual_partial_kernel_3d(
    const Real* phi,
    const Real* rhs,
    std::size_t array_n,
    Real h2,
    ResidualPair* partial_sums
) {
    extern __shared__ unsigned char shared_bytes[];
    auto* shared = reinterpret_cast<ResidualPair*>(shared_bytes);

    const std::size_t interior_n = array_n - 2;
    const std::size_t total_points = interior_n * interior_n * interior_n;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    double residual_sum = 0.0;

    for (std::size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total_points;
         linear += stride) {
        const std::size_t plane = interior_n * interior_n;
        const std::size_t i = linear / plane + 1;
        const std::size_t rem = linear % plane;
        const std::size_t j = rem / interior_n + 1;
        const std::size_t k = rem % interior_n + 1;
        const std::size_t idx = offset(array_n, i, j, k);
        const Real lap = (
            Real{6} * phi[idx] -
            phi[offset(array_n, i + 1, j, k)] -
            phi[offset(array_n, i - 1, j, k)] -
            phi[offset(array_n, i, j + 1, k)] -
            phi[offset(array_n, i, j - 1, k)] -
            phi[offset(array_n, i, j, k + 1)] -
            phi[offset(array_n, i, j, k - 1)]
        );
        const Real b = h2 * rhs[idx];
        const Real diff = b - lap;

        residual_sum += static_cast<double>(diff) * static_cast<double>(diff);
    }

    shared[threadIdx.x] = ResidualPair{residual_sum, 0.0};
    __syncthreads();

    for (unsigned int offset_value = blockDim.x / 2; offset_value > 0; offset_value >>= 1U) {
        if (threadIdx.x < offset_value) {
            shared[threadIdx.x].residual += shared[threadIdx.x + offset_value].residual;
            shared[threadIdx.x].rhs += shared[threadIdx.x + offset_value].rhs;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

template <int Dummy = 0>
__global__ void reduce_residual_pairs_kernel(
    const ResidualPair* input, ResidualPair* output, std::size_t count
) {
    extern __shared__ unsigned char shared_bytes[];
    auto* shared = reinterpret_cast<ResidualPair*>(shared_bytes);

    const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x * 2 + threadIdx.x;
    ResidualPair sum{};

    if (first < count) {
        sum = input[first];
    }

    const std::size_t second = first + blockDim.x;
    if (second < count) {
        sum.residual += input[second].residual;
        sum.rhs += input[second].rhs;
    }

    shared[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int offset_value = blockDim.x / 2; offset_value > 0; offset_value >>= 1U) {
        if (threadIdx.x < offset_value) {
            shared[threadIdx.x].residual += shared[threadIdx.x + offset_value].residual;
            shared[threadIdx.x].rhs += shared[threadIdx.x + offset_value].rhs;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[blockIdx.x] = shared[0];
    }
}

template <int Dummy = 0>
__global__ void store_rhs_norm_kernel(const ResidualPair* totals, double* rhs_norm2) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        rhs_norm2[0] = totals[0].rhs;
    }
}

template <int Dummy = 0>
__global__ void finalize_relative_residual_pair_kernel(
    const ResidualPair* totals, double* relative_residual
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const double residual_total = totals[0].residual;
        const double rhs_total = totals[0].rhs;
        relative_residual[0] = (rhs_total == 0.0)
            ? std::sqrt(residual_total)
            : std::sqrt(residual_total / rhs_total);
    }
}

template <int Dummy = 0>
__global__ void finalize_relative_residual_kernel(
    const ResidualPair* totals, const double* rhs_norm2, double* relative_residual
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const double residual_total = totals[0].residual;
        const double rhs_total = rhs_norm2[0];
        relative_residual[0] = (rhs_total == 0.0)
            ? std::sqrt(residual_total)
            : std::sqrt(residual_total / rhs_total);
    }
}

} // namespace poisson::cuda_kernels
