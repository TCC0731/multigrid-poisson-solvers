#pragma once

#include "common.cuh"

namespace poisson::cuda_kernels {

template <typename Real>
__global__ void relative_residual_partial_kernel(
    const Real* phi,
    const Real* rhs,
    std::size_t array_n,
    Real h2,
    double* partial_residuals,
    double* partial_rhs
) {
    extern __shared__ double shared[];
    double* shared_residuals = shared;
    double* shared_rhs = shared + blockDim.x;

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

    shared_residuals[threadIdx.x] = residual_sum;
    shared_rhs[threadIdx.x] = rhs_sum;
    __syncthreads();

    for (unsigned int offset_value = blockDim.x / 2; offset_value > 0; offset_value >>= 1U) {
        if (threadIdx.x < offset_value) {
            shared_residuals[threadIdx.x] += shared_residuals[threadIdx.x + offset_value];
            shared_rhs[threadIdx.x] += shared_rhs[threadIdx.x + offset_value];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_residuals[blockIdx.x] = shared_residuals[0];
        partial_rhs[blockIdx.x] = shared_rhs[0];
    }
}

} // namespace poisson::cuda_kernels
