#pragma once

#include "common.cuh"

namespace poisson::cuda_kernels {

template <typename Real>
__global__ void jacobi_update_kernel(
    const Real* phi,
    const Real* rhs,
    Real* work,
    std::size_t array_n,
    Real h2
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const std::size_t j = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    if (i >= array_n - 1 || j >= array_n - 1) {
        return;
    }

    work[offset(array_n, i, j)] = Real{0.25} * (
        phi[offset(array_n, i + 1, j)] +
        phi[offset(array_n, i - 1, j)] +
        phi[offset(array_n, i, j + 1)] +
        phi[offset(array_n, i, j - 1)] +
        h2 * rhs[offset(array_n, i, j)]
    );
}

} // namespace poisson::cuda_kernels
