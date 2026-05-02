#pragma once

#include "common.cuh"

namespace poisson::cuda_kernels {

template <typename Real>
__global__ void rb_sor_color_kernel(
    Real* phi,
    const Real* rhs,
    std::size_t array_n,
    Real h2,
    Real omega,
    int color
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const std::size_t j = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    if (i >= array_n - 1 || j >= array_n - 1) {
        return;
    }
    if (static_cast<int>((i + j) & 1U) != color) {
        return;
    }

    const std::size_t idx = offset(array_n, i, j);
    const Real update = Real{0.25} * (
        phi[offset(array_n, i + 1, j)] +
        phi[offset(array_n, i - 1, j)] +
        phi[offset(array_n, i, j + 1)] +
        phi[offset(array_n, i, j - 1)] +
        h2 * rhs[idx]
    );
    phi[idx] = (Real{1} - omega) * phi[idx] + omega * update;
}

} // namespace poisson::cuda_kernels
