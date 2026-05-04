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
    const std::size_t interior_n = array_n - 2;
    const std::size_t i = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const std::size_t active_j =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i > interior_n) {
        return;
    }

    // Map x to the k-th active cell on the selected checkerboard color.
    const std::size_t j0 = 1 + ((i + static_cast<std::size_t>(color)) & 1U);
    const std::size_t j = j0 + 2 * active_j;
    if (j > interior_n) {
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
