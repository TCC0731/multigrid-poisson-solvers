#pragma once

#include "common.cuh"

namespace poisson::cuda_kernels {

inline constexpr std::size_t kFusedCoarseSorMaxArrayN = 6;
inline constexpr std::size_t kFusedCoarseSorMaxElements =
    kFusedCoarseSorMaxArrayN * kFusedCoarseSorMaxArrayN;

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

template <typename Real>
__global__ void rb_sor_fused_coarse_kernel(
    Real* phi,
    const Real* rhs,
    std::size_t array_n,
    Real h2,
    Real omega,
    std::size_t steps
) {
    // The coarsest MG grid is tiny (at most 6x6), so one block can keep the
    // whole stencil state in shared memory and perform all sweeps in place.
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) {
        return;
    }
    if (array_n < 3 || array_n > kFusedCoarseSorMaxArrayN) {
        return;
    }

    const std::size_t i = static_cast<std::size_t>(threadIdx.y);
    const std::size_t j = static_cast<std::size_t>(threadIdx.x);
    const std::size_t idx = offset(array_n, i, j);
    const std::size_t interior_n = array_n - 2;

    __shared__ Real shared_phi[kFusedCoarseSorMaxElements];
    __shared__ Real shared_rhs[kFusedCoarseSorMaxElements];

    shared_phi[idx] = phi[idx];
    shared_rhs[idx] = rhs[idx];
    __syncthreads();

    for (std::size_t step = 0; step < steps; ++step) {
        for (int color = 0; color < 2; ++color) {
            if (
                i >= 1 && i <= interior_n &&
                j >= 1 && j <= interior_n &&
                (((i + j) & 1U) == static_cast<std::size_t>(color))
            ) {
                const Real update = Real{0.25} * (
                    shared_phi[offset(array_n, i + 1, j)] +
                    shared_phi[offset(array_n, i - 1, j)] +
                    shared_phi[offset(array_n, i, j + 1)] +
                    shared_phi[offset(array_n, i, j - 1)] +
                    h2 * shared_rhs[idx]
                );
                shared_phi[idx] = (Real{1} - omega) * shared_phi[idx] + omega * update;
            }
            __syncthreads();
        }
    }

    phi[idx] = shared_phi[idx];
}

} // namespace poisson::cuda_kernels
