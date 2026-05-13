#pragma once

#include "common.cuh"

namespace poisson::cuda_kernels {

inline constexpr std::size_t kFusedSmallGridSorMaxArrayN2D = 32;
inline constexpr std::size_t kFusedSmallGridSorMaxElements2D =
    kFusedSmallGridSorMaxArrayN2D * kFusedSmallGridSorMaxArrayN2D;
inline constexpr std::size_t kFusedCoarseSorMaxArrayN3D = 6;
inline constexpr std::size_t kFusedCoarseSorMaxElements3D =
    kFusedCoarseSorMaxArrayN3D * kFusedCoarseSorMaxArrayN3D * kFusedCoarseSorMaxArrayN3D;

template <typename Real>
__global__ void rb_sor_color_kernel(
    Real* __restrict__ phi,
    const Real* __restrict__ rhs,
    std::size_t array_n,
    Real h2,
    Real omega,
    int color
) {
    // This smoother only targets practical grid sizes, so 32-bit indices help
    // reduce integer instruction count and register pressure in the hot path.
    const std::uint32_t stride = static_cast<std::uint32_t>(array_n);
    const std::uint32_t interior_n = stride - 2U;
    const std::uint32_t i = blockIdx.y * blockDim.y + threadIdx.y + 1U;
    const std::uint32_t active_j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > interior_n) {
        return;
    }

    // Map x to the k-th active cell on the selected checkerboard color.
    const std::uint32_t j0 = 1U + ((i + static_cast<std::uint32_t>(color)) & 1U);
    const std::uint32_t j = j0 + (active_j << 1U);
    if (j > interior_n) {
        return;
    }

    const std::uint32_t idx = offset_u32(stride, i, j);
    const Real update = Real{0.25} * (
        phi[offset_u32(stride, i + 1U, j)] +
        phi[offset_u32(stride, i - 1U, j)] +
        phi[offset_u32(stride, i, j + 1U)] +
        phi[offset_u32(stride, i, j - 1U)] +
        h2 * rhs[idx]
    );
    phi[idx] = (Real{1} - omega) * phi[idx] + omega * update;
}

template <typename Real>
__global__ void rb_sor_color_kernel_3d(
    Real* phi,
    const Real* rhs,
    std::size_t array_n,
    Real h2,
    Real omega,
    int color
) {
    const std::size_t interior_n = array_n - 2;
    const std::size_t i = static_cast<std::size_t>(blockIdx.z) * blockDim.z + threadIdx.z + 1;
    const std::size_t j = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const std::size_t active_k =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i > interior_n || j > interior_n) {
        return;
    }

    const std::size_t k0 = 1 + ((i + j + static_cast<std::size_t>(color)) & 1U);
    const std::size_t k = k0 + 2 * active_k;
    if (k > interior_n) {
        return;
    }

    const std::size_t idx = offset(array_n, i, j, k);
    const Real update = Real{1} / Real{6} * (
        phi[offset(array_n, i + 1, j, k)] +
        phi[offset(array_n, i - 1, j, k)] +
        phi[offset(array_n, i, j + 1, k)] +
        phi[offset(array_n, i, j - 1, k)] +
        phi[offset(array_n, i, j, k + 1)] +
        phi[offset(array_n, i, j, k - 1)] +
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
    // Small 2D MG grids fit in one block, so we can keep the whole stencil
    // state in shared memory and perform all sweeps in place.
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) {
        return;
    }
    if (array_n < 3 || array_n > kFusedSmallGridSorMaxArrayN2D) {
        return;
    }

    const std::size_t i = static_cast<std::size_t>(threadIdx.y);
    const std::size_t j = static_cast<std::size_t>(threadIdx.x);
    const std::size_t idx = offset(array_n, i, j);
    const std::size_t interior_n = array_n - 2;

    __shared__ Real shared_phi[kFusedSmallGridSorMaxElements2D];
    __shared__ Real shared_rhs[kFusedSmallGridSorMaxElements2D];

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

template <typename Real>
__global__ void rb_sor_fused_coarse_kernel_3d(
    Real* phi,
    const Real* rhs,
    std::size_t array_n,
    Real h2,
    Real omega,
    std::size_t steps
) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) {
        return;
    }
    if (array_n < 3 || array_n > kFusedCoarseSorMaxArrayN3D) {
        return;
    }

    const std::size_t i = static_cast<std::size_t>(threadIdx.z);
    const std::size_t j = static_cast<std::size_t>(threadIdx.y);
    const std::size_t k = static_cast<std::size_t>(threadIdx.x);
    const std::size_t idx = offset(array_n, i, j, k);
    const std::size_t interior_n = array_n - 2;

    __shared__ Real shared_phi[kFusedCoarseSorMaxElements3D];
    __shared__ Real shared_rhs[kFusedCoarseSorMaxElements3D];

    shared_phi[idx] = phi[idx];
    shared_rhs[idx] = rhs[idx];
    __syncthreads();

    for (std::size_t step = 0; step < steps; ++step) {
        for (int color = 0; color < 2; ++color) {
            if (
                i >= 1 && i <= interior_n &&
                j >= 1 && j <= interior_n &&
                k >= 1 && k <= interior_n &&
                (((i + j + k) & 1U) == static_cast<std::size_t>(color))
            ) {
                const Real update = Real{1} / Real{6} * (
                    shared_phi[offset(array_n, i + 1, j, k)] +
                    shared_phi[offset(array_n, i - 1, j, k)] +
                    shared_phi[offset(array_n, i, j + 1, k)] +
                    shared_phi[offset(array_n, i, j - 1, k)] +
                    shared_phi[offset(array_n, i, j, k + 1)] +
                    shared_phi[offset(array_n, i, j, k - 1)] +
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
