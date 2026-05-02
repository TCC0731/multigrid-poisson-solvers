#pragma once

#include "common.cuh"

namespace poisson::cuda_kernels {

template <typename Real>
__global__ void residual_full_kernel(
    const Real* phi,
    const Real* rhs,
    Real* residual_out,
    std::size_t array_n,
    Real inv_h2
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const std::size_t j = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    if (i >= array_n - 1 || j >= array_n - 1) {
        return;
    }

    const std::size_t idx = offset(array_n, i, j);
    residual_out[idx] = rhs[idx] - (
        Real{4} * phi[idx] -
        phi[offset(array_n, i + 1, j)] -
        phi[offset(array_n, i - 1, j)] -
        phi[offset(array_n, i, j + 1)] -
        phi[offset(array_n, i, j - 1)]
    ) * inv_h2;
}

template <typename Real>
__global__ void restrict_full_weighting_kernel(
    const Real* fine,
    Real* coarse,
    std::size_t coarse_array_n
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const std::size_t j = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    if (i >= coarse_array_n - 1 || j >= coarse_array_n - 1) {
        return;
    }

    const std::size_t fine_array_n = 2 * coarse_array_n - 1;
    const std::size_t fi = 2 * i;
    const std::size_t fj = 2 * j;

    coarse[offset(coarse_array_n, i, j)] = (
        Real{4} * fine[offset(fine_array_n, fi, fj)] +
        Real{2} * (
            fine[offset(fine_array_n, fi - 1, fj)] +
            fine[offset(fine_array_n, fi + 1, fj)] +
            fine[offset(fine_array_n, fi, fj - 1)] +
            fine[offset(fine_array_n, fi, fj + 1)]
        ) +
        fine[offset(fine_array_n, fi - 1, fj - 1)] +
        fine[offset(fine_array_n, fi - 1, fj + 1)] +
        fine[offset(fine_array_n, fi + 1, fj - 1)] +
        fine[offset(fine_array_n, fi + 1, fj + 1)]
    ) / Real{16};
}

template <typename Real>
__global__ void prolong_add_kernel(const Real* coarse, Real* fine, std::size_t fine_array_n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const std::size_t j = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    if (i >= fine_array_n - 1 || j >= fine_array_n - 1) {
        return;
    }

    const std::size_t coarse_array_n = (fine_array_n + 1) / 2;
    Real correction = Real{};

    if (((i & 1U) == 0U) && ((j & 1U) == 0U)) {
        correction = coarse[offset(coarse_array_n, i / 2, j / 2)];
    } else if (((i & 1U) == 1U) && ((j & 1U) == 0U)) {
        correction = Real{0.5} * (
            coarse[offset(coarse_array_n, (i - 1) / 2, j / 2)] +
            coarse[offset(coarse_array_n, (i + 1) / 2, j / 2)]
        );
    } else if (((i & 1U) == 0U) && ((j & 1U) == 1U)) {
        correction = Real{0.5} * (
            coarse[offset(coarse_array_n, i / 2, (j - 1) / 2)] +
            coarse[offset(coarse_array_n, i / 2, (j + 1) / 2)]
        );
    } else {
        correction = Real{0.25} * (
            coarse[offset(coarse_array_n, (i - 1) / 2, (j - 1) / 2)] +
            coarse[offset(coarse_array_n, (i + 1) / 2, (j - 1) / 2)] +
            coarse[offset(coarse_array_n, (i - 1) / 2, (j + 1) / 2)] +
            coarse[offset(coarse_array_n, (i + 1) / 2, (j + 1) / 2)]
        );
    }

    fine[offset(fine_array_n, i, j)] += correction;
}

} // namespace poisson::cuda_kernels
