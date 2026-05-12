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
__global__ void residual_full_kernel_3d(
    const Real* phi,
    const Real* rhs,
    Real* residual_out,
    std::size_t array_n,
    Real inv_h2
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.z) * blockDim.z + threadIdx.z + 1;
    const std::size_t j = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    if (i >= array_n - 1 || j >= array_n - 1 || k >= array_n - 1) {
        return;
    }

    const std::size_t idx = offset(array_n, i, j, k);
    residual_out[idx] = rhs[idx] - (
        Real{6} * phi[idx] -
        phi[offset(array_n, i + 1, j, k)] -
        phi[offset(array_n, i - 1, j, k)] -
        phi[offset(array_n, i, j + 1, k)] -
        phi[offset(array_n, i, j - 1, k)] -
        phi[offset(array_n, i, j, k + 1)] -
        phi[offset(array_n, i, j, k - 1)]
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
__device__ inline Real restriction_weight_3d(int offset_value) {
    return offset_value == 0 ? Real{2} : Real{1};
}

template <typename Real>
__global__ void restrict_full_weighting_kernel_3d(
    const Real* fine,
    Real* coarse,
    std::size_t coarse_array_n
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.z) * blockDim.z + threadIdx.z + 1;
    const std::size_t j = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    if (i >= coarse_array_n - 1 || j >= coarse_array_n - 1 || k >= coarse_array_n - 1) {
        return;
    }

    const std::size_t fine_array_n = 2 * coarse_array_n - 1;
    const std::size_t fi = 2 * i;
    const std::size_t fj = 2 * j;
    const std::size_t fk = 2 * k;

    Real total{};
    for (int di = -1; di <= 1; ++di) {
        const Real wi = restriction_weight_3d<Real>(di);
        for (int dj = -1; dj <= 1; ++dj) {
            const Real wij = wi * restriction_weight_3d<Real>(dj);
            for (int dk = -1; dk <= 1; ++dk) {
                total += wij
                    * restriction_weight_3d<Real>(dk)
                    * fine[offset(
                        fine_array_n,
                        static_cast<std::size_t>(static_cast<int>(fi) + di),
                        static_cast<std::size_t>(static_cast<int>(fj) + dj),
                        static_cast<std::size_t>(static_cast<int>(fk) + dk)
                    )];
            }
        }
    }

    coarse[offset(coarse_array_n, i, j, k)] = total / Real{64};
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

template <typename Real>
__global__ void prolong_add_kernel_3d(const Real* coarse, Real* fine, std::size_t fine_array_n) {
    const std::size_t fi = static_cast<std::size_t>(blockIdx.z) * blockDim.z + threadIdx.z + 1;
    const std::size_t fj = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const std::size_t fk = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    if (fi >= fine_array_n - 1 || fj >= fine_array_n - 1 || fk >= fine_array_n - 1) {
        return;
    }

    const std::size_t coarse_array_n = (fine_array_n + 1) / 2;
    const bool i_even = (fi & 1U) == 0U;
    const bool j_even = (fj & 1U) == 0U;
    const bool k_even = (fk & 1U) == 0U;

    const std::size_t i0 = fi / 2;
    const std::size_t i1 = i_even ? i0 : i0 + 1;
    const std::size_t j0 = fj / 2;
    const std::size_t j1 = j_even ? j0 : j0 + 1;
    const std::size_t k0 = fk / 2;
    const std::size_t k1 = k_even ? k0 : k0 + 1;

    const Real wi0 = i_even ? Real{1} : Real{0.5};
    const Real wi1 = i_even ? Real{} : Real{0.5};
    const Real wj0 = j_even ? Real{1} : Real{0.5};
    const Real wj1 = j_even ? Real{} : Real{0.5};
    const Real wk0 = k_even ? Real{1} : Real{0.5};
    const Real wk1 = k_even ? Real{} : Real{0.5};

    fine[offset(fine_array_n, fi, fj, fk)] +=
        wi0 * wj0 * wk0 * coarse[offset(coarse_array_n, i0, j0, k0)] +
        wi1 * wj0 * wk0 * coarse[offset(coarse_array_n, i1, j0, k0)] +
        wi0 * wj1 * wk0 * coarse[offset(coarse_array_n, i0, j1, k0)] +
        wi1 * wj1 * wk0 * coarse[offset(coarse_array_n, i1, j1, k0)] +
        wi0 * wj0 * wk1 * coarse[offset(coarse_array_n, i0, j0, k1)] +
        wi1 * wj0 * wk1 * coarse[offset(coarse_array_n, i1, j0, k1)] +
        wi0 * wj1 * wk1 * coarse[offset(coarse_array_n, i0, j1, k1)] +
        wi1 * wj1 * wk1 * coarse[offset(coarse_array_n, i1, j1, k1)];
}

} // namespace poisson::cuda_kernels
