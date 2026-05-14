#pragma once

#include "common.cuh"

namespace poisson::cuda_kernels {

inline constexpr std::size_t kExactCoarseSolveMaxArrayN = 6;
inline constexpr std::size_t kExactCoarseSolveMaxElements =
    kExactCoarseSolveMaxArrayN * kExactCoarseSolveMaxArrayN;
inline constexpr std::size_t kExactCoarseSolveMaxInteriorN = kExactCoarseSolveMaxArrayN - 2;
inline constexpr std::size_t kExactCoarseSolveMaxUnknowns =
    kExactCoarseSolveMaxInteriorN * kExactCoarseSolveMaxInteriorN;
inline constexpr std::size_t kExactCoarseSolveMaxElements3D =
    kExactCoarseSolveMaxArrayN * kExactCoarseSolveMaxArrayN * kExactCoarseSolveMaxArrayN;
inline constexpr std::size_t kExactCoarseSolveMaxUnknowns3D =
    kExactCoarseSolveMaxInteriorN * kExactCoarseSolveMaxInteriorN *
    kExactCoarseSolveMaxInteriorN;

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

template <typename Real>
__global__ void exact_coarse_solve_kernel_2d(
    Real* phi,
    const Real* rhs,
    std::size_t array_n,
    Real h2
) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) {
        return;
    }
    if (array_n < 3 || array_n > kExactCoarseSolveMaxArrayN) {
        return;
    }

    const std::size_t i = static_cast<std::size_t>(threadIdx.y);
    const std::size_t j = static_cast<std::size_t>(threadIdx.x);
    const std::size_t idx = offset(array_n, i, j);
    const std::size_t interior_n = array_n - 2;

    __shared__ Real shared_phi[kExactCoarseSolveMaxElements];
    __shared__ Real shared_rhs[kExactCoarseSolveMaxElements];
    __shared__ Real system_matrix[kExactCoarseSolveMaxUnknowns * kExactCoarseSolveMaxUnknowns];
    __shared__ Real rhs_vector[kExactCoarseSolveMaxUnknowns];
    __shared__ Real solution[kExactCoarseSolveMaxUnknowns];

    shared_phi[idx] = phi[idx];
    shared_rhs[idx] = rhs[idx];
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        const std::size_t unknown_count = interior_n * interior_n;

        for (std::size_t entry = 0; entry < unknown_count * unknown_count; ++entry) {
            system_matrix[entry] = Real{};
        }

        // Use the current device-side boundary values directly. The top-level
        // tiny-grid solve already carries physical boundaries in phi, while
        // recursive coarse error equations keep zero boundaries in phi.
        for (std::size_t row_i = 0; row_i < interior_n; ++row_i) {
            for (std::size_t row_j = 0; row_j < interior_n; ++row_j) {
                const std::size_t row = row_i * interior_n + row_j;
                const std::size_t grid_i = row_i + 1;
                const std::size_t grid_j = row_j + 1;

                Real rhs_entry = h2 * shared_rhs[offset(array_n, grid_i, grid_j)];
                system_matrix[row * unknown_count + row] = Real{4};

                if (row_i > 0) {
                    system_matrix[row * unknown_count + (row_i - 1) * interior_n + row_j] =
                        Real{-1};
                } else {
                    rhs_entry += shared_phi[offset(array_n, grid_i - 1, grid_j)];
                }

                if (row_i + 1 < interior_n) {
                    system_matrix[row * unknown_count + (row_i + 1) * interior_n + row_j] =
                        Real{-1};
                } else {
                    rhs_entry += shared_phi[offset(array_n, grid_i + 1, grid_j)];
                }

                if (row_j > 0) {
                    system_matrix[row * unknown_count + row_i * interior_n + (row_j - 1)] =
                        Real{-1};
                } else {
                    rhs_entry += shared_phi[offset(array_n, grid_i, grid_j - 1)];
                }

                if (row_j + 1 < interior_n) {
                    system_matrix[row * unknown_count + row_i * interior_n + (row_j + 1)] =
                        Real{-1};
                } else {
                    rhs_entry += shared_phi[offset(array_n, grid_i, grid_j + 1)];
                }

                rhs_vector[row] = rhs_entry;
            }
        }

        for (std::size_t pivot_col = 0; pivot_col < unknown_count; ++pivot_col) {
            const Real pivot = system_matrix[pivot_col * unknown_count + pivot_col];
            for (std::size_t row = pivot_col + 1; row < unknown_count; ++row) {
                const Real factor =
                    system_matrix[row * unknown_count + pivot_col] / pivot;
                for (std::size_t col = pivot_col; col < unknown_count; ++col) {
                    system_matrix[row * unknown_count + col] -=
                        factor * system_matrix[pivot_col * unknown_count + col];
                }
                rhs_vector[row] -= factor * rhs_vector[pivot_col];
            }
        }

        for (std::size_t row = unknown_count; row-- > 0;) {
            Real sum = rhs_vector[row];
            for (std::size_t col = row + 1; col < unknown_count; ++col) {
                sum -= system_matrix[row * unknown_count + col] * solution[col];
            }
            solution[row] = sum / system_matrix[row * unknown_count + row];
        }
    }
    __syncthreads();

    if (i >= 1 && i <= interior_n && j >= 1 && j <= interior_n) {
        phi[idx] = solution[(i - 1) * interior_n + (j - 1)];
    }
}

template <typename Real>
__global__ void exact_coarse_solve_kernel_3d(
    Real* phi,
    const Real* rhs,
    std::size_t array_n,
    Real h2
) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) {
        return;
    }
    if (array_n < 3 || array_n > kExactCoarseSolveMaxArrayN) {
        return;
    }

    const std::size_t i = static_cast<std::size_t>(threadIdx.z);
    const std::size_t j = static_cast<std::size_t>(threadIdx.y);
    const std::size_t k = static_cast<std::size_t>(threadIdx.x);
    const std::size_t idx = offset(array_n, i, j, k);
    const std::size_t interior_n = array_n - 2;

    __shared__ Real shared_phi[kExactCoarseSolveMaxElements3D];
    __shared__ Real shared_rhs[kExactCoarseSolveMaxElements3D];
    __shared__ Real
        system_matrix[kExactCoarseSolveMaxUnknowns3D * kExactCoarseSolveMaxUnknowns3D];
    __shared__ Real rhs_vector[kExactCoarseSolveMaxUnknowns3D];
    __shared__ Real solution[kExactCoarseSolveMaxUnknowns3D];

    shared_phi[idx] = phi[idx];
    shared_rhs[idx] = rhs[idx];
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        const std::size_t unknown_count = interior_n * interior_n * interior_n;

        for (std::size_t entry = 0; entry < unknown_count * unknown_count; ++entry) {
            system_matrix[entry] = Real{};
        }

        for (std::size_t row_i = 0; row_i < interior_n; ++row_i) {
            for (std::size_t row_j = 0; row_j < interior_n; ++row_j) {
                for (std::size_t row_k = 0; row_k < interior_n; ++row_k) {
                    const std::size_t row =
                        (row_i * interior_n + row_j) * interior_n + row_k;
                    const std::size_t grid_i = row_i + 1;
                    const std::size_t grid_j = row_j + 1;
                    const std::size_t grid_k = row_k + 1;

                    Real rhs_entry = h2 * shared_rhs[offset(array_n, grid_i, grid_j, grid_k)];
                    system_matrix[row * unknown_count + row] = Real{6};

                    if (row_i > 0) {
                        system_matrix
                            [row * unknown_count + ((row_i - 1) * interior_n + row_j) * interior_n + row_k] =
                            Real{-1};
                    } else {
                        rhs_entry += shared_phi[offset(array_n, grid_i - 1, grid_j, grid_k)];
                    }

                    if (row_i + 1 < interior_n) {
                        system_matrix
                            [row * unknown_count + ((row_i + 1) * interior_n + row_j) * interior_n + row_k] =
                            Real{-1};
                    } else {
                        rhs_entry += shared_phi[offset(array_n, grid_i + 1, grid_j, grid_k)];
                    }

                    if (row_j > 0) {
                        system_matrix
                            [row * unknown_count + (row_i * interior_n + (row_j - 1)) * interior_n + row_k] =
                            Real{-1};
                    } else {
                        rhs_entry += shared_phi[offset(array_n, grid_i, grid_j - 1, grid_k)];
                    }

                    if (row_j + 1 < interior_n) {
                        system_matrix
                            [row * unknown_count + (row_i * interior_n + (row_j + 1)) * interior_n + row_k] =
                            Real{-1};
                    } else {
                        rhs_entry += shared_phi[offset(array_n, grid_i, grid_j + 1, grid_k)];
                    }

                    if (row_k > 0) {
                        system_matrix
                            [row * unknown_count + (row_i * interior_n + row_j) * interior_n + (row_k - 1)] =
                            Real{-1};
                    } else {
                        rhs_entry += shared_phi[offset(array_n, grid_i, grid_j, grid_k - 1)];
                    }

                    if (row_k + 1 < interior_n) {
                        system_matrix
                            [row * unknown_count + (row_i * interior_n + row_j) * interior_n + (row_k + 1)] =
                            Real{-1};
                    } else {
                        rhs_entry += shared_phi[offset(array_n, grid_i, grid_j, grid_k + 1)];
                    }

                    rhs_vector[row] = rhs_entry;
                }
            }
        }

        for (std::size_t pivot_col = 0; pivot_col < unknown_count; ++pivot_col) {
            const Real pivot = system_matrix[pivot_col * unknown_count + pivot_col];
            for (std::size_t row = pivot_col + 1; row < unknown_count; ++row) {
                const Real factor = system_matrix[row * unknown_count + pivot_col] / pivot;
                for (std::size_t col = pivot_col; col < unknown_count; ++col) {
                    system_matrix[row * unknown_count + col] -=
                        factor * system_matrix[pivot_col * unknown_count + col];
                }
                rhs_vector[row] -= factor * rhs_vector[pivot_col];
            }
        }

        for (std::size_t row = unknown_count; row-- > 0;) {
            Real sum = rhs_vector[row];
            for (std::size_t col = row + 1; col < unknown_count; ++col) {
                sum -= system_matrix[row * unknown_count + col] * solution[col];
            }
            solution[row] = sum / system_matrix[row * unknown_count + row];
        }
    }
    __syncthreads();

    if (
        i >= 1 && i <= interior_n &&
        j >= 1 && j <= interior_n &&
        k >= 1 && k <= interior_n
    ) {
        phi[idx] = solution[((i - 1) * interior_n + (j - 1)) * interior_n + (k - 1)];
    }
}

} // namespace poisson::cuda_kernels
