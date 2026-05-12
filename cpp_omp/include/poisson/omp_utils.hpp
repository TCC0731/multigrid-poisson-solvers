#pragma once

#include <cstddef>

namespace poisson::omp_config {

inline constexpr std::size_t kParallel1DThreshold = 1;
inline constexpr std::size_t kParallel2DThreshold = 1;
inline constexpr std::size_t kParallel3DThreshold = 1;

[[nodiscard]] constexpr bool should_parallel_1d(std::size_t n) noexcept {
    return n >= kParallel1DThreshold;
}

[[nodiscard]] constexpr bool should_parallel_2d(std::size_t rows, std::size_t cols) noexcept {
    return rows * cols >= kParallel2DThreshold;
}

[[nodiscard]] constexpr bool should_parallel_3d(
    std::size_t rows, std::size_t cols, std::size_t depth
) noexcept {
    return rows * cols * depth >= kParallel3DThreshold;
}

} // namespace poisson::omp_config
