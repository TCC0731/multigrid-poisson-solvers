#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace poisson::cuda_kernels {

inline constexpr int kBlockX = 16;
inline constexpr int kBlockY = 16;
inline constexpr int kReductionThreads = 256;

__device__ inline std::size_t offset(std::size_t stride, std::size_t i, std::size_t j) {
    return i * stride + j;
}

inline dim3 make_block_2d() {
    return dim3{static_cast<unsigned int>(kBlockX), static_cast<unsigned int>(kBlockY), 1U};
}

inline dim3 make_grid_2d(std::size_t width, std::size_t height, dim3 block) {
    return dim3{
        static_cast<unsigned int>((width + block.x - 1) / block.x),
        static_cast<unsigned int>((height + block.y - 1) / block.y),
        1U,
    };
}

} // namespace poisson::cuda_kernels
