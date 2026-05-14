#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace poisson::cuda_kernels {

inline constexpr int kBlockX = 16;
inline constexpr int kBlockY = 16;
inline constexpr int kBlock3DX = 8;
inline constexpr int kBlock3DY = 8;
inline constexpr int kBlock3DZ = 4;
inline constexpr int kReductionThreads = 256;

__device__ __forceinline__ std::size_t offset(std::size_t stride, std::size_t i, std::size_t j) {
    return i * stride + j;
}

__device__ __forceinline__ std::size_t offset(
    std::size_t stride, std::size_t i, std::size_t j, std::size_t k
) {
    return (i * stride + j) * stride + k;
}

__device__ __forceinline__ std::uint32_t offset_u32(
    std::uint32_t stride, std::uint32_t i, std::uint32_t j
) {
    return i * stride + j;
}

__device__ __forceinline__ std::uint32_t offset_u32(
    std::uint32_t stride, std::uint32_t i, std::uint32_t j, std::uint32_t k
) {
    return (i * stride + j) * stride + k;
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

inline dim3 make_block_3d() {
    return dim3{
        static_cast<unsigned int>(kBlock3DX),
        static_cast<unsigned int>(kBlock3DY),
        static_cast<unsigned int>(kBlock3DZ),
    };
}

inline dim3 make_grid_3d(std::size_t width, std::size_t height, std::size_t depth, dim3 block) {
    return dim3{
        static_cast<unsigned int>((width + block.x - 1) / block.x),
        static_cast<unsigned int>((height + block.y - 1) / block.y),
        static_cast<unsigned int>((depth + block.z - 1) / block.z),
    };
}

} // namespace poisson::cuda_kernels
