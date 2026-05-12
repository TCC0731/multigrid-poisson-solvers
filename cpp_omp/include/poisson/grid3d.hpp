#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace poisson {

template <typename Real>
class Grid3D {
public:
    using value_type = Real;

    Grid3D() = default;

    explicit Grid3D(std::size_t size, Real value = Real{})
        : size_{size}, values_(size * size * size, value) {}

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t elements() const noexcept { return values_.size(); }
    [[nodiscard]] bool empty() const noexcept { return values_.empty(); }
    [[nodiscard]] std::size_t stride() const noexcept { return size_; }

    [[nodiscard]] Real& operator()(std::size_t i, std::size_t j, std::size_t k) {
        return values_.at(offset(i, j, k));
    }

    [[nodiscard]] const Real& operator()(std::size_t i, std::size_t j, std::size_t k) const {
        return values_.at(offset(i, j, k));
    }

    [[nodiscard]] Real& unchecked(std::size_t i, std::size_t j, std::size_t k) noexcept {
        return values_[offset_unchecked(i, j, k)];
    }

    [[nodiscard]] const Real& unchecked(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return values_[offset_unchecked(i, j, k)];
    }

    void fill(Real value) {
        std::fill(values_.begin(), values_.end(), value);
    }

    [[nodiscard]] std::vector<Real>& data() noexcept { return values_; }
    [[nodiscard]] const std::vector<Real>& data() const noexcept { return values_; }

private:
    [[nodiscard]] std::size_t offset(std::size_t i, std::size_t j, std::size_t k) const {
        if (i >= size_ || j >= size_ || k >= size_) {
            throw std::out_of_range("Grid3D index out of range");
        }
        return offset_unchecked(i, j, k);
    }

    [[nodiscard]] std::size_t offset_unchecked(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return (i * size_ + j) * size_ + k;
    }

    std::size_t size_{0};
    std::vector<Real> values_{};
};

template <typename ToReal, typename FromReal>
[[nodiscard]] Grid3D<ToReal> cast_grid(const Grid3D<FromReal>& source) {
    Grid3D<ToReal> result{source.size()};
    for (std::size_t i = 0; i < source.size(); ++i) {
        for (std::size_t j = 0; j < source.size(); ++j) {
            for (std::size_t k = 0; k < source.size(); ++k) {
                result.unchecked(i, j, k) = static_cast<ToReal>(source.unchecked(i, j, k));
            }
        }
    }
    return result;
}

} // namespace poisson
