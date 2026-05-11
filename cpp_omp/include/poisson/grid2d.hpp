#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace poisson {

template <typename Real>
class Grid2D {
public:
    using value_type = Real;

    Grid2D() = default;

    explicit Grid2D(std::size_t size, Real value = Real{})
        : size_{size}, values_(size * size, value) {}

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t elements() const noexcept { return values_.size(); }
    [[nodiscard]] bool empty() const noexcept { return values_.empty(); }
    [[nodiscard]] std::size_t stride() const noexcept { return size_; }

    [[nodiscard]] Real& operator()(std::size_t i, std::size_t j) {
        return values_.at(offset(i, j));
    }

    [[nodiscard]] const Real& operator()(std::size_t i, std::size_t j) const {
        return values_.at(offset(i, j));
    }

    [[nodiscard]] Real& unchecked(std::size_t i, std::size_t j) noexcept {
        return values_[offset_unchecked(i, j)];
    }

    [[nodiscard]] const Real& unchecked(std::size_t i, std::size_t j) const noexcept {
        return values_[offset_unchecked(i, j)];
    }

    void fill(Real value) {
        std::fill(values_.begin(), values_.end(), value);
    }

    [[nodiscard]] std::vector<Real>& data() noexcept { return values_; }
    [[nodiscard]] const std::vector<Real>& data() const noexcept { return values_; }

private:
    [[nodiscard]] std::size_t offset(std::size_t i, std::size_t j) const {
        if (i >= size_ || j >= size_) {
            throw std::out_of_range("Grid2D index out of range");
        }
        return offset_unchecked(i, j);
    }

    [[nodiscard]] std::size_t offset_unchecked(std::size_t i, std::size_t j) const noexcept {
        return i * size_ + j;
    }

    std::size_t size_{0};
    std::vector<Real> values_{};
};

template <typename ToReal, typename FromReal>
[[nodiscard]] Grid2D<ToReal> cast_grid(const Grid2D<FromReal>& source) {
    Grid2D<ToReal> result{source.size()};
    for (std::size_t i = 0; i < source.size(); ++i) {
        for (std::size_t j = 0; j < source.size(); ++j) {
            result(i, j) = static_cast<ToReal>(source(i, j));
        }
    }
    return result;
}

} // namespace poisson
