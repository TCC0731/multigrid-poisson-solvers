#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace poisson {

using Real = double;

class Grid2D {
public:
    Grid2D() = default;

    explicit Grid2D(std::size_t size, Real value = 0.0)
        : size_{size}, values_(size * size, value) {}

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t elements() const noexcept { return values_.size(); }
    [[nodiscard]] bool empty() const noexcept { return values_.empty(); }

    [[nodiscard]] Real& operator()(std::size_t i, std::size_t j) {
        return values_.at(offset(i, j));
    }

    [[nodiscard]] const Real& operator()(std::size_t i, std::size_t j) const {
        return values_.at(offset(i, j));
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
        return i * size_ + j;
    }

    std::size_t size_{0};
    std::vector<Real> values_{};
};

} // namespace poisson
