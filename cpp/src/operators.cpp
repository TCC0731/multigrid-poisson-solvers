#include "poisson/operators.hpp"

#include <stdexcept>

namespace poisson {

Grid2D apply_A(const Grid2D& phi, Real h) {
    if (h <= 0.0) {
        throw std::invalid_argument("grid spacing h must be positive");
    }
    if (phi.size() < 3) {
        throw std::invalid_argument("phi must include at least one interior cell");
    }

    const std::size_t array_n = phi.size();
    Grid2D result{array_n - 2};
    const Real inv_h2 = 1.0 / (h * h);

    for (std::size_t i = 1; i + 1 < array_n; ++i) {
        for (std::size_t j = 1; j + 1 < array_n; ++j) {
            result(i - 1, j - 1) = (
                4.0 * phi(i, j)
                - phi(i + 1, j)
                - phi(i - 1, j)
                - phi(i, j + 1)
                - phi(i, j - 1)
            ) * inv_h2;
        }
    }

    return result;
}

Grid2D residual(const Grid2D& phi, const Grid2D& rhs, Real h) {
    if (phi.size() != rhs.size()) {
        throw std::invalid_argument("phi and rhs must have the same grid size");
    }
    if (h <= 0.0) {
        throw std::invalid_argument("grid spacing h must be positive");
    }
    if (phi.size() < 3) {
        throw std::invalid_argument("phi must include at least one interior cell");
    }

    const std::size_t array_n = phi.size();
    Grid2D result{array_n - 2};
    const Real inv_h2 = 1.0 / (h * h);

    for (std::size_t i = 1; i + 1 < array_n; ++i) {
        for (std::size_t j = 1; j + 1 < array_n; ++j) {
            const Real a_phi = (
                4.0 * phi(i, j)
                - phi(i + 1, j)
                - phi(i - 1, j)
                - phi(i, j + 1)
                - phi(i, j - 1)
            ) * inv_h2;
            result(i - 1, j - 1) = rhs(i, j) - a_phi;
        }
    }

    return result;
}

} // namespace poisson

