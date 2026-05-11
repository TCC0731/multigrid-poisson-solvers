#include "poisson/operators.hpp"

#include <stdexcept>

namespace poisson {

template <typename Real>
Grid2D<Real> apply_A(const Grid2D<Real>& phi, Real h) {
    if (h <= Real{}) {
        throw std::invalid_argument("grid spacing h must be positive");
    }
    if (phi.size() < 3) {
        throw std::invalid_argument("phi must include at least one interior cell");
    }

    const std::size_t array_n = phi.size();
    const std::size_t interior_end = array_n - 1;
    Grid2D<Real> result{array_n - 2};
    const Real inv_h2 = Real{1} / (h * h);

    for (std::size_t i = 1; i < interior_end; ++i) {
        for (std::size_t j = 1; j < interior_end; ++j) {
            result.unchecked(i - 1, j - 1) = (
                Real{4} * phi.unchecked(i, j)
                - phi.unchecked(i + 1, j)
                - phi.unchecked(i - 1, j)
                - phi.unchecked(i, j + 1)
                - phi.unchecked(i, j - 1)
            ) * inv_h2;
        }
    }

    return result;
}

template <typename Real>
Grid2D<Real> residual(const Grid2D<Real>& phi, const Grid2D<Real>& rhs, Real h) {
    if (phi.size() != rhs.size()) {
        throw std::invalid_argument("phi and rhs must have the same grid size");
    }
    if (h <= Real{}) {
        throw std::invalid_argument("grid spacing h must be positive");
    }
    if (phi.size() < 3) {
        throw std::invalid_argument("phi must include at least one interior cell");
    }

    const std::size_t array_n = phi.size();
    const std::size_t interior_end = array_n - 1;
    Grid2D<Real> result{array_n - 2};
    const Real inv_h2 = Real{1} / (h * h);

    for (std::size_t i = 1; i < interior_end; ++i) {
        for (std::size_t j = 1; j < interior_end; ++j) {
            const Real a_phi = (
                Real{4} * phi.unchecked(i, j)
                - phi.unchecked(i + 1, j)
                - phi.unchecked(i - 1, j)
                - phi.unchecked(i, j + 1)
                - phi.unchecked(i, j - 1)
            ) * inv_h2;
            result.unchecked(i - 1, j - 1) = rhs.unchecked(i, j) - a_phi;
        }
    }

    return result;
}

template Grid2D<float> apply_A<float>(const Grid2D<float>&, float);
template Grid2D<double> apply_A<double>(const Grid2D<double>&, double);
template Grid2D<float> residual<float>(const Grid2D<float>&, const Grid2D<float>&, float);
template Grid2D<double> residual<double>(const Grid2D<double>&, const Grid2D<double>&, double);

} // namespace poisson
