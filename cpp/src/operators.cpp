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
    Grid2D<Real> result{array_n - 2};
    const Real inv_h2 = Real{1} / (h * h);

    for (std::size_t i = 1; i + 1 < array_n; ++i) {
        for (std::size_t j = 1; j + 1 < array_n; ++j) {
            result(i - 1, j - 1) = (
                Real{4} * phi(i, j)
                - phi(i + 1, j)
                - phi(i - 1, j)
                - phi(i, j + 1)
                - phi(i, j - 1)
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
    Grid2D<Real> result{array_n - 2};
    const Real inv_h2 = Real{1} / (h * h);

    for (std::size_t i = 1; i + 1 < array_n; ++i) {
        for (std::size_t j = 1; j + 1 < array_n; ++j) {
            const Real a_phi = (
                Real{4} * phi(i, j)
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

template <typename Real>
Grid3D<Real> apply_A(const Grid3D<Real>& phi, Real h) {
    if (h <= Real{}) {
        throw std::invalid_argument("grid spacing h must be positive");
    }
    if (phi.size() < 3) {
        throw std::invalid_argument("phi must include at least one interior cell");
    }

    const std::size_t array_n = phi.size();
    Grid3D<Real> result{array_n - 2};
    const Real inv_h2 = Real{1} / (h * h);

    for (std::size_t i = 1; i + 1 < array_n; ++i) {
        for (std::size_t j = 1; j + 1 < array_n; ++j) {
            for (std::size_t k = 1; k + 1 < array_n; ++k) {
                result(i - 1, j - 1, k - 1) = (
                    Real{6} * phi(i, j, k)
                    - phi(i + 1, j, k)
                    - phi(i - 1, j, k)
                    - phi(i, j + 1, k)
                    - phi(i, j - 1, k)
                    - phi(i, j, k + 1)
                    - phi(i, j, k - 1)
                ) * inv_h2;
            }
        }
    }

    return result;
}

template <typename Real>
Grid3D<Real> residual(const Grid3D<Real>& phi, const Grid3D<Real>& rhs, Real h) {
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
    Grid3D<Real> result{array_n - 2};
    const Real inv_h2 = Real{1} / (h * h);

    for (std::size_t i = 1; i + 1 < array_n; ++i) {
        for (std::size_t j = 1; j + 1 < array_n; ++j) {
            for (std::size_t k = 1; k + 1 < array_n; ++k) {
                const Real a_phi = (
                    Real{6} * phi(i, j, k)
                    - phi(i + 1, j, k)
                    - phi(i - 1, j, k)
                    - phi(i, j + 1, k)
                    - phi(i, j - 1, k)
                    - phi(i, j, k + 1)
                    - phi(i, j, k - 1)
                ) * inv_h2;
                result(i - 1, j - 1, k - 1) = rhs(i, j, k) - a_phi;
            }
        }
    }

    return result;
}

template Grid2D<float> apply_A<float>(const Grid2D<float>&, float);
template Grid2D<double> apply_A<double>(const Grid2D<double>&, double);
template Grid2D<float> residual<float>(const Grid2D<float>&, const Grid2D<float>&, float);
template Grid2D<double> residual<double>(const Grid2D<double>&, const Grid2D<double>&, double);
template Grid3D<float> apply_A<float>(const Grid3D<float>&, float);
template Grid3D<double> apply_A<double>(const Grid3D<double>&, double);
template Grid3D<float> residual<float>(const Grid3D<float>&, const Grid3D<float>&, float);
template Grid3D<double> residual<double>(const Grid3D<double>&, const Grid3D<double>&, double);

} // namespace poisson
