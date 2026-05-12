#pragma once

#include "poisson/operators.hpp"

namespace poisson {

struct ErrorMetrics {
    double error_l2{};
    double error_linf{};
};

// Legacy absolute residual L2 norm.
template <typename Real>
[[nodiscard]] Real residual_l2(const Problem2D<Real>& problem, const Grid2D<Real>& phi);

template <typename Real>
[[nodiscard]] Real residual_l2(const Problem3D<Real>& problem, const Grid3D<Real>& phi);

// Matches python/solvers/utils.py::_relative_physical_residual_l2.
template <typename Real>
[[nodiscard]] Real relative_physical_residual_l2(
    const Problem2D<Real>& problem, const Grid2D<Real>& phi
);

template <typename Real>
[[nodiscard]] Real relative_physical_residual_l2(
    const Problem3D<Real>& problem, const Grid3D<Real>& phi
);

template <typename Real>
[[nodiscard]] ErrorMetrics error_metrics(const Problem2D<Real>& problem, const Grid2D<Real>& phi);

template <typename Real>
[[nodiscard]] ErrorMetrics error_metrics(const Problem3D<Real>& problem, const Grid3D<Real>& phi);

template <typename Real>
[[nodiscard]] ErrorMetrics metrics(const Problem2D<Real>& problem, const Grid2D<Real>& phi) {
    return error_metrics(problem, phi);
}

template <typename Real>
[[nodiscard]] ErrorMetrics metrics(const Problem3D<Real>& problem, const Grid3D<Real>& phi) {
    return error_metrics(problem, phi);
}

} // namespace poisson
