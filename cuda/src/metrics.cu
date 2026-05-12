#include "poisson/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace poisson {

template <typename Real>
Real residual_l2(const Problem2D<Real>& problem, const Grid2D<Real>& phi) {
    const Grid2D<Real> r = residual(phi, problem.rhs, problem.h);
    Real sum_sq = Real{};

    for (std::size_t i = 0; i < r.size(); ++i) {
        for (std::size_t j = 0; j < r.size(); ++j) {
            const Real value = r.unchecked(i, j);
            sum_sq += value * value;
        }
    }

    return problem.h * std::sqrt(sum_sq);
}

template <typename Real>
Real residual_l2(const Problem3D<Real>& problem, const Grid3D<Real>& phi) {
    const Grid3D<Real> r = residual(phi, problem.rhs, problem.h);
    Real sum_sq = Real{};

    for (std::size_t i = 0; i < r.size(); ++i) {
        for (std::size_t j = 0; j < r.size(); ++j) {
            for (std::size_t k = 0; k < r.size(); ++k) {
                const Real value = r.unchecked(i, j, k);
                sum_sq += value * value;
            }
        }
    }

    return std::sqrt(problem.h * problem.h * problem.h) * std::sqrt(sum_sq);
}

template <typename Real>
Real relative_physical_residual_l2(const Problem2D<Real>& problem, const Grid2D<Real>& phi) {
    const Real h2 = problem.h * problem.h;
    const std::size_t interior_end = phi.size() - 1;
    Real res_total = Real{};
    Real rhs_total = Real{};

    for (std::size_t i = 1; i < interior_end; ++i) {
        for (std::size_t j = 1; j < interior_end; ++j) {
            const Real lap = (
                Real{4} * phi.unchecked(i, j) -
                phi.unchecked(i + 1, j) -
                phi.unchecked(i - 1, j) -
                phi.unchecked(i, j + 1) -
                phi.unchecked(i, j - 1)
            );
            const Real b = h2 * problem.rhs.unchecked(i, j);
            const Real d = b - lap;

            res_total += d * d;
            rhs_total += b * b;
        }
    }

    if (rhs_total == Real{}) {
        return std::sqrt(res_total);
    }

    return std::sqrt(res_total / rhs_total);
}

template <typename Real>
Real relative_physical_residual_l2(const Problem3D<Real>& problem, const Grid3D<Real>& phi) {
    const Real h2 = problem.h * problem.h;
    const std::size_t interior_end = phi.size() - 1;
    Real res_total = Real{};
    Real rhs_total = Real{};

    for (std::size_t i = 1; i < interior_end; ++i) {
        for (std::size_t j = 1; j < interior_end; ++j) {
            for (std::size_t k = 1; k < interior_end; ++k) {
                const Real lap = (
                    Real{6} * phi.unchecked(i, j, k) -
                    phi.unchecked(i + 1, j, k) -
                    phi.unchecked(i - 1, j, k) -
                    phi.unchecked(i, j + 1, k) -
                    phi.unchecked(i, j - 1, k) -
                    phi.unchecked(i, j, k + 1) -
                    phi.unchecked(i, j, k - 1)
                );
                const Real b = h2 * problem.rhs.unchecked(i, j, k);
                const Real d = b - lap;

                res_total += d * d;
                rhs_total += b * b;
            }
        }
    }

    if (rhs_total == Real{}) {
        return std::sqrt(res_total);
    }

    return std::sqrt(res_total / rhs_total);
}

template <typename Real>
ErrorMetrics error_metrics(const Problem2D<Real>& problem, const Grid2D<Real>& phi) {
    if (phi.size() != problem.exact.size()) {
        throw std::invalid_argument("solution grid size does not match the problem");
    }

    Real sum_sq = Real{};
    Real max_abs = Real{};
    const std::size_t interior_end = phi.size() - 1;

    for (std::size_t i = 1; i < interior_end; ++i) {
        for (std::size_t j = 1; j < interior_end; ++j) {
            const Real diff = phi.unchecked(i, j) - problem.exact.unchecked(i, j);
            const Real abs_diff = std::abs(diff);
            sum_sq += diff * diff;
            max_abs = std::max(max_abs, abs_diff);
        }
    }

    return {
        static_cast<double>(problem.h * std::sqrt(sum_sq)),
        static_cast<double>(max_abs),
    };
}

template <typename Real>
ErrorMetrics error_metrics(const Problem3D<Real>& problem, const Grid3D<Real>& phi) {
    if (phi.size() != problem.exact.size()) {
        throw std::invalid_argument("solution grid size does not match the problem");
    }

    Real sum_sq = Real{};
    Real max_abs = Real{};
    const std::size_t interior_end = phi.size() - 1;

    for (std::size_t i = 1; i < interior_end; ++i) {
        for (std::size_t j = 1; j < interior_end; ++j) {
            for (std::size_t k = 1; k < interior_end; ++k) {
                const Real diff =
                    phi.unchecked(i, j, k) - problem.exact.unchecked(i, j, k);
                const Real abs_diff = std::abs(diff);
                sum_sq += diff * diff;
                max_abs = std::max(max_abs, abs_diff);
            }
        }
    }

    return {
        static_cast<double>(std::sqrt(problem.h * problem.h * problem.h) * std::sqrt(sum_sq)),
        static_cast<double>(max_abs),
    };
}

template float residual_l2<float>(const Problem2D<float>&, const Grid2D<float>&);
template double residual_l2<double>(const Problem2D<double>&, const Grid2D<double>&);
template float residual_l2<float>(const Problem3D<float>&, const Grid3D<float>&);
template double residual_l2<double>(const Problem3D<double>&, const Grid3D<double>&);
template float relative_physical_residual_l2<float>(const Problem2D<float>&, const Grid2D<float>&);
template double relative_physical_residual_l2<double>(const Problem2D<double>&, const Grid2D<double>&);
template float relative_physical_residual_l2<float>(const Problem3D<float>&, const Grid3D<float>&);
template double relative_physical_residual_l2<double>(const Problem3D<double>&, const Grid3D<double>&);
template ErrorMetrics error_metrics<float>(const Problem2D<float>&, const Grid2D<float>&);
template ErrorMetrics error_metrics<double>(const Problem2D<double>&, const Grid2D<double>&);
template ErrorMetrics error_metrics<float>(const Problem3D<float>&, const Grid3D<float>&);
template ErrorMetrics error_metrics<double>(const Problem3D<double>&, const Grid3D<double>&);

} // namespace poisson
