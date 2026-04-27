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
            sum_sq += r(i, j) * r(i, j);
        }
    }

    return problem.h * std::sqrt(sum_sq);
}

template <typename Real>
ErrorMetrics error_metrics(const Problem2D<Real>& problem, const Grid2D<Real>& phi) {
    if (phi.size() != problem.exact.size()) {
        throw std::invalid_argument("solution grid size does not match the problem");
    }

    Real sum_sq = Real{};
    Real max_abs = Real{};

    for (std::size_t i = 1; i + 1 < phi.size(); ++i) {
        for (std::size_t j = 1; j + 1 < phi.size(); ++j) {
            const Real diff = phi(i, j) - problem.exact(i, j);
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

template float residual_l2<float>(const Problem2D<float>&, const Grid2D<float>&);
template double residual_l2<double>(const Problem2D<double>&, const Grid2D<double>&);
template ErrorMetrics error_metrics<float>(const Problem2D<float>&, const Grid2D<float>&);
template ErrorMetrics error_metrics<double>(const Problem2D<double>&, const Grid2D<double>&);

} // namespace poisson
