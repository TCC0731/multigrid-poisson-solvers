#include "poisson/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace poisson {

Real residual_l2(const Problem2D& problem, const Grid2D& phi) {
    const Grid2D r = residual(phi, problem.rhs, problem.h);
    Real sum_sq = 0.0;

    for (std::size_t i = 0; i < r.size(); ++i) {
        for (std::size_t j = 0; j < r.size(); ++j) {
            sum_sq += r(i, j) * r(i, j);
        }
    }

    return problem.h * std::sqrt(sum_sq);
}

ErrorMetrics error_metrics(const Problem2D& problem, const Grid2D& phi) {
    if (phi.size() != problem.exact.size()) {
        throw std::invalid_argument("solution grid size does not match the problem");
    }

    Real sum_sq = 0.0;
    Real max_abs = 0.0;

    for (std::size_t i = 1; i + 1 < phi.size(); ++i) {
        for (std::size_t j = 1; j + 1 < phi.size(); ++j) {
            const Real diff = phi(i, j) - problem.exact(i, j);
            const Real abs_diff = std::abs(diff);
            sum_sq += diff * diff;
            max_abs = std::max(max_abs, abs_diff);
        }
    }

    return {
        problem.h * std::sqrt(sum_sq),
        max_abs,
    };
}

} // namespace poisson

