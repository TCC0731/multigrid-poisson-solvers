#pragma once

#include "poisson/operators.hpp"

namespace poisson {

struct ErrorMetrics {
    Real error_l2{};
    Real error_linf{};
};

[[nodiscard]] Real residual_l2(const Problem2D& problem, const Grid2D& phi);
[[nodiscard]] ErrorMetrics error_metrics(const Problem2D& problem, const Grid2D& phi);

} // namespace poisson

