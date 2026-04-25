#pragma once

#include "poisson/problem.hpp"

namespace poisson {

struct ValidationReport {
    bool size_ok{false};
    bool h_ok{false};
    bool finite_ok{false};
    bool boundary_ok{false};
    bool interior_zero_ok{false};
    bool ok{false};
    Real boundary_error{0.0};
    Real interior_max_abs{0.0};
};

[[nodiscard]] ValidationReport validate_problem(const Problem2D& problem);

} // namespace poisson

