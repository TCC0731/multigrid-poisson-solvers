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
    double boundary_error{0.0};
    double interior_max_abs{0.0};
};

template <typename Real>
[[nodiscard]] ValidationReport validate_problem(const Problem2D<Real>& problem);

template <typename Real>
[[nodiscard]] ValidationReport validate_problem(const Problem3D<Real>& problem);

} // namespace poisson
