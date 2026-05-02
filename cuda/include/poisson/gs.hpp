#pragma once

#include "poisson/solver.hpp"

namespace poisson {

template <typename Real>
[[nodiscard]] SolveResult solve_gs(
    const Problem2D<Real>& problem, const SolveOptions<Real>& options
);

} // namespace poisson
