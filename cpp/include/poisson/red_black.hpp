#pragma once

#include "poisson/solver.hpp"

namespace poisson {

[[nodiscard]] SolveResult solve_red_black_relaxation(
    const Problem2D& problem, const SolveOptions& options, Real omega
);

} // namespace poisson
