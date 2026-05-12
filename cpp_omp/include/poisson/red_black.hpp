#pragma once

#include "poisson/solver.hpp"

namespace poisson {

template <typename Real>
[[nodiscard]] SolveResult solve_red_black_relaxation(
    const Problem2D<Real>& problem, const SolveOptions<Real>& options, Real omega
);

template <typename Real>
[[nodiscard]] SolveResult3D solve_red_black_relaxation(
    const Problem3D<Real>& problem, const SolveOptions<Real>& options, Real omega
);

} // namespace poisson
