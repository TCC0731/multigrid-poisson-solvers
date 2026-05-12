#pragma once

#include "poisson/solver.hpp"

namespace poisson {

template <typename Real>
[[nodiscard]] SolveResult solve_sor(const Problem2D<Real>& problem, const SolveOptions<Real>& options);

template <typename Real>
[[nodiscard]] SolveResult3D solve_sor(
    const Problem3D<Real>& problem, const SolveOptions<Real>& options
);

} // namespace poisson
