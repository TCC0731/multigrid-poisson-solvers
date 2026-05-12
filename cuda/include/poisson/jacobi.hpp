#pragma once

#include "poisson/solver.hpp"

namespace poisson {

template <typename Real>
[[nodiscard]] SolveResult solve_jacobi(
    const Problem2D<Real>& problem, const SolveOptions<Real>& options
);

template <typename Real>
[[nodiscard]] SolveResult3D solve_jacobi(
    const Problem3D<Real>& problem, const SolveOptions<Real>& options
);

} // namespace poisson
