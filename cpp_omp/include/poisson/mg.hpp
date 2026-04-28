#pragma once

#include "poisson/solver.hpp"

namespace poisson {

enum class MGCycle {
    V,
    W,
};

template <typename Real>
struct MGOptions {
    Real tol{Real{1e-10}};
    std::size_t max_iter{100};
    std::size_t nu{2};
    MGCycle cycle{MGCycle::V};
    std::size_t coarse_steps{16};
    Real omega{Real{1}};
    bool omega_is_auto{false};
};

template <typename Real>
[[nodiscard]] SolveResult solve_mg_exact(
    const Problem2D<Real>& problem, const MGOptions<Real>& options = MGOptions<Real>{}
);

template <typename Real>
[[nodiscard]] SolveResult solve_mg_sor(
    const Problem2D<Real>& problem, const MGOptions<Real>& options = MGOptions<Real>{}
);

} // namespace poisson
