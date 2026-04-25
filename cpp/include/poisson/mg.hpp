#pragma once

#include "poisson/solver.hpp"

namespace poisson {

enum class MGCycle {
    V,
    W,
};

struct MGOptions {
    Real tol{1e-10};
    std::size_t max_iter{100};
    std::size_t nu{2};
    MGCycle cycle{MGCycle::V};
    std::size_t coarse_steps{16};
};

[[nodiscard]] SolveResult solve_mg_exact(const Problem2D& problem, const MGOptions& options = {});
[[nodiscard]] SolveResult solve_mg_sor(const Problem2D& problem, const MGOptions& options = {});

} // namespace poisson
