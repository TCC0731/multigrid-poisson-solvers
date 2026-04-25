#include "poisson/gs.hpp"

namespace poisson {

SolveResult solve_gs(const Problem2D& problem, const SolveOptions& options) {
    return solve_red_black_relaxation(problem, options, 1.0);
}

std::string_view GaussSeidelSolver2D::name() const noexcept {
    return "gs";
}

SolveResult GaussSeidelSolver2D::solve(
    const Problem2D& problem, const SolveOptions& options
) const {
    return solve_gs(problem, options);
}

} // namespace poisson
