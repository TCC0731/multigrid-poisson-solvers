#include "poisson/sor.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace poisson {
namespace {

Real default_sor_omega(std::size_t interior_n) {
    if (interior_n < 1) {
        throw std::invalid_argument("interior_n must be positive");
    }

    const Real angle = std::numbers::pi_v<Real> / (static_cast<Real>(interior_n) + 1.0);
    return 2.0 / (1.0 + std::sin(angle));
}

} // namespace

SolveResult solve_sor(const Problem2D& problem, const SolveOptions& options) {
    return solve_red_black_relaxation(problem, options, default_sor_omega(problem.interior_n));
}

std::string_view SorSolver2D::name() const noexcept {
    return "sor";
}

SolveResult SorSolver2D::solve(
    const Problem2D& problem, const SolveOptions& options
) const {
    return solve_sor(problem, options);
}

} // namespace poisson
