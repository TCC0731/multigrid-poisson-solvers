#include "poisson/sor.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace poisson {
namespace {

template <typename Real>
Real default_sor_omega(std::size_t interior_n) {
    if (interior_n < 1) {
        throw std::invalid_argument("interior_n must be positive");
    }

    const Real angle = std::numbers::pi_v<Real> / (static_cast<Real>(interior_n) + Real{1});
    return Real{2} / (Real{1} + std::sin(angle));
}

} // namespace

std::string_view SorSolver2D::name() const noexcept {
    return "sor";
}

SolveResult SorSolver2D::solve(
    const Problem2D<double>& problem, const SolveOptions<double>& options
) const {
    return solve_sor<double>(problem, options);
}

std::string_view SorSolver3D::name() const noexcept {
    return "sor";
}

SolveResult3D SorSolver3D::solve(
    const Problem3D<double>& problem, const SolveOptions<double>& options
) const {
    return solve_sor<double>(problem, options);
}

template <typename Real>
SolveResult solve_sor(const Problem2D<Real>& problem, const SolveOptions<Real>& options) {
    return solve_red_black_relaxation<Real>(problem, options, default_sor_omega<Real>(problem.interior_n));
}

template <typename Real>
SolveResult3D solve_sor(const Problem3D<Real>& problem, const SolveOptions<Real>& options) {
    return solve_red_black_relaxation<Real>(problem, options, default_sor_omega<Real>(problem.interior_n));
}

template SolveResult solve_sor<float>(const Problem2D<float>&, const SolveOptions<float>&);
template SolveResult solve_sor<double>(const Problem2D<double>&, const SolveOptions<double>&);
template SolveResult3D solve_sor<float>(const Problem3D<float>&, const SolveOptions<float>&);
template SolveResult3D solve_sor<double>(const Problem3D<double>&, const SolveOptions<double>&);

} // namespace poisson
