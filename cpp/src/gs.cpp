#include "poisson/gs.hpp"

namespace poisson {

std::string_view GaussSeidelSolver2D::name() const noexcept {
    return "gs";
}

SolveResult GaussSeidelSolver2D::solve(
    const Problem2D<double>& problem, const SolveOptions<double>& options
) const {
    return solve_gs<double>(problem, options);
}

std::string_view GaussSeidelSolver3D::name() const noexcept {
    return "gs";
}

SolveResult3D GaussSeidelSolver3D::solve(
    const Problem3D<double>& problem, const SolveOptions<double>& options
) const {
    return solve_gs<double>(problem, options);
}

template <typename Real>
SolveResult solve_gs(const Problem2D<Real>& problem, const SolveOptions<Real>& options) {
    return solve_red_black_relaxation<Real>(problem, options, Real{1});
}

template <typename Real>
SolveResult3D solve_gs(const Problem3D<Real>& problem, const SolveOptions<Real>& options) {
    return solve_red_black_relaxation<Real>(problem, options, Real{1});
}

template SolveResult solve_gs<float>(const Problem2D<float>&, const SolveOptions<float>&);
template SolveResult solve_gs<double>(const Problem2D<double>&, const SolveOptions<double>&);
template SolveResult3D solve_gs<float>(const Problem3D<float>&, const SolveOptions<float>&);
template SolveResult3D solve_gs<double>(const Problem3D<double>&, const SolveOptions<double>&);

} // namespace poisson
