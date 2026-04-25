#include "poisson/jacobi.hpp"

#include "poisson/metrics.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace poisson {
namespace {

SolveResult solve_jacobi_impl(const Problem2D& problem, const SolveOptions& options) {
    if (options.tol <= 0.0) {
        throw std::invalid_argument("tol must be positive");
    }
    if (options.max_iter < 1) {
        throw std::invalid_argument("max_iter must be positive");
    }
    if (problem.h <= 0.0) {
        throw std::invalid_argument("grid spacing h must be positive");
    }
    if (
        problem.exact.size() != problem.rhs.size() ||
        problem.phi0.size() != problem.rhs.size()
    ) {
        throw std::invalid_argument("problem grids must have the same size");
    }
    if (problem.array_n() < 3) {
        throw std::invalid_argument("problem must include at least one interior cell");
    }

    Grid2D phi = problem.phi0;
    Grid2D work = phi;
    const Real h2 = problem.h * problem.h;
    const std::size_t array_n = problem.array_n();

    for (std::size_t iteration = 1; iteration <= options.max_iter; ++iteration) {
        for (std::size_t i = 1; i + 1 < array_n; ++i) {
            for (std::size_t j = 1; j + 1 < array_n; ++j) {
                work(i, j) = 0.25 * (
                    phi(i + 1, j) +
                    phi(i - 1, j) +
                    phi(i, j + 1) +
                    phi(i, j - 1) +
                    h2 * problem.rhs(i, j)
                );
            }
        }

        const Real res = residual_l2(problem, work);
        if (res <= options.tol) {
            return {std::move(work), iteration, res};
        }

        std::swap(phi, work);
    }

    return {std::move(phi), options.max_iter, residual_l2(problem, phi)};
}

} // namespace

std::string_view JacobiSolver2D::name() const noexcept {
    return "jacobi";
}

SolveResult JacobiSolver2D::solve(
    const Problem2D& problem, const SolveOptions& options
) const {
    return solve_jacobi_impl(problem, options);
}

SolveResult solve_jacobi(const Problem2D& problem, const SolveOptions& options) {
    return solve_jacobi_impl(problem, options);
}

} // namespace poisson
