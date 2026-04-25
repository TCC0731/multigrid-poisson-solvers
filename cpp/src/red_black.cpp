#include "poisson/red_black.hpp"

#include "poisson/metrics.hpp"

#include <stdexcept>
#include <utility>

namespace poisson {
namespace {

SolveResult solve_red_black_impl(
    const Problem2D& problem, const SolveOptions& options, Real omega
) {
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
    if (omega <= 0.0 || omega > 2.0) {
        throw std::invalid_argument("omega must be in (0, 2]");
    }

    Grid2D phi = problem.phi0;
    const Real h2 = problem.h * problem.h;
    const std::size_t array_n = problem.array_n();
    for (std::size_t iteration = 1; iteration <= options.max_iter; ++iteration) {
        for (std::size_t color = 0; color < 2; ++color) {
            for (std::size_t i = 1; i + 1 < array_n; ++i) {
                const std::size_t j0 = 1 + ((i + color) & 1);
                for (std::size_t j = j0; j + 1 < array_n; j += 2) {
                    const Real update = 0.25 * (
                        phi(i + 1, j) +
                        phi(i - 1, j) +
                        phi(i, j + 1) +
                        phi(i, j - 1) +
                        h2 * problem.rhs(i, j)
                    );
                    phi(i, j) = (1.0 - omega) * phi(i, j) + omega * update;
                }
            }
        }

        const Real res = residual_l2(problem, phi);
        if (res <= options.tol) {
            return {std::move(phi), iteration, res};
        }
    }

    return {std::move(phi), options.max_iter, residual_l2(problem, phi)};
}

} // namespace

SolveResult solve_red_black_relaxation(
    const Problem2D& problem, const SolveOptions& options, Real omega
) {
    return solve_red_black_impl(problem, options, omega);
}

} // namespace poisson
