#include "poisson/red_black.hpp"

#include "poisson/metrics.hpp"

#include <stdexcept>
#include <utility>

namespace poisson {
namespace {

template <typename Real>
SolveResult solve_red_black_impl(
    const Problem2D<Real>& problem, const SolveOptions<Real>& options, Real omega
) {
    if (options.tol <= Real{}) {
        throw std::invalid_argument("tol must be positive");
    }
    if (options.max_iter < 1) {
        throw std::invalid_argument("max_iter must be positive");
    }
    if (problem.h <= Real{}) {
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
    if (omega <= Real{} || omega > Real{2}) {
        throw std::invalid_argument("omega must be in (0, 2]");
    }

    Grid2D<Real> phi = problem.phi0;
    const Real h2 = problem.h * problem.h;
    const std::size_t array_n = problem.array_n();
    for (std::size_t iteration = 1; iteration <= options.max_iter; ++iteration) {
        for (std::size_t color = 0; color < 2; ++color) {
            for (std::size_t i = 1; i + 1 < array_n; ++i) {
                const std::size_t j0 = 1 + ((i + color) & 1);
                for (std::size_t j = j0; j + 1 < array_n; j += 2) {
                    const Real update = Real{1} / Real{4} * (
                        phi(i + 1, j) +
                        phi(i - 1, j) +
                        phi(i, j + 1) +
                        phi(i, j - 1) +
                        h2 * problem.rhs(i, j)
                    );
                    phi(i, j) = (Real{1} - omega) * phi(i, j) + omega * update;
                }
            }
        }

        const Real res = relative_physical_residual_l2(problem, phi);
        if (res <= options.tol) {
            return make_solve_result(std::move(phi), iteration, res);
        }
    }

    return make_solve_result(
        std::move(phi),
        options.max_iter,
        relative_physical_residual_l2(problem, phi)
    );
}

} // namespace

template <typename Real>
SolveResult solve_red_black_relaxation(
    const Problem2D<Real>& problem, const SolveOptions<Real>& options, Real omega
) {
    return solve_red_black_impl(problem, options, omega);
}

template SolveResult solve_red_black_relaxation<float>(
    const Problem2D<float>&,
    const SolveOptions<float>&,
    float
);
template SolveResult solve_red_black_relaxation<double>(
    const Problem2D<double>&,
    const SolveOptions<double>&,
    double
);

} // namespace poisson
