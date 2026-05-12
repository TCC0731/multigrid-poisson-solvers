#include "poisson/jacobi.hpp"

#include "poisson/metrics.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace poisson {
namespace {

template <typename Real>
SolveResult solve_jacobi_impl(const Problem2D<Real>& problem, const SolveOptions<Real>& options) {
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

    Grid2D<Real> phi = problem.phi0;
    Grid2D<Real> work = phi;
    const Real h2 = problem.h * problem.h;
    const std::size_t array_n = problem.array_n();

    for (std::size_t iteration = 1; iteration <= options.max_iter; ++iteration) {
        for (std::size_t i = 1; i + 1 < array_n; ++i) {
            for (std::size_t j = 1; j + 1 < array_n; ++j) {
                work(i, j) = Real{1} / Real{4} * (
                    phi(i + 1, j) +
                    phi(i - 1, j) +
                    phi(i, j + 1) +
                    phi(i, j - 1) +
                    h2 * problem.rhs(i, j)
                );
            }
        }

        const Real res = relative_physical_residual_l2(problem, work);
        if (res <= options.tol) {
            return make_solve_result(std::move(work), iteration, res);
        }

        std::swap(phi, work);
    }

    return make_solve_result(
        std::move(phi),
        options.max_iter,
        relative_physical_residual_l2(problem, phi)
    );
}

template <typename Real>
SolveResult3D solve_jacobi_3d_impl(const Problem3D<Real>& problem, const SolveOptions<Real>& options) {
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

    Grid3D<Real> phi = problem.phi0;
    Grid3D<Real> work = phi;
    const Real h2 = problem.h * problem.h;
    const std::size_t array_n = problem.array_n();

    for (std::size_t iteration = 1; iteration <= options.max_iter; ++iteration) {
        for (std::size_t i = 1; i + 1 < array_n; ++i) {
            for (std::size_t j = 1; j + 1 < array_n; ++j) {
                for (std::size_t k = 1; k + 1 < array_n; ++k) {
                    work(i, j, k) = Real{1} / Real{6} * (
                        phi(i + 1, j, k) +
                        phi(i - 1, j, k) +
                        phi(i, j + 1, k) +
                        phi(i, j - 1, k) +
                        phi(i, j, k + 1) +
                        phi(i, j, k - 1) +
                        h2 * problem.rhs(i, j, k)
                    );
                }
            }
        }

        const Real res = relative_physical_residual_l2(problem, work);
        if (res <= options.tol) {
            return make_solve_result(std::move(work), iteration, res);
        }

        std::swap(phi, work);
    }

    return make_solve_result(
        std::move(phi),
        options.max_iter,
        relative_physical_residual_l2(problem, phi)
    );
}

} // namespace

std::string_view JacobiSolver2D::name() const noexcept {
    return "jacobi";
}

SolveResult JacobiSolver2D::solve(
    const Problem2D<double>& problem, const SolveOptions<double>& options
) const {
    return solve_jacobi<double>(problem, options);
}

std::string_view JacobiSolver3D::name() const noexcept {
    return "jacobi";
}

SolveResult3D JacobiSolver3D::solve(
    const Problem3D<double>& problem, const SolveOptions<double>& options
) const {
    return solve_jacobi<double>(problem, options);
}

template <typename Real>
SolveResult solve_jacobi(const Problem2D<Real>& problem, const SolveOptions<Real>& options) {
    return solve_jacobi_impl(problem, options);
}

template <typename Real>
SolveResult3D solve_jacobi(const Problem3D<Real>& problem, const SolveOptions<Real>& options) {
    return solve_jacobi_3d_impl(problem, options);
}

template SolveResult solve_jacobi<float>(const Problem2D<float>&, const SolveOptions<float>&);
template SolveResult solve_jacobi<double>(const Problem2D<double>&, const SolveOptions<double>&);
template SolveResult3D solve_jacobi<float>(const Problem3D<float>&, const SolveOptions<float>&);
template SolveResult3D solve_jacobi<double>(const Problem3D<double>&, const SolveOptions<double>&);

} // namespace poisson
