#include "poisson/jacobi.hpp"

#include "poisson/cuda_utils.hpp"
#include "poisson/validation.hpp"

#include <stdexcept>
#include <utility>

namespace poisson {
namespace {

template <typename Real>
void validate_jacobi_inputs(const Problem2D<Real>& problem, const SolveOptions<Real>& options) {
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
    const ValidationReport report = validate_problem(problem);
    if (!report.ok) {
        throw std::invalid_argument("problem is invalid");
    }
}

template <typename Real>
void validate_jacobi_inputs(const Problem3D<Real>& problem, const SolveOptions<Real>& options) {
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
    const ValidationReport report = validate_problem(problem);
    if (!report.ok) {
        throw std::invalid_argument("problem is invalid");
    }
}

} // namespace

template <typename Real>
SolveResult solve_jacobi(const Problem2D<Real>& problem, const SolveOptions<Real>& options) {
    const cuda::detail::ScopedNvtxRange solve_range{"cuda::solve_jacobi"};
    validate_jacobi_inputs(problem, options);
    cuda::ensure_device_available();

    cuda::DeviceGrid2D<Real> phi{problem.phi0};
    const cuda::DeviceGrid2D<Real> rhs{problem.rhs};
    cuda::DeviceGrid2D<Real> work{problem.phi0};
    cuda::RelativeResidualWorkspace residual_workspace{problem.array_n()};

    for (std::size_t iteration = 1; iteration <= options.max_iter; ++iteration) {
        const cuda::detail::ScopedNvtxRange iteration_range{"cuda::jacobi_iteration"};
        cuda::run_jacobi_step(phi, rhs, problem.h, work);

        if (cuda::should_check_non_mg_residual(iteration, options.max_iter)) {
            const double residual =
                cuda::compute_relative_residual(work, rhs, problem.h, residual_workspace);
            if (residual <= static_cast<double>(options.tol) || iteration == options.max_iter) {
                return make_solve_result(work.download(), iteration, static_cast<Real>(residual));
            }
        }

        std::swap(phi, work);
    }

    throw std::logic_error("jacobi iteration loop exited unexpectedly");
}

template SolveResult solve_jacobi<float>(const Problem2D<float>&, const SolveOptions<float>&);
template SolveResult solve_jacobi<double>(const Problem2D<double>&, const SolveOptions<double>&);

template <typename Real>
SolveResult3D solve_jacobi(const Problem3D<Real>& problem, const SolveOptions<Real>& options) {
    const cuda::detail::ScopedNvtxRange solve_range{"cuda::solve_jacobi_3d"};
    validate_jacobi_inputs(problem, options);
    cuda::ensure_device_available();

    cuda::DeviceGrid3D<Real> phi{problem.phi0};
    const cuda::DeviceGrid3D<Real> rhs{problem.rhs};
    cuda::DeviceGrid3D<Real> work{problem.phi0};
    cuda::RelativeResidualWorkspace3D residual_workspace{problem.array_n()};

    for (std::size_t iteration = 1; iteration <= options.max_iter; ++iteration) {
        const cuda::detail::ScopedNvtxRange iteration_range{"cuda::jacobi_iteration_3d"};
        cuda::run_jacobi_step(phi, rhs, problem.h, work);

        if (cuda::should_check_non_mg_residual(iteration, options.max_iter)) {
            const double residual =
                cuda::compute_relative_residual(work, rhs, problem.h, residual_workspace);
            if (residual <= static_cast<double>(options.tol) || iteration == options.max_iter) {
                return make_solve_result(work.download(), iteration, static_cast<Real>(residual));
            }
        }

        std::swap(phi, work);
    }

    throw std::logic_error("jacobi iteration loop exited unexpectedly");
}

template SolveResult3D solve_jacobi<float>(const Problem3D<float>&, const SolveOptions<float>&);
template SolveResult3D solve_jacobi<double>(const Problem3D<double>&, const SolveOptions<double>&);

} // namespace poisson
