#include "poisson/gs.hpp"

#include "poisson/cuda_utils.hpp"
#include "poisson/validation.hpp"

#include <stdexcept>

namespace poisson {
namespace {

template <typename Real>
void validate_gs_inputs(const Problem2D<Real>& problem, const SolveOptions<Real>& options) {
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
SolveResult solve_gs(const Problem2D<Real>& problem, const SolveOptions<Real>& options) {
    validate_gs_inputs(problem, options);
    cuda::ensure_device_available();

    cuda::DeviceGrid2D<Real> phi{problem.phi0};
    const cuda::DeviceGrid2D<Real> rhs{problem.rhs};

    for (std::size_t iteration = 1; iteration <= options.max_iter; ++iteration) {
        cuda::run_rb_sor_steps(phi, rhs, problem.h, Real{1}, 1);

        const double residual = cuda::compute_relative_residual(phi, rhs, problem.h);
        if (residual <= static_cast<double>(options.tol)) {
            return make_solve_result(phi.download(), iteration, static_cast<Real>(residual));
        }
    }

    const double residual = cuda::compute_relative_residual(phi, rhs, problem.h);
    return make_solve_result(phi.download(), options.max_iter, static_cast<Real>(residual));
}

template SolveResult solve_gs<float>(const Problem2D<float>&, const SolveOptions<float>&);
template SolveResult solve_gs<double>(const Problem2D<double>&, const SolveOptions<double>&);

} // namespace poisson
