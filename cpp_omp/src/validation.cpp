#include "poisson/validation.hpp"
#include "poisson/omp_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace poisson {
namespace {

template <typename Real>
bool finite_grid(const Grid2D<Real>& grid) {
    bool all_finite = true;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if(omp_config::should_parallel_2d(grid.size(), grid.size())) reduction(&&:all_finite) schedule(static)
#endif
    for (std::size_t i = 0; i < grid.size(); ++i) {
        for (std::size_t j = 0; j < grid.size(); ++j) {
            all_finite = all_finite && std::isfinite(grid.unchecked(i, j));
        }
    }
    return all_finite;
}

} // namespace

template <typename Real>
ValidationReport validate_problem(const Problem2D<Real>& problem) {
    ValidationReport report{};

    const std::size_t expected_n = problem.array_n();
    report.size_ok =
        problem.exact.size() == expected_n &&
        problem.rhs.size() == expected_n &&
        problem.phi0.size() == expected_n;
    report.h_ok = problem.h > Real{};

    if (!report.size_ok || !report.h_ok) {
        report.ok = false;
        return report;
    }

    report.finite_ok =
        finite_grid(problem.exact) &&
        finite_grid(problem.rhs) &&
        finite_grid(problem.phi0);

    if (!report.finite_ok) {
        report.ok = false;
        return report;
    }

    const std::size_t array_n = problem.array_n();
    const std::size_t interior_end = array_n - 1;
    double boundary_error = 0.0;
#ifdef _OPENMP
#pragma omp parallel for if(omp_config::should_parallel_1d(array_n)) reduction(max:boundary_error) schedule(static)
#endif
    for (std::size_t i = 0; i < array_n; ++i) {
        boundary_error = std::max(
            boundary_error,
            static_cast<double>(std::abs(problem.phi0.unchecked(i, 0) - problem.exact.unchecked(i, 0)))
        );
        boundary_error = std::max(
            boundary_error,
            static_cast<double>(std::abs(problem.phi0.unchecked(i, array_n - 1) - problem.exact.unchecked(i, array_n - 1)))
        );
    }

#ifdef _OPENMP
#pragma omp parallel for if(omp_config::should_parallel_1d(array_n)) reduction(max:boundary_error) schedule(static)
#endif
    for (std::size_t j = 0; j < array_n; ++j) {
        boundary_error = std::max(
            boundary_error,
            static_cast<double>(std::abs(problem.phi0.unchecked(0, j) - problem.exact.unchecked(0, j)))
        );
        boundary_error = std::max(
            boundary_error,
            static_cast<double>(std::abs(problem.phi0.unchecked(array_n - 1, j) - problem.exact.unchecked(array_n - 1, j)))
        );
    }

    double interior_max_abs = 0.0;
#ifdef _OPENMP
#pragma omp parallel for if(omp_config::should_parallel_2d(interior_end - 1, interior_end - 1)) reduction(max:interior_max_abs) schedule(static)
#endif
    for (std::size_t i = 1; i < interior_end; ++i) {
#ifdef _OPENMP
#pragma omp simd reduction(max:interior_max_abs)
#endif
        for (std::size_t j = 1; j < interior_end; ++j) {
            interior_max_abs = std::max(
                interior_max_abs,
                static_cast<double>(std::abs(problem.phi0.unchecked(i, j)))
            );
        }
    }

    report.boundary_error = boundary_error;
    report.interior_max_abs = interior_max_abs;

    const double tol = static_cast<double>(Real{100} * std::numeric_limits<Real>::epsilon());
    report.boundary_ok = report.boundary_error <= tol;
    report.interior_zero_ok = report.interior_max_abs <= tol;
    report.ok = report.size_ok && report.h_ok && report.finite_ok && report.boundary_ok && report.interior_zero_ok;

    return report;
}

template ValidationReport validate_problem<float>(const Problem2D<float>&);
template ValidationReport validate_problem<double>(const Problem2D<double>&);

} // namespace poisson
