#include "poisson/validation.hpp"

#include <algorithm>
#include <cmath>

namespace poisson {
namespace {

bool finite_grid(const Grid2D& grid) {
    for (std::size_t i = 0; i < grid.size(); ++i) {
        for (std::size_t j = 0; j < grid.size(); ++j) {
            if (!std::isfinite(grid(i, j))) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

ValidationReport validate_problem(const Problem2D& problem) {
    ValidationReport report{};

    const std::size_t expected_n = problem.array_n();
    report.size_ok =
        problem.exact.size() == expected_n &&
        problem.rhs.size() == expected_n &&
        problem.phi0.size() == expected_n;
    report.h_ok = problem.h > 0.0;

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
    for (std::size_t i = 0; i < array_n; ++i) {
        report.boundary_error = std::max(report.boundary_error, std::abs(problem.phi0(i, 0) - problem.exact(i, 0)));
        report.boundary_error = std::max(report.boundary_error, std::abs(problem.phi0(i, array_n - 1) - problem.exact(i, array_n - 1)));
    }

    for (std::size_t j = 0; j < array_n; ++j) {
        report.boundary_error = std::max(report.boundary_error, std::abs(problem.phi0(0, j) - problem.exact(0, j)));
        report.boundary_error = std::max(report.boundary_error, std::abs(problem.phi0(array_n - 1, j) - problem.exact(array_n - 1, j)));
    }

    for (std::size_t i = 1; i + 1 < array_n; ++i) {
        for (std::size_t j = 1; j + 1 < array_n; ++j) {
            report.interior_max_abs = std::max(report.interior_max_abs, std::abs(problem.phi0(i, j)));
        }
    }

    constexpr Real tol = 1e-12;
    report.boundary_ok = report.boundary_error <= tol;
    report.interior_zero_ok = report.interior_max_abs <= tol;
    report.ok = report.size_ok && report.h_ok && report.finite_ok && report.boundary_ok && report.interior_zero_ok;

    return report;
}

} // namespace poisson
