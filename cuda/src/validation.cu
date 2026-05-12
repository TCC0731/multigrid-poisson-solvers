#include "poisson/validation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace poisson {
namespace {

template <typename Real>
bool finite_grid(const Grid2D<Real>& grid) {
    for (std::size_t i = 0; i < grid.size(); ++i) {
        for (std::size_t j = 0; j < grid.size(); ++j) {
            if (!std::isfinite(grid.unchecked(i, j))) {
                return false;
            }
        }
    }
    return true;
}

template <typename Real>
bool finite_grid(const Grid3D<Real>& grid) {
    for (std::size_t i = 0; i < grid.size(); ++i) {
        for (std::size_t j = 0; j < grid.size(); ++j) {
            for (std::size_t k = 0; k < grid.size(); ++k) {
                if (!std::isfinite(grid.unchecked(i, j, k))) {
                    return false;
                }
            }
        }
    }
    return true;
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
    for (std::size_t i = 1; i < interior_end; ++i) {
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

template <typename Real>
ValidationReport validate_problem(const Problem3D<Real>& problem) {
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
    for (std::size_t i = 0; i < array_n; ++i) {
        for (std::size_t j = 0; j < array_n; ++j) {
            report.boundary_error = std::max(
                report.boundary_error,
                static_cast<double>(
                    std::abs(problem.phi0.unchecked(i, j, 0) - problem.exact.unchecked(i, j, 0))
                )
            );
            report.boundary_error = std::max(
                report.boundary_error,
                static_cast<double>(
                    std::abs(
                        problem.phi0.unchecked(i, j, array_n - 1) -
                        problem.exact.unchecked(i, j, array_n - 1)
                    )
                )
            );
        }
    }

    for (std::size_t i = 0; i < array_n; ++i) {
        for (std::size_t k = 0; k < array_n; ++k) {
            report.boundary_error = std::max(
                report.boundary_error,
                static_cast<double>(
                    std::abs(problem.phi0.unchecked(i, 0, k) - problem.exact.unchecked(i, 0, k))
                )
            );
            report.boundary_error = std::max(
                report.boundary_error,
                static_cast<double>(
                    std::abs(
                        problem.phi0.unchecked(i, array_n - 1, k) -
                        problem.exact.unchecked(i, array_n - 1, k)
                    )
                )
            );
        }
    }

    for (std::size_t j = 0; j < array_n; ++j) {
        for (std::size_t k = 0; k < array_n; ++k) {
            report.boundary_error = std::max(
                report.boundary_error,
                static_cast<double>(
                    std::abs(problem.phi0.unchecked(0, j, k) - problem.exact.unchecked(0, j, k))
                )
            );
            report.boundary_error = std::max(
                report.boundary_error,
                static_cast<double>(
                    std::abs(
                        problem.phi0.unchecked(array_n - 1, j, k) -
                        problem.exact.unchecked(array_n - 1, j, k)
                    )
                )
            );
        }
    }

    for (std::size_t i = 1; i + 1 < array_n; ++i) {
        for (std::size_t j = 1; j + 1 < array_n; ++j) {
            for (std::size_t k = 1; k + 1 < array_n; ++k) {
                report.interior_max_abs = std::max(
                    report.interior_max_abs,
                    static_cast<double>(std::abs(problem.phi0.unchecked(i, j, k)))
                );
            }
        }
    }

    const double tol = static_cast<double>(Real{100} * std::numeric_limits<Real>::epsilon());
    report.boundary_ok = report.boundary_error <= tol;
    report.interior_zero_ok = report.interior_max_abs <= tol;
    report.ok =
        report.size_ok &&
        report.h_ok &&
        report.finite_ok &&
        report.boundary_ok &&
        report.interior_zero_ok;

    return report;
}

template ValidationReport validate_problem<float>(const Problem2D<float>&);
template ValidationReport validate_problem<double>(const Problem2D<double>&);
template ValidationReport validate_problem<float>(const Problem3D<float>&);
template ValidationReport validate_problem<double>(const Problem3D<double>&);

} // namespace poisson
