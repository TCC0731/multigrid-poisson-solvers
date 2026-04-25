#include "poisson/mg.hpp"

#include "poisson/metrics.hpp"
#include "poisson/validation.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace poisson {
namespace {

enum class CoarseSolve {
    Exact,
    Sor,
};

Real default_sor_omega(std::size_t interior_n) {
    if (interior_n < 1) {
        throw std::invalid_argument("interior_n must be positive");
    }

    const Real angle = std::numbers::pi_v<Real> / (static_cast<Real>(interior_n) + 1.0);
    return 2.0 / (1.0 + std::sin(angle));
}

void smooth_red_black(
    Grid2D& phi, const Grid2D& rhs, Real h, Real omega, std::size_t steps
) {
    const Real h2 = h * h;
    const std::size_t array_n = phi.size();

    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t color = 0; color < 2; ++color) {
            for (std::size_t i = 1; i + 1 < array_n; ++i) {
                const std::size_t j0 = 1 + ((i + color) & 1);
                for (std::size_t j = j0; j + 1 < array_n; j += 2) {
                    const Real update = 0.25 * (
                        phi(i + 1, j) +
                        phi(i - 1, j) +
                        phi(i, j + 1) +
                        phi(i, j - 1) +
                        h2 * rhs(i, j)
                    );
                    phi(i, j) = (1.0 - omega) * phi(i, j) + omega * update;
                }
            }
        }
    }
}

Grid2D residual_full(const Grid2D& phi, const Grid2D& rhs, Real h) {
    Grid2D result{phi.size()};
    const Real inv_h2 = 1.0 / (h * h);

    for (std::size_t i = 1; i + 1 < phi.size(); ++i) {
        for (std::size_t j = 1; j + 1 < phi.size(); ++j) {
            result(i, j) = rhs(i, j) - (
                4.0 * phi(i, j) -
                phi(i + 1, j) -
                phi(i - 1, j) -
                phi(i, j + 1) -
                phi(i, j - 1)
            ) * inv_h2;
        }
    }

    return result;
}

Grid2D restrict_full_weighting(const Grid2D& fine) {
    const std::size_t n = fine.size() - 2;
    const std::size_t nc = (n - 1) / 2;
    Grid2D coarse{nc + 2};

    for (std::size_t i = 1; i <= nc; ++i) {
        const std::size_t fi = 2 * i;
        for (std::size_t j = 1; j <= nc; ++j) {
            const std::size_t fj = 2 * j;
            coarse(i, j) = (
                4.0 * fine(fi, fj) +
                2.0 * (
                    fine(fi - 1, fj) +
                    fine(fi + 1, fj) +
                    fine(fi, fj - 1) +
                    fine(fi, fj + 1)
                ) +
                fine(fi - 1, fj - 1) +
                fine(fi - 1, fj + 1) +
                fine(fi + 1, fj - 1) +
                fine(fi + 1, fj + 1)
            ) / 16.0;
        }
    }

    return coarse;
}

void prolong_add(const Grid2D& coarse, Grid2D& fine) {
    const std::size_t nc = coarse.size() - 2;

    for (std::size_t i = 1; i <= nc; ++i) {
        const std::size_t fi = 2 * i;
        for (std::size_t j = 1; j <= nc; ++j) {
            fine(fi, j * 2) += coarse(i, j);
        }
    }

    for (std::size_t i = 1; i < nc; ++i) {
        const std::size_t fi = 2 * i + 1;
        for (std::size_t j = 1; j <= nc; ++j) {
            const std::size_t fj = 2 * j;
            fine(fi, fj) += 0.5 * (coarse(i, j) + coarse(i + 1, j));
        }
    }

    for (std::size_t i = 1; i <= nc; ++i) {
        const std::size_t fi = 2 * i;
        for (std::size_t j = 1; j < nc; ++j) {
            const std::size_t fj = 2 * j + 1;
            fine(fi, fj) += 0.5 * (coarse(i, j) + coarse(i, j + 1));
        }
    }

    for (std::size_t i = 1; i < nc; ++i) {
        const std::size_t fi = 2 * i + 1;
        for (std::size_t j = 1; j < nc; ++j) {
            const std::size_t fj = 2 * j + 1;
            fine(fi, fj) += 0.25 * (
                coarse(i, j) +
                coarse(i + 1, j) +
                coarse(i, j + 1) +
                coarse(i + 1, j + 1)
            );
        }
    }
}

void solve_coarsest_exact(Grid2D& phi, const Grid2D& rhs, Real h) {
    const std::size_t n = rhs.size() - 2;
    const std::size_t m = n * n;
    const Real inv_h2 = 1.0 / (h * h);

    std::vector<Real> a(m * m, 0.0);
    std::vector<Real> b(m, 0.0);
    std::vector<Real> x(m, 0.0);

    const auto index = [n](std::size_t i, std::size_t j) {
        return i * n + j;
    };

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const std::size_t row = index(i, j);
            a[row * m + row] = 4.0 * inv_h2;
            if (i > 0) {
                a[row * m + index(i - 1, j)] = -inv_h2;
            }
            if (i + 1 < n) {
                a[row * m + index(i + 1, j)] = -inv_h2;
            }
            if (j > 0) {
                a[row * m + index(i, j - 1)] = -inv_h2;
            }
            if (j + 1 < n) {
                a[row * m + index(i, j + 1)] = -inv_h2;
            }
            b[row] = rhs(i + 1, j + 1);
        }
    }

    for (std::size_t k = 0; k < m; ++k) {
        const Real pivot = a[k * m + k];
        for (std::size_t row = k + 1; row < m; ++row) {
            const Real factor = a[row * m + k] / pivot;
            for (std::size_t col = k; col < m; ++col) {
                a[row * m + col] -= factor * a[k * m + col];
            }
            b[row] -= factor * b[k];
        }
    }

    for (std::size_t row = m; row-- > 0;) {
        Real sum = b[row];
        for (std::size_t col = row + 1; col < m; ++col) {
            sum -= a[row * m + col] * x[col];
        }
        x[row] = sum / a[row * m + row];
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            phi(i + 1, j + 1) = x[index(i, j)];
        }
    }
}

void solve_coarsest_sor(
    Grid2D& phi, const Grid2D& rhs, Real h, Real omega, std::size_t steps
) {
    smooth_red_black(phi, rhs, h, omega, steps);
}

void mg_cycle(
    Grid2D& phi,
    const Grid2D& rhs,
    Real h,
    Real omega,
    std::size_t nu,
    MGCycle cycle,
    CoarseSolve coarse_mode,
    std::size_t coarse_steps
) {
    const std::size_t n = phi.size() - 2;
    if (n <= 4) {
        if (coarse_mode == CoarseSolve::Exact) {
            solve_coarsest_exact(phi, rhs, h);
        } else {
            solve_coarsest_sor(phi, rhs, h, omega, coarse_steps);
        }
        return;
    }

    smooth_red_black(phi, rhs, h, omega, nu);

    Grid2D coarse_rhs = restrict_full_weighting(residual_full(phi, rhs, h));
    Grid2D coarse_err{coarse_rhs.size()};
    mg_cycle(coarse_err, coarse_rhs, 2.0 * h, omega, nu, cycle, coarse_mode, coarse_steps);
    if (cycle == MGCycle::W) {
        mg_cycle(coarse_err, coarse_rhs, 2.0 * h, omega, nu, cycle, coarse_mode, coarse_steps);
    }
    prolong_add(coarse_err, phi);

    smooth_red_black(phi, rhs, h, omega, nu);
}

void validate_mg_inputs(
    const Problem2D& problem, const MGOptions& options, CoarseSolve coarse_mode
) {
    if (options.tol <= 0.0) {
        throw std::invalid_argument("tol must be positive");
    }
    if (options.max_iter < 1) {
        throw std::invalid_argument("max_iter must be positive");
    }
    if (options.nu < 1) {
        throw std::invalid_argument("nu must be positive");
    }
    if (coarse_mode == CoarseSolve::Sor && options.coarse_steps < 1) {
        throw std::invalid_argument("coarse_steps must be positive");
    }
    if (problem.array_n() < 3) {
        throw std::invalid_argument("problem must include at least one interior cell");
    }
    const ValidationReport report = validate_problem(problem);
    if (!report.ok) {
        throw std::invalid_argument("problem is invalid");
    }
}

SolveResult solve_mg_impl(
    const Problem2D& problem, const MGOptions& options, CoarseSolve coarse_mode
) {
    validate_mg_inputs(problem, options, coarse_mode);

    const Real omega = default_sor_omega(problem.interior_n);
    Grid2D phi = problem.phi0;

    for (std::size_t iteration = 1; iteration <= options.max_iter; ++iteration) {
        mg_cycle(
            phi,
            problem.rhs,
            problem.h,
            omega,
            options.nu,
            options.cycle,
            coarse_mode,
            options.coarse_steps
        );

        const Real res = residual_l2(problem, phi);
        if (res <= options.tol) {
            return {std::move(phi), iteration, res};
        }
    }

    return {std::move(phi), options.max_iter, residual_l2(problem, phi)};
}

} // namespace

SolveResult solve_mg_exact(const Problem2D& problem, const MGOptions& options) {
    return solve_mg_impl(problem, options, CoarseSolve::Exact);
}

SolveResult solve_mg_sor(const Problem2D& problem, const MGOptions& options) {
    return solve_mg_impl(problem, options, CoarseSolve::Sor);
}

} // namespace poisson
