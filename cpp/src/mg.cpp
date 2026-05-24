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

template <typename Real>
Real default_sor_omega(std::size_t interior_n) {
    if (interior_n < 1) {
        throw std::invalid_argument("interior_n must be positive");
    }

    const Real angle = std::numbers::pi_v<Real> / (static_cast<Real>(interior_n) + Real{1});
    return Real{2} / (Real{1} + std::sin(angle));
}

template <typename Real>
Real effective_mg_omega(const Problem2D<Real>& problem, const MGOptions<Real>& options) {
    if (options.omega_is_auto) {
        return default_sor_omega<Real>(problem.interior_n);
    }
    return options.omega;
}

template <typename Real>
Real effective_mg_omega(const Problem3D<Real>& problem, const MGOptions<Real>& options) {
    if (options.omega_is_auto) {
        return default_sor_omega<Real>(problem.interior_n);
    }
    return options.omega;
}

template <typename Real>
void smooth_red_black(
    Grid2D<Real>& phi, const Grid2D<Real>& rhs, Real h, Real omega, std::size_t steps
) {
    const Real h2 = h * h;
    const std::size_t array_n = phi.size();

    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t color = 0; color < 2; ++color) {
            for (std::size_t i = 1; i + 1 < array_n; ++i) {
                const std::size_t j0 = 1 + ((i + color) & 1);
                for (std::size_t j = j0; j + 1 < array_n; j += 2) {
                    const Real update = Real{1} / Real{4} * (
                        phi(i + 1, j) +
                        phi(i - 1, j) +
                        phi(i, j + 1) +
                        phi(i, j - 1) +
                        h2 * rhs(i, j)
                    );
                    phi(i, j) = (Real{1} - omega) * phi(i, j) + omega * update;
                }
            }
        }
    }
}

template <typename Real>
Grid2D<Real> residual_full(const Grid2D<Real>& phi, const Grid2D<Real>& rhs, Real h) {
    Grid2D<Real> result{phi.size()};
    const Real inv_h2 = Real{1} / (h * h);

    for (std::size_t i = 1; i + 1 < phi.size(); ++i) {
        for (std::size_t j = 1; j + 1 < phi.size(); ++j) {
            result(i, j) = rhs(i, j) - (
                Real{4} * phi(i, j) -
                phi(i + 1, j) -
                phi(i - 1, j) -
                phi(i, j + 1) -
                phi(i, j - 1)
            ) * inv_h2;
        }
    }

    return result;
}

template <typename Real>
Grid2D<Real> restrict_full_weighting(const Grid2D<Real>& fine) {
    const std::size_t n = fine.size() - 2;
    const std::size_t nc = (n - 1) / 2;
    Grid2D<Real> coarse{nc + 2};

    for (std::size_t i = 1; i <= nc; ++i) {
        const std::size_t fi = 2 * i;
        for (std::size_t j = 1; j <= nc; ++j) {
            const std::size_t fj = 2 * j;
            coarse(i, j) = (
                Real{4} * fine(fi, fj) +
                Real{2} * (
                    fine(fi - 1, fj) +
                    fine(fi + 1, fj) +
                    fine(fi, fj - 1) +
                    fine(fi, fj + 1)
                ) +
                fine(fi - 1, fj - 1) +
                fine(fi - 1, fj + 1) +
                fine(fi + 1, fj - 1) +
                fine(fi + 1, fj + 1)
            ) / Real{16};
        }
    }

    return coarse;
}

template <typename Real>
void prolong_add(const Grid2D<Real>& coarse, Grid2D<Real>& fine) {
    const std::size_t nc = coarse.size() - 2;

    for (std::size_t i = 1; i <= nc; ++i) {
        const std::size_t fi = 2 * i;
        for (std::size_t j = 1; j <= nc; ++j) {
            fine(fi, j * 2) += coarse(i, j);
        }
    }

    // Include boundary-adjacent odd fine points. Those interpolate against the
    // coarse-grid zero boundary and must not be skipped.
    for (std::size_t i = 0; i <= nc; ++i) {
        const std::size_t fi = 2 * i + 1;
        for (std::size_t j = 1; j <= nc; ++j) {
            const std::size_t fj = 2 * j;
            fine(fi, fj) += Real{1} / Real{2} * (coarse(i, j) + coarse(i + 1, j));
        }
    }

    for (std::size_t i = 1; i <= nc; ++i) {
        const std::size_t fi = 2 * i;
        for (std::size_t j = 0; j <= nc; ++j) {
            const std::size_t fj = 2 * j + 1;
            fine(fi, fj) += Real{1} / Real{2} * (coarse(i, j) + coarse(i, j + 1));
        }
    }

    for (std::size_t i = 0; i <= nc; ++i) {
        const std::size_t fi = 2 * i + 1;
        for (std::size_t j = 0; j <= nc; ++j) {
            const std::size_t fj = 2 * j + 1;
            fine(fi, fj) += Real{1} / Real{4} * (
                coarse(i, j) +
                coarse(i + 1, j) +
                coarse(i, j + 1) +
                coarse(i + 1, j + 1)
            );
        }
    }
}

template <typename Real>
void solve_coarsest_exact(Grid2D<Real>& phi, const Grid2D<Real>& rhs, Real h) {
    const std::size_t n = rhs.size() - 2;
    const std::size_t m = n * n;
    const Real inv_h2 = Real{1} / (h * h);

    std::vector<Real> a(m * m, Real{});
    std::vector<Real> b(m, Real{});
    std::vector<Real> x(m, Real{});

    const auto index = [n](std::size_t i, std::size_t j) {
        return i * n + j;
    };

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const std::size_t row = index(i, j);
            a[row * m + row] = Real{4} * inv_h2;
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

template <typename Real>
void solve_coarsest_sor(
    Grid2D<Real>& phi, const Grid2D<Real>& rhs, Real h, Real omega, std::size_t steps
) {
    smooth_red_black(phi, rhs, h, omega, steps);
}

template <typename Real>
void smooth_red_black_3d(
    Grid3D<Real>& phi, const Grid3D<Real>& rhs, Real h, Real omega, std::size_t steps
) {
    const Real h2 = h * h;
    const std::size_t array_n = phi.size();

    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t color = 0; color < 2; ++color) {
            for (std::size_t i = 1; i + 1 < array_n; ++i) {
                for (std::size_t j = 1; j + 1 < array_n; ++j) {
                    const std::size_t k0 = 1 + ((i + j + color) & 1);
                    for (std::size_t k = k0; k + 1 < array_n; k += 2) {
                        const Real update = Real{1} / Real{6} * (
                            phi(i + 1, j, k) +
                            phi(i - 1, j, k) +
                            phi(i, j + 1, k) +
                            phi(i, j - 1, k) +
                            phi(i, j, k + 1) +
                            phi(i, j, k - 1) +
                            h2 * rhs(i, j, k)
                        );
                        phi(i, j, k) = (Real{1} - omega) * phi(i, j, k) + omega * update;
                    }
                }
            }
        }
    }
}

template <typename Real>
Grid3D<Real> residual_full_3d(const Grid3D<Real>& phi, const Grid3D<Real>& rhs, Real h) {
    Grid3D<Real> result{phi.size()};
    const Real inv_h2 = Real{1} / (h * h);

    for (std::size_t i = 1; i + 1 < phi.size(); ++i) {
        for (std::size_t j = 1; j + 1 < phi.size(); ++j) {
            for (std::size_t k = 1; k + 1 < phi.size(); ++k) {
                result(i, j, k) = rhs(i, j, k) - (
                    Real{6} * phi(i, j, k) -
                    phi(i + 1, j, k) -
                    phi(i - 1, j, k) -
                    phi(i, j + 1, k) -
                    phi(i, j - 1, k) -
                    phi(i, j, k + 1) -
                    phi(i, j, k - 1)
                ) * inv_h2;
            }
        }
    }

    return result;
}

template <typename Real>
Real weight_3(int offset) {
    return offset == 0 ? Real{2} : Real{1};
}

template <typename Real>
Grid3D<Real> restrict_full_weighting_3d(const Grid3D<Real>& fine) {
    const std::size_t n = fine.size() - 2;
    const std::size_t nc = (n - 1) / 2;
    Grid3D<Real> coarse{nc + 2};

    for (std::size_t i = 1; i <= nc; ++i) {
        const std::size_t fi = 2 * i;
        for (std::size_t j = 1; j <= nc; ++j) {
            const std::size_t fj = 2 * j;
            for (std::size_t k = 1; k <= nc; ++k) {
                const std::size_t fk = 2 * k;
                Real total{};
                for (int di = -1; di <= 1; ++di) {
                    const Real wi = weight_3<Real>(di);
                    for (int dj = -1; dj <= 1; ++dj) {
                        const Real wij = wi * weight_3<Real>(dj);
                        for (int dk = -1; dk <= 1; ++dk) {
                            total += wij
                                * weight_3<Real>(dk)
                                * fine(
                                    static_cast<std::size_t>(static_cast<int>(fi) + di),
                                    static_cast<std::size_t>(static_cast<int>(fj) + dj),
                                    static_cast<std::size_t>(static_cast<int>(fk) + dk)
                                );
                        }
                    }
                }
                coarse(i, j, k) = total / Real{64};
            }
        }
    }

    return coarse;
}

template <typename Real>
void prolong_add_3d(const Grid3D<Real>& coarse, Grid3D<Real>& fine) {
    const std::size_t nc = coarse.size() - 2;
    const std::size_t nf = 2 * nc + 1;

    for (std::size_t fi = 1; fi <= nf; ++fi) {
        const bool i_even = (fi & 1) == 0;
        const std::size_t i0 = fi / 2;
        const std::size_t i1 = i_even ? i0 : i0 + 1;
        const Real wi0 = i_even ? Real{1} : Real{1} / Real{2};
        const Real wi1 = i_even ? Real{} : Real{1} / Real{2};

        for (std::size_t fj = 1; fj <= nf; ++fj) {
            const bool j_even = (fj & 1) == 0;
            const std::size_t j0 = fj / 2;
            const std::size_t j1 = j_even ? j0 : j0 + 1;
            const Real wj0 = j_even ? Real{1} : Real{1} / Real{2};
            const Real wj1 = j_even ? Real{} : Real{1} / Real{2};

            for (std::size_t fk = 1; fk <= nf; ++fk) {
                const bool k_even = (fk & 1) == 0;
                const std::size_t k0 = fk / 2;
                const std::size_t k1 = k_even ? k0 : k0 + 1;
                const Real wk0 = k_even ? Real{1} : Real{1} / Real{2};
                const Real wk1 = k_even ? Real{} : Real{1} / Real{2};

                fine(fi, fj, fk) +=
                    wi0 * wj0 * wk0 * coarse(i0, j0, k0) +
                    wi1 * wj0 * wk0 * coarse(i1, j0, k0) +
                    wi0 * wj1 * wk0 * coarse(i0, j1, k0) +
                    wi1 * wj1 * wk0 * coarse(i1, j1, k0) +
                    wi0 * wj0 * wk1 * coarse(i0, j0, k1) +
                    wi1 * wj0 * wk1 * coarse(i1, j0, k1) +
                    wi0 * wj1 * wk1 * coarse(i0, j1, k1) +
                    wi1 * wj1 * wk1 * coarse(i1, j1, k1);
            }
        }
    }
}

template <typename Real>
void solve_coarsest_exact_3d(Grid3D<Real>& phi, const Grid3D<Real>& rhs, Real h) {
    const std::size_t n = rhs.size() - 2;
    const std::size_t m = n * n * n;
    const Real inv_h2 = Real{1} / (h * h);

    std::vector<Real> a(m * m, Real{});
    std::vector<Real> b(m, Real{});
    std::vector<Real> x(m, Real{});

    const auto index = [n](std::size_t i, std::size_t j, std::size_t k) {
        return (i * n + j) * n + k;
    };

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                const std::size_t row = index(i, j, k);
                a[row * m + row] = Real{6} * inv_h2;
                if (i > 0) {
                    a[row * m + index(i - 1, j, k)] = -inv_h2;
                }
                if (i + 1 < n) {
                    a[row * m + index(i + 1, j, k)] = -inv_h2;
                }
                if (j > 0) {
                    a[row * m + index(i, j - 1, k)] = -inv_h2;
                }
                if (j + 1 < n) {
                    a[row * m + index(i, j + 1, k)] = -inv_h2;
                }
                if (k > 0) {
                    a[row * m + index(i, j, k - 1)] = -inv_h2;
                }
                if (k + 1 < n) {
                    a[row * m + index(i, j, k + 1)] = -inv_h2;
                }
                b[row] = rhs(i + 1, j + 1, k + 1);
            }
        }
    }

    for (std::size_t pivot_col = 0; pivot_col < m; ++pivot_col) {
        const Real pivot = a[pivot_col * m + pivot_col];
        for (std::size_t row = pivot_col + 1; row < m; ++row) {
            const Real factor = a[row * m + pivot_col] / pivot;
            for (std::size_t col = pivot_col; col < m; ++col) {
                a[row * m + col] -= factor * a[pivot_col * m + col];
            }
            b[row] -= factor * b[pivot_col];
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
            for (std::size_t k = 0; k < n; ++k) {
                phi(i + 1, j + 1, k + 1) = x[index(i, j, k)];
            }
        }
    }
}

template <typename Real>
void solve_coarsest_sor_3d(
    Grid3D<Real>& phi, const Grid3D<Real>& rhs, Real h, Real omega, std::size_t steps
) {
    smooth_red_black_3d(phi, rhs, h, omega, steps);
}

template <typename Real>
void mg_cycle(
    Grid2D<Real>& phi,
    const Grid2D<Real>& rhs,
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

    Grid2D<Real> coarse_rhs = restrict_full_weighting(residual_full(phi, rhs, h));
    Grid2D<Real> coarse_err{coarse_rhs.size()};
    const std::size_t coarse_n = coarse_rhs.size() - 2;
    mg_cycle(coarse_err, coarse_rhs, Real{2} * h, omega, nu, cycle, coarse_mode, coarse_steps);
    if (cycle == MGCycle::W && coarse_n > 4) {
        mg_cycle(coarse_err, coarse_rhs, Real{2} * h, omega, nu, cycle, coarse_mode, coarse_steps);
    }
    prolong_add(coarse_err, phi);

    smooth_red_black(phi, rhs, h, omega, nu);
}

template <typename Real>
void mg_cycle_3d(
    Grid3D<Real>& phi,
    const Grid3D<Real>& rhs,
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
            solve_coarsest_exact_3d(phi, rhs, h);
        } else {
            solve_coarsest_sor_3d(phi, rhs, h, omega, coarse_steps);
        }
        return;
    }

    smooth_red_black_3d(phi, rhs, h, omega, nu);

    Grid3D<Real> coarse_rhs = restrict_full_weighting_3d(residual_full_3d(phi, rhs, h));
    Grid3D<Real> coarse_err{coarse_rhs.size()};
    const std::size_t coarse_n = coarse_rhs.size() - 2;
    mg_cycle_3d(coarse_err, coarse_rhs, Real{2} * h, omega, nu, cycle, coarse_mode, coarse_steps);
    if (cycle == MGCycle::W && coarse_n > 4) {
        mg_cycle_3d(coarse_err, coarse_rhs, Real{2} * h, omega, nu, cycle, coarse_mode, coarse_steps);
    }
    prolong_add_3d(coarse_err, phi);

    smooth_red_black_3d(phi, rhs, h, omega, nu);
}

template <typename Real>
void validate_mg_inputs(
    const Problem2D<Real>& problem, const MGOptions<Real>& options, CoarseSolve coarse_mode
) {
    if (options.tol <= Real{}) {
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
    if (!options.omega_is_auto && (options.omega <= Real{} || options.omega > Real{2})) {
        throw std::invalid_argument("omega must be in (0, 2]");
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
void validate_mg_inputs(
    const Problem3D<Real>& problem, const MGOptions<Real>& options, CoarseSolve coarse_mode
) {
    if (options.tol <= Real{}) {
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
    if (!options.omega_is_auto && (options.omega <= Real{} || options.omega > Real{2})) {
        throw std::invalid_argument("omega must be in (0, 2]");
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
SolveResult solve_mg_impl(
    const Problem2D<Real>& problem, const MGOptions<Real>& options, CoarseSolve coarse_mode
) {
    validate_mg_inputs(problem, options, coarse_mode);

    const Real omega = effective_mg_omega(problem, options);
    Grid2D<Real> phi = problem.phi0;

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

template <typename Real>
SolveResult3D solve_mg_3d_impl(
    const Problem3D<Real>& problem, const MGOptions<Real>& options, CoarseSolve coarse_mode
) {
    validate_mg_inputs(problem, options, coarse_mode);

    const Real omega = effective_mg_omega(problem, options);
    Grid3D<Real> phi = problem.phi0;

    for (std::size_t iteration = 1; iteration <= options.max_iter; ++iteration) {
        mg_cycle_3d(
            phi,
            problem.rhs,
            problem.h,
            omega,
            options.nu,
            options.cycle,
            coarse_mode,
            options.coarse_steps
        );

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
SolveResult solve_mg_exact(const Problem2D<Real>& problem, const MGOptions<Real>& options) {
    return solve_mg_impl(problem, options, CoarseSolve::Exact);
}

template <typename Real>
SolveResult solve_mg_sor(const Problem2D<Real>& problem, const MGOptions<Real>& options) {
    return solve_mg_impl(problem, options, CoarseSolve::Sor);
}

template <typename Real>
SolveResult3D solve_mg_exact(const Problem3D<Real>& problem, const MGOptions<Real>& options) {
    return solve_mg_3d_impl(problem, options, CoarseSolve::Exact);
}

template <typename Real>
SolveResult3D solve_mg_sor(const Problem3D<Real>& problem, const MGOptions<Real>& options) {
    return solve_mg_3d_impl(problem, options, CoarseSolve::Sor);
}

template SolveResult solve_mg_exact<float>(const Problem2D<float>&, const MGOptions<float>&);
template SolveResult solve_mg_exact<double>(const Problem2D<double>&, const MGOptions<double>&);
template SolveResult solve_mg_sor<float>(const Problem2D<float>&, const MGOptions<float>&);
template SolveResult solve_mg_sor<double>(const Problem2D<double>&, const MGOptions<double>&);
template SolveResult3D solve_mg_exact<float>(const Problem3D<float>&, const MGOptions<float>&);
template SolveResult3D solve_mg_exact<double>(const Problem3D<double>&, const MGOptions<double>&);
template SolveResult3D solve_mg_sor<float>(const Problem3D<float>&, const MGOptions<float>&);
template SolveResult3D solve_mg_sor<double>(const Problem3D<double>&, const MGOptions<double>&);

} // namespace poisson
