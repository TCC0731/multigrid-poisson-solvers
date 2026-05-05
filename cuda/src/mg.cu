#include "poisson/mg.hpp"

#include "poisson/cuda_utils.hpp"
#include "poisson/validation.hpp"

#include <cmath>
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
constexpr Real pi_v = static_cast<Real>(3.14159265358979323846264338327950288L);

template <typename Real>
Real default_sor_omega(std::size_t interior_n) {
    if (interior_n < 1) {
        throw std::invalid_argument("interior_n must be positive");
    }

    const Real angle = pi_v<Real> / (static_cast<Real>(interior_n) + Real{1});
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
void solve_coarsest_exact(
    cuda::DeviceGrid2D<Real>& phi,
    const cuda::DeviceGrid2D<Real>& rhs,
    Real h
) {
    Grid2D<Real> host_phi = phi.download();
    const Grid2D<Real> host_rhs = rhs.download();

    const std::size_t n = host_rhs.size() - 2;
    const std::size_t m = n * n;
    const Real inv_h2 = Real{1} / (h * h);

    std::vector<Real> a(m * m, Real{});
    std::vector<Real> b(m, Real{});
    std::vector<Real> x(m, Real{});

    const auto index = [n](std::size_t i, std::size_t j) {
        return i * n + j;
    };

    // The coarsest MG solve is applied to the error equation, so the coarse
    // correction uses homogeneous Dirichlet boundary conditions.
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
            b[row] = host_rhs.unchecked(i + 1, j + 1);
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
            host_phi.unchecked(i + 1, j + 1) = x[index(i, j)];
        }
    }

    phi.upload(host_phi);
}

template <typename Real>
void mg_cycle(
    cuda::DeviceGrid2D<Real>& phi,
    const cuda::DeviceGrid2D<Real>& rhs,
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
            cuda::run_rb_sor_steps(phi, rhs, h, omega, coarse_steps);
        }
        return;
    }

    cuda::run_rb_sor_steps(phi, rhs, h, omega, nu);

    cuda::DeviceGrid2D<Real> fine_residual{phi.size()};
    cuda::compute_residual_full(phi, rhs, h, fine_residual);

    const std::size_t coarse_interior_n = (n - 1) / 2;
    cuda::DeviceGrid2D<Real> coarse_rhs{coarse_interior_n + 2};
    cuda::restrict_full_weighting(fine_residual, coarse_rhs);

    cuda::DeviceGrid2D<Real> coarse_error{coarse_rhs.size()};
    coarse_error.zero();
    mg_cycle(
        coarse_error,
        coarse_rhs,
        Real{2} * h,
        omega,
        nu,
        cycle,
        coarse_mode,
        coarse_steps
    );
    if (cycle == MGCycle::W) {
        mg_cycle(
            coarse_error,
            coarse_rhs,
            Real{2} * h,
            omega,
            nu,
            cycle,
            coarse_mode,
            coarse_steps
        );
    }

    cuda::prolong_add(coarse_error, phi);
    cuda::run_rb_sor_steps(phi, rhs, h, omega, nu);
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
SolveResult solve_mg_impl(
    const Problem2D<Real>& problem, const MGOptions<Real>& options, CoarseSolve coarse_mode
) {
    validate_mg_inputs(problem, options, coarse_mode);
    cuda::ensure_device_available();

    const Real omega = effective_mg_omega(problem, options);
    cuda::DeviceGrid2D<Real> phi{problem.phi0};
    const cuda::DeviceGrid2D<Real> rhs{problem.rhs};
    cuda::RelativeResidualWorkspace residual_workspace{problem.array_n()};

    for (std::size_t iteration = 1; iteration <= options.max_iter; ++iteration) {
        mg_cycle(
            phi,
            rhs,
            problem.h,
            omega,
            options.nu,
            options.cycle,
            coarse_mode,
            options.coarse_steps
        );

        const double residual =
            cuda::compute_relative_residual_uncached(phi, rhs, problem.h, residual_workspace);
        if (residual <= static_cast<double>(options.tol)) {
            return make_solve_result(phi.download(), iteration, static_cast<Real>(residual));
        }
    }

    const double residual =
        cuda::compute_relative_residual_uncached(phi, rhs, problem.h, residual_workspace);
    return make_solve_result(phi.download(), options.max_iter, static_cast<Real>(residual));
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

template SolveResult solve_mg_exact<float>(const Problem2D<float>&, const MGOptions<float>&);
template SolveResult solve_mg_exact<double>(const Problem2D<double>&, const MGOptions<double>&);
template SolveResult solve_mg_sor<float>(const Problem2D<float>&, const MGOptions<float>&);
template SolveResult solve_mg_sor<double>(const Problem2D<double>&, const MGOptions<double>&);

} // namespace poisson
