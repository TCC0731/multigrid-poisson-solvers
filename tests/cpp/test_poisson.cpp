#include "poisson/gs.hpp"
#include "poisson/jacobi.hpp"
#include "poisson/mg.hpp"
#include "poisson/metrics.hpp"
#include "poisson/operators.hpp"
#include "poisson/problem.hpp"
#include "poisson/sor.hpp"
#include "poisson/validation.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace {

using namespace poisson;

template <typename Real>
using GridT = poisson::Grid2D<Real>;

template <typename Real>
using ProblemT = poisson::Problem2D<Real>;

template <typename Real>
using SolveOptionsT = poisson::SolveOptions<Real>;

template <typename Real>
using MGOptionsT = poisson::MGOptions<Real>;

constexpr std::array<Case2D, 5> kCases{
    Case2D::Sine,
    Case2D::MixedSine,
    Case2D::Bubble,
    Case2D::Exp,
    Case2D::Cosine,
};
constexpr std::array<MGCycle, 2> kCycles{MGCycle::V, MGCycle::W};

template <typename Real>
constexpr Real zero_tol() {
    if constexpr (std::is_same_v<Real, float>) {
        return Real{1e-6};
    }
    return Real{1e-12};
}

template <typename Real>
constexpr Real convergence_tol() {
    if constexpr (std::is_same_v<Real, float>) {
        return Real{5e-2};
    }
    return Real{1e-10};
}

template <typename Real>
constexpr Real interface_tol() {
    if constexpr (std::is_same_v<Real, float>) {
        return Real{5e-2};
    }
    return Real{1e-8};
}

template <typename Real>
constexpr double residual_compare_tol() {
    if constexpr (std::is_same_v<Real, float>) {
        return 1e-5;
    }
    return 1e-12;
}

template <typename Real>
SolveOptionsT<Real> make_solve_options(std::size_t max_iter = 20'000) {
    return SolveOptionsT<Real>{convergence_tol<Real>(), max_iter};
}

template <typename Real>
SolveOptionsT<Real> make_interface_solve_options(std::size_t max_iter = 5'000) {
    return SolveOptionsT<Real>{interface_tol<Real>(), max_iter};
}

template <typename Real>
MGOptionsT<Real> make_mg_options(
    MGCycle cycle,
    std::size_t coarse_steps = 16,
    std::size_t max_iter = 100
) {
    return MGOptionsT<Real>{convergence_tol<Real>(), max_iter, 2, cycle, coarse_steps};
}

template <typename Real>
bool grid_all_close_to_zero(const GridT<Real>& grid, Real tol = zero_tol<Real>()) {
    for (std::size_t i = 0; i < grid.size(); ++i) {
        for (std::size_t j = 0; j < grid.size(); ++j) {
            if (std::abs(grid(i, j)) > tol) {
                return false;
            }
        }
    }
    return true;
}

template <typename Real>
bool boundary_has_nonzero(const GridT<Real>& grid, Real tol = zero_tol<Real>()) {
    const std::size_t n = grid.size();

    for (std::size_t i = 0; i < n; ++i) {
        if (std::abs(grid(i, 0)) > tol || std::abs(grid(i, n - 1)) > tol) {
            return true;
        }
    }

    for (std::size_t j = 0; j < n; ++j) {
        if (std::abs(grid(0, j)) > tol || std::abs(grid(n - 1, j)) > tol) {
            return true;
        }
    }

    return false;
}

template <typename Real>
void expect_boundaries_equal(const GridT<Real>& lhs, const GridT<Real>& rhs) {
    ASSERT_EQ(lhs.size(), rhs.size());

    const std::size_t n = lhs.size();
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(lhs(i, 0)), static_cast<double>(rhs(i, 0)));
        EXPECT_DOUBLE_EQ(static_cast<double>(lhs(i, n - 1)), static_cast<double>(rhs(i, n - 1)));
    }
    for (std::size_t j = 0; j < n; ++j) {
        EXPECT_DOUBLE_EQ(static_cast<double>(lhs(0, j)), static_cast<double>(rhs(0, j)));
        EXPECT_DOUBLE_EQ(static_cast<double>(lhs(n - 1, j)), static_cast<double>(rhs(n - 1, j)));
    }
}

template <typename Real, typename SolveFn, typename Options>
SolveResult expect_solver_smoke(SolveFn&& solve, const ProblemT<Real>& problem, const Options& options) {
    const SolveResult result = solve(problem, options);
    const GridT<Real> phi = cast_grid<Real>(result.phi);
    const Real recomputed_residual = residual_l2(problem, phi);
    const ErrorMetrics metrics = error_metrics(problem, phi);

    EXPECT_EQ(result.phi.size(), problem.phi0.size());
    EXPECT_GT(result.iterations, 0u);
    EXPECT_LE(result.iterations, options.max_iter);
    EXPECT_TRUE(std::isfinite(result.residual_l2));
    EXPECT_GE(result.residual_l2, 0.0);
    EXPECT_LE(result.residual_l2, static_cast<double>(options.tol));
    EXPECT_NEAR(result.residual_l2, static_cast<double>(recomputed_residual), residual_compare_tol<Real>());
    expect_boundaries_equal(phi, problem.phi0);

    EXPECT_TRUE(std::isfinite(metrics.error_l2));
    EXPECT_TRUE(std::isfinite(metrics.error_linf));
    EXPECT_GE(metrics.error_l2, 0.0);
    EXPECT_GE(metrics.error_linf, 0.0);

    return result;
}

template <typename Real, typename SolveFn, typename Options>
void expect_solver_converges(SolveFn&& solve, const ProblemT<Real>& problem, const Options& options) {
    const SolveResult result = expect_solver_smoke(solve, problem, options);
    const GridT<Real> phi = cast_grid<Real>(result.phi);
    const ErrorMetrics metrics = error_metrics(problem, phi);

    EXPECT_LT(metrics.error_l2, 1e-2);
    EXPECT_LT(metrics.error_linf, 1e-2);
}

std::string case_name(const ::testing::TestParamInfo<Case2D>& info) {
    return std::string(poisson::to_string(info.param));
}

std::string cycle_name(const ::testing::TestParamInfo<MGCycle>& info) {
    return info.param == MGCycle::V ? "V" : "W";
}

using MgCaseParam = std::tuple<Case2D, MGCycle>;

std::string mg_case_name(const ::testing::TestParamInfo<MgCaseParam>& info) {
    const auto& [case_id, cycle] = info.param;
    return std::string(poisson::to_string(case_id)) + "_" + (cycle == MGCycle::V ? "V" : "W");
}

template <typename Real>
void run_grid2d_construction_fill_and_bounds() {
    GridT<Real> empty;
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.size(), 0u);
    EXPECT_EQ(empty.elements(), 0u);

    GridT<Real> grid{4, Real{1.5}};
    EXPECT_FALSE(grid.empty());
    EXPECT_EQ(grid.size(), 4u);
    EXPECT_EQ(grid.elements(), 16u);
    EXPECT_DOUBLE_EQ(static_cast<double>(grid(2, 3)), 1.5);

    grid.fill(Real{2});
    EXPECT_DOUBLE_EQ(static_cast<double>(grid(0, 0)), 2.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(grid(3, 3)), 2.0);
    EXPECT_THROW((void)grid(4, 0), std::out_of_range);
}

template <typename Real>
void run_apply_a_and_residual_stencil() {
    GridT<Real> phi{5};
    phi.fill(Real{});
    phi(2, 2) = Real{1};

    const GridT<Real> aphi = apply_A(phi, Real{1});
    ASSERT_EQ(aphi.size(), 3u);

    constexpr Real expected[3][3] = {
        {Real{0}, Real{-1}, Real{0}},
        {Real{-1}, Real{4}, Real{-1}},
        {Real{0}, Real{-1}, Real{0}},
    };

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(static_cast<double>(aphi(i, j)), static_cast<double>(expected[i][j]));
        }
    }

    const GridT<Real> zero_rhs{5};
    const GridT<Real> r0 = residual(phi, zero_rhs, Real{1});
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(static_cast<double>(r0(i, j)), -static_cast<double>(expected[i][j]));
        }
    }

    GridT<Real> rhs{5};
    rhs.fill(Real{});
    for (std::size_t i = 1; i + 1 < rhs.size(); ++i) {
        for (std::size_t j = 1; j + 1 < rhs.size(); ++j) {
            rhs(i, j) = aphi(i - 1, j - 1);
        }
    }

    const GridT<Real> r1 = residual(phi, rhs, Real{1});
    for (std::size_t i = 0; i < r1.size(); ++i) {
        for (std::size_t j = 0; j < r1.size(); ++j) {
            EXPECT_DOUBLE_EQ(static_cast<double>(r1(i, j)), 0.0);
        }
    }
}

template <typename Real>
void run_metrics_exact_discrete_problem_has_zero_metrics() {
    GridT<Real> exact{5};
    exact.fill(Real{});
    exact(2, 2) = Real{1};

    GridT<Real> rhs{5};
    rhs.fill(Real{});
    const GridT<Real> aphi = apply_A(exact, Real{1});
    for (std::size_t i = 1; i + 1 < rhs.size(); ++i) {
        for (std::size_t j = 1; j + 1 < rhs.size(); ++j) {
            rhs(i, j) = aphi(i - 1, j - 1);
        }
    }

    const ProblemT<Real> problem{
        Case2D::Sine,
        3,
        Real{1},
        exact,
        rhs,
        exact,
    };

    EXPECT_DOUBLE_EQ(static_cast<double>(residual_l2(problem, exact)), 0.0);

    const ErrorMetrics metrics = error_metrics(problem, exact);
    EXPECT_DOUBLE_EQ(metrics.error_l2, 0.0);
    EXPECT_DOUBLE_EQ(metrics.error_linf, 0.0);
}

template <typename Real>
void run_manufactured_problem_shape_and_boundary(Case2D case_id) {
    const ProblemT<Real> problem = make_problem<Real>(case_id, 7);
    const ValidationReport report = validate_problem(problem);

    EXPECT_EQ(problem.case_id, case_id);
    EXPECT_EQ(problem.interior_n, 7u);
    EXPECT_DOUBLE_EQ(static_cast<double>(problem.h), 1.0 / 8.0);
    EXPECT_EQ(problem.array_n(), 9u);
    EXPECT_EQ(problem.exact.size(), 9u);
    EXPECT_EQ(problem.rhs.size(), 9u);
    EXPECT_EQ(problem.phi0.size(), 9u);
    EXPECT_TRUE(report.ok);
    EXPECT_TRUE(report.size_ok);
    EXPECT_TRUE(report.h_ok);
    EXPECT_TRUE(report.finite_ok);
    EXPECT_TRUE(report.boundary_ok);
    EXPECT_TRUE(report.interior_zero_ok);
    EXPECT_DOUBLE_EQ(report.boundary_error, 0.0);
    EXPECT_DOUBLE_EQ(report.interior_max_abs, 0.0);

    for (std::size_t i = 0; i < problem.array_n(); ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(problem.phi0(i, 0)), static_cast<double>(problem.exact(i, 0)));
        EXPECT_DOUBLE_EQ(
            static_cast<double>(problem.phi0(i, problem.array_n() - 1)),
            static_cast<double>(problem.exact(i, problem.array_n() - 1))
        );
    }
    for (std::size_t j = 0; j < problem.array_n(); ++j) {
        EXPECT_DOUBLE_EQ(static_cast<double>(problem.phi0(0, j)), static_cast<double>(problem.exact(0, j)));
        EXPECT_DOUBLE_EQ(
            static_cast<double>(problem.phi0(problem.array_n() - 1, j)),
            static_cast<double>(problem.exact(problem.array_n() - 1, j))
        );
    }

    for (std::size_t i = 1; i + 1 < problem.array_n(); ++i) {
        for (std::size_t j = 1; j + 1 < problem.array_n(); ++j) {
            EXPECT_DOUBLE_EQ(static_cast<double>(problem.phi0(i, j)), 0.0);
        }
    }

    if (case_id == Case2D::Sine || case_id == Case2D::MixedSine || case_id == Case2D::Bubble) {
        EXPECT_TRUE(grid_all_close_to_zero(problem.phi0));
    } else {
        EXPECT_TRUE(boundary_has_nonzero(problem.phi0));
        EXPECT_FALSE(grid_all_close_to_zero(problem.phi0));
    }
}

template <typename Real>
void run_validation_detects_tampered_problem_fields() {
    ProblemT<Real> problem = make_problem<Real>(Case2D::Sine, 7);

    problem.phi0(2, 2) = Real{1};
    ValidationReport report = validate_problem(problem);
    EXPECT_FALSE(report.ok);
    EXPECT_FALSE(report.interior_zero_ok);
    EXPECT_GT(report.interior_max_abs, 0.0);

    problem = make_problem<Real>(Case2D::Sine, 7);
    problem.phi0(0, 0) = problem.phi0(0, 0) + Real{1};
    report = validate_problem(problem);
    EXPECT_FALSE(report.ok);
    EXPECT_FALSE(report.boundary_ok);
    EXPECT_GT(report.boundary_error, 0.0);
}

template <typename Real>
void run_problem_error_rejects_invalid_construction_arguments() {
    EXPECT_THROW(static_cast<void>(make_problem<Real>("unknown_case", 7)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(make_problem<Real>(Case2D::Sine, 0)), std::invalid_argument);
}

template <typename Real>
void run_parameter_error_grid_access_rejects_out_of_range() {
    GridT<Real> grid{4};
    EXPECT_THROW((void)grid(4, 0), std::out_of_range);
    EXPECT_THROW((void)grid(0, 4), std::out_of_range);
}

template <typename Real>
void run_jacobi_solver_interface() {
    const ProblemT<Real> problem = make_problem<Real>(Case2D::Sine, 7);
    const SolveOptionsT<Real> options = make_interface_solve_options<Real>();

    const SolveResult direct = expect_solver_smoke(
        [](const ProblemT<Real>& p, const SolveOptionsT<Real>& o) {
            return solve_jacobi(p, o);
        },
        problem,
        options
    );

    if constexpr (std::is_same_v<Real, double>) {
        const JacobiSolver2D solver{};
        const SolveResult via_interface = solver.solve(problem, options);

        EXPECT_EQ(solver.name(), "jacobi");
        EXPECT_EQ(direct.iterations, via_interface.iterations);
        EXPECT_DOUBLE_EQ(direct.residual_l2, via_interface.residual_l2);
        ASSERT_EQ(direct.phi.size(), via_interface.phi.size());
        for (std::size_t i = 0; i < direct.phi.size(); ++i) {
            for (std::size_t j = 0; j < direct.phi.size(); ++j) {
                EXPECT_DOUBLE_EQ(direct.phi(i, j), via_interface.phi(i, j));
            }
        }
    }
}

template <typename Real>
void run_jacobi_solver_converges(Case2D case_id) {
    const ProblemT<Real> problem = make_problem<Real>(case_id, 32);
    const SolveOptionsT<Real> options = make_solve_options<Real>();
    expect_solver_converges(
        [](const ProblemT<Real>& p, const SolveOptionsT<Real>& o) {
            return solve_jacobi(p, o);
        },
        problem,
        options
    );
}

template <typename Real>
void run_jacobi_solver_regression() {
    const ProblemT<Real> problem = make_problem<Real>(Case2D::Sine, 31);
    const SolveOptionsT<Real> options = make_solve_options<Real>();

    const SolveResult result = expect_solver_smoke(
        [](const ProblemT<Real>& p, const SolveOptionsT<Real>& o) {
            return solve_jacobi(p, o);
        },
        problem,
        options
    );

    const GridT<Real> phi = cast_grid<Real>(result.phi);
    const ErrorMetrics metrics = error_metrics(problem, phi);

    if constexpr (std::is_same_v<Real, float>) {
        EXPECT_EQ(result.iterations, 1'095u);
        EXPECT_NEAR(result.residual_l2, 4.999456e-02, 1e-6);
        EXPECT_NEAR(metrics.error_l2, 2.132575e-03, 1e-8);
        EXPECT_NEAR(metrics.error_linf, 4.265308e-03, 1e-8);
    } else {
        EXPECT_EQ(result.iterations, 5'245u);
        EXPECT_NEAR(result.residual_l2, 9.981977e-11, 1e-15);
        EXPECT_NEAR(metrics.error_l2, 4.017888e-04, 5e-11);
        EXPECT_NEAR(metrics.error_linf, 8.035777e-04, 5e-11);
    }

    EXPECT_NEAR(
        result.residual_l2,
        static_cast<double>(residual_l2(problem, phi)),
        residual_compare_tol<Real>()
    );
}

template <typename Real>
void run_gs_solver_interface() {
    const ProblemT<Real> problem = make_problem<Real>(Case2D::Sine, 7);
    const SolveOptionsT<Real> options = make_interface_solve_options<Real>();

    const SolveResult direct = expect_solver_smoke(
        [](const ProblemT<Real>& p, const SolveOptionsT<Real>& o) {
            return solve_gs(p, o);
        },
        problem,
        options
    );

    if constexpr (std::is_same_v<Real, double>) {
        const GaussSeidelSolver2D solver{};
        const SolveResult via_interface = solver.solve(problem, options);

        EXPECT_EQ(solver.name(), "gs");
        EXPECT_EQ(direct.iterations, via_interface.iterations);
        EXPECT_DOUBLE_EQ(direct.residual_l2, via_interface.residual_l2);
        ASSERT_EQ(direct.phi.size(), via_interface.phi.size());
        for (std::size_t i = 0; i < direct.phi.size(); ++i) {
            for (std::size_t j = 0; j < direct.phi.size(); ++j) {
                EXPECT_DOUBLE_EQ(direct.phi(i, j), via_interface.phi(i, j));
            }
        }
    }
}

template <typename Real>
void run_gs_solver_converges(Case2D case_id) {
    const ProblemT<Real> problem = make_problem<Real>(case_id, 32);
    const SolveOptionsT<Real> options = make_solve_options<Real>();
    expect_solver_converges(
        [](const ProblemT<Real>& p, const SolveOptionsT<Real>& o) {
            return solve_gs(p, o);
        },
        problem,
        options
    );
}

template <typename Real>
void run_sor_solver_interface() {
    const ProblemT<Real> problem = make_problem<Real>(Case2D::Sine, 7);
    const SolveOptionsT<Real> options = make_interface_solve_options<Real>();

    const SolveResult direct = expect_solver_smoke(
        [](const ProblemT<Real>& p, const SolveOptionsT<Real>& o) {
            return solve_sor(p, o);
        },
        problem,
        options
    );

    if constexpr (std::is_same_v<Real, double>) {
        const SorSolver2D solver{};
        const SolveResult via_interface = solver.solve(problem, options);

        EXPECT_EQ(solver.name(), "sor");
        EXPECT_EQ(direct.iterations, via_interface.iterations);
        EXPECT_DOUBLE_EQ(direct.residual_l2, via_interface.residual_l2);
        ASSERT_EQ(direct.phi.size(), via_interface.phi.size());
        for (std::size_t i = 0; i < direct.phi.size(); ++i) {
            for (std::size_t j = 0; j < direct.phi.size(); ++j) {
                EXPECT_DOUBLE_EQ(direct.phi(i, j), via_interface.phi(i, j));
            }
        }
    }
}

template <typename Real>
void run_sor_solver_converges(Case2D case_id) {
    const ProblemT<Real> problem = make_problem<Real>(case_id, 32);
    const SolveOptionsT<Real> options = make_solve_options<Real>();
    expect_solver_converges(
        [](const ProblemT<Real>& p, const SolveOptionsT<Real>& o) {
            return solve_sor(p, o);
        },
        problem,
        options
    );
}

template <typename Real>
void run_mg_exact_converges(const MgCaseParam& param) {
    const auto& [case_id, cycle] = param;
    const ProblemT<Real> problem = make_problem<Real>(case_id, 32);
    const MGOptionsT<Real> options = make_mg_options<Real>(cycle);
    expect_solver_converges(
        [](const ProblemT<Real>& p, const MGOptionsT<Real>& o) {
            return solve_mg_exact(p, o);
        },
        problem,
        options
    );
}

template <typename Real>
void run_mg_sor_converges(const MgCaseParam& param) {
    const auto& [case_id, cycle] = param;
    const ProblemT<Real> problem = make_problem<Real>(case_id, 32);
    const MGOptionsT<Real> options = make_mg_options<Real>(cycle, 32, 150);
    expect_solver_converges(
        [](const ProblemT<Real>& p, const MGOptionsT<Real>& o) {
            return solve_mg_sor(p, o);
        },
        problem,
        options
    );
}

template <typename Real>
void run_mg_regression(MGCycle cycle) {
    const ProblemT<Real> problem = make_problem<Real>(Case2D::Sine, 64);
    const MGOptionsT<Real> options = make_mg_options<Real>(cycle, 16, 100);

    const SolveResult result = expect_solver_smoke(
        [](const ProblemT<Real>& p, const MGOptionsT<Real>& o) {
            return solve_mg_exact(p, o);
        },
        problem,
        options
    );

    const GridT<Real> phi = cast_grid<Real>(result.phi);
    const ErrorMetrics metrics = error_metrics(problem, phi);

    if constexpr (std::is_same_v<Real, float>) {
        if (cycle == MGCycle::V) {
            EXPECT_EQ(result.iterations, 26u);
            EXPECT_NEAR(result.residual_l2, 4.222215e-02, 1e-6);
            EXPECT_NEAR(metrics.error_l2, 6.726872e-05, 1e-8);
            EXPECT_NEAR(metrics.error_linf, 1.611114e-04, 1e-8);
        } else {
            EXPECT_EQ(result.iterations, 19u);
            EXPECT_NEAR(result.residual_l2, 3.223472e-02, 1e-6);
            EXPECT_NEAR(metrics.error_l2, 9.712945e-05, 1e-8);
            EXPECT_NEAR(metrics.error_linf, 2.052188e-04, 1e-8);
        }
    } else {
        if (cycle == MGCycle::V) {
            EXPECT_EQ(result.iterations, 83u);
            EXPECT_NEAR(result.residual_l2, 8.472344985850555e-11, 1e-12);
            EXPECT_NEAR(metrics.error_l2, 9.734474635149315e-05, 5e-11);
            EXPECT_NEAR(metrics.error_linf, 1.945758162948952e-04, 5e-11);
        } else {
            EXPECT_EQ(result.iterations, 68u);
            EXPECT_NEAR(result.residual_l2, 9.117932594196834e-11, 1e-12);
            EXPECT_NEAR(metrics.error_l2, 9.734474633251941e-05, 5e-11);
            EXPECT_NEAR(metrics.error_linf, 1.945758160654121e-04, 5e-11);
        }
    }

    EXPECT_NEAR(
        result.residual_l2,
        static_cast<double>(residual_l2(problem, phi)),
        residual_compare_tol<Real>()
    );
}

template <typename Real>
void run_invalid_input_solvers_reject_bad_options_and_problems() {
    const ProblemT<Real> valid_problem = make_problem<Real>(Case2D::Sine, 7);
    const SolveOptionsT<Real> valid_options{convergence_tol<Real>(), 100};

    EXPECT_THROW(static_cast<void>(solve_jacobi(valid_problem, SolveOptionsT<Real>{Real{}, 100})), std::invalid_argument);
    EXPECT_THROW(
        static_cast<void>(solve_jacobi(valid_problem, SolveOptionsT<Real>{-convergence_tol<Real>(), 100})),
        std::invalid_argument
    );
    EXPECT_THROW(static_cast<void>(solve_jacobi(valid_problem, SolveOptionsT<Real>{convergence_tol<Real>(), 0})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(solve_gs(valid_problem, SolveOptionsT<Real>{Real{}, 100})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(solve_sor(valid_problem, SolveOptionsT<Real>{Real{}, 100})), std::invalid_argument);

    ProblemT<Real> bad_h = valid_problem;
    bad_h.h = Real{};
    EXPECT_THROW(static_cast<void>(solve_jacobi(bad_h, valid_options)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(solve_gs(bad_h, valid_options)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(solve_sor(bad_h, valid_options)), std::invalid_argument);

    ProblemT<Real> bad_rhs = valid_problem;
    bad_rhs.rhs = GridT<Real>{valid_problem.rhs.size() + 1};
    EXPECT_THROW(static_cast<void>(solve_jacobi(bad_rhs, valid_options)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(solve_gs(bad_rhs, valid_options)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(solve_sor(bad_rhs, valid_options)), std::invalid_argument);

    ProblemT<Real> bad_phi0 = valid_problem;
    bad_phi0.phi0 = GridT<Real>{valid_problem.phi0.size() + 1};
    EXPECT_THROW(static_cast<void>(solve_jacobi(bad_phi0, valid_options)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(solve_gs(bad_phi0, valid_options)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(solve_sor(bad_phi0, valid_options)), std::invalid_argument);

    ProblemT<Real> bad_n = valid_problem;
    bad_n.interior_n = 0;
    EXPECT_THROW(static_cast<void>(solve_jacobi(bad_n, valid_options)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(solve_gs(bad_n, valid_options)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(solve_sor(bad_n, valid_options)), std::invalid_argument);
}

template <typename Real>
void run_invalid_input_mg_rejects_bad_options_and_problems() {
    const ProblemT<Real> valid_problem = make_problem<Real>(Case2D::Sine, 7);

    const MGOptionsT<Real> exact_options = make_mg_options<Real>(MGCycle::V);
    const MGOptionsT<Real> sor_options = make_mg_options<Real>(MGCycle::V, 16, 100);

    EXPECT_THROW(
        static_cast<void>(solve_mg_exact(valid_problem, MGOptionsT<Real>{Real{}, 100, 2, MGCycle::V, 16})),
        std::invalid_argument
    );
    EXPECT_THROW(
        static_cast<void>(solve_mg_exact(valid_problem, MGOptionsT<Real>{-convergence_tol<Real>(), 100, 2, MGCycle::V, 16})),
        std::invalid_argument
    );
    EXPECT_THROW(
        static_cast<void>(solve_mg_exact(valid_problem, MGOptionsT<Real>{convergence_tol<Real>(), 0, 2, MGCycle::V, 16})),
        std::invalid_argument
    );
    EXPECT_THROW(
        static_cast<void>(solve_mg_exact(valid_problem, MGOptionsT<Real>{convergence_tol<Real>(), 100, 0, MGCycle::V, 16})),
        std::invalid_argument
    );

    EXPECT_THROW(
        static_cast<void>(solve_mg_sor(valid_problem, MGOptionsT<Real>{Real{}, 100, 2, MGCycle::V, 16})),
        std::invalid_argument
    );
    EXPECT_THROW(
        static_cast<void>(solve_mg_sor(valid_problem, MGOptionsT<Real>{convergence_tol<Real>(), 0, 2, MGCycle::V, 16})),
        std::invalid_argument
    );
    EXPECT_THROW(
        static_cast<void>(solve_mg_sor(valid_problem, MGOptionsT<Real>{convergence_tol<Real>(), 100, 0, MGCycle::V, 16})),
        std::invalid_argument
    );
    EXPECT_THROW(
        static_cast<void>(solve_mg_sor(valid_problem, MGOptionsT<Real>{convergence_tol<Real>(), 100, 2, MGCycle::V, 0})),
        std::invalid_argument
    );

    ProblemT<Real> bad_h = valid_problem;
    bad_h.h = Real{};
    EXPECT_THROW(static_cast<void>(solve_mg_exact(bad_h, exact_options)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(solve_mg_sor(bad_h, sor_options)), std::invalid_argument);

    ProblemT<Real> bad_rhs = valid_problem;
    bad_rhs.rhs = GridT<Real>{valid_problem.rhs.size() + 1};
    EXPECT_THROW(static_cast<void>(solve_mg_exact(bad_rhs, exact_options)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(solve_mg_sor(bad_rhs, sor_options)), std::invalid_argument);

    ProblemT<Real> bad_phi0 = valid_problem;
    bad_phi0.phi0 = GridT<Real>{valid_problem.phi0.size() + 1};
    EXPECT_THROW(static_cast<void>(solve_mg_exact(bad_phi0, exact_options)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(solve_mg_sor(bad_phi0, sor_options)), std::invalid_argument);

    ProblemT<Real> bad_n = valid_problem;
    bad_n.interior_n = 0;
    EXPECT_THROW(static_cast<void>(solve_mg_exact(bad_n, exact_options)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(solve_mg_sor(bad_n, sor_options)), std::invalid_argument);
}

template <typename Real>
class ProblemCaseTestBase : public ::testing::TestWithParam<Case2D> {};

template <typename Real>
class JacobiCaseTestBase : public ::testing::TestWithParam<Case2D> {};

template <typename Real>
class GsCaseTestBase : public ::testing::TestWithParam<Case2D> {};

template <typename Real>
class SorCaseTestBase : public ::testing::TestWithParam<Case2D> {};

template <typename Real>
class MgExactCaseTestBase : public ::testing::TestWithParam<MgCaseParam> {};

template <typename Real>
class MgSorCaseTestBase : public ::testing::TestWithParam<MgCaseParam> {};

template <typename Real>
class MgRegressionTestBase : public ::testing::TestWithParam<MGCycle> {};

using ProblemCaseTestFloat = ProblemCaseTestBase<float>;
using ProblemCaseTestDouble = ProblemCaseTestBase<double>;
using JacobiCaseTestFloat = JacobiCaseTestBase<float>;
using JacobiCaseTestDouble = JacobiCaseTestBase<double>;
using GsCaseTestFloat = GsCaseTestBase<float>;
using GsCaseTestDouble = GsCaseTestBase<double>;
using SorCaseTestFloat = SorCaseTestBase<float>;
using SorCaseTestDouble = SorCaseTestBase<double>;
using MgExactCaseTestFloat = MgExactCaseTestBase<float>;
using MgExactCaseTestDouble = MgExactCaseTestBase<double>;
using MgSorCaseTestFloat = MgSorCaseTestBase<float>;
using MgSorCaseTestDouble = MgSorCaseTestBase<double>;
using MgRegressionTestFloat = MgRegressionTestBase<float>;
using MgRegressionTestDouble = MgRegressionTestBase<double>;

#define DEFINE_REAL_TEST(SUITE, TEST_NAME, HELPER) \
TEST(SUITE##Float, TEST_NAME) { HELPER<float>(); } \
TEST(SUITE##Double, TEST_NAME) { HELPER<double>(); }

#define DEFINE_REAL_PARAM_TEST(SUITE, TEST_NAME, HELPER) \
TEST_P(SUITE##Float, TEST_NAME) { HELPER<float>(GetParam()); } \
TEST_P(SUITE##Double, TEST_NAME) { HELPER<double>(GetParam()); }

DEFINE_REAL_TEST(Grid2DTest, ConstructionFillAndBounds, run_grid2d_construction_fill_and_bounds);
DEFINE_REAL_TEST(OperatorsTest, ApplyAAndResidualStencil, run_apply_a_and_residual_stencil);
DEFINE_REAL_TEST(MetricsTest, ExactDiscreteProblemHasZeroMetrics, run_metrics_exact_discrete_problem_has_zero_metrics);
DEFINE_REAL_PARAM_TEST(ProblemCaseTest, ManufacturedProblemShapeAndBoundary, run_manufactured_problem_shape_and_boundary);
DEFINE_REAL_TEST(ValidationTest, DetectsTamperedProblemFields, run_validation_detects_tampered_problem_fields);
DEFINE_REAL_TEST(ProblemErrorTest, RejectsInvalidConstructionArguments, run_problem_error_rejects_invalid_construction_arguments);
DEFINE_REAL_TEST(ParameterErrorTest, GridAccessRejectsOutOfRange, run_parameter_error_grid_access_rejects_out_of_range);
DEFINE_REAL_TEST(JacobiSolverInterfaceTest, DirectFunctionAndClassWrapperAgree, run_jacobi_solver_interface);
DEFINE_REAL_PARAM_TEST(JacobiCaseTest, ConvergesAndPreservesBoundaries, run_jacobi_solver_converges);
DEFINE_REAL_TEST(JacobiSolverRegressionTest, SineCaseMatchesReferenceValues, run_jacobi_solver_regression);
DEFINE_REAL_TEST(GaussSeidelSolverInterfaceTest, DirectFunctionAndClassWrapperAgree, run_gs_solver_interface);
DEFINE_REAL_PARAM_TEST(GsCaseTest, ConvergesAndPreservesBoundaries, run_gs_solver_converges);
DEFINE_REAL_TEST(SorSolverInterfaceTest, DirectFunctionAndClassWrapperAgree, run_sor_solver_interface);
DEFINE_REAL_PARAM_TEST(SorCaseTest, ConvergesAndPreservesBoundaries, run_sor_solver_converges);
DEFINE_REAL_PARAM_TEST(MgExactCaseTest, ConvergesAndPreservesBoundaries, run_mg_exact_converges);
DEFINE_REAL_PARAM_TEST(MgSorCaseTest, ConvergesAndPreservesBoundaries, run_mg_sor_converges);
DEFINE_REAL_PARAM_TEST(MgRegressionTest, SineCaseMatchesReferenceValues, run_mg_regression);
DEFINE_REAL_TEST(InvalidInputTest, SolversRejectBadOptionsAndProblems, run_invalid_input_solvers_reject_bad_options_and_problems);
DEFINE_REAL_TEST(InvalidInputTest, MgRejectsBadOptionsAndProblems, run_invalid_input_mg_rejects_bad_options_and_problems);

#undef DEFINE_REAL_TEST
#undef DEFINE_REAL_PARAM_TEST

INSTANTIATE_TEST_SUITE_P(Float, ProblemCaseTestFloat, ::testing::ValuesIn(kCases), case_name);
INSTANTIATE_TEST_SUITE_P(Double, ProblemCaseTestDouble, ::testing::ValuesIn(kCases), case_name);
INSTANTIATE_TEST_SUITE_P(Float, JacobiCaseTestFloat, ::testing::ValuesIn(kCases), case_name);
INSTANTIATE_TEST_SUITE_P(Double, JacobiCaseTestDouble, ::testing::ValuesIn(kCases), case_name);
INSTANTIATE_TEST_SUITE_P(Float, GsCaseTestFloat, ::testing::ValuesIn(kCases), case_name);
INSTANTIATE_TEST_SUITE_P(Double, GsCaseTestDouble, ::testing::ValuesIn(kCases), case_name);
INSTANTIATE_TEST_SUITE_P(Float, SorCaseTestFloat, ::testing::ValuesIn(kCases), case_name);
INSTANTIATE_TEST_SUITE_P(Double, SorCaseTestDouble, ::testing::ValuesIn(kCases), case_name);
INSTANTIATE_TEST_SUITE_P(
    Float,
    MgExactCaseTestFloat,
    ::testing::Combine(::testing::ValuesIn(kCases), ::testing::ValuesIn(kCycles)),
    mg_case_name
);
INSTANTIATE_TEST_SUITE_P(
    Double,
    MgExactCaseTestDouble,
    ::testing::Combine(::testing::ValuesIn(kCases), ::testing::ValuesIn(kCycles)),
    mg_case_name
);
INSTANTIATE_TEST_SUITE_P(
    Float,
    MgSorCaseTestFloat,
    ::testing::Combine(::testing::ValuesIn(kCases), ::testing::ValuesIn(kCycles)),
    mg_case_name
);
INSTANTIATE_TEST_SUITE_P(
    Double,
    MgSorCaseTestDouble,
    ::testing::Combine(::testing::ValuesIn(kCases), ::testing::ValuesIn(kCycles)),
    mg_case_name
);
INSTANTIATE_TEST_SUITE_P(Float, MgRegressionTestFloat, ::testing::ValuesIn(kCycles), cycle_name);
INSTANTIATE_TEST_SUITE_P(Double, MgRegressionTestDouble, ::testing::ValuesIn(kCycles), cycle_name);

} // namespace
