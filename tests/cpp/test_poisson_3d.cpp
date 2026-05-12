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
#include <tuple>
#include <type_traits>

namespace {

using namespace poisson;

template <typename Real>
using GridT = poisson::Grid3D<Real>;

template <typename Real>
using ProblemT = poisson::Problem3D<Real>;

template <typename Real>
using SolveOptionsT = poisson::SolveOptions<Real>;

template <typename Real>
using MGOptionsT = poisson::MGOptions<Real>;

constexpr std::array<Case3D, 5> kCases{
    Case3D::Sine,
    Case3D::MixedSine,
    Case3D::Bubble,
    Case3D::Exp,
    Case3D::Cosine,
};
constexpr std::array<MGCycle, 2> kCycles{MGCycle::V, MGCycle::W};

template <typename Real>
constexpr Real zero_tol() {
    if constexpr (std::is_same_v<Real, float>) {
        return Real{1e-5};
    }
    return Real{1e-12};
}

template <typename Real>
constexpr Real convergence_tol() {
    if constexpr (std::is_same_v<Real, float>) {
        return Real{1e-3};
    }
    return Real{1e-8};
}

template <typename Real>
constexpr double residual_compare_tol() {
    if constexpr (std::is_same_v<Real, float>) {
        return 1e-4;
    }
    return 1e-10;
}

template <typename Real>
SolveOptionsT<Real> make_solve_options(std::size_t max_iter = 20'000) {
    return SolveOptionsT<Real>{convergence_tol<Real>(), max_iter};
}

template <typename Real>
MGOptionsT<Real> make_mg_options(
    MGCycle cycle,
    std::size_t coarse_steps = 32,
    std::size_t max_iter = 80,
    Real omega = Real{1},
    bool omega_is_auto = false
) {
    return MGOptionsT<Real>{
        convergence_tol<Real>(),
        max_iter,
        2,
        cycle,
        coarse_steps,
        omega,
        omega_is_auto,
    };
}

template <typename Real>
bool grid_all_close_to_zero(const GridT<Real>& grid, Real tol = zero_tol<Real>()) {
    for (std::size_t i = 0; i < grid.size(); ++i) {
        for (std::size_t j = 0; j < grid.size(); ++j) {
            for (std::size_t k = 0; k < grid.size(); ++k) {
                if (std::abs(grid(i, j, k)) > tol) {
                    return false;
                }
            }
        }
    }
    return true;
}

template <typename Real>
bool boundary_has_nonzero(const GridT<Real>& grid, Real tol = zero_tol<Real>()) {
    const std::size_t n = grid.size();

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (std::abs(grid(i, j, 0)) > tol || std::abs(grid(i, j, n - 1)) > tol) {
                return true;
            }
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            if (std::abs(grid(i, 0, k)) > tol || std::abs(grid(i, n - 1, k)) > tol) {
                return true;
            }
        }
    }
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t k = 0; k < n; ++k) {
            if (std::abs(grid(0, j, k)) > tol || std::abs(grid(n - 1, j, k)) > tol) {
                return true;
            }
        }
    }

    return false;
}

template <typename Real>
void expect_boundaries_equal(const GridT<Real>& lhs, const GridT<Real>& rhs) {
    ASSERT_EQ(lhs.size(), rhs.size());

    const std::size_t n = lhs.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            EXPECT_DOUBLE_EQ(static_cast<double>(lhs(i, j, 0)), static_cast<double>(rhs(i, j, 0)));
            EXPECT_DOUBLE_EQ(static_cast<double>(lhs(i, j, n - 1)), static_cast<double>(rhs(i, j, n - 1)));
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            EXPECT_DOUBLE_EQ(static_cast<double>(lhs(i, 0, k)), static_cast<double>(rhs(i, 0, k)));
            EXPECT_DOUBLE_EQ(static_cast<double>(lhs(i, n - 1, k)), static_cast<double>(rhs(i, n - 1, k)));
        }
    }
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t k = 0; k < n; ++k) {
            EXPECT_DOUBLE_EQ(static_cast<double>(lhs(0, j, k)), static_cast<double>(rhs(0, j, k)));
            EXPECT_DOUBLE_EQ(static_cast<double>(lhs(n - 1, j, k)), static_cast<double>(rhs(n - 1, j, k)));
        }
    }
}

template <typename Real, typename SolveFn, typename Options>
SolveResult3D expect_solver_smoke(SolveFn&& solve, const ProblemT<Real>& problem, const Options& options) {
    const SolveResult3D result = solve(problem, options);
    const GridT<Real> phi = cast_grid<Real>(result.phi);
    const Real recomputed_residual = relative_physical_residual_l2(problem, phi);
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
    const SolveResult3D result = expect_solver_smoke(solve, problem, options);
    const GridT<Real> phi = cast_grid<Real>(result.phi);
    const ErrorMetrics metrics = error_metrics(problem, phi);

    EXPECT_LT(metrics.error_l2, 1e-1);
    EXPECT_LT(metrics.error_linf, 2e-1);
}

std::string case_name(const ::testing::TestParamInfo<Case3D>& info) {
    return std::string(poisson::to_string(info.param));
}

std::string cycle_name(const ::testing::TestParamInfo<MGCycle>& info) {
    return info.param == MGCycle::V ? "V" : "W";
}

using MgCaseParam = std::tuple<Case3D, MGCycle>;

std::string mg_case_name(const ::testing::TestParamInfo<MgCaseParam>& info) {
    const auto& [case_id, cycle] = info.param;
    return std::string(poisson::to_string(case_id)) + "_" + (cycle == MGCycle::V ? "V" : "W");
}

template <typename Real>
void run_grid3d_construction_fill_and_bounds() {
    GridT<Real> empty;
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.size(), 0u);
    EXPECT_EQ(empty.elements(), 0u);

    GridT<Real> grid{4, Real{1.5}};
    EXPECT_FALSE(grid.empty());
    EXPECT_EQ(grid.size(), 4u);
    EXPECT_EQ(grid.elements(), 64u);
    EXPECT_DOUBLE_EQ(static_cast<double>(grid(2, 3, 1)), 1.5);

    grid.fill(Real{2});
    EXPECT_DOUBLE_EQ(static_cast<double>(grid(0, 0, 0)), 2.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(grid(3, 3, 3)), 2.0);
    EXPECT_THROW((void)grid(4, 0, 0), std::out_of_range);
}

template <typename Real>
void run_apply_a_and_residual_stencil_3d() {
    GridT<Real> phi{5};
    phi.fill(Real{});
    phi(2, 2, 2) = Real{1};

    const GridT<Real> aphi = apply_A(phi, Real{1});
    ASSERT_EQ(aphi.size(), 3u);

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 3; ++k) {
                Real expected{};
                if (i == 1 && j == 1 && k == 1) {
                    expected = Real{6};
                } else if (
                    (i == 1 && j == 1 && (k == 0 || k == 2)) ||
                    (i == 1 && k == 1 && (j == 0 || j == 2)) ||
                    (j == 1 && k == 1 && (i == 0 || i == 2))
                ) {
                    expected = Real{-1};
                }
                EXPECT_DOUBLE_EQ(static_cast<double>(aphi(i, j, k)), static_cast<double>(expected));
            }
        }
    }

    const GridT<Real> zero_rhs{5};
    const GridT<Real> r0 = residual(phi, zero_rhs, Real{1});
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 3; ++k) {
                EXPECT_DOUBLE_EQ(static_cast<double>(r0(i, j, k)), -static_cast<double>(aphi(i, j, k)));
            }
        }
    }

    GridT<Real> rhs{5};
    rhs.fill(Real{});
    for (std::size_t i = 1; i + 1 < rhs.size(); ++i) {
        for (std::size_t j = 1; j + 1 < rhs.size(); ++j) {
            for (std::size_t k = 1; k + 1 < rhs.size(); ++k) {
                rhs(i, j, k) = aphi(i - 1, j - 1, k - 1);
            }
        }
    }

    const GridT<Real> r1 = residual(phi, rhs, Real{1});
    for (std::size_t i = 0; i < r1.size(); ++i) {
        for (std::size_t j = 0; j < r1.size(); ++j) {
            for (std::size_t k = 0; k < r1.size(); ++k) {
                EXPECT_DOUBLE_EQ(static_cast<double>(r1(i, j, k)), 0.0);
            }
        }
    }
}

template <typename Real>
void run_metrics_exact_discrete_problem_has_zero_metrics_3d() {
    GridT<Real> exact{5};
    exact.fill(Real{});
    exact(2, 2, 2) = Real{1};

    GridT<Real> rhs{5};
    rhs.fill(Real{});
    const GridT<Real> aphi = apply_A(exact, Real{1});
    for (std::size_t i = 1; i + 1 < rhs.size(); ++i) {
        for (std::size_t j = 1; j + 1 < rhs.size(); ++j) {
            for (std::size_t k = 1; k + 1 < rhs.size(); ++k) {
                rhs(i, j, k) = aphi(i - 1, j - 1, k - 1);
            }
        }
    }

    const ProblemT<Real> problem{
        Case3D::Sine,
        3,
        Real{1},
        exact,
        rhs,
        exact,
    };

    EXPECT_DOUBLE_EQ(static_cast<double>(residual_l2(problem, exact)), 0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(relative_physical_residual_l2(problem, exact)), 0.0);

    const ErrorMetrics metrics = error_metrics(problem, exact);
    EXPECT_DOUBLE_EQ(metrics.error_l2, 0.0);
    EXPECT_DOUBLE_EQ(metrics.error_linf, 0.0);
}

template <typename Real>
void run_manufactured_problem_shape_and_boundary_3d(Case3D case_id) {
    const ProblemT<Real> problem = make_problem_3d<Real>(case_id, 31);
    const ValidationReport report = validate_problem(problem);

    EXPECT_EQ(problem.case_id, case_id);
    EXPECT_EQ(problem.interior_n, 31u);
    EXPECT_DOUBLE_EQ(static_cast<double>(problem.h), 1.0 / 32.0);
    EXPECT_EQ(problem.array_n(), 33u);
    EXPECT_EQ(problem.exact.size(), 33u);
    EXPECT_EQ(problem.rhs.size(), 33u);
    EXPECT_EQ(problem.phi0.size(), 33u);
    EXPECT_TRUE(report.ok);
    EXPECT_TRUE(report.size_ok);
    EXPECT_TRUE(report.h_ok);
    EXPECT_TRUE(report.finite_ok);
    EXPECT_TRUE(report.boundary_ok);
    EXPECT_TRUE(report.interior_zero_ok);
    EXPECT_DOUBLE_EQ(report.boundary_error, 0.0);
    EXPECT_DOUBLE_EQ(report.interior_max_abs, 0.0);

    for (std::size_t i = 1; i + 1 < problem.array_n(); ++i) {
        for (std::size_t j = 1; j + 1 < problem.array_n(); ++j) {
            for (std::size_t k = 1; k + 1 < problem.array_n(); ++k) {
                EXPECT_DOUBLE_EQ(static_cast<double>(problem.phi0(i, j, k)), 0.0);
            }
        }
    }

    if (case_id == Case3D::Sine || case_id == Case3D::MixedSine || case_id == Case3D::Bubble) {
        EXPECT_TRUE(grid_all_close_to_zero(problem.phi0));
    } else {
        EXPECT_TRUE(boundary_has_nonzero(problem.phi0));
        EXPECT_FALSE(grid_all_close_to_zero(problem.phi0));
    }
}

template <typename Real>
void run_validation_detects_tampered_problem_fields_3d() {
    ProblemT<Real> problem = make_problem_3d<Real>(Case3D::Sine, 7);

    problem.phi0(2, 2, 2) = Real{1};
    ValidationReport report = validate_problem(problem);
    EXPECT_FALSE(report.ok);
    EXPECT_FALSE(report.interior_zero_ok);
    EXPECT_GT(report.interior_max_abs, 0.0);

    problem = make_problem_3d<Real>(Case3D::Sine, 7);
    problem.phi0(0, 0, 0) = problem.phi0(0, 0, 0) + Real{1};
    report = validate_problem(problem);
    EXPECT_FALSE(report.ok);
    EXPECT_FALSE(report.boundary_ok);
    EXPECT_GT(report.boundary_error, 0.0);
}

template <typename Real>
void run_problem_error_rejects_invalid_construction_arguments_3d() {
    EXPECT_THROW(static_cast<void>(make_problem_3d<Real>("unknown_case", 7)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(make_problem_3d<Real>(Case3D::Sine, 0)), std::invalid_argument);
}

template <typename Real>
void run_jacobi_solver_interface_3d() {
    const ProblemT<Real> problem = make_problem_3d<Real>(Case3D::Sine, 7);
    const SolveOptionsT<Real> options = make_solve_options<Real>();

    const SolveResult3D direct = expect_solver_smoke(
        [](const ProblemT<Real>& p, const SolveOptionsT<Real>& o) {
            return solve_jacobi(p, o);
        },
        problem,
        options
    );

    if constexpr (std::is_same_v<Real, double>) {
        const JacobiSolver3D solver{};
        const SolveResult3D via_interface = solver.solve(problem, options);

        EXPECT_EQ(solver.name(), "jacobi");
        EXPECT_EQ(direct.iterations, via_interface.iterations);
        EXPECT_DOUBLE_EQ(direct.residual_l2, via_interface.residual_l2);
        ASSERT_EQ(direct.phi.size(), via_interface.phi.size());
        for (std::size_t i = 0; i < direct.phi.size(); ++i) {
            for (std::size_t j = 0; j < direct.phi.size(); ++j) {
                for (std::size_t k = 0; k < direct.phi.size(); ++k) {
                    EXPECT_DOUBLE_EQ(direct.phi(i, j, k), via_interface.phi(i, j, k));
                }
            }
        }
    }
}

template <typename Real>
void run_jacobi_solver_converges_3d(Case3D case_id) {
    const ProblemT<Real> problem = make_problem_3d<Real>(case_id, 7);
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
void run_gs_solver_converges_3d(Case3D case_id) {
    const ProblemT<Real> problem = make_problem_3d<Real>(case_id, 7);
    const SolveOptionsT<Real> options = make_solve_options<Real>();
    expect_solver_converges(
        [](const ProblemT<Real>& p, const SolveOptionsT<Real>& o) {
            return solve_gs(p, o);
        },
        problem,
        options
    );

    if constexpr (std::is_same_v<Real, double>) {
        const GaussSeidelSolver3D solver{};
        EXPECT_EQ(solver.name(), "gs");
        (void)solver.solve(problem, options);
    }
}

template <typename Real>
void run_sor_solver_converges_3d(Case3D case_id) {
    const ProblemT<Real> problem = make_problem_3d<Real>(case_id, 7);
    const SolveOptionsT<Real> options = make_solve_options<Real>();
    expect_solver_converges(
        [](const ProblemT<Real>& p, const SolveOptionsT<Real>& o) {
            return solve_sor(p, o);
        },
        problem,
        options
    );

    if constexpr (std::is_same_v<Real, double>) {
        const SorSolver3D solver{};
        EXPECT_EQ(solver.name(), "sor");
        (void)solver.solve(problem, options);
    }
}

template <typename Real>
void run_mg_exact_converges_3d(const MgCaseParam& param) {
    const auto& [case_id, cycle] = param;
    const ProblemT<Real> problem = make_problem_3d<Real>(case_id, 15);
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
void run_mg_sor_converges_3d(const MgCaseParam& param) {
    const auto& [case_id, cycle] = param;
    const ProblemT<Real> problem = make_problem_3d<Real>(case_id, 15);
    const MGOptionsT<Real> options = make_mg_options<Real>(cycle, 64, 100);
    expect_solver_converges(
        [](const ProblemT<Real>& p, const MGOptionsT<Real>& o) {
            return solve_mg_sor(p, o);
        },
        problem,
        options
    );
}

template <typename Real>
void run_mg_coarse_modes_3d(MGCycle cycle) {
    const ProblemT<Real> problem = make_problem_3d<Real>(Case3D::Sine, 3);
    const MGOptionsT<Real> exact_options = make_mg_options<Real>(cycle, 16, 20);
    const MGOptionsT<Real> sor_options = make_mg_options<Real>(cycle, 64, 40);

    expect_solver_converges(
        [](const ProblemT<Real>& p, const MGOptionsT<Real>& o) {
            return solve_mg_exact(p, o);
        },
        problem,
        exact_options
    );
    expect_solver_converges(
        [](const ProblemT<Real>& p, const MGOptionsT<Real>& o) {
            return solve_mg_sor(p, o);
        },
        problem,
        sor_options
    );
}

template <typename Real>
void run_invalid_input_solvers_reject_bad_options_and_problems_3d() {
    const ProblemT<Real> valid_problem = make_problem_3d<Real>(Case3D::Sine, 7);
    const SolveOptionsT<Real> valid_options{convergence_tol<Real>(), 100};

    EXPECT_THROW(static_cast<void>(solve_jacobi(valid_problem, SolveOptionsT<Real>{Real{}, 100})), std::invalid_argument);
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
}

template <typename Real>
void run_invalid_input_mg_rejects_bad_options_and_problems_3d() {
    const ProblemT<Real> valid_problem = make_problem_3d<Real>(Case3D::Sine, 7);
    const MGOptionsT<Real> exact_options = make_mg_options<Real>(MGCycle::V);

    EXPECT_THROW(
        static_cast<void>(solve_mg_exact(valid_problem, MGOptionsT<Real>{Real{}, 100, 2, MGCycle::V, 16})),
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
        static_cast<void>(solve_mg_sor(valid_problem, MGOptionsT<Real>{convergence_tol<Real>(), 100, 2, MGCycle::V, 0})),
        std::invalid_argument
    );
    EXPECT_THROW(
        static_cast<void>(solve_mg_exact(valid_problem, MGOptionsT<Real>{convergence_tol<Real>(), 100, 2, MGCycle::V, 16, Real{0}, false})),
        std::invalid_argument
    );

    ProblemT<Real> bad_h = valid_problem;
    bad_h.h = Real{};
    EXPECT_THROW(static_cast<void>(solve_mg_exact(bad_h, exact_options)), std::invalid_argument);
}

template <typename Real>
class ProblemCase3DTestBase : public ::testing::TestWithParam<Case3D> {};

template <typename Real>
class JacobiCase3DTestBase : public ::testing::TestWithParam<Case3D> {};

template <typename Real>
class GsCase3DTestBase : public ::testing::TestWithParam<Case3D> {};

template <typename Real>
class SorCase3DTestBase : public ::testing::TestWithParam<Case3D> {};

template <typename Real>
class MgExactCase3DTestBase : public ::testing::TestWithParam<MgCaseParam> {};

template <typename Real>
class MgSorCase3DTestBase : public ::testing::TestWithParam<MgCaseParam> {};

template <typename Real>
class MgCoarseMode3DTestBase : public ::testing::TestWithParam<MGCycle> {};

using ProblemCase3DTestFloat = ProblemCase3DTestBase<float>;
using ProblemCase3DTestDouble = ProblemCase3DTestBase<double>;
using JacobiCase3DTestFloat = JacobiCase3DTestBase<float>;
using JacobiCase3DTestDouble = JacobiCase3DTestBase<double>;
using GsCase3DTestFloat = GsCase3DTestBase<float>;
using GsCase3DTestDouble = GsCase3DTestBase<double>;
using SorCase3DTestFloat = SorCase3DTestBase<float>;
using SorCase3DTestDouble = SorCase3DTestBase<double>;
using MgExactCase3DTestFloat = MgExactCase3DTestBase<float>;
using MgExactCase3DTestDouble = MgExactCase3DTestBase<double>;
using MgSorCase3DTestFloat = MgSorCase3DTestBase<float>;
using MgSorCase3DTestDouble = MgSorCase3DTestBase<double>;
using MgCoarseMode3DTestFloat = MgCoarseMode3DTestBase<float>;
using MgCoarseMode3DTestDouble = MgCoarseMode3DTestBase<double>;

#define DEFINE_REAL_TEST(SUITE, TEST_NAME, HELPER) \
TEST(SUITE##Float, TEST_NAME) { HELPER<float>(); } \
TEST(SUITE##Double, TEST_NAME) { HELPER<double>(); }

#define DEFINE_REAL_PARAM_TEST(SUITE, TEST_NAME, HELPER) \
TEST_P(SUITE##Float, TEST_NAME) { HELPER<float>(GetParam()); } \
TEST_P(SUITE##Double, TEST_NAME) { HELPER<double>(GetParam()); }

DEFINE_REAL_TEST(Grid3DTest, ConstructionFillAndBounds, run_grid3d_construction_fill_and_bounds);
DEFINE_REAL_TEST(Operators3DTest, ApplyAAndResidualStencil, run_apply_a_and_residual_stencil_3d);
DEFINE_REAL_TEST(Metrics3DTest, ExactDiscreteProblemHasZeroMetrics, run_metrics_exact_discrete_problem_has_zero_metrics_3d);
DEFINE_REAL_PARAM_TEST(ProblemCase3DTest, ManufacturedProblemShapeAndBoundary, run_manufactured_problem_shape_and_boundary_3d);
DEFINE_REAL_TEST(Validation3DTest, DetectsTamperedProblemFields, run_validation_detects_tampered_problem_fields_3d);
DEFINE_REAL_TEST(ProblemError3DTest, RejectsInvalidConstructionArguments, run_problem_error_rejects_invalid_construction_arguments_3d);
DEFINE_REAL_TEST(JacobiSolver3DInterfaceTest, DirectFunctionAndClassWrapperAgree, run_jacobi_solver_interface_3d);
DEFINE_REAL_PARAM_TEST(JacobiCase3DTest, ConvergesAndPreservesBoundaries, run_jacobi_solver_converges_3d);
DEFINE_REAL_PARAM_TEST(GsCase3DTest, ConvergesAndPreservesBoundaries, run_gs_solver_converges_3d);
DEFINE_REAL_PARAM_TEST(SorCase3DTest, ConvergesAndPreservesBoundaries, run_sor_solver_converges_3d);
DEFINE_REAL_PARAM_TEST(MgExactCase3DTest, ConvergesAndPreservesBoundaries, run_mg_exact_converges_3d);
DEFINE_REAL_PARAM_TEST(MgSorCase3DTest, ConvergesAndPreservesBoundaries, run_mg_sor_converges_3d);
DEFINE_REAL_PARAM_TEST(MgCoarseMode3DTest, ExactAndSorModesConverge, run_mg_coarse_modes_3d);
DEFINE_REAL_TEST(InvalidInput3DTest, SolversRejectBadOptionsAndProblems, run_invalid_input_solvers_reject_bad_options_and_problems_3d);
DEFINE_REAL_TEST(InvalidInput3DTest, MgRejectsBadOptionsAndProblems, run_invalid_input_mg_rejects_bad_options_and_problems_3d);

#undef DEFINE_REAL_TEST
#undef DEFINE_REAL_PARAM_TEST

INSTANTIATE_TEST_SUITE_P(Float, ProblemCase3DTestFloat, ::testing::ValuesIn(kCases), case_name);
INSTANTIATE_TEST_SUITE_P(Double, ProblemCase3DTestDouble, ::testing::ValuesIn(kCases), case_name);
INSTANTIATE_TEST_SUITE_P(Float, JacobiCase3DTestFloat, ::testing::ValuesIn(kCases), case_name);
INSTANTIATE_TEST_SUITE_P(Double, JacobiCase3DTestDouble, ::testing::ValuesIn(kCases), case_name);
INSTANTIATE_TEST_SUITE_P(Float, GsCase3DTestFloat, ::testing::ValuesIn(kCases), case_name);
INSTANTIATE_TEST_SUITE_P(Double, GsCase3DTestDouble, ::testing::ValuesIn(kCases), case_name);
INSTANTIATE_TEST_SUITE_P(Float, SorCase3DTestFloat, ::testing::ValuesIn(kCases), case_name);
INSTANTIATE_TEST_SUITE_P(Double, SorCase3DTestDouble, ::testing::ValuesIn(kCases), case_name);
INSTANTIATE_TEST_SUITE_P(
    Float,
    MgExactCase3DTestFloat,
    ::testing::Combine(::testing::ValuesIn(kCases), ::testing::ValuesIn(kCycles)),
    mg_case_name
);
INSTANTIATE_TEST_SUITE_P(
    Double,
    MgExactCase3DTestDouble,
    ::testing::Combine(::testing::ValuesIn(kCases), ::testing::ValuesIn(kCycles)),
    mg_case_name
);
INSTANTIATE_TEST_SUITE_P(
    Float,
    MgSorCase3DTestFloat,
    ::testing::Combine(::testing::ValuesIn(kCases), ::testing::ValuesIn(kCycles)),
    mg_case_name
);
INSTANTIATE_TEST_SUITE_P(
    Double,
    MgSorCase3DTestDouble,
    ::testing::Combine(::testing::ValuesIn(kCases), ::testing::ValuesIn(kCycles)),
    mg_case_name
);
INSTANTIATE_TEST_SUITE_P(Float, MgCoarseMode3DTestFloat, ::testing::ValuesIn(kCycles), cycle_name);
INSTANTIATE_TEST_SUITE_P(Double, MgCoarseMode3DTestDouble, ::testing::ValuesIn(kCycles), cycle_name);

} // namespace
