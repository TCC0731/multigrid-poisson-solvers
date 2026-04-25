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

namespace {

using namespace poisson;

constexpr Real kZeroTol = 1e-12;
constexpr std::array<Case2D, 5> kCases{
    Case2D::Sine,
    Case2D::MixedSine,
    Case2D::Bubble,
    Case2D::Exp,
    Case2D::Cosine,
};
constexpr std::array<MGCycle, 2> kCycles{MGCycle::V, MGCycle::W};

bool grid_all_close_to_zero(const Grid2D& grid, Real tol = kZeroTol) {
    for (std::size_t i = 0; i < grid.size(); ++i) {
        for (std::size_t j = 0; j < grid.size(); ++j) {
            if (std::abs(grid(i, j)) > tol) {
                return false;
            }
        }
    }
    return true;
}

bool boundary_has_nonzero(const Grid2D& grid, Real tol = kZeroTol) {
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

void expect_boundaries_equal(const Grid2D& lhs, const Grid2D& rhs) {
    ASSERT_EQ(lhs.size(), rhs.size());

    const std::size_t n = lhs.size();
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(lhs(i, 0), rhs(i, 0));
        EXPECT_DOUBLE_EQ(lhs(i, n - 1), rhs(i, n - 1));
    }
    for (std::size_t j = 0; j < n; ++j) {
        EXPECT_DOUBLE_EQ(lhs(0, j), rhs(0, j));
        EXPECT_DOUBLE_EQ(lhs(n - 1, j), rhs(n - 1, j));
    }
}

template <typename SolveFn, typename Options>
void expect_solver_converges(SolveFn&& solve, const Problem2D& problem, const Options& options) {
    const SolveResult result = solve(problem, options);
    const Real recomputed_residual = residual_l2(problem, result.phi);
    const ErrorMetrics metrics = error_metrics(problem, result.phi);

    EXPECT_EQ(result.phi.size(), problem.phi0.size());
    EXPECT_GT(result.iterations, 0u);
    EXPECT_LE(result.iterations, options.max_iter);
    EXPECT_TRUE(std::isfinite(result.residual_l2));
    EXPECT_GE(result.residual_l2, 0.0);
    EXPECT_LE(result.residual_l2, options.tol);
    EXPECT_DOUBLE_EQ(result.residual_l2, recomputed_residual);

    expect_boundaries_equal(result.phi, problem.phi0);

    EXPECT_TRUE(std::isfinite(metrics.error_l2));
    EXPECT_TRUE(std::isfinite(metrics.error_linf));
    EXPECT_GE(metrics.error_l2, 0.0);
    EXPECT_GE(metrics.error_linf, 0.0);
    EXPECT_LT(metrics.error_l2, 1e-2);
    EXPECT_LT(metrics.error_linf, 1e-2);
}

std::string case_name(const ::testing::TestParamInfo<Case2D>& info) {
    return std::string(poisson::to_string(info.param));
}

std::string cycle_name(const ::testing::TestParamInfo<MGCycle>& info) {
    return info.param == MGCycle::V ? "V" : "W";
}

MGOptions make_mg_options(MGCycle cycle, std::size_t coarse_steps = 16, std::size_t max_iter = 100) {
    return MGOptions{1e-10, max_iter, 2, cycle, coarse_steps};
}

using MgCaseParam = std::tuple<Case2D, MGCycle>;

std::string mg_case_name(const ::testing::TestParamInfo<MgCaseParam>& info) {
    const auto& [case_id, cycle] = info.param;
    return std::string(poisson::to_string(case_id)) + "_" + (cycle == MGCycle::V ? "V" : "W");
}

class ProblemCaseTest : public ::testing::TestWithParam<Case2D> {};
class JacobiCaseTest : public ::testing::TestWithParam<Case2D> {};
class GsCaseTest : public ::testing::TestWithParam<Case2D> {};
class SorCaseTest : public ::testing::TestWithParam<Case2D> {};
class MgExactCaseTest : public ::testing::TestWithParam<MgCaseParam> {};
class MgSorCaseTest : public ::testing::TestWithParam<MgCaseParam> {};
class MgRegressionTest : public ::testing::TestWithParam<MGCycle> {};

INSTANTIATE_TEST_SUITE_P(AllCases, ProblemCaseTest, ::testing::ValuesIn(kCases), case_name);
INSTANTIATE_TEST_SUITE_P(AllCases, JacobiCaseTest, ::testing::ValuesIn(kCases), case_name);
INSTANTIATE_TEST_SUITE_P(AllCases, GsCaseTest, ::testing::ValuesIn(kCases), case_name);
INSTANTIATE_TEST_SUITE_P(AllCases, SorCaseTest, ::testing::ValuesIn(kCases), case_name);
INSTANTIATE_TEST_SUITE_P(
    AllCases,
    MgExactCaseTest,
    ::testing::Combine(::testing::ValuesIn(kCases), ::testing::ValuesIn(kCycles)),
    mg_case_name
);
INSTANTIATE_TEST_SUITE_P(
    AllCases,
    MgSorCaseTest,
    ::testing::Combine(::testing::ValuesIn(kCases), ::testing::ValuesIn(kCycles)),
    mg_case_name
);
INSTANTIATE_TEST_SUITE_P(AllCycles, MgRegressionTest, ::testing::ValuesIn(kCycles), cycle_name);

TEST(Grid2DTest, ConstructionFillAndBounds) {
    Grid2D empty;
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.size(), 0u);
    EXPECT_EQ(empty.elements(), 0u);

    Grid2D grid{4, 1.5};
    EXPECT_FALSE(grid.empty());
    EXPECT_EQ(grid.size(), 4u);
    EXPECT_EQ(grid.elements(), 16u);
    EXPECT_DOUBLE_EQ(grid(2, 3), 1.5);

    grid.fill(2.0);
    EXPECT_DOUBLE_EQ(grid(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(grid(3, 3), 2.0);
    EXPECT_THROW((void)grid(4, 0), std::out_of_range);
}

TEST(OperatorsTest, ApplyAAndResidualStencil) {
    Grid2D phi{5};
    phi.fill(0.0);
    phi(2, 2) = 1.0;

    const Grid2D aphi = apply_A(phi, 1.0);
    ASSERT_EQ(aphi.size(), 3u);

    constexpr Real expected[3][3] = {
        {0.0, -1.0, 0.0},
        {-1.0, 4.0, -1.0},
        {0.0, -1.0, 0.0},
    };

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(aphi(i, j), expected[i][j]);
        }
    }

    const Grid2D zero_rhs{5};
    const Grid2D r0 = residual(phi, zero_rhs, 1.0);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(r0(i, j), -expected[i][j]);
        }
    }

    Grid2D rhs{5};
    rhs.fill(0.0);
    for (std::size_t i = 1; i + 1 < rhs.size(); ++i) {
        for (std::size_t j = 1; j + 1 < rhs.size(); ++j) {
            rhs(i, j) = aphi(i - 1, j - 1);
        }
    }

    const Grid2D r1 = residual(phi, rhs, 1.0);
    for (std::size_t i = 0; i < r1.size(); ++i) {
        for (std::size_t j = 0; j < r1.size(); ++j) {
            EXPECT_DOUBLE_EQ(r1(i, j), 0.0);
        }
    }
}

TEST(MetricsTest, ExactDiscreteProblemHasZeroMetrics) {
    Grid2D exact{5};
    exact.fill(0.0);
    exact(2, 2) = 1.0;

    Grid2D rhs{5};
    rhs.fill(0.0);
    const Grid2D aphi = apply_A(exact, 1.0);
    for (std::size_t i = 1; i + 1 < rhs.size(); ++i) {
        for (std::size_t j = 1; j + 1 < rhs.size(); ++j) {
            rhs(i, j) = aphi(i - 1, j - 1);
        }
    }

    const Problem2D problem{
        Case2D::Sine,
        3,
        1.0,
        exact,
        rhs,
        exact,
    };

    EXPECT_DOUBLE_EQ(residual_l2(problem, exact), 0.0);

    const ErrorMetrics metrics = error_metrics(problem, exact);
    EXPECT_DOUBLE_EQ(metrics.error_l2, 0.0);
    EXPECT_DOUBLE_EQ(metrics.error_linf, 0.0);
}

TEST_P(ProblemCaseTest, ManufacturedProblemShapeAndBoundary) {
    const Case2D case_id = GetParam();
    const Problem2D problem = make_problem(case_id, 7);
    const ValidationReport report = validate_problem(problem);

    EXPECT_EQ(problem.case_id, case_id);
    EXPECT_EQ(problem.interior_n, 7u);
    EXPECT_DOUBLE_EQ(problem.h, 1.0 / 8.0);
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
        EXPECT_DOUBLE_EQ(problem.phi0(i, 0), problem.exact(i, 0));
        EXPECT_DOUBLE_EQ(problem.phi0(i, problem.array_n() - 1), problem.exact(i, problem.array_n() - 1));
    }
    for (std::size_t j = 0; j < problem.array_n(); ++j) {
        EXPECT_DOUBLE_EQ(problem.phi0(0, j), problem.exact(0, j));
        EXPECT_DOUBLE_EQ(problem.phi0(problem.array_n() - 1, j), problem.exact(problem.array_n() - 1, j));
    }

    for (std::size_t i = 1; i + 1 < problem.array_n(); ++i) {
        for (std::size_t j = 1; j + 1 < problem.array_n(); ++j) {
            EXPECT_DOUBLE_EQ(problem.phi0(i, j), 0.0);
        }
    }

    if (case_id == Case2D::Sine || case_id == Case2D::MixedSine || case_id == Case2D::Bubble) {
        EXPECT_TRUE(grid_all_close_to_zero(problem.phi0));
    } else {
        EXPECT_TRUE(boundary_has_nonzero(problem.phi0));
        EXPECT_FALSE(grid_all_close_to_zero(problem.phi0));
    }
}

TEST(ValidationTest, DetectsTamperedProblemFields) {
    Problem2D problem = make_problem(Case2D::Sine, 7);

    problem.phi0(2, 2) = 1.0;
    ValidationReport report = validate_problem(problem);
    EXPECT_FALSE(report.ok);
    EXPECT_FALSE(report.interior_zero_ok);
    EXPECT_GT(report.interior_max_abs, 0.0);

    problem = make_problem(Case2D::Sine, 7);
    problem.phi0(0, 0) = problem.phi0(0, 0) + 1.0;
    report = validate_problem(problem);
    EXPECT_FALSE(report.ok);
    EXPECT_FALSE(report.boundary_ok);
    EXPECT_GT(report.boundary_error, 0.0);
}

TEST(ProblemErrorTest, RejectsInvalidConstructionArguments) {
    EXPECT_THROW(make_problem("unknown_case", 7), std::invalid_argument);
    EXPECT_THROW(make_problem(Case2D::Sine, 0), std::invalid_argument);
}

TEST(ParameterErrorTest, GridAccessRejectsOutOfRange) {
    Grid2D grid{4};
    EXPECT_THROW((void)grid(4, 0), std::out_of_range);
    EXPECT_THROW((void)grid(0, 4), std::out_of_range);
}

TEST(JacobiSolverInterfaceTest, DirectFunctionAndClassWrapperAgree) {
    const Problem2D problem = make_problem(Case2D::Sine, 7);
    const SolveOptions options{1e-8, 5'000};

    const SolveResult direct = solve_jacobi(problem, options);
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

TEST_P(JacobiCaseTest, ConvergesAndPreservesBoundaries) {
    const Problem2D problem = make_problem(GetParam(), 32);
    const SolveOptions options{1e-10, 20'000};
    expect_solver_converges(solve_jacobi, problem, options);
}

TEST(JacobiSolverRegressionTest, SineCaseMatchesReferenceValues) {
    const Problem2D problem = make_problem(Case2D::Sine, 31);
    const SolveOptions options{1e-10, 20'000};

    const SolveResult result = solve_jacobi(problem, options);
    const ErrorMetrics metrics = error_metrics(problem, result.phi);

    EXPECT_EQ(result.iterations, 5'245u);
    EXPECT_DOUBLE_EQ(result.residual_l2, residual_l2(problem, result.phi));
    EXPECT_NEAR(result.residual_l2, 9.981977e-11, 1e-15);
    EXPECT_NEAR(metrics.error_l2, 4.017888e-04, 5e-11);
    EXPECT_NEAR(metrics.error_linf, 8.035777e-04, 5e-11);
}

TEST(GaussSeidelSolverInterfaceTest, DirectFunctionAndClassWrapperAgree) {
    const Problem2D problem = make_problem(Case2D::Sine, 7);
    const SolveOptions options{1e-8, 5'000};

    const SolveResult direct = solve_gs(problem, options);
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

TEST_P(GsCaseTest, ConvergesAndPreservesBoundaries) {
    const Problem2D problem = make_problem(GetParam(), 32);
    const SolveOptions options{1e-10, 20'000};
    expect_solver_converges(solve_gs, problem, options);
}

TEST(SorSolverInterfaceTest, DirectFunctionAndClassWrapperAgree) {
    const Problem2D problem = make_problem(Case2D::Sine, 7);
    const SolveOptions options{1e-8, 5'000};

    const SolveResult direct = solve_sor(problem, options);
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

TEST_P(SorCaseTest, ConvergesAndPreservesBoundaries) {
    const Problem2D problem = make_problem(GetParam(), 32);
    const SolveOptions options{1e-10, 20'000};
    expect_solver_converges(solve_sor, problem, options);
}

TEST_P(MgExactCaseTest, ConvergesAndPreservesBoundaries) {
    const auto& [case_id, cycle] = GetParam();
    const Problem2D problem = make_problem(case_id, 32);
    const MGOptions options = make_mg_options(cycle);
    expect_solver_converges(solve_mg_exact, problem, options);
}

TEST_P(MgSorCaseTest, ConvergesAndPreservesBoundaries) {
    const auto& [case_id, cycle] = GetParam();
    const Problem2D problem = make_problem(case_id, 32);
    const MGOptions options = make_mg_options(cycle, 32, 150);
    expect_solver_converges(solve_mg_sor, problem, options);
}

TEST_P(MgRegressionTest, SineCaseMatchesReferenceValues) {
    const MGCycle cycle = GetParam();
    const Problem2D problem = make_problem(Case2D::Sine, 64);
    const MGOptions options = make_mg_options(cycle, 16, 100);

    const SolveResult result = solve_mg_exact(problem, options);
    const ErrorMetrics metrics = error_metrics(problem, result.phi);

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

    EXPECT_DOUBLE_EQ(result.residual_l2, residual_l2(problem, result.phi));
}

TEST(InvalidInputTest, SolversRejectBadOptionsAndProblems) {
    const Problem2D valid_problem = make_problem(Case2D::Sine, 7);
    const SolveOptions valid_options{1e-10, 100};

    EXPECT_THROW(solve_jacobi(valid_problem, SolveOptions{0.0, 100}), std::invalid_argument);
    EXPECT_THROW(solve_jacobi(valid_problem, SolveOptions{-1e-10, 100}), std::invalid_argument);
    EXPECT_THROW(solve_jacobi(valid_problem, SolveOptions{1e-10, 0}), std::invalid_argument);
    EXPECT_THROW(solve_gs(valid_problem, SolveOptions{0.0, 100}), std::invalid_argument);
    EXPECT_THROW(solve_sor(valid_problem, SolveOptions{0.0, 100}), std::invalid_argument);

    Problem2D bad_h = valid_problem;
    bad_h.h = 0.0;
    EXPECT_THROW(solve_jacobi(bad_h, valid_options), std::invalid_argument);
    EXPECT_THROW(solve_gs(bad_h, valid_options), std::invalid_argument);
    EXPECT_THROW(solve_sor(bad_h, valid_options), std::invalid_argument);

    Problem2D bad_rhs = valid_problem;
    bad_rhs.rhs = Grid2D{valid_problem.rhs.size() + 1};
    EXPECT_THROW(solve_jacobi(bad_rhs, valid_options), std::invalid_argument);
    EXPECT_THROW(solve_gs(bad_rhs, valid_options), std::invalid_argument);
    EXPECT_THROW(solve_sor(bad_rhs, valid_options), std::invalid_argument);

    Problem2D bad_phi0 = valid_problem;
    bad_phi0.phi0 = Grid2D{valid_problem.phi0.size() + 1};
    EXPECT_THROW(solve_jacobi(bad_phi0, valid_options), std::invalid_argument);
    EXPECT_THROW(solve_gs(bad_phi0, valid_options), std::invalid_argument);
    EXPECT_THROW(solve_sor(bad_phi0, valid_options), std::invalid_argument);

    Problem2D bad_n = valid_problem;
    bad_n.interior_n = 0;
    EXPECT_THROW(solve_sor(bad_n, valid_options), std::invalid_argument);
}

TEST(InvalidInputTest, MgRejectsBadOptionsAndProblems) {
    const Problem2D valid_problem = make_problem(Case2D::Sine, 7);

    const MGOptions exact_options = make_mg_options(MGCycle::V);
    const MGOptions sor_options = make_mg_options(MGCycle::V, 16, 100);

    EXPECT_THROW(solve_mg_exact(valid_problem, MGOptions{0.0, 100, 2, MGCycle::V, 16}), std::invalid_argument);
    EXPECT_THROW(solve_mg_exact(valid_problem, MGOptions{-1e-10, 100, 2, MGCycle::V, 16}), std::invalid_argument);
    EXPECT_THROW(solve_mg_exact(valid_problem, MGOptions{1e-10, 0, 2, MGCycle::V, 16}), std::invalid_argument);
    EXPECT_THROW(solve_mg_exact(valid_problem, MGOptions{1e-10, 100, 0, MGCycle::V, 16}), std::invalid_argument);

    EXPECT_THROW(solve_mg_sor(valid_problem, MGOptions{0.0, 100, 2, MGCycle::V, 16}), std::invalid_argument);
    EXPECT_THROW(solve_mg_sor(valid_problem, MGOptions{1e-10, 0, 2, MGCycle::V, 16}), std::invalid_argument);
    EXPECT_THROW(solve_mg_sor(valid_problem, MGOptions{1e-10, 100, 0, MGCycle::V, 16}), std::invalid_argument);
    EXPECT_THROW(solve_mg_sor(valid_problem, MGOptions{1e-10, 100, 2, MGCycle::V, 0}), std::invalid_argument);

    Problem2D bad_h = valid_problem;
    bad_h.h = 0.0;
    EXPECT_THROW(solve_mg_exact(bad_h, exact_options), std::invalid_argument);
    EXPECT_THROW(solve_mg_sor(bad_h, sor_options), std::invalid_argument);

    Problem2D bad_rhs = valid_problem;
    bad_rhs.rhs = Grid2D{valid_problem.rhs.size() + 1};
    EXPECT_THROW(solve_mg_exact(bad_rhs, exact_options), std::invalid_argument);
    EXPECT_THROW(solve_mg_sor(bad_rhs, sor_options), std::invalid_argument);

    Problem2D bad_phi0 = valid_problem;
    bad_phi0.phi0 = Grid2D{valid_problem.phi0.size() + 1};
    EXPECT_THROW(solve_mg_exact(bad_phi0, exact_options), std::invalid_argument);
    EXPECT_THROW(solve_mg_sor(bad_phi0, sor_options), std::invalid_argument);

    Problem2D bad_n = valid_problem;
    bad_n.interior_n = 0;
    EXPECT_THROW(solve_mg_exact(bad_n, exact_options), std::invalid_argument);
    EXPECT_THROW(solve_mg_sor(bad_n, sor_options), std::invalid_argument);
}

} // namespace
