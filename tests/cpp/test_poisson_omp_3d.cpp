// Reuse the full 3D Poisson test suite against the OpenMP build.
#include <vector>

#include "test_poisson_3d.cpp"

TEST(OpenMPResidualHistory3DTest, SorRecordsResidualHistory) {
    using Real = double;

    const auto problem = poisson::make_problem_3d<Real>("sine", 7);
    std::vector<Real> residual_history;
    poisson::SolveOptions<Real> options{};
    options.tol = Real{1e-8};
    options.max_iter = 2'000;
    options.residual_history = &residual_history;

    const auto result = poisson::solve_sor<Real>(problem, options);

    ASSERT_FALSE(residual_history.empty());
    EXPECT_EQ(residual_history.size(), result.iterations + 1);
    EXPECT_GE(residual_history.front(), residual_history.back());
    EXPECT_NEAR(residual_history.back(), result.residual_l2, 1e-12);
}

TEST(OpenMPResidualHistory3DTest, MgExactRecordsResidualHistory) {
    using Real = double;

    const auto problem = poisson::make_problem_3d<Real>("sine", 15);
    std::vector<Real> residual_history;
    poisson::MGOptions<Real> options{};
    options.tol = Real{1e-10};
    options.max_iter = 20;
    options.nu = 2;
    options.cycle = poisson::MGCycle::V;
    options.residual_history = &residual_history;

    const auto result = poisson::solve_mg_exact<Real>(problem, options);

    ASSERT_FALSE(residual_history.empty());
    EXPECT_EQ(residual_history.size(), result.iterations + 1);
    EXPECT_GE(residual_history.front(), residual_history.back());
    EXPECT_NEAR(residual_history.back(), result.residual_l2, 1e-12);
}
