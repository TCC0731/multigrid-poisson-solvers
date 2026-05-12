#include "poisson/gs.hpp"
#include "poisson/jacobi.hpp"
#include "poisson/metrics.hpp"
#include "poisson/mg.hpp"
#include "poisson/operators.hpp"
#include "poisson/problem.hpp"
#include "poisson/sor.hpp"
#include "poisson/validation.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <type_traits>

namespace {

using namespace poisson;

void skip_without_cuda_device() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count < 1) {
        GTEST_SKIP() << "CUDA device is not available";
    }
}

template <typename Real>
constexpr Real solver_tol() {
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
Grid3D<Real> cast_solution(const SolveResult3D& result) {
    return cast_grid<Real>(result.phi);
}

template <typename Real, typename SolveFn, typename Options>
void expect_cuda_solver_converges(SolveFn&& solve, const Problem3D<Real>& problem, const Options& options) {
    skip_without_cuda_device();

    const SolveResult3D result = solve(problem, options);
    const Grid3D<Real> phi = cast_solution<Real>(result);
    const Real recomputed_residual = relative_physical_residual_l2(problem, phi);
    const ErrorMetrics metrics = error_metrics(problem, phi);

    EXPECT_EQ(result.phi.size(), problem.phi0.size());
    EXPECT_GT(result.iterations, 0u);
    EXPECT_LE(result.iterations, options.max_iter);
    EXPECT_TRUE(std::isfinite(result.residual_l2));
    EXPECT_LE(result.residual_l2, static_cast<double>(options.tol));
    EXPECT_NEAR(result.residual_l2, static_cast<double>(recomputed_residual), residual_compare_tol<Real>());
    EXPECT_LT(metrics.error_l2, 1e-1);
    EXPECT_LT(metrics.error_linf, 2e-1);
}

template <typename Real>
void run_problem_generation_and_stencil() {
    const Problem3D<Real> problem = make_problem_3d<Real>(Case3D::Sine, 7);
    const ValidationReport report = validate_problem(problem);

    EXPECT_TRUE(report.ok);
    EXPECT_EQ(problem.array_n(), 9u);
    EXPECT_DOUBLE_EQ(static_cast<double>(problem.h), 1.0 / 8.0);

    Grid3D<Real> phi{5};
    phi.fill(Real{});
    phi(2, 2, 2) = Real{1};
    const Grid3D<Real> aphi = apply_A(phi, Real{1});
    EXPECT_DOUBLE_EQ(static_cast<double>(aphi(1, 1, 1)), 6.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(aphi(0, 1, 1)), -1.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(aphi(1, 0, 1)), -1.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(aphi(1, 1, 0)), -1.0);
}

template <typename Real>
void run_jacobi_and_rb_solvers() {
    const Problem3D<Real> problem = make_problem_3d<Real>(Case3D::Sine, 7);
    const SolveOptions<Real> jacobi_options{solver_tol<Real>(), 20'000};
    const SolveOptions<Real> rb_options{solver_tol<Real>(), 10'000};

    expect_cuda_solver_converges<Real>(
        [](const auto& p, const auto& o) { return solve_jacobi<Real>(p, o); },
        problem,
        jacobi_options
    );
    expect_cuda_solver_converges<Real>(
        [](const auto& p, const auto& o) { return solve_gs<Real>(p, o); },
        problem,
        rb_options
    );
    expect_cuda_solver_converges<Real>(
        [](const auto& p, const auto& o) { return solve_sor<Real>(p, o); },
        problem,
        rb_options
    );
}

template <typename Real>
void run_mg_v_and_w_cycles() {
    const Problem3D<Real> problem = make_problem_3d<Real>(Case3D::Sine, 15);
    const MGOptions<Real> v_exact{solver_tol<Real>(), 80, 2, MGCycle::V, 32, Real{1}, false};
    const MGOptions<Real> w_sor{solver_tol<Real>(), 100, 2, MGCycle::W, 64, Real{1}, false};

    expect_cuda_solver_converges<Real>(
        [](const auto& p, const auto& o) { return solve_mg_exact<Real>(p, o); },
        problem,
        v_exact
    );
    expect_cuda_solver_converges<Real>(
        [](const auto& p, const auto& o) { return solve_mg_sor<Real>(p, o); },
        problem,
        w_sor
    );
}

TEST(Cuda3DProblemFloat, ProblemGenerationAndStencil) {
    run_problem_generation_and_stencil<float>();
}

TEST(Cuda3DProblemDouble, ProblemGenerationAndStencil) {
    run_problem_generation_and_stencil<double>();
}

TEST(Cuda3DSolversFloat, JacobiGsSorConverge) {
    run_jacobi_and_rb_solvers<float>();
}

TEST(Cuda3DSolversDouble, JacobiGsSorConverge) {
    run_jacobi_and_rb_solvers<double>();
}

TEST(Cuda3DMgFloat, VExactAndWSorConverge) {
    run_mg_v_and_w_cycles<float>();
}

TEST(Cuda3DMgDouble, VExactAndWSorConverge) {
    run_mg_v_and_w_cycles<double>();
}

} // namespace
