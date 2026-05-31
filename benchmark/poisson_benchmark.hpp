#pragma once

#include "poisson/gs.hpp"
#include "poisson/jacobi.hpp"
#include "poisson/metrics.hpp"
#include "poisson/mg.hpp"
#include "poisson/problem.hpp"
#include "poisson/sor.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cctype>
#include <iomanip>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace poisson::benchmark {
namespace detail {

inline constexpr std::string_view kCaseName{"sine"};
inline constexpr std::string_view kSolverComparisonSuite{"solver_comparison"};
inline constexpr std::string_view kMgCompareSuite{"mg_compare"};
inline constexpr std::size_t kWarmupRuns{1};
inline constexpr std::size_t kTimedRuns{50};
inline constexpr std::size_t kWarmupMaxIter{10};
inline constexpr double kTol{1e-9};
inline constexpr double kMgOmega{1.25};
inline constexpr std::size_t kMgNu{3};
inline constexpr std::size_t kMgCoarseSteps{16};
inline constexpr std::size_t kMgMaxIter{150};
inline constexpr std::array<std::size_t, 4> kSolverSizes{15, 31, 63, 127};
inline constexpr std::array<std::size_t, 6> kRbSorSizes{15, 31, 63, 127, 256, 512};
inline constexpr std::array<std::size_t, 9> kMgSizes{15, 31, 63, 127, 255, 511, 1023, 2047, 4095};
inline constexpr std::array<std::size_t, 4> kSolverSizes3D{15, 31, 47, 63};
inline constexpr std::array<std::size_t, 5> kRbSorSizes3D{31, 63, 95, 127, 159};
inline constexpr std::array<std::size_t, 6> kMgSizes3D{15, 31, 63, 127, 255, 383};

inline bool size_allowed(std::size_t grid_size, std::size_t max_grid_size) noexcept {
    return max_grid_size == 0 || grid_size <= max_grid_size;
}

template <typename Real>
constexpr std::string_view dtype_name() noexcept {
    if constexpr (std::is_same_v<Real, float>) {
        return "float";
    }
    return "double";
}

inline std::string normalize_token(std::string_view value) {
    std::string normalized;
    normalized.reserve(value.size());
    for (const char ch : value) {
        if (ch == '-') {
            normalized.push_back('_');
        } else {
            normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
    }
    return normalized;
}

inline std::string csv_escape(std::string_view value) {
    const bool needs_quotes = value.find_first_of(",\"\n\r") != std::string_view::npos;
    if (!needs_quotes) {
        return std::string(value);
    }

    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char ch : value) {
        if (ch == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}

inline void write_escaped(std::ostream& os, std::string_view value) {
    os << csv_escape(value);
}

inline void write_scientific(std::ostream& os, double value, int precision = 6) {
    const auto old_flags = os.flags();
    const auto old_precision = os.precision();
    os.setf(std::ios::scientific, std::ios::floatfield);
    os.precision(precision);
    os << value;
    os.flags(old_flags);
    os.precision(old_precision);
}

inline void write_fixed(std::ostream& os, double value, int precision = 6) {
    const auto old_flags = os.flags();
    const auto old_precision = os.precision();
    os.setf(std::ios::fixed, std::ios::floatfield);
    os.precision(precision);
    os << value;
    os.flags(old_flags);
    os.precision(old_precision);
}

} // namespace detail

enum class Suite {
    SolverComparison,
    MgCompare,
    All,
};

[[nodiscard]] inline Suite parse_suite(std::string_view suite_name) {
    const std::string normalized = detail::normalize_token(suite_name);
    if (normalized == "all") {
        return Suite::All;
    }
    if (normalized == "solver" || normalized == "solver_comparison") {
        return Suite::SolverComparison;
    }
    if (normalized == "mg" || normalized == "mg_compare") {
        return Suite::MgCompare;
    }
    throw std::invalid_argument("unknown benchmark suite: " + std::string(suite_name));
}

[[nodiscard]] inline std::string_view to_string(Suite suite) noexcept {
    switch (suite) {
    case Suite::SolverComparison:
        return detail::kSolverComparisonSuite;
    case Suite::MgCompare:
        return detail::kMgCompareSuite;
    case Suite::All:
        return "all";
    }
    return "unknown";
}

struct TimingStats {
    double mean_ms{0.0};
    double std_ms{0.0};
};

struct SolverTimingStats {
    TimingStats benchmark_ms{};
    TimingStats including_graph_ms{};
};

struct BenchmarkRow {
    std::string suite;
    std::string backend;
    std::string dtype;
    std::string case_name;
    std::string solver;
    std::size_t grid_size{0};
    std::size_t max_iter{0};
    double tol{0.0};
    std::size_t warmup_runs{0};
    std::size_t timed_runs{0};
    std::string cycle;
    std::string omega;
    std::string nu;
    std::size_t iterations{0};
    double residual_l2{0.0};
    double error_l2{0.0};
    double error_linf{0.0};
    double mean_time_ms{0.0};
    double std_time_ms{0.0};
    double mean_time_including_graph_ms{0.0};
    double std_time_including_graph_ms{0.0};
};

// Mirror the solver binaries: warm up first, then average repeated timed runs.
struct BenchmarkTiming {
    std::size_t warmup_runs{detail::kWarmupRuns};
    std::size_t timed_runs{detail::kTimedRuns};
};

[[nodiscard]] inline TimingStats summarize_samples(const std::vector<double>& samples) {
    double mean = 0.0;
    for (const double sample : samples) {
        mean += sample;
    }
    mean /= static_cast<double>(samples.size());

    double variance = 0.0;
    if (samples.size() > 1) {
        for (const double sample : samples) {
            const double diff = sample - mean;
            variance += diff * diff;
        }
        variance /= static_cast<double>(samples.size() - 1);
    }

    return {mean, std::sqrt(variance)};
}

template <typename Result>
[[nodiscard]] double benchmark_sample_time_ms(const Result& result, double fallback_ms) {
    return result.benchmark_compute_time_ms >= 0.0 ? result.benchmark_compute_time_ms : fallback_ms;
}

template <typename Result>
[[nodiscard]] double including_graph_sample_time_ms(const Result& result, double fallback_ms) {
    return result.benchmark_including_graph_time_ms >= 0.0
        ? result.benchmark_including_graph_time_ms
        : fallback_ms;
}

template <typename WarmupSolverFn, typename TimedSolverFn>
[[nodiscard]] auto time_solver(
    WarmupSolverFn&& warmup_solver,
    TimedSolverFn&& timed_solver,
    const BenchmarkTiming& timing
) {
    if (timing.timed_runs < 1) {
        throw std::invalid_argument("timed_runs must be positive");
    }

    for (std::size_t i = 0; i < timing.warmup_runs; ++i) {
        (void)warmup_solver();
    }

    std::vector<double> samples;
    samples.reserve(timing.timed_runs);
    std::vector<double> including_graph_samples;
    including_graph_samples.reserve(timing.timed_runs);
    using Result = std::invoke_result_t<TimedSolverFn&>;
    std::optional<Result> last_result;
    double total_benchmark_time_ms = 0.0;
    double total_including_graph_time_ms = 0.0;

    for (std::size_t i = 0; i < timing.timed_runs; ++i) {
        const auto start = std::chrono::steady_clock::now();
        last_result = timed_solver();
        const auto end = std::chrono::steady_clock::now();
        const double full_sample_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        const double benchmark_time_ms = benchmark_sample_time_ms(*last_result, full_sample_ms);
        const double including_graph_time_ms =
            including_graph_sample_time_ms(*last_result, full_sample_ms);
        samples.push_back(benchmark_time_ms);
        including_graph_samples.push_back(including_graph_time_ms);
        total_benchmark_time_ms += benchmark_time_ms;
        total_including_graph_time_ms += including_graph_time_ms;
    }

    TimingStats benchmark_stats = summarize_samples(samples);
    TimingStats including_graph_stats = summarize_samples(including_graph_samples);
    benchmark_stats.mean_ms = total_benchmark_time_ms / static_cast<double>(timing.timed_runs);
    including_graph_stats.mean_ms =
        total_including_graph_time_ms / static_cast<double>(timing.timed_runs);

    return std::pair<Result, SolverTimingStats>{
        std::move(*last_result),
        SolverTimingStats{
            benchmark_stats,
            including_graph_stats,
        },
    };
}

template <typename Real, typename WarmupSolverFn, typename TimedSolverFn>
[[nodiscard]] BenchmarkRow make_row(
    std::string_view suite,
    std::string_view backend,
    std::string_view solver_name,
    std::size_t grid_size,
    std::size_t max_iter,
    Real tol,
    std::string_view cycle,
    std::string_view omega,
    std::string_view nu,
    const BenchmarkTiming& timing,
    WarmupSolverFn&& warmup_solver_fn,
    TimedSolverFn&& timed_solver_fn
) {
    const auto problem = poisson::make_problem<Real>(poisson::Case2D::Sine, grid_size);
    auto warmup_solver = [&]() {
        return warmup_solver_fn(problem);
    };
    auto timed_solver = [&]() {
        return timed_solver_fn(problem);
    };

    const auto [result, stats] = time_solver(warmup_solver, timed_solver, timing);
    const auto metrics_problem = poisson::make_problem<double>(poisson::Case2D::Sine, grid_size);
    const auto metrics = poisson::metrics(metrics_problem, result.phi);

    return {
        std::string(suite),
        std::string(backend),
        std::string(detail::dtype_name<Real>()),
        std::string(detail::kCaseName),
        std::string(solver_name),
        grid_size,
        max_iter,
        static_cast<double>(tol),
        timing.warmup_runs,
        timing.timed_runs,
        std::string(cycle),
        std::string(omega),
        std::string(nu),
        result.iterations,
        result.residual_l2,
        metrics.error_l2,
        metrics.error_linf,
        stats.benchmark_ms.mean_ms,
        stats.benchmark_ms.std_ms,
        stats.including_graph_ms.mean_ms,
        stats.including_graph_ms.std_ms,
    };
}

#ifdef POISSON_ENABLE_3D
template <typename Real, typename WarmupSolverFn, typename TimedSolverFn>
[[nodiscard]] BenchmarkRow make_row_3d(
    std::string_view suite,
    std::string_view backend,
    std::string_view solver_name,
    std::size_t grid_size,
    std::size_t max_iter,
    Real tol,
    std::string_view cycle,
    std::string_view omega,
    std::string_view nu,
    const BenchmarkTiming& timing,
    WarmupSolverFn&& warmup_solver_fn,
    TimedSolverFn&& timed_solver_fn
) {
    const auto problem = poisson::make_problem_3d<Real>(poisson::Case3D::Sine, grid_size);
    auto warmup_solver = [&]() {
        return warmup_solver_fn(problem);
    };
    auto timed_solver = [&]() {
        return timed_solver_fn(problem);
    };

    const auto [result, stats] = time_solver(warmup_solver, timed_solver, timing);
    const auto metrics_problem = poisson::make_problem_3d<double>(poisson::Case3D::Sine, grid_size);
    const auto metrics = poisson::metrics(metrics_problem, result.phi);

    return {
        std::string(suite),
        std::string(backend),
        std::string(detail::dtype_name<Real>()),
        std::string(detail::kCaseName),
        std::string(solver_name),
        grid_size,
        max_iter,
        static_cast<double>(tol),
        timing.warmup_runs,
        timing.timed_runs,
        std::string(cycle),
        std::string(omega),
        std::string(nu),
        result.iterations,
        result.residual_l2,
        metrics.error_l2,
        metrics.error_linf,
        stats.benchmark_ms.mean_ms,
        stats.benchmark_ms.std_ms,
        stats.including_graph_ms.mean_ms,
        stats.including_graph_ms.std_ms,
    };
}
#endif

template <typename Real, typename SolverFn>
void append_mg_rows(
    std::vector<BenchmarkRow>& rows,
    std::string_view backend,
    std::string_view solver_name,
    std::string_view cycle,
    std::string_view omega,
    std::string_view nu,
    std::size_t max_iter,
    Real tol,
    const poisson::MGOptions<Real>& warmup_opts,
    const poisson::MGOptions<Real>& timed_opts,
    std::size_t max_grid_size,
    const BenchmarkTiming& timing,
    SolverFn&& solve_fn
) {
    for (const std::size_t grid_size : detail::kMgSizes) {
        if (!detail::size_allowed(grid_size, max_grid_size)) {
            continue;
        }
        rows.push_back(make_row<Real>(
            detail::kMgCompareSuite,
            backend,
            solver_name,
            grid_size,
            max_iter,
            tol,
            cycle,
            omega,
            nu,
            timing,
            [&](const auto& problem) {
                return solve_fn(problem, warmup_opts);
            },
            [&](const auto& problem) {
                return solve_fn(problem, timed_opts);
            }
        ));
    }
}

#ifdef POISSON_ENABLE_3D
template <typename Real, typename SolverFn>
void append_mg_rows_3d(
    std::vector<BenchmarkRow>& rows,
    std::string_view backend,
    std::string_view solver_name,
    std::string_view cycle,
    std::string_view omega,
    std::string_view nu,
    std::size_t max_iter,
    Real tol,
    const poisson::MGOptions<Real>& warmup_opts,
    const poisson::MGOptions<Real>& timed_opts,
    std::size_t max_grid_size,
    const BenchmarkTiming& timing,
    SolverFn&& solve_fn
) {
    for (const std::size_t grid_size : detail::kMgSizes3D) {
        if (!detail::size_allowed(grid_size, max_grid_size)) {
            continue;
        }
        rows.push_back(make_row_3d<Real>(
            detail::kMgCompareSuite,
            backend,
            solver_name,
            grid_size,
            max_iter,
            tol,
            cycle,
            omega,
            nu,
            timing,
            [&](const auto& problem) {
                return solve_fn(problem, warmup_opts);
            },
            [&](const auto& problem) {
                return solve_fn(problem, timed_opts);
            }
        ));
    }
}
#endif

template <typename Real>
[[nodiscard]] std::vector<BenchmarkRow> make_solver_comparison_rows(
    std::string_view backend,
    std::size_t max_grid_size = 0,
    const BenchmarkTiming& timing = BenchmarkTiming{}
) {
    std::vector<BenchmarkRow> rows;
    rows.reserve(14);

    const Real tol = static_cast<Real>(detail::kTol);
    const poisson::SolveOptions<Real> jacobi_opts{tol, 100000};
    const poisson::SolveOptions<Real> jacobi_warmup_opts{tol, detail::kWarmupMaxIter};
    const poisson::SolveOptions<Real> gs_opts{tol, 100000};
    const poisson::SolveOptions<Real> gs_warmup_opts{tol, detail::kWarmupMaxIter};
    const poisson::SolveOptions<Real> sor_opts{tol, 10000};
    const poisson::SolveOptions<Real> sor_warmup_opts{tol, detail::kWarmupMaxIter};

    for (const std::size_t grid_size : detail::kSolverSizes) {
        if (!detail::size_allowed(grid_size, max_grid_size)) {
            continue;
        }
        rows.push_back(make_row<Real>(
            detail::kSolverComparisonSuite,
            backend,
            "Jacobi",
            grid_size,
            100000,
            tol,
            "",
            "",
            "",
            timing,
            [&](const auto& problem) {
                return poisson::solve_jacobi<Real>(problem, jacobi_warmup_opts);
            },
            [&](const auto& problem) {
                return poisson::solve_jacobi<Real>(problem, jacobi_opts);
            }
        ));
    }

    for (const std::size_t grid_size : detail::kSolverSizes) {
        if (!detail::size_allowed(grid_size, max_grid_size)) {
            continue;
        }
        rows.push_back(make_row<Real>(
            detail::kSolverComparisonSuite,
            backend,
            "RB GS",
            grid_size,
            100000,
            tol,
            "",
            "",
            "",
            timing,
            [&](const auto& problem) {
                return poisson::solve_gs<Real>(problem, gs_warmup_opts);
            },
            [&](const auto& problem) {
                return poisson::solve_gs<Real>(problem, gs_opts);
            }
        ));
    }

    for (const std::size_t grid_size : detail::kRbSorSizes) {
        if (!detail::size_allowed(grid_size, max_grid_size)) {
            continue;
        }
        rows.push_back(make_row<Real>(
            detail::kSolverComparisonSuite,
            backend,
            "RB SOR",
            grid_size,
            10000,
            tol,
            "",
            "",
            "",
            timing,
            [&](const auto& problem) {
                return poisson::solve_sor<Real>(problem, sor_warmup_opts);
            },
            [&](const auto& problem) {
                return poisson::solve_sor<Real>(problem, sor_opts);
            }
        ));
    }

    return rows;
}

template <typename Real>
[[nodiscard]] std::vector<BenchmarkRow> make_mg_compare_rows(
    std::string_view backend,
    std::size_t max_grid_size = 0,
    const BenchmarkTiming& timing = BenchmarkTiming{}
) {
    std::vector<BenchmarkRow> rows;
    rows.reserve(36);

    const Real tol = static_cast<Real>(detail::kTol);
    const poisson::MGOptions<Real> mg_v_warmup_opts{
        tol,
        detail::kWarmupMaxIter,
        detail::kMgNu,
        poisson::MGCycle::V,
        detail::kMgCoarseSteps,
        static_cast<Real>(detail::kMgOmega),
        false,
    };
    const poisson::MGOptions<Real> mg_v_opts{
        tol,
        detail::kMgMaxIter,
        detail::kMgNu,
        poisson::MGCycle::V,
        detail::kMgCoarseSteps,
        static_cast<Real>(detail::kMgOmega),
        false,
    };
    const poisson::MGOptions<Real> mg_w_warmup_opts{
        tol,
        detail::kWarmupMaxIter,
        detail::kMgNu,
        poisson::MGCycle::W,
        detail::kMgCoarseSteps,
        static_cast<Real>(detail::kMgOmega),
        false,
    };
    const poisson::MGOptions<Real> mg_w_opts{
        tol,
        detail::kMgMaxIter,
        detail::kMgNu,
        poisson::MGCycle::W,
        detail::kMgCoarseSteps,
        static_cast<Real>(detail::kMgOmega),
        false,
    };

    append_mg_rows(
        rows,
        backend,
        "MG(v,w=1.25,coarse=exact)",
        "v",
        "1.25",
        "3",
        detail::kMgMaxIter,
        tol,
        mg_v_warmup_opts,
        mg_v_opts,
        max_grid_size,
        timing,
        [&](const auto& problem, const auto& options) {
            return poisson::solve_mg_exact<Real>(problem, options);
        }
    );
    append_mg_rows(
        rows,
        backend,
        "MG(v,w=1.25,coarse=sor)",
        "v",
        "1.25",
        "3",
        detail::kMgMaxIter,
        tol,
        mg_v_warmup_opts,
        mg_v_opts,
        max_grid_size,
        timing,
        [&](const auto& problem, const auto& options) {
            return poisson::solve_mg_sor<Real>(problem, options);
        }
    );
    append_mg_rows(
        rows,
        backend,
        "MG(w,w=1.25,coarse=exact)",
        "w",
        "1.25",
        "3",
        detail::kMgMaxIter,
        tol,
        mg_w_warmup_opts,
        mg_w_opts,
        max_grid_size,
        timing,
        [&](const auto& problem, const auto& options) {
            return poisson::solve_mg_exact<Real>(problem, options);
        }
    );
    append_mg_rows(
        rows,
        backend,
        "MG(w,w=1.25,coarse=sor)",
        "w",
        "1.25",
        "3",
        detail::kMgMaxIter,
        tol,
        mg_w_warmup_opts,
        mg_w_opts,
        max_grid_size,
        timing,
        [&](const auto& problem, const auto& options) {
            return poisson::solve_mg_sor<Real>(problem, options);
        }
    );

    return rows;
}

#ifdef POISSON_ENABLE_3D
template <typename Real>
[[nodiscard]] std::vector<BenchmarkRow> make_solver_comparison_rows_3d(
    std::string_view backend,
    std::size_t max_grid_size = 0,
    const BenchmarkTiming& timing = BenchmarkTiming{}
) {
    std::vector<BenchmarkRow> rows;
    rows.reserve(13);

    const Real tol = static_cast<Real>(detail::kTol);
    const poisson::SolveOptions<Real> jacobi_opts{tol, 100000};
    const poisson::SolveOptions<Real> jacobi_warmup_opts{tol, detail::kWarmupMaxIter};
    const poisson::SolveOptions<Real> gs_opts{tol, 100000};
    const poisson::SolveOptions<Real> gs_warmup_opts{tol, detail::kWarmupMaxIter};
    const poisson::SolveOptions<Real> sor_opts{tol, 10000};
    const poisson::SolveOptions<Real> sor_warmup_opts{tol, detail::kWarmupMaxIter};

    for (const std::size_t grid_size : detail::kSolverSizes3D) {
        if (!detail::size_allowed(grid_size, max_grid_size)) {
            continue;
        }
        rows.push_back(make_row_3d<Real>(
            detail::kSolverComparisonSuite,
            backend,
            "Jacobi 3D",
            grid_size,
            100000,
            tol,
            "",
            "",
            "",
            timing,
            [&](const auto& problem) {
                return poisson::solve_jacobi<Real>(problem, jacobi_warmup_opts);
            },
            [&](const auto& problem) {
                return poisson::solve_jacobi<Real>(problem, jacobi_opts);
            }
        ));
    }

    for (const std::size_t grid_size : detail::kSolverSizes3D) {
        if (!detail::size_allowed(grid_size, max_grid_size)) {
            continue;
        }
        rows.push_back(make_row_3d<Real>(
            detail::kSolverComparisonSuite,
            backend,
            "RB GS 3D",
            grid_size,
            100000,
            tol,
            "",
            "",
            "",
            timing,
            [&](const auto& problem) {
                return poisson::solve_gs<Real>(problem, gs_warmup_opts);
            },
            [&](const auto& problem) {
                return poisson::solve_gs<Real>(problem, gs_opts);
            }
        ));
    }

    for (const std::size_t grid_size : detail::kRbSorSizes3D) {
        if (!detail::size_allowed(grid_size, max_grid_size)) {
            continue;
        }
        rows.push_back(make_row_3d<Real>(
            detail::kSolverComparisonSuite,
            backend,
            "RB SOR 3D",
            grid_size,
            10000,
            tol,
            "",
            "",
            "",
            timing,
            [&](const auto& problem) {
                return poisson::solve_sor<Real>(problem, sor_warmup_opts);
            },
            [&](const auto& problem) {
                return poisson::solve_sor<Real>(problem, sor_opts);
            }
        ));
    }

    return rows;
}

template <typename Real>
[[nodiscard]] std::vector<BenchmarkRow> make_mg_compare_rows_3d(
    std::string_view backend,
    std::size_t max_grid_size = 0,
    const BenchmarkTiming& timing = BenchmarkTiming{}
) {
    std::vector<BenchmarkRow> rows;
    rows.reserve(24);

    const Real tol = static_cast<Real>(detail::kTol);
    const poisson::MGOptions<Real> mg_v_warmup_opts{
        tol,
        detail::kWarmupMaxIter,
        detail::kMgNu,
        poisson::MGCycle::V,
        detail::kMgCoarseSteps,
        static_cast<Real>(detail::kMgOmega),
        false,
    };
    const poisson::MGOptions<Real> mg_v_opts{
        tol,
        detail::kMgMaxIter,
        detail::kMgNu,
        poisson::MGCycle::V,
        detail::kMgCoarseSteps,
        static_cast<Real>(detail::kMgOmega),
        false,
    };
    const poisson::MGOptions<Real> mg_w_warmup_opts{
        tol,
        detail::kWarmupMaxIter,
        detail::kMgNu,
        poisson::MGCycle::W,
        detail::kMgCoarseSteps,
        static_cast<Real>(detail::kMgOmega),
        false,
    };
    const poisson::MGOptions<Real> mg_w_opts{
        tol,
        detail::kMgMaxIter,
        detail::kMgNu,
        poisson::MGCycle::W,
        detail::kMgCoarseSteps,
        static_cast<Real>(detail::kMgOmega),
        false,
    };

    append_mg_rows_3d(
        rows,
        backend,
        "MG 3D(v,w=1.25,coarse=exact)",
        "v",
        "1.25",
        "3",
        detail::kMgMaxIter,
        tol,
        mg_v_warmup_opts,
        mg_v_opts,
        max_grid_size,
        timing,
        [&](const auto& problem, const auto& options) {
            return poisson::solve_mg_exact<Real>(problem, options);
        }
    );
    append_mg_rows_3d(
        rows,
        backend,
        "MG 3D(v,w=1.25,coarse=sor)",
        "v",
        "1.25",
        "3",
        detail::kMgMaxIter,
        tol,
        mg_v_warmup_opts,
        mg_v_opts,
        max_grid_size,
        timing,
        [&](const auto& problem, const auto& options) {
            return poisson::solve_mg_sor<Real>(problem, options);
        }
    );
    append_mg_rows_3d(
        rows,
        backend,
        "MG 3D(w,w=1.25,coarse=exact)",
        "w",
        "1.25",
        "3",
        detail::kMgMaxIter,
        tol,
        mg_w_warmup_opts,
        mg_w_opts,
        max_grid_size,
        timing,
        [&](const auto& problem, const auto& options) {
            return poisson::solve_mg_exact<Real>(problem, options);
        }
    );
    append_mg_rows_3d(
        rows,
        backend,
        "MG 3D(w,w=1.25,coarse=sor)",
        "w",
        "1.25",
        "3",
        detail::kMgMaxIter,
        tol,
        mg_w_warmup_opts,
        mg_w_opts,
        max_grid_size,
        timing,
        [&](const auto& problem, const auto& options) {
            return poisson::solve_mg_sor<Real>(problem, options);
        }
    );

    return rows;
}
#endif

template <typename Real>
[[nodiscard]] std::vector<BenchmarkRow> run_suite(
    Suite suite,
    std::string_view backend,
    std::size_t dimension = 2,
    std::size_t max_grid_size = 0,
    const BenchmarkTiming& timing = BenchmarkTiming{}
) {
    std::vector<BenchmarkRow> rows;

    if (dimension == 3) {
#ifdef POISSON_ENABLE_3D
        if (suite == Suite::SolverComparison || suite == Suite::All) {
            const auto solver_rows = make_solver_comparison_rows_3d<Real>(
                backend, max_grid_size, timing
            );
            rows.insert(rows.end(), solver_rows.begin(), solver_rows.end());
        }

        if (suite == Suite::MgCompare || suite == Suite::All) {
            const auto mg_rows = make_mg_compare_rows_3d<Real>(backend, max_grid_size, timing);
            rows.insert(rows.end(), mg_rows.begin(), mg_rows.end());
        }

        return rows;
#else
        throw std::invalid_argument("3D benchmarks are not enabled for this backend");
#endif
    }

    if (dimension != 2) {
        throw std::invalid_argument("dimension must be 2 or 3");
    }

    if (suite == Suite::SolverComparison || suite == Suite::All) {
        const auto solver_rows =
            make_solver_comparison_rows<Real>(backend, max_grid_size, timing);
        rows.insert(rows.end(), solver_rows.begin(), solver_rows.end());
    }

    if (suite == Suite::MgCompare || suite == Suite::All) {
        const auto mg_rows = make_mg_compare_rows<Real>(backend, max_grid_size, timing);
        rows.insert(rows.end(), mg_rows.begin(), mg_rows.end());
    }

    return rows;
}

inline void write_csv_header(std::ostream& os) {
    os << "suite,backend,dtype,case,solver,grid_size,max_iter,tol,warmup_runs,timed_runs,cycle,omega,nu,iterations,residual_l2,error_l2,error_linf,mean_time_ms,std_time_ms,mean_time_including_graph_ms,std_time_including_graph_ms\n";
}

inline void write_csv_row(std::ostream& os, const BenchmarkRow& row) {
    detail::write_escaped(os, row.suite);
    os << ',';
    detail::write_escaped(os, row.backend);
    os << ',';
    detail::write_escaped(os, row.dtype);
    os << ',';
    detail::write_escaped(os, row.case_name);
    os << ',';
    detail::write_escaped(os, row.solver);
    os << ',';
    os << row.grid_size << ',';
    os << row.max_iter << ',';
    detail::write_scientific(os, row.tol);
    os << ',';
    os << row.warmup_runs << ',';
    os << row.timed_runs << ',';
    detail::write_escaped(os, row.cycle);
    os << ',';
    detail::write_escaped(os, row.omega);
    os << ',';
    detail::write_escaped(os, row.nu);
    os << ',';
    os << row.iterations << ',';
    detail::write_scientific(os, row.residual_l2);
    os << ',';
    detail::write_scientific(os, row.error_l2);
    os << ',';
    detail::write_scientific(os, row.error_linf);
    os << ',';
    detail::write_fixed(os, row.mean_time_ms);
    os << ',';
    detail::write_fixed(os, row.std_time_ms);
    os << ',';
    detail::write_fixed(os, row.mean_time_including_graph_ms);
    os << ',';
    detail::write_fixed(os, row.std_time_including_graph_ms);
    os << '\n';
}

inline void write_csv(std::ostream& os, const std::vector<BenchmarkRow>& rows) {
    write_csv_header(os);
    for (const auto& row : rows) {
        write_csv_row(os, row);
    }
}

inline void write_simple_csv_header(std::ostream& os) {
    os << "solver,grid_size,iterations,mean_time_ms,std_time_ms\n";
}

inline void write_simple_csv_row(std::ostream& os, const BenchmarkRow& row) {
    detail::write_escaped(os, row.solver);
    os << ',';
    os << row.grid_size << ',';
    os << row.iterations << ',';
    detail::write_fixed(os, row.mean_time_ms);
    os << ',';
    detail::write_fixed(os, row.std_time_ms);
    os << '\n';
}

inline void write_simple_csv(std::ostream& os, const std::vector<BenchmarkRow>& rows) {
    write_simple_csv_header(os);
    for (const auto& row : rows) {
        write_simple_csv_row(os, row);
    }
}

} // namespace poisson::benchmark
