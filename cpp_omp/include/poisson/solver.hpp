#pragma once

#include <cstddef>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "poisson/problem.hpp"

namespace poisson {

template <typename Real>
struct SolveOptions {
    Real tol{Real{1e-10}};
    std::size_t max_iter{10'000};
    std::vector<Real>* residual_history{nullptr};
};

struct SolveResult {
    Grid2D<double> phi{};
    std::size_t iterations{0};
    double residual_l2{0.0};
    double benchmark_compute_time_ms{-1.0};
    double benchmark_including_graph_time_ms{-1.0};
};

struct SolveResult3D {
    Grid3D<double> phi{};
    std::size_t iterations{0};
    double residual_l2{0.0};
    double benchmark_compute_time_ms{-1.0};
    double benchmark_including_graph_time_ms{-1.0};
};

template <typename Real>
[[nodiscard]] SolveResult make_solve_result(
    Grid2D<Real>&& phi,
    std::size_t iterations,
    Real residual_l2,
    double benchmark_compute_time_ms = -1.0,
    double benchmark_including_graph_time_ms = -1.0
) {
    if constexpr (std::is_same_v<Real, double>) {
        return {
            std::move(phi),
            iterations,
            static_cast<double>(residual_l2),
            benchmark_compute_time_ms,
            benchmark_including_graph_time_ms,
        };
    } else {
        return {
            cast_grid<double>(phi),
            iterations,
            static_cast<double>(residual_l2),
            benchmark_compute_time_ms,
            benchmark_including_graph_time_ms,
        };
    }
}

template <typename Real>
[[nodiscard]] SolveResult3D make_solve_result(
    Grid3D<Real>&& phi,
    std::size_t iterations,
    Real residual_l2,
    double benchmark_compute_time_ms = -1.0,
    double benchmark_including_graph_time_ms = -1.0
) {
    if constexpr (std::is_same_v<Real, double>) {
        return {
            std::move(phi),
            iterations,
            static_cast<double>(residual_l2),
            benchmark_compute_time_ms,
            benchmark_including_graph_time_ms,
        };
    } else {
        return {
            cast_grid<double>(phi),
            iterations,
            static_cast<double>(residual_l2),
            benchmark_compute_time_ms,
            benchmark_including_graph_time_ms,
        };
    }
}

class ISolver2D {
public:
    virtual ~ISolver2D() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual SolveResult solve(
        const Problem2D<double>& problem, const SolveOptions<double>& options
    ) const = 0;
};

class ISolver3D {
public:
    virtual ~ISolver3D() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual SolveResult3D solve(
        const Problem3D<double>& problem, const SolveOptions<double>& options
    ) const = 0;
};

} // namespace poisson
