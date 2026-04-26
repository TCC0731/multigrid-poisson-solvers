#pragma once

#include <cstddef>
#include <string_view>
#include <type_traits>
#include <utility>

#include "poisson/problem.hpp"

namespace poisson {

template <typename Real>
struct SolveOptions {
    Real tol{Real{1e-10}};
    std::size_t max_iter{10'000};
};

struct SolveResult {
    Grid2D<double> phi{};
    std::size_t iterations{0};
    double residual_l2{0.0};
};

template <typename Real>
[[nodiscard]] SolveResult make_solve_result(
    Grid2D<Real>&& phi, std::size_t iterations, Real residual_l2
) {
    if constexpr (std::is_same_v<Real, double>) {
        return {std::move(phi), iterations, static_cast<double>(residual_l2)};
    } else {
        return {cast_grid<double>(phi), iterations, static_cast<double>(residual_l2)};
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

} // namespace poisson
