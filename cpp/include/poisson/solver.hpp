#pragma once

#include <cstddef>
#include <string_view>

#include "poisson/problem.hpp"

namespace poisson {

struct SolveOptions {
    Real tol{1e-10};
    std::size_t max_iter{10'000};
};

struct SolveResult {
    Grid2D phi{};
    std::size_t iterations{0};
    Real residual_l2{0.0};
};

class ISolver2D {
public:
    virtual ~ISolver2D() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual SolveResult solve(
        const Problem2D& problem, const SolveOptions& options
    ) const = 0;
};

} // namespace poisson

