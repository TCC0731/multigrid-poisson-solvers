#pragma once

#include "poisson/solver.hpp"

namespace poisson {

[[nodiscard]] SolveResult solve_jacobi(const Problem2D& problem, const SolveOptions& options);

class JacobiSolver2D final : public ISolver2D {
public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] SolveResult solve(
        const Problem2D& problem, const SolveOptions& options
    ) const override;
};

} // namespace poisson
