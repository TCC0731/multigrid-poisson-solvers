#pragma once

#include "poisson/red_black.hpp"

namespace poisson {

[[nodiscard]] SolveResult solve_gs(const Problem2D& problem, const SolveOptions& options);

class GaussSeidelSolver2D final : public ISolver2D {
public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] SolveResult solve(
        const Problem2D& problem, const SolveOptions& options
    ) const override;
};

} // namespace poisson
