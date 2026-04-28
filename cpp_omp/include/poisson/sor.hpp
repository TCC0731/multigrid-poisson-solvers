#pragma once

#include "poisson/red_black.hpp"

namespace poisson {

template <typename Real>
[[nodiscard]] SolveResult solve_sor(const Problem2D<Real>& problem, const SolveOptions<Real>& options);

class SorSolver2D final : public ISolver2D {
public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] SolveResult solve(
        const Problem2D<double>& problem, const SolveOptions<double>& options
    ) const override;
};

} // namespace poisson
