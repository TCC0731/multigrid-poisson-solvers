#pragma once

#include "poisson/red_black.hpp"

namespace poisson {

template <typename Real>
[[nodiscard]] SolveResult solve_gs(const Problem2D<Real>& problem, const SolveOptions<Real>& options);

template <typename Real>
[[nodiscard]] SolveResult3D solve_gs(const Problem3D<Real>& problem, const SolveOptions<Real>& options);

class GaussSeidelSolver2D final : public ISolver2D {
public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] SolveResult solve(
        const Problem2D<double>& problem, const SolveOptions<double>& options
    ) const override;
};

class GaussSeidelSolver3D final : public ISolver3D {
public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] SolveResult3D solve(
        const Problem3D<double>& problem, const SolveOptions<double>& options
    ) const override;
};

} // namespace poisson
