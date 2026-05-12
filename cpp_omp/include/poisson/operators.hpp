#pragma once

#include "poisson/problem.hpp"

namespace poisson {

template <typename Real>
[[nodiscard]] Grid2D<Real> apply_A(const Grid2D<Real>& phi, Real h);

template <typename Real>
[[nodiscard]] Grid2D<Real> residual(const Grid2D<Real>& phi, const Grid2D<Real>& rhs, Real h);

template <typename Real>
[[nodiscard]] Grid3D<Real> apply_A(const Grid3D<Real>& phi, Real h);

template <typename Real>
[[nodiscard]] Grid3D<Real> residual(const Grid3D<Real>& phi, const Grid3D<Real>& rhs, Real h);

} // namespace poisson
