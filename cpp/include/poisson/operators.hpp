#pragma once

#include "poisson/problem.hpp"

namespace poisson {

[[nodiscard]] Grid2D apply_A(const Grid2D& phi, Real h);
[[nodiscard]] Grid2D residual(const Grid2D& phi, const Grid2D& rhs, Real h);

} // namespace poisson

