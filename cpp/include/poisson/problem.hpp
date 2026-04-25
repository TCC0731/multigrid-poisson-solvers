#pragma once

#include <cstddef>
#include <string_view>

#include "poisson/grid2d.hpp"

namespace poisson {

enum class Case2D {
    Sine,
    MixedSine,
    Bubble,
    Exp,
    Cosine,
};

[[nodiscard]] std::string_view to_string(Case2D case_id);
[[nodiscard]] Case2D parse_case(std::string_view case_name);

struct Problem2D {
    Case2D case_id{};
    std::size_t interior_n{};
    Real h{};
    Grid2D exact{};
    Grid2D rhs{};
    Grid2D phi0{};

    [[nodiscard]] std::size_t array_n() const noexcept { return interior_n + 2; }
};

[[nodiscard]] Problem2D make_problem(Case2D case_id, std::size_t interior_n);
[[nodiscard]] Problem2D make_problem(std::string_view case_name, std::size_t interior_n);

} // namespace poisson

