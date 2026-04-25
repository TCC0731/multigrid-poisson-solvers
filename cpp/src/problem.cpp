#include "poisson/problem.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace poisson {
namespace {

using Fn = Real (*)(Real, Real);

constexpr Real pi = std::numbers::pi_v<Real>;

Real sine_exact(Real x, Real y) {
    return std::sin(pi * x) * std::sin(pi * y);
}

Real sine_rhs(Real x, Real y) {
    return 2.0 * pi * pi * std::sin(pi * x) * std::sin(pi * y);
}

Real mixed_sine_exact(Real x, Real y) {
    return std::sin(2.0 * pi * x) * std::sin(3.0 * pi * y);
}

Real mixed_sine_rhs(Real x, Real y) {
    return 13.0 * pi * pi * std::sin(2.0 * pi * x) * std::sin(3.0 * pi * y);
}

Real bubble_exact(Real x, Real y) {
    return x * (1.0 - x) * y * (1.0 - y);
}

Real bubble_rhs(Real x, Real y) {
    return 2.0 * x * (1.0 - x) + 2.0 * y * (1.0 - y);
}

Real exp_exact(Real x, Real y) {
    return std::exp(x + y);
}

Real exp_rhs(Real x, Real y) {
    return -2.0 * std::exp(x + y);
}

Real cosine_exact(Real x, Real y) {
    return std::cos(pi * x) * std::cos(pi * y);
}

Real cosine_rhs(Real x, Real y) {
    return 2.0 * pi * pi * std::cos(pi * x) * std::cos(pi * y);
}

struct CaseFunctions {
    Fn exact;
    Fn rhs;
};

CaseFunctions functions_for(Case2D case_id) {
    switch (case_id) {
    case Case2D::Sine:
        return {sine_exact, sine_rhs};
    case Case2D::MixedSine:
        return {mixed_sine_exact, mixed_sine_rhs};
    case Case2D::Bubble:
        return {bubble_exact, bubble_rhs};
    case Case2D::Exp:
        return {exp_exact, exp_rhs};
    case Case2D::Cosine:
        return {cosine_exact, cosine_rhs};
    }
    throw std::invalid_argument("unsupported case");
}

} // namespace

std::string_view to_string(Case2D case_id) {
    switch (case_id) {
    case Case2D::Sine:
        return "sine";
    case Case2D::MixedSine:
        return "mixed_sine";
    case Case2D::Bubble:
        return "bubble";
    case Case2D::Exp:
        return "exp";
    case Case2D::Cosine:
        return "cosine";
    }
    return "unknown";
}

Case2D parse_case(std::string_view case_name) {
    if (case_name == "sine") {
        return Case2D::Sine;
    }
    if (case_name == "mixed_sine") {
        return Case2D::MixedSine;
    }
    if (case_name == "bubble") {
        return Case2D::Bubble;
    }
    if (case_name == "exp") {
        return Case2D::Exp;
    }
    if (case_name == "cosine") {
        return Case2D::Cosine;
    }
    throw std::invalid_argument("unknown case: " + std::string(case_name));
}

Problem2D make_problem(Case2D case_id, std::size_t interior_n) {
    if (interior_n < 1) {
        throw std::invalid_argument("interior_n must be positive");
    }

    const std::size_t array_n = interior_n + 2;
    const Real h = 1.0 / static_cast<Real>(interior_n + 1);
    const auto functions = functions_for(case_id);

    Problem2D problem{
        case_id,
        interior_n,
        h,
        Grid2D{array_n},
        Grid2D{array_n},
        Grid2D{array_n},
    };

    problem.phi0.fill(0.0);

    for (std::size_t i = 0; i < array_n; ++i) {
        const Real x = static_cast<Real>(i) * h;
        for (std::size_t j = 0; j < array_n; ++j) {
            const Real y = static_cast<Real>(j) * h;
            problem.exact(i, j) = functions.exact(x, y);
            problem.rhs(i, j) = functions.rhs(x, y);
        }
    }

    for (std::size_t i = 0; i < array_n; ++i) {
        problem.phi0(i, 0) = problem.exact(i, 0);
        problem.phi0(i, array_n - 1) = problem.exact(i, array_n - 1);
    }

    for (std::size_t j = 0; j < array_n; ++j) {
        problem.phi0(0, j) = problem.exact(0, j);
        problem.phi0(array_n - 1, j) = problem.exact(array_n - 1, j);
    }

    return problem;
}

Problem2D make_problem(std::string_view case_name, std::size_t interior_n) {
    return make_problem(parse_case(case_name), interior_n);
}

} // namespace poisson

