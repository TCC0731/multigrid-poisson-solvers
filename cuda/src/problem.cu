#include "poisson/problem.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace poisson {
namespace {

template <typename Real>
using Fn = Real (*)(Real, Real);

template <typename Real>
using Fn3D = Real (*)(Real, Real, Real);

template <typename Real>
struct CaseFunctions {
    Fn<Real> exact;
    Fn<Real> rhs;
};

template <typename Real>
struct CaseFunctions3D {
    Fn3D<Real> exact;
    Fn3D<Real> rhs;
};

template <typename Real>
constexpr Real pi_v = static_cast<Real>(3.14159265358979323846264338327950288L);

template <typename Real>
Real sine_exact(Real x, Real y) {
    return std::sin(pi_v<Real> * x) * std::sin(pi_v<Real> * y);
}

template <typename Real>
Real sine_rhs(Real x, Real y) {
    return Real{2} * pi_v<Real> * pi_v<Real> * std::sin(pi_v<Real> * x) * std::sin(pi_v<Real> * y);
}

template <typename Real>
Real sine_exact_3d(Real x, Real y, Real z) {
    return std::sin(pi_v<Real> * x) * std::sin(pi_v<Real> * y) * std::sin(pi_v<Real> * z);
}

template <typename Real>
Real sine_rhs_3d(Real x, Real y, Real z) {
    return Real{3} * pi_v<Real> * pi_v<Real>
        * std::sin(pi_v<Real> * x)
        * std::sin(pi_v<Real> * y)
        * std::sin(pi_v<Real> * z);
}

template <typename Real>
Real mixed_sine_exact(Real x, Real y) {
    return std::sin(Real{2} * pi_v<Real> * x) * std::sin(Real{3} * pi_v<Real> * y);
}

template <typename Real>
Real mixed_sine_rhs(Real x, Real y) {
    return Real{13} * pi_v<Real> * pi_v<Real> *
        std::sin(Real{2} * pi_v<Real> * x) *
        std::sin(Real{3} * pi_v<Real> * y);
}

template <typename Real>
Real mixed_sine_exact_3d(Real x, Real y, Real z) {
    return std::sin(Real{2} * pi_v<Real> * x)
        * std::sin(Real{3} * pi_v<Real> * y)
        * std::sin(Real{4} * pi_v<Real> * z);
}

template <typename Real>
Real mixed_sine_rhs_3d(Real x, Real y, Real z) {
    return Real{29} * pi_v<Real> * pi_v<Real>
        * std::sin(Real{2} * pi_v<Real> * x)
        * std::sin(Real{3} * pi_v<Real> * y)
        * std::sin(Real{4} * pi_v<Real> * z);
}

template <typename Real>
Real bubble_exact(Real x, Real y) {
    return x * (Real{1} - x) * y * (Real{1} - y);
}

template <typename Real>
Real bubble_rhs(Real x, Real y) {
    return Real{2} * x * (Real{1} - x) + Real{2} * y * (Real{1} - y);
}

template <typename Real>
Real bubble_exact_3d(Real x, Real y, Real z) {
    return x * (Real{1} - x) * y * (Real{1} - y) * z * (Real{1} - z);
}

template <typename Real>
Real bubble_rhs_3d(Real x, Real y, Real z) {
    return Real{2} * (
        y * (Real{1} - y) * z * (Real{1} - z) +
        x * (Real{1} - x) * z * (Real{1} - z) +
        x * (Real{1} - x) * y * (Real{1} - y)
    );
}

template <typename Real>
Real exp_exact(Real x, Real y) {
    return std::exp(x + y);
}

template <typename Real>
Real exp_rhs(Real x, Real y) {
    return -Real{2} * std::exp(x + y);
}

template <typename Real>
Real exp_exact_3d(Real x, Real y, Real z) {
    return std::exp(x + y + z);
}

template <typename Real>
Real exp_rhs_3d(Real x, Real y, Real z) {
    return -Real{3} * std::exp(x + y + z);
}

template <typename Real>
Real cosine_exact(Real x, Real y) {
    return std::cos(pi_v<Real> * x) * std::cos(pi_v<Real> * y);
}

template <typename Real>
Real cosine_rhs(Real x, Real y) {
    return Real{2} * pi_v<Real> * pi_v<Real> * std::cos(pi_v<Real> * x) * std::cos(pi_v<Real> * y);
}

template <typename Real>
Real cosine_exact_3d(Real x, Real y, Real z) {
    return std::cos(pi_v<Real> * x) * std::cos(pi_v<Real> * y) * std::cos(pi_v<Real> * z);
}

template <typename Real>
Real cosine_rhs_3d(Real x, Real y, Real z) {
    return Real{3} * pi_v<Real> * pi_v<Real>
        * std::cos(pi_v<Real> * x)
        * std::cos(pi_v<Real> * y)
        * std::cos(pi_v<Real> * z);
}

template <typename Real>
CaseFunctions<Real> functions_for(Case2D case_id) {
    switch (case_id) {
    case Case2D::Sine:
        return {sine_exact<Real>, sine_rhs<Real>};
    case Case2D::MixedSine:
        return {mixed_sine_exact<Real>, mixed_sine_rhs<Real>};
    case Case2D::Bubble:
        return {bubble_exact<Real>, bubble_rhs<Real>};
    case Case2D::Exp:
        return {exp_exact<Real>, exp_rhs<Real>};
    case Case2D::Cosine:
        return {cosine_exact<Real>, cosine_rhs<Real>};
    }
    throw std::invalid_argument("unsupported case");
}

template <typename Real>
CaseFunctions3D<Real> functions_for(Case3D case_id) {
    switch (case_id) {
    case Case3D::Sine:
        return {sine_exact_3d<Real>, sine_rhs_3d<Real>};
    case Case3D::MixedSine:
        return {mixed_sine_exact_3d<Real>, mixed_sine_rhs_3d<Real>};
    case Case3D::Bubble:
        return {bubble_exact_3d<Real>, bubble_rhs_3d<Real>};
    case Case3D::Exp:
        return {exp_exact_3d<Real>, exp_rhs_3d<Real>};
    case Case3D::Cosine:
        return {cosine_exact_3d<Real>, cosine_rhs_3d<Real>};
    }
    throw std::invalid_argument("unsupported case");
}

template <typename Real>
Problem2D<Real> make_problem_impl(Case2D case_id, std::size_t interior_n) {
    if (interior_n < 1) {
        throw std::invalid_argument("interior_n must be positive");
    }

    const std::size_t array_n = interior_n + 2;
    const Real h = Real{1} / static_cast<Real>(interior_n + 1);
    const auto functions = functions_for<Real>(case_id);

    Problem2D<Real> problem{
        case_id,
        interior_n,
        h,
        Grid2D<Real>{array_n},
        Grid2D<Real>{array_n},
        Grid2D<Real>{array_n},
    };

    problem.phi0.fill(Real{});

    for (std::size_t i = 0; i < array_n; ++i) {
        const Real x = static_cast<Real>(i) * h;
        for (std::size_t j = 0; j < array_n; ++j) {
            const Real y = static_cast<Real>(j) * h;
            problem.exact.unchecked(i, j) = functions.exact(x, y);
            problem.rhs.unchecked(i, j) = functions.rhs(x, y);
        }
    }

    for (std::size_t i = 0; i < array_n; ++i) {
        problem.phi0.unchecked(i, 0) = problem.exact.unchecked(i, 0);
        problem.phi0.unchecked(i, array_n - 1) = problem.exact.unchecked(i, array_n - 1);
    }

    for (std::size_t j = 0; j < array_n; ++j) {
        problem.phi0.unchecked(0, j) = problem.exact.unchecked(0, j);
        problem.phi0.unchecked(array_n - 1, j) = problem.exact.unchecked(array_n - 1, j);
    }

    return problem;
}

template <typename Real>
Problem3D<Real> make_problem_3d_impl(Case3D case_id, std::size_t interior_n) {
    if (interior_n < 1) {
        throw std::invalid_argument("interior_n must be positive");
    }

    const std::size_t array_n = interior_n + 2;
    const Real h = Real{1} / static_cast<Real>(interior_n + 1);
    const auto functions = functions_for<Real>(case_id);

    Problem3D<Real> problem{
        case_id,
        interior_n,
        h,
        Grid3D<Real>{array_n},
        Grid3D<Real>{array_n},
        Grid3D<Real>{array_n},
    };

    problem.phi0.fill(Real{});

    for (std::size_t i = 0; i < array_n; ++i) {
        const Real x = static_cast<Real>(i) * h;
        for (std::size_t j = 0; j < array_n; ++j) {
            const Real y = static_cast<Real>(j) * h;
            for (std::size_t k = 0; k < array_n; ++k) {
                const Real z = static_cast<Real>(k) * h;
                problem.exact.unchecked(i, j, k) = functions.exact(x, y, z);
                problem.rhs.unchecked(i, j, k) = functions.rhs(x, y, z);
            }
        }
    }

    for (std::size_t i = 0; i < array_n; ++i) {
        for (std::size_t j = 0; j < array_n; ++j) {
            problem.phi0.unchecked(i, j, 0) = problem.exact.unchecked(i, j, 0);
            problem.phi0.unchecked(i, j, array_n - 1) =
                problem.exact.unchecked(i, j, array_n - 1);
        }
    }

    for (std::size_t i = 0; i < array_n; ++i) {
        for (std::size_t k = 0; k < array_n; ++k) {
            problem.phi0.unchecked(i, 0, k) = problem.exact.unchecked(i, 0, k);
            problem.phi0.unchecked(i, array_n - 1, k) =
                problem.exact.unchecked(i, array_n - 1, k);
        }
    }

    for (std::size_t j = 0; j < array_n; ++j) {
        for (std::size_t k = 0; k < array_n; ++k) {
            problem.phi0.unchecked(0, j, k) = problem.exact.unchecked(0, j, k);
            problem.phi0.unchecked(array_n - 1, j, k) =
                problem.exact.unchecked(array_n - 1, j, k);
        }
    }

    return problem;
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

std::string_view to_string(Case3D case_id) {
    switch (case_id) {
    case Case3D::Sine:
        return "sine";
    case Case3D::MixedSine:
        return "mixed_sine";
    case Case3D::Bubble:
        return "bubble";
    case Case3D::Exp:
        return "exp";
    case Case3D::Cosine:
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

Case3D parse_case_3d(std::string_view case_name) {
    if (case_name == "sine") {
        return Case3D::Sine;
    }
    if (case_name == "mixed_sine") {
        return Case3D::MixedSine;
    }
    if (case_name == "bubble") {
        return Case3D::Bubble;
    }
    if (case_name == "exp") {
        return Case3D::Exp;
    }
    if (case_name == "cosine") {
        return Case3D::Cosine;
    }
    throw std::invalid_argument("unknown case: " + std::string(case_name));
}

template <typename Real>
Problem2D<Real> make_problem(Case2D case_id, std::size_t interior_n) {
    return make_problem_impl<Real>(case_id, interior_n);
}

template <typename Real>
Problem2D<Real> make_problem(std::string_view case_name, std::size_t interior_n) {
    return make_problem_impl<Real>(parse_case(case_name), interior_n);
}

template <typename Real>
Problem3D<Real> make_problem_3d(Case3D case_id, std::size_t interior_n) {
    return make_problem_3d_impl<Real>(case_id, interior_n);
}

template <typename Real>
Problem3D<Real> make_problem_3d(std::string_view case_name, std::size_t interior_n) {
    return make_problem_3d_impl<Real>(parse_case_3d(case_name), interior_n);
}

Problem2D<double> make_problem(Case2D case_id, std::size_t interior_n) {
    return make_problem<double>(case_id, interior_n);
}

Problem2D<double> make_problem(std::string_view case_name, std::size_t interior_n) {
    return make_problem<double>(case_name, interior_n);
}

Problem3D<double> make_problem_3d(Case3D case_id, std::size_t interior_n) {
    return make_problem_3d<double>(case_id, interior_n);
}

Problem3D<double> make_problem_3d(std::string_view case_name, std::size_t interior_n) {
    return make_problem_3d<double>(case_name, interior_n);
}

template Problem2D<float> make_problem<float>(Case2D, std::size_t);
template Problem2D<double> make_problem<double>(Case2D, std::size_t);
template Problem2D<float> make_problem<float>(std::string_view, std::size_t);
template Problem2D<double> make_problem<double>(std::string_view, std::size_t);
template Problem3D<float> make_problem_3d<float>(Case3D, std::size_t);
template Problem3D<double> make_problem_3d<double>(Case3D, std::size_t);
template Problem3D<float> make_problem_3d<float>(std::string_view, std::size_t);
template Problem3D<double> make_problem_3d<double>(std::string_view, std::size_t);

} // namespace poisson
