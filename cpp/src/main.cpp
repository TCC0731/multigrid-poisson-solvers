#include "poisson/jacobi.hpp"
#include "poisson/metrics.hpp"
#include "poisson/validation.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct Options {
    std::string case_name{"sine"};
    std::size_t grid_size{31};
    double tol{1e-10};
    std::size_t max_iter{20'000};
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--case NAME] [--grid-size N] [--tol T] [--max-iter N]\n";
    std::cerr << "Cases: sine, mixed_sine, bubble, exp, cosine\n";
}

Options parse_args(int argc, char** argv) {
    Options options{};

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }

        if (arg == "--case") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--case requires a value");
            }
            options.case_name = argv[++i];
            continue;
        }

        if (arg == "-n" || arg == "--grid-size") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--grid-size requires a value");
            }
            const long long parsed = std::stoll(argv[++i]);
            if (parsed < 1) {
                throw std::invalid_argument("grid-size must be positive");
            }
            options.grid_size = static_cast<std::size_t>(parsed);
            continue;
        }

        if (arg == "--tol") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--tol requires a value");
            }
            options.tol = std::stod(argv[++i]);
            continue;
        }

        if (arg == "--max-iter") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--max-iter requires a value");
            }
            const long long parsed = std::stoll(argv[++i]);
            if (parsed < 1) {
                throw std::invalid_argument("max-iter must be positive");
            }
            options.max_iter = static_cast<std::size_t>(parsed);
            continue;
        }

        throw std::invalid_argument("unknown argument: " + std::string(arg));
    }

    if (options.tol <= 0.0) {
        throw std::invalid_argument("tol must be positive");
    }

    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const poisson::Problem2D problem =
            poisson::make_problem(options.case_name, options.grid_size);
        const poisson::ValidationReport report = poisson::validate_problem(problem);
        if (!report.ok) {
            std::cerr << "error: generated problem failed validation\n";
            return 1;
        }

        const poisson::JacobiSolver2D solver{};
        const poisson::SolveOptions solve_options{
            options.tol,
            options.max_iter,
        };

        const auto start = std::chrono::steady_clock::now();
        const poisson::SolveResult result = solver.solve(problem, solve_options);
        const auto end = std::chrono::steady_clock::now();
        const double time_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        const poisson::ErrorMetrics metrics = poisson::error_metrics(problem, result.phi);

        std::cout << "solver,backend,dtype,grid_size,iterations,residual_l2,error_l2,error_linf,time_ms\n";
        std::cout << std::scientific;
        std::cout.precision(6);
        std::cout << solver.name() << ",cpp20,float64," << problem.interior_n << ','
                  << result.iterations << ',' << result.residual_l2 << ','
                  << metrics.error_l2 << ',' << metrics.error_linf << ',';
        std::cout << std::fixed;
        std::cout.precision(3);
        std::cout << time_ms << '\n';

        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
}
