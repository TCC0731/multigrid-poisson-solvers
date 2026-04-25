#include "poisson/gs.hpp"
#include "poisson/jacobi.hpp"
#include "poisson/metrics.hpp"
#include "poisson/mg.hpp"
#include "poisson/sor.hpp"
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
    std::string solver_name{"jacobi"};
    std::string mg_coarse_name{"exact"};
    std::string cycle_name{"v"};
    std::size_t grid_size{31};
    double tol{1e-10};
    std::size_t max_iter{20'000};
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--solver NAME] [--case NAME] [--grid-size N]"
                 " [--tol T] [--max-iter N] [--cycle v|w] [--mg-coarse exact|sor]\n";
    std::cerr << "Solvers: jacobi, gs, sor, mg\n";
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

        if (arg == "--solver") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--solver requires a value");
            }
            options.solver_name = argv[++i];
            continue;
        }

        if (arg == "--cycle") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--cycle requires a value");
            }
            options.cycle_name = argv[++i];
            continue;
        }

        if (arg == "--mg-coarse") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--mg-coarse requires a value");
            }
            options.mg_coarse_name = argv[++i];
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

    if (options.cycle_name != "v" && options.cycle_name != "w") {
        throw std::invalid_argument("cycle must be v or w");
    }

    if (options.mg_coarse_name != "exact" && options.mg_coarse_name != "sor") {
        throw std::invalid_argument("mg-coarse must be exact or sor");
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

        const poisson::SolveOptions solve_options{options.tol, options.max_iter};
        const poisson::MGCycle mg_cycle =
            options.cycle_name == "w" ? poisson::MGCycle::W : poisson::MGCycle::V;
        const poisson::MGOptions mg_options{
            options.tol,
            options.max_iter,
            2,
            mg_cycle,
            16,
        };

        std::string solver_name = options.solver_name;
        poisson::SolveResult result{};

        const auto start = std::chrono::steady_clock::now();
        if (options.solver_name == "jacobi") {
            result = poisson::solve_jacobi(problem, solve_options);
        } else if (options.solver_name == "gs") {
            result = poisson::solve_gs(problem, solve_options);
        } else if (options.solver_name == "sor") {
            result = poisson::solve_sor(problem, solve_options);
        } else if (options.solver_name == "mg") {
            if (options.mg_coarse_name == "exact") {
                result = poisson::solve_mg_exact(problem, mg_options);
                solver_name = "mg_exact";
            } else if (options.mg_coarse_name == "sor") {
                result = poisson::solve_mg_sor(problem, mg_options);
                solver_name = "mg_sor";
            } else {
                throw std::invalid_argument("mg-coarse must be exact or sor");
            }
        } else {
            throw std::invalid_argument("unknown solver: " + options.solver_name);
        }
        const auto end = std::chrono::steady_clock::now();
        const double time_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        const poisson::ErrorMetrics metrics = poisson::error_metrics(problem, result.phi);

        std::cout << "solver,backend,dtype,grid_size,iterations,residual_l2,error_l2,error_linf,time_ms\n";
        std::cout << std::scientific;
        std::cout.precision(6);
        std::cout << solver_name << ",cpp20,float64," << problem.interior_n << ','
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
