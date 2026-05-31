#include "poisson/cuda_utils.hpp"
#include "poisson/gs.hpp"
#include "poisson/jacobi.hpp"
#include "poisson/metrics.hpp"
#include "poisson/mg.hpp"
#include "poisson/problem.hpp"
#include "poisson/sor.hpp"
#include "poisson/validation.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

struct Options {
    std::string dtype{"double"};
    std::string case_name{"sine"};
    std::string solver_name{"jacobi"};
    std::string mg_coarse_name{"exact"};
    std::string cycle_name{"v"};
    std::size_t grid_size{31};
    std::size_t dimension{2};
    std::optional<double> tol{};
    std::size_t max_iter{20'000};
    std::size_t repeat_runs{1};
    std::size_t nu{2};
    std::size_t coarse_steps{16};
    std::optional<double> omega{};
    bool omega_is_auto{false};
    bool dump_phi{false};
    std::optional<std::string> dump_phi_path{};
};

template <typename Real>
Real default_tol() {
    if constexpr (std::is_same_v<Real, float>) {
        return Real{1e-6};
    }
    return Real{1e-10};
}

template <typename Result>
[[nodiscard]] double benchmark_compute_time_ms(const Result& result, double fallback_ms) {
    return result.benchmark_compute_time_ms >= 0.0 ? result.benchmark_compute_time_ms : fallback_ms;
}

template <typename SolveFn>
[[nodiscard]] auto time_solver_runs(SolveFn&& solve_fn, std::size_t repeat_runs) {
    if (repeat_runs < 1) {
        throw std::invalid_argument("repeat-runs must be positive");
    }

    using Result = std::invoke_result_t<SolveFn&>;

    std::optional<Result> last_result;
    double total_time_ms = 0.0;
    for (std::size_t i = 0; i < repeat_runs; ++i) {
        const auto start = std::chrono::steady_clock::now();
        last_result = solve_fn();
        const auto end = std::chrono::steady_clock::now();
        const double wall_time_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        total_time_ms += benchmark_compute_time_ms(*last_result, wall_time_ms);
    }

    if (!last_result.has_value()) {
        throw std::logic_error("solver did not produce any result");
    }

    return std::pair<Result, double>{
        std::move(*last_result),
        total_time_ms / static_cast<double>(repeat_runs),
    };
}

[[nodiscard]] std::string make_phi_dump_filename(
    const Options& options, std::string_view solver_name
) {
    std::string filename{"phi_cuda_"};
    filename += std::to_string(options.dimension);
    filename += "d_";
    filename += solver_name;
    filename += '_';
    filename += options.case_name;
    filename += "_n";
    filename += std::to_string(options.grid_size);
    filename += '_';
    filename += options.dtype;
    filename += ".bin";
    return filename;
}

constexpr char kPhiDumpMagic[8] = {'P', 'H', 'I', 'D', 'U', 'M', 'P', '1'};
constexpr std::uint32_t kPhiDumpVersion = 1;

template <typename T>
void write_binary_value(std::ofstream& out, const T& value, const std::string& path) {
    static_assert(std::is_trivially_copyable_v<T>);
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) {
        throw std::runtime_error("failed to write phi dump file: " + path);
    }
}

template <typename Grid>
void dump_phi_binary(const Grid& phi, const std::string& path, std::size_t dimension, std::size_t interior_n) {
    using Value = typename Grid::value_type;
    static_assert(std::is_trivially_copyable_v<Value>);

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open phi dump file: " + path);
    }

    out.write(kPhiDumpMagic, sizeof(kPhiDumpMagic));
    if (!out) {
        throw std::runtime_error("failed to write phi dump file: " + path);
    }

    write_binary_value(out, kPhiDumpVersion, path);
    write_binary_value(out, static_cast<std::uint32_t>(dimension), path);
    write_binary_value(out, static_cast<std::uint64_t>(interior_n), path);
    write_binary_value(out, static_cast<std::uint64_t>(phi.size()), path);
    write_binary_value(out, static_cast<std::uint64_t>(phi.elements()), path);
    write_binary_value(out, static_cast<std::uint64_t>(sizeof(Value)), path);

    if (phi.elements() > 0) {
        out.write(
            reinterpret_cast<const char*>(phi.data().data()),
            static_cast<std::streamsize>(phi.elements() * sizeof(Value))
        );
        if (!out) {
            throw std::runtime_error("failed to write phi dump file: " + path);
        }
    }
}

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--dtype float|double] [--solver jacobi|gs|sor|mg] [--case NAME] [--grid-size N]"
                 " [--dim 2|3] [--tol T] [--max-iter N] [--repeat-runs N] [--cycle v|w]"
                 " [--nu N] [--omega auto|VALUE] [--mg-coarse exact|sor]"
                 " [--coarse-steps N] [--dump-phi [PATH]]\n";
    std::cerr << "Dtypes: float, double\n";
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

        if (arg == "--dtype") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--dtype requires a value");
            }
            options.dtype = argv[++i];
            continue;
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

        if (arg == "--nu") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--nu requires a value");
            }
            const long long parsed = std::stoll(argv[++i]);
            if (parsed < 1) {
                throw std::invalid_argument("nu must be positive");
            }
            options.nu = static_cast<std::size_t>(parsed);
            continue;
        }

        if (arg == "--omega") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--omega requires a value");
            }
            const std::string_view omega_value{argv[++i]};
            if (omega_value == "auto" || omega_value == "default" || omega_value == "none") {
                options.omega.reset();
                options.omega_is_auto = true;
                continue;
            }
            options.omega = std::stod(std::string(omega_value));
            options.omega_is_auto = false;
            continue;
        }

        if (arg == "--mg-coarse") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--mg-coarse requires a value");
            }
            options.mg_coarse_name = argv[++i];
            continue;
        }

        if (arg == "--coarse-steps") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--coarse-steps requires a value");
            }
            const long long parsed = std::stoll(argv[++i]);
            if (parsed < 1) {
                throw std::invalid_argument("coarse-steps must be positive");
            }
            options.coarse_steps = static_cast<std::size_t>(parsed);
            continue;
        }

        if (arg == "--dump-phi") {
            options.dump_phi = true;
            if (i + 1 < argc) {
                const std::string_view maybe_path{argv[i + 1]};
                if (!maybe_path.empty() && maybe_path.front() != '-') {
                    options.dump_phi_path = std::string(maybe_path);
                    ++i;
                }
            }
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

        if (arg == "--dim" || arg == "--dimension") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--dim requires a value");
            }
            const long long parsed = std::stoll(argv[++i]);
            if (parsed != 2 && parsed != 3) {
                throw std::invalid_argument("dim must be 2 or 3");
            }
            options.dimension = static_cast<std::size_t>(parsed);
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

        if (arg == "--repeat-runs") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--repeat-runs requires a value");
            }
            const long long parsed = std::stoll(argv[++i]);
            if (parsed < 1) {
                throw std::invalid_argument("repeat-runs must be positive");
            }
            options.repeat_runs = static_cast<std::size_t>(parsed);
            continue;
        }

        throw std::invalid_argument("unknown argument: " + std::string(arg));
    }

    if (options.dtype != "float" && options.dtype != "double") {
        throw std::invalid_argument("dtype must be float or double");
    }
    if (options.tol.has_value() && *options.tol <= 0.0) {
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

template <typename Real>
int run_2d(const Options& options) {
    const poisson::cuda::detail::ScopedNvtxRange run_range{"cuda::run"};
    using Problem = poisson::Problem2D<Real>;
    using SolveOpts = poisson::SolveOptions<Real>;
    using MgOpts = poisson::MGOptions<Real>;

    poisson::cuda::ensure_device_available();

    const Problem problem = poisson::make_problem<Real>(options.case_name, options.grid_size);
    const poisson::ValidationReport report = poisson::validate_problem(problem);
    if (!report.ok) {
        std::cerr << "error: generated problem failed validation\n";
        return 1;
    }

    const Real tol = options.tol.has_value()
        ? static_cast<Real>(*options.tol)
        : default_tol<Real>();

    const SolveOpts solve_options{tol, options.max_iter};
    const poisson::MGCycle mg_cycle =
        options.cycle_name == "w" ? poisson::MGCycle::W : poisson::MGCycle::V;
    MgOpts mg_options{tol, options.max_iter, options.nu, mg_cycle, options.coarse_steps};
    if (options.omega_is_auto) {
        mg_options.omega_is_auto = true;
    } else if (options.omega.has_value()) {
        mg_options.omega = static_cast<Real>(*options.omega);
    }

    std::string solver_name = options.solver_name;

    const auto solve_once = [&]() -> poisson::SolveResult {
        if (options.solver_name == "jacobi") {
            return poisson::solve_jacobi<Real>(problem, solve_options);
        }
        if (options.solver_name == "gs") {
            return poisson::solve_gs<Real>(problem, solve_options);
        }
        if (options.solver_name == "sor") {
            return poisson::solve_sor<Real>(problem, solve_options);
        }
        if (options.solver_name == "mg") {
            if (options.mg_coarse_name == "exact") {
                solver_name = "mg_exact";
                return poisson::solve_mg_exact<Real>(problem, mg_options);
            }
            if (options.mg_coarse_name == "sor") {
                solver_name = "mg_sor";
                return poisson::solve_mg_sor<Real>(problem, mg_options);
            }
            throw std::invalid_argument("mg-coarse must be exact or sor");
        }
        throw std::invalid_argument("unknown solver: " + options.solver_name);
    };
    const auto timed_run = time_solver_runs(solve_once, options.repeat_runs);
    const auto& result = timed_run.first;
    const double time_ms = timed_run.second;

    const auto metrics_problem = poisson::make_problem<double>(options.case_name, options.grid_size);
    const poisson::ErrorMetrics metrics = poisson::metrics(metrics_problem, result.phi);

    if (options.dump_phi) {
        const std::string dump_path = options.dump_phi_path.has_value()
            ? *options.dump_phi_path
            : make_phi_dump_filename(options, solver_name);
        dump_phi_binary(result.phi, dump_path, 2, problem.interior_n);
        std::cerr << "phi dumped to " << dump_path << '\n';
    }

    std::cout << "solver,backend,dtype,grid_size,iterations,residual_l2,error_l2,error_linf,time_ms\n";
    std::cout << std::scientific;
    std::cout.precision(6);
    std::cout << solver_name << ",cuda," << options.dtype << "," << problem.interior_n << ','
              << result.iterations << ',' << result.residual_l2 << ','
              << metrics.error_l2 << ',' << metrics.error_linf << ',';
    std::cout << std::fixed;
    std::cout.precision(3);
    std::cout << time_ms << '\n';

    return 0;
}

template <typename Real>
int run_3d(const Options& options) {
    const poisson::cuda::detail::ScopedNvtxRange run_range{"cuda::run_3d"};
    using Problem = poisson::Problem3D<Real>;
    using SolveOpts = poisson::SolveOptions<Real>;
    using MgOpts = poisson::MGOptions<Real>;

    poisson::cuda::ensure_device_available();

    const Problem problem = poisson::make_problem_3d<Real>(options.case_name, options.grid_size);
    const poisson::ValidationReport report = poisson::validate_problem(problem);
    if (!report.ok) {
        std::cerr << "error: generated 3D problem failed validation\n";
        return 1;
    }

    const Real tol = options.tol.has_value()
        ? static_cast<Real>(*options.tol)
        : default_tol<Real>();

    const SolveOpts solve_options{tol, options.max_iter};
    const poisson::MGCycle mg_cycle =
        options.cycle_name == "w" ? poisson::MGCycle::W : poisson::MGCycle::V;
    MgOpts mg_options{tol, options.max_iter, options.nu, mg_cycle, options.coarse_steps};
    if (options.omega_is_auto) {
        mg_options.omega_is_auto = true;
    } else if (options.omega.has_value()) {
        mg_options.omega = static_cast<Real>(*options.omega);
    }

    std::string solver_name = options.solver_name;

    const auto solve_once = [&]() -> poisson::SolveResult3D {
        if (options.solver_name == "jacobi") {
            return poisson::solve_jacobi<Real>(problem, solve_options);
        }
        if (options.solver_name == "gs") {
            return poisson::solve_gs<Real>(problem, solve_options);
        }
        if (options.solver_name == "sor") {
            return poisson::solve_sor<Real>(problem, solve_options);
        }
        if (options.solver_name == "mg") {
            if (options.mg_coarse_name == "exact") {
                solver_name = "mg_exact";
                return poisson::solve_mg_exact<Real>(problem, mg_options);
            }
            if (options.mg_coarse_name == "sor") {
                solver_name = "mg_sor";
                return poisson::solve_mg_sor<Real>(problem, mg_options);
            }
            throw std::invalid_argument("mg-coarse must be exact or sor");
        }
        throw std::invalid_argument("unknown solver: " + options.solver_name);
    };
    const auto timed_run = time_solver_runs(solve_once, options.repeat_runs);
    const auto& result = timed_run.first;
    const double time_ms = timed_run.second;
    poisson::cuda::check(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

    const auto metrics_problem =
        poisson::make_problem_3d<double>(options.case_name, options.grid_size);
    const poisson::ErrorMetrics metrics = poisson::metrics(metrics_problem, result.phi);

    if (options.dump_phi) {
        const std::string dump_path = options.dump_phi_path.has_value()
            ? *options.dump_phi_path
            : make_phi_dump_filename(options, solver_name);
        dump_phi_binary(result.phi, dump_path, 3, problem.interior_n);
        std::cerr << "phi dumped to " << dump_path << '\n';
    }

    std::cout << "solver,backend,dtype,grid_size,iterations,residual_l2,error_l2,error_linf,time_ms\n";
    std::cout << std::scientific;
    std::cout.precision(6);
    std::cout << solver_name << ",cuda," << options.dtype << "," << problem.interior_n << ','
              << result.iterations << ',' << result.residual_l2 << ','
              << metrics.error_l2 << ',' << metrics.error_linf << ',';
    std::cout << std::fixed;
    std::cout.precision(3);
    std::cout << time_ms << '\n';

    return 0;
}

template <typename Real>
int run(const Options& options) {
    if (options.dimension == 3) {
        return run_3d<Real>(options);
    }
    return run_2d<Real>(options);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        if (options.dtype == "float") {
            return run<float>(options);
        }
        if (options.dtype == "double") {
            return run<double>(options);
        }
        throw std::invalid_argument("dtype must be float or double");
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
}
