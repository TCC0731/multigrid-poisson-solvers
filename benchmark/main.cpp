#include "poisson_benchmark.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef POISSON_BENCHMARK_BACKEND_LABEL
#define POISSON_BENCHMARK_BACKEND_LABEL cpp20
#endif

#define POISSON_BENCHMARK_STRINGIFY_IMPL(x) #x
#define POISSON_BENCHMARK_STRINGIFY(x) POISSON_BENCHMARK_STRINGIFY_IMPL(x)

namespace {

constexpr std::string_view kBackendLabel{
    POISSON_BENCHMARK_STRINGIFY(POISSON_BENCHMARK_BACKEND_LABEL)
};

struct Options {
    poisson::benchmark::Suite suite{poisson::benchmark::Suite::All};
    std::size_t dimension{2};
    std::size_t max_grid_size{0};
    poisson::benchmark::BenchmarkTiming timing{};
    std::filesystem::path output_base_path{};
    bool has_output{false};
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--suite all|solver_comparison|mg_compare] [--dim 2|3]"
                 " [--max-grid-size N] [--warmup-runs N] [--timed-runs N]"
                 " [--repeat-runs N]"
                 " [--output BASE]\n";
    std::cerr << "When BASE is provided, the program writes BASE.csv and BASE_all.csv.\n";
    std::cerr << "Default suite: all (runs solver_comparison + mg_compare)\n";
    std::cerr << "Default dimension: 2\n";
    std::cerr << "Default warmup runs: " << poisson::benchmark::detail::kWarmupRuns << '\n';
    std::cerr << "Default timed runs: " << poisson::benchmark::detail::kTimedRuns << '\n';
    std::cerr << "Backend label: " << kBackendLabel << '\n';
}

Options parse_args(int argc, char** argv) {
    Options options{};

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }

        if (arg == "--suite") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--suite requires a value");
            }
            options.suite = poisson::benchmark::parse_suite(argv[++i]);
            continue;
        }

        if (arg == "--dim") {
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

        if (arg == "--max-grid-size") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--max-grid-size requires a value");
            }
            const long long parsed = std::stoll(argv[++i]);
            if (parsed < 1) {
                throw std::invalid_argument("max-grid-size must be positive");
            }
            options.max_grid_size = static_cast<std::size_t>(parsed);
            continue;
        }

        if (arg == "--warmup-runs") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--warmup-runs requires a value");
            }
            const long long parsed = std::stoll(argv[++i]);
            if (parsed < 0) {
                throw std::invalid_argument("warmup-runs must be non-negative");
            }
            options.timing.warmup_runs = static_cast<std::size_t>(parsed);
            continue;
        }

        if (arg == "--timed-runs" || arg == "--repeat-runs") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--timed-runs requires a value");
            }
            const long long parsed = std::stoll(argv[++i]);
            if (parsed < 1) {
                throw std::invalid_argument("timed-runs must be positive");
            }
            options.timing.timed_runs = static_cast<std::size_t>(parsed);
            continue;
        }

        if (arg == "--output" || arg == "-o") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--output requires a path");
            }
            options.output_base_path = argv[++i];
            options.has_output = true;
            continue;
        }

        throw std::invalid_argument("unknown argument: " + std::string(arg));
    }

    return options;
}

struct OutputPaths {
    std::filesystem::path summary_path;
    std::filesystem::path full_path;
};

[[nodiscard]] std::filesystem::path normalize_output_base(std::filesystem::path path) {
    if (path.has_extension() && path.extension() == ".csv") {
        path.replace_extension();
    }
    return path;
}

[[nodiscard]] OutputPaths resolve_output_paths(const std::filesystem::path& requested_base) {
    const std::filesystem::path base = normalize_output_base(requested_base);

    OutputPaths paths{};
    paths.summary_path = base;
    paths.summary_path += ".csv";
    paths.full_path = base;
    paths.full_path += "_all.csv";
    return paths;
}

void write_full_rows(std::ostream& os, const std::vector<poisson::benchmark::BenchmarkRow>& rows) {
    poisson::benchmark::write_csv(os, rows);
}

void write_summary_rows(std::ostream& os, const std::vector<poisson::benchmark::BenchmarkRow>& rows) {
    poisson::benchmark::write_simple_csv(os, rows);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);

#ifdef _OPENMP
        omp_set_dynamic(0);
#endif

        std::cerr << "Running benchmark suite '" << poisson::benchmark::to_string(options.suite)
                  << "' in " << options.dimension << "D on backend '" << kBackendLabel << "'\n";

        const auto rows = poisson::benchmark::run_suite<double>(
            options.suite,
            kBackendLabel,
            options.dimension,
            options.max_grid_size,
            options.timing
        );

        if (options.has_output) {
            const OutputPaths paths = resolve_output_paths(options.output_base_path);
            const auto parent = paths.full_path.parent_path();
            if (!parent.empty()) {
                std::filesystem::create_directories(parent);
            }

            std::ofstream full_file(paths.full_path);
            if (!full_file.is_open()) {
                throw std::runtime_error("failed to open output file: " + paths.full_path.string());
            }
            std::ofstream summary_file(paths.summary_path);
            if (!summary_file.is_open()) {
                throw std::runtime_error("failed to open output file: " + paths.summary_path.string());
            }

            write_full_rows(full_file, rows);
            write_summary_rows(summary_file, rows);

            std::cerr << "Wrote full CSV to " << paths.full_path << '\n';
            std::cerr << "Wrote summary CSV to " << paths.summary_path << '\n';
            return 0;
        }

        write_full_rows(std::cout, rows);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
}
