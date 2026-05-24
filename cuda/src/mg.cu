#include "poisson/mg.hpp"

#include "poisson/cuda_utils.hpp"
#include "poisson/validation.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace poisson {
namespace {

enum class CoarseSolve {
    Exact,
    Sor,
};

class ScopedCudaStream {
public:
    ScopedCudaStream() {
        cuda::check(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags",
            __FILE__,
            __LINE__
        );
    }

    ~ScopedCudaStream() {
        if (stream_ != nullptr) {
            (void)cudaStreamDestroy(stream_);
        }
    }

    ScopedCudaStream(const ScopedCudaStream&) = delete;
    ScopedCudaStream& operator=(const ScopedCudaStream&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_{nullptr};
};

class ScopedCudaGraph {
public:
    ScopedCudaGraph() = default;

    ~ScopedCudaGraph() {
        if (graph_ != nullptr) {
            (void)cudaGraphDestroy(graph_);
        }
    }

    ScopedCudaGraph(const ScopedCudaGraph&) = delete;
    ScopedCudaGraph& operator=(const ScopedCudaGraph&) = delete;

    [[nodiscard]] cudaGraph_t get() const noexcept { return graph_; }

    void reset(cudaGraph_t graph = nullptr) noexcept {
        if (graph_ != nullptr) {
            (void)cudaGraphDestroy(graph_);
        }
        graph_ = graph;
    }

private:
    cudaGraph_t graph_{nullptr};
};

class ScopedCudaGraphExec {
public:
    ScopedCudaGraphExec() = default;

    ~ScopedCudaGraphExec() {
        if (graph_exec_ != nullptr) {
            (void)cudaGraphExecDestroy(graph_exec_);
        }
    }

    ScopedCudaGraphExec(const ScopedCudaGraphExec&) = delete;
    ScopedCudaGraphExec& operator=(const ScopedCudaGraphExec&) = delete;

    [[nodiscard]] cudaGraphExec_t get() const noexcept { return graph_exec_; }

    [[nodiscard]] cudaGraphExec_t* put() noexcept {
        if (graph_exec_ != nullptr) {
            (void)cudaGraphExecDestroy(graph_exec_);
            graph_exec_ = nullptr;
        }
        return &graph_exec_;
    }

private:
    cudaGraphExec_t graph_exec_{nullptr};
};

template <typename Real>
constexpr Real pi_v = static_cast<Real>(3.14159265358979323846264338327950288L);

using MgClock = std::chrono::steady_clock;

[[nodiscard]] double elapsed_ms(
    const MgClock::time_point& start, const MgClock::time_point& end
) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

__global__ void update_mg_loop_state_kernel(
    const double* residual,
    double tol,
    std::size_t max_iter,
    std::size_t* iteration_count,
    cudaGraphConditionalHandle handle
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    const std::size_t next_iteration = iteration_count[0] + 1;
    iteration_count[0] = next_iteration;
    const unsigned int keep_iterating =
        (next_iteration < max_iter && residual[0] > tol) ? 1U : 0U;
    cudaGraphSetConditional(handle, keep_iterating);
}

template <typename Real>
Real default_sor_omega(std::size_t interior_n) {
    if (interior_n < 1) {
        throw std::invalid_argument("interior_n must be positive");
    }

    const Real angle = pi_v<Real> / (static_cast<Real>(interior_n) + Real{1});
    return Real{2} / (Real{1} + std::sin(angle));
}

template <typename Real>
Real effective_mg_omega(const Problem2D<Real>& problem, const MGOptions<Real>& options) {
    if (options.omega_is_auto) {
        return default_sor_omega<Real>(problem.interior_n);
    }
    return options.omega;
}

template <typename Real>
Real effective_mg_omega(const Problem3D<Real>& problem, const MGOptions<Real>& options) {
    if (options.omega_is_auto) {
        return default_sor_omega<Real>(problem.interior_n);
    }
    return options.omega;
}

[[nodiscard]] std::string make_mg_cycle_label(
    std::size_t level_index, std::size_t interior_n, MGCycle cycle, CoarseSolve coarse_mode
) {
    std::string label{"mg::cycle[level="};
    label += std::to_string(level_index);
    label += ",n=";
    label += std::to_string(interior_n);
    label += ",cycle=";
    label += ((cycle == MGCycle::V) ? "V" : "W");
    label += ",coarse=";
    label += ((coarse_mode == CoarseSolve::Exact) ? "exact" : "sor");
    label += "]";
    return label;
}

[[nodiscard]] bool use_fused_small_grid_smoother_2d(std::size_t array_n) {
    return array_n >= 3 && array_n <= cuda_kernels::kFusedSmallGridSorMaxArrayN2D;
}

template <typename Real, typename PhiGrid, typename RhsGrid>
void run_mg_smoother_2d(
    PhiGrid& phi,
    const RhsGrid& rhs,
    Real h,
    Real omega,
    std::size_t steps,
    cudaStream_t stream
) {
    if (use_fused_small_grid_smoother_2d(phi.size())) {
        cuda::run_fused_rb_sor_steps(phi, rhs, h, omega, steps, stream);
        return;
    }
    cuda::run_rb_sor_steps(phi, rhs, h, omega, steps, stream);
}

[[nodiscard]] bool use_fused_small_grid_smoother_3d(std::size_t array_n) {
    return array_n >= 3 && array_n <= cuda_kernels::kFusedSmallGridSorMaxArrayN3D;
}

template <typename Real, typename PhiGrid, typename RhsGrid>
void run_mg_smoother_3d(
    PhiGrid& phi,
    const RhsGrid& rhs,
    Real h,
    Real omega,
    std::size_t steps,
    cudaStream_t stream
) {
    if (use_fused_small_grid_smoother_3d(phi.size())) {
        cuda::run_fused_rb_sor_steps(phi, rhs, h, omega, steps, stream);
        return;
    }
    cuda::run_rb_sor_steps(phi, rhs, h, omega, steps, stream);
}

template <typename Real>
struct MGLevelWorkspace {
    // The 2D path now fuses residual computation and restriction, so each
    // level only needs the coarse rhs plus the coarse error used during
    // recursion.
    cuda::DeviceGridView2D<Real> coarse_rhs;
    cuda::DeviceGridView2D<Real> coarse_error;
};

template <typename Real>
struct MGLevelWorkspace3D {
    // The 3D path now fuses residual computation and restriction, so each
    // level only needs the coarse rhs plus the coarse error used during
    // recursion.
    cuda::DeviceGridView3D<Real> coarse_rhs;
    cuda::DeviceGridView3D<Real> coarse_error;
};

template <typename Real>
class MGWorkspace {
public:
    MGWorkspace() = default;

    explicit MGWorkspace(std::size_t fine_array_n) {
        reserve_for(fine_array_n);
    }

    void reserve_for(std::size_t fine_array_n) {
        const cuda::detail::ScopedNvtxRange range{"mg::workspace_reserve"};
        if (fine_array_n == configured_array_n_ && !levels_.empty()) {
            return;
        }

        configured_array_n_ = fine_array_n;
        levels_.clear();
        layouts_.clear();

        const std::size_t required_elements = append_layout(fine_array_n, 0);
        if (required_elements > pool_capacity_elements_) {
            const cuda::detail::ScopedNvtxRange grow_range{"mg::workspace_grow_pool"};
            pool_ = cuda::DeviceBuffer<Real>{required_elements};
            pool_capacity_elements_ = required_elements;
        }

        levels_.reserve(layouts_.size());
        Real* const pool_data = pool_.data();
        for (const auto& layout : layouts_) {
            levels_.push_back(MGLevelWorkspace<Real>{
                cuda::DeviceGridView2D<Real>{
                    pool_data + layout.coarse_rhs_offset,
                    layout.coarse_array_n,
                },
                cuda::DeviceGridView2D<Real>{
                    pool_data + layout.coarse_error_offset,
                    layout.coarse_array_n,
                },
            });
        }
    }

    [[nodiscard]] MGLevelWorkspace<Real>& level(std::size_t index) {
        if (index >= levels_.size()) {
            throw std::logic_error("MG workspace level index out of range");
        }
        return levels_[index];
    }

private:
    struct MGLevelLayout {
        std::size_t coarse_array_n{};
        std::size_t coarse_rhs_offset{};
        std::size_t coarse_error_offset{};
    };

    [[nodiscard]] std::size_t append_layout(std::size_t fine_array_n, std::size_t base_offset) {
        const std::size_t interior_n = fine_array_n - 2;
        if (interior_n <= 4) {
            return 0;
        }

        const std::size_t coarse_interior_n = (interior_n - 1) / 2;
        const std::size_t coarse_array_n = coarse_interior_n + 2;
        const std::size_t coarse_elements = coarse_array_n * coarse_array_n;

        const std::size_t level_index = layouts_.size();
        layouts_.push_back(MGLevelLayout{
            coarse_array_n,
            0,
            base_offset,
        });

        const std::size_t child_elements =
            append_layout(coarse_array_n, base_offset + coarse_elements);

        layouts_[level_index].coarse_rhs_offset = base_offset + coarse_elements + child_elements;
        return 2 * coarse_elements + child_elements;
    }

    std::size_t configured_array_n_{0};
    std::size_t pool_capacity_elements_{0};
    cuda::DeviceBuffer<Real> pool_{};
    std::vector<MGLevelLayout> layouts_{};
    std::vector<MGLevelWorkspace<Real>> levels_{};
};

template <typename Real>
class MGWorkspace3D {
public:
    MGWorkspace3D() = default;

    explicit MGWorkspace3D(std::size_t fine_array_n) {
        reserve_for(fine_array_n);
    }

    void reserve_for(std::size_t fine_array_n) {
        const cuda::detail::ScopedNvtxRange range{"mg::workspace_reserve_3d"};
        if (fine_array_n == configured_array_n_ && !levels_.empty()) {
            return;
        }

        configured_array_n_ = fine_array_n;
        levels_.clear();
        layouts_.clear();

        const std::size_t required_elements = append_layout(fine_array_n, 0);
        if (required_elements > pool_capacity_elements_) {
            const cuda::detail::ScopedNvtxRange grow_range{"mg::workspace_grow_pool_3d"};
            pool_ = cuda::DeviceBuffer<Real>{required_elements};
            pool_capacity_elements_ = required_elements;
        }

        levels_.reserve(layouts_.size());
        Real* const pool_data = pool_.data();
        for (const auto& layout : layouts_) {
            levels_.push_back(MGLevelWorkspace3D<Real>{
                cuda::DeviceGridView3D<Real>{
                    pool_data + layout.coarse_rhs_offset,
                    layout.coarse_array_n,
                },
                cuda::DeviceGridView3D<Real>{
                    pool_data + layout.coarse_error_offset,
                    layout.coarse_array_n,
                },
            });
        }
    }

    [[nodiscard]] MGLevelWorkspace3D<Real>& level(std::size_t index) {
        if (index >= levels_.size()) {
            throw std::logic_error("MG workspace level index out of range");
        }
        return levels_[index];
    }

private:
    struct MGLevelLayout {
        std::size_t coarse_array_n{};
        std::size_t coarse_rhs_offset{};
        std::size_t coarse_error_offset{};
    };

    [[nodiscard]] std::size_t append_layout(std::size_t fine_array_n, std::size_t base_offset) {
        const std::size_t interior_n = fine_array_n - 2;
        if (interior_n <= 4) {
            return 0;
        }

        const std::size_t coarse_interior_n = (interior_n - 1) / 2;
        const std::size_t coarse_array_n = coarse_interior_n + 2;
        const std::size_t coarse_elements = coarse_array_n * coarse_array_n * coarse_array_n;

        const std::size_t level_index = layouts_.size();
        layouts_.push_back(MGLevelLayout{
            coarse_array_n,
            base_offset,
            base_offset,
        });

        const std::size_t child_elements =
            append_layout(coarse_array_n, base_offset + coarse_elements);

        layouts_[level_index].coarse_rhs_offset = base_offset + coarse_elements + child_elements;
        return 2 * coarse_elements + child_elements;
    }

    std::size_t configured_array_n_{0};
    std::size_t pool_capacity_elements_{0};
    cuda::DeviceBuffer<Real> pool_{};
    std::vector<MGLevelLayout> layouts_{};
    std::vector<MGLevelWorkspace3D<Real>> levels_{};
};

template <typename Real, typename PhiGrid, typename RhsGrid>
void solve_coarsest_exact(PhiGrid& phi, const RhsGrid& rhs, Real h, cudaStream_t stream) {
    const cuda::detail::ScopedNvtxRange range{"mg::solve_coarsest_exact"};
    cuda::run_exact_coarse_solve(phi, rhs, h, stream);
}

template <typename Real, typename PhiGrid, typename RhsGrid>
void solve_coarsest_exact_3d(PhiGrid& phi, const RhsGrid& rhs, Real h, cudaStream_t stream) {
    const cuda::detail::ScopedNvtxRange range{"mg::solve_coarsest_exact_3d"};
    cuda::run_exact_coarse_solve(phi, rhs, h, stream);
}

template <typename Real, typename PhiGrid, typename RhsGrid>
void mg_cycle(
    PhiGrid& phi,
    const RhsGrid& rhs,
    Real h,
    Real omega,
    std::size_t nu,
    MGCycle cycle,
    CoarseSolve coarse_mode,
    std::size_t coarse_steps,
    MGWorkspace<Real>& workspace,
    std::size_t level_index,
    cudaStream_t stream
) {
    const std::size_t n = phi.size() - 2;
    const std::string cycle_label = make_mg_cycle_label(level_index, n, cycle, coarse_mode);
    const cuda::detail::ScopedNvtxRange cycle_range{cycle_label};

    if (n <= 4) {
        if (coarse_mode == CoarseSolve::Exact) {
            solve_coarsest_exact(phi, rhs, h, stream);
        } else {
            cuda::run_fused_rb_sor_steps(phi, rhs, h, omega, coarse_steps, stream);
        }
        return;
    }

    {
        const cuda::detail::ScopedNvtxRange pre_smooth_range{"mg::pre_smooth"};
        run_mg_smoother_2d(phi, rhs, h, omega, nu, stream);
    }

    auto& level = workspace.level(level_index);
    auto& coarse_rhs = level.coarse_rhs;
    {
        const cuda::detail::ScopedNvtxRange fused_range{"mg::compute_residual_restrict"};
        cuda::compute_residual_restrict_full_weighting<Real>(phi, rhs, h, coarse_rhs, stream);
    }

    auto& coarse_error = level.coarse_error;
    {
        const cuda::detail::ScopedNvtxRange coarse_correction_range{"mg::coarse_correction"};
        const std::size_t coarse_n = coarse_rhs.size() - 2;
        coarse_error.zero(stream);
        mg_cycle<Real>(
            coarse_error,
            coarse_rhs,
            Real{2} * h,
            omega,
            nu,
            cycle,
            coarse_mode,
            coarse_steps,
            workspace,
            level_index + 1,
            stream
        );
        if (cycle == MGCycle::W && coarse_n > 4) {
            mg_cycle<Real>(
                coarse_error,
                coarse_rhs,
                Real{2} * h,
                omega,
                nu,
                cycle,
                coarse_mode,
                coarse_steps,
                workspace,
                level_index + 1,
                stream
            );
        }
    }

    {
        const cuda::detail::ScopedNvtxRange prolong_range{"mg::prolongate"};
        cuda::prolong_add<Real>(coarse_error, phi, stream);
    }
    {
        const cuda::detail::ScopedNvtxRange post_smooth_range{"mg::post_smooth"};
        run_mg_smoother_2d(phi, rhs, h, omega, nu, stream);
    }
}

template <typename Real, typename PhiGrid, typename RhsGrid>
void mg_cycle_3d(
    PhiGrid& phi,
    const RhsGrid& rhs,
    Real h,
    Real omega,
    std::size_t nu,
    MGCycle cycle,
    CoarseSolve coarse_mode,
    std::size_t coarse_steps,
    MGWorkspace3D<Real>& workspace,
    std::size_t level_index,
    cudaStream_t stream
) {
    const std::size_t n = phi.size() - 2;
    const std::string cycle_label = make_mg_cycle_label(level_index, n, cycle, coarse_mode);
    const cuda::detail::ScopedNvtxRange cycle_range{cycle_label};

    if (n <= 4) {
        if (coarse_mode == CoarseSolve::Exact) {
            solve_coarsest_exact_3d(phi, rhs, h, stream);
        } else {
            cuda::run_fused_rb_sor_steps(phi, rhs, h, omega, coarse_steps, stream);
        }
        return;
    }

    {
        const cuda::detail::ScopedNvtxRange pre_smooth_range{"mg::pre_smooth_3d"};
        run_mg_smoother_3d(phi, rhs, h, omega, nu, stream);
    }

    auto& level = workspace.level(level_index);
    auto& coarse_rhs = level.coarse_rhs;
    {
        const cuda::detail::ScopedNvtxRange fused_range{"mg::compute_residual_restrict_3d"};
        cuda::compute_residual_restrict_full_weighting<Real>(phi, rhs, h, coarse_rhs, stream);
    }

    auto& coarse_error = level.coarse_error;
    {
        const cuda::detail::ScopedNvtxRange coarse_correction_range{"mg::coarse_correction_3d"};
        const std::size_t coarse_n = coarse_rhs.size() - 2;
        coarse_error.zero(stream);
        mg_cycle_3d<Real>(
            coarse_error,
            coarse_rhs,
            Real{2} * h,
            omega,
            nu,
            cycle,
            coarse_mode,
            coarse_steps,
            workspace,
            level_index + 1,
            stream
        );
        if (cycle == MGCycle::W && coarse_n > 4) {
            mg_cycle_3d<Real>(
                coarse_error,
                coarse_rhs,
                Real{2} * h,
                omega,
                nu,
                cycle,
                coarse_mode,
                coarse_steps,
                workspace,
                level_index + 1,
                stream
            );
        }
    }

    {
        const cuda::detail::ScopedNvtxRange prolong_range{"mg::prolongate_3d"};
        cuda::prolong_add(coarse_error, phi, stream);
    }
    {
        const cuda::detail::ScopedNvtxRange post_smooth_range{"mg::post_smooth_3d"};
        run_mg_smoother_3d(phi, rhs, h, omega, nu, stream);
    }
}

template <typename Real>
void validate_mg_inputs(
    const Problem2D<Real>& problem, const MGOptions<Real>& options, CoarseSolve coarse_mode
) {
    if (options.tol <= Real{}) {
        throw std::invalid_argument("tol must be positive");
    }
    if (options.max_iter < 1) {
        throw std::invalid_argument("max_iter must be positive");
    }
    if (options.nu < 1) {
        throw std::invalid_argument("nu must be positive");
    }
    if (coarse_mode == CoarseSolve::Sor && options.coarse_steps < 1) {
        throw std::invalid_argument("coarse_steps must be positive");
    }
    if (!options.omega_is_auto && (options.omega <= Real{} || options.omega > Real{2})) {
        throw std::invalid_argument("omega must be in (0, 2]");
    }
    if (problem.array_n() < 3) {
        throw std::invalid_argument("problem must include at least one interior cell");
    }
    const ValidationReport report = validate_problem(problem);
    if (!report.ok) {
        throw std::invalid_argument("problem is invalid");
    }
}

template <typename Real>
void validate_mg_inputs(
    const Problem3D<Real>& problem, const MGOptions<Real>& options, CoarseSolve coarse_mode
) {
    if (options.tol <= Real{}) {
        throw std::invalid_argument("tol must be positive");
    }
    if (options.max_iter < 1) {
        throw std::invalid_argument("max_iter must be positive");
    }
    if (options.nu < 1) {
        throw std::invalid_argument("nu must be positive");
    }
    if (coarse_mode == CoarseSolve::Sor && options.coarse_steps < 1) {
        throw std::invalid_argument("coarse_steps must be positive");
    }
    if (!options.omega_is_auto && (options.omega <= Real{} || options.omega > Real{2})) {
        throw std::invalid_argument("omega must be in (0, 2]");
    }
    if (problem.array_n() < 3) {
        throw std::invalid_argument("problem must include at least one interior cell");
    }
    const ValidationReport report = validate_problem(problem);
    if (!report.ok) {
        throw std::invalid_argument("problem is invalid");
    }
}

template <typename Real>
SolveResult solve_mg_impl(
    const Problem2D<Real>& problem, const MGOptions<Real>& options, CoarseSolve coarse_mode
) {
    const cuda::detail::ScopedNvtxRange solve_range{"mg::solve"};
    validate_mg_inputs(problem, options, coarse_mode);
    cuda::ensure_device_available();

    const Real omega = effective_mg_omega(problem, options);
    cuda::DeviceGrid2D<Real> phi{problem.phi0};
    const cuda::DeviceGrid2D<Real> rhs{problem.rhs};
    thread_local MGWorkspace<Real> workspace{};
    {
        const cuda::detail::ScopedNvtxRange workspace_range{"mg::workspace_setup"};
        workspace.reserve_for(problem.array_n());
    }
    cuda::RelativeResidualWorkspace residual_workspace{problem.array_n()};
    const MgClock::time_point graph_timing_start = MgClock::now();
    ScopedCudaStream graph_stream{};
    cuda::ensure_rhs_norm_cached(rhs, problem.h, residual_workspace);
    ScopedCudaGraph graph{};
    cudaGraph_t created_graph = nullptr;
    cuda::check(cudaGraphCreate(&created_graph, 0), "cudaGraphCreate", __FILE__, __LINE__);
    graph.reset(created_graph);
    cuda::DeviceBuffer<std::size_t> iteration_count{1};
    cudaGraphConditionalHandle while_handle{};
    cuda::check(
        cudaGraphConditionalHandleCreate(
            &while_handle, graph.get(), 1, cudaGraphCondAssignDefault
        ),
        "cudaGraphConditionalHandleCreate",
        __FILE__,
        __LINE__
    );
    cudaGraphNode_t conditional_node = nullptr;
    cudaGraphNodeParams conditional_params{};
    conditional_params.type = cudaGraphNodeTypeConditional;
    conditional_params.conditional.handle = while_handle;
    conditional_params.conditional.type = cudaGraphCondTypeWhile;
    conditional_params.conditional.size = 1;
    cuda::check(
        cudaGraphAddNode(&conditional_node, graph.get(), nullptr, 0, &conditional_params),
        "cudaGraphAddNode",
        __FILE__,
        __LINE__
    );
    cudaGraph_t body_graph = conditional_params.conditional.phGraph_out[0];
    {
        const cuda::detail::ScopedNvtxRange capture_range{"mg::graph_capture"};
        cuda::check(
            cudaStreamBeginCaptureToGraph(
                graph_stream.get(),
                body_graph,
                nullptr,
                nullptr,
                0,
                cudaStreamCaptureModeRelaxed
            ),
            "cudaStreamBeginCaptureToGraph",
            __FILE__,
            __LINE__
        );
        try {
            mg_cycle<Real>(
                phi,
                rhs,
                problem.h,
                omega,
                options.nu,
                options.cycle,
                coarse_mode,
                options.coarse_steps,
                workspace,
                0,
                graph_stream.get()
            );
            cuda::compute_relative_residual_device(
                phi, rhs, problem.h, residual_workspace, graph_stream.get()
            );
            update_mg_loop_state_kernel<<<1, 1, 0, graph_stream.get()>>>(
                residual_workspace.final_residual(),
                static_cast<double>(options.tol),
                options.max_iter,
                iteration_count.data(),
                while_handle
            );
            cuda::check_kernel("update_mg_loop_state_kernel", graph_stream.get());
        } catch (...) {
            (void)cudaStreamEndCapture(graph_stream.get(), nullptr);
            throw;
        }

        cuda::check(
            cudaStreamEndCapture(graph_stream.get(), nullptr),
            "cudaStreamEndCapture",
            __FILE__,
            __LINE__
        );
    }

    ScopedCudaGraphExec graph_exec{};
    cuda::check(
        cudaGraphInstantiate(graph_exec.put(), graph.get(), nullptr, nullptr, 0),
        "cudaGraphInstantiate",
        __FILE__,
        __LINE__
    );
    iteration_count.zero(graph_stream.get());
    cuda::check(
        cudaStreamSynchronize(graph_stream.get()),
        "cudaStreamSynchronize",
        __FILE__,
        __LINE__
    );
    const MgClock::time_point compute_timing_start = MgClock::now();
    cuda::check(
        cudaGraphLaunch(graph_exec.get(), graph_stream.get()),
        "cudaGraphLaunch",
        __FILE__,
        __LINE__
    );
    cuda::check(
        cudaStreamSynchronize(graph_stream.get()),
        "cudaStreamSynchronize",
        __FILE__,
        __LINE__
    );

    std::array<std::size_t, 1> host_iterations{};
    iteration_count.download(host_iterations.data(), 1);
    std::array<double, 1> host_residual{};
    cuda::check(
        cudaMemcpy(
            host_residual.data(),
            residual_workspace.final_residual(),
            sizeof(double),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpyDeviceToHost",
        __FILE__,
        __LINE__
    );
    auto host_phi = phi.download();
    const MgClock::time_point solve_end = MgClock::now();
    return make_solve_result(
        std::move(host_phi),
        host_iterations[0],
        static_cast<Real>(host_residual[0]),
        elapsed_ms(compute_timing_start, solve_end),
        elapsed_ms(graph_timing_start, solve_end)
    );
}

template <typename Real>
SolveResult3D solve_mg_3d_impl(
    const Problem3D<Real>& problem, const MGOptions<Real>& options, CoarseSolve coarse_mode
) {
    const cuda::detail::ScopedNvtxRange solve_range{"mg::solve_3d"};
    validate_mg_inputs(problem, options, coarse_mode);
    cuda::ensure_device_available();

    const Real omega = effective_mg_omega(problem, options);
    cuda::DeviceGrid3D<Real> phi{problem.phi0};
    const cuda::DeviceGrid3D<Real> rhs{problem.rhs};
    thread_local MGWorkspace3D<Real> workspace{};
    {
        const cuda::detail::ScopedNvtxRange workspace_range{"mg::workspace_setup_3d"};
        workspace.reserve_for(problem.array_n());
    }
    cuda::RelativeResidualWorkspace3D residual_workspace{problem.array_n()};
    const MgClock::time_point graph_timing_start = MgClock::now();
    ScopedCudaStream graph_stream{};
    cuda::ensure_rhs_norm_cached(rhs, problem.h, residual_workspace);
    ScopedCudaGraph graph{};
    cudaGraph_t created_graph = nullptr;
    cuda::check(cudaGraphCreate(&created_graph, 0), "cudaGraphCreate", __FILE__, __LINE__);
    graph.reset(created_graph);
    cuda::DeviceBuffer<std::size_t> iteration_count{1};
    cudaGraphConditionalHandle while_handle{};
    cuda::check(
        cudaGraphConditionalHandleCreate(
            &while_handle, graph.get(), 1, cudaGraphCondAssignDefault
        ),
        "cudaGraphConditionalHandleCreate",
        __FILE__,
        __LINE__
    );
    cudaGraphNode_t conditional_node = nullptr;
    cudaGraphNodeParams conditional_params{};
    conditional_params.type = cudaGraphNodeTypeConditional;
    conditional_params.conditional.handle = while_handle;
    conditional_params.conditional.type = cudaGraphCondTypeWhile;
    conditional_params.conditional.size = 1;
    cuda::check(
        cudaGraphAddNode(&conditional_node, graph.get(), nullptr, 0, &conditional_params),
        "cudaGraphAddNode",
        __FILE__,
        __LINE__
    );
    cudaGraph_t body_graph = conditional_params.conditional.phGraph_out[0];
    {
        const cuda::detail::ScopedNvtxRange capture_range{"mg::graph_capture_3d"};
        cuda::check(
            cudaStreamBeginCaptureToGraph(
                graph_stream.get(),
                body_graph,
                nullptr,
                nullptr,
                0,
                cudaStreamCaptureModeRelaxed
            ),
            "cudaStreamBeginCaptureToGraph",
            __FILE__,
            __LINE__
        );
        try {
            mg_cycle_3d<Real>(
                phi,
                rhs,
                problem.h,
                omega,
                options.nu,
                options.cycle,
                coarse_mode,
                options.coarse_steps,
                workspace,
                0,
                graph_stream.get()
            );
            cuda::compute_relative_residual_device(
                phi, rhs, problem.h, residual_workspace, graph_stream.get()
            );
            update_mg_loop_state_kernel<<<1, 1, 0, graph_stream.get()>>>(
                residual_workspace.final_residual(),
                static_cast<double>(options.tol),
                options.max_iter,
                iteration_count.data(),
                while_handle
            );
            cuda::check_kernel("update_mg_loop_state_kernel", graph_stream.get());
        } catch (...) {
            (void)cudaStreamEndCapture(graph_stream.get(), nullptr);
            throw;
        }

        cuda::check(
            cudaStreamEndCapture(graph_stream.get(), nullptr),
            "cudaStreamEndCapture",
            __FILE__,
            __LINE__
        );
    }

    ScopedCudaGraphExec graph_exec{};
    cuda::check(
        cudaGraphInstantiate(graph_exec.put(), graph.get(), nullptr, nullptr, 0),
        "cudaGraphInstantiate",
        __FILE__,
        __LINE__
    );
    iteration_count.zero(graph_stream.get());
    cuda::check(
        cudaStreamSynchronize(graph_stream.get()),
        "cudaStreamSynchronize",
        __FILE__,
        __LINE__
    );
    const MgClock::time_point compute_timing_start = MgClock::now();
    cuda::check(
        cudaGraphLaunch(graph_exec.get(), graph_stream.get()),
        "cudaGraphLaunch",
        __FILE__,
        __LINE__
    );
    cuda::check(
        cudaStreamSynchronize(graph_stream.get()),
        "cudaStreamSynchronize",
        __FILE__,
        __LINE__
    );

    std::array<std::size_t, 1> host_iterations{};
    iteration_count.download(host_iterations.data(), 1);
    std::array<double, 1> host_residual{};
    cuda::check(
        cudaMemcpy(
            host_residual.data(),
            residual_workspace.final_residual(),
            sizeof(double),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpyDeviceToHost",
        __FILE__,
        __LINE__
    );
    auto host_phi = phi.download();
    const MgClock::time_point solve_end = MgClock::now();
    return make_solve_result(
        std::move(host_phi),
        host_iterations[0],
        static_cast<Real>(host_residual[0]),
        elapsed_ms(compute_timing_start, solve_end),
        elapsed_ms(graph_timing_start, solve_end)
    );
}

} // namespace

template <typename Real>
SolveResult solve_mg_exact(const Problem2D<Real>& problem, const MGOptions<Real>& options) {
    return solve_mg_impl(problem, options, CoarseSolve::Exact);
}

template <typename Real>
SolveResult solve_mg_sor(const Problem2D<Real>& problem, const MGOptions<Real>& options) {
    return solve_mg_impl(problem, options, CoarseSolve::Sor);
}

template <typename Real>
SolveResult3D solve_mg_exact(const Problem3D<Real>& problem, const MGOptions<Real>& options) {
    return solve_mg_3d_impl(problem, options, CoarseSolve::Exact);
}

template <typename Real>
SolveResult3D solve_mg_sor(const Problem3D<Real>& problem, const MGOptions<Real>& options) {
    return solve_mg_3d_impl(problem, options, CoarseSolve::Sor);
}

template SolveResult solve_mg_exact<float>(const Problem2D<float>&, const MGOptions<float>&);
template SolveResult solve_mg_exact<double>(const Problem2D<double>&, const MGOptions<double>&);
template SolveResult solve_mg_sor<float>(const Problem2D<float>&, const MGOptions<float>&);
template SolveResult solve_mg_sor<double>(const Problem2D<double>&, const MGOptions<double>&);
template SolveResult3D solve_mg_exact<float>(const Problem3D<float>&, const MGOptions<float>&);
template SolveResult3D solve_mg_exact<double>(const Problem3D<double>&, const MGOptions<double>&);
template SolveResult3D solve_mg_sor<float>(const Problem3D<float>&, const MGOptions<float>&);
template SolveResult3D solve_mg_sor<double>(const Problem3D<double>&, const MGOptions<double>&);

} // namespace poisson
