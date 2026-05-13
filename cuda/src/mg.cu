#include "poisson/mg.hpp"

#include "poisson/cuda_utils.hpp"
#include "poisson/validation.hpp"

#include <algorithm>
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

template <typename Real>
constexpr Real pi_v = static_cast<Real>(3.14159265358979323846264338327950288L);

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
void run_mg_smoother_2d(PhiGrid& phi, const RhsGrid& rhs, Real h, Real omega, std::size_t steps) {
    if (use_fused_small_grid_smoother_2d(phi.size())) {
        cuda::run_fused_rb_sor_steps(phi, rhs, h, omega, steps);
        return;
    }
    cuda::run_rb_sor_steps(phi, rhs, h, omega, steps);
}

template <typename Real>
struct MGLevelWorkspace {
    cuda::DeviceGridView2D<Real> fine_residual;
    cuda::DeviceGridView2D<Real> coarse_rhs;
    cuda::DeviceGridView2D<Real> coarse_error;
};

template <typename Real>
struct MGLevelWorkspace3D {
    cuda::DeviceGridView3D<Real> fine_residual;
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
                    pool_data + layout.fine_residual_offset,
                    layout.fine_array_n,
                },
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
        std::size_t fine_array_n{};
        std::size_t coarse_array_n{};
        std::size_t fine_residual_offset{};
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
        const std::size_t fine_elements = fine_array_n * fine_array_n;
        const std::size_t coarse_elements = coarse_array_n * coarse_array_n;

        const std::size_t level_index = layouts_.size();
        layouts_.push_back(MGLevelLayout{
            fine_array_n,
            coarse_array_n,
            base_offset,
            0,
            base_offset,
        });

        const std::size_t child_elements =
            append_layout(coarse_array_n, base_offset + coarse_elements);
        // Reuse the fine-residual slot for coarse_error plus the entire child
        // subtree once restriction has finished.
        const std::size_t transient_or_child_elements =
            std::max(fine_elements, coarse_elements + child_elements);

        layouts_[level_index].coarse_rhs_offset = base_offset + transient_or_child_elements;
        return transient_or_child_elements + coarse_elements;
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
                    pool_data + layout.fine_residual_offset,
                    layout.fine_array_n,
                },
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
        std::size_t fine_array_n{};
        std::size_t coarse_array_n{};
        std::size_t fine_residual_offset{};
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
        const std::size_t fine_elements = fine_array_n * fine_array_n * fine_array_n;
        const std::size_t coarse_elements = coarse_array_n * coarse_array_n * coarse_array_n;

        const std::size_t level_index = layouts_.size();
        layouts_.push_back(MGLevelLayout{
            fine_array_n,
            coarse_array_n,
            base_offset,
            0,
            base_offset,
        });

        const std::size_t child_elements =
            append_layout(coarse_array_n, base_offset + coarse_elements);
        const std::size_t transient_or_child_elements =
            std::max(fine_elements, coarse_elements + child_elements);

        layouts_[level_index].coarse_rhs_offset = base_offset + transient_or_child_elements;
        return transient_or_child_elements + coarse_elements;
    }

    std::size_t configured_array_n_{0};
    std::size_t pool_capacity_elements_{0};
    cuda::DeviceBuffer<Real> pool_{};
    std::vector<MGLevelLayout> layouts_{};
    std::vector<MGLevelWorkspace3D<Real>> levels_{};
};

template <typename Real>
void copy_boundary_values(const Grid3D<Real>& source, Grid3D<Real>& target) {
    if (source.size() != target.size()) {
        throw std::invalid_argument("boundary template size mismatch");
    }

    if (source.size() == 0) {
        return;
    }

    const std::size_t last = source.size() - 1;
    for (std::size_t i = 0; i <= last; ++i) {
        for (std::size_t j = 0; j <= last; ++j) {
            target.unchecked(i, j, 0) = source.unchecked(i, j, 0);
            target.unchecked(i, j, last) = source.unchecked(i, j, last);
        }
    }
    for (std::size_t i = 0; i <= last; ++i) {
        for (std::size_t k = 0; k <= last; ++k) {
            target.unchecked(i, 0, k) = source.unchecked(i, 0, k);
            target.unchecked(i, last, k) = source.unchecked(i, last, k);
        }
    }
    for (std::size_t j = 0; j <= last; ++j) {
        for (std::size_t k = 0; k <= last; ++k) {
            target.unchecked(0, j, k) = source.unchecked(0, j, k);
            target.unchecked(last, j, k) = source.unchecked(last, j, k);
        }
    }
}

template <typename Real, typename PhiGrid, typename RhsGrid>
void solve_coarsest_exact(PhiGrid& phi, const RhsGrid& rhs, Real h) {
    const cuda::detail::ScopedNvtxRange range{"mg::solve_coarsest_exact"};
    cuda::run_exact_coarse_solve(phi, rhs, h);
}

template <typename Real, typename PhiGrid, typename RhsGrid>
void solve_coarsest_exact_3d(
    PhiGrid& phi,
    const RhsGrid& rhs,
    Real h,
    const Grid3D<Real>* boundary_template
) {
    const cuda::detail::ScopedNvtxRange range{"mg::solve_coarsest_exact_3d"};
    Grid3D<Real> host_phi{phi.size()};
    if (boundary_template != nullptr) {
        copy_boundary_values(*boundary_template, host_phi);
    }
    const Grid3D<Real> host_rhs = rhs.download();

    const std::size_t n = host_rhs.size() - 2;
    const std::size_t m = n * n * n;
    const Real inv_h2 = Real{1} / (h * h);

    std::vector<Real> a(m * m, Real{});
    std::vector<Real> b(m, Real{});
    std::vector<Real> x(m, Real{});

    const auto index = [n](std::size_t i, std::size_t j, std::size_t k) {
        return (i * n + j) * n + k;
    };

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                const std::size_t row = index(i, j, k);
                a[row * m + row] = Real{6} * inv_h2;
                if (i > 0) {
                    a[row * m + index(i - 1, j, k)] = -inv_h2;
                }
                if (i + 1 < n) {
                    a[row * m + index(i + 1, j, k)] = -inv_h2;
                }
                if (j > 0) {
                    a[row * m + index(i, j - 1, k)] = -inv_h2;
                }
                if (j + 1 < n) {
                    a[row * m + index(i, j + 1, k)] = -inv_h2;
                }
                if (k > 0) {
                    a[row * m + index(i, j, k - 1)] = -inv_h2;
                }
                if (k + 1 < n) {
                    a[row * m + index(i, j, k + 1)] = -inv_h2;
                }
                b[row] = host_rhs.unchecked(i + 1, j + 1, k + 1);
            }
        }
    }

    for (std::size_t pivot_col = 0; pivot_col < m; ++pivot_col) {
        const Real pivot = a[pivot_col * m + pivot_col];
        for (std::size_t row = pivot_col + 1; row < m; ++row) {
            const Real factor = a[row * m + pivot_col] / pivot;
            for (std::size_t col = pivot_col; col < m; ++col) {
                a[row * m + col] -= factor * a[pivot_col * m + col];
            }
            b[row] -= factor * b[pivot_col];
        }
    }

    for (std::size_t row = m; row-- > 0;) {
        Real sum = b[row];
        for (std::size_t col = row + 1; col < m; ++col) {
            sum -= a[row * m + col] * x[col];
        }
        x[row] = sum / a[row * m + row];
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                host_phi.unchecked(i + 1, j + 1, k + 1) = x[index(i, j, k)];
            }
        }
    }

    phi.upload(host_phi);
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
    std::size_t level_index
) {
    const std::size_t n = phi.size() - 2;
    const std::string cycle_label = make_mg_cycle_label(level_index, n, cycle, coarse_mode);
    const cuda::detail::ScopedNvtxRange cycle_range{cycle_label};

    if (n <= 4) {
        if (coarse_mode == CoarseSolve::Exact) {
            solve_coarsest_exact(phi, rhs, h);
        } else {
            cuda::run_fused_rb_sor_steps(phi, rhs, h, omega, coarse_steps);
        }
        return;
    }

    {
        const cuda::detail::ScopedNvtxRange pre_smooth_range{"mg::pre_smooth"};
        run_mg_smoother_2d(phi, rhs, h, omega, nu);
    }

    auto& level = workspace.level(level_index);
    auto& fine_residual = level.fine_residual;
    {
        const cuda::detail::ScopedNvtxRange residual_range{"mg::compute_fine_residual"};
        cuda::compute_residual_full(phi, rhs, h, fine_residual);
    }

    auto& coarse_rhs = level.coarse_rhs;
    {
        const cuda::detail::ScopedNvtxRange restriction_range{"mg::restrict_to_coarse"};
        cuda::restrict_full_weighting<Real>(fine_residual, coarse_rhs);
    }

    auto& coarse_error = level.coarse_error;
    {
        const cuda::detail::ScopedNvtxRange coarse_correction_range{"mg::coarse_correction"};
        coarse_error.zero();
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
            level_index + 1
        );
        if (cycle == MGCycle::W) {
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
                level_index + 1
            );
        }
    }

    {
        const cuda::detail::ScopedNvtxRange prolong_range{"mg::prolongate"};
        cuda::prolong_add<Real>(coarse_error, phi);
    }
    {
        const cuda::detail::ScopedNvtxRange post_smooth_range{"mg::post_smooth"};
        run_mg_smoother_2d(phi, rhs, h, omega, nu);
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
    const Grid3D<Real>* boundary_template
) {
    const std::size_t n = phi.size() - 2;
    const std::string cycle_label = make_mg_cycle_label(level_index, n, cycle, coarse_mode);
    const cuda::detail::ScopedNvtxRange cycle_range{cycle_label};

    if (n <= 4) {
        if (coarse_mode == CoarseSolve::Exact) {
            solve_coarsest_exact_3d(phi, rhs, h, boundary_template);
        } else {
            cuda::run_fused_rb_sor_steps(phi, rhs, h, omega, coarse_steps);
        }
        return;
    }

    {
        const cuda::detail::ScopedNvtxRange pre_smooth_range{"mg::pre_smooth_3d"};
        cuda::run_rb_sor_steps(phi, rhs, h, omega, nu);
    }

    auto& level = workspace.level(level_index);
    auto& fine_residual = level.fine_residual;
    {
        const cuda::detail::ScopedNvtxRange residual_range{"mg::compute_fine_residual_3d"};
        cuda::compute_residual_full(phi, rhs, h, fine_residual);
    }

    auto& coarse_rhs = level.coarse_rhs;
    {
        const cuda::detail::ScopedNvtxRange restriction_range{"mg::restrict_to_coarse_3d"};
        cuda::restrict_full_weighting(fine_residual, coarse_rhs);
    }

    auto& coarse_error = level.coarse_error;
    {
        const cuda::detail::ScopedNvtxRange coarse_correction_range{"mg::coarse_correction_3d"};
        coarse_error.zero();
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
            static_cast<const Grid3D<Real>*>(nullptr)
        );
        if (cycle == MGCycle::W) {
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
                static_cast<const Grid3D<Real>*>(nullptr)
            );
        }
    }

    {
        const cuda::detail::ScopedNvtxRange prolong_range{"mg::prolongate_3d"};
        cuda::prolong_add(coarse_error, phi);
    }
    {
        const cuda::detail::ScopedNvtxRange post_smooth_range{"mg::post_smooth_3d"};
        cuda::run_rb_sor_steps(phi, rhs, h, omega, nu);
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

    for (std::size_t iteration = 1; iteration <= options.max_iter; ++iteration) {
        const cuda::detail::ScopedNvtxRange iteration_range{"mg::iteration"};
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
            0
        );

        const double residual =
            cuda::compute_relative_residual_uncached(phi, rhs, problem.h, residual_workspace);
        if (residual <= static_cast<double>(options.tol)) {
            return make_solve_result(phi.download(), iteration, static_cast<Real>(residual));
        }
    }

    const double residual =
        cuda::compute_relative_residual_uncached(phi, rhs, problem.h, residual_workspace);
    return make_solve_result(phi.download(), options.max_iter, static_cast<Real>(residual));
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

    for (std::size_t iteration = 1; iteration <= options.max_iter; ++iteration) {
        const cuda::detail::ScopedNvtxRange iteration_range{"mg::iteration_3d"};
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
            &problem.phi0
        );

        const double residual =
            cuda::compute_relative_residual_uncached_3d(phi, rhs, problem.h, residual_workspace);
        if (residual <= static_cast<double>(options.tol)) {
            return make_solve_result(phi.download(), iteration, static_cast<Real>(residual));
        }
    }

    const double residual =
        cuda::compute_relative_residual_uncached_3d(phi, rhs, problem.h, residual_workspace);
    return make_solve_result(phi.download(), options.max_iter, static_cast<Real>(residual));
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
