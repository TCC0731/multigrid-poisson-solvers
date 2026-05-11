from __future__ import annotations

import numpy as np

from solvers.utils import (
    _relative_physical_residual_l2,
    _relative_physical_residual_l2_3d,
)


def residual_l2(problem, phi: np.ndarray) -> float:
    return float(
        _relative_physical_residual_l2(phi, problem.rhs, problem.h)
    )


def error_metrics(problem, phi: np.ndarray) -> tuple[float, float]:
    err = phi[1:-1, 1:-1] - problem.exact[1:-1, 1:-1]
    return problem.h * np.linalg.norm(err), float(np.max(np.abs(err)))


def residual_l2_3d(problem, phi: np.ndarray) -> float:
    return float(
        _relative_physical_residual_l2_3d(phi, problem.rhs, problem.h)
    )


def error_metrics_3d(problem, phi: np.ndarray) -> tuple[float, float]:
    err = phi[1:-1, 1:-1, 1:-1] - problem.exact[1:-1, 1:-1, 1:-1]
    return problem.h**1.5 * np.linalg.norm(err), float(np.max(np.abs(err)))
