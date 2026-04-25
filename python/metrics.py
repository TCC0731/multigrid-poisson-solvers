from __future__ import annotations

import numpy as np

from operators import residual


def residual_l2(problem, phi: np.ndarray) -> float:
    return problem.h * np.linalg.norm(residual(phi, problem.rhs, problem.h))


def error_metrics(problem, phi: np.ndarray) -> tuple[float, float]:
    err = phi[1:-1, 1:-1] - problem.exact[1:-1, 1:-1]
    return problem.h * np.linalg.norm(err), float(np.max(np.abs(err)))
