from __future__ import annotations

import numpy as np


def apply_A(phi: np.ndarray, h: float) -> np.ndarray:
    return (
        4.0 * phi[1:-1, 1:-1]
        - phi[2:, 1:-1]
        - phi[:-2, 1:-1]
        - phi[1:-1, 2:]
        - phi[1:-1, :-2]
    ) / (h * h)


def residual(phi: np.ndarray, rhs: np.ndarray, h: float) -> np.ndarray:
    return rhs[1:-1, 1:-1] - apply_A(phi, h)
