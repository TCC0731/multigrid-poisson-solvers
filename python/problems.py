from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CASE_FUNCS = {
    "sine": (
        lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y),
        lambda x, y: 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y),
    ),
    "mixed_sine": (
        lambda x, y: np.sin(2.0 * np.pi * x) * np.sin(3.0 * np.pi * y),
        lambda x, y: 13.0 * np.pi**2 * np.sin(2.0 * np.pi * x) * np.sin(3.0 * np.pi * y),
    ),
    "bubble": (
        lambda x, y: x * (1.0 - x) * y * (1.0 - y),
        lambda x, y: 2.0 * x * (1.0 - x) + 2.0 * y * (1.0 - y),
    ),
    "exp": (
        lambda x, y: np.exp(x + y),
        lambda x, y: -2.0 * np.exp(x + y),
    ),
    "cosine": (
        lambda x, y: np.cos(np.pi * x) * np.cos(np.pi * y),
        lambda x, y: 2.0 * np.pi**2 * np.cos(np.pi * x) * np.cos(np.pi * y),
    ),
}
CASES = tuple(CASE_FUNCS)


@dataclass(frozen=True)
class Problem2D:
    case: str
    grid_size: int
    h: float
    exact: np.ndarray
    rhs: np.ndarray
    phi0: np.ndarray


def _coerce_dtype(dtype):
    try:
        dtype_type = np.dtype(dtype).type
    except TypeError as exc:
        raise ValueError(f"unsupported dtype {dtype!r}") from exc
    if dtype_type not in (np.float32, np.float64):
        raise ValueError("dtype must be float32 or float64")
    return dtype_type


def _problem(exact_fn, rhs_fn, case: str, grid_size: int, dtype=np.float64) -> Problem2D:
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    dtype_type = _coerce_dtype(dtype)
    x = np.linspace(0.0, 1.0, grid_size + 2, dtype=dtype_type)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    exact = np.asarray(exact_fn(xx, yy), dtype=dtype_type)
    rhs = np.asarray(rhs_fn(xx, yy), dtype=dtype_type)
    phi0 = np.zeros_like(exact)
    phi0[0, :] = exact[0, :]
    phi0[-1, :] = exact[-1, :]
    phi0[:, 0] = exact[:, 0]
    phi0[:, -1] = exact[:, -1]
    h = dtype_type(1.0 / (grid_size + 1))
    return Problem2D(case, grid_size, h, exact, rhs, phi0)


def make_problem(case: str, grid_size: int, dtype=np.float64) -> Problem2D:
    if case not in CASE_FUNCS:
        raise ValueError(f"unknown case {case!r}; expected one of {CASES}")
    exact_fn, rhs_fn = CASE_FUNCS[case]
    return _problem(exact_fn, rhs_fn, case, grid_size, dtype=dtype)
