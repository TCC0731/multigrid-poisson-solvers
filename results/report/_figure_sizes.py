from __future__ import annotations

"""Shared figure-size constants for the report plots.

The report scripts import these helpers so that figure geometry can be changed
from one place without hunting through every plotting file.
"""

from typing import Final

FIGSIZE_1X3_MG_2D_3D_SCALING: Final[tuple[float, float]] = (12.0, 5.0)
FIGSIZE_1X3_MG_CONVERGENCE: Final[tuple[float, float]] = (12.0, 5.0)
FIGSIZE_2X2_2_ROWS: Final[tuple[float, float]] = (10.0, 6.0)
FIGSIZE_2X3_2_ROWS: Final[tuple[float, float]] = (12.0, 6.0)

# These two 1x2 figures keep their current proportions, but the size is now
# centralized so they can be tuned from one place later if needed.
FIGSIZE_1X2_CUDA_MG_CONVERGENCE: Final[tuple[float, float]] = (10.0, 5)
FIGSIZE_1X2_CUDA_MG_SOR_TIME_FIT: Final[tuple[float, float]] = (10.0, 5)


def figsize_for_rows(
    base_figsize: tuple[float, float],
    nrows: int,
    *,
    reference_rows: int = 2,
) -> tuple[float, float]:
    """Scale the height of a 2-row reference figure to ``nrows`` rows."""

    if nrows < 1:
        raise ValueError("nrows must be at least 1")
    if reference_rows < 1:
        raise ValueError("reference_rows must be at least 1")

    width, height = base_figsize
    scale = nrows / reference_rows
    return width, height * scale


def figsize_for_columns(
    base_figsize: tuple[float, float],
    ncols: int,
    *,
    reference_columns: int = 2,
) -> tuple[float, float]:
    """Scale the width of a 2-column reference figure to ``ncols`` columns."""

    if ncols < 1:
        raise ValueError("ncols must be at least 1")
    if reference_columns < 1:
        raise ValueError("reference_columns must be at least 1")

    width, height = base_figsize
    scale = ncols / reference_columns
    return width * scale, height
