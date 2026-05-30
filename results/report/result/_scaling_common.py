from __future__ import annotations

"""Shared helpers for the 2D/3D and CUDA scaling report scripts."""

from collections import defaultdict
from typing import Iterable, Mapping


DEFAULT_GRID_SIZES_BY_DIM = {
    2: (127, 255, 511, 1023, 2047, 4095),
    3: (31, 63, 127, 255),
}


def cells_for_grid_size(dim: int, grid_size: int) -> int:
    return int(grid_size) ** int(dim)


def unique_positive_ints(values: Iterable[object], *, name: str) -> tuple[int, ...]:
    items: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if item < 1:
            raise ValueError(f"{name} must be a positive integer")
        if item in seen:
            continue
        seen.add(item)
        items.append(item)
    if not items:
        raise ValueError(f"at least one {name} is required")
    return tuple(items)


def group_rows_by_case(rows: Iterable[Mapping[str, object]]) -> dict[tuple[int, str], list[dict[str, object]]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["dimension"]), str(row["case"]))].append(dict(row))
    return grouped


def group_rows_by_dimension(rows: Iterable[Mapping[str, object]]) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["dimension"])].append(dict(row))
    return grouped


def group_rows_by_dimension_case_grid(
    rows: Iterable[Mapping[str, object]],
) -> dict[tuple[int, str, int], list[dict[str, object]]]:
    grouped: dict[tuple[int, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["dimension"]), str(row["case"]), int(row["grid_size"]))].append(dict(row))
    return grouped
