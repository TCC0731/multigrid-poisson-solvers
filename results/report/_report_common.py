from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def positive_int(value: object, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def positive_float(value: object, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if parsed <= 0.0:
        raise ValueError(f"{name} must be a positive number")
    return parsed


def normalize_choices(values: Iterable[object], *, valid: Sequence[object], name: str) -> tuple[object, ...]:
    items = tuple(values)
    if not items:
        raise ValueError(f"at least one {name} is required")
    invalid = [item for item in items if item not in valid]
    if invalid:
        raise ValueError(f"unsupported {name}(s): {invalid!r}")
    return items


def csv_float(value: float) -> str:
    return f"{float(value):.6e}"


def load_pyplot():
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for these reports. "
            "Please run them inside `conda activate mg`."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def finalize_figure_header(
    fig,
    *,
    title: str,
    handles: Sequence[object],
    labels: Sequence[str],
    ncol: int,
    legend_y: float = 0.875,
    title_y: float = 0.985,
    tight_top: float = 0.80,
    legend_kwargs: Mapping[str, object] | None = None,
) -> None:
    """Place a figure legend and title in separate header bands.

    Figure-level legends are not handled by ``tight_layout()``, so callers need
    to reserve explicit top space and keep the legend below the title.
    """

    if handles and labels:
        options: dict[str, object] = {"frameon": False, "borderaxespad": 0.0}
        if legend_kwargs:
            options.update(legend_kwargs)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=ncol,
            **options,
        )
    fig.suptitle(title, y=title_y)
    fig.tight_layout(rect=(0, 0, 1, tight_top))


def power_law_fit_curve(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    num_points: int = 200,
) -> tuple[list[float], list[float], float] | None:
    """Fit y = a * x^b in log-log space and return a smooth curve.

    The helper filters out non-positive points before fitting, since those
    cannot be represented on log axes.
    """

    points = [(float(x), float(y)) for x, y in zip(x_values, y_values) if float(x) > 0.0 and float(y) > 0.0]
    if len(points) < 2:
        return None

    log_x = [math.log10(x) for x, _ in points]
    log_y = [math.log10(y) for _, y in points]
    n = len(points)
    sum_x = sum(log_x)
    sum_y = sum(log_y)
    sum_xx = sum(value * value for value in log_x)
    sum_xy = sum(x * y for x, y in zip(log_x, log_y))
    denom = n * sum_xx - sum_x * sum_x
    slope = 0.0 if abs(denom) < 1e-12 else (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    x_min = min(x for x, _ in points)
    x_max = max(x for x, _ in points)
    if x_min == x_max:
        fit_x = [x_min, x_max]
    else:
        steps = max(2, num_points)
        log_min = math.log10(x_min)
        log_max = math.log10(x_max)
        fit_x = [
            10 ** (log_min + (log_max - log_min) * index / (steps - 1))
            for index in range(steps)
        ]

    fit_y = [10 ** (intercept + slope * math.log10(x)) for x in fit_x]
    return fit_x, fit_y, slope


def write_rows_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    columns: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
