"""The private rendering leaf: the plotting stack enters here only.

This module is not part of the public surface.  Its single callable
imports the plotting stack inside the function body at call time, so a
deployment without the optional ``plot`` extra never pays for — and
never imports — it, while every report and every spec stays fully
importable and computable.  Rendering never opens an interactive
window: the callable builds one figure and returns it.

Research datasets only; NOT investable.
"""

from __future__ import annotations

from typing import Any

__all__ = ["render"]


def _as_float(item: Any) -> float:
    if isinstance(item, bool):
        return float(item)
    if isinstance(item, (int, float)):
        return float(item)
    return float("nan")


def render(spec: Any) -> Any:
    """Render one PlotSpec into a figure (no window, no display call)."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ImportError(
            "rendering a PlotSpec requires the optional plot extra; "
            "install portlearn[plot] to render — every report and every "
            "spec stays importable and computable without it"
        ) from error
    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    positions = list(range(len(spec.labels)))
    for run in spec.series:
        axis.plot(
            positions,
            [_as_float(item) for item in run.y],
            marker="o",
            label=run.name,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels([str(label) for label in spec.labels])
    axis.set_title(spec.title)
    if spec.series:
        axis.legend()
    figure.supxlabel(spec.footnote, fontsize=8)
    return figure
