"""The diagnostics package: five standalone report callables and PlotSpec.

Every public callable is a pure, stdlib-only report builder over one
sealed dataset: importing this package registers no dataset module, no
provider facade, and no plotting stack.  The five blocks stay private
and load lazily through PEP 562 attribute access, so the public
surface is exactly ``describe``, ``correlation``, ``coverage``,
``missingness``, ``plot``, and the ``PlotSpec`` type — nothing else.

Research datasets only; NOT investable.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "PlotSpec",
    "correlation",
    "coverage",
    "describe",
    "missingness",
    "plot",
]


def __getattr__(name: str) -> Any:
    """Lazily import one public name, or fail with the module error."""
    if name == "describe":
        from portlearn.data.diagnostics._describe import describe

        return describe
    if name == "correlation":
        from portlearn.data.diagnostics._correlation import correlation

        return correlation
    if name == "coverage":
        from portlearn.data.diagnostics._coverage import coverage

        return coverage
    if name == "missingness":
        from portlearn.data.diagnostics._missingness import missingness

        return missingness
    if name == "plot":
        from portlearn.data.diagnostics._plot import plot

        return plot
    if name == "PlotSpec":
        from portlearn.data.diagnostics._plot import PlotSpec

        return PlotSpec
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}; the public "
        "diagnostics surface is the five report callables 'describe', "
        "'correlation', 'coverage', 'missingness', and 'plot', plus the "
        "'PlotSpec' type."
    )


def __dir__() -> list[str]:
    return sorted(__all__)
