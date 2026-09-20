"""The researcher-facing data facade (progressive disclosure, level 1).

Importing this package is side-effect free: it eagerly imports no
provider facade and no adapter module.  The two provider facades
(:mod:`~portlearn.data.fama_french` and :mod:`~portlearn.data.fred`)
are exposed lazily through PEP 562 attribute access, so the import
surface stays minimal and the import graph stays acyclic — importing
``portlearn.data`` registers this package alone.

Research datasets only; NOT investable.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ResearchDataset", "diagnostics", "fama_french", "fred"]


def __getattr__(name: str) -> Any:
    """Lazily import one public name, or fail with the module error.

    ``ResearchDataset`` — the provider-neutral sealed two-state
    container — imports lazily like the provider facades, preserving the package's import side-effect guarantees: a bare
    ``import portlearn.data`` registers this package alone (the
    dataset module, and with it pandas, loads only on first use).  The
    state types (``UnqualifiedDataset``/``QualifiedDataset``) stay
    internal to :mod:`portlearn.data.dataset`.  The diagnostics
    subpackage (five standalone stdlib-only report callables plus
    ``PlotSpec``) is itself stdlib-only, so its lazy import registers
    no dataset module and no pandas either.
    """
    if name == "ResearchDataset":
        from portlearn.data.dataset import ResearchDataset

        return ResearchDataset
    if name == "diagnostics":
        from importlib import import_module

        return import_module("portlearn.data.diagnostics")
    if name in ("fama_french", "fred"):
        from importlib import import_module

        return import_module(f"portlearn.data.{name}")
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}; the public data "
        "surface is 'ResearchDataset' plus the provider facades "
        "'fama_french' and 'fred'."
    )


def __dir__() -> list[str]:
    return sorted(__all__)
