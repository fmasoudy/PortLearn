"""PortLearn — research infrastructure for portfolio-learning research.

This package ships its public identity only: importing this module is
side-effect-free — it performs no filesystem writes, network access,
configuration changes, logging initialization, data retrieval, or
application computation — and it eagerly imports no other PortLearn
module, so the import surface stays minimal and stable.

``pyproject.toml`` is the single version authority: ``__version__`` is
derived from the installed distribution metadata rather than
hard-coded, so it can never drift from the declared release.
"""

from __future__ import annotations

from importlib import metadata
from typing import Any

__version__ = metadata.version("portlearn")

__all__ = ["__version__", "data", "weights"]


def __getattr__(name: str) -> Any:
    """Lazily import the public module facades (PEP 562)."""
    if name == "data":
        from importlib import import_module

        return import_module("portlearn.data")
    if name == "weights":
        from importlib import import_module

        return import_module("portlearn.weights")
    if name == "rebalance":
        from importlib import import_module

        return import_module("portlearn.rebalance")
    if name == "trades":
        from importlib import import_module

        return import_module("portlearn.trades")
    if name == "turnover":
        from importlib import import_module

        return import_module("portlearn.turnover")
    if name == "costs":
        from importlib import import_module

        return import_module("portlearn.costs")
    if name == "ledger":
        from importlib import import_module

        return import_module("portlearn.ledger")
    if name == "strategies":
        from importlib import import_module

        return import_module("portlearn.strategies")
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}; the lazily "
        "exposed public subpackage is 'data'; the public modules are "
        "'weights', 'rebalance', 'trades', 'turnover', 'costs', "
        "'ledger', and 'strategies'."
    )


def __dir__() -> list[str]:
    return sorted(__all__)
