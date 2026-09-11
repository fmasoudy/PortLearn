"""PortLearn — research infrastructure for portfolio-learning research.

This package ships its public identity only: importing this module is
side-effect-free — it performs no filesystem writes, network access,
configuration mutation, logging initialization, data retrieval, or
application computation — and it eagerly imports no other PortLearn
module, so the import surface stays minimal and stable.

``pyproject.toml`` is the single version authority: ``__version__`` is
derived from the installed distribution metadata rather than
hard-coded, so it can never drift from the declared release.
"""

from __future__ import annotations

from importlib import metadata

__version__ = metadata.version("portlearn")
