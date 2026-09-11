# PortLearn

PortLearn is a finance-first research framework for controlled, reproducible, and modular experimentation in machine-learned portfolio choice.

> Change the research component without accidentally changing the financial experiment.

PortLearn is being developed to support comparable portfolio-learning research across classical methods, forecasting-based machine learning, direct deep learning, reinforcement learning, optimization-based approaches, and user-defined research components under common financial experiment contracts.

## Status

PortLearn is in early development and has not yet been published as a stable package release. The current public surface exposes the research foundation and API contracts. The public API is not yet stable.

## Design Direction

PortLearn aims to provide reusable infrastructure for:

- financial data and information-set construction;
- modular feature engineering;
- portfolio strategy composition;
- common portfolio accounting;
- controlled comparison of alternative methods;
- research reproducibility and provenance.

PortLearn is a research toolkit, not a repository for individual paper-specific models or unpublished research architectures.

## Roadmap

PortLearn is under active research development. This roadmap is intentionally high-level: it communicates broad direction only, is subject to change as the research framework develops, and does not promise dates or specific functionality.

**Available now**

- Research foundation: time/chronology contracts, observation handling, information sets, and the core research interfaces that define how portfolio research components compose.
- Validation utilities for detecting violations of point-in-time information contracts.
- Canonical run manifests for recording environment and command provenance.

**Next**

- Datasets and data access: adapters and alignment for market and macro data.
- Experiment and reproducibility infrastructure.

**Planned**

- Forecasting methods.
- Portfolio construction and accounting.
- Later deep-learning and reinforcement-learning research capabilities.

Entries move forward on this roadmap as the underlying research foundation stabilizes; nothing here is a dated commitment.

## Development

The local development battery, from a fresh clone:

```console
uv sync --locked
uv run ruff check .
uv run pytest
uv build
uv run python scripts/verify_built_wheel.py
```
