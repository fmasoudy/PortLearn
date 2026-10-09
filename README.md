<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/fmasoudy/PortLearn/main/docs/assets/brand/png/monochrome/portlearn-logo-horizontal-white.png">
    <img src="https://raw.githubusercontent.com/fmasoudy/PortLearn/main/docs/assets/brand/png/primary/portlearn-logo-horizontal.png" alt="PortLearn logo: a rounded navy-and-teal PL monogram with a segmented circular motif, beside the PortLearn wordmark" width="460">
  </picture>
</p>

# PortLearn

PortLearn is a finance-first research framework for controlled, reproducible, and modular experimentation in machine-learned portfolio choice.

> Change the research component without accidentally changing the financial experiment.

PortLearn is being developed to support comparable portfolio-learning research across classical methods, forecasting-based machine learning, direct deep learning, reinforcement learning, optimization-based approaches, and user-defined research components under common financial experiment contracts.

## Status

PortLearn is in early development and has not yet been published as a stable package release. The current public surface exposes the research foundation, the portfolio-construction and accounting contracts, and the evaluation, experiment, and comparison surfaces. The public API is not yet stable.

## Design Direction

PortLearn aims to provide reusable infrastructure for:

- financial data and information-set construction;
- modular feature engineering;
- portfolio strategy composition;
- common portfolio accounting;
- controlled comparison of alternative methods;
- research reproducibility and provenance.

PortLearn is a research toolkit, not a repository for individual paper-specific models or unpublished research architectures.

## Brand Assets

The PortLearn logo and icon set live in [`docs/assets/brand/`](docs/assets/brand/), with usage guidance (which lockup for which context, including PyPI presentation) in [`docs/assets/brand/README.md`](docs/assets/brand/README.md).

## Roadmap

PortLearn is under active research development. This roadmap is intentionally high-level: it communicates broad direction only, is subject to change as the research framework develops, and does not promise dates or specific functionality.

**Available now**

- Research foundation: time/chronology contracts, observation handling, information sets, and the core research interfaces that define how portfolio research components compose.
- Validation utilities for detecting violations of point-in-time information contracts.
- Canonical run manifests for recording environment and command provenance.
- Data access: a researcher-facing data facade (`portlearn.data`) with Fama/French and FRED provider adapters, a sealed research-dataset container with pandas conversion, and availability-aware alignment of mixed-frequency observations on a period calendar.
- Chronologically valid feature transforms: lags, rolling statistics, scalers, and carry-forward.
- Contracts for the forecasting and estimation lifecycle (fitting, refitting, forecast timing, tuning, seeds, determinism, provenance); estimators are not provided yet.
- Descriptive research-dataset diagnostics (summary, correlation, coverage, missingness) with renderer-neutral plotting; rendering is available through the optional `plot` extra.
- Portfolio weights: target-weight validation, portfolio weights, and the closed portfolio-role vocabulary, under the `portlearn.weights` module.
- Rebalancing: schedule policies and the drift law that carries held weights across holding segments (`portlearn.rebalance`).
- Transaction ledger: segment-composed accounting over the wealth path, built on immutable per-period ledger records with retained execution details, and the reference accounting engine (`portlearn.ledger`).
- Trading and cost accounting: cost-aware transaction and turnover accounting with proportional cost models (`portlearn.trades`, `portlearn.turnover`, `portlearn.costs`).
- The strategy decision interface: `Strategy.decide(context) -> DecisionResult` over `DecisionContext` — the decision-time aggregate of forecast, information, holdings, and strategy state — validated by `require_decision_result_compatible`.
- Strategies: the built-in classical strategies — equal weight, inverse volatility, minimum variance, and mean-variance (`portlearn.strategies`).
- Portfolio evaluation metrics: finance-standard point statistics over one recorded wealth path — including annualized return, volatility, Sharpe ratio, certainty-equivalent return, and maximum drawdown — with the return basis, annualization frequency, and risk-free reference declared per call (`portlearn.metrics`).
- Results and provenance: immutable result records with canonical JSON serialization, forecast and information-set provenance links, conformance records with sanctioned deviations, and a ledger digest helper (`portlearn.results`).
- Experiments: declarative experiment configuration — process-local strategy registration, stable identifiers, benchmark metadata, shared experimental conditions — and the runtime that applies one common information supply, universe, and accounting contract to every selected strategy (`portlearn.experiments`).
- Temporal research protocols: declarative estimation, selection, refit, and preprocessing timing, with sparse per-strategy overrides, normalizing to one canonical protocol per run (`portlearn.protocols`).
- Comparison outputs: structured comparison tables and a plain-text renderer over recorded experiment results; recorded values, verdicts, and provenance are shown verbatim, never ranked or summarized (`portlearn.comparisons`).
- The classical research route: a single documented entry point that composes protocols, experiments, metrics, and results into one declared run with fit-provenance checks at entry (`portlearn.research`).
- Research examples: an instructional end-to-end classical research replay over committed sources, and an external composition challenge demonstrating scikit-learn estimation and cvxpy optimization composed through the public contracts under the optional `external-composition` extra.

**Next**

- Forecasting and estimation methods that implement the lifecycle contracts.

**Planned**

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
