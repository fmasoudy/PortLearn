# Changelog

All notable public changes to PortLearn are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning follows the project's PEP 440 policy: early-development `0.0.x` public releases.

## [Unreleased]

## [0.0.1.dev4]

### Added

- Portfolio evaluation metrics: the `portlearn.metrics` module with finance-standard point statistics over one recorded wealth path — annualized return, volatility, Sharpe ratio, certainty-equivalent return, and maximum drawdown — with the return basis, annualization frequency, and risk-free reference declared per call.
- Results and provenance schema: the `portlearn.results` module with immutable result records, canonical JSON serialization, forecast and information-set provenance links, conformance records with sanctioned deviations, parameter relations, and the ledger digest helper.
- Experiments: the `portlearn.experiments` module with process-local strategy registration, the serializable experiment configuration with stable identifiers and benchmark metadata, shared experimental conditions, four declarative selection modes, and the runtime that applies one common information supply, universe, and accounting contract to every selected strategy.
- Temporal research protocols: the `portlearn.protocols` module with the declarative temporal protocol layer — canonical protocols, sparse per-strategy override composition, and the ordinary authoring layer — normalizing to one canonical resolution.
- Comparison outputs: the `portlearn.comparisons` module with structured comparison tables and a plain-text renderer over recorded experiment results; recorded values, declarations, verdicts, deviations, and provenance are copied verbatim, never ranked, converted, or summarized.
- The classical research route: the `portlearn.research` entry point composing protocols, experiments, and results into one declared run, rejecting missing or incomplete fit provenance at entry.
- Research examples: `examples/classical_research_smoke.py`, an instructional offline replay over committed sources with a reference run snapshot manifest, and `examples/external_composition_challenge.py`, a synthetic demonstration composing scikit-learn estimation and cvxpy optimization through the public contracts only.
- The optional `external-composition` dependency group (scikit-learn and cvxpy) for the composition example; neither distribution is ever imported by the core package.
- Public test modules covering the metrics, results, experiments, protocols, comparison, and research-entry surfaces, plus the example replay sources.

## [0.0.1.dev3]

### Added

- Strategies: the `portlearn.strategies` module with the four built-in classical strategies — `EqualWeight`, `InverseVolatility`, `MinimumVariance`, and `MeanVariance` — implementing the strategy decision contract.
- SciPy ships as a core runtime dependency: the constrained optimization solvers used by `MinimumVariance` and `MeanVariance` are available from a plain `pip install portlearn`. SciPy is lazily imported inside the internal optimizer adapter, never at package import.

## [0.0.1.dev2]

### Added

- Portfolio weights: the `portlearn.weights` module with the closed portfolio-role vocabulary, target-weight validation, and immutable portfolio weights.
- Rebalancing: schedule policies and the drift law carrying held weights across holding segments (`portlearn.rebalance`).
- Transaction and turnover accounting with proportional transaction-cost models (`portlearn.trades`, `portlearn.turnover`, `portlearn.costs`).
- The transaction ledger: segment-composed accounting over the wealth path with the reference accounting engine (`portlearn.ledger`).
- The strategy decision interface — `Strategy.decide(context) -> DecisionResult` over `DecisionContext`, validated by `require_decision_result_compatible`.
- Lazy module facades for the portfolio modules under the top-level package namespace.
- Public test modules covering the portfolio-weight, rebalance, transaction, cost, ledger, and decision-contract surfaces.

### Changed

### Fixed

### Deprecated

### Removed

## [0.0.1.dev1]

### Added

- Data access: the `portlearn.data` facade with a sealed two-state research-dataset container with pandas conversion, and an optional `parquet` extra for Arrow/parquet interoperability.
- Fama/French and FRED provider adapters (retrieval, decoding, retrieval provenance), introduced under the `portlearn.data.adapters` namespace.
- Availability-aware alignment of mixed-frequency observations, built on a period calendar that maps period-end dates to time-zone-aware instants.
- Chronologically valid feature transforms: lags, rolling statistics, scalers, and carry-forward.
- Forecasting and estimation lifecycle contracts around the `Forecaster` protocol — fitting, refitting, forecast timing, tuning, seeds, determinism, and provenance; contracts only, no estimators.
- Descriptive research-dataset diagnostics — summary, correlation, coverage, and missingness reports — plus renderer-neutral plotting specs, with rendering provided by the optional `plot` extra (matplotlib).
- pandas as the sole unconditional core runtime dependency.

### Changed

### Fixed

### Deprecated

### Removed

## [0.0.1.dev0]

### Added

- Foundation research contracts: time/chronology contracts, observation handling, information sets, core research interfaces, information-leakage validation, and run manifests, with a synthetic contract-wiring example, public test suite, and public CI.

### Changed

### Fixed

### Deprecated

### Removed
