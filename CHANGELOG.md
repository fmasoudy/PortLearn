# Changelog

All notable public changes to PortLearn are documented in this file.

Git history remains the detailed development record. This changelog is the curated user/researcher-facing history: it tracks meaningful public changes and public software releases, with release sections recording releases and the entries within them recording user-visible changes.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning follows the project's PEP 440 policy: early-development `0.0.x` public releases.

## [Unreleased]

## [0.0.1.dev2]

### Added

- Portfolio weights: the `portlearn.weights` module with the closed portfolio-role vocabulary, target-weight validation, and immutable weight books.
- Rebalancing: schedule policies and the drift law carrying held weights across holding segments (`portlearn.rebalance`).
- Transaction and turnover accounting with proportional transaction-cost models (`portlearn.trades`, `portlearn.turnover`, `portlearn.costs`).
- The transaction ledger: segment-composed accounting over the wealth path with the reference accounting engine (`portlearn.ledger`).
- The strategy decision-contract seam — `Strategy.decide(context) -> DecisionResult` over `DecisionContext`, validated by `require_decision_result_compatible`.
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
