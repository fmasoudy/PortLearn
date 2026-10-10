"""Results contract tests: the universal result and provenance schema (§8 floors).

The surface under test is the ``portlearn.results`` public module: the
eight frozen value objects (``ResultRecord``, ``PathResult``,
``MetricsBlock``, ``ForecastLink``, ``InformationSetAnchor``,
``ConformanceRecord``, ``SanctionedDeviation``, ``ParameterRelation``),
the module-owned ``ResultStructureError``, the two serialization entry
points (``ResultRecord.to_json`` / ``ResultRecord.from_json``), and the
one public digest helper ``ledger_digest`` — exactly as frozen by the
Specification approval (specification §§4–8 at accepted
candidate ``23fa6b0``; §§4–§8 byte-identical at the
implementation baseline ``c5fe968``).

Floors pinned here (22 behavior floors, results §8): the module surface
and lazy facade (T1); the nine-field frozen record with its
fail-closed string/version/naive-instant/path-name laws (T2/T2a); the
declaration-complete metric block (T3); the digest-plus-structural-
summary ledger link with canonical metric ordering (T4); the
ten-field forecast link, the empty-or-canonical forecast sequence, and
the information anchor (T5); the computed conformance verdict over
structured sanctions (T6); the recursive settings grammar with deep
snapshots, signed-zero normalization, and insertion-order blindness
(T6a); the canonical sorted/compact/ASCII fail-closed serialization
(T7); content-equality → byte-identity (T8); the derived manifest link
with no embedding and no stray seed (T9); the never-rows wire and the
engine-internals fence (T10); the strictly-positive ``str(timedelta)``
horizon grammar (T11); offset-preserving aware instants (T12); the
stdlib-only import fence (T13); serialization determinism (T14); the
error-surface ownership law (T15); exact-key/duplicate-key/version/
nonfinite/naive/noncanonical parse rejection and round-trip (T16);
verdict tamper-evidence (T17); the γ/λ co-occurrence relation law with
the effective-only ``risk_aversion`` lookup (T18); immutability,
canonical equality, and deliberate unhashability (T19); and the
pinned §5.6a ledger-digest payload law — sensitivity, insertion-order
blindness, order sensitivity, no wrapper (T20).

Per §13a every ledger, summary, manifest, provenance, and information
fixture is rebuilt through the frozen public constructors
(``build_ledger``, ``core_metrics``, ``RunManifest``,
``fitting_provenance``, ``InformationSet``) — never hand-inlined
literals without provenance.  Digest and canonical-JSON oracles are
derived independently in this module from the §5.1–§5.6a law text
using the stdlib alone (``json`` + ``hashlib``), never from any
production helper; they read only the frozen seams' public fields.

Constructor-route readings pinned here (the spec names the consumed
objects and pins every stored-field keyword, but does not spell the
consumed-object parameter names): the consumed ``RunManifest``,
``LedgerPath``, and ``InformationSet`` are passed positionally first
to ``ResultRecord`` / ``PathResult`` / ``InformationSetAnchor``; the
``MetricsSummary`` is passed positionally first to ``MetricsBlock``
with the four declaration keywords mirroring ``core_metrics``
keyword-for-keyword; ``schema_version`` is the ``"1.0.0"`` literal at
construction (§5a) and any other supplied value rejects.

RED-first: the module is absent at this tranche, so every
``portlearn.results`` import happens INSIDE the tests and builders —
pytest reports the causal absence (``ModuleNotFoundError: No module
named 'portlearn.results'``) for every node instead of aborting
collection.  Floor names are behavior names; the §8 floor each node
pins is recorded in its docstring.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import importlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from hashlib import sha256
from typing import Any

import pytest

from portlearn.costs import Proportional
from portlearn.forecasting import (
    DeterminismClass,
    FittingPlan,
    FitWindow,
    fitting_provenance,
)
from portlearn.interfaces import InformationSet, PortfolioDecision
from portlearn.ledger import LedgerPath, LedgerRow, build_ledger
from portlearn.manifest import RunManifest
from portlearn.metrics import MetricsSummary, core_metrics
from portlearn.observations import TimedObservation
from portlearn.timing import NaiveTimestampError
from portlearn.turnover import two_sided

# --------------------------------------------------------------------------- #
# The module under test — imported inside every node so collection succeeds   #
# while the module is absent (the causal RED signal).                         #
# --------------------------------------------------------------------------- #


@functools.cache
def _results() -> Any:
    """Import ``portlearn.results`` on demand (RED: ModuleNotFoundError)."""
    return importlib.import_module("portlearn.results")


# --------------------------------------------------------------------------- #
# Generation isolation (harness plumbing — no assertion or floor changes).    #
# --------------------------------------------------------------------------- #
#
# ``tests/test_package_contract.py`` runs a registry-purge probe
# (``test_import_triggers_no_eager_portlearn_submodules``) that deletes
# every ``portlearn``/``portlearn.*`` entry from ``sys.modules`` and
# re-imports the root package fresh, leaving that new generation in the
# registry.  When this module is collected before that probe runs, the
# names bound at import time above (``Proportional``, ``LedgerPath``,
# ``MetricsSummary``, ``NaiveTimestampError``, ...) belong to the
# collection-time generation, while ``_results()`` — imported lazily at
# test time — binds classes from a post-purge generation.  Nominal
# identity (``type(...) is NaiveTimestampError``), enum membership
# (``DeterminismClass.SEED_REPRODUCIBLE``), ``isinstance`` admission of
# consumed fixtures, and dataclass equality then compare objects from
# two different module generations and fail purely on import order.
#
# The cure is local isolation plumbing: capture the collection-time
# generation at import, pin it back into ``sys.modules`` for the duration
# of this module's nodes — so ``_results()``, every fixture builder, and
# the consumed objects share one coherent generation — and restore the
# ambient registry exactly afterwards.  No behavioral assertion, floor,
# production validation, or foreign test changes: with one coherent
# generation pinned the original assertions run unchanged, and without
# the pin the same assertions fail on generation skew — this fixture
# cannot manufacture a pass.


def _portlearn_generation() -> dict[str, Any]:
    """Snapshot the registered ``portlearn`` root/submodule generation."""
    return {
        name: module
        for name, module in sys.modules.items()
        if name == "portlearn" or name.startswith("portlearn.")
    }


_COLLECTION_GENERATION = _portlearn_generation()


@pytest.fixture(autouse=True, scope="module")
def _one_coherent_module_generation():
    """Run this module's nodes against one coherent module generation."""
    ambient = _portlearn_generation()
    sys.modules.update(_COLLECTION_GENERATION)
    try:
        yield
    finally:
        for name in sorted(_portlearn_generation()):
            del sys.modules[name]
        sys.modules.update(ambient)


# --------------------------------------------------------------------------- #
# Independent canonical-JSON and digest oracles (stdlib only; §5.1–§5.6a).    #
# --------------------------------------------------------------------------- #


def _canonical_json(value: Any) -> str:
    """The §5.1–§5.4 byte law: sorted keys, compact, ASCII, no nonfinite."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _weight_map(weights: Any) -> dict[str, float]:
    """A §5.6a weight map: exact asset keys, floats, signed zero unified."""
    return {asset: float(weights[asset]) + 0.0 for asset in weights}


def _trade_image(trade: Any) -> dict[str, Any]:
    """The §5.6a trade projection: exactly source/destination/delta."""
    return {
        "source": {
            "state": trade.source.state.value,
            "weights": _weight_map(trade.source.weights),
        },
        "destination": {
            "state": trade.destination.state.value,
            "weights": _weight_map(trade.destination.weights),
        },
        "delta": _weight_map(trade.delta),
    }


def _execution_image(execution: Any) -> dict[str, Any]:
    """The §5.6a execution projection: exactly the eight Execution keys."""
    return {
        "decision_time": execution.decision_time.isoformat(),
        "execution_time": execution.execution_time.isoformat(),
        "target_weights": _weight_map(execution.target_weights),
        "pre_trade_weights": _weight_map(execution.pre_trade_weights),
        "trade": _trade_image(execution.trade),
        "turnover": float(execution.turnover) + 0.0,
        "transaction_cost": float(execution.transaction_cost) + 0.0,
        "post_trade_weights": _weight_map(execution.post_trade_weights),
    }


def _row_image(row: LedgerRow) -> dict[str, Any]:
    """The §5.6a row projection: exactly the twelve LedgerRow keys."""
    return {
        "period_start": row.period_start.isoformat(),
        "period_end": row.period_end.isoformat(),
        "opening_weights": _weight_map(row.opening_weights),
        "executions": [_execution_image(e) for e in row.executions],
        "closing_weights": _weight_map(row.closing_weights),
        "gross_return": float(row.gross_return) + 0.0,
        "net_return": float(row.net_return) + 0.0,
        "turnover": float(row.turnover) + 0.0,
        "transaction_cost": float(row.transaction_cost) + 0.0,
        "wealth_open": float(row.wealth_open) + 0.0,
        "wealth_close": float(row.wealth_close) + 0.0,
        "universe": sorted(row.universe),
    }


def _row_payload(path: LedgerPath) -> list[dict[str, Any]]:
    """The §5.6a payload: the bare row-object array, no wrapper."""
    return [_row_image(row) for row in path.rows]


def _oracle_ledger_digest(path: LedgerPath) -> str:
    """Independent §5.6a oracle: sha256 over the bare canonical row array."""
    payload = _canonical_json(_row_payload(path))
    return sha256(payload.encode("utf-8")).hexdigest()


def _oracle_membership_digest(information_set: InformationSet) -> str:
    """Independent D7 oracle: sha256 over the sorted identity-triple array."""
    triples = [
        _canonical_json(
            [
                record.series_id,
                record.observation_time.isoformat(),
                record.available_time.isoformat(),
            ]
        )
        for record in information_set
    ]
    payload = "[" + ",".join(sorted(triples)) + "]"
    return sha256(payload.encode("utf-8")).hexdigest()


def _oracle_manifest_digest(manifest: RunManifest) -> str:
    """Independent §5.3 oracle: sha256 over the manifest's own to_json."""
    return sha256(manifest.to_json().encode("utf-8")).hexdigest()


def _schema_keys(
    node: Any, skip: tuple[str, ...] = ("declared", "effective")
) -> set[str]:
    """Every object key on the wire, optionally skipping settings subtrees."""
    keys: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            keys.add(key)
            if key not in skip:
                keys |= _schema_keys(value, skip)
    elif isinstance(node, list):
        for value in node:
            keys |= _schema_keys(value, skip)
    return keys


def _iter_objects(node: Any):
    """Yield every JSON object on the wire, in document order."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_objects(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_objects(value)


# --------------------------------------------------------------------------- #
# Frozen-seam fixtures — rebuilt through the public constructors (§13a).      #
# --------------------------------------------------------------------------- #

_MONTHS = tuple(datetime(2026, m, 1, tzinfo=UTC) for m in range(1, 6))
_TZ_PLUS_TEN = timezone(timedelta(hours=10))
_CREATED_AT = datetime(2026, 9, 27, 9, 30, tzinfo=_TZ_PLUS_TEN)
_REBALANCE_FEB = PortfolioDecision(
    decision_time=_MONTHS[1],
    execution_time=_MONTHS[1],
    target_weights={"A": 0.5, "B": 0.5},
)


@functools.cache
def _manifest() -> RunManifest:
    """The consumed run identity (§13a: the frozen public constructor)."""
    return RunManifest(
        run_id="m42-red-run",
        created_at=datetime(2026, 9, 27, 8, 15, tzinfo=_TZ_PLUS_TEN),
        python_version="3.13.5",
        package_version="0.0.1.dev3",
        dependency_pins={"scipy": "1.14.0"},
        commands=["portlearn-run m42"],
    )


@functools.cache
def _path_two_assets() -> LedgerPath:
    """A two-asset, four-period path with one boundary rebalance.

    F1-shaped (§14 probe): initial 40/60 book, one execution at the
    period-1 boundary, 10bp two-sided cost, ``W_0 = 100``.
    """
    return build_ledger(
        initial_weights={"A": 0.4, "B": 0.6},
        accounting_instants=_MONTHS,
        growth_factors={
            _MONTHS[0]: {"A": 0.995, "B": 1.01},
            _MONTHS[1]: {"A": 0.995, "B": 1.011},
            _MONTHS[2]: {"A": 0.995, "B": 1.012},
            _MONTHS[3]: {"A": 0.995, "B": 1.013},
        },
        decisions=[_REBALANCE_FEB],
        cost_model=Proportional(rate=0.001, turnover=two_sided),
        initial_wealth=100.0,
    )


@functools.cache
def _path_single_asset() -> LedgerPath:
    """A single-asset, three-period path with no executions, ``W_0 = 1``."""
    return build_ledger(
        initial_weights={"S": 1.0},
        accounting_instants=_MONTHS[:4],
        growth_factors={
            _MONTHS[0]: {"S": 1.10},
            _MONTHS[1]: {"S": 0.50},
            _MONTHS[2]: {"S": 1.20},
        },
        decisions=[],
        cost_model=Proportional(rate=0.001, turnover=two_sided),
        initial_wealth=1.0,
    )


@functools.cache
def _path_two_executions_one_period() -> LedgerPath:
    """A path whose first period retains TWO executions (T20 order law).

    Intra-month decide-and-execute pairs at Jan 10 and Jan 20 both land
    inside the first accounting period, so one row carries a two-element
    ``executions`` tuple in execution order.
    """
    first = datetime(2026, 1, 10, tzinfo=UTC)
    second = datetime(2026, 1, 20, tzinfo=UTC)
    return build_ledger(
        initial_weights={"A": 0.5, "B": 0.5},
        accounting_instants=(_MONTHS[0], _MONTHS[1], _MONTHS[2]),
        growth_factors={
            _MONTHS[0]: {"A": 1.001, "B": 1.002},
            first: {"A": 1.001, "B": 1.001},
            second: {"A": 1.0, "B": 1.004},
            _MONTHS[1]: {"A": 0.999, "B": 1.003},
        },
        decisions=[
            PortfolioDecision(
                decision_time=first,
                execution_time=first,
                target_weights={"A": 0.25, "B": 0.75},
            ),
            PortfolioDecision(
                decision_time=second,
                execution_time=second,
                target_weights={"A": 0.6, "B": 0.4},
            ),
        ],
        cost_model=Proportional(rate=0.001, turnover=two_sided),
        initial_wealth=1.0,
    )


@functools.cache
def _summary_two_assets_net() -> MetricsSummary:
    return core_metrics(
        _path_two_assets(), periods_per_year=12, gamma=2.0, risk_free=0.001
    )


@functools.cache
def _summary_two_assets_gross() -> MetricsSummary:
    return core_metrics(
        _path_two_assets(),
        periods_per_year=12,
        gamma=2.0,
        risk_free=0.001,
        basis="gross",
    )


@functools.cache
def _summary_single_asset_net() -> MetricsSummary:
    return core_metrics(
        _path_single_asset(), periods_per_year=12, gamma=5.0, risk_free=0.002
    )


@functools.cache
def _fit_set_primary() -> InformationSet:
    """Three admitted records at an April cutoff (§13a constructor)."""
    return InformationSet(
        [
            TimedObservation(
                series_id="px.A",
                observation_time=datetime(2026, 3, 31, tzinfo=UTC),
                available_time=datetime(2026, 4, 1, tzinfo=UTC),
                value=101.25,
            ),
            TimedObservation(
                series_id="px.B",
                observation_time=datetime(2026, 3, 31, tzinfo=UTC),
                available_time=datetime(2026, 4, 1, tzinfo=UTC),
                value=42.0,
            ),
            TimedObservation(
                series_id="rate.cash",
                observation_time=datetime(2026, 3, 31, tzinfo=UTC),
                available_time=datetime(2026, 4, 2, tzinfo=UTC),
                value=0.0011,
            ),
        ],
        as_of=datetime(2026, 4, 30, tzinfo=UTC),
    )


@functools.cache
def _fit_set_secondary() -> InformationSet:
    """Two admitted records at a May cutoff, permuted submission order."""
    return InformationSet(
        [
            TimedObservation(
                series_id="rate.cash",
                observation_time=datetime(2026, 4, 30, tzinfo=UTC),
                available_time=datetime(2026, 5, 1, tzinfo=UTC),
                value=0.0012,
            ),
            TimedObservation(
                series_id="px.A",
                observation_time=datetime(2026, 4, 30, tzinfo=UTC),
                available_time=datetime(2026, 5, 1, tzinfo=UTC),
                value=103.5,
            ),
        ],
        as_of=datetime(2026, 5, 31, tzinfo=UTC),
    )


@functools.cache
def _mlp_provenance() -> Any:
    """One fit's provenance through ``fitting_provenance`` (§13a)."""
    plan = FittingPlan(
        model="mlp",
        config={"hidden": 8, "layers": (2, 3)},
        seed=7,
        fit_window=FitWindow(window_start=datetime(2026, 1, 1, tzinfo=UTC)),
        determinism=DeterminismClass.SEED_REPRODUCIBLE,
    )
    return fitting_provenance(plan, _fit_set_primary())


@functools.cache
def _ridge_provenance() -> Any:
    """A second, distinct fit's provenance through the same constructor."""
    plan = FittingPlan(
        model="ridge",
        config={"alpha": 0.5},
        seed=None,
        fit_window=FitWindow(window_start=datetime(2026, 1, 1, tzinfo=UTC)),
        determinism=DeterminismClass.DETERMINISTIC,
    )
    return fitting_provenance(plan, _fit_set_secondary())


# --------------------------------------------------------------------------- #
# Results-dependent builders — every one imports the module on demand.        #
# --------------------------------------------------------------------------- #


def _block_net_two_assets():
    results = _results()
    return results.MetricsBlock(
        _summary_two_assets_net(),
        basis="net",
        periods_per_year=12,
        gamma=2.0,
        risk_free=0.001,
    )


def _block_gross_two_assets():
    results = _results()
    return results.MetricsBlock(
        _summary_two_assets_gross(),
        basis="gross",
        periods_per_year=12,
        gamma=2.0,
        risk_free=0.001,
    )


def _block_net_single_asset():
    results = _results()
    return results.MetricsBlock(
        _summary_single_asset_net(),
        basis="net",
        periods_per_year=12,
        gamma=5.0,
        risk_free=(0.002, 0.002, 0.002),
    )


def _path_result_two_assets(blocks=None):
    results = _results()
    metrics = (
        blocks
        if blocks is not None
        else (
            _block_net_two_assets(),
            _block_gross_two_assets(),
        )
    )
    return results.PathResult(_path_two_assets(), tuple(metrics))


def _path_result_single_asset(blocks=None):
    results = _results()
    metrics = blocks if blocks is not None else (_block_net_single_asset(),)
    return results.PathResult(_path_single_asset(), tuple(metrics))


def _anchor(information_set: InformationSet):
    return _results().InformationSetAnchor(information_set)


def _link_mlp():
    results = _results()
    provenance = _mlp_provenance()
    return results.ForecastLink(
        model=provenance.model,
        config_hash=provenance.config_hash,
        cutoff=provenance.cutoff,
        seed=provenance.seed,
        determinism=provenance.determinism,
        fit_window_start=provenance.fit_window.window_start,
        refit_instants=(
            datetime(2026, 5, 31, tzinfo=UTC),
            datetime(2026, 6, 30, tzinfo=UTC),
        ),
        forecast_target="expected_return",
        forecast_horizon=timedelta(days=1, hours=2, minutes=30),
        information=(_anchor(_fit_set_primary()), _anchor(_fit_set_secondary())),
    )


def _link_ridge():
    results = _results()
    provenance = _ridge_provenance()
    return results.ForecastLink(
        model=provenance.model,
        config_hash=provenance.config_hash,
        cutoff=provenance.cutoff,
        seed=provenance.seed,
        determinism=provenance.determinism,
        fit_window_start=provenance.fit_window.window_start,
        refit_instants=(),
        forecast_target="volatility_forecast",
        forecast_horizon=timedelta(hours=6),
        information=(),
    )


def _consistent_conformance():
    results = _results()
    declared = {
        "cost_rate": 0.001,
        "max_weight": 0.5,
        "universe": ("A", "B"),
        "label": "base",
        "note": None,
        "nested": {"depth": 2, "flags": (True, False)},
    }
    effective = {
        "label": "base",
        "nested": {"flags": (True, False), "depth": 2},
        "universe": ("A", "B"),
        "max_weight": 0.5,
        "cost_rate": 0.001,
        "note": None,
    }
    return results.ConformanceRecord(
        declared=declared, effective=effective, deviations=()
    )


def _sanction(field: str):
    results = _results()
    return results.SanctionedDeviation(
        field=field,
        authority="portfolio owner",
        reference="approval-log",
        reason="raised for the cost-sensitivity check",
        approved_at=datetime(2026, 9, 27, 7, 0, tzinfo=_TZ_PLUS_TEN),
    )


def _sanctioned_conformance():
    results = _results()
    return results.ConformanceRecord(
        declared={"cost_rate": 0.001, "max_weight": 0.5},
        effective={"max_weight": 0.25, "cost_rate": 0.002},
        deviations=(_sanction("cost_rate"), _sanction("max_weight")),
    )


def _default_paths():
    return {
        "ew": _path_result_two_assets(),
        "iv": _path_result_single_asset(),
    }


NAIVE_TS = datetime(2026, 9, 27, 9, 30)  # noqa: DTZ001 -- deliberate naive probe


def _naive_ts():
    """A deliberately naive timestamp for the aware-instant laws."""
    return NAIVE_TS


def _record(
    *,
    paths=None,
    forecasts=None,
    conformance=None,
    parameter_relations=(),
    created_at=_CREATED_AT,
    strategy_id="equal-weight",
    manifest=None,
):
    results = _results()
    return results.ResultRecord(
        manifest if manifest is not None else _manifest(),
        strategy_id=strategy_id,
        paths=paths if paths is not None else _default_paths(),
        forecasts=forecasts if forecasts is not None else (_link_mlp(), _link_ridge()),
        conformance=conformance
        if conformance is not None
        else _consistent_conformance(),
        parameter_relations=parameter_relations,
        created_at=created_at,
    )


def _relation(gamma: float, risk_aversion: float):
    results = _results()
    return results.ParameterRelation(
        role_a="evaluation_gamma",
        role_b="decision_risk_aversion",
        value_a=gamma,
        value_b=risk_aversion,
    )


def _relations_record():
    """A record whose effective settings declare the decision risk aversion."""
    results = _results()
    conformance = results.ConformanceRecord(
        declared={"risk_aversion": 2.0},
        effective={"risk_aversion": 2.0},
        deviations=(),
    )
    # §5.10/T18: the default paths carry gamma 2.0 AND 5.0, so an effective
    # risk_aversion of 2.0 requires a covering relation for both pairings.
    return _record(
        conformance=conformance,
        parameter_relations=(_relation(2.0, 2.0), _relation(5.0, 2.0)),
    )


# --------------------------------------------------------------------------- #
# T1 — module surface and lazy facade                                         #
# --------------------------------------------------------------------------- #


def test_results_module_imports_and_exports_exactly_the_pinned_surface() -> None:
    """Floor T1 — the module exports exactly the ten declared names."""
    module = _results()
    assert module.__all__ == [
        "ConformanceRecord",
        "ForecastLink",
        "InformationSetAnchor",
        "MetricsBlock",
        "ParameterRelation",
        "PathResult",
        "ResultRecord",
        "ResultStructureError",
        "SanctionedDeviation",
        "ledger_digest",
    ]
    for name in module.__all__:
        assert getattr(module, name) is not None, f"{name} does not resolve"


def test_root_facade_lazily_exposes_results_and_directory_lists_it() -> None:
    """Floor T1 — the lazy facade resolves the module and ``__dir__`` lists it."""
    direct = _results()
    import portlearn as pl
    from portlearn import results as via_facade

    assert pl.results is direct
    assert via_facade is direct
    assert "results" in dir(pl)
    # __all__ untouched (D1: module facades are lazy attributes).
    assert pl.__all__ == ["__version__", "data", "weights"]


def test_root_facade_error_message_names_results_among_public_modules() -> None:
    """Floor T1 — the unknown-attribute error enumerates ``results``."""
    _results()
    import portlearn as pl

    with pytest.raises(AttributeError) as excinfo:
        _discard = pl.Definitely_not_a_portlearn_surface  # noqa: B950 -- deliberate miss
    assert "results" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# T2 / T2a — the nine-field frozen record and its admission laws               #
# --------------------------------------------------------------------------- #


def test_result_record_builds_from_a_consumed_manifest_with_nine_frozen_fields() -> (
    None
):
    """Floor T2 — exactly nine fields, frozen, no unknown kwargs."""
    record = _record()
    names = [field.name for field in dataclasses.fields(record)]
    assert sorted(names) == [
        "conformance",
        "created_at",
        "forecasts",
        "manifest_digest",
        "parameter_relations",
        "paths",
        "run_id",
        "schema_version",
        "strategy_id",
    ]
    assert record.schema_version == "1.0.0"
    with pytest.raises(AttributeError):
        record.strategy_id = "other"
    with pytest.raises(TypeError):
        _record(manifest=_manifest(), strategy_id="equal-weight")  # sanity build
        _results().ResultRecord(
            _manifest(),
            strategy_id="equal-weight",
            paths=_default_paths(),
            forecasts=(),
            conformance=_consistent_conformance(),
            parameter_relations=(),
            created_at=_CREATED_AT,
            surprise=1,
        )


def test_record_rejects_blank_strategy_id_and_nonliteral_schema_version() -> None:
    """Floor T2a — blank/non-string ids and a wrong version token reject."""
    results = _results()
    for bad in ("", "   ", 7, None):
        with pytest.raises(results.ResultStructureError):
            _record(strategy_id=bad)
    with pytest.raises(results.ResultStructureError):
        _results().ResultRecord(
            _manifest(),
            strategy_id="equal-weight",
            paths=_default_paths(),
            forecasts=(),
            conformance=_consistent_conformance(),
            parameter_relations=(),
            created_at=_CREATED_AT,
            schema_version="2.0.0",
        )


def test_record_rejects_naive_created_at_with_the_timing_module_error() -> None:
    """Floor T2a/T12/T15 — naive instants reject with timing's own error."""
    _results()
    with pytest.raises(NaiveTimestampError) as excinfo:
        _record(created_at=_naive_ts())
    assert type(excinfo.value) is NaiveTimestampError


def test_record_rejects_empty_paths_and_blank_or_duplicate_path_names() -> None:
    """Floor T2a — no zero-path record; blank or duplicate names reject."""
    _results()
    with pytest.raises(ValueError):
        _record(paths={})
    with pytest.raises(ValueError):
        _record(paths={"   ": _path_result_single_asset()})
    # A Python mapping cannot carry duplicate keys; the spec pins duplicate
    # name rejection on the wire, exercised floor-side at T16 via duplicate
    # JSON object keys and here via the collapse-to-one guard on names.


# --------------------------------------------------------------------------- #
# T9 — the derived manifest link                                              #
# --------------------------------------------------------------------------- #


def test_record_derives_run_identity_from_the_consumed_manifest_only() -> None:
    """Floor T9 — run_id and manifest_digest are derived, never accepted."""
    results = _results()
    record = _record()
    assert record.run_id == _manifest().run_id
    assert record.manifest_digest == _oracle_manifest_digest(_manifest())
    with pytest.raises(results.ResultStructureError):
        _results().ResultRecord(
            _manifest(),
            strategy_id="equal-weight",
            paths=_default_paths(),
            forecasts=(),
            conformance=_consistent_conformance(),
            parameter_relations=(),
            created_at=_CREATED_AT,
            run_id="smuggled",
        )
    with pytest.raises(results.ResultStructureError):
        _results().ResultRecord(
            _manifest(),
            strategy_id="equal-weight",
            paths=_default_paths(),
            forecasts=(),
            conformance=_consistent_conformance(),
            parameter_relations=(),
            created_at=_CREATED_AT,
            manifest_digest="0" * 64,
        )


def test_wire_run_link_is_the_exact_flat_pair_with_no_manifest_embedding() -> None:
    """Floor T9 — the run link serializes as the flat digest/run_id pair."""
    _results()
    parsed = json.loads(_record().to_json())
    assert parsed["run_id"] == _manifest().run_id
    assert parsed["manifest_digest"] == _oracle_manifest_digest(_manifest())
    text = _record().to_json()
    for embedded in (
        "portlearn-run m42",
        "1.14.0",
        "2026-09-27T08:15:00+10:00",
        "python_version",
        "package_version",
        "dependency_pins",
        "commands",
    ):
        assert embedded not in text, f"manifest content embedded: {embedded!r}"


def test_wire_seed_appears_only_inside_forecast_links() -> None:
    """Floor T9 — no seed token outside the forecast link's seed field."""
    _results()
    parsed = json.loads(_record().to_json())
    seeded = [obj for obj in _iter_objects(parsed) if "seed" in obj]
    assert seeded, "the fixture record carries forecast links with seeds"
    for obj in seeded:
        assert "forecast_horizon" in obj, "a seed appears outside a forecast link"


# --------------------------------------------------------------------------- #
# T3 — the declaration-complete metric block                                  #
# --------------------------------------------------------------------------- #


def test_metrics_block_carries_the_six_values_and_four_declarations_exactly() -> None:
    """Floor T3 — the block is the summary plus its producing declarations."""
    block = _block_net_two_assets()
    names = [field.name for field in dataclasses.fields(block)]
    assert sorted(names) == [
        "annualized_mean_return",
        "annualized_volatility",
        "basis",
        "certainty_equivalent_return",
        "cumulative_return",
        "gamma",
        "max_drawdown",
        "periods_per_year",
        "risk_free",
        "sharpe_ratio",
    ]
    summary = _summary_two_assets_net()
    for value_name in (
        "cumulative_return",
        "annualized_mean_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "certainty_equivalent_return",
    ):
        assert getattr(block, value_name) == getattr(summary, value_name)
    assert block.basis == "net"
    assert block.periods_per_year == 12
    assert block.gamma == 2.0
    assert block.risk_free == 0.001
    series_block = _block_net_single_asset()
    assert series_block.risk_free == (0.002, 0.002, 0.002)


def test_metrics_block_rejects_unknown_kwargs_and_unlawful_declarations() -> None:
    """Floor T3 — unknown kwargs TypeError; the metrics declaration laws carried."""
    _results()
    with pytest.raises(TypeError):
        _results().MetricsBlock(
            _summary_two_assets_net(),
            basis="net",
            periods_per_year=12,
            gamma=2.0,
            risk_free=0.001,
            surprise=1,
        )
    for gamma in (0.0, -2.0):
        with pytest.raises(ValueError):
            _results().MetricsBlock(
                _summary_two_assets_net(),
                basis="net",
                periods_per_year=12,
                gamma=gamma,
                risk_free=0.001,
            )
    with pytest.raises(ValueError):
        _results().MetricsBlock(
            _summary_two_assets_net(),
            basis="net",
            periods_per_year=0,
            gamma=2.0,
            risk_free=0.001,
        )
    with pytest.raises(ValueError):
        _results().MetricsBlock(
            _summary_two_assets_net(),
            basis="total",
            periods_per_year=12,
            gamma=2.0,
            risk_free=0.001,
        )


# --------------------------------------------------------------------------- #
# T4 — the digest-plus-summary ledger link                                    #
# --------------------------------------------------------------------------- #


def test_path_result_derives_the_ledger_digest_and_structural_summary() -> None:
    """Floor T4 — digest and summary derive from the consumed LedgerPath."""
    results = _results()
    path = _path_two_assets()
    path_result = _path_result_two_assets()
    assert path_result.ledger_digest == _oracle_ledger_digest(path)
    assert path_result.ledger_digest == results.ledger_digest(path)
    assert path_result.n_periods == path.n_periods == len(path.rows)
    assert path_result.period_start == path.rows[0].period_start
    assert path_result.period_end == path.rows[-1].period_end
    assert path_result.final_wealth == path.final_wealth
    names = [field.name for field in dataclasses.fields(path_result)]
    assert sorted(names) == [
        "final_wealth",
        "ledger_digest",
        "metrics",
        "n_periods",
        "period_end",
        "period_start",
    ]
    for supplied in (
        {"ledger_digest": "0" * 64},
        {"n_periods": 4},
        {"period_start": _MONTHS[0]},
        {"period_end": _MONTHS[4]},
        {"final_wealth": 100.0},
    ):
        with pytest.raises(results.ResultStructureError):
            _results().PathResult(path, (_block_net_two_assets(),), **supplied)


def test_path_result_rejects_degenerate_paths_and_empty_metric_sequences() -> None:
    """Floor T4 — empty path, naive bounds, nonfinite wealth, empty metrics."""
    results = _results()
    block = _block_net_two_assets()
    empty_path = LedgerPath(rows=(), final_wealth=1.0, n_periods=0)
    with pytest.raises(ValueError):
        results.PathResult(empty_path, (block,))
    naive_row = dataclasses.replace(
        _path_two_assets().rows[0], period_start=_naive_ts()
    )
    naive_path = LedgerPath(
        rows=(naive_row, *_path_two_assets().rows[1:]),
        final_wealth=100.0,
        n_periods=4,
    )
    with pytest.raises(NaiveTimestampError):
        results.PathResult(naive_path, (block,))
    nonfinite_path = LedgerPath(
        rows=_path_two_assets().rows,
        final_wealth=float("inf"),
        n_periods=4,
    )
    with pytest.raises(results.ResultStructureError):
        results.PathResult(nonfinite_path, (block,))
    with pytest.raises(ValueError):
        results.PathResult(_path_two_assets(), ())


def test_path_metrics_sequence_is_canonical_and_blind_to_submission_order() -> None:
    """Floor T4/§5.13 — metric blocks store canonically; order cannot alter bytes."""
    _results()
    first = _path_result_two_assets(
        (_block_net_two_assets(), _block_gross_two_assets())
    )
    second = _path_result_two_assets(
        (_block_gross_two_assets(), _block_net_two_assets())
    )
    assert tuple(first.metrics) == tuple(second.metrics)
    with pytest.raises(ValueError):
        _path_result_two_assets((_block_net_two_assets(), _block_net_two_assets()))


def test_ledger_digest_is_blind_to_weight_map_insertion_order() -> None:
    """Floor T4/T20b — content-equal paths hash identically (§5.6a)."""
    results = _results()
    rebalance = PortfolioDecision(
        decision_time=_MONTHS[1],
        execution_time=_MONTHS[1],
        target_weights={"B": 0.5, "A": 0.5},
    )
    permuted = build_ledger(
        initial_weights={"B": 0.6, "A": 0.4},
        accounting_instants=_MONTHS,
        growth_factors={
            _MONTHS[0]: {"B": 1.01, "A": 0.995},
            _MONTHS[1]: {"B": 1.011, "A": 0.995},
            _MONTHS[2]: {"B": 1.012, "A": 0.995},
            _MONTHS[3]: {"B": 1.013, "A": 0.995},
        },
        decisions=[rebalance],
        cost_model=Proportional(rate=0.001, turnover=two_sided),
        initial_wealth=100.0,
    )
    base = _path_two_assets()
    assert results.ledger_digest(permuted) == results.ledger_digest(base)
    assert results.ledger_digest(permuted) == _oracle_ledger_digest(base)


# --------------------------------------------------------------------------- #
# T5 — the forecast link, the anchor, and the forecast sequence                #
# --------------------------------------------------------------------------- #


def test_forecast_link_carries_the_ten_field_provenance_image() -> None:
    """Floor T5 — the link is the schema-owned image of one fit's provenance."""
    _results()
    link = _link_mlp()
    names = [field.name for field in dataclasses.fields(link)]
    assert sorted(names) == [
        "config_hash",
        "cutoff",
        "determinism",
        "fit_window_start",
        "forecast_horizon",
        "forecast_target",
        "information",
        "model",
        "refit_instants",
        "seed",
    ]
    provenance = _mlp_provenance()
    assert link.model == provenance.model == "mlp"
    assert link.config_hash == provenance.config_hash
    assert link.cutoff == _fit_set_primary().as_of
    assert link.seed == 7
    assert link.determinism is DeterminismClass.SEED_REPRODUCIBLE
    assert link.fit_window_start == provenance.fit_window.window_start
    assert tuple(link.refit_instants) == (
        datetime(2026, 5, 31, tzinfo=UTC),
        datetime(2026, 6, 30, tzinfo=UTC),
    )
    assert link.forecast_target == "expected_return"
    assert link.forecast_horizon == timedelta(days=1, hours=2, minutes=30)
    assert len(tuple(link.information)) == 2
    empty_refits = _link_ridge()
    assert tuple(empty_refits.refit_instants) == ()
    assert tuple(empty_refits.information) == ()
    assert empty_refits.seed is None


def test_forecast_link_rejects_degenerate_horizons_and_blank_strings() -> None:
    """Floor T5/T11 — zero/negative horizons, naive cutoff, blank names."""
    results = _results()
    provenance = _ridge_provenance()
    common = {
        "model": provenance.model,
        "config_hash": provenance.config_hash,
        "cutoff": provenance.cutoff,
        "seed": None,
        "determinism": provenance.determinism,
        "fit_window_start": provenance.fit_window.window_start,
        "refit_instants": (),
        "information": (),
    }
    with pytest.raises(results.ResultStructureError):
        results.ForecastLink(
            forecast_horizon=timedelta(0), forecast_target="t", **common
        )
    with pytest.raises(ValueError):
        results.ForecastLink(
            forecast_horizon=timedelta(days=-2, hours=2),
            forecast_target="t",
            **common,
        )
    with pytest.raises(NaiveTimestampError):
        results.ForecastLink(
            forecast_horizon=timedelta(hours=6),
            forecast_target="t",
            cutoff=_naive_ts(),
            **{k: v for k, v in common.items() if k != "cutoff"},
        )
    for blank in ("", "   "):
        with pytest.raises(results.ResultStructureError):
            results.ForecastLink(
                forecast_horizon=timedelta(hours=6),
                forecast_target=blank,
                **common,
            )
        with pytest.raises(results.ResultStructureError):
            results.ForecastLink(
                forecast_horizon=timedelta(hours=6),
                forecast_target="expected_return",
                model=blank,
                **{k: v for k, v in common.items() if k != "model"},
            )


def test_information_anchor_derives_as_of_and_membership_digest() -> None:
    """Floor T5/D7 — both anchor fields derive from the consumed set."""
    results = _results()
    information_set = _fit_set_primary()
    anchor = _anchor(information_set)
    names = [field.name for field in dataclasses.fields(anchor)]
    assert sorted(names) == ["as_of", "membership_digest"]
    assert anchor.as_of == information_set.as_of
    assert anchor.membership_digest == _oracle_membership_digest(information_set)
    permuted_set = InformationSet(
        list(_fit_set_primary())[::-1], as_of=information_set.as_of
    )
    assert _anchor(permuted_set).membership_digest == anchor.membership_digest
    assert anchor.membership_digest != _oracle_membership_digest(_fit_set_secondary())
    with pytest.raises(results.ResultStructureError):
        results.InformationSetAnchor(information_set, membership_digest="0" * 64)
    with pytest.raises(results.ResultStructureError):
        results.InformationSetAnchor(information_set, as_of=information_set.as_of)


def test_forecasts_may_be_empty_reject_duplicates_and_store_canonical_order() -> None:
    """Floor T5/§5.11 — empty-or-canonical sequence; duplicates reject."""
    _results()
    empty = _record(forecasts=())
    assert tuple(empty.forecasts) == ()
    assert json.loads(empty.to_json())["forecasts"] == []
    with pytest.raises(ValueError):
        _record(forecasts=(_link_mlp(), _link_mlp()))
    one_order = _record(forecasts=(_link_mlp(), _link_ridge()))
    other_order = _record(forecasts=(_link_ridge(), _link_mlp()))
    assert tuple(one_order.forecasts) == tuple(other_order.forecasts)
    assert one_order.to_json() == other_order.to_json()


# --------------------------------------------------------------------------- #
# T6 — the computed conformance verdict                                       #
# --------------------------------------------------------------------------- #


def test_conformance_verdict_computes_consistent_from_canonical_equality() -> None:
    """Floor T6 — canonical-equal settings with no deviations are consistent."""
    _results()
    conformance = _consistent_conformance()
    assert conformance.verdict == "consistent"
    assert tuple(conformance.deviations) == ()
    names = [field.name for field in dataclasses.fields(conformance)]
    assert sorted(names) == ["declared", "deviations", "effective", "verdict"]
    with pytest.raises(TypeError):
        _results().ConformanceRecord(
            declared={"cost_rate": 0.001},
            effective={"cost_rate": 0.001},
            deviations=(),
            verdict="consistent",
        )


def test_conformance_verdict_computes_sanctioned_deviation_from_full_coverage() -> None:
    """Floor T6 — every difference covered by a complete sanction sanctions."""
    _results()
    conformance = _sanctioned_conformance()
    assert conformance.verdict == "sanctioned_deviation"
    assert sorted(d.field for d in conformance.deviations) == [
        "cost_rate",
        "max_weight",
    ]


def test_conformance_rejects_an_uncovered_difference() -> None:
    """Floor T6 — a difference with no sanctioning coverage rejects."""
    _results()
    with pytest.raises(ValueError):
        _results().ConformanceRecord(
            declared={"cost_rate": 0.001},
            effective={"cost_rate": 0.002},
            deviations=(),
        )


def test_conformance_rejects_incomplete_sanctions_and_naive_approval() -> None:
    """Floor T6 — blank sanction fields and naive approval instants reject."""
    results = _results()
    for incomplete in (
        {"field": "cost_rate", "authority": "", "reference": "r", "reason": "ok"},
        {"field": "cost_rate", "authority": "a", "reference": "  ", "reason": "ok"},
        {"field": "cost_rate", "authority": "a", "reference": "r", "reason": ""},
    ):
        with pytest.raises(ValueError):
            results.ConformanceRecord(
                declared={"cost_rate": 0.001},
                effective={"cost_rate": 0.002},
                deviations=(
                    results.SanctionedDeviation(
                        approved_at=datetime(2026, 9, 27, 7, 0, tzinfo=_TZ_PLUS_TEN),
                        **incomplete,
                    ),
                ),
            )
    with pytest.raises(NaiveTimestampError):
        results.ConformanceRecord(
            declared={"cost_rate": 0.001},
            effective={"cost_rate": 0.002},
            deviations=(
                results.SanctionedDeviation(
                    field="cost_rate",
                    authority="portfolio owner",
                    reference="approval-log",
                    reason="raised for the cost-sensitivity check",
                    approved_at=_naive_ts(),
                ),
            ),
        )


def test_conformance_rejects_phantom_and_duplicate_sanctions() -> None:
    """Floor T6 — sanctions must name differing fields, exactly one each."""
    results = _results()
    phantom = results.SanctionedDeviation(
        field="label",
        authority="portfolio owner",
        reference="approval-log",
        reason="no difference to sanction",
        approved_at=datetime(2026, 9, 27, 7, 0, tzinfo=_TZ_PLUS_TEN),
    )
    with pytest.raises(ValueError):
        results.ConformanceRecord(
            declared={"cost_rate": 0.001, "label": "base"},
            effective={"cost_rate": 0.002, "label": "base"},
            deviations=(_sanction("cost_rate"), phantom),
        )
    with pytest.raises(ValueError):
        results.ConformanceRecord(
            declared={"cost_rate": 0.001},
            effective={"cost_rate": 0.002},
            deviations=(_sanction("cost_rate"), _sanction("cost_rate")),
        )


# --------------------------------------------------------------------------- #
# T6a — the recursive settings grammar                                        #
# --------------------------------------------------------------------------- #


class _FakeMode(enum.Enum):
    TRAINING = "training"


def test_settings_grammar_round_trips_every_lawful_value_kind() -> None:
    """Floor T6a — each lawful kind survives the record round trip."""
    results = _results()
    declared = {
        "none_value": None,
        "bool_value": True,
        "int_value": 5,
        "float_value": 0.25,
        "str_value": "monthly",
        "tuple_value": (1, "a", (2.5, False)),
        "map_value": {"inner": {"deep": False, "count": 2}},
    }
    conformance = results.ConformanceRecord(
        declared=declared, effective=dict(declared), deviations=()
    )
    record = _record(conformance=conformance)
    restored = results.ResultRecord.from_json(record.to_json())
    assert restored == record
    effective = restored.conformance.effective
    assert effective["none_value"] is None
    assert effective["bool_value"] is True
    assert not isinstance(effective["int_value"], bool)
    assert effective["int_value"] == 5
    assert effective["float_value"] == 0.25
    assert effective["str_value"] == "monthly"
    assert effective["tuple_value"] == (1, "a", (2.5, False))
    assert effective["map_value"] == {"inner": {"deep": False, "count": 2}}


def test_settings_grammar_rejects_every_unlawful_value_kind() -> None:
    """Floor T6a — bytes, enums, datetimes, sets, nonfinite, bad keys reject."""
    results = _results()
    unlawful_values = [
        b"bytes",
        _FakeMode.TRAINING,
        datetime(2026, 9, 27, tzinfo=UTC),
        {"a", "b"},
        float("inf"),
        float("nan"),
        object(),
    ]
    for bad in unlawful_values:
        with pytest.raises(ValueError):
            results.ConformanceRecord(
                declared={"setting": bad}, effective={"setting": bad}, deviations=()
            )
    for bad_mappings in (
        {3: "value"},
        {" ": 1},
        {"outer": {" ": 1}},
        {"outer": {7: 1}},
        {"outer": ({"nested_bad": b"bytes"},)},
    ):
        with pytest.raises(ValueError):
            results.ConformanceRecord(
                declared=bad_mappings, effective=bad_mappings, deviations=()
            )


def test_settings_snapshots_are_deeply_immutable_and_blind_to_insertion_order() -> None:
    """Floor T6a — deep snapshots; caller mutation cannot reach the record."""
    results = _results()
    declared = {"cost_rate": 0.001, "nested": {"layers": [1, 2]}}
    conformance = results.ConformanceRecord(
        declared=declared, effective=dict(declared), deviations=()
    )
    before = _record(conformance=conformance).to_json()
    declared["cost_rate"] = 9.0
    declared["nested"]["layers"].append(3)
    after = _record(conformance=conformance).to_json()
    assert before == after
    assert conformance.declared["cost_rate"] == 0.001
    assert conformance.declared["nested"]["layers"] == (1, 2)
    with pytest.raises(TypeError):
        conformance.declared["cost_rate"] = 9.0
    reordered = results.ConformanceRecord(
        declared={"nested": {"layers": (1, 2)}, "cost_rate": 0.001},
        effective={"cost_rate": 0.001, "nested": {"layers": (1, 2)}},
        deviations=(),
    )
    assert reordered.verdict == "consistent"
    assert _record(conformance=reordered).to_json() == before


def test_signed_zero_is_one_settings_value_and_cannot_be_sanctioned() -> None:
    """Floor T6a — −0.0 and +0.0 are one value; zero sign is no difference."""
    results = _results()
    zero_conformance = results.ConformanceRecord(
        declared={"floor_weight": -0.0},
        effective={"floor_weight": 0.0},
        deviations=(),
    )
    assert zero_conformance.verdict == "consistent"
    negative_record = _record(conformance=zero_conformance)
    positive_record = _record(
        conformance=results.ConformanceRecord(
            declared={"floor_weight": 0.0},
            effective={"floor_weight": 0.0},
            deviations=(),
        )
    )
    assert negative_record.to_json() == positive_record.to_json()
    with pytest.raises(ValueError):
        results.ConformanceRecord(
            declared={"floor_weight": -0.0},
            effective={"floor_weight": 0.0},
            deviations=(
                results.SanctionedDeviation(
                    field="floor_weight",
                    authority="portfolio owner",
                    reference="approval-log",
                    reason="zero sign is not a settings difference",
                    approved_at=datetime(2026, 9, 27, 7, 0, tzinfo=_TZ_PLUS_TEN),
                ),
            ),
        )


# --------------------------------------------------------------------------- #
# T7 / T8 / T14 — canonical form, byte identity, determinism                   #
# --------------------------------------------------------------------------- #


def test_to_json_emits_the_canonical_sorted_compact_ascii_form() -> None:
    """Floor T7 — sorted keys, compact separators, ASCII, no nonfinite tokens."""
    _results()
    text = _record().to_json()
    parsed = json.loads(text)
    assert _canonical_json(parsed) == text
    assert text.isascii()
    assert "NaN" not in text
    assert "Infinity" not in text


def test_nonfinite_metric_content_fails_closed_at_serialization() -> None:
    """Floor T7/§5.4 — a nonfinite metric value rejects, never serializes."""
    results = _results()
    nonfinite_summary = MetricsSummary(
        cumulative_return=0.01,
        annualized_mean_return=0.02,
        annualized_volatility=0.03,
        sharpe_ratio=float("inf"),
        max_drawdown=-0.1,
        certainty_equivalent_return=0.04,
    )
    with pytest.raises(results.ResultStructureError):
        block = results.MetricsBlock(
            nonfinite_summary,
            basis="net",
            periods_per_year=12,
            gamma=2.0,
            risk_free=0.001,
        )
        path_result = results.PathResult(_path_two_assets(), (block,))
        _record(paths={"ew": path_result}).to_json()


def test_content_equal_records_serialize_to_identical_bytes() -> None:
    """Floor T8 — insertion order and forecast submission order are invisible."""
    _results()
    record = _record()
    parsed = json.loads(record.to_json())
    assert _canonical_json(parsed) == record.to_json()
    reordered_forecasts = _record(forecasts=(_link_ridge(), _link_mlp()))
    assert reordered_forecasts.to_json() == record.to_json()


def test_serialization_is_deterministic_with_no_injected_timestamps() -> None:
    """Floor T14 — the same content serializes identically, every time."""
    _results()
    record = _record()
    assert record.to_json() == record.to_json()
    assert _record().to_json() == _record().to_json()
    assert _record().created_at == _CREATED_AT


# --------------------------------------------------------------------------- #
# T10 — the never-rows wire and the engine-internals fence                     #
# --------------------------------------------------------------------------- #

_FORBIDDEN_WIRE_TOKENS = frozenset(
    {
        "executions",
        "opening_weights",
        "closing_weights",
        "universe",
        "wealth_open",
        "wealth_close",
        "target_weights",
        "pre_trade_weights",
        "post_trade_weights",
        "trade",
        "delta",
        "gross_return",
        "net_return",
        "turnover",
        "transaction_cost",
        "segment_growth",
        "realized_over",
    }
)


def test_wire_never_carries_ledger_rows_or_engine_internals() -> None:
    """Floor T10 — no row/execution/weights/universe/engine token on the wire."""
    results = _results()
    record = _record()
    parsed = json.loads(record.to_json())
    keys = _schema_keys(parsed)
    assert keys.isdisjoint(_FORBIDDEN_WIRE_TOKENS), sorted(
        keys & _FORBIDDEN_WIRE_TOKENS
    )
    assert parsed["paths"]["ew"]["ledger_digest"] == results.ledger_digest(
        _path_two_assets()
    )
    assert parsed["paths"]["iv"]["ledger_digest"] == results.ledger_digest(
        _path_single_asset()
    )


# --------------------------------------------------------------------------- #
# T11 / T12 — horizon grammar and instant offsets                              #
# --------------------------------------------------------------------------- #


def test_horizon_wire_form_is_the_documented_timedelta_grammar() -> None:
    """Floor T11 — str(timedelta) wire forms; never an ISO-8601 duration."""
    _results()
    parsed = json.loads(_record().to_json())
    horizons = [link["forecast_horizon"] for link in parsed["forecasts"]]
    assert sorted(horizons) == ["1 day, 2:30:00", "6:00:00"]
    for horizon in horizons:
        assert not horizon.startswith("P")
    restored = _results().ResultRecord.from_json(_record().to_json())
    assert {link.forecast_horizon for link in restored.forecasts} == {
        timedelta(days=1, hours=2, minutes=30),
        timedelta(hours=6),
    }


def test_aware_instant_offsets_survive_serialization_byte_exactly() -> None:
    """Floor T12 — the original UTC offset round-trips byte-exactly."""
    _results()
    record = _record()
    text = record.to_json()
    assert "2026-09-27T09:30:00+10:00" in text
    restored = _results().ResultRecord.from_json(text)
    assert restored.created_at == _CREATED_AT
    assert restored.created_at.utcoffset() == timedelta(hours=10)
    assert restored.forecasts[0].cutoff.utcoffset() == timedelta(0)


# --------------------------------------------------------------------------- #
# T13 — the stdlib-only import fence                                          #
# --------------------------------------------------------------------------- #

_FENCE_PROGRAM = """
import sys

baseline = frozenset(sys.modules)
import portlearn.results

delta = {name.split(".")[0] for name in set(sys.modules) - baseline}
allowed = set(sys.stdlib_module_names) | {"portlearn"}
assert delta <= allowed, f"non-stdlib import delta: {sorted(delta - allowed)}"
print("RESULTS_STDLIB_FENCE_OK")
"""


def test_importing_results_pulls_no_third_party_module() -> None:
    """Floor T13 — a clean subprocess census of the import delta."""
    _results()
    completed = subprocess.run(
        [sys.executable, "-c", _FENCE_PROGRAM],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        "the stdlib-fence subprocess failed:\n"
        f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
    )
    assert "RESULTS_STDLIB_FENCE_OK" in completed.stdout


# --------------------------------------------------------------------------- #
# T15 — error-surface ownership                                               #
# --------------------------------------------------------------------------- #


def test_module_error_surface_ownership() -> None:
    """Floor T15 — ResultStructureError is a ValueError; naive stays timing's."""
    results = _results()
    assert issubclass(results.ResultStructureError, ValueError)
    # results owns the structural error arm outright (audit finding F3): the
    # class is a module-owned subclass of ValueError, never the stdlib
    # alias itself — ``issubclass`` alone cannot separate the two.
    assert results.ResultStructureError is not ValueError
    for naive_builder in (
        lambda: _record(created_at=_naive_ts()),
        lambda: _results().ConformanceRecord(
            declared={"cost_rate": 0.001},
            effective={"cost_rate": 0.002},
            deviations=(
                _results().SanctionedDeviation(
                    field="cost_rate",
                    authority="portfolio owner",
                    reference="approval-log",
                    reason="raised for the cost-sensitivity check",
                    approved_at=_naive_ts(),
                ),
            ),
        ),
    ):
        with pytest.raises(NaiveTimestampError) as excinfo:
            naive_builder()
        assert type(excinfo.value) is NaiveTimestampError


# --------------------------------------------------------------------------- #
# T16 — round trip and the fail-closed parser                                 #
# --------------------------------------------------------------------------- #


def _fixture_records():
    return {
        "base": _record(),
        "no-forecasts": _record(forecasts=()),
        "sanctioned": _record(conformance=_sanctioned_conformance()),
        "relations-covered": _relations_record(),
    }


def test_round_trip_restores_every_fixture_record() -> None:
    """Floor T16 — from_json(to_json(r)) == r for every fixture record."""
    results = _results()
    for label, record in _fixture_records().items():
        restored = results.ResultRecord.from_json(record.to_json())
        assert restored == record, label


def test_parse_rejects_extra_and_missing_keys_at_every_object() -> None:
    """Floor T16 — exact key-set equality, extra and missing keys alike."""
    results = _results()
    text = _record().to_json()
    parsed = json.loads(text)
    parsed["surplus"] = 1
    with pytest.raises(results.ResultStructureError):
        results.ResultRecord.from_json(_canonical_json(parsed))
    parsed = json.loads(text)
    del parsed["conformance"]
    with pytest.raises(results.ResultStructureError):
        results.ResultRecord.from_json(_canonical_json(parsed))
    parsed = json.loads(text)
    parsed["paths"]["ew"]["surplus"] = 1
    with pytest.raises(results.ResultStructureError):
        results.ResultRecord.from_json(_canonical_json(parsed))
    parsed = json.loads(text)
    del parsed["paths"]["ew"]["final_wealth"]
    with pytest.raises(results.ResultStructureError):
        results.ResultRecord.from_json(_canonical_json(parsed))
    parsed = json.loads(text)
    parsed["paths"]["ew"]["metrics"][0]["surplus"] = 1
    with pytest.raises(results.ResultStructureError):
        results.ResultRecord.from_json(_canonical_json(parsed))


def test_parse_rejects_duplicate_keys_at_every_nesting_depth() -> None:
    """Floor T16 — duplicate JSON object keys reject, top level to depth."""
    results = _results()
    text = _record().to_json()
    top_level = text.replace(
        '"schema_version":"1.0.0"',
        '"schema_version":"1.0.0","schema_version":"1.0.0"',
        1,
    )
    assert top_level != text
    with pytest.raises(results.ResultStructureError):
        results.ResultRecord.from_json(top_level)
    in_block = text.replace('"basis":"net"', '"basis":"net","basis":"net"', 1)
    assert in_block != text
    with pytest.raises(results.ResultStructureError):
        results.ResultRecord.from_json(in_block)
    in_settings = text.replace(
        '"cost_rate":0.001', '"cost_rate":0.001,"cost_rate":0.001', 1
    )
    assert in_settings != text
    with pytest.raises(results.ResultStructureError):
        results.ResultRecord.from_json(in_settings)


def test_parse_rejects_unknown_version_nonfinite_literals_and_naive_instants() -> None:
    """Floor T16 — unknown schema_version, NaN literals, naive instants."""
    results = _results()
    text = _record().to_json()
    parsed = json.loads(text)
    parsed["schema_version"] = "0.9.0"
    with pytest.raises(results.ResultStructureError):
        results.ResultRecord.from_json(_canonical_json(parsed))
    parsed = json.loads(text)
    parsed["paths"]["ew"]["metrics"][0]["sharpe_ratio"] = float("nan")
    # The §5.9 nonfinite law is a text-level law (parse_constant): splice
    # the literal into the wire bytes via a nonfinite-permissive dump —
    # re-serializing through the canonical oracle would reject inside
    # the oracle itself (allow_nan=False), never reaching from_json.
    nan_literal = json.dumps(parsed, separators=(",", ":"))
    with pytest.raises(results.ResultStructureError):
        results.ResultRecord.from_json(nan_literal)
    parsed = json.loads(text)
    parsed["paths"]["iv"]["metrics"][0]["max_drawdown"] = float("-inf")
    negative_infinity_literal = json.dumps(parsed, separators=(",", ":"))
    with pytest.raises(results.ResultStructureError):
        results.ResultRecord.from_json(negative_infinity_literal)
    parsed = json.loads(text)
    parsed["created_at"] = "2026-09-27T09:30:00"
    with pytest.raises(NaiveTimestampError):
        results.ResultRecord.from_json(_canonical_json(parsed))


def test_parse_rejects_noncanonical_sequence_order() -> None:
    """Floor T16/§5.11/§5.13 — the canonical order of every sequence is total."""
    results = _results()
    base = _record()
    text = base.to_json()
    parsed = json.loads(text)
    parsed["forecasts"] = list(reversed(parsed["forecasts"]))
    with pytest.raises(ValueError):
        results.ResultRecord.from_json(_canonical_json(parsed))
    parsed = json.loads(text)
    parsed["paths"]["ew"]["metrics"] = list(reversed(parsed["paths"]["ew"]["metrics"]))
    with pytest.raises(ValueError):
        results.ResultRecord.from_json(_canonical_json(parsed))
    sanctioned = json.loads(_record(conformance=_sanctioned_conformance()).to_json())
    sanctioned["conformance"]["deviations"] = list(
        reversed(sanctioned["conformance"]["deviations"])
    )
    with pytest.raises(ValueError):
        results.ResultRecord.from_json(_canonical_json(sanctioned))


# --------------------------------------------------------------------------- #
# T17 — tamper evidence                                                       #
# --------------------------------------------------------------------------- #


def test_parse_rejects_a_tampered_verdict_or_stored_deviation() -> None:
    """Floor T17 — the verdict is recomputed; stored deviations revalidate."""
    results = _results()
    consistent = json.loads(_record().to_json())
    consistent["conformance"]["verdict"] = "sanctioned_deviation"
    with pytest.raises(results.ResultStructureError):
        results.ResultRecord.from_json(_canonical_json(consistent))
    sanctioned = json.loads(_record(conformance=_sanctioned_conformance()).to_json())
    sanctioned["conformance"]["verdict"] = "consistent"
    with pytest.raises(results.ResultStructureError):
        results.ResultRecord.from_json(_canonical_json(sanctioned))
    blanked = json.loads(_record(conformance=_sanctioned_conformance()).to_json())
    blanked["conformance"]["deviations"][0]["authority"] = ""
    with pytest.raises(ValueError):
        results.ResultRecord.from_json(_canonical_json(blanked))
    redirected = json.loads(_record(conformance=_sanctioned_conformance()).to_json())
    redirected["conformance"]["deviations"][0]["field"] = "label"
    with pytest.raises(ValueError):
        results.ResultRecord.from_json(_canonical_json(redirected))


# --------------------------------------------------------------------------- #
# T18 — the γ/λ co-occurrence relation law                                    #
# --------------------------------------------------------------------------- #


def _relations_conformance(risk_aversion: float):
    results = _results()
    return results.ConformanceRecord(
        declared={"risk_aversion": risk_aversion},
        effective={"risk_aversion": risk_aversion},
        deviations=(),
    )


def test_gamma_lambda_co_occurrence_requires_a_relation_for_every_distinct_pairing() -> (
    None
):
    """Floor T18/§5.10 — every distinct (gamma, effective risk_aversion) pair."""
    results = _results()
    conformance = _relations_conformance(4.0)

    # No effective risk aversion: no co-occurrence, no relation required.
    assert tuple(_record().parameter_relations) == ()

    # Co-occurrence without a covering relation rejects.
    with pytest.raises(ValueError):
        _record(conformance=conformance)

    # The default paths carry gamma 2.0 AND 5.0, so an effective risk_aversion
    # of 4.0 needs a covering relation for BOTH distinct pairings; each
    # missing pairing rejects on its own.
    distinct = _relation(2.0, 4.0)
    wide = _relation(5.0, 4.0)
    assert distinct.relation == "distinct"
    assert wide.relation == "distinct"
    with pytest.raises(ValueError):
        _record(conformance=conformance, parameter_relations=(wide,))  # no (2, 4)
    with pytest.raises(ValueError):
        _record(conformance=conformance, parameter_relations=(distinct,))  # no (5, 4)
    covered = _record(conformance=conformance, parameter_relations=(distinct, wide))
    assert tuple(covered.parameter_relations) == (distinct, wide)

    # A "distinct pairing" is a unique role/value pair, not an unequal numeric
    # value: an effective risk_aversion of 2.0 likewise needs both entries.
    identical_pair = _relations_conformance(2.0)
    identical = _relation(2.0, 2.0)
    matching = _relation(5.0, 2.0)
    assert identical.relation == "identical"
    assert matching.relation == "distinct"
    with pytest.raises(ValueError):
        _record(
            conformance=identical_pair,
            parameter_relations=(identical,),  # missing the (5, 2) pairing
        )
    _record(conformance=identical_pair, parameter_relations=(identical, matching))

    # The token is computed, never an input, and never disagrees with values.
    with pytest.raises(TypeError):
        results.ParameterRelation(
            role_a="evaluation_gamma",
            role_b="decision_risk_aversion",
            value_a=2.0,
            value_b=4.0,
            relation="identical",
        )
    tampered = json.loads(
        _record(
            conformance=conformance,
            parameter_relations=(distinct, wide),
        ).to_json()
    )
    # Flipping index [1] ("distinct" -> "identical") keeps the sequence in
    # canonical order (the relation token leads each element's sorted-key
    # bytes, and "distinct" < "identical"), so the rejection is causally the
    # token disagreeing with its own values, not the §5.13 order law.
    tampered["parameter_relations"][1]["relation"] = "identical"
    with pytest.raises(ValueError):
        results.ResultRecord.from_json(_canonical_json(tampered))


def test_nested_risk_aversion_is_never_the_decision_lookup() -> None:
    """Floor T18/§5.10 — only the top-level effective key is consulted."""
    results = _results()
    nested_only = results.ConformanceRecord(
        declared={"layers": {"risk_aversion": 4.0}},
        effective={"layers": {"risk_aversion": 4.0}},
        deviations=(),
    )
    # The nested declaration does not trigger the co-occurrence law...
    _record(conformance=nested_only)
    hidden = results.ConformanceRecord(
        declared={"layers": {"risk_aversion": 9.0}, "risk_aversion": 4.0},
        effective={"layers": {"risk_aversion": 9.0}, "risk_aversion": 4.0},
        deviations=(),
    )
    # ...and never discharges it: the top-level value still requires coverage.
    with pytest.raises(ValueError):
        _record(conformance=hidden)
    _record(
        conformance=hidden,
        parameter_relations=(_relation(2.0, 4.0), _relation(5.0, 4.0)),
    )


# --------------------------------------------------------------------------- #
# T19 — immutability, equality, and the no-hash law                           #
# --------------------------------------------------------------------------- #


def test_record_sequences_are_immutable_and_the_record_is_deliberately_unhashable() -> (
    None
):
    """Floor T19 — immutable sequences; canonical equality; hash rejects."""
    _results()
    record = _record()
    with pytest.raises(TypeError):
        record.paths["new"] = _path_result_single_asset()
    with pytest.raises(TypeError):
        record.forecasts[0] = _link_ridge()
    # §5.13/D2: parameter_relations is an immutable canonical sequence — item
    # mutation against a nonempty, fully covering sequence must reject.
    relations_record = _relations_record()
    assert len(tuple(relations_record.parameter_relations)) == 2
    with pytest.raises(TypeError):
        relations_record.parameter_relations[0] = _relation(2.0, 2.0)
    with pytest.raises(TypeError):
        record.conformance.deviations[0] = _sanction("cost_rate")
    with pytest.raises(TypeError):
        _path_result_two_assets().metrics[0] = _block_gross_two_assets()
    with pytest.raises(TypeError):
        hash(record)
    assert _record() == _record()
    assert _record() != _record(strategy_id="inverse-volatility")
    sanctioned = _record(conformance=_sanctioned_conformance())
    with pytest.raises(TypeError):
        sanctioned.conformance.deviations[0] = _sanction("max_weight")


# --------------------------------------------------------------------------- #
# T20 — the pinned §5.6a ledger-digest payload law                            #
# --------------------------------------------------------------------------- #


def test_ledger_digest_tracks_every_payload_mutation() -> None:
    """Floor T20a — scalar, key, and instant mutations all change the digest."""
    results = _results()
    base = _path_two_assets()
    base_digest = results.ledger_digest(base)
    assert base_digest == _oracle_ledger_digest(base)
    rows = base.rows

    swapped_open = dataclasses.replace(
        rows[0],
        opening_weights={"A": 0.6, "B": 0.4},
    )
    shifted_gross = dataclasses.replace(
        rows[0], gross_return=rows[0].gross_return + 0.5
    )
    shifted_turnover = dataclasses.replace(rows[1], turnover=rows[1].turnover + 0.25)
    widened_close = dataclasses.replace(
        rows[1],
        closing_weights={**dict(rows[1].closing_weights), "C": 0.0},
    )
    renamed_open = dataclasses.replace(
        rows[0],
        opening_weights={"Z": 0.4, "B": 0.6},
    )
    shifted_start = dataclasses.replace(
        rows[0], period_start=rows[0].period_start - timedelta(minutes=1)
    )
    shifted_execution = dataclasses.replace(
        rows[1],
        executions=(
            dataclasses.replace(rows[1].executions[0], turnover=0.125),
            *rows[1].executions[1:],
        ),
    )
    mutants = {
        "weights-value swap": (swapped_open, 0),
        "gross_return scalar": (shifted_gross, 0),
        "row turnover scalar": (shifted_turnover, 1),
        "weight-map key added": (widened_close, 1),
        "weight-map key renamed": (renamed_open, 0),
        "period_start instant": (shifted_start, 0),
        "execution turnover scalar": (shifted_execution, 1),
    }
    for label, (mutated_row, index) in mutants.items():
        mutated_rows = list(rows)
        mutated_rows[index] = mutated_row
        mutated_path = LedgerPath(
            rows=tuple(mutated_rows),
            final_wealth=base.final_wealth,
            n_periods=base.n_periods,
        )
        expected = _oracle_ledger_digest(mutated_path)
        assert expected != base_digest, f"oracle is insensitive: {label}"
        assert results.ledger_digest(mutated_path) == expected, label
        assert results.ledger_digest(mutated_path) != base_digest, label


def test_ledger_digest_distinguishes_row_and_execution_order() -> None:
    """Floor T20c — row order and execution tuple order are content."""
    results = _results()
    base = _path_two_assets()
    reversed_rows = LedgerPath(
        rows=tuple(reversed(base.rows)),
        final_wealth=base.final_wealth,
        n_periods=base.n_periods,
    )
    assert results.ledger_digest(reversed_rows) == _oracle_ledger_digest(reversed_rows)
    assert results.ledger_digest(reversed_rows) != results.ledger_digest(base)

    multi = _path_two_executions_one_period()
    first_row = multi.rows[0]
    assert len(first_row.executions) == 2
    swapped_executions = dataclasses.replace(
        first_row, executions=tuple(reversed(first_row.executions))
    )
    swapped_path = LedgerPath(
        rows=(swapped_executions, *multi.rows[1:]),
        final_wealth=multi.final_wealth,
        n_periods=multi.n_periods,
    )
    assert (
        results.ledger_digest(swapped_path)
        == _oracle_ledger_digest(swapped_path)
        != results.ledger_digest(multi)
    )


def test_ledger_digest_hashes_the_bare_row_array_with_no_wrapper() -> None:
    """Floor T20d — the payload is the bare row array, never a wrapped form."""
    results = _results()
    base = _path_two_assets()
    payload = _row_payload(base)
    bare = sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    assert results.ledger_digest(base) == bare
    wrapped = {
        "rows": payload,
    }
    wrapped_digest = sha256(_canonical_json(wrapped).encode("utf-8")).hexdigest()
    assert results.ledger_digest(base) != wrapped_digest
    with_summary = {
        "rows": payload,
        "n_periods": base.n_periods,
        "final_wealth": base.final_wealth,
    }
    summary_digest = sha256(_canonical_json(with_summary).encode("utf-8")).hexdigest()
    assert results.ledger_digest(base) != summary_digest
    spliced = sha256(
        _canonical_json([*payload, base.n_periods, base.final_wealth]).encode("utf-8")
    ).hexdigest()
    assert results.ledger_digest(base) != spliced


# --------------------------------------------------------------------------- #
# §11 — the exclusion fence                                                   #
# --------------------------------------------------------------------------- #


def test_module_exposes_no_persistence_comparison_or_archive_surface() -> None:
    """§11 — no persistence, comparison, archive, or multi-run surface."""
    module = _results()
    for name in (
        "save",
        "load",
        "dump",
        "write",
        "read",
        "to_csv",
        "to_frame",
        "to_parquet",
        "read_json",
        "from_csv",
        "compare",
        "comparison",
        "table",
        "archive",
        "ExperimentResults",
    ):
        assert not hasattr(module, name), f"module exposes {name!r}"
        assert not hasattr(module.ResultRecord, name), f"record exposes {name!r}"


def test_one_run_maps_named_paths_to_per_path_declaration_complete_blocks() -> None:
    """Floor D2/D3 (V1) — blocks live on named paths, nowhere else."""
    _results()
    record = _record()
    assert set(record.paths) == {"ew", "iv"}
    ew_blocks = tuple(record.paths["ew"].metrics)
    iv_blocks = tuple(record.paths["iv"].metrics)
    assert len(ew_blocks) == 2
    assert len(iv_blocks) == 1
    assert not hasattr(record, "metrics")
    assert ew_blocks[0].basis == "net"
    assert ew_blocks[1].basis == "gross"
    assert iv_blocks[0].gamma == 5.0
    for block in (*ew_blocks, *iv_blocks):
        for declaration in ("basis", "periods_per_year", "gamma", "risk_free"):
            assert getattr(block, declaration) is not None


# --------------------------------------------------------------------------- #
# results HOLD cure — fresh-process facade probe (F1) and the declared-fallback
# discriminator (F2), per the owner's bounded acceptance-cure instruction.
# --------------------------------------------------------------------------- #


_FACADE_PROBE_PROGRAM = """
import sys

import portlearn

assert "portlearn.results" not in sys.modules, (
    "portlearn.results resolved before the facade was touched"
)
module = portlearn.results
assert getattr(portlearn, "results") is sys.modules["portlearn.results"], (
    "the facade attribute is not the module it resolved"
)
print("RESULTS_FACADE_PROBE_OK")
"""


def test_root_facade_resolves_results_without_prior_submodule_import_fresh_process() -> (
    None
):
    """Floor T1 — a fresh process resolves ``results`` through the facade.

    In-process the lazy ``__getattr__`` branch can already have been
    bypassed by an earlier direct import of ``portlearn.results``, so
    the causal probe runs in a child interpreter: the assert ordering
    below fails if any submodule import preceded the facade touch.
    """
    _results()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    completed = subprocess.run(
        [sys.executable, "-c", _FACADE_PROBE_PROGRAM],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, (
        "the facade-probe subprocess failed:\n"
        f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
    )
    assert "RESULTS_FACADE_PROBE_OK" in completed.stdout
    assert "ImportError" not in completed.stderr
    assert "AttributeError" not in completed.stderr


def test_settings_risk_aversion_effective_only_lookup_no_relation_required() -> None:
    """Floor T18/§5.10 — declared-only risk_aversion never triggers coverage.

    A top-level ``risk_aversion`` present in ``declared`` yet absent from
    ``effective`` is a sanctioned difference, never a decision input: the
    co-occurrence law consults only the top-level effective value, so
    constructing the record needs no ParameterRelation at all.
    """
    results = _results()
    # The top-level key is declared but not effective — the difference is
    # sanctioned, and no gamma pairing exists to cover.
    declared_only = results.ConformanceRecord(
        declared={"risk_aversion": 4.0},
        effective={},
        deviations=(_sanction("risk_aversion"),),
    )
    assert declared_only.verdict == "sanctioned_deviation"
    # The default paths carry gamma 2.0 and 5.0: a declared-fallback
    # lookup (consulting declared before effective) would demand
    # coverage here; the effective-only law requires none.
    bare = _record(conformance=declared_only)
    assert tuple(bare.parameter_relations) == ()
    # A declared/effective split under a nested key proves the fallback
    # never consults a nested declared value either.
    nested_declared = results.ConformanceRecord(
        declared={"layers": {"risk_aversion": 4.0}},
        effective={},
        deviations=(_sanction("layers"),),
    )
    _record(conformance=nested_declared)
