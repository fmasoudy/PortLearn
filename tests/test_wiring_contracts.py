"""Synthetic contract-wiring check: composition only.

A deterministic, seeded synthetic walk through the seven protocols —
minimal implementations of ``FeatureTransform``, ``Forecaster``,
``Strategy``, ``RebalancePolicy``, ``CostModel``, ``AccountingEngine``,
``Evaluator`` (pure functions/datas: no I/O, no external data, and no
randomness beyond the fixed recorded seed) — asserting composition facts
only: every ``FeatureTransform.transform`` input set admits every
``Forecaster.forecast`` information set it feeds; forecast/decision
compatibility via ``require_forecast_decision_compatible``;
``forecast.decision_time == information_set.as_of`` (the implementer
obligation, verified here); decision→accounting; accounting→evaluation;
and a manifest instant per run, aware-datetime serialized.  The wiring
check is interface composition evidence, **not** an end-to-end research
smoke test and not evidence that any later engine, alignment, or
experiment is correct.  Weight values are inert identifiers of
composition, not portfolio claims.

The manifest serialization floors are pinned here too: unconditional
identity, naive/date rejection, canonical sorted-key JSON bytes, and the
exact offset-preserving round trip.  Nodes assert error classes and laws
only, never error-message tails.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta, timezone

import pytest

import portlearn
from portlearn.interfaces import (
    AccountingEngine,
    AccountingResult,
    CostModel,
    DecisionContext,
    DecisionResult,
    Evaluator,
    Forecast,
    InformationSet,
    PortfolioDecision,
    RebalancePolicy,
    require_decision_result_compatible,
    require_feature_lineage,
    require_forecast_decision_compatible,
)
from portlearn.observations import TimedObservation
from portlearn.weights import PortfolioWeights, WeightState

# The seed is fixed and recorded in this file only — the minimal public
# RunManifest schema deliberately carries no seed field.
WIRING_SEED = 20260430

# Fixed synthetic demonstration instant (aware, non-UTC offset): every
# manifest instant is an aware-datetime serialization, and the +10:00
# offset exercises the offset-preserving round trip.
WIRED_DEMONSTRATION_INSTANT = datetime(
    2026, 4, 30, 10, 30, tzinfo=timezone(timedelta(hours=10))
)

Q1_OBSERVATION = datetime(2026, 3, 31, 9, 0, tzinfo=UTC)
Q1_FIRST_RELEASE = datetime(2026, 4, 1, 9, 0, tzinfo=UTC)
APRIL_DECISION = datetime(2026, 4, 30, 9, 0, tzinfo=UTC)

# Inert composition fixtures: the pre-trade book and the realized returns
# of the decision's outcome carry no accounting convention.
PRE_TRADE_WEIGHTS = {"AUD_CPI": 0.5, "AUD_EQ": 0.5}
REALIZED_RETURNS = {"AUD_CPI": 0.01, "AUD_EQ": -0.02}
_COST_PER_UNIT_TURNOVER = 0.0005


def _seeded_unit(seed: int, label: str) -> float:
    """A deterministic unit-interval value derived from the seed.

    SHA-256 over ``"{seed}:{label}"`` mapped onto ``[0, 1)`` by exact
    integer arithmetic: no randomness beyond the seed and no interpreter-
    dependent hashing anywhere in the walk.
    """
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _cpi_first_release() -> TimedObservation:
    return TimedObservation(
        series_id="AUD_CPI_QOQ",
        observation_time=Q1_OBSERVATION,
        available_time=Q1_FIRST_RELEASE,
        value=0.6,
    )


def _unemployment_first_release() -> TimedObservation:
    return TimedObservation(
        series_id="AUD_UNEMP_QOQ",
        observation_time=Q1_OBSERVATION,
        available_time=Q1_FIRST_RELEASE,
        value=4.2,
    )


# ---------------------------------------------------------------------------
# Minimal seeded implementations of the seven protocols
# ---------------------------------------------------------------------------


class _QuarterTrendTransform:
    """``FeatureTransform``: derive one trend feature per admitted record.

    Derived features are observations that re-enter every timing law: each
    preserves its input's ``observation_time`` and ``available_time``
    (lineage-monotone with equality) under a fresh ``FEAT_`` identity,
    with a seed-derived inert value.
    """

    def __init__(self, seed: int) -> None:
        self._seed = seed

    def transform(
        self, observations: Iterable[TimedObservation]
    ) -> tuple[TimedObservation, ...]:
        return tuple(
            TimedObservation(
                series_id=f"FEAT_{record.series_id}",
                observation_time=record.observation_time,
                available_time=record.available_time,
                value=_seeded_unit(
                    self._seed,
                    f"feature:{record.series_id}:"
                    f"{record.available_time.isoformat()}",
                ),
            )
            for record in observations
        )


class _SeededForecaster:
    """``Forecaster``: seeded expectations over an admitted set, deciding
    exactly at the set's ``as_of`` (the implementer obligation)."""

    def __init__(self, seed: int) -> None:
        self._seed = seed

    def forecast(self, information_set: InformationSet) -> Forecast:
        return Forecast(
            values={
                record.series_id: _seeded_unit(
                    self._seed,
                    f"forecast:{record.series_id}:"
                    f"{information_set.as_of.isoformat()}",
                )
                for record in information_set
            },
            target="expected_return",
            decision_time=information_set.as_of,
            produced_by="wiring.seeded-forecaster",
        )


class _SeededStrategy:
    """``Strategy``: one seeded target book over the context's declared
    universe, executing one hour after the context's authoritative
    decision instant (the decision seam)."""

    def __init__(self, seed: int) -> None:
        self._seed = seed

    def decide(self, context: DecisionContext) -> DecisionResult:
        return DecisionResult(
            decision=PortfolioDecision(
                decision_time=context.decision_time,
                execution_time=context.decision_time + timedelta(hours=1),
                target_weights={
                    identifier: _seeded_unit(
                        self._seed, f"weight:{identifier}"
                    )
                    for identifier in context.universe
                },
            )
        )


class _CalendarRebalancePolicy:
    """``RebalancePolicy``: rebalance at or after the last rebalance."""

    def should_rebalance(
        self, decision_time: datetime, last_rebalance_time: datetime
    ) -> bool:
        return decision_time >= last_rebalance_time


class _LinearCostModel:
    """``CostModel``: provisional linear turnover cost (no units,
    currency, or turnover semantics implied)."""

    def estimate_trade_cost(
        self,
        pre_trade_weights: Mapping[str, float],
        target_weights: Mapping[str, float],
    ) -> float:
        identifiers = sorted(set(pre_trade_weights) | set(target_weights))
        turnover = sum(
            abs(target_weights.get(i, 0.0) - pre_trade_weights.get(i, 0.0))
            for i in identifiers
        )
        return _COST_PER_UNIT_TURNOVER * turnover


class _RebalanceAccountingEngine:
    """``AccountingEngine``: the deliberately thin accounting — the
    post-trade book is the decision's target book, nothing more."""

    def account(
        self,
        decision: PortfolioDecision,
        pre_trade_weights: Mapping[str, float],
        realized_returns: Mapping[str, float],
    ) -> AccountingResult:
        return AccountingResult(post_trade_weights=decision.target_weights)


class _PositionCountEvaluator:
    """``Evaluator``: one open metric — the position count of the
    post-trade book (open vocabulary; no metric catalogue)."""

    def evaluate(
        self, accounting_result: AccountingResult
    ) -> Mapping[str, float]:
        return {"position_count": float(len(accounting_result.post_trade_weights))}


def _wired_walk() -> tuple[
    InformationSet, InformationSet, Forecast, PortfolioDecision, AccountingResult
]:
    """The full seeded composition: admitted vintages → transform →
    feature information set → forecast → decision → accounting."""
    information_set = InformationSet(
        [_cpi_first_release(), _unemployment_first_release()], APRIL_DECISION
    )
    transform = _QuarterTrendTransform(WIRING_SEED)
    features = transform.transform(tuple(information_set))
    feature_set = InformationSet(features, information_set.as_of)
    forecaster = _SeededForecaster(WIRING_SEED)
    forecast = forecaster.forecast(feature_set)
    context = DecisionContext(
        decision_time=forecast.decision_time,
        information=feature_set,
        universe=tuple(forecast.values),
        current_weights=PortfolioWeights(
            dict(PRE_TRADE_WEIGHTS), WeightState.PRE_TRADE
        ),
        current_weights_as_of=APRIL_DECISION,
        forecast=forecast,
    )
    result = _SeededStrategy(WIRING_SEED).decide(context)
    require_decision_result_compatible(context, result)
    decision = result.decision
    accounting = _RebalanceAccountingEngine().account(
        decision, PRE_TRADE_WEIGHTS, REALIZED_RETURNS
    )
    return information_set, feature_set, forecast, decision, accounting


# ---------------------------------------------------------------------------
# Protocol composition facts
# ---------------------------------------------------------------------------


def test_transform_outputs_admit_as_the_forecaster_information_set() -> None:
    """Composition fact 1 (transform→forecast): every ``transform`` input
    set admits the information set it feeds — the derived features
    satisfy lineage monotonicity and are admissible at the very same
    ``as_of`` as their inputs, so the forecaster consumes exactly what
    the transform produced."""
    information_set = InformationSet(
        [_cpi_first_release(), _unemployment_first_release()], APRIL_DECISION
    )
    features = _QuarterTrendTransform(WIRING_SEED).transform(
        tuple(information_set)
    )
    assert require_feature_lineage(tuple(information_set), features) is None
    feature_set = InformationSet(features, information_set.as_of)
    assert {record.series_id for record in feature_set} == {
        "FEAT_AUD_CPI_QOQ",
        "FEAT_AUD_UNEMP_QOQ",
    }
    assert all(
        record.available_time <= feature_set.as_of for record in feature_set
    )


def test_forecast_feeds_a_compatible_decision() -> None:
    """Composition fact 2 (forecast→decision compatibility): the wired
    strategy's decision passes ``require_forecast_decision_compatible``
    — deciding on information requires the information to exist first."""
    _, _, forecast, decision, _ = _wired_walk()
    assert (
        require_forecast_decision_compatible(forecast, decision) is None
    )
    assert decision.execution_time >= decision.decision_time


def test_forecast_origin_equals_information_set_as_of() -> None:
    """Composition fact 3 (``decision_time == as_of``): the wired
    forecaster discharges the implementer obligation — the
    forecast's ``decision_time`` is exactly the ``as_of`` of the
    information set it consumed."""
    information_set, feature_set, forecast, _, _ = _wired_walk()
    assert forecast.decision_time == feature_set.as_of
    assert forecast.decision_time == information_set.as_of
    assert set(forecast.values) == {
        "FEAT_AUD_CPI_QOQ",
        "FEAT_AUD_UNEMP_QOQ",
    }


def test_decision_accounts_through_policy_cost_and_engine() -> None:
    """Composition fact 4 (decision→accounting): the rebalance policy
    admits the decision instant, the cost model prices the move as a
    bare float, and the accounting engine produces an identifiable
    ``AccountingResult`` whose post-trade book is the target book."""
    _, _, _forecast, decision, _ = _wired_walk()
    policy = _CalendarRebalancePolicy()
    assert isinstance(policy, RebalancePolicy)
    assert policy.should_rebalance(decision.decision_time, APRIL_DECISION)
    cost_model = _LinearCostModel()
    assert isinstance(cost_model, CostModel)
    estimated_cost = cost_model.estimate_trade_cost(
        PRE_TRADE_WEIGHTS, decision.target_weights
    )
    assert isinstance(estimated_cost, float)
    assert estimated_cost >= 0.0
    engine = _RebalanceAccountingEngine()
    assert isinstance(engine, AccountingEngine)
    result = engine.account(decision, PRE_TRADE_WEIGHTS, REALIZED_RETURNS)
    assert isinstance(result, AccountingResult)
    assert result.post_trade_weights == decision.target_weights


def test_accounting_result_feeds_the_evaluator() -> None:
    """Composition fact 5 (accounting→evaluation): the evaluator consumes
    the accounting result and yields an explicitly open metric mapping —
    composition, no metric semantics."""
    _, _, _, _, accounting = _wired_walk()
    evaluator = _PositionCountEvaluator()
    assert isinstance(evaluator, Evaluator)
    metrics = evaluator.evaluate(accounting)
    assert isinstance(metrics, Mapping)
    assert metrics == {"position_count": 2.0}


def test_each_wired_run_carries_a_manifest_instant() -> None:
    """Composition fact 6 (manifest instant per run): the wired run
    records one ``RunManifest`` at the fixed demonstration instant —
    aware-datetime serialized with its numeric offset, identical on
    serialization, and reconstructable — and the walk is deterministic
    from the seed alone."""
    from portlearn.manifest import RunManifest

    _, _, forecast, decision, _ = _wired_walk()
    replayed = _wired_walk()
    assert replayed[2].values == forecast.values
    assert replayed[3].target_weights == decision.target_weights
    manifest = RunManifest(
        run_id=f"wiring-seed-{WIRING_SEED}",
        created_at=WIRED_DEMONSTRATION_INSTANT,
        python_version=platform.python_version(),
        package_version=portlearn.__version__,
        dependency_pins={"portlearn": portlearn.__version__},
        commands=["uv sync --locked && uv run ruff check . && uv run pytest"],
    )
    serialized = manifest.to_json()
    assert serialized == manifest.to_json()
    assert WIRED_DEMONSTRATION_INSTANT.utcoffset() == timedelta(hours=10)
    assert WIRED_DEMONSTRATION_INSTANT.isoformat() in serialized
    assert RunManifest.from_json(serialized) == manifest


# ---------------------------------------------------------------------------
# Manifest serialization floors — error classes and laws only, never
# message tails: the pins are plain ``pytest.raises`` calls.
# ---------------------------------------------------------------------------


def _fixed_manifest_arguments() -> dict:
    """The fixed demonstration manifest arguments (deterministic bytes)."""
    return {
        "run_id": "wiring-seed-20260430",
        "created_at": WIRED_DEMONSTRATION_INSTANT,
        "python_version": "3.11.15",
        "package_version": "0.0.1.dev0",
        "dependency_pins": {"portlearn": "0.0.1.dev0"},
        "commands": ["uv sync --locked && uv run ruff check . && uv run pytest"],
    }


def test_manifest_identity_is_fail_closed_and_seed_free() -> None:
    """Identity floor: ``run_id``/``python_version``/``package_version``
    must be non-blank strings, ``created_at`` an aware instant, every
    ``(distribution, pin)`` pair non-blank with at least one pin, and
    ``commands`` at least one non-blank entry; the minimal public schema
    is exactly the six fields — no ``seed`` field exists."""
    from portlearn.manifest import RunManifest

    good = RunManifest(**_fixed_manifest_arguments())
    assert good.run_id == "wiring-seed-20260430"

    def _rejected(**overrides: object) -> None:
        arguments = _fixed_manifest_arguments()
        arguments.update(overrides)
        with pytest.raises(ValueError):
            RunManifest(**arguments)

    _rejected(run_id="   ")
    _rejected(python_version="")
    _rejected(package_version="  ")
    _rejected(dependency_pins={})
    _rejected(dependency_pins={"   ": "0.0.1.dev0"})
    _rejected(dependency_pins={"portlearn": ""})
    _rejected(commands=[])

    schema_keys = set(json.loads(good.to_json()))
    assert schema_keys == {
        "run_id",
        "created_at",
        "python_version",
        "package_version",
        "dependency_pins",
        "commands",
    }


def test_manifest_rejects_naive_instants_and_calendar_dates() -> None:
    """Aware-ness law: a naive ``created_at`` or a bare
    ``datetime.date`` rejects with ``NaiveTimestampError`` — the class
    from ``portlearn.timing``, never a message tail."""
    from datetime import date

    from portlearn.manifest import RunManifest
    from portlearn.timing import NaiveTimestampError

    naive_arguments = _fixed_manifest_arguments()
    naive_arguments["created_at"] = datetime(  # noqa: DTZ001 — naive by design
        2026, 4, 30, 10, 30
    )
    with pytest.raises(NaiveTimestampError):
        RunManifest(**naive_arguments)
    dated_arguments = _fixed_manifest_arguments()
    dated_arguments["created_at"] = date(2026, 4, 30)
    with pytest.raises(NaiveTimestampError):
        RunManifest(**dated_arguments)


def test_manifest_serialization_is_canonical_sorted_json_bytes() -> None:
    """Canonical serialization floor: ``to_json()`` is deterministic
    canonical JSON — sorted keys, fixed separators — asserted as the
    exact expected bytes for a fixed manifest (not merely twice-call
    identity); ``from_json`` reconstructs, and an offset-free serialized
    ``created_at`` rejects with ``NaiveTimestampError``."""
    import json

    from portlearn.manifest import RunManifest
    from portlearn.timing import NaiveTimestampError

    manifest = RunManifest(**_fixed_manifest_arguments())
    expected = (
        '{"commands":["uv sync --locked && uv run ruff check . '
        '&& uv run pytest"],'
        '"created_at":"2026-04-30T10:30:00+10:00",'
        '"dependency_pins":{"portlearn":"0.0.1.dev0"},'
        '"package_version":"0.0.1.dev0",'
        '"python_version":"3.11.15",'
        '"run_id":"wiring-seed-20260430"}'
    )
    assert manifest.to_json() == expected
    assert RunManifest.from_json(expected) == manifest
    offset_free = expected.replace("2026-04-30T10:30:00+10:00", "2026-04-30T10:30:00")
    assert "+10:00" not in offset_free
    with pytest.raises(NaiveTimestampError):
        RunManifest.from_json(offset_free)
    assert json.loads(manifest.to_json()) == json.loads(expected)


def test_manifest_round_trip_preserves_the_exact_utc_offset() -> None:
    """Round-trip floor: ``from_json(to_json(m)) == m`` exactly —
    preserving the original +10:00 UTC offset, not merely equal
    instants, and identical on re-serialization."""
    from portlearn.manifest import RunManifest

    manifest = RunManifest(**_fixed_manifest_arguments())
    reconstructed = RunManifest.from_json(manifest.to_json())
    assert reconstructed == manifest
    assert reconstructed.created_at == manifest.created_at
    assert reconstructed.created_at.utcoffset() == timedelta(hours=10)
    assert reconstructed.to_json() == manifest.to_json()


# ---------------------------------------------------------------------------
# Manifest immutability floors (cure C2) — the manifest is genuinely
# immutable: nested mutables cannot be reached for in-place mutation,
# and the constructor copies the caller's mapping.  Floors are pinned
# on error classes and laws only, never message tails.
# ---------------------------------------------------------------------------


def test_manifest_dependency_pins_reject_in_place_mutation() -> None:
    """Immutability floor: item assignment on the exposed
    ``dependency_pins`` mapping raises ``TypeError`` — the manifest
    stores a read-only mapping, never the caller's (or any) mutable
    dict."""
    from portlearn.manifest import RunManifest

    manifest = RunManifest(**_fixed_manifest_arguments())
    with pytest.raises(TypeError):
        manifest.dependency_pins["portlearn"] = "MUTATED"


def test_manifest_hash_and_serialization_survive_mutation_attempts() -> None:
    """Immutability floor: after any attempted mutation of the nested
    mapping (each individually rejected), ``hash(manifest)``,
    ``manifest == same manifest``, and ``to_json()`` bytes are
    unchanged — a stable hash needs an unreachable nested mutable."""
    from portlearn.manifest import RunManifest

    manifest = RunManifest(**_fixed_manifest_arguments())
    hash_before = hash(manifest)
    json_before = manifest.to_json()
    with pytest.raises(TypeError):
        manifest.dependency_pins["portlearn"] = "MUTATED"
    with pytest.raises(TypeError):
        manifest.dependency_pins["injected"] = "9.9.9"
    with pytest.raises(TypeError):
        del manifest.dependency_pins["portlearn"]  # type: ignore[union-attr]
    assert hash(manifest) == hash_before
    assert manifest.to_json() == json_before
    twin = RunManifest(**_fixed_manifest_arguments())
    assert manifest == twin


def test_manifest_constructor_copies_the_callers_mapping() -> None:
    """Immutability floor: mutating the caller's mapping after
    construction does not affect the manifest — the constructor copies
    its input, so manifest identity never aliases external state."""
    from portlearn.manifest import RunManifest

    pins: dict[str, str] = {"portlearn": "0.0.1.dev0"}
    manifest = RunManifest(**{**_fixed_manifest_arguments(), "dependency_pins": pins})
    pins["portlearn"] = "MUTATED-ALIAS"
    pins["injected"] = "9.9.9"
    assert manifest.dependency_pins["portlearn"] == "0.0.1.dev0"
    assert "injected" not in manifest.dependency_pins
    assert manifest == RunManifest(**_fixed_manifest_arguments())


# ---------------------------------------------------------------------------
# Duplicate-key JSON floors (cure C3) — unconditional duplicate-key
# rejection on manifest JSON parsing.
# ---------------------------------------------------------------------------


def test_manifest_json_rejects_duplicate_top_level_keys() -> None:
    """Parse floor: JSON text with a duplicated top-level schema key
    rejects with ``ValueError`` — never silently take the last value —
    and the schema-keys check still applies after unconditional parsing."""
    from portlearn.manifest import RunManifest

    manifest = RunManifest(**_fixed_manifest_arguments())
    canonical = manifest.to_json()
    duplicated = canonical.replace(
        '"package_version":"0.0.1.dev0"',
        '"package_version":"0.0.1.dev0","package_version":"OVERRIDDEN"',
    )
    assert duplicated != canonical
    with pytest.raises(ValueError):
        RunManifest.from_json(duplicated)


def test_manifest_json_rejects_duplicate_nested_keys() -> None:
    """Parse floor: JSON text with a duplicated key inside the nested
    ``dependency_pins`` object also rejects with ``ValueError``."""
    from portlearn.manifest import RunManifest

    manifest = RunManifest(**_fixed_manifest_arguments())
    canonical = manifest.to_json()
    duplicated = canonical.replace(
        '"dependency_pins":{"portlearn":"0.0.1.dev0"}',
        '"dependency_pins":{"portlearn":"0.0.1.dev0",'
        '"portlearn":"OVERRIDDEN"}',
    )
    assert duplicated != canonical
    with pytest.raises(ValueError):
        RunManifest.from_json(duplicated)


def test_manifest_json_single_key_payload_still_parses_and_round_trips() -> None:
    """Parse floor: unconditional duplicate detection does not reject
    legitimate single-key payloads: the canonical serialization parses,
    round-trips to an equal manifest, and re-serialization is
    byte-stable."""
    from portlearn.manifest import RunManifest

    manifest = RunManifest(**_fixed_manifest_arguments())
    parsed = RunManifest.from_json(manifest.to_json())
    assert parsed == manifest
    assert parsed.to_json() == manifest.to_json()


def test_manifest_json_rejects_duplicate_keys_case_insensitively() -> None:
    """Parse floor: duplicates that differ only by case do not evade —
    keys are compared exactly, so a case-differing duplicate is still
    a distinct key pair; this pins that the fix is exact-match
    unconditional, not case-folding."""
    from portlearn.manifest import RunManifest

    manifest = RunManifest(**_fixed_manifest_arguments())
    duplicated = manifest.to_json().replace(
        '"run_id":"wiring-seed-20260430"',
        '"run_id":"wiring-seed-20260430","Run_Id":"EVASION"',
    )
    # "Run_Id" is a distinct key (not a schema key), so the schema-keys
    # check rejects it — unconditional either way.
    with pytest.raises(ValueError):
        RunManifest.from_json(duplicated)
