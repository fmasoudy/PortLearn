"""Interface-contract tests for portlearn.

These tests freeze the core research interface contract: InformationSet,
FeatureTransform, Forecaster/Forecast, Strategy with the
DecisionContext/DecisionResult decision seam, PortfolioDecision, the
four runtime-checkable structural contracts, AccountingResult, and the
three validators.  Coverage is organized as a composition matrix over
synthetic collaborators, a rejection matrix of malformed inputs and
contract violations, and structural and behavioral pins on the public
surface — the exact exported-name set, the fixed five-time vocabulary,
delegation of admission to the timing law, stdlib-and-package-only
imports, and error-class ownership.
"""

from __future__ import annotations

import ast
import inspect
import sys
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from portlearn.observations import (
    AmbiguousObservationError,
    FeatureLineageError,
    TimedObservation,
    vintage_as_of,
)
from portlearn.timing import (
    FutureInformationError,
    InvalidChronologyError,
    NaiveTimestampError,
    require_available_for_decision,
)
from portlearn.weights import PortfolioWeights, WeightState

MELBOURNE = timezone(timedelta(hours=11))

# Synthetic publication and decision chronology (all aware instants):
# one quarterly observation published after quarter end, revised the next
# trading day, decided the following Monday, executed half an hour later.
QUARTER_END = datetime(2026, 6, 30, tzinfo=UTC)
FIRST_PUBLICATION = datetime(2026, 7, 2, 9, 0, tzinfo=UTC)
SECOND_PUBLICATION = datetime(2026, 7, 3, 9, 0, tzinfo=UTC)
DECISION_INSTANT = datetime(2026, 7, 6, 9, 0, tzinfo=UTC)
AFTER_DECISION = DECISION_INSTANT + timedelta(seconds=1)
BEFORE_DECISION = datetime(2026, 7, 6, 8, 59, tzinfo=UTC)
EXECUTION_INSTANT = datetime(2026, 7, 6, 9, 30, tzinfo=UTC)
LATE_PUBLICATION = datetime(2026, 8, 3, 9, 0, tzinfo=UTC)
# The same instant as DECISION_INSTANT, declared in Melbourne winter
# time (UTC+11 as of July 2026): instants compare across zones.
MELBOURNE_EQUAL_INSTANT = datetime(2026, 7, 6, 20, 0, tzinfo=MELBOURNE)

# Malformed instant fixtures (aware-instant discipline).
NAIVE_INSTANT = datetime(2026, 7, 6, 9, 0)  # noqa: DTZ001  # intentional naive instant
CALENDAR_DATE = date(2026, 7, 6)

WEIGHTS_EQUAL_PAIR = {"EQUITY.ASX.WOW": 0.5, "EQUITY.ASX.CBA": 0.5}


def observation(
    series_id: str = "MACRO.GDPQ.REAL",
    value: float = 1.0,
    observation_time: datetime = QUARTER_END,
    available_time: datetime = FIRST_PUBLICATION,
) -> TimedObservation:
    """One admitted point-in-time record for the synthetic fixtures."""
    return TimedObservation(
        series_id=series_id,
        observation_time=observation_time,
        available_time=available_time,
        value=value,
    )


# ---------------------------------------------------------------------------
# Composition matrix
# ---------------------------------------------------------------------------


def test_synthetic_transform_produces_lineage_valid_outputs() -> None:
    from portlearn.interfaces import require_feature_lineage

    class _QuarterlyMomentumTransform:
        def transform(self, observations):
            latest_input_availability = max(
                item.available_time for item in observations
            )
            return [
                TimedObservation(
                    series_id="FEAT.MOM.GDPQ.252D",
                    observation_time=QUARTER_END,
                    available_time=latest_input_availability,
                    value=0.8,
                ),
                TimedObservation(
                    series_id="FEAT.MOM.CPI.252D",
                    observation_time=QUARTER_END,
                    available_time=latest_input_availability,
                    value=-0.3,
                ),
            ]

    inputs = [
        observation(series_id="MACRO.GDPQ.REAL", available_time=FIRST_PUBLICATION),
        observation(series_id="MACRO.CPI.YOY", available_time=SECOND_PUBLICATION),
    ]
    outputs = _QuarterlyMomentumTransform().transform(inputs)
    assert all(isinstance(output, TimedObservation) for output in outputs)
    assert all(
        output.available_time >= FIRST_PUBLICATION for output in outputs
    )
    assert require_feature_lineage(inputs, outputs) is None


def test_require_feature_lineage_accepts_monotone_outputs() -> None:
    from portlearn.interfaces import require_feature_lineage

    inputs = [
        observation(available_time=FIRST_PUBLICATION),
        observation(available_time=SECOND_PUBLICATION),
    ]
    boundary_output = observation(
        series_id="FEAT.MOM.GDPQ.252D", available_time=SECOND_PUBLICATION
    )
    later_output = observation(
        series_id="FEAT.MOM.CPI.252D", available_time=DECISION_INSTANT
    )
    assert (
        require_feature_lineage(inputs, [boundary_output, later_output]) is None
    )


def test_information_set_of_admitted_observations_is_iterable_with_as_of() -> None:
    from portlearn.interfaces import InformationSet

    admitted = [
        observation(series_id="MACRO.GDPQ.REAL"),
        observation(series_id="MACRO.CPI.YOY"),
        observation(series_id="FX.AUDUSD.SPOT", value=0.6612),
    ]
    information_set = InformationSet(admitted, DECISION_INSTANT)
    assert list(information_set) == admitted
    assert list(information_set) == admitted
    assert all(
        carried is record for carried, record in zip(information_set, admitted)
    )
    assert information_set.as_of == DECISION_INSTANT


def test_information_set_empty_is_admissible_empty_set() -> None:
    from portlearn.interfaces import InformationSet

    empty_from_list = InformationSet([], DECISION_INSTANT)
    empty_from_tuple = InformationSet((), DECISION_INSTANT)
    empty_from_generator = InformationSet(
        (record for record in ()), DECISION_INSTANT
    )
    for empty_set in (empty_from_list, empty_from_tuple, empty_from_generator):
        assert list(empty_set) == []
        assert empty_set.as_of == DECISION_INSTANT


def test_synthetic_forecaster_returns_forecast_dated_at_set_as_of() -> None:
    from portlearn.interfaces import Forecast, InformationSet

    class _MeanForecaster:
        def forecast(self, admitted_information):
            return Forecast(
                values={"EQUITY.ASX.WOW": 0.031, "EQUITY.ASX.CBA": 0.018},
                target="expected_return",
                decision_time=admitted_information.as_of,
                produced_by="synthetic-mean-forecaster",
            )

    information_set = InformationSet(
        [
            observation(series_id="MACRO.GDPQ.REAL"),
            observation(series_id="MACRO.CPI.YOY"),
        ],
        DECISION_INSTANT,
    )
    forecast = _MeanForecaster().forecast(information_set)
    assert forecast.decision_time == information_set.as_of


def test_synthetic_strategy_decision_dated_at_or_after_forecast_and_compatible() -> None:
    from portlearn.interfaces import (
        DecisionContext,
        DecisionResult,
        Forecast,
        InformationSet,
        PortfolioDecision,
        require_decision_result_compatible,
        require_forecast_decision_compatible,
    )

    class _EqualWeightStrategy:
        def decide(self, context):
            decided_at = context.decision_time
            return DecisionResult(
                decision=PortfolioDecision(
                    decision_time=decided_at,
                    execution_time=decided_at + timedelta(minutes=30),
                    target_weights=dict(WEIGHTS_EQUAL_PAIR),
                )
            )

    information_set = InformationSet([observation()], DECISION_INSTANT)
    forecast = Forecast(
        values={"EQUITY.ASX.WOW": 0.031, "EQUITY.ASX.CBA": 0.018},
        target="expected_return",
        decision_time=information_set.as_of,
        produced_by="synthetic-mean-forecaster",
    )
    context = DecisionContext(
        decision_time=information_set.as_of,
        information=information_set,
        universe=("EQUITY.ASX.WOW", "EQUITY.ASX.CBA"),
        current_weights=PortfolioWeights(
            dict(WEIGHTS_EQUAL_PAIR), WeightState.PRE_TRADE
        ),
        current_weights_as_of=BEFORE_DECISION,
        forecast=forecast,
    )
    result = _EqualWeightStrategy().decide(context)
    decision = result.decision
    assert forecast.decision_time == decision.decision_time
    assert decision.decision_time <= decision.execution_time
    assert result.next_strategy_state is None
    assert require_forecast_decision_compatible(forecast, decision) is None
    assert require_decision_result_compatible(context, result) is None


def test_rebalance_policy_runtime_checkable_presence_discriminates() -> None:
    from portlearn.interfaces import RebalancePolicy

    class _QuarterlyReviewPolicy:
        def should_rebalance(self, decision_time, last_rebalance_time) -> bool:
            return decision_time >= last_rebalance_time

    class _CalendarBlindPolicy:
        def review_schedule(self) -> str:
            return "undated"

    policy = _QuarterlyReviewPolicy()
    assert isinstance(policy, RebalancePolicy)
    assert not isinstance(_CalendarBlindPolicy(), RebalancePolicy)
    for base in type(policy).__mro__:
        assert not base.__module__.startswith("portlearn")


def test_cost_model_runtime_checkable_presence_discriminates() -> None:
    from portlearn.interfaces import CostModel

    class _SpreadAndCommissionModel:
        def estimate_trade_cost(self, pre_trade_weights, target_weights) -> float:
            return 12.5

    class _NoCostQuotation:
        def quote_spread(self) -> float:
            return 0.0

    model = _SpreadAndCommissionModel()
    assert isinstance(model, CostModel)
    assert not isinstance(_NoCostQuotation(), CostModel)
    for base in type(model).__mro__:
        assert not base.__module__.startswith("portlearn")


def test_accounting_engine_runtime_checkable_presence_discriminates() -> None:
    from portlearn.interfaces import AccountingEngine

    class _TwoWayFeeEngine:
        def account(self, decision, pre_trade_weights, realized_returns):
            return decision.target_weights

    class _NoAccountingBook:
        def summarize_fees(self) -> float:
            return 0.0

    engine = _TwoWayFeeEngine()
    assert isinstance(engine, AccountingEngine)
    assert not isinstance(_NoAccountingBook(), AccountingEngine)
    for base in type(engine).__mro__:
        assert not base.__module__.startswith("portlearn")


def test_evaluator_runtime_checkable_presence_discriminates() -> None:
    from portlearn.interfaces import Evaluator

    class _RatioEvaluator:
        def evaluate(self, accounting_result):
            return {"weight_gross_exposure": 1.0}

    class _NoVerdictReport:
        def summarize(self) -> str:
            return "no verdict"

    evaluator = _RatioEvaluator()
    assert isinstance(evaluator, Evaluator)
    assert not isinstance(_NoVerdictReport(), Evaluator)
    for base in type(evaluator).__mro__:
        assert not base.__module__.startswith("portlearn")


def test_full_chain_composes_end_to_end_on_synthetic_data() -> None:
    from portlearn.interfaces import (
        AccountingResult,
        DecisionContext,
        DecisionResult,
        Forecast,
        InformationSet,
        PortfolioDecision,
    )

    class _MeanReversionForecaster:
        def forecast(self, admitted_information):
            return Forecast(
                values={"EQUITY.ASX.WOW": -0.004, "EQUITY.ASX.CBA": 0.011},
                target="expected_return",
                decision_time=admitted_information.as_of,
                produced_by="synthetic-mean-reversion-forecaster",
            )

    class _EqualWeightStrategy:
        def decide(self, context):
            decided_at = context.decision_time
            return DecisionResult(
                decision=PortfolioDecision(
                    decision_time=decided_at,
                    execution_time=decided_at + timedelta(minutes=30),
                    target_weights=dict(WEIGHTS_EQUAL_PAIR),
                )
            )

    class _TwoWayFeeAccountingEngine:
        def account(self, decision, pre_trade_weights, realized_returns):
            return AccountingResult(post_trade_weights=dict(decision.target_weights))

    class _RatioEvaluator:
        def evaluate(self, accounting_result):
            return {"weight_gross_exposure": 1.0}

    information_set = InformationSet(
        [
            observation(series_id="MACRO.GDPQ.REAL"),
            observation(series_id="MACRO.CPI.YOY"),
        ],
        DECISION_INSTANT,
    )
    forecast = _MeanReversionForecaster().forecast(information_set)

    # The current-holdings snapshot and realized returns below are
    # fabricated literally in this test body — no interface in this
    # module produces or validates them. The snapshot enters the
    # decision context as the decision-time current_weights book; the
    # accounting engine still receives the plain mapping.
    current_snapshot = {"EQUITY.ASX.WOW": 0.6, "EQUITY.ASX.CBA": 0.4}
    realized_returns = {"EQUITY.ASX.WOW": -0.021, "EQUITY.ASX.CBA": 0.017}
    context = DecisionContext(
        decision_time=information_set.as_of,
        information=information_set,
        universe=("EQUITY.ASX.WOW", "EQUITY.ASX.CBA"),
        current_weights=PortfolioWeights(
            dict(current_snapshot), WeightState.PRE_TRADE
        ),
        current_weights_as_of=BEFORE_DECISION,
        forecast=forecast,
    )
    decision = _EqualWeightStrategy().decide(context).decision

    accounting_result = _TwoWayFeeAccountingEngine().account(
        decision, current_snapshot, realized_returns
    )
    assert isinstance(accounting_result, AccountingResult)
    evaluation = _RatioEvaluator().evaluate(accounting_result)
    assert isinstance(evaluation, Mapping)
    assert all(
        isinstance(metric, str) and isinstance(value, float)
        for metric, value in evaluation.items()
    )


# ---------------------------------------------------------------------------
# Rejection matrix
# ---------------------------------------------------------------------------


def test_information_set_rejects_lookahead_item() -> None:
    from portlearn.interfaces import InformationSet

    admitted = observation(series_id="MACRO.GDPQ.REAL")
    not_yet_published = observation(
        series_id="MACRO.CPI.YOY", available_time=AFTER_DECISION
    )
    with pytest.raises(FutureInformationError):
        InformationSet([admitted, not_yet_published], DECISION_INSTANT)


def test_information_set_rejects_duplicate_identity_equal_values() -> None:
    from portlearn.interfaces import InformationSet

    first_copy = observation(series_id="MACRO.GDPQ.REAL", value=2.5)
    identical_copy = observation(series_id="MACRO.GDPQ.REAL", value=2.5)
    with pytest.raises(AmbiguousObservationError):
        InformationSet([first_copy, identical_copy], DECISION_INSTANT)


def test_information_set_rejects_duplicate_identity_conflicting_values() -> None:
    from portlearn.interfaces import InformationSet

    advance_estimate = observation(series_id="MACRO.GDPQ.REAL", value=2.5)
    conflicting_rerelease = observation(series_id="MACRO.GDPQ.REAL", value=3.0)
    with pytest.raises(AmbiguousObservationError):
        InformationSet(
            [advance_estimate, conflicting_rerelease], DECISION_INSTANT
        )


def test_information_set_rejects_duplicate_identity_that_individually_pass_admission() -> None:
    from portlearn.interfaces import InformationSet

    # Both records sit exactly at the inclusive admission boundary, so
    # each would individually be admitted; the shared identity triple
    # still rejects (no value-equality exception).
    boundary_first = observation(
        series_id="MACRO.GDPQ.REAL",
        value=2.5,
        available_time=DECISION_INSTANT,
    )
    boundary_second = observation(
        series_id="MACRO.GDPQ.REAL",
        value=2.5,
        available_time=DECISION_INSTANT,
    )
    with pytest.raises(AmbiguousObservationError):
        InformationSet([boundary_first, boundary_second], DECISION_INSTANT)


def test_information_set_rejects_revision_pair() -> None:
    from portlearn.interfaces import InformationSet

    first_release = observation(
        series_id="MACRO.GDPQ.REAL", value=1.0, available_time=FIRST_PUBLICATION
    )
    next_day_revision = observation(
        series_id="MACRO.GDPQ.REAL", value=2.0, available_time=SECOND_PUBLICATION
    )
    with pytest.raises(AmbiguousObservationError):
        InformationSet([first_release, next_day_revision], DECISION_INSTANT)


def test_information_set_rejects_revision_pair_with_inadmissible_later_member_still_ambiguous() -> None:
    from portlearn.interfaces import InformationSet

    # The later vintage is itself not yet admissible (its availability
    # is after as_of), yet the input-domain refusal fires before any
    # admission verdict: the pair is ambiguous, never silently filtered
    # down to its admissible member.
    first_release = observation(
        series_id="MACRO.GDPQ.REAL", value=1.0, available_time=FIRST_PUBLICATION
    )
    unreleased_revision = observation(
        series_id="MACRO.GDPQ.REAL", value=3.0, available_time=LATE_PUBLICATION
    )
    with pytest.raises(AmbiguousObservationError):
        InformationSet([first_release, unreleased_revision], DECISION_INSTANT)


def test_naive_or_date_instants_rejected_at_every_interface_boundary() -> None:
    from portlearn.interfaces import (
        Forecast,
        InformationSet,
        PortfolioDecision,
        require_forecast_decision_compatible,
    )

    class _UndatedForecast:
        decision_time = NAIVE_INSTANT

    class _UndatedDecision:
        decision_time = NAIVE_INSTANT

    for malformed_instant in (NAIVE_INSTANT, CALENDAR_DATE):
        with pytest.raises(NaiveTimestampError):
            InformationSet([], malformed_instant)
        with pytest.raises(NaiveTimestampError):
            Forecast(
                values={"EQUITY.ASX.WOW": 0.031},
                target="expected_return",
                decision_time=malformed_instant,
                produced_by="synthetic-mean-forecaster",
            )
        with pytest.raises(NaiveTimestampError):
            PortfolioDecision(
                decision_time=malformed_instant,
                execution_time=EXECUTION_INSTANT,
                target_weights=dict(WEIGHTS_EQUAL_PAIR),
            )
        with pytest.raises(NaiveTimestampError):
            PortfolioDecision(
                decision_time=DECISION_INSTANT,
                execution_time=malformed_instant,
                target_weights=dict(WEIGHTS_EQUAL_PAIR),
            )

    dated_forecast = Forecast(
        values={"EQUITY.ASX.WOW": 0.031},
        target="expected_return",
        decision_time=DECISION_INSTANT,
        produced_by="synthetic-mean-forecaster",
    )
    dated_decision = PortfolioDecision(
        decision_time=DECISION_INSTANT,
        execution_time=EXECUTION_INSTANT,
        target_weights=dict(WEIGHTS_EQUAL_PAIR),
    )
    with pytest.raises(NaiveTimestampError):
        require_forecast_decision_compatible(
            _UndatedForecast(), dated_decision
        )
    with pytest.raises(NaiveTimestampError):
        require_forecast_decision_compatible(
            dated_forecast, _UndatedDecision()
        )


def test_portfolio_decision_enforces_decision_le_execution() -> None:
    from portlearn.interfaces import PortfolioDecision

    with pytest.raises(InvalidChronologyError):
        PortfolioDecision(
            decision_time=EXECUTION_INSTANT,
            execution_time=DECISION_INSTANT,
            target_weights=dict(WEIGHTS_EQUAL_PAIR),
        )
    same_instant = PortfolioDecision(
        decision_time=DECISION_INSTANT,
        execution_time=DECISION_INSTANT,
        target_weights=dict(WEIGHTS_EQUAL_PAIR),
    )
    assert same_instant.execution_time == same_instant.decision_time


def test_require_forecast_decision_compatible_rejects_decision_before_forecast() -> None:
    from portlearn.interfaces import (
        Forecast,
        PortfolioDecision,
        require_forecast_decision_compatible,
    )

    forecast = Forecast(
        values={"EQUITY.ASX.WOW": 0.031},
        target="expected_return",
        decision_time=DECISION_INSTANT,
        produced_by="synthetic-mean-forecaster",
    )
    pre_information_decision = PortfolioDecision(
        decision_time=BEFORE_DECISION,
        execution_time=DECISION_INSTANT,
        target_weights=dict(WEIGHTS_EQUAL_PAIR),
    )
    with pytest.raises(InvalidChronologyError):
        require_forecast_decision_compatible(forecast, pre_information_decision)


def test_require_feature_lineage_rejects_output_before_latest_input() -> None:
    from portlearn.interfaces import require_feature_lineage

    inputs = [
        observation(available_time=FIRST_PUBLICATION),
        observation(available_time=SECOND_PUBLICATION),
    ]
    premature_output = observation(
        series_id="FEAT.MOM.GDPQ.252D", available_time=FIRST_PUBLICATION
    )
    with pytest.raises(FeatureLineageError):
        require_feature_lineage(inputs, [premature_output])


def test_require_feature_lineage_rejects_empty_inputs() -> None:
    from portlearn.interfaces import require_feature_lineage

    orphan_output = observation(series_id="FEAT.MOM.GDPQ.252D")
    with pytest.raises(FeatureLineageError):
        require_feature_lineage([], [orphan_output])


def test_blank_target_produced_by_and_identifier_keys_rejected() -> None:
    from portlearn.interfaces import Forecast

    quoted_values = {"EQUITY.ASX.WOW": 0.031}
    for blank_target in ("", "   ", "\t\n"):
        with pytest.raises(ValueError):
            Forecast(
                values=dict(quoted_values),
                target=blank_target,
                decision_time=DECISION_INSTANT,
                produced_by="synthetic-mean-forecaster",
            )
    for blank_provenance in ("", "  "):
        with pytest.raises(ValueError):
            Forecast(
                values=dict(quoted_values),
                target="expected_return",
                decision_time=DECISION_INSTANT,
                produced_by=blank_provenance,
            )
    with pytest.raises(ValueError):
        Forecast(
            values={"   ": 0.031},
            target="expected_return",
            decision_time=DECISION_INSTANT,
            produced_by="synthetic-mean-forecaster",
        )


def test_blank_identifier_keys_rejected() -> None:
    from portlearn.interfaces import Forecast, PortfolioDecision

    with pytest.raises(ValueError):
        Forecast(
            values={"": 0.02},
            target="expected_return",
            decision_time=DECISION_INSTANT,
            produced_by="synthetic-mean-forecaster",
        )
    with pytest.raises(ValueError):
        Forecast(
            values={"EQUITY.ASX.WOW": 0.02, "  ": 0.01},
            target="expected_return",
            decision_time=DECISION_INSTANT,
            produced_by="synthetic-mean-forecaster",
        )
    with pytest.raises(ValueError):
        PortfolioDecision(
            decision_time=DECISION_INSTANT,
            execution_time=EXECUTION_INSTANT,
            target_weights={"": 0.5},
        )
    with pytest.raises(ValueError):
        PortfolioDecision(
            decision_time=DECISION_INSTANT,
            execution_time=EXECUTION_INSTANT,
            target_weights={"EQUITY.ASX.WOW": 0.5, "\t": 0.5},
        )


# ---------------------------------------------------------------------------
# Structural and behavioral pins
# ---------------------------------------------------------------------------


def test_surface_exports_exactly_thirteen_names_plus_three_validators() -> None:
    import portlearn.interfaces

    interfaces = portlearn.interfaces
    expected_surface = frozenset(
        {
            "AccountingResult",
            "FeatureTransform",
            "Forecast",
            "Forecaster",
            "InformationSet",
            "PortfolioDecision",
            "RebalancePolicy",
            "Strategy",
            "CostModel",
            "AccountingEngine",
            "Evaluator",
            "DecisionContext",
            "DecisionResult",
            "require_decision_result_compatible",
            "require_feature_lineage",
            "require_forecast_decision_compatible",
        }
    )
    exported = interfaces.__all__
    assert len(exported) == 16
    assert set(exported) == expected_surface
    assert list(exported) == sorted(exported)

    # The import ceiling: every name an implementation may bind by
    # import (plus the __future__ binding) is explicitly allowed, so any
    # other public module-owned name must live in __all__.
    ceiling_imports = frozenset(
        {
            "annotations",
            "Iterable",
            "Mapping",
            "Sequence",
            "dataclass",
            "date",
            "datetime",
            "Protocol",
            "runtime_checkable",
            "AmbiguousObservationError",
            "TimedObservation",
            "require_lineage_monotone",
            "InvalidChronologyError",
            "NaiveTimestampError",
            "require_available_for_decision",
            "PortfolioWeights",
            "WeightState",
        }
    )
    public_names = {
        name for name in dir(interfaces) if not name.startswith("_")
    }
    module_owned = public_names - ceiling_imports
    assert module_owned <= set(exported)
    assert set(exported) <= public_names


def test_no_time_fields_beyond_the_five_time_law() -> None:
    import portlearn.interfaces

    interfaces = portlearn.interfaces
    time_nouns = frozenset(
        {
            "observation_time",
            "available_time",
            "decision_time",
            "execution_time",
            "start_time",
            "end_time",
            "as_of",
        }
    )
    for exported in (
        "InformationSet",
        "Forecast",
        "PortfolioDecision",
        "AccountingResult",
        "DecisionContext",
        "DecisionResult",
    ):
        contract = getattr(interfaces, exported)
        for parameter in inspect.signature(contract).parameters:
            if "time" in parameter or parameter.endswith("_at"):
                assert parameter in time_nouns, parameter

    forecast = interfaces.Forecast(
        values={"EQUITY.ASX.WOW": 0.02},
        target="expected_return",
        decision_time=DECISION_INSTANT,
        produced_by="synthetic-mean-forecaster",
    )
    decision = interfaces.PortfolioDecision(
        decision_time=DECISION_INSTANT,
        execution_time=EXECUTION_INSTANT,
        target_weights=dict(WEIGHTS_EQUAL_PAIR),
    )
    accounting_result = interfaces.AccountingResult(
        post_trade_weights=dict(WEIGHTS_EQUAL_PAIR)
    )
    information_set = interfaces.InformationSet(
        [observation()], DECISION_INSTANT
    )
    for value_object in (forecast, decision, accounting_result, information_set):
        for attribute in dir(value_object):
            if not attribute.startswith("_") and (
                "time" in attribute or attribute.endswith("_at")
            ):
                assert attribute in time_nouns, attribute


def test_information_set_delegates_admission_to_timing_predicate() -> None:
    import portlearn.interfaces

    interfaces = portlearn.interfaces
    boundary_item = observation(available_time=DECISION_INSTANT)
    boundary_set = interfaces.InformationSet([boundary_item], DECISION_INSTANT)
    assert list(boundary_set) == [boundary_item]

    not_yet_published = observation(
        series_id="MACRO.CPI.YOY.FLASH", available_time=AFTER_DECISION
    )
    with pytest.raises(FutureInformationError) as direct_law:
        require_available_for_decision(not_yet_published, DECISION_INSTANT)
    with pytest.raises(FutureInformationError) as through_set:
        interfaces.InformationSet([not_yet_published], DECISION_INSTANT)
    assert str(direct_law.value) == str(through_set.value)

    # No local availability comparison may exist inside InformationSet —
    # admission is the delegated timing predicate only.
    source = Path(interfaces.__file__).read_text(encoding="utf-8")
    class_definitions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "InformationSet"
    ]
    assert len(class_definitions) == 1
    for comparison in ast.walk(class_definitions[0]):
        if isinstance(comparison, ast.Compare):
            for operand in (comparison.left, *comparison.comparators):
                referenced = {
                    name.id
                    for name in ast.walk(operand)
                    if isinstance(name, ast.Name)
                } | {
                    attribute.attr
                    for attribute in ast.walk(operand)
                    if isinstance(attribute, ast.Attribute)
                }
                assert "available_time" not in referenced


def test_interfaces_module_defines_no_error_classes() -> None:
    import portlearn.interfaces

    interfaces = portlearn.interfaces
    for name in dir(interfaces):
        attribute = getattr(interfaces, name)
        if isinstance(attribute, type) and issubclass(attribute, BaseException):
            assert attribute.__module__ != "portlearn.interfaces", name
    # Reused errors that only propagate through the delegated laws are
    # never re-exported by this module.
    assert not hasattr(interfaces, "FutureInformationError")
    assert not hasattr(interfaces, "FeatureLineageError")
    assert not hasattr(interfaces, "MissingAvailabilityError")
    for name, owner_module in (
        ("NaiveTimestampError", "portlearn.timing"),
        ("InvalidChronologyError", "portlearn.timing"),
        ("AmbiguousObservationError", "portlearn.observations"),
    ):
        exposed = getattr(interfaces, name, None)
        if exposed is not None:
            assert exposed is getattr(sys.modules[owner_module], name)


def test_interfaces_imports_are_stdlib_and_portlearn_only() -> None:
    import portlearn.interfaces

    interfaces = portlearn.interfaces
    source = Path(interfaces.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in sys.stdlib_module_names
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                assert node.module in {"observations", "timing", "weights"}
            else:
                assert (node.module or "").split(".")[0] in (
                    sys.stdlib_module_names
                )


def test_information_set_constructs_from_iterable_of_timed_observations() -> None:
    from portlearn.interfaces import InformationSet

    gdp_estimate = observation(series_id="MACRO.GDPQ.REAL")
    cpi_estimate = observation(series_id="MACRO.CPI.YOY")
    information_set = InformationSet(
        iter([gdp_estimate, cpi_estimate]), DECISION_INSTANT
    )
    carried = list(information_set)
    assert carried == [gdp_estimate, cpi_estimate]
    assert carried[0] is gdp_estimate
    assert carried[1] is cpi_estimate


def test_information_set_admits_items_at_as_of() -> None:
    from portlearn.interfaces import InformationSet

    published_last_week = observation(
        series_id="MACRO.GDPQ.REAL", available_time=FIRST_PUBLICATION
    )
    published_exactly_now = observation(
        series_id="FX.AUDUSD.SPOT",
        value=0.6612,
        available_time=MELBOURNE_EQUAL_INSTANT,
    )
    information_set = InformationSet(
        [published_last_week, published_exactly_now], DECISION_INSTANT
    )
    assert list(information_set) == [published_last_week, published_exactly_now]
    assert information_set.as_of == DECISION_INSTANT


def test_information_set_admission_follows_input_order() -> None:
    from portlearn.interfaces import InformationSet

    first_offender = observation(
        series_id="MACRO.CPI.YOY.FLASH", available_time=AFTER_DECISION
    )
    second_offender = observation(
        series_id="MACRO.GDPQ.REAL.ADVANCE", available_time=LATE_PUBLICATION
    )
    with pytest.raises(FutureInformationError) as rejection:
        InformationSet([first_offender, second_offender], DECISION_INSTANT)
    assert "MACRO.CPI.YOY.FLASH" in str(rejection.value)
    assert "MACRO.GDPQ.REAL.ADVANCE" not in str(rejection.value)

    cpi_estimate = observation(series_id="MACRO.CPI.YOY")
    gdp_estimate = observation(series_id="MACRO.GDPQ.REAL")
    admitted = InformationSet(
        [cpi_estimate, gdp_estimate], DECISION_INSTANT
    )
    assert list(admitted) == [cpi_estimate, gdp_estimate]


def test_information_set_is_iterable_and_as_of_read_only() -> None:
    from portlearn.interfaces import InformationSet

    admitted_gdp = observation(series_id="MACRO.GDPQ.REAL")
    source_items = [admitted_gdp]
    information_set = InformationSet(source_items, DECISION_INSTANT)
    with pytest.raises(AttributeError):
        information_set.as_of = AFTER_DECISION
    source_items.append(observation(series_id="MACRO.CPI.YOY"))
    assert list(information_set) == [admitted_gdp]
    assert list(information_set) == [admitted_gdp]
    assert information_set.as_of == DECISION_INSTANT


def test_information_set_has_no_query_or_selection_api() -> None:
    from portlearn.interfaces import InformationSet

    information_set = InformationSet([observation()], DECISION_INSTANT)
    public_surface = {
        name for name in dir(information_set) if not name.startswith("_")
    }
    assert public_surface == {"as_of"}
    assert not hasattr(information_set, "__contains__")
    assert not hasattr(information_set, "__getitem__")
    assert not hasattr(information_set, "__len__")


def test_information_set_naive_as_of_rejected_before_item_checks() -> None:
    from portlearn.interfaces import InformationSet

    not_yet_published = observation(
        series_id="MACRO.CPI.YOY", available_time=AFTER_DECISION
    )
    duplicate_pair = [observation(), observation()]
    with pytest.raises(NaiveTimestampError):
        InformationSet([not_yet_published], NAIVE_INSTANT)
    with pytest.raises(NaiveTimestampError):
        InformationSet(duplicate_pair, NAIVE_INSTANT)


def test_information_set_never_selects_among_vintages() -> None:
    from portlearn.interfaces import InformationSet

    revision_history = [
        observation(value=1.0, available_time=FIRST_PUBLICATION),
        observation(value=2.0, available_time=SECOND_PUBLICATION),
        observation(value=3.0, available_time=LATE_PUBLICATION),
    ]
    selected_vintage = vintage_as_of(revision_history, DECISION_INSTANT)
    assert selected_vintage is not None
    assert selected_vintage.value == 2.0
    with pytest.raises(AmbiguousObservationError):
        InformationSet(revision_history, DECISION_INSTANT)
    point_in_time = InformationSet([selected_vintage], DECISION_INSTANT)
    assert list(point_in_time) == [selected_vintage]


def test_feature_transform_protocol_is_static_only() -> None:
    from portlearn.interfaces import FeatureTransform

    class _WindowedTransform:
        def transform(self, observations):
            return list(observations)

    with pytest.raises(TypeError):
        isinstance(_WindowedTransform(), FeatureTransform)


def test_forecaster_protocol_is_static_only() -> None:
    from portlearn.interfaces import Forecaster

    class _ConstantForecaster:
        def forecast(self, information_set):
            return information_set

    with pytest.raises(TypeError):
        isinstance(_ConstantForecaster(), Forecaster)


def test_forecast_carries_values_target_decision_time_produced_by() -> None:
    from portlearn.interfaces import Forecast

    values = {"EQUITY.ASX.WOW": 0.031, "EQUITY.ASX.CBA": 0.018}
    forecast = Forecast(
        values=values,
        target="expected_return",
        decision_time=DECISION_INSTANT,
        produced_by="synthetic-mean-forecaster",
    )
    assert forecast.values is values
    assert forecast.target == "expected_return"
    assert forecast.decision_time == DECISION_INSTANT
    assert forecast.produced_by == "synthetic-mean-forecaster"
    with pytest.raises(FrozenInstanceError):
        forecast.target = "value_at_risk"


def test_forecast_values_unconstrained() -> None:
    from portlearn.interfaces import Forecast

    for unconstrained_values in (
        {"EQUITY.ASX.WOW": -0.75},
        {"MACRO.CPI.YOY": 2.5},
        {"FX.AUDUSD.SPOT": 1e12},
    ):
        forecast = Forecast(
            values=unconstrained_values,
            target="expected_return",
            decision_time=DECISION_INSTANT,
            produced_by="synthetic-extreme-forecaster",
        )
        assert forecast.values == unconstrained_values


def test_require_forecast_decision_compatible_accepts_equal_and_later() -> None:
    from portlearn.interfaces import (
        Forecast,
        PortfolioDecision,
        require_forecast_decision_compatible,
    )

    equal_forecast = Forecast(
        values={"EQUITY.ASX.WOW": 0.031},
        target="expected_return",
        decision_time=DECISION_INSTANT,
        produced_by="synthetic-mean-forecaster",
    )
    equal_decision = PortfolioDecision(
        decision_time=DECISION_INSTANT,
        execution_time=EXECUTION_INSTANT,
        target_weights=dict(WEIGHTS_EQUAL_PAIR),
    )
    assert equal_forecast.decision_time == equal_decision.decision_time
    assert (
        require_forecast_decision_compatible(equal_forecast, equal_decision)
        is None
    )

    later_decision = PortfolioDecision(
        decision_time=AFTER_DECISION,
        execution_time=EXECUTION_INSTANT,
        target_weights=dict(WEIGHTS_EQUAL_PAIR),
    )
    assert equal_forecast.decision_time < later_decision.decision_time
    assert later_decision.decision_time <= later_decision.execution_time
    assert (
        require_forecast_decision_compatible(equal_forecast, later_decision)
        is None
    )


def test_strategy_protocol_is_static_only() -> None:
    from portlearn.interfaces import Strategy

    class _BuyAndHoldStrategy:
        def decide(self, context):
            return context

    with pytest.raises(TypeError):
        isinstance(_BuyAndHoldStrategy(), Strategy)


def test_portfolio_decision_naive_instants_rejected() -> None:
    from portlearn.interfaces import PortfolioDecision

    for naive_instant in (NAIVE_INSTANT, CALENDAR_DATE):
        with pytest.raises(NaiveTimestampError):
            PortfolioDecision(
                decision_time=naive_instant,
                execution_time=EXECUTION_INSTANT,
                target_weights=dict(WEIGHTS_EQUAL_PAIR),
            )
        with pytest.raises(NaiveTimestampError):
            PortfolioDecision(
                decision_time=DECISION_INSTANT,
                execution_time=naive_instant,
                target_weights=dict(WEIGHTS_EQUAL_PAIR),
            )


def test_portfolio_decision_weight_sum_not_validated() -> None:
    from portlearn.interfaces import PortfolioDecision

    levered_book = {"EQUITY.ASX.WOW": 1.25, "EQUITY.ASX.CBA": 0.75}
    light_book = {"EQUITY.ASX.WOW": 0.2, "EQUITY.ASX.CBA": 0.1}
    for unconstrained_weights in (levered_book, light_book):
        decision = PortfolioDecision(
            decision_time=DECISION_INSTANT,
            execution_time=EXECUTION_INSTANT,
            target_weights=unconstrained_weights,
        )
        assert decision.target_weights == unconstrained_weights


def test_accounting_result_is_frozen_dataclass_with_exactly_one_field() -> None:
    from portlearn.interfaces import AccountingResult

    result_fields = fields(AccountingResult)
    assert len(result_fields) == 1
    assert result_fields[0].name == "post_trade_weights"
    accounting_result = AccountingResult(
        post_trade_weights=dict(WEIGHTS_EQUAL_PAIR)
    )
    with pytest.raises(FrozenInstanceError):
        accounting_result.post_trade_weights = {"EQUITY.ASX.WOW": 1.0}


def test_exactly_four_runtime_checkable_protocols() -> None:
    from portlearn.interfaces import (
        AccountingEngine,
        CostModel,
        Evaluator,
        FeatureTransform,
        Forecaster,
        RebalancePolicy,
        Strategy,
    )

    class _SyntheticDesk:
        def should_rebalance(self, decision_time, last_rebalance_time) -> bool:
            return False

        def estimate_trade_cost(
            self, pre_trade_weights, target_weights
        ) -> float:
            return 0.0

        def account(self, decision, pre_trade_weights, realized_returns):
            return decision.target_weights

        def evaluate(self, accounting_result):
            return {"weight_gross_exposure": 1.0}

    desk = _SyntheticDesk()
    for runtime_checkable in (
        RebalancePolicy,
        CostModel,
        AccountingEngine,
        Evaluator,
    ):
        assert isinstance(desk, runtime_checkable)
    for static_only in (FeatureTransform, Forecaster, Strategy):
        with pytest.raises(TypeError):
            isinstance(desk, static_only)


# ---------------------------------------------------------------------------
# DecisionContext and DecisionResult contract laws (the decision seam)
# ---------------------------------------------------------------------------


def decision_context(**overrides):
    """One lawful decision context for the decision-law tests."""
    from typing import Any

    from portlearn.interfaces import DecisionContext, InformationSet

    kwargs: dict[str, Any] = {
        "decision_time": DECISION_INSTANT,
        "information": InformationSet([observation()], DECISION_INSTANT),
        "universe": ("EQUITY.ASX.WOW", "EQUITY.ASX.CBA"),
        "current_weights": PortfolioWeights(
            dict(WEIGHTS_EQUAL_PAIR), WeightState.PRE_TRADE
        ),
        "current_weights_as_of": BEFORE_DECISION,
    }
    kwargs.update(overrides)
    return DecisionContext(**kwargs)


def test_decision_context_requires_aware_instants() -> None:
    for naive_instant in (NAIVE_INSTANT, CALENDAR_DATE):
        with pytest.raises(NaiveTimestampError):
            decision_context(decision_time=naive_instant)
        with pytest.raises(NaiveTimestampError):
            decision_context(current_weights_as_of=naive_instant)


def test_decision_context_rejects_post_decision_information() -> None:
    from portlearn.interfaces import InformationSet

    late_set = InformationSet([observation()], AFTER_DECISION)
    with pytest.raises(InvalidChronologyError):
        decision_context(information=late_set)


def test_decision_context_rejects_late_holdings_snapshot() -> None:
    with pytest.raises(InvalidChronologyError):
        decision_context(current_weights_as_of=AFTER_DECISION)


def test_decision_context_rejects_misaligned_forecast() -> None:
    from portlearn.interfaces import Forecast

    for other_origin in (BEFORE_DECISION, AFTER_DECISION):
        misaligned = Forecast(
            values={"EQUITY.ASX.WOW": 0.031},
            target="expected_return",
            decision_time=other_origin,
            produced_by="synthetic-mean-forecaster",
        )
        with pytest.raises(InvalidChronologyError):
            decision_context(forecast=misaligned)


def test_decision_context_admits_boundary_and_cross_zone_alignment() -> None:
    from portlearn.interfaces import Forecast

    # MELBOURNE_EQUAL_INSTANT is the same instant as DECISION_INSTANT in
    # another zone: law (d) holds by instant, not by wall clock.
    forecast = Forecast(
        values={"EQUITY.ASX.WOW": 0.031},
        target="expected_return",
        decision_time=MELBOURNE_EQUAL_INSTANT,
        produced_by="synthetic-mean-forecaster",
    )
    context = decision_context(
        current_weights_as_of=DECISION_INSTANT, forecast=forecast
    )
    assert context.forecast is forecast
    assert context.strategy_state is None
    assert context.universe == ("EQUITY.ASX.WOW", "EQUITY.ASX.CBA")


def test_decision_context_current_weights_must_be_pre_trade() -> None:
    for wrong_role in (WeightState.TARGET, WeightState.POST_TRADE):
        with pytest.raises(ValueError):
            decision_context(
                current_weights=PortfolioWeights(
                    dict(WEIGHTS_EQUAL_PAIR), wrong_role
                )
            )


def test_decision_context_universe_is_exact_unique_ordered() -> None:
    ordered = ("EQUITY.ASX.CBA", "EQUITY.ASX.WOW", "EQUITY.ASX.TLS")
    context = decision_context(universe=list(ordered))
    assert context.universe == ordered
    assert isinstance(context.universe, tuple)

    with pytest.raises(ValueError):
        decision_context(universe=("EQUITY.ASX.WOW", "EQUITY.ASX.WOW"))
    with pytest.raises(ValueError):
        decision_context(universe=("EQUITY.ASX.WOW", "   "))
    with pytest.raises(ValueError):
        decision_context(universe=(42, "EQUITY.ASX.WOW"))
    with pytest.raises(ValueError):
        decision_context(universe="EQUITY.ASX.WOW")


def test_decision_context_universe_and_holdings_need_not_match() -> None:
    # Universe-vs-holdings law: held CBA outside the investable
    # universe and newly eligible TLS inside it are both lawful.
    context = decision_context(
        universe=("EQUITY.ASX.WOW", "EQUITY.ASX.TLS"),
        current_weights=PortfolioWeights(
            dict(WEIGHTS_EQUAL_PAIR), WeightState.PRE_TRADE
        ),
    )
    assert "EQUITY.ASX.CBA" in context.current_weights.weights
    assert "EQUITY.ASX.CBA" not in context.universe


def test_decision_context_is_frozen_and_stores_state_as_given() -> None:
    carried = {"step": 3}
    context = decision_context(strategy_state=carried)
    assert context.strategy_state is carried
    with pytest.raises(FrozenInstanceError):
        context.universe = ("EQUITY.ASX.WOW",)


def test_require_decision_result_compatible_rejects_unanchored() -> None:
    from portlearn.interfaces import (
        DecisionResult,
        PortfolioDecision,
        require_decision_result_compatible,
    )

    early = PortfolioDecision(
        decision_time=BEFORE_DECISION,
        execution_time=DECISION_INSTANT,
        target_weights=dict(WEIGHTS_EQUAL_PAIR),
    )
    with pytest.raises(InvalidChronologyError):
        require_decision_result_compatible(
            decision_context(), DecisionResult(decision=early)
        )

    late = PortfolioDecision(
        decision_time=AFTER_DECISION,
        execution_time=AFTER_DECISION + timedelta(minutes=30),
        target_weights=dict(WEIGHTS_EQUAL_PAIR),
    )
    with pytest.raises(InvalidChronologyError):
        require_decision_result_compatible(
            decision_context(), DecisionResult(decision=late)
        )


def test_require_decision_result_compatible_admits_boundaries() -> None:
    from portlearn.interfaces import (
        DecisionResult,
        PortfolioDecision,
        require_decision_result_compatible,
    )

    for execution_instant in (DECISION_INSTANT, EXECUTION_INSTANT):
        decision = PortfolioDecision(
            decision_time=DECISION_INSTANT,
            execution_time=execution_instant,
            target_weights=dict(WEIGHTS_EQUAL_PAIR),
        )
        assert (
            require_decision_result_compatible(
                decision_context(), DecisionResult(decision=decision)
            )
            is None
        )


def test_stateful_strategy_transitions_only_via_next_state() -> None:
    from portlearn.interfaces import (
        DecisionResult,
        PortfolioDecision,
        require_decision_result_compatible,
    )

    class _SteppingStrategy:
        def decide(self, context):
            state = context.strategy_state
            step = 0 if state is None else state["step"]
            return DecisionResult(
                decision=PortfolioDecision(
                    decision_time=context.decision_time,
                    execution_time=context.decision_time
                    + timedelta(minutes=30),
                    target_weights=dict(WEIGHTS_EQUAL_PAIR),
                ),
                next_strategy_state={"step": step + 1},
            )

    carried = {"step": 3}
    context = decision_context(strategy_state=carried)
    result = _SteppingStrategy().decide(context)
    assert carried == {"step": 3}
    assert context.strategy_state == {"step": 3}
    assert result.next_strategy_state == {"step": 4}
    assert require_decision_result_compatible(context, result) is None
