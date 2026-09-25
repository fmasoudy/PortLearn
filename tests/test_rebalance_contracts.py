"""Behavior-named conformance suite for the rebalancing and drift contract.

These tests freeze the rebalancing and portfolio-drift behavioral
contract: strictly increasing aware-instant rebalance schedules (calendar
generated or explicitly enumerated — the same object either way), holding
plans binding two independently declared instant sequences (rebalance
decisions and accounting), buy-and-hold drift over caller-supplied per-asset
gross growth factors with a unconditional strictly-positive-finite
portfolio-growth denominator, exact-fill execution of a target at its
execution instant with strictly increasing executions across rebalances,
availability checking of factor inputs at the instant of use, and full
immutability of every input and output.

Every expected value is hand-computed on exact cases (see each test); the
arithmetic floors are pinned against the design preflight reference, including
edge-case counterexamples: the total-loss edge ``(0.5, 0.5) × (0, 1.1)
→ (0, 1)``, the leveraged-book denominator rejections ``(2, −1) × (0.1, 2)
→ D = −1.8`` and ``(2, −1) × (0.5, 1) → D = 0``, and the path-identity
domain case where the staged drift fails at its intermediate denominator
while the direct product stays admissible.
"""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from portlearn.calendar import month_end_instant, quarter_end_instant
from portlearn.rebalance import (
    GrowthFactors,
    HoldingPlan,
    RebalanceSchedule,
    drift_weights,
    execute_rebalance,
    month_end_schedule,
    quarter_end_schedule,
    require_factors_available,
    require_valid_rebalances,
)
from portlearn.timing import (
    DecisionTiming,
    FutureInformationError,
    InvalidChronologyError,
    NaiveTimestampError,
    ReturnRealizationPeriod,
)
from portlearn.weights import (
    PortfolioWeights,
    WeightState,
    gross_exposure,
)

TOL = 1e-12
MELB = timezone(timedelta(hours=11))  # a fixed +11:00 zone, no DST fold here


def _timing(decision: datetime, execution: datetime) -> DecisionTiming:
    """A lawful decision timing: realization starts at the execution."""
    return DecisionTiming(
        decision_time=decision,
        execution_time=execution,
        return_realization_period=ReturnRealizationPeriod(
            start_time=execution, end_time=execution + timedelta(hours=1)
        ),
    )


def _post(weights: dict[str, float]) -> PortfolioWeights:
    return PortfolioWeights(weights, WeightState.POST_TRADE)


def _target(weights: dict[str, float]) -> PortfolioWeights:
    return PortfolioWeights(weights, WeightState.TARGET)


# ---------------------------------------------------------------------------
# Root-package exposure
# ---------------------------------------------------------------------------


def test_root_package_lazily_exposes_rebalance_module() -> None:
    """The root package exposes the rebalance module lazily: a fresh import
    registers only the root package, and the attribute resolution loads
    exactly the portlearn.rebalance module."""
    for name in [
        name
        for name in list(sys.modules)
        if name == "portlearn" or name.startswith("portlearn.")
    ]:
        del sys.modules[name]
    import portlearn

    assert "portlearn.rebalance" not in sys.modules, (
        "the rebalance module must be lazy: a bare root import must not "
        "register it"
    )
    resolved = portlearn.rebalance
    import portlearn.rebalance as direct

    assert resolved is direct
    assert direct.__name__ == "portlearn.rebalance"


# ---------------------------------------------------------------------------
# Schedule construction (explicit and calendar routes)
# ---------------------------------------------------------------------------


def test_calendar_generated_month_end_schedule_equals_enumerated_route() -> None:
    """Calendar-generated and explicitly enumerated schedules with identical
    instant sets are the same object and normalize to UTC instants."""
    calendar_route = month_end_schedule(2026, 1, 3, UTC)
    enumerated_route = RebalanceSchedule(
        [
            month_end_instant(2026, 1, UTC),
            month_end_instant(2026, 2, UTC),
            month_end_instant(2026, 3, UTC),
        ]
    )
    assert calendar_route == enumerated_route
    assert calendar_route.instants == enumerated_route.instants
    assert list(calendar_route) == list(enumerated_route.instants)
    assert len(calendar_route) == 3
    assert all(instant.tzinfo is UTC for instant in calendar_route.instants)
    # Year rollover rolls the calendar forward, still equal to enumeration.
    rolling = month_end_schedule(2026, 11, 3, UTC)
    assert rolling.instants == (
        month_end_instant(2026, 11, UTC),
        month_end_instant(2026, 12, UTC),
        month_end_instant(2027, 1, UTC),
    )


def test_calendar_and_enumerated_routes_produce_identical_plan_paths() -> None:
    """Schedules with identical instant sets produce identical plan paths
    regardless of construction route."""
    calendar_route = month_end_schedule(2026, 1, 3, UTC)
    enumerated_route = RebalanceSchedule(
        [month_end_instant(2026, m, UTC) for m in (1, 2, 3)]
    )
    accounting = month_end_schedule(2026, 1, 6, UTC)
    plan_calendar = HoldingPlan(calendar_route, accounting)
    plan_enumerated = HoldingPlan(enumerated_route, accounting)
    assert plan_calendar == plan_enumerated
    assert (
        plan_calendar.rebalance_accounting_instants
        == plan_enumerated.rebalance_accounting_instants
        == calendar_route.instants
    )


def test_quarter_end_schedule_rolls_years_and_matches_enumerated_route() -> None:
    """Quarter-end schedules compose with the period calendar and roll
    across year boundaries exactly as enumeration does."""
    calendar_route = quarter_end_schedule(2026, 3, 3, UTC)
    enumerated_route = RebalanceSchedule(
        [
            quarter_end_instant(2026, 3, UTC),
            quarter_end_instant(2026, 4, UTC),
            quarter_end_instant(2027, 1, UTC),
        ]
    )
    assert calendar_route == enumerated_route
    for bad_quarter in (0, 5, -1, True, "2", 2.0):
        with pytest.raises(ValueError, match="quarter"):
            quarter_end_schedule(2026, bad_quarter, 1, UTC)
    for bad_count in (0, -1, True, "3", 2.5):
        with pytest.raises(ValueError, match="count"):
            month_end_schedule(2026, 1, bad_count, UTC)


def test_schedule_normalizes_zone_expressions_to_utc_instants() -> None:
    """A schedule stores UTC-normalized instants: the same true instant
    expressed in another zone is the same schedule entry."""
    melbourne_wall = datetime(2026, 1, 31, 23, 59, 59, 999999, tzinfo=MELB)
    schedule = RebalanceSchedule([melbourne_wall])
    assert schedule.instants == (
        datetime(2026, 1, 31, 12, 59, 59, 999999, tzinfo=UTC),
    )
    assert schedule.instants[0].tzinfo is UTC


def test_single_instant_schedule_is_legal() -> None:
    """A single-instant schedule is lawful: one decision, then drift to the
    horizon end (open and hold)."""
    only = datetime(2026, 1, 31, tzinfo=UTC)
    schedule = RebalanceSchedule([only])
    assert len(schedule) == 1
    assert schedule.instants == (only,)
    assert list(schedule) == [only]


def test_empty_schedule_rejected() -> None:
    """An empty schedule names no decisions and is rejected unconditionally."""
    with pytest.raises(ValueError, match="at least one"):
        RebalanceSchedule([])


def test_schedule_rejects_duplicate_instants() -> None:
    """A schedule must be strictly increasing: duplicates reject with the
    chronology error, including duplicates expressed in different zones."""
    first = datetime(2026, 1, 31, tzinfo=UTC)
    with pytest.raises(InvalidChronologyError):
        RebalanceSchedule([first, first])
    same_instant_other_zone = datetime(2026, 1, 31, 11, 0, tzinfo=MELB)
    assert same_instant_other_zone == first  # same true instant
    with pytest.raises(InvalidChronologyError):
        RebalanceSchedule([first, same_instant_other_zone])


def test_schedule_rejects_non_increasing_instants() -> None:
    """A schedule that reverses order is malformed and rejects."""
    with pytest.raises(InvalidChronologyError):
        RebalanceSchedule(
            [datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)]
        )


def test_schedule_rejects_naive_or_non_instant_entries() -> None:
    """Naive datetimes, bare dates, and non-instant entries all reject with
    the fixed naive-timestamp error; no default zone is ever assumed."""
    with pytest.raises(NaiveTimestampError):
        RebalanceSchedule([datetime(2026, 1, 1)])  # noqa: DTZ001 — deliberately naive
    with pytest.raises(NaiveTimestampError):
        RebalanceSchedule(
            [datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2)]  # noqa: DTZ001
        )
    with pytest.raises(NaiveTimestampError):
        RebalanceSchedule(["2026-01-01T00:00:00+00:00"])
    with pytest.raises(NaiveTimestampError):
        RebalanceSchedule([42])
    with pytest.raises(ValueError):
        RebalanceSchedule(42)  # not even iterable


# ---------------------------------------------------------------------------
# The drift law over growth factors
# ---------------------------------------------------------------------------


def test_flat_growth_factors_return_realized_weights_exactly() -> None:
    """Flat factors g = 1 return the prior weights exactly, including sign
    structure and explicit zero weights."""
    for book in (
        {"A": 0.5, "B": 0.5},
        {"A": 0.25, "B": 0.75},
        {"A": 0.3, "B": 0.7},
        {"A": 0.1, "B": 0.2, "C": 0.7},
        {"SHORT": -0.3, "LONG": 1.3},
        {"A": 0.5, "ZERO": 0.0, "B": 0.5},
    ):
        drifted = drift_weights(_post(book), {key: 1.0 for key in book})
        assert dict(drifted.weights) == book
        assert drifted.assets == frozenset(book)


def test_total_loss_asset_drifts_to_exact_zero_weight() -> None:
    """An individual g_i = 0 on a held asset is lawful arithmetic: the
    asset's weight becomes exactly 0 and the book renormalizes to (0, 1)."""
    drifted = drift_weights(
        _post({"A": 0.5, "B": 0.5}), {"A": 0.0, "B": 1.1}
    )
    assert dict(drifted.weights) == {"A": 0.0, "B": 1.0}


def test_zero_factor_on_a_short_position_normalizes_to_positive_zero() -> None:
    """A short wiped out by g_i = 0 maps to +0.0, never a signed -0.0."""
    drifted = drift_weights(
        _post({"S": -0.2, "A": 0.5, "B": 0.7}),
        {"S": 0.0, "A": 1.1, "B": 1.0},
    )
    assert drifted.weight_of("S") == 0.0
    assert math.copysign(1.0, drifted.weight_of("S")) > 0
    assert abs(math.fsum(drifted.weights.values()) - 1.0) < TOL


def test_positive_factors_preserve_prior_weight_signs() -> None:
    """Under a strictly positive denominator, every asset with g_i > 0
    preserves the sign of its prior weight and the budget identity holds."""
    drifted = drift_weights(
        _post({"S": -0.3, "A": 0.7, "B": 0.6}),
        {"S": 1.2, "A": 0.9, "B": 1.0},
    )
    assert drifted.weight_of("S") < 0
    assert drifted.weight_of("A") > 0
    assert drifted.weight_of("B") > 0
    assert abs(math.fsum(drifted.weights.values()) - 1.0) < TOL


def test_negative_denominator_fails_closed() -> None:
    """A leveraged long/short book whose portfolio growth denominator is
    negative has an undefined state and fails closed — never a
    sign-flipped result: (2, -1) x (0.1, 2) -> D = -1.8."""
    with pytest.raises(ValueError, match="denominator"):
        drift_weights(_post({"L": 2.0, "S": -1.0}), {"L": 0.1, "S": 2.0})


def test_zero_denominator_fails_closed() -> None:
    """A zero denominator (exact long/short growth cancellation, or an empty
    book) is undefined and fails closed: (2, -1) x (0.5, 1) -> D = 0."""
    with pytest.raises(ValueError, match="denominator"):
        drift_weights(_post({"L": 2.0, "S": -1.0}), {"L": 0.5, "S": 1.0})
    with pytest.raises(ValueError, match="denominator"):
        drift_weights(_post({}), {})


def test_negative_growth_factor_rejected() -> None:
    """A negative gross growth factor is malformed (g_i >= 0) and rejects."""
    with pytest.raises(ValueError, match="nonnegative"):
        drift_weights(_post({"A": 0.5, "B": 0.5}), {"A": -1.1, "B": 1.0})


def test_non_finite_growth_factors_rejected() -> None:
    """Infinite and NaN growth factors reject unconditionally."""
    for bad in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError, match="finite"):
            drift_weights(_post({"A": 0.5, "B": 0.5}), {"A": bad, "B": 1.0})


def test_huge_int_growth_factor_fails_closed_without_overflow_leak() -> None:
    """A hostile huge-int factor beyond the float range rejects with
    ValueError — no OverflowError ever leaks."""
    with pytest.raises(ValueError, match="finite"):
        drift_weights(
            _post({"A": 0.5, "B": 0.5}), {"A": 10**400, "B": 1.0}
        )


def test_term_overflow_to_infinity_fails_closed() -> None:
    """Individually finite inputs whose product terms overflow to infinity
    reject unconditionally before any aggregation."""
    with pytest.raises(ValueError, match="finite"):
        drift_weights(
            _post({"A": 1e155, "B": 1.0 - 1e155}),
            {"A": 1e155, "B": 1e155},
        )


def test_denominator_sum_overflow_fails_closed() -> None:
    """Finite terms whose exact denominator sum overflows the float range
    reject with ValueError, never an OverflowError from the aggregation."""
    with pytest.raises(ValueError, match="finite"):
        drift_weights(
            _post({"A": 1.5, "B": 1.5}),
            {"A": 1e308, "B": 1e308},
        )


def test_huge_proportional_factors_are_lawful_arithmetic() -> None:
    """Huge-but-representable proportional factors are admissible: the
    ratio is exact and the budget identity holds."""
    drifted = drift_weights(
        _post({"A": 0.6, "B": 0.4}), {"A": 1.7e308, "B": 1.7e308}
    )
    assert abs(drifted.weight_of("A") - 0.6) < TOL
    assert abs(drifted.weight_of("B") - 0.4) < TOL
    assert abs(math.fsum(drifted.weights.values()) - 1.0) < TOL


def test_growth_factor_universe_must_match_weights_exactly() -> None:
    """The universe is fixed over the holding interval: factors naming
    absent assets or omitting held assets both reject."""
    book = _post({"A": 0.6, "B": 0.4})
    with pytest.raises(ValueError, match="universe"):
        drift_weights(book, {"A": 1.1})  # omits B
    with pytest.raises(ValueError, match="universe"):
        drift_weights(book, {"A": 1.1, "B": 1.0, "C": 1.0})  # names C


def test_bool_and_non_real_growth_factors_rejected() -> None:
    """A bool is not a factor and a non-real is not a factor; both reject."""
    with pytest.raises(ValueError, match="bool"):
        drift_weights(_post({"A": 0.5, "B": 0.5}), {"A": True, "B": 1.0})
    with pytest.raises(ValueError, match="real"):
        drift_weights(_post({"A": 0.5, "B": 0.5}), {"A": "1.1", "B": 1.0})
    with pytest.raises(ValueError, match="real"):
        drift_weights(_post({"A": 0.5, "B": 0.5}), {"A": None, "B": 1.0})


def test_drift_rejects_target_state_input() -> None:
    """Drift applies to realized states only: a TARGET is a desire, not a
    portfolio that can drift; a non-weights input is a shape error."""
    with pytest.raises(ValueError, match="TARGET"):
        drift_weights(_target({"A": 0.6, "B": 0.4}), {"A": 1.0, "B": 1.0})
    with pytest.raises(ValueError, match="PortfolioWeights"):
        drift_weights({"A": 0.6, "B": 0.4}, {"A": 1.0, "B": 1.0})
    with pytest.raises(ValueError, match="mapping"):
        drift_weights(_post({"A": 0.6, "B": 0.4}), [("A", 1.0), ("B", 1.0)])


def test_drift_returns_fresh_pre_trade_object_never_mutating_inputs() -> None:
    """Drift outputs a fresh PRE_TRADE value object; inputs are never
    mutated and later source mutation cannot corrupt anything."""
    source_weights = {"A": 0.6, "B": 0.4}
    source_factors = {"A": 1.1, "B": 0.95}
    post = _post(source_weights)
    drifted = drift_weights(post, source_factors)
    assert drifted is not post
    assert drifted.state is WeightState.PRE_TRADE
    assert post.state is WeightState.POST_TRADE
    assert dict(post.weights) == {"A": 0.6, "B": 0.4}
    source_weights["A"] = 99.0
    source_factors["A"] = 99.0
    assert dict(post.weights) == {"A": 0.6, "B": 0.4}
    assert abs(drifted.weight_of("A") - 0.6 * 1.1 / (0.6 * 1.1 + 0.4 * 0.95)) < TOL
    with pytest.raises(TypeError):
        drifted.weights["A"] = 0.5  # type: ignore[index]


# ---------------------------------------------------------------------------
# Path identity and composition
# ---------------------------------------------------------------------------


def test_staged_drift_equals_direct_product_within_tolerance() -> None:
    """Drifting in two stages equals drifting once over the elementwise
    product, wherever both paths are admissible."""
    post = _post({"A": 0.6, "B": 0.4})
    staged = drift_weights(
        drift_weights(post, {"A": 1.1, "B": 0.95}), {"A": 1.05, "B": 1.2}
    )
    direct = drift_weights(post, {"A": 1.1 * 1.05, "B": 0.95 * 1.2})
    for asset in ("A", "B"):
        assert abs(staged.weight_of(asset) - direct.weight_of(asset)) < TOL


def test_direct_product_is_elementwise_order_independent() -> None:
    """The direct product of growth factors is order-independent: applying
    g1 then g2 product-wise equals applying g2 then g1 product-wise."""
    post = _post({"A": 0.6, "B": 0.4})
    forward = drift_weights(post, {"A": 1.1 * 1.05, "B": 0.95 * 1.2})
    backward = drift_weights(post, {"A": 1.05 * 1.1, "B": 1.2 * 0.95})
    for asset in ("A", "B"):
        assert abs(forward.weight_of(asset) - backward.weight_of(asset)) < TOL


def test_staged_rejection_is_not_an_equality_claim() -> None:
    """Where the staged path fails at its intermediate denominator, the
    rejection stands even though the direct product is admissible: equality
    is never claimed against a path the law requires to reject."""
    book = _post({"L": 2.0, "S": -1.0})
    with pytest.raises(ValueError, match="denominator"):
        drift_weights(drift_weights(book, {"L": 0.1, "S": 2.0}), {"L": 12.0, "S": 1.0})
    direct = drift_weights(book, {"L": 0.1 * 12.0, "S": 2.0 * 1.0})
    assert abs(math.fsum(direct.weights.values()) - 1.0) < TOL


# ---------------------------------------------------------------------------
# Decision/execution separation and execution sequences
# ---------------------------------------------------------------------------


def test_post_trade_state_lives_at_delayed_execution_instant() -> None:
    """With execution after the decision, the executed portfolio exists at
    the execution instant as a POST_TRADE state equal to the target."""
    decision = datetime(2026, 1, 31, 23, 59, 59, 999999, tzinfo=UTC)
    execution = datetime(2026, 2, 2, 16, 0, tzinfo=UTC)
    target = _target({"A": 0.6, "B": 0.4})
    executed = execute_rebalance(target, _timing(decision, execution))
    assert executed.state is WeightState.POST_TRADE
    assert dict(executed.weights) == {"A": 0.6, "B": 0.4}


def test_pre_trade_drift_runs_to_the_execution_instant() -> None:
    """The portfolio drifts until the moment of execution: the state just
    before execution is the drift of the last post-trade state, and
    executing transforms it exactly into the new target."""
    first_execution = datetime(2026, 1, 31, 23, 59, 59, 999999, tzinfo=UTC)
    second_decision = datetime(2026, 2, 28, 23, 59, 59, 999999, tzinfo=UTC)
    second_execution = datetime(2026, 3, 2, 16, 0, tzinfo=UTC)
    post = execute_rebalance(
        _target({"A": 0.5, "B": 0.5}), _timing(first_execution, first_execution)
    )
    drifted_to_execution = drift_weights(post, {"A": 1.2, "B": 0.9})
    assert drifted_to_execution.state is WeightState.PRE_TRADE
    assert drifted_to_execution.weight_of("A") > 0.5  # relative growth moved it
    executed = execute_rebalance(
        _target({"A": 0.4, "B": 0.6}), _timing(second_decision, second_execution)
    )
    assert executed.state is WeightState.POST_TRADE
    assert dict(executed.weights) == {"A": 0.4, "B": 0.6}


def test_same_instant_decide_and_execute_is_admissible() -> None:
    """Deciding and executing at one instant is the admissible special
    case, alone and inside a strictly increasing execution sequence."""
    instant = datetime(2026, 1, 31, tzinfo=UTC)
    executed = execute_rebalance(_target({"A": 1.0}), _timing(instant, instant))
    assert executed.state is WeightState.POST_TRADE
    later = instant + timedelta(days=28)
    assert (
        require_valid_rebalances(
            [_timing(instant, instant), _timing(later, later)]
        )
        is None
    )


def test_execution_before_decision_rejected() -> None:
    """A trade cannot execute before its decision: the fixed timing object
    rejects it, and so do the execution surfaces standing alone."""
    decision = datetime(2026, 2, 1, tzinfo=UTC)
    execution = datetime(2026, 1, 31, tzinfo=UTC)
    with pytest.raises(InvalidChronologyError):
        _timing(decision, execution)
    with pytest.raises(InvalidChronologyError):
        execute_rebalance(
            _target({"A": 1.0}),
            SimpleNamespace(decision_time=decision, execution_time=execution),
        )
    with pytest.raises(InvalidChronologyError):
        require_valid_rebalances(
            [SimpleNamespace(decision_time=decision, execution_time=execution)]
        )


def test_duplicate_execution_instant_across_rebalances_rejected() -> None:
    """Two distinct rebalances cannot share one execution instant: that
    would create two post-trade states at one instant with no ordering
    semantics. Distinct increasing executions are accepted, including
    delayed ones, and an empty sequence is vacuously valid."""
    first_decision = datetime(2026, 1, 31, tzinfo=UTC)
    second_decision = datetime(2026, 2, 28, tzinfo=UTC)
    shared_execution = datetime(2026, 3, 1, tzinfo=UTC)
    with pytest.raises(InvalidChronologyError, match="strictly increasing"):
        require_valid_rebalances(
            [
                _timing(first_decision, shared_execution),
                _timing(second_decision, shared_execution),
            ]
        )
    assert (
        require_valid_rebalances(
            [
                _timing(first_decision, datetime(2026, 2, 2, tzinfo=UTC)),
                _timing(second_decision, datetime(2026, 3, 2, tzinfo=UTC)),
            ]
        )
        is None
    )
    assert require_valid_rebalances([]) is None


def test_executed_portfolio_equals_target_exactly_under_exact_fill() -> None:
    """Under exact fill the post-trade weights equal the target exactly, as
    a fresh object, with the target itself left untouched; executing a
    non-target state is a category error."""
    target = _target({"A": 0.25, "B": 0.75})
    timing = _timing(
        datetime(2026, 1, 31, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    )
    executed = execute_rebalance(target, timing)
    assert executed is not target
    assert dict(executed.weights) == dict(target.weights)
    assert executed.assets == target.assets
    assert target.state is WeightState.TARGET  # untouched
    with pytest.raises(ValueError, match="TARGET"):
        execute_rebalance(_post({"A": 0.5, "B": 0.5}), timing)
    with pytest.raises(ValueError, match="decision_time"):
        execute_rebalance(target, object())


# ---------------------------------------------------------------------------
# Holding plans: independently declared sequences
# ---------------------------------------------------------------------------


def test_sparser_rebalance_than_accounting_forms_no_targets_between_rebalances() -> None:
    """A daily-accounting plan with one mid-window rebalance records drift
    states on the other accounting instants and forms no targets there."""
    accounting = tuple(
        datetime(2026, 3, day, tzinfo=UTC) for day in range(1, 11)
    )
    schedule = RebalanceSchedule([datetime(2026, 3, 5, tzinfo=UTC)])
    plan = HoldingPlan(schedule, accounting)
    assert plan.rebalance_accounting_instants == (accounting[4],)
    non_decision = [
        instant for instant in accounting if not plan.is_decision_instant(instant)
    ]
    assert len(non_decision) == 9
    for instant in non_decision:
        assert plan.is_accounting_instant(instant)
        assert not plan.is_decision_instant(instant)
    assert not plan.is_decision_instant(datetime(2026, 3, 5, 12, tzinfo=UTC))
    assert not plan.is_accounting_instant(datetime(2026, 3, 11, tzinfo=UTC))


def test_cross_frequency_pairings_are_independently_representable() -> None:
    """Every cross-frequency pairing is first-class: daily+daily,
    daily+yearly-style, monthly+monthly, monthly+quarterly, and monthly
    accounting with a single rebalance are all declared, none inferred."""
    daily = tuple(datetime(2026, 3, d, tzinfo=UTC) for d in range(1, 11))
    # daily accounting + rebalance every accounting instant
    identical = HoldingPlan(RebalanceSchedule(daily), daily)
    assert identical.rebalance_accounting_instants == daily
    # daily accounting + one sparse rebalance
    sparse = HoldingPlan(RebalanceSchedule([daily[4]]), daily)
    assert sparse.rebalance_accounting_instants == (daily[4],)
    # monthly accounting + one yearly-style rebalance
    monthly = month_end_schedule(2026, 1, 12, UTC)
    yearly_style = HoldingPlan(RebalanceSchedule([monthly.instants[5]]), monthly)
    assert yearly_style.rebalance_accounting_instants == (monthly.instants[5],)
    # monthly accounting + monthly rebalance (identical sequences)
    monthly_identical = HoldingPlan(monthly, monthly)
    assert monthly_identical.rebalance_accounting_instants == monthly.instants
    # monthly accounting + quarterly rebalance
    quarterly = quarter_end_schedule(2026, 1, 4, UTC)
    mixed = HoldingPlan(quarterly, monthly)
    assert mixed.rebalance_accounting_instants == quarterly.instants


def test_plan_with_identical_sequences_rebalances_at_every_accounting_instant() -> None:
    """Declaring the sequences identical rebalances at every accounting
    instant: every accounting instant is a decision instant."""
    monthly = month_end_schedule(2026, 1, 4, UTC)
    plan = HoldingPlan(monthly, monthly)
    assert plan.rebalance_accounting_instants == plan.accounting_instants
    assert all(plan.is_decision_instant(i) for i in plan.accounting_instants)
    assert plan.decision_instants == monthly.instants


def test_plan_allows_decision_instants_outside_the_accounting_sequence() -> None:
    """Neither sequence is inferred from the other: a plan whose rebalance
    decision instants never coincide with an accounting instant is
    representable and simply records no rebalance states."""
    accounting = month_end_schedule(2026, 1, 3, UTC)
    disjoint = RebalanceSchedule(
        [
            month_end_instant(2026, 1, UTC) - timedelta(days=1),
            month_end_instant(2026, 2, UTC) - timedelta(days=1),
        ]
    )
    plan = HoldingPlan(disjoint, accounting)
    assert plan.rebalance_accounting_instants == ()
    assert plan.decision_instants == disjoint.instants


def test_plan_rejects_naive_and_non_monotone_accounting_instants() -> None:
    """The accounting sequence obeys the same instant laws as the schedule:
    naive entries reject, duplicates and reversals reject, and empty
    rejects; so does a schedule that is not a schedule object."""
    schedule = RebalanceSchedule([datetime(2026, 1, 31, tzinfo=UTC)])
    with pytest.raises(NaiveTimestampError):
        HoldingPlan(schedule, [datetime(2026, 1, 31)])  # noqa: DTZ001 — naive probe
    with pytest.raises(InvalidChronologyError):
        HoldingPlan(
            schedule,
            [
                datetime(2026, 1, 31, tzinfo=UTC),
                datetime(2026, 1, 31, tzinfo=UTC),
            ],
        )
    with pytest.raises(InvalidChronologyError):
        HoldingPlan(
            schedule,
            [
                datetime(2026, 2, 28, tzinfo=UTC),
                datetime(2026, 1, 31, tzinfo=UTC),
            ],
        )
    with pytest.raises(ValueError, match="at least one"):
        HoldingPlan(schedule, [])
    with pytest.raises(ValueError, match="RebalanceSchedule"):
        HoldingPlan(
            [datetime(2026, 1, 31, tzinfo=UTC)],  # type: ignore[arg-type]
            [datetime(2026, 1, 31, tzinfo=UTC)],
        )
    with pytest.raises(NaiveTimestampError):
        schedule.is_decision_instant(datetime(2026, 1, 31))  # noqa: DTZ001 — naive


def test_no_rebalance_accounting_state_equals_drift_of_last_post_trade_state() -> None:
    """On accounting instants that are not decision instants the recorded
    state is the drift of the last post-trade state — a PRE_TRADE object,
    composable exactly as the staged path — and no constraint validation
    occurs: drift may violate declared ceilings, that is its economic
    content."""
    post = execute_rebalance(
        _target({"A": -0.2, "B": 1.2}),
        _timing(
            datetime(2026, 1, 31, tzinfo=UTC), datetime(2026, 1, 31, tzinfo=UTC)
        ),
    )
    first = drift_weights(post, {"A": 1.5, "B": 1.0})
    second = drift_weights(first, {"A": 1.1, "B": 1.0})
    assert first.state is WeightState.PRE_TRADE
    assert second.state is WeightState.PRE_TRADE
    direct = drift_weights(post, {"A": 1.5 * 1.1, "B": 1.0})
    for asset in ("A", "B"):
        assert abs(second.weight_of(asset) - direct.weight_of(asset)) < TOL
    gross_ceiling = 1.2
    drifted_gross = gross_exposure(second.weights)
    assert drifted_gross > gross_ceiling  # the ceiling is violated
    # ... and drift raised nothing: no-rebalance states are never validated


def test_schedule_and_plan_objects_are_immutable() -> None:
    """Schedules and plans are fixed declarations: attribute assignment
    rejects, and mutating the source iterables after construction changes
    nothing."""
    instants = [datetime(2026, 1, 31, tzinfo=UTC)]
    schedule = RebalanceSchedule(instants)
    accounting = [datetime(2026, 1, 31, tzinfo=UTC)]
    plan = HoldingPlan(schedule, accounting)
    with pytest.raises(AttributeError):
        schedule.instants = ()  # type: ignore[misc]
    with pytest.raises(AttributeError):
        plan.accounting_instants = ()  # type: ignore[misc]
    instants.append(datetime(2026, 3, 31, tzinfo=UTC))
    accounting.append(datetime(2026, 3, 31, tzinfo=UTC))
    assert schedule.instants == (datetime(2026, 1, 31, tzinfo=UTC),)
    assert plan.accounting_instants == (datetime(2026, 1, 31, tzinfo=UTC),)


# ---------------------------------------------------------------------------
# Availability checking of factor inputs
# ---------------------------------------------------------------------------


def test_factors_available_after_decision_rejected_as_future_information() -> None:
    """Growth factors are information items: consuming at a decision instant
    factors that first could have been known later is look-ahead and
    rejects, including arrival inside the decision-to-execution gap."""
    decision = datetime(2026, 1, 31, tzinfo=UTC)
    execution = datetime(2026, 2, 2, tzinfo=UTC)
    late = GrowthFactors(
        {"A": 1.1, "B": 0.95}, datetime(2026, 2, 1, tzinfo=UTC)
    )
    with pytest.raises(FutureInformationError):
        require_factors_available(late, decision)
    in_gap = GrowthFactors(
        {"A": 1.1, "B": 0.95}, datetime(2026, 2, 1, 12, tzinfo=UTC)
    )
    with pytest.raises(FutureInformationError):
        require_factors_available(in_gap, decision)
    assert in_gap.available_time < execution  # genuinely inside the gap
    with pytest.raises(NaiveTimestampError):
        require_factors_available(late, datetime(2026, 1, 31))  # noqa: DTZ001 — naive check


def test_factors_available_at_or_before_decision_admitted() -> None:
    """Factors available exactly at the decision instant are admitted
    (availability at the decision admits), as are earlier ones."""
    decision = datetime(2026, 1, 31, tzinfo=UTC)
    exactly_at = GrowthFactors(
        {"A": 1.1, "B": 0.95}, datetime(2026, 1, 31, tzinfo=UTC)
    )
    earlier = GrowthFactors(
        {"A": 1.1, "B": 0.95}, datetime(2026, 1, 30, tzinfo=UTC)
    )
    assert require_factors_available(exactly_at, decision) is None
    assert require_factors_available(earlier, decision) is None


def test_factor_provenance_objects_validate_and_snapshot_fail_closed() -> None:
    """Factor provenance objects validate their mapping unconditional (blank
    identifiers, bools, non-reals, negative, non-finite factors), require
    an aware availability instant, and snapshot immutably."""
    with pytest.raises(ValueError, match="identifier"):
        GrowthFactors({" ": 1.0}, datetime(2026, 1, 30, tzinfo=UTC))
    for bad in (True, "1.1", None, -0.5, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            GrowthFactors({"A": bad}, datetime(2026, 1, 30, tzinfo=UTC))
    with pytest.raises(NaiveTimestampError):
        GrowthFactors({"A": 1.0}, datetime(2026, 1, 30))  # noqa: DTZ001 — naive
    source = {"A": 1.1}
    factors = GrowthFactors(source, datetime(2026, 1, 30, tzinfo=UTC))
    source["A"] = 99.0
    source["EVIL"] = -5.0
    assert dict(factors.factors) == {"A": 1.1}
    with pytest.raises(TypeError):
        factors.factors["A"] = 1.0  # type: ignore[index]
    with pytest.raises(AttributeError):
        factors.available_time = datetime(2026, 1, 1, tzinfo=UTC)  # type: ignore[misc]
    assert factors.available_time == datetime(2026, 1, 30, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Availability at the instant of consumption (the use instant)
# ---------------------------------------------------------------------------


def test_factors_consumed_before_availability_reject_at_the_use_instant() -> None:
    """Growth factors are admitted at the instant of consumption: the
    same factors first available 2026-02-01 UTC reject as look-ahead when
    consumed at the earlier use instant 2026-01-31 UTC (the use instant of
    factors forming a target at a decision). The keyword ``use_time``
    names the check."""
    factors = GrowthFactors({"A": 1.1, "B": 0.95}, datetime(2026, 2, 1, tzinfo=UTC))
    with pytest.raises(FutureInformationError):
        require_factors_available(
            factors, use_time=datetime(2026, 1, 31, tzinfo=UTC)
        )


def test_same_factors_admitted_at_a_later_use_instant() -> None:
    """The identical factor object consumed at a later use instant admits:
    realized growth through a later execution legitimately contains
    information unavailable at the decision, so factors first available
    2026-02-01 UTC consumed at use instant 2026-02-02 UTC (pre-trade
    drift at the execution) pass the check and return None."""
    factors = GrowthFactors({"A": 1.1, "B": 0.95}, datetime(2026, 2, 1, tzinfo=UTC))
    assert (
        require_factors_available(
            factors, use_time=datetime(2026, 2, 2, tzinfo=UTC)
        )
        is None
    )
