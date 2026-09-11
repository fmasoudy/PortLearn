"""Timing-surface contract tests for portlearn.

These tests freeze the documented timing semantics: timezone-aware
instants only (no naive datetimes, no date-to-midnight coercion), the
five-time chronology law with equal-adjacent boundaries admissible, the
inclusive decision-time-only admission law, and the typed fail-closed
error taxonomy owned by ``portlearn.timing``.  Sections below cover the
valid-admission matrix (boundary cases, cross-zone instants, selected
vintages, and determinism) and the rejection matrix (malformed instants,
chronology violations, and missing availability).

Malformed-item fixtures built from ``types.SimpleNamespace`` appear only
where fail-closed field validation of ``available_time`` on the
admission pair must be exercised (attribute absence, ``None``, naive
datetime, ``datetime.date``).  Real ``TimedObservation`` instances
cannot represent those malformed states because their own constructor
rejects them; the namespaces are malformed *inputs*, never
implementations.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from portlearn.observations import TimedObservation, vintage_as_of
from portlearn.timing import (
    DecisionTiming,
    FutureInformationError,
    InvalidChronologyError,
    MissingAvailabilityError,
    NaiveTimestampError,
    ReturnRealizationPeriod,
    is_available_for_decision,
    require_available_for_decision,
)

MELBOURNE_SUMMER = timezone(timedelta(hours=11))  # AEDT fixed offset
IST = timezone(timedelta(hours=5, minutes=30))  # half-hour offset zone
MARQUESAS = timezone(timedelta(hours=-9, minutes=-30))  # negative half-hour
NEW_YORK = timezone(timedelta(hours=-5))

SERIES_CLOSE = "PX.XJO.DAILY.CLOSE"
SERIES_MACRO = "MACRO.GDPQ.REAL"

# Reference example A: a month-end decision consuming a
# same-day daily close observation under zero publication lag.
A_OBSERVATION = datetime(2026, 3, 31, 14, 0, tzinfo=UTC)
A_AVAILABLE = datetime(2026, 3, 31, 14, 0, tzinfo=UTC)
A_DECISION = datetime(2026, 3, 31, 23, 0, tzinfo=UTC)
A_EXECUTION = datetime(2026, 4, 1, 0, 30, tzinfo=UTC)

# Reference example B: a lagged macro release consumed at a
# later monthly decision.
B_OBSERVATION = datetime(2026, 3, 16, 9, 0, tzinfo=UTC)
B_AVAILABLE = datetime(2026, 4, 7, 9, 0, tzinfo=UTC)

DECISION_END_OF_APRIL = datetime(2026, 4, 30, 23, 0, tzinfo=UTC)


def close_item(
    available_time: datetime = A_AVAILABLE,
    observation_time: datetime = A_OBSERVATION,
    series_id: str = SERIES_CLOSE,
    value: float = 6543.21,
) -> TimedObservation:
    """A real daily-close observation composed for the timing contracts."""
    return TimedObservation(
        series_id=series_id,
        observation_time=observation_time,
        available_time=available_time,
        value=value,
    )


def valid_period(
    start: datetime, end: datetime
) -> ReturnRealizationPeriod:
    return ReturnRealizationPeriod(start_time=start, end_time=end)


# ---------------------------------------------------------------------------
# Valid-admission matrix (timing surface)
# ---------------------------------------------------------------------------


def test_full_strict_chronology_composite_chain() -> None:
    """All five instants strictly ordered, composite chain asserted."""
    observation = datetime(2026, 3, 31, 10, 0, tzinfo=UTC)
    available = datetime(2026, 3, 31, 12, 0, tzinfo=UTC)
    decision = datetime(2026, 3, 31, 22, 0, tzinfo=UTC)
    execution = datetime(2026, 3, 31, 22, 30, tzinfo=UTC)
    start = datetime(2026, 3, 31, 22, 45, tzinfo=UTC)
    end = datetime(2026, 4, 30, 22, 45, tzinfo=UTC)

    item = close_item(
        observation_time=observation, available_time=available, value=100.0
    )
    timing = DecisionTiming(
        decision_time=decision,
        execution_time=execution,
        return_realization_period=valid_period(start, end),
    )

    assert item.observation_time == observation
    assert item.available_time == available
    assert timing.decision_time == decision
    assert timing.execution_time == execution
    assert timing.return_realization_period.start_time == start
    assert timing.return_realization_period.end_time == end

    # The composite five-instant chain is asserted directly.
    period = timing.return_realization_period
    assert item.observation_time < item.available_time
    assert item.available_time < timing.decision_time
    assert timing.decision_time < timing.execution_time
    assert timing.execution_time < period.start_time
    assert period.start_time < period.end_time

    assert is_available_for_decision(item, decision) is True
    assert require_available_for_decision(item, decision) is None


def test_observation_equals_available_zero_lag_is_admissible() -> None:
    """Zero publication lag is a legitimate explicit design."""
    item = close_item()  # reference example A: same-instant availability
    assert item.observation_time == item.available_time
    assert is_available_for_decision(item, A_DECISION) is True
    assert require_available_for_decision(item, A_DECISION) is None


def test_available_equals_decision_admits_at_the_boundary() -> None:
    """The admission boundary is inclusive: an item available exactly at
    the decision time is admissible."""
    item = close_item(available_time=A_DECISION)
    assert is_available_for_decision(item, A_DECISION) is True
    assert require_available_for_decision(item, A_DECISION) is None


def test_decision_equals_execution_same_instant_trade_is_admissible():
    """Same-instant decide-and-execute is an explicit design."""
    timing = DecisionTiming(
        decision_time=A_DECISION,
        execution_time=A_DECISION,
        return_realization_period=valid_period(
            A_EXECUTION, A_EXECUTION + timedelta(days=30)
        ),
    )
    assert timing.decision_time == timing.execution_time
    item = close_item()
    assert is_available_for_decision(item, timing.decision_time) is True
    assert require_available_for_decision(item, timing.decision_time) is None


def test_execution_equals_realization_start_is_admissible() -> None:
    """Equality at the execution/realization-start boundary."""
    timing = DecisionTiming(
        decision_time=A_DECISION,
        execution_time=A_EXECUTION,
        return_realization_period=valid_period(
            A_EXECUTION, datetime(2026, 5, 1, 0, 30, tzinfo=UTC)
        ),
    )
    assert (
        timing.execution_time == timing.return_realization_period.start_time
    )


ZONE_CASES = [
    pytest.param(UTC, id="utc"),
    pytest.param(MELBOURNE_SUMMER, id="plus-eleven"),
    pytest.param(IST, id="plus-five-thirty"),
    pytest.param(NEW_YORK, id="minus-five"),
    pytest.param(MARQUESAS, id="minus-nine-thirty"),
]


@pytest.mark.parametrize("zone", ZONE_CASES)
def test_same_instant_across_zones_admits_equally(zone: timezone) -> None:
    """Wall clock and zone name never affect ordering or equality."""
    decision_utc = datetime(2026, 3, 31, 23, 0, tzinfo=UTC)
    item = close_item(available_time=decision_utc.astimezone(zone))
    assert item.available_time == decision_utc  # instant equality
    assert is_available_for_decision(item, decision_utc) is True
    # One minute earlier, expressed in a third zone: strict cross-zone
    # ordering must still reject.
    one_minute_earlier = (decision_utc - timedelta(minutes=1)).astimezone(
        MELBOURNE_SUMMER
    )
    assert is_available_for_decision(item, one_minute_earlier) is False


@pytest.mark.parametrize(
    "zone",
    [pytest.param(IST, id="plus-0530"), pytest.param(MARQUESAS, id="minus-0930")],
)
def test_half_hour_offset_zones_compare_as_instants(zone: timezone) -> None:
    """Non-whole-hour stdlib fixed offsets, both signs, stdlib only."""
    base = datetime(2026, 3, 31, 23, 0, tzinfo=UTC)
    shifted = base.astimezone(zone)
    assert shifted.utcoffset() not in (None, timedelta(0)) or zone is UTC
    assert shifted == base  # equality independent of offset sign
    item = close_item(available_time=shifted)
    assert is_available_for_decision(item, base) is True
    assert is_available_for_decision(item, shifted) is True
    earlier = (base - timedelta(minutes=1)).astimezone(MELBOURNE_SUMMER)
    assert is_available_for_decision(item, earlier) is False


def test_selected_vintage_satisfies_the_admission_law() -> None:
    """The selected vintage then admits cleanly."""
    observation = datetime(2026, 2, 28, 9, 0, tzinfo=UTC)
    first = TimedObservation(
        series_id=SERIES_MACRO,
        observation_time=observation,
        available_time=datetime(2026, 3, 2, 9, 0, tzinfo=UTC),
        value=1.0,
    )
    second = TimedObservation(
        series_id=SERIES_MACRO,
        observation_time=observation,
        available_time=datetime(2026, 3, 16, 9, 0, tzinfo=UTC),
        value=2.0,
    )
    third = TimedObservation(
        series_id=SERIES_MACRO,
        observation_time=observation,
        available_time=datetime(2026, 4, 2, 9, 0, tzinfo=UTC),
        value=3.0,
    )
    history = [first, second, third]

    decision = datetime(2026, 3, 20, 9, 0, tzinfo=UTC)
    vintage = vintage_as_of(history, decision)
    assert vintage is not None
    assert vintage.available_time == second.available_time
    assert require_available_for_decision(vintage, decision) is None
    assert is_available_for_decision(vintage, decision) is True

    earlier_decision = datetime(2026, 3, 10, 9, 0, tzinfo=UTC)
    earlier_vintage = vintage_as_of(history, earlier_decision)
    assert earlier_vintage is not None
    assert earlier_vintage.available_time == first.available_time
    assert is_available_for_decision(earlier_vintage, earlier_decision) is True


def test_determinism_verdicts_and_error_messages_are_stable() -> None:
    """Identical inputs yield identical verdicts and messages."""
    item = close_item(available_time=A_DECISION + timedelta(minutes=1))
    verdicts = [is_available_for_decision(item, A_DECISION) for _ in range(5)]
    assert verdicts == [False, False, False, False, False]

    def rejection_message() -> str:
        with pytest.raises(FutureInformationError) as excinfo:
            require_available_for_decision(item, A_DECISION)
        return str(excinfo.value)

    assert rejection_message() == rejection_message()

    def chronology_message() -> str:
        with pytest.raises(InvalidChronologyError) as excinfo:
            DecisionTiming(
                decision_time=A_DECISION,
                execution_time=A_DECISION - timedelta(minutes=1),
                return_realization_period=valid_period(
                    A_EXECUTION, A_EXECUTION + timedelta(days=30)
                ),
            )
        return str(excinfo.value)

    assert chronology_message() == chronology_message()

    def naive_message() -> str:
        with pytest.raises(NaiveTimestampError) as excinfo:
            valid_period(datetime(2026, 4, 1), A_EXECUTION + timedelta(days=30))  # noqa: DTZ001  # intentional naive instant
        return str(excinfo.value)

    assert naive_message() == naive_message()


# ---------------------------------------------------------------------------
# Rejection matrix (timing surface)
# ---------------------------------------------------------------------------

NAIVE_INSTANT = datetime(2026, 3, 31, 10, 0)  # no tzinfo — never admissible  # noqa: DTZ001  # intentional naive instant


def _valid_period_for_r_cases() -> ReturnRealizationPeriod:
    return valid_period(A_EXECUTION, A_EXECUTION + timedelta(days=30))


NAIVE_SURFACES = [
    pytest.param(
        lambda: ReturnRealizationPeriod(
            start_time=NAIVE_INSTANT,
            end_time=A_EXECUTION + timedelta(days=30),
        ),
        id="realization-start",
    ),
    pytest.param(
        lambda: ReturnRealizationPeriod(
            start_time=A_EXECUTION, end_time=NAIVE_INSTANT
        ),
        id="realization-end",
    ),
    pytest.param(
        lambda: DecisionTiming(
            decision_time=NAIVE_INSTANT,
            execution_time=A_EXECUTION,
            return_realization_period=_valid_period_for_r_cases(),
        ),
        id="decision-time",
    ),
    pytest.param(
        lambda: DecisionTiming(
            decision_time=A_DECISION,
            execution_time=NAIVE_INSTANT,
            return_realization_period=_valid_period_for_r_cases(),
        ),
        id="execution-time",
    ),
    pytest.param(
        lambda: is_available_for_decision(close_item(), NAIVE_INSTANT),
        id="admission-decision-time",
    ),
    pytest.param(
        lambda: is_available_for_decision(
            SimpleNamespace(
                series_id=SERIES_CLOSE,
                observation_time=A_OBSERVATION,
                available_time=NAIVE_INSTANT,
                value=1.0,
            ),
            A_DECISION,
        ),
        id="admission-item-available-time",
    ),
]


@pytest.mark.parametrize("malformed", NAIVE_SURFACES)
def test_naive_datetimes_are_rejected_on_every_timing_surface(
    malformed: object,
) -> None:
    """No default timezone is ever assumed."""
    with pytest.raises(NaiveTimestampError):
        malformed()  # type: ignore[operator]


def test_naive_timestamp_message_explains_the_offset_requirement() -> None:
    """The message must explain the explicit-UTC-offset requirement."""
    with pytest.raises(NaiveTimestampError) as excinfo:
        valid_period(datetime(2026, 4, 1), A_EXECUTION + timedelta(days=30))  # noqa: DTZ001  # intentional naive instant
    assert "offset" in str(excinfo.value).lower()


DATE_SURFACES = [
    pytest.param(
        lambda: ReturnRealizationPeriod(
            start_time=date(2026, 4, 1),
            end_time=A_EXECUTION + timedelta(days=30),
        ),
        id="realization-start",
    ),
    pytest.param(
        lambda: ReturnRealizationPeriod(
            start_time=A_EXECUTION, end_time=date(2026, 5, 1)
        ),
        id="realization-end",
    ),
    pytest.param(
        lambda: DecisionTiming(
            decision_time=date(2026, 3, 31),
            execution_time=A_EXECUTION,
            return_realization_period=_valid_period_for_r_cases(),
        ),
        id="decision-time",
    ),
    pytest.param(
        lambda: DecisionTiming(
            decision_time=A_DECISION,
            execution_time=date(2026, 4, 1),
            return_realization_period=_valid_period_for_r_cases(),
        ),
        id="execution-time",
    ),
    pytest.param(
        lambda: is_available_for_decision(close_item(), date(2026, 3, 31)),
        id="admission-decision-time",
    ),
    pytest.param(
        lambda: is_available_for_decision(
            SimpleNamespace(
                series_id=SERIES_CLOSE,
                observation_time=A_OBSERVATION,
                available_time=date(2026, 3, 31),
                value=1.0,
            ),
            A_DECISION,
        ),
        id="admission-item-available-time",
    ),
]


@pytest.mark.parametrize("malformed", DATE_SURFACES)
def test_date_inputs_are_rejected_without_midnight_coercion(
    malformed: object,
) -> None:
    """``datetime.date`` never becomes that date's midnight."""
    with pytest.raises(NaiveTimestampError):
        malformed()  # type: ignore[operator]


def test_available_before_observation_is_rejected() -> None:
    """Availability may not precede the observation it describes."""
    with pytest.raises(InvalidChronologyError) as excinfo:
        close_item(observation_time=B_AVAILABLE, available_time=B_OBSERVATION)
    message = str(excinfo.value)
    assert SERIES_CLOSE in message  # names the offending series_id
    assert str(B_OBSERVATION) in message or B_OBSERVATION.isoformat() in message


def test_future_information_is_a_typed_rejection() -> None:
    """An end-March decision cannot consume the April macro release."""
    item = close_item(
        observation_time=B_OBSERVATION,
        available_time=B_AVAILABLE,
        series_id=SERIES_MACRO,
        value=2.5,
    )
    decision = datetime(2026, 3, 31, 23, 0, tzinfo=UTC)
    # "Not yet available" is a legitimate False, never a silent drop.
    assert is_available_for_decision(item, decision) is False
    with pytest.raises(FutureInformationError) as excinfo:
        require_available_for_decision(item, decision)
    assert SERIES_MACRO in str(excinfo.value)


def test_decision_execution_gap_information_is_still_rejected() -> None:
    """The gate is decision_time only.

    Information arriving inside the decision-to-execution gap satisfies
    ``available_time <= execution_time`` yet must still be rejected: no
    admission verdict may depend on ``execution_time``.  Two valid
    ``DecisionTiming`` objects differing only in execution_time (one
    exactly at the item's availability, one later) must produce the
    identical rejection.
    """
    decision = datetime(2026, 3, 31, 22, 0, tzinfo=UTC)
    in_the_gap = datetime(2026, 3, 31, 22, 30, tzinfo=UTC)
    item = close_item(available_time=in_the_gap)

    timings = [
        DecisionTiming(
            decision_time=decision,
            execution_time=in_the_gap,
            return_realization_period=valid_period(
                in_the_gap + timedelta(minutes=15),
                datetime(2026, 4, 30, 22, 45, tzinfo=UTC),
            ),
        ),
        DecisionTiming(
            decision_time=decision,
            execution_time=in_the_gap + timedelta(hours=1),
            return_realization_period=valid_period(
                in_the_gap + timedelta(hours=1, minutes=15),
                datetime(2026, 4, 30, 23, 45, tzinfo=UTC),
            ),
        ),
    ]
    for timing in timings:
        # The timing objects are themselves valid; the gap is the only
        # reason for rejection, so the rejection is causal.
        assert decision < item.available_time <= timing.execution_time
        assert is_available_for_decision(item, timing.decision_time) is False
        with pytest.raises(FutureInformationError):
            require_available_for_decision(item, timing.decision_time)


def test_execution_before_decision_is_rejected() -> None:
    """A trade cannot execute before its decision is made."""
    with pytest.raises(InvalidChronologyError):
        DecisionTiming(
            decision_time=A_DECISION,
            execution_time=A_DECISION - timedelta(minutes=30),
            return_realization_period=_valid_period_for_r_cases(),
        )


def test_zero_length_realization_period_is_rejected() -> None:
    """The realization period needs strictly positive length."""
    start = datetime(2026, 4, 1, 0, 30, tzinfo=UTC)
    with pytest.raises(InvalidChronologyError):
        valid_period(start, start)  # start_time == end_time


def test_reversed_realization_period_is_rejected() -> None:
    """A reversed realization period is malformed."""
    with pytest.raises(InvalidChronologyError):
        valid_period(
            A_EXECUTION + timedelta(days=1),
            A_EXECUTION,
        )


def test_realization_start_before_execution_is_rejected() -> None:
    """Outcomes cannot start realizing before the trade executes."""
    execution = datetime(2026, 3, 31, 23, 0, tzinfo=UTC)
    with pytest.raises(InvalidChronologyError):
        DecisionTiming(
            decision_time=datetime(2026, 3, 31, 22, 0, tzinfo=UTC),
            execution_time=execution,
            return_realization_period=valid_period(
                execution - timedelta(minutes=30),  # itself a valid period
                execution + timedelta(days=30),
            ),
        )


def test_missing_availability_rejects_fail_closed() -> None:
    """Absent availability is never defaulted.

    The boolean query must raise rather than answer ``False`` so that
    "not yet available" is never conflated with "malformed".
    """
    no_availability_field = SimpleNamespace(
        series_id=SERIES_CLOSE, observation_time=A_OBSERVATION, value=1.0
    )
    null_availability = SimpleNamespace(
        series_id=SERIES_CLOSE,
        observation_time=A_OBSERVATION,
        available_time=None,
        value=1.0,
    )
    for malformed in (no_availability_field, null_availability):
        with pytest.raises(MissingAvailabilityError):
            is_available_for_decision(malformed, A_DECISION)
        with pytest.raises(MissingAvailabilityError):
            require_available_for_decision(malformed, A_DECISION)
    with pytest.raises(MissingAvailabilityError):
        TimedObservation(
            series_id=SERIES_CLOSE,
            observation_time=A_OBSERVATION,
            available_time=None,
            value=1.0,
        )


def test_reversed_chronology_is_rejected_at_each_value_object() -> None:
    """Construction itself is fail-closed on every value object."""
    with pytest.raises(InvalidChronologyError):
        close_item(
            observation_time=B_AVAILABLE, available_time=B_OBSERVATION
        )  # observation after availability
    with pytest.raises(InvalidChronologyError):
        DecisionTiming(
            decision_time=A_DECISION,
            execution_time=A_DECISION - timedelta(minutes=1),
            return_realization_period=_valid_period_for_r_cases(),
        )  # execution before decision
    with pytest.raises(InvalidChronologyError):
        valid_period(
            A_EXECUTION + timedelta(days=30), A_EXECUTION
        )  # start after end
