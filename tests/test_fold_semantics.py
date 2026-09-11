"""DST-fold instant-semantics regression battery for portlearn.

These tests freeze the fold-corrected timing semantics: every semantic
timestamp ordering,
equality, and identity-key comparison in ``portlearn.timing`` and
``portlearn.observations`` must operate on the normalized UTC instant
the datetime denotes — honoring ``fold`` — never on raw aware-datetime
comparison.

Why raw comparison is untrustworthy: when two aware datetimes share the
same ``tzinfo`` object, Python's comparison ignores ``fold``, so the
Melbourne 2026-04-05 02:30 ambiguity — fold=0 (AEDT, UTC+11,
2026-04-04T15:30Z) and fold=1 (AEST, UTC+10, 2026-04-04T16:30Z) —
compares *equal* (and hashes equal) although the two denote instants an
hour apart.  On the raw comparison an observation available at the
later fold=1 instant is wrongly visible to a decision made at the
earlier fold=0 instant — exactly the no-look-ahead violation this
battery reproduces.

The battery below walks that one fold through every semantic surface:
the admission law (pure query and gate), vintage visibility and
selection, observation identity keys, the decision→execution→
realization chronology, feature-lineage ordering, and the
normalization helpers themselves (``to_instant`` / ``instant_key``).
Tests marked *regression* fail on a tree without the fold fix; tests
marked *boundary pin* hold on both trees and guard the corrected
semantics against future regressions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from portlearn.interfaces import (
    Forecast,
    InformationSet,
    PortfolioDecision,
    require_forecast_decision_compatible,
)
from portlearn.observations import (
    AmbiguousObservationError,
    FeatureLineageError,
    TimedObservation,
    require_lineage_monotone,
    vintage_as_of,
)
from portlearn.timing import (
    DecisionTiming,
    FutureInformationError,
    InvalidChronologyError,
    ReturnRealizationPeriod,
    is_available_for_decision,
    require_available_for_decision,
)

MELBOURNE = ZoneInfo("Australia/Melbourne")

#: The reproduced ambiguity: Melbourne's 2026-04-05 02:30 wall
#: time occurs twice (AEDT→AEST transition).  Raw aware-datetime
#: comparison with a shared tzinfo ignores ``fold`` and treats these as
#: equal instants; truly they are one hour apart.
FOLD_FIRST = datetime(2026, 4, 5, 2, 30, tzinfo=MELBOURNE, fold=0)  # 15:30Z
FOLD_SECOND = datetime(2026, 4, 5, 2, 30, tzinfo=MELBOURNE, fold=1)  # 16:30Z
FOLD_FIRST_UTC = datetime(2026, 4, 4, 15, 30, tzinfo=UTC)
FOLD_SECOND_UTC = datetime(2026, 4, 4, 16, 30, tzinfo=UTC)

#: 03:30 Melbourne on 2026-04-05 is unambiguous AEST (UTC+10): 17:30Z,
#: strictly after both fold instants.
AFTER_BOTH_FOLDS = datetime(2026, 4, 5, 3, 30, tzinfo=MELBOURNE)

SERIES = "PX.XJO.DAILY.CLOSE"


# ---------------------------------------------------------------------------
# (a) The admission law across the fold — the exact look-ahead reproduced
# ---------------------------------------------------------------------------


def test_fold_later_instant_is_not_available_at_earlier_fold_decision() -> None:
    """Regression (C1 core): an item available at the fold=1 instant
    (16:30Z) is NOT available to a decision made at the fold=0 instant
    (15:30Z) — raw comparison says ``<=`` and admits it, which is the
    look-ahead this battery reproduces."""
    item = SimpleNamespace(available_time=FOLD_SECOND)
    assert is_available_for_decision(item, FOLD_FIRST) is False


def test_fold_later_instant_rejects_at_the_admission_gate() -> None:
    """Regression: the admission gate raises ``FutureInformationError``
    for an item whose availability is the later fold instant when the
    decision is the earlier fold instant."""
    item = SimpleNamespace(available_time=FOLD_SECOND)
    with pytest.raises(FutureInformationError):
        require_available_for_decision(item, FOLD_FIRST)


def test_fold_later_vintage_is_invisible_at_earlier_fold_decision() -> None:
    """Regression: ``vintage_as_of`` returns ``None`` for a record whose
    ``available_time`` is the later fold instant when the decision is
    the earlier fold instant — the revised value must be invisible
    before its own availability."""
    record = TimedObservation(SERIES, FOLD_FIRST_UTC, FOLD_SECOND, 101.0)
    assert vintage_as_of([record], FOLD_FIRST) is None


def test_fold_first_instant_is_admissible_at_its_own_decision_instant() -> None:
    """Boundary pin: availability exactly at the decision instant still
    admits — the inclusive ``<=`` boundary survives fold correction."""
    item = SimpleNamespace(available_time=FOLD_FIRST)
    assert is_available_for_decision(item, FOLD_FIRST) is True
    require_available_for_decision(item, FOLD_FIRST)  # no raise
    record = TimedObservation(SERIES, FOLD_FIRST_UTC, FOLD_FIRST, 100.0)
    assert vintage_as_of([record], FOLD_FIRST) is record


# ---------------------------------------------------------------------------
# (b) Strict ordering of the two fold instants
# ---------------------------------------------------------------------------


def test_normalized_instants_order_the_two_folds_strictly() -> None:
    """Regression: ``to_instant`` places fold=1 strictly after fold=0 —
    2026-04-04T16:30Z > 2026-04-04T15:30Z — where the raw comparison
    reports equality."""
    from portlearn.timing import to_instant

    assert to_instant(FOLD_SECOND) > to_instant(FOLD_FIRST)
    assert to_instant(FOLD_FIRST) < to_instant(FOLD_SECOND)
    assert to_instant(FOLD_FIRST) != to_instant(FOLD_SECOND)


def test_realization_period_spanning_the_fold_has_positive_length() -> None:
    """Regression: ``[fold=0, fold=1)`` is a genuinely positive-length
    realization period (one true hour) and must construct — the raw
    equal comparison wrongly rejects it as zero-length."""
    period = ReturnRealizationPeriod(FOLD_FIRST, FOLD_SECOND)
    assert period.start_time is FOLD_FIRST  # originals kept for provenance
    assert period.end_time is FOLD_SECOND


def test_reversed_realization_period_across_the_fold_still_rejects() -> None:
    """Boundary pin: ``[fold=1, fold=0)`` has negative true length and
    rejects with ``InvalidChronologyError`` after fold correction."""
    with pytest.raises(InvalidChronologyError):
        ReturnRealizationPeriod(FOLD_SECOND, FOLD_FIRST)


# ---------------------------------------------------------------------------
# (c) Identity keys distinguish the folds — no silent dedup
# ---------------------------------------------------------------------------


def test_instant_keys_distinguish_the_two_folds() -> None:
    """Regression: ``instant_key`` yields distinct hashable identities
    for the two fold instants, so timestamp-keyed mappings never
    silently merge them."""
    from portlearn.timing import instant_key

    key_first = instant_key(FOLD_FIRST)
    key_second = instant_key(FOLD_SECOND)
    assert key_first != key_second
    assert hash(key_first) != hash(key_second)
    mapping = {key_first: "fold-0", key_second: "fold-1"}
    assert len(mapping) == 2


def test_fold_distinct_availabilities_are_not_duplicate_identities() -> None:
    """Regression: two revisions whose ``available_time`` values differ
    only by fold are DISTINCT identity triples — not the duplicate the
    raw equal comparison reports — and the visible vintage at a later
    decision is the true-latest (fold=1) revision, regardless of input
    order."""
    earlier_revision = TimedObservation(SERIES, FOLD_FIRST_UTC, FOLD_FIRST, 1.0)
    later_revision = TimedObservation(SERIES, FOLD_FIRST_UTC, FOLD_SECOND, 2.0)
    assert vintage_as_of(
        [earlier_revision, later_revision], AFTER_BOTH_FOLDS
    ) is later_revision
    assert vintage_as_of(
        [later_revision, earlier_revision], AFTER_BOTH_FOLDS
    ) is later_revision
    # Before the fold: only the fold=0 revision is visible.
    assert vintage_as_of(
        [earlier_revision, later_revision], FOLD_FIRST
    ) is earlier_revision


def test_observation_times_differing_only_by_fold_are_distinct_groups() -> None:
    """Boundary pin: two records whose ``observation_time`` values
    differ only by fold are different observation instants — different
    ``(series_id, observation_time)`` groups — so submitting both to
    one ``vintage_as_of`` query rejects with
    ``AmbiguousObservationError`` (mixed groups), never a silent
    merge."""
    fold_zero_observed = TimedObservation(SERIES, FOLD_FIRST, FOLD_FIRST, 1.0)
    fold_one_observed = TimedObservation(SERIES, FOLD_SECOND, FOLD_SECOND, 2.0)
    with pytest.raises(AmbiguousObservationError):
        vintage_as_of([fold_zero_observed, fold_one_observed], AFTER_BOTH_FOLDS)


# ---------------------------------------------------------------------------
# (e) Decision→execution→realization ordering across the fold
# ---------------------------------------------------------------------------


def test_decision_to_execution_across_the_fold_orders_correctly() -> None:
    """Regression: deciding at fold=0 (15:30Z) and executing at fold=1
    (16:30Z) is a valid one-hour-later execution; the reversed pair —
    executing at fold=0 a decision made at fold=1 — violates the
    chronology law on true instants and must reject."""
    valid = DecisionTiming(
        decision_time=FOLD_FIRST,
        execution_time=FOLD_SECOND,
        return_realization_period=ReturnRealizationPeriod(
            FOLD_SECOND, AFTER_BOTH_FOLDS
        ),
    )
    assert valid.decision_time is FOLD_FIRST  # originals kept for display
    with pytest.raises(InvalidChronologyError):
        DecisionTiming(
            decision_time=FOLD_SECOND,
            execution_time=FOLD_FIRST,
            return_realization_period=ReturnRealizationPeriod(
                FOLD_FIRST, AFTER_BOTH_FOLDS
            ),
        )


def test_realization_start_before_execution_across_the_fold_rejects() -> None:
    """Regression: realizing from the fold=0 instant (15:30Z) a trade
    executed at the fold=1 instant (16:30Z) starts before the trade
    executes on true instants and must reject, though the raw
    comparison calls the two equal and admits it."""
    with pytest.raises(InvalidChronologyError):
        DecisionTiming(
            decision_time=FOLD_FIRST,
            execution_time=FOLD_SECOND,
            return_realization_period=ReturnRealizationPeriod(
                FOLD_FIRST, AFTER_BOTH_FOLDS
            ),
        )


# ---------------------------------------------------------------------------
# (f) Feature lineage ordering across the fold
# ---------------------------------------------------------------------------


def test_lineage_feature_before_fold_later_input_rejects() -> None:
    """Regression: a feature declared available at fold=0 (15:30Z)
    deriving from an input available at fold=1 (16:30Z) is declared
    before its latest input and rejects with ``FeatureLineageError`` —
    the raw equal comparison admits it as simultaneous."""
    with pytest.raises(FeatureLineageError):
        require_lineage_monotone(FOLD_FIRST, [FOLD_SECOND])


def test_lineage_latest_input_is_the_true_latest_instant() -> None:
    """Regression: among inputs at both folds the latest input is the
    fold=1 instant, so a feature at fold=0 rejects even when a fold=0
    input is also present (the raw ``max`` collapses the two folds and
    picks the first)."""
    with pytest.raises(FeatureLineageError):
        require_lineage_monotone(FOLD_FIRST, [FOLD_FIRST, FOLD_SECOND])


def test_lineage_feature_at_fold_second_with_mixed_fold_inputs_admits() -> None:
    """Boundary pin: a feature available exactly at the true-latest
    (fold=1) input instant satisfies lineage monotonicity."""
    require_lineage_monotone(FOLD_SECOND, [FOLD_FIRST, FOLD_SECOND])  # no raise
    require_lineage_monotone(FOLD_SECOND, [FOLD_FIRST])  # no raise


# ---------------------------------------------------------------------------
# (g) Equality holds only for the same true instant
# ---------------------------------------------------------------------------


def test_instant_equality_holds_only_for_the_same_true_instant() -> None:
    """Regression: ``to_instant`` renders each fold equal only to the
    UTC datetime denoting the same true instant — fold=0 is
    2026-04-04T15:30Z, fold=1 is 2026-04-04T16:30Z, and the two folds
    are not equal to each other."""
    from portlearn.timing import to_instant

    assert to_instant(FOLD_FIRST) == to_instant(FOLD_FIRST_UTC)
    assert to_instant(FOLD_SECOND) == to_instant(FOLD_SECOND_UTC)
    assert to_instant(FOLD_FIRST) != to_instant(FOLD_SECOND)


def test_admission_across_zones_matches_the_true_instant() -> None:
    """Boundary pin: the admission law evaluated across different
    tzinfo objects (UTC decision, Melbourne availability) follows the
    true instant on both trees and after the fix — fold=0 availability
    admits at 15:30Z, fold=1 availability does not."""
    fold_zero_item = SimpleNamespace(available_time=FOLD_FIRST)
    fold_one_item = SimpleNamespace(available_time=FOLD_SECOND)
    assert is_available_for_decision(fold_zero_item, FOLD_FIRST_UTC) is True
    assert is_available_for_decision(fold_one_item, FOLD_FIRST_UTC) is False


# --------------------------------------------------------------------------- #
# (h) The interface surfaces — PortfolioDecision chronology and forecast
#     compatibility across the fold (the two residual raw comparisons
#     corrected in interfaces.py)
# --------------------------------------------------------------------------- #


def portfolio_decision_at(
    decision_time: datetime, execution_time: datetime
) -> PortfolioDecision:
    """A minimal well-formed-elsewhere PortfolioDecision for testing."""
    return PortfolioDecision(
        decision_time=decision_time,
        execution_time=execution_time,
        target_weights={"ASX:XJO": 1.0},
    )


def forecast_at(decision_time: datetime) -> Forecast:
    """A minimal well-formed-elsewhere Forecast for testing."""
    return Forecast(
        values={"ASX:XJO": 0.012},
        target="expected_return",
        decision_time=decision_time,
        produced_by="test-forecaster",
    )


def test_portfolio_reversed_rejects() -> None:
    """Regression (C1 residual, interfaces.py): a decision made at the
    fold=1 instant (16:30Z) executing at the fold=0 instant (15:30Z)
    is a genuinely reversed chronology one true hour apart and must
    reject with ``InvalidChronologyError`` — the raw comparison calls
    the two folds equal and wrongly admits it."""
    with pytest.raises(InvalidChronologyError):
        portfolio_decision_at(FOLD_SECOND, FOLD_FIRST)


def test_forecast_lookahead_rejects() -> None:
    """Regression (C1 residual, interfaces.py): a forecast originated
    at the fold=1 instant (16:30Z) consumed by a decision dated at the
    fold=0 instant (15:30Z) is look-ahead leakage one true hour wide
    and must reject with ``InvalidChronologyError`` — the raw
    comparison calls the two folds equal and wrongly admits it."""
    forecast = forecast_at(FOLD_SECOND)
    decision = portfolio_decision_at(FOLD_FIRST, FOLD_FIRST)
    with pytest.raises(InvalidChronologyError):
        require_forecast_decision_compatible(forecast, decision)


def test_portfolio_fold_zero_to_fold_one_execution_admits() -> None:
    """Boundary pin: deciding at fold=0 (15:30Z) and executing at
    fold=1 (16:30Z) is a valid one-hour-later execution and constructs
    — with both original datetimes stored exactly as given, never
    UTC-converted."""
    decision = portfolio_decision_at(FOLD_FIRST, FOLD_SECOND)
    assert decision.decision_time is FOLD_FIRST
    assert decision.execution_time is FOLD_SECOND


def test_portfolio_same_instant_across_zones_admits() -> None:
    """Boundary pin: decision and execution at the same true instant
    expressed in different zones — Melbourne fold=1 02:30 and
    2026-04-04T16:30Z — is the admissible same-instant
    decide-and-execute, and the stored datetimes keep their original
    zones."""
    decision = portfolio_decision_at(FOLD_SECOND, FOLD_SECOND_UTC)
    assert decision.decision_time is FOLD_SECOND
    assert decision.execution_time is FOLD_SECOND_UTC


def test_forecast_compatibility_same_instant_across_zones_admits() -> None:
    """Boundary pin: a forecast originated at the same true instant as
    the decision, expressed in a different zone, is the admissible
    decide-exactly-at-the-origin case."""
    forecast = forecast_at(FOLD_FIRST_UTC)
    decision = portfolio_decision_at(FOLD_FIRST, FOLD_FIRST)
    assert require_forecast_decision_compatible(forecast, decision) is None


def test_forecast_compatibility_later_decision_admits() -> None:
    """Boundary pin: a forecast originated at fold=0 consumed by a
    decision dated at the later fold=1 instant is ordinary
    decide-on-existing-information and admits."""
    forecast = forecast_at(FOLD_FIRST)
    decision = portfolio_decision_at(FOLD_SECOND, FOLD_SECOND)
    assert require_forecast_decision_compatible(forecast, decision) is None


# --------------------------------------------------------------------------- #
# (i) InformationSet identity/group keys across the fold (the residual
#     raw datetime keys corrected in interfaces.py)
# --------------------------------------------------------------------------- #


def test_information_set_admits_fold_distinct_observation_instants() -> None:
    """Regression (C1 residual, interfaces.py): two records whose
    observation_time and available_time differ only by fold are two
    DISTINCT true observation instants an hour apart — not the
    duplicates the raw aware-datetime comparison reports — so an
    ``InformationSet`` formed after both are available admits both as
    two records, retaining the original datetimes exactly as given."""
    fold_zero_observed = TimedObservation(SERIES, FOLD_FIRST, FOLD_FIRST, 1.0)
    fold_one_observed = TimedObservation(SERIES, FOLD_SECOND, FOLD_SECOND, 2.0)
    information_set = InformationSet(
        [fold_zero_observed, fold_one_observed], AFTER_BOTH_FOLDS
    )
    admitted = tuple(information_set)
    assert len(admitted) == 2
    assert admitted[0] is fold_zero_observed  # originals retained, in
    assert admitted[1] is fold_one_observed  # submission order
    assert admitted[0].observation_time is FOLD_FIRST
    assert admitted[0].available_time is FOLD_FIRST
    assert admitted[1].observation_time is FOLD_SECOND
    assert admitted[1].available_time is FOLD_SECOND


def test_information_set_same_instant_across_zones_is_duplicate_identity() -> None:
    """Regression (C1 residual, interfaces.py): two records carrying the
    SAME true observation and availability instant, one expressed in
    Melbourne wall time and the other in UTC, are one identity triple —
    instant identity, not wall-time/zone identity — so submitting both
    to one ``InformationSet`` rejects with
    ``AmbiguousObservationError`` even though each datetime object
    compares unequal and the raw tuples never collide."""
    melbourne_observed = TimedObservation(SERIES, FOLD_FIRST, FOLD_FIRST, 1.0)
    utc_observed = TimedObservation(
        SERIES, FOLD_FIRST_UTC, FOLD_FIRST_UTC, 1.0
    )
    assert melbourne_observed.observation_time != utc_observed.observation_time
    assert melbourne_observed.available_time != utc_observed.available_time
    with pytest.raises(AmbiguousObservationError):
        InformationSet(
            [melbourne_observed, utc_observed], AFTER_BOTH_FOLDS
        )
