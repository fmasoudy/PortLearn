"""Time-index and admission contracts for portfolio decisions.

This module defines the timing semantics:

- every financial time is a timezone-aware instant, compared as instants
  (wall clock and zone name never affect ordering or equality);
- naive datetimes and ``datetime.date`` inputs are rejected fail-closed —
  no default zone is ever assumed and no date-to-midnight coercion is
  ever performed, because silent coercion is the classic daily-data
  look-ahead leakage vector;
- the chronology law

  ``observation_time ≤ available_time ≤ decision_time ≤ execution_time
  ≤ realization start_time < realization end_time``

  holds with equal adjacent instants admissible at every ``≤`` boundary
  (zero publication lag, availability exactly at the decision,
  same-instant decide-and-execute) and strictly positive realization
  length;
- the admission law: an item of information may enter the information
  set for a decision at ``decision_time`` iff its ``available_time`` is
  at or before ``decision_time``, evaluated on aware instants.  The gate
  is the decision instant only, never the execution instant, so
  information arriving in the decision-to-execution gap is look-ahead
  leakage and is rejected.

This module defines four fail-closed errors:
``NaiveTimestampError``, ``InvalidChronologyError``,
``FutureInformationError``, and ``MissingAvailabilityError``.  It is
stdlib-only and imports nothing from ``portlearn.observations``, so no
import cycle exists on this contract surface.  Every semantic
ordering, equality, and identity-key comparison operates on the
normalized UTC instant (``to_instant`` / ``instant_key``), so DST-fold
ambiguity can never make two distinct instants compare equal.
PortLearn renders instants in no local timezone of its own;
formatting an instant for display in any zone is the caller's
concern, and no contract comparison depends on any local timezone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

__all__ = [
    "DecisionTiming",
    "FutureInformationError",
    "InvalidChronologyError",
    "MissingAvailabilityError",
    "NaiveTimestampError",
    "ReturnRealizationPeriod",
    "instant_key",
    "is_available_for_decision",
    "require_available_for_decision",
    "to_instant",
]


# --------------------------------------------------------------------------- #
# Fail-closed error taxonomy — the timing-owned arm
# --------------------------------------------------------------------------- #


class NaiveTimestampError(Exception):
    """A time input lacks an explicit timezone, or a date was supplied.

    Financial times are compared as instants on a single timeline.  An
    instant without an explicit UTC offset cannot be placed on that
    timeline without assuming a default zone, and a ``datetime.date``
    cannot be placed on it without silently coercing it to that date's
    midnight — the classic daily-data leakage vector.  PortLearn does
    neither, so the input is rejected.
    """


class InvalidChronologyError(Exception):
    """The chronology law was violated.

    Availability may not precede the observation it describes, a trade
    may not execute before its decision is made, outcomes may not start
    realizing before the trade executes, and the return-realization
    period must have strictly positive length.
    """


class FutureInformationError(Exception):
    """An item failing the admission law was submitted for a decision.

    This is the look-ahead rejection: the item's ``available_time`` is
    after the ``decision_time`` of the decision that would consume it,
    so admitting it would let the decision see information that did not
    exist yet.
    """


class MissingAvailabilityError(Exception):
    """``available_time`` is absent or ``None`` on a submitted item.

    Availability is mandatory: a source unable to declare when the
    information first could have been known must fail, never default to
    the observation instant, "immediately available", or "never
    available".
    """


# --------------------------------------------------------------------------- #
# Instant validation
# --------------------------------------------------------------------------- #


def _require_aware_instant(value: Any, field_name: str) -> datetime:
    """Return ``value`` as a timezone-aware instant, or fail closed.

    Naive datetimes, ``datetime.date`` inputs, and non-instant inputs
    are all rejected with ``NaiveTimestampError``: no default timezone
    is assumed and no date-to-midnight coercion is performed anywhere.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise NaiveTimestampError(
                f"{field_name} is a naive timestamp — it carries no explicit "
                f"UTC offset, so it cannot be placed on the single timeline "
                f"on which financial times are compared without assuming a "
                f"default timezone; got {value!r}. Supply a timezone-aware "
                "datetime (an explicit offset) instead."
            )
        return value
    if isinstance(value, date):
        raise NaiveTimestampError(
            f"{field_name} is a calendar date ({value.isoformat()}), not an "
            "instant: PortLearn never coerces a date to that date's "
            "midnight, because such coercion is the classic daily-data "
            "look-ahead leakage vector. Supply a timezone-aware datetime "
            "with an explicit UTC offset declaring when the information "
            "first could have been known."
        )
    raise NaiveTimestampError(
        f"{field_name} must be a timezone-aware datetime instant carrying "
        f"an explicit UTC offset; got {type(value).__name__}: {value!r}. "
        "Financial times are compared as instants, so an input that is not "
        "an aware instant is rejected fail-closed."
    )


def _item_available_time(item: Any) -> datetime:
    """Extract and validate ``item.available_time`` for the admission law.

    Attribute absence and ``None`` are ``MissingAvailabilityError``; a
    naive datetime or a ``datetime.date`` is ``NaiveTimestampError``.
    """
    try:
        available = item.available_time
    except AttributeError:
        raise MissingAvailabilityError(
            "the submitted information item declares no available_time; "
            "availability is mandatory on every contract surface — a source "
            "unable to declare when the information first could have been "
            "known must fail, never default"
        ) from None
    if available is None:
        raise MissingAvailabilityError(
            "the submitted information item declares available_time=None; "
            "availability is mandatory on every contract surface — a source "
            "unable to declare when the information first could have been "
            "known must fail, never default"
        )
    return _require_aware_instant(available, "available_time")


def _item_series_id(item: Any) -> Any:
    """The item's series identifier when one is in context."""
    return getattr(item, "series_id", None)


# --------------------------------------------------------------------------- #
# Instant normalization — the single comparison basis
# --------------------------------------------------------------------------- #


def to_instant(value: Any, field_name: str = "timestamp") -> datetime:
    """Return the UTC datetime denoting ``value``'s true instant.

    ``value`` is validated fail-closed exactly as
    ``_require_aware_instant`` validates it, then normalized through
    ``astimezone(timezone.utc)``, which resolves ``fold`` through the
    zone's ``utcoffset``: the two Melbourne 2026-04-05 02:30
    ambiguities normalize to ``2026-04-04T15:30Z`` (fold=0, UTC+11)
    and ``2026-04-04T16:30Z`` (fold=1, UTC+10).  Every semantic
    ordering and equality comparison in this package compares these
    normalized instants, never the raw datetimes, because raw
    aware-datetime comparison ignores ``fold`` whenever both sides
    share one ``tzinfo`` object and can therefore call two distinct
    instants equal.  Original datetime objects stay stored where
    they are held for provenance and display; normalization serves
    comparison only — stored data is never UTC-converted.
    """
    moment = _require_aware_instant(value, field_name)
    return moment.astimezone(UTC)


def instant_key(value: Any, field_name: str = "timestamp") -> datetime:
    """Hashable normalized-UTC identity for timestamp keys and sorting.

    The same normalization as ``to_instant``, named for its role:
    use it wherever a datetime becomes a dictionary key, a set
    member, or a sort key, so two fold-distinct instants never
    silently merge into one identity.  Sorting on ``instant_key``
    values is safe because they all carry ``timezone.utc``, where no
    fold ambiguity exists.
    """
    return to_instant(value, field_name)


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReturnRealizationPeriod:
    """The half-open instant interval ``[start_time, end_time)`` over
    which a decision's outcome realizes.

    Both bounds must be aware instants and the period must have strictly
    positive length: ``start_time < end_time``.  Equal adjacent bounds
    are admissible everywhere else in the chronology law, but a
    zero-length or reversed realization period is malformed.
    """

    start_time: datetime
    end_time: datetime

    def __post_init__(self) -> None:
        start = to_instant(self.start_time, "realization start_time")
        end = to_instant(self.end_time, "realization end_time")
        if not start < end:
            raise InvalidChronologyError(
                "malformed return-realization period: start_time must "
                "strictly precede end_time so the half-open period "
                f"[start_time, end_time) has strictly positive length; got "
                f"start_time={start.isoformat()}, end_time={end.isoformat()}"
            )


@dataclass(frozen=True)
class DecisionTiming:
    """When a single portfolio decision is made, executed, and realized.

    Validates the decision-event portion of the chronology law:
    ``decision_time ≤ execution_time ≤ realization start_time``,
    with equal adjacent instants admissible (same-instant
    decide-and-execute is an explicit design) and all instants aware.
    Rebalance schedules and decision frequency are out of scope here.
    """

    decision_time: datetime
    execution_time: datetime
    return_realization_period: ReturnRealizationPeriod

    def __post_init__(self) -> None:
        decision = to_instant(self.decision_time, "decision_time")
        execution = to_instant(self.execution_time, "execution_time")
        start = to_instant(
            self.return_realization_period.start_time, "realization start_time"
        )
        if execution < decision:
            raise InvalidChronologyError(
                "chronology violation: a trade cannot execute before its "
                "decision is made, but execution_time precedes "
                f"decision_time; got decision_time={decision.isoformat()}, "
                f"execution_time={execution.isoformat()}"
            )
        if start < execution:
            raise InvalidChronologyError(
                "chronology violation: outcomes cannot start realizing "
                "before the trade executes, but realization start_time "
                "precedes execution_time; got execution_time="
                f"{execution.isoformat()}, realization start_time="
                f"{start.isoformat()}"
            )


# --------------------------------------------------------------------------- #
# The admission law
# --------------------------------------------------------------------------- #


def is_available_for_decision(item: Any, decision_time: Any) -> bool:
    """Pure admission query: is ``item``'s information visible yet?

    Returns ``True`` iff ``item.available_time <= decision_time`` on
    aware instants — availability exactly at the decision admits the
    item.  ``False`` means "not yet available", a legitimate verdict,
    never a silent drop.  Malformed inputs still raise: the
    ``decision_time`` is validated before any comparison, and an item
    whose ``available_time`` is absent, ``None``, naive, or a date fails
    closed with its typed error so "not yet available" is never
    conflated with "malformed".
    """
    decision = to_instant(decision_time, "decision_time")
    available = to_instant(_item_available_time(item), "available_time")
    return available <= decision


def require_available_for_decision(item: Any, decision_time: Any) -> None:
    """Admission gate: raise on look-ahead, return ``None`` on admission.

    Raises ``FutureInformationError`` when ``item.available_time`` is
    after ``decision_time`` — including information arriving inside the
    decision-to-execution gap, which satisfies no admission verdict
    because the gate is the decision instant only, never the execution
    instant.  Malformed inputs raise their typed
    errors exactly as the boolean query does.
    """
    decision = to_instant(decision_time, "decision_time")
    available = to_instant(_item_available_time(item), "available_time")
    if available > decision:
        series_id = _item_series_id(item)
        raise FutureInformationError(
            f"look-ahead rejection: the information item {series_id!r} is "
            "not available at the decision instant — available_time="
            f"{available.isoformat()} is after decision_time="
            f"{decision.isoformat()}. Information that first could have "
            "been known after the decision (including inside the "
            "decision-to-execution gap) is future information and may not "
            "enter the information set."
        )
