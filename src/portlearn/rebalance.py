"""Rebalancing schedules and the drift law.

This module implements the rebalancing and portfolio-drift contract:
explicit :class:`RebalanceSchedule` declarations of strictly increasing
aware decision instants — calendar-generated or explicitly enumerated,
the same object either way — :class:`HoldingPlan` bindings of two
independently declared instant sequences (rebalance decisions and
accounting), buy-and-hold drift over caller-supplied per-asset gross
growth factors, exact-fill execution of a target at its execution
instant, and the availability gate for factor information at the
instant of use.

The laws, in summary:

- **Declarations, never inference.** A schedule validates strict
  monotonicity and instant awareness, nothing else; it is never
  constructed from data. A holding plan binds two independently
  declared sequences — neither inferred from the other, from the
  dataset, or from a frequency adjective; every cross-frequency
  pairing is first-class, and rebalance-at-every-accounting-instant is
  expressed by declaring the sequences identical.
- **The growth law.** ``w(t⁻)_i = w_i·g_i / Σ_j w_j·g_j`` over a frozen
  universe: buy-and-hold reweighting by caller-supplied gross growth
  factors ``g_i = 1 + r_i``, no trade, no cost, no cash flow, no
  normalization beyond the budget identity itself. The module never
  constructs, infers, or interprets the factors — their provenance is
  owned by the caller's data contract.
- **Fail-closed numerics.** Every factor is finite and nonnegative;
  every weight-factor product must be finite; the portfolio-growth
  denominator must be strictly positive and finite — the binding
  short/leverage rule. An individual ``g_i = 0`` is lawful arithmetic
  (a total loss maps the asset to exactly zero); a non-positive,
  zero, or non-finite denominator is an undefined portfolio state and
  rejects. No ``OverflowError`` leaks on hostile inputs.
- **Decisions and executions.** The accounting instants that coincide
  with rebalance decisions are exactly that — instants shared by the
  two declared sequences; actual POST_TRADE timing is governed by
  ``DecisionTiming.execution_time``. The executed book equals
  the target exactly under exact fill at the execution instant
  ``e ≥ t``, with same-instant decide-and-execute the admissible
  special case; execution instants strictly increase across distinct
  rebalances. Between executions the book drifts passively.
- **Composition, never restatement.** Instant admission reuses the
  timing module's admission rule, calendar anchors reuse the
  period-calendar substrate, and every weight is the ``PortfolioWeights`` value object; the error surface is
  ``ValueError``/``InvalidChronologyError``/
  ``NaiveTimestampError`` only, with no new error types.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from types import MappingProxyType
from typing import Any

from portlearn.calendar import month_end_instant, quarter_end_instant
from portlearn.timing import (
    InvalidChronologyError,
    require_available_for_decision,
    to_instant,
)
from portlearn.weights import PortfolioWeights, WeightState

__all__ = [
    "GrowthFactors",
    "HoldingPlan",
    "RebalanceSchedule",
    "drift_weights",
    "execute_rebalance",
    "month_end_schedule",
    "quarter_end_schedule",
    "require_factors_available",
    "require_valid_rebalances",
]


# --------------------------------------------------------------------------- #
# Numeric discipline (checked-real / guarded-aggregation discipline)
# --------------------------------------------------------------------------- #


def _checked_float(value: float, name: str) -> float:
    """Checked real→float conversion: an int beyond the float range
    raises ``OverflowError`` from ``float()`` — mapped here onto the
    fail-closed finiteness check so the module error surface stays
    ``ValueError`` only. No clipping, no normalization: an out-of-range
    real is rejected, never rescaled."""
    try:
        converted = float(value)
    except OverflowError:
        converted = math.inf if value > 0 else -math.inf
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return converted


def _finite_fsum(values: list[float], name: str) -> float:
    """``math.fsum`` guarded so the aggregation itself can never leak
    ``OverflowError`` or return a non-finite total: individually finite
    values whose exact sum overflows the float range are rejected
    fail-closed with ``ValueError`` (ValueError-only error surface)."""
    try:
        total = math.fsum(values)
    except OverflowError:
        raise ValueError(f"{name} must be finite; got non-finite sum") from None
    if not math.isfinite(total):
        raise ValueError(f"{name} must be finite; got non-finite sum")
    return float(total)


def _require_identifier(asset: object) -> None:
    """An asset identifier is a non-blank string; blank or whitespace-only
    identifiers (and non-strings) reject fail-closed."""
    if not isinstance(asset, str) or not asset.strip():
        raise ValueError(
            "an asset identifier must be a non-blank string naming one "
            f"universe member; got {asset!r}"
        )


def _require_nonnegative_factor(value: object) -> float:
    """A gross growth factor is a finite nonnegative real; ``bool`` is
    not a factor and non-reals are not factors — no coercion."""
    if isinstance(value, bool):
        raise ValueError(  # noqa: TRY004 — bool rejection is contract, not type discipline
            f"bool is not a growth factor; got {value!r}"
        )
    if not isinstance(value, (int, float)):
        raise ValueError(  # noqa: TRY004 — the error surface is ValueError-only by contract
            "a growth factor must be a real number; got "
            f"{type(value).__name__}: {value!r}"
        )
    checked = _checked_float(value, "growth factor")
    if checked < 0:
        raise ValueError(
            "growth factors are gross (g = 1 + r) and must be nonnegative; "
            f"got {value!r}"
        )
    return checked


def _validated_factors(factors: object) -> dict[str, float]:
    """Validate a growth-factor mapping fail-closed and return a fresh
    snapshot dict: identifiers non-blank, factors finite nonnegative
    reals, and the input itself a mapping."""
    if not isinstance(factors, Mapping):
        raise ValueError(  # noqa: TRY004 — the error surface is ValueError-only by contract
            "growth factors must be a mapping from asset identifier to "
            f"gross growth factor; got {type(factors).__name__}: {factors!r}"
        )
    snapshot: dict[str, float] = {}
    for asset, factor in factors.items():
        _require_identifier(asset)
        snapshot[asset] = _require_nonnegative_factor(factor)
    return snapshot


def _require_stride_count(count: object) -> None:
    """A calendar generator emits a positive integer number of
    consecutive period ends; anything else rejects fail-closed."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError(
            "count must be a positive integer naming how many consecutive "
            f"period ends to generate; got {count!r}"
        )


def _normalized_instants(
    source: object, kind: str, what: str
) -> tuple[datetime, ...]:
    """Validate an instant sequence into a UTC-normalized tuple.

    A string/bytes pseudo-sequence, a non-iterable, or an empty
    sequence rejects with ``ValueError``; a naive or non-instant entry
    rejects with ``NaiveTimestampError``; a repeated or reversed
    instant rejects with ``InvalidChronologyError``. Entries normalize
    through ``to_instant``, so equal instants expressed in different
    zones still reject as equal.
    """
    if isinstance(source, (str, bytes)):
        raise ValueError(  # noqa: TRY004 — the error surface is ValueError-only by contract
            f"{what} must be a sequence of aware instants, not a single "
            f"{type(source).__name__}: {source!r}."
        )
    try:
        entries = tuple(source)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(
            f"{what} must be an iterable of aware instants; got "
            f"{type(source).__name__}: {source!r}."
        ) from error
    if not entries:
        raise ValueError(
            f"{what} must contain at least one aware instant; an empty "
            f"{kind} names nothing."
        )
    normalized = [to_instant(entry, f"{kind} entry") for entry in entries]
    for previous, current in pairwise(normalized):
        if not previous < current:
            raise InvalidChronologyError(
                f"chronology violation: {what} must be strictly increasing "
                f"aware instants, but {current.isoformat()} does not "
                f"strictly follow {previous.isoformat()}. A {kind} that "
                "repeats or reverses an instant names no valid sequence."
            )
    return tuple(normalized)


def _decision_execution_instants(timing: Any) -> tuple[datetime, datetime]:
    """Extract and validate one rebalance's (decision, execution) pair.

    Accepts any object declaring ``decision_time``/``execution_time``
    (``DecisionTiming`` is the canonical carrier) and enforces the
    chronology requirement: an execution may never precede its
    decision, while same-instant decide-and-execute is admissible.
    """
    try:
        decision_raw = timing.decision_time
        execution_raw = timing.execution_time
    except AttributeError as error:
        raise ValueError(
            "a rebalance timing must declare decision_time and "
            "execution_time (composing with the DecisionTiming law); got "
            f"{type(timing).__name__}: {timing!r}"
        ) from error
    decision = to_instant(decision_raw, "decision_time")
    execution = to_instant(execution_raw, "execution_time")
    if execution < decision:
        raise InvalidChronologyError(
            "chronology violation: a trade cannot execute before its "
            "decision is made, but execution_time precedes decision_time; "
            f"got decision_time={decision.isoformat()}, "
            f"execution_time={execution.isoformat()}"
        )
    return decision, execution


# --------------------------------------------------------------------------- #
# The rebalance schedule
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RebalanceSchedule:
    """A strictly increasing finite sequence of aware decision instants.

    A pure declaration: construction validates strict monotonicity and
    instant awareness, nothing else. Calendar-generated and explicitly
    enumerated routes produce the same object — equality is on the
    normalized instant sequence. The stored ``instants`` tuple is
    UTC-normalized (comparison basis) and snapshotted: later mutation
    of the source iterable is invisible.
    """

    instants: tuple[datetime, ...]

    def __post_init__(self) -> None:
        normalized = _normalized_instants(
            self.instants, "schedule", "a rebalance schedule"
        )
        object.__setattr__(self, "instants", normalized)

    def __iter__(self) -> Iterator[datetime]:
        return iter(self.instants)

    def __len__(self) -> int:
        return len(self.instants)

    def is_decision_instant(self, instant: Any) -> bool:
        """Does this schedule declare a rebalance decision at ``instant``?

        The query instant is validated and normalized by the shared
        instant normalizer first — a naive instant rejects with
        ``NaiveTimestampError``, never a silent mismatch.
        """
        return to_instant(instant, "schedule entry") in self.instants


def month_end_schedule(
    year: int, month: int, count: int, tzinfo: Any
) -> RebalanceSchedule:
    """Consecutive month-end rebalance decision instants.

    Generates ``count`` successive month-end instants starting with
    ``month`` of ``year``, rolling across year boundaries, each
    composed through the period-calendar substrate (so the declared
    zone must be an aware timezone object or the construction
    rejects with ``NaiveTimestampError``). A calendar generator
    convenience — the produced schedule is identical to explicitly
    enumerating the same instants.
    """
    _require_stride_count(count)
    if not isinstance(month, int) or isinstance(month, bool) or not (
        1 <= month <= 12
    ):
        raise ValueError(
            "month must be an integer 1-12 naming a calendar month; got "
            f"{month!r}"
        )
    return RebalanceSchedule(
        tuple(
            month_end_instant(
                year + (month - 1 + step) // 12,
                (month - 1 + step) % 12 + 1,
                tzinfo,
            )
            for step in range(count)
        )
    )


def quarter_end_schedule(
    year: int, quarter: int, count: int, tzinfo: Any
) -> RebalanceSchedule:
    """Consecutive quarter-end rebalance decision instants.

    Generates ``count`` successive quarter-end instants starting with
    ``quarter`` (1–4) of ``year``, rolling across year boundaries,
    through the period-calendar substrate with the same fail-closed
    timezone rule as :func:`month_end_schedule`.
    """
    _require_stride_count(count)
    if not isinstance(quarter, int) or isinstance(quarter, bool) or not (
        1 <= quarter <= 4
    ):
        raise ValueError(
            "quarter must be an integer 1-4 naming a calendar quarter; got "
            f"{quarter!r}"
        )
    return RebalanceSchedule(
        tuple(
            quarter_end_instant(
                year + (quarter - 1 + step) // 4,
                (quarter - 1 + step) % 4 + 1,
                tzinfo,
            )
            for step in range(count)
        )
    )


# --------------------------------------------------------------------------- #
# Independently declared instant sequences
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HoldingPlan:
    """Two independently declared instant sequences bound into one plan.

    ``schedule`` carries the rebalance decision instants;
    ``accounting_instants`` carries the accounting instants. Neither
    sequence is inferred from the other: cross-frequency pairings are
    first-class, decision instants may fall outside the accounting
    sequence entirely, and rebalancing at every accounting instant is
    expressed by declaring the sequences identical. An accounting
    instant that is also a decision instant is exactly an accounting
    instant that coincides with a rebalance decision; actual
    POST_TRADE timing is governed by ``DecisionTiming.execution_time``.
    Every other accounting instant records drift.
    """

    schedule: RebalanceSchedule
    accounting_instants: tuple[datetime, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, RebalanceSchedule):
            raise ValueError(  # noqa: TRY004 — the error surface is ValueError-only by contract
                "a holding plan binds a RebalanceSchedule as its rebalance "
                f"decision declaration; got {type(self.schedule).__name__}: "
                f"{self.schedule!r}"
            )
        normalized = _normalized_instants(
            self.accounting_instants, "sequence", "an accounting sequence"
        )
        object.__setattr__(self, "accounting_instants", normalized)

    @property
    def decision_instants(self) -> tuple[datetime, ...]:
        """The rebalance decision instants, exactly as declared."""
        return self.schedule.instants

    @property
    def rebalance_accounting_instants(self) -> tuple[datetime, ...]:
        """The accounting instants on which a rebalance decision falls.

        The intersection of the two declared sequences, in accounting
        order — the accounting instants that coincide with rebalance
        decisions. Actual POST_TRADE timing is governed by
        ``DecisionTiming.execution_time``. Empty when the
        sequences are disjoint: a plan whose decisions never coincide
        with accounting records only drift states.
        """
        decisions = frozenset(self.schedule.instants)
        return tuple(
            instant
            for instant in self.accounting_instants
            if instant in decisions
        )

    def is_decision_instant(self, instant: Any) -> bool:
        """Does a rebalance decision fall at ``instant``?"""
        return self.schedule.is_decision_instant(instant)

    def is_accounting_instant(self, instant: Any) -> bool:
        """Does the accounting sequence declare a state at ``instant``?"""
        return to_instant(instant, "accounting instant") in (
            self.accounting_instants
        )


# --------------------------------------------------------------------------- #
# The drift law over growth factors
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GrowthFactors:
    """Per-asset gross growth factors carrying their availability.

    A frozen value object pairing a validated factor mapping (finite
    nonnegative reals on non-blank identifiers, snapshotted immutably)
    with the aware ``available_time`` declaring when the factors first
    could have been known. The factor *provenance* — raw prices,
    adjusted prices, or a return series — is owned by the caller's
    data contract, never by this module.
    """

    factors: Mapping[str, float]
    available_time: datetime

    def __post_init__(self) -> None:
        # Validate the availability instant through the timing module
        # (naive or non-instant values reject with NaiveTimestampError).
        to_instant(self.available_time, "available_time")
        snapshot = _validated_factors(self.factors)
        object.__setattr__(self, "factors", MappingProxyType(snapshot))


def require_factors_available(factors: Any, use_time: Any) -> None:
    """Admission gate for growth-factor information at an instant of use.

    Composes the timing module's admission gate directly — the gate is
    the instant of consumption: factors whose ``available_time`` is
    after the use instant reject with ``FutureInformationError``.
    Factors forming a TARGET at a decision instant ``t`` are consumed
    at ``use_time=t``; factors computing PRE_TRADE drift at an
    execution instant ``e`` are consumed at ``use_time=e`` — realized
    growth through a later execution legitimately contains information
    unavailable at the decision. Availability exactly at the use
    instant admits. This module enforces admission at this boundary;
    it cannot and does not claim to certify caller-side provenance.
    """
    require_available_for_decision(factors, use_time)


def drift_weights(
    weights: Any, factors: Mapping[str, float]
) -> PortfolioWeights:
    """Buy-and-hold drift of a realized portfolio over growth factors.

    ``w(t⁻)_i = w_i·g_i / Σ_j w_j·g_j`` on a frozen universe: the
    factors must name exactly the weight object's assets. Inputs
    validate fail-closed (factors finite and nonnegative; every
    weight-factor product finite; the denominator aggregated under the
    guarded exact-sum discipline), the portfolio-growth denominator
    must be strictly positive and finite — a non-positive, zero, or
    non-finite denominator is an undefined portfolio state and the
    binding short/leverage rule — and the result is a fresh
    ``PRE_TRADE`` value object with ``−0.0`` normalized to ``+0.0``.
    A ``TARGET`` input rejects: a target is a desire, not a portfolio
    that can drift.
    """
    if not isinstance(weights, PortfolioWeights):
        raise ValueError(  # noqa: TRY004 — the error surface is ValueError-only by contract
            "drift_weights operates on a PortfolioWeights value object; "
            f"got {type(weights).__name__}: {weights!r}"
        )
    if weights.state is WeightState.TARGET:
        raise ValueError(
            "drift operates on a realized portfolio state (POST_TRADE or a "
            "previously drifted PRE_TRADE), never a TARGET: a target is a "
            "desire, not a portfolio that can drift; execute it first."
        )
    checked = _validated_factors(factors)
    if set(checked) != set(weights.assets):
        raise ValueError(
            "the growth-factor universe must equal the weights universe "
            "exactly (the universe is frozen over the holding interval); "
            f"factors name {sorted(checked)}, weights name "
            f"{sorted(weights.assets)}"
        )
    terms = [
        weight * checked[asset] for asset, weight in weights.weights.items()
    ]
    if not all(math.isfinite(term) for term in terms):
        raise ValueError(
            "every weight-factor product must be finite; a term overflowed "
            "the float range — the drifted state is undefined, never "
            "rescaled."
        )
    denominator = _finite_fsum(terms, "portfolio-growth denominator")
    if denominator <= 0:
        raise ValueError(
            "the portfolio-growth denominator D = Σ_j w_j·g_j must be "
            "strictly positive and finite — the binding short/leverage "
            f"rule; got D={denominator!r}. The portfolio state is "
            "undefined and fails closed."
        )
    return PortfolioWeights(
        {
            asset: weight * checked[asset] / denominator + 0.0
            for asset, weight in weights.weights.items()
        },
        WeightState.PRE_TRADE,
    )


# --------------------------------------------------------------------------- #
# Execution and the rebalance-event sequence
# --------------------------------------------------------------------------- #


def execute_rebalance(target: Any, timing: Any) -> PortfolioWeights:
    """Execute a target at its execution instant under exact fill.

    The trade transforms the drifted book into ``w(e⁺) = w*(t)`` at the
    execution instant ``e ≥ t`` (the timing requirement composes with
    ``DecisionTiming``; same-instant decide-and-execute is the
    admissible special case). The result is a fresh ``POST_TRADE``
    value object equal to the target exactly — the target itself is
    never mutated, and execution realism (partial fills, refusals) is
    deliberately out of scope. Only a ``TARGET`` may be executed.
    """
    if not isinstance(target, PortfolioWeights) or (
        target.state is not WeightState.TARGET
    ):
        state = getattr(target, "state", None)
        raise ValueError(
            "execute_rebalance executes a PortfolioWeights TARGET under "
            f"exact fill; got {type(target).__name__} in state {state!r}. "
            "Only a TARGET is executable."
        )
    _decision_execution_instants(timing)  # chronology law, fail-closed
    return PortfolioWeights(dict(target.weights), WeightState.POST_TRADE)


def require_valid_rebalances(timings: Iterable[Any]) -> None:
    """Validate a rebalance-event sequence: chronology within each event,
    strictly increasing execution instants across distinct events.

    Every event must satisfy ``execution_time ≥ decision_time`` (never
    reversed; same-instant admissible). No two distinct rebalances may
    share one execution instant — that would create two POST_TRADE
    states at one instant with no ordering semantics — so execution
    instants strictly increase. An empty sequence is vacuously valid.
    Returns ``None`` on success.
    """
    if isinstance(timings, (str, bytes)):
        raise ValueError(  # noqa: TRY004 — the error surface is ValueError-only by contract
            "require_valid_rebalances takes a sequence of rebalance "
            f"timings, not a single {type(timings).__name__}: {timings!r}."
        )
    try:
        entries = tuple(timings)
    except TypeError as error:
        raise ValueError(
            "require_valid_rebalances takes an iterable of rebalance "
            f"timings; got {type(timings).__name__}: {timings!r}."
        ) from error
    previous_execution: datetime | None = None
    for timing in entries:
        _decision, execution = _decision_execution_instants(timing)
        if previous_execution is not None and not previous_execution < execution:
            raise InvalidChronologyError(
                "chronology violation: execution instants must be strictly "
                "increasing across distinct rebalance events — two "
                "rebalances cannot share one execution instant, which "
                "would create two POST_TRADE states at one instant with "
                f"no ordering semantics; got {execution.isoformat()} not "
                f"strictly after {previous_execution.isoformat()}."
            )
        previous_execution = execution
