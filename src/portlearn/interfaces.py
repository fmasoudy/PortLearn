"""Core research interfaces for portfolio decisions.

This module defines the core interface contract: the
admitted-information value object ``InformationSet``, the
provenance-bearing ``Forecast`` and ``PortfolioDecision`` value
objects with the deliberately thin ``AccountingResult``, the
decision-seam value objects ``DecisionContext`` and
``DecisionResult``, the seven structural contracts,
and the three compatibility validators
``require_feature_lineage``,
``require_forecast_decision_compatible``, and
``require_decision_result_compatible``.

Four laws are inherited from the timing and observations modules
without change:

- **No new time semantics** — every instant an interface carries is one
  of this package's financial timestamps with its fixed meaning, and a
  naive ``datetime`` or a ``datetime.date`` is rejected strict at
  every boundary with the reused ``NaiveTimestampError``: no default
  timezone is assumed and no date-to-midnight coercion is performed.
- **No second admission rule** — admission into an ``InformationSet``
  is exactly the inclusive law
  ``available_time <= decision_time``, delegated to
  ``require_available_for_decision``; this module never re-implements,
  widens, or shadows it.
- **No new error classes** — the error surface reuses the timing
  and observations errors with their fixed module ownership
  (``portlearn.timing`` and ``portlearn.observations``); the only
  built-in error raised is ``ValueError`` for blank identifiers, blank
  targets, and blank provenance strings, mirroring the ``series_id``
  discipline.
- **No weight arithmetic** — ``Forecast.values`` and
  ``PortfolioDecision.target_weights`` are stored exactly as given;
  sum-to-one, gross and net exposure, and feasibility conventions are left to the portfolio-weight contracts of :mod`portlearn.weights`, not this module's.

Exactly four contracts (``RebalancePolicy``, ``CostModel``,
``AccountingEngine``, ``Evaluator``) carry ``@runtime_checkable``,
because runtime structural checks genuinely exist on those surfaces;
``FeatureTransform``, ``Forecaster``, and ``Strategy`` remain
static-only and are validated behavioral. The module is stdlib-only
and imports only from the timing, observations, and weights
modules: the decision context reuses the ``WeightState`` portfolio-role vocabulary instead of
defining a second one; ``portlearn.weights`` itself is unchanged).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from .observations import (
    AmbiguousObservationError,
    TimedObservation,
    require_lineage_monotone,
)
from .timing import (
    InvalidChronologyError,
    NaiveTimestampError,  # noqa: F401  # re-exported: raised at this surface by the validator from portlearn.timing
    _require_aware_instant,
    require_available_for_decision,
)
from .timing import (
    to_instant as _to_instant,  # timing-owned normalizer reused for comparison only; the private alias avoids widening the public namespace
)
from .weights import (  # reuse the weights module's weight-role vocabulary; portlearn.weights itself stays unchanged
    PortfolioWeights,
    WeightState,
)

__all__ = [
    "AccountingEngine",
    "AccountingResult",
    "CostModel",
    "DecisionContext",
    "DecisionResult",
    "Evaluator",
    "FeatureTransform",
    "Forecast",
    "Forecaster",
    "InformationSet",
    "PortfolioDecision",
    "RebalancePolicy",
    "Strategy",
    "require_decision_result_compatible",
    "require_feature_lineage",
    "require_forecast_decision_compatible",
]


# --------------------------------------------------------------------------- #
# Unconditional input checks — reused from portlearn.timing
# --------------------------------------------------------------------------- #

# The aware-instant validator is implemented in ``portlearn.timing``
# and imported above: shared invariant validation has exactly one
# implementation per public interface, and every surface that needs
# it re-exports that exact function object, preserving object
# identity package-wide. The canonical
# invalid-input wording — including its calendar-date tail — is
# timing.py's and is reused verbatim rather than shortened; the
# rejection itself is unchanged in kind: naive datetimes,
# ``datetime.date`` inputs, and non-instant inputs all still fail
# closed with the reused ``NaiveTimestampError``.


def _require_identifier(identifier: object, field_name: str) -> None:
    """Reject blank or non-string identifiers unconditional.

    Identifiers are exact strings — no case folding, trimming, or
    Unicode normalization of any kind — and a blank or non-string
    identifier names no instrument or series, so the value is rejected
    with the built-in ``ValueError`` exactly as a malformed
    ``series_id`` is rejected.
    """
    if not isinstance(identifier, str):
        raise ValueError(  # noqa: TRY004 — identifier discipline reuses ValueError
            f"{field_name} must be an exact string identifier naming "
            f"the instrument or series it refers to; got "
            f"{type(identifier).__name__}: {identifier!r}. A non-string "
            "identifier names no instrument or series, so the value is "
            "rejected unconditionally."
        )
    if not identifier.strip():
        raise ValueError(
            f"{field_name} must be a non-empty, non-blank string "
            f"identifier naming the instrument or series it refers to; "
            f"got {identifier!r}. A blank identifier names no "
            "instrument or series, so the value is rejected "
            "unconditional."
        )


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, init=False)
class InformationSet:
    """The point-in-time information admitted for one portfolio decision.

    An ``InformationSet`` is the set of ``TimedObservation`` records
    admitted at one decision instant ``as_of``: every record's
    ``available_time`` is at or before ``as_of`` (the inclusive
    admission law, delegated to ``require_available_for_decision``), so
    the set is exactly what could have been known when the decision was
    made — never information from the decision-to-execution gap.

    Construction validates strictly over the whole submitted collection, in
    this exact order:

    1. ``as_of`` must be an aware instant (naive datetimes and
       ``datetime.date`` reject with ``NaiveTimestampError``) — checked
       before any item is inspected;
    2. two or more records sharing the full identity triple
       ``(series_id, observation_time, available_time)`` reject with
       ``AmbiguousObservationError`` — no last-write-wins, no
       deduplication, no value-equality exception;
    3. two or more records sharing one ``(series_id,
       observation_time)`` group with differing ``available_time`` — a
       revision pair — reject with the same ``AmbiguousObservationError``:
       the set is given at most one vintage per series-observation and
       never selects among vintages itself. Vintage selection is the
       caller's, performed beforehand with the ``vintage_as_of``
       operation;
    4. each record is admitted in submission order by the admission
       predicate; any look-ahead record rejects with
       ``FutureInformationError``.

    Empty input is admissible as the empty information set: when
    nothing is available at ``as_of``, the admitted set is empty, and
    empty is not ambiguous. The object is immutable and iterable (its
    records, in submission order); ``as_of`` is exposed read-only.
    There is no query, filter, transformation, lazy-admission, or
    vintage-selection mechanism on this type.
    """

    _records: tuple[TimedObservation, ...]
    _as_of: datetime

    def __init__(
        self, items: Iterable[TimedObservation], as_of: datetime
    ) -> None:
        """Admit ``items`` at ``as_of`` or reject invalid input."""
        decision = _require_aware_instant(as_of, "as_of")
        submitted = tuple(items)

        seen_identities: set[tuple[str, datetime, datetime]] = set()
        seen_groups: set[tuple[str, datetime]] = set()
        for record in submitted:
            # Identity and group keys carry the normalized UTC instant
            # (the timing-owned normalizer), never the raw datetime:
            # raw comparison ignores ``fold`` within one zone and calls
            # two distinct instants equal, while across zones it calls
            # one instant two. Keys only — the stored records keep
            # their original datetimes.
            observation_instant = _to_instant(
                record.observation_time, "observation_time"
            )
            available_instant = _to_instant(
                record.available_time, "available_time"
            )
            identity = (
                record.series_id,
                observation_instant,
                available_instant,
            )
            if identity in seen_identities:
                raise AmbiguousObservationError(
                    "ambiguous information-set input: two submitted "
                    "records share the full identity triple "
                    f"(series_id={record.series_id!r}, "
                    f"observation_time="
                    f"{record.observation_time.isoformat()}, "
                    f"available_time="
                    f"{record.available_time.isoformat()}) — the "
                    "identity triple is a key, so the admitted "
                    "information set accepts no duplicate records: no "
                    "last-write-wins, no deduplication, and no "
                    "value-equality exception."
                )
            seen_identities.add(identity)
            group = (record.series_id, observation_instant)
            if group in seen_groups:
                raise AmbiguousObservationError(
                    "ambiguous information-set input: the submitted "
                    "records carry more than one vintage of the same "
                    f"series-observation (series_id="
                    f"{record.series_id!r}, observation_time="
                    f"{record.observation_time.isoformat()}) — the "
                    "admitted information set is given at most one "
                    "vintage per series-observation and never selects "
                    "among vintages itself. Select the point-in-time "
                    "vintage with the ``vintage_as_of`` operation "
                    "before constructing the set."
                )
            seen_groups.add(group)

        for record in submitted:
            require_available_for_decision(record, decision)

        object.__setattr__(self, "_records", submitted)
        object.__setattr__(self, "_as_of", decision)

    @property
    def as_of(self) -> datetime:
        """The decision instant the records were admitted at."""
        return self._as_of

    def __iter__(self):
        """Iterate the admitted records in submission order."""
        return iter(self._records)


@dataclass(frozen=True)
class Forecast:
    """A provenance-bearing forecast for one portfolio decision.

    ``values`` maps exact-string instrument or series identifiers to
    forecasted floats with no numeric, distributional, or behavioral
    constraint at this interface — negative expected returns,
    one, and extreme magnitudes are all legitimate forecast content.
    ``target`` declares the financial quantity the values speak about
    (an expected return, a risk measure, or any other declared target —
    an open vocabulary: the surface guarantees target identity as an
    exact non-blank string, never a taxonomy). ``decision_time`` is the
    decision instant of the ``InformationSet`` the forecast was
    formed from — equality with that set's ``as_of`` is the
    ``Forecaster`` implementer's obligation, verified in contract
    tests, not a constructor-enforced invariant. ``produced_by`` names
    the producing forecaster for provenance.

    Construction validates strictly: ``decision_time`` must be an aware
    instant, and ``target``, ``produced_by``, and every key of
    ``values`` must be a non-blank exact string. ``values`` is stored
    exactly as given — no copy, no deep freeze.
    """

    values: Mapping[str, float]
    target: str
    decision_time: datetime
    produced_by: str

    def __post_init__(self) -> None:
        _require_aware_instant(self.decision_time, "decision_time")
        for identifier in self.values:
            _require_identifier(identifier, "the forecast's values key")
        _require_identifier(self.target, "the forecast's target")
        _require_identifier(self.produced_by, "the forecast's produced_by")


@dataclass(frozen=True)
class PortfolioDecision:
    """One portfolio decision: when it is made, executed, and targeted.

    ``decision_time`` and ``execution_time`` are aware instants
    satisfying the chronology law ``decision_time <=
    execution_time`` — same-instant decide-and-execute is admissible, a
    reversed order is not. ``target_weights`` maps exact-string
    instrument identifiers to target portfolio weights and is stored
    exactly as given: this interface places no weight constraint of
    any kind — sum-to-one, gross and net exposure, long/short, and
    feasibility conventions are left to the portfolio-weight contracts of :mod`portlearn.weights`.

    Construction validates strictly: both instants must be aware, every
    ``target_weights`` key must be a non-blank exact string, and the
    chronology law must hold.
    """

    decision_time: datetime
    execution_time: datetime
    target_weights: Mapping[str, float]

    def __post_init__(self) -> None:
        _require_aware_instant(self.decision_time, "decision_time")
        _require_aware_instant(self.execution_time, "execution_time")
        for identifier in self.target_weights:
            _require_identifier(
                identifier, "the decision's target_weights key"
            )
        decision_instant = _to_instant(self.decision_time, "decision_time")
        execution_instant = _to_instant(self.execution_time, "execution_time")
        if execution_instant < decision_instant:
            raise InvalidChronologyError(
                "chronology violation: a trade cannot execute before "
                "its decision is made, but execution_time precedes "
                f"decision_time; got decision_time="
                f"{decision_instant.isoformat()}, execution_time="
                f"{execution_instant.isoformat()}. Same-instant "
                "decide-and-execute is admissible; a reversed order is "
                "not."
            )


@dataclass(frozen=True)
class AccountingResult:
    """The result of accounting one portfolio decision (deliberately thin).

    A immutable value object with exactly one fact: accounting produces an
    identifiable accounting result containing the post-trade portfolio
    weights. It is deliberately not a ledger schema — no lot, cash,
    fee, or realized-P&L field exists on it, and ``post_trade_weights``
    carries no timing, realization-period, or accounting convention at
    this interface. No validation is applied to its contents: which
    accounting rules produced the weights is the accounting engine's
    to define.
    """

    post_trade_weights: Mapping[str, float]


@dataclass(frozen=True)
class DecisionContext:
    """The decision-time aggregate for one portfolio decision.

    Everything dynamic a strategy may condition on at the decision
    instant, and nothing from after it: the authoritative timing
    anchor ``decision_time``, the admitted ``information``, the
    caller-declared ``universe``, the decision-time portfolio
    snapshot ``current_weights`` with its own ``current_weights_as_of``
    timestamp, the optional carried ``strategy_state``, and an
    optional ``forecast``. The timestamped portfolio is exactly these
    two concrete fields - there is no pseudo-type wrapping them.

    Construction validates strictly over the admission
    and alignment laws, in this order:

    1. ``decision_time`` must be an aware instant (naive datetimes
       and ``datetime.date`` reject with ``NaiveTimestampError``);
    2. every ``universe`` entry must be a non-blank exact string and
       no entry may duplicate another (``ValueError``); the
       caller-declared order is preserved and the sequence is stored
       as a tuple;
    3. ``current_weights`` must carry ``WeightState.PRE_TRADE`` - the
       decision-time snapshot role; ``TARGET`` and ``POST_TRADE``
       reject with ``ValueError``;
    4. ``current_weights_as_of`` must be an aware instant satisfying
       ``current_weights_as_of <= decision_time``
       (``InvalidChronologyError``);
    5. ``information.as_of <= decision_time``
       (``InvalidChronologyError``) - no post-decision information;
    6. a present ``forecast`` must satisfy
       ``forecast.decision_time == decision_time``
       (``InvalidChronologyError``).

    Two laws hold by construction rather than by check. The
    universe-vs-holdings law: nothing here forces the holdings keys
    of ``current_weights`` to equal ``universe`` - holdings partially
    outside the currently investable universe, and vice versa, are
    lawful. The role law's other half: this snapshot is the
    decision-time book, not the execution-time pre-trade book -
    accounting continues to drift the portfolio and determines the
    execution-time book separately.

    ``strategy_state`` is stored exactly as given and is never
    modify by this object; a stateful strategy's transitions are
    represented solely through ``DecisionResult.next_strategy_state``.
    """

    decision_time: datetime
    information: InformationSet
    universe: tuple[str, ...]
    current_weights: PortfolioWeights
    current_weights_as_of: datetime
    strategy_state: object = None
    forecast: Forecast | None = None

    def __post_init__(self) -> None:
        decision_aware = _require_aware_instant(
            self.decision_time, "decision_time"
        )
        decision_instant = _to_instant(
            decision_aware, "decision_time"
        )
        if isinstance(self.universe, str):
            raise ValueError(  # noqa: TRY004 — the rejection surface is ValueError-only, mirroring the blank-identifier law
                "universe must be a sequence of exact-string "
                "instrument identifiers in caller-declared order, not "
                f"a single string; got {self.universe!r}. A single "
                "string names one instrument, not a universe, so the "
                "value is rejected unconditionally."
            )
        normalized_universe = tuple(self.universe)
        seen_universe: set[str] = set()
        for identifier in normalized_universe:
            _require_identifier(identifier, "the context's universe entry")
            if identifier in seen_universe:
                raise ValueError(
                    f"duplicate universe entry: {identifier!r} appears "
                    "more than once in the decision context's universe "
                    "- the universe is a sequence of exact-string "
                    "instrument identifiers in caller-declared order, "
                    "so a duplicate names no additional instrument and "
                    "is rejected unconditionally."
                )
            seen_universe.add(identifier)
        object.__setattr__(self, "universe", normalized_universe)
        if self.current_weights.state != WeightState.PRE_TRADE:
            raise ValueError(
                "current-portfolio role violation: the decision "
                "context's current_weights must carry "
                "WeightState.PRE_TRADE - the decision-time snapshot "
                "of what the portfolio holds - got WeightState."
                f"{self.current_weights.state.name}. TARGET is what a "
                "strategy desires and POST_TRADE is what accounting "
                "produced; neither is a lawful decision-time "
                "holdings snapshot, so the value is rejected "
                "unconditional (the decision-context role law)."
            )
        holdings_aware = _require_aware_instant(
            self.current_weights_as_of, "current_weights_as_of"
        )
        holdings_instant = _to_instant(
            holdings_aware, "current_weights_as_of"
        )
        if holdings_instant > decision_instant:
            raise InvalidChronologyError(
                "chronology violation: the decision context admits no "
                "post-decision information, but current_weights_as_of="
                f"{holdings_instant.isoformat()} follows decision_time="
                f"{decision_instant.isoformat()}. The current-holdings "
                "snapshot must be known at or before the decision "
                "instant it is decided on."
            )
        information_instant = _to_instant(
            self.information.as_of, "the information set's as_of"
        )
        if information_instant > decision_instant:
            raise InvalidChronologyError(
                "chronology violation: the decision context admits no "
                "post-decision information, but information.as_of="
                f"{information_instant.isoformat()} follows "
                f"decision_time={decision_instant.isoformat()}. The "
                "information set must be admitted at or before the "
                "decision instant it is decided on."
            )
        if self.forecast is not None:
            forecast_instant = _to_instant(
                self.forecast.decision_time,
                "the forecast's decision_time",
            )
            if forecast_instant != decision_instant:
                raise InvalidChronologyError(
                    "alignment violation: a forecast carried by the "
                    "decision context must be dated exactly at the "
                    "context's authoritative decision_time, but "
                    "forecast.decision_time="
                    f"{forecast_instant.isoformat()} differs from "
                    f"decision_time={decision_instant.isoformat()}. A "
                    "forecast dated elsewhere belongs to a different "
                    "decision instant, so the value is rejected "
                    "unconditional."
                )


@dataclass(frozen=True)
class DecisionResult:
    """The outcome of one strategy decision.

    Exactly two facts: the ``decision`` (a ``PortfolioDecision``) and
    the ``next_strategy_state`` the strategy carries forward (``None``
    for a stateless strategy). The state law is fixed: the context's
    input ``strategy_state`` is never modified in place - every
    transition is represented solely by ``next_strategy_state``, and
    randomness affecting replay is carried explicitly through that
    state rather than hidden mutable RNG state. The cross-object rules - ``decision.decision_time == context.decision_time``
    and ``decision.execution_time >= context.decision_time`` - are
    enforced at this seam by the exported
    ``require_decision_result_compatible`` validator.
    """

    decision: PortfolioDecision
    next_strategy_state: object = None


# --------------------------------------------------------------------------- #
# Structural contracts
# --------------------------------------------------------------------------- #


class FeatureTransform(Protocol):
    """A lineage-monotone derivation of feature observations.

    A transform consumes admitted ``TimedObservation`` records and
    produces new ``TimedObservation`` records — derived features are
    observations, carrying their own ``series_id``,
    ``observation_time``, ``available_time``, and ``value``, and
    re-entering every observation rule (identity, chronology,
    admission) as such. Lineage monotonicity is enforced at this
    surface through the exported ``require_feature_lineage``
    validator, never inside the transform. Static-only: validated
    behaviorally, not by ``isinstance``. This interface ships no
    concrete transform, registry, or feature library.
    """

    def transform(
        self, observations: Iterable[TimedObservation]
    ) -> Sequence[TimedObservation]:
        """Derive feature observations from admitted input records."""


class Forecaster(Protocol):
    """A forecaster producing provenance-bearing forecasts.

    The ``information_set`` passed in has already enforced admission at
    its ``as_of``; the forecaster treats that instant as the forecast
    origin and returns a ``Forecast`` whose ``decision_time`` equals it
    (the implementer's obligation, verified in contract tests). The
    interface performs no admission of its own and accepts no
    information outside the ``InformationSet``. No architecture —
    LSTM, statistical, econometric, or otherwise — is prescribed;
    fitting-window, estimation, and state semantics belong to the
    implementer. Static-only: validated behaviorally, not by
    ``isinstance``.
    """

    def forecast(self, information_set: InformationSet) -> Forecast:
        """Produce the forecast for one admitted information set."""


class Strategy(Protocol):
    """A strategy deciding one portfolio decision from its context.

    ``decide`` consumes exactly one ``DecisionContext`` - the
    decision-time aggregate of the authoritative
    ``decision_time``, the admitted ``information``, the
    caller-declared ``universe``, the decision-time
    ``current_weights`` snapshot, the carried ``strategy_state``,
    and an optional ``forecast`` - and returns exactly one
    ``DecisionResult``. The forecast-only signature
    ``decide(forecast)`` is not a lawful alternative: the contract
    admits no compatibility shim, no dual protocol, and no
    adapter around this seam. The context's ``strategy_state``
    is never modified in place; every state transition is
    represented solely by the result's ``next_strategy_state``.
    Static-only: validated behaviorally, not by ``isinstance``.
    """

    def decide(self, context: DecisionContext) -> DecisionResult:
        """Decide the target portfolio from one decision context."""


@runtime_checkable
class RebalancePolicy(Protocol):
    """A policy deciding whether a decision instant is a rebalance.

    One pure query on aware instants: should the portfolio be
    rebalanced at ``decision_time``, given the last rebalance happened
    at ``last_rebalance_time``. No default schedule, calendar, or
    frequency is implied — scheduling policy belongs to
    implementations. Runtime-checkable because a runtime structural
    check genuinely exists on this surface.
    """

    def should_rebalance(
        self, decision_time: datetime, last_rebalance_time: datetime
    ) -> bool:
        """Answer whether the portfolio should rebalance at this instant."""


@runtime_checkable
class CostModel(Protocol):
    """A model estimating the cost of moving between two weight books.

    One method estimating the trade cost of moving from
    ``pre_trade_weights`` to ``target_weights`` as a bare float — a
    deliberately provisional typing. No functional form, units,
    currency, or turnover definition is implied; cost semantics
    belong to implementations. Runtime-checkable because a runtime
    structural check genuinely exists on this surface.
    """

    def estimate_trade_cost(
        self,
        pre_trade_weights: Mapping[str, float],
        target_weights: Mapping[str, float],
    ) -> float:
        """Estimate the cost of trading to the target weight book."""


@runtime_checkable
class AccountingEngine(Protocol):
    """An engine accounting one executed portfolio decision.

    One method consuming the decision, the pre-trade weight book, and
    the realized returns of the decision's outcome, and producing an
    ``AccountingResult``. Neither ``pre_trade_weights`` nor
    ``realized_returns`` carries timing, realization-period, or
    accounting convention at this interface — which realization
    period and which accounting rules they must correspond to is
    the implementer's to define. Runtime-checkable because a runtime
    structural check genuinely exists on this surface.
    """

    def account(
        self,
        decision: PortfolioDecision,
        pre_trade_weights: Mapping[str, float],
        realized_returns: Mapping[str, float],
    ) -> AccountingResult:
        """Account one decision into an accounting result."""


@runtime_checkable
class Evaluator(Protocol):
    """An evaluator computing metrics from accounting results.

    One method consuming an ``AccountingResult`` and returning an
    explicitly open metric mapping — no metric catalogue, statistic,
    benchmark, or significance semantics at this interface.
    Runtime-checkable because a runtime structural check genuinely
    exists on this surface.
    """

    def evaluate(
        self, accounting_result: AccountingResult
    ) -> Mapping[str, float]:
        """Compute the metric mapping for one accounting result."""


# --------------------------------------------------------------------------- #
# Compatibility validators
# --------------------------------------------------------------------------- #


def require_feature_lineage(
    inputs: Iterable[TimedObservation], outputs: Iterable[TimedObservation]
) -> None:
    """Assert no derived output is declared available before its inputs.

    Thin reuse of the lineage rule: a
    derived feature may not be declared available before the latest
    input it derives from, and a feature with an empty input collection
    has no defensible availability at all. The input availabilities
    are collected once and each output's availability is checked by
    ``require_lineage_monotone`` — the rule, the errors, and the empty-input rejection all remain the observations module's;
    nothing is redefined here.

    Raises ``FeatureLineageError`` (from ``portlearn.observations``)
    when any output's ``available_time`` precedes the latest input
    availability or when ``inputs`` is empty.
    """
    input_available_times = [item.available_time for item in inputs]
    for output in outputs:
        require_lineage_monotone(output.available_time, input_available_times)


def require_forecast_decision_compatible(
    forecast: Forecast, decision: PortfolioDecision
) -> None:
    """Assert a decision is not dated before the forecast it consumes.

    A decision made on a forecast must satisfy ``forecast.decision_time
    <= decision.decision_time`` — deciding exactly at the forecast
    origin is admissible, and deciding strictly later is admissible
    with information admitted at the later instant. The impossible
    ordering alone is forbidden: a decision dated before the
    information it consumed is look-ahead leakage. Both instants are
    aware-checked first, so naive or date-valued instants reject with
    ``NaiveTimestampError`` before any comparison.

    Raises ``InvalidChronologyError`` (from ``portlearn.timing``) on
    violation; returns ``None`` otherwise.
    """
    forecast_origin = _to_instant(
        forecast.decision_time, "the forecast's decision_time"
    )
    decision_instant = _to_instant(
        decision.decision_time, "the decision's decision_time"
    )
    if forecast_origin > decision_instant:
        raise InvalidChronologyError(
            "chronology violation: a decision cannot be dated before "
            "the forecast it consumes, but the decision's "
            f"decision_time={decision_instant.isoformat()} precedes "
            f"the forecast's decision_time="
            f"{forecast_origin.isoformat()}. Deciding on information "
            "requires the information to exist first."
        )


def require_decision_result_compatible(
    context: DecisionContext, result: DecisionResult
) -> None:
    """Assert a decision result is anchored at its decision context.

    The context's ``decision_time`` is the single authoritative
    timing anchor of the decision instant, so the returned decision
    must be dated exactly there:
    ``result.decision.decision_time == context.decision_time`` - and
    may not execute before it:
    ``result.decision.execution_time >= context.decision_time``
    (same-instant decide-and-execute is admissible). All compared
    instants are normalized with the timing-owned normalizer, so
    naive or date-valued instants reject with ``NaiveTimestampError``
    before any comparison.

    Raises ``InvalidChronologyError`` (from ``portlearn.timing``) on
    violation; returns ``None`` otherwise.
    """
    context_instant = _to_instant(
        context.decision_time, "the context's decision_time"
    )
    result_decision_instant = _to_instant(
        result.decision.decision_time,
        "the result decision's decision_time",
    )
    if result_decision_instant != context_instant:
        raise InvalidChronologyError(
            "chronology violation: a strategy's decision must be "
            "dated exactly at its decision context's authoritative "
            "decision_time, but result.decision.decision_time="
            f"{result_decision_instant.isoformat()} differs from "
            f"context.decision_time={context_instant.isoformat()}. "
            "The decision instant must not become implicit again."
        )
    execution_instant = _to_instant(
        result.decision.execution_time,
        "the result decision's execution_time",
    )
    if execution_instant < context_instant:
        raise InvalidChronologyError(
            "chronology violation: a strategy's decision cannot "
            "execute before its decision context's decision_time, "
            "but result.decision.execution_time="
            f"{execution_instant.isoformat()} precedes "
            f"context.decision_time={context_instant.isoformat()}. "
            "Same-instant decide-and-execute is admissible; a "
            "reversed order is not."
        )
