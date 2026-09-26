"""The portfolio ledger: segment-composed accounting over the wealth path.

``build_ledger`` composes the PortLearn accounting contract into one
immutable ledger value: the weight-book contract, the growth-factor
drift law over holding segments (composed through ``pl.rebalance``),
the trade, turnover, and proportional-cost laws, and the
accounting-engine interface. The result — a ``LedgerPath`` — carries
exactly one ``LedgerRow`` per accounting period in finance-table
language, with zero, one, or many ``Execution`` records per row.

The contract, in summary:

- **Segment-composed growth.** Holding time splits into segments at
  execution instants; gross growth is ``G_gross,t = Π_m D_m`` with the
  fixed-universe denominator ``D_m = Σᵢ wᵢ·gᵢ`` per segment — never a
  single denominator over a multi-execution period.
- **Multiplicative costs.** ``F_cost,t = Π_k (1 − q_k)`` under the
  cost law; ``G_net = G_gross × F_cost``; costs never enter
  additively, and ``W_{t+1} = W_t × G_net,t`` from the caller-supplied
  initial wealth ``W_0`` (default 1).
- **Half-open ownership.** An execution belongs to the accounting
  period containing its instant — ``[period_start, period_end)`` — and
  closing weights are the portfolio at ``period_end⁻``: an execution
  exactly at ``period_end`` belongs to the next row.
- **The engine law.** Every supplied ``AccountingEngine`` must return
  ``POST_TRADE = TARGET`` on every execution — anything else rejects
  strict.
- **Realized returns by holding interval.** The engine receives each
  decision's realized per-asset returns over ``[execution, next
  execution)``, the final execution ending at the ledger horizon;
  accounting boundaries subdivide that interval, never redefine it.

The module imports only PortLearn modules and the stdlib,
defines no new exception classes (the timing-law errors and
``ValueError`` carry every failure), and records no strategy,
experiment, or run identity anywhere on the record.
"""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from portlearn import (
    interfaces as _interfaces,
)
from portlearn import (
    rebalance as _rebalance,
)
from portlearn import (
    timing as _timing,
)
from portlearn import (
    trades as _trades,
)
from portlearn import (
    turnover as _turnover,
)
from portlearn import (
    weights as _weights,
)

__all__ = [
    "ExactFillAccounting",
    "Execution",
    "LedgerPath",
    "LedgerRow",
    "build_ledger",
]

_INFINITY = float("inf")

class _FrozenBook(Mapping[str, float]):
    """An immutable weight-book view (the snapshot discipline).

    Wraps a snapshot ``dict`` and exposes only the read surface of a
    ``Mapping``: item assignment and deletion are structurally
    impossible (no mutator exists), so a book placed on a ledger row
    can never be edited after the row is assembled. Value equality
    against any other ``Mapping`` (a plain ``dict``, another view) is
    by entries — deterministic value equality.
    """

    __slots__ = ("_entries",)

    def __init__(self, entries: Mapping[str, float]) -> None:
        object.__setattr__(self, "_entries", dict(entries))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"{type(self).__name__} is an immutable value object; "
            f"attribute {name!r} cannot be assigned"
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            f"{type(self).__name__} is an immutable value object; "
            f"attribute {name!r} cannot be deleted"
        )

    def __getitem__(self, key: str) -> float:
        return self._entries[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self._entries) == dict(other)
        return NotImplemented

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._entries!r})"


def _exact_total(values: Iterable[float]) -> float:
    """Compensated (Neumaier) summation — exact, order-independent
    accumulation for the budget and denominator aggregates, so a
    lawful book is never rejected or mis-measured by naive float
    accumulation order. Pure stdlib arithmetic; no rounding of the
    mathematical sum beyond the float representation itself."""
    total = 0.0
    compensation = 0.0
    for value in values:
        current = total + value
        if abs(total) >= abs(value):
            compensation += (total - current) + value
        else:
            compensation += (value - current) + total
        total = current
    return total + compensation

def _checked_instant(value: Any, role: str) -> datetime:
    """Normalize a time input through the timing module: an aware
    datetime returns its UTC instant; anything else (naive datetime,
    date, non-datetime) fails closed with
    ``NaiveTimestampError`` — the ledger defines no time coercion of
    its own."""
    return _timing.to_instant(value, role)


def _require_factor_value(asset: str, factor: Any, start: datetime) -> float:
    """A growth factor must be a non-bool finite real ``g ≥ 0`` (gross
    factors are ``1 + r``); anything else fails closed on the
    ``ValueError`` domain surface before any arithmetic runs."""
    if isinstance(factor, bool) or not isinstance(factor, (int, float)):
        raise ValueError(  # noqa: TRY004 — ValueError-only surface is the error law
            f"the growth factor on {asset!r} for the holding segment "
            f"starting at {start.isoformat()} must be a real number; "
            f"got {type(factor).__name__}: {factor!r}"
        )
    value = float(factor)
    if value != value or value in (_INFINITY, -_INFINITY):  # noqa: PLR0124 — deliberate NaN probe; math is outside the stdlib allowlist
        raise ValueError(
            f"the growth factor on {asset!r} for the holding segment "
            f"starting at {start.isoformat()} must be finite; got "
            f"{value!r} — a non-finite portfolio state is undefined "
            "and fails closed."
        )
    if value < 0.0:
        raise ValueError(
            f"growth factors are gross (g = 1 + r) and must be "
            f"nonnegative; got {value!r} on {asset!r} for the holding "
            f"segment starting at {start.isoformat()}."
        )
    return value


def _segment_denominator(
    book: Mapping[str, float],
    factors: Mapping[str, float],
    start: datetime,
) -> tuple[float, dict[str, float]]:
    """The holding-segment denominator for one segment:
    factors are supplied for exactly the held universe, each factor is
    a finite nonnegative real, and the denominator ``D = Σᵢ wᵢ·gᵢ`` is
    exact, finite, and strictly positive — the binding short/leverage
    rule. Returns ``D`` together with the validated per-asset factor
    values so the caller never re-derives them.

    By design, the
    weight *transition* itself is NOT computed here — it is composed
    through ``pl.rebalance.drift_weights``, which normalizes
    ``D`` away inside its own renormalization ``wᵢ·gᵢ/D``. The ledger
    retains this denominator calculation because the segment-composed
    gross law ``G_gross,t = Π_m D_m`` needs each segment's
    denominator as a value in its own right; without it the period's
    gross growth would be unrecoverable from the drifted weights
    alone. No drift-law formula beyond this ``D_m`` is
    restated as this module's own."""
    if set(factors) != set(book):
        raise ValueError(
            "the growth-factor universe must equal the held book's "
            f"universe exactly for the holding segment starting at "
            f"{start.isoformat()}; got factors {sorted(map(str, factors))} "
            f"for the held book {sorted(book)} — the universe is fixed "
            "within one drift interval and an unmatched supply fails "
            "closed."
        )
    checked: dict[str, float] = {
        asset: _require_factor_value(asset, factors[asset], start)
        for asset in book
    }
    terms: list[float] = []
    for asset, weight in book.items():
        product = weight * checked[asset]
        if product != product or product in (_INFINITY, -_INFINITY):  # noqa: PLR0124 — deliberate NaN probe; math is outside the stdlib allowlist
            raise ValueError(
                f"every weight-factor product must be finite; the term on "
                f"{asset!r} for the holding segment starting at "
                f"{start.isoformat()} overflowed the float range — the "
                "portfolio state is undefined, never rescaled."
            )
        terms.append(product)
    denominator = _exact_total(terms)
    if denominator != denominator or denominator in (_INFINITY, -_INFINITY):  # noqa: PLR0124 — deliberate NaN probe; math is outside the stdlib allowlist
        raise ValueError(
            "the portfolio-growth denominator D = Σᵢ wᵢ·gᵢ must be "
            f"finite for the holding segment starting at "
            f"{start.isoformat()}; got a non-finite sum."
        )
    if denominator <= 0.0:
        raise ValueError(
            "the portfolio-growth denominator D = Σᵢ wᵢ·gᵢ must be "
            "strictly positive and finite for the holding segment "
            f"starting at {start.isoformat()} — the binding short/"
            f"leverage rule; got D={denominator!r}. The portfolio state "
            "is undefined and fails closed."
        )
    return denominator, checked


def _drift(
    book: "_weights.PortfolioWeights",
    factors: Mapping[str, float],
) -> "_weights.PortfolioWeights":
    """Drift the held book across one holding segment — the derived
    book of the segment walk, never a supplied book. The transition is
    COMPOSED through the drift function
    ``portlearn.rebalance.drift_weights``: no local
    ``wᵢ·gᵢ/D`` arithmetic remains in this
    module. The internal book is carried as a ``PortfolioWeights``
    value object (POST_TRADE at birth from the caller's initial
    mapping, PRE_TRADE after every drift), so a TARGET can never be
    drifted — ``drift_weights`` itself enforces that law."""
    return _rebalance.drift_weights(book, factors)

def _checked_weight(asset: str, weight: Any, role: str) -> float:
    """A supplied weight must be a non-bool finite real; signed weights
    (shorts) and explicit zeros are lawful — only the representation
    fails closed here (the budget law is applied to the whole book)."""
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError(  # noqa: TRY004 — ValueError-only surface is the error law
            f"the {role} weight on {asset!r} must be a real number; "
            f"got {type(weight).__name__}: {weight!r}"
        )
    value = float(weight)
    if value != value or value in (_INFINITY, -_INFINITY):  # noqa: PLR0124 — deliberate NaN probe; math is outside the stdlib allowlist
        raise ValueError(
            f"the {role} weight on {asset!r} must be finite; got "
            f"{value!r} — a non-finite portfolio book is undefined and "
            "fails closed."
        )
    return value


def _require_unit_budget(book: Mapping[str, float], role: str) -> None:
    """The wealth path is a fully represented unit-NAV account —
    every supplied book on it (the initial book and every target) must
    carry budget exactly 1.0. A budget-0.5 book is not silently read
    as a 50% loss and cash/financing cannot hide off the book: both
    reject unconditionally. Fully invested signed long/short books
    summing exactly to 1 remain admissible."""
    total = _exact_total(book.values())
    if total != 1.0:
        raise ValueError(
            f"the {role} budget must be exactly 1.0 for the unit-NAV "
            f"wealth path; got {total!r} — a non-unit book is not "
            "silently rescaled and fails closed. Cash and financing "
            "positions must be explicit instruments on the budget-1.0 "
            "book."
        )


def _supplied_book(value: Any, role: str) -> dict[str, float]:
    """Admit a supplied weight book (the initial book or a decision's
    target): a mapping of identifiers to finite real weights with the
    unit-NAV budget enforced — the unit-NAV check on every supplied
    book."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Mapping):
        raise ValueError(  # noqa: TRY004 — ValueError-only surface is the error law
            f"the {role} weights must be a mapping of asset identifiers "
            f"to weights; got {type(value).__name__}: {value!r}"
        )
    book = {
        asset: _checked_weight(asset, weight, role)
        for asset, weight in value.items()
    }
    _require_unit_budget(book, role)
    return book

@dataclass(frozen=True)
class Execution:
    """One executed decision retained on a row.

    The full audit primitive: when it was decided, when it executed,
    the desired target, the drifted pre-trade book it ran from, the
    canonical weight-space trade, the turnover measure and cost
    fraction under the cost model's declared convention, and the
    post-trade book — which the engine law pins to the target exactly
    (POST_TRADE = TARGET, re-verified). An immutable value
    object with deterministic value equality; no hash guarantee.
    """

    decision_time: datetime
    execution_time: datetime
    target_weights: Mapping[str, float]
    pre_trade_weights: Mapping[str, float]
    trade: Any
    turnover: float
    transaction_cost: float
    post_trade_weights: Mapping[str, float]


@dataclass(frozen=True)
class LedgerRow:
    """One accounting period of the wealth path (finance table).

    ``period_start``/``period_end`` bound the half-open ownership
    interval ``[period_start, period_end)``; ``executions`` is the
    zero/one/many tuple in execution order; ``closing_weights`` is the
    portfolio at ``period_end⁻``; ``gross_return``/``net_return`` are
    simple returns under the segment-composed and cost-composed
    identities; ``turnover`` sums the execution turnovers;
    ``transaction_cost`` is ``1 − F_cost,t``; ``wealth_open``/
    ``wealth_close`` carry the unit-NAV account value at the period
    boundaries; ``universe`` is the union of identifiers across ALL
    retained books of the row. No strategy, experiment, or run
    identity exists on the record.
    """

    period_start: datetime
    period_end: datetime
    opening_weights: Mapping[str, float]
    executions: tuple[Execution, ...]
    closing_weights: Mapping[str, float]
    gross_return: float
    net_return: float
    turnover: float
    transaction_cost: float
    wealth_open: float
    wealth_close: float
    universe: frozenset[str]


@dataclass(frozen=True)
class LedgerPath:
    """The complete record of one portfolio path.

    An immutable value object — not a live cursor — so replay is
    trivially exact: identical inputs produce a value-equal path. The
    engine that produced it is not part of the value; only the rows
    and the path-level summary are.
    """

    rows: tuple[LedgerRow, ...]
    final_wealth: float
    n_periods: int


class ExactFillAccounting:
    """The reference accounting engine: exact fill, POST_TRADE =
    TARGET.

    Accounts one executed decision by returning the decision's target
    book as the post-trade book — the execution identity,
    with no execution realism (partial fills, refusals) applied. The
    ``realized_returns`` argument is received and deliberately not
    consumed: under exact fill the post-trade book is the target
    regardless of the outcome, so the reference engine is pure with
    respect to it.
    """

    __slots__ = ()

    def account(
        self,
        decision: Any,
        pre_trade_weights: Any,
        realized_returns: Any,
    ) -> Any:
        return _interfaces.AccountingResult(dict(decision.target_weights))

def _decision_instants(decision: Any) -> tuple[datetime, datetime]:
    """Read a decision's ``decision_time``/``execution_time`` through
    the timing module (duck-typed, the event-sequence rule):
    both must be aware instants and the trade may not execute before
    it is decided — same-instant decide-and-execute is admissible."""
    decision_time = getattr(decision, "decision_time", None)
    execution_time = getattr(decision, "execution_time", None)
    if decision_time is None or execution_time is None:
        raise ValueError(
            "every decision must carry 'decision_time' and "
            "'execution_time' attributes; got "
            f"{type(decision).__name__}: {decision!r}"
        )
    decided = _checked_instant(decision_time, "decision_time")
    executes = _checked_instant(execution_time, "execution_time")
    if executes < decided:
        raise _timing.InvalidChronologyError(
            "chronology violation: a trade cannot execute before its "
            "decision is made, but execution_time precedes "
            f"decision_time; got decision_time={decided.isoformat()}, "
            f"execution_time={executes.isoformat()}. Same-instant "
            "decide-and-execute is admissible; a reversed order is not."
        )
    return decided, executes


def _turnover_measure(cost_model: Any):
    """The cost model's bound turnover convention, by its stable
    canonical identity — the record carries the same convention the
    cost law binds, never a re-derived or defaulted one."""
    convention = getattr(cost_model, "convention", None)
    if convention == "one_way":
        return _turnover.one_way
    if convention == "two_sided":
        return _turnover.two_sided
    raise ValueError(
        "the cost model must bind one of PortLearn's recognized named "
        "turnover conventions ('one_way' or 'two_sided') so the "
        "execution record can carry it; got "
        f"{type(cost_model).__name__} with convention {convention!r}."
    )

def build_ledger(
    initial_weights: Mapping[str, float],
    accounting_instants: Iterable[Any],
    growth_factors: Mapping[Any, Mapping[str, float]],
    decisions: Iterable[Any],
    cost_model: Any,
    accounting_engine: Any = None,
    initial_wealth: float = 1.0,
) -> LedgerPath:
    """Compose the accounting contract into one immutable ledger value.

    Walks the wealth path period by period, splitting holding time
    into segments at execution instants: gross growth composes the
    per-segment denominators ``G_gross,t = Π_m D_m``, costs compose
    ``F_cost,t = Π_k (1 − q_k)`` under the cost law, and the
    unit-NAV account recurses ``W_{t+1} = W_t × G_net,t`` from the
    caller-supplied initial wealth ``initial_wealth`` (default 1).

    ``initial_wealth`` is the account's opening wealth ``W_0`` and is
    SEPARATE from the unit-NAV weight budget: the budget
    law is a *weight* identity — every book on the wealth path must
    carry budget exactly 1.0, whatever currency or scale the account
    is denominated in — while ``initial_wealth`` only scales the
    reported ``wealth_open``/``wealth_close``/``final_wealth`` values.
    A positive finite real; anything else (0, negative, NaN, ±inf,
    bool, non-real) is rejected with ``ValueError``.

    The cost model is the ``Proportional`` model:
    ``build_ledger`` composes it through ``pl.costs`` and accepts no
    substitute. Custom or duck-typed cost models are OUTSIDE the
    ledger public contract.
    """
    engine = accounting_engine if accounting_engine is not None else ExactFillAccounting()
    measure = _turnover_measure(cost_model)

    if isinstance(accounting_instants, (str, bytes)) or not isinstance(
        accounting_instants, Iterable
    ):
        raise ValueError(  # noqa: TRY004 — ValueError-only surface is the error law
            "accounting_instants must be an iterable of aware instants "
            f"bounding the accounting periods; got "
            f"{type(accounting_instants).__name__}: {accounting_instants!r}"
        )
    boundaries = [
        _checked_instant(instant, "accounting instant")
        for instant in accounting_instants
    ]
    if len(boundaries) < 2:
        raise ValueError(
            "accounting_instants must carry at least a horizon pair "
            f"(start, end); got {len(boundaries)} instant(s)"
        )
    for earlier, later in zip(boundaries, boundaries[1:]):  # noqa: RUF007 — itertools is outside the stdlib allowlist
        if not earlier < later:
            raise _timing.InvalidChronologyError(
                "chronology violation: accounting instants must be "
                "strictly increasing — two periods may not overlap or "
                "share a boundary instant; got "
                f"{later.isoformat()} not strictly after "
                f"{earlier.isoformat()}."
            )
    horizon_start, horizon_end = boundaries[0], boundaries[-1]

    initial = _supplied_book(initial_weights, "initial")

    if isinstance(initial_wealth, bool) or not isinstance(
        initial_wealth, (int, float)
    ):
        raise ValueError(  # noqa: TRY004 — ValueError-only surface is the error law
            "initial_wealth must be a real number, not "
            f"{type(initial_wealth).__name__}: {initial_wealth!r}"
        )
    if initial_wealth != initial_wealth or initial_wealth in (  # noqa: PLR0124 — deliberate NaN probe; math is outside the stdlib allowlist
        _INFINITY,
        -_INFINITY,
    ):
        raise ValueError(
            "initial_wealth must be finite; got "
            f"{initial_wealth!r} — a non-finite account value is "
            "undefined and fails closed."
        )
    if initial_wealth <= 0:
        raise ValueError(
            "initial_wealth must be strictly positive; got "
            f"{initial_wealth!r} — the wealth path is an account-value "
            "recursion and has no zero or negative starting point."
        )
    wealth = float(initial_wealth)

    if isinstance(decisions, (str, bytes)):
        raise ValueError(  # noqa: TRY004 — ValueError-only surface is the error law
            "decisions must be an iterable of decisions, not a single "
            f"{type(decisions).__name__}: {decisions!r}"
        )
    events: list[dict[str, Any]] = []
    previous_execution: datetime | None = None
    for decision in decisions:
        decided, executes = _decision_instants(decision)
        if previous_execution is not None and not previous_execution < executes:
            raise _timing.InvalidChronologyError(
                "chronology violation: execution instants must be "
                "strictly increasing across distinct decisions — two "
                "executions cannot share one instant with no ordering "
                f"semantics; got {executes.isoformat()} not strictly "
                f"after {previous_execution.isoformat()}."
            )
        previous_execution = executes
        if executes < horizon_start or executes >= horizon_end:
            raise _timing.InvalidChronologyError(
                "chronology violation: every execution must fall inside "
                "the ledger horizon "
                f"[{horizon_start.isoformat()}, "
                f"{horizon_end.isoformat()}); got "
                f"{executes.isoformat()} — an execution outside the "
                "horizon has no accounting period to belong to."
            )
        events.append(
            {
                "decision": decision,
                "decision_time": decided,
                "execution_time": executes,
                "target": _supplied_book(
                    decision.target_weights, "decision target"
                ),
            }
        )

    if isinstance(growth_factors, (str, bytes)) or not isinstance(
        growth_factors, Mapping
    ):
        raise ValueError(  # noqa: TRY004 — ValueError-only surface is the error law
            "growth_factors must be a mapping of segment-start instant "
            "to per-asset gross factors; got "
            f"{type(growth_factors).__name__}: {growth_factors!r}"
        )
    supplies: dict[datetime, Mapping[str, float]] = {}
    for key, factors in growth_factors.items():
        start = _checked_instant(key, "growth_factors key")
        if start in supplies:
            raise ValueError(
                "growth_factors must supply exactly one factor book per "
                f"segment-start instant; {start.isoformat()} is supplied "
                "more than once."
            )
        if isinstance(factors, (str, bytes)) or not isinstance(factors, Mapping):
            raise ValueError(  # noqa: TRY004 — ValueError-only surface is the error law
                "the growth factors supplied for the segment starting at "
                f"{start.isoformat()} must be a mapping of asset "
                f"identifiers to gross factors; got "
                f"{type(factors).__name__}: {factors!r}"
            )
        supplies[start] = factors

    def segment_factors(start: datetime) -> Mapping[str, float]:
        if start not in supplies:
            raise ValueError(
                "growth_factors must supply the gross factors for every "
                f"holding segment; none is supplied for the segment "
                f"starting at {start.isoformat()}."
            )
        return supplies[start]

    def segment_growth(
        book: "_weights.PortfolioWeights", start: datetime
    ) -> tuple[float, "_weights.PortfolioWeights"]:
        # Design split: the denominator D_m
        # is retained locally because G_gross,t = Π_m D_m needs it as a
        # value (drift_weights normalizes D away); the weight transition
        # itself is composed — never restated — through the drift
        # module itself.
        factors = segment_factors(start)
        denominator, _ = _segment_denominator(book.weights, factors, start)
        return denominator, _drift(book, factors)

    def realized_over(
        target: dict[str, float], start: datetime, end: datetime
    ) -> dict[str, float]:
        """Per-asset compounded gross growth of the held target over
        ``[start, end)``: every factor supply at ``start`` and at
        each accounting instant strictly inside the interval
        compounds; accounting boundaries subdivide the interval,
        never redefine it (interval start included, interval end
        excluded). The per-asset
        product starts at ``1.0`` — realized growth is a per-asset
        factor product over the interval, independent of weights."""
        universe = set(target)
        compounded = {asset: 1.0 for asset in universe}
        walking = True
        while walking:
            factors = segment_factors(start)
            _, checked = _segment_denominator(target, factors, start)
            compounded = {
                asset: value * checked[asset] for asset, value in compounded.items()
            }
            following = [
                instant
                for instant in boundaries
                if start < instant < end
            ]
            if following:
                start = following[0]
            else:
                walking = False
        return {asset: value - 1.0 for asset, value in compounded.items()}

    rows: list[LedgerRow] = []
    # The internal book is carried as a PortfolioWeights value object —
    # POST_TRADE at birth (the initial book, exactly executed), PRE_TRADE
    # after every composed drift — so a TARGET can never be drifted:
    # pl.rebalance.drift_weights enforces that rule.
    book = _weights.PortfolioWeights(
        initial, _weights.WeightState.POST_TRADE
    )
    next_event = 0

    for period_start, period_end in zip(boundaries, boundaries[1:]):  # noqa: RUF007 — itertools is outside the stdlib allowlist
        opening = _FrozenBook(book.weights)
        row_executions: list[Execution] = []
        identifiers: set[str] = set(book.weights)
        segment_start = period_start
        gross_factor = 1.0
        cost_factor = 1.0
        turnover_total = 0.0

        while (
            next_event < len(events)
            and period_start <= events[next_event]["execution_time"] < period_end
        ):
            event = events[next_event]
            target = event["target"]
            if segment_start < event["execution_time"]:
                denominator, drifted = segment_growth(book, segment_start)
                gross_factor *= denominator
                book = drifted
            pre_trade = _FrozenBook(book.weights)
            identifiers.update(target)
            identifiers.update(book.weights)
            trade = _trades.from_weights(
                _weights.PortfolioWeights(
                    book.weights, _weights.WeightState.PRE_TRADE
                ),
                _weights.PortfolioWeights(
                    target, _weights.WeightState.TARGET
                ),
            )
            turnover_value = measure(trade)
            # The record's uniform convention: every transaction_cost on
            # the record (execution and row) is one minus the matching
            # cost factor — here 1 − F_cost,k, identical to q_k under
            # the identity F_cost,k = 1 − q_k, computed through the
            # cost model.
            execution_cost = 1.0 - cost_model.f_cost(trade)
            if next_event + 1 < len(events):
                interval_end = events[next_event + 1]["execution_time"]
            else:
                interval_end = horizon_end
            realized = realized_over(
                dict(target), event["execution_time"], interval_end
            )
            result = engine.account(event["decision"], pre_trade, realized)
            post_trade_raw = getattr(result, "post_trade_weights", None)
            if not isinstance(post_trade_raw, Mapping):
                raise ValueError(  # noqa: TRY004 — ValueError-only surface is the error law
                    "the accounting engine must return an AccountingResult "
                    "carrying a post_trade_weights mapping; got "
                    f"{type(result).__name__}: {result!r}"
                )
            post_trade = dict(post_trade_raw)
            if post_trade != target:
                raise ValueError(
                    "the accounting engine must return POST_TRADE = "
                    "TARGET — the post-trade book must equal the "
                    "decision's target weights exactly; got "
                    f"{post_trade!r} for target {target!r}. Execution "
                    "realism (partial fills, refusals) is out of scope "
                    "for this ledger and fails closed."
                )
            row_executions.append(
                Execution(
                    decision_time=event["decision_time"],
                    execution_time=event["execution_time"],
                    target_weights=_FrozenBook(target),
                    pre_trade_weights=pre_trade,
                    trade=trade,
                    turnover=turnover_value,
                    transaction_cost=execution_cost,
                    post_trade_weights=_FrozenBook(post_trade),
                )
            )
            cost_factor *= 1.0 - execution_cost
            turnover_total += turnover_value
            # The executed book IS the target — carried POST_TRADE (the
            # engine identity above already verified POST_TRADE = TARGET),
            # never a state the drift law may be handed as a desire.
            book = _weights.PortfolioWeights(
                target, _weights.WeightState.POST_TRADE
            )
            segment_start = event["execution_time"]
            next_event += 1

        denominator, drifted = segment_growth(book, segment_start)
        gross_factor *= denominator
        book = drifted
        closing = _FrozenBook(book.weights)
        identifiers.update(book.weights)

        net_factor = gross_factor * cost_factor
        wealth_close = wealth * net_factor
        rows.append(
            LedgerRow(
                period_start=period_start,
                period_end=period_end,
                opening_weights=opening,
                executions=tuple(row_executions),
                closing_weights=closing,
                gross_return=gross_factor - 1.0,
                net_return=net_factor - 1.0,
                turnover=turnover_total + 0.0,
                transaction_cost=1.0 - cost_factor,
                wealth_open=wealth,
                wealth_close=wealth_close,
                universe=frozenset(identifiers),
            )
        )
        wealth = wealth_close

    return LedgerPath(
        rows=tuple(rows), final_wealth=wealth, n_periods=len(rows)
    )
