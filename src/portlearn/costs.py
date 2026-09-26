"""Proportional transaction costs, convention-bound and strict.

This module defines the proportional cost model: the
standard linear cost model ``Proportional(rate, turnover=...)`` bound
to a **named turnover convention**, because a rate statement such as
"25 bps transaction cost" is incomplete unless the library also
records whether the rate applies to ``one_way`` or ``two_sided``
turnover. The convention is a first-class binding input — a bare
scalar turnover without a named convention is an incomplete cost
specification.

The laws, in summary:

- **Convention binding.** The keyword is exactly ``turnover`` (never
  alternated publicly with ``turnover_convention``) and accepts only
  PortLearn's two recognized turnover conventions,
  ``portlearn.turnover.one_way`` and
  ``portlearn.turnover.two_sided``; arbitrary lambdas or unknown
  callables reject invalid input. Internally the model preserves a
  stable canonical identity — ``"one_way"`` / ``"two_sided"`` — so
  future experiment provenance (e.g. YAML ``costs: {model:
  proportional, rate: 0.0025, turnover: one_way}``) can represent the
  convention without redesigning the public API. No registries, no
  plugin machinery: the two conventions of the turnover module are
  the only selectable measures.
- **Cost fraction and factor.** ``q = rate ×
  selected_turnover_measure(trade)`` and ``F_cost = 1 − q``; the
  domain is ``0 ≤ q < 1``, enforced strict on non-finite,
  negative, or ``q ≥ 1`` inputs (``F_cost = 0`` or negative is not a
  lawful portfolio state). The domain check is the safety net
  especially for leveraged/short books, whose two-sided turnover may
  exceed 2 and push ``q`` to 1 even at moderate rates.
- **Rate law.** ``rate ≥ 0`` real finite; negative, non-real
  (``bool``/``Decimal``/string/complex), and non-finite rates reject;
  ``rate = 0`` is lawful (the frictionless model).
- **Baseline cost-to-weights separation.** Under this declared
  simplified baseline convention: weights track — ``POST_TRADE =
  TARGET`` exactly, the rebalancing law unaltered; value track
  — multiplied by ``F_cost``. Costs multiply value, never weights.
  Cash financing, execution-level fee funding, partial fills,
  spreads, slippage, market impact, and share-space execution are
  outside this baseline and would alter it. The model is timeless:
  no execution-timestamp binding, accounting-period assignment,
  ledger row placement, or wealth path (downstream execution
  accounting owns those).
- **Composition and purity.** Over a trade sequence the total cost
  factor is ``F_cost,total = Π_k F_cost,k``, and where a growth basis
  is needed ``G_after_cost = G_before_cost × F_cost,total``; no
  ``r_net`` return-record form is defined here. Models are pure with
  respect to the trade: evaluating ``q``/``f_cost`` never modify or
  consumes it, so one stored trade can be evaluated under many cost
  models (rates 0.0010/0.0025/0.0050, say) without rerunning the
  optimizer.

The error surface is ``ValueError`` only (plus ``TypeError`` for a
missing required constructor argument); numerics follow the
checked-real discipline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from portlearn import turnover as _turnover
from portlearn.trades import WeightTrade, from_weights
from portlearn.weights import PortfolioWeights, WeightState

__all__ = ["Proportional"]

# The closed convention registry of record: the two named measures of
# the turnover module, keyed by their canonical identity strings. No
# plugin machinery — this fixed mapping is the entire selectable set.
_RECOGNIZED_CONVENTIONS: dict[str, Any] = {
    "one_way": _turnover.one_way,
    "two_sided": _turnover.two_sided,
}


def _require_rate(value: object) -> float:
    """The rate law: a finite real ≥ 0; ``bool`` is not a number; no
    clipping or normalization — out-of-domain inputs reject."""
    if isinstance(value, bool):
        raise ValueError(  # noqa: TRY004 — bool rejection is law, not type discipline
            f"rate must be a real number, not bool; got {value!r}"
        )
    if not isinstance(value, (int, float)):
        raise ValueError(  # noqa: TRY004 — ValueError-only surface is the error law
            f"rate must be a real number; got {type(value).__name__}: "
            f"{value!r}. Use float(rate)."
        )
    try:
        converted = float(value)
    except OverflowError:
        converted = math.inf
    if not math.isfinite(converted):
        raise ValueError(f"rate must be finite; got {value!r}")
    if converted < 0.0:
        raise ValueError(f"rate must be nonnegative; got {converted!r}")
    return converted


def _require_recognized_convention(value: object) -> tuple[str, Any]:
    """Only PortLearn's two named turnover conventions are selectable;
    arbitrary lambdas/callables/scalars reject unconditionally."""
    for identity, measure in _RECOGNIZED_CONVENTIONS.items():
        if value is measure:
            return identity, measure
    raise ValueError(
        "the turnover convention must be one of PortLearn's recognized "
        f"named conventions (portlearn.turnover.one_way or "
        f"portlearn.turnover.two_sided); got {value!r}. A bare scalar or "
        "arbitrary callable is an incomplete cost specification and "
        "fails closed."
    )


def _require_weight_mapping(value: object, role: str) -> Mapping[str, float]:
    """A weight book operand must be a mapping of identifiers to
    weights; anything else rejects unconditionally on the ValueError-only
    surface before book construction is attempted."""
    if not isinstance(value, Mapping):
        raise ValueError(  # noqa: TRY004 — ValueError-only surface is the error law
            f"the {role} weights must be a mapping of asset identifiers "
            f"to weights; got {type(value).__name__}: {value!r}"
        )
    return value


class Proportional:
    """The convention-bound proportional cost model.

    Public shape: constructor ``Proportional(rate,
    turnover)`` — both required, the keyword name exactly ``turnover``;
    public read attributes ``rate: float`` and ``convention: str`` (the
    stable canonical identity ``"one_way"``/``"two_sided"``); methods
    ``q(trade) -> float``, ``f_cost(trade) -> float``, and
    ``estimate_trade_cost(pre_trade_weights, target_weights) -> float``
    (the public ``CostModel`` protocol surface: it builds the same
    canonical ``WeightTrade`` from the two weight mappings —
    ``PRE_TRADE`` and ``TARGET`` books under the same validation —
    and returns ``q`` of it, with no duplicated turnover arithmetic).
    Immutable
    from the instant it exists; pure with respect to every trade it
    evaluates.
    """

    __slots__ = ("_convention", "_measure", "_rate")

    def __init__(self, rate: object, turnover: object) -> None:
        # Both inputs are first-class binding inputs: neither may be
        # omitted — a bare rate is as incomplete as a missing one.
        # (A missing argument is Python's own TypeError; an explicitly
        # supplied non-convention rejects as a ValueError domain
        # failure below, keeping the domain error surface ValueError-only.)
        checked_rate = _require_rate(rate)
        identity, measure = _require_recognized_convention(turnover)
        object.__setattr__(self, "_rate", checked_rate)
        object.__setattr__(self, "_convention", identity)
        object.__setattr__(self, "_measure", measure)

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

    @property
    def rate(self) -> float:
        """The nonnegative finite real proportional rate."""
        return self._rate

    @property
    def convention(self) -> str:
        """The stable canonical convention identity string."""
        return self._convention

    def q(self, trade: WeightTrade) -> float:
        """The cost fraction ``q = rate × turnover_measure(trade)``.

        Domain-checked ``0 ≤ q < 1``: a non-finite,
        negative, or ``q ≥ 1`` result rejects — ``F_cost = 0`` or
        negative is not a lawful portfolio state, and leveraged books
        can reach the boundary even at moderate rates. Pure with
        respect to the trade.
        """
        measure_q = self._checked_measure(trade)
        fraction = self._rate * measure_q
        if not math.isfinite(fraction):
            raise ValueError(
                f"the cost fraction q must be finite; got {fraction!r} "
                f"(rate {self._rate!r} × {self._convention} turnover "
                f"{measure_q!r})"
            )
        if fraction < 0.0:
            raise ValueError(
                f"the cost fraction q must be nonnegative; got {fraction!r}"
            )
        if fraction >= 1.0:
            raise ValueError(
                f"the cost fraction q must satisfy q < 1 (F_cost > 0); "
                f"got q = {fraction!r} (rate {self._rate!r} × "
                f"{self._convention} turnover {measure_q!r}) — F_cost = 0 "
                "or negative is not a lawful portfolio state and fails "
                "closed."
            )
        return fraction

    def f_cost(self, trade: WeightTrade) -> float:
        """The cost factor ``F_cost = 1 − q`` on the domain ``0 <
        F_cost ≤ 1``; identical domain checks."""
        return 1.0 - self.q(trade)

    def estimate_trade_cost(
        self,
        pre_trade_weights: Mapping[str, float],
        target_weights: Mapping[str, float],
    ) -> float:
        """Estimate the cost of trading to the target weight book.

        The public ``CostModel`` surface over raw weight mappings:
        constructs the canonical ``WeightTrade`` with exactly the
        ``from_weights`` semantics (books validated as ``PRE_TRADE``
        and ``TARGET``, deltas over the union of assets) and returns
        ``q`` of that trade — the bound convention is applied through
        the canonical path, with no turnover arithmetic duplicated
        here. Domain checks are therefore identical
        to ``q``'s.
        """
        trade = from_weights(
            PortfolioWeights(
                _require_weight_mapping(pre_trade_weights, "pre-trade"),
                WeightState.PRE_TRADE,
            ),
            PortfolioWeights(
                _require_weight_mapping(target_weights, "target"),
                WeightState.TARGET,
            ),
        )
        return self.q(trade)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(rate={self._rate!r}, "
            f"turnover={self._convention!r})"
        )

    def _checked_measure(self, trade: WeightTrade) -> float:
        """The bound convention evaluated on the trade, unconditional on
        non-finite turnover intermediates (ValueError-only surface)."""
        if not isinstance(trade, WeightTrade):
            raise ValueError(  # noqa: TRY004 — ValueError-only surface is the error law
                "the cost model operates on a WeightTrade value object; "
                f"got {type(trade).__name__}: {trade!r}. Construct one "
                "with portlearn.trades.from_weights(pre_trade, target)."
            )
        measured = self._measure(trade)
        if not math.isfinite(measured):
            raise ValueError(
                f"the {self._convention} turnover must be finite; got "
                f"{measured!r} — a non-finite turnover intermediate is "
                "undefined and fails closed."
            )
        if measured < 0.0:
            raise ValueError(
                f"the {self._convention} turnover must be nonnegative; "
                f"got {measured!r}"
            )
        return measured
