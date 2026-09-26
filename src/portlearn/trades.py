"""Weight-space trade transitions: the ``WeightTrade`` primitive.

This module defines the weight-space trade transition: the
``WeightTrade`` value object, constructed only through
``from_weights(pre_trade, target)``, is the explicit ``PRE_TRADE →
TARGET`` transition with ``Δw = TARGET − PRE_TRADE`` aligned **by
asset identifier, never mapping order**.

The laws, in summary:

- **State pair.** The source is a realized ``PRE_TRADE`` book and the
  destination is a ``TARGET``; any other state pairing fails closed.
  Both operands satisfy the basic ``PortfolioWeights``
  representation contract — finite, real, well-formed mappings — and
  the pre-trade book is **never revalidated against TARGET
  constraints**: rebalancing drift may legitimately displace
  PRE_TRADE weights outside the target's allocation limits, and the
  trade from that displaced book is lawful.
- **Identifier alignment over the union of assets.** The trade
  universe is ``U = assets(PRE_TRADE) ∪ assets(TARGET)`` and
  ``Δwᵢ = w_target(i) − w_pre(i)`` for every ``i`` in ``U``, using
  the effective weight ``0.0`` for an identifier absent from a book
  (the weights module's lookup rule). Absence is never redefined as
  an explicit zero: zero is effective-for-calculation only, and the
  retained books keep their original asset sets. Differences are
  computed per asset identifier, so mapping insertion order can
  never change the trade.
- **Sign semantics.** ``Δwᵢ > 0`` buys, ``Δwᵢ < 0`` sells, ``Δwᵢ = 0``
  holds. Deltas are checked finite strict — individually finite
  weights whose difference overflows the float range reject, never
  rescale.
- **Retention.** The object retains the PRE_TRADE weights, the
  TARGET weights, and the delta mapping, and is never reduced to a
  turnover scalar, so alternative turnover conventions stay
  computable on the same realized trade without rerunning the
  optimizer.
- **Optimizer-agnostic and timeless.** No optimizer identity belongs
  to the object — every TARGET-generating method terminates in the
  same TARGET contract — and the object carries no execution
  timestamp, accounting period, ledger placement, or wealth path;
  those bindings belong to downstream execution accounting, not to
  this weight-space value.

The error surface is ``ValueError`` only; the module is stdlib-only
and reuses the numeric discipline (checked reals, guarded
summation) of the weight and rebalancing modules.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any

from portlearn.weights import PortfolioWeights, WeightState

__all__ = ["WeightTrade", "from_weights"]


def _require_book(value: Any, *, expected: WeightState, role: str) -> PortfolioWeights:
    """A trade operand must be a ``PortfolioWeights`` value object in
    exactly the expected state; anything else fails closed."""
    if not isinstance(value, PortfolioWeights):
        raise ValueError(  # noqa: TRY004 — ValueError-only surface is the error law
            f"the trade {role} must be a PortfolioWeights value object; "
            f"got {type(value).__name__}: {value!r}"
        )
    if value.state is not expected:
        raise ValueError(
            f"the trade {role} must be in state {expected.value}; got "
            f"{value.state.value if isinstance(value.state, WeightState) else value.state!r}. "
            "A trade runs from a realized PRE_TRADE book into a TARGET "
            "desire — execute_rebalance produces the POST_TRADE book, and "
            "a TARGET is not a portfolio that can hold a pre-trade position."
        )
    return value


class WeightTrade:
    """The immutable ``PRE_TRADE → TARGET`` weight-space transition.

    Public shape: constructor-of-record is
    ``from_weights(pre_trade, target)`` (direct construction applies
    the same validation); public read attributes ``source`` (the
    ``PRE_TRADE`` ``PortfolioWeights``), ``destination`` (the
    ``TARGET`` ``PortfolioWeights``), and ``delta`` — the immutable
    mapping view of ``Δw = TARGET − PRE_TRADE`` per asset identifier
    over the union of the two asset sets (effective weight ``0.0``
    for an identifier absent from a book), buys positive, sells
    negative. The object is a timeless
    weight-space value: equality and hashing are by value (states
    plus weight mappings, order-insensitive), it is immutable from
    the instant it exists, and it retains everything needed to
    recompute any turnover convention later.
    """

    __slots__ = ("_delta", "destination", "source")

    def __init__(
        self, pre_trade: PortfolioWeights, target: PortfolioWeights
    ) -> None:
        """Construct the transition with full strict validation:
        state pair, union-of-assets delta law, and per-identifier
        finite deltas — the same laws ``from_weights`` applies, so no
        inconsistent instance can ever exist."""
        pre = _require_book(pre_trade, expected=WeightState.PRE_TRADE, role="source")
        post = _require_book(target, expected=WeightState.TARGET, role="destination")
        delta: dict[str, float] = {}
        # Union of assets: an identifier absent from one book
        # participates with its effective weight 0.0 (the weights
        # module's lookup rule). Zero is effective-for-calculation
        # only — the retained books below keep their original asset
        # sets, so absence is never redefined as an explicit zero.
        universe = pre.assets | post.assets
        for asset in universe:
            difference = post.weight_of(asset) - pre.weight_of(asset)
            if not math.isfinite(difference):
                raise ValueError(
                    f"the weight delta on {asset!r} must be finite; got "
                    f"{difference!r} — the transition between individually "
                    "finite weights overflowed the float range and the trade "
                    "state is undefined, never rescaled."
                )
            delta[asset] = difference + 0.0  # normalize −0.0 to +0.0
        object.__setattr__(self, "source", pre)
        object.__setattr__(self, "destination", post)
        # Only the immutable MappingProxyType view is retained; no
        # mutable backing dict alias exists on the object.
        object.__setattr__(self, "_delta", MappingProxyType(delta))

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
    def delta(self) -> MappingProxyType[str, float]:
        """``Δw = TARGET − PRE_TRADE`` per asset identifier."""
        return self._delta  # type: ignore[no-any-return]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WeightTrade):
            return NotImplemented
        return (
            self.source.state is other.source.state
            and self.destination.state is other.destination.state
            and dict(self.source.weights) == dict(other.source.weights)
            and dict(self.destination.weights) == dict(other.destination.weights)
            and dict(self._delta) == dict(other._delta)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.source.state,
                self.destination.state,
                tuple(sorted(self.source.weights.items())),
                tuple(sorted(self.destination.weights.items())),
            )
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(source=PRE_TRADE, destination=TARGET, "
            f"delta={dict(self._delta)!r})"
        )


def from_weights(pre_trade: Any, target: Any) -> WeightTrade:
    """Construct the ``PRE_TRADE → TARGET`` trade, ``Δw = TARGET −
    PRE_TRADE``, aligned by asset identifier.

    Both operands are validated as ``PortfolioWeights`` value objects
    in exactly the states ``PRE_TRADE`` and ``TARGET`` (the basic
    representation contract already guarantees finite real
    well-formed weights; the pre-trade book is deliberately **not**
    revalidated against TARGET constraints — drift may displace it).
    The trade universe is the union of the two asset sets, and the
    per-asset delta ``w*ᵢ − w⁻ᵢ`` uses the effective weight ``0.0``
    for an identifier absent from a book (so a pure entry buys the
    whole target and a pure exit sells the whole pre-trade book). It
    is computed by identifier — never by mapping order — and must be
    finite: an overflowing difference is an undefined trade state and
    rejects. The result is a fresh immutable ``WeightTrade`` that
    retains the pre-trade book, the target, and the delta mapping,
    each book keeping its original asset set.
    """
    pre = _require_book(pre_trade, expected=WeightState.PRE_TRADE, role="source")
    post = _require_book(target, expected=WeightState.TARGET, role="destination")
    return WeightTrade(pre, post)
