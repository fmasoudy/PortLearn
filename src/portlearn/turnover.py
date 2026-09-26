"""Portfolio turnover under two named conventions.

This module defines the portfolio-turnover measure over a
``WeightTrade``: turnover is a **trading measure, not itself a cost**,
computed in weight space, per trade (never a period statistic), and
exposed under exactly two named conventions — a generic
convention-hidden ``turnover()`` is prohibited; every computation must
name its convention.

- ``two_sided(trade) = Σᵢ|Δwᵢ|`` — buys and sells both count.
- ``one_way(trade) = 0.5 · Σᵢ|Δwᵢ|`` — the half convention common in
  empirical asset pricing; some literature calls this
  "half-turnover", but no second public alias is exposed.

**No universal bound** is claimed on turnover: with short and
leveraged portfolios the measure is unbounded. A bound of 2 is
documented **only** as a derived property of ``two_sided`` for
fully-invested long-only portfolios: moving an entire allocation from
one disjoint long-only support to another trades a total absolute
weight of exactly ``Σ|Δw| = 2`` — one whole portfolio out, another
whole portfolio in (illustrated by ``{"A": 1.0}`` → ``{"B": 1.0}``).
Turnover is exactly ``0.0`` iff every ``Δwᵢ = 0.0`` exactly.

Both functions are pure with respect to the trade: they never modify
or consume it, and repeated evaluation reproduces the same result.
The error surface is ``ValueError`` only; numerics follow the
checked-real / guarded-aggregation discipline (``math.fsum``, no
``OverflowError`` leaks).
"""

from __future__ import annotations

import math

from portlearn.trades import WeightTrade

__all__ = ["one_way", "two_sided"]


def _absolute_delta_total(trade: WeightTrade) -> float:
    """``Σᵢ|Δwᵢ|`` under the guarded exact-sum discipline, unconditional
    on a non-finite intermediate or total (ValueError-only surface)."""
    if not isinstance(trade, WeightTrade):
        raise ValueError(  # noqa: TRY004 — ValueError-only surface is the error law
            "turnover operates on a WeightTrade value object; got "
            f"{type(trade).__name__}: {trade!r}. Construct one with "
            "portlearn.trades.from_weights(pre_trade, target)."
        )
    terms: list[float] = []
    for asset, difference in trade.delta.items():
        magnitude = abs(difference)
        if not math.isfinite(magnitude):
            raise ValueError(
                f"the absolute delta on {asset!r} must be finite; got "
                f"{magnitude!r} — a non-finite turnover intermediate is "
                "undefined and fails closed."
            )
        terms.append(magnitude)
    try:
        total = math.fsum(terms)
    except OverflowError:
        raise ValueError(
            "the absolute-delta total must be finite; got non-finite sum"
        ) from None
    if not math.isfinite(total):
        raise ValueError(
            "the absolute-delta total must be finite; got non-finite sum"
        )
    return float(total)


def two_sided(trade: WeightTrade) -> float:
    """``two_sided(trade) = Σᵢ|Δwᵢ|`` — buys and sells both count.

    The two-sided convention: every absolute delta of the trade's
    universe is summed with ``math.fsum`` for exact, order-independent
    accumulation. Pure with respect to the trade; exactly ``0.0`` iff
    every delta is exactly zero. The only bound ever claimed: for
    fully-invested long-only portfolios ``two_sided`` is bounded by
    the derived value 2, with equality on a disjoint-allocation move
    (one whole portfolio out, another in); with shorts and leverage
    turnover is unbounded, and no universal bound is implied.
    """
    return _absolute_delta_total(trade)


def one_way(trade: WeightTrade) -> float:
    """``one_way(trade) = 0.5 · Σᵢ|Δwᵢ|`` — the half convention.

    The one-way convention common in empirical asset pricing (some
    literature calls it "half-turnover"; no second public alias is
    exposed). Exactly half of ``two_sided`` on the same trade. Pure
    with respect to the trade; exactly ``0.0`` iff every delta is
    exactly zero.
    """
    return 0.5 * _absolute_delta_total(trade) + 0.0  # −0.0 → +0.0 on the zero trade
