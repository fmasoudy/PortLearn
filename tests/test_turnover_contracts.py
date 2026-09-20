"""Behavior-named conformance suite for the turnover contract.

These tests freeze the approved portfolio-turnover laws: exactly two
named conventions over a ``WeightTrade`` — ``two_sided(trade) =
Σᵢ|Δwᵢ|`` and ``one_way(trade) = 0.5·Σᵢ|Δwᵢ|`` (some literature
calls this "half-turnover") — weight-space and per-trade by
construction, the exact ``one_way = 0.5 × two_sided`` identity, zero
turnover exactly 0.0 iff every delta is exactly 0.0, no universal
bound claimed (turnover is unbounded for short/leveraged books) with
2 documented only as the derived disjoint-allocation property for
fully-invested long-only portfolios, purity (evaluation never mutates
or consumes the trade), repeated-evaluation determinism, and the
absence of any generic convention-hidden ``turnover()`` entry point.
"""

from __future__ import annotations

import math

import pytest

import portlearn.turnover as turnover_module
from portlearn.trades import from_weights
from portlearn.turnover import one_way, two_sided
from portlearn.weights import PortfolioWeights, WeightState


def _pre(weights: dict[str, float]) -> PortfolioWeights:
    return PortfolioWeights(weights, WeightState.PRE_TRADE)


def _target(weights: dict[str, float]) -> PortfolioWeights:
    return PortfolioWeights(weights, WeightState.TARGET)


def _trade(pre: dict[str, float], target: dict[str, float]):
    return from_weights(_pre(pre), _target(target))


# ---------------------------------------------------------------------------
# The two named conventions
# ---------------------------------------------------------------------------


def test_two_sided_sums_absolute_deltas() -> None:
    trade = _trade({"A": 0.25, "B": 0.75}, {"A": 0.75, "B": 0.25})
    assert two_sided(trade) == 1.0  # |+0.5| + |-0.5|, binary-exact


def test_one_way_is_half_the_absolute_sum() -> None:
    trade = _trade({"A": 0.25, "B": 0.75}, {"A": 0.75, "B": 0.25})
    assert one_way(trade) == 0.5  # 0.5 · (0.5 + 0.5), binary-exact


def test_one_way_equals_half_of_two_sided_exactly() -> None:
    trade = _trade(
        {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4},
        {"A": 0.4, "B": 0.1, "C": 0.35, "D": 0.15},
    )
    assert one_way(trade) == 0.5 * two_sided(trade)
    assert one_way(trade) == two_sided(trade) / 2.0


def test_disjoint_long_only_allocation_reaches_two_sided_two() -> None:
    # fully-invested long-only books over disjoint supports: the derived
    # two_sided bound of 2 for that class (not a universal law)
    trade = _trade({"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0})
    assert two_sided(trade) == 2.0
    assert one_way(trade) == 1.0


def test_short_leveraged_turnover_is_not_bounded_by_two() -> None:
    # a short/leveraged book moves far more than 2 — no universal bound
    trade = _trade({"A": 5.0, "B": -4.0}, {"A": -4.0, "B": 5.0})
    assert two_sided(trade) == pytest.approx(18.0)
    assert one_way(trade) == pytest.approx(9.0)


def test_zero_trade_gives_exactly_zero_under_both_conventions() -> None:
    trade = _trade({"A": 0.25, "B": 0.75}, {"A": 0.25, "B": 0.75})
    assert two_sided(trade) == 0.0
    assert one_way(trade) == 0.0
    assert all(delta == 0.0 for delta in trade.delta.values())


def test_turnover_zero_iff_every_delta_exactly_zero() -> None:
    zero_everywhere = _trade({"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5})
    assert one_way(zero_everywhere) == 0.0 and two_sided(zero_everywhere) == 0.0
    one_flat = _trade({"A": 0.5, "B": 0.5}, {"A": 0.5 + 1e-15, "B": 0.5 - 1e-15})
    assert one_way(one_flat) > 0.0  # any nonzero delta ⇒ nonzero turnover


def test_turnover_is_symmetric_under_trade_reversal() -> None:
    forward = _trade({"A": 0.25, "B": 0.75}, {"A": 0.75, "B": 0.25})
    reversed_ = _trade({"A": 0.75, "B": 0.25}, {"A": 0.25, "B": 0.75})
    assert one_way(forward) == one_way(reversed_)
    assert two_sided(forward) == two_sided(reversed_)


# ---------------------------------------------------------------------------
# Purity and determinism
# ---------------------------------------------------------------------------


def test_evaluation_does_not_mutate_or_consume_the_trade() -> None:
    trade = _trade({"A": 0.25, "B": 0.75}, {"A": 0.75, "B": 0.25})
    before = (dict(trade.source.weights), dict(trade.destination.weights), dict(trade.delta))
    assert one_way(trade) == 0.5
    assert two_sided(trade) == 1.0
    assert one_way(trade) == 0.5  # reusable after costing-basis reads
    after = (dict(trade.source.weights), dict(trade.destination.weights), dict(trade.delta))
    assert before == after
    assert trade == _trade({"A": 0.25, "B": 0.75}, {"A": 0.75, "B": 0.25})


def test_repeated_evaluation_reproduces_the_same_result() -> None:
    trade = _trade(
        {"A": 0.1, "B": 0.2, "C": 0.7}, {"A": 0.5, "B": 0.25, "C": 0.25}
    )
    assert [one_way(trade) for _ in range(5)] == [one_way(trade)] * 5
    assert [two_sided(trade) for _ in range(5)] == [two_sided(trade)] * 5


def test_result_is_a_finite_real_float() -> None:
    trade = _trade({"A": 0.25, "B": 0.75}, {"A": 0.75, "B": 0.25})
    for measure in (one_way, two_sided):
        value = measure(trade)
        assert isinstance(value, float)
        assert not math.isinf(value) and not math.isnan(value)


# ---------------------------------------------------------------------------
# Convention naming discipline
# ---------------------------------------------------------------------------


def test_no_generic_convention_hidden_turnover_entry_point() -> None:
    assert not hasattr(turnover_module, "turnover"), (
        "every turnover computation must name its convention; a generic "
        "convention-hidden turnover() is prohibited"
    )
    assert sorted(turnover_module.__all__) == ["one_way", "two_sided"]


def test_two_conventions_are_independent_on_the_same_trade() -> None:
    trade = _trade({"A": 0.2, "B": 0.8}, {"A": 0.7, "B": 0.3})
    # same retained trade, both conventions, no cross-contamination
    assert two_sided(trade) == pytest.approx(1.0)
    assert one_way(trade) == pytest.approx(0.5)
    assert two_sided(trade) == pytest.approx(1.0)  # unchanged after one_way read


def test_fail_closed_on_non_weighttrade_inputs() -> None:
    for measure in (one_way, two_sided):
        with pytest.raises(ValueError):
            measure({"A": 0.5})  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            measure("not a trade")  # type: ignore[arg-type]
