"""Behavior-named conformance suite for the weight-trade contract.

These tests freeze the weight-space trade transition: the
``WeightTrade`` primitive constructed by ``portlearn.trades.from_weights``
as the explicit ``PRE_TRADE → TARGET`` transition with ``Δw =
TARGET − PRE_TRADE``, alignment by asset identifier (never mapping
order), buys positive and sells negative, union-of-assets universe law
(an identifier absent from one book participates with effective weight
zero; the retained books keep their original asset sets), no
revalidation of the drifted book
against TARGET constraints, full retention of the pre-trade book, the
target, and the delta mapping, optimizer-agnostic and timeless
carriage (no optimizer identity, no execution timestamp, ledger row,
accounting period, or wealth path), and unconditional numerics on
non-finite deltas.
"""

from __future__ import annotations

import math

import pytest

from portlearn.costs import Proportional
from portlearn.rebalance import drift_weights
from portlearn.trades import WeightTrade, from_weights
from portlearn.turnover import one_way, two_sided
from portlearn.weights import (
    PortfolioWeights,
    WeightConstraints,
    WeightState,
    require_valid_target,
)


def _pre(weights: dict[str, float]) -> PortfolioWeights:
    return PortfolioWeights(weights, WeightState.PRE_TRADE)


def _post(weights: dict[str, float]) -> PortfolioWeights:
    return PortfolioWeights(weights, WeightState.POST_TRADE)


def _target(weights: dict[str, float]) -> PortfolioWeights:
    return PortfolioWeights(weights, WeightState.TARGET)


# ---------------------------------------------------------------------------
# T1 core: states, delta law, sign semantics
# ---------------------------------------------------------------------------


def test_source_is_pre_trade_and_destination_is_target() -> None:
    trade = from_weights(_pre({"A": 0.25, "B": 0.75}), _target({"A": 0.75, "B": 0.25}))
    assert trade.source.state is WeightState.PRE_TRADE
    assert trade.destination.state is WeightState.TARGET


def test_delta_is_target_minus_pre_trade() -> None:
    trade = from_weights(
        _pre({"A": 0.25, "B": 0.75, "C": 0.5}),
        _target({"A": 0.75, "B": 0.25, "C": 0.5}),
    )
    # binary-exact hand computation: 0.75-0.25, 0.25-0.75, 0.5-0.5
    assert trade.delta["A"] == 0.5
    assert trade.delta["B"] == -0.5
    assert trade.delta["C"] == 0.0


def test_positive_delta_buys_negative_sells_zero_holds() -> None:
    trade = from_weights(
        _pre({"A": 0.25, "B": 0.75, "C": 0.5}),
        _target({"A": 0.75, "B": 0.25, "C": 0.5}),
    )
    assert trade.delta["A"] > 0.0  # a buy
    assert trade.delta["B"] < 0.0  # a sell
    assert trade.delta["C"] == 0.0  # no trade


def test_retains_pre_trade_target_and_delta() -> None:
    pre = _pre({"A": 0.25, "B": 0.75})
    target = _target({"A": 0.75, "B": 0.25})
    trade = from_weights(pre, target)
    assert dict(trade.source.weights) == {"A": 0.25, "B": 0.75}
    assert dict(trade.destination.weights) == {"A": 0.75, "B": 0.25}
    assert dict(trade.delta) == {"A": 0.5, "B": -0.5}


def test_trade_is_immutable_value_object() -> None:
    trade = from_weights(_pre({"A": 0.25, "B": 0.75}), _target({"A": 0.75, "B": 0.25}))
    with pytest.raises(AttributeError):
        trade.source = _pre({"A": 1.0})  # type: ignore[misc]
    with pytest.raises(AttributeError):
        trade.delta = {"A": 0.0}  # type: ignore[misc]
    with pytest.raises(TypeError):
        trade.delta["A"] = 0.0  # type: ignore[index]


# ---------------------------------------------------------------------------
# Alignment by asset identifier, never mapping order
# ---------------------------------------------------------------------------


def test_mapping_order_invariance_aligns_by_asset_identifier() -> None:
    mixed = from_weights(
        _pre({"A": 0.6, "B": 0.4}), _target({"B": 0.5, "A": 0.5})
    )
    same_order = from_weights(
        _pre({"A": 0.6, "B": 0.4}), _target({"A": 0.5, "B": 0.5})
    )
    both_reordered = from_weights(
        _pre({"B": 0.4, "A": 0.6}), _target({"B": 0.5, "A": 0.5})
    )
    assert mixed == same_order
    assert mixed == both_reordered
    assert dict(mixed.delta) == dict(same_order.delta) == dict(both_reordered.delta)
    # per-asset sign semantics survive reordering: A sold, B bought
    assert mixed.delta["A"] < 0.0
    assert mixed.delta["B"] > 0.0


def test_equal_trades_hash_equal_and_unequal_differ() -> None:
    pre = _pre({"A": 0.25, "B": 0.75})
    target = _target({"A": 0.75, "B": 0.25})
    assert from_weights(pre, target) == from_weights(pre, target)
    assert hash(from_weights(pre, target)) == hash(from_weights(pre, target))
    other = from_weights(pre, _target({"A": 0.5, "B": 0.5}))
    assert other != from_weights(pre, target)


# ---------------------------------------------------------------------------
# Universe law: union of the two asset sets, effective zero per side
# ---------------------------------------------------------------------------


def test_union_of_assets_canonical_example() -> None:
    # PRE {A: .6, B: .4} -> TARGET {A: .5, C: .5}: B exits the target
    # book and C enters it; every identifier participates.
    trade = from_weights(_pre({"A": 0.6, "B": 0.4}), _target({"A": 0.5, "C": 0.5}))
    assert trade.delta["A"] == 0.5 - 0.6
    assert trade.delta["B"] == 0.0 - 0.4  # absent from the target: sold out
    assert trade.delta["C"] == 0.5 - 0.0  # absent from the pre-trade book: bought in
    # every delta is finite and the universe is the union
    assert set(trade.delta) == {"A", "B", "C"}


def test_union_of_assets_turnover_canonical_example() -> None:
    # the same canonical example, priced under both conventions:
    # two_sided = |−.1| + |−.4| + |+.5| = 1.0, one_way = 0.5
    trade = from_weights(_pre({"A": 0.6, "B": 0.4}), _target({"A": 0.5, "C": 0.5}))
    assert two_sided(trade) == 1.0
    assert one_way(trade) == 0.5


def test_union_trade_retains_original_asset_sets() -> None:
    # absence stays absence: zero is effective-for-calculation only, so
    # the retained books are never redefined with explicit zeros.
    pre = _pre({"A": 0.6, "B": 0.4})
    target = _target({"A": 0.5, "C": 0.5})
    trade = from_weights(pre, target)
    assert set(trade.source.weights) == {"A", "B"}
    assert set(trade.destination.weights) == {"A", "C"}
    assert trade.source.assets == frozenset({"A", "B"})
    assert trade.destination.assets == frozenset({"A", "C"})
    # the effective-weight lookup rule is the only zero source
    assert trade.source.weight_of("C") == 0.0
    assert trade.destination.weight_of("B") == 0.0


def test_pure_entry_all_deltas_equal_target_weights() -> None:
    # an empty pre-trade book (a portfolio holding nothing yet) buying
    # the whole target: every delta is exactly the target weight.
    trade = from_weights(_pre({}), _target({"A": 0.5, "B": 0.5}))
    assert dict(trade.delta) == {"A": 0.5, "B": 0.5}
    assert two_sided(trade) == 1.0
    assert one_way(trade) == 0.5


def test_pure_exit_all_deltas_equal_minus_pre_weights() -> None:
    # an empty target book (liquidating everything): every delta is
    # exactly minus the pre-trade weight, via absent identifiers.
    trade = from_weights(_pre({"A": 0.25, "B": 0.75}), _target({}))
    assert dict(trade.delta) == {"A": -0.25, "B": -0.75}
    assert two_sided(trade) == 1.0
    assert one_way(trade) == 0.5


def test_union_mapping_order_invariance() -> None:
    # identifier alignment extends over the union: reordering either
    # book's insertion order never changes the trade.
    baseline = from_weights(
        _pre({"A": 0.6, "B": 0.4}), _target({"A": 0.5, "C": 0.5})
    )
    reordered = from_weights(
        _pre({"B": 0.4, "A": 0.6}), _target({"C": 0.5, "A": 0.5})
    )
    assert reordered == baseline
    assert dict(reordered.delta) == dict(baseline.delta)
    assert hash(reordered) == hash(baseline)


def test_union_trade_is_immutable_value_object() -> None:
    trade = from_weights(_pre({"A": 0.6, "B": 0.4}), _target({"A": 0.5, "C": 0.5}))
    with pytest.raises(AttributeError):
        trade.delta = {"A": 0.0}  # type: ignore[misc]
    with pytest.raises(TypeError):
        trade.delta["B"] = 0.0  # type: ignore[index]


def test_union_trade_equality_and_hashing_by_value() -> None:
    pre = _pre({"A": 0.6, "B": 0.4})
    target = _target({"A": 0.5, "C": 0.5})
    assert from_weights(pre, target) == from_weights(pre, target)
    assert hash(from_weights(pre, target)) == hash(from_weights(pre, target))
    other = from_weights(_pre({"A": 0.6, "B": 0.4}), _target({"A": 0.6, "C": 0.4}))
    assert other != from_weights(pre, target)


def test_union_cost_fraction_uses_union_turnover() -> None:
    # the union trade flows through the canonical cost path unchanged
    model = Proportional(rate=0.0025, turnover=one_way)
    trade = from_weights(_pre({"A": 0.6, "B": 0.4}), _target({"A": 0.5, "C": 0.5}))
    assert one_way(trade) == 0.5
    assert model.q(trade) == pytest.approx(0.0025 * 0.5)
    assert model.f_cost(trade) == pytest.approx(1.0 - 0.00125)


def test_unchanged_universe_behavior_unchanged() -> None:
    # identical asset sets behave exactly as before the union law
    trade = from_weights(_pre({"A": 0.25, "B": 0.75}), _target({"A": 0.75, "B": 0.25}))
    assert dict(trade.delta) == {"A": 0.5, "B": -0.5}
    assert two_sided(trade) == 1.0
    assert one_way(trade) == 0.5


def test_non_weights_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        from_weights({"A": 0.5}, {"A": 0.5})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        from_weights(_post({"A": 0.5}), _target({"A": 0.5}))  # POST_TRADE is not a pre-trade book
    with pytest.raises(ValueError):
        from_weights(_pre({"A": 0.5}), _pre({"A": 0.5}))  # a PRE_TRADE is not a target
    with pytest.raises(ValueError):
        from_weights(_target({"A": 0.5}), _target({"A": 0.5}))  # a TARGET cannot drift into a trade source


# ---------------------------------------------------------------------------
# Drift-displaced pre-trade book: no TARGET-constraint revalidation
# ---------------------------------------------------------------------------


def test_drift_displaced_pre_trade_outside_target_limits_is_lawful() -> None:
    constraints = WeightConstraints(max_gross_exposure=1.5)
    target = _target({"A": 0.5, "B": 0.5})
    require_valid_target(target, constraints)  # the target itself is admissible
    # the drifted book sits outside those same TARGET limits...
    drifted = drift_weights(
        _post({"A": 2.0, "B": -1.0}), {"A": 2.0, "B": 1.0}
    )
    assert drifted.state is WeightState.PRE_TRADE
    assert drifted.weights["A"] == pytest.approx(4.0 / 3.0)
    assert drifted.weights["B"] == pytest.approx(-1.0 / 3.0)
    as_target = PortfolioWeights(dict(drifted.weights), WeightState.TARGET)
    with pytest.raises(ValueError):
        require_valid_target(as_target, constraints)  # ...would reject as a TARGET
    # ...yet the trade from it is lawful: the drifted book is never
    # revalidated against TARGET constraints.
    trade = from_weights(drifted, target)
    assert trade.source.state is WeightState.PRE_TRADE
    assert trade.delta["A"] == pytest.approx(0.5 - 4.0 / 3.0)
    assert trade.delta["B"] == pytest.approx(0.5 + 1.0 / 3.0)


def test_explicit_zeros_participate_in_the_universe() -> None:
    trade = from_weights(
        _pre({"A": 0.0, "B": 1.0}), _target({"A": 1.0, "B": 0.0})
    )
    assert trade.delta["A"] == 1.0
    assert trade.delta["B"] == -1.0


# ---------------------------------------------------------------------------
# Optimizer-agnostic and timeless carriage
# ---------------------------------------------------------------------------


def test_optimizer_agnostic_surface() -> None:
    universe = ["A", "B", "C", "D"]
    equal_weight_target = _target({asset: 1.0 / len(universe) for asset in universe})
    user_supplied_target = _target({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    pre = _pre({"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0})
    # identical targets from different provenance produce the identical trade
    assert from_weights(pre, equal_weight_target) == from_weights(pre, user_supplied_target)
    # and no optimizer identity belongs to the trade object
    trade = from_weights(pre, equal_weight_target)
    for absent in ("optimizer", "method", "strategy", "provenance", "model"):
        assert not hasattr(trade, absent)


def test_timeless_object_carries_no_execution_or_ledger_fields() -> None:
    trade = from_weights(_pre({"A": 0.25, "B": 0.75}), _target({"A": 0.75, "B": 0.25}))
    for absent in (
        "timestamp",
        "time",
        "instant",
        "executed_at",
        "ledger",
        "period",
        "wealth",
        "value",
    ):
        assert not hasattr(trade, absent)


# ---------------------------------------------------------------------------
# Validated numerics
# ---------------------------------------------------------------------------


def test_non_finite_delta_fails_closed() -> None:
    # both weights are finite reals, but the difference overflows
    with pytest.raises(ValueError, match="finite"):
        from_weights(
            _pre({"A": -1.7e308}), _target({"A": 1.7e308})
        )


def test_repr_is_informative() -> None:
    trade = from_weights(_pre({"A": 0.25, "B": 0.75}), _target({"A": 0.75, "B": 0.25}))
    text = repr(trade)
    assert "WeightTrade" in text
    assert "PRE_TRADE" in text and "TARGET" in text


def test_weighttrade_type_is_exposed_as_the_primitive() -> None:
    trade = from_weights(_pre({"A": 0.5}), _target({"A": 0.5}))
    assert isinstance(trade, WeightTrade)
    assert math.isfinite(trade.delta["A"])
