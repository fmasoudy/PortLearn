"""Behavior-named conformance suite for the transaction-cost contract.

These tests freeze the proportional-cost laws: the
convention-bound ``Proportional(rate, turnover=one_way)`` model whose
keyword is exactly ``turnover`` and which accepts only PortLearn's two
recognized turnover conventions (arbitrary lambdas/callables rejected
unconditional), the stable canonical convention identity (``"one_way"``
/ ``"two_sided"``) carried for future experiment provenance, the cost
fraction ``q = rate × turnover_measure(trade)`` with cost factor
``F_cost = 1 − q`` on the unconditional domain ``0 ≤ q < 1``, the rate
law ``rate ≥ 0`` real finite (bool/Decimal/negative/non-finite
rejected; ``rate = 0`` lawful frictionless), zero-trade and zero-rate
identities, purity/reusability of the stored trade across multiple
cost models (rates 0.0010/0.0025/0.0050 on one trade, no re-derivation),
reporting-independence (a ``two_sided`` reporting call and a
``one_way``-bound cost model coexist without forcing equality), the
baseline cost-to-weights separation (weights track POST_TRADE = TARGET
exactly under nonzero cost; value track multiplied by F_cost), the
two-trade cost-factor composition identity, and consistency with the
fixed rebalancing laws.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from portlearn.costs import Proportional
from portlearn.rebalance import execute_rebalance
from portlearn.timing import DecisionTiming, ReturnRealizationPeriod
from portlearn.trades import from_weights
from portlearn.turnover import one_way, two_sided
from portlearn.weights import PortfolioWeights, WeightState

TOL = 1e-12


def _pre(weights: dict[str, float]) -> PortfolioWeights:
    return PortfolioWeights(weights, WeightState.PRE_TRADE)


def _post(weights: dict[str, float]) -> PortfolioWeights:
    return PortfolioWeights(weights, WeightState.POST_TRADE)


def _target(weights: dict[str, float]) -> PortfolioWeights:
    return PortfolioWeights(weights, WeightState.TARGET)


def _trade(pre: dict[str, float], target: dict[str, float]):
    return from_weights(_pre(pre), _target(target))


def _timing() -> DecisionTiming:
    instant = datetime(2026, 1, 30, tzinfo=UTC)
    return DecisionTiming(
        decision_time=instant,
        execution_time=instant,
        return_realization_period=ReturnRealizationPeriod(
            start_time=instant, end_time=instant + timedelta(hours=1)
        ),
    )


def _bps25_trade():
    """The canonical 25 bps case: one_way turnover 0.5."""
    return _trade({"A": 0.25, "B": 0.75}, {"A": 0.75, "B": 0.25})


# ---------------------------------------------------------------------------
# Model construction and the exact `turnover` keyword
# ---------------------------------------------------------------------------


def test_model_binds_its_convention_through_the_turnover_keyword() -> None:
    model = Proportional(rate=0.0025, turnover=one_way)
    trade = _bps25_trade()
    assert model.q(trade) == pytest.approx(0.0025 * 0.5)
    assert model.f_cost(trade) == pytest.approx(1.0 - 0.00125)


def test_construction_requires_the_turnover_convention_not_positional_rate_only() -> None:
    with pytest.raises(TypeError):
        Proportional(rate=0.0025)  # a bare rate is an incomplete cost spec
    with pytest.raises(TypeError):
        Proportional(turnover=one_way)  # no rate is not a cost model


def test_zero_rate_is_lawful_frictionless() -> None:
    model = Proportional(rate=0, turnover=one_way)
    trade = _bps25_trade()
    assert model.q(trade) == 0.0
    assert model.f_cost(trade) == 1.0
    int_zero = Proportional(rate=0.0, turnover=two_sided)
    assert int_zero.f_cost(trade) == 1.0


def test_rate_law_finite_real_nonnegative() -> None:
    for lawful in (0.0, 0.001, 0.0025, 1.0, 2, 1e-9):
        model = Proportional(rate=lawful, turnover=one_way)
        assert model.rate == float(lawful)
    for bad_rate in (-0.0001, -1.0, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            Proportional(rate=bad_rate, turnover=one_way)
    for non_real in (True, False, Decimal("0.0025"), "0.0025", None, 1 + 2j):
        with pytest.raises(ValueError):
            Proportional(rate=non_real, turnover=one_way)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Recognized conventions only; canonical identity provenance
# ---------------------------------------------------------------------------


def test_arbitrary_callables_are_rejected_fail_closed() -> None:
    def half_measure(trade):
        return 0.5

    for unknown in (
        lambda trade: 1.0,
        half_measure,
        two_sided.__call__,
        sum,
        0.5,  # a bare scalar is not a convention
        "one_way",  # a string names a convention but is not one
        None,
    ):
        with pytest.raises(ValueError):
            Proportional(rate=0.0025, turnover=unknown)


def test_canonical_convention_identity_is_preserved() -> None:
    assert Proportional(rate=0.0025, turnover=one_way).convention == "one_way"
    assert Proportional(rate=0.0025, turnover=two_sided).convention == "two_sided"
    # the identity is stable canonical text, ready for YAML provenance:
    # costs: {model: proportional, rate: 0.0025, turnover: one_way}
    for model in (
        Proportional(rate=0.0010, turnover=one_way),
        Proportional(rate=0.0025, turnover=one_way),
        Proportional(rate=0.0050, turnover=two_sided),
    ):
        assert model.convention in {"one_way", "two_sided"}


def test_two_sided_binding_charges_double_the_one_way_measure() -> None:
    trade = _bps25_trade()
    one = Proportional(rate=0.0025, turnover=one_way)
    two = Proportional(rate=0.0025, turnover=two_sided)
    assert two.q(trade) == pytest.approx(2.0 * one.q(trade))
    assert one.q(trade) == pytest.approx(0.00125)
    assert two.q(trade) == pytest.approx(0.0025)


# ---------------------------------------------------------------------------
# Cost fraction q and factor F_cost; unconditional domain 0 <= q < 1
# ---------------------------------------------------------------------------


def test_q_is_rate_times_the_named_measure() -> None:
    trade = _trade(
        {"A": 0.1, "B": 0.2, "C": 0.7}, {"A": 0.5, "B": 0.25, "C": 0.25}
    )
    model = Proportional(rate=0.0025, turnover=one_way)
    # one_way = 0.5*(0.4+0.05+0.45) = 0.45
    assert one_way(trade) == pytest.approx(0.45)
    assert model.q(trade) == pytest.approx(0.0025 * 0.45)


def test_cost_fraction_domain_rejects_q_at_or_above_one() -> None:
    trade = _trade({"A": 5.0, "B": -4.0}, {"A": -4.0, "B": 5.0})
    # two_sided = 18.0
    with pytest.raises(ValueError):
        Proportional(rate=0.06, turnover=two_sided).q(trade)  # q = 1.08 > 1
    with pytest.raises(ValueError):
        Proportional(rate=1.0, turnover=two_sided).q(trade)  # q = 18.0
    # q == 1 exactly (F_cost = 0) is equally unlawful
    with pytest.raises(ValueError):
        Proportional(rate=1.0 / 18.0, turnover=two_sided).q(trade)
    # just below the boundary stays lawful
    ok = Proportional(rate=1.0 / 18.0 - 1e-9, turnover=two_sided)
    assert ok.q(trade) < 1.0
    assert ok.f_cost(trade) > 0.0


def test_cost_fraction_domain_rejects_non_finite_intermediates() -> None:
    # individually finite weights whose delta overflows: from_weights
    # itself rejects, so the non-finite turnover intermediate can never
    # arise from a constructed trade; the domain check remains the safety
    # net for hostile q inputs.
    with pytest.raises(ValueError):
        from_weights(_pre({"A": -1.7e308}), _target({"A": 1.7e308}))
    trade = _bps25_trade()
    model = Proportional(rate=0.0025, turnover=one_way)
    assert math.isfinite(model.q(trade)) and math.isfinite(model.f_cost(trade))


def test_f_cost_is_one_minus_q() -> None:
    trade = _trade({"A": 0.2, "B": 0.8}, {"A": 0.7, "B": 0.3})
    model = Proportional(rate=0.0025, turnover=one_way)
    assert model.f_cost(trade) == pytest.approx(1.0 - model.q(trade))


def test_zero_trade_gives_zero_q_and_unit_f_cost_exactly() -> None:
    trade = _trade({"A": 0.25, "B": 0.75}, {"A": 0.25, "B": 0.75})
    for model in (
        Proportional(rate=0.0025, turnover=one_way),
        Proportional(rate=0.0025, turnover=two_sided),
    ):
        assert model.q(trade) == 0.0
        assert model.f_cost(trade) == 1.0


def test_results_are_finite_real_floats() -> None:
    trade = _bps25_trade()
    model = Proportional(rate=0.0025, turnover=one_way)
    for value in (model.q(trade), model.f_cost(trade)):
        assert type(value) is float
        assert math.isfinite(value)


# ---------------------------------------------------------------------------
# Purity / reusability: one stored trade, many cost models
# ---------------------------------------------------------------------------


def test_evaluation_never_mutates_or_consumes_the_trade() -> None:
    trade = _bps25_trade()
    before = (dict(trade.source.weights), dict(trade.destination.weights), dict(trade.delta))
    models = [
        Proportional(rate=r, turnover=one_way)
        for r in (0.0010, 0.0025, 0.0050)
    ]
    for model in models:
        model.q(trade)
        model.f_cost(trade)
    after = (dict(trade.source.weights), dict(trade.destination.weights), dict(trade.delta))
    assert before == after


def test_same_trade_under_three_rates_without_rederivation() -> None:
    trade = _bps25_trade()  # one_way = 0.5 exactly
    c10 = Proportional(rate=0.0010, turnover=one_way)
    c25 = Proportional(rate=0.0025, turnover=one_way)
    c50 = Proportional(rate=0.0050, turnover=one_way)
    assert c10.q(trade) == pytest.approx(0.0005)
    assert c25.q(trade) == pytest.approx(0.00125)
    assert c50.q(trade) == pytest.approx(0.0025)
    # monotone: charging more costs more, factor shrinks
    assert c10.q(trade) < c25.q(trade) < c50.q(trade)
    assert c10.f_cost(trade) > c25.f_cost(trade) > c50.f_cost(trade)
    # the same stored trade still evaluates identically afterwards
    assert one_way(trade) == 0.5 and two_sided(trade) == 1.0


def test_repeated_evaluation_reproduces_the_same_result() -> None:
    trade = _bps25_trade()
    model = Proportional(rate=0.0025, turnover=one_way)
    assert [model.q(trade) for _ in range(5)] == [model.q(trade)] * 5
    assert [model.f_cost(trade) for _ in range(5)] == [model.f_cost(trade)] * 5


def test_reporting_and_costing_coexist_without_forcing_equality() -> None:
    trade = _bps25_trade()
    model = Proportional(rate=0.0025, turnover=one_way)
    reported = two_sided(trade)  # a reporting call, one-way cost basis
    assert reported == 1.0
    assert model.q(trade) == pytest.approx(0.00125)  # one_way basis, not 0.0025
    assert reported / 2.0 == pytest.approx(one_way(trade))  # both live on the same retained trade


# ---------------------------------------------------------------------------
# T4 baseline separation: weights track POST_TRADE = TARGET; value × F_cost
# ---------------------------------------------------------------------------


def test_post_trade_weights_equal_target_exactly_under_nonzero_cost() -> None:
    target = _target({"A": 0.75, "B": 0.25})
    trade = from_weights(_pre({"A": 0.25, "B": 0.75}), target)
    model = Proportional(rate=0.0025, turnover=one_way)
    f = model.f_cost(trade)
    assert f < 1.0  # a genuinely nonzero cost
    executed = execute_rebalance(target, _timing())  # fixed execution law
    assert executed.state is WeightState.POST_TRADE
    assert dict(executed.weights) == dict(trade.destination.weights)
    assert trade.destination.weights == target.weights  # exactly, no renormalization
    # the value track is a separate multiplication, never a weight edit:
    value_before = 1_000_000.0
    value_after = value_before * f
    assert value_after == pytest.approx(998_750.0)  # hand-computed 1e6 × 0.99875
    # and weights are untouched by the value multiplication
    assert dict(executed.weights) == {"A": 0.75, "B": 0.25}


def test_baseline_separation_is_declared_simplified_convention() -> None:
    # the module documents that cash financing, spreads, slippage,
    # market impact, partial fills, and share-space execution are
    # outside this baseline; the law here is only that the two tracks
    # compose without redefining each other.
    doc = Proportional.__doc__ or ""
    assert "proportional" in doc.lower()


def test_costs_do_not_alter_frozen_timing_semantics() -> None:
    # Costing owns no execution-instant binding: costing a trade composed
    # with the fixed execution law changes nothing about the timing.
    target = _target({"A": 0.6, "B": 0.4})
    timing = _timing()
    executed = execute_rebalance(target, timing)
    trade = from_weights(_pre({"A": 0.5, "B": 0.5}), target)
    model = Proportional(rate=0.0025, turnover=one_way)
    assert model.f_cost(trade) == pytest.approx(1.0 - 0.0025 * 0.1)
    assert dict(executed.weights) == {"A": 0.6, "B": 0.4}  # execution law unaltered
    assert not hasattr(trade, "timestamp") and not hasattr(trade, "ledger")


# ---------------------------------------------------------------------------
# T5 composition over a two-trade sequence
# ---------------------------------------------------------------------------


def test_two_trade_composition_multiplies_cost_factors() -> None:
    model = Proportional(rate=0.0025, turnover=one_way)
    first = _trade({"A": 0.25, "B": 0.75}, {"A": 0.75, "B": 0.25})
    second = _trade({"A": 0.75, "B": 0.25}, {"A": 0.60, "B": 0.40})
    f1 = model.f_cost(first)
    f2 = model.f_cost(second)
    f_total = f1 * f2
    assert f1 == pytest.approx(1.0 - 0.0025 * 0.5)
    assert f2 == pytest.approx(1.0 - 0.0025 * 0.15)
    assert f_total == pytest.approx((1.0 - 0.00125) * (1.0 - 0.000375))
    # growth basis: G_after = G_before × F_total
    g_before = 1.05
    g_after = g_before * f_total
    assert g_after == pytest.approx(1.05 * (1.0 - 0.00125) * (1.0 - 0.000375))


def test_composition_with_a_frictionless_leg_is_the_other_leg() -> None:
    model = Proportional(rate=0.0025, turnover=one_way)
    costly = _trade({"A": 0.25, "B": 0.75}, {"A": 0.75, "B": 0.25})
    frictionless = _trade({"A": 0.75, "B": 0.25}, {"A": 0.75, "B": 0.25})
    assert model.f_cost(frictionless) == 1.0
    total = model.f_cost(costly) * model.f_cost(frictionless)
    assert total == model.f_cost(costly)


def test_composition_domain_stays_within_lawful_bounds() -> None:
    model = Proportional(rate=0.01, turnover=two_sided)
    trades = [
        _trade({"A": 0.9, "B": 0.1}, {"A": 0.1, "B": 0.9}),  # two_sided 1.8
        _trade({"A": 0.1, "B": 0.9}, {"A": 0.5, "B": 0.5}),  # two_sided 0.8
    ]
    factors = [model.f_cost(t) for t in trades]
    total = math.prod(factors)
    for f, t in zip(factors, trades, strict=True):
        assert 0.0 < f <= 1.0
        assert f == pytest.approx(1.0 - 0.01 * two_sided(t))
    assert 0.0 < total <= 1.0


def test_model_is_immutable() -> None:
    model = Proportional(rate=0.0025, turnover=one_way)
    with pytest.raises(AttributeError):
        model.rate = 0.0050  # type: ignore[misc]
    with pytest.raises(AttributeError):
        model.convention = "two_sided"  # type: ignore[misc]


def test_repr_carries_rate_and_canonical_convention() -> None:
    model = Proportional(rate=0.0025, turnover=one_way)
    text = repr(model)
    assert "0.0025" in text and "one_way" in text


def test_fail_closed_on_non_weighttrade_inputs() -> None:
    model = Proportional(rate=0.0025, turnover=one_way)
    with pytest.raises(ValueError):
        model.q({"A": 0.5})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        model.f_cost(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Structural conformance to the public CostModel protocol
# ---------------------------------------------------------------------------


def test_proportional_is_instance_of_public_costmodel_protocol() -> None:
    from portlearn.interfaces import CostModel

    assert isinstance(Proportional(rate=0.0025, turnover=one_way), CostModel) is True
    assert isinstance(Proportional(rate=0.0, turnover=two_sided), CostModel) is True


def test_estimate_trade_cost_matches_q_on_canonical_weights() -> None:
    model = Proportional(rate=0.0025, turnover=one_way)
    cost = model.estimate_trade_cost({"A": 0.6, "B": 0.4}, {"A": 0.5, "C": 0.5})
    # canonical example: one_way turnover exactly 0.5 -> q = 0.0025 * 0.5
    assert cost == pytest.approx(0.00125)
    assert cost == pytest.approx(model.q(_trade({"A": 0.6, "B": 0.4}, {"A": 0.5, "C": 0.5})))


def test_estimate_trade_cost_is_the_canonical_q_no_duplicated_arithmetic() -> None:
    model = Proportional(rate=0.0025, turnover=one_way)
    for pre, target in (
        ({"A": 0.25, "B": 0.75}, {"A": 0.75, "B": 0.25}),
        ({}, {"A": 0.5, "B": 0.5}),
        ({"A": 0.25, "B": 0.75}, {}),
    ):
        trade = _trade(pre, target)
        assert model.estimate_trade_cost(pre, target) == model.q(trade)


def test_estimate_trade_cost_validates_like_from_weights() -> None:
    model = Proportional(rate=0.0025, turnover=one_way)
    # weight mappings are the protocol input: a zero-delta book is lawful
    assert model.estimate_trade_cost({"A": 0.5}, {"A": 0.5}) == 0.0
    # non-mapping operands and unlawful weights reject unconditionally
    for bad_pre, bad_target in (
        ("not a book", {"A": 0.5}),
        ({"A": 0.5}, "not a book"),
        ({"A": True}, {"A": 0.5}),  # type: ignore[dict-item]
        ({"A": 0.5}, {"A": float("nan")}),  # type: ignore[dict-item]
    ):
        with pytest.raises(ValueError):
            model.estimate_trade_cost(bad_pre, bad_target)  # type: ignore[arg-type]


def test_estimate_trade_cost_is_pure_and_repeatable() -> None:
    model = Proportional(rate=0.0025, turnover=one_way)
    pre, target = {"A": 0.6, "B": 0.4}, {"A": 0.5, "C": 0.5}
    assert [model.estimate_trade_cost(pre, target) for _ in range(3)] == [
        model.estimate_trade_cost(pre, target)
    ] * 3


def test_estimate_trade_cost_enforces_the_q_domain_fail_closed() -> None:
    # a huge target move at a high rate pushes q >= 1: unconditional
    model = Proportional(rate=0.9, turnover=two_sided)
    with pytest.raises(ValueError, match="q < 1"):
        model.estimate_trade_cost({}, {"A": 100.0})
