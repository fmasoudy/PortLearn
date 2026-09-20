"""Behavior-named conformance suite for the portfolio-ledger contract.

These tests freeze the portfolio-ledger laws:
one frozen ``LedgerRow`` per accounting
period in finance-table language with zero/one/many ``Execution``
records; segment-composed gross growth ``G_gross,t = Π_m D_m`` over
holding segments split at execution instants; the composed cost factor
``F_cost,t = Π_k (1 − q_k)`` from the cost law; ``G_net =
G_gross × F_cost``; the multiplicative unit-NAV wealth recursion
``W_{t+1} = W_t × G_net,t``; the all-retained-books row universe; the
half-open ``[period_start, period_end)`` execution-ownership rule;
closing weights at ``period_end⁻`` before any execution exactly at
``period_end``; POST_TRADE = TARGET enforced fail-closed on every
supplied ``AccountingEngine``; the held-target
``realized_returns`` law compounding per-asset factors over the
holding interval; and no strategy name
anywhere on the record (D10). All expected values are hand-calculated.
"""

from __future__ import annotations

import dataclasses
import inspect
from datetime import UTC, datetime
from typing import Any

import pytest

import portlearn.ledger as ledger_module
from portlearn.costs import Proportional
from portlearn.interfaces import AccountingResult, PortfolioDecision
from portlearn.ledger import (
    ExactFillAccounting,
    build_ledger,
)
from portlearn.timing import InvalidChronologyError, NaiveTimestampError
from portlearn.turnover import one_way, two_sided
from portlearn.weights import PortfolioWeights

# ---------------------------------------------------------------------------
# Shared fixtures — a three-asset world with hand-calculable growth.
# ---------------------------------------------------------------------------
# UTC instants on fixed month boundaries. UTC throughout keeps every
# hand calculation exact in binary floats where the values admit it;
# pytest.approx is used where they do not.

T0 = datetime(2026, 1, 1, tzinfo=UTC)
T1 = datetime(2026, 2, 1, tzinfo=UTC)
T2 = datetime(2026, 3, 1, tzinfo=UTC)
T3 = datetime(2026, 4, 1, tzinfo=UTC)
T4 = datetime(2026, 5, 1, tzinfo=UTC)
T5 = datetime(2026, 6, 1, tzinfo=UTC)

PERIODS = (T0, T1, T2, T3, T4, T5)


def _book(weights: dict[str, float], state: str) -> PortfolioWeights:
    from portlearn.weights import WeightState

    return PortfolioWeights(weights, WeightState(state))


def _decision(
    target: dict[str, float], decision_time: datetime, execution_time: datetime
) -> PortfolioDecision:
    return PortfolioDecision(
        decision_time=decision_time,
        execution_time=execution_time,
        target_weights=target,
    )


class _SpyEngine:
    """An admissible stand-in engine: returns the TARGET book while
    recording every call's arguments (floor 8/17 twin)."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any, Any]] = []

    def account(
        self,
        decision: PortfolioDecision,
        pre_trade_weights: Any,
        realized_returns: Any,
    ) -> AccountingResult:
        self.calls.append((decision, pre_trade_weights, realized_returns))
        return AccountingResult(dict(decision.target_weights))


class _SkewEngine:
    """A NON-target engine: mutates the post-trade book (floor 9)."""

    def account(
        self,
        decision: PortfolioDecision,
        pre_trade_weights: Any,
        realized_returns: Any,
    ) -> AccountingResult:
        skewed = dict(decision.target_weights)
        key = next(iter(skewed))
        skewed[key] = skewed[key] + 1e-9
        return AccountingResult(skewed)
# ---------------------------------------------------------------------------
# Hand-calculated worlds.
#
# World X (two executions inside ONE period; floors 3, 4, 15):
#   period [T0, T2), executions at M1 (Jan 16) and M2 (Feb 16), so the
#   three segments are [T0,M1), [M1,M2), [M2,T2).  Initial {A: 0.5,
#   B: 0.5}; cost model rate 0.3 on one_way.
#     seg1 {A: 2.0, B: 1.0}  -> D1 = 0.5*2 + 0.5*1 = 1.5
#     exec M1: pre {2/3, 1/3} -> target {0.5, 0.5}; one_way = 1/6,
#              q1 = 0.3 * 1/6 = 0.05, F1 = 0.95
#     seg2 {A: 1.0, B: 1.0}  -> D2 = 1.0 (drift keeps {0.5, 0.5})
#     exec M2: pre {0.5, 0.5} -> target {A: 0.25, B: 0.75};
#              one_way = 0.25, q2 = 0.075, F2 = 0.925
#     seg3 {A: 1.5, B: 1.0}  -> D3 = 0.25*1.5 + 0.75*1.0 = 1.125
#   G_gross = 1.5 * 1.0 * 1.125 = 1.6875   (exact in binary)
#   F_cost  = 0.95 * 0.925     = 0.87875   (exact)
#   G_net   = 1.6875 * 0.87875 = 1.482890625 (exact)
#   Rejected single-denominator form: A compounds 2*1*1.5 = 3.0,
#   B compounds 1.0, so 0.5*3.0 + 0.5*1.0 = 2.0 != 1.6875.
# ---------------------------------------------------------------------------

MID1 = datetime(2026, 1, 16, tzinfo=UTC)
MID2 = datetime(2026, 2, 16, tzinfo=UTC)


def _world_x(decisions: list[PortfolioDecision] | None = None):
    """The two-executions-one-period world as explicit inputs."""
    return {
        "initial_weights": {"A": 0.5, "B": 0.5},
        "accounting_instants": (T0, T2),
        "growth_factors": {
            T0: {"A": 2.0, "B": 1.0},
            MID1: {"A": 1.0, "B": 1.0},
            MID2: {"A": 1.5, "B": 1.0},
        },
        "decisions": decisions
        if decisions is not None
        else [
            _decision({"A": 0.5, "B": 0.5}, MID1, MID1),
            _decision({"A": 0.25, "B": 0.75}, MID2, MID2),
        ],
        "cost_model": Proportional(0.3, one_way),
    }
# ---------------------------------------------------------------------------
# Floor 1 — row shape is the finance table, nothing else (D9/D10).
# ---------------------------------------------------------------------------


def test_row_shape_fields_finance_table() -> None:
    row = build_ledger(**_world_x()).rows[0]
    assert dataclasses.is_dataclass(type(row))
    fields = {f.name for f in dataclasses.fields(row)}
    assert fields == {
        "period_start",
        "period_end",
        "opening_weights",
        "executions",
        "closing_weights",
        "gross_return",
        "net_return",
        "turnover",
        "transaction_cost",
        "wealth_open",
        "wealth_close",
        "universe",
    }
    execution = row.executions[0]
    assert dataclasses.is_dataclass(type(execution))
    execution_fields = {f.name for f in dataclasses.fields(execution)}
    assert execution_fields == {
        "decision_time",
        "execution_time",
        "target_weights",
        "pre_trade_weights",
        "trade",
        "turnover",
        "transaction_cost",
        "post_trade_weights",
    }
    # D10: no strategy name / experiment identifier anywhere on the record.
    for banned in ("strategy", "experiment", "name", "label", "run_id"):
        assert banned not in fields
        assert banned not in execution_fields
    path = build_ledger(**_world_x())
    assert dataclasses.is_dataclass(type(path))
    path_fields = {f.name for f in dataclasses.fields(path)}
    assert path_fields == {"rows", "final_wealth", "n_periods"}
    for banned in ("strategy", "experiment", "name", "label", "run_id"):
        assert banned not in path_fields


# ---------------------------------------------------------------------------
# Floor 2 — zero / one / many executions per period.
# ---------------------------------------------------------------------------


def test_executions_tuple_zero_one_many() -> None:
    # Zero: a drift-only period records the EMPTY tuple and no costs.
    # D = 0.5*2.0 + 0.5*0.5 = 1.25, so r_gross = 0.25; F_cost = 1.
    zero = build_ledger(
        initial_weights={"A": 0.5, "B": 0.5},
        accounting_instants=(T0, T1),
        growth_factors={T0: {"A": 2.0, "B": 0.5}},
        decisions=[],
        cost_model=Proportional(0.001, two_sided),
    ).rows[0]
    assert zero.executions == ()
    assert zero.turnover == 0
    assert zero.transaction_cost == 0
    assert zero.gross_return == pytest.approx(0.25)
    assert zero.net_return == pytest.approx(0.25)  # no execution, F = 1

    # One: a single mid-period execution, tuple of length one.
    one = build_ledger(**_world_x(
        [_decision({"A": 0.5, "B": 0.5}, MID1, MID1)]
    )).rows[0]
    assert len(one.executions) == 1
    assert one.executions[0].execution_time == MID1

    # Many: World X's two mid-period executions, in execution order.
    many = build_ledger(**_world_x()).rows[0]
    assert len(many.executions) == 2
    assert [e.execution_time for e in many.executions] == [MID1, MID2]

# ---------------------------------------------------------------------------
# Floor 3 — gross is segment-composed, never single-denominator (T3).
# ---------------------------------------------------------------------------


def test_gross_return_segment_composed() -> None:
    row = build_ledger(**_world_x()).rows[0]
    # G_gross = 1.5 * 1.0 * 1.125 = 1.6875, all binary-exact.
    assert row.gross_return == pytest.approx(1.6875 - 1.0)
    # The rejected single-denominator form gives a DIFFERENT number:
    # 0.5*(2*1*1.5) + 0.5*(1*1*1) = 2.0 — must not be what we recorded.
    assert row.gross_return != pytest.approx(2.0 - 1.0)
    # Execution-free period: segment law and single-denominator coincide.
    flat = build_ledger(
        initial_weights={"A": 0.5, "B": 0.5},
        accounting_instants=(T0, T1),
        growth_factors={T0: {"A": 2.0, "B": 1.0}},
        decisions=[],
        cost_model=Proportional(0.0, one_way),
    ).rows[0]
    assert flat.gross_return == pytest.approx(1.5 - 1.0)


# ---------------------------------------------------------------------------
# Floor 4 — two executions in one period compose both factor tracks (T3/T4).
# ---------------------------------------------------------------------------


def test_two_executions_one_period_composes_factors() -> None:
    row = build_ledger(**_world_x()).rows[0]
    assert row.gross_return == pytest.approx(1.6875 - 1.0)  # Π_m D_m
    # F_cost = 0.95 * 0.925 = 0.87875; transaction_cost = 1 - F = 0.12125.
    assert row.transaction_cost == pytest.approx(1.0 - 0.87875)
    # G_net = 1.6875 * 0.87875, r_net = G_net - 1.
    assert row.net_return == pytest.approx(1.6875 * 0.87875 - 1.0)
    # Row turnover aggregates the two executions (floor 15's convention).
    assert row.turnover == pytest.approx(1.0 / 6.0 + 0.25)


# ---------------------------------------------------------------------------
# Floor 5 — costs enter multiplicatively, never additively (T3/T5).
# ---------------------------------------------------------------------------


def test_cost_factor_multiplicative_only() -> None:
    row = build_ledger(**_world_x()).rows[0]
    g_gross = 1.0 + row.gross_return
    g_net = 1.0 + row.net_return
    f_cost = 1.0 - row.transaction_cost
    assert g_net == pytest.approx(g_gross * f_cost)
    # The additive form r_net = r_gross − q gives a different number:
    # 0.6875 − 0.12125 = 0.56625 vs the true 0.482890625 — reject it.
    assert row.net_return != pytest.approx(
        row.gross_return - row.transaction_cost
    )


# ---------------------------------------------------------------------------
# Floor 6 — the wealth recursion is multiplicative only (T5).
# ---------------------------------------------------------------------------


def test_wealth_recursion_multiplicative_only() -> None:
    path = build_ledger(**_world_x())
    assert path.n_periods == 1
    assert path.rows[0].wealth_open == 1.0  # unit-NAV basis W_0
    # W_1 = 1.0 × G_net,0 = 1.482890625.
    assert path.rows[0].wealth_close == pytest.approx(1.482890625)
    assert path.final_wealth == pytest.approx(1.482890625)
    # The additive recursion W_1 = W_0 (1 + r_gross − q) = 1.56625
    # differs — it is the rejected alternative.
    additive = 1.0 * (1.0 + 0.6875 - 0.12125)
    assert path.final_wealth != pytest.approx(additive)
    # Multi-row linkage: W_{t+1} = W_t × G_net,t row for row.
    multi = build_ledger(**_multi_period_world())
    for previous, current in zip(multi.rows, multi.rows[1:]):
        assert current.wealth_open == pytest.approx(previous.wealth_close)
        implied = previous.wealth_close * (1.0 + current.net_return)
        assert current.wealth_close == pytest.approx(implied)

# ---------------------------------------------------------------------------
# World Y — monthly accounting [T0,T1),[T1,T2),[T2,T3); horizon [T0,T3).
#   d0 decided T0, executed MID1 (row 0 interior);
#   d1 decided T0, executed MID2 — DELAYED past an accounting boundary:
#      charged to row 1 ([T1,T2)), not to row 0 where it was decided;
#   d2 decided T1, executed T2 — EXACTLY at row 1's period_end: it
#      belongs to row 2, and row 1's closing_weights stops at T2⁻.
# Row 2's first execution has pre_trade_weights == row 2's opening.
# Hand values (rate 0.01, one_way):
#   seg[T0,MID1)   {A:1.1,B:1.0} on {.5,.5}    -> D=1.05, drift {11/21,10/21}
#   seg[MID1,T1)   {A:1.0,B:1.1} on {.4,.6}    -> D=1.06
#   seg[T1,MID2)   {A:1.2,B:0.9} on {20/53,33/53} -> D=53.7/53
#   seg[MID2,T2)   {A:1.0,B:1.0}               -> D=1.0
#   seg[T2,T3)     {A:0.5,B:1.5} on {.3,.7}    -> D=1.2
# Realized-return law (floor 19, spy-recorded):
#   d0 interval [MID1,MID2): A: 1.0*1.2-1 = +0.20; B: 1.1*0.9-1 = -0.01
#     (crosses the T1 accounting boundary — NOT row 0's own segment
#      returns A:1.0-1=0.0, B:1.1-1=0.1);
#   d1 interval [MID2,T2):   A: 0.0, B: 0.0;
#   d2 interval [T2,T3) (final -> ledger horizon): A: -0.5, B: +0.5.
# ---------------------------------------------------------------------------
# World Z — within-period enter+exit (floor 11): period [T0,T2) opens
# {A,B}, executes at MID1 into {A,B,C}, at MID2 back to {A,B}; C is in
# an execution book only, yet must appear in the row's universe.
# (Implementation-tranche note: the horizon must end at T2 — MID2 is
# Feb 16, past T1 = Feb 1, and floor 16 / §6(iii) reject an execution
# outside the horizon; the fixture's own three-segment factor supply
# {T0, MID1, MID2} spans [T0, MID2) + a final drift segment.)
# ---------------------------------------------------------------------------


def _multi_period_world() -> dict[str, Any]:
    return {
        "initial_weights": {"A": 0.5, "B": 0.5},
        "accounting_instants": (T0, T1, T2, T3),
        "growth_factors": {
            T0: {"A": 1.1, "B": 1.0},
            MID1: {"A": 1.0, "B": 1.1},
            T1: {"A": 1.2, "B": 0.9},
            MID2: {"A": 1.0, "B": 1.0},
            T2: {"A": 0.5, "B": 1.5},
        },
        "decisions": [
            _decision({"A": 0.4, "B": 0.6}, T0, MID1),
            _decision({"A": 0.25, "B": 0.75}, T0, MID2),
            _decision({"A": 0.3, "B": 0.7}, T1, T2),
        ],
        "cost_model": Proportional(0.01, one_way),
    }


def _enter_exit_world() -> dict[str, Any]:
    return {
        "initial_weights": {"A": 0.5, "B": 0.5},
        "accounting_instants": (T0, T2),
        # The final segment re-enters the {A,B} book: factors name
        # exactly the held universe (the frozen within-segment law).
        "growth_factors": {
            T0: {"A": 1.0, "B": 1.0},
            MID1: {"A": 1.0, "B": 1.0, "C": 1.0},
            MID2: {"A": 1.0, "B": 1.0},
        },
        "decisions": [
            _decision({"A": 0.3, "B": 0.3, "C": 0.4}, MID1, MID1),
            _decision({"A": 0.5, "B": 0.5}, MID2, MID2),
        ],
        "cost_model": Proportional(0.001, one_way),
    }

# ---------------------------------------------------------------------------
# Floor 7 — unit-NAV budget is required, fail-closed (T5/D8).
# ---------------------------------------------------------------------------


def test_nav_budget_one_required_fail_closed() -> None:
    with pytest.raises(ValueError, match="budget"):
        build_ledger(**_world_x() | {"initial_weights": {"A": 0.5}})
    # A non-unit TARGET book rejects too (every book on the wealth path).
    with pytest.raises(ValueError, match="budget"):
        build_ledger(**_world_x(
            [_decision({"A": 0.25, "B": 0.25}, MID1, MID1),
             _decision({"A": 0.25, "B": 0.75}, MID2, MID2)]
        ))
    # Signed long/short summing to exactly 1 IS admissible.
    ls = build_ledger(
        initial_weights={"A": 1.5, "B": -0.5},
        accounting_instants=(T0, T1),
        growth_factors={T0: {"A": 1.0, "B": 1.0}},
        decisions=[],
        cost_model=Proportional(0.0, one_way),
    )
    assert ls.rows[0].gross_return == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Floors 8/9/17 — the engine law: composed once per execution, must
# return POST_TRADE = TARGET, non-target rejected (T4).
# ---------------------------------------------------------------------------


def test_engine_must_return_target_spy_returns_target() -> None:
    spy = _SpyEngine()
    path = build_ledger(**_world_x(), accounting_engine=spy)
    # One account() call per execution, in execution order.
    assert len(spy.calls) == 2
    called_targets = [
        dict(call[0].target_weights) for call in spy.calls
    ]
    assert called_targets == [
        {"A": 0.5, "B": 0.5},
        {"A": 0.25, "B": 0.75},
    ]
    # post_trade_weights equals the TARGET book (value equality).
    for execution, call in zip(path.rows[0].executions, spy.calls):
        assert dict(execution.post_trade_weights) == dict(call[0].target_weights)


def test_engine_non_target_rejected_fail_closed() -> None:
    with pytest.raises(ValueError, match="target"):
        build_ledger(**_world_x(), accounting_engine=_SkewEngine())


def test_ledger_composes_accounting_engine() -> None:
    # The default engine IS the reference ExactFillAccounting.
    reference = build_ledger(**_world_x())
    explicit = build_ledger(
        **_world_x(), accounting_engine=ExactFillAccounting()
    )
    assert reference == explicit
    for execution in reference.rows[0].executions:
        assert dict(execution.post_trade_weights) == dict(
            execution.target_weights
        )
    # ExactFillAccounting itself returns the target through the frozen
    # AccountingResult surface.
    result = ExactFillAccounting().account(
        _decision({"A": 0.4, "B": 0.6}, MID1, MID1),
        {"A": 0.5, "B": 0.5},
        {"A": 0.1, "B": -0.1},
    )
    assert isinstance(result, AccountingResult)
    assert dict(result.post_trade_weights) == {"A": 0.4, "B": 0.6}

# ---------------------------------------------------------------------------
# Floor 10 — closing_weights boundary semantics.
# ---------------------------------------------------------------------------


def test_closing_weights_boundary_semantics() -> None:
    path = build_ledger(**_multi_period_world())
    row0, row1, row2 = path.rows
    assert row0.period_start == T0 and row0.period_end == T1
    assert row1.period_start == T1 and row1.period_end == T2
    assert row2.period_start == T2 and row2.period_end == T3
    # First row opens on the initial book.
    assert dict(row0.opening_weights) == {"A": 0.5, "B": 0.5}
    # The T2 execution belongs to row 2, NOT row 1: row 1's closing is
    # the drifted book at T2⁻ (before that execution).
    assert len(row1.executions) == 1  # only the MID2 execution
    assert row1.executions[0].execution_time == MID2
    assert len(row2.executions) == 1  # the boundary execution
    assert row2.executions[0].execution_time == T2
    assert row2.executions[0].pre_trade_weights == row2.opening_weights
    # Exact adjacent-row linkage.
    assert row1.opening_weights == row0.closing_weights
    assert row2.opening_weights == row1.closing_weights


# ---------------------------------------------------------------------------
# Floor 11 — universe retains a within-period enter+exit identifier (T6).
# ---------------------------------------------------------------------------


def test_universe_retains_within_period_enter_exit() -> None:
    row = build_ledger(**_enter_exit_world()).rows[0]
    assert "C" in row.universe
    assert set(row.universe) == {"A", "B", "C"}
    # The rejected alternative — opening ∪ closing — would lose C.
    assert "C" not in set(row.opening_weights) | set(row.closing_weights)


# ---------------------------------------------------------------------------
# Floor 16 — an execution belongs to the period CONTAINING it (v1 floor).
# ---------------------------------------------------------------------------


def test_execution_assigned_to_period_containing_it() -> None:
    path = build_ledger(**_multi_period_world())
    row0, row1, row2 = path.rows
    # d0: decided T0, executed MID1 -> row 0 (mid-period interior).
    assert row0.executions[0].decision_time == T0
    assert row0.executions[0].execution_time == MID1
    # d1: decided T0, executed MID2 -> row 1, NOT row 0 (delayed past
    # the T1 boundary; charged where it executes).
    assert row1.executions[0].decision_time == T0
    assert row1.executions[0].execution_time == MID2
    # d2: executed exactly at T2 = row 1's end / row 2's start -> row 2.
    assert row2.executions[0].execution_time == T2
    # An execution outside the horizon rejects (chronology law).
    with pytest.raises(InvalidChronologyError):
        build_ledger(**_world_x(
            [_decision({"A": 0.5, "B": 0.5}, MID1, MID1),
             _decision({"A": 0.25, "B": 0.75}, MID2, MID2),
             _decision({"A": 0.5, "B": 0.5}, T2, T2)]
        ))
    # Non-monotone executions reject too (frozen sequence law).
    with pytest.raises(InvalidChronologyError):
        build_ledger(**_world_x(
            [_decision({"A": 0.25, "B": 0.75}, MID2, MID2),
             _decision({"A": 0.5, "B": 0.5}, MID1, MID1)]
        ))

# ---------------------------------------------------------------------------
# Floor 12 — q boundary rejects through the frozen cost law (v1 floor).
# ---------------------------------------------------------------------------


def test_q_boundary_rejects() -> None:
    # q = rate × one_way(turnover) must stay < 1.  one_way of this
    # full-flip trade (A: 1.0 -> 0.0, B: 0.0 -> 1.0) is exactly 1.0, so
    # rate 1.0 gives q = 1 (boundary) and 1.5 gives q > 1: both reject.
    for rate in (1.0, 1.5):
        with pytest.raises(ValueError):
            build_ledger(
                initial_weights={"A": 1.0, "B": 0.0},
                accounting_instants=(T0, T1),
                growth_factors={T0: {"A": 1.0, "B": 1.0}},
                decisions=[
                    _decision({"A": 0.0, "B": 1.0}, T0, T0)
                ],
                cost_model=Proportional(rate, one_way),
            )
    # q just under the boundary stays lawful.
    lawful = build_ledger(
        initial_weights={"A": 1.0, "B": 0.0},
        accounting_instants=(T0, T1),
        growth_factors={T0: {"A": 1.0, "B": 1.0}},
        decisions=[_decision({"A": 0.0, "B": 1.0}, T0, T0)],
        cost_model=Proportional(0.99, one_way),
    )
    assert lawful.rows[0].transaction_cost == pytest.approx(0.99)


def test_row_aggregate_conventions_declared() -> None:
    row = build_ledger(**_world_x()).rows[0]
    q1, q2 = 0.05, 0.075
    # Row turnover = Σ execution turnover (NOT per-execution-only, NOT 0.5×).
    assert row.turnover == pytest.approx(1.0 / 6.0 + 0.25)
    assert row.turnover == pytest.approx(
        sum(e.turnover for e in row.executions)
    )
    # Row transaction_cost = 1 − F_cost,t (NOT Σ q_k = 0.125).
    assert row.transaction_cost == pytest.approx(1.0 - (1 - q1) * (1 - q2))
    assert row.transaction_cost != pytest.approx(q1 + q2)
    # Per-execution records carry their own conventions.
    assert row.executions[0].turnover == pytest.approx(1.0 / 6.0)
    assert row.executions[0].transaction_cost == pytest.approx(q1)
    assert row.executions[1].turnover == pytest.approx(0.25)
    assert row.executions[1].transaction_cost == pytest.approx(q2)
    # The per-execution q values are the frozen cost law on the
    # retained trade — recompute them from the record's own primitives.
    from portlearn.turnover import one_way as ow

    for execution, expected_q in zip(row.executions, (0.05, 0.075)):
        assert ow(execution.trade) == pytest.approx(execution.turnover)
        assert 0.05 <= execution.transaction_cost < 1.0
        assert expected_q == pytest.approx(execution.transaction_cost)

# ---------------------------------------------------------------------------
# Floor 13 — inherited exceptions, no new classes.
# ---------------------------------------------------------------------------


def test_exceptions_inherited_no_new_classes() -> None:
    # Naive instants anywhere reject with the frozen NaiveTimestampError.
    naive_world = _world_x() | {
        "accounting_instants": (
            datetime(2026, 1, 1), datetime(2026, 3, 1)  # noqa: DTZ001 — deliberately naive instants (NaiveTimestampError probe)
        )
    }
    with pytest.raises(NaiveTimestampError):
        build_ledger(**naive_world)
    # Overlapping/non-increasing accounting periods reject with the
    # frozen InvalidChronologyError (strictly increasing law).
    with pytest.raises(InvalidChronologyError):
        build_ledger(**_world_x() | {
            "accounting_instants": (T0, T0, T2)
        })
    # Domain violations raise plain ValueError through frozen laws:
    # a negative growth factor rejects through the frozen factor law.
    with pytest.raises(ValueError):
        build_ledger(**_world_x() | {
            "growth_factors": {
                T0: {"A": -0.5, "B": 1.0},
                MID1: {"A": 1.0, "B": 1.0},
                MID2: {"A": 1.5, "B": 1.0},
            }
        })
    # A non-finite factor rejects too.
    with pytest.raises(ValueError):
        build_ledger(**_world_x() | {
            "growth_factors": {
                T0: {"A": float("inf"), "B": 1.0},
                MID1: {"A": 1.0, "B": 1.0},
                MID2: {"A": 1.5, "B": 1.0},
            }
        })
    # No ledger module defines a new exception class.
    for name, member in inspect.getmembers(
        ledger_module, inspect.isclass
    ):
        assert not issubclass(
            member, Exception
        ), f"pl.ledger must not define exception classes; found {name}"


# ---------------------------------------------------------------------------
# Floor 14 — frozen rows, deterministic value equality, no hash promises.
# ---------------------------------------------------------------------------


def test_rows_frozen_value_equal_not_hash_guaranteed() -> None:
    first = build_ledger(**_world_x())
    second = build_ledger(**_world_x())
    # Replay: identical inputs produce value-equal, row-for-row
    # identical ledgers.
    assert first == second
    assert first.rows == second.rows
    assert first.rows[0] == second.rows[0]
    # Immutable: attribute assignment rejects on row and execution.
    with pytest.raises(AttributeError):
        first.rows[0].gross_return = 0.5
    with pytest.raises(AttributeError):
        first.rows[0].executions[0].turnover = 0.0
    # The books are immutable MappingProxyType views (weight-book law retained).
    with pytest.raises(TypeError):
        first.rows[0].opening_weights["A"] = 0.25
    # No hash guarantee: hash() is never asserted anywhere in the ledger —
    # but the objects must still support deterministic value equality.
    assert first.rows[0].executions[0].trade.source.weights == (
        second.rows[0].executions[0].trade.source.weights
    )
    # LedgerPath exposes exactly the path-level summary fields.
    assert first.n_periods == 1
    assert first.final_wealth == first.rows[0].wealth_close


# ---------------------------------------------------------------------------
# Floor 19 — realized returns follow the execution holding interval
# (the realized-returns holding-interval law; the spy records what the engine received).
# ---------------------------------------------------------------------------


def test_engine_realized_returns_follow_execution_holding_interval() -> None:
    spy = _SpyEngine()
    build_ledger(**_multi_period_world(), accounting_engine=spy)
    assert len(spy.calls) == 3
    d0_returns = dict(spy.calls[0][2])
    d1_returns = dict(spy.calls[1][2])
    d2_returns = dict(spy.calls[2][2])
    # d0 [MID1, MID2) CROSSES the T1 boundary: compounded, not one row.
    assert d0_returns == pytest.approx({"A": 0.2, "B": -0.01})
    # ...and NOT the first segment's own factors (A: 0.0, B: 0.1).
    assert d0_returns != pytest.approx({"A": 0.0, "B": 0.1})
    # d1 [MID2, T2): identity factors.
    assert d1_returns == pytest.approx({"A": 0.0, "B": 0.0})
    # d2 [T2, T3) is FINAL: its interval ends at the LEDGER HORIZON.
    assert d2_returns == pytest.approx({"A": -0.5, "B": 0.5})


# ---------------------------------------------------------------------------
# Floor 20 — initial_wealth is positive, finite, and real, fail-closed
# (T5/W0; the §8 line-103 W₀ ≤ 0 edge probe).
# ---------------------------------------------------------------------------


def test_initial_wealth_positive_fail_closed() -> None:
    # W0 must be strictly positive and finite: 0, negative, NaN, and
    # +inf each reject with ValueError naming 'initial_wealth' (and
    # -inf / bool / non-real reject through the same gate).
    for bad in (0, -1, float("nan"), float("inf"), float("-inf"), True, "1"):
        with pytest.raises(ValueError, match="initial_wealth"):
            build_ledger(**_world_x(), initial_wealth=bad)
    # The default call keeps the unit basis: wealth_open == 1.0.
    default = build_ledger(**_world_x()).rows[0]
    assert default.wealth_open == 1.0
    # A researcher-scale W0 scales every wealth field by exactly W0 —
    # the multiplicative recursion is unchanged (W_close = W0 × G_net;
    # World X G_net = 1.482890625, binary-exact).
    w0 = 1_000_000
    scaled = build_ledger(**_world_x(), initial_wealth=w0).rows[0]
    assert scaled.wealth_open == w0
    assert scaled.wealth_close == pytest.approx(w0 * 1.482890625)
    assert scaled.wealth_close == pytest.approx(w0 * (1.0 + default.net_return))
    # W0 is NOT the weight budget: the unit-NAV book law (D8) is a
    # separate gate that still rejects a budget-0.5 book at any W0.
    with pytest.raises(ValueError, match="budget"):
        build_ledger(**_world_x() | {"initial_weights": {"A": 0.5}},
                     initial_wealth=w0)


# ---------------------------------------------------------------------------
# Floor 18 — pl.ledger imports only frozen modules (composition law).
# ---------------------------------------------------------------------------


def test_ledger_imports_only_frozen_modules() -> None:
    import ast

    tree = ast.parse(inspect.getsource(ledger_module))
    frozen = {
        "portlearn.weights",
        "portlearn.timing",
        "portlearn.calendar",
        "portlearn.trades",
        "portlearn.turnover",
        "portlearn.costs",
        "portlearn.interfaces",
        "portlearn.rebalance",
    }
    stdlib_roots = {"dataclasses", "datetime", "typing", "collections"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert (
                    root in stdlib_roots or alias.name in frozen
                ), f"illegal import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                local = module.split(".")[-1] if module else ""
                assert local in {
                    m.split(".")[-1] for m in frozen
                }, f"illegal relative import: {module}"
            elif module == "portlearn":
                for alias in node.names:
                    qualified = f"portlearn.{alias.name}"
                    assert qualified in frozen, f"illegal from-import: {qualified}"
            else:
                assert module.split(".")[0] in stdlib_roots, (
                    f"illegal import: {module}"
                )
    # The mutable siblings are NOT composed anywhere in the module text.
    # pl.rebalance IS composed (the drift law), so it is
    # absent from this banned tuple; the mutable siblings stay banned.
    source = inspect.getsource(ledger_module)
    for banned in ("backtest", "reporting", "strategies"):
        assert banned not in source, f"pl.ledger must not reference {banned!r}"
