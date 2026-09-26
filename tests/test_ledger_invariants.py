"""Hand-calculated invariant battery for the fixed portfolio ledger.

This module is a test-only module: it adds no production code,
edits no fixed surface, and exercises only the fixed ledger public
interface (``portlearn.ledger.build_ledger`` with the fixed weight,
drift, and cost primitives it composes).  Where ``tests/test_ledger_contracts.py``
freezes the ledger behavioral contracts, this battery independently
re-derives the eleven invariant families I1–I11
from first principles:

* I1  wealth closure ``W_{t+1} = W_t × G_net,t`` on the unit-NAV basis
  and the unit budget of every recorded book;
* I2  segment-composed gross ``G_gross,t = Π_m D_m`` (never a single
  denominator), with execution-free rows reducing to ``D`` itself;
* I3  multiplicative costs ``F_cost,t = Π_k (1 − q_k)``, with a
  cost-free row paying exactly nothing;
* I4  POST_TRADE = TARGET on every execution, unconditional, with any
  compliant engine producing a value-equal ledger;
* I5  the per-holding segment drift identity chained across
  executions and periods, opening → drift → execution → … → closing;
* I5b execution turnover summing to row turnover under both fixed
  conventions (one_way and two_sided);
* I6  execution ownership by EXECUTION instant on ``[start, end)`` —
  delayed executions charge the executing period, a boundary
  execution opens the next period;
* I7  the row universe as the union of ALL retained books, retaining
  an entered-and-exited asset at effective weight zero;
* I8  purity/replay: independent assemblies of identical inputs
  produce value-equal, row-for-row identical ledgers;
* I9  behavioral immutability of the whole record;
* I10 unconditional admission minimums with the fixed named exceptions;
* I11 an anti-reference meta-floor: every expected value is produced by
  in-module ``_hand_*`` helpers using only literals, ``Fraction``,
  and stdlib arithmetic — never by calling the library under test.

Numerical discipline (D5/D5b).  Every fixture below is a dyadic
rational world: all weights, growth factors, segment denominators,
drift endpoints, cost fractions, and wealth products have
denominators that are powers of two, so every intermediate and
recorded float is bit-exact.  Consequently NO tolerance is used
anywhere in this file — there is not a single ``approx`` assertion.
The dyadic guarantee is itself enforced at fixture-build time by
``_hand_dyadic_float``, which refuses to convert any non-dyadic
rational.
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Mapping
from datetime import UTC, datetime
from fractions import Fraction
from typing import Any

import pytest

from portlearn.costs import Proportional
from portlearn.interfaces import AccountingResult, PortfolioDecision
from portlearn.ledger import ExactFillAccounting, build_ledger
from portlearn.timing import InvalidChronologyError, NaiveTimestampError
from portlearn.turnover import one_way, two_sided

# ---------------------------------------------------------------------------
# Calendar (UTC month boundaries and mid-month instants).
# ---------------------------------------------------------------------------

T0 = datetime(2026, 1, 1, tzinfo=UTC)
T1 = datetime(2026, 2, 1, tzinfo=UTC)
T2 = datetime(2026, 3, 1, tzinfo=UTC)
T3 = datetime(2026, 4, 1, tzinfo=UTC)

JAN8 = datetime(2026, 1, 8, tzinfo=UTC)
JAN15 = datetime(2026, 1, 15, tzinfo=UTC)
JAN20 = datetime(2026, 1, 20, tzinfo=UTC)
JAN24 = datetime(2026, 1, 24, tzinfo=UTC)
JAN25 = datetime(2026, 1, 25, tzinfo=UTC)
FEB12 = datetime(2026, 2, 12, tzinfo=UTC)
FEB20 = datetime(2026, 2, 20, tzinfo=UTC)
FEB25 = datetime(2026, 2, 25, tzinfo=UTC)

# ---------------------------------------------------------------------------
# Hand-derived arithmetic helpers (the I11 anti-reference perimeter).
#
# Every expected value in this battery is computed below using ONLY
# literal rationals, ``Fraction``, and stdlib arithmetic.  None of
# these helpers touches the library under test; the meta-floor at the
# end of the file walks this module's own AST to keep that true.
# ---------------------------------------------------------------------------


def _hand_dyadic_float(value: Fraction) -> float:
    """Convert an exact rational to its bit-exact float.

    Refuses any non-dyadic rational (denominator not a power of two):
    the no-tolerance discipline of this battery is valid exactly as
    long as every expected value is dyadic, so the guard makes that
    claim self-enforcing rather than aspirational.
    """
    ratio = Fraction(value)
    denominator = ratio.denominator
    if denominator & (denominator - 1):
        raise AssertionError(
            f"non-dyadic expected value {ratio}: the battery's "
            "exact-equality discipline requires power-of-two "
            "denominators everywhere"
        )
    return float(ratio)


def _hand_book(book: Mapping[str, Fraction]) -> dict[str, float]:
    """A float weight book from an exact rational fixture."""
    return {
        asset: _hand_dyadic_float(Fraction(weight))
        for asset, weight in book.items()
    }


def _hand_segment_denominator(
    book: Mapping[str, Fraction], factors: Mapping[str, Fraction]
) -> Fraction:
    """The holding-segment denominator ``D = Σᵢ wᵢ·gᵢ`` by hand.

    The factor supply must name exactly the held universe (the fixed
    within-segment law); a fixture violating it is a fixture bug.
    """
    if set(book) != set(factors):
        raise AssertionError(
            f"fixture universe mismatch: book {sorted(book)} vs "
            f"factors {sorted(factors)}"
        )
    total = Fraction(0)
    for asset in book:
        total += Fraction(book[asset]) * Fraction(factors[asset])
    return total


def _hand_segment_growth(
    book: Mapping[str, Fraction], factors: Mapping[str, Fraction]
) -> tuple[Fraction, dict[str, Fraction]]:
    """One segment by hand: ``(D, drifted book)`` with ``wᵢ·gᵢ/D``."""
    denominator = _hand_segment_denominator(book, factors)
    drifted = {
        asset: Fraction(book[asset]) * Fraction(factors[asset])
        / denominator
        for asset in book
    }
    return denominator, drifted


def _hand_deltas(
    source: Mapping[str, Fraction], destination: Mapping[str, Fraction]
) -> list[Fraction]:
    """``Δw = destination − source`` per asset over the union."""
    return [
        Fraction(destination.get(asset, Fraction(0)))
        - Fraction(source.get(asset, Fraction(0)))
        for asset in sorted(set(source) | set(destination))
    ]


def _hand_one_way(
    source: Mapping[str, Fraction], destination: Mapping[str, Fraction]
) -> Fraction:
    """``one_way = 0.5 · Σᵢ|Δwᵢ|`` — the half convention, by hand."""
    total = Fraction(0)
    for delta in _hand_deltas(source, destination):
        total += Fraction(1, 2) * abs(delta)
    return total


def _hand_two_sided(
    source: Mapping[str, Fraction], destination: Mapping[str, Fraction]
) -> Fraction:
    """``two_sided = Σᵢ|Δwᵢ|`` — buys and sells both count, by hand."""
    total = Fraction(0)
    for delta in _hand_deltas(source, destination):
        total += abs(delta)
    return total


def _hand_cost_factor(cost_fractions: list[Fraction]) -> Fraction:
    """``F_cost = Π_k (1 − q_k)`` — multiplicative composition, by hand."""
    factor = Fraction(1)
    for q in cost_fractions:
        factor *= Fraction(1) - Fraction(q)
    return factor


def _hand_product(values: list[Fraction]) -> Fraction:
    """Exact product of exact rationals."""
    result = Fraction(1)
    for value in values:
        result *= Fraction(value)
    return result


def _hand_book_at_row_open(
    world: Mapping[str, Any], row_index: int
) -> dict[str, Fraction]:
    """Hand-chain the drift walk up to the opening of the given row.

    Mirrors the ownership and segment laws in exact rationals: within
    each earlier period the book drifts from segment start to the next
    execution instant, the execution replaces the book with its
    target, and the target seeds the next segment; the final segment
    of each period ends at the closing book, which is the next
    period's opening.
    """
    current = {
        asset: Fraction(weight) for asset, weight in world["initial"].items()
    }
    boundaries = list(world["instants"])
    window = boundaries[: row_index + 1]
    for start, end in zip(window, window[1:]):  # noqa: RUF007 — itertools is outside the floor-18 stdlib allowlist
        segment_start = start
        for _, executes, target in world["executions"]:
            if not start <= executes < end:
                continue
            if segment_start < executes:
                _, current = _hand_segment_growth(
                    current, world["factors"][segment_start]
                )
            current = {
                asset: Fraction(weight) for asset, weight in target.items()
            }
            segment_start = executes
        _, current = _hand_segment_growth(
            current, world["factors"][segment_start]
        )
    return current


# ---------------------------------------------------------------------------
# World M (multi-execution) — three assets {A, B, C}, three monthly
# periods [T0,T1), [T1,T2), [T2,T3); TWO executions inside period 0
# (I2/I3), one inside period 1, period 2 drift-only (I2b/I3b).
# Rate 1/4 on one_way throughout.
#
# Hand derivation (every value dyadic, hence float-exact):
#   open₀ {A: 1/2, B: 1/4, C: 1/4}
#   seg [T0, JAN8)   g {3, 1, 1}        terms {3/2, 1/4, 1/4}   D₁ = 2
#       drift {3/4, 1/8, 1/8}
#   exec JAN8 → {1/2, 1/4, 1/4}: |Δ| = {1/4, 1/8, 1/8}
#       one_way = 1/4, q₁ = 1/16, F₁ = 15/16
#   seg [JAN8, JAN24) g {1, 3/2, 1/2}   terms {1/2, 3/8, 1/8}   D₂ = 1
#       drift {1/2, 3/8, 1/8}
#   exec JAN24 → {1/4, 1/4, 1/2}: |Δ| = {1/4, 1/8, 3/8}
#       one_way = 3/8, q₂ = 3/32, F₂ = 29/32
#   seg [JAN24, T1)  g {3, 3/2, 3/4}    terms {3/4, 3/8, 3/8}   D₃ = 3/2
#       drift {1/2, 1/4, 1/4}  = close₀ = open₁
#   Row 0: G_gross = 2·1·3/2 = 3;  F_cost = 15/16·29/32 = 435/512;
#          G_net = 1305/512; turnover = 1/4 + 3/8 = 5/8; W: 1 → 1305/512.
#   seg [T1, FEB12)  g {1/2, 3/2, 3/2}  terms {1/4, 3/8, 3/8}   D₁ = 1
#       drift {1/4, 3/8, 3/8}
#   exec FEB12 → {1/8, 3/8, 1/2}: |Δ| = {1/8, 0, 1/8}
#       one_way = 1/8, q = 1/32, F = 31/32
#   seg [FEB12, T2)  g {4, 2, 3/2}      terms {1/2, 3/4, 3/4}   D₂ = 2
#       drift {1/4, 3/8, 3/8}  = close₁ = open₂
#   Row 1: G_gross = 2; F_cost = 31/32; G_net = 31/16; turnover 1/8.
#   seg [T2, T3)     g {2, 2, 2}        terms {1/2, 3/4, 3/4}   D = 2
#       drift {1/4, 3/8, 3/8} unchanged = close₂
#   Row 2 (drift-only): G_gross = 2; F_cost = 1; G_net = 2; turnover 0.
#   Wealth: 1 → 1305/512 → 40455/8192 → 40455/4096 (final_wealth).
#   Rejected single-denominator form for row 0 (per-asset compounding
#   on the opening book): A 3·1·3 = 9, B 1·3/2·3/2 = 9/4,
#   C 1·1/2·3/4 = 3/8; 1/2·9 + 1/4·9/4 + 1/4·3/8 = 165/32 ≠ 3.
# ---------------------------------------------------------------------------

_WORLD_MULTI = {
    "initial": {"A": Fraction(1, 2), "B": Fraction(1, 4), "C": Fraction(1, 4)},
    "instants": (T0, T1, T2, T3),
    "factors": {
        T0: {"A": Fraction(3), "B": Fraction(1), "C": Fraction(1)},
        JAN8: {"A": Fraction(1), "B": Fraction(3, 2), "C": Fraction(1, 2)},
        JAN24: {"A": Fraction(3), "B": Fraction(3, 2), "C": Fraction(3, 4)},
        T1: {"A": Fraction(1, 2), "B": Fraction(3, 2), "C": Fraction(3, 2)},
        FEB12: {"A": Fraction(4), "B": Fraction(2), "C": Fraction(3, 2)},
        T2: {"A": Fraction(2), "B": Fraction(2), "C": Fraction(2)},
    },
    "executions": (
        (
            JAN8,
            JAN8,
            {"A": Fraction(1, 2), "B": Fraction(1, 4), "C": Fraction(1, 4)},
        ),
        (
            JAN24,
            JAN24,
            {"A": Fraction(1, 4), "B": Fraction(1, 4), "C": Fraction(1, 2)},
        ),
        (
            FEB12,
            FEB12,
            {"A": Fraction(1, 8), "B": Fraction(3, 8), "C": Fraction(1, 2)},
        ),
    ),
    "rate": Fraction(1, 4),
}

# Per-row hand tables: (gross_return, net_return, transaction_cost,
# turnover, wealth_open, wealth_close).
_MULTI_ROWS = (
    (
        Fraction(2),
        Fraction(793, 512),
        Fraction(77, 512),
        Fraction(5, 8),
        Fraction(1),
        Fraction(1305, 512),
    ),
    (
        Fraction(1),
        Fraction(15, 16),
        Fraction(1, 32),
        Fraction(1, 8),
        Fraction(1305, 512),
        Fraction(40455, 8192),
    ),
    (
        Fraction(1),
        Fraction(1),
        Fraction(0),
        Fraction(0),
        Fraction(40455, 8192),
        Fraction(40455, 4096),
    ),
)

# ---------------------------------------------------------------------------
# World D (delayed and boundary executions) — two assets {A, B}, three
# monthly periods; I6 ownership semantics.
#   d0 decided T0,    executed JAN20 (row 0 interior);
#   d1 decided T0,    executed FEB20 — DELAYED past the T1 boundary:
#        charged to row 1, the period CONTAINING the execution, not to
#        row 0 where it was decided;
#   d2 decided FEB25, executed T2 — EXACTLY on a boundary: it belongs
#        to row 2 ([T2,T3)), at that row's period_open, with NO drift
#        before it (pre_trade == row 2 opening == row 1 closing).
# Hand derivation (rate 1/4, one_way):
#   open₀ {A: 1/2, B: 1/2}
#   seg [T0, JAN20)   g {3/2, 1/2}  terms {3/4, 1/4}  D = 1, drift {3/4, 1/4}
#   exec JAN20 → {1/4, 3/4}: one_way = 1/2, q = 1/8, F = 7/8
#   seg [JAN20, T1)   g {1, 1}     D = 1, drift {1/4, 3/4} = close₀
#   Row 0: G = 1, F = 7/8, G_net = 7/8, turnover 1/2, W: 1 → 7/8.
#   seg [T1, FEB20)   g {3/2, 1/2}  terms {3/8, 3/8}  D = 3/4,
#       drift {1/2, 1/2}  (pre-trade of the DELAYED d1)
#   exec FEB20 → {3/4, 1/4}: one_way = 1/4, q = 1/16, F = 15/16
#   seg [FEB20, T2)   g {1/2, 3/2}  terms {3/8, 3/8}  D = 3/4,
#       drift {1/2, 1/2} = close₁
#   Row 1: G = 9/16, F = 15/16, G_net = 135/256, turnover 1/4,
#          W: 7/8 → 945/2048.
#   exec T2 (boundary, period_open of row 2): pre == {1/2, 1/2} == open₂
#       → {1/4, 3/4}: one_way = 1/4, q = 1/16, F = 15/16
#   seg [T2, T3)      g {1, 1}     D = 1, drift {1/4, 3/4} = close₂
#   Row 2: G = 1, F = 15/16, G_net = 15/16, turnover 1/4,
#          W: 945/2048 → 14175/32768.
#   Rejected single-denominator form for row 1: compounded per-asset
#   factors A 3/2·1/2 = 3/4, B 1/2·3/2 = 3/4 on {1/4, 3/4} give D = 3/4
#   ≠ 9/16.
# ---------------------------------------------------------------------------

_WORLD_DELAYED = {
    "initial": {"A": Fraction(1, 2), "B": Fraction(1, 2)},
    "instants": (T0, T1, T2, T3),
    "factors": {
        T0: {"A": Fraction(3, 2), "B": Fraction(1, 2)},
        JAN20: {"A": Fraction(1), "B": Fraction(1)},
        T1: {"A": Fraction(3, 2), "B": Fraction(1, 2)},
        FEB20: {"A": Fraction(1, 2), "B": Fraction(3, 2)},
        T2: {"A": Fraction(1), "B": Fraction(1)},
    },
    "executions": (
        (T0, JAN20, {"A": Fraction(1, 4), "B": Fraction(3, 4)}),
        (T0, FEB20, {"A": Fraction(3, 4), "B": Fraction(1, 4)}),
        (FEB25, T2, {"A": Fraction(1, 4), "B": Fraction(3, 4)}),
    ),
    "rate": Fraction(1, 4),
}

_DELAYED_ROWS = (
    (
        Fraction(0),
        Fraction(-1, 8),
        Fraction(1, 8),
        Fraction(1, 2),
        Fraction(1),
        Fraction(7, 8),
    ),
    (
        Fraction(-7, 16),
        Fraction(-121, 256),
        Fraction(1, 16),
        Fraction(1, 4),
        Fraction(7, 8),
        Fraction(945, 2048),
    ),
    (
        Fraction(0),
        Fraction(-1, 16),
        Fraction(1, 16),
        Fraction(1, 4),
        Fraction(945, 2048),
        Fraction(14175, 32768),
    ),
)

# ---------------------------------------------------------------------------
# World E (enter-and-exit within one period) — assets {A, B} plus C,
# TWO periods [T0,T1), [T1,T2); C enters at JAN15 and exits at JAN25,
# both inside period 0, so C appears in NO opening/closing book of the
# row yet MUST be retained in the row universe (I7); period 1 is
# drift-only.  Rate 1/4 on one_way.
# Hand derivation:
#   open₀ {A: 1/2, B: 1/2}
#   seg [T0, JAN15)  g {3/2, 1/2}       terms {3/4, 1/4}  D = 1,
#       drift {3/4, 1/4}
#   exec JAN15 → ENTER C {A: 1/4, B: 1/4, C: 1/2}:
#       |Δ| = {1/2, 0, 1/2}, one_way = 1/2, q = 1/8, F = 7/8
#   seg [JAN15, JAN25) g {2, 2, 2}      terms {1/2, 1/2, 1}  D = 2,
#       drift {1/4, 1/4, 1/2} (uniform growth leaves weights fixed)
#   exec JAN25 → EXIT C {A: 1/2, B: 1/2}:
#       |Δ| = {1/4, 1/4, 1/2}, one_way = 1/2, q = 1/8, F = 7/8
#   seg [JAN25, T1)  g {3/2, 1/2}       terms {3/4, 1/4}  D = 1,
#       drift {3/4, 1/4} = close₀ (C is gone from the closing book)
#   Row 0: G = 1·2·1 = 2; F = 7/8·7/8 = 49/64; G_net = 49/32;
#          turnover = 1; universe {A, B, C}; W: 1 → 49/32.
#   seg [T1, T2)     g {1/2, 3/2}       terms {3/8, 3/8}  D = 3/4,
#       drift {1/2, 1/2} = close₁
#   Row 1 (drift-only): G = 3/4, F = 1, G_net = 3/4, turnover 0,
#          universe {A, B}; W: 49/32 → 147/128.
# Under two_sided (I5b): each trade's Σᵢ|Δwᵢ| = 1, so q = 1/4 twice,
# F = 9/16, row turnover = 2, G_net = 2·9/16 = 9/8.
# ---------------------------------------------------------------------------

_WORLD_ENTER_EXIT = {
    "initial": {"A": Fraction(1, 2), "B": Fraction(1, 2)},
    "instants": (T0, T1, T2),
    "factors": {
        T0: {"A": Fraction(3, 2), "B": Fraction(1, 2)},
        JAN15: {"A": Fraction(2), "B": Fraction(2), "C": Fraction(2)},
        JAN25: {"A": Fraction(3, 2), "B": Fraction(1, 2)},
        T1: {"A": Fraction(1, 2), "B": Fraction(3, 2)},
    },
    "executions": (
        (
            JAN15,
            JAN15,
            {"A": Fraction(1, 4), "B": Fraction(1, 4), "C": Fraction(1, 2)},
        ),
        (JAN25, JAN25, {"A": Fraction(1, 2), "B": Fraction(1, 2)}),
    ),
    "rate": Fraction(1, 4),
}

_ENTER_EXIT_ROWS = (
    (
        Fraction(1),
        Fraction(17, 32),
        Fraction(15, 64),
        Fraction(1),
        Fraction(1),
        Fraction(49, 32),
    ),
    (
        Fraction(-1, 4),
        Fraction(-1, 4),
        Fraction(0),
        Fraction(0),
        Fraction(49, 32),
        Fraction(147, 128),
    ),
)

_ALL_WORLDS = (_WORLD_MULTI, _WORLD_DELAYED, _WORLD_ENTER_EXIT)
_ALL_ROW_TABLES = (_MULTI_ROWS, _DELAYED_ROWS, _ENTER_EXIT_ROWS)


# ---------------------------------------------------------------------------
# Construction helpers (public API used normally — outside the
# ``_hand_*`` anti-reference perimeter, per I11).
# ---------------------------------------------------------------------------


def _ledger_inputs(
    world: Mapping[str, Any],
    convention: Any = one_way,
    initial_wealth: float = 1.0,
) -> dict[str, Any]:
    """Fresh ``build_ledger`` keyword inputs from a rational world.

    Every call assembles fresh mutable mappings, so two calls model
    two genuinely independent assemblies (I8).
    """
    return {
        "initial_weights": _hand_book(world["initial"]),
        "accounting_instants": tuple(world["instants"]),
        "growth_factors": {
            start: _hand_book(book) for start, book in world["factors"].items()
        },
        "decisions": [
            PortfolioDecision(
                decision_time=decided,
                execution_time=executes,
                target_weights=_hand_book(target),
            )
            for decided, executes, target in world["executions"]
        ],
        "cost_model": Proportional(
            _hand_dyadic_float(world["rate"]), convention
        ),
        "initial_wealth": initial_wealth,
    }


def _build(
    world: Mapping[str, Any],
    convention: Any = one_way,
    initial_wealth: float = 1.0,
) -> Any:
    return build_ledger(**_ledger_inputs(world, convention, initial_wealth))


class _RecordingEngine:
    """A compliant admissible engine: returns the TARGET book while
    recording every call (I4b twin of the exact-fill default)."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any, Any]] = []

    def account(
        self,
        decision: Any,
        pre_trade_weights: Any,
        realized_returns: Any,
    ) -> AccountingResult:
        self.calls.append((decision, pre_trade_weights, realized_returns))
        return AccountingResult(dict(decision.target_weights))


class _SkewEngine:
    """A NON-compliant engine: nudges one target weight, so the
    post-trade book is no longer the target (I4 unconditional probe)."""

    def account(
        self,
        decision: Any,
        pre_trade_weights: Any,
        realized_returns: Any,
    ) -> AccountingResult:
        skewed = dict(decision.target_weights)
        key = min(skewed)
        skewed[key] = skewed[key] + 2**-10
        return AccountingResult(skewed)


def _assert_paths_value_equal(first: Any, second: Any) -> None:
    """Row-for-row, field-for-field value equality of two paths."""
    assert first.n_periods == second.n_periods
    assert len(first.rows) == len(second.rows)
    for left, right in zip(first.rows, second.rows):
        assert left.period_start == right.period_start
        assert left.period_end == right.period_end
        assert dict(left.opening_weights) == dict(right.opening_weights)
        assert dict(left.closing_weights) == dict(right.closing_weights)
        assert left.gross_return == right.gross_return
        assert left.net_return == right.net_return
        assert left.turnover == right.turnover
        assert left.transaction_cost == right.transaction_cost
        assert left.wealth_open == right.wealth_open
        assert left.wealth_close == right.wealth_close
        assert set(left.universe) == set(right.universe)
        assert len(left.executions) == len(right.executions)
        for exec_left, exec_right in zip(left.executions, right.executions):
            assert exec_left.decision_time == exec_right.decision_time
            assert exec_left.execution_time == exec_right.execution_time
            assert dict(exec_left.target_weights) == dict(
                exec_right.target_weights
            )
            assert dict(exec_left.pre_trade_weights) == dict(
                exec_right.pre_trade_weights
            )
            assert dict(exec_left.post_trade_weights) == dict(
                exec_right.post_trade_weights
            )
            assert exec_left.turnover == right_turnover(exec_right)
            assert exec_left.transaction_cost == exec_right.transaction_cost
            assert dict(exec_left.trade.delta) == dict(exec_right.trade.delta)
            assert dict(exec_left.trade.source.weights) == dict(
                exec_right.trade.source.weights
            )
            assert dict(exec_left.trade.destination.weights) == dict(
                exec_right.trade.destination.weights
            )


def right_turnover(execution: Any) -> float:
    """Read the counterpart execution's turnover (small indirection so
    the paired-assertion line stays within the line-length budget)."""
    return execution.turnover


def _assert_drift_chain(path: Any, world: Mapping[str, Any]) -> None:
    """I5: walk every row segment-by-segment against the hand arithmetic.

    Chained across the whole path: each row opens on the previous
    row's hand closing book; within a row, each segment endpoint feeds
    the next execution's pre-trade book (an interior execution), or IS
    the opening book (a boundary execution at period_open — no drift
    before it, I6); each target becomes the next segment's start; the
    final segment's endpoint is the closing book.
    """
    current = {
        asset: Fraction(weight) for asset, weight in world["initial"].items()
    }
    for row in path.rows:
        assert dict(row.opening_weights) == _hand_book(current)
        expected = [
            (executes, target)
            for _, executes, target in world["executions"]
            if row.period_start <= executes < row.period_end
        ]
        assert [e.execution_time for e in row.executions] == [
            t for t, _ in expected
        ]
        segment_start = row.period_start
        for execution, (wanted, target) in zip(row.executions, expected):
            if segment_start < wanted:
                _, current = _hand_segment_growth(
                    current, world["factors"][segment_start]
                )
            assert dict(execution.pre_trade_weights) == _hand_book(current)
            assert dict(execution.target_weights) == _hand_book(target)
            assert dict(execution.post_trade_weights) == _hand_book(target)
            current = {
                asset: Fraction(weight) for asset, weight in target.items()
            }
            segment_start = wanted
        _, current = _hand_segment_growth(
            current, world["factors"][segment_start]
        )
        assert dict(row.closing_weights) == _hand_book(current)


# ---------------------------------------------------------------------------
# I1 — wealth closure on the unit-NAV basis.
# ---------------------------------------------------------------------------


def test_wealth_closes_across_periods() -> None:
    # W_{t+1} = W_t × G_net,t exactly, row by row, in every world; the
    # first row opens at the unit-NAV W₀ = 1 and final_wealth is the
    # exact product of the hand-derived net growth factors.
    for world, table in zip(_ALL_WORLDS, _ALL_ROW_TABLES):
        path = _build(world)
        assert path.rows[0].wealth_open == 1.0
        previous_close = Fraction(1)
        net_factors = []
        for row, expected in zip(path.rows, table):
            opens, closes = expected[4], expected[5]
            assert row.wealth_open == _hand_dyadic_float(opens)
            assert row.wealth_close == _hand_dyadic_float(closes)
            # The recursion itself, in exact float arithmetic.
            growth = Fraction(1) + Fraction(expected[1])
            assert row.wealth_close == row.wealth_open * float(growth)
            assert row.wealth_open == _hand_dyadic_float(previous_close)
            net_factors.append(growth)
            previous_close = closes
        assert path.final_wealth == path.rows[-1].wealth_close
        assert path.final_wealth == _hand_dyadic_float(
            _hand_product(net_factors)
        )
        assert path.n_periods == len(path.rows)


def test_every_recorded_book_sums_to_unit_budget() -> None:
    # Every book anywhere on the wealth path — openings, closings, and
    # each execution's target, pre-trade, and post-trade books —
    # carries budget exactly 1 (the weight identity behind the
    # unit-NAV recursion), checked in exact rational arithmetic.
    for world in _ALL_WORLDS:
        path = _build(world)
        for row in path.rows:
            books = [row.opening_weights, row.closing_weights]
            books.extend(e.target_weights for e in row.executions)
            books.extend(e.pre_trade_weights for e in row.executions)
            books.extend(e.post_trade_weights for e in row.executions)
            for book in books:
                total = Fraction(0)
                for weight in dict(book).values():
                    total += Fraction(weight)
                assert total == 1, f"non-unit budget in {dict(book)!r}"


def test_initial_wealth_scales_the_path_exactly() -> None:
    # A researcher-scale W₀ multiplies every wealth field by exactly
    # W₀ and changes nothing else — the multiplicative recursion is
    # scale-free (2^10 keeps every product dyadic and exact).
    scale = 1024.0
    default = _build(_WORLD_MULTI)
    scaled = _build(_WORLD_MULTI, initial_wealth=scale)
    assert scaled.final_wealth == scale * default.final_wealth
    for scaled_row, default_row, expected in zip(
        scaled.rows, default.rows, _MULTI_ROWS
    ):
        assert scaled_row.wealth_open == scale * default_row.wealth_open
        assert scaled_row.wealth_close == scale * default_row.wealth_close
        assert scaled_row.wealth_open == _hand_dyadic_float(
            Fraction(1024) * expected[4]
        )
        assert scaled_row.gross_return == default_row.gross_return
        assert scaled_row.net_return == default_row.net_return
        assert dict(scaled_row.opening_weights) == dict(
            default_row.opening_weights
        )


# ---------------------------------------------------------------------------
# I2 — segment-composed gross growth.
# ---------------------------------------------------------------------------


def test_gross_return_composes_across_segments() -> None:
    # Row gross growth is the product of the hand-derived segment
    # denominators D_m, exactly — including the three-segment,
    # two-execution row of World M and the two-segment delayed row of
    # World D.
    multi = _build(_WORLD_MULTI).rows
    assert multi[0].gross_return == _hand_dyadic_float(
        _hand_product([Fraction(2), Fraction(1), Fraction(3, 2)])
        - Fraction(1)
    )
    assert multi[0].gross_return == 2.0
    assert multi[1].gross_return == _hand_dyadic_float(
        _hand_product([Fraction(1), Fraction(2)]) - Fraction(1)
    )
    delayed = _build(_WORLD_DELAYED).rows
    assert delayed[1].gross_return == _hand_dyadic_float(
        _hand_product([Fraction(3, 4), Fraction(3, 4)]) - Fraction(1)
    )
    assert delayed[1].gross_return == -0.4375
    enter_exit = _build(_WORLD_ENTER_EXIT).rows
    assert enter_exit[0].gross_return == _hand_dyadic_float(
        _hand_product([Fraction(1), Fraction(2), Fraction(1)]) - Fraction(1)
    )
    # The full hand tables close for every row of every world.
    for world, table in zip(_ALL_WORLDS, _ALL_ROW_TABLES):
        for row, expected in zip(_build(world).rows, table):
            assert row.gross_return == _hand_dyadic_float(expected[0])


def test_gross_return_is_not_the_single_denominator_form() -> None:
    # The rejected alternative — compounding per-asset factors over
    # the WHOLE period on the opening book (one denominator for the
    # period) — gives a different, wrong number in both multi-segment
    # rows; the recorded value must equal the segment product and
    # differ from the single-denominator value.  A last-segment-only
    # collapse (another single-denominator degeneration) differs too.
    multi_row0 = _build(_WORLD_MULTI).rows[0]
    assert multi_row0.gross_return != _hand_dyadic_float(
        Fraction(165, 32) - Fraction(1)
    )
    assert multi_row0.gross_return != _hand_dyadic_float(
        Fraction(3, 2) - Fraction(1)
    )
    delayed_row1 = _build(_WORLD_DELAYED).rows[1]
    assert delayed_row1.gross_return != _hand_dyadic_float(
        Fraction(3, 4) - Fraction(1)
    )


def test_drift_only_period_gross_equals_its_segment_denominator() -> None:
    # With no execution the segment law collapses to one segment: the
    # row's gross growth IS that segment's denominator, exactly (I2b).
    multi_row2 = _build(_WORLD_MULTI).rows[2]
    assert multi_row2.executions == ()
    assert multi_row2.gross_return == _hand_dyadic_float(
        Fraction(2) - Fraction(1)
    )
    enter_exit_row1 = _build(_WORLD_ENTER_EXIT).rows[1]
    assert enter_exit_row1.executions == ()
    assert enter_exit_row1.gross_return == _hand_dyadic_float(
        Fraction(3, 4) - Fraction(1)
    )


# ---------------------------------------------------------------------------
# I3 — multiplicative costs.
# ---------------------------------------------------------------------------


def test_costs_compose_multiplicatively_within_a_period() -> None:
    # F_cost,t = Π_k (1 − q_k) with the hand q values; the row's
    # transaction_cost is 1 − F_cost and G_net = G_gross × F_cost —
    # exactly, in every row of every world, never the additive form.
    multi_row0 = _build(_WORLD_MULTI).rows[0]
    q_values = [Fraction(1, 16), Fraction(3, 32)]
    assert multi_row0.transaction_cost == _hand_dyadic_float(
        Fraction(1) - _hand_cost_factor(q_values)
    )
    assert multi_row0.transaction_cost != _hand_dyadic_float(
        Fraction(1, 16) + Fraction(3, 32)
    )
    for world, table in zip(_ALL_WORLDS, _ALL_ROW_TABLES):
        for row, expected in zip(_build(world).rows, table):
            assert row.transaction_cost == _hand_dyadic_float(expected[2])
            # The multiplicative identity, exactly:
            # (1 + r_net) == (1 + r_gross) · (1 − cost).
            gross = Fraction(1) + Fraction(expected[0])
            net = Fraction(1) + Fraction(expected[1])
            cost_factor = Fraction(1) - Fraction(expected[2])
            assert net == gross * cost_factor
            assert (1.0 + row.net_return) == (1.0 + row.gross_return) * (
                1.0 - row.transaction_cost
            )
            # The additive law r_net = r_gross − q omits the
            # cross-term r_gross·q, so it is a different number
            # wherever BOTH growth and cost exist (they coincide in
            # the degenerate r_gross = 0 row of the delayed world,
            # where there is no cross-term to omit).
            if expected[2] > 0 and expected[0] != 0:
                assert row.net_return != (
                    row.gross_return - row.transaction_cost
                )


def test_execution_free_period_charges_no_costs_exactly() -> None:
    # I3b: an execution-free row pays EXACTLY nothing — cost and
    # turnover are the exact floats 0.0 and net_return IS gross_return
    # (bit-identical, no tolerance).
    for world, row_index in ((_WORLD_MULTI, 2), (_WORLD_ENTER_EXIT, 1)):
        row = _build(world).rows[row_index]
        assert row.executions == ()
        assert row.turnover == 0.0
        assert row.transaction_cost == 0.0
        assert row.net_return == row.gross_return


# ---------------------------------------------------------------------------
# I4 — POST_TRADE = TARGET, unconditional; engines are interchangeable.
# ---------------------------------------------------------------------------


def test_post_trade_weights_equal_target_weights() -> None:
    # Every execution on every row of every world records the target
    # as its post-trade book, exactly, and the target matches the
    # hand-derived book for that execution instant.
    for world in _ALL_WORLDS:
        path = _build(world)
        hand_targets = {
            executes: book for _, executes, book in world["executions"]
        }
        for row in path.rows:
            for execution in row.executions:
                assert dict(execution.post_trade_weights) == dict(
                    execution.target_weights
                )
                assert dict(execution.post_trade_weights) == _hand_book(
                    hand_targets[execution.execution_time]
                )


def test_non_target_accounting_engine_fails_closed() -> None:
    # An engine whose post-trade book is not the target rejects with
    # ValueError at composition time — execution realism is out of
    # scope and fails closed.
    with pytest.raises(ValueError, match="target"):
        build_ledger(
            **_ledger_inputs(_WORLD_MULTI), accounting_engine=_SkewEngine()
        )


def test_engine_choice_does_not_change_the_ledger_value() -> None:
    # I4b: the default engine, an explicit ExactFillAccounting, and a
    # compliant recording engine produce economically value-equal
    # ledgers, row for row and field for field; the recording itself
    # is engine state, NOT part of the ledger value.
    default = _build(_WORLD_MULTI)
    explicit = build_ledger(
        **_ledger_inputs(_WORLD_MULTI),
        accounting_engine=ExactFillAccounting(),
    )
    first_spy = _RecordingEngine()
    spied = build_ledger(
        **_ledger_inputs(_WORLD_MULTI), accounting_engine=first_spy
    )
    assert default == explicit
    assert default == spied
    _assert_paths_value_equal(default, explicit)
    _assert_paths_value_equal(default, spied)
    # The spy really was composed once per execution...
    assert len(first_spy.calls) == len(_WORLD_MULTI["executions"])
    # ...two fresh spies record independently (distinct call lists)...
    second_spy = _RecordingEngine()
    build_ledger(**_ledger_inputs(_WORLD_MULTI), accounting_engine=second_spy)
    assert first_spy.calls is not second_spy.calls
    # ...and the ledger value is untouched by that side channel.
    assert build_ledger(
        **_ledger_inputs(_WORLD_MULTI), accounting_engine=second_spy
    ) == default


# ---------------------------------------------------------------------------
# I5 — the per-holding segment drift identity, chained.
# ---------------------------------------------------------------------------


def test_segment_drift_identity_chains_across_executions() -> None:
    # For every world: segment end = start × g_i / D_m per holding
    # (hand Fraction arithmetic), pre_trade is the segment endpoint,
    # post_trade is the target, the target seeds the next segment, the
    # final endpoint is the closing book, and each row opens on the
    # previous row's closing book.
    for world in _ALL_WORLDS:
        _assert_drift_chain(_build(world), world)


def test_execution_free_period_drifts_opening_into_closing() -> None:
    # The execution-free reduction: opening → one drift → closing.
    row = _build(_WORLD_MULTI).rows[2]
    opening = _hand_book_at_row_open(_WORLD_MULTI, 2)
    assert dict(row.opening_weights) == _hand_book(opening)
    _, drifted = _hand_segment_growth(
        opening, _WORLD_MULTI["factors"][row.period_start]
    )
    assert dict(row.closing_weights) == _hand_book(drifted)

    row_e = _build(_WORLD_ENTER_EXIT).rows[1]
    opening_e = _hand_book_at_row_open(_WORLD_ENTER_EXIT, 1)
    assert dict(row_e.opening_weights) == _hand_book(opening_e)
    _, drifted_e = _hand_segment_growth(
        opening_e, _WORLD_ENTER_EXIT["factors"][row_e.period_start]
    )
    assert dict(row_e.closing_weights) == _hand_book(drifted_e)


# ---------------------------------------------------------------------------
# I5b — turnover aggregation and conventions.
# ---------------------------------------------------------------------------


def test_execution_turnover_sums_to_row_turnover() -> None:
    # Row turnover is the plain sum of its executions' turnovers under
    # the model's convention, exactly; each execution's turnover is
    # the hand one_way measure of (hand pre-trade book → target).
    multi = _build(_WORLD_MULTI)
    for row, expected in zip(multi.rows, _MULTI_ROWS):
        assert row.turnover == _hand_dyadic_float(expected[3])
        assert row.turnover == sum(e.turnover for e in row.executions)
    # Hand per-execution measures for the two-execution row.
    row0 = multi.rows[0]
    pre_first = {"A": Fraction(3, 4), "B": Fraction(1, 8), "C": Fraction(1, 8)}
    pre_second = {
        "A": Fraction(1, 2),
        "B": Fraction(3, 8),
        "C": Fraction(1, 8),
    }
    assert row0.executions[0].turnover == _hand_dyadic_float(
        _hand_one_way(pre_first, _WORLD_MULTI["executions"][0][2])
    )
    assert row0.executions[1].turnover == _hand_dyadic_float(
        _hand_one_way(pre_second, _WORLD_MULTI["executions"][1][2])
    )


def test_turnover_conventions_are_pinned_one_way_and_two_sided() -> None:
    # The SAME world under both fixed conventions: one_way = 0.5·Σ|Δ|
    # and two_sided = Σ|Δ| — exactly doubles on these books — with
    # the cost fraction following the convention-bound measure and
    # the row turnover remaining the sum of execution turnovers.
    one_way_path = _build(_WORLD_ENTER_EXIT, convention=one_way)
    two_sided_path = _build(_WORLD_ENTER_EXIT, convention=two_sided)
    enter_target = _WORLD_ENTER_EXIT["executions"][0][2]
    exit_target = _WORLD_ENTER_EXIT["executions"][1][2]
    pre_enter = {"A": Fraction(3, 4), "B": Fraction(1, 4)}
    pre_exit = {"A": Fraction(1, 4), "B": Fraction(1, 4), "C": Fraction(1, 2)}
    for row in (one_way_path.rows[0], two_sided_path.rows[0]):
        assert row.turnover == sum(e.turnover for e in row.executions)
    assert one_way_path.rows[0].executions[0].turnover == (
        _hand_dyadic_float(_hand_one_way(pre_enter, enter_target))
    )
    assert one_way_path.rows[0].executions[1].turnover == (
        _hand_dyadic_float(_hand_one_way(pre_exit, exit_target))
    )
    assert two_sided_path.rows[0].executions[0].turnover == (
        _hand_dyadic_float(_hand_two_sided(pre_enter, enter_target))
    )
    assert two_sided_path.rows[0].executions[1].turnover == (
        _hand_dyadic_float(_hand_two_sided(pre_exit, exit_target))
    )
    for left, right in zip(
        one_way_path.rows[0].executions, two_sided_path.rows[0].executions
    ):
        assert right.turnover == 2.0 * left.turnover
    # Convention-bound costs: q = 1/4 twice under two_sided, so
    # F_cost = 9/16, cost = 7/16, G_net = 2 · 9/16 = 9/8 exactly.
    assert two_sided_path.rows[0].transaction_cost == _hand_dyadic_float(
        Fraction(1) - _hand_cost_factor([Fraction(1, 4), Fraction(1, 4)])
    )
    assert two_sided_path.rows[0].net_return == _hand_dyadic_float(
        Fraction(9, 8) - Fraction(1)
    )
    assert two_sided_path.rows[0].wealth_close == _hand_dyadic_float(
        Fraction(9, 8)
    )
    assert one_way_path.rows[0].turnover == 1.0
    assert two_sided_path.rows[0].turnover == 2.0


# ---------------------------------------------------------------------------
# I6 — execution ownership by execution instant, [start, end).
# ---------------------------------------------------------------------------


def test_delayed_execution_is_charged_to_its_execution_period() -> None:
    # d1 is decided at T0 but executes at FEB20, inside row 1: it is
    # recorded on row 1 (with its decision time intact), row 0 keeps
    # only its own JAN20 execution, and row 1's pre-trade book is the
    # hand drift of row 1's opening over [T1, FEB20).
    path = _build(_WORLD_DELAYED)
    row0, row1, _ = path.rows
    assert [e.execution_time for e in row0.executions] == [JAN20]
    assert [e.execution_time for e in row1.executions] == [FEB20]
    delayed = row1.executions[0]
    assert delayed.decision_time == T0
    assert delayed.execution_time == FEB20
    _, drifted = _hand_segment_growth(
        {"A": Fraction(1, 4), "B": Fraction(3, 4)},
        _WORLD_DELAYED["factors"][T1],
    )
    assert dict(delayed.pre_trade_weights) == _hand_book(drifted)
    # Row 0's own wealth bears only its own execution's cost.
    assert row0.transaction_cost == _hand_dyadic_float(Fraction(1, 8))


def test_boundary_execution_opens_the_next_period() -> None:
    # d2 executes exactly at T2 — row 1's period_end and row 2's
    # period_open. It belongs to ROW 2: row 1's closing book stops at
    # the drifted T2⁻ state, and row 2's execution runs from that
    # book with no drift before it.
    path = _build(_WORLD_DELAYED)
    row1, row2 = path.rows[1], path.rows[2]
    assert [e.execution_time for e in row1.executions] == [FEB20]
    assert [e.execution_time for e in row2.executions] == [T2]
    boundary = row2.executions[0]
    assert boundary.decision_time == FEB25
    assert dict(boundary.pre_trade_weights) == dict(row2.opening_weights)
    assert dict(row2.opening_weights) == dict(row1.closing_weights)
    assert dict(row1.closing_weights) == {"A": 0.5, "B": 0.5}


# ---------------------------------------------------------------------------
# I7 — the all-retained-books row universe.
# ---------------------------------------------------------------------------


def test_universe_unions_every_retained_book_of_the_row() -> None:
    # The row universe is the union over ALL retained books — opening,
    # every execution's target/pre-trade/post-trade, and closing — not
    # merely opening ∪ closing.
    multi = _build(_WORLD_MULTI)
    for row in multi.rows:
        assert set(row.universe) == {"A", "B", "C"}
    delayed = _build(_WORLD_DELAYED)
    for row in delayed.rows:
        assert set(row.universe) == {"A", "B"}
    enter_exit = _build(_WORLD_ENTER_EXIT)
    assert set(enter_exit.rows[0].universe) == {"A", "B", "C"}
    assert set(enter_exit.rows[1].universe) == {"A", "B"}


def test_entered_and_exited_asset_is_retained_at_effective_zero() -> None:
    # C enters at JAN15 and exits at JAN25, both inside period 0: C
    # is in the row universe and in the mid-period execution books,
    # but in NEITHER the opening nor the closing book — absent from a
    # book means effective weight exactly zero there — and once
    # exited it is gone from later rows entirely.
    path = _build(_WORLD_ENTER_EXIT)
    row0 = path.rows[0]
    assert "C" in row0.universe
    assert "C" not in dict(row0.opening_weights)
    assert "C" not in dict(row0.closing_weights)
    assert row0.opening_weights.get("C", 0.0) == 0.0
    assert row0.closing_weights.get("C", 0.0) == 0.0
    enter, exit_c = row0.executions
    assert "C" in dict(enter.target_weights)
    assert "C" in dict(enter.post_trade_weights)
    assert "C" in dict(exit_c.pre_trade_weights)
    assert "C" not in dict(exit_c.target_weights)
    assert "C" not in set(path.rows[1].universe)


# ---------------------------------------------------------------------------
# I8 — purity and replay.
# ---------------------------------------------------------------------------


def test_identical_inputs_replay_to_value_equal_ledgers() -> None:
    # Two independent assemblies from identically-valued fresh inputs
    # produce value-equal paths: == holds, every row and scalar field
    # matches exactly, and no record object is shared between the
    # assemblies.  (No pickling is required or asserted — the ledger
    # record is a value object, not a serialization artifact.)
    for world in _ALL_WORLDS:
        first = _build(world)
        second = _build(world)
        assert first is not second
        assert first == second
        assert first.rows == second.rows
        _assert_paths_value_equal(first, second)
        assert first.rows[0] is not second.rows[0]
        assert first.rows[0].opening_weights is not (
            second.rows[0].opening_weights
        )
        assert first.final_wealth == second.final_wealth


# ---------------------------------------------------------------------------
# I9 — behavioral immutability of the record.
# ---------------------------------------------------------------------------


def test_ledger_records_are_immutable_value_objects() -> None:
    # Attribute writes and deletes fail on the path, every row, every
    # execution, and the retained trade; the row/executions tuples
    # reject item assignment, so nothing on the record is replaceable
    # after construction.
    path = _build(_WORLD_MULTI)
    row = path.rows[0]
    execution = row.executions[0]
    for target, attribute, value in (
        (path, "final_wealth", 0.0),
        (row, "gross_return", 0.0),
        (row, "universe", frozenset()),
        (execution, "turnover", 0.0),
        (execution.trade, "source", None),
    ):
        with pytest.raises(AttributeError):
            setattr(target, attribute, value)
    with pytest.raises(AttributeError):
        del row.universe
    with pytest.raises(AttributeError):
        del execution.turnover
    with pytest.raises(TypeError):
        path.rows[0] = path.rows[1]
    with pytest.raises(TypeError):
        row.executions[0] = row.executions[1]


def test_books_are_read_only_through_the_public_surface() -> None:
    # Every weight book on the record is a read-only mapping: item
    # assignment and deletion both fail, for openings, closings, and
    # every execution book.
    path = _build(_WORLD_MULTI)
    books = [path.rows[0].opening_weights, path.rows[0].closing_weights]
    for execution in path.rows[0].executions:
        books.extend(
            (
                execution.target_weights,
                execution.pre_trade_weights,
                execution.post_trade_weights,
            )
        )
    for book in books:
        assert isinstance(book, Mapping)
        with pytest.raises(TypeError):
            book["A"] = 0.25
        with pytest.raises(TypeError):
            del book["A"]


def test_input_mapping_mutation_cannot_reach_the_record() -> None:
    # The snapshot discipline, behaviorally: mutating the caller's
    # input mappings AFTER construction leaves the fixed record
    # untouched.
    inputs = _ledger_inputs(_WORLD_MULTI)
    path = build_ledger(**inputs)
    inputs["initial_weights"]["A"] = 0.9
    inputs["growth_factors"][T0]["A"] = 99.0
    assert dict(path.rows[0].opening_weights) == _hand_book(
        _WORLD_MULTI["initial"]
    )
    assert path.rows[0].gross_return == _hand_dyadic_float(_MULTI_ROWS[0][0])


# ---------------------------------------------------------------------------
# I10 — unconditional admission minimums with the fixed named exceptions.
# ---------------------------------------------------------------------------


def test_chronology_violations_reject_with_named_exception() -> None:
    # Overlapping / non-increasing accounting instants, non-monotone
    # executions, and executions outside the horizon all reject with
    # the fixed InvalidChronologyError — the named class, not a bare
    # ValueError.
    with pytest.raises(InvalidChronologyError):
        build_ledger(
            **_ledger_inputs(_WORLD_MULTI)
            | {"accounting_instants": (T0, T0, T1)}
        )
    with pytest.raises(InvalidChronologyError):
        build_ledger(
            **_ledger_inputs(_WORLD_MULTI)
            | {"accounting_instants": (T1, T0, T2)}
        )
    reversed_executions = _ledger_inputs(_WORLD_MULTI)
    reversed_executions["accounting_instants"] = (T0, T1)
    reversed_executions["decisions"] = [
        PortfolioDecision(
            decision_time=JAN24,
            execution_time=JAN24,
            target_weights=_hand_book(_WORLD_MULTI["executions"][1][2]),
        ),
        PortfolioDecision(
            decision_time=JAN8,
            execution_time=JAN8,
            target_weights=_hand_book(_WORLD_MULTI["executions"][0][2]),
        ),
    ]
    with pytest.raises(InvalidChronologyError):
        build_ledger(**reversed_executions)
    # An execution past the horizon end has no owning period.
    with pytest.raises(InvalidChronologyError):
        build_ledger(
            **_ledger_inputs(_WORLD_ENTER_EXIT)
            | {"accounting_instants": (T0, JAN20)}
        )


def test_naive_timestamps_reject_with_named_exception() -> None:
    # A naive accounting instant, a naive growth-factor key, and a
    # naive decision instant each reject with the fixed
    # NaiveTimestampError.
    with pytest.raises(NaiveTimestampError):
        build_ledger(
            **_ledger_inputs(_WORLD_MULTI)
            | {
                "accounting_instants": (
                    datetime(2026, 1, 1),  # noqa: DTZ001 — deliberately naive instants (NaiveTimestampError probe)
                    datetime(2026, 3, 1),  # noqa: DTZ001 — deliberately naive instants (NaiveTimestampError probe)
                )
            }
        )
    with pytest.raises(NaiveTimestampError):
        build_ledger(
            **_ledger_inputs(_WORLD_MULTI)
            | {
                "growth_factors": dict(_WORLD_MULTI["factors"])
                | {
                    datetime(2026, 1, 8): {"A": 1.0, "B": 1.0, "C": 1.0}  # noqa: DTZ001 — deliberately naive instant (NaiveTimestampError probe)
                }
            }
        )
    with pytest.raises(NaiveTimestampError):
        inputs = _ledger_inputs(_WORLD_MULTI)
        inputs["decisions"] = [
            PortfolioDecision(
                decision_time=datetime(2026, 1, 8),  # noqa: DTZ001 — deliberately naive instants (NaiveTimestampError probe)
                execution_time=datetime(2026, 1, 8),  # noqa: DTZ001
                target_weights=_hand_book(_WORLD_MULTI["executions"][0][2]),
            )
        ]
        build_ledger(**inputs)


def test_non_unit_budget_books_reject_on_the_wealth_path() -> None:
    # Every book on the wealth path must carry budget exactly 1: a
    # half-budget initial book and a three-quarters-budget target both
    # reject with ValueError naming the budget.
    with pytest.raises(ValueError, match="budget"):
        build_ledger(
            **_ledger_inputs(_WORLD_MULTI) | {"initial_weights": {"A": 0.5}}
        )
    short_budget_world = dict(_WORLD_MULTI)
    short_budget_world["executions"] = (
        (
            JAN8,
            JAN8,
            {
                "A": Fraction(1, 4),
                "B": Fraction(1, 4),
                "C": Fraction(1, 4),
            },
        ),
    )
    with pytest.raises(ValueError, match="budget"):
        build_ledger(**_ledger_inputs(short_budget_world))


def test_non_finite_growth_factors_reject() -> None:
    # NaN, infinity, and negative gross factors all reject with
    # ValueError through the fixed factor law.
    factors = dict(_WORLD_MULTI["factors"])
    healthy = {
        start: _hand_book(book) for start, book in factors.items()
    }
    for bad in (float("nan"), float("inf"), -0.5):
        with pytest.raises(ValueError):
            build_ledger(
                **_ledger_inputs(_WORLD_MULTI)
                | {
                    "growth_factors": healthy
                    | {T0: {"A": bad, "B": 1.0, "C": 1.0}}
                }
            )


# ---------------------------------------------------------------------------
# I11 — the anti-reference meta-floor.
# ---------------------------------------------------------------------------


def test_expected_values_are_hand_derived_not_oracle_recomputed() -> None:
    # Walk THIS file's own AST: every in-module ``_hand_*`` helper is
    # the only place expected values come from, and none of them may
    # reference the library under test — no portlearn import, no
    # imported public name, no string aliasing — only literals,
    # Fraction, and stdlib arithmetic.
    source = pathlib.Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    hand_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_hand_")
    ]
    assert len(hand_functions) >= 8, "the hand-derived floor lost its helpers"
    banned = {
        "portlearn",
        "build_ledger",
        "ExactFillAccounting",
        "Proportional",
        "PortfolioDecision",
        "AccountingResult",
        "InvalidChronologyError",
        "NaiveTimestampError",
        "one_way",
        "two_sided",
    }
    for function in hand_functions:
        uses_exact_arithmetic = False
        for node in ast.walk(function):
            assert not isinstance(node, (ast.Import, ast.ImportFrom)), (
                f"{function.name} must not import anything: expected "
                "values come from literals, Fraction, and stdlib "
                "arithmetic only"
            )
            if isinstance(node, ast.Name):
                assert node.id not in banned, (
                    f"{function.name} references {node.id!r}: the "
                    "library under test must not compute its own "
                    "expected values"
                )
                if node.id == "Fraction" or node.id.startswith("_hand_"):
                    uses_exact_arithmetic = True
            if isinstance(node, ast.Attribute):
                assert node.attr not in banned, (
                    f"{function.name} references attribute "
                    f"{node.attr!r} of the library under test"
                )
            if isinstance(node, ast.Constant) and isinstance(
                node.value, str
            ):
                assert "portlearn" not in node.value, (
                    f"{function.name} mentions the package under test "
                    "in a string"
                )
        assert uses_exact_arithmetic, (
            f"{function.name} must do its arithmetic in exact rationals "
            "or delegate to another hand helper"
        )


def test_battery_exercises_public_surface_only() -> None:
    # The battery never reaches around the public API: no private
    # type of the ledger implementation and no dunder-level object
    # surgery appears anywhere in this file.  (The banned tokens are
    # assembled from fragments so this scan cannot trip on itself.)
    source = pathlib.Path(__file__).read_text(encoding="utf-8")
    banned_tokens = (
        "Mapping" + "ProxyType",
        "_Frozen" + "Book",
        "portlearn" + "._",
        "__dict" + "__",
        "object.__set" + "attr__",
    )
    for token in banned_tokens:
        assert token not in source, (
            f"the battery must stay on the public surface; found "
            f"{token!r}"
        )
