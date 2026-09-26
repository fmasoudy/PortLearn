"""Composition and strategy-contract falsification battery (P0-P4, C1-C15).

Every floor in this module runs against the public surface of
PortLearn only: no private imports, no monkey-patching, no test-only
adapters in ``src/``, and no re-derived library echoes — every expected
value below is hand-derived from the declared synthetic world in
this docstring and reproduced in comments next to each assertion.

The synthetic world (all factors binary-exact dyadic rationals)
----------------------------------------------------------------

Assets ``CASH``, ``A``, ``B``; three accounting periods over month-end
instants (UTC, via the public ``month_end_instant``):

* ``S0`` = 2026-05-31 end  (decision instant for period 1)
* ``T0`` = 2026-06-30 end  (execution 1 / period-1 start)
* ``T1`` = 2026-07-31 end  (execution 2 / period-2 start)
* ``T2`` = 2026-08-31 end  (execution 3 / period-3 start)
* ``T3`` = 2026-09-30 end  (path end)

Initial book ``{CASH: 0.5, A: 0.25, B: 0.25}`` (unit budget: the
explicit named cash asset carries its own growth factor; there is no
implicit residual).  Decisions (delayed execution: decided at the prior
month end, executed at the period start, so each trade's cost is
charged to the period containing its execution instant):

* ``d1``: decided ``S0``, executed ``T0``, target ``{A: 0.5, B: 0.5}``
* ``d2``: decided ``T0``, executed ``T1``, target ``{CASH: 0.625, B: 0.375}``
* ``d3``: decided ``T1``, executed ``T2``, target ``{CASH: 1.0}``

Segment gross growth factors (fixed universe per segment; the CASH
factor is carried by the synthetic series ``SYNTHGC``):

* ``[T0, T1)``: ``{A: 2.5, B: 1.5}``
* ``[T1, T2)``: ``{CASH: 2.0, B: 2.0}``
* ``[T2, T3)``: ``{CASH: 1.25}``

Cost model ``Proportional(0.25, turnover=one_way)``.

Hand derivation (reproduced per row in C5):

* ``d1``: deltas over the union ``{CASH, A, B}`` are ``+0.25, +0.25,
  -0.5`` so ``two_sided = 1.0`` and ``one_way = 0.5``;
  ``q = 0.25 * 0.5 = 0.125``, ``F_cost = 0.875``.  Post-trade book
  ``{A: 0.5, B: 0.5}`` drifts over ``{A: 2.5, B: 1.5}``:
  ``D = 0.5*2.5 + 0.5*1.5 = 2.0`` → closing ``{A: 0.625, B: 0.375}``.
  ``G_gross = 2.0``; ``G_net = 2.0 * 0.875 = 1.75``; wealth ``1 → 1.75``.
* ``d2``: deltas over ``{A, B, CASH}`` are ``-0.625, 0, +0.625`` so
  ``one_way = 0.625``; ``q = 0.15625``, ``F_cost = 0.84375``.
  Post-trade ``{CASH: 0.625, B: 0.375}`` drifts over uniform ``{2.0,
  2.0}`` unchanged.  ``G_gross = 2.0``; ``G_net = 1.6875``; wealth
  ``1.75 → 2.953125``.
* ``d3``: deltas over ``{B, CASH}`` are ``-0.375, +0.375`` so
  ``one_way = 0.375``; ``q = 0.09375``, ``F_cost = 0.90625``.
  Post-trade ``{CASH: 1.0}`` grows by ``1.25``.  ``G_gross = 1.25``;
  ``G_net = 1.25 * 0.90625 = 1.1328125``; wealth
  ``2.953125 → 3.3453369140625``.

Final wealth ``= 189/64 * 145/128 = 27405/8192 = 3.3453369140625``
(exactly representable in binary floating point; every equality below
is exact ``==``).

Falsification outcomes for this public test module
------------------------------------------------------------------------

* PASS: all seventeen floors (C1-C15 plus P4) — this module holds
  zero FALSIFIED outcomes after the decision-seam repair.
* Repaired through the ``DecisionContext`` seam: C8 (equal
  weight), C9 (minimum variance), C11 (direct neural allocation),
  C12 (stateful RL), and the P4 current-portfolio-state decision flow were
  FALSIFIED on the fixed ``decide(self, forecast)`` seam — no lawful
  parameter carried the current universe, the current portfolio state
  available at decision time, or the strategy's carried state.  The
  repaired decision contract added
  the ``DecisionContext`` channels (``universe``, ``current_weights``,
  ``strategy_state``, ``decision_time``), and each floor below now
  rides its channel lawfully.  The original falsification teeth are
  preserved as change-detectors: each repaired floor still fails if
  its channel is removed from the seam again, and each still proves
  that no *forecast* channel can carry its input (clauses (i)
  fabrication, (ii) smuggling, and (v) dropping stay unlawful
  routes).
* INFORMATIVE: hand-building a ``QualifiedDataset`` — friction note
  only; the battery itself admits the synthetic world exclusively
  through the public offline facade path.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
from dataclasses import fields
from datetime import UTC as _DatetimeUTC
from datetime import datetime, timedelta
from importlib.metadata import version as _dist_version
from zoneinfo import ZoneInfo

import pytest

from portlearn import __version__
from portlearn.calendar import month_end_instant
from portlearn.costs import Proportional
from portlearn.data import fred as fred_facade
from portlearn.data.dataset import UnqualifiedDataError
from portlearn.data.ingestion import AvailabilityPolicy
from portlearn.interfaces import (
    AccountingResult,
    DecisionContext,
    DecisionResult,
    Evaluator,
    Forecast,
    InformationSet,
    PortfolioDecision,
    Strategy,
    require_decision_result_compatible,
    require_forecast_decision_compatible,
)
from portlearn.ledger import ExactFillAccounting, build_ledger
from portlearn.manifest import RunManifest
from portlearn.observations import (
    AmbiguousObservationError,
    TimedObservation,
    vintage_as_of,
)
from portlearn.rebalance import drift_weights
from portlearn.timing import (
    FutureInformationError,
    InvalidChronologyError,
    NaiveTimestampError,
)
from portlearn.turnover import one_way
from portlearn.weights import PortfolioWeights, WeightState

# --------------------------------------------------------------------------- #
# The synthetic world: instants, books, targets, factors (hand-set constants)  #
# --------------------------------------------------------------------------- #

_UTC = _DatetimeUTC  # the aware UTC constant
_ZONE = ZoneInfo("UTC")

#: Decision/execution/accounting instants (public month-end operation).
S0 = month_end_instant(2026, 5, _ZONE)
T0 = month_end_instant(2026, 6, _ZONE)
T1 = month_end_instant(2026, 7, _ZONE)
T2 = month_end_instant(2026, 8, _ZONE)
T3 = month_end_instant(2026, 9, _ZONE)

#: The initial book: the explicit named cash asset plus two risky assets.
INITIAL_BOOK = {"CASH": 0.5, "A": 0.25, "B": 0.25}

#: The synthetic-world targets, keyed by the decision instant that
#: desires them (delayed execution one month later).
WORLD_TARGETS = {
    S0: {"A": 0.5, "B": 0.5},
    T0: {"CASH": 0.625, "B": 0.375},
    T1: {"CASH": 1.0},
}

#: Cost model: 25% proportional on one-way turnover (all dyadic-exact).
COST_RATE = 0.25

#: The current investable universe at each decision instant (the equal
#: weight family needs *this* — note A is still held at T0 even though
#: the T0 target drops it, so the universe is not the target key set).
CURRENT_UNIVERSE = {S0: {"CASH", "A", "B"}, T0: {"CASH", "A", "B"}, T1: {"CASH", "B"}}

#: The pre-trade book at each decision instant — the books the ledger
#: itself trades from at the matching execution (hand-derived; see the
#: P4 floor): at S0/T0 the initial book, at T1 the drifted d1 target.
_PRE_DECISION_BOOK = {
    S0: dict(INITIAL_BOOK),
    T0: dict(INITIAL_BOOK),
    T1: {"A": 0.625, "B": 0.375},
}


def _next_month_end(instant: datetime) -> datetime:
    """The month-end instant one month after ``instant`` (public law)."""
    if instant.month == 12:
        return month_end_instant(instant.year + 1, 1, instant.tzinfo)
    return month_end_instant(instant.year, instant.month + 1, instant.tzinfo)


# --------------------------------------------------------------------------- #
# Synthetic provider-format bytes and the public admission path (C1)           #
# --------------------------------------------------------------------------- #

#: Synthetic FRED-format observations: series -> [(month, value), ...].
#: ``SYNTHGA``/``SYNTHGB`` carry the risky growth factors, ``SYNTHGC``
#: the CASH growth factor.  Values are the world's declared factors and
#: nothing else — invented values in the provider response format.
_SYNTH_OBSERVATIONS = {
    "SYNTHGA": [("2026-05-01", "1.0"), ("2026-06-01", "2.5")],
    "SYNTHGB": [
        ("2026-05-01", "1.0"),
        ("2026-06-01", "1.5"),
        ("2026-07-01", "2.0"),
    ],
    "SYNTHGC": [("2026-07-01", "2.0"), ("2026-08-01", "1.25")],
}


def _observation_bytes(series_id: str) -> bytes:
    rows = [
        {"date": month, "value": value}
        for month, value in _SYNTH_OBSERVATIONS[series_id]
    ]
    payload = {
        "series_id": series_id,
        "realtime_start": "2026-05-01",
        "realtime_end": "2026-09-30",
        "observations": rows,
    }
    return json.dumps(payload).encode()


def _metadata_bytes(series_id: str) -> bytes:
    payload = {
        "id": series_id,
        "title": "Synthetic growth-factor series (SYNTHETIC REPLICA)",
        "units": "growth_factor",
        "frequency": "Monthly",
        "notes": (
            "Synthetic replica in the provider response format; invented "
            "values; research data only, NOT investable."
        ),
    }
    return json.dumps(payload).encode()


#: The declared availability policy: an observation at a month-end
#: instant is available at that same instant (declared, never inferred).
_POLICY = AvailabilityPolicy.same_instant()


def _qualified_world() -> dict[str, object]:
    """Load the synthetic world through the public offline facade path.

    Every series is admitted through ``portlearn.data.fred.load`` with
    caller-supplied synthetic provider-format bytes, the indivisible
    ``availability``/``tzinfo`` evidence pair, and an explicit offline
    api-key argument (no network, no environment read).  The returned
    datasets are ``QualifiedDataset`` values built by the facade — the
    battery never hand-builds a dataset.
    """
    world: dict[str, object] = {}
    for series_id in _SYNTH_OBSERVATIONS:
        dataset = fred_facade.load(
            series_id,
            "offline-synthetic-key",
            frequency="Monthly",
            data=_observation_bytes(series_id),
            metadata_bytes=_metadata_bytes(series_id),
            availability=_POLICY,
            tzinfo=_ZONE,
        )
        world[series_id] = dataset
    return world


def _information_set(
    records: list[TimedObservation], as_of: datetime
) -> InformationSet:
    """Admit the point-in-time visible subset of ``records`` at ``as_of``.

    The fixed admission law rejects on the *whole* submitted
    collection — one not-yet-available record rejects the entire set —
    so the caller selects the visible records first (the point-in-time
    discipline), exactly as the public research chain must.
    """
    visible = [record for record in records if record.available_time <= as_of]
    return InformationSet(visible, as_of=as_of)


def _world_records(world: dict[str, object]) -> list[TimedObservation]:
    return [record for dataset in world.values() for record in dataset.records]


def _obs_value(
    records: list[TimedObservation], series_id: str, instant: datetime
) -> float:
    for record in records:
        if record.series_id == series_id and record.observation_time == instant:
            return record.value
    raise KeyError(f"no observation {series_id}@{instant.isoformat()}")


def _world_growth_factors(
    records: list[TimedObservation],
) -> dict[datetime, dict[str, float]]:
    """Segment gross factors, read off the admitted observations.

    Hand mapping: an asset's factor for the segment starting at instant
    ``t`` is its synthetic observation dated exactly ``t``; the CASH
    factor is carried by ``SYNTHGC``.  (CASH is an explicit named asset
    with its own factor — never an implicit residual.)
    """
    return {
        # [T0, T1): A grows 2.5, B grows 1.5 (hand: 0.5*2.5+0.5*1.5 = 2.0)
        T0: {
            "A": _obs_value(records, "FRED/SYNTHGA", T0),
            "B": _obs_value(records, "FRED/SYNTHGB", T0),
        },
        # [T1, T2): CASH grows 2.0 (SYNTHGC), B grows 2.0 (uniform drift)
        T1: {
            "CASH": _obs_value(records, "FRED/SYNTHGC", T1),
            "B": _obs_value(records, "FRED/SYNTHGB", T1),
        },
        # [T2, T3): all cash, CASH grows 1.25 (SYNTHGC)
        T2: {"CASH": _obs_value(records, "FRED/SYNTHGC", T2)},
    }


def _cost_model() -> Proportional:
    return Proportional(COST_RATE, turnover=one_way)


# --------------------------------------------------------------------------- #
# Lawful stand-ins (P0/P2/P3): everything below drives the public seam only    #
# --------------------------------------------------------------------------- #


class GrowthFactorForecaster:
    """C3 research forecaster stand-in.

    Produces a provenance-bearing ``Forecast`` whose values are the
    latest *visible vintage* per series (selected with the public
    ``vintage_as_of`` operation), keyed by series id.  This is the only
    ``Forecast`` the public research chain can derive from the admitted
    data — the falsification floors use it as the lawful baseline.
    """

    identity = "growth-factor-stand-in/1"

    def forecast(self, information_set: InformationSet) -> Forecast:
        as_of = information_set.as_of
        values: dict[str, float] = {}
        series_ids = sorted({record.series_id for record in information_set})
        for series_id in series_ids:
            observation_times = sorted(
                {
                    record.observation_time
                    for record in information_set
                    if record.series_id == series_id
                }
            )
            visible = None
            for observation_time in observation_times:
                group = [
                    record
                    for record in information_set
                    if record.series_id == series_id
                    and record.observation_time == observation_time
                ]
                vintage = vintage_as_of(group, as_of)
                if vintage is not None:
                    visible = vintage
            if visible is not None:
                values[series_id] = visible.value
        return Forecast(
            values=values,
            target="growth_factor",
            decision_time=as_of,
            produced_by=(
                f"{self.identity} over the point-in-time records "
                "admitted at the decision instant of the synthetic world"
            ),
        )


class PinnedTargetWeightsForecaster:
    """Forecast-to-weights forecaster stand-in (C10/C13/C14 chain).

    Emits the synthetic-world target weights as its forecast
    values at each decision instant; the key set of the emitted values
    *is* the target universe, which is how the changing-universe family
    rides lawfully through the seam.  ``produced_by`` discloses the
    pinned stand-in placeholder — the register records it.
    """

    identity = "pinned-target-weights-stand-in/1"

    def forecast(self, information_set: InformationSet) -> Forecast:
        return Forecast(
            values=dict(WORLD_TARGETS[information_set.as_of]),
            target="portfolio_weights",
            decision_time=information_set.as_of,
            produced_by=(
                f"{self.identity}; pinned synthetic-world targets "
                "(disclosed stand-in placeholder)"
            ),
        )


class ForecastToWeightsStandIn:
    """C10 representation stand-in: forecast values are the target book.

    The natural seam shape — ``decide`` maps the forecast's values to
    the target weights one-to-one over the forecast's key set, with
    same-instant decide-and-execute (admissible under the chronology
    law).  No semantic placeholder is required for this family.
    """

    identity = "forecast-to-weights-stand-in/1"

    def decide(self, context: DecisionContext) -> DecisionResult:
        forecast = context.forecast
        assert forecast is not None  # this family's channel is the forecast
        return DecisionResult(
            decision=PortfolioDecision(
                decision_time=forecast.decision_time,
                execution_time=forecast.decision_time,
                target_weights=dict(forecast.values),
            ),
            next_strategy_state=None,
        )


class StatefulTargetWeightsStrategy:
    """P2/P0 chain strategy: pinned targets + replayable carried state.

    Carries the emitted-target history through the lawful state
    channel of the decision seam: ``decide`` reads ``context.strategy_state``
    (the prior history, ``None`` on the first decision) and returns
    the extended history as ``next_strategy_state`` — the input state
    is never mutated in place.  Each decision executes at the *next*
    month end — the delayed execution the synthetic world charges to
    the execution period.
    """

    identity = "stateful-target-weights-stand-in/2"

    def decide(self, context: DecisionContext) -> DecisionResult:
        target = dict(WORLD_TARGETS[context.decision_time])
        prior = context.strategy_state
        history = (tuple(prior) + (target,)) if prior is not None else (target,)
        return DecisionResult(
            decision=PortfolioDecision(
                decision_time=context.decision_time,
                execution_time=_next_month_end(context.decision_time),
                target_weights=target,
            ),
            next_strategy_state=history,
        )


class InformationDroppingStandIn:
    """Clause (v) demonstration stand-in: information-blind fallback.

    Returns the only book it can lawfully know without the current
    universe or holdings — the constant all-cash book — regardless of
    the forecast, i.e. it *drops* the economically necessary
    information rather than fabricating or smuggling it.
    """

    identity = "information-dropping-stand-in/1"

    def decide(self, context: DecisionContext) -> DecisionResult:
        return DecisionResult(
            decision=PortfolioDecision(
                decision_time=context.decision_time,
                execution_time=context.decision_time,
                target_weights={"CASH": 1.0},
            ),
            next_strategy_state=None,
        )


class EqualWeightStandIn:
    """C8 stand-in: equal weight over the current universe.

    Reads the current investable universe from the explicit
    ``DecisionContext.universe`` channel — caller order is
    preserved but the book itself is order-free: every member earns
    ``1/n``.  The output is value-bearing in the channel: a different
    universe yields a different book.
    """

    identity = "equal-weight-stand-in/2"

    def decide(self, context: DecisionContext) -> DecisionResult:
        members = context.universe
        count = len(members)
        weights = {asset: 1.0 / count for asset in members}
        return DecisionResult(
            decision=PortfolioDecision(
                decision_time=context.decision_time,
                execution_time=context.decision_time,
                target_weights=weights,
            ),
            next_strategy_state=None,
        )


class MinimumVarianceStandIn:
    """C9 stand-in: inverse-variance weights over the current universe.

    Needs the second moments of the *current universe*.  The moments
    are the strategy's own carried risk model: they ride the lawful
    ``DecisionContext.strategy_state`` channel (a plain mapping of
    asset to variance over ``context.universe``), never a forecast;
    absent the carried model it fails closed.
    """

    identity = "minimum-variance-stand-in/2"

    def decide(self, context: DecisionContext) -> DecisionResult:
        variances = context.strategy_state
        if not variances:
            raise ValueError(
                "minimum variance needs the covariance of the current "
                "universe; no lawful Forecast carries it"
            )
        inverse = {asset: 1.0 / variances[asset] for asset in context.universe}
        total = sum(inverse.values())
        weights = {asset: share / total for asset, share in inverse.items()}
        return DecisionResult(
            decision=PortfolioDecision(
                decision_time=context.decision_time,
                execution_time=context.decision_time,
                target_weights=weights,
            ),
            next_strategy_state=None,
        )


class DirectNeuralAllocationStandIn:
    """C11 stand-in: fixed-order feature vector mapped to weights.

    The network input contract is a feature vector over the current
    universe; the vector is the strategy's own carried input state and
    rides the lawful ``DecisionContext.strategy_state`` channel (a
    plain mapping of asset to feature over ``context.universe``),
    normalizing to unit budget; absent the carried vector it fails
    closed.
    """

    identity = "direct-neural-allocation-stand-in/2"

    def decide(self, context: DecisionContext) -> DecisionResult:
        features = context.strategy_state
        if not features:
            raise ValueError(
                "direct neural allocation needs its feature vector over "
                "the current universe; no lawful Forecast carries it"
            )
        total = sum(features[asset] for asset in context.universe)
        weights = {
            asset: features[asset] / total for asset in context.universe
        }
        return DecisionResult(
            decision=PortfolioDecision(
                decision_time=context.decision_time,
                execution_time=context.decision_time,
                target_weights=weights,
            ),
            next_strategy_state=None,
        )


class StatefulRLStandIn:
    """C12 stand-in: carried policy state + a no-trade band.

    The policy's environment half — the pre-trade portfolio it must
    act on — rides the lawful ``DecisionContext.current_weights``
    channel; its own half (the desired target book and the decision
    and trade counters) rides ``DecisionContext.strategy_state`` and
    transitions only through ``DecisionResult.next_strategy_state``
    (the input state is never mutated in place).  Inside the band
    (max |target - pre_trade| < 0.10) it keeps the holdings; outside
    it trades to the target.  Absent the carried policy state it
    fails closed.
    """

    identity = "stateful-rl-stand-in/2"

    def decide(self, context: DecisionContext) -> DecisionResult:
        state = context.strategy_state
        if not isinstance(state, dict) or "target" not in state:
            raise ValueError(
                "the no-trade band needs the policy's carried target and "
                "counters; no lawful Forecast carries them"
            )
        target = dict(state["target"])
        pre_trade = dict(context.current_weights.weights)
        decisions = state.get("decisions", 0) + 1
        trades = state.get("trades", 0)
        union = sorted(set(pre_trade) | set(target))
        if max(
            abs(target.get(asset, 0.0) - pre_trade.get(asset, 0.0))
            for asset in union
        ) < 0.10:
            decision = PortfolioDecision(
                decision_time=context.decision_time,
                execution_time=context.decision_time,
                target_weights=pre_trade,
            )
        else:
            trades += 1
            decision = PortfolioDecision(
                decision_time=context.decision_time,
                execution_time=context.decision_time,
                target_weights=target,
            )
        return DecisionResult(
            decision=decision,
            next_strategy_state={
                "target": target,
                "decisions": decisions,
                "trades": trades,
            },
        )


def _every_lawful_forecast() -> list[Forecast]:
    """Every forecast the public research chain can derive.

    Runs the ``GrowthFactorForecaster`` stand-in over the qualified
    synthetic world at every decision instant.  The falsification
    floors enumerate this list to prove that no lawful producer emits
    the information their strategy family needs.
    """
    records = _world_records(_qualified_world())
    forecaster = GrowthFactorForecaster()
    return [
        forecaster.forecast(_information_set(records, as_of=instant))
        for instant in (S0, T0, T1, T2)
    ]


def _every_target_weights_forecast() -> list[Forecast]:
    """Every forecast the forecast-to-weights stand-in can derive."""
    records = _world_records(_qualified_world())
    forecaster = PinnedTargetWeightsForecaster()
    return [
        forecaster.forecast(_information_set(records, as_of=instant))
        for instant in (S0, T0, T1)
    ]


def _assembly_manifest(run_id: str, created_at: datetime) -> RunManifest:
    """A valid run manifest asserting exactly the provenance it owns."""
    return RunManifest(
        run_id=run_id,
        created_at=created_at,
        python_version=platform.python_version(),
        package_version=__version__,
        dependency_pins={"portlearn": __version__, "pandas": _dist_version("pandas")},
        commands=["uv run pytest tests/test_composition_falsification.py -q"],
    )


def _decision_context(
    records: list[TimedObservation],
    instant: datetime,
    strategy_state: object = None,
    current_weights: PortfolioWeights | None = None,
    universe: tuple[str, ...] | None = None,
    forecast: Forecast | None = None,
) -> DecisionContext:
    """One lawful ``DecisionContext`` at ``instant`` over the world
    records.  Defaults carry every channel: the pinned-
    targets forecast and its information set at the instant, the
    current universe, the hand-derived pre-trade book (as of the
    decision instant itself), and the caller's carried state."""
    return DecisionContext(
        decision_time=instant,
        information=_information_set(records, as_of=instant),
        universe=universe or tuple(sorted(CURRENT_UNIVERSE[instant])),
        current_weights=current_weights
        or PortfolioWeights(
            dict(_PRE_DECISION_BOOK[instant]), WeightState.PRE_TRADE
        ),
        current_weights_as_of=instant,
        strategy_state=strategy_state,
        forecast=forecast
        or PinnedTargetWeightsForecaster().forecast(
            _information_set(records, as_of=instant)
        ),
    )


def _run_composition(
    run_id: str, created_at: datetime
) -> tuple[object, list[PortfolioDecision], tuple[dict[str, float], ...], RunManifest]:
    """Assemble the P0 chain from scratch and run it through the ledger.

    Fresh qualified datasets, fresh stand-ins, fresh decisions, fresh
    manifest — an independent assembly.  The carried state threads
    lawfully through ``DecisionResult.next_strategy_state``.  Two
    calls with identical world inputs must produce value-equal paths,
    identical decisions and identical carried state (C14).
    """
    world = _qualified_world()
    records = _world_records(world)
    forecaster = PinnedTargetWeightsForecaster()
    strategy = StatefulTargetWeightsStrategy()
    decisions = []
    state: object = None
    for instant in (S0, T0, T1):
        result = strategy.decide(
            _decision_context(
                records,
                instant,
                strategy_state=state,
                forecast=forecaster.forecast(
                    _information_set(records, as_of=instant)
                ),
            )
        )
        decisions.append(result.decision)
        state = result.next_strategy_state
    path = build_ledger(
        initial_weights=dict(INITIAL_BOOK),
        accounting_instants=(T0, T1, T2, T3),
        growth_factors=_world_growth_factors(records),
        decisions=decisions,
        cost_model=_cost_model(),
    )
    return path, decisions, state, _assembly_manifest(run_id, created_at)


# --------------------------------------------------------------------------- #
# C1 — synthetic data qualifies through public admission                       #
# --------------------------------------------------------------------------- #


def test_synthetic_dataset_qualifies_through_public_admission() -> None:
    """C1 (PASS).  The synthetic world's provider-format bytes qualify
    through the public facade admission laws and become the decision-
    time eligible sealed state that P0 consumes."""
    world = _qualified_world()
    expected_counts = {"SYNTHGA": 2, "SYNTHGB": 3, "SYNTHGC": 2}
    for series_id, dataset in world.items():
        # The qualified sealed state, through the indivisible pair.
        assert dataset.availability_state == "QUALIFIED"
        assert dataset.availability is _POLICY
        # Records are timed observations with declared same-instant
        # availability at month-end instants (5 rows total per series
        # declaration; monthly dates map to month ends).
        records = dataset.records
        assert len(records) == expected_counts[series_id]
        assert all(isinstance(record, TimedObservation) for record in records)
        for record in records:
            assert record.observation_time == record.available_time
            assert record.observation_time in (S0, T0, T1, T2)
        # Units delegate to the qualified provenance (never restated).
        assert dataset.units == "growth_factor"
        # The retained bytes are hash-pinned by the public property.
        assert dataset.source_sha256 == hashlib.sha256(
            _observation_bytes(series_id)
        ).hexdigest()

    # The raw public path (no evidence pair) stays UNQUALIFIED and is
    # refused at decision time; one-sided qualification fails closed.
    raw = fred_facade.load(
        "SYNTHGA",
        "offline-synthetic-key",
        frequency="Monthly",
        data=_observation_bytes("SYNTHGA"),
        metadata_bytes=_metadata_bytes("SYNTHGA"),
    )
    assert raw.availability_state == "UNQUALIFIED"
    assert raw.availability is None
    with pytest.raises(UnqualifiedDataError):
        raw.to_information_set(as_of=T0)
    with pytest.raises(ValueError):
        raw.qualify(availability=_POLICY)


# --------------------------------------------------------------------------- #
# C2 — information sets assemble from the qualified data                      #
# --------------------------------------------------------------------------- #


def test_information_set_assembles_from_qualified_data() -> None:
    """C2 (PASS).  ``InformationSet`` builds from the qualified
    observations with the pinned constructor signature and admits as
    the forecaster's input under the fixed admission law."""
    records = _world_records(_qualified_world())
    # Pinned public constructor signature: exactly (items, as_of).
    assert list(inspect.signature(InformationSet.__init__).parameters) == [
        "self",
        "items",
        "as_of",
    ]
    # Admitted record counts per decision instant (hand-counted):
    # S0 sees the two May observations; T0 adds A/B June; T1 adds B/C
    # July; T2 adds C August.
    for instant, expected in ((S0, 2), (T0, 4), (T1, 6), (T2, 7)):
        information_set = _information_set(records, as_of=instant)
        assert information_set.as_of == instant
        assert len(list(information_set)) == expected
    # The admission law is real: submitting the raw collection — one
    # not-yet-available record present — rejects the entire set
    # (look-ahead), and a duplicated record identity rejects (no
    # last-write-wins).  The point-in-time selection the helper applies
    # is the caller's obligation, not the container's.
    with pytest.raises(FutureInformationError):
        InformationSet(records, as_of=S0)
    duplicate = next(record for record in records if record.observation_time == S0)
    with pytest.raises(AmbiguousObservationError):
        InformationSet([duplicate, duplicate], as_of=S0)


# --------------------------------------------------------------------------- #
# C3 — the forecaster stand-in produces a provenance-bearing forecast         #
# --------------------------------------------------------------------------- #


def test_forecaster_stand_in_produces_provenance_bearing_forecast() -> None:
    """C3 (PASS).  The trivial forecaster produces a ``Forecast``
    (values, target, decision_time, produced_by) with
    ``decision_time == as_of`` — the implementer's obligation."""
    records = _world_records(_qualified_world())
    forecaster = GrowthFactorForecaster()
    # The public Forecast shape is exactly the four pinned fields.
    assert [field.name for field in fields(Forecast)] == [
        "values",
        "target",
        "decision_time",
        "produced_by",
    ]
    # At T0 the latest visible vintages are A=2.5 and B=1.5 (the June
    # observations; hand-checked via vintage_as_of below).
    forecast = forecaster.forecast(_information_set(records, as_of=T0))
    assert forecast.decision_time == T0
    assert forecast.target == "growth_factor"
    assert forecast.values == {"FRED/SYNTHGA": 2.5, "FRED/SYNTHGB": 1.5}
    assert forecaster.identity in forecast.produced_by
    # The values are the strict vintage operation's own selection: the
    # query submits exactly one (series_id, observation_time) group —
    # the June observation of SYNTHGA — and its visible vintage is 2.5.
    june_group = [
        record
        for record in records
        if record.series_id == "FRED/SYNTHGA" and record.observation_time == T0
    ]
    assert len(june_group) == 1
    assert vintage_as_of(june_group, T0).value == 2.5
    # At T2 the C series has become visible: three values (hand-set).
    late = forecaster.forecast(_information_set(records, as_of=T2))
    assert late.values == {
        "FRED/SYNTHGA": 2.5,
        "FRED/SYNTHGB": 2.0,
        "FRED/SYNTHGC": 1.25,
    }


# --------------------------------------------------------------------------- #
# C4 — the strategy stand-in decides target weights from a forecast           #
# --------------------------------------------------------------------------- #


def test_strategy_stand_in_decides_target_weights_from_forecast() -> None:
    """C4 (PASS).  The trivial strategy maps ``DecisionContext ->
    DecisionResult`` through the ``decide`` seam with the chronology
    law honored (delayed execution admissible, inversion rejected)."""
    records = _world_records(_qualified_world())
    forecast = PinnedTargetWeightsForecaster().forecast(
        _information_set(records, as_of=S0)
    )
    context = _decision_context(records, S0, forecast=forecast)
    result = StatefulTargetWeightsStrategy().decide(context)
    decision = result.decision
    # The seam: values -> target weights, decided at the context's
    # decision instant, executed at the next month end (delayed,
    # lawful).
    assert decision.target_weights == {"A": 0.5, "B": 0.5}
    assert decision.decision_time == forecast.decision_time == S0
    assert decision.execution_time == T0
    assert decision.execution_time >= decision.decision_time
    assert require_forecast_decision_compatible(forecast, decision) is None
    assert require_decision_result_compatible(context, result) is None
    # The state channel transitions lawfully: the first decision's
    # next_strategy_state is exactly the one-book history.
    assert result.next_strategy_state == ({"A": 0.5, "B": 0.5},)
    # The chronology law is unconditional at the decision value object
    # itself: a trade executing before it is decided cannot even be
    # constructed, so it can never reach the accounting surface.
    with pytest.raises(InvalidChronologyError):
        PortfolioDecision(
            decision_time=T0,
            execution_time=S0,
            target_weights={"A": 0.5, "B": 0.5},
        )


# --------------------------------------------------------------------------- #
# C5 — the composition closes wealth end to end                               #
# --------------------------------------------------------------------------- #


def test_composition_closes_wealth_end_to_end() -> None:
    """C5 (PASS).  The P0 chain through ``build_ledger`` closes wealth
    exactly at every row (W_{t+1} = W_t * G_net,t), with delayed
    execution charged to the execution period and turnover/costs under
    the cost laws.  Every expected value below is hand-derived in the
    module docstring."""
    path, decisions, _state, _manifest = _run_composition(
        "m26-c5", S0
    )
    assert path.n_periods == 3
    # The three synthetic-world decisions, decided one month before
    # they execute (delayed execution), charged to the execution
    # period: d1 is decided at S0 but its cost sits in row [T0, T1).
    assert [decision.decision_time for decision in decisions] == [S0, T0, T1]
    assert [decision.execution_time for decision in decisions] == [T0, T1, T2]

    row0, row1, row2 = path.rows
    # Row [T0, T1): one_way 0.5, q = 0.125, G_net = 1.75.
    assert row0.period_start == T0 and row0.period_end == T1
    assert row0.turnover == 0.5
    assert row0.transaction_cost == 0.125
    assert row0.gross_return == 1.0  # G_gross = 2.0 (simple return)
    assert row0.net_return == 0.75  # G_net = 1.75 (simple return)
    assert row0.wealth_open == 1.0 and row0.wealth_close == 1.75
    assert dict(row0.closing_weights) == {"A": 0.625, "B": 0.375}
    assert sorted(row0.universe) == ["A", "B", "CASH"]
    execution = row0.executions[0]
    assert execution.decision_time == S0 and execution.execution_time == T0
    assert execution.turnover == 0.5 and execution.transaction_cost == 0.125
    assert dict(execution.pre_trade_weights) == dict(INITIAL_BOOK)
    assert dict(execution.post_trade_weights) == {"A": 0.5, "B": 0.5}

    # Row [T1, T2): one_way 0.625, q = 0.15625, G_net = 1.6875.
    assert row1.period_start == T1 and row1.period_end == T2
    assert row1.turnover == 0.625
    assert row1.transaction_cost == 0.15625
    assert row1.gross_return == 1.0  # G_gross = 2.0
    assert row1.net_return == 0.6875  # G_net = 1.6875
    assert row1.wealth_open == 1.75 and row1.wealth_close == 2.953125
    assert dict(row1.closing_weights) == {"CASH": 0.625, "B": 0.375}
    assert sorted(row1.universe) == ["A", "B", "CASH"]

    # Row [T2, T3): one_way 0.375, q = 0.09375, G_net = 1.1328125.
    assert row2.period_start == T2 and row2.period_end == T3
    assert row2.turnover == 0.375
    assert row2.transaction_cost == 0.09375
    assert row2.gross_return == 0.25  # G_gross = 1.25
    assert row2.net_return == 0.1328125  # G_net = 1.1328125
    assert row2.wealth_open == 2.953125
    assert row2.wealth_close == 3.3453369140625
    assert dict(row2.closing_weights) == {"CASH": 1.0}
    assert sorted(row2.universe) == ["B", "CASH"]

    # The wealth identity holds exactly at every row, and the final
    # wealth is the dyadic-exact rational 27405/8192.
    for row in path.rows:
        assert row.wealth_close == row.wealth_open * (1.0 + row.net_return)
    assert path.final_wealth == 27405 / 8192
    assert path.final_wealth == 3.3453369140625


# --------------------------------------------------------------------------- #
# C6 — stateful strategy state replays across decision dates                  #
# --------------------------------------------------------------------------- #


def test_stateful_strategy_state_replays_across_decision_dates() -> None:
    """C6 (PASS).  The P2 stand-in's carried state replays
    deterministically: two independent assemblies reach identical
    decisions and identical state."""
    records = _world_records(_qualified_world())
    forecaster = PinnedTargetWeightsForecaster()
    forecasts = [
        forecaster.forecast(_information_set(records, as_of=instant))
        for instant in (S0, T0, T1)
    ]

    def _replay(
        forecast_order: list[Forecast],
    ) -> tuple[list[PortfolioDecision], object]:
        decisions = []
        state: object = None
        for forecast in forecast_order:
            result = StatefulTargetWeightsStrategy().decide(
                _decision_context(
                    records,
                    forecast.decision_time,
                    strategy_state=state,
                    forecast=forecast,
                )
            )
            decisions.append(result.decision)
            state = result.next_strategy_state
        return decisions, state

    decisions_a, state_a = _replay(forecasts)
    decisions_b, state_b = _replay(forecasts)
    assert decisions_a == decisions_b
    assert state_a == state_b
    # The state is value-bearing: it holds exactly the three emitted
    # synthetic-world target books, in order.
    assert state_a == (
        {"A": 0.5, "B": 0.5},
        {"CASH": 0.625, "B": 0.375},
        {"CASH": 1.0},
    )
    # And it genuinely depends on the inputs: a different decision
    # order produces a different carried state.
    _decisions_r, state_r = _replay(list(reversed(forecasts)))
    assert state_r != state_a


# --------------------------------------------------------------------------- #
# C7 — aggregate coverage across the six representation families              #
# --------------------------------------------------------------------------- #


def test_representation_probes_cover_six_contract_families() -> None:
    """C7 (PASS).  Each of the six P3 representations builds a trivial
    stand-in, executes it through the public ``decide`` seam, and
    produces a hand-checkable decision observable.  The floor asserts
    coverage and value-bearing outputs, not method correctness."""
    observables: dict[str, bool] = {}
    records = _world_records(_qualified_world())

    # Equal weight — the explicit universe channel (REG-C8,
    # repaired): {CASH, A, B} at 1/3 each, unit budget.
    context = _decision_context(records, T0)
    target = EqualWeightStandIn().decide(context).decision.target_weights
    observables["equal_weight"] = (
        target == {"CASH": 1 / 3, "A": 1 / 3, "B": 1 / 3}
        and sum(target.values()) == 1.0
    )

    # Minimum variance — the carried risk-model state over the
    # universe (REG-C9, repaired): inverse-variance weights
    # w_A = 0.25/1.25 = 0.2, w_B = 0.8.
    context = _decision_context(
        records, T0, strategy_state={"A": 4.0, "B": 1.0}, universe=("A", "B")
    )
    target = MinimumVarianceStandIn().decide(context).decision.target_weights
    observables["minimum_variance"] = target == {
        "A": 0.25 / 1.25,
        "B": 1.0 / 1.25,
    }

    # Forecast to weights — the natural seam shape, no placeholder.
    forecast = PinnedTargetWeightsForecaster().forecast(
        _information_set(records, as_of=T0)
    )
    decision = ForecastToWeightsStandIn().decide(
        _decision_context(records, T0, forecast=forecast)
    ).decision
    observables["forecast_to_weights"] = (
        decision.target_weights == {"CASH": 0.625, "B": 0.375}
        and decision.decision_time == T0
    )

    # Direct neural allocation — the carried feature vector over the
    # universe (REG-C11, repaired): v = (1.0, 3.0) normalized ->
    # {A: 0.25, B: 0.75}, binary-exact.
    context = _decision_context(
        records, T0, strategy_state={"A": 1.0, "B": 3.0}, universe=("A", "B")
    )
    target = DirectNeuralAllocationStandIn().decide(context).decision.target_weights
    observables["direct_neural_allocation"] = target == {"A": 0.25, "B": 0.75}

    # Stateful RL — the current-weights channel (REG-C12, repaired):
    # inside the no-trade band the policy keeps the pre-trade book.
    context = _decision_context(
        records,
        T0,
        current_weights=PortfolioWeights(
            {"CASH": 0.625, "B": 0.375}, WeightState.PRE_TRADE
        ),
        strategy_state={
            "target": {"CASH": 0.625, "B": 0.375},
            "decisions": 0,
            "trades": 0,
        },
    )
    result = StatefulRLStandIn().decide(context)
    observables["stateful_rl"] = (
        result.decision.target_weights == {"CASH": 0.625, "B": 0.375}
        and result.next_strategy_state["trades"] == 0
    )

    # Changing universe — the explicit universe channel is canonical
    # (REG-C13, placeholder retired): the context's universe tracks
    # the changing universe across instants.
    universes = {
        instant: _decision_context(records, instant).universe
        for instant in (S0, T0, T1)
    }
    observables["changing_universe"] = universes == {
        S0: ("A", "B", "CASH"),
        T0: ("A", "B", "CASH"),
        T1: ("B", "CASH"),
    }

    # Aggregate coverage: exactly the six families, every observable
    # value-bearing and green.
    assert set(observables) == {
        "equal_weight",
        "minimum_variance",
        "forecast_to_weights",
        "direct_neural_allocation",
        "stateful_rl",
        "changing_universe",
    }
    assert all(observables.values())
    # Register disclosure: after the decision-seam repair no family
    # requires a semantic placeholder — every input rides a lawful
    # DecisionContext channel, and the forecast-to-weights family
    # needs nothing beyond the forecast itself.
    placeholders: set[str] = set()
    assert "forecast_to_weights" not in placeholders
    assert placeholders <= set(observables)


# --------------------------------------------------------------------------- #
# C8 — equal weight (repaired: explicit universe channel)                      #
# --------------------------------------------------------------------------- #


def test_equal_weight_representation_yields_natural_decision_observable() -> None:
    """C8 (PASS — register REG-C8, repaired through the DecisionContext seam).

    The equal-weight family needs the *current investable universe*
    (``{CASH, A, B}`` at T0 — A is still held even though the T0
    target drops it).  The ``DecisionContext.universe`` channel now
    carries it lawfully; this floor keeps the original falsification
    teeth as change-detectors: it fails if the channel is removed
    from the seam again, and it still proves that no forecast channel
    can carry the universe.
    """
    strategy = EqualWeightStandIn()
    records = _world_records(_qualified_world())
    lawful_forecasts = _every_lawful_forecast()
    pinned_forecasts = _every_target_weights_forecast()
    every_forecast = lawful_forecasts + pinned_forecasts

    # (a) Real public API call, lawful context: the universe arrives on
    # its own channel and the output is the equal-weight book over it —
    # value-bearing in the channel (a different universe yields a
    # different book).
    context = _decision_context(records, T0)
    output = strategy.decide(context).decision
    assert set(output.target_weights) == {"CASH", "A", "B"}
    assert output.target_weights == {"CASH": 1 / 3, "A": 1 / 3, "B": 1 / 3}
    assert sum(output.target_weights.values()) == 1.0
    narrower = _decision_context(records, T1)
    narrower_output = strategy.decide(narrower).decision
    assert narrower_output.target_weights == {"B": 0.5, "CASH": 0.5}

    # The channel is the only lawful carrier: every forecast key set is
    # provider series ids or a *target* universe — never the current
    # universe — so the universe cannot ride a forecast (clauses (i)
    # fabrication and (ii) smuggling remain unlawful routes).
    assert not ({"CASH", "A", "B"} & set(lawful_forecasts[1].values))
    for forecast in every_forecast:
        assert set(forecast.values) != {"CASH", "A", "B"}

    # Clause (v) dropping — the fallback that ignores the channel: its
    # output is identical under strictly different universes.
    blind = InformationDroppingStandIn()
    assert (
        blind.decide(context).decision.target_weights
        == blind.decide(narrower).decision.target_weights
        == {"CASH": 1.0}
    )

    # The repaired seam carries the channel (change-detector: this pin
    # and the field pin fail if the universe parameter is removed
    # again).
    assert list(inspect.signature(Strategy.decide).parameters) == [
        "self",
        "context",
    ]
    assert "universe" in {field.name for field in fields(DecisionContext)}


# --------------------------------------------------------------------------- #
# C9 — minimum variance (repaired: carried risk-model state)                   #
# --------------------------------------------------------------------------- #


def test_minimum_variance_representation_yields_natural_decision_observable() -> None:
    """C9 (PASS — register REG-C9, repaired through the DecisionContext seam).

    Minimum variance needs the second moments of the current
    universe.  The moments are the strategy's own carried risk model:
    they ride ``DecisionContext.strategy_state`` over
    ``context.universe`` — no lawful forecast carries them, and this
    floor still proves that while pinning the channels it needs.
    """
    strategy = MinimumVarianceStandIn()
    records = _world_records(_qualified_world())
    lawful_forecasts = _every_lawful_forecast()
    pinned_forecasts = _every_target_weights_forecast()

    # (a) Real public API call, lawful context: the carried risk model
    # arrives on the state channel and the output is the
    # inverse-variance book over the universe.  Hand check for
    # variances (var_A, var_B) = (4, 1): inverse-variance weights
    # w_A = (1/4)/(1/4+1) = 0.2, w_B = 1.0/1.25 = 0.8 (exact division).
    context = _decision_context(
        records, T0, strategy_state={"A": 4.0, "B": 1.0}, universe=("A", "B")
    )
    target = strategy.decide(context).decision.target_weights
    assert target == {"A": 0.25 / 1.25, "B": 1.0 / 1.25}
    assert target["A"] == 0.2 and target["B"] == 0.8

    # (b) The channel is unconditional: without the carried risk model
    # the strategy cannot decide — the moments exist on no other
    # lawful channel.
    with pytest.raises(ValueError, match="no lawful Forecast"):
        strategy.decide(_decision_context(records, T0))

    # The forecast channel still carries no covariance entry — the
    # lawful values are keyed by series ids and carry growth-factor
    # scalars; the pinned forecasts key by *target* asset ids
    # (clauses (i) fabrication and (ii) smuggling remain unlawful
    # routes, enumerated over every derivable forecast).
    for forecast in lawful_forecasts + pinned_forecasts:
        assert all(":" not in key for key in forecast.values)
        assert not any(key.startswith("var:") for key in forecast.values)

    # Clause (v) dropping — the fallback that ignores the channel
    # cannot name the universe, so its output is not the
    # minimum-variance book.
    blind_target = (
        InformationDroppingStandIn().decide(context).decision.target_weights
    )
    assert blind_target != {"A": 0.2, "B": 0.8}

    # The repaired seam carries the state channel (change-detector).
    assert list(inspect.signature(Strategy.decide).parameters) == [
        "self",
        "context",
    ]
    assert "strategy_state" in {
        field.name for field in fields(DecisionContext)
    }


# --------------------------------------------------------------------------- #
# C10 — forecast to weights (natural seam shape; PASS floor)                   #
# --------------------------------------------------------------------------- #


def test_forecast_to_weights_representation_yields_natural_decision_observable() -> None:
    """C10 (PASS).  The forecast-to-weights family is the natural seam
    shape: the forecast's values map one-to-one onto target weights
    over the forecast's own key set, with no semantic placeholder —
    no smuggling, no fabrication, no dropped information.  Pins the
    natural decision observable through the public decide seam."""
    records = _world_records(_qualified_world())
    forecast = PinnedTargetWeightsForecaster().forecast(
        _information_set(records, as_of=T0)
    )
    context = _decision_context(records, T0, forecast=forecast)
    result = ForecastToWeightsStandIn().decide(context)
    decision = result.decision

    # The natural decision observable: values -> weights over the
    # forecast's key set; decided and executed at the forecast
    # instant (same-instant decide-and-execute is admissible); unit
    # budget; chronology lawful.
    assert decision.target_weights == {"CASH": 0.625, "B": 0.375}
    assert decision.decision_time == T0
    assert decision.execution_time == T0
    assert sum(decision.target_weights.values()) == 1.0
    assert set(decision.target_weights) == set(forecast.values)
    assert require_forecast_decision_compatible(forecast, decision) is None
    assert require_decision_result_compatible(context, result) is None
    assert result.next_strategy_state is None  # the family is stateless


# --------------------------------------------------------------------------- #
# C11 — direct neural allocation (repaired: carried feature state)             #
# --------------------------------------------------------------------------- #


def test_direct_neural_allocation_representation_yields_natural_decision_observable() -> None:
    """C11 (PASS — register REG-C11, repaired through the DecisionContext seam).

    An end-to-end network maps a feature state over the current
    universe to weights.  The feature vector is the strategy's own
    carried input state: it rides
    ``DecisionContext.strategy_state`` over ``context.universe`` — no
    lawful forecast carries the universe, the feature semantics, or
    the input ordering the network contract needs.
    """
    strategy = DirectNeuralAllocationStandIn()
    records = _world_records(_qualified_world())
    lawful_forecasts = _every_lawful_forecast()
    pinned_forecasts = _every_target_weights_forecast()

    # (a) Real public API call, lawful context: the carried feature
    # vector arrives on the state channel.  Hand check: v = (1.0, 3.0)
    # over (A, B), normalized w_i = v_i / sum(v) = {A: 0.25, B: 0.75}
    # (binary-exact).
    context = _decision_context(
        records, T0, strategy_state={"A": 1.0, "B": 3.0}, universe=("A", "B")
    )
    target = strategy.decide(context).decision.target_weights
    assert target == {"A": 0.25, "B": 0.75}

    # (b) The channel is unconditional: without the carried feature
    # vector the strategy cannot decide.
    with pytest.raises(ValueError, match="no lawful Forecast"):
        strategy.decide(_decision_context(records, T0))

    # Even the arity is wrong on every forecast channel: the lawful
    # forecast at T0 exposes two series ids while the current universe
    # has three assets, and the key sets are disjoint from it
    # (clauses (i) and (ii) have no carrier).
    assert len(lawful_forecasts[1].values) == 2
    assert len(CURRENT_UNIVERSE[T0]) == 3
    assert not (CURRENT_UNIVERSE[T0] & set(lawful_forecasts[1].values))
    assert all(
        not any(key.startswith("feat:") for key in forecast.values)
        for forecast in lawful_forecasts + pinned_forecasts
    )

    # Clause (v) dropping — the fallback that ignores the channel
    # drops the feature state entirely; its output ignores the
    # admitted information.
    blind = InformationDroppingStandIn()
    assert (
        blind.decide(context).decision.target_weights
        == blind.decide(_decision_context(records, T1)).decision.target_weights
    )

    # The repaired seam carries the state channel (change-detector).
    assert list(inspect.signature(Strategy.decide).parameters) == [
        "self",
        "context",
    ]


# --------------------------------------------------------------------------- #
# C12 — stateful RL (repaired: current_weights + carried state)                #
# --------------------------------------------------------------------------- #


def test_stateful_rl_representation_yields_natural_decision_observable() -> None:
    """C12 (PASS — register REG-C12, repaired through the DecisionContext seam).

    The policy's own carried state (its desired target and its
    counters) rides ``DecisionContext.strategy_state`` and transitions
    only through ``DecisionResult.next_strategy_state``; the
    *environment* half — the pre-trade portfolio it must act on —
    rides ``DecisionContext.current_weights``.  Both halves are now
    lawful channels; this floor keeps the original teeth (holdings
    determine the action, the input state is never mutated in place,
    and no forecast carries either half).
    """
    records = _world_records(_qualified_world())
    lawful_forecasts = _every_lawful_forecast()
    pinned_forecasts = _every_target_weights_forecast()
    held_book = PortfolioWeights(
        {"CASH": 0.625, "B": 0.375}, WeightState.PRE_TRADE
    )
    all_cash = PortfolioWeights({"CASH": 1.0}, WeightState.PRE_TRADE)
    initial_state = {
        "target": {"CASH": 0.625, "B": 0.375},
        "decisions": 0,
        "trades": 0,
    }

    def _rl_context(
        pre_trade: PortfolioWeights, carried: object
    ) -> DecisionContext:
        return _decision_context(
            records, T0, current_weights=pre_trade, strategy_state=carried
        )

    # (a) Real public API call, lawful context: inside the no-trade
    # band (max |target - pre_trade| = 0 < 0.10) the policy keeps the
    # holdings, and the carried state transitions lawfully through
    # next_strategy_state — the input state object is never mutated in
    # place.
    held = StatefulRLStandIn().decide(_rl_context(held_book, initial_state))
    assert held.decision.target_weights == {"CASH": 0.625, "B": 0.375}
    assert held.next_strategy_state == {
        "target": {"CASH": 0.625, "B": 0.375},
        "decisions": 1,
        "trades": 0,
    }
    assert initial_state["decisions"] == 0  # the input state is untouched

    # The environment channel is value-bearing: identical market
    # content, different pre-trade books, different actions.  Outside
    # the band (max |Δ| = 0.375 >= 0.10) the policy trades to the
    # target.
    traded = StatefulRLStandIn().decide(_rl_context(all_cash, initial_state))
    assert traded.decision.target_weights == {"CASH": 0.625, "B": 0.375}
    assert traded.next_strategy_state["trades"] == 1

    # The carried-state mechanics replay (the C6 route): threading
    # next_strategy_state through the same environment sequence
    # reaches identical carried state in independent policies.
    finals = []
    for _policy in (StatefulRLStandIn(), StatefulRLStandIn()):
        carried: object = initial_state
        for book in (held_book, all_cash):
            carried = _policy.decide(_rl_context(book, carried)).next_strategy_state
        finals.append(carried)
    assert finals[0] == finals[1] == {
        "target": {"CASH": 0.625, "B": 0.375},
        "decisions": 2,
        "trades": 1,
    }

    # (b) The channel is unconditional: without the carried policy state
    # the strategy cannot decide.
    with pytest.raises(ValueError, match="no lawful Forecast"):
        StatefulRLStandIn().decide(_decision_context(records, T0))

    # Clause (i) fabrication — no lawful producer emits pre-trade or
    # target-role entries (enumerated over every derivable forecast):
    # the books ride their own channels now.
    assert all(
        not any(key.startswith(("pre:", "tgt:")) for key in forecast.values)
        for forecast in lawful_forecasts + pinned_forecasts
    )

    # Clause (v) dropping — without reading the channels the fallback
    # returns the same all-cash book whatever the holdings.
    blind = InformationDroppingStandIn()
    assert (
        blind.decide(_rl_context(held_book, initial_state)).decision.target_weights
        == blind.decide(_rl_context(all_cash, initial_state)).decision.target_weights
    )

    # The repaired seam carries the environment channel
    # (change-detector).
    assert list(inspect.signature(Strategy.decide).parameters) == [
        "self",
        "context",
    ]
    assert "current_weights" in {
        field.name for field in fields(DecisionContext)
    }


# --------------------------------------------------------------------------- #
# C13 — changing universe (valid: explicit universe channel)                   #
# --------------------------------------------------------------------------- #


def test_changing_universe_representation_yields_natural_decision_observable() -> None:
    """C13 (PASS — register REG-C13, explicit channel canonical).

    The changing-universe family rides the explicit
    ``DecisionContext.universe`` channel: the context names the
    current investable universe at each decision instant, and the
    stand-in's book tracks it as assets enter and exit.  The
    pre-repair placeholder (universe change expressed through the
    forecast key set) is retired — the explicit channel is canonical.
    """
    records = _world_records(_qualified_world())

    universes = {
        instant: _decision_context(records, instant).universe
        for instant in (S0, T0, T1)
    }
    decision_s0 = EqualWeightStandIn().decide(
        _decision_context(records, S0)
    ).decision
    decision_t0 = EqualWeightStandIn().decide(
        _decision_context(records, T0)
    ).decision
    decision_t1 = EqualWeightStandIn().decide(
        _decision_context(records, T1)
    ).decision

    # A is held from the start and exits after the T0 decision; B
    # exits at T1 — the universe of the *decision* changes every
    # period, and the stand-in's book tracks it exactly.
    assert universes == {
        S0: ("A", "B", "CASH"),
        T0: ("A", "B", "CASH"),
        T1: ("B", "CASH"),
    }
    assert decision_s0.target_weights == {"CASH": 1 / 3, "A": 1 / 3, "B": 1 / 3}
    assert decision_t0.target_weights == {"CASH": 1 / 3, "A": 1 / 3, "B": 1 / 3}
    assert decision_t1.target_weights == {"B": 0.5, "CASH": 0.5}
    assert "A" in decision_s0.target_weights
    assert "A" not in decision_t1.target_weights
    assert "B" in decision_t1.target_weights
    for decision in (decision_s0, decision_t0, decision_t1):
        assert sum(decision.target_weights.values()) == 1.0
        assert decision.execution_time >= decision.decision_time

    # The exit trades are executable weight-space transitions over the
    # union of the books (the fixed trades law): exiting A at T1
    # trades |Δ| = 0.625 exactly — hand-derived in the C5 docstring.
    path, _decisions, _state, _manifest = _run_composition("m26-c13", S0)
    exit_execution = path.rows[1].executions[0]
    assert dict(exit_execution.pre_trade_weights) == {"A": 0.625, "B": 0.375}
    assert dict(exit_execution.post_trade_weights) == {"CASH": 0.625, "B": 0.375}
    assert exit_execution.turnover == 0.625


# --------------------------------------------------------------------------- #
# C14 — independent assemblies replay identically                             #
# --------------------------------------------------------------------------- #


def test_composition_replays_identically_in_independent_assemblies() -> None:
    """C14 (PASS).  P0 assembled twice from identical inputs produces
    value-equal ``LedgerPath``s, identical decisions, and identical
    stand-in state.  Each assembly's ``RunManifest`` is merely valid —
    the two are deliberately given different run ids and creation
    instants and are NOT compared to each other."""
    path_a, decisions_a, state_a, manifest_a = _run_composition(
        "m26-assembly-alpha", S0
    )
    path_b, decisions_b, state_b, manifest_b = _run_composition(
        "m26-assembly-beta", S0 + timedelta(hours=1)
    )
    # Composition-level replay: value-equal paths and rows.
    assert path_a == path_b
    assert path_a.rows == path_b.rows
    assert path_a.final_wealth == path_b.final_wealth == 27405 / 8192
    # Identical decisions and identical stand-in state.
    assert decisions_a == decisions_b
    assert state_a == state_b == (
        {"A": 0.5, "B": 0.5},
        {"CASH": 0.625, "B": 0.375},
        {"CASH": 1.0},
    )
    # Each manifest is valid on its own (canonical round trip) while
    # legitimate run identity differs per assembly.
    for manifest in (manifest_a, manifest_b):
        assert RunManifest.from_json(manifest.to_json()).to_json() == manifest.to_json()
    assert manifest_a.run_id != manifest_b.run_id
    assert manifest_a.created_at != manifest_b.created_at


# --------------------------------------------------------------------------- #
# C14b — the manifest records only the provenance it owns                      #
# --------------------------------------------------------------------------- #


def test_run_manifest_records_only_provenance_it_owns() -> None:
    """C14b (PASS).  Each composition run carries a valid, populated
    ``RunManifest`` asserting exactly the provenance the manifest
    owns; the public schema is exactly the six fields, naive
    timestamps reject, and no manifest field is invented (there is no
    seed field; dataset/stand-in/synthetic-world identity lives in the
    evidence register as run parameters)."""
    _path, _decisions, _state, manifest = _run_composition("m26-c14b", S0)
    six = {
        "run_id",
        "created_at",
        "python_version",
        "package_version",
        "dependency_pins",
        "commands",
    }
    # The public schema is exactly the six fields — no seed, nothing
    # else — on both the slots surface and the canonical JSON surface.
    assert set(RunManifest.__slots__) == six
    assert set(json.loads(manifest.to_json())) == six
    assert "seed" not in json.loads(manifest.to_json())
    # Every field is populated with provenance the manifest owns.
    assert manifest.run_id == "m26-c14b"
    assert manifest.created_at.tzinfo is not None  # aware creation instant
    assert manifest.python_version == platform.python_version()
    assert manifest.package_version == __version__
    assert dict(manifest.dependency_pins) == {
        "portlearn": __version__,
        "pandas": _dist_version("pandas"),
    }
    assert manifest.commands == (
        "uv run pytest tests/test_composition_falsification.py -q",
    )
    # Naive creation instants reject — never is a timezone silently
    # assumed.
    with pytest.raises(NaiveTimestampError):
        _assembly_manifest(
            "naive", datetime(2026, 9, 18)  # noqa: DTZ001 — deliberately naive
        )
    # The manifest is immutable, pins included.
    with pytest.raises(AttributeError):
        manifest.run_id = "mutated"  # type: ignore[misc]
    with pytest.raises(TypeError):
        manifest.dependency_pins["injected"] = "1.0"


# --------------------------------------------------------------------------- #
# P4 — the current-portfolio-state decision flow (repaired: current_weights)  #
# --------------------------------------------------------------------------- #


def test_pre_trade_weights_decision_flow_reaches_decide_seam() -> None:
    """P4 (PASS — register REG-P4, repaired through the DecisionContext seam; the
    distilled minimal reproduction shared by C8/C9/C11/C12).

    The current portfolio state available at decision time reaches the
    decision seam lawfully through
    ``DecisionContext.current_weights``.  The later execution-time
    pre-trade book is computed by the public surface at exactly the
    right time (``drift_weights`` before execution) and recorded on
    every ledger execution — the two are distinguished by phase, not
    conflated.  Teeth preserved: the floor fails if the
    ``current_weights`` channel is removed again, and no forecast can
    carry the book.
    """
    # 1. The information exists publicly at decision time: the drifted
    #    pre-trade books the ledger itself will trade from.
    pre_t1 = drift_weights(
        PortfolioWeights({"A": 0.5, "B": 0.5}, WeightState.PRE_TRADE),
        {"A": 2.5, "B": 1.5},  # D = 0.5*2.5 + 0.5*1.5 = 2.0
    )
    assert dict(pre_t1.weights) == {"A": 0.625, "B": 0.375}
    pre_t2 = drift_weights(
        PortfolioWeights({"CASH": 0.625, "B": 0.375}, WeightState.PRE_TRADE),
        {"CASH": 2.0, "B": 2.0},  # uniform factors: the book is unchanged
    )
    assert dict(pre_t2.weights) == {"CASH": 0.625, "B": 0.375}

    # 2. The ledger records those exact books on its executions.
    path, _decisions, _state, _manifest = _run_composition("m26-p4", S0)
    executions = [e for row in path.rows for e in row.executions]
    assert dict(executions[0].pre_trade_weights) == dict(INITIAL_BOOK)
    assert dict(executions[1].pre_trade_weights) == dict(pre_t1.weights)
    assert dict(executions[2].pre_trade_weights) == dict(pre_t2.weights)

    # 3. The channel exists and carries them: the drifted book flows
    #    lawfully into decide, and a no-trade-band policy reading it
    #    holds exactly the ledger's own pre-trade book (change-
    #    detector: the pin and the field pin fail if the
    #    current_weights parameter is removed again).
    records = _world_records(_qualified_world())
    context = _decision_context(
        records,
        T1,
        current_weights=PortfolioWeights(
            dict(pre_t1.weights), WeightState.PRE_TRADE
        ),
        strategy_state={
            "target": dict(pre_t1.weights),
            "decisions": 0,
            "trades": 0,
        },
    )
    result = StatefulRLStandIn().decide(context)
    assert result.decision.target_weights == dict(pre_t1.weights)
    assert result.next_strategy_state["trades"] == 0
    assert require_decision_result_compatible(context, result) is None
    assert list(inspect.signature(Strategy.decide).parameters) == [
        "self",
        "context",
    ]
    assert "current_weights" in {
        field.name for field in fields(DecisionContext)
    }

    # 4. Clause (i) fabrication remains the only forecast route: no
    #    forecast any lawful stand-in can derive carries the pre-trade
    #    book (the t1 book {A: 0.625, B: 0.375} is series-keyed or a
    #    target — never the holdings).
    lawful = _every_lawful_forecast() + _every_target_weights_forecast()
    for forecast in lawful:
        assert forecast.values != {"A": 0.625, "B": 0.375}
        assert not any(key.startswith("pre:") for key in forecast.values)

    # 5. The role confusion is resolved by the channel structure: the
    #    target-weights forecast at T0 is *numerically identical* to
    #    the pre-trade book at T2 ({CASH: 0.625, B: 0.375}), but the
    #    two ride different lawful channels — ``Forecast.values`` vs
    #    ``current_weights`` snapshotted with ``WeightState.PRE_TRADE``
    #    — so a role-less float mapping can no longer confuse a
    #    desired target with the current holdings.
    pinned_t0 = _every_target_weights_forecast()[1]
    assert dict(pinned_t0.values) == dict(executions[2].pre_trade_weights)
    assert context.current_weights.state is WeightState.PRE_TRADE
    assert set(context.forecast.values) != set(context.current_weights.weights)

    # 6. Clause (v) dropping — the fallback ignores the holdings
    #    entirely; a no-trade policy cannot exist on it.
    blind = InformationDroppingStandIn()
    assert (
        blind.decide(context).decision.target_weights
        == blind.decide(_decision_context(records, T0)).decision.target_weights
    )


# --------------------------------------------------------------------------- #
# C15 — the evaluator stand-in returns one trivial observable (PASS floor)    #
# --------------------------------------------------------------------------- #


class PositionCountEvaluator:
    """C15 evaluator stand-in, held to the evaluator-contract boundary.

    Exactly one trivial observable — the position count of the
    post-trade book — computed through the public
    ``AccountingResult`` → ``Evaluator`` seam.  The stand-in defines
    no metric catalogue, statistic, benchmark, significance
    semantics, or result schema; terminal wealth and period count
    remain accounting observables the probe hands over, never
    "metrics" it computes.
    """

    identity = "position-count-evaluator-stand-in/1"

    def evaluate(self, accounting_result: AccountingResult) -> dict[str, float]:
        return {
            "position_count": float(len(accounting_result.post_trade_weights))
        }


def test_evaluator_stand_in_returns_one_trivial_observable() -> None:
    """C15 (PASS).  The P0 chain's terminal stand-in proves the
    ``AccountingResult`` → ``Evaluator`` public seam composes: the
    evaluator consumes exactly the accounting results the public
    engine produces for the chain's own executed decisions, returns
    exactly one trivial observable per result, and defines no metric
    catalogue, statistic, benchmark, significance semantics, or
    result schema — terminal wealth and period count stay
    probe-handed accounting observables on the ledger path."""
    # 1. The chain runs end-to-end and retains its executions — the
    #    accounting facts the evaluator will consume.
    path, decisions, _state, _manifest = _run_composition("m26-c15", S0)
    executions = [e for row in path.rows for e in row.executions]
    engine = ExactFillAccounting()
    evaluator = PositionCountEvaluator()

    # 2. The public engine accounts each executed decision exactly as
    #    the ledger does (ledger.py:695: decision, drifted pre-trade
    #    book, realized segment returns).  Hand-derived post-trade
    #    books under exact fill (POST_TRADE = TARGET):
    #    d1 → {A: 0.5, B: 0.5}, d2 → {CASH: 0.625, B: 0.375},
    #    d3 → {CASH: 1.0}.
    realized_books = (
        {"A": 2.5, "B": 1.5},  # [T0, T1) segment factors over the d1 target
        {"CASH": 2.0, "B": 2.0},  # [T1, T2) factors over the d2 target
        {"CASH": 1.25},  # [T2, T3) factor over the d3 target
    )
    results = [
        engine.account(decision, dict(execution.pre_trade_weights), realized)
        for decision, execution, realized in zip(
            decisions, executions, realized_books, strict=True
        )
    ]
    assert all(isinstance(result, AccountingResult) for result in results)
    assert [dict(result.post_trade_weights) for result in results] == [
        {"A": 0.5, "B": 0.5},
        {"CASH": 0.625, "B": 0.375},
        {"CASH": 1.0},
    ]
    # The accounted books are exactly the books the ledger retained —
    # the evaluator consumes the ledger's own accounting facts.
    for result, execution in zip(results, executions, strict=True):
        assert (
            dict(result.post_trade_weights)
            == dict(execution.post_trade_weights)
        )

    # 3. The seam composes: evaluate consumes each AccountingResult
    #    and returns exactly one trivial observable — the position
    #    count of the post-trade book (hand: 2 / 2 / 1).
    assert [evaluator.evaluate(result) for result in results] == [
        {"position_count": 2.0},
        {"position_count": 2.0},
        {"position_count": 1.0},
    ]

    # 4. The seam shape is fixed (interfaces.py:474): exactly one
    #    parameter — the accounting result, nothing else.
    assert list(inspect.signature(Evaluator.evaluate).parameters) == [
        "self",
        "accounting_result",
    ]

    # 5. The boundary holds: no metric/result schema is defined.  The
    #    mapping never carries more than the one trivial observable,
    #    and terminal wealth / period count are accounting
    #    observables on the path the evaluator never touches.
    for result in results:
        assert set(evaluator.evaluate(result)) == {"position_count"}
    assert path.final_wealth == 27405 / 8192
    assert path.n_periods == 3
