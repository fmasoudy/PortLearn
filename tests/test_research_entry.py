"""Contract-first RED floors for the entry-adapter laws.

Targets the absent public adapter ``portlearn.research.run_declared_experiment``
(public entry-point contract).  At authoring time the module does not
exist, so the targeted pytest run must end in the expected contract-first RED
collection failure ``ModuleNotFoundError: No module named 'portlearn.research'``.
Every floor here is written RED against the absent implementation; nothing
creates, mocks, or monkeypatches the target into existence (adapter laws 1-7).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from portlearn.costs import Proportional
from portlearn.experiments import (
    DecisionSchedule,
    ExecutionError,
    Experiment,
    ExperimentConditions,
    ExperimentConfiguration,
    StrategyRegistration,
    select_all,
)
from portlearn.forecasting import DeterminismClass, FitWindow, ForecastProvenance
from portlearn.interfaces import DecisionResult, InformationSet, PortfolioDecision
from portlearn.ledger import ExactFillAccounting
from portlearn.protocols import (
    EvaluationLearning,
    ProtocolStructureError,
    ProtocolVocabularyError,
    StrategyTemporalDeclaration,
    TemporalDesign,
    WindowBasis,
    WindowMode,
    WindowSpec,
    annual,
    resolve_temporal_protocols,
)
from portlearn.protocols import (
    none as absence,
)
from portlearn.research import run_declared_experiment
from portlearn.turnover import two_sided

# --------------------------------------------------------------------------- #
# Synthetic constants: deterministic, no filesystem/network/clock reads.      #
# --------------------------------------------------------------------------- #

MEL = ZoneInfo("Australia/Melbourne")
SPAN = "2015-01-01..2020-12-31"
HORIZON = timedelta(days=91)
ESTIMATION_WINDOW = timedelta(days=365)
STRATEGY_IDS = ("EW", "MV")

DECISION_DATES = (
    datetime(2018, 1, 2, tzinfo=UTC),
    datetime(2018, 2, 1, tzinfo=UTC),
    datetime(2018, 3, 1, tzinfo=UTC),
)
UNIVERSES: Mapping[datetime, tuple[str, ...]] = {
    instant: ("A", "B") for instant in DECISION_DATES
}

RECORD_INSTANTS: Mapping[str, Sequence[datetime]] = {
    "EW": (
        datetime(2013, 1, 7, tzinfo=MEL),
        datetime(2016, 1, 4, tzinfo=MEL),
        datetime(2019, 1, 7, tzinfo=MEL),
    ),
    "MV": (
        datetime(2013, 1, 7, tzinfo=MEL),
        datetime(2016, 1, 4, tzinfo=MEL),
        datetime(2019, 1, 7, tzinfo=MEL),
    ),
}


def _absent_arm() -> Any:
    """The no-fit arm: every temporal field declares its absence."""
    return StrategyTemporalDeclaration(
        estimation=absence,
        selection=absence,
        refit=absence,
        preprocessing=absence,
    )


def _fitful_arm() -> Any:
    """A focused fitful arm: a rolling estimation window plus annual refits."""
    return StrategyTemporalDeclaration(
        estimation=WindowSpec(
            basis=WindowBasis.CALENDAR,
            length=ESTIMATION_WINDOW,
            mode=WindowMode.ROLLING,
        ),
        selection=absence,
        refit=annual,
        preprocessing=absence,
    )


def _design(strategies: Sequence[tuple[str, Any]]) -> TemporalDesign:
    return TemporalDesign(
        evaluation_span=SPAN,
        evaluation_learning=EvaluationLearning.WALK_FORWARD,
        estimand="mean_return",
        target_horizon=HORIZON,
        timezone=MEL,
        strategies=list(strategies),
    )


def _nofit_design() -> TemporalDesign:
    return _design([("EW", _absent_arm()), ("MV", _absent_arm())])


def _image(cutoff: datetime) -> ForecastProvenance:
    return ForecastProvenance(
        model="synthetic-model",
        config_hash="0" * 64,
        cutoff=cutoff,
        seed=0,
        determinism=DeterminismClass.SEED_REPRODUCIBLE,
        fit_window=FitWindow(window_start=cutoff - ESTIMATION_WINDOW),
        produced_by="F3F4-contract-RED",
    )


# --------------------------------------------------------------------------- #
# Synthetic decide-spy and frozen public runtime wiring.                      #
# --------------------------------------------------------------------------- #


class _DecideSpy:
    """A real strategy object counting decide calls.

    For every invocation the floors reject, the counter must remain zero:
    protocol resolution and provenance validation precede the first decision
    (adapter laws 2-3 and 6).
    """

    def __init__(self) -> None:
        self.calls: int = 0

    def decide(self, context: Any) -> DecisionResult:
        self.calls += 1
        per_weight = 1.0 / len(context.universe)
        return DecisionResult(
            decision=PortfolioDecision(
                decision_time=context.decision_time,
                execution_time=context.decision_time,
                target_weights={asset: per_weight for asset in context.universe},
            )
        )


class _Info:
    def __call__(self, instant: datetime) -> InformationSet:
        return InformationSet(items=(), as_of=instant)


class _Outcome:
    def __init__(self, universes: Mapping[datetime, Sequence[str]]) -> None:
        self._universes = universes

    def __call__(self, instant: datetime) -> Mapping[str, float]:
        return {asset: 1.0 for asset in self._universes[instant]}


class _Rebalance:
    def should_rebalance(self, decision_time: Any, last: Any) -> bool:
        return True


def _spies() -> Mapping[str, _DecideSpy]:
    return {name: _DecideSpy() for name in STRATEGY_IDS}


def _registrations(spies: Mapping[str, _DecideSpy]) -> tuple[Any, ...]:
    return tuple(StrategyRegistration(name, spies[name]) for name in STRATEGY_IDS)


def _configuration() -> ExperimentConfiguration:
    return ExperimentConfiguration(
        strategy_ids=STRATEGY_IDS,
        target="EW",
        comparators=("MV",),
        conditions={"periods_per_year": 12},
    )


def _base_kwargs(
    spies: Mapping[str, _DecideSpy],
    temporal: Any,
    fit_provenances: Any,
) -> dict[str, Any]:
    """The complete lawful invocation of the absent public adapter."""
    return {
        "configuration": _configuration(),
        "registrations": _registrations(spies),
        "schedule": DecisionSchedule(
            decision_dates=list(DECISION_DATES), universes=dict(UNIVERSES)
        ),
        "conditions": ExperimentConditions(
            information_provider=_Info(),
            outcome_provider=_Outcome(UNIVERSES),
            rebalance_policy=_Rebalance(),
            accounting_engine=ExactFillAccounting(),
            cost_model=Proportional(rate=0.001, turnover=two_sided),
            initial_weights={"A": 0.5, "B": 0.5},
            initial_wealth=100.0,
            initial_last_rebalance_time=datetime(2017, 12, 1, tzinfo=UTC),
        ),
        "temporal": temporal,
        "record_instants_by_strategy": {
            name: list(instants) for name, instants in RECORD_INSTANTS.items()
        },
        "fit_provenances_by_strategy": fit_provenances,
    }


def _invoke(
    spies: Mapping[str, _DecideSpy],
    temporal: Any,
    fit_provenances: Any,
    **overrides: Any,
) -> Any:
    """Call the absent adapter through the frozen public runtime wiring."""
    kwargs = _base_kwargs(spies, temporal, fit_provenances)
    kwargs.update(overrides)
    return run_declared_experiment(**kwargs)


def _invoke_omitting(
    spies: Mapping[str, _DecideSpy],
    temporal: Any,
    fit_provenances: Any,
    *,
    omitted: tuple[str, ...],
) -> Any:
    """Call the adapter with the named required keyword(s) entirely absent."""
    kwargs = _base_kwargs(spies, temporal, fit_provenances)
    for name in omitted:
        del kwargs[name]
    return run_declared_experiment(**kwargs)


def _decisions(spies: Mapping[str, _DecideSpy]) -> int:
    return sum(spy.calls for spy in spies.values())


# --------------------------------------------------------------------------- #
# F3 — missing temporal declaration or arm provenance rejects pre-decision.   #
# --------------------------------------------------------------------------- #


def test_missing_temporal_declaration_or_arm_provenance_rejects_before_any_decision() -> (
    None
):
    """Adapter law 1: ``temporal`` and ``fit_provenances_by_strategy`` are
    required (``None`` rejects ``TypeError``; absence is never inferred and
    never defaulted); the fit-provenance mapping must have exact strategy-name
    coverage, so omitting either arm rejects; malformed or incomplete temporal
    declarations propagate the frozen protocol errors unwrapped; a decide-spy
    asserts zero strategy decisions for every rejected invocation."""
    spies = _spies()

    # (a) temporal=None rejects TypeError before any decision.
    with pytest.raises(TypeError):
        _invoke(
            spies,
            temporal=None,
            fit_provenances={"EW": (), "MV": ()},
        )
    assert _decisions(spies) == 0

    # (b) fit_provenances_by_strategy=None rejects TypeError.
    with pytest.raises(TypeError):
        _invoke(
            spies,
            temporal=_nofit_design(),
            fit_provenances=None,
        )
    assert _decisions(spies) == 0

    # (c) omitting the required keywords entirely is never inferred and never
    # defaulted: the adapter call itself rejects before any decision.
    with pytest.raises(TypeError):
        _invoke_omitting(
            spies,
            temporal=_nofit_design(),
            fit_provenances={"EW": (), "MV": ()},
            omitted=("temporal",),
        )
    assert _decisions(spies) == 0
    with pytest.raises(TypeError):
        _invoke_omitting(
            spies,
            temporal=_nofit_design(),
            fit_provenances={"EW": (), "MV": ()},
            omitted=("fit_provenances_by_strategy",),
        )
    assert _decisions(spies) == 0

    # (d) omitting either arm from the fit-provenance mapping rejects
    # (exact names coverage; omitted and empty are never conflated).
    with pytest.raises(ExecutionError):
        _invoke(
            spies,
            temporal=_nofit_design(),
            fit_provenances={"EW": ()},  # MV arm omitted
        )
    assert _decisions(spies) == 0
    with pytest.raises(ExecutionError):
        _invoke(
            spies,
            temporal=_nofit_design(),
            fit_provenances={"MV": ()},  # EW arm omitted
        )
    assert _decisions(spies) == 0

    # (e) malformed declarations: a non-declaration object, and a design
    # whose declared pairs do not cover the named strategies, both fail in
    # the frozen resolver and must propagate unwrapped (adapter law 2) —
    # never wrapped, defaulted, or silently accepted.
    for malformed in (
        object(),
        _design([("EW", _absent_arm())]),  # MV never declared
    ):
        with pytest.raises((ProtocolStructureError, ProtocolVocabularyError)):
            _invoke(
                spies,
                temporal=malformed,
                fit_provenances={"EW": (), "MV": ()},
            )
        assert _decisions(spies) == 0


# --------------------------------------------------------------------------- #
# F4 — fit provenance must match every resolved event before any decision.    #
# --------------------------------------------------------------------------- #


def test_fit_provenance_must_match_every_resolved_event_before_any_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adapter law 3: on a focused fitful declaration resolved through the
    frozen protocol resolver, zero/missing, duplicate, extra, reordered, and
    cutoff-mismatched ``ForecastProvenance`` sequences reject ``ExecutionError``
    before any decision.  The no-fit two-arm declaration separately proves
    explicitly supplied empty sequences are complete, exercising the frozen
    public runtime ``Experiment.run`` through the adapter."""
    spies = _spies()

    fitful_design = _design([("EW", _absent_arm()), ("MV", _fitful_arm())])
    resolved = resolve_temporal_protocols(
        fitful_design,
        list(STRATEGY_IDS),
        {name: list(instants) for name, instants in RECORD_INSTANTS.items()},
    )
    fit_events = tuple(resolved["MV"].fit_events)
    assert fit_events, "the fitful arm must resolve to canonical fit events"
    canonical = [_image(event.cutoff) for event in fit_events]

    def mutated(kind: str) -> tuple[ForecastProvenance, ...]:
        if kind == "zero":
            return ()
        if kind == "missing":
            return tuple(canonical[:-1])
        if kind == "duplicate":
            duplicated = list(canonical)
            duplicated[-1] = canonical[0]
            return tuple(duplicated)
        if kind == "extra":
            return (
                *canonical,
                _image(fit_events[-1].cutoff + timedelta(days=1)),
            )
        if kind == "reordered":
            swapped = list(canonical)
            swapped[0], swapped[1] = swapped[1], swapped[0]
            return tuple(swapped)
        if kind == "cutoff-mismatch":
            mismatched = list(canonical)
            mismatched[0] = _image(fit_events[0].cutoff + timedelta(days=1))
            return tuple(mismatched)
        raise AssertionError(kind)

    # Every defective image sequence rejects ExecutionError pre-decision.
    for kind in (
        "zero",
        "missing",
        "duplicate",
        "extra",
        "reordered",
        "cutoff-mismatch",
    ):
        with pytest.raises(ExecutionError):
            _invoke(
                spies,
                temporal=fitful_design,
                fit_provenances={"EW": (), "MV": mutated(kind)},
            )
        assert _decisions(spies) == 0

    # Acceptance/control: the no-fit two-arm declaration accepts explicitly
    # supplied empty sequences as complete, and runs through Experiment.run
    # (adapter law 4) — both arms execute every scheduled decision.  The
    # instrumented public runtime proves invocation plus consumption of the
    # public-path result: the adapter must call ``Experiment.run`` exactly
    # once, with the exact named selection object, and must return that
    # call's exact result object — a populated call log whose result is
    # discarded, or a locally reconstructed equivalent result, rejects here.
    selection = select_all(_configuration())
    original_run = Experiment.run
    run_calls: list[tuple[Any, Any, Any]] = []

    def instrumented_run(self: Any, selection: Any) -> Any:
        returned_result = original_run(self, selection)
        run_calls.append((self, selection, returned_result))
        return returned_result

    monkeypatch.setattr(Experiment, "run", instrumented_run)
    results, provenances = _invoke(
        spies,
        temporal=_nofit_design(),
        fit_provenances={"EW": (), "MV": ()},
        selection=selection,
    )
    assert len(run_calls) == 1
    assert run_calls[0][1] is selection
    assert results is run_calls[0][2]
    executed = sorted(record.strategy_id for record in results.records)
    assert executed == ["EW", "MV"]
    nofit_resolved = resolve_temporal_protocols(
        _nofit_design(),
        list(STRATEGY_IDS),
        {name: list(instants) for name, instants in RECORD_INSTANTS.items()},
    )
    for name in STRATEGY_IDS:
        assert spies[name].calls == len(DECISION_DATES)
        arm = provenances[name]
        assert arm.protocol == nofit_resolved[name]
        assert tuple(arm.fit_provenances) == ()
    assert provenances.keys() == {"EW", "MV"}
