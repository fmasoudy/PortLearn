"""Contract tests for ``portlearn.protocols`` — the temporal research protocol.

These tests freeze the approved temporal-protocol contract as behavior
floors: a temporal declaration is required with no default and no
protocol-free run; the declaration vocabularies are closed and typed;
partitions are chronological and role-declared; fixed-origin admits no
in-span learning event while walk-forward admits exactly the declared
ones at lawful cutoffs; every declared fit/refit/selection event is the
unit the frozen forecasting-lifecycle admission gates apply against; the estimand and
target horizon are common anchors carried verbatim into every arm; the
declared fit-event grid is complete and canonically ordered; no
post-decision information may reach pre-decision estimation; selection
consumes only the trailing derived validation span; the experiment runtime
consumes protocol provenance opaquely; rejections are typed fail-closed
errors (the two new protocol errors plus the frozen timing/forecasting
arms, reused — never paralleled); sparse overrides inherit every
unspecified field and cannot move the common anchors; the ordinary and
advanced declaration routes resolve to canonically equal frozen
protocols; and resolution is non-mutating, order-invariant, and exact
about per-strategy record-timeline coverage.

RED-first: the module is absent at this tranche, so every
``portlearn.protocols`` access happens INSIDE the nodes through the
lazy ``_protocols()`` importer — pytest reports the causal absence
(``ModuleNotFoundError: No module named 'portlearn.protocols'``) for
every node instead of aborting collection.  Every node in this file is
a designed causal-RED node while the module is absent; the frozen-seam
names imported at module level (``month_end_instant``,
``instant_key``, ``NaiveTimestampError``) belong to live modules and
only ever support the assertions.

Nodes assert error classes and laws only, never error-message tails.
Floor names are behavior names; the law each node pins is recorded in
its docstring.  Vocabulary tokens (``INITIAL_FIT``, ``REFIT``,
``SELECTION``) are read as module attributes, the spelling the
authority's own executable example uses.

Mutation-tooth map (design intent — which node is the intended killer
for each named tooth; no mutation battery is run in this tranche):

* no-default tooth            -> the missing-choice node (class 1)
* masking tooth               -> the trailing-validation-boundary node
* fixed-origin masking tooth  -> the in-span-event rejection nodes
* one-clock tooth             -> the naive-instant rejection nodes
* provenance tooth            -> the grid completeness/order nodes
* horizon tooth               -> the estimand/horizon anchor nodes
* first-fit tooth             -> the no-back-fill first-decision node
* inheritance tooth           -> the sparse-inheritance node
* sparse-policing tooth       -> the override-key policing nodes
* anchor tooth                -> the non-overridable-anchor nodes
* route-equivalence tooth     -> the canonical-equality node
* provenance-route tooth      -> the no-route-fields node
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import sys
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from portlearn.calendar import month_end_instant
from portlearn.timing import NaiveTimestampError, instant_key

# --------------------------------------------------------------------------- #
# The scenario — the authority's own fully-specified illustration.             #
# --------------------------------------------------------------------------- #
# Melbourne zone; evaluation span 2019-01-01..2024-12-31; annual grid anchored
# at the span start, so the initial cutoff c0 is the December-2018 month-end
# instant (the latest annual grid instant strictly before the span start) and
# a five-calendar-year estimation window opens at the December-2013 month-end
# instant.  All expected instants are derived here through the frozen calendar
# seam, never restated as magic wall-clock literals.

MEL = ZoneInfo("Australia/Melbourne")

EVALUATION_SPAN = "2019-01-01..2024-12-31"
E_START = datetime(2019, 1, 1, tzinfo=MEL)
E_END = datetime(2024, 12, 31, 23, 59, 59, 999999, tzinfo=MEL)
C_ZERO = month_end_instant(2018, 12, MEL)
T_START = month_end_instant(2013, 12, MEL)
VALIDATION_START = month_end_instant(2016, 12, MEL)

ESTIMAND = "oos_sharpe"
HORIZON = timedelta(days=21)

# The annual grid instants strictly inside the evaluation span (half-open at
# the span end: no claim is made about a span-end-coincident grid instant).
ANNUAL_IN_SPAN = tuple(month_end_instant(y, 12, MEL) for y in range(2019, 2024))

# Admitted per-strategy record timelines: timezone-aware month-end instants,
# strictly increasing on the single clock.
ADMITTED = tuple(
    month_end_instant(y, m, MEL) for y in range(2013, 2025) for m in range(1, 13)
)
ADMITTED_LATE = tuple(
    month_end_instant(y, m, MEL) for y in range(2016, 2025) for m in range(1, 13)
)
ADMITTED_SPARSE = tuple(
    month_end_instant(y, m, MEL) for y in range(2013, 2025) for m in (1, 3, 5, 7, 9, 11)
)

# The pinned public surface: the frozen value objects, the ordinary authoring
# vocabulary, the advanced constructors, the one resolution step, and exactly
# the two new typed errors — nothing else may be public.
PINNED_PUBLIC_NAMES = frozenset(
    {
        # Canonical value objects.
        "CalendarSpan",
        "ExperimentTemporalDeclaration",
        "FitEvent",
        "HistoryPolicy",
        "ObservationRefitCadence",
        "ProtocolProvenance",
        "RefitCadence",
        "StrategyProtocolOverride",
        "StrategyTemporalDeclaration",
        "TemporalDesign",
        "TemporalPartition",
        "TemporalResearchProtocol",
        "WindowSpec",
        # Frozen enums.
        "EvaluationLearning",
        "HistoryBasis",
        "HistoryMode",
        "PartitionRole",
        "PreprocessingPolicy",
        "SelectionPolicy",
        "WindowBasis",
        "WindowMode",
        # Advanced constructors.
        "expanding_calendar_history",
        "expanding_observation_history",
        "month_end_refits",
        "observation_refits",
        "quarter_end_refits",
        "rolling_calendar_history",
        "rolling_observation_history",
        # Ordinary authoring vocabulary.
        "annual",
        "at_each_fit",
        "monthly",
        "none",
        "quarterly",
        "window",
        # The one resolution step.
        "resolve_temporal_protocols",
        # The two new typed errors (the frozen arms are reused, never re-defined).
        "ProtocolStructureError",
        "ProtocolVocabularyError",
    }
)

# No protocol-provenance field may record how the protocol was authored.
_ROUTE_TOKENS = (
    "route",
    "authoring",
    "spelling",
    "concise",
    "ordinary",
    "advanced",
    "declaration",
)


# --------------------------------------------------------------------------- #
# The module under test — imported inside every node so collection succeeds  #
# while the module is absent (the causal RED signal).                         #
# --------------------------------------------------------------------------- #


@functools.cache
def _protocols() -> Any:
    """Import ``portlearn.protocols`` on demand (RED: ModuleNotFoundError)."""
    return importlib.import_module("portlearn.protocols")


# --------------------------------------------------------------------------- #
# Generation isolation (harness plumbing — no assertion or floor changes).    #
# --------------------------------------------------------------------------- #
#
# ``tests/test_package_contract.py`` runs a registry-purge probe that deletes
# every ``portlearn``/``portlearn.*`` entry from ``sys.modules`` and re-imports
# the root package fresh.  When this module is collected before that probe
# runs, the frozen-seam names bound at import time above belong to the
# collection-time generation, while the lazily imported module under test
# binds a post-purge generation — and error-class identity checks would then
# compare classes from two different generations.  The fixture below pins the
# collection-time generation back for the duration of this module's nodes and
# restores the ambient registry exactly afterwards.  Mirrors the
# ``tests/test_results.py`` / ``tests/test_experiments.py`` precedent.


def _portlearn_generation() -> dict[str, Any]:
    """Snapshot the registered ``portlearn`` root/submodule generation."""
    return {
        name: module
        for name, module in sys.modules.items()
        if name == "portlearn" or name.startswith("portlearn.")
    }


_COLLECTION_GENERATION = _portlearn_generation()


@pytest.fixture(autouse=True, scope="module")
def _one_coherent_module_generation():
    """Run this module's nodes against one coherent module generation."""
    ambient = _portlearn_generation()
    sys.modules.update(_COLLECTION_GENERATION)
    try:
        yield
    finally:
        for name in sorted(_portlearn_generation()):
            del sys.modules[name]
        sys.modules.update(ambient)


# --------------------------------------------------------------------------- #
# Declaration builders — pure assembly over the module under test; every     #
# builder takes the module ``p`` so nothing protocol-side binds at collection.#
# --------------------------------------------------------------------------- #


def _standard_partitions(p: Any) -> tuple[Any, ...]:
    """The TRAIN [t_start, c0] + TEST [E_start, E_end] partition pair."""
    return (
        p.TemporalPartition(role=p.PartitionRole.TRAIN, start=T_START, end=C_ZERO),
        p.TemporalPartition(role=p.PartitionRole.TEST, start=E_START, end=E_END),
    )


def _common_protocol(
    p: Any,
    *,
    evaluation_learning: Any,
    partitions: Any = None,
    fit_events: Any = (),
    history_policy: Any = None,
    selection_history: Any = None,
    preprocessing_policy: Any = None,
    selection_policy: Any = None,
    estimand: str = ESTIMAND,
    target_horizon: timedelta = HORIZON,
) -> Any:
    """A canonical common protocol over the standard scenario."""
    return p.TemporalResearchProtocol(
        evaluation_learning=evaluation_learning,
        partitions=partitions if partitions is not None else _standard_partitions(p),
        target_horizon=target_horizon,
        estimand=estimand,
        history_policy=history_policy,
        selection_history=selection_history,
        fit_events=fit_events,
        preprocessing_policy=preprocessing_policy,
        selection_policy=selection_policy,
    )


def _initial_fit_event(p: Any) -> Any:
    """The one declared initial fit at c0 on the TRAIN role."""
    return p.FitEvent(kind=p.INITIAL_FIT, cutoff=C_ZERO, role=p.PartitionRole.TRAIN)


def _mv_arm(p: Any) -> Any:
    """The rolling five-calendar-year estimation arm; everything else absent."""
    return p.StrategyTemporalDeclaration(
        estimation=p.window(years=5, mode=p.WindowMode.ROLLING),
        selection=p.none,
        refit=p.none,
        preprocessing=p.none,
    )


def _mv_annual_arm(p: Any) -> Any:
    """The rolling five-year arm with an annual refit cadence declared."""
    return p.StrategyTemporalDeclaration(
        estimation=p.window(years=5, mode=p.WindowMode.ROLLING),
        selection=p.none,
        refit=p.annual,
        preprocessing=p.none,
    )


def _selection_arm(p: Any) -> Any:
    """Three-year estimation, two-year selection, annual refits, preprocessing."""
    return p.StrategyTemporalDeclaration(
        estimation=p.window(years=3, mode=p.WindowMode.ROLLING),
        selection=p.window(years=2, mode=p.WindowMode.ROLLING),
        refit=p.annual,
        preprocessing=p.at_each_fit,
    )


def _ew_arm(p: Any) -> Any:
    """The no-fit arm: every strategy field declares its absence."""
    return p.StrategyTemporalDeclaration(
        estimation=p.none,
        selection=p.none,
        refit=p.none,
        preprocessing=p.none,
    )


def _design(
    p: Any,
    *,
    evaluation_learning: Any,
    strategies: Any,
    estimand: str = ESTIMAND,
    target_horizon: timedelta = HORIZON,
) -> Any:
    """An ordinary design over the standard evaluation span and zone."""
    return p.TemporalDesign(
        evaluation_span=EVALUATION_SPAN,
        evaluation_learning=evaluation_learning,
        estimand=estimand,
        target_horizon=target_horizon,
        timezone=MEL,
        strategies=strategies,
    )


def _resolve(p: Any, declaration: Any, names: Any, timeline: Any) -> Any:
    """Run the one resolution step over the given admitted timelines."""
    return p.resolve_temporal_protocols(declaration, list(names), dict(timeline))


def _keys(instants: Any) -> list[Any]:
    """Normalized single-clock identities for ordering and set comparisons."""
    return [instant_key(value) for value in instants]


# --------------------------------------------------------------------------- #
# Floor: a temporal declaration is required — no default, no protocol-free run#
# --------------------------------------------------------------------------- #


class TestRequiredDeclaration:
    """A temporal declaration is mandatory — absence rejects before execution.

    Pins the no-default law (the missing evaluation-learning choice rejects
    at construction, never a default fill) and the no-protocol-free-run law
    (the resolution step rejects a non-declaration before any target is
    produced).  Two designed causal-RED nodes.
    """

    def test_missing_evaluation_learning_choice_rejects_at_construction(self) -> None:
        p = _protocols()
        with pytest.raises(Exception):  # noqa: B017 — causal RED node
            p.TemporalDesign(
                evaluation_span=EVALUATION_SPAN,
                estimand=ESTIMAND,
                target_horizon=HORIZON,
                timezone=MEL,
                strategies=[("MV", _mv_arm(p))],
            )

    def test_resolution_rejects_a_non_declaration_object(self) -> None:
        p = _protocols()
        with pytest.raises(p.ProtocolStructureError):
            _resolve(p, object(), ["MV"], {"MV": ADMITTED})


class TestUnknownVocabulary:
    """Every enum-bearing field is closed — unknown tokens reject typed.

    Pins the closed-vocabulary law: an unknown evaluation-learning member
    and an unknown partition-role token each reject with the protocol
    vocabulary error, never a coerced or nearest-neighbor value.  Two
    designed causal-RED nodes.
    """

    def test_unknown_evaluation_learning_token_rejects(self) -> None:
        p = _protocols()
        with pytest.raises(p.ProtocolVocabularyError):
            _design(
                p,
                evaluation_learning="sometimes",
                strategies=[("MV", _mv_arm(p))],
            )

    def test_unknown_partition_role_token_rejects(self) -> None:
        p = _protocols()
        with pytest.raises(p.ProtocolVocabularyError):
            p.TemporalPartition(
                role="production",  # type: ignore[arg-type]
                start=T_START,
                end=C_ZERO,
            )


class TestIncompatibleWindow:
    """Declared windows carry their own basis and must be internally coherent.

    Pins the history-basis law: an observation-count basis paired with a
    calendar-duration parameter rejects at construction, and the ordinary
    window constructor rejects two basis lengths supplied together.  Two
    designed causal-RED nodes.
    """

    def test_observation_basis_with_calendar_parameter_rejects(self) -> None:
        p = _protocols()
        with pytest.raises(p.ProtocolStructureError):
            p.HistoryPolicy(
                mode=p.HistoryMode.ROLLING,
                basis=p.HistoryBasis.OBSERVATION_COUNT,
                minimum_history=p.CalendarSpan(years=5),
            )

    def test_window_constructor_rejects_two_basis_lengths(self) -> None:
        p = _protocols()
        with pytest.raises(p.ProtocolStructureError):
            p.window(years=5, observations=252, mode=p.WindowMode.ROLLING)


class TestChronologicalPartitions:
    """Partitions are ordered, non-overlapping, role-declared intervals.

    Pins the chronological-partition law: a TEST span placed before a TRAIN
    span rejects at construction, and a declared fit-event cutoff outside
    the union of the declared partition spans rejects at construction.  Two
    designed causal-RED nodes.
    """

    def test_test_role_before_train_role_rejects_at_construction(self) -> None:
        p = _protocols()
        with pytest.raises(p.ProtocolStructureError):
            p.TemporalResearchProtocol(
                evaluation_learning=p.EvaluationLearning.WALK_FORWARD,
                partitions=(
                    p.TemporalPartition(
                        role=p.PartitionRole.TEST, start=E_START, end=E_END
                    ),
                    p.TemporalPartition(
                        role=p.PartitionRole.TRAIN, start=T_START, end=C_ZERO
                    ),
                ),
                target_horizon=HORIZON,
                estimand=ESTIMAND,
                history_policy=None,
                selection_history=None,
                fit_events=(),
                preprocessing_policy=p.PreprocessingPolicy.NONE,
                selection_policy=p.SelectionPolicy.NONE,
            )

    def test_fit_event_cutoff_outside_declared_partitions_rejects(self) -> None:
        p = _protocols()
        with pytest.raises(p.ProtocolStructureError):
            p.TemporalResearchProtocol(
                evaluation_learning=p.EvaluationLearning.WALK_FORWARD,
                partitions=_standard_partitions(p),
                target_horizon=HORIZON,
                estimand=ESTIMAND,
                history_policy=p.rolling_calendar_history(timedelta(days=365)),
                selection_history=None,
                fit_events=(
                    p.FitEvent(
                        kind=p.INITIAL_FIT,
                        cutoff=month_end_instant(2025, 6, MEL),
                        role=p.PartitionRole.TRAIN,
                    ),
                ),
                preprocessing_policy=p.PreprocessingPolicy.NONE,
                selection_policy=p.SelectionPolicy.NONE,
            )


# --------------------------------------------------------------------------- #
# Floor: fixed-origin admits no in-span learning event.                       #
# --------------------------------------------------------------------------- #


class TestFixedOriginInSpan:
    """A fixed-origin protocol rejects any learning event inside the span.

    Pins the fixed-origin law: an ordinary annual-refit declaration
    rejects under fixed-origin, and the advanced form rejects an in-span
    fit event likewise.  Two designed causal-RED nodes.
    """

    def test_ordinary_in_span_refit_rejects_under_fixed_origin(self) -> None:
        p = _protocols()
        with pytest.raises(p.ProtocolStructureError):
            _resolve(
                p,
                _design(
                    p,
                    evaluation_learning=p.FIXED_ORIGIN,
                    strategies=[("MV", _mv_annual_arm(p))],
                ),
                ["MV"],
                {"MV": ADMITTED},
            )

    def test_advanced_in_span_event_rejects_under_fixed_origin(self) -> None:
        p = _protocols()
        with pytest.raises(p.ProtocolStructureError):
            _resolve(
                p,
                p.ExperimentTemporalDeclaration(
                    common=_common_protocol(
                        p,
                        evaluation_learning=p.FIXED_ORIGIN,
                        fit_events=(
                            p.FitEvent(
                                kind=p.REFIT,
                                cutoff=ANNUAL_IN_SPAN[0],
                                role=p.PartitionRole.TRAIN,
                            ),
                        ),
                    ),
                    strategy_overrides=[],
                ),
                ["MV"],
                {"MV": ADMITTED},
            )


# --------------------------------------------------------------------------- #
# Floor: walk-forward executes exactly the declared in-span events.           #
# --------------------------------------------------------------------------- #


class TestWalkForwardUndeclaredEvent:
    """A walk-forward protocol executes exactly its declared events.

    Pins the walk-forward lawfulness law: the materialized refit grid is
    exactly the declared annual cadence inside the evaluation span (no
    undeclared in-span event exists), and an arm with no refit declared
    carries no in-span learning event.  Two designed causal-RED nodes.
    """

    def test_declared_annual_refits_are_exactly_the_in_span_events(self) -> None:
        p = _protocols()
        resolved = _resolve(
            p,
            _design(
                p,
                evaluation_learning=p.WALK_FORWARD,
                strategies=[("MV", _mv_annual_arm(p))],
            ),
            ["MV"],
            {"MV": ADMITTED},
        )
        refits = [event for event in resolved["MV"].fit_events if event.kind == p.REFIT]
        assert [instant_key(event.cutoff) for event in refits] == _keys(ANNUAL_IN_SPAN)

    def test_no_refit_arm_carries_no_in_span_learning_event(self) -> None:
        p = _protocols()
        resolved = _resolve(
            p,
            _design(
                p,
                evaluation_learning=p.WALK_FORWARD,
                strategies=[("MV", _mv_arm(p))],
            ),
            ["MV"],
            {"MV": ADMITTED},
        )
        in_span = [
            event
            for event in resolved["MV"].fit_events
            if instant_key(E_START) <= instant_key(event.cutoff) < instant_key(E_END)
        ]
        assert in_span == []


# --------------------------------------------------------------------------- #
# Floor: walk-forward label availability at the selection cutoff.             #
# --------------------------------------------------------------------------- #
# The selection arm declares a two-calendar-year selection window trailing
# its initial cutoff ([December 2016, December 2018] here) and a three-year
# estimation window trailing that.  The gap timeline below covers the
# estimation span fully but contains no record inside the selection span, so
# the selection event at the initial cutoff has no labels realizable at that
# cutoff — the label-availability rejection.  It is distinct from the
# first-fit floor's timeline, which starves the estimation window itself.

_SELECTION_GAP_TIMELINE = tuple(
    month_end_instant(year, month, MEL)
    for year in range(2013, 2017)
    for month in range(1, 13)
)


class TestWalkForwardLabelAvailability:
    """Selection labels must be realizable at the declared selection cutoff.

    Pins the walk-forward label-availability law: a selection event whose
    consumed validation-span labels are absent from the strategy's admitted
    timeline rejects at resolution (no masked-but-missing labels), and a
    lawful selection arm resolves with its selection events on the declared
    annual grid.  Two designed causal-RED nodes.
    """

    def test_selection_labels_absent_from_timeline_reject(self) -> None:
        p = _protocols()
        with pytest.raises(p.ProtocolStructureError):
            _resolve(
                p,
                _design(
                    p,
                    evaluation_learning=p.WALK_FORWARD,
                    strategies=[("LSTM", _selection_arm(p))],
                ),
                ["LSTM"],
                {"LSTM": _SELECTION_GAP_TIMELINE},
            )

    def test_lawful_selection_events_resolve_on_the_declared_grid(self) -> None:
        p = _protocols()
        resolved = _resolve(
            p,
            _design(
                p,
                evaluation_learning=p.WALK_FORWARD,
                strategies=[("LSTM", _selection_arm(p))],
            ),
            ["LSTM"],
            {"LSTM": ADMITTED},
        )
        selection_events = [
            event for event in resolved["LSTM"].fit_events if event.kind == p.SELECTION
        ]
        assert len(selection_events) >= 1
        for event in selection_events:
            assert instant_key(event.cutoff) in _keys([C_ZERO, *ANNUAL_IN_SPAN])


# --------------------------------------------------------------------------- #
# Floor: the forecasting-lifecycle gates apply at every declared fit/refit/selection event.    #
# --------------------------------------------------------------------------- #
# Each node derives a declared event cutoff from the resolved protocol and
# applies the frozen forecasting-lifecycle gate AT that event — the composition the law
# requires the protocol to perform, never to re-implement.  The nodes are
# causal-RED while the module is absent (the resolution fails first) and
# carry the frozen-seam teeth afterwards.


class TestM17GateApplication:
    """The frozen fitting-boundary gates bind at every declared event.

    Pins the forecasting-lifecycle-application law at the three event kinds: the admission
    gate at a fit event, the one-clock gate at a declared cutoff, and the
    selection-label gate at a selection event.  Three designed causal-RED
    nodes.
    """

    def test_admission_gate_binds_at_the_initial_fit_event(self) -> None:
        from portlearn.forecasting import require_fit_inputs_admitted
        from portlearn.timing import FutureInformationError

        p = _protocols()
        resolved = _resolve(
            p,
            _design(
                p,
                evaluation_learning=p.WALK_FORWARD,
                strategies=[("MV", _mv_arm(p))],
            ),
            ["MV"],
            {"MV": ADMITTED},
        )
        initial = next(
            event for event in resolved["MV"].fit_events if event.kind == p.INITIAL_FIT
        )

        class _Record:
            series_id = "AUD_CPI"

            def __init__(self, available_time: Any) -> None:
                self.available_time = available_time

        class _FitSet:
            def __init__(self, as_of: Any) -> None:
                self.as_of = as_of

        # A record available after the event's declared cutoff is not
        # admissible for the fit: the frozen gate rejects, fail-closed.
        with pytest.raises(FutureInformationError):
            require_fit_inputs_admitted(
                [_Record(month_end_instant(2019, 3, MEL))],
                _FitSet(initial.cutoff),
            )

    def test_one_clock_gate_rejects_a_divergent_cutoff_assertion(self) -> None:
        from portlearn.forecasting import (
            ProvenanceStructureError,
            require_cutoff_from_fit_set,
        )

        p = _protocols()
        resolved = _resolve(
            p,
            _design(
                p,
                evaluation_learning=p.WALK_FORWARD,
                strategies=[("MV", _mv_arm(p))],
            ),
            ["MV"],
            {"MV": ADMITTED},
        )
        initial = next(
            event for event in resolved["MV"].fit_events if event.kind == p.INITIAL_FIT
        )

        class _FitSet:
            def __init__(self, as_of: Any) -> None:
                self.as_of = as_of

        # Any cutoff assertion diverging from the fit set's own as_of is a
        # second clock and rejects — there is no lawful divergence.
        with pytest.raises(ProvenanceStructureError):
            require_cutoff_from_fit_set(
                month_end_instant(2019, 1, MEL), _FitSet(initial.cutoff)
            )

    def test_selection_label_gate_binds_at_a_selection_event(self) -> None:
        from portlearn.forecasting import require_selection_labels_available
        from portlearn.timing import FutureInformationError

        p = _protocols()
        resolved = _resolve(
            p,
            _design(
                p,
                evaluation_learning=p.WALK_FORWARD,
                strategies=[("LSTM", _selection_arm(p))],
            ),
            ["LSTM"],
            {"LSTM": ADMITTED},
        )
        selection = next(
            event for event in resolved["LSTM"].fit_events if event.kind == p.SELECTION
        )

        class _Label:
            series_id = "AUD_CPI"

            def __init__(self, available_time: Any) -> None:
                self.available_time = available_time

        # A selection label available after the selection cutoff is future
        # information at the tuning boundary: the frozen gate rejects.
        with pytest.raises(FutureInformationError):
            require_selection_labels_available(
                [_Label(month_end_instant(2019, 3, MEL))],
                selection.cutoff,
            )


# --------------------------------------------------------------------------- #
# Floor: one clock — every protocol instant lives on the declared zone.       #
# --------------------------------------------------------------------------- #
# Law 1: the experiment declares exactly one clock.  A record timeline that
# carries a naive instant rejects at resolution through the frozen timing
# arm, and every resolved fit-event cutoff is *expressed* on the declared
# zone's own wall clock — the MT4 declarative half (single-zone grid
# anchoring); MT4's execution-stage half is out of scope here.


class TestOneClockLaw:
    """Law 1 pins one declared clock for the whole protocol.

    Two designed causal-RED nodes: the naive-instant rejection (the
    one-clock tooth) and the zone-expressed cutoff grid.
    """

    def test_naive_record_instant_rejects_at_resolution(self) -> None:
        p = _protocols()
        mixed = (
            month_end_instant(2018, 1, MEL).replace(tzinfo=None),
            month_end_instant(2018, 2, MEL),
        )
        with pytest.raises(NaiveTimestampError):
            _resolve(
                p,
                _design(
                    p,
                    evaluation_learning=p.WALK_FORWARD,
                    strategies=[("MV", _mv_arm(p))],
                ),
                ["MV"],
                {"MV": mixed},
            )

    def test_fit_event_cutoffs_are_expressed_on_the_declared_zone_clock(self) -> None:
        p = _protocols()
        resolved = _resolve(
            p,
            _design(
                p,
                evaluation_learning=p.WALK_FORWARD,
                strategies=[("MV", _mv_annual_arm(p))],
            ),
            ["MV"],
            {"MV": ADMITTED},
        )
        cutoffs = [event.cutoff for event in resolved["MV"].fit_events]
        assert cutoffs
        keys = _keys(cutoffs)
        assert keys == sorted(keys)
        assert len(set(keys)) == len(keys)
        assert keys[0] == instant_key(C_ZERO)
        # The declared annual grid strictly inside the span is fully
        # materialized; no claim is made about an E_end-coincident refit.
        assert set(keys) >= set(_keys((C_ZERO,) + ANNUAL_IN_SPAN))
        assert keys[-1] <= instant_key(E_END)
        for cutoff in cutoffs:
            assert cutoff.tzinfo is not None
            assert cutoff.utcoffset() == MEL.utcoffset(cutoff.replace(tzinfo=None))


# --------------------------------------------------------------------------- #
# Floor: horizon/estimand pinning — the two forecast anchors are common.      #
# --------------------------------------------------------------------------- #
# Law 9: estimand and target horizon are common anchors of the experiment.
# The protocol carries them verbatim; a strategy override has NO anchor
# fields, so they cannot be moved per-strategy.  The horizon floor is a
# protocol-vs-forecast TOKEN mismatch (the estimand names the target the
# horizon measures), never an invented global vocabulary.  MT6.


class TestHorizonEstimandPinning:
    """Law 9 pins the estimand and horizon as unmoving common anchors.

    Two designed causal-RED nodes: verbatim carriage into every resolved
    arm, and no per-strategy anchor fields on the override value object.
    """

    def test_common_anchors_carried_verbatim_into_every_arm(self) -> None:
        p = _protocols()
        resolved = _resolve(
            p,
            _design(
                p,
                evaluation_learning=p.WALK_FORWARD,
                strategies=[
                    ("MV", _mv_arm(p)),
                    ("LSTM", _selection_arm(p)),
                ],
            ),
            ["MV", "LSTM"],
            {"MV": ADMITTED, "LSTM": ADMITTED_LATE},
        )
        for name in ("MV", "LSTM"):
            assert resolved[name].estimand == ESTIMAND
            assert resolved[name].target_horizon == HORIZON

    def test_override_has_no_anchor_fields_to_move(self) -> None:
        p = _protocols()
        field_names = {
            field.name for field in dataclasses.fields(p.StrategyProtocolOverride)
        }
        assert field_names == {
            "history_policy",
            "selection_history",
            "fit_events",
            "preprocessing_policy",
            "selection_policy",
        }
        with pytest.raises(TypeError):
            p.StrategyProtocolOverride(estimand=ESTIMAND)


# --------------------------------------------------------------------------- #
# Floor: provenance completeness — one canonical image per declared event.    #
# --------------------------------------------------------------------------- #
# Law 10: every executed fit event yields exactly one ForecastProvenance; a
# ProtocolProvenance with a missing, duplicated, or unordered fit-event
# image rejects.  The full MT5 kill is execution-stage (image assembly);
# the declarative halves pinned here are the image SOURCE — the resolved
# grid is a complete, deduplicated, canonically ordered FitEvent sequence,
# so one-to-one imaging is satisfiable — and the image CARRIER, read by
# class-level introspection only (the constructor is not §4-pinned and is
# never constructed; recorded adjudication).


class TestProtocolProvenanceCompleteness:
    """Law 10 pins a complete, ordered, one-to-one provenance image.

    Two designed causal-RED nodes: the resolved grid as a complete
    canonical event sequence, and the provenance class's ordered
    multiplicity-preserving image carrier.
    """

    def test_resolved_grid_is_a_complete_canonical_event_sequence(self) -> None:
        p = _protocols()
        resolved = _resolve(
            p,
            _design(
                p,
                evaluation_learning=p.WALK_FORWARD,
                strategies=[("MV", _mv_annual_arm(p))],
            ),
            ["MV"],
            {"MV": ADMITTED},
        )
        events = resolved["MV"].fit_events
        # The declared cadence is materialized at resolution into concrete
        # events — no cadence object survives into the resolved protocol.
        assert not isinstance(events, p.ObservationRefitCadence)
        assert events
        assert all(isinstance(event, p.FitEvent) for event in events)
        keys = _keys([event.cutoff for event in events])
        assert keys == sorted(keys)
        assert len(set(keys)) == len(keys)
        # Completeness against dropping: one event for every declared
        # cutoff (c0 plus every in-span annual instant; no E_end claim).
        assert len(events) >= 1 + len(ANNUAL_IN_SPAN)
        for event in events:
            assert event.kind is not None
            assert event.cutoff is not None
            assert event.role is not None

    def test_provenance_declares_an_ordered_image_carrier(self) -> None:
        p = _protocols()

        def _ordered(annotation: Any) -> bool:
            text = str(annotation).lower()
            return any(token in text for token in ("tuple", "list", "sequence"))

        # Class-level introspection only — never construction.
        assert p.ProtocolProvenance.__dataclass_params__.frozen
        declared = dataclasses.fields(p.ProtocolProvenance)
        assert declared
        # Law 10 rejects a missing, DUPLICATED, or UNORDERED image; only
        # an ordered multiplicity-preserving carrier can represent one.
        assert any(_ordered(field.type) for field in declared)


# --------------------------------------------------------------------------- #
# Floor: the contamination backstop — no post-decision information anywhere.  #
# --------------------------------------------------------------------------- #
# Law 10a is the backstop over Laws 7-8: no future outcome, post-decision
# observation, evaluation-span datum, or transformation of them may enter
# any pre-decision fit, refit, selection, preprocessing fit, or tuning
# decision — even where a family's internal mechanics would make the
# contamination convenient.  Every node composes the frozen forecasting-lifecycle gates at a
# resolved declared event; MT2's masking shape is the first node: a
# surface-correct mask does not launder timing.


class TestContaminationBackstop:
    """Law 10a binds on availability timing, never on the surface.

    Three designed causal-RED nodes: the MT2 masking shape at a selection
    event, an evaluation-span record at an in-span refit, and the one-clock
    backstop at that same refit.
    """

    def test_surface_masked_post_cutoff_selection_label_still_rejects(self) -> None:
        from portlearn.forecasting import require_selection_labels_available
        from portlearn.timing import FutureInformationError

        p = _protocols()
        resolved = _resolve(
            p,
            _design(
                p,
                evaluation_learning=p.WALK_FORWARD,
                strategies=[("LSTM", _selection_arm(p))],
            ),
            ["LSTM"],
            {"LSTM": ADMITTED},
        )
        selection = next(
            event for event in resolved["LSTM"].fit_events if event.kind == p.SELECTION
        )

        class _MaskedLabel:
            series_id = "AUD_CPI"
            masked = True  # the surface-correct half: content masked

            def __init__(self, available_time: Any) -> None:
                self.available_time = available_time

        # MT2: the label surface declares itself masked, yet its
        # availability is post-cutoff — the mask cannot launder the
        # information into the tuning decision.
        with pytest.raises(FutureInformationError):
            require_selection_labels_available(
                [_MaskedLabel(month_end_instant(2019, 3, MEL))],
                selection.cutoff,
            )

    def test_evaluation_span_record_rejects_at_an_in_span_refit(self) -> None:
        from portlearn.forecasting import require_fit_inputs_admitted
        from portlearn.timing import FutureInformationError

        p = _protocols()
        resolved = _resolve(
            p,
            _design(
                p,
                evaluation_learning=p.WALK_FORWARD,
                strategies=[("MV", _mv_annual_arm(p))],
            ),
            ["MV"],
            {"MV": ADMITTED},
        )
        refit = next(
            event
            for event in resolved["MV"].fit_events
            if instant_key(E_START) < instant_key(event.cutoff) < instant_key(E_END)
        )

        class _Record:
            series_id = "AUD_CPI"

            def __init__(self, available_time: Any) -> None:
                self.available_time = available_time

        class _FitSet:
            def __init__(self, as_of: Any) -> None:
                self.as_of = as_of

        # The in-span walk-forward event may consume only records available
        # at or before its own cutoff; anything later is post-decision.
        with pytest.raises(FutureInformationError):
            require_fit_inputs_admitted(
                [_Record(refit.cutoff + timedelta(days=45))],
                _FitSet(refit.cutoff),
            )

    def test_naive_available_time_still_rejects_at_an_in_span_refit(self) -> None:
        from portlearn.forecasting import require_fit_inputs_admitted
        from portlearn.timing import NaiveTimestampError

        p = _protocols()
        resolved = _resolve(
            p,
            _design(
                p,
                evaluation_learning=p.WALK_FORWARD,
                strategies=[("MV", _mv_annual_arm(p))],
            ),
            ["MV"],
            {"MV": ADMITTED},
        )
        refit = next(
            event
            for event in resolved["MV"].fit_events
            if instant_key(E_START) < instant_key(event.cutoff) < instant_key(E_END)
        )

        class _Record:
            series_id = "AUD_CPI"

            def __init__(self, available_time: Any) -> None:
                self.available_time = available_time

        class _FitSet:
            def __init__(self, as_of: Any) -> None:
                self.as_of = as_of

        # The backstop composes with the one-clock law: an unavailable
        # (naive) availability instant is not comparable on the single
        # clock, so it is unavailable full stop — no masking pass.
        with pytest.raises(NaiveTimestampError):
            require_fit_inputs_admitted(
                [_Record(month_end_instant(2019, 6, MEL).replace(tzinfo=None))],
                _FitSet(refit.cutoff),
            )


# --------------------------------------------------------------------------- #
# Floor: preprocessing/selection boundary — where each fit may consume.       #
# --------------------------------------------------------------------------- #
# Law 11: preprocessing is fitted only at declared fit events on lawful
# pre-cutoff TRAIN-role data; selection/tuning consumes only VALIDATION-
# role information available at the selection cutoff; a preprocessing fit
# or tuning step that consumes evaluation-span or post-cutoff information
# rejects (Law 10a).  The declarative halves here: the policies are frozen
# closed two-member vocabularies with no defaults (no invented third
# policy), and the resolved protocol carries the declared policy verbatim
# into the arm where the boundary binds.


class TestPreprocessingSelectionBoundary:
    """Law 11 pins declared policies at declared events on lawful data.

    Two designed causal-RED nodes: the closed vocabularies with a declared
    member carried verbatim, and the no-default requirement for both
    policies on the resolved protocol.
    """

    def test_policies_are_closed_two_member_vocabularies_carried_verbatim(self) -> None:
        p = _protocols()
        for enum_cls, members in (
            (p.PreprocessingPolicy, ("NONE", "TRAIN_ONLY_AT_EACH_FIT")),
            (p.SelectionPolicy, ("NONE", "VALIDATION_ONLY_AT_DECLARED_EVENTS")),
        ):
            names = [member.name for member in enum_cls]
            assert names == list(members)
        resolved = _resolve(
            p,
            _design(
                p,
                evaluation_learning=p.WALK_FORWARD,
                strategies=[("LSTM", _selection_arm(p))],
            ),
            ["LSTM"],
            {"LSTM": ADMITTED},
        )
        arm = resolved["LSTM"]
        assert arm.preprocessing_policy is p.PreprocessingPolicy.TRAIN_ONLY_AT_EACH_FIT
        assert (
            arm.selection_policy is p.SelectionPolicy.VALIDATION_ONLY_AT_DECLARED_EVENTS
        )

    def test_declared_policy_is_required_not_defaulted(self) -> None:
        p = _protocols()
        # The advanced common protocol has no default for either policy:
        # the field is required at construction (missing → TypeError, the
        # frozen no-default rejection), and the enum itself has no default
        # member to fill silently.
        with pytest.raises(TypeError):
            p.TemporalResearchProtocol(
                evaluation_learning=p.WALK_FORWARD,
                partitions=_standard_partitions(p),
                target_horizon=HORIZON,
                estimand=ESTIMAND,
                history_policy=None,
                selection_history=None,
                fit_events=(),
            )
        for enum_cls in (p.PreprocessingPolicy, p.SelectionPolicy):
            sentinel = object()
            assert getattr(enum_cls, "_missing_", sentinel) is not sentinel
            assert enum_cls._missing_(None) is None


# --------------------------------------------------------------------------- #
# Floor: event-order determinism on both branches (Law 12 / §6).              #
# --------------------------------------------------------------------------- #
# §6 pins the per-decision order exhaustively for both branches: any
# fit/refit/selection uses only information lawful at or before its
# declared cutoff; under WALK_FORWARD learning events continue at declared
# in-span cutoffs; under FIXED_ORIGIN step (1) executes no learning event
# inside the evaluation span.  The first decision is special: the initial
# fit must be declared AND satisfiable — an unavailable minimum history
# rejects at resolution with NO BACK-FILL (advisory 2, MT7).


class TestEventOrderDeterminism:
    """Law 12/§6 pins one deterministic event order on both branches.

    Four designed causal-RED nodes: the first-decision no-back-fill
    rejection (MT7), the walk-forward in-span order, the fixed-origin
    no-in-span-event order, and identical re-resolution determinism.
    """

    def test_unavailable_initial_window_rejects_with_no_back_fill(self) -> None:
        p = _protocols()
        # The arm declares a five-calendar-year estimation window; this
        # timeline's earliest admitted record is years after the window
        # opens, so the initial fit is declared but unsatisfiable.  The
        # only lawful outcome is rejection at resolution — MT7: no
        # back-filled (shortened) initial window may be invented.
        with pytest.raises(p.ProtocolStructureError):
            _resolve(
                p,
                _design(
                    p,
                    evaluation_learning=p.WALK_FORWARD,
                    strategies=[("MV", _mv_arm(p))],
                ),
                ["MV"],
                {"MV": ADMITTED_LATE},
            )

    def test_walk_forward_events_march_strictly_increasing_in_span(self) -> None:
        p = _protocols()
        resolved = _resolve(
            p,
            _design(
                p,
                evaluation_learning=p.WALK_FORWARD,
                strategies=[("MV", _mv_annual_arm(p))],
            ),
            ["MV"],
            {"MV": ADMITTED},
        )
        events = resolved["MV"].fit_events
        kinds = [event.kind for event in events]
        keys = _keys([event.cutoff for event in events])
        assert kinds[0] == p.INITIAL_FIT
        assert keys[0] == instant_key(C_ZERO)
        assert instant_key(keys[0]) < instant_key(E_START)
        assert keys == sorted(keys)
        assert len(set(keys)) == len(keys)
        refits = [event for event in events if event.kind == p.REFIT]
        assert refits
        for event in refits:
            assert (
                instant_key(E_START) <= instant_key(event.cutoff) < instant_key(E_END)
            )

    def test_fixed_origin_resolves_one_pre_span_event_and_no_in_span_event(
        self,
    ) -> None:
        p = _protocols()
        resolved = _resolve(
            p,
            _design(
                p,
                evaluation_learning=p.FIXED_ORIGIN,
                strategies=[("MV", _mv_arm(p))],
            ),
            ["MV"],
            {"MV": ADMITTED},
        )
        events = resolved["MV"].fit_events
        assert len(events) == 1
        only = events[0]
        assert only.kind == p.INITIAL_FIT
        assert only.role == p.PartitionRole.TRAIN
        assert instant_key(only.cutoff) == instant_key(C_ZERO)
        assert instant_key(only.cutoff) < instant_key(E_START)

    def test_resolution_is_deterministic_on_both_branches(self) -> None:
        p = _protocols()
        for learning in (p.WALK_FORWARD, p.FIXED_ORIGIN):
            arm = _mv_annual_arm(p) if learning == p.WALK_FORWARD else _mv_arm(p)
            first = _resolve(
                p,
                _design(
                    p,
                    evaluation_learning=learning,
                    strategies=[("MV", arm)],
                ),
                ["MV"],
                {"MV": ADMITTED},
            )
            second = _resolve(
                p,
                _design(
                    p,
                    evaluation_learning=learning,
                    strategies=[("MV", arm)],
                ),
                ["MV"],
                {"MV": ADMITTED},
            )
            assert _keys([e.cutoff for e in first["MV"].fit_events]) == _keys(
                [e.cutoff for e in second["MV"].fit_events]
            )
            assert [e.kind for e in first["MV"].fit_events] == [
                e.kind for e in second["MV"].fit_events
            ]


# ---------------------------------------------------------------------------
# Floor: opaque consumption by the experiment runtime (Law 13).
# ---------------------------------------------------------------------------
# The per-run protocol image is what runtime consumes, and it consumes it
# opaquely: the per-event images are the frozen forecasting-lifecycle ForecastProvenance
# (the same ten-field image ForecastLink already consumes), reused and
# never paralleled protocol-side, and the protocol module exposes no
# forecasting-lifecycle validator call route for the runtime to reach through.  No runtime
# W3 runtime behavior is asserted here.


class TestOpaqueConsumption:
    """Law 13 pins opaque reuse of the frozen provenance image.

    Two designed causal-RED nodes: the image carrier is typed by the
    frozen forecasting-lifecycle ForecastProvenance with no protocol-side parallel, and
    the module surface carries no forecasting-lifecycle validator the runtime could
    call through the protocol layer.
    """

    def test_provenance_images_reuse_the_frozen_m17_forecast_type(self) -> None:
        p = _protocols()
        from portlearn.forecasting import ForecastProvenance

        assert isinstance(ForecastProvenance, type)
        # Class-level introspection only: the constructor is not
        # section-4 pinned and is never constructed (recorded
        # adjudication from the provenance-completeness floor).
        declared = dataclasses.fields(p.ProtocolProvenance)
        assert declared
        assert any("ForecastProvenance" in str(field.type) for field in declared)
        # Reused, never paralleled: the frozen image type is not
        # re-exported on the protocol surface.
        assert not hasattr(p, "ForecastProvenance")

    def test_protocol_surface_exposes_no_m17_validator_call_route(self) -> None:
        p = _protocols()
        from portlearn.forecasting import (
            require_fit_inputs_admitted,
            require_selection_labels_available,
        )

        for validator in (
            require_fit_inputs_admitted,
            require_selection_labels_available,
        ):
            assert callable(validator)
            # Law 13: runtime never calls forecasting-lifecycle validators; the protocol
            # layer hands over an image, not a validator call route.
            assert not hasattr(p, validator.__name__)


# ---------------------------------------------------------------------------
# Floor: typed fail-closed error arms (Law 14).
# ---------------------------------------------------------------------------
# Every protocol rejection raises a typed error before execution: the
# two new protocol errors are ValueError-style classes distinct from
# every frozen arm, and the frozen timing arm is reused verbatim for
# the naive-zone rejection, never paralleled protocol-side.


class TestTypedErrorArms:
    """Law 14 pins the typed arms for every protocol rejection.

    Two designed causal-RED nodes: the two new errors are distinct
    ValueError-style classes and never aliases of the frozen arms, and
    the naive-zone construction rejection raises exactly the frozen
    timing error class.
    """

    def test_new_errors_are_valuetype_and_not_frozen_aliases(self) -> None:
        p = _protocols()
        from portlearn.forecasting import ProvenanceStructureError

        assert issubclass(p.ProtocolVocabularyError, ValueError)
        assert issubclass(p.ProtocolStructureError, ValueError)
        assert p.ProtocolVocabularyError is not p.ProtocolStructureError
        # The frozen arms are reused, never re-defined: both new
        # errors are distinct classes from the frozen forecasting arm.
        assert p.ProtocolVocabularyError is not ProvenanceStructureError
        assert p.ProtocolStructureError is not ProvenanceStructureError

    def test_naive_zone_rejection_raises_the_frozen_timing_error(self) -> None:
        p = _protocols()
        with pytest.raises(NaiveTimestampError) as caught:
            p.TemporalDesign(
                evaluation_span=EVALUATION_SPAN,
                evaluation_learning=p.WALK_FORWARD,
                estimand=ESTIMAND,
                target_horizon=HORIZON,
                timezone=None,
                strategies=[("EW", _ew_arm(p))],
            )
        # Exactly the frozen timing class: not a protocol-side
        # parallel and not a subclass declared in the new module.
        assert type(caught.value) is NaiveTimestampError


# ---------------------------------------------------------------------------
# Floor: sparse inheritance and no repetition (Law 15; MT8).
# ---------------------------------------------------------------------------
# An unspecified override field inherits the common value (nothing is
# invented), a fully unspecified override resolves to exactly the
# common protocol, and the ordinary per-strategy declaration carries
# only the four strategy-neutral fields so common anchors cannot be
# repeated there.


def _overridable_common(p: Any) -> Any:
    """A lawful advanced-route common protocol with one initial fit."""
    return _common_protocol(
        p,
        evaluation_learning=p.WALK_FORWARD,
        history_policy=p.HistoryPolicy(
            mode=p.HistoryMode.ROLLING,
            basis=p.HistoryBasis.CALENDAR_DURATION,
            minimum_history=p.CalendarSpan(years=5),
        ),
        fit_events=(_initial_fit_event(p),),
        preprocessing_policy=p.PreprocessingPolicy.NONE,
        selection_policy=p.SelectionPolicy.NONE,
    )


class TestSparseInheritance:
    """Law 15 pins inherit-don't-invent for unspecified fields.

    Three designed causal-RED nodes: field-by-field sparse
    inheritance, the no-anchor-fields census of the ordinary arm
    declaration, and whole-protocol equality for a fully unspecified
    override.  The MT8 killers: an invented value for any unspecified
    field breaks the inheritance assertions and the equality.
    """

    def test_sparse_override_inherits_every_unspecified_field(self) -> None:
        p = _protocols()
        common = _overridable_common(p)
        resolved = _resolve(
            p,
            p.ExperimentTemporalDeclaration(
                common=common,
                strategy_overrides=[
                    (
                        "MV",
                        p.StrategyProtocolOverride(
                            preprocessing_policy=(
                                p.PreprocessingPolicy.TRAIN_ONLY_AT_EACH_FIT
                            )
                        ),
                    )
                ],
            ),
            ["MV"],
            {"MV": ADMITTED},
        )
        arm = resolved["MV"]
        assert arm.preprocessing_policy is p.PreprocessingPolicy.TRAIN_ONLY_AT_EACH_FIT
        assert arm.history_policy == common.history_policy
        assert arm.selection_history == common.selection_history
        assert arm.fit_events == common.fit_events
        assert arm.selection_policy is common.selection_policy

    def test_ordinary_arm_declaration_has_no_anchor_fields_to_repeat(self) -> None:
        p = _protocols()
        declared = {
            field.name for field in dataclasses.fields(p.StrategyTemporalDeclaration)
        }
        assert declared == {"estimation", "selection", "refit", "preprocessing"}
        anchors = (
            "evaluation_span",
            "evaluation_learning",
            "estimand",
            "target_horizon",
            "timezone",
            "partitions",
        )
        assert declared.isdisjoint(anchors)

    def test_all_none_override_resolves_to_the_common_protocol(self) -> None:
        p = _protocols()
        common = _overridable_common(p)
        resolved = _resolve(
            p,
            p.ExperimentTemporalDeclaration(
                common=common,
                strategy_overrides=[("MV", p.StrategyProtocolOverride())],
            ),
            ["MV"],
            {"MV": ADMITTED},
        )
        assert resolved["MV"] == common


# ---------------------------------------------------------------------------
# Floor: override policing at resolution (Law 15; MT9).
# ---------------------------------------------------------------------------
# Resolution is exact about strategy coverage and override integrity:
# an unknown override key, a named strategy with no pair, a duplicated
# key in the pair sequence, and an override whose spliced protocol
# fails whole-object canonical re-validation each reject fail-closed.


class TestOverridePolicing:
    """Law 15 pins fail-closed policing of the override sequence.

    Two designed causal-RED nodes (the MT9 killers): unknown and
    uncovered strategy keys reject, and duplicate pairs plus a
    conflicting splice reject.
    """

    def test_unknown_and_uncovered_strategy_keys_reject(self) -> None:
        p = _protocols()
        common = _overridable_common(p)
        # An override key outside strategy_names is unknown: reject.
        with pytest.raises(p.ProtocolStructureError):
            _resolve(
                p,
                p.ExperimentTemporalDeclaration(
                    common=common,
                    strategy_overrides=[
                        ("MV", p.StrategyProtocolOverride()),
                        ("LSTM", p.StrategyProtocolOverride()),
                    ],
                ),
                ["MV"],
                {"MV": ADMITTED},
            )
        # A duplicated strategy_names entry: reject (coverage must be
        # exactly one record timeline per named strategy).
        with pytest.raises(p.ProtocolStructureError):
            _resolve(
                p,
                p.ExperimentTemporalDeclaration(
                    common=common,
                    strategy_overrides=[("MV", p.StrategyProtocolOverride())],
                ),
                ["MV", "MV"],
                {"MV": ADMITTED},
            )

    def test_duplicate_pairs_and_conflicting_splices_reject(self) -> None:
        p = _protocols()
        common = _overridable_common(p)
        # The pair sequence retains duplicate evidence: the same
        # strategy key twice rejects before canonicalization.
        duplicate = p.StrategyProtocolOverride()
        with pytest.raises(p.ProtocolStructureError):
            _resolve(
                p,
                p.ExperimentTemporalDeclaration(
                    common=common,
                    strategy_overrides=[("MV", duplicate), ("MV", duplicate)],
                ),
                ["MV"],
                {"MV": ADMITTED},
            )
        # A conflicting override: the spliced protocol carries a
        # SELECTION event beside SelectionPolicy.NONE and a None
        # selection_history, failing whole-object re-validation.
        selection = p.FitEvent(
            kind=p.SELECTION,
            cutoff=VALIDATION_START,
            role=p.PartitionRole.VALIDATION,
        )
        with pytest.raises(p.ProtocolStructureError):
            _resolve(
                p,
                p.ExperimentTemporalDeclaration(
                    common=common,
                    strategy_overrides=[
                        ("MV", p.StrategyProtocolOverride(fit_events=(selection,)))
                    ],
                ),
                ["MV"],
                {"MV": ADMITTED},
            )


# ---------------------------------------------------------------------------
# Floor: non-overridable common anchors (Law 15; MT10).
# ---------------------------------------------------------------------------
# The common decision and evaluation anchors cannot move: resolution
# copies them onto every arm and asserts identity against the common
# protocol, and an override whose instants escape the common research
# timeline (the only indirect move route) rejects fail-closed.


class TestNonOverridableAnchors:
    """Law 15 pins anchor immovability across the override route.

    Two designed causal-RED nodes (the MT10 killers): resolved anchors
    are identical to the common protocol's, and override fit events
    escaping the common research timeline reject.
    """

    def test_resolution_asserts_anchor_identity_onto_every_arm(self) -> None:
        p = _protocols()
        common = _overridable_common(p)
        override = p.StrategyProtocolOverride(
            preprocessing_policy=p.PreprocessingPolicy.TRAIN_ONLY_AT_EACH_FIT
        )
        arm = _resolve(
            p,
            p.ExperimentTemporalDeclaration(
                common=common, strategy_overrides=[("MV", override)]
            ),
            ["MV"],
            {"MV": ADMITTED},
        )["MV"]
        assert arm.evaluation_learning is common.evaluation_learning
        assert arm.estimand == common.estimand
        assert arm.target_horizon == common.target_horizon
        assert tuple(arm.partitions) == tuple(common.partitions)

    def test_override_events_escaping_the_common_timeline_reject(self) -> None:
        p = _protocols()
        common = _overridable_common(p)
        # Instants outside the union of the declared partition spans:
        # after the evaluation-span end and before the TRAIN start.
        for rogue_cutoff in (
            month_end_instant(2026, 6, MEL),
            month_end_instant(2010, 12, MEL),
        ):
            rogue = p.FitEvent(
                kind=p.REFIT,
                cutoff=rogue_cutoff,
                role=p.PartitionRole.TRAIN,
            )
            with pytest.raises(p.ProtocolStructureError):
                _resolve(
                    p,
                    p.ExperimentTemporalDeclaration(
                        common=common,
                        strategy_overrides=[
                            (
                                "MV",
                                p.StrategyProtocolOverride(fit_events=(rogue,)),
                            )
                        ],
                    ),
                    ["MV"],
                    {"MV": ADMITTED},
                )


# ---------------------------------------------------------------------------
# Floor: canonical route equality (Law 16; MT11).
# ---------------------------------------------------------------------------
# The ordinary declaration and its advanced spelling resolve to the
# same canonical protocol (the authority's own executable example
# assertion), and an observation refit cadence materializes at
# resolution into concrete events read from each strategy's own
# admitted sequence.


class TestCanonicalRouteEquality:
    """Law 16 pins one canonical resolution across both routes.

    Two designed causal-RED nodes: concise-vs-advanced whole-protocol
    equality (the MT11 killer: any route-dependent validation, event
    order, cutoff, or output breaks it), and observation-cadence
    materialization from each strategy's own sequence.
    """

    def test_concise_declaration_resolves_equal_to_its_advanced_spelling(self):
        p = _protocols()
        ordinary = _resolve(
            p,
            _design(
                p,
                evaluation_learning=p.WALK_FORWARD,
                strategies=[("MV", _mv_arm(p))],
            ),
            ["MV"],
            {"MV": ADMITTED},
        )
        advanced = _resolve(
            p,
            p.ExperimentTemporalDeclaration(
                common=_common_protocol(
                    p,
                    evaluation_learning=p.WALK_FORWARD,
                    history_policy=p.HistoryPolicy(
                        mode=p.HistoryMode.ROLLING,
                        basis=p.HistoryBasis.CALENDAR_DURATION,
                        minimum_history=p.CalendarSpan(years=5),
                    ),
                    fit_events=(_initial_fit_event(p),),
                    preprocessing_policy=p.PreprocessingPolicy.NONE,
                    selection_policy=p.SelectionPolicy.NONE,
                ),
                strategy_overrides=[],
            ),
            ["MV"],
            {"MV": ADMITTED},
        )
        assert ordinary["MV"] == advanced["MV"]
        # The comparator-fairness anchor itself: identical partitions.
        assert list(ordinary["MV"].partitions) == list(advanced["MV"].partitions)

    def test_observation_cadence_materializes_from_each_own_sequence(self):
        p = _protocols()
        wide_start = month_end_instant(2012, 12, MEL)
        common = p.TemporalResearchProtocol(
            evaluation_learning=p.WALK_FORWARD,
            partitions=(
                p.TemporalPartition(
                    role=p.PartitionRole.TRAIN, start=wide_start, end=C_ZERO
                ),
                p.TemporalPartition(
                    role=p.PartitionRole.TEST, start=E_START, end=E_END
                ),
            ),
            target_horizon=HORIZON,
            estimand=ESTIMAND,
            history_policy=p.HistoryPolicy(
                mode=p.HistoryMode.ROLLING,
                basis=p.HistoryBasis.OBSERVATION_COUNT,
                minimum_history=1,
            ),
            selection_history=None,
            fit_events=p.observation_refits(
                every_observations=6, role=p.PartitionRole.TRAIN
            ),
            preprocessing_policy=p.PreprocessingPolicy.NONE,
            selection_policy=p.SelectionPolicy.NONE,
        )
        resolved = _resolve(
            p,
            p.ExperimentTemporalDeclaration(common=common, strategy_overrides=[]),
            ["DENSE", "SPARSE"],
            {"DENSE": ADMITTED, "SPARSE": ADMITTED_SPARSE},
        )
        materialized = {}
        for name, own in (("DENSE", ADMITTED), ("SPARSE", ADMITTED_SPARSE)):
            events = resolved[name].fit_events
            # The cadence never survives resolution: concrete events.
            assert events
            assert not isinstance(events, p.ObservationRefitCadence)
            assert all(isinstance(event, p.FitEvent) for event in events)
            # The declared role is stamped on every materialized event
            # (the kind is not section-4 pinned and is not asserted).
            assert all(event.role is p.PartitionRole.TRAIN for event in events)
            keys = _keys([event.cutoff for event in events])
            assert keys == sorted(keys)
            assert len(set(keys)) == len(keys)
            assert set(keys) <= set(_keys(own))
            materialized[name] = list(keys)
        # A common cadence reads each strategy's OWN admitted sequence:
        # the two materialized cutoff sets differ.
        assert materialized["DENSE"] != materialized["SPARSE"]


# ---------------------------------------------------------------------------
# Floor: provenance equality across routes (Law 16; MT12).
# ---------------------------------------------------------------------------
# ProtocolProvenance records the resolved protocol only: never the
# declaration route and never the pre-resolution shape, so canonically
# equal inputs are provenance-identical.  The constructor is not
# section-4 pinned, so both nodes are class-level introspection (the
# recorded adjudication); the full MT12 kill is execution-stage.


class TestProvenanceRouteIndependence:
    """Law 16 pins a route-blind provenance image.

    Two designed causal-RED declarative halves (the MT12 killers
    available before execution-stage provenance exists): no route-typed
    or route-named field, and no field carrying any pre-resolution
    declaration shape.
    """

    def test_provenance_declares_no_route_typed_or_route_named_fields(self):
        p = _protocols()
        declared = dataclasses.fields(p.ProtocolProvenance)
        assert declared
        for field in declared:
            name = field.name.lower()
            declared_type = str(field.type).lower()
            for token in _ROUTE_TOKENS:
                assert token not in name
                assert token not in declared_type

    def test_provenance_fields_exclude_every_pre_resolution_shape(self):
        p = _protocols()
        declared = dataclasses.fields(p.ProtocolProvenance)
        assert declared
        pre_resolution = (
            "TemporalDesign",
            "ExperimentTemporalDeclaration",
            "StrategyProtocolOverride",
            "StrategyTemporalDeclaration",
        )
        texts = [field.name for field in declared]
        texts.extend(str(field.type) for field in declared)
        for shape in pre_resolution:
            assert all(shape not in text for text in texts)


# ---------------------------------------------------------------------------
# Floor: resolution non-mutation, order invariance, and record-timeline
# coverage (Law 16 / section 4).
# ---------------------------------------------------------------------------
# Resolution is non-mutating and input-order invariant on both routes,
# and it is exact about record-timeline coverage: a missing record key,
# an extra record key, and a non-increasing admitted sequence each
# reject fail-closed.  (Naive record instants reject on the frozen
# timing arm, pinned already by the one-clock floor.)


class TestResolutionPurity:
    """Law 16 pins pure, order-invariant, exact-coverage resolution.

    Two designed causal-RED nodes: resolving twice in reversed input
    order leaves every arm and every input untouched, and record
    timeline coverage mismatches reject.
    """

    def test_resolution_is_non_mutating_and_order_invariant(self):
        p = _protocols()
        timeline = {"MV": ADMITTED, "EW": ADMITTED}
        snapshot = {name: tuple(value) for name, value in timeline.items()}
        design = _design(
            p,
            evaluation_learning=p.WALK_FORWARD,
            strategies=[("MV", _mv_arm(p)), ("EW", _ew_arm(p))],
        )
        names = ["MV", "EW"]
        forward = p.resolve_temporal_protocols(design, list(names), dict(timeline))
        reverse = p.resolve_temporal_protocols(
            design, list(names)[::-1], dict(timeline)
        )
        assert forward["MV"] == reverse["MV"]
        assert forward["EW"] == reverse["EW"]
        # Non-mutation: the caller's admitted-timeline mapping survives
        # resolution unchanged (key set and every sequence value).
        assert set(timeline) == set(snapshot)
        for name, value in timeline.items():
            assert tuple(value) == snapshot[name]
        assert design == _design(
            p,
            evaluation_learning=p.WALK_FORWARD,
            strategies=[("MV", _mv_arm(p)), ("EW", _ew_arm(p))],
        )
        # Advanced route: override-pair order cannot change resolution.
        common = _overridable_common(p)
        mv_override = p.StrategyProtocolOverride(
            preprocessing_policy=p.PreprocessingPolicy.TRAIN_ONLY_AT_EACH_FIT
        )
        ew_override = p.StrategyProtocolOverride()
        straight = _resolve(
            p,
            p.ExperimentTemporalDeclaration(
                common=common,
                strategy_overrides=[("MV", mv_override), ("EW", ew_override)],
            ),
            ["MV", "EW"],
            timeline,
        )
        swapped = _resolve(
            p,
            p.ExperimentTemporalDeclaration(
                common=common,
                strategy_overrides=[("EW", ew_override), ("MV", mv_override)],
            ),
            ["MV", "EW"],
            timeline,
        )
        assert straight["MV"] == swapped["MV"]
        assert straight["EW"] == swapped["EW"]

    def test_record_timeline_coverage_mismatch_rejects(self):
        p = _protocols()
        design = _design(
            p,
            evaluation_learning=p.WALK_FORWARD,
            strategies=[("MV", _mv_arm(p)), ("EW", _ew_arm(p))],
        )
        # A named strategy with no record key: reject.
        with pytest.raises(p.ProtocolStructureError):
            _resolve(p, design, ["MV", "EW"], {"MV": ADMITTED})
        # A record key outside strategy_names: reject.
        with pytest.raises(p.ProtocolStructureError):
            _resolve(
                p,
                design,
                ["MV", "EW"],
                {"MV": ADMITTED, "EW": ADMITTED, "LSTM": ADMITTED},
            )
        # A non-increasing admitted sequence on the single clock: a
        # real sequence carrying a descent, representable as written.
        descending = (ADMITTED[1], ADMITTED[0], *ADMITTED[2:])
        with pytest.raises(p.ProtocolStructureError):
            _resolve(
                p,
                design,
                ["MV", "EW"],
                {"MV": descending, "EW": ADMITTED},
            )
