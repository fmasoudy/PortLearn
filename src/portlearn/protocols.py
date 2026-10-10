"""The temporal-protocol temporal research protocol layer.

This module implements the temporal research protocol layer
(specification approved 2026-09-30, design SHA ``5a484127…``): the declarative
temporal research protocol that every governed experiment resolves
before its first decision.

Three authoring layers, one canonical resolution:

* the canonical layer — :class:`TemporalResearchProtocol` and its public
  value objects (the advanced, fully explicit form);
* the advanced composition layer — :class:`ExperimentTemporalDeclaration`
  with sparse :class:`StrategyProtocolOverride` pairs (declare once,
  override sparsely);
* the ordinary authoring layer — :class:`TemporalDesign` with per-strategy
  :class:`StrategyTemporalDeclaration` objects speaking only estimation,
  selection, refit, and preprocessing.

Both authoring routes normalize to canonically equal resolved protocols
through the one resolution step, :func:`resolve_temporal_protocols`, which
is declarative, deterministic, non-mutating, order-invariant, and
fail-closed. Public interfaces are consumed read-only and never paralleled:
aware-instant handling and the single clock come from ``portlearn.timing``,
calendar edges from ``portlearn.calendar``, and the provenance image type
from ``portlearn.forecasting`` (reused, never re-exported here).
"""

from __future__ import annotations

import calendar as _stdlib_calendar
import re
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from itertools import pairwise
from typing import Any

from . import calendar as _calendar
from . import forecasting as _forecasting
from . import timing as _timing

__all__ = [
    "CalendarSpan",
    "EvaluationLearning",
    "ExperimentTemporalDeclaration",
    "FitEvent",
    "HistoryBasis",
    "HistoryMode",
    "HistoryPolicy",
    "ObservationRefitCadence",
    "PartitionRole",
    "PreprocessingPolicy",
    "ProtocolProvenance",
    "ProtocolStructureError",
    "ProtocolVocabularyError",
    "RefitCadence",
    "SelectionPolicy",
    "StrategyProtocolOverride",
    "StrategyTemporalDeclaration",
    "TemporalDesign",
    "TemporalPartition",
    "TemporalResearchProtocol",
    "WindowBasis",
    "WindowMode",
    "WindowSpec",
    "annual",
    "at_each_fit",
    "expanding_calendar_history",
    "expanding_observation_history",
    "month_end_refits",
    "monthly",
    "none",
    "observation_refits",
    "quarter_end_refits",
    "quarterly",
    "resolve_temporal_protocols",
    "rolling_calendar_history",
    "rolling_observation_history",
    "window",
]


# --------------------------------------------------------------------------- #
# Typed fail-closed errors (Law 14) — the two new protocol arms.
# --------------------------------------------------------------------------- #


class ProtocolVocabularyError(ValueError):
    """A declared token is outside its closed vocabulary.

    Every enum-bearing field is closed: an unknown or missing member
    rejects at construction with this error, never a coerced or
    nearest-neighbor value.
    """


class ProtocolStructureError(ValueError):
    """A declaration or resolution is structurally invalid.

    Covers every non-vocabulary protocol rejection: chronological,
    containment, coherence, coverage, cross-field, and conflict rules.
    """


# --------------------------------------------------------------------------- #
# Closed vocabularies (frozen enums; no default member anywhere).
# --------------------------------------------------------------------------- #


class _ClosedEnum(Enum):
    """A closed vocabulary: unknown tokens reject typed, ``None`` is absent."""

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if value is None:
            return None
        raise ProtocolVocabularyError(
            f"{cls.__name__} is a closed vocabulary: the token {value!r} "
            "is not a member and no value is ever coerced or defaulted"
        )


class EvaluationLearning(_ClosedEnum):
    """Where learning events may occur relative to the evaluation span."""

    FIXED_ORIGIN = "fixed_origin"
    WALK_FORWARD = "walk_forward"


class PreprocessingPolicy(_ClosedEnum):
    """Where protocol-declared preprocessing may be fitted."""

    NONE = "none"
    TRAIN_ONLY_AT_EACH_FIT = "train_only_at_each_fit"


class SelectionPolicy(_ClosedEnum):
    """What protocol-declared selection/tuning may consume."""

    NONE = "none"
    VALIDATION_ONLY_AT_DECLARED_EVENTS = "validation_only_at_declared_events"


class PartitionRole(_ClosedEnum):
    """The declared role linkage of a partition or fit event."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class HistoryMode(_ClosedEnum):
    """Rolling (exact trailing window) or expanding (fixed origin)."""

    ROLLING = "rolling"
    EXPANDING = "expanding"


class HistoryBasis(_ClosedEnum):
    """Calendar-duration or observation-count window basis."""

    CALENDAR_DURATION = "calendar_duration"
    OBSERVATION_COUNT = "observation_count"


class WindowBasis(_ClosedEnum):
    """The ordinary window basis: calendar or observations."""

    CALENDAR = "calendar"
    OBSERVATIONS = "observations"


class WindowMode(_ClosedEnum):
    """The ordinary window mode: rolling or expanding (never assumed)."""

    ROLLING = "rolling"
    EXPANDING = "expanding"


class _EventKind(_ClosedEnum):
    """The kind of one declared fit event (module-level tokens below)."""

    INITIAL_FIT = "initial_fit"
    REFIT = "refit"
    SELECTION = "selection"


# Module-level vocabulary tokens — the spelling the authority's own
# executable example uses (importable attributes, deliberately not in
# ``__all__``, which pins exactly the §4 public surface).
FIXED_ORIGIN = EvaluationLearning.FIXED_ORIGIN
WALK_FORWARD = EvaluationLearning.WALK_FORWARD
INITIAL_FIT = _EventKind.INITIAL_FIT
REFIT = _EventKind.REFIT
SELECTION = _EventKind.SELECTION

_EVENT_KIND_ORDER = {
    _EventKind.INITIAL_FIT: 0,
    _EventKind.REFIT: 1,
    _EventKind.SELECTION: 2,
}
_ROLE_ORDER = {
    PartitionRole.TRAIN: 0,
    PartitionRole.VALIDATION: 1,
    PartitionRole.TEST: 2,
}


# --------------------------------------------------------------------------- #
# Shared validation helpers (single clock on the frozen timing seam).
# --------------------------------------------------------------------------- #


def _set(obj: Any, name: str, value: Any) -> None:
    """Immutable-safe field normalization inside ``__post_init__``."""
    object.__setattr__(obj, name, value)


def _require_member(value: Any, enum_cls: type[Enum], field_name: str) -> Any:
    """Return ``value`` as a member of ``enum_cls`` or reject typed."""
    if isinstance(value, enum_cls):
        return value
    if value is None:
        # A required closed-vocabulary field carries no default member
        # (§4): its absence is a construction rejection — a structural
        # fault, distinct from an unknown token.
        raise ProtocolStructureError(
            f"{field_name} must be declared explicitly with a member of "
            f"{enum_cls.__name__}; None is not a member and no default "
            "member exists to fill silently"
        )
    try:
        return enum_cls(value)
    except ValueError as exc:
        raise ProtocolVocabularyError(
            f"{field_name} must be a {enum_cls.__name__} member; "
            f"got {type(value).__name__}: {value!r}"
        ) from exc


def _require_aware(value: Any, field_name: str) -> datetime:
    """Validate an aware instant on the public interface, returning it verbatim."""
    _timing.to_instant(value, field_name)
    # Stored verbatim; instant normalization is comparison-only.
    return value


def _require_positive_timedelta(value: Any, field_name: str) -> timedelta:
    if not isinstance(value, timedelta) or value <= timedelta(0):
        raise ProtocolStructureError(
            f"{field_name} must be a strictly positive timedelta; got "
            f"{type(value).__name__}: {value!r}"
        )
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ProtocolStructureError(
            f"{field_name} must be a strictly positive non-bool integer; "
            f"got {type(value).__name__}: {value!r}"
        )
    return value


def _require_non_blank_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolVocabularyError(
            f"{field_name} must be a non-blank vocabulary token string; "
            f"got {type(value).__name__}: {value!r}"
        )
    return value


def _require_strategy_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolStructureError(
            f"{field_name} must be a non-blank exact strategy id string; "
            f"got {type(value).__name__}: {value!r}"
        )
    return value


def _require_pairs(
    value: Any, element_cls: type, field_name: str
) -> tuple[tuple[str, Any], ...]:
    """Validate an ordered ``(strategy_id, declaration)`` pair sequence."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProtocolStructureError(
            f"{field_name} must be a sequence of "
            f"(strategy_id, {element_cls.__name__}) pairs; got "
            f"{type(value).__name__}: {value!r}"
        )
    pairs: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for entry in value:
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence):
            raise ProtocolStructureError(
                f"{field_name} entries must be (strategy_id, "
                f"{element_cls.__name__}) pairs; got {entry!r}"
            )
        if len(entry) != 2:
            raise ProtocolStructureError(
                f"{field_name} entries must be exactly two-element pairs; got {entry!r}"
            )
        key, declaration = entry[0], entry[1]
        _require_strategy_id(key, f"{field_name} strategy id")
        if not isinstance(declaration, element_cls):
            raise ProtocolStructureError(
                f"{field_name} declarations must be {element_cls.__name__} "
                f"instances; got {type(declaration).__name__}: {declaration!r}"
            )
        if key in seen:
            raise ProtocolStructureError(
                f"{field_name} contains the duplicated strategy id {key!r}; "
                "duplicates reject before canonicalization"
            )
        seen.add(key)
        pairs.append((key, declaration))
    return tuple(pairs)


# --------------------------------------------------------------------------- #
# Calendar arithmetic on the frozen calendar seam (one declared zone).
# --------------------------------------------------------------------------- #


def _month_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def _index_year_month(index: int) -> tuple[int, int]:
    year, month_zero = divmod(index, 12)
    return year, month_zero + 1


def _shift_months(moment: datetime, months: int) -> datetime:
    """Shift a wall-clock datetime by exact Gregorian calendar months."""
    year, month = _index_year_month(_month_index(moment.year, moment.month) + months)
    day = min(moment.day, _stdlib_calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)


def _calendar_span_months(span: CalendarSpan) -> int:
    return span.years * 12 + span.months


def _month_end_wall(index: int, zone: Any) -> datetime:
    """The month-end instant of ``index`` expressed on the declared zone."""
    year, month = _index_year_month(index)
    return _calendar.month_end_instant(year, month, zone).astimezone(zone)


def _window_start(
    policy: HistoryPolicy, anchor_end: datetime, own_keys: list[datetime]
) -> datetime:
    """The instant where the declared trailing window before ``anchor_end`` opens.

    Calendar windows use exact Gregorian arithmetic on the anchor's own
    wall clock; observation-count windows materialize against the
    strategy's own admitted sequence (never a calendar guess).
    """
    if policy.basis is HistoryBasis.CALENDAR_DURATION:
        length = policy.minimum_history
        if isinstance(length, CalendarSpan):
            return _shift_months(anchor_end, -_calendar_span_months(length))
        return anchor_end - length
    count = int(policy.minimum_history)
    position = bisect_right(own_keys, _timing.instant_key(anchor_end, "window end"))
    if position < count:
        raise ProtocolStructureError(
            "the declared observation-count window is unavailable at the "
            f"anchor {anchor_end.isoformat()}: the strategy's own admitted "
            f"sequence provides {position} records but {count} are required "
            "(no back-fill)"
        )
    return own_keys[position - count]


def _sequence_instants(value: Any, field_name: str) -> tuple[datetime, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProtocolStructureError(
            f"{field_name} must be a sequence of aware instants; got "
            f"{type(value).__name__}: {value!r}"
        )
    return tuple(value)


# --------------------------------------------------------------------------- #
# Canonical value objects (the advanced, fully explicit form).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CalendarSpan:
    """An exact calendar-period length (never 365 days, never 252 obs)."""

    years: int = 0
    months: int = 0

    def __post_init__(self) -> None:
        for name, value in (("years", self.years), ("months", self.months)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ProtocolStructureError(
                    f"CalendarSpan.{name} must be a non-negative non-bool "
                    f"integer; got {type(value).__name__}: {value!r}"
                )
        if self.years == 0 and self.months == 0:
            raise ProtocolStructureError(
                "CalendarSpan must not be zero: years and months are both 0"
            )


@dataclass(frozen=True)
class HistoryPolicy:
    """A history-window rule: mode, basis, and exact minimum history.

    No ``step`` field exists and none may be added: history-window
    semantics and refit cadence are disjoint concepts.
    """

    mode: HistoryMode
    basis: HistoryBasis
    minimum_history: CalendarSpan | timedelta | int

    def __post_init__(self) -> None:
        _set(
            self, "mode", _require_member(self.mode, HistoryMode, "HistoryPolicy.mode")
        )
        _set(
            self,
            "basis",
            _require_member(self.basis, HistoryBasis, "HistoryPolicy.basis"),
        )
        length = self.minimum_history
        if self.basis is HistoryBasis.CALENDAR_DURATION:
            lawful = isinstance(length, CalendarSpan) or (
                isinstance(length, timedelta) and length > timedelta(0)
            )
            if not lawful:
                raise ProtocolStructureError(
                    "CALENDAR_DURATION history requires a CalendarSpan or a "
                    "strictly positive timedelta minimum_history; got "
                    f"{type(length).__name__}: {length!r}"
                )
        else:
            _require_positive_int(length, "OBSERVATION_COUNT minimum_history")


@dataclass(frozen=True)
class TemporalPartition:
    """A declared, role-linked, chronological instant interval."""

    role: PartitionRole
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        _set(self, "role", _require_member(self.role, PartitionRole, "role"))
        _require_aware(self.start, "TemporalPartition.start")
        _require_aware(self.end, "TemporalPartition.end")
        if _timing.instant_key(self.start, "start") >= _timing.instant_key(
            self.end, "end"
        ):
            raise ProtocolStructureError(
                "a TemporalPartition must satisfy start < end on the single "
                f"clock; got [{self.start.isoformat()}, "
                f"{self.end.isoformat()}] with role {self.role.name}"
            )


@dataclass(frozen=True)
class FitEvent:
    """One declared fit/refit/selection event at a valid cutoff."""

    kind: _EventKind
    cutoff: datetime
    role: PartitionRole

    def __post_init__(self) -> None:
        _set(self, "kind", _require_member(self.kind, _EventKind, "kind"))
        _require_aware(self.cutoff, "cutoff")
        _set(self, "role", _require_member(self.role, PartitionRole, "role"))

    def _canonical_key(self) -> tuple[datetime, int, int]:
        return (
            _timing.instant_key(self.cutoff, "cutoff"),
            _EVENT_KIND_ORDER[self.kind],
            _ROLE_ORDER[self.role],
        )


@dataclass(frozen=True)
class ObservationRefitCadence:
    """The declarative observation-count refit cadence (no instants yet)."""

    every_observations: int
    role: PartitionRole

    def __post_init__(self) -> None:
        _require_positive_int(
            self.every_observations, "ObservationRefitCadence.every_observations"
        )
        _set(
            self,
            "role",
            _require_member(self.role, PartitionRole, "ObservationRefitCadence.role"),
        )


@dataclass(frozen=True)
class TemporalResearchProtocol:
    """The one canonical temporal declaration runtime consumes.

    Declarative only: no fitting logic, no estimator references, no data.
    Construction validates the whole object in declaration order with the
    typed errors; a cadence-bearing ``fit_events`` grid validates
    shape-only at construction and fully at the one resolution step.
    """

    evaluation_learning: EvaluationLearning
    partitions: Sequence[TemporalPartition]
    target_horizon: timedelta
    estimand: str
    history_policy: HistoryPolicy | None
    selection_history: HistoryPolicy | None
    fit_events: Sequence[FitEvent] | ObservationRefitCadence
    preprocessing_policy: PreprocessingPolicy
    selection_policy: SelectionPolicy

    def __post_init__(self) -> None:
        _set(
            self,
            "evaluation_learning",
            _require_member(
                self.evaluation_learning, EvaluationLearning, "evaluation_learning"
            ),
        )
        _set(self, "partitions", self._validated_partitions(self.partitions))
        if not isinstance(
            self.target_horizon, timedelta
        ) or self.target_horizon <= timedelta(0):
            raise ProtocolStructureError(
                "target_horizon must be a strictly positive timedelta; got "
                f"{type(self.target_horizon).__name__}: {self.target_horizon!r}"
            )
        _require_non_blank_str(self.estimand, "estimand")
        for name, policy in (
            ("history_policy", self.history_policy),
            ("selection_history", self.selection_history),
        ):
            if policy is not None and not isinstance(policy, HistoryPolicy):
                raise ProtocolStructureError(
                    f"{name} must be a HistoryPolicy or None; got "
                    f"{type(policy).__name__}: {policy!r}"
                )
        _set(
            self,
            "preprocessing_policy",
            _require_member(
                self.preprocessing_policy, PreprocessingPolicy, "preprocessing_policy"
            ),
        )
        _set(
            self,
            "selection_policy",
            _require_member(self.selection_policy, SelectionPolicy, "selection_policy"),
        )
        self._validate_selection_coherence()
        _set(self, "fit_events", self._validated_fit_events(self.fit_events))

    # -- validation ------------------------------------------------------- #

    @staticmethod
    def _validated_partitions(partitions: Any) -> tuple[TemporalPartition, ...]:
        if isinstance(partitions, (str, bytes)) or not isinstance(partitions, Sequence):
            raise ProtocolStructureError(
                "partitions must be a sequence of TemporalPartition objects; "
                f"got {type(partitions).__name__}: {partitions!r}"
            )
        normalized = tuple(partitions)
        if not normalized:
            raise ProtocolStructureError(
                "partitions must declare at least one TemporalPartition"
            )
        for partition in normalized:
            if not isinstance(partition, TemporalPartition):
                raise ProtocolStructureError(
                    "partitions must be TemporalPartition instances; got "
                    f"{type(partition).__name__}: {partition!r}"
                )
        for earlier, later in pairwise(normalized):
            # Law 3 chronology: strictly ordered spans; exact boundary
            # adjacency (an earlier span ending exactly where the later
            # opens, rule 5's TRAIN/VALIDATION seam) is lawful.
            if _timing.instant_key(earlier.end, "partition end") > _timing.instant_key(
                later.start, "partition start"
            ):
                raise ProtocolStructureError(
                    "partitions must be chronological and non-overlapping on "
                    f"the single clock: {earlier.role.name} ends "
                    f"{earlier.end.isoformat()} not before {later.role.name} "
                    f"starts {later.start.isoformat()}"
                )
        return normalized

    def _validate_selection_coherence(self) -> None:
        selection_declared = self.selection_history is not None
        none_policy = self.selection_policy is SelectionPolicy.NONE
        if selection_declared == none_policy:
            state = (
                "a selection history policy beside SelectionPolicy.NONE"
                if selection_declared
                else "SelectionPolicy.VALIDATION_ONLY_AT_DECLARED_EVENTS "
                "beside a None selection_history"
            )
            raise ProtocolStructureError(
                f"selection coherence rejects {state}: selection_history and "
                "selection_policy must be declared together"
            )

    def _validated_fit_events(
        self, fit_events: Any
    ) -> tuple[FitEvent, ...] | ObservationRefitCadence:
        if isinstance(fit_events, ObservationRefitCadence):
            return fit_events
        if isinstance(fit_events, (str, bytes)) or not isinstance(fit_events, Sequence):
            raise ProtocolStructureError(
                "fit_events must be a sequence of FitEvent objects or one "
                f"ObservationRefitCadence; got {type(fit_events).__name__}: "
                f"{fit_events!r}"
            )
        events = tuple(fit_events)
        for event in events:
            if not isinstance(event, FitEvent):
                raise ProtocolStructureError(
                    "fit_events entries must be FitEvent instances; got "
                    f"{type(event).__name__}: {event!r}"
                )
        if not events:
            return events
        if self.history_policy is None:
            raise ProtocolStructureError(
                "a strategy with any fit event must declare a history policy; "
                "history_policy None is valid only for a strategy that fits "
                "nothing"
            )
        ordered = tuple(sorted(events, key=FitEvent._canonical_key))
        for earlier, later in pairwise(ordered):
            if earlier._canonical_key() == later._canonical_key():
                raise ProtocolStructureError(
                    "fit_events contains a duplicated event at cutoff "
                    f"{earlier.cutoff.isoformat()}; the declared grid is "
                    "deduplicated evidence"
                )
        spans = [
            (
                _timing.instant_key(partition.start, "partition start"),
                _timing.instant_key(partition.end, "partition end"),
            )
            for partition in self.partitions
        ]
        for event in ordered:
            key = _timing.instant_key(event.cutoff, "cutoff")
            if not any(start <= key <= end for start, end in spans):
                raise ProtocolStructureError(
                    "every fit-event cutoff must lie inside the union of the "
                    f"declared partition spans; the {event.kind.value} event "
                    f"at {event.cutoff.isoformat()} escapes the research "
                    "timeline"
                )
        if any(event.kind is _EventKind.SELECTION for event in ordered) and (
            self.selection_policy
            is not SelectionPolicy.VALIDATION_ONLY_AT_DECLARED_EVENTS
        ):
            raise ProtocolStructureError(
                "a SELECTION fit event beside SelectionPolicy.NONE "
                "rejects: selection events require "
                "VALIDATION_ONLY_AT_DECLARED_EVENTS and a declared "
                "selection_history"
            )
        if self.evaluation_learning is EvaluationLearning.FIXED_ORIGIN:
            span_start, span_end = self._evaluation_span()
            for event in ordered:
                key = _timing.instant_key(event.cutoff, "cutoff")
                if span_start <= key < span_end:
                    raise ProtocolStructureError(
                        "FIXED_ORIGIN admits no learning event inside the "
                        f"evaluation span; the {event.kind.value} event at "
                        f"{event.cutoff.isoformat()} is in-span"
                    )
        return ordered

    def _evaluation_span(self) -> tuple[datetime, datetime]:
        """The evaluation span implied by the declared TEST partition."""
        for partition in self.partitions:
            if partition.role is PartitionRole.TEST:
                return (
                    _timing.instant_key(partition.start, "span start"),
                    _timing.instant_key(partition.end, "span end"),
                )
        return (
            _timing.instant_key(self.partitions[0].start, "span start"),
            _timing.instant_key(self.partitions[-1].end, "span end"),
        )


@dataclass(frozen=True)
class ProtocolProvenance:
    """The per-strategy resolved protocol image runtime consumes opaquely.

    Records the resolved canonical protocol only — never the declaration
    route and never the pre-resolution shape — plus, per declared fit
    event actually executed, the public forecasting-lifecycle ``ForecastProvenance``
    produced by the ``fitting_provenance`` route (reused, never
    paralleled protocol-side).
    """

    protocol: TemporalResearchProtocol
    fit_provenances: tuple[_forecasting.ForecastProvenance, ...] = ()


# --------------------------------------------------------------------------- #
# Advanced composition layer — declare once, override sparsely.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StrategyProtocolOverride:
    """A sparse per-strategy protocol override; ``None`` inherits the common.

    There is deliberately no field for ``partitions``,
    ``evaluation_learning``, ``estimand``, or ``target_horizon``: the
    common decision and evaluation anchors are non-overridable.
    """

    history_policy: HistoryPolicy | None = None
    selection_history: HistoryPolicy | None = None
    fit_events: Sequence[FitEvent] | ObservationRefitCadence | None = None
    preprocessing_policy: PreprocessingPolicy | None = None
    selection_policy: SelectionPolicy | None = None

    def __post_init__(self) -> None:
        if self.history_policy is not None and not isinstance(
            self.history_policy, HistoryPolicy
        ):
            raise ProtocolStructureError(
                "history_policy override must be a HistoryPolicy or None; "
                f"got {type(self.history_policy).__name__}"
            )
        if self.selection_history is not None and not isinstance(
            self.selection_history, HistoryPolicy
        ):
            raise ProtocolStructureError(
                "selection_history override must be a HistoryPolicy or None; "
                f"got {type(self.selection_history).__name__}"
            )
        if self.fit_events is not None:
            lawful = isinstance(self.fit_events, ObservationRefitCadence) or (
                not isinstance(self.fit_events, (str, bytes))
                and isinstance(self.fit_events, Sequence)
                and all(isinstance(event, FitEvent) for event in self.fit_events)
            )
            if not lawful:
                raise ProtocolStructureError(
                    "fit_events override must be a FitEvent sequence or one "
                    f"ObservationRefitCadence; got {self.fit_events!r}"
                )
        if self.preprocessing_policy is not None:
            _require_member(
                self.preprocessing_policy,
                PreprocessingPolicy,
                "preprocessing_policy override",
            )
        if self.selection_policy is not None:
            _require_member(
                self.selection_policy,
                SelectionPolicy,
                "selection_policy override",
            )


@dataclass(frozen=True)
class ExperimentTemporalDeclaration:
    """The advanced experiment-level declaration: common plus sparse overrides."""

    common: TemporalResearchProtocol
    strategy_overrides: Sequence[tuple[str, StrategyProtocolOverride]]

    def __post_init__(self) -> None:
        if not isinstance(self.common, TemporalResearchProtocol):
            raise ProtocolStructureError(
                "common must be a TemporalResearchProtocol; got "
                f"{type(self.common).__name__}: {self.common!r}"
            )
        _set(
            self,
            "strategy_overrides",
            _require_pairs(
                self.strategy_overrides,
                StrategyProtocolOverride,
                "strategy_overrides",
            ),
        )


# --------------------------------------------------------------------------- #
# Ordinary authoring layer — the one concise researcher-facing surface.
# --------------------------------------------------------------------------- #


class _AbsenceToken:
    """The explicit absence token (never Python ``None``)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "none"


class _AtEachFitToken:
    """The preprocessing-at-every-fit token."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "at_each_fit"


none = _AbsenceToken()
at_each_fit = _AtEachFitToken()


@dataclass(frozen=True)
class WindowSpec:
    """A declared ordinary window: basis, exact length, and required mode."""

    basis: WindowBasis
    length: CalendarSpan | timedelta | int
    mode: WindowMode

    def __post_init__(self) -> None:
        _set(self, "basis", _require_member(self.basis, WindowBasis, "basis"))
        _set(self, "mode", _require_member(self.mode, WindowMode, "mode"))
        if self.basis is WindowBasis.CALENDAR:
            lawful = isinstance(self.length, CalendarSpan) or (
                isinstance(self.length, timedelta) and self.length > timedelta(0)
            )
            if not lawful:
                raise ProtocolStructureError(
                    "a CALENDAR window requires a CalendarSpan or strictly "
                    "positive timedelta length; got "
                    f"{type(self.length).__name__}: {self.length!r}"
                )
        else:
            _require_positive_int(self.length, "OBSERVATIONS window length")


@dataclass(frozen=True)
class RefitCadence:
    """A calendar refit cadence: one refit per ``months``-month period end."""

    months: int

    def __post_init__(self) -> None:
        _require_positive_int(self.months, "RefitCadence.months")


annual = RefitCadence(months=12)
quarterly = RefitCadence(months=3)
monthly = RefitCadence(months=1)


def window(
    *,
    years: int = 0,
    months: int = 0,
    observations: int = 0,
    mode: WindowMode,
) -> WindowSpec:
    """The one ordinary window constructor; exactly one length is supplied."""
    supplied = [
        name
        for name, value in (
            ("years", years),
            ("months", months),
            ("observations", observations),
        )
        if value != 0
    ]
    if len(supplied) != 1:
        raise ProtocolStructureError(
            "window(..) requires exactly one of years/months/observations; "
            f"supplied: {sorted(supplied) or 'none'}"
        )
    _require_member(mode, WindowMode, "mode")
    name = supplied[0]
    value = {"years": years, "months": months, "observations": observations}[name]
    _require_positive_int(value, f"window {name}")
    if name == "years":
        return WindowSpec(
            basis=WindowBasis.CALENDAR,
            length=CalendarSpan(years=value, months=0),
            mode=mode,
        )
    if name == "months":
        return WindowSpec(
            basis=WindowBasis.CALENDAR,
            length=CalendarSpan(years=0, months=value),
            mode=mode,
        )
    return WindowSpec(basis=WindowBasis.OBSERVATIONS, length=value, mode=mode)


_SPAN_PATTERN = re.compile(r"^(\d{4})-(\d{2})-(\d{2})\.\.(\d{4})-(\d{2})-(\d{2})$")
_LAST_TIME_OF_DAY = (23, 59, 59, 999999)


def _parse_evaluation_span(literal: Any, zone: Any) -> tuple[datetime, datetime]:
    """Parse the public ``YYYY-MM-DD..YYYY-MM-DD`` literal in ``zone``."""
    if not isinstance(literal, str):
        raise ProtocolStructureError(
            "evaluation_span must be the literal string "
            f"'YYYY-MM-DD..YYYY-MM-DD'; got {type(literal).__name__}: "
            f"{literal!r}"
        )
    match = _SPAN_PATTERN.match(literal)
    if match is None:
        raise ProtocolStructureError(
            "evaluation_span must match 'YYYY-MM-DD..YYYY-MM-DD' exactly; "
            f"got {literal!r}"
        )
    try:
        start = datetime(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            0,
            0,
            0,
            0,
            tzinfo=zone,
        )
        end = datetime(
            int(match.group(4)),
            int(match.group(5)),
            int(match.group(6)),
            *_LAST_TIME_OF_DAY,
            tzinfo=zone,
        )
    except ValueError as exc:
        raise ProtocolStructureError(
            "evaluation_span carries an impossible calendar boundary: "
            f"{literal!r} ({exc})"
        ) from exc
    if _timing.instant_key(start, "span start") >= _timing.instant_key(end, "span end"):
        raise ProtocolStructureError(
            "evaluation_span must be a positive span: start < end on the "
            f"single clock; got {literal!r}"
        )
    return start, end


def _validated_zone(zone: Any) -> Any:
    """The one declared zone; None or a naive substitute rejects."""
    _timing.to_instant(datetime(2000, 1, 1, tzinfo=zone), "timezone")
    return zone


@dataclass(frozen=True)
class StrategyTemporalDeclaration:
    """The ordinary per-strategy declaration: four strategy-neutral fields.

    All four are required; absence is declared with the public ``none``
    token, never Python ``None`` and never inferred from the strategy.
    """

    estimation: WindowSpec | _AbsenceToken
    selection: WindowSpec | _AbsenceToken
    refit: RefitCadence | _AbsenceToken
    preprocessing: _AtEachFitToken | _AbsenceToken

    def __post_init__(self) -> None:
        checks = (
            ("estimation", self.estimation, (WindowSpec, _AbsenceToken)),
            ("selection", self.selection, (WindowSpec, _AbsenceToken)),
            ("refit", self.refit, (RefitCadence, _AbsenceToken)),
            (
                "preprocessing",
                self.preprocessing,
                (_AtEachFitToken, _AbsenceToken),
            ),
        )
        for name, value, lawful in checks:
            if isinstance(value, lawful):
                continue
            if value is None:
                raise ProtocolStructureError(
                    f"{name} requires its explicit declaration or the public "
                    "none token; Python None rejects (absence is declared, "
                    "never defaulted)"
                )
            raise ProtocolStructureError(
                f"{name} must be a declared {name} value or the none token; "
                f"got {type(value).__name__}: {value!r}"
            )
        absent = none
        if self.estimation is absent and any(
            (
                self.selection is not absent,
                self.refit is not absent,
                self.preprocessing is not absent,
            )
        ):
            raise ProtocolStructureError(
                "cross-field rule: selection, refit, and preprocessing each "
                "require a declared estimation window"
            )


@dataclass(frozen=True)
class TemporalDesign:
    """The ordinary experiment-level temporal declaration (the concise surface)."""

    evaluation_span: str
    evaluation_learning: EvaluationLearning
    estimand: str
    target_horizon: timedelta
    timezone: Any
    strategies: Sequence[tuple[str, StrategyTemporalDeclaration]]

    def __post_init__(self) -> None:
        _set(
            self,
            "evaluation_learning",
            _require_member(
                self.evaluation_learning, EvaluationLearning, "evaluation_learning"
            ),
        )
        _set(self, "timezone", _validated_zone(self.timezone))
        _require_non_blank_str(self.estimand, "estimand")
        if not isinstance(
            self.target_horizon, timedelta
        ) or self.target_horizon <= timedelta(0):
            raise ProtocolStructureError(
                "target_horizon must be a strictly positive timedelta; got "
                f"{self.target_horizon!r}"
            )
        span_start, span_end = _parse_evaluation_span(
            self.evaluation_span, self.timezone
        )
        _set(
            self,
            "strategies",
            _require_pairs(self.strategies, StrategyTemporalDeclaration, "strategies"),
        )
        _set(self, "_span_start", span_start)
        _set(self, "_span_end", span_end)

    def span(self) -> tuple[datetime, datetime]:
        """The parsed evaluation-span instants (fixed at construction)."""
        return self._span_start, self._span_end  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Advanced constructors — explicit low-level canonical pieces.
# --------------------------------------------------------------------------- #


def rolling_calendar_history(window_length: timedelta) -> HistoryPolicy:
    """ROLLING on CALENDAR_DURATION with an exact elapsed minimum history."""
    _require_positive_timedelta(window_length, "window_length")
    return HistoryPolicy(
        mode=HistoryMode.ROLLING,
        basis=HistoryBasis.CALENDAR_DURATION,
        minimum_history=window_length,
    )


def expanding_calendar_history(first_window: timedelta) -> HistoryPolicy:
    """EXPANDING on CALENDAR_DURATION with the first window as the floor."""
    _require_positive_timedelta(first_window, "first_window")
    return HistoryPolicy(
        mode=HistoryMode.EXPANDING,
        basis=HistoryBasis.CALENDAR_DURATION,
        minimum_history=first_window,
    )


def rolling_observation_history(window_count: int) -> HistoryPolicy:
    """ROLLING on OBSERVATION_COUNT with an exact trailing record count."""
    _require_positive_int(window_count, "window_count")
    return HistoryPolicy(
        mode=HistoryMode.ROLLING,
        basis=HistoryBasis.OBSERVATION_COUNT,
        minimum_history=window_count,
    )


def expanding_observation_history(first_window: int) -> HistoryPolicy:
    """EXPANDING on OBSERVATION_COUNT with the first count as the floor."""
    _require_positive_int(first_window, "first_window")
    return HistoryPolicy(
        mode=HistoryMode.EXPANDING,
        basis=HistoryBasis.OBSERVATION_COUNT,
        minimum_history=first_window,
    )


def month_end_refits(
    start: datetime, end: datetime, every_months: int, role: PartitionRole
) -> tuple[FitEvent, ...]:
    """A calendar month-end refit grid on the public calendar interface."""
    _require_aware(start, "start")
    _require_aware(end, "end")
    _require_positive_int(every_months, "every_months")
    stamped_role = _require_member(role, PartitionRole, "role")
    zone = start.tzinfo
    start_key = _timing.instant_key(start, "start")
    end_key = _timing.instant_key(end, "end")
    events: list[FitEvent] = []
    index = _month_index(start.year, start.month)
    while True:
        cutoff = _month_end_wall(index, zone)
        key = _timing.instant_key(cutoff, "cutoff")
        if key > end_key:
            break
        if key >= start_key:
            events.append(
                FitEvent(kind=_EventKind.REFIT, cutoff=cutoff, role=stamped_role)
            )
        index += every_months
    return tuple(events)


def quarter_end_refits(
    start: datetime, end: datetime, every_quarters: int, role: PartitionRole
) -> tuple[FitEvent, ...]:
    """A calendar quarter-end refit grid on the public calendar interface."""
    _require_aware(start, "start")
    _require_aware(end, "end")
    _require_positive_int(every_quarters, "every_quarters")
    stamped_role = _require_member(role, PartitionRole, "role")
    zone = start.tzinfo
    start_key = _timing.instant_key(start, "start")
    end_key = _timing.instant_key(end, "end")
    quarter_index = start.year * 4 + (start.month - 1) // 3
    events: list[FitEvent] = []
    while True:
        year, quarter_zero = divmod(quarter_index, 4)
        cutoff = _calendar.quarter_end_instant(year, quarter_zero + 1, zone).astimezone(
            zone
        )
        key = _timing.instant_key(cutoff, "cutoff")
        if key > end_key:
            break
        if key >= start_key:
            events.append(
                FitEvent(kind=_EventKind.REFIT, cutoff=cutoff, role=stamped_role)
            )
        quarter_index += every_quarters
    return tuple(events)


def observation_refits(
    every_observations: int, role: PartitionRole
) -> ObservationRefitCadence:
    """The observation-count refit cadence (instants materialize at resolution)."""
    return ObservationRefitCadence(every_observations=every_observations, role=role)


# --------------------------------------------------------------------------- #
# The one resolution step.
# --------------------------------------------------------------------------- #


_HISTORY_MODE_FROM_WINDOW_MODE = {
    WindowMode.ROLLING: HistoryMode.ROLLING,
    WindowMode.EXPANDING: HistoryMode.EXPANDING,
}
_HISTORY_BASIS_FROM_WINDOW_BASIS = {
    WindowBasis.CALENDAR: HistoryBasis.CALENDAR_DURATION,
    WindowBasis.OBSERVATIONS: HistoryBasis.OBSERVATION_COUNT,
}


def _validated_names(strategy_names: Any) -> tuple[str, ...]:
    if isinstance(strategy_names, (str, bytes)) or not isinstance(
        strategy_names, Sequence
    ):
        raise ProtocolStructureError(
            "strategy_names must be a sequence of strategy id strings; got "
            f"{type(strategy_names).__name__}: {strategy_names!r}"
        )
    names = tuple(
        _require_strategy_id(name, "strategy_names entry") for name in strategy_names
    )
    if len(set(names)) != len(names):
        raise ProtocolStructureError(
            "strategy_names must contain exactly one record timeline per "
            f"named strategy; duplicated names in {list(names)!r} reject"
        )
    return names


def _validated_timelines(
    record_instants_by_strategy: Any, names: tuple[str, ...]
) -> dict[str, tuple[list[datetime], tuple[datetime, ...]]]:
    if not isinstance(record_instants_by_strategy, Mapping):
        raise ProtocolStructureError(
            "record_instants_by_strategy must be a mapping of strategy id "
            f"to admitted instant sequence; got "
            f"{type(record_instants_by_strategy).__name__}"
        )
    provided = set(record_instants_by_strategy)
    expected = set(names)
    if provided != expected:
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        raise ProtocolStructureError(
            "resolution requires exact per-strategy record-timeline "
            f"coverage; missing keys {missing}, unmatched keys {extra}"
        )
    timelines: dict[str, tuple[list[datetime], tuple[datetime, ...]]] = {}
    for name in names:
        instants = _sequence_instants(
            record_instants_by_strategy[name],
            f"record_instants_by_strategy[{name!r}]",
        )
        keys = [
            _timing.instant_key(instant, f"record instant for {name!r}")
            for instant in instants
        ]
        for earlier, later in pairwise(keys):
            if earlier >= later:
                raise ProtocolStructureError(
                    "each admitted record sequence must be strictly "
                    f"increasing on the single clock; strategy {name!r} "
                    f"descends or repeats at {later.isoformat()}"
                )
        timelines[name] = (keys, instants)
    return timelines


def _materialize_cadence(
    cadence: ObservationRefitCadence,
    partitions: Sequence[TemporalPartition],
    own_keys: list[datetime],
    own_instants: tuple[datetime, ...],
) -> tuple[FitEvent, ...]:
    """Every N-th admitted record, deduplicated, inside the research timeline."""
    spans = [
        (
            _timing.instant_key(partition.start, "partition start"),
            _timing.instant_key(partition.end, "partition end"),
        )
        for partition in partitions
    ]
    events: list[FitEvent] = []
    seen: set[datetime] = set()
    step = cadence.every_observations
    for position in range(step - 1, len(own_keys), step):
        key = own_keys[position]
        if key in seen:
            continue
        seen.add(key)
        if not any(start <= key <= end for start, end in spans):
            continue
        events.append(
            FitEvent(
                kind=_EventKind.REFIT,
                cutoff=own_instants[position],
                role=cadence.role,
            )
        )
    return tuple(events)


def _grid_instant(span_start: datetime, months: int, step: int, zone: Any) -> datetime:
    """The cadence-grid instant ``step`` periods from the span anchor.

    The declared zone's time axis is partitioned into consecutive
    ``months``-month periods whose boundaries pass through the evaluation
    span start; grid instants are those periods' month-end instants, in
    both directions from the anchor, on the public calendar interface.
    """
    boundary = _shift_months(span_start, step * months)
    previous_microsecond = boundary - timedelta(microseconds=1)
    return _month_end_wall(
        _month_index(previous_microsecond.year, previous_microsecond.month),
        zone,
    )


def _history_policy_from_spec(spec: WindowSpec) -> HistoryPolicy:
    return HistoryPolicy(
        mode=_HISTORY_MODE_FROM_WINDOW_MODE[spec.mode],
        basis=_HISTORY_BASIS_FROM_WINDOW_BASIS[spec.basis],
        minimum_history=spec.length,
    )


def _ordinary_bundles(
    design: TemporalDesign,
    names: tuple[str, ...],
    timelines: Mapping[str, tuple[list[datetime], tuple[datetime, ...]]],
) -> dict[str, dict[str, Any]]:
    """Derive each named strategy's canonical field bundle (the pinned rules)."""
    declared = {key: value for key, value in design.strategies}
    unknown = sorted(set(declared) - set(names))
    if unknown:
        raise ProtocolStructureError(
            "every declared strategy id must be a named strategy; unknown "
            f"ids {unknown} reject fail-closed"
        )
    missing = sorted(set(names) - set(declared))
    if missing:
        raise ProtocolStructureError(
            "every named strategy requires one ordinary declaration pair; "
            f"strategies without a pair {missing} reject fail-closed"
        )
    span_start, span_end = design.span()
    zone = design.timezone
    span_start_key = _timing.instant_key(span_start, "span start")
    span_end_key = _timing.instant_key(span_end, "span end")
    absent = none  # the frozen singleton, not the class
    bundles: dict[str, dict[str, Any]] = {}
    for name in names:
        declaration = declared[name]
        own_keys = timelines[name][0]
        absent = none
        test_partition = TemporalPartition(
            role=PartitionRole.TEST, start=span_start, end=span_end
        )
        if declaration.estimation is absent:
            # The frozen absence semantics: an arm that neither fits nor
            # selects, still carrying every common anchor.
            bundles[name] = {
                "evaluation_learning": design.evaluation_learning,
                "partitions": (test_partition,),
                "target_horizon": design.target_horizon,
                "estimand": design.estimand,
                "history_policy": None,
                "selection_history": None,
                "fit_events": (),
                "preprocessing_policy": PreprocessingPolicy.NONE,
                "selection_policy": SelectionPolicy.NONE,
            }
            continue
        months = 12 if declaration.refit is absent else declaration.refit.months
        step = 0
        while (
            _timing.instant_key(_grid_instant(span_start, months, step, zone), "cutoff")
            >= span_start_key
        ):
            step -= 1
        c_zero = _grid_instant(span_start, months, step, zone)
        estimation_policy = _history_policy_from_spec(declaration.estimation)
        partitions: list[TemporalPartition] = []
        events: list[FitEvent] = [
            FitEvent(
                kind=_EventKind.INITIAL_FIT,
                cutoff=c_zero,
                role=PartitionRole.TRAIN,
            )
        ]
        selection_policy = SelectionPolicy.NONE
        selection_history: HistoryPolicy | None = None
        if declaration.selection is not absent:
            spec = declaration.selection
            selection_history = _history_policy_from_spec(spec)
            selection_policy = SelectionPolicy.VALIDATION_ONLY_AT_DECLARED_EVENTS
            validation_start = _window_start(selection_history, c_zero, own_keys)
            partitions.append(
                TemporalPartition(
                    role=PartitionRole.VALIDATION,
                    start=validation_start,
                    end=c_zero,
                )
            )
            events.append(
                FitEvent(
                    kind=_EventKind.SELECTION,
                    cutoff=c_zero,
                    role=PartitionRole.VALIDATION,
                )
            )
        else:
            validation_start = c_zero
        estimation_end = validation_start
        train_start = _window_start(estimation_policy, estimation_end, own_keys)
        partitions.insert(
            0,
            TemporalPartition(
                role=PartitionRole.TRAIN, start=train_start, end=estimation_end
            ),
        )
        partitions.append(test_partition)
        if (
            design.evaluation_learning is EvaluationLearning.WALK_FORWARD
            and declaration.refit is not absent
        ):
            forward = step + 1
            while True:
                cutoff = _grid_instant(span_start, months, forward, zone)
                key = _timing.instant_key(cutoff, "cutoff")
                if key >= span_end_key:
                    break
                events.append(
                    FitEvent(
                        kind=_EventKind.REFIT,
                        cutoff=cutoff,
                        role=PartitionRole.TRAIN,
                    )
                )
                if declaration.selection is not absent:
                    events.append(
                        FitEvent(
                            kind=_EventKind.SELECTION,
                            cutoff=cutoff,
                            role=PartitionRole.VALIDATION,
                        )
                    )
                forward += 1
        elif declaration.refit is not absent:
            raise ProtocolStructureError(
                "FIXED_ORIGIN admits no learning event inside the evaluation "
                "span; a declared refit cadence necessarily enters the span "
                f"and rejects (strategy {name!r})"
            )
        preprocessing = (
            PreprocessingPolicy.NONE
            if declaration.preprocessing is absent
            else PreprocessingPolicy.TRAIN_ONLY_AT_EACH_FIT
        )
        bundles[name] = {
            "evaluation_learning": design.evaluation_learning,
            "partitions": tuple(partitions),
            "target_horizon": design.target_horizon,
            "estimand": design.estimand,
            "history_policy": estimation_policy,
            "selection_history": selection_history,
            "fit_events": tuple(sorted(events, key=FitEvent._canonical_key)),
            "preprocessing_policy": preprocessing,
            "selection_policy": selection_policy,
        }
    return bundles


def _advanced_bundles(
    declaration: ExperimentTemporalDeclaration, names: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """Splice each declared override over the common protocol's fields."""
    common = declaration.common
    overrides = {key: value for key, value in declaration.strategy_overrides}
    unknown = sorted(set(overrides) - set(names))
    if unknown:
        raise ProtocolStructureError(
            "every override strategy id must be a named strategy; unknown "
            f"override keys {unknown} reject fail-closed"
        )
    bundles: dict[str, dict[str, Any]] = {}
    for name in names:
        override = overrides.get(name)
        if override is None:
            bundles[name] = {
                "evaluation_learning": common.evaluation_learning,
                "partitions": common.partitions,
                "target_horizon": common.target_horizon,
                "estimand": common.estimand,
                "history_policy": common.history_policy,
                "selection_history": common.selection_history,
                "fit_events": common.fit_events,
                "preprocessing_policy": common.preprocessing_policy,
                "selection_policy": common.selection_policy,
            }
            continue
        bundles[name] = {
            # Non-overridable common anchors, copied and asserted.
            "evaluation_learning": common.evaluation_learning,
            "partitions": common.partitions,
            "target_horizon": common.target_horizon,
            "estimand": common.estimand,
            "history_policy": (
                common.history_policy
                if override.history_policy is None
                else override.history_policy
            ),
            "selection_history": (
                common.selection_history
                if override.selection_history is None
                else override.selection_history
            ),
            "fit_events": (
                common.fit_events
                if override.fit_events is None
                else override.fit_events
            ),
            "preprocessing_policy": (
                common.preprocessing_policy
                if override.preprocessing_policy is None
                else override.preprocessing_policy
            ),
            "selection_policy": (
                common.selection_policy
                if override.selection_policy is None
                else override.selection_policy
            ),
        }
    return bundles


def _check_first_decision(
    protocol: TemporalResearchProtocol, own_keys: list[datetime]
) -> None:
    """The initial fit must be declared AND satisfiable — no back-fill.

    Availability follows the authority's research-timeline rule: the strategy's
    required start is its selection-window start where one is declared,
    else its estimation-window start; the strategy's own admitted timeline
    must reach back to it. Every selection event additionally requires at
    least one admitted record strictly inside its consumed window.
    """
    events = protocol.fit_events
    if not isinstance(events, tuple) or not events:
        return
    selection_events = [event for event in events if event.kind is _EventKind.SELECTION]
    estimation_events = [
        event for event in events if event.kind is not _EventKind.SELECTION
    ]
    if selection_events:
        selection_policy = protocol.selection_history
        assert selection_policy is not None  # guaranteed by construction laws
        for event in selection_events:
            if selection_policy.mode is HistoryMode.EXPANDING:
                window_start = _window_start(
                    selection_policy, selection_events[0].cutoff, own_keys
                )
            else:
                window_start = _window_start(selection_policy, event.cutoff, own_keys)
            start_key = _timing.instant_key(window_start, "window start")
            end_key = _timing.instant_key(event.cutoff, "cutoff")
            if not any(start_key < key <= end_key for key in own_keys):
                raise ProtocolStructureError(
                    "selection labels must be realizable at the declared "
                    f"selection cutoff {event.cutoff.isoformat()}: no "
                    "admitted record lies strictly inside the consumed "
                    "validation window"
                )
        return
    if estimation_events and protocol.history_policy is not None:
        window_start = _window_start(
            protocol.history_policy, estimation_events[0].cutoff, own_keys
        )
        start_key = _timing.instant_key(window_start, "window start")
        if not own_keys or own_keys[0] > start_key:
            raise ProtocolStructureError(
                "the initial fit is declared but unsatisfiable: the "
                "strategy's admitted timeline opens "
                f"{own_keys[0].isoformat() if own_keys else 'never'}, after "
                f"the declared minimum history opens "
                f"{window_start.isoformat()}; no back-filled (shortened) "
                "initial window may be invented"
            )


def resolve_temporal_protocols(
    declaration: TemporalDesign | ExperimentTemporalDeclaration,
    strategy_names: Sequence[str],
    record_instants_by_strategy: Mapping[str, Sequence[datetime]],
) -> Mapping[str, TemporalResearchProtocol]:
    """The single normalization step before execution.

    Accepts exactly one of the ordinary :class:`TemporalDesign` or the
    advanced :class:`ExperimentTemporalDeclaration`; produces each named
    strategy's canonical resolved protocol — materializing any observation
    cadence against that strategy's own admitted sequence first, splicing
    declared overrides over the common protocol, and re-validating every
    strategy whole under the same canonical construction rules. Fails closed,
    non-mutating, and order-invariant.
    """
    if not isinstance(declaration, (TemporalDesign, ExperimentTemporalDeclaration)):
        raise ProtocolStructureError(
            "resolution accepts exactly one of TemporalDesign or "
            "ExperimentTemporalDeclaration; got "
            f"{type(declaration).__name__}: {declaration!r}"
        )
    names = _validated_names(strategy_names)
    timelines = _validated_timelines(record_instants_by_strategy, names)
    if isinstance(declaration, TemporalDesign):
        bundles = _ordinary_bundles(declaration, names, timelines)
    else:
        bundles = _advanced_bundles(declaration, names)
    resolved: dict[str, TemporalResearchProtocol] = {}
    for name in names:
        bundle = bundles[name]
        fit_events = bundle["fit_events"]
        own_keys, own_instants = timelines[name]
        if isinstance(fit_events, ObservationRefitCadence):
            fit_events = _materialize_cadence(
                fit_events, bundle["partitions"], own_keys, own_instants
            )
        protocol = TemporalResearchProtocol(
            evaluation_learning=bundle["evaluation_learning"],
            partitions=bundle["partitions"],
            target_horizon=bundle["target_horizon"],
            estimand=bundle["estimand"],
            history_policy=bundle["history_policy"],
            selection_history=bundle["selection_history"],
            fit_events=fit_events,
            preprocessing_policy=bundle["preprocessing_policy"],
            selection_policy=bundle["selection_policy"],
        )
        _check_first_decision(protocol, own_keys)
        resolved[name] = protocol
    return resolved
