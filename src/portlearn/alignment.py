"""Availability-aware alignment of mixed-frequency observations.

This module implements the alignment laws as a thin composition
surface over the frozen observation, timing, and period-calendar
laws: an immutable
:class:`ObservationStore` indexed by ``(series_id, observation_time)``
group, a single visibility query that delegates vintage selection to
the frozen ``vintage_as_of`` operation, an :func:`align` output of the
caller's own vintage records, exactly one convenience month-end
decision-calendar builder composed over the shared period-calendar
substrate, and one grid-chronology validator reusing the frozen
chronology and timestamp errors.

The laws, in summary:

- **Alignment is admission, parameterized by availability.** The only
  question a decision instant asks of any series — daily, monthly, or
  quarterly — is *what was knowable at this instant?* Visibility is
  decided by each record's own declared ``available_time`` through the
  frozen vintage operation and the frozen inclusive admission law;
  this module never re-implements admission, never selects among
  vintages of one observation, and contains no date-matching or
  calendar-proximity join of any kind.
- **Exclusion with explicit absence.** A group with no visible vintage
  at the decision instant contributes no record; the visibility query
  answers ``None`` in that slot — no error, no imputation, no
  placeholder, and never a silently carried stale value.
- **The decision calendar is input, not machinery.** Alignment
  consumes one aware instant; decision grids are researcher inputs.
  One convenience builder ships — month-end instants through the
  shared substrate — and nothing else; schedule machinery is a named,
  deferred seam owned elsewhere.
- **No aggregation, no resampling, no implicit publication lag.**
  Alignment aligns; frequency transformation and bounded carry-forward
  belong to the transforms module and are composed by the researcher,
  declared and auditable, never defaulted here. Availability comes
  from each record's own ``available_time``; this module adds none,
  assumes none, and cannot be configured with one.
- **Chronology of the grid.** Calendar-builder output and any
  researcher-supplied grid must be strictly increasing aware instants;
  non-monotone grids reject with the frozen chronology error, and
  naive instants reject with the frozen timestamp error at every entry
  point.

**RESEARCHER WARNING — align on availability, never on dates.**
Aligning on *observation dates* or series labels, or on naive
``merge_asof``-style joins, can *leak* information whenever
availability differs from observation timing: a macro print dated
before a mid-month decision but published weeks after it is silently
admitted by every date-matching idiom in the ecosystem.
Availability-aware admission through this module is the
*authoritative* path and its only alignment semantics — there is no
date-matching join here to fall back on. Using an older quarter's
value at a monthly decision remains available, but only through the
transforms module's declared, bounded carry-forward transform —
explicit and auditable, never a silent default inside a visibility
query.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from itertools import pairwise
from types import MappingProxyType
from typing import Any

from portlearn.calendar import month_end_instant
from portlearn.observations import (
    AmbiguousObservationError,
    TimedObservation,
    vintage_as_of,
)
from portlearn.timing import (
    InvalidChronologyError,
    instant_key,
    to_instant,
)

__all__ = [
    "GridDeclarationError",
    "ObservationStore",
    "UnknownSeriesError",
    "align",
    "monthly_decision_calendar",
    "require_increasing_instants",
]


class UnknownSeriesError(ValueError):
    """A request names a series or observation the store does not hold.

    Structural, not informational: an unknown series identifier or an
    unknown ``(series_id, observation_time)`` group means the request
    itself is malformed relative to the store — distinct from a known
    group whose latest vintage is not yet visible, which is the honest
    ``None`` absence outcome, never this error.
    """


class GridDeclarationError(ValueError):
    """A decision grid or request is structurally malformed.

    Raised for malformed calendar declarations (bad year/month/count
    shapes), malformed request elements, and grids that are not
    non-empty sequences of entries. Chronology violations and naive
    instants are not declaration failures — they reject with the
    frozen chronology and timestamp errors respectively.
    """


def _observation_key(record: TimedObservation) -> tuple[str, datetime]:
    """The normalized ``(series_id, observation_time)`` group key."""
    return (
        record.series_id,
        instant_key(record.observation_time, "observation_time"),
    )


class ObservationStore:
    """An immutable point-in-time index over declared observations.

    Built from :class:`~portlearn.observations.TimedObservation`
    records under the frozen record discipline: every record's own
    constructor laws apply (aware instants, mandatory availability,
    availability never preceding the observation), and two records
    sharing the full identity triple ``(series_id, observation_time,
    available_time)`` reject with ``AmbiguousObservationError`` — no
    last-write-wins, no value-equality exception. A revision — the
    same observation with a later ``available_time`` — is a separate
    record and is exactly what the store holds; selecting among
    vintages of one observation is the frozen vintage operation's job
    at query time, never the store's at build time.

    The index maps each ``(series_id, observation_time)`` group to its
    records, and each series to its group keys in observation order.
    Inputs are copied at construction, so later mutation of the
    caller's collection cannot change what the store indexed; the
    store exposes no mutation surface of its own. ``provenance`` is a
    read-only mapping recording the store's shape and the availability
    basis: availability is each record's own declared
    ``available_time`` — alignment adds no publication lag.
    """

    _groups: Mapping[tuple[str, datetime], tuple[TimedObservation, ...]]
    _series_keys: Mapping[str, tuple[tuple[str, datetime], ...]]
    _provenance: Mapping[str, Any]

    def __init__(self, records: Iterable[TimedObservation]) -> None:
        materialized = tuple(records)
        grouped: dict[tuple[str, datetime], list[TimedObservation]] = {}
        seen_identities: set[tuple[str, datetime, datetime]] = set()
        for record in materialized:
            if not isinstance(record, TimedObservation):
                raise TypeError(
                    "the store indexes declared observations only; got "
                    f"{type(record).__name__}: {record!r}. Submit "
                    "TimedObservation records."
                )
            identity = (
                record.series_id,
                instant_key(record.observation_time, "observation_time"),
                instant_key(record.available_time, "available_time"),
            )
            if identity in seen_identities:
                raise AmbiguousObservationError(
                    "ambiguous observation input: two records share the "
                    "full identity triple (series_id="
                    f"{record.series_id!r}, observation_time="
                    f"{record.observation_time.isoformat()}, "
                    f"available_time={record.available_time.isoformat()}) "
                    "— the identity triple is a key, so there is no "
                    "last-write-wins and no value-equality exception."
                )
            seen_identities.add(identity)
            grouped.setdefault(_observation_key(record), []).append(record)

        series_keys: dict[str, list[tuple[str, datetime]]] = {}
        for key in grouped:
            series_keys.setdefault(key[0], []).append(key)
        self._groups = MappingProxyType(
            {key: tuple(members) for key, members in grouped.items()}
        )
        self._series_keys = MappingProxyType(
            {series_id: tuple(sorted(keys))
             for series_id, keys in series_keys.items()}
        )
        self._provenance = MappingProxyType(
            {
                "record_count": len(materialized),
                "series_count": len(self._series_keys),
                "observation_group_count": len(self._groups),
                "availability_basis": (
                    "record available_time declared at ingestion; "
                    "alignment adds no publication lag"
                ),
            }
        )

    @property
    def provenance(self) -> Mapping[str, Any]:
        """A read-only record of the store's shape and availability
        basis (availability is each record's own declared
        ``available_time``; the store assumes none)."""
        return self._provenance

    def _visible_group_record(
        self, key: tuple[str, datetime], decision: datetime
    ) -> TimedObservation | None:
        """The frozen vintage operation's answer for one group."""
        visible = vintage_as_of(self._groups[key], decision)
        if visible is not None:
            return visible
        return None

    def _require_series_keys(
        self, series_id: str
    ) -> tuple[tuple[str, datetime], ...]:
        keys = self._series_keys.get(series_id)
        if keys is None:
            raise UnknownSeriesError(
                f"unknown series {series_id!r}: the store holds no "
                "observation group for this identifier, so no vintage of "
                "it can exist at any decision instant. A series that is "
                "known but not yet visible at a decision instant is the "
                "None absence outcome instead — never this error."
            )
        return keys

    def visible_at(
        self, decision_instant: Any, series_ids: Sequence[str]
    ) -> list[TimedObservation | None]:
        """What each requested series shows at one decision instant.

        For each requested series identifier, in request order, the
        latest of its observation groups that has a visible vintage at
        ``decision_instant`` — selected by issuing the frozen
        ``vintage_as_of`` operation per group and admitting by the
        frozen inclusive law (``available_time <= decision_instant``).
        A series with no visible vintage at the instant answers
        ``None`` in its slot: explicit absence, not an error, not an
        imputation, and never a silently carried stale value. The
        store never selects among vintages of one observation and
        never re-implements admission.

        ``decision_instant`` must be an aware instant (naive inputs
        reject with ``NaiveTimestampError``). Each identifier must be
        a non-blank string (malformed shape rejects with
        ``GridDeclarationError``); an identifier the store does not
        hold rejects with ``UnknownSeriesError`` — a structural
        failure, distinct from absence.
        """
        decision = to_instant(decision_instant, "decision_instant")
        if isinstance(series_ids, (str, bytes)):
            raise GridDeclarationError(
                "series_ids must be a sequence of series identifiers, not "
                f"a single {type(series_ids).__name__}: {series_ids!r}. "
                "Submit one identifier per requested slot."
            )
        try:
            requested = tuple(series_ids)
        except TypeError as error:
            raise GridDeclarationError(
                "series_ids must be an iterable of series identifiers; "
                f"got {type(series_ids).__name__}: {series_ids!r}."
            ) from error
        answers: list[TimedObservation | None] = []
        for series_id in requested:
            if not isinstance(series_id, str) or not series_id.strip():
                raise GridDeclarationError(
                    "each requested series identifier must be a non-empty, "
                    f"non-blank string; got {series_id!r}. A blank "
                    "identifier names no series."
                )
            keys = self._require_series_keys(series_id)
            latest: TimedObservation | None = None
            for key in reversed(keys):
                record = self._visible_group_record(key, decision)
                if record is not None:
                    latest = record
                    break
            answers.append(latest)
        return answers


def _materialize_requests(requested_groups: Iterable[Any]) -> tuple[Any, ...]:
    """Requests as a tuple, rejecting non-sequences fail-closed."""
    if isinstance(requested_groups, (str, bytes)):
        raise GridDeclarationError(
            "requested_groups must be a sequence of "
            "(series_id, observation_time) pairs, not a single "
            f"{type(requested_groups).__name__}: {requested_groups!r}."
        )
    try:
        return tuple(requested_groups)
    except TypeError as error:
        raise GridDeclarationError(
            "requested_groups must be an iterable of "
            "(series_id, observation_time) pairs; got "
            f"{type(requested_groups).__name__}: {requested_groups!r}."
        ) from error


def _parse_group_request(request: Any) -> tuple[str, Any]:
    """One ``(series_id, observation_time)`` request, shape-validated."""
    if not isinstance(request, (tuple, list)) or len(request) != 2:
        raise GridDeclarationError(
            "each requested group must be a (series_id, observation_time) "
            f"pair; got {type(request).__name__}: {request!r}. Alignment "
            "admits named observations, so the request must name both the "
            "series and the observation being aligned."
        )
    series_id, observation_time = request
    if not isinstance(series_id, str) or not series_id.strip():
        raise GridDeclarationError(
            "each requested group's series_id must be a non-empty, "
            f"non-blank string; got {series_id!r}. A blank identifier "
            "names no series."
        )
    return series_id, observation_time


def align(
    store: ObservationStore,
    decision_instant: Any,
    requested_groups: Iterable[Any],
) -> list[TimedObservation]:
    """The admissible vintage records for one decision instant.

    For each requested ``(series_id, observation_time)`` group, in
    request order, the frozen vintage operation's visible vintage at
    ``decision_instant`` (the latest ``available_time`` at or before
    the instant — availability exactly at the decision admits). A
    group with no visible vintage contributes no record: explicit
    absence, never an error, never an imputation, never a carried
    value. The output is deterministic — the caller's declared request
    order with absence removed — and holds the store's own record
    objects.

    The output is the caller's to submit to the frozen
    ``InformationSet(items, as_of=decision_instant)`` constructor for
    fail-closed admission; this function never constructs an
    information set and never bypasses its constructor laws.

    ``decision_instant`` must be an aware instant and each requested
    observation an aware instant (naive inputs reject with
    ``NaiveTimestampError``); malformed request shape rejects with
    ``GridDeclarationError``; a request naming a series or observation
    the store does not hold rejects with ``UnknownSeriesError``.
    """
    if not isinstance(store, ObservationStore):
        raise TypeError(
            "align composes an ObservationStore; got "
            f"{type(store).__name__}: {store!r}. Build the store from "
            "declared observations first."
        )
    decision = to_instant(decision_instant, "decision_instant")
    requests = _materialize_requests(requested_groups)
    admitted: list[TimedObservation] = []
    for request in requests:
        series_id, observation_time = _parse_group_request(request)
        key = (
            series_id,
            instant_key(observation_time, "observation_time"),
        )
        if key not in store._groups:
            raise UnknownSeriesError(
                f"unknown observation group ({series_id!r}, "
                f"{to_instant(observation_time, 'observation_time').isoformat()})"
                ": the store holds no such (series_id, observation_time) "
                "group, so the request is malformed relative to the store. "
                "A held group whose vintage is not yet visible at the "
                "decision instant is the honest absence — no record — "
                "instead, never this error."
            )
        visible = store._visible_group_record(key, decision)
        if visible is None:
            continue
        admitted.append(visible)
    return admitted


def require_increasing_instants(grid: Iterable[Any]) -> None:
    """Assert a decision grid is strictly increasing aware instants.

    Every grid a researcher supplies — and every grid the calendar
    builder emits — must be a non-empty sequence of aware instants in
    strictly increasing order: a decision calendar that repeats or
    reverses an instant is malformed. Equal adjacent instants reject
    (strict increase); a non-monotone grid rejects with the frozen
    ``InvalidChronologyError``; a naive entry rejects with the frozen
    ``NaiveTimestampError``; a non-sequence, a string, or an empty
    grid rejects with ``GridDeclarationError``. Comparisons use
    normalized instants, so equal instants expressed in different
    timezones still reject as equal. Returns ``None`` on success.
    """
    if isinstance(grid, (str, bytes)):
        raise GridDeclarationError(
            "a decision grid must be a sequence of aware instants, not a "
            f"single {type(grid).__name__}: {grid!r}."
        )
    try:
        entries = tuple(grid)
    except TypeError as error:
        raise GridDeclarationError(
            "a decision grid must be an iterable of aware instants; got "
            f"{type(grid).__name__}: {grid!r}."
        ) from error
    if not entries:
        raise GridDeclarationError(
            "a decision grid must contain at least one aware instant; an "
            "empty grid names no decisions."
        )
    normalized = [to_instant(entry, "grid entry") for entry in entries]
    for previous, current in pairwise(normalized):
        if not previous < current:
            raise InvalidChronologyError(
                "chronology violation: a decision grid must be strictly "
                "increasing aware instants, but "
                f"{current.isoformat()} does not strictly follow "
                f"{previous.isoformat()}. A decision calendar that "
                "repeats or reverses an instant names no valid sequence "
                "of decisions."
            )


def _parse_start(start_year_month: Any) -> tuple[int, int]:
    """The ``(year, month)`` calendar start, shape-validated."""
    if not isinstance(start_year_month, (tuple, list)) or len(
        start_year_month
    ) != 2:
        raise GridDeclarationError(
            "start_year_month must be a (year, month) pair naming the "
            "first month of the calendar; got "
            f"{type(start_year_month).__name__}: {start_year_month!r}."
        )
    year, month = start_year_month
    for name, value in (("year", year), ("month", month)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise GridDeclarationError(
                f"start_year_month's {name} must be an integer; got "
                f"{type(value).__name__}: {value!r}."
            )
    if not 1 <= year <= 9999:
        raise GridDeclarationError(
            f"start_year_month's year must be within 1-9999; got {year!r}."
        )
    if not 1 <= month <= 12:
        raise GridDeclarationError(
            "start_year_month's month must be an integer 1-12 naming a "
            f"calendar month; got {month!r}."
        )
    return year, month


def _parse_count(count: Any) -> int:
    """The calendar length, shape-validated."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise GridDeclarationError(
            "count must be a positive integer naming how many month-end "
            f"instants the calendar spans; got {type(count).__name__}: "
            f"{count!r}."
        )
    return count


def monthly_decision_calendar(
    start_year_month: Any, count: Any, tz: Any
) -> list[datetime]:
    """Month-end decision instants, composed over the shared substrate.

    Exactly one convenience builder: ``count`` consecutive month-end
    instants starting at ``start_year_month``, each produced by the
    shared period-calendar substrate's ``month_end_instant`` — the
    last instant of the month (``23:59:59.999999`` on its last
    calendar day, leap February included) in the declared timezone
    ``tz``, UTC-normalized at storage. The period-end mapping is the
    substrate's, imported here and never restated; no quarterly,
    weekly, or custom-frequency builder ships, and no schedule
    machinery exists in this module (that seam is named and owned
    elsewhere).

    ``tz`` must be an aware timezone object — a naive or invalid zone
    rejects fail-closed with ``NaiveTimestampError`` through the
    substrate's own law; no default zone is ever assumed. The builder's
    output is validated by the same grid law researchers' grids obey
    (:func:`require_increasing_instants`) before it is returned, so
    every calendar this module emits is strictly increasing aware
    instants by construction. ``SAME_INSTANT`` availability under this
    period-end convention is a researcher declaration stamped in
    provenance, never an inference made here.
    """
    start_year, start_month = _parse_start(start_year_month)
    _parse_count(count)
    start_index = start_year * 12 + (start_month - 1)
    instants: list[datetime] = []
    for offset in range(count):
        total = start_index + offset
        year = total // 12
        month = total % 12 + 1
        instant = month_end_instant(year, month, tz)
        instants.append(instant)
    require_increasing_instants(instants)
    return instants
