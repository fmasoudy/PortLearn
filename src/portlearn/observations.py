"""Point-in-time observations, vintages, and feature lineage.

This module defines the information semantics:

- **Data law** — data is what observations exist: records carrying both
  ``observation_time`` (when the underlying phenomenon is dated) and
  ``available_time`` (when the information first could have been known);
  an observation without declared availability is not data PortLearn can
  admit.
- **Identity law** — an observation's identity is the triple
  ``(series_id, observation_time, available_time)``.  A revision is a
  separate record with the same ``series_id`` and ``observation_time``
  and a later ``available_time``; revisions are never in-place changes.
  ``series_id`` is an opaque, non-empty, non-blank string compared by
  exact string equality — no case folding, whitespace stripping, or
  Unicode normalization of any kind.
- **Vintage law** — at a decision time, the visible vintage of a
  series-observation is the record with the latest ``available_time``
  at or before the decision; a revised value is invisible before its
  own availability.  ``vintage_as_of`` accepts exactly one
  ``(series_id, observation_time)`` group and rejects duplicate
  full-identity records and mixed-group input strict.
- **Feature lineage law** — a derived feature may not be declared
  available before its latest input: ``feature available_time ≥
  max(input available_times)``, and an empty input collection has no
  defensible availability.

This module owns ``AmbiguousObservationError`` and
``FeatureLineageError`` and imports the timing errors it raises;
``portlearn.timing`` imports nothing from here, so no
import cycle exists on this contract surface.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .timing import (
    InvalidChronologyError,
    MissingAvailabilityError,
    NaiveTimestampError,  # noqa: F401  # re-exported: raised at this surface by the validator from portlearn.timing
    _require_aware_instant,  # noqa: F401  # re-exported: the shared validator from portlearn.timing, imported to preserve object identity
    instant_key,
    to_instant,
)

__all__ = [
    "AmbiguousObservationError",
    "FeatureLineageError",
    "TimedObservation",
    "require_lineage_monotone",
    "vintage_as_of",
]


# --------------------------------------------------------------------------- #
# Rejection taxonomy — the observations-owned arm
# --------------------------------------------------------------------------- #


class AmbiguousObservationError(Exception):
    """Observation identity is ambiguous: duplicate records or mixed input.

    Raised when two records share the full identity triple
    ``(series_id, observation_time, available_time)`` — whether or not
    their values agree, since the identity triple is a key with no
    last-write-wins and no value-equality exception — or when input to
    ``vintage_as_of`` spans more than one ``(series_id,
    observation_time)`` group.  No preference rule is applied.
    """


class FeatureLineageError(Exception):
    """A feature's declared availability violates lineage monotonicity.

    A derived feature may not be declared available before its latest
    input's availability, and a feature with an empty input collection
    has no defensible availability at all: both unconditional rather than
    warn, because a feature visible before the data it derives from is
    look-ahead leakage.
    """


# --------------------------------------------------------------------------- #
# Instant validation — reused, not re-implemented
# --------------------------------------------------------------------------- #

# The aware-instant validator is implemented in ``portlearn.timing``
# and imported above: shared invariant validation has exactly one
# implementation, and every surface that needs it re-exports that
# exact function object by import, so re-exports preserve object
# identity package-wide.  The canonical invalid-input wording —
# including its calendar-date tail — is timing.py's.


# --------------------------------------------------------------------------- #
# The observation value object
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TimedObservation:
    """One point-in-time record of a series observation.

    ``observation_time`` is when the underlying phenomenon is dated;
    ``available_time`` is when the information first could have been
    known (publication is one source of availability); ``value`` is the
    observed figure.  The identity of the record is the triple
    ``(series_id, observation_time, available_time)``: a revision is a
    separate record with a later ``available_time``, never an in-place change.

    Construction validates strictly: ``series_id`` must be a non-empty,
    non-blank string (validated with the built-in ``ValueError`` because
    a blank identifier names no series); both instants must be aware;
    ``available_time`` may not be absent, ``None``, or precede the
    observation it describes.  The identifier is preserved exactly —
    no case folding, trimming, or Unicode normalization.
    """

    series_id: str
    observation_time: datetime
    available_time: datetime
    value: Any

    def __post_init__(self) -> None:
        # Non-string identifiers raise ValueError rather than TypeError:
        # the built-in ValueError is the closed-taxonomy choice for every
        # malformed series_id, non-string included.
        if not isinstance(self.series_id, str):
            raise ValueError(  # noqa: TRY004 — malformed identifiers raise ValueError
                "series_id must be a string identifier naming the series "
                "the observation belongs to; got "
                f"{type(self.series_id).__name__}: {self.series_id!r}. "
                "A non-string identifier names no series, so the "
                "observation is rejected unconditionally."
            )
        if not self.series_id.strip():
            raise ValueError(
                "series_id must be a non-empty, non-blank string identifier "
                f"naming the series the observation belongs to; got "
                f"{self.series_id!r}. A blank identifier names no series, "
                "so the observation is rejected unconditionally."
            )
        observation = to_instant(
            self.observation_time, "observation_time"
        )
        if self.available_time is None:
            raise MissingAvailabilityError(
                f"the observation for series {self.series_id!r} declares "
                "available_time=None; availability is mandatory — a source "
                "unable to declare when the information first could have "
                "been known must fail, never default"
            )
        available = to_instant(self.available_time, "available_time")
        if available < observation:
            raise InvalidChronologyError(
                "chronology violation: availability may not precede the "
                f"observation it describes, but series {self.series_id!r} "
                f"declares observation_time={observation.isoformat()} after "
                f"available_time={available.isoformat()}. Information "
                "cannot be published before the phenomenon it reports."
            )


# --------------------------------------------------------------------------- #
# Point-in-time vintage selection
# --------------------------------------------------------------------------- #


def vintage_as_of(
    observations: Sequence[TimedObservation], decision_time: Any
) -> TimedObservation | None:
    """The visible vintage of one series-observation at a decision time.

    Selection follows the point-in-time rule: among
    exactly one ``(series_id, observation_time)`` group, the visible
    vintage is the record with the latest ``available_time`` at or
    before ``decision_time`` — availability exactly at the decision is
    visible.  A revised value is invisible before its own availability.

    Evaluation validates strictly in this order: ``decision_time`` is
    validated as an aware instant before
    any branching; empty input returns ``None`` (no vintage, explicitly
    not an error); input spanning more than one group rejects with
    ``AmbiguousObservationError``; duplicate full-identity records
    reject with ``AmbiguousObservationError`` whether or not their
    values agree; when no record is visible, the outcome is ``None``,
    again not an error.
    """
    decision = to_instant(decision_time, "decision_time")

    if not observations:
        return None

    groups: dict[tuple[str, datetime], list[TimedObservation]] = {}
    for record in observations:
        groups.setdefault(
            (record.series_id, instant_key(record.observation_time)), []
        ).append(record)

    if len(groups) > 1:
        named = sorted({series_id for series_id, _ in groups})
        raise AmbiguousObservationError(
            "ambiguous observation input: the records span more than one "
            f"(series_id, observation_time) group — series {named!r} — so "
            "no single series-observation vintage can be selected without "
            "an illegitimate cross-group preference rule. Submit exactly "
            "one group per vintage query."
        )

    group = next(iter(groups.values()))
    seen: set[tuple[str, datetime, datetime]] = set()
    for record in group:
        identity = (
            record.series_id,
            instant_key(record.observation_time),
            instant_key(record.available_time),
        )
        if identity in seen:
            raise AmbiguousObservationError(
                "ambiguous observation input: two records share the full "
                f"identity triple (series_id={record.series_id!r}, "
                f"observation_time={record.observation_time.isoformat()}, "
                f"available_time={record.available_time.isoformat()}) — the "
                "identity triple is a key, so there is no last-write-wins "
                "and no value-equality exception; conflicting or "
                "indeterminate availability fails closed."
            )
        seen.add(identity)

    visible = [
        record
        for record in group
        if instant_key(record.available_time) <= decision
    ]
    if not visible:
        return None
    return max(visible, key=lambda record: instant_key(record.available_time))


# --------------------------------------------------------------------------- #
# Feature lineage monotonicity
# --------------------------------------------------------------------------- #


def require_lineage_monotone(
    feature_available_time: Any, input_available_times: Iterable[Any]
) -> None:
    """Assert a feature is not declared available before its latest input.

    A derived feature may enter an information set no earlier than the
    latest input it derives from: ``feature_available_time`` must be at
    or after ``max(input_available_times)``.  An
    empty input collection also rejects — a feature with no declared
    inputs has no defensible availability.  Both the feature instant
    and every input instant must be aware; naive or date inputs fail
    closed with ``NaiveTimestampError`` on this surface too.
    """
    feature_available = to_instant(
        feature_available_time, "feature_available_time"
    )

    inputs = list(input_available_times)
    if not inputs:
        raise FeatureLineageError(
            "feature lineage violation: the feature declares an empty "
            "input collection, so no input availability bounds its own — "
            "a feature with no declared inputs has no defensible "
            "availability and cannot be admitted."
        )

    input_instants = [
        to_instant(instant, "input_available_time") for instant in inputs
    ]
    latest_input = max(input_instants)

    if feature_available < latest_input:
        raise FeatureLineageError(
            "feature lineage violation: the feature is declared available "
            "before its latest input — feature_available_time="
            f"{feature_available.isoformat()} precedes the latest "
            f"input_available_time={latest_input.isoformat()}. A derived "
            "value visible before the data it derives from is look-ahead "
            "leakage, so the declaration is rejected."
        )
