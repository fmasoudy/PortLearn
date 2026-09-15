"""Generic user-data ingestion: the declared-schema normalization chokepoint.

This module freezes the generic-user-data-interface laws:

- **Declared schema is mandatory.** :class:`DeclaredTableSchema`
  names every column the source table must carry; blank or non-string
  identifiers reject ``ValueError``.  No column is ever inferred.
- **One availability convention.** :class:`AvailabilityPolicy` has
  exactly three modes — ``EXPLICIT_COLUMN(column)``,
  ``SAME_INSTANT``, and ``FIXED_LAG(delta, justification)``.  The
  policy is required: a schema without one rejects, and no default
  exists, because a defaulted availability policy is a silent
  leakage assumption.
- **Duck-typed DataFrame admission.** :func:`from_dataframe`
  accepts any table exposing column names and row iteration; it never
  imports a dataframe library.  Values are admitted from the closed
  domain ``str | int | float | bool | None`` — anything else rejects
  ``TypeError`` (no object passthrough, no dtype inference).
- **Time fields are frozen-law-only.** Every time cell on every
  input path decodes exclusively through the frozen
  :func:`portlearn.timing.to_instant` law: naive datetimes,
  calendar dates, date-only strings, and offset-less strings all
  reject ``NaiveTimestampError`` — no default zone, no date-to-midnight
  coercion, the classic daily-data leakage vector.
- **CSV via the standard library.** :func:`from_csv` accepts CSV
  text or a path, decodes it with :mod:`csv` only, and rejects a
  declared column missing from the header.  Row count, header order,
  and a byte fingerprint enter provenance.
- **Parquet behind the guarded extra.** :func:`from_parquet`
  requires the ``portlearn[parquet]`` extra (pyarrow); its absence
  raises :class:`MissingParquetExtraError` naming the extra.  The
  core import never touches pyarrow.
- **One row→record chokepoint.** All four input paths
  (DataFrame, CSV, Parquet, explicit record sequence) construct
  :class:`~portlearn.observations.TimedObservation` records through
  one builder that enforces the frozen identity, chronology, and
  availability laws and rejects duplicate full-identity rows with
  ``AmbiguousObservationError`` at ingestion — early and loud.
- **Frequency is carried metadata, never semantics.**  Nothing
  converts, infers, or resamples; a declared frequency inconsistent
  with the observation spacing is the researcher's declaration.
- **Provenance.** Every load returns records plus a
  :class:`SourceProvenance` value object — a sibling artifact that
  never widens ``RunManifest``.

Contract errors are owned by their frozen modules: this module
imports them from :mod:`portlearn.timing` and
:mod:`portlearn.observations` and defines none of its own beyond the
extra-guard error.  The module is stdlib-only: the core
package carries zero runtime dependencies.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import cache
from importlib import metadata
from typing import Any

from portlearn.observations import AmbiguousObservationError, TimedObservation
from portlearn.timing import (
    MissingAvailabilityError,
    NaiveTimestampError,
    instant_key,
    to_instant,
)

__all__ = [
    "AvailabilityPolicy",
    "DeclaredTableSchema",
    "MissingParquetExtraError",
    "SourceProvenance",
    "from_csv",
    "from_dataframe",
    "from_parquet",
    "from_records",
]


# --------------------------------------------------------------------------- #
# The guarded-extra error — the one error this module owns
# --------------------------------------------------------------------------- #


class MissingParquetExtraError(Exception):
    """``from_parquet`` was called without the ``portlearn[parquet]`` extra.

    The Parquet path is an optional capability: the core package
    deliberately declares zero runtime dependencies, so
    pyarrow ships only behind the ``portlearn[parquet]`` extra and its
    absence is an actionable installation instruction, never a silent
    capability gap.
    """


# --------------------------------------------------------------------------- #
# The availability policy (no default exists)
# --------------------------------------------------------------------------- #

_MODE_EXPLICIT_COLUMN = "EXPLICIT_COLUMN"
_MODE_SAME_INSTANT = "SAME_INSTANT"
_MODE_FIXED_LAG = "FIXED_LAG"


@dataclass(frozen=True)
class AvailabilityPolicy:
    """How a source declares when each row first could have been known.

    Exactly three modes exist:

    - ``explicit_column(column)`` — the table carries a column naming
      each row's availability instant;
    - ``same_instant()`` — the observation instant is itself the
      availability instant (zero publication lag, declared);
    - ``fixed_lag(delta, justification=...)`` — availability is the
      observation instant shifted by a declared, justified lag.

    Instances are interned per declaration: equal declarations return
    the identical object, so provenance can be compared by identity.
    There is no default policy and no inference — a schema without an
    explicit policy rejects fail-closed.
    """

    mode: str
    column_name: str | None = None
    delta: timedelta | None = None
    justification: str | None = None

    @classmethod
    def explicit_column(cls, column_name: str) -> AvailabilityPolicy:
        """Availability is declared per row by ``column_name``."""
        if not isinstance(column_name, str) or not column_name.strip():
            raise ValueError(
                "the EXPLICIT_COLUMN availability policy requires the "
                "non-blank name of the column declaring when each row "
                "first could have been known; got "
                f"{column_name!r}. Availability is declaration or "
                "rejection — never inference."
            )
        return _interned_explicit_column(column_name)

    @classmethod
    def same_instant(cls) -> AvailabilityPolicy:
        """Availability is the observation instant itself (declared)."""
        return _SAME_INSTANT

    @classmethod
    def fixed_lag(
        cls, delta: timedelta, *, justification: str
    ) -> AvailabilityPolicy:
        """Availability is the observation instant shifted by ``delta``."""
        if not isinstance(delta, timedelta):
            raise ValueError(  # noqa: TRY004 — admission rejects with ValueError
                "the FIXED_LAG availability policy requires a "
                "datetime.timedelta as its declared lag; got "
                f"{type(delta).__name__}: {delta!r}."
            )
        if delta < timedelta(0):
            raise ValueError(
                "the FIXED_LAG availability policy requires a "
                "non-negative declared lag — information cannot become "
                f"available before the phenomenon it reports; got {delta!r}."
            )
        if not isinstance(justification, str) or not justification.strip():
            raise ValueError(
                "the FIXED_LAG availability policy requires a non-blank "
                "justification stating where the declared lag comes from "
                "(a release calendar, a publication convention); got "
                f"{justification!r}. An unjustified lag is an unauditable "
                "leakage assumption."
            )
        return _interned_fixed_lag(delta, justification)


@cache
def _interned_explicit_column(column_name: str) -> AvailabilityPolicy:
    return AvailabilityPolicy(
        mode=_MODE_EXPLICIT_COLUMN, column_name=column_name
    )


@cache
def _interned_fixed_lag(
    delta: timedelta, justification: str
) -> AvailabilityPolicy:
    return AvailabilityPolicy(
        mode=_MODE_FIXED_LAG, delta=delta, justification=justification
    )


_SAME_INSTANT = AvailabilityPolicy(mode=_MODE_SAME_INSTANT)


# --------------------------------------------------------------------------- #
# The declared table schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, kw_only=True)
class DeclaredTableSchema:
    """The declared column contract a source table must satisfy.

    Ingestion is declared-schema-only: the schema names the series,
    observation-time, and value columns, carries the mandatory
    :class:`AvailabilityPolicy`, and declares units and frequency as
    carried metadata (never semantics).  Blank or non-string column
    identifiers reject ``ValueError``; a missing availability policy
    rejects — no default exists.
    """

    series_id_column: str
    observation_time_column: str
    availability: AvailabilityPolicy
    value_column: str
    units: str
    frequency: str
    description_column: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "series_id_column",
            "observation_time_column",
            "value_column",
        ):
            declared = getattr(self, name)
            if not isinstance(declared, str) or not declared.strip():
                raise ValueError(
                    f"{name} must be a non-empty, non-blank string naming "
                    f"a declared column of the source table; got "
                    f"{declared!r}. Ingestion is declared-schema-only — no "
                    "column is ever inferred."
                )
        for name in ("units", "frequency"):
            declared = getattr(self, name)
            if not isinstance(declared, str) or not declared.strip():
                raise ValueError(
                    f"{name} must be a non-empty, non-blank string "
                    "declaring the units / frequency metadata carried in "
                    f"provenance; got {declared!r}."
                )
        if self.description_column is not None and (
            not isinstance(self.description_column, str)
            or not self.description_column.strip()
        ):
            raise ValueError(
                "description_column, when declared, must be a non-empty, "
                f"non-blank string; got {self.description_column!r}."
            )
        if not isinstance(self.availability, AvailabilityPolicy):
            raise ValueError(  # noqa: TRY004 — admission is uniformly ValueError
                "availability is mandatory: pass an explicit "
                "AvailabilityPolicy (explicit_column, same_instant, or "
                "fixed_lag with a justification). No default availability "
                "policy exists, because a defaulted availability is a "
                f"silent leakage assumption; got {self.availability!r}."
            )


# --------------------------------------------------------------------------- #
# Ingestion provenance (a sibling artifact, never a RunManifest)
# --------------------------------------------------------------------------- #

_LICENCE_NOTE = (
    "no licence declared by this source; PortLearn ingestion records the "
    "declaration duty — it neither asserts nor verifies one"
)


@dataclass(frozen=True)
class SourceProvenance:
    """Where a loaded table came from and under which declared laws.

    A sibling value object: provenance travels beside the
    records so any loaded table is reconstructable from source id,
    content hash, and retrieval instant, while ``RunManifest``'s frozen
    schema is never widened.
    """

    source_id: str
    retrieval_instant: datetime
    content_sha256: str | None
    licence_note: str
    availability: AvailabilityPolicy
    adapter_identity: str
    adapter_version: str
    row_count: int
    header_order: list[str]
    units: str
    frequency: str


# --------------------------------------------------------------------------- #
# Time cells decode exclusively through the frozen law
# --------------------------------------------------------------------------- #


def _cell_instant(cell: Any, field_name: str) -> datetime:
    """Decode one time cell to a UTC-normalized instant, fail-closed.

    ISO-8601 strings decode via :meth:`datetime.datetime.fromisoformat`
    and then — like every datetime or date input — pass through the
    frozen :func:`portlearn.timing.to_instant` law, which rejects naive
    datetimes, calendar dates, and date-only strings with
    ``NaiveTimestampError``.  No default zone and no date-to-midnight
    coercion exists on any input path.
    """
    if isinstance(cell, str):
        try:
            cell = datetime.fromisoformat(cell)
        except ValueError:
            raise NaiveTimestampError(
                f"{field_name} is not a parseable ISO-8601 timestamp "
                f"carrying an explicit UTC offset; got {cell!r}. Time "
                "fields decode exclusively through the frozen instant "
                "law — a string without an offset cannot be placed on "
                "the single timeline without assuming a default "
                "timezone, and PortLearn never does."
            ) from None
    return to_instant(cell, field_name)


# --------------------------------------------------------------------------- #
# The single row→record chokepoint
# --------------------------------------------------------------------------- #


def _build_record(
    row: Any,
    schema: DeclaredTableSchema,
    seen_identities: set[tuple[str, datetime, datetime]],
) -> TimedObservation:
    """Construct one :class:`TimedObservation` from one decoded row.

    This is the only construction site: every input path (DataFrame,
    CSV, Parquet, explicit record sequence) funnels through it, so the
    frozen identity, chronology, and availability laws hold once for
    the whole package.  Duplicate full-identity rows reject here,
    early and loud, with ``AmbiguousObservationError``.
    """
    value = row.get(schema.value_column)
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise TypeError(
            f"value must be within the closed domain "
            "str | int | float | bool | None; got "
            f"{type(value).__name__}: {value!r}. Ingestion performs no "
            "object passthrough and no dtype inference — a value outside "
            "the domain is rejected fail-closed."
        )
    observation = _cell_instant(
        row.get(schema.observation_time_column), "observation_time"
    )
    available = _availability_instant(schema, row, observation)
    record = TimedObservation(
        series_id=row.get(schema.series_id_column),
        observation_time=observation,
        available_time=available,
        value=value,
    )
    identity = (
        record.series_id,
        instant_key(record.observation_time),
        instant_key(record.available_time),
    )
    if identity in seen_identities:
        raise AmbiguousObservationError(
            "ambiguous observation input: two rows share the full "
            f"identity triple (series_id={record.series_id!r}, "
            f"observation_time={record.observation_time.isoformat()}, "
            f"available_time={record.available_time.isoformat()}) — the "
            "identity triple is a key, so ingestion fails closed here, "
            "early and loud, exactly as the information-set layer would "
            "later; there is no last-write-wins and no value-equality "
            "exception."
        )
    seen_identities.add(identity)
    return record


def _availability_instant(
    schema: DeclaredTableSchema, row: Any, observation: datetime
) -> datetime | None:
    """Apply the declared :class:`AvailabilityPolicy` to one row.

    Returns ``None`` when an EXPLICIT_COLUMN declaration is absent or
    blank, so the frozen observation constructor raises
    ``MissingAvailabilityError`` — a source unable to declare when the
    information first could have been known must fail, never default.
    """
    policy = schema.availability
    if policy.mode == _MODE_SAME_INSTANT:
        return observation
    if policy.mode == _MODE_FIXED_LAG:
        return observation + policy.delta
    raw = row.get(policy.column_name)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    return _cell_instant(raw, "available_time")


def _finish(
    rows: Iterable[Any],
    schema: DeclaredTableSchema,
    *,
    adapter_identity: str,
    source_id: str,
    content_sha256: str | None,
    header_order: list[str],
) -> tuple[list[TimedObservation], SourceProvenance]:
    """Run every row through the chokepoint and assemble provenance."""
    seen: set[tuple[str, datetime, datetime]] = set()
    records = [_build_record(row, schema, seen) for row in rows]
    provenance = SourceProvenance(
        source_id=source_id,
        retrieval_instant=datetime.now(UTC),
        content_sha256=content_sha256,
        licence_note=_LICENCE_NOTE,
        availability=schema.availability,
        adapter_identity=adapter_identity,
        adapter_version=metadata.version("portlearn"),
        row_count=len(records),
        header_order=list(header_order),
        units=schema.units,
        frequency=schema.frequency,
    )
    return records, provenance


def _require_schema(schema: Any) -> DeclaredTableSchema:
    if not isinstance(schema, DeclaredTableSchema):
        raise TypeError(
            "schema must be a DeclaredTableSchema declaring the source "
            "table's columns, availability policy, units, and frequency; "
            f"got {type(schema).__name__}: {schema!r}. Ingestion is "
            "declared-schema-only — no column is ever inferred."
        )
    return schema


# --------------------------------------------------------------------------- #
# Duck-typed DataFrame admission
# --------------------------------------------------------------------------- #


def _frame_columns(frame: Any) -> list[str]:
    """Resolve a duck-typed frame's column names, or reject ``TypeError``."""
    candidate = getattr(frame, "columns", None)
    if candidate is not None:
        if callable(candidate):
            candidate = candidate()
        try:
            names = [str(name) for name in candidate]
        except TypeError:
            names = []
        if names:
            return names
    try:
        names = [str(name) for name in frame]
    except TypeError:
        raise TypeError(
            "from_dataframe accepts any duck-typed table exposing its "
            "column names — a columns attribute, a columns() method, or "
            f"iteration over column names — plus row iteration; got "
            f"{type(frame).__name__}, which exposes no column interface. "
            "Ingestion never imports a dataframe library, so the frame "
            "must satisfy the protocol itself."
        ) from None
    if not names:
        raise TypeError(
            "from_dataframe accepts any duck-typed table exposing its "
            f"column names, but {type(frame).__name__} declares an empty "
            "column set — no declared column can be looked up."
        )
    return names


def _is_row_mapping(candidate: Any) -> bool:
    return hasattr(candidate, "keys") and hasattr(candidate, "__getitem__")


def _normalized_rows(rows: Iterable[Any]) -> list[Any]:
    """Accept mapping rows, or ``(index, mapping)`` pairs (pandas style)."""
    normalized: list[Any] = []
    for yielded in rows:
        if (
            isinstance(yielded, tuple)
            and len(yielded) == 2
            and not _is_row_mapping(yielded[0])
            and _is_row_mapping(yielded[1])
        ):
            yielded = yielded[1]
        if not _is_row_mapping(yielded):
            raise TypeError(
                "row iteration must yield per-row mappings keyed by "
                "column name (or (index, mapping) pairs); got "
                f"{type(yielded).__name__}: {yielded!r}."
            )
        normalized.append(yielded)
    return normalized


def from_dataframe(
    frame: Any, schema: DeclaredTableSchema
) -> tuple[list[TimedObservation], SourceProvenance]:
    """Load records from any duck-typed column-name + row-iteration table.

    ``frame`` must expose its column names (a ``columns`` attribute, a
    ``columns()`` method, or iteration over column names) and row
    iteration yielding per-row mappings (or ``(index, mapping)``
    pairs).  Protocol failure rejects ``TypeError``.  No dataframe
    library is ever imported; values are admitted from the closed
    domain ``str | int | float | bool | None`` only.
    """
    _require_schema(schema)
    columns = _frame_columns(frame)
    try:
        raw_rows = frame.iterrows()
    except AttributeError:
        raise TypeError(
            "from_dataframe accepts any duck-typed table exposing column "
            f"names plus row iteration, but {type(frame).__name__} "
            "declares no iterrows() row iterator."
        ) from None
    rows = _normalized_rows(raw_rows)
    return _finish(
        rows,
        schema,
        adapter_identity="dataframe",
        source_id="dataframe:in-memory",
        content_sha256=None,
        header_order=columns,
    )


# --------------------------------------------------------------------------- #
# CSV via the standard library
# --------------------------------------------------------------------------- #


class _MissingAvailabilityColumnError(MissingAvailabilityError, ValueError):
    """The declared availability column is absent from the CSV header.

    Private to this module: the condition is simultaneously a missing
    declared column (a ``ValueError``) and an undeclared
    availability (the frozen ``MissingAvailabilityError``), so the
    raised error is an instance of both — the frozen error reused as
    a base class, never redefined.
    """


def _csv_materialize(source: str | os.PathLike[str]) -> tuple[str, str, str]:
    """Return ``(text, content_sha256, source_id)`` for text or path."""
    path: str | None = None
    if isinstance(source, os.PathLike):
        path = os.fspath(source)
    elif isinstance(source, str) and "\n" not in source and "\r" not in source:
        if os.path.exists(source):
            path = source
    elif not isinstance(source, str):
        raise TypeError(
            "from_csv accepts CSV text or a path (str or os.PathLike) to "
            f"a CSV file; got {type(source).__name__}: {source!r}."
        )
    if path is not None:
        with open(path, "rb") as handle:
            raw_bytes = handle.read()
        return (
            raw_bytes.decode("utf-8-sig"),
            hashlib.sha256(raw_bytes).hexdigest(),
            f"csv:{os.path.abspath(path)}",
        )
    assert isinstance(source, str)
    return (
        source,
        hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "csv:inline-text",
    )


def _require_declared_columns(
    header: list[str], schema: DeclaredTableSchema
) -> None:
    """Reject (fail-closed) any declared column absent from the header."""
    present = set(header)
    for column in (
        schema.series_id_column,
        schema.observation_time_column,
        schema.value_column,
    ):
        if column not in present:
            raise ValueError(
                f"the declared column {column!r} is missing from the CSV "
                f"header {header!r}; ingestion is declared-schema-only — "
                "no column is ever inferred."
            )
    policy = schema.availability
    if policy.mode == _MODE_EXPLICIT_COLUMN and policy.column_name not in present:
        raise _MissingAvailabilityColumnError(
            f"the declared availability column {policy.column_name!r} is "
            f"missing from the CSV header {header!r}; the EXPLICIT_COLUMN "
            "availability policy cannot be applied, so available_time is "
            "undeclared — a source unable to declare when the information "
            "first could have been known must fail, never default. Pass a "
            "header carrying the declared availability column or declare "
            "a different AvailabilityPolicy."
        )


def _decode_csv_value(cell: Any) -> Any:
    """Decode one CSV value cell: integer, float, blank→None, else str."""
    if not isinstance(cell, str):
        return cell
    stripped = cell.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return cell


def from_csv(
    source: str | os.PathLike[str], schema: DeclaredTableSchema
) -> tuple[list[TimedObservation], SourceProvenance]:
    """Load records from CSV text or a CSV file path.

    Decodes with the :mod:`csv` module only (no third-party parser),
    validates the header against the declared schema, records row
    count, header order, and the SHA-256 fingerprint of the exact
    decoded bytes in provenance, and funnels every row through the
    single chokepoint.
    """
    _require_schema(schema)
    text, content_sha256, source_id = _csv_materialize(source)
    parsed = list(csv.reader(io.StringIO(text)))
    if not parsed:
        raise ValueError(
            "the CSV source declares no header row; a declared schema "
            "cannot be matched against an empty source."
        )
    header = parsed[0]
    header_order = list(header)
    _require_declared_columns(header, schema)
    rows: list[dict[str, Any]] = []
    for raw_row in parsed[1:]:
        padded = list(raw_row) + [""] * (len(header) - len(raw_row))
        row = dict(zip(header, padded, strict=False))
        row[schema.value_column] = _decode_csv_value(
            row.get(schema.value_column)
        )
        rows.append(row)
    return _finish(
        rows,
        schema,
        adapter_identity="csv",
        source_id=source_id,
        content_sha256=content_sha256,
        header_order=header_order,
    )


# --------------------------------------------------------------------------- #
# The explicit record sequence path (same chokepoint)
# --------------------------------------------------------------------------- #


def from_records(
    records: Iterable[Any], schema: DeclaredTableSchema
) -> tuple[list[TimedObservation], SourceProvenance]:
    """Load records from an explicit sequence of per-row mappings.

    Each item must be a mapping keyed by the declared column names
    (string time cells decode through the same frozen instant law as
    every other path); every row funnels through the single
    chokepoint with the full identity, chronology, availability, and
    duplicate-identity laws enforced.
    """
    _require_schema(schema)
    rows = _normalized_rows(records)
    return _finish(
        rows,
        schema,
        adapter_identity="records",
        source_id="records:in-memory",
        content_sha256=None,
        header_order=[],
    )


# --------------------------------------------------------------------------- #
# Parquet behind the guarded extra
# --------------------------------------------------------------------------- #


def from_parquet(
    path: str | os.PathLike[str], schema: DeclaredTableSchema
) -> tuple[list[TimedObservation], SourceProvenance]:
    """Load records from a Parquet file via the ``portlearn[parquet]`` extra.

    Requires pyarrow, shipped only behind the optional
    ``portlearn[parquet]`` extra; when it is absent this raises
    :class:`MissingParquetExtraError` naming the extra — the core
    import never touches pyarrow.  Decode reduces to the same
    duck-typed row path as every other input.
    """
    _require_schema(schema)
    try:
        import pyarrow  # noqa: F401  # presence probe only
    except ImportError:
        raise MissingParquetExtraError(
            "from_parquet requires pyarrow, provided by the "
            'portlearn[parquet] optional extra — install it with '
            'pip install "portlearn[parquet]" (or uv sync --extra '
            "parquet). The core package deliberately declares zero "
            "runtime dependencies, so the Parquet path stays guarded "
            "behind the extra."
        ) from None
    from pyarrow import parquet

    location = os.fspath(path)
    table = parquet.read_table(location)
    header_order = list(table.column_names)
    rows = _normalized_rows(table.to_pylist())
    return _finish(
        rows,
        schema,
        adapter_identity="parquet",
        source_id=f"parquet:{location}",
        content_sha256=None,
        header_order=header_order,
    )
