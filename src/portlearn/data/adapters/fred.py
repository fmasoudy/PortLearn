"""FRED research-data adapter — a pure decoder plus a thin fetcher.

The decoder reduces the Federal Reserve Economic Data (FRED) API's JSON
observations response to the ingestion declared-table form: every record
is constructed and validated by the single ingestion chokepoint
(:func:`portlearn.data.ingestion.from_records`) — this module never constructs
a record, never restates the identity/chronology/availability laws, and
never stamps a default availability.  Monthly and quarterly period-start
observation dates map to period-END instants through the shared
period-calendar substrate (:func:`portlearn.calendar.month_end_instant`,
:func:`portlearn.calendar.quarter_end_instant`); daily dates map to the
calendar day's last instant in the caller-declared zone — never a market
close.  The provider's missing-data marker (``"."``) decodes to an absent
record, never to ``None`` or an invented zero.

**The false-vintage fence:** FRED serves revised
snapshots.  Every retrieval is labeled with a data mode from the closed
set ``{CURRENT_SNAPSHOT, POINT_IN_TIME}``, carried in provenance.  A
``CURRENT_SNAPSHOT`` decode bounds every record's ``available_time``
below by the retrieval snapshot instant: a revised snapshot cannot receive an
assumed historical availability timestamp and cannot support point-in-time
claims.  A declared per-series lag may be recorded as **non-binding
research metadata** (an assumption about release practice) but can never
move ``available_time`` earlier than the retrieval instant.  A declared
``POINT_IN_TIME`` mode rejects fail-closed: no source-supplied vintage
evidence path exists, and none may be improvised.

Decoded series are research inputs and are **not investable**; nothing in
this adapter is an investability claim.

Fetching is a separate stdlib-only function (``urllib.request``) whose
only outputs are bytes plus retrieval provenance.  The API key is a
fetch-time argument only: it is sent to the provider, never recorded in
any provenance field, any URL, or any fixture — the provenance value
object structurally refuses key-bearing material.  True-vintage retrieval
parameters (``vintage_dates``; non-default ``output_type``) reject
fail-closed: this adapter implements no vintage path.  The decoder is
network-independent and pure: identical bytes decode to identical
records.  All committed fixtures are clearly labeled SYNTHETIC
provider-format replicas (see the fixture MANIFEST) containing invented
values — no provider-owned data is redistributed.

Beside the qualified surface this module also hosts the
retrieval-only (UNQUALIFIED) sibling surface: :func:`fetch_raw` returns
the provider's exact bytes with retrieval-only provenance (retrieval
facts only — no availability field at all), and
:func:`decode_unqualified` reduces locally-held bytes to
:class:`PeriodKeyObservation` records carrying the provider's own period
labels verbatim — never an instant, never a fabricated vintage, and
never a default policy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version as _distribution_version
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from portlearn.calendar import month_end_instant, quarter_end_instant
from portlearn.data import ingestion
from portlearn.timing import to_instant

from .._records import PeriodKeyObservation, RetrievalProvenance

__all__ = [
    "ADAPTER_IDENTITY",
    "ADAPTER_VERSION",
    "CURRENT_SNAPSHOT",
    "DATASET_MODES",
    "POINT_IN_TIME",
    "FREDProvenance",
    "FREDRetrievalProvenance",
    "PeriodKeyObservation",
    "ProviderResponseError",
    "PseudoVintageError",
    "RetrievalProvenance",
    "UnknownSeriesError",
    "UnsupportedFrequencyError",
    "decode",
    "decode_unqualified",
    "fetch",
    "fetch_raw",
    "require_current_snapshot_recency",
]

ADAPTER_IDENTITY = "portlearn.data.adapters.fred"

#: Adapter revision (manifest-recorded); bumped on any surface change.
ADAPTER_VERSION = 1

#: The closed data-mode set: every retrieval is labeled
#: with exactly one mode, carried in provenance, never inferred.
CURRENT_SNAPSHOT = "CURRENT_SNAPSHOT"
POINT_IN_TIME = "POINT_IN_TIME"
DATASET_MODES = (CURRENT_SNAPSHOT, POINT_IN_TIME)

_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

#: The period frequencies this adapter maps through the shared substrate; any
#: other provider frequency (Weekly, Annual, ...) is outside the frozen
#: period-mapping scope and rejects fail-closed.
SUPPORTED_FREQUENCIES = ("Monthly", "Quarterly", "Daily")

_LICENCE_NOTE = (
    "FRED API terms of use: attribution mandatory, third-party "
    "copyright respected, termination at will (committed fixtures are "
    "synthetic replicas holding invented values — no provider-owned "
    "data redistributed). Research use only; not investable."
)

_ATTRIBUTION = (
    "This product uses the FRED(R) API but is not endorsed or certified "
    "by the Federal Reserve Bank of St. Louis."
)

_NO_LAG_NOTE = "no declared lag; availability bounded below by the retrieval instant"

#: The provider's missing-data marker: an observation carrying it is an
#: explicit absence — no record is emitted for it, ever.
_MISSING_MARKER = "."

_SERIES_PREFIX = "FRED/"

#: The deterministic offline retrieval instant (UTC epoch).  An offline
#: ``decode_unqualified`` (no ``retrieval`` supplied) must stay pure —
#: identical bytes decode to identical provenance — so it never reads
#: the clock; it pins this one fixed timezone-aware instant instead, so
#: a later qualified re-decode that adopts the record can still place it
#: on the single timeline.
_OFFLINE_RETRIEVAL_INSTANT = datetime(1970, 1, 1, tzinfo=UTC)

#: The single sanctioned availability-policy plumbing in this module:
#: internal plumbing that hands the bounded availability column to the
#: ingestion chokepoint.  It is NOT a default availability — the caller's
#: declared policy remains a mandatory argument on every decode.
_BOUNDED_EXPLICIT_COLUMN = ingestion.AvailabilityPolicy.explicit_column(
    "available_time"
)


class ProviderResponseError(ValueError):
    """The provider response is unusable (shape, JSON, HTTP) — fail-closed."""


class UnknownSeriesError(ProviderResponseError):
    """The provider reports the requested series does not exist."""


class UnsupportedFrequencyError(ValueError):
    """The frequency is outside the closed period-mapping set."""


class PseudoVintageError(ValueError):
    """A revised snapshot was about to become a pseudo-vintage — barred.

    Raised when a ``CURRENT_SNAPSHOT`` record would carry an
    ``available_time`` preceding its retrieval lower bound, or when a
    ``POINT_IN_TIME`` mode is requested without the source-supplied
    vintage evidence the mode demands.
    """


def _lag_note(policy: ingestion.AvailabilityPolicy) -> str:
    if policy.mode == "FIXED_LAG":
        return (
            f"declared fixed lag {policy.delta!s} is non-binding research "
            "metadata about release practice; it never moves available_time "
            f"earlier than the retrieval lower bound. Justification: "
            f"{policy.justification}"
        )
    return _NO_LAG_NOTE


@dataclass(frozen=True)
class FREDProvenance(ingestion.SourceProvenance):
    """Retrieval provenance for a fetched or decoded FRED dataset.

    A sibling value object extending the frozen ingestion provenance with
    the provider-layer retrieval facts: the sanitized
    request URL (never containing the API key), the realtime window, the
    attribution line, the closed ``data_mode``, the sanitized request
    parameters, and the declared-lag note.  The frozen
    ``SourceProvenance`` shape is never mutated — extension is by
    subclass, never by editing the ingestion module.
    """

    url: str = ""
    realtime_start: str | None = None
    realtime_end: str | None = None
    attribution: str = _ATTRIBUTION
    data_mode: str = CURRENT_SNAPSHOT
    request_params: dict[str, str] = field(default_factory=dict)
    declared_lag_note: str = _NO_LAG_NOTE

    def __post_init__(self) -> None:
        if self.data_mode not in DATASET_MODES:
            raise ValueError(
                "data_mode must be one of the closed set "
                f"{DATASET_MODES!r} — every retrieval is labeled with "
                "exactly one data mode; got "
                f"{self.data_mode!r}."
            )
        lowered = {str(key).lower() for key in self.request_params}
        if "api_key" in lowered:
            raise ValueError(
                "api_key may never ride provenance: request_params carries "
                "key material, but the API key is fetch-time only and any "
                "record of it is a disclosure, fail-closed."
            )
        if "api_key=" in self.url:
            raise ValueError(
                "api_key may never ride provenance: the url carries key "
                "material, but the API key is fetch-time only and any "
                "record of it is a disclosure, fail-closed."
            )


@dataclass(frozen=True)
class FREDRetrievalProvenance(RetrievalProvenance):
    """Retrieval-only provenance: retrieval facts, no availability field.

    The retrieval-only sibling of :class:`FREDProvenance`: the
    same provider retrieval facts (sanitized URL, realtime window,
    attribution, closed data mode, sanitized request parameters, the
    requested series id) minus every availability fact.  There is
    deliberately **no** ``availability`` field and this record is
    deliberately *not* a
    :class:`~portlearn.data.ingestion.SourceProvenance`: a retrieval-only
    load can never satisfy the qualified provenance contract.  Frozen
    and hashable end-to-end: ``request_params`` is a canonical
    immutable tuple of key-sorted string pairs (never a dict, never
    key-bearing), which is exactly what the frozen qualified decoder's
    ``dict(retrieval.request_params)`` adoption consumes unchanged.
    """

    source_id: str = ""
    retrieval_instant: datetime = _OFFLINE_RETRIEVAL_INSTANT
    content_sha256: str = ""
    licence_note: str = _LICENCE_NOTE
    adapter_identity: str = ADAPTER_IDENTITY
    adapter_version: str = ""
    units: str = "unspecified"
    frequency: str = "unspecified"
    url: str = ""
    last_modified: str | None = None
    attribution: str = _ATTRIBUTION
    data_mode: str = CURRENT_SNAPSHOT
    request_params: tuple[tuple[str, str], ...] = ()
    realtime_start: str | None = None
    realtime_end: str | None = None
    series_id: str = ""

    def __post_init__(self) -> None:
        if self.data_mode not in DATASET_MODES:
            raise ValueError(
                "data_mode must be one of the closed set "
                f"{DATASET_MODES!r} — every retrieval is labeled with "
                "exactly one data mode; got "
                f"{self.data_mode!r}."
            )
        canonical = tuple(
            sorted(
                (str(key), str(value)) for key, value in self.request_params
            )
        )
        lowered = {key.lower() for key, _ in canonical}
        if "api_key" in lowered:
            raise ValueError(
                "api_key may never ride provenance: request_params carries "
                "key material, but the API key is fetch-time only and any "
                "record of it is a disclosure, fail-closed."
            )
        if "api_key=" in self.url:
            raise ValueError(
                "api_key may never ride provenance: the url carries key "
                "material, but the API key is fetch-time only and any "
                "record of it is a disclosure, fail-closed."
            )
        object.__setattr__(self, "request_params", canonical)


# --------------------------------------------------------------------------- #
# Admission (fail-closed, no defaults)
# --------------------------------------------------------------------------- #


def _require_policy(availability: Any) -> ingestion.AvailabilityPolicy:
    if not isinstance(availability, ingestion.AvailabilityPolicy):
        raise ValueError(  # noqa: TRY004 — admission is uniformly ValueError
            "availability is mandatory and this adapter supplies no "
            "default: pass an explicit availability policy (an explicit "
            "column, same-instant, or fixed-lag declaration). A silent "
            "default availability would be a silent leakage assumption; "
            f"got {availability!r}."
        )
    return availability


def _require_mode(data_mode: Any) -> str:
    if data_mode not in DATASET_MODES:
        raise ValueError(
            f"data_mode must be one of the closed set {DATASET_MODES!r} "
            "; got {data_mode!r}."
        )
    return data_mode


def _require_frequency(frequency: Any) -> str:
    if frequency not in SUPPORTED_FREQUENCIES:
        raise UnsupportedFrequencyError(
            f"frequency {frequency!r} is outside the closed period-"
            f"mapping set {SUPPORTED_FREQUENCIES!r}; Weekly/Annual and "
            "other provider frequencies reject fail-closed — they have "
            "no frozen period-instant mapping."
        )
    return frequency


def _require_timezone(tzinfo: Any) -> None:
    if tzinfo is None:
        raise ValueError(
            "decode requires an explicit aware timezone (tzinfo=...); "
            "no default zone is ever assumed — a naive mapping cannot "
            "be placed on the single timeline."
        )


# --------------------------------------------------------------------------- #
# Provider-format parsing (structure only — records are the chokepoint's)
# --------------------------------------------------------------------------- #


def _parse_payload(data: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderResponseError(
            f"provider bytes are not a JSON document ({exc}); the decoder "
            "rejects unparseable provider bytes fail-closed."
        ) from None
    if not isinstance(payload, dict):
        raise ProviderResponseError(
            "the provider JSON document is not an object; the decoder "
            "rejects unrecognized provider bytes fail-closed."
        )
    return payload


def _require_observation_payload(
    payload: dict[str, Any], series_id: str
) -> list[dict[str, Any]]:
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message", "")
        raise UnknownSeriesError(
            f"the provider returned an error payload for {series_id!r}: "
            f"{message} The request rejects fail-closed."
        )
    for window_key in ("realtime_start", "realtime_end"):
        if not isinstance(payload.get(window_key), str):
            raise ProviderResponseError(
                f"the payload is missing its {window_key} realtime window "
                "label; a silent realtime window is barred — "
                "the retrieval window is always labeled, never assumed."
            )
    payload_series = payload.get("series_id")
    if payload_series != series_id:
        raise ProviderResponseError(
            f"the payload series_id {payload_series!r} does not match the "
            f"requested series {series_id!r}; a mismatched series rejects "
            "fail-closed."
        )
    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ProviderResponseError(
            "the provider payload carries no observations list; the "
            "provider format is not recognized, fail-closed."
        )
    return observations


def _day_last_instant(date_key: str, tzinfo: Any) -> datetime:
    """The calendar day's LAST aware instant in the declared zone."""
    _require_date_key(date_key)
    return datetime(
        int(date_key[:4]),
        int(date_key[5:7]),
        int(date_key[8:10]),
        23,
        59,
        59,
        999999,
        tzinfo=tzinfo,
    )


def _require_date_key(date_key: Any) -> str:
    if (
        not isinstance(date_key, str)
        or len(date_key) != 10
        or date_key[4] != "-"
        or date_key[7] != "-"
        or not date_key[:4].isdigit()
        or not date_key[5:7].isdigit()
        or not date_key[8:10].isdigit()
    ):
        raise ProviderResponseError(
            f"observation date {date_key!r} is not a provider YYYY-MM-DD "
            "date key; the decoder rejects it fail-closed."
        )
    return date_key


def _period_instant(date_key: str, frequency: str, tzinfo: Any) -> datetime:
    """Map one provider observation date to its aware instant.

    Monthly and quarterly dates are period STARTS (FRED's own dating);
    they map to the period's END instant through the shared substrate
    for accounting consistency.  A date that is not its
    period's start rejects — it names no whole period.  Daily dates map
    to the day's last instant: no trading-session time is assumed.
    """
    year, month, day = (
        int(date_key[:4]),
        int(date_key[5:7]),
        int(date_key[8:10]),
    )
    if frequency == "Daily":
        return _day_last_instant(date_key, tzinfo)
    if day != 1:
        raise ValueError(
            f"{frequency} observation date {date_key!r} is not a period "
            "start (day 01); a mid-period date names no whole period and "
            "rejects fail-closed."
        )
    if frequency == "Monthly":
        if not 1 <= month <= 12:
            raise ProviderResponseError(
                f"observation date {date_key!r} names no calendar month."
            )
        return month_end_instant(year, month, tzinfo)
    if month not in (1, 4, 7, 10):
        raise ValueError(
            f"Quarterly observation date {date_key!r} is not a period "
            "start (month 01/04/07/10); a mid-quarter date names no "
            "whole quarter and rejects fail-closed."
        )
    return quarter_end_instant(year, (month - 1) // 3 + 1, tzinfo)


def _parse_value(cell: Any) -> float:
    """One provider value cell; the missing marker never reaches here."""
    try:
        return float(cell)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ProviderResponseError(
            f"provider value cell {cell!r} is neither a numeric "
            "observation nor the missing-data marker; the decoder "
            "rejects unparseable provider bytes fail-closed."
        ) from None


def _realtime_day_last_instant(day: str, tzinfo: Any) -> datetime:
    """The last instant of the payload's realtime-end day, declared zone."""
    _require_date_key(day)
    return datetime(
        int(day[:4]),
        int(day[5:7]),
        int(day[8:10]),
        23,
        59,
        59,
        999999,
        tzinfo=tzinfo,
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now_utc() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- #
# The thin fetch layer — bytes plus retrieval provenance, nothing else
# --------------------------------------------------------------------------- #


def _require_api_key(api_key: Any) -> str:
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(
            "api_key is mandatory for fetch and none was supplied: pass "
            "the FRED API key as the api_key argument — it is used for "
            "this request only and is never recorded in provenance; got "
            f"{api_key!r}."
        )
    return api_key


def _require_params(params: Any) -> dict[str, str]:
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise ValueError(  # noqa: TRY004 — admission is uniformly ValueError
            f"params must be a mapping of extra query parameters; got "
            f"{type(params).__name__}: {params!r}."
        )
    for key, value in params.items():
        if str(key).lower() == "api_key":
            raise ValueError(
                "api_key may not ride params: the API key is the "
                "fetch-time api_key argument only and may never be "
                "smuggled through the extra-parameters mapping."
            )
        if key == "vintage_dates":
            raise ValueError(
                "vintage_dates is a true-vintage retrieval parameter: "
                "this adapter implements no vintage path (the POINT_IN_TIME "
                "boundary), so vintage retrieval rejects fail-closed."
            )
        if key == "output_type" and str(value) != "1":
            raise ValueError(
                f"output_type={value!r} selects a vintage/realtime output "
                "shape: this adapter implements no vintage path (the "
                "POINT_IN_TIME boundary), so non-default output types "
                "reject fail-closed."
            )
    return {str(key): str(value) for key, value in params.items()}


def _request_observations(
    series_id: str,
    api_key: str,
    params: dict[str, str] | None,
) -> tuple[bytes, dict[str, str], str]:
    """Shared fetch machinery (qualified ``fetch`` and raw ``fetch_raw``).

    Validates the API key **before** any socket opens, always sends the
    labeled realtime window, sends the key to the provider inside the
    request URL only, and returns ``(exact bytes, sanitized query,
    sanitized URL)`` — the key never rides any returned value.  This is
    the single in-module home of the credential-validation and
    sanitization rules (no duplicated rule homes).
    """
    key = _require_api_key(api_key)
    extra = _require_params(params)
    realtime_day = _now_utc().date().isoformat()
    query: dict[str, str] = {
        "series_id": str(series_id),
        "realtime_start": extra.get("realtime_start", realtime_day),
        "realtime_end": extra.get("realtime_end", realtime_day),
        "file_type": "json",
    }
    for name, value in extra.items():
        if name not in query:
            query[name] = value
    sanitized = urlencode(query)
    full = urlencode({**query, "api_key": key})
    # The full (key-bearing) URL travels to the provider as a plain
    # string argument — the credential rides the request URL only, and
    # the sanitized twin below is the sole URL any provenance records.
    full_url = f"{_BASE_URL}?{full}"
    try:
        with urlopen(full_url, timeout=60) as response:
            data = response.read()
    except HTTPError as exc:
        raise ProviderResponseError(
            f"provider HTTP {exc.code} failure for series {series_id!r} "
            f"({exc.reason}); the fetch rejects fail-closed."
        ) from None
    except URLError as exc:
        raise ProviderResponseError(
            f"provider transport failure for series {series_id!r} "
            f"({exc.reason}); the fetch rejects fail-closed."
        ) from None
    return data, query, f"{_BASE_URL}?{sanitized}"


def fetch(
    series_id: str,
    api_key: str,
    params: dict[str, str] | None = None,
) -> tuple[bytes, FREDProvenance]:
    """Fetch one series' observations bytes plus retrieval provenance.

    Stdlib-only (``urllib.request``); the provider's bytes are returned
    exactly as received — no parsing of observations happens here.  The
    realtime window (``realtime_start``/``realtime_end``) is always sent
    (D5: never a silent window), defaulting to today.  The API key is
    validated before any socket is opened, sent to the provider, and
    never recorded: the returned provenance carries a sanitized URL and
    sanitized request parameters only.  True-vintage parameters reject
    fail-closed.  The retrieval is labeled ``CURRENT_SNAPSHOT``.
    """
    data, query, sanitized_url = _request_observations(
        series_id, api_key, params
    )
    provenance = FREDProvenance(
        source_id=f"fred:{series_id}",
        retrieval_instant=_now_utc(),
        content_sha256=_sha256(data),
        licence_note=_LICENCE_NOTE,
        availability=_BOUNDED_EXPLICIT_COLUMN,
        adapter_identity=ADAPTER_IDENTITY,
        adapter_version=_distribution_version("portlearn"),
        row_count=0,
        header_order=[],
        units="unspecified",
        frequency="unspecified",
        url=sanitized_url,
        realtime_start=query["realtime_start"],
        realtime_end=query["realtime_end"],
        attribution=_ATTRIBUTION,
        data_mode=CURRENT_SNAPSHOT,
        request_params=dict(query),
        declared_lag_note=_NO_LAG_NOTE,
    )
    return data, provenance


# --------------------------------------------------------------------------- #
# The retrieval-only (UNQUALIFIED) fetch — exact bytes, retrieval facts only
# --------------------------------------------------------------------------- #


def fetch_raw(
    series_id: str,
    api_key: str,
    params: dict[str, str] | None = None,
) -> tuple[bytes, FREDRetrievalProvenance]:
    """Fetch one series' exact bytes plus retrieval-only provenance.

    The retrieval-only sibling of :func:`fetch`: the same shared
    machinery (key validated before any socket opens, labeled realtime
    window always sent, key sent to the provider inside the request URL
    only, sanitized URL/params, no vintage parameters), with the
    provenance carrying retrieval facts only — there is no
    ``availability`` field and no policy argument anywhere on this
    path, because a raw retrieval claims nothing about decision-time
    eligibility.  ``api_key`` is a required positional argument: this
    surface never reads credentials from the environment.
    """
    data, query, sanitized_url = _request_observations(
        series_id, api_key, params
    )
    return data, FREDRetrievalProvenance(
        source_id=f"fred:{series_id}",
        retrieval_instant=_now_utc(),
        content_sha256=_sha256(data),
        licence_note=_LICENCE_NOTE,
        adapter_identity=ADAPTER_IDENTITY,
        adapter_version=_distribution_version("portlearn"),
        url=sanitized_url,
        realtime_start=query["realtime_start"],
        realtime_end=query["realtime_end"],
        attribution=_ATTRIBUTION,
        data_mode=CURRENT_SNAPSHOT,
        request_params=tuple(sorted(query.items())),
        series_id=str(series_id),
    )


# --------------------------------------------------------------------------- #
# The pure decoder — local bytes to declared-table form
# --------------------------------------------------------------------------- #


def decode(
    data: bytes,
    series_id: str,
    availability: ingestion.AvailabilityPolicy,
    *,
    data_mode: str = CURRENT_SNAPSHOT,
    tzinfo: Any = None,
    frequency: str | None = None,
    metadata_bytes: bytes | None = None,
    retrieval: FREDProvenance | None = None,
) -> tuple[list[Any], FREDProvenance]:
    """Decode provider-format JSON bytes to records plus provenance.

    Pure and network-free: identical bytes decode to identical records.
    Every record is constructed, validated, and availability-stamped by
    the single ingestion chokepoint through a declared table schema; this
    module performs no private record validation and stamps no default
    availability.  Monthly/quarterly period starts map to period-END
    instants through the shared substrate; daily dates map to the day's
    last instant in the declared zone.  The missing marker ``"."`` decodes
    to an absent record.

    The declared availability policy is honored only **above** the
    retrieval lower bound: every ``CURRENT_SNAPSHOT`` record's
    ``available_time`` is bounded below by the retrieval snapshot instant (the
    ``retrieval`` provenance's instant when supplied, else the last
    instant of the payload's realtime-end day in the declared zone), and
    a declared fixed lag rides solely as non-binding research metadata
    release practice.  ``POINT_IN_TIME`` rejects fail-closed.  When
    ``retrieval`` provenance from :func:`fetch` is supplied, its content
    hash must match ``data`` exactly and its data mode must agree.
    """
    mode = _require_mode(data_mode)
    policy = _require_policy(availability)
    _require_timezone(tzinfo)
    if frequency is None:
        raise ValueError(
            "frequency is mandatory and this adapter supplies no default: "
            "declare the series frequency (Monthly, Quarterly, or Daily) "
            "so the period mapping is never inferred."
        )
    declared_frequency = _require_frequency(frequency)
    if retrieval is not None:
        if retrieval.data_mode != mode:
            raise ValueError(
                f"decode data_mode {mode!r} disagrees with the retrieval "
                f"provenance data_mode {retrieval.data_mode!r}; a decode "
                "may never relabel the mode its retrieval carried."
            )
        if retrieval.content_sha256 != _sha256(data):
            raise ValueError(
                "retrieval provenance hash pin mismatch: the bytes handed "
                "to decode are not the bytes the fetch recorded; "
                "hash-pinned decoding fails closed."
            )
    if mode == POINT_IN_TIME:
        raise PseudoVintageError(
            "POINT_IN_TIME requires source-supplied, verifiable "
            "realtime/vintage evidence (realtime_start/realtime_end/"
            "vintage_dates or an equivalent source mechanism), and this adapter "
            "implements no such evidence path; the data mode rejects "
            "fail-closed (an unsupported operation for this provider)."
        )
    payload = _parse_payload(data)
    observations = _require_observation_payload(payload, series_id)
    units = "unspecified"
    if metadata_bytes is not None:
        metadata = _parse_payload(metadata_bytes)
        metadata_id = metadata.get("id")
        if metadata_id != series_id:
            raise ProviderResponseError(
                f"the metadata series id {metadata_id!r} does not match "
                f"the requested series {series_id!r}; mismatched "
                "metadata rejects fail-closed."
            )
        metadata_units = metadata.get("units")
        if isinstance(metadata_units, str) and metadata_units.strip():
            units = metadata_units
        metadata_frequency = metadata.get("frequency")
        if metadata_frequency is not None:
            _require_frequency(metadata_frequency)
            if metadata_frequency != declared_frequency:
                raise ValueError(
                    f"declared frequency {declared_frequency!r} disagrees "
                    f"with the metadata frequency {metadata_frequency!r}; "
                    "a contradicting declaration rejects fail-closed."
                )
    if retrieval is not None:
        lower_bound: datetime = to_instant(
            retrieval.retrieval_instant, "retrieval bound"
        )
    else:
        lower_bound = to_instant(
            _realtime_day_last_instant(
                str(payload["realtime_end"]), tzinfo
            ),
            "retrieval bound",
        )
    rows: list[dict[str, Any]] = []
    for observation in observations:
        if not isinstance(observation, dict):
            raise ProviderResponseError(
                f"observation entry {observation!r} is not an object; the "
                "provider format is not recognized, fail-closed."
            )
        cell = observation.get("value")
        if cell is None or cell == _MISSING_MARKER:
            continue  # an explicit absence — never imputed
        date_key = _require_date_key(observation.get("date"))
        observation_time = _period_instant(
            date_key, declared_frequency, tzinfo
        )
        declared_instant: datetime | None = None
        if policy.mode == "FIXED_LAG":
            declared_instant = observation_time + policy.delta
        elif policy.mode == "SAME_INSTANT":
            declared_instant = observation_time
        if declared_instant is None or to_instant(
            declared_instant, "declared"
        ) < lower_bound:
            available_time = lower_bound
        else:
            available_time = to_instant(declared_instant, "declared")
        rows.append(
            {
                "series_id": _SERIES_PREFIX + series_id,
                "observation_time": observation_time.isoformat(),
                "available_time": available_time.isoformat(),
                "value": _parse_value(cell),
            }
        )
    schema = ingestion.DeclaredTableSchema(
        series_id_column="series_id",
        observation_time_column="observation_time",
        value_column="value",
        availability=_BOUNDED_EXPLICIT_COLUMN,
        units=units,
        frequency=declared_frequency,
    )
    records, _ = ingestion.from_records(rows, schema)
    del _
    provenance = FREDProvenance(
        source_id=f"fred:{series_id}",
        retrieval_instant=lower_bound,
        content_sha256=_sha256(data),
        licence_note=_LICENCE_NOTE,
        availability=policy,
        adapter_identity=ADAPTER_IDENTITY,
        adapter_version=_distribution_version("portlearn"),
        row_count=len(records),
        header_order=["value"],
        units=units,
        frequency=declared_frequency,
        url=retrieval.url if retrieval is not None else "",
        realtime_start=str(payload["realtime_start"]),
        realtime_end=str(payload["realtime_end"]),
        attribution=_ATTRIBUTION,
        data_mode=mode,
        request_params=(
            dict(retrieval.request_params) if retrieval is not None else {}
        ),
        declared_lag_note=_lag_note(policy),
    )
    return records, provenance


# --------------------------------------------------------------------------- #
# The retrieval-only (UNQUALIFIED) decoder — period labels, never instants
# --------------------------------------------------------------------------- #


def _period_label(date_key: str, frequency: str | None) -> str:
    """The provider's own period label at the declared frequency.

    The provider labels Monthly and Quarterly observations with their
    period-start date (``YYYY-MM-01``); the period itself is named by
    the provider's ``YYYY-MM`` period label, so the boilerplate day
    component is dropped — a truncation of the provider's own label,
    never a calendar interpretation and never an instant.  Daily
    observations carry the provider's full ``YYYY-MM-DD`` date label
    verbatim: a day is its own period.  With no declared frequency the
    provider date string is carried exactly as received.
    """
    if frequency in ("Monthly", "Quarterly"):
        return date_key[:7]
    return date_key


def decode_unqualified(
    data: bytes,
    series_id: str,
    *,
    frequency: str | None = None,
    metadata_bytes: bytes | None = None,
    retrieval: FREDRetrievalProvenance | None = None,
) -> tuple[list[PeriodKeyObservation], FREDRetrievalProvenance]:
    """Decode provider-format JSON bytes to period-key records.

    The retrieval-only sibling of :func:`decode`: pure,
    network-free, and deterministic — identical bytes decode to
    identical records and identical provenance, so no clock is read on
    this path (the offline retrieval instant is the fixed UTC epoch).
    Each record carries the provider's own period label in
    ``period_key`` (``YYYY-MM`` for Monthly/Quarterly period starts,
    the full ``YYYY-MM-DD`` date for Daily): placing a period on the
    single timeline is exactly the qualification step this decode
    deliberately omits, so no record defines an ``observation_time``
    or an ``available_time`` attribute and no availability policy is
    consumed, defaulted, or bounded.  The missing marker ``"."``
    decodes to an absent record exactly as the qualified decoder does,
    reusing the same private parse primitives.

    When ``retrieval`` provenance from :func:`fetch_raw` is supplied,
    its content hash must match ``data`` exactly (hash-pinned decode,
    the frozen-decoder pin mirrored by this sibling) and its retrieval
    facts (URL, retrieval instant, request parameters, realtime
    window) are adopted into the decode provenance.
    """
    declared_frequency: str | None = None
    if frequency is not None:
        declared_frequency = _require_frequency(frequency)
    if retrieval is not None and retrieval.content_sha256 != _sha256(data):
        raise ValueError(
            "retrieval provenance hash pin mismatch: the bytes handed to "
            "decode are not the bytes the fetch recorded "
            f"({retrieval.content_sha256!r} pinned, "
            f"{_sha256(data)!r} received); hash-pinned decoding "
            "fails closed."
        )
    payload = _parse_payload(data)
    observations = _require_observation_payload(payload, series_id)
    units = "unspecified"
    if metadata_bytes is not None:
        metadata = _parse_payload(metadata_bytes)
        metadata_id = metadata.get("id")
        if metadata_id != series_id:
            raise ProviderResponseError(
                f"the metadata series id {metadata_id!r} does not match "
                f"the requested series {series_id!r}; mismatched "
                "metadata rejects fail-closed."
            )
        metadata_units = metadata.get("units")
        if isinstance(metadata_units, str) and metadata_units.strip():
            units = metadata_units
        metadata_frequency = metadata.get("frequency")
        if metadata_frequency is not None:
            checked = _require_frequency(metadata_frequency)
            if (
                declared_frequency is not None
                and metadata_frequency != declared_frequency
            ):
                raise ValueError(
                    f"declared frequency {declared_frequency!r} disagrees "
                    f"with the metadata frequency {metadata_frequency!r}; "
                    "a contradicting declaration rejects fail-closed."
                )
            declared_frequency = checked
    records: list[PeriodKeyObservation] = []
    for observation in observations:
        if not isinstance(observation, dict):
            raise ProviderResponseError(
                f"observation entry {observation!r} is not an object; the "
                "provider format is not recognized, fail-closed."
            )
        cell = observation.get("value")
        if cell is None or cell == _MISSING_MARKER:
            continue  # an explicit absence — never imputed
        date_key = _require_date_key(observation.get("date"))
        records.append(
            PeriodKeyObservation(
                series_id=_SERIES_PREFIX + series_id,
                period_key=_period_label(date_key, declared_frequency),
                value=_parse_value(cell),
            )
        )
    provenance = FREDRetrievalProvenance(
        source_id=f"fred:{series_id}",
        retrieval_instant=(
            retrieval.retrieval_instant if retrieval is not None
            else _OFFLINE_RETRIEVAL_INSTANT
        ),
        content_sha256=_sha256(data),
        licence_note=_LICENCE_NOTE,
        adapter_identity=ADAPTER_IDENTITY,
        adapter_version=_distribution_version("portlearn"),
        units=units,
        frequency=declared_frequency if declared_frequency else "unspecified",
        url=retrieval.url if retrieval is not None else "",
        last_modified=(
            retrieval.last_modified if retrieval is not None else None
        ),
        attribution=_ATTRIBUTION,
        data_mode=CURRENT_SNAPSHOT,
        request_params=(
            tuple(retrieval.request_params) if retrieval is not None else ()
        ),
        realtime_start=str(payload["realtime_start"]),
        realtime_end=str(payload["realtime_end"]),
        series_id=str(series_id),
    )
    return records, provenance


# --------------------------------------------------------------------------- #
# The public false-vintage fence
# --------------------------------------------------------------------------- #


def require_current_snapshot_recency(
    records: Any, retrieval_instant: datetime
) -> None:
    """Permit only post-retrieval availability on snapshot records, or reject.

    Every ``CURRENT_SNAPSHOT`` record's ``available_time`` must sit at or
    after ``retrieval_instant``: a revised snapshot
    carrying an assumed historical — pre-retrieval — availability is a
    pseudo-vintage, and constructing one rejects with
    ``PseudoVintageError``.  Accepts a single record or an iterable.
    """
    items = (
        [records]
        if hasattr(records, "available_time")
        else list(records)
    )
    bound = to_instant(retrieval_instant, "retrieval bound")
    for item in items:
        available = to_instant(item.available_time, "available_time")
        if available < bound:
            raise PseudoVintageError(
                "pseudo-vintage rejected: a CURRENT_SNAPSHOT record may "
                "not carry availability "
                f"{available.isoformat()} preceding the retrieval bound "
                f"{bound.isoformat()} — a revised snapshot "
                "cannot receive an assumed historical availability "
                "timestamp and cannot support point-in-time claims."
            )
