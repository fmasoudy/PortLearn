"""Fama/French research-data adapter — a pure decoder plus a thin fetcher.

The decoder reduces the provider's ZIP-wrapped CSV/TXT publication format
(monthly/daily research factors and the 49-industry portfolio monthly
returns) to the ingestion declared-table form: every record is constructed
and validated by the single ingestion chokepoint
(:func:`portlearn.data.ingestion.from_records`) — this module never constructs a
record, never restates the identity/chronology/availability validation, and
never stamps a default availability.  Monthly period keys map to period
end instants through the shared period-calendar substrate
(:func:`portlearn.calendar.month_end_instant`); daily date keys map to the
calendar day's last instant in the caller-declared zone — never a market
close.  Missing-data sentinels (``-99.99`` / ``-999``) decode to absent
records, never to ``None`` or invented zeros.

Dataset kinds are a closed set: research factors (``factor``) and research
portfolio returns (``portfolio_returns``).  The kind is semantic, never an
investability claim: **factor datasets are not investable** and can never
enter a candidate asset-return path; ``portfolio_returns`` datasets carry
no automatic-investability claim either — they only permit candidate
asset-return input semantics for researcher-side code.  No dataset
delivered by this adapter is investable, and none may ever be claimed to
be.

Fetching is a separate stdlib-only function (``urllib.request`` +
``zipfile``) whose only outputs are bytes plus retrieval provenance
(URL, retrieval UTC instant, sha256, ``Last-Modified`` when present,
copyright line, dataset kind, units/frequency, availability declaration,
not-investable warning).  The decoder is network-independent and pure:
identical bytes decode to identical records.  All committed fixtures are
clearly labeled SYNTHETIC provider-format replicas (see the fixture
MANIFEST) containing invented values — no provider-owned data is
redistributed.

Beside the qualified surface this module also hosts the
retrieval-only (UNQUALIFIED) sibling surface: :func:`fetch_raw` returns
the provider's exact bytes with retrieval-only provenance (retrieval
facts only — no availability field at all), and
:func:`decode_unqualified` reduces locally-held bytes to
:class:`PeriodKeyObservation` records carrying the provider's own period
labels verbatim — never an instant, never a fabricated availability, and
never a default policy.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version as _distribution_version
from typing import Any
from urllib.request import Request, urlopen

from portlearn.calendar import month_end_instant
from portlearn.data import ingestion
from portlearn.observations import TimedObservation

from .._records import PeriodKeyObservation, RetrievalProvenance

__all__ = [
    "ADAPTER_IDENTITY",
    "ADAPTER_VERSION",
    "DATASET_KINDS",
    "FF_DATASETS",
    "FFDatasetEntry",
    "FFProvenance",
    "FFRetrievalProvenance",
    "PeriodKeyObservation",
    "RetrievalProvenance",
    "UnknownDatasetError",
    "decode",
    "decode_unqualified",
    "fetch",
    "fetch_raw",
    "require_candidate_asset_returns",
]

ADAPTER_IDENTITY = "portlearn.data.adapters.ff"

#: Catalog/decoder revision (manifest-recorded); bumped on any catalog change.
ADAPTER_VERSION = 1

#: Closed dataset-kind set: a semantic separation of research
#: factors from research portfolio returns — never an investability claim.
DATASET_KINDS = ("factor", "portfolio_returns")

_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

_LICENCE_NOTE = (
    "Provider-format data; upstream copyright Eugene F. Fama and Kenneth "
    "R. French (committed fixtures are synthetic replicas holding invented "
    "values — no provider-owned data redistributed). Research use only; "
    "not investable and never an investability claim."
)

_NOT_INVESTABLE = (
    "Research dataset decoded by portlearn.data.adapters.ff; "
    "NOT investable — "
    "no factor or industry-portfolio series may be treated as a tradable "
    "asset return; portfolio_returns kinds permit candidate asset-return "
    "input semantics only, without any automatic investability claim."
)

#: Provider missing-data sentinels: an observation marked with one of these
#: is an explicit absence — no record is emitted for it, ever.
_MISSING_SENTINELS = (-99.99, -999.0, -999)

_SERIES_PREFIX = "FF/"

#: The deterministic offline retrieval instant (UTC epoch).  An offline
#: ``decode_unqualified`` (no ``retrieval`` supplied) must stay pure —
#: identical bytes decode to identical provenance — so it never reads the
#: clock; it pins this one fixed timezone-aware instant instead.  The
#: qualified decoder that adopts such a provenance as its ``retrieval=``
#: pin adopts this same aware instant.
_OFFLINE_RETRIEVAL_INSTANT = datetime(1970, 1, 1, tzinfo=UTC)


class UnknownDatasetError(ValueError):
    """The dataset identifier names no catalog entry (strict)."""


@dataclass(frozen=True)
class FFDatasetEntry:
    """An immutable catalog entry: URL, member, kind, units, frequency."""

    dataset_id: str
    url: str
    member_name: str
    dataset_kind: str
    units: str
    frequency: str
    delimiter: str
    fixed_width: bool
    section_title: str | None
    notes: str
    licence_note: str

    def __post_init__(self) -> None:
        if self.dataset_kind not in DATASET_KINDS:
            raise ValueError(
                "dataset_kind must be one of the closed set "
                f"{DATASET_KINDS!r} — a semantic separation of research "
                "factors from research portfolio returns, never an "
                f"investability claim; got {self.dataset_kind!r}."
            )


FF_DATASETS: dict[str, FFDatasetEntry] = {
    "factors_monthly_csv": FFDatasetEntry(
        dataset_id="factors_monthly_csv",
        url=_BASE_URL + "F-F_Research_Data_Factors_CSV.zip",
        member_name="F-F_Research_Data_Factors.csv",
        dataset_kind="factor",
        units="percent",
        frequency="monthly",
        delimiter=",",
        fixed_width=False,
        section_title=None,
        notes=(
            "Monthly research factors (Mkt-RF, SMB, HML, RF) in the "
            "comma-separated variant. The provider's 2025 research-file "
            "methodology revision is recorded here as a regime fact, "
            "never adjusted for."
        ),
        licence_note=_LICENCE_NOTE,
    ),
    "factors_monthly_txt": FFDatasetEntry(
        dataset_id="factors_monthly_txt",
        url=_BASE_URL + "F-F_Research_Data_Factors_TXT.zip",
        member_name="F-F_Research_Data_Factors.txt",
        dataset_kind="factor",
        units="percent",
        frequency="monthly",
        delimiter=" ",
        fixed_width=True,
        section_title=None,
        notes=(
            "Monthly research factors in the fixed-width text variant; "
            "must decode to observations identical to the CSV variant."
        ),
        licence_note=_LICENCE_NOTE,
    ),
    "factors_daily_csv": FFDatasetEntry(
        dataset_id="factors_daily_csv",
        url=_BASE_URL + "F-F_Research_Data_Factors_daily_CSV.zip",
        member_name="F-F_Research_Data_Factors_daily.csv",
        dataset_kind="factor",
        units="percent",
        frequency="daily",
        delimiter=",",
        fixed_width=False,
        section_title=None,
        notes=(
            "Daily research factors; each trading date maps to that "
            "calendar day's last instant in the caller-declared zone."
        ),
        licence_note=_LICENCE_NOTE,
    ),
    "industry49_monthly_csv": FFDatasetEntry(
        dataset_id="industry49_monthly_csv",
        url=_BASE_URL + "49_Industry_Portfolios_CSV.zip",
        member_name="49_Industry_Portfolios.csv",
        dataset_kind="portfolio_returns",
        units="percent",
        frequency="monthly",
        delimiter=",",
        fixed_width=False,
        section_title="Average Value Weighted Returns -- Monthly",
        notes=(
            "49 industry portfolios, average value-weighted monthly "
            "returns (first monthly section only; later sections are "
            "never decoded). Candidate asset-return inputs only; NOT "
            "investable."
        ),
        licence_note=_LICENCE_NOTE,
    ),
}


@dataclass(frozen=True)
class FFProvenance(ingestion.SourceProvenance):
    """Retrieval provenance for a fetched or decoded dataset.

    A sibling value object extending the ingestion module's ``SourceProvenance`` with
    the provider-layer retrieval facts: source URL,
    ``Last-Modified`` when the provider supplied it, the copyright/
    attribution line, the closed ``dataset_kind``, the catalog
    ``dataset_id``, and the not-investable warning.  The ``SourceProvenance`` shape is never modify — extension is by
    subclass, never by editing the ingestion module.
    """

    url: str = ""
    last_modified: str | None = None
    copyright_line: str | None = None
    dataset_kind: str = ""
    dataset_id: str = ""
    not_investable_warning: str = _NOT_INVESTABLE

    def __post_init__(self) -> None:
        if self.dataset_kind not in DATASET_KINDS:
            raise ValueError(
                "dataset_kind must be one of the closed set "
                f"{DATASET_KINDS!r} — a semantic separation of research "
                "factors from research portfolio returns, never an "
                "investability claim; got "
                f"{self.dataset_kind!r}."
            )


@dataclass(frozen=True)
class FFRetrievalProvenance(RetrievalProvenance):
    """Retrieval-only provenance: retrieval facts, no availability field.

    The retrieval-only sibling of :class:`FFProvenance`: the same
    provider retrieval facts (URL, retrieval instant, content hash,
    ``Last-Modified``, attribution trailer, dataset kind, units,
    frequency, row count, header order) minus every availability fact.
    There is deliberately **no** ``availability`` field and this record is
    deliberately *not* a
    :class:`~portlearn.data.ingestion.SourceProvenance`: a retrieval-only
    load can never satisfy the qualified provenance contract.  Immutable
    and hashable end-to-end (``header_order`` is an immutable tuple of
    column names).
    """

    source_id: str = ""
    retrieval_instant: datetime = _OFFLINE_RETRIEVAL_INSTANT
    content_sha256: str = ""
    licence_note: str = _LICENCE_NOTE
    adapter_identity: str = ADAPTER_IDENTITY
    adapter_version: str = ""
    units: str = ""
    frequency: str = ""
    url: str = ""
    last_modified: str | None = None
    copyright_line: str | None = None
    dataset_kind: str = ""
    dataset_id: str = ""
    not_investable_warning: str = _NOT_INVESTABLE
    row_count: int = 0
    header_order: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dataset_kind not in DATASET_KINDS:
            raise ValueError(
                "dataset_kind must be one of the closed set "
                f"{DATASET_KINDS!r} — a semantic separation of research "
                "factors from research portfolio returns, never an "
                "investability claim; got "
                f"{self.dataset_kind!r}."
            )


# --------------------------------------------------------------------------- #
# Admission (unconditional, no defaults)
# --------------------------------------------------------------------------- #


def _require_catalog_entry(dataset_id: Any) -> FFDatasetEntry:
    if not isinstance(dataset_id, str) or dataset_id not in FF_DATASETS:
        raise UnknownDatasetError(
            f"no catalog entry named {dataset_id!r}: the fixed catalog "
            f"holds exactly {sorted(FF_DATASETS)}; dataset identifiers "
            "are exact-match only and reject otherwise."
        )
    return FF_DATASETS[dataset_id]


def _require_policy(availability: Any) -> ingestion.AvailabilityPolicy:
    if not isinstance(availability, ingestion.AvailabilityPolicy):
        raise ValueError(  # noqa: TRY004 — admission is uniformly ValueError
            "availability is mandatory and this adapter supplies no "
            "default: pass an explicit AvailabilityPolicy "
            "(explicit_column, same_instant, or fixed_lag with a "
            "justification). A silent default availability would be a "
            f"silent leakage assumption; got {availability!r}."
        )
    return availability


# --------------------------------------------------------------------------- #
# Provider-format parsing (structure only — records are the chokepoint's)
# --------------------------------------------------------------------------- #


def _parse_header_line(line: str, fixed_width: bool) -> list[str]:
    """Column names from a provider header line, padding stripped."""
    if fixed_width:
        return [name.strip() for name in line.split()]
    names = [name.strip() for name in line.split(",")]
    return [name for name in names if name]


def _cell_values(line: str, fixed_width: bool) -> list[str]:
    """Observation cells from one provider data line (key excluded)."""
    if fixed_width:
        return line.split()[1:]
    return line.split(",")[1:]


def _parse_value(cell: str) -> tuple[bool, float]:
    """(present, value) for one cell; sentinels and blanks are absent."""
    stripped = cell.strip()
    if not stripped:
        return False, 0.0
    try:
        value = float(stripped)
    except ValueError:
        raise ValueError(
            f"provider cell {stripped!r} is neither a numeric "
            "observation nor a missing-data sentinel; the decoder "
            "rejects unparseable provider bytes unconditional."
        ) from None
    if value in _MISSING_SENTINELS:
        return False, 0.0
    return True, value


def _monthly_key(yyyymm: str) -> tuple[int, int]:
    """(year, month) from a six-digit monthly period key, ."""
    if len(yyyymm) != 6 or not yyyymm.isdigit():
        raise ValueError(
            f"monthly period key {yyyymm!r} must be six digits YYYYMM."
        )
    year, month = int(yyyymm[:4]), int(yyyymm[4:])
    if not 1 <= month <= 12:
        raise ValueError(
            f"monthly period key {yyyymm!r} names no calendar month."
        )
    return year, month


def _daily_instant(yyyymmdd: str, tzinfo: Any) -> datetime:
    """The calendar day's LAST aware instant in the declared zone.

    Deliberately not a market close: no trading-session time (such as a
    16:00 bell) is assumed anywhere in this adapter — only the declared
    zone's final microsecond of the calendar day.
    """
    if len(yyyymmdd) != 8 or not yyyymmdd.isdigit():
        raise ValueError(
            f"daily date key {yyyymmdd!r} must be eight digits YYYYMMDD."
        )
    return datetime(
        int(yyyymmdd[:4]),
        int(yyyymmdd[4:6]),
        int(yyyymmdd[6:]),
        23,
        59,
        59,
        999999,
        tzinfo=tzinfo,
    )


def _find_factor_header(lines: list[str]) -> tuple[int, list[str], bool]:
    """(index, names, fixed_width) of the first factor header line."""
    for index, line in enumerate(lines):
        if "Mkt-RF" in line and "SMB" in line and "RF" in line:
            fixed_width = "," not in line
            return index, _parse_header_line(line, fixed_width), fixed_width
    raise ValueError(
        "no factor column header found in the member text; the provider "
        "format is not recognized, and reject."
    )


def _find_section_header(
    lines: list[str], section_title: str
) -> tuple[int, list[str]]:
    """(header index, names) for the named section's column header."""
    target = section_title.strip()
    for index, line in enumerate(lines[:-1]):
        if line.strip() == target:
            header = lines[index + 1]
            return index + 1, _parse_header_line(header, False)
    raise ValueError(
        f"no section titled {section_title!r} found in the member text; "
        "the provider format is not recognized, and reject."
    )


def _data_rows(
    text: str, entry: FFDatasetEntry
) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """(column names, [(period key, cells)]) from the decoded data block.

    The block ends at its first blank line after data begins — later
    sections (annual blocks, other weighting schemes, firm counts) are
    never decoded as observations.
    """
    lines = text.splitlines()
    if entry.section_title is None:
        start, names, fixed_width = _find_factor_header(lines)
    else:
        start, names = _find_section_header(lines, entry.section_title)
        fixed_width = False
    rows: list[tuple[str, list[str]]] = []
    for line in lines[start + 1 :]:
        if not line.strip():
            if rows:
                break
            continue
        key = line.split(",")[0].strip() if not fixed_width else (
            line.split()[0]
        )
        if not key.isdigit():
            break
        rows.append((key, _cell_values(line, fixed_width)))
    if not rows:
        raise ValueError(
            "the decoded data block holds no observation rows; the "
            "provider format is not recognized, and reject."
        )
    return names, rows


def _copyright_line(data: bytes, member_name: str) -> str | None:
    """The provider attribution trailer line, when present (provenance)."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            text = archive.read(member_name).decode("ascii")
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError):
        return None
    for line in text.splitlines():
        if line.strip().startswith("Copyright"):
            return line.strip()
    return None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now_utc() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- #
# The thin fetch layer — bytes plus retrieval provenance, nothing else
# --------------------------------------------------------------------------- #


def fetch(
    dataset_id: str, availability: ingestion.AvailabilityPolicy
) -> tuple[bytes, FFProvenance]:
    """Fetch one catalog dataset's ZIP bytes plus retrieval provenance.

    Stdlib-only (``urllib.request``); the provider's bytes are returned
    exactly as received — no parsing of observations happens here.  The
    returned provenance records the URL, the retrieval UTC instant, the
    sha256 of the exact bytes, ``Last-Modified`` when the provider sent
    it, the attribution trailer line, the closed dataset kind, units and
    frequency, the declared availability policy, and the not-investable
    warning.  ``availability`` is mandatory: this function never supplies
    a default availability.
    """
    entry = _require_catalog_entry(dataset_id)
    policy = _require_policy(availability)
    request = Request(entry.url, headers={"User-Agent": "portlearn-adapter"})
    with urlopen(request, timeout=60) as response:
        data = response.read()
        last_modified = response.headers.get("Last-Modified")
    return data, FFProvenance(
        source_id=f"ff:{entry.dataset_id}",
        retrieval_instant=_now_utc(),
        content_sha256=_sha256(data),
        licence_note=entry.licence_note,
        availability=policy,
        adapter_identity=ADAPTER_IDENTITY,
        adapter_version=_distribution_version("portlearn"),
        row_count=0,
        header_order=[],
        units=entry.units,
        frequency=entry.frequency,
        url=entry.url,
        last_modified=last_modified,
        copyright_line=_copyright_line(data, entry.member_name),
        dataset_kind=entry.dataset_kind,
        dataset_id=entry.dataset_id,
        not_investable_warning=_NOT_INVESTABLE,
    )


# --------------------------------------------------------------------------- #
# The retrieval-only (UNQUALIFIED) fetch — exact bytes, retrieval facts only
# --------------------------------------------------------------------------- #


def fetch_raw(dataset_id: str) -> tuple[bytes, FFRetrievalProvenance]:
    """Fetch one catalog dataset's exact ZIP bytes plus retrieval facts.

    The retrieval-only sibling of :func:`fetch`: stdlib-only, the
    provider's bytes returned exactly as received, and the provenance
    carries retrieval facts only — there is no ``availability`` field
    and no policy argument anywhere on this path, because a raw
    retrieval claims nothing about decision-time eligibility.  An
    unknown dataset id rejects before any socket is opened.
    """
    entry = _require_catalog_entry(dataset_id)
    request = Request(entry.url, headers={"User-Agent": "portlearn-adapter"})
    with urlopen(request, timeout=60) as response:
        data = response.read()
        last_modified = response.headers.get("Last-Modified")
    return data, FFRetrievalProvenance(
        source_id=f"ff:{entry.dataset_id}",
        retrieval_instant=_now_utc(),
        content_sha256=_sha256(data),
        licence_note=entry.licence_note,
        adapter_identity=ADAPTER_IDENTITY,
        adapter_version=_distribution_version("portlearn"),
        units=entry.units,
        frequency=entry.frequency,
        url=entry.url,
        last_modified=last_modified,
        copyright_line=_copyright_line(data, entry.member_name),
        dataset_kind=entry.dataset_kind,
        dataset_id=entry.dataset_id,
        not_investable_warning=_NOT_INVESTABLE,
        row_count=0,
        header_order=(),
    )


# --------------------------------------------------------------------------- #
# The pure decoder — local bytes to declared-table form
# --------------------------------------------------------------------------- #


def decode(
    data: bytes,
    dataset_id: str,
    availability: ingestion.AvailabilityPolicy,
    *,
    tzinfo: Any = None,
    tz: Any = None,
    retrieval: FFProvenance | None = None,
) -> tuple[list[TimedObservation], FFProvenance]:
    """Decode provider-format ZIP bytes to records plus provenance.

    Pure and network-free: identical bytes decode to identical records.
    Every record is constructed, validated, and availability-stamped by
    the single ingestion chokepoint through a declared table schema; this
    module performs no private record validation and stamps no default
    availability.  Monthly keys map to period end instants through the
    shared period-calendar substrate; daily keys map to the day's last
    instant in the declared zone.  Sentinels decode to absent records.

    When ``retrieval`` provenance from :func:`fetch` is supplied, its
    content hash must match ``data`` exactly (hash-pinned decode) and its
    retrieval facts (URL, retrieval instant, ``Last-Modified``) are
    adopted into the decode provenance.
    """
    entry = _require_catalog_entry(dataset_id)
    policy = _require_policy(availability)
    if tzinfo is None and tz is None:
        raise ValueError(
            "decode requires an explicit aware timezone (tzinfo=...); "
            "no default zone is ever assumed — a naive mapping cannot "
            "be placed on the single timeline."
        )
    zone = tzinfo if tzinfo is not None else tz
    if retrieval is not None and retrieval.content_sha256 != _sha256(data):
        raise ValueError(
            "retrieval provenance hash pin mismatch: the bytes "
            "handed to decode are not the bytes the fetch recorded "
            f"({retrieval.content_sha256!r} pinned, "
            f"{_sha256(data)!r} received); hash-pinned decoding "
            "fails closed."
        )
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.namelist()
        if members != [entry.member_name]:
            raise ValueError(
                f"provider ZIP members {members!r} do not match the "
                f"catalog member {entry.member_name!r}, and reject."
            )
        text = archive.read(entry.member_name).decode("ascii")
    names, rows = _data_rows(text, entry)
    schema = ingestion.DeclaredTableSchema(
        series_id_column="series_id",
        observation_time_column="observation_time",
        value_column="value",
        availability=policy,
        units=entry.units,
        frequency=entry.frequency,
    )
    table: list[dict[str, Any]] = []
    for key, cells in rows:
        if entry.frequency == "monthly":
            year, month = _monthly_key(key)
            observation_time: datetime = month_end_instant(year, month, zone)
        else:
            observation_time = _daily_instant(key, zone)
        for name, cell in zip(names, cells, strict=False):
            present, value = _parse_value(cell)
            if present:
                table.append(
                    {
                        "series_id": _SERIES_PREFIX + name,
                        "observation_time": observation_time.isoformat(),
                        "value": value,
                    }
                )
    records, _ = ingestion.from_records(table, schema)
    del _
    provenance = FFProvenance(
        source_id=f"ff:{entry.dataset_id}",
        retrieval_instant=(
            retrieval.retrieval_instant if retrieval is not None
            else _now_utc()
        ),
        content_sha256=_sha256(data),
        licence_note=entry.licence_note,
        availability=policy,
        adapter_identity=ADAPTER_IDENTITY,
        adapter_version=_distribution_version("portlearn"),
        row_count=len(records),
        header_order=list(names),
        units=entry.units,
        frequency=entry.frequency,
        url=retrieval.url if retrieval is not None else entry.url,
        last_modified=(
            retrieval.last_modified if retrieval is not None else None
        ),
        copyright_line=_copyright_line(data, entry.member_name),
        dataset_kind=entry.dataset_kind,
        dataset_id=entry.dataset_id,
        not_investable_warning=_NOT_INVESTABLE,
    )
    return records, provenance


# --------------------------------------------------------------------------- #
# The retrieval-only (UNQUALIFIED) decoder — period labels, never instants
# --------------------------------------------------------------------------- #


def decode_unqualified(
    data: bytes,
    dataset_id: str,
    *,
    retrieval: FFRetrievalProvenance | None = None,
) -> tuple[list[PeriodKeyObservation], FFRetrievalProvenance]:
    """Decode provider-format ZIP bytes to period-key records.

    The retrieval-only sibling of :func:`decode`: pure,
    network-free, and deterministic — identical bytes decode to
    identical records and identical provenance, so no clock is read on
    this path (the offline retrieval instant is the fixed UTC epoch).
    Each record carries the provider's own period label verbatim
    (``"202301"`` monthly, ``"20250817"`` daily) in ``period_key``:
    placing a period on the single timeline is exactly the
    qualification step this decode deliberately omits, so no record
    defines an ``observation_time`` or an ``available_time`` attribute
    and no availability policy is consumed or defaulted.  Sentinel
    cells decode to absent records exactly as the qualified decoder
    does, reusing the same private parse primitives.

    When ``retrieval`` provenance from :func:`fetch_raw` is supplied,
    its content hash must match ``data`` exactly (hash-pinned decode,
    mirroring the qualified decoder's pin) and its retrieval
    facts (URL, retrieval instant, ``Last-Modified``) are adopted into
    the decode provenance.
    """
    entry = _require_catalog_entry(dataset_id)
    if retrieval is not None and retrieval.content_sha256 != _sha256(data):
        raise ValueError(
            "retrieval provenance hash pin mismatch: the bytes handed to "
            "decode are not the bytes the fetch recorded "
            f"({retrieval.content_sha256!r} pinned, "
            f"{_sha256(data)!r} received); hash-pinned decoding "
            "fails closed."
        )
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.namelist()
        if members != [entry.member_name]:
            raise ValueError(
                f"provider ZIP members {members!r} do not match the "
                f"catalog member {entry.member_name!r}, and reject."
            )
        text = archive.read(entry.member_name).decode("ascii")
    names, rows = _data_rows(text, entry)
    records: list[PeriodKeyObservation] = []
    for key, cells in rows:
        for name, cell in zip(names, cells, strict=False):
            present, value = _parse_value(cell)
            if present:
                records.append(
                    PeriodKeyObservation(
                        series_id=_SERIES_PREFIX + name,
                        period_key=key,
                        value=value,
                    )
                )
    provenance = FFRetrievalProvenance(
        source_id=f"ff:{entry.dataset_id}",
        retrieval_instant=(
            retrieval.retrieval_instant if retrieval is not None
            else _OFFLINE_RETRIEVAL_INSTANT
        ),
        content_sha256=_sha256(data),
        licence_note=entry.licence_note,
        adapter_identity=ADAPTER_IDENTITY,
        adapter_version=_distribution_version("portlearn"),
        units=entry.units,
        frequency=entry.frequency,
        url=retrieval.url if retrieval is not None else entry.url,
        last_modified=(
            retrieval.last_modified if retrieval is not None else None
        ),
        copyright_line=_copyright_line(data, entry.member_name),
        dataset_kind=entry.dataset_kind,
        dataset_id=entry.dataset_id,
        not_investable_warning=_NOT_INVESTABLE,
        row_count=len(records),
        header_order=tuple(names),
    )
    return records, provenance


# --------------------------------------------------------------------------- #
# The asset-return fence
# --------------------------------------------------------------------------- #


def require_candidate_asset_returns(provenance: ingestion.SourceProvenance) -> None:
    """Permit candidate asset-return input semantics or reject invalid input.

    Only a ``portfolio_returns`` dataset may pass this fence, and passing
    it claims nothing more than candidate input semantics — never
    investability.  A ``factor`` dataset rejects: research factors must
    never flow through an asset-return path merely because they are
    numeric return columns.
    """
    kind = getattr(provenance, "dataset_kind", None)
    if kind != "portfolio_returns":
        raise ValueError(
            "dataset_kind is "
            f"{kind!r}; only portfolio_returns datasets may enter a "
            "candidate asset-return path — factor datasets are research "
            "factors, NOT investable, and never asset returns."
        )
