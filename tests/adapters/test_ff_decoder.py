"""Contract tests for the Fama/French provider-format adapter.

The adapter under test is a pure decoder over locally-held provider-format
bytes plus a thin fetch layer (stdlib ``urllib.request`` + ``zipfile`` only).
All fixtures are clearly labeled SYNTHETIC provider-format replicas built
from invented values; no provider-owned data is redistributed, and the suite
never touches the network.

Covered laws: decode-to-declared-table-form through the single ingestion
chokepoint, fetch/decode split, frozen catalog with closed dataset kinds,
sentinel-to-absence mapping, period-end and day-last instant mapping through
the shared period-calendar substrate, declared-availability stamping with no
code default, fixture hash pinning, naive-free instants, the not-investable
fence, and the factor-kind asset-return-path rejection.
"""

from __future__ import annotations

import ast
import hashlib
import zipfile
from datetime import UTC, datetime, timedelta, timezone
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Self

import pytest

from portlearn.calendar import month_end_instant
from portlearn.data.adapters import ff
from portlearn.data.ingestion import AvailabilityPolicy, SourceProvenance
from portlearn.observations import TimedObservation

# --------------------------------------------------------------------------- #
# Shared constants
# --------------------------------------------------------------------------- #

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ff"

#: Fixed southern-hemisphere-style offset (no DST movement; a declared zone
#: is supplied by the caller — the adapter never assumes a default zone).
TZ = timezone(timedelta(hours=10))

PINNED_ZIP_SHA256 = {
    "ff_factors_monthly_csv.zip": (
        "3d8737ac49bc95bed8e8fa097be07f90533b8ea667ae12e14e7fe37cf73c51c7"
    ),
    "ff_factors_monthly_txt.zip": (
        "e4c297e7b91e259fb7d5fd4fa5bbbdbeb5b53edb7517fd31c237f05457ed4a2c"
    ),
    "ff_factors_daily_csv.zip": (
        "74f9cd110c43ca2ae3714a4efdebaf5d0e94f547d6f8eed1fefc6669ff0d160f"
    ),
    "ff_industry49_monthly_csv.zip": (
        "eb4831ad8e0f32d40ef8a6b1775b2e25a229d52b6d9d835559a515f0465cb9bf"
    ),
}

FACTORS_CSV = "factors_monthly_csv"
FACTORS_TXT = "factors_monthly_txt"
FACTORS_DAILY = "factors_daily_csv"
INDUSTRY49 = "industry49_monthly_csv"

FIXTURE_FOR_DATASET = {
    FACTORS_CSV: "ff_factors_monthly_csv.zip",
    FACTORS_TXT: "ff_factors_monthly_txt.zip",
    FACTORS_DAILY: "ff_factors_daily_csv.zip",
    INDUSTRY49: "ff_industry49_monthly_csv.zip",
}


def _fixture_bytes(dataset_id: str) -> bytes:
    return (FIXTURES / FIXTURE_FOR_DATASET[dataset_id]).read_bytes()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _policy(
    delta: timedelta = timedelta(days=45),
) -> AvailabilityPolicy:
    """A caller-declared FIXED_LAG availability policy (posting practice)."""
    return AvailabilityPolicy.fixed_lag(
        delta,
        justification=(
            "Researcher declaration: provider practice regenerates the "
            "monthly research files during the middle of the following "
            "month, so a 45-day declared lag approximates first "
            "knowability of a revised snapshot."
        ),
    )


def _decode(
    dataset_id: str = FACTORS_CSV,
    *,
    data: bytes | None = None,
    availability: Any = None,
    tz: timezone = TZ,
):
    if availability is None:
        availability = _policy()
    return ff.decode(
        data if data is not None else _fixture_bytes(dataset_id),
        dataset_id,
        availability,
        tz=tz,
    )


def _find(
    records: list[TimedObservation], series_id: str, when: datetime
) -> TimedObservation | None:
    for record in records:
        if record.series_id == series_id and record.observation_time == when:
            return record
    return None


# --------------------------------------------------------------------------- #
# L1 — decode returns records + provenance, purely, without network
# --------------------------------------------------------------------------- #


def test_decode_returns_records_and_provenance_from_local_bytes() -> None:
    """Decode reduces provider-format bytes to the declared-table form."""
    records, provenance = _decode()
    assert isinstance(records, list)
    assert len(records) == 141  # 36 months x 4 series - 3 sentinel absences
    assert all(isinstance(record, TimedObservation) for record in records)
    assert isinstance(provenance, ff.FFProvenance)
    assert isinstance(provenance, SourceProvenance)  # sibling value object
    assert provenance.row_count == len(records)


def test_decode_never_touches_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """The decoder is a pure function over bytes; it never fetches."""

    def _explode(url: object) -> None:  # pragma: no cover - guard only
        raise AssertionError(f"decode must not fetch; got {url!r}")

    monkeypatch.setattr(ff, "urlopen", _explode)
    records, _ = _decode()
    assert records


def test_decoder_is_pure_over_identical_bytes() -> None:
    """Same bytes + same declaration -> identical records, every run."""
    first_records, first_provenance = _decode()
    second_records, second_provenance = _decode()
    assert first_records == second_records
    assert (
        first_provenance.content_sha256 == second_provenance.content_sha256
    )


# --------------------------------------------------------------------------- #
# L5 — period mapping through the shared substrate
# --------------------------------------------------------------------------- #


def test_monthly_period_maps_to_period_end_instant() -> None:
    """A monthly period key dates the observation at the period's end."""
    records, _ = _decode()
    leap_feb_end = month_end_instant(2024, 2, TZ)
    record = _find(records, "FF/RF", leap_feb_end)
    assert record is not None
    assert record.observation_time == datetime(
        2024, 2, 29, 23, 59, 59, 999999, tzinfo=TZ
    )


def test_monthly_mapping_matches_shared_period_calendar_substrate() -> None:
    """Every monthly instant equals the shared substrate's mapping."""
    records, _ = _decode()
    months = {
        (2023, 3),
        (2023, 1),
        (2024, 2),
        (2025, 11),
        (2025, 12),
    }
    expected = {month_end_instant(year, month, TZ) for year, month in months}
    assert {record.observation_time for record in records} >= expected


def test_leap_february_and_short_months_map_to_correct_last_day() -> None:
    """Leap February lands on the 29th; 2023 February on the 28th."""
    records, _ = _decode()
    assert _find(records, "FF/RF", month_end_instant(2024, 2, TZ)) is not None
    assert _find(records, "FF/RF", month_end_instant(2023, 2, TZ)) is not None
    # there is no Feb 29 in 2023; the substrate maps Feb 2023 to the 28th
    feb_2023 = _find(records, "FF/RF", month_end_instant(2023, 2, TZ))
    assert feb_2023 is not None
    assert feb_2023.observation_time == datetime(
        2023, 2, 28, 23, 59, 59, 999999, tzinfo=TZ
    )


def test_daily_date_maps_to_calendar_day_last_instant() -> None:
    """A daily key maps to that calendar day's last instant, declared zone.

    The trading day's close is deliberately not assumed — no market-close
    instant (e.g. 16:00) may appear.
    """
    records, _ = _decode(FACTORS_DAILY)
    first = datetime(2026, 7, 1, 23, 59, 59, 999999, tzinfo=TZ)
    record = _find(records, "FF/Mkt-RF", first)
    assert record is not None
    local = record.observation_time.astimezone(TZ)
    assert (local.hour, local.minute, local.second, local.microsecond) == (
        23,
        59,
        59,
        999999,
    )
    market_close_style = datetime(2026, 7, 1, 16, 0, 0, tzinfo=TZ)
    assert record.observation_time != market_close_style


# --------------------------------------------------------------------------- #
# L4 — missing sentinels are explicit absences
# --------------------------------------------------------------------------- #


def test_sentinels_map_to_absent_records_in_both_variants() -> None:
    """-99.99 / -999 cells emit no record — CSV and fixed-width TXT alike."""
    for dataset_id in (FACTORS_CSV, FACTORS_TXT):
        records, _ = _decode(dataset_id)
        feb_2024 = month_end_instant(2024, 2, TZ)
        assert _find(records, "FF/SMB", feb_2024) is None  # -99.99 sentinel
        assert _find(records, "FF/HML", feb_2024) is None  # -999 sentinel
        assert _find(records, "FF/Mkt-RF", feb_2024) is not None
        assert _find(records, "FF/RF", feb_2024) is not None
        nov_2025 = month_end_instant(2025, 11, TZ)
        assert _find(records, "FF/Mkt-RF", nov_2025) is None

    daily_records, _ = _decode(FACTORS_DAILY)
    jul_8 = datetime(2026, 7, 8, 23, 59, 59, 999999, tzinfo=TZ)
    assert _find(daily_records, "FF/HML", jul_8) is None
    assert _find(daily_records, "FF/SMB", jul_8) is not None

    industry_records, _ = _decode(INDUSTRY49)
    feb_2023 = month_end_instant(2023, 2, TZ)
    assert _find(industry_records, "FF/Soda", feb_2023) is None
    apr_2023 = month_end_instant(2023, 4, TZ)
    assert _find(industry_records, "FF/Hlth", apr_2023) is None
    assert _find(industry_records, "FF/Agric", feb_2023) is not None


def test_sentinels_never_become_none_or_zero_records() -> None:
    """No emitted record carries a sentinel, None, or an invented 0.0."""
    for dataset_id in FIXTURE_FOR_DATASET:
        records, _ = _decode(dataset_id)
        for record in records:
            assert record.value is not None
            assert record.value not in ff._MISSING_SENTINELS
    records, _ = _decode(FACTORS_CSV)
    feb_2024 = month_end_instant(2024, 2, TZ)
    for series_id in ("FF/SMB", "FF/HML"):
        matches = [
            record
            for record in records
            if record.series_id == series_id
            and record.observation_time == feb_2024
        ]
        assert matches == []


# --------------------------------------------------------------------------- #
# L2 — fetch provenance completeness (network monkeypatched, never live)
# --------------------------------------------------------------------------- #


class _FakeResponse:
    """Minimal response stand-in: bytes plus header lookup."""

    def __init__(self, payload: bytes, last_modified: str | None) -> None:
        self._payload = payload
        self.headers: dict[str, str] = (
            {"Last-Modified": last_modified} if last_modified else {}
        )

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def _fake_urlopen(payload: bytes, last_modified: str | None):
    def _urlopen(*args: object, **kwargs: object) -> _FakeResponse:
        return _FakeResponse(payload, last_modified)

    return _urlopen


_LAST_MODIFIED = "Fri, 04 Sep 2026 14:47:39 GMT"


def test_fetch_records_full_retrieval_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fetch provenance carries URL, instant, sha256, Last-Modified, ©."""
    payload = _fixture_bytes(FACTORS_CSV)
    monkeypatch.setattr(ff, "urlopen", _fake_urlopen(payload, _LAST_MODIFIED))
    data, provenance = ff.fetch(FACTORS_CSV, _policy())
    assert data == payload
    entry = ff.FF_DATASETS[FACTORS_CSV]
    assert provenance.url == entry.url
    assert provenance.content_sha256 == _sha256(payload)
    assert provenance.last_modified == _LAST_MODIFIED
    assert provenance.copyright_line is not None
    assert provenance.copyright_line.startswith("Copyright 2026")
    assert provenance.retrieval_instant.utcoffset() is not None
    assert provenance.dataset_kind == "factor"
    assert provenance.row_count == 0  # the fetcher never parses
    assert provenance.header_order == []


def test_fetch_records_last_modified_absent_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absent Last-Modified header is recorded as None, never guessed."""
    monkeypatch.setattr(
        ff, "urlopen", _fake_urlopen(_fixture_bytes(FACTORS_CSV), None)
    )
    _, provenance = ff.fetch(FACTORS_CSV, _policy())
    assert provenance.last_modified is None


def test_decode_adopts_retrieval_provenance_and_detects_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fetch->decode round trip carries retrieval fields; tampering fails."""
    payload = _fixture_bytes(FACTORS_CSV)
    monkeypatch.setattr(ff, "urlopen", _fake_urlopen(payload, _LAST_MODIFIED))
    _, fetched = ff.fetch(FACTORS_CSV, _policy())

    records, _ = _decode(FACTORS_CSV, availability=_policy(), data=payload)
    # good path: decode may adopt a matching retrieval provenance
    _, adopted = ff.decode(
        payload,
        FACTORS_CSV,
        _policy(),
        tz=TZ,
        retrieval=fetched,
    )
    assert adopted.last_modified == _LAST_MODIFIED
    assert adopted.retrieval_instant == fetched.retrieval_instant
    assert records

    tampered = b"drifted" + payload
    with pytest.raises(ValueError, match="hash"):
        ff.decode(
            tampered,
            FACTORS_CSV,
            _policy(),
            tz=TZ,
            retrieval=fetched,
        )


# --------------------------------------------------------------------------- #
# L3 — catalog lookup
# --------------------------------------------------------------------------- #


def test_unknown_dataset_identifier_rejects() -> None:
    """An unknown dataset id rejects fail-closed with ValueError."""
    with pytest.raises(ValueError, match="no_such_dataset"):
        ff.decode(
            _fixture_bytes(FACTORS_CSV),
            "no_such_dataset",
            _policy(),
            tz=TZ,
        )
    with pytest.raises(ValueError, match="no_such_dataset"):
        ff.fetch("no_such_dataset", _policy())
    assert issubclass(ff.UnknownDatasetError, ValueError)


# --------------------------------------------------------------------------- #
# L10 — dataset kinds: closed, provenance-carried, load-bearing
# --------------------------------------------------------------------------- #


def test_catalog_is_frozen_with_closed_dataset_kinds() -> None:
    """The frozen catalog names exactly the supported research files."""
    assert set(ff.FF_DATASETS) == {
        FACTORS_CSV,
        FACTORS_TXT,
        FACTORS_DAILY,
        INDUSTRY49,
    }
    for entry in ff.FF_DATASETS.values():
        assert entry.dataset_kind in ff.DATASET_KINDS
        assert entry.url.startswith("https://")
        assert entry.units == "percent"
        assert entry.frequency in {"monthly", "daily"}
        assert "Fama" in entry.licence_note
        assert "not investable" in entry.licence_note
    assert ff.FF_DATASETS[FACTORS_CSV].dataset_kind == "factor"
    assert ff.FF_DATASETS[FACTORS_TXT].dataset_kind == "factor"
    assert ff.FF_DATASETS[FACTORS_DAILY].dataset_kind == "factor"
    assert ff.FF_DATASETS[INDUSTRY49].dataset_kind == "portfolio_returns"
    # the known methodology-break regime fact is recorded, not adjusted
    assert "2025" in ff.FF_DATASETS[FACTORS_CSV].notes


def test_dataset_kind_round_trips_into_decode_provenance() -> None:
    """The catalog kind lands in decode provenance for both kinds."""
    _, factor_provenance = _decode(FACTORS_CSV)
    assert factor_provenance.dataset_kind == "factor"
    _, portfolio_provenance = _decode(INDUSTRY49)
    assert portfolio_provenance.dataset_kind == "portfolio_returns"
    assert factor_provenance.dataset_id == FACTORS_CSV


def test_dataset_kind_closed_set_rejects_other_values() -> None:
    """Provenance construction rejects any kind outside the closed set."""

    def _build(kind: str) -> None:
        ff.FFProvenance(
            source_id="ff:test",
            retrieval_instant=datetime.now(UTC),
            content_sha256="0" * 64,
            licence_note="test licence note",
            availability=_policy(),
            adapter_identity=ff.ADAPTER_IDENTITY,
            adapter_version=distribution_version("portlearn"),
            row_count=0,
            header_order=[],
            units="percent",
            frequency="monthly",
            dataset_id=FACTORS_CSV,
            dataset_kind=kind,
            url=ff.FF_DATASETS[FACTORS_CSV].url,
            last_modified=None,
            copyright_line=None,
        )

    _build("factor")
    _build("portfolio_returns")
    with pytest.raises(ValueError, match="dataset_kind"):
        _build("benchmark")


def test_factor_kind_rejected_from_asset_return_path() -> None:
    """A factor-kind dataset can never enter the asset-return path."""
    _, factor_provenance = _decode(FACTORS_CSV)
    with pytest.raises(ValueError, match="factor"):
        ff.require_candidate_asset_returns(factor_provenance)

    _, portfolio_provenance = _decode(INDUSTRY49)
    assert ff.require_candidate_asset_returns(portfolio_provenance) is None


# --------------------------------------------------------------------------- #
# L6 — availability is declared, never inferred
# --------------------------------------------------------------------------- #


def test_declared_fixed_lag_stamps_records_and_provenance() -> None:
    """A declared FIXED_LAG stamps every record's availability."""
    policy = _policy(timedelta(days=45))
    records, provenance = _decode(availability=policy)
    for record in records:
        assert record.available_time == record.observation_time + timedelta(
            days=45
        )
    assert provenance.availability is policy  # interned identity, carried
    assert provenance.availability.delta == timedelta(days=45)


def test_absent_availability_policy_fails_closed_with_no_default() -> None:
    """No policy -> fail-closed. The adapter never supplies a default."""
    with pytest.raises(ValueError, match="availability is mandatory"):
        ff.decode(
            _fixture_bytes(FACTORS_CSV),
            FACTORS_CSV,
            None,  # type: ignore[arg-type]
            tz=TZ,
        )
    with pytest.raises(ValueError, match="no default"):
        ff.fetch(FACTORS_CSV, None)  # type: ignore[arg-type]


def test_same_instant_policy_is_caller_declared_not_adapter_default() -> None:
    """SAME_INSTANT is admitted only when the caller declares it."""
    policy = AvailabilityPolicy.same_instant()
    records, provenance = _decode(availability=policy)
    for record in records:
        assert record.available_time == record.observation_time
    assert provenance.availability.mode == "SAME_INSTANT"


# --------------------------------------------------------------------------- #
# L9 — fixtures are hash-pinned and the manifest agrees
# --------------------------------------------------------------------------- #


def test_fixture_hashes_are_pinned_and_manifest_agrees() -> None:
    """Every fixture byte stream matches its pinned sha256 and manifest."""
    manifest = (FIXTURES / "MANIFEST.md").read_text(encoding="utf-8")
    assert "synthetic" in manifest.lower()
    assert "no provider-owned data" in manifest.lower()
    for name, pinned in PINNED_ZIP_SHA256.items():
        data = (FIXTURES / name).read_bytes()
        assert _sha256(data) == pinned, f"fixture {name} drifted"
        assert pinned in manifest
        assert name in manifest
    for path in FIXTURES.glob("*.zip"):
        assert path.name in PINNED_ZIP_SHA256


# --------------------------------------------------------------------------- #
# Naive-free + L8 not-investable + L7 series naming
# --------------------------------------------------------------------------- #


def test_every_instant_is_aware() -> None:
    """All produced instants are aware by construction."""
    records, provenance = _decode()
    for record in records:
        assert record.observation_time.utcoffset() is not None
        assert record.available_time.utcoffset() is not None
    assert provenance.retrieval_instant.utcoffset() is not None


def test_not_investable_warning_in_docstring_and_provenance() -> None:
    """The not-investable fence is carried in docs and provenance."""
    assert "not investable" in ff.__doc__.lower()
    _, provenance = _decode()
    assert "not investable" in provenance.not_investable_warning.lower()
    assert "not investable" in provenance.licence_note.lower()
    assert "Fama" in provenance.licence_note


def test_series_identifiers_follow_documented_convention() -> None:
    """Series ids are the documented FF/<column> convention — opaque ids."""
    records, _ = _decode(FACTORS_CSV)
    assert {record.series_id for record in records} == {
        "FF/Mkt-RF",
        "FF/SMB",
        "FF/HML",
        "FF/RF",
    }
    industry_records, _ = _decode(INDUSTRY49)
    ids = {record.series_id for record in industry_records}
    assert len(ids) == 49  # fixed-width header padding is stripped
    assert "FF/Agric" in ids
    assert "FF/Other" in ids


def test_txt_and_csv_variants_decode_to_identical_records() -> None:
    """The two provider format variants reduce to identical observations."""
    csv_records, _ = _decode(FACTORS_CSV)
    txt_records, _ = _decode(FACTORS_TXT)
    assert csv_records == txt_records


def test_header_order_row_count_and_units_frequency_in_provenance() -> None:
    """Provenance carries header order, row count, units, frequency."""
    records, provenance = _decode(FACTORS_CSV)
    assert provenance.header_order == ["Mkt-RF", "SMB", "HML", "RF"]
    assert provenance.row_count == len(records) == 141
    assert provenance.units == "percent"
    assert provenance.frequency == "monthly"
    assert provenance.adapter_identity == "portlearn.data.adapters.ff"
    assert provenance.adapter_version == distribution_version("portlearn")


# --------------------------------------------------------------------------- #
# Structural (AST) laws — split, delegation, no local restatement
# --------------------------------------------------------------------------- #

_SOURCE = Path(ff.__file__).read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)


def _function(name: str) -> ast.FunctionDef:
    for node in _TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found at module level")


def test_fetch_never_parses_and_decode_never_fetches() -> None:
    """The fetch/decode split holds at source level."""
    decode_body = ast.dump(_function("decode"))
    fetch_body = ast.dump(_function("fetch"))
    assert "urlopen" not in decode_body
    for parser_marker in ("from_records", "TimedObservation", "_parse"):
        assert parser_marker not in fetch_body


def test_adapter_never_constructs_availability_policies() -> None:
    """The adapter receives declarations; it never builds or defaults one."""
    assert "AvailabilityPolicy(" not in _SOURCE
    assert ".same_instant(" not in _SOURCE
    assert ".fixed_lag(" not in _SOURCE


def test_period_mapping_imported_never_restated() -> None:
    """The monthly mapping imports the shared substrate; none restated."""
    imports = {
        node.name
        for node in ast.walk(_TREE)
        if isinstance(node, ast.ImportFrom)
        for node in node.names
    }
    assert "month_end_instant" in imports
    assert "month_end_instant(" in _SOURCE
    assert "monthrange" not in _SOURCE  # no parallel calendar arithmetic


def test_record_construction_delegates_to_ingestion_chokepoint() -> None:
    """Records are built by the ingestion chokepoint, never locally."""
    assert "TimedObservation(" not in _SOURCE
    assert "AmbiguousObservationError" not in _SOURCE
    assert "from_records(" in _SOURCE


def test_adapters_package_init_is_docstring_only() -> None:
    """The adapters package identity module has no eager imports."""
    init_path = Path(ff.__file__).resolve().parent / "__init__.py"
    init_tree = ast.parse(init_path.read_text(encoding="utf-8"))
    for node in init_tree.body:
        assert not isinstance(node, (ast.Import, ast.ImportFrom))


def test_zip_container_layout_matches_catalog_members() -> None:
    """Each fixture ZIP holds exactly the catalog's member file name."""
    for dataset_id, fixture_name in FIXTURE_FOR_DATASET.items():
        with zipfile.ZipFile(FIXTURES / fixture_name) as archive:
            assert archive.namelist() == [
                ff.FF_DATASETS[dataset_id].member_name
            ]
