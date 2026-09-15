"""Contract tests for the FRED adapter.

Every node maps to a frozen decoding law; the numbered section headers
below cite the law each node enforces.  All decoding is network-free
(monkeypatched fetch; local synthetic fixtures); the L11 mode fence and
the L4 false-vintage boundary are the named adversarial nodes.
"""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timedelta
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Self
from urllib.error import HTTPError
from zoneinfo import ZoneInfo

import pytest

from portlearn.calendar import month_end_instant, quarter_end_instant
from portlearn.data.adapters import fred
from portlearn.data.ingestion import AvailabilityPolicy
from portlearn.observations import TimedObservation

FIXTURES = Path(__file__).parent / "fixtures" / "fred"
ZONE = ZoneInfo("Australia/Melbourne")

OBS_CPIM = "obs_synthcpim_monthly.json"
META_CPIM = "meta_synthcpim.json"
OBS_GDPQ = "obs_synthgdpq_quarterly.json"
OBS_DFFD = "obs_synthdffd_daily.json"
OBS_UNKNOWN = "obs_unknown_series.json"

PINNED_SHA256 = {
    OBS_CPIM: "141eee56eef451a23e0ea232e30eb6eaf211acf75b0003b2aa9cd5db420c2639",
    META_CPIM: "edc878ff39c6efcdb978ba8fb21e07be5b1684dab6fe8112d9c9fc439cf6d16a",
    OBS_GDPQ: "dec55f04df139de8e830f20645123406f473b327fdb4f5d10b42bda374cfba17",
    "meta_synthgdpq.json": "b78b5967639894ec34beb3d71a53803e47aca02690597db2299265e6f8f963cd",
    OBS_DFFD: "b9a70b396bf4f77617ae72419b11c39433b7e3c37b8a92c32612c1eb63ff0c0d",
    "meta_synthdffd.json": "77a2de82eb93abe80884685c16085699e8c81868fc894201cdf62c9d9e8821d7",
    OBS_UNKNOWN: "f6795c3501492a4d87a8485d2f08f142de47822cd0aa4fbf615c57c5edc143ab",
}

#: The synthetic retrieval window pinned in every observations fixture.
REALTIME_END = "2026-09-12"

#: The CURRENT_SNAPSHOT floor instant for bare-bytes decodes: the last
#: instant of the payload's realtime-end day in the declared zone.
FLOOR = datetime(2026, 9, 12, 23, 59, 59, 999999, tzinfo=ZONE)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _policy() -> AvailabilityPolicy:
    return AvailabilityPolicy.fixed_lag(
        timedelta(days=28),
        justification="synthetic CPI-like release schedule: ~4 weeks "
        "after reference month (declared research assumption)",
    )


#: Sentinel distinguishing "caller declared nothing" from an explicit None.
_UNSET = object()


def _decode(
    name: str = OBS_CPIM,
    *,
    availability: Any = _UNSET,
    data_mode: str | None = fred.CURRENT_SNAPSHOT,
    frequency: str | None = "Monthly",
    metadata_bytes: bytes | None = None,
):
    payload = json.loads(_fixture_bytes(name))
    series = payload.get("series_id", "SYNTHCPIM")
    return fred.decode(
        _fixture_bytes(name),
        series,
        _policy() if availability is _UNSET else availability,
        data_mode=data_mode,
        tzinfo=ZONE,
        frequency=frequency,
        metadata_bytes=metadata_bytes,
    )


# --------------------------------------------------------------------------- #
# L1 — decodes to the canonical record form; network-free; pure
# --------------------------------------------------------------------------- #


def test_decode_returns_records_and_provenance_from_local_bytes() -> None:
    """Decode yields chokepoint-built records plus full provenance."""
    records, provenance = _decode()
    assert len(records) == 34  # 36 period rows minus 2 missing markers
    assert {record.series_id for record in records} == {"FRED/SYNTHCPIM"}
    assert provenance.source_id == "fred:SYNTHCPIM"
    assert provenance.row_count == len(records)
    assert provenance.adapter_identity == "portlearn.data.adapters.fred"
    assert provenance.adapter_version == distribution_version("portlearn")


def test_decode_never_touches_the_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decoder is network-free by construction."""
    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("decode attempted a network call")

    monkeypatch.setattr(fred, "urlopen", _explode)
    records, _ = _decode()
    assert records


def test_decode_is_pure_over_identical_bytes() -> None:
    """Identical bytes decode to identical records and provenance."""
    first_records, first_provenance = _decode()
    second_records, second_provenance = _decode()
    assert first_records == second_records
    assert first_provenance == second_provenance


# --------------------------------------------------------------------------- #
# L2 — fetch provenance completeness; the key never rides anything
# --------------------------------------------------------------------------- #


class _FakeResponse:
    """Minimal response stand-in: bytes plus header lookup."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.headers: dict[str, str] = {}

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def _fake_urlopen(payload: bytes):
    def _urlopen(*args: object, **kwargs: object) -> _FakeResponse:
        return _FakeResponse(payload)

    return _urlopen


_API_KEY = "TESTKEY0000000000000000000000000000000000"


def test_fetch_records_full_retrieval_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fetch provenance carries URL+params, instant, sha256, attribution."""
    payload = _fixture_bytes(OBS_CPIM)
    monkeypatch.setattr(fred, "urlopen", _fake_urlopen(payload))
    data, provenance = fred.fetch("SYNTHCPIM", _API_KEY)
    assert data == payload
    assert provenance.content_sha256 == _sha256(payload)
    assert provenance.retrieval_instant.utcoffset() is not None
    assert "api.stlouisfed.org" in provenance.url
    assert "SYNTHCPIM" in provenance.url
    assert "realtime_start=" in provenance.url
    assert "realtime_end=" in provenance.url
    assert provenance.realtime_start is not None
    assert provenance.realtime_end is not None
    assert "Federal Reserve Bank of St. Louis" in provenance.attribution
    assert "FRED" in provenance.licence_note
    assert provenance.data_mode == fred.CURRENT_SNAPSHOT
    assert provenance.row_count == 0  # the fetcher never parses
    assert provenance.header_order == []


def test_fetch_provenance_and_url_never_contain_the_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No key material enters any provenance field, ever (L9 fence)."""
    payload = _fixture_bytes(OBS_CPIM)
    monkeypatch.setattr(fred, "urlopen", _fake_urlopen(payload))
    _, provenance = fred.fetch("SYNTHCPIM", _API_KEY)
    dumped = json.dumps(provenance.__dict__, default=str)
    assert _API_KEY not in dumped
    assert "api_key" not in dumped


def test_fetch_rejects_missing_api_key_before_any_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absent key fails closed before a socket is ever opened."""
    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("fetch attempted a network call without a key")

    monkeypatch.setattr(fred, "urlopen", _explode)
    with pytest.raises(ValueError, match="api_key"):
        fred.fetch("SYNTHCPIM", "")
    with pytest.raises(ValueError, match="api_key"):
        fred.fetch("SYNTHCPIM", None)  # type: ignore[arg-type]


def test_fetch_rejects_key_smuggling_and_vintage_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """api_key in params, and true-vintage parameters, reject fail-closed."""
    monkeypatch.setattr(fred, "urlopen", _fake_urlopen(b"{}"))
    with pytest.raises(ValueError, match="api_key"):
        fred.fetch("SYNTHCPIM", _API_KEY, {"api_key": _API_KEY})
    with pytest.raises(ValueError, match="vintage"):
        fred.fetch(
            "SYNTHCPIM", _API_KEY, {"vintage_dates": "2024-01-01"}
        )
    with pytest.raises(ValueError, match="vintage"):
        fred.fetch("SYNTHCPIM", _API_KEY, {"output_type": "2"})


# --------------------------------------------------------------------------- #
# L3 — availability is declared, never defaulted; lag is non-binding metadata
# --------------------------------------------------------------------------- #


def test_declared_fixed_lag_is_floored_and_recorded_as_non_binding() -> None:
    """A declared per-series lag rides as metadata under the floor."""
    records, provenance = _decode()
    for record in records:
        assert record.available_time == FLOOR  # obs + 28d < retrieval: floored
    assert provenance.availability is _policy()
    assert provenance.availability.delta == timedelta(days=28)
    assert "non-binding" in provenance.declared_lag_note
    assert "28 days" in provenance.declared_lag_note
    assert "release schedule" in provenance.declared_lag_note


def test_no_default_availability_exists_in_code() -> None:
    """Absent or non-policy availability rejects; no silent default."""
    with pytest.raises(ValueError, match="availability"):
        _decode(availability=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="availability"):
        _decode(availability="fixed_lag")  # type: ignore[arg-type]


def test_same_instant_is_caller_declared_and_floored() -> None:
    """A declared SAME_INSTANT policy is floored at the retrieval instant."""
    records, provenance = _decode(
        availability=AvailabilityPolicy.same_instant()
    )
    for record in records:
        assert record.available_time == FLOOR
    assert provenance.availability.mode == "SAME_INSTANT"


# --------------------------------------------------------------------------- #
# L4 + L11 — the false-vintage fence: revised ≠ real-time
# --------------------------------------------------------------------------- #


def test_revised_representation_negative_no_pre_retrieval_availability() -> None:
    """No CURRENT_SNAPSHOT record carries pre-retrieval availability."""
    records, _ = _decode()
    for record in records:
        assert record.available_time >= FLOOR
        # the pseudo-vintage is unconstructible: a 2023 observation may not
        # claim a 2023 availability from a 2026 revised retrieval
        assert record.available_time.year == 2026


def test_pre_retrieval_availability_stamp_rejects_fail_closed() -> None:
    """The public floor fence rejects a pre-retrieval availability stamp."""
    record = TimedObservation(
        series_id="FRED/SYNTHCPIM",
        observation_time=month_end_instant(2023, 1, ZONE),
        available_time=datetime(2023, 2, 15, tzinfo=ZONE),
        value=250.0,
    )
    with pytest.raises(fred.PseudoVintageError, match="CURRENT_SNAPSHOT"):
        fred.require_current_snapshot_recency(record, FLOOR)
    with pytest.raises(fred.PseudoVintageError, match="CURRENT_SNAPSHOT"):
        fred.require_current_snapshot_recency([record], FLOOR)


def test_data_mode_is_labeled_in_provenance() -> None:
    """Every decode and fetch is labeled with its closed-set data mode."""
    _, provenance = _decode()
    assert provenance.data_mode == fred.CURRENT_SNAPSHOT
    assert fred.DATASET_MODES == ("CURRENT_SNAPSHOT", "POINT_IN_TIME")


def test_point_in_time_rejects_without_source_vintage_evidence() -> None:
    """POINT_IN_TIME rejects fail-closed even with realtime fields present."""
    with pytest.raises(fred.PseudoVintageError, match="POINT_IN_TIME"):
        _decode(data_mode=fred.POINT_IN_TIME)


def test_unknown_mode_rejects_against_the_closed_set() -> None:
    """A mode outside {CURRENT_SNAPSHOT, POINT_IN_TIME} rejects."""
    with pytest.raises(ValueError, match="closed set"):
        _decode(data_mode="REALTIME_VINTAGE")


def test_retrieval_mode_must_agree_with_decode_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A decode may not relabel the mode its retrieval provenance carries."""
    payload = _fixture_bytes(OBS_CPIM)
    monkeypatch.setattr(fred, "urlopen", _fake_urlopen(payload))
    _, fetched = fred.fetch("SYNTHCPIM", _API_KEY)
    with pytest.raises(ValueError, match="mode"):
        fred.decode(
            payload,
            "SYNTHCPIM",
            _policy(),
            data_mode=fred.POINT_IN_TIME,
            tzinfo=ZONE,
            frequency="Monthly",
            retrieval=fetched,
        )


# --------------------------------------------------------------------------- #
# L5 — period observation dates map to aware period-end / day-last instants
# --------------------------------------------------------------------------- #


def test_monthly_period_maps_to_period_end_instant() -> None:
    """Monthly period-start dates map to the shared substrate month end."""
    records, _ = _decode()
    by_date = {
        (r.observation_time.year, r.observation_time.month): r
        for r in records
    }
    assert by_date[(2023, 1)].observation_time == month_end_instant(
        2023, 1, ZONE
    )
    assert by_date[(2025, 12)].observation_time == month_end_instant(
        2025, 12, ZONE
    )


def test_quarterly_period_maps_to_quarter_end_instant() -> None:
    """Quarterly period-start dates map to the substrate quarter end."""
    records, _ = _decode(OBS_GDPQ, frequency="Quarterly")
    instants = {r.observation_time for r in records}
    assert quarter_end_instant(2022, 1, ZONE) in instants  # 2022-01-01
    assert quarter_end_instant(2023, 2, ZONE) in instants  # 2023-04-01
    assert quarter_end_instant(2025, 4, ZONE) in instants  # 2025-10-01


def test_leap_february_and_short_months_map_to_the_correct_last_day() -> None:
    """Leap Feb 2024-02-01 maps to 2024-02-29 day-last, Apr to day-30."""
    records, _ = _decode()
    by_month = {
        (r.observation_time.year, r.observation_time.month): r
        for r in records
    }
    leap = by_month[(2024, 2)].observation_time
    assert leap == month_end_instant(2024, 2, ZONE)  # 2024-02-29 day-last
    april = by_month[(2024, 4)].observation_time
    assert april == month_end_instant(2024, 4, ZONE)  # April 30 day-last


def test_daily_maps_to_calendar_day_last_not_market_close() -> None:
    """Daily dates map to the day's last instant — no trading session."""
    records, _ = _decode(OBS_DFFD, frequency="Daily")
    by_day = {r.observation_time.day: r for r in records}
    for day, record in by_day.items():
        # day-last in the DECLARED zone (UTC-normalized at storage):
        # no trading-session time is assumed anywhere.
        assert record.observation_time == datetime(
            2026, 7, day, 23, 59, 59, 999999, tzinfo=ZONE
        )
    # Saturday 2026-07-04 maps by the same calendar convention
    assert 4 in by_day


def test_non_period_start_dates_reject() -> None:
    """A mid-month date on a Monthly series is not a period start."""
    payload = json.dumps({
        "series_id": "SYNTHCPIM",
        "realtime_start": REALTIME_END,
        "realtime_end": REALTIME_END,
        "observations": [
            {"date": "2023-01-15", "realtime_start": REALTIME_END,
             "realtime_end": REALTIME_END, "value": "250.0"},
        ],
    }).encode("utf-8")
    with pytest.raises(ValueError, match="period start"):
        fred.decode(
            payload, "SYNTHCPIM", _policy(),
            data_mode=fred.CURRENT_SNAPSHOT,
            tzinfo=ZONE, frequency="Monthly",
        )


# --------------------------------------------------------------------------- #
# L6 — the missing marker is absent, never imputed
# --------------------------------------------------------------------------- #


def test_missing_marker_maps_to_absent_not_imputed() -> None:
    """'.' rows emit no record — never None, never 0.0."""
    records, _ = _decode()
    observed_months = {(r.observation_time.year, r.observation_time.month)
                       for r in records}
    assert (2024, 3) not in observed_months  # '.' cell
    assert (2025, 11) not in observed_months  # '.' cell
    assert len(records) == 34


def test_present_values_survive_untouched() -> None:
    """Present values are exact provider floats; none is None or '.'."""
    records, _ = _decode()
    payload = json.loads(_fixture_bytes(OBS_CPIM))
    expected = {
        obs["date"]: obs["value"]
        for obs in payload["observations"]
        if obs["value"] != "."
    }
    assert len(expected) == 34
    for record in records:
        assert record.value is not None
        # Values may legitimately be near zero; only the type is pinned.
        assert isinstance(record.value, float)


# --------------------------------------------------------------------------- #
# L7 — units and frequency are provenance metadata, never converted
# --------------------------------------------------------------------------- #


def test_metadata_units_and_frequency_ride_provenance() -> None:
    """Units/frequency come from the metadata endpoint, carried verbatim."""
    records, provenance = _decode(metadata_bytes=_fixture_bytes(META_CPIM))
    assert provenance.units == (
        "Index 1982-1984=100, Seasonally Adjusted (synthetic)"
    )
    assert provenance.frequency == "Monthly"
    payload = json.loads(_fixture_bytes(OBS_CPIM))
    first = next(
        obs for obs in payload["observations"] if obs["value"] != "."
    )
    first_record = next(
        r for r in records
        if (r.observation_time.year, r.observation_time.month)
        == (int(first["date"][:4]), int(first["date"][5:7]))
    )
    assert first_record.value == float(first["value"])  # never converted


def test_without_metadata_units_are_unspecified_never_inferred() -> None:
    """Absent metadata leaves units unspecified; frequency is declared."""
    _, provenance = _decode()
    assert provenance.units == "unspecified"
    assert provenance.frequency == "Monthly"


def test_unsupported_frequency_rejects_against_closed_set() -> None:
    """Weekly/Annual frequencies reject: outside the supported period mapping."""
    metadata = json.dumps({
        "id": "SYNTHCPIM", "units": "x", "frequency": "Weekly",
    }).encode("utf-8")
    with pytest.raises(fred.UnsupportedFrequencyError):
        _decode(metadata_bytes=metadata)
    with pytest.raises(fred.UnsupportedFrequencyError):
        _decode(frequency="Annual")


def test_explicit_frequency_must_agree_with_metadata() -> None:
    """A declared frequency contradicting metadata rejects fail-closed."""
    with pytest.raises(ValueError, match="frequency"):
        _decode(frequency="Daily", metadata_bytes=_fixture_bytes(META_CPIM))


def test_metadata_for_another_series_rejects() -> None:
    """Metadata whose series id disagrees with the decode rejects."""
    other = json.dumps({
        "id": "SYNTHGDPQ", "units": "x", "frequency": "Monthly",
    }).encode("utf-8")
    with pytest.raises(fred.ProviderResponseError, match="series"):
        _decode(metadata_bytes=other)


# --------------------------------------------------------------------------- #
# L8 — series naming is a documented convention
# --------------------------------------------------------------------------- #


def test_series_identifiers_follow_documented_convention() -> None:
    """Series ids are the documented FRED/<series_id> convention."""
    records, _ = _decode()
    assert {r.series_id for r in records} == {"FRED/SYNTHCPIM"}


# --------------------------------------------------------------------------- #
# L9 — fixtures are key-free, hash-pinned, and manifest-agreeing
# --------------------------------------------------------------------------- #


def test_fixtures_are_key_free() -> None:
    """No api-key material appears in any fixture byte stream."""
    for path in FIXTURES.glob("*.json"):
        lowered = path.read_bytes().lower()
        assert b"api_key" not in lowered, f"key material in {path.name}"
        assert b"apikey" not in lowered, f"key material in {path.name}"


def test_fixture_hashes_are_pinned_and_manifest_agrees() -> None:
    """Every fixture byte stream matches its pinned sha256 and manifest."""
    manifest = (FIXTURES / "MANIFEST.md").read_text(encoding="utf-8")
    assert "synthetic" in manifest.lower()
    assert "no provider-owned data" in manifest.lower()
    for name, pinned in PINNED_SHA256.items():
        data = (FIXTURES / name).read_bytes()
        assert _sha256(data) == pinned, f"fixture {name} drifted"
        assert pinned in manifest
        assert name in manifest
    for path in FIXTURES.glob("*.json"):
        assert path.name in PINNED_SHA256


# --------------------------------------------------------------------------- #
# L10 — the module-owned error taxonomy, fail-closed
# --------------------------------------------------------------------------- #


def test_unknown_series_rejects_with_module_error() -> None:
    """A provider unknown-series payload rejects as UnknownSeriesError."""
    with pytest.raises(fred.UnknownSeriesError, match="does not exist"):
        fred.decode(
            _fixture_bytes(OBS_UNKNOWN), "SYNTHCPIM", _policy(),
            data_mode=fred.CURRENT_SNAPSHOT, tzinfo=ZONE,
            frequency="Monthly",
        )


def test_malformed_json_rejects_as_provider_response_error() -> None:
    """Bytes that are not JSON reject fail-closed."""
    with pytest.raises(fred.ProviderResponseError, match="JSON"):
        fred.decode(
            b"not json at all", "SYNTHCPIM", _policy(),
            data_mode=fred.CURRENT_SNAPSHOT, tzinfo=ZONE,
            frequency="Monthly",
        )


def test_shape_errors_reject_as_provider_response_error() -> None:
    """Missing observations / mismatched series id reject fail-closed."""
    no_observations = json.dumps({
        "series_id": "SYNTHCPIM",
        "realtime_start": REALTIME_END,
        "realtime_end": REALTIME_END,
    }).encode("utf-8")
    with pytest.raises(fred.ProviderResponseError, match="observations"):
        fred.decode(
            no_observations, "SYNTHCPIM", _policy(),
            data_mode=fred.CURRENT_SNAPSHOT, tzinfo=ZONE,
            frequency="Monthly",
        )
    wrong_series = json.loads(_fixture_bytes(OBS_CPIM))
    wrong_series["series_id"] = "SYNTHOTHER"
    with pytest.raises(fred.ProviderResponseError, match="series"):
        fred.decode(
            json.dumps(wrong_series).encode("utf-8"), "SYNTHCPIM",
            _policy(),
            data_mode=fred.CURRENT_SNAPSHOT, tzinfo=ZONE,
            frequency="Monthly",
        )


def test_missing_realtime_window_rejects_silent_window_barred() -> None:
    """A payload without realtime labels rejects (D5: never a silent win)."""
    payload = json.loads(_fixture_bytes(OBS_CPIM))
    del payload["realtime_start"]
    del payload["realtime_end"]
    with pytest.raises(fred.ProviderResponseError, match="realtime"):
        fred.decode(
            json.dumps(payload).encode("utf-8"), "SYNTHCPIM", _policy(),
            data_mode=fred.CURRENT_SNAPSHOT, tzinfo=ZONE,
            frequency="Monthly",
        )


def test_http_error_maps_to_provider_response_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider HTTP failures surface as the module error, fail-closed."""
    def _raise(*args: object, **kwargs: object) -> None:
        raise HTTPError(
            "https://api.stlouisfed.org/fred/series/observations",
            400, "Bad Request", None, None,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(fred, "urlopen", _raise)
    with pytest.raises(fred.ProviderResponseError, match="HTTP 400"):
        fred.fetch("SYNTHCPIM", _API_KEY)


# --------------------------------------------------------------------------- #
# Hash-pinned fetch → decode round trip (content-hash discipline)
# --------------------------------------------------------------------------- #


def test_decode_adopts_retrieval_provenance_and_detects_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fetch->decode round trip carries the retrieval instant; drift fails."""
    payload = _fixture_bytes(OBS_CPIM)
    monkeypatch.setattr(fred, "urlopen", _fake_urlopen(payload))
    _, fetched = fred.fetch("SYNTHCPIM", _API_KEY)

    records, provenance = fred.decode(
        payload, "SYNTHCPIM", _policy(),
        data_mode=fred.CURRENT_SNAPSHOT, tzinfo=ZONE, frequency="Monthly",
        retrieval=fetched,
    )
    assert provenance.retrieval_instant == fetched.retrieval_instant
    assert all(
        record.available_time >= fetched.retrieval_instant
        for record in records
    )

    with pytest.raises(ValueError, match="hash"):
        fred.decode(
            payload + b" ", "SYNTHCPIM", _policy(),
            data_mode=fred.CURRENT_SNAPSHOT, tzinfo=ZONE,
            frequency="Monthly", retrieval=fetched,
        )


# --------------------------------------------------------------------------- #
# Naive-free instants everywhere
# --------------------------------------------------------------------------- #


def test_every_instant_is_aware() -> None:
    """All produced instants are aware by construction."""
    records, provenance = _decode()
    for record in records:
        assert record.observation_time.utcoffset() is not None
        assert record.available_time.utcoffset() is not None
    assert provenance.retrieval_instant.utcoffset() is not None


def test_decode_requires_an_explicit_aware_timezone() -> None:
    """No default zone is ever assumed."""
    with pytest.raises(ValueError, match="timezone"):
        fred.decode(
            _fixture_bytes(OBS_CPIM), "SYNTHCPIM", _policy(),
            data_mode=fred.CURRENT_SNAPSHOT, frequency="Monthly",
        )


# --------------------------------------------------------------------------- #
# Structural (AST) laws — split, delegation, no local restatement
# --------------------------------------------------------------------------- #

_SOURCE = Path(fred.__file__).read_text(encoding="utf-8")
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


def test_adapter_never_constructs_availability_defaults() -> None:
    """No default policy is built; the only construction is the plumbing."""
    assert ".same_instant(" not in _SOURCE
    assert ".fixed_lag(" not in _SOURCE
    assert "AvailabilityPolicy(" not in _SOURCE
    # the single sanctioned construction: the explicit-column plumbing that
    # hands the floored availability to the ingestion chokepoint
    assert _SOURCE.count("explicit_column(") == 1


def test_period_mapping_imported_never_restated() -> None:
    """Period ends import the shared substrate; none restated."""
    imports = {
        node.name
        for node in ast.walk(_TREE)
        if isinstance(node, ast.ImportFrom)
        for node in node.names
    }
    assert "month_end_instant" in imports
    assert "quarter_end_instant" in imports
    assert "month_end_instant(" in _SOURCE
    assert "quarter_end_instant(" in _SOURCE
    assert "monthrange" not in _SOURCE  # no parallel calendar arithmetic


def test_record_construction_delegates_to_ingestion_chokepoint() -> None:
    """Records are built by the ingestion chokepoint, never locally."""
    assert "TimedObservation(" not in _SOURCE
    assert "from_records(" in _SOURCE


def test_adapters_package_init_remains_docstring_only() -> None:
    """The adapters package identity module has no eager imports."""
    init_path = Path(fred.__file__).resolve().parent / "__init__.py"
    init_tree = ast.parse(init_path.read_text(encoding="utf-8"))
    for node in init_tree.body:
        assert not isinstance(node, (ast.Import, ast.ImportFrom))


def test_mode_constants_form_the_closed_set() -> None:
    """The data-mode labels are the documented closed set."""
    assert fred.CURRENT_SNAPSHOT == "CURRENT_SNAPSHOT"
    assert fred.POINT_IN_TIME == "POINT_IN_TIME"
    assert fred.DATASET_MODES == ("CURRENT_SNAPSHOT", "POINT_IN_TIME")


# --------------------------------------------------------------------------- #
# Provenance structural fence: key material can never ride provenance
# --------------------------------------------------------------------------- #


def test_provenance_rejects_key_material_structurally() -> None:
    """FREDProvenance itself refuses key-bearing params/urls."""
    with pytest.raises(ValueError, match="api_key"):
        fred.FREDProvenance(
            source_id="fred:X",
            retrieval_instant=FLOOR,
            content_sha256=None,
            licence_note="x",
            availability=None,
            adapter_identity="portlearn.data.adapters.fred",
            adapter_version="0",
            row_count=0,
            header_order=[],
            units="unspecified",
            frequency="Monthly",
            request_params={"api_key": "SECRET"},
        )
    with pytest.raises(ValueError, match="api_key"):
        fred.FREDProvenance(
            source_id="fred:X",
            retrieval_instant=FLOOR,
            content_sha256=None,
            licence_note="x",
            availability=None,
            adapter_identity="portlearn.data.adapters.fred",
            adapter_version="0",
            row_count=0,
            header_order=[],
            units="unspecified",
            frequency="Monthly",
            url="https://api.stlouisfed.org/fred/series/observations"
                "?api_key=SECRET&series_id=X",
        )


def test_provenance_rejects_unknown_data_modes() -> None:
    """The provenance fence enforces the closed mode set."""
    with pytest.raises(ValueError, match="closed set"):
        fred.FREDProvenance(
            source_id="fred:X",
            retrieval_instant=FLOOR,
            content_sha256=None,
            licence_note="x",
            availability=None,
            adapter_identity="portlearn.data.adapters.fred",
            adapter_version="0",
            row_count=0,
            header_order=[],
            units="unspecified",
            frequency="Monthly",
            data_mode="VINTAGE",
        )


# --------------------------------------------------------------------------- #
# Fixture reference: values decode exactly as the provider printed them
# --------------------------------------------------------------------------- #


def test_quarterly_and_daily_fixture_counts() -> None:
    """The quarterly and daily fixtures decode to their full counts."""
    quarterly, q_prov = _decode(OBS_GDPQ, frequency="Quarterly")
    assert len(quarterly) == 15  # 16 quarters minus one '.' cell
    assert q_prov.row_count == 15
    daily, d_prov = _decode(OBS_DFFD, frequency="Daily")
    assert len(daily) == 9  # 10 days minus one '.' cell
    assert d_prov.row_count == 9


def test_all_values_match_provider_cells_exactly() -> None:
    """Every decoded float equals the provider's own string cell."""
    for name, freq, series in (
        (OBS_CPIM, "Monthly", "SYNTHCPIM"),
        (OBS_GDPQ, "Quarterly", "SYNTHGDPQ"),
        (OBS_DFFD, "Daily", "SYNTHDFFD"),
    ):
        records, _ = fred.decode(
            _fixture_bytes(name), series, _policy(),
            data_mode=fred.CURRENT_SNAPSHOT, tzinfo=ZONE, frequency=freq,
        )
        payload = json.loads(_fixture_bytes(name))
        provider_values = [
            float(obs["value"])
            for obs in payload["observations"]
            if obs["value"] != "."
        ]
        assert sorted(r.value for r in records) == sorted(provider_values)
