"""Contract tests for the FRED retrieval-only sibling surfaces.

The surfaces under test are the additive retrieval-only entry points of the
relocated FRED provider module (``portlearn.data.adapters.fred``):
``fetch_raw``, ``decode_unqualified``, and the
``FREDRetrievalProvenance`` typed provenance with CURRENT_SNAPSHOT
credential sanitization.  All tests are fixture-backed and network-free:
fetching is exercised through a monkeypatched ``urlopen`` over committed
synthetic replicas, and every decode runs on locally-held fixture bytes.
No provider-owned data is redistributed; the fake key material below is
invented for the sanitization checks alone.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Self

import pytest

from portlearn.data.ingestion import SourceProvenance

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "fred"

OBS_MONTHLY = "obs_synthcpim_monthly.json"
META_MONTHLY = "meta_synthcpim.json"
SERIES = "SYNTHCPIM"

#: An invented key: exists only to prove the sanitization law, never read
#: from any environment or file.
EXPLICIT_KEY = "SYNTHETIC_INVENDED_KEY_0000000000000000"

#: L3 field set for FRED retrieval provenance — retrieval facts only, no
#: availability field, and the realtime window plus sanitized request
#: parameters.
FRED_RETRIEVAL_FIELDS = frozenset(
    {
        "source_id",
        "retrieval_instant",
        "content_sha256",
        "licence_note",
        "adapter_identity",
        "adapter_version",
        "units",
        "frequency",
        "url",
        "last_modified",
        "attribution",
        "data_mode",
        "request_params",
        "realtime_start",
        "realtime_end",
        "series_id",
    }
)


def _fred() -> Any:
    """Import the relocated FRED module under portlearn.data.adapters."""
    from portlearn.data.adapters import fred

    return fred


def _obs_bytes() -> bytes:
    return (FIXTURES / OBS_MONTHLY).read_bytes()


def _meta_bytes() -> bytes:
    return (FIXTURES / META_MONTHLY).read_bytes()


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.headers = {"Last-Modified": None}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def test_fetch_raw_sends_explicit_key_and_sanitizes_all_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``fetch_raw`` demands an explicit key, sends the realtime window,
    and returns sanitized retrieval-only provenance (L1, L3)."""
    fred = _fred()
    captured: dict[str, Any] = {}

    def _capture(url: str, *args: object, **kwargs: object) -> _FakeResponse:
        captured["url"] = url
        return _FakeResponse(_obs_bytes())

    monkeypatch.setattr(fred, "urlopen", _capture)

    data, provenance = fred.fetch_raw(SERIES, EXPLICIT_KEY)

    assert data == _obs_bytes()
    assert isinstance(provenance, fred.FREDRetrievalProvenance)
    # The key traveled to the provider inside the request URL only.
    assert EXPLICIT_KEY in captured["url"]
    # ... and never into any provenance fact.
    dumped = repr(provenance)
    assert EXPLICIT_KEY not in dumped
    assert "api_key" not in dumped
    assert provenance.data_mode == "CURRENT_SNAPSHOT"
    assert provenance.realtime_start <= provenance.realtime_end
    assert not hasattr(provenance, "availability")


def test_fetch_raw_rejects_missing_key_before_any_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing or empty key rejects unconditionally before any socket."""
    fred = _fred()

    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("no network may be touched on rejection")

    monkeypatch.setattr(fred, "urlopen", _explode)
    with pytest.raises(ValueError, match="api_key"):
        fred.fetch_raw(SERIES, "")
    with pytest.raises(ValueError, match="api_key"):
        fred.fetch_raw(SERIES, None)  # type: ignore[arg-type]


def test_fetch_raw_never_reads_credentials_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider surface never sources a key from the environment:
    the API key must be supplied explicitly and is never sourced
    from environment variables or ``.env`` files — removing every
    conventional variable still demands the explicit argument."""
    fred = _fred()
    for name in (
        "FRED_API_KEY",
        "FRED_API_KEY_FILE",
        "PORTLEARN_FRED_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("no network may be touched on rejection")

    monkeypatch.setattr(fred, "urlopen", _explode)
    with pytest.raises(TypeError):
        fred.fetch_raw(SERIES)  # type: ignore[call-arg]


def test_decode_unqualified_maps_periods_to_provider_labels() -> None:
    """``decode_unqualified`` reduces fixture bytes to period-key records
    carrying the provider's own date labels, never an instant (L2)."""
    fred = _fred()

    records, provenance = fred.decode_unqualified(
        _obs_bytes(), SERIES, metadata_bytes=_meta_bytes()
    )

    assert isinstance(records, list)
    assert records
    # 36 payload rows minus the 2 missing-marker rows.
    assert len(records) == 34
    for record in records:
        assert record.period_key
        assert not hasattr(record, "observation_time")
        assert not hasattr(record, "available_time")
    # Provider labels are carried exactly: YYYY-MM monthly starts.
    labels = {record.period_key for record in records}
    assert "2023-01" in labels
    # The missing marker "." produced no record.
    assert not any(record.period_key == "2024-03" for record in records)
    assert provenance.data_mode == "CURRENT_SNAPSHOT"
    assert not hasattr(provenance, "availability")


def test_decode_unqualified_is_pure_over_identical_bytes() -> None:
    """Identical bytes decode to identical records and provenance (L2)."""
    fred = _fred()
    payload = _obs_bytes()

    first = fred.decode_unqualified(payload, SERIES)
    second = fred.decode_unqualified(payload, SERIES)

    assert first[0] == second[0]
    assert first[1] == second[1]


def test_decode_unqualified_hash_pin_rejects_mutated_bytes() -> None:
    """Supplying ``retrieval`` pins the decode to the exact bytes (L3)."""
    fred = _fred()
    payload = _obs_bytes()
    _, provenance = fred.decode_unqualified(payload, SERIES)
    mutated = payload[:-4] + b"0000"

    with pytest.raises(ValueError, match="hash pin mismatch"):
        fred.decode_unqualified(mutated, SERIES, retrieval=provenance)


def test_retrieval_provenance_field_set_is_pinned_and_request_params_frozen() -> None:
    """The FRED retrieval field set is exactly the fixed L3 list; the
    object is hashable with canonical immutable ``request_params`` (L3,
    D6)."""
    fred = _fred()

    class _Capture:
        def __init__(self) -> None:
            self.provenance: object = None

        def __call__(self, url: str, *args: object, **kwargs: object
                     ) -> _FakeResponse:
            return _FakeResponse(_obs_bytes())

    capture = _Capture()
    monkeypatch_proxy = pytest.MonkeyPatch()
    monkeypatch_proxy.setattr(fred, "urlopen", capture)
    try:
        _, provenance = fred.fetch_raw(SERIES, EXPLICIT_KEY)
    finally:
        monkeypatch_proxy.undo()

    assert dataclasses.is_dataclass(provenance)
    assert {f.name for f in dataclasses.fields(provenance)} == set(
        FRED_RETRIEVAL_FIELDS
    )
    assert hash(provenance) == hash(provenance)
    assert not isinstance(provenance, SourceProvenance)
    assert provenance.adapter_identity == "portlearn.data.adapters.fred"
    # Canonical immutable request parameters (D6).
    with pytest.raises(Exception):  # noqa: B017 — immutability is the law
        provenance.request_params["file_type"] = "alien"  # type: ignore[index]
    with pytest.raises(Exception):  # noqa: B017 — immutability is the law
        provenance.request_params = {}  # type: ignore[misc]


def test_current_snapshot_limitation_is_carried_not_invented() -> None:
    """CURRENT_SNAPSHOT is the only data mode this provider surface
    yields: the retrieval-only path never invents a vintage or a
    pseudo-lag."""
    fred = _fred()
    records, provenance = fred.decode_unqualified(_obs_bytes(), SERIES)
    assert provenance.data_mode == fred.CURRENT_SNAPSHOT
    assert not hasattr(provenance, "availability")
    assert all(not hasattr(r, "available_time") for r in records)
