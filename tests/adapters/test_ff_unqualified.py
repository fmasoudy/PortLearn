"""Contract tests for the FF retrieval-only sibling surfaces.

The surfaces under test are the additive retrieval-only entry points of the
relocated FF provider module (``portlearn.data.adapters.ff``): ``fetch_raw``
and ``decode_unqualified``, the ``PeriodKeyObservation`` record, and the
``FFRetrievalProvenance`` typed provenance.  All tests are fixture-backed
and network-free: fetching is exercised through a monkeypatched
``urlopen`` over committed synthetic replicas, and every decode runs on
locally-held fixture bytes.  No provider-owned data is redistributed and
no key material exists on this provider's path.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any, Self

import pytest

from portlearn.data.ingestion import SourceProvenance
from portlearn.observations import TimedObservation

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ff"

INDUSTRY49_ZIP = "ff_industry49_monthly_csv.zip"
INDUSTRY49 = "industry49_monthly_csv"

#: Pinned retrieval-provenance field set (retrieval facts only — no availability).
FF_RETRIEVAL_FIELDS = frozenset(
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
        "copyright_line",
        "dataset_kind",
        "dataset_id",
        "not_investable_warning",
        "row_count",
        "header_order",
    }
)

#: Pinned fixture hash (pinned by the decoder tests; bytes are read, never
#: changed).
PINNED_INDUSTRY49_SHA256 = (
    "eb4831ad8e0f32d40ef8a6b1775b2e25a229d52b6d9d835559a515f0465cb9bf"
)


def _fixture_bytes() -> bytes:
    return (FIXTURES / INDUSTRY49_ZIP).read_bytes()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ff() -> Any:
    """Import the relocated FF module under portlearn.data.adapters."""
    from portlearn.data.adapters import ff

    return ff


class _FakeResponse:
    """A context-manager stand-in for a ``urlopen`` response object."""

    def __init__(self, payload: bytes, last_modified: str | None) -> None:
        self._payload = payload
        self.headers = {"Last-Modified": last_modified}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def _patch_network(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> Any:
    """Point the relocated module's ``urlopen`` at local fixture bytes."""
    ff = _ff()
    monkeypatch.setattr(
        ff, "urlopen", lambda *a, **k: _FakeResponse(payload, None)
    )
    return ff


def test_fetch_raw_returns_exact_bytes_and_retrieval_only_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``fetch_raw`` yields the provider's exact bytes plus retrieval-only
    typed provenance carrying no availability field (L1, L3)."""
    ff = _patch_network(monkeypatch, _fixture_bytes())

    data, provenance = ff.fetch_raw(INDUSTRY49)

    assert data == _fixture_bytes()
    assert isinstance(provenance, ff.FFRetrievalProvenance)
    assert provenance.content_sha256 == _sha256(data)
    assert provenance.source_id == f"ff:{INDUSTRY49}"
    assert provenance.dataset_id == INDUSTRY49
    assert provenance.dataset_kind == "portfolio_returns"
    assert provenance.units == "percent"
    assert provenance.frequency == "monthly"
    assert provenance.url.endswith("49_Industry_Portfolios_CSV.zip")
    # L3: retrieval-only provenance carries no availability field at all.
    assert not hasattr(provenance, "availability")


def test_fetch_raw_rejects_unknown_dataset_id_before_any_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown dataset id rejects before any socket is opened."""

    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("no network may be touched on rejection")

    ff = _ff()
    monkeypatch.setattr(ff, "urlopen", _explode)
    with pytest.raises(ValueError, match="no catalog entry"):
        ff.fetch_raw("not_a_catalog_dataset")


def test_decode_unqualified_yields_period_key_records_without_instants() -> None:
    """``decode_unqualified`` reduces fixture bytes to provider-labelled
    period-key records that define no instant attribute (L1, L2)."""
    ff = _ff()

    records, provenance = ff.decode_unqualified(_fixture_bytes(), INDUSTRY49)

    assert records
    assert isinstance(records, list)
    assert isinstance(provenance, ff.FFRetrievalProvenance)
    # L2: the exact provider temporal label, never normalized.
    assert records[0].period_key == "202301"
    series_ids = {record.series_id for record in records}
    assert len(series_ids) == 49
    assert all(record.series_id for record in records)
    # L2: no instant exists to fabricate — the attributes do not exist.
    for record in records:
        assert not hasattr(record, "observation_time")
        assert not hasattr(record, "available_time")
        assert not isinstance(record, TimedObservation)
    # The sibling parses the same rows as the qualified decoder.
    assert len(records) == 292
    agric = next(
        r for r in records if r.period_key == "202301" and "Agric" in r.series_id
    )
    assert agric.value == 9.66
    assert provenance.row_count == len(records)
    assert provenance.header_order[0] == "Agric"
    assert provenance.units == "percent"
    assert not hasattr(provenance, "availability")


def test_decode_unqualified_maps_sentinels_to_absence_and_is_deterministic() -> None:
    """Sentinel cells produce no record, exactly as the qualified decoder
    does, and identical bytes decode to identical records (L2)."""
    ff = _ff()
    payload = _fixture_bytes()

    first_records, first_prov = ff.decode_unqualified(payload, INDUSTRY49)
    second_records, second_prov = ff.decode_unqualified(payload, INDUSTRY49)

    assert first_records == second_records
    assert first_prov == second_prov
    # The 202302 Soda cell is a -99.99 sentinel in the fixture: absent.
    assert not any(
        r.period_key == "202302" and "Soda" in r.series_id
        for r in first_records
    )


def test_decode_unqualified_hash_pin_rejects_mutated_bytes_with_retrieval() -> None:
    """With ``retrieval`` supplied the sibling is hash-pinned: mutated
    bytes reject (L3 — the fixed-decoder pin, mirrored by the sibling)."""
    ff = _ff()
    payload = _fixture_bytes()
    _, provenance = ff.decode_unqualified(payload, INDUSTRY49)
    mutated = b"drifted" + payload

    with pytest.raises(ValueError, match="hash pin mismatch"):
        ff.decode_unqualified(mutated, INDUSTRY49, retrieval=provenance)


def test_period_key_observation_is_frozen_hashable_and_instant_free() -> None:
    """``PeriodKeyObservation`` is a fixed hashable value object whose
    only fields are the provider label triple (L2)."""
    from portlearn.data._records import PeriodKeyObservation

    record = PeriodKeyObservation(
        series_id="FF/Agric", period_key="202301", value=9.66
    )
    twin = PeriodKeyObservation(
        series_id="FF/Agric", period_key="202301", value=9.66
    )
    assert record == twin
    assert hash(record) == hash(twin)
    with pytest.raises(Exception):  # noqa: B017 — immutability is the law
        record.value = 1.0  # type: ignore[misc]
    # Blank identifiers reject at construction.
    with pytest.raises(ValueError):
        PeriodKeyObservation(series_id="  ", period_key="202301", value=1.0)
    with pytest.raises(ValueError):
        PeriodKeyObservation(series_id="FF/Agric", period_key="", value=1.0)
    # The closed ingestion value domain holds at the record boundary.
    with pytest.raises((TypeError, ValueError)):
        PeriodKeyObservation(
            series_id="FF/Agric", period_key="202301", value=object()
        )


def test_retrieval_provenance_field_set_is_pinned_and_adoptable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retrieval field set is exactly the fixed L3 list, the object is
    hashable, is not a ``SourceProvenance``, and the fixed qualified
    decoder adopts it as its ``retrieval=`` pin (L3)."""
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    from portlearn.data.ingestion import AvailabilityPolicy

    ff = _ff()
    payload = _fixture_bytes()
    _, provenance = ff.decode_unqualified(payload, INDUSTRY49)

    assert dataclasses.is_dataclass(provenance)
    assert {field.name for field in dataclasses.fields(prov)} == set(
        FF_RETRIEVAL_FIELDS
    ) if (prov := provenance) else False
    assert hash(provenance) == hash(provenance)
    assert not isinstance(provenance, SourceProvenance)
    assert provenance.adapter_identity == "portlearn.data.adapters.ff"
    assert provenance.content_sha256 == PINNED_INDUSTRY49_SHA256

    # The fixed qualified decoder adopts the retrieval-only provenance.
    policy = AvailabilityPolicy.fixed_lag(
        timedelta(days=31), justification="declared research assumption"
    )
    timed, qualified_provenance = ff.decode(
        payload,
        INDUSTRY49,
        policy,
        tzinfo=ZoneInfo("America/New_York"),
        retrieval=provenance,
    )
    assert len(timed) == 292
    assert qualified_provenance.content_sha256 == PINNED_INDUSTRY49_SHA256


def test_retrieval_provenance_base_lives_in_the_shared_leaf_module() -> None:
    """The ``RetrievalProvenance`` base is defined in the shared
    ``_unqualified`` leaf module beside ``PeriodKeyObservation`` (L2, L3)."""
    from portlearn.data._records import (
        PeriodKeyObservation,
        RetrievalProvenance,
    )
    from portlearn.data.adapters import ff

    assert issubclass(ff.FFRetrievalProvenance, RetrievalProvenance)
    assert ff.PeriodKeyObservation is PeriodKeyObservation
