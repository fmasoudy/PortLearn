"""Contract tests for the public data facade.

The surface under test is the researcher-facing data API: the lazy
``pl.data`` facade, the two provider facades, the ``ResearchDataset``
two-state value object, qualification, pandas conversion, the alias
catalog, the FRED CURRENT_SNAPSHOT carry, import-DAG laziness, the
pandas dependency/metadata contract, public-constructor sufficiency,
and the relocation identity of the adapter layer.

The tests are fixture-backed and network-free: every facade ``load`` is
exercised through caller-supplied fixture bytes or a monkeypatched
provider fetch over committed synthetic replicas.  Public surfaces
are imported inside the tests so pytest can report the causal absence
(``ModuleNotFoundError: portlearn.data``) instead of aborting
collection.  Research datasets only; NOT investable.

Test names follow the facade contract nodes verbatim; the packaging and
dependency contract lives in ``tests/test_package_contract.py``).

"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import runpy
import socket
import subprocess
import sys
import tomllib
from contextlib import redirect_stdout
from datetime import timedelta
from importlib import metadata
from pathlib import Path
from typing import Any, Self
from zoneinfo import ZoneInfo

import pytest

from portlearn.data.ingestion import AvailabilityPolicy
from portlearn.observations import TimedObservation

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "portlearn"
FIXTURES = REPOSITORY_ROOT / "tests" / "adapters" / "fixtures"
EXAMPLE_PATH = REPOSITORY_ROOT / "examples" / "information_set_smoke.py"
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"

INDUSTRY49_ZIP = FIXTURES / "ff" / "ff_industry49_monthly_csv.zip"
FACTORS_DAILY_ZIP = FIXTURES / "ff" / "ff_factors_daily_csv.zip"
FRED_MONTHLY_JSON = FIXTURES / "fred" / "obs_synthcpim_monthly.json"
FRED_META_JSON = FIXTURES / "fred" / "meta_synthcpim.json"

TZ = ZoneInfo("America/New_York")

PINNED_INDUSTRY49_SHA256 = (
    "eb4831ad8e0f32d40ef8a6b1775b2e25a229d52b6d9d835559a515f0465cb9bf"
)

FACADE_SOURCES = [
    PACKAGE_ROOT / "data" / "fama_french.py",
    PACKAGE_ROOT / "data" / "fred.py",
    PACKAGE_ROOT / "data" / "dataset.py",
    PACKAGE_ROOT / "data" / "_records.py",
    PACKAGE_ROOT / "data" / "ingestion.py",
]

ADAPTER_SOURCES = [
    PACKAGE_ROOT / "data" / "adapters" / "ff.py",
    PACKAGE_ROOT / "data" / "adapters" / "fred.py",
    PACKAGE_ROOT / "data" / "_records.py",
]

DATA_PACKAGE_MODULES = [
    "portlearn.data",
    "portlearn.data.fama_french",
    "portlearn.data.fred",
    "portlearn.data.dataset",
    "portlearn.data._records",
    "portlearn.data.ingestion",
    "portlearn.data.adapters",
    "portlearn.data.adapters.ff",
    "portlearn.data.adapters.fred",
]

#: The sole module where the pandas import may live.
PANDAS_HOME = "portlearn.data.dataset"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _policy() -> AvailabilityPolicy:
    return AvailabilityPolicy.fixed_lag(
        timedelta(days=31),
        justification=(
            "Ken French monthly files publish in the first days of the "
            "following month (declared research assumption)."
        ),
    )


def _industry49_bytes() -> bytes:
    return INDUSTRY49_ZIP.read_bytes()


def _fred_payload() -> bytes:
    return FRED_MONTHLY_JSON.read_bytes()


def _fred_metadata() -> bytes:
    return FRED_META_JSON.read_bytes()


def _ff_module() -> Any:
    from portlearn.data.adapters import ff

    return ff


def _fred_module() -> Any:
    from portlearn.data.adapters import fred

    return fred


class _FakeResponse:
    """A context-manager stand-in for a ``urlopen`` response object."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.headers: dict[str, str] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def _request_url(request: object) -> str:
    """Extract the target URL from a str or ``urllib.request.Request``."""
    if isinstance(request, str):
        return request
    getter = getattr(request, "get_full_url", None)
    return str(getter()) if callable(getter) else str(request)


def _patch_ff_network(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> Any:
    ff = _ff_module()
    monkeypatch.setattr(ff, "urlopen", lambda *a, **k: _FakeResponse(payload))
    return ff


def _patch_ff_network_routed(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Serve fixture bytes by catalog URL token (multi-artifact tests)."""

    routes = {
        "49_Industry_Portfolios_CSV.zip": _industry49_bytes(),
        "F-F_Research_Data_Factors_daily_CSV.zip": (FACTORS_DAILY_ZIP.read_bytes()),
    }

    def _routed(request: object, *args: object, **kwargs: object):
        url = _request_url(request)
        for token, payload in routes.items():
            if token in url:
                return _FakeResponse(payload)
        raise AssertionError(f"unexpected fetch url: {url!r}")

    ff = _ff_module()
    monkeypatch.setattr(ff, "urlopen", _routed)
    return ff


def _patch_fred_network(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Route FRED fetches at the synthetic observations/metadata bytes."""

    def _routed(request: object, *args: object, **kwargs: object):
        url = _request_url(request)
        if "series/observations" in url:
            return _FakeResponse(_fred_payload())
        return _FakeResponse(_fred_metadata())

    fred = _fred_module()
    monkeypatch.setattr(fred, "urlopen", _routed)
    return fred


def _load_industry49(**kwargs: Any) -> Any:
    import portlearn as pl

    return pl.data.fama_french.load("industry49", frequency="monthly", **kwargs)


def _qualified_industry49(monkeypatch: pytest.MonkeyPatch) -> Any:
    _patch_ff_network(monkeypatch, _industry49_bytes())
    return _load_industry49(availability=_policy(), tzinfo=TZ)


# --------------------------------------------------------------------------- #
# Law 1 — the UNQUALIFIED load carries no availability anywhere
# --------------------------------------------------------------------------- #


def test_unqualified_ff49_load_carries_no_availability_anywhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The beginner load: state UNQUALIFIED, availability None, and no
    record, provenance, or dataset field carries an availability field,
    policy, or fabricated instant (L1, L4, L5)."""
    _patch_ff_network(monkeypatch, _industry49_bytes())
    ds = _load_industry49()

    assert ds.availability_state == "UNQUALIFIED"
    assert ds.availability is None
    assert ds.provider == "fama_french"
    assert ds.name == "industry49"
    assert ds.provider_dataset_id == "industry49_monthly_csv"
    assert isinstance(ds.adapter_version, str)
    assert not hasattr(ds.retrieval_provenance, "availability")
    assert ds.qualified_provenance is None
    for record in ds.records:
        assert not hasattr(record, "available_time")
        assert not hasattr(record, "observation_time")
        assert not isinstance(record, TimedObservation)
    first = ds.records[0]
    assert first.period_key == "202301"
    assert "Agric" in first.series_id
    assert first.value == 9.66
    assert len(ds.records) == 292
    assert ds.units == "percent"
    assert ds.frequency == "monthly"
    assert ds.source_sha256 == PINNED_INDUSTRY49_SHA256
    assert ds.source_bytes == _industry49_bytes()
    assert ds.auxiliary_bytes is None


# --------------------------------------------------------------------------- #
# Floors 2-5 — every decision-time refusal surface
# --------------------------------------------------------------------------- #


def test_unqualified_dataset_cannot_enter_information_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Floor 2: the dataset-owned refusal raises UnqualifiedDataError (a
    TypeError) naming the qualify path; raw InformationSet construction
    over the same records raises the fixed MissingAvailabilityError
    admission law (L6)."""
    from datetime import UTC, datetime

    from portlearn.interfaces import InformationSet
    from portlearn.timing import MissingAvailabilityError

    _patch_ff_network(monkeypatch, _industry49_bytes())
    ds = _load_industry49()
    as_of = datetime(2026, 9, 12, tzinfo=UTC)

    with pytest.raises(Exception) as facade_refusal:
        ds.to_information_set(as_of=as_of)
    facade_error = facade_refusal.value
    assert type(facade_error).__name__ == "UnqualifiedDataError"
    assert isinstance(facade_error, TypeError)
    assert "qualify(" in str(facade_error)
    assert "AvailabilityPolicy" in str(facade_error)

    with pytest.raises(MissingAvailabilityError):
        InformationSet(ds.records, as_of=as_of)


def test_observation_store_refuses_unqualified_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Floor 3: the fixed isinstance fence rejects the period-key
    records at store construction (L6)."""
    from portlearn.alignment import ObservationStore

    _patch_ff_network(monkeypatch, _industry49_bytes())
    ds = _load_industry49()

    with pytest.raises(TypeError):
        ObservationStore(ds.records)


def test_transforms_refuse_unqualified_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Floor 4: every public transform rejects records with no
    available_time (availability keying is the law)."""
    from portlearn.transforms import (
        CarryForward,
        Lag,
        MinMaxScaler,
        RollingMean,
        RollingVolatility,
        StandardScaler,
    )

    _patch_ff_network(monkeypatch, _industry49_bytes())
    ds = _load_industry49()
    records = ds.records

    with pytest.raises(Exception):  # noqa: B017 — refusal is the law
        RollingMean(2).transform(records)
    with pytest.raises(Exception):  # noqa: B017 — refusal is the law
        RollingVolatility(2).transform(records)
    with pytest.raises(Exception):  # noqa: B017 — refusal is the law
        Lag(1, timedelta(days=1)).transform(records)
    with pytest.raises(Exception):  # noqa: B017 — refusal is the law
        StandardScaler(records)
    with pytest.raises(Exception):  # noqa: B017 — refusal is the law
        MinMaxScaler(records)
    with pytest.raises(Exception):  # noqa: B017 — refusal is the law
        CarryForward(
            timedelta(days=1), reference_instants=[], real_time_surface=True
        ).transform(records)


class _AsOfSet:
    """A minimal as_of-bearing fit set for the forecasting fence probe."""

    def __init__(self, as_of: Any) -> None:
        self.as_of = as_of


def test_no_unqualified_smuggling_into_decision_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Floor 5: the decision-time chain is fenced at every entry —
    forecasting fit admission, transforms, the store, information-set
    admission, and the dataset-owned surface (L6)."""
    from datetime import UTC, datetime

    from portlearn.alignment import ObservationStore
    from portlearn.forecasting import require_fit_inputs_admitted
    from portlearn.interfaces import InformationSet
    from portlearn.timing import MissingAvailabilityError

    _patch_ff_network(monkeypatch, _industry49_bytes())
    ds = _load_industry49()
    records = ds.records
    as_of = datetime(2026, 9, 12, tzinfo=UTC)

    with pytest.raises(Exception):  # noqa: B017 — refusal is the law
        require_fit_inputs_admitted(records, _AsOfSet(as_of))
    with pytest.raises(TypeError):
        ObservationStore(records)
    with pytest.raises(MissingAvailabilityError):
        InformationSet(records, as_of=as_of)
    with pytest.raises(Exception) as refusal:
        ds.to_information_set(as_of=as_of)
    assert "qualify(" in str(refusal.value)


# --------------------------------------------------------------------------- #
# Floors 6-8 — qualification, the evidence pair, equivalence
# --------------------------------------------------------------------------- #


def test_qualify_is_one_canonical_hash_pinned_redecode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Floor 6: the raw load plus qualify yields QUALIFIED records
    stamped under exactly the caller's policy over the retained bytes
    (L8, L9, L3)."""
    _patch_ff_network(monkeypatch, _industry49_bytes())
    ff = _ff_module()
    data, retrieval = ff.fetch_raw("industry49_monthly_csv")

    import portlearn as pl

    raw = pl.data.fama_french.load(
        "industry49", frequency="monthly", data=data, retrieval=retrieval
    )
    assert raw.availability_state == "UNQUALIFIED"

    policy = _policy()
    qualified = raw.qualify(availability=policy, tzinfo=TZ)

    assert qualified.availability_state == "QUALIFIED"
    assert qualified.availability is policy
    assert qualified.availability is not None
    for record in qualified.records:
        assert isinstance(record, TimedObservation)
        assert record.available_time >= record.observation_time
    assert qualified.qualified_provenance.content_sha256 == PINNED_INDUSTRY49_SHA256
    assert qualified.source_bytes == data == _industry49_bytes()
    assert qualified.source_sha256 == _sha256(qualified.source_bytes)
    assert qualified.qualified_provenance.url == retrieval.url
    assert qualified.qualified_provenance.retrieval_instant == (
        retrieval.retrieval_instant
    )
    assert qualified.qualified_provenance.last_modified == (retrieval.last_modified)

    # The receiver is unchanged: qualification returns a new dataset.
    assert raw.availability_state == "UNQUALIFIED"
    assert raw.qualified_provenance is None
    assert raw.records is not qualified.records

    # Mutated bytes reject through the fixed decoder's hash pin — the
    # re-decode can never adopt retrieval facts over different bytes.
    with pytest.raises(ValueError):
        ff.decode(
            data + b"\x00",
            "industry49_monthly_csv",
            policy,
            tzinfo=TZ,
            retrieval=retrieval,
        )
    with pytest.raises(Exception):  # noqa: B017 — immutability is the law
        object.__setattr__(qualified, "source_bytes", b"mutated")


def test_qualify_requires_the_full_evidence_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Floor 7: one-sided evidence pairs reject ValueError — availability
    without tzinfo, and tzinfo without availability (L5, L9)."""
    _patch_ff_network(monkeypatch, _industry49_bytes())

    with pytest.raises(ValueError):
        _load_industry49(availability=_policy())
    with pytest.raises(ValueError):
        _load_industry49(tzinfo=TZ)


def _assert_qualified_identical(first: Any, second: Any) -> None:
    assert first.availability_state == "QUALIFIED"
    assert second.availability_state == "QUALIFIED"
    assert first == second
    assert first.records == second.records
    assert first.source_bytes == second.source_bytes
    assert first.qualified_provenance == second.qualified_provenance


def test_one_step_and_two_step_qualification_are_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Floor 8a: one-step qualified load equals the raw-load-plus-qualify
    two-step path over the same bytes (L9)."""
    from datetime import UTC, datetime

    import portlearn as pl

    _patch_ff_network(monkeypatch, _industry49_bytes())
    # Fixed-clock pin: this test makes two live fetches, and every fetch
    # stamps its own wall-clock retrieval instant — two fetches, two
    # instants — so the field-for-field provenance equality below could
    # never hold deterministically. Pinning the adapter's clock seam to
    # one fixed aware instant shared by both retrievals restores the
    # intended condition ("no wall-clock field differs because retrieval
    # is shared"); no assertion is weakened.
    fixed_instant = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(_ff_module(), "_now_utc", lambda: fixed_instant)
    one_step = _load_industry49(availability=_policy(), tzinfo=TZ)

    ff = _ff_module()
    data, retrieval = ff.fetch_raw("industry49_monthly_csv")
    raw = pl.data.fama_french.load(
        "industry49", frequency="monthly", data=data, retrieval=retrieval
    )
    two_step = raw.qualify(availability=_policy(), tzinfo=TZ)

    _assert_qualified_identical(one_step, two_step)
    assert _sha256(bytes(two_step.source_bytes)) == (
        _sha256(bytes(one_step.source_bytes))
    )
    # L9: the facade fetch path begins at fetch_raw — the fixed qualified
    # entry points are the independent equivalence comparator, not the
    # facade path.
    assert one_step.provider_dataset_id == two_step.provider_dataset_id
    assert one_step.name == two_step.name
    assert one_step.adapter_version == two_step.adapter_version


def test_qualified_behavior_preserved_full_suite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Floor 8b: the facade's qualified output matches the fixed
    adapter-qualified path record-for-record over the same bytes (L9:
    the qualified decoders are the independent equivalence comparator)."""
    from datetime import timedelta as _td

    policy = AvailabilityPolicy.fixed_lag(
        _td(days=31), justification="declared research assumption"
    )
    ff = _ff_module()
    payload = _industry49_bytes()
    timed, provenance = ff.decode(
        payload,
        "industry49_monthly_csv",
        policy,
        tzinfo=TZ,
    )
    assert len(timed) == 292

    _patch_ff_network(monkeypatch, payload)
    ds = _load_industry49(availability=policy, tzinfo=TZ)

    assert list(ds.records) == timed
    assert ds.units == provenance.units
    assert ds.frequency == provenance.frequency
    assert ds.qualified_provenance.content_sha256 == (provenance.content_sha256)


# --------------------------------------------------------------------------- #
# Law 9 — forbidden mixed states unrepresentable
# --------------------------------------------------------------------------- #


def test_forbidden_mixed_states_unrepresentable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Floor 10: every L5 forbidden mixed state rejects at the public
    constructors (unconditional validation)."""
    import portlearn as pl

    _patch_ff_network(monkeypatch, _industry49_bytes())
    ds = _load_industry49()
    policy = _policy()
    record = ds.records[0]

    from portlearn.data._records import PeriodKeyObservation

    with pytest.raises(ValueError):
        PeriodKeyObservation(series_id="  ", period_key="202301", value=1.0)
    with pytest.raises(ValueError):
        PeriodKeyObservation(series_id="FF/Agric", period_key="", value=1.0)
    with pytest.raises((TypeError, ValueError)):
        PeriodKeyObservation(series_id="FF/Agric", period_key="202301", value=object())
    assert not hasattr(record, "observation_time")
    assert not hasattr(record, "available_time")

    ff = _ff_module()
    data, retrieval = ff.fetch_raw("industry49_monthly_csv")

    with pytest.raises(ValueError):
        pl.data.fama_french.load(
            "industry49",
            frequency="monthly",
            data=data,
            retrieval=retrieval,
            availability=policy,
        )
    with pytest.raises(ValueError):
        pl.data.fama_french.load(
            "industry49",
            frequency="monthly",
            data=data,
            retrieval=retrieval,
            tzinfo=TZ,
        )
    qualified = ds.qualify(availability=policy, tzinfo=TZ)
    with pytest.raises(Exception):  # noqa: B017 — one-way state is the law
        qualified.qualify(availability=policy, tzinfo=TZ)


# --------------------------------------------------------------------------- #
# Law 10 — to_pandas is normal core interoperability
# --------------------------------------------------------------------------- #


def test_to_pandas_is_normal_core_interoperability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Floor 11: both states convert with exactly the L5 pinned column
    sets; pandas is a normal core dependency here (not optional, not
    guarded) and lives at module scope in exactly one module (single-module law)."""
    import pandas as pd

    _patch_ff_network(monkeypatch, _industry49_bytes())
    ds = _load_industry49()

    frame = ds.to_pandas()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["series_id", "period_key", "value"]
    assert len(frame) == len(ds.records)
    assert frame.iloc[0]["series_id"] == ds.records[0].series_id
    assert frame.iloc[0]["period_key"] == ds.records[0].period_key
    assert frame.iloc[0]["value"] == ds.records[0].value

    qualified = _load_industry49(availability=_policy(), tzinfo=TZ)
    qframe = qualified.to_pandas()
    assert list(qframe.columns) == [
        "series_id",
        "observation_time",
        "value",
        "available_time",
    ]
    assert len(qframe) == len(qualified.records)

    # The pandas import lives at module scope in exactly one module and
    # nowhere else in the package (L11, D9).
    sources = sorted(PACKAGE_ROOT.rglob("*.py"))
    assert sources, "the package source tree unexpectedly declares no files"
    import_locations = [
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in sources
        if re.search(
            r"^\s*import pandas\b",
            path.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    ]
    assert import_locations == ["src/portlearn/data/dataset.py"], (
        "the module-scope pandas import must exist exactly once, in "
        "portlearn/data/dataset.py (the sole pandas home); found "
        f"{import_locations!r}"
    )

    # No optional-pandas machinery survives anywhere in the package: no
    # guarded-import symbol, no pandas-extra string, no try/except-
    # ImportError pandas path (floor 18's source-scan teeth).
    for path in sources:
        text = path.read_text(encoding="utf-8")
        assert "MissingPandasExtraError" not in text, (
            f"{path.name}: MissingPandasExtraError is forbidden — pandas "
            "is a normal core dependency (L11)"
        )
        assert "[pandas]" not in text, (
            f"{path.name}: a [pandas] extra string is forbidden "
            "optional-pandas machinery"
        )
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Import) and any(
                        alias.name == "pandas" for alias in inner.names
                    ):
                        raise AssertionError(
                            f"{path.name}: a try/except-ImportError pandas "
                            "import is a guarded-import path — pandas is "
                            "unconditional (L11)"
                        )


# --------------------------------------------------------------------------- #
# Law 11 — alias selection is frequency-scoped, versioned
# --------------------------------------------------------------------------- #


def test_alias_frequency_is_selector_only_and_versioned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Floor 12: aliases map research names to provider dataset ids as a
    selector only — unknown aliases reject before any network, ambiguity
    and wrong-frequency reject listing the artifacts, and no artifact is
    ever resampled."""
    import portlearn as pl

    _patch_ff_network_routed(monkeypatch)
    ds = _load_industry49()
    assert ds.name == "industry49"
    assert ds.provider_dataset_id == "industry49_monthly_csv"

    _patch_ff_network_routed(monkeypatch)
    daily = pl.data.fama_french.load("factors", frequency="daily")
    assert daily.provider_dataset_id == "factors_daily_csv"
    assert daily.frequency == "daily"
    # Selector only: daily period labels stay daily — never resampled.
    assert daily.records
    assert all(len(record.period_key) == 8 for record in daily.records)

    # Ambiguity: an alias with multiple artifacts and no frequency
    # rejects, listing the artifacts.
    _patch_ff_network_routed(monkeypatch)
    with pytest.raises(ValueError) as ambiguity:
        pl.data.fama_french.load("factors")
    message = str(ambiguity.value)
    assert "factors_monthly_csv" in message
    assert "factors_daily_csv" in message

    # Wrong frequency: not among the alias's artifacts rejects.
    with pytest.raises(ValueError):
        pl.data.fama_french.load("factors", frequency="weekly")

    # Unknown alias rejects before any network is touched.
    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("no network may be touched on rejection")

    ff = _ff_module()
    monkeypatch.setattr(ff, "urlopen", _explode)
    with pytest.raises(ValueError):
        pl.data.fama_french.load("not_a_research_alias", frequency="monthly")


# --------------------------------------------------------------------------- #
# Law 12 — FRED CURRENT_SNAPSHOT carried, never invented
# --------------------------------------------------------------------------- #


def test_fred_unqualified_carries_current_snapshot_limitation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Floor 13: the FRED facade carries the CURRENT_SNAPSHOT limitation:
    unqualified records carry no invented availability; the sanitized
    provenance never contains the key; qualification carries the mode."""
    import portlearn as pl

    _patch_fred_network(monkeypatch)
    ds = pl.data.fred.load(
        "SYNTHCPIM",
        api_key="SYNTHETIC-KEY-123",
        frequency="monthly",
    )

    assert ds.data_mode == "CURRENT_SNAPSHOT"
    assert ds.provider == "fred"
    assert ds.auxiliary_bytes is not None
    for record in ds.records:
        assert not hasattr(record, "available_time")
    dumped = repr(ds.retrieval_provenance)
    assert "SYNTHETIC-KEY-123" not in dumped
    assert "api_key" not in dumped
    assert ds.qualified_provenance is None


# --------------------------------------------------------------------------- #
# Law 13 — no default availability constructed anywhere
# --------------------------------------------------------------------------- #


def test_no_default_availability_constructs_anywhere() -> None:
    """Floor 14 (with L14): the data-package sources construct no
    AvailabilityPolicy and stamp no availability-shaped value: the
    test-side check scans the data-package sources for any availability
    construction call.  Availability enters only as a caller-supplied
    argument."""
    forbidden_calls = {
        "AvailabilityPolicy.fixed_lag",
        "AvailabilityPolicy.same_instant",
        "AvailabilityPolicy.explicit_column",
    }
    for source in FACADE_SOURCES + ADAPTER_SOURCES:
        assert source.is_file(), f"the data-package source is missing: {source}"
        text = source.read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                rendered = ast.unparse(node.func)
                assert rendered not in forbidden_calls, (
                    f"{source.name} constructs an availability policy "
                    f"({rendered}) — availability enters only as a "
                    "caller-supplied argument"
                )
            if isinstance(node, ast.ImportFrom):
                targets = {alias.name for alias in node.names}
                assert not targets & {"AvailabilityPolicy"}, (
                    f"{source.name} imports AvailabilityPolicy — no "
                    "default availability may be constructed"
                )


# --------------------------------------------------------------------------- #
# Law 14 — hash contract and immutable retrieval params
# --------------------------------------------------------------------------- #


def test_researchdataset_hash_contract_and_immutable_retrieval_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Floor 15: value equality over all fields; hash derives solely
    from the fixed identity tuple (L8); no unhashable provenance field
    can break hashing; FRED request_params is canonically immutable."""
    import portlearn as pl

    _patch_ff_network(monkeypatch, _industry49_bytes())
    first = _load_industry49()
    second = _load_industry49()
    assert first == second
    assert hash(first) == hash(second)

    qualified_first = first.qualify(availability=_policy(), tzinfo=TZ)
    qualified_second = second.qualify(availability=_policy(), tzinfo=TZ)
    equal_q = qualified_first == qualified_second
    assert equal_q
    assert hash(qualified_first) == hash(qualified_second)

    # L8: provenance and records are excluded from the hash while
    # equality remains value-based: different policies over the same
    # bytes hash differently or collide — never break the contract.
    other_policy = AvailabilityPolicy.fixed_lag(
        timedelta(days=45),
        justification="a different declared research assumption",
    )
    other = first.qualify(availability=other_policy, tzinfo=TZ)
    assert other != qualified_first

    # Mutations remain impossible in both states (floor 16 folded here).
    with pytest.raises(Exception):  # noqa: B017 — immutability is the law
        object.__setattr__(first, "availability_state", "QUALIFIED")
    with pytest.raises(Exception):  # noqa: B017 — immutability is the law
        object.__setattr__(qualified_first, "records", ())

    # No unhashable provenance field can break hashing: the FRED
    # qualified provenance legitimately carries a dict request_params
    # and raises on hash() — hashing the dataset never touches it.
    _patch_fred_network(monkeypatch)
    fred_ds = pl.data.fred.load(
        "SYNTHCPIM", api_key="SYNTHETIC-KEY-123", frequency="monthly"
    )
    fred_qualified = fred_ds.qualify(availability=_policy(), tzinfo=TZ)
    hash(fred_qualified)
    params = fred_qualified.qualified_provenance.request_params
    with pytest.raises(Exception):  # noqa: B017 — dict unhashable by design
        hash(params)


# --------------------------------------------------------------------------- #
# Law 17 — the undeclared top-level adapter namespace is absent
# --------------------------------------------------------------------------- #


_UNDECLARED_TOP_LEVEL_NAMESPACE_PROBE = (
    "import importlib, json\n"
    "\n"
    "\n"
    "def absent(name):\n"
    "    try:\n"
    "        importlib.import_module(name)\n"
    "    except ModuleNotFoundError:\n"
    "        return True\n"
    "    return False\n"
    "\n"
    "\n"
    "import portlearn\n"
    "\n"
    "print(\n"
    "    json.dumps(\n"
    "        {\n"
    '            "portlearn.adapters": absent("portlearn.adapters"),\n'
    '            "portlearn.adapters.ff": absent(\n'
    '                "portlearn.adapters.ff"\n'
    "            ),\n"
    '            "portlearn.adapters.fred": absent(\n'
    '                "portlearn.adapters.fred"\n'
    "            ),\n"
    '            "root adapters attribute": hasattr(portlearn, "adapters"),\n'
    "        }\n"
    "    )\n"
    ")\n"
)


def _undeclared_top_level_namespace_state() -> dict[str, Any]:
    """Probe the undeclared top-level adapter namespace in a fresh
    interpreter.

    The reference runs out-of-process (the Floor 19 probe pattern) so it
    needs no in-process ``sys.modules`` purge: purging and reimporting
    ``portlearn*`` inside the pytest interpreter re-executes package
    modules and drifts class identity for every later test — the
    harness-isolation defect this probe pattern avoids.
    """
    probe = subprocess.run(
        [sys.executable, "-c", _UNDECLARED_TOP_LEVEL_NAMESPACE_PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        pytest.fail(
            "fresh-interpreter undeclared-namespace probe failed: "
            f"{probe.stderr.strip()}"
        )
    return json.loads(probe.stdout)


def test_undeclared_top_level_adapter_namespace_is_absent() -> None:
    """Floor 17: the undeclared top-level adapter namespace is absent
    from the public package surface — importing ``portlearn.adapters``
    and its provider submodules raises ``ModuleNotFoundError`` in a
    fresh interpreter, ``portlearn`` carries no attribute or submodule
    of that name, and ``src/portlearn/adapters/`` does not exist on
    disk."""
    state = _undeclared_top_level_namespace_state()
    for name in (
        "portlearn.adapters",
        "portlearn.adapters.ff",
        "portlearn.adapters.fred",
    ):
        assert state[name], (
            f"importing {name!r} must raise ModuleNotFoundError — the "
            "top-level adapter namespace is not part of the public "
            "package surface"
        )
    assert not state["root adapters attribute"], (
        "portlearn.adapters is not part of the public package surface: "
        "no attribute or submodule of that name may exist on portlearn"
    )
    assert not (PACKAGE_ROOT / "adapters").exists(), (
        "src/portlearn/adapters/ must not exist on disk — the top-level "
        "adapter namespace is not part of the public package surface"
    )


# --------------------------------------------------------------------------- #
# Law 18 — no optional pandas machinery survives
# --------------------------------------------------------------------------- #


def _requirement_name(entry: str) -> str:
    """The distribution name of one ``Requires-Dist`` entry."""
    head = entry.split(";", 1)[0]
    for marker in ("<", ">", "=", "!", "[", " "):
        head = head.split(marker, 1)[0]
    return head.strip().lower()


def test_no_optional_pandas_machinery_survives() -> None:
    """Floor 18: pandas is core — the optional table is exactly the
    parquet and plot extras, the installed distribution carries pandas
    and SciPy as unconditional ``Requires-Dist`` entries,
    and no guarded-import symbol or pandas-extra message survives in
    the package (L11, tooth W11)."""
    with PYPROJECT_PATH.open("rb") as handle:
        pyproject = tomllib.load(handle)
    optional = pyproject.get("project", {}).get("optional-dependencies", {})
    assert "pandas" not in optional, (
        "pandas is a core runtime dependency — a [pandas] extra is "
        "forbidden optional-pandas machinery"
    )
    assert set(optional) == {"parquet", "plot"}, (
        "optional capability groups are exactly the parquet and plot "
        f"extras; found {sorted(optional)}"
    )

    requires = list(metadata.requires("portlearn") or [])
    pandas_entries = [
        entry for entry in requires if _requirement_name(entry) == "pandas"
    ]
    assert pandas_entries, (
        "the installed distribution must carry pandas as an "
        "unconditional core Requires-Dist entry; "
        f"found {requires!r}"
    )
    for entry in pandas_entries:
        assert "extra ==" not in entry, (
            f"the pandas requirement must be unconditional — no extra "
            f"marker; found {entry!r}"
        )

    # No guarded-import error symbol survives in the package namespace.
    import portlearn

    assert not hasattr(portlearn, "MissingPandasExtraError"), (
        "MissingPandasExtraError must not exist — pandas is a normal "
        "core dependency with no guarded-import error path"
    )


# --------------------------------------------------------------------------- #
# Law 19 — the import DAG is acyclic and laziness holds
# --------------------------------------------------------------------------- #


def _fresh_registry(statement: str) -> list[str]:
    """Execute one import statement in a fresh interpreter and return
    the resulting ``sys.modules`` snapshot (offline, network-free)."""
    script = f"import json, sys\n{statement}\nprint(json.dumps(sorted(sys.modules)))\n"
    probe = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        pytest.fail(
            f"fresh-interpreter import probe failed for {statement!r}: "
            f"{probe.stderr.strip()}"
        )
    return json.loads(probe.stdout)


def _portlearn_side(registry: list[str]) -> list[str]:
    return [name for name in registry if name.split(".", 1)[0] == "portlearn"]


def test_import_dag_is_acyclic_and_laziness_holds() -> None:
    """Floor 19: the import graph is the pinned L17 DAG — a lazy root, a
    lazy data package, no cross-facade edge, no upward adapter edge, and
    pandas triggered by ``dataset`` alone (tooth W12)."""
    # (i) the root stays lazy: a fresh import registers the root only.
    assert _portlearn_side(_fresh_registry("import portlearn")) == ["portlearn"], (
        "a fresh root import must register exactly the root module"
    )

    # (ii) the data package registers itself and no provider submodule.
    assert _portlearn_side(_fresh_registry("import portlearn.data")) == [
        "portlearn",
        "portlearn.data",
    ], "import portlearn.data must register no facade/provider submodule"

    # (iii) importing one facade never registers the sibling provider.
    for statement, banned in (
        ("import portlearn.data.fama_french", "fred"),
        ("import portlearn.data.fred", "fama_french"),
    ):
        registered = _portlearn_side(_fresh_registry(statement))
        smuggled = [name for name in registered if banned in name]
        assert not smuggled, (
            f"{statement} registered sibling-provider modules {smuggled} "
            "— no facade imports its sibling facade"
        )

    # (iv) an adapter import never climbs to a facade module (no cycle;
    # L17 rule ii: nothing below portlearn.data imports anything at or
    # above the facade layer).
    banned_facades = {
        "portlearn.data.fama_french",
        "portlearn.data.fred",
        "portlearn.data.dataset",
    }
    for module in (
        "portlearn.data.adapters",
        "portlearn.data._records",
        "portlearn.data.adapters.ff",
        "portlearn.data.adapters.fred",
    ):
        registered = _portlearn_side(_fresh_registry(f"import {module}"))
        climbed = set(registered) & banned_facades
        assert not climbed, (
            f"import {module} registered facade modules {sorted(climbed)} "
            "— no module below portlearn.data imports anything at or "
            "above it (L17 rule ii)"
        )

    # (v) pandas is triggered by dataset's single home and by the two
    # facades that import it — and by nothing else in the data package.
    pandas_trigger_modules = {
        PANDAS_HOME,
        "portlearn.data.fama_french",
        "portlearn.data.fred",
    }
    for module in DATA_PACKAGE_MODULES:
        has_pandas = "pandas" in _fresh_registry(f"import {module}")
        assert has_pandas == (module in pandas_trigger_modules), (
            f"import {module} pandas-in-registry={has_pandas} — pandas "
            "is triggered only through the dataset module's single "
            "module-scope home"
        )

    # (vi) the root import alone never registers pandas.
    assert "pandas" not in _fresh_registry("import portlearn"), (
        "importing the root package must never import pandas"
    )


# --------------------------------------------------------------------------- #
# Law 21 — public constructors suffice; no privileged construction
# --------------------------------------------------------------------------- #


def test_public_constructors_suffice_no_privileged_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Floor 21: a valid ``ResearchDataset`` in both states is
    constructed exclusively through the documented public contracts —
    caller bytes, retrieval provenance, the public ``load``, and the
    canonical ``qualify`` — and the facade modules contain no
    construction site external code cannot use (L16: built-ins have no
    privileged dataset-construction access)."""
    import portlearn as pl

    ff = _ff_module()
    _patch_ff_network(monkeypatch, _industry49_bytes())
    data, retrieval = ff.fetch_raw("industry49_monthly_csv")

    unqualified = pl.data.fama_french.load(
        "industry49", frequency="monthly", data=data, retrieval=retrieval
    )
    assert unqualified.availability_state == "UNQUALIFIED"

    qualified = unqualified.qualify(availability=_policy(), tzinfo=TZ)
    assert qualified.availability_state == "QUALIFIED"

    for dataset in (unqualified, qualified):
        assert dataset.provider == "fama_french"
        assert dataset.source_bytes == data
        assert dataset.source_sha256 == _sha256(data)

    # The facade modules construct datasets through exactly the public
    # contracts: no local dataset redefinition, no direct record
    # fabrication bypassing the fixed decoders and the canonical
    # constructor any external integrator can call.
    for source in (
        PACKAGE_ROOT / "data" / "fama_french.py",
        PACKAGE_ROOT / "data" / "fred.py",
    ):
        assert source.is_file(), f"the facade module is missing: {source}"
        text = source.read_text(encoding="utf-8")
        assert "class ResearchDataset" not in text, (
            f"{source.name}: the facade may not redefine the dataset "
            "container — one public constructor exists"
        )
        assert "TimedObservation(" not in text, (
            f"{source.name}: the facade may not fabricate qualified "
            "records — record construction belongs to the fixed "
            "decoders behind the public constructor"
        )
        assert "PeriodKeyObservation(" not in text, (
            f"{source.name}: the facade may not fabricate unqualified "
            "records — record construction belongs to the fixed "
            "decoders behind the public constructor"
        )


# --------------------------------------------------------------------------- #
# Provider neutrality — the container is a provider-neutral kernel
# --------------------------------------------------------------------------- #


def _kernel_dataset(provider: Any, **overrides: Any) -> Any:
    """Construct one UNQUALIFIED dataset through the internal generic
    constructor over real fixture bytes — no facade, no provider branch.

    The bytes, records, provenance, and decoder are FF's; only the
    provider id varies, which is exactly the capability under test.
    """
    from portlearn.data.adapters import ff as ff_adapter
    from portlearn.data.dataset import UnqualifiedDataset

    data = _industry49_bytes()
    records, provenance = ff_adapter.decode_unqualified(data, "industry49_monthly_csv")
    overrides.setdefault("provider", provider)
    overrides.setdefault("name", "industry49")
    overrides.setdefault("provider_dataset_id", "industry49_monthly_csv")
    overrides.setdefault("adapter_version", provenance.adapter_version)
    overrides.setdefault("source_bytes", data)
    overrides.setdefault("auxiliary_bytes", None)
    overrides.setdefault("records", tuple(records))
    overrides.setdefault("retrieval_provenance", provenance)
    overrides.setdefault("decoder", ff_adapter.decode)
    return UnqualifiedDataset(**overrides)


def test_dataset_kernel_accepts_any_opaque_provider_id() -> None:
    """The provider id is an opaque, structurally validated string: a
    private source or an unsupported public provider composes into the
    same sealed kernel with no kernel edit, no registry, and no
    provider subclass."""
    for provider in ("my_private_source", "crsp"):
        ds = _kernel_dataset(provider)
        assert ds.provider == provider
        assert ds.availability_state == "UNQUALIFIED"
        qualified = ds.qualify(availability=_policy(), tzinfo=TZ)
        assert qualified.provider == provider
        assert qualified.availability_state == "QUALIFIED"


@pytest.mark.parametrize(
    "bad_provider",
    ["", "   ", "\t\n", 7, 3.5, None, b"fama_french", ["fred"]],
)
def test_dataset_kernel_rejects_blank_and_non_string_provider_ids(
    bad_provider: Any,
) -> None:
    """Blank and non-string provider ids reject unconditionally at
    construction; every non-blank string is valid."""
    with pytest.raises(ValueError):
        _kernel_dataset(bad_provider)


def test_researchdataset_imports_from_the_public_data_package() -> None:
    """``from portlearn.data import ResearchDataset`` is the public
    import surface; the state types stay internal, and the lazy import
    laws of the package are preserved."""
    import portlearn.data as data_package
    from portlearn.data import ResearchDataset

    assert ResearchDataset.__name__ == "ResearchDataset"
    assert "ResearchDataset" in data_package.__all__
    assert "UnqualifiedDataset" not in data_package.__all__
    assert "QualifiedDataset" not in data_package.__all__
    assert not hasattr(data_package, "UnqualifiedDataset")
    assert not hasattr(data_package, "QualifiedDataset")


def test_dataset_provenance_routes_to_the_active_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``dataset.provenance`` is a read-only routing surface: the
    retrieval provenance while UNQUALIFIED, the qualified provenance
    once QUALIFIED; the advanced named surfaces remain; no setter
    exists."""
    _patch_ff_network(monkeypatch, _industry49_bytes())
    ds = _load_industry49()

    assert ds.provenance is ds.retrieval_provenance
    qualified = ds.qualify(availability=_policy(), tzinfo=TZ)
    assert qualified.provenance is qualified.qualified_provenance
    assert qualified.retrieval_provenance is ds.retrieval_provenance

    with pytest.raises(AttributeError):
        ds.provenance = ds.retrieval_provenance  # type: ignore[misc]


def test_data_mode_delegates_to_the_active_typed_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``data_mode`` carries no kernel storage: it reads the active
    typed provenance with no provider-name branch — present for FRED
    (CURRENT_SNAPSHOT), absent for FF whose provenance carries none."""
    import portlearn as pl

    _patch_fred_network(monkeypatch)
    fred_ds = pl.data.fred.load(
        "SYNTHCPIM", api_key="SYNTHETIC-KEY-123", frequency="monthly"
    )
    assert fred_ds.data_mode == "CURRENT_SNAPSHOT"
    assert fred_ds.data_mode == fred_ds.retrieval_provenance.data_mode
    fred_qualified = fred_ds.qualify(availability=_policy(), tzinfo=TZ)
    assert fred_qualified.data_mode == (fred_qualified.qualified_provenance.data_mode)

    _patch_ff_network(monkeypatch, _industry49_bytes())
    ff_ds = _load_industry49()
    assert not hasattr(ff_ds, "data_mode")


# --------------------------------------------------------------------------- #
# Law 22 — relocated adapters preserve qualified behavior and output
# --------------------------------------------------------------------------- #


def _run_smoke_example(monkeypatch: pytest.MonkeyPatch) -> str:
    """Execute the canonical smoke example offline, capturing stdout."""

    def _refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "the canonical smoke example attempted a network connection"
        )

    monkeypatch.setattr(socket, "socket", _refuse)
    with redirect_stdout(io.StringIO()) as buffer:
        runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")
    return buffer.getvalue()


def test_relocated_adapters_preserve_qualified_behavior_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Floor 22: qualified behavior and output survive the relocation —
    canonical ``adapter_identity`` labels at the new namespace,
    the fixed
    qualified decode at the new namespace over the pinned fixture, and
    the canonical smoke example executing unchanged apart from its
    single relocated import line."""
    ff = _ff_module()
    fred = _fred_module()

    assert ff.ADAPTER_IDENTITY == "portlearn.data.adapters.ff"
    assert fred.ADAPTER_IDENTITY == "portlearn.data.adapters.fred"

    _patch_ff_network(monkeypatch, _industry49_bytes())
    data, retrieval = ff.fetch_raw("industry49_monthly_csv")
    timed, provenance = ff.decode(
        data,
        "industry49_monthly_csv",
        _policy(),
        tzinfo=TZ,
        retrieval=retrieval,
    )
    assert len(timed) == 292
    assert isinstance(timed[0], TimedObservation)
    assert provenance.adapter_identity == "portlearn.data.adapters.ff"
    assert provenance.content_sha256 == PINNED_INDUSTRY49_SHA256

    output = _run_smoke_example(monkeypatch)
    assert "portlearn.data.adapters.ff" in output
    assert "portlearn.data.adapters.fred" in output
