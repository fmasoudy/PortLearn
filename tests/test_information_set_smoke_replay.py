"""Offline replay contracts for the information-set smoke example.

The example under test composes PortLearn's public decoder, transform,
alignment, information-set, leakage, and manifest surfaces over committed
synthetic provider-format fixtures, entirely offline.  These nodes pin that
composition: an end-to-end offline replay with the network unavailable,
provenance summary content, availability facts at the publication boundary,
a fully blocked leakage battery, identical determinism across repeated
runs, the example's import discipline, and four-family admission at the
decision instant.
"""

from __future__ import annotations

import ast
import hashlib
import io
import runpy
import socket
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from portlearn.alignment import ObservationStore, align
from portlearn.calendar import month_end_instant
from portlearn.data.adapters import ff, fred
from portlearn.data.ingestion import AvailabilityPolicy
from portlearn.interfaces import InformationSet
from portlearn.leakage import run_leakage_cases
from portlearn.manifest import RunManifest
from portlearn.observations import (
    AmbiguousObservationError,
    FeatureLineageError,
)
from portlearn.timing import FutureInformationError, NaiveTimestampError

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "information_set_smoke.py"
FF_FIXTURES = REPO / "tests" / "adapters" / "fixtures" / "ff"
FRED_FIXTURES = REPO / "tests" / "adapters" / "fixtures" / "fred"
DIGEST_MARKER = "[h] replay output sha256: "

ZONE = ZoneInfo("Australia/Melbourne")


def _run_example() -> str:
    """Execute the example as a script and capture its output."""
    with redirect_stdout(io.StringIO()) as buffer:
        runpy.run_path(str(EXAMPLE), run_name="__main__")
    return buffer.getvalue()


def _import_example() -> dict[str, Any]:
    """Import the example module body without running ``main``."""
    with redirect_stdout(io.StringIO()) as buffer:
        namespace = runpy.run_path(
            str(EXAMPLE), run_name="information_set_smoke_module"
        )
    assert buffer.getvalue() == ""
    return namespace


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_offline_replay_runs_end_to_end_with_the_network_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _refuse_connection(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "the offline replay attempted a network connection"
        )

    monkeypatch.setattr(socket, "socket", _refuse_connection)
    output = _run_example()
    namespace = _import_example()
    assert namespace["FENCE"] in output
    assert "[1] decision " in output
    assert "9 records admitted from 9 requested observation groups" in output
    assert "[b] leakage battery: 4/4 attempted leaks blocked" in output


def test_source_summary_carries_every_family_hash_policy_and_adapter() -> (
    None
):
    output = _run_example()
    fixtures = (
        FF_FIXTURES / "ff_factors_monthly_csv.zip",
        FF_FIXTURES / "ff_factors_daily_csv.zip",
        FF_FIXTURES / "ff_industry49_monthly_csv.zip",
        FRED_FIXTURES / "obs_synthcpim_monthly.json",
    )
    for fixture in fixtures:
        assert _digest(fixture.read_bytes()) in output
    assert "portlearn.data.adapters.ff" in output
    assert "portlearn.data.adapters.fred" in output
    assert "FIXED_LAG" in output
    assert "CURRENT_SNAPSHOT" in output
    assert "portfolio_returns" in output
    assert "NOT investable" in output
    assert "RollingMean" in output
    assert "window=5" in output
    manifest_line = next(
        line
        for line in output.splitlines()
        if line.startswith("[m] manifest: ")
    )
    manifest = RunManifest.from_json(
        manifest_line.split("[m] manifest: ", 1)[1]
    )
    assert manifest.run_id == "information-set-smoke-offline-replay"
    assert manifest.dependency_pins["portlearn"]
    assert RunManifest.from_json(manifest.to_json()) == manifest


def test_pre_publication_print_stays_invisible_until_the_boundary() -> None:
    policy = AvailabilityPolicy.fixed_lag(
        timedelta(days=28),
        justification=(
            "synthetic release schedule declared for the committed "
            "fixture replica"
        ),
    )
    records, _ = fred.decode(
        (FRED_FIXTURES / "obs_synthcpim_monthly.json").read_bytes(),
        "SYNTHCPIM",
        policy,
        data_mode=fred.CURRENT_SNAPSHOT,
        tzinfo=ZONE,
        frequency="Monthly",
        metadata_bytes=(
            FRED_FIXTURES / "meta_synthcpim.json"
        ).read_bytes(),
    )
    floor = datetime(2026, 9, 12, 23, 59, 59, 999999, tzinfo=ZONE)
    december = month_end_instant(2025, 12, ZONE)
    macro = [r for r in records if r.observation_time == december]
    assert len(macro) == 1
    assert macro[0].available_time == floor
    store = ObservationStore(records)
    pre_publication = month_end_instant(2026, 8, ZONE)
    assert align(store, pre_publication, [("FRED/SYNTHCPIM", december)]) == []
    boundary = align(store, floor, [("FRED/SYNTHCPIM", december)])
    assert len(boundary) == 1
    InformationSet(boundary, as_of=floor)
    with pytest.raises(FutureInformationError):
        InformationSet(boundary, as_of=floor - timedelta(microseconds=1))
    output = _run_example()
    assert "0 of 1 requested groups admitted" in output
    assert boundary[0].available_time.isoformat() in output
    assert "FutureInformationError" in output


def test_leakage_battery_blocks_every_attempted_leak() -> None:
    namespace = _import_example()
    replay = namespace["compose_replay"]()
    report = run_leakage_cases(replay.leakage_cases)
    assert report.all_blocked
    expected = {case.expected_error for case in replay.leakage_cases}
    assert expected == {
        FutureInformationError,
        AmbiguousObservationError,
        NaiveTimestampError,
        FeatureLineageError,
    }
    output = _run_example()
    assert "[b] leakage battery: 4/4 attempted leaks blocked" in output


def test_replay_output_is_byte_identical_across_two_runs() -> None:
    first = _run_example()
    second = _run_example()
    assert first == second
    index = first.rindex(DIGEST_MARKER)
    digest = first[index + len(DIGEST_MARKER) :].strip()
    assert len(digest) == 64
    assert digest == hashlib.sha256(first[:index].encode()).hexdigest()


def test_example_import_is_side_effect_free_and_public_only() -> None:
    source = EXAMPLE.read_text()
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
    allowed = {
        "__future__",
        "os",
        "hashlib",
        "platform",
        "datetime",
        "pathlib",
        "types",
        "zoneinfo",
        "portlearn",
    }
    assert roots <= allowed, roots - allowed
    for banned in ("urllib", "socket", "http", "ssl", "requests"):
        assert banned not in roots
    for claim in (
        "performance",
        "strategy",
        "backtest",
        "sharpe",
        "evaluation",
        "weighting",
    ):
        assert claim not in source.lower()
    assert "def main() -> None:" in source
    namespace = _import_example()
    assert callable(namespace["main"])
    assert callable(namespace["compose_replay"])


def test_information_set_admits_all_four_families_at_the_decision() -> None:
    namespace = _import_example()
    replay = namespace["compose_replay"]()
    admitted = InformationSet(
        replay.aligned_records, as_of=replay.decision_instant
    )
    series = sorted(record.series_id for record in admitted)
    assert len(series) == 9
    for name in (
        "FF/Mkt-RF",
        "FF/SMB",
        "FF/HML",
        "FF/RF",
        "FF/Banks",
        "FF/Oil",
        "FF/Util",
        "FRED/SYNTHCPIM",
    ):
        assert name in series
    assert any("|rolling_mean[5]" in name for name in series)
    assert all(
        record.available_time <= replay.decision_instant
        for record in admitted
    )
    ff.require_candidate_asset_returns(replay.portfolio_provenance)
    with pytest.raises(ValueError):
        ff.require_candidate_asset_returns(replay.factor_provenance)
