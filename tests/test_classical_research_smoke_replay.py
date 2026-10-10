"""Offline replay contracts for the classical research smoke example.

The example under test composes PortLearn's complete public research
runtime -- declared entry adapter, temporal protocol resolution, the
experiment runtime, accounting, metrics, and descriptive comparison
rendering -- over a committed long synthetic provider-format fixture,
entirely offline.  These nodes pin that composition: an end-to-end offline
replay through the documented ``--fixture`` entry point with sockets and
``urlopen`` refused, anti-bypass sentinels on the live public runtime
seams, import-time side-effect discipline, and synthetic self-declaration
in fixture mode (fence, ``synthetic=yes``, and the committed fixture's own
synthetic labelling in bytes and manifest).

The tranche-B extension floors pin the recorded-path invariants:
point-in-time visibility at every live decision instant with the pinned
first-instant visibility facts and the frozen admission validators (F5);
the common-conditions wire, strictly positive first inverse-volatility
rebalance turnover, and the zero-rate mutation kill on the net ledger
digest (F6 -- under the classical-research section-15 authorized narrowing: net-digest
mutation plus conditions-wire visibility, because the frozen report
surface declares no gross-basis MetricsBlock); fail-closed snapshot
identity with both the pinned and the observed SHA-256 in the rejection
and zero executed decisions (F7); and byte-identical deterministic
replay across repeated CLI executions in fixture mode, with the
manifest-pinned real-snapshot leg exercised only when those exact bytes
are locally possessed and never substituted (F8).

All instrumentation is test-local and non-public: call-through spies on
the frozen ``Experiment`` construction and strategy ``decide`` surfaces,
and a temporary zero-rate subclass substituted for the frozen public
``portlearn.costs.Proportional`` constructor symbol.  No production or
example surface is altered and no new public API is invented.

The comparator-drop readiness leg is closed without any private example
seam: on a separate actual ``--fixture`` CLI execution the public
``portlearn.research.run_declared_experiment`` call-through lets the real
function execute and then returns a type-correct ``ExperimentResults``
with one declared arm removed while the provenance mapping is retained;
the frozen readiness chain must reject by name before any comparison
block or completed report.

The final batch pins the frozen data and surface floors: genuine
retained provider bytes decoded through the public adapter route with
the network refused, admitted by the asset-return fence with the
factor-kind rejection demonstrated, and a transparent conditional
custody boundary when the manifest-pinned bytes are absent (F9); a
descriptive-only report and rendered table -- none of the frozen
forbidden tokens, exactly the six frozen metrics in declared order,
no derived contrast, and no example post-processing of the rendered
table (F10); and refresh as the only network path -- the plain run
opens no socket and completes offline, while the documented
``--refresh`` CLI path fetches once, writes the snapshot, prints the
derived manifest JSON, and runs no experiment (F11).
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import re
import runpy
import socket
import sys
import urllib.request
import zipfile
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout, suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

import portlearn
import portlearn.comparisons
import portlearn.costs
import portlearn.experiments
import portlearn.ledger
import portlearn.research
import portlearn.strategies
from portlearn.calendar import month_end_instant
from portlearn.data import fama_french
from portlearn.data.adapters import ff
from portlearn.interfaces import InformationSet
from portlearn.timing import FutureInformationError, NaiveTimestampError

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "classical_research_smoke.py"
FF_FIXTURES = REPO / "tests" / "adapters" / "fixtures" / "ff"
MANIFEST = FF_FIXTURES / "MANIFEST.md"

#: The four short-history synthetic replicas committed before classical-research; the
#: long fixture authorized by classical-research is every committed zip beyond this set.
SHORT_HISTORY_FIXTURES = frozenset(
    {
        "ff_factors_monthly_csv.zip",
        "ff_factors_monthly_txt.zip",
        "ff_factors_daily_csv.zip",
        "ff_industry49_monthly_csv.zip",
    }
)

#: Distinctive phrases of the one-line boundary fence (classical-research section 7); a
#: fence line must carry all three.
FENCE_PHRASES = (
    "instructional framework-operation smoke only",
    "NOT investable",
    "no performance, superiority, or empirical claim is made or possible",
)

#: The frozen metric names every arm must restate (classical-research section 6).
METRIC_NAMES = (
    "cumulative_return",
    "annualized_mean_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "certainty_equivalent_return",
)

#: The two declared arms (classical-research section 6) and the declared research zone.
ARMS = ("equal_weight", "inverse_volatility")
ZONE = ZoneInfo("Australia/Melbourne")

#: The committed source manifest pinning the run-of-record snapshot hash
#: and the gitignored local snapshot location (classical-research section 4).
SOURCES_MANIFEST = EXAMPLE.parent / "classical_research_sources.json"
REAL_SNAPSHOT = (
    REPO / "scratch" / "classical_research_snapshot" / "industry49_monthly_csv.zip"
)

#: A digest-stamped report line: a sha256 mention carrying a 64-hex digest.
_DIGEST_LINE = re.compile(r"sha256\D*?(?P<d>[0-9a-f]{64})")

#: The exact digest marker line that must close every report (F8).
_MARKER_LINE = re.compile(r"DIGEST_MARKER sha256=(?P<d>[0-9a-f]{64})")

#: The one permitted volatile footer line (classical-research section 10): it must
#: begin with ``created_at`` and carry a separator before its value; it
#: is never part of the hashed deterministic prefix.
_CREATED_AT_LINE = re.compile(r"created_at\W.*")

#: The forbidden claim vocabulary, exactly as frozen in classical-research section 8
#: floor F10: the rendered table and the report carry none of it.
FORBIDDEN_CLAIM_TOKENS = (
    "rank",
    "winner",
    "best",
    "superior",
    "outperform",
    "beats",
    "significant",
    "p-value",
    "confidence interval",
)

#: Directories whose contents an import of the example must not disturb.
_WATCHED_DIRS = (REPO, EXAMPLE.parent, FF_FIXTURES)


class _PublicPathSentinel(RuntimeError):
    """Raised through a public seam; surfacing proves the seam executed."""


def _refuse_connection(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("the offline replay attempted a network connection")


def _refuse_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "socket", _refuse_connection)
    monkeypatch.setattr(urllib.request, "urlopen", _refuse_connection)


def _sentinel_experiment_run(self: Any, selection: Any = None) -> Any:
    raise _PublicPathSentinel("SENTINEL: public Experiment.run executed")


def _sentinel_resolve(*args: Any, **kwargs: Any) -> Any:
    raise _PublicPathSentinel("SENTINEL: public resolve_temporal_protocols executed")


def _run_fixture_replay(monkeypatch: pytest.MonkeyPatch) -> str:
    """Execute the example as ``__main__`` via the documented fixture entry."""
    _require_example()
    monkeypatch.setattr(sys, "argv", ["classical_research_smoke.py", "--fixture"])
    with redirect_stdout(io.StringIO()) as buffer:
        runpy.run_path(str(EXAMPLE), run_name="__main__")
    return buffer.getvalue()


def _require_example() -> None:
    """Fail with the designed causal marker while the example is absent."""
    if not EXAMPLE.is_file():
        pytest.fail(
            f"EXPECTED_RED absent authorized artifact: {EXAMPLE.relative_to(REPO)}",
            pytrace=False,
        )


def _fence_lines(output: str) -> list[str]:
    return [
        line
        for line in output.splitlines()
        if all(phrase in line for phrase in FENCE_PHRASES)
    ]


def _is_fence_line(line: str) -> bool:
    return all(phrase in line for phrase in FENCE_PHRASES)


def _snapshot(path: Path) -> list[tuple[str, int, int, bool]]:
    return sorted(
        (
            child.name,
            child.stat().st_mtime_ns,
            child.stat().st_size,
            child.is_dir(),
        )
        for child in path.iterdir()
    )


# --------------------------------------------------------------------------- #
# Tranche-B instrumentation (non-public, call-through, architecture-neutral)   #
# --------------------------------------------------------------------------- #


def _spy_decisions(
    monkeypatch: pytest.MonkeyPatch,
    sink: list[tuple[str, Any, Any]],
) -> None:
    """Wrap both frozen strategy ``decide`` seams, recording every call."""

    for strategy in (
        portlearn.strategies.EqualWeight,
        portlearn.strategies.InverseVolatility,
    ):

        def _wrapped(
            self: Any,
            context: Any,
            _original: Any = strategy.decide,
        ) -> Any:
            result = _original(self, context)
            sink.append((type(self).__name__, context, result))
            return result

        monkeypatch.setattr(strategy, "decide", _wrapped)


def _spy_conditions(monkeypatch: pytest.MonkeyPatch, sink: list[Any]) -> None:
    """Wrap frozen ``Experiment.__init__``, recording injected conditions."""

    original = portlearn.experiments.Experiment.__init__

    def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        sink.append(args[3] if len(args) == 4 else kwargs.get("conditions"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(portlearn.experiments.Experiment, "__init__", _wrapped)


class _ZeroRateProportional(portlearn.costs.Proportional):
    """The M4 mutation seam: constructs the frozen model at rate ``0.0``."""

    def __init__(self, rate: object, turnover: object) -> None:
        super().__init__(0.0, turnover)


def _run_snapshot_replay(monkeypatch: pytest.MonkeyPatch, path: Path) -> str:
    """Execute the example as ``__main__`` via the documented snapshot entry."""
    monkeypatch.setattr(
        sys, "argv", ["classical_research_smoke.py", "--snapshot", str(path)]
    )
    with redirect_stdout(io.StringIO()) as buffer:
        runpy.run_path(str(EXAMPLE), run_name="__main__")
    return buffer.getvalue()


def _pinned_content_sha256() -> str:
    """The committed source manifest's pinned 64-hex ``content_sha256``."""

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "content_sha256":
                    return value
                found = _walk(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = _walk(item)
                if found is not None:
                    return found
        return None

    assert SOURCES_MANIFEST.is_file(), (
        f"the committed source manifest is absent: {SOURCES_MANIFEST.relative_to(REPO)}"
    )
    pinned = _walk(json.loads(SOURCES_MANIFEST.read_text()))
    assert isinstance(pinned, str) and re.fullmatch(r"[0-9a-f]{64}", pinned), (
        "the committed source manifest does not pin a 64-hex content_sha256"
    )
    return pinned


def _arm_lines(output: str, arm: str) -> list[str]:
    return [line for line in output.splitlines() if arm in line]


def _arm_ledger_digests(output: str, arm: str) -> list[str]:
    """Per-arm ledger digests as stamped by the run facts section."""
    digests = []
    for line in _arm_lines(output, arm):
        if "ledger_digest" in line:
            match = _DIGEST_LINE.search(line)
            if match is not None:
                digests.append(match.group("d"))
    return digests


def _arm_period_count(output: str, arm: str) -> int | None:
    for line in _arm_lines(output, arm):
        if "n_periods" in line:
            match = re.search(r"n_periods\D*(\d+)", line)
            if match is not None:
                return int(match.group(1))
    return None


def _arm_metric_values(output: str, arm: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in METRIC_NAMES:
        for line in _arm_lines(output, arm):
            match = re.search(rf"{name}\D*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", line)
            if match is not None:
                values[name] = float(match.group(1))
                break
    return values


def _split_report(output: str) -> tuple[str, str | None, str]:
    """Byte-exact report splitter (classical-research section 10).

    Returns ``(deterministic_prefix, optional_created_at_line_or_None,
    marker)`` carved from the exact ``output`` bytes: ``marker`` is the
    final line and must be exactly ``DIGEST_MARKER sha256=<64hex>``;
    the deterministic prefix is every byte before the marker, less the
    one permitted volatile ``created_at`` footer line, which, when
    present, must be the sole line between the prefix and the marker.
    The prefix itself must close on the boundary fence; no other
    volatile or tail line is admitted.
    """

    lines = output.splitlines()
    assert lines, "the report is empty"
    marker = lines[-1]
    assert _MARKER_LINE.fullmatch(marker) is not None, (
        "the report must end with the exact digest marker line "
        f"'DIGEST_MARKER sha256=<64hex>'; got {marker!r}"
    )
    body = lines[:-1]
    assert body, "the report carries no content before the digest marker"
    created_at_line: str | None = None
    if _CREATED_AT_LINE.fullmatch(body[-1]):
        created_at_line = body[-1]
        body = body[:-1]
    assert body and _is_fence_line(body[-1]), (
        "the deterministic prefix must close on the boundary fence, with "
        "at most the single permitted created_at line beyond it"
    )
    prefix = "".join(f"{line}\n" for line in body)
    assert "created_at" not in prefix, (
        "a volatile created_at fact leaked into the deterministic prefix"
    )
    return prefix, created_at_line, marker


def _require_comparison_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    """F1 deferred leg, now closed on the public adapter seam alone.

    On a separate actual ``--fixture`` CLI execution the public
    ``portlearn.research.run_declared_experiment`` is instrumented
    call-through: the real function executes unchanged, then returns a
    type-correct ``ExperimentResults`` carrying one declared arm fewer
    (a dataclass replacement, so every surviving record and the record
    type are exactly the real ones) while the returned provenance
    mapping retains every arm.  The frozen readiness chain must then
    reject by name — ``readiness`` in the message of a
    ``ValueError``/``ExecutionError`` — before any comparison block is
    rendered or any report completes: no target fabrication, no
    source-text reads, no private globals.
    """

    _require_example()
    original = portlearn.research.run_declared_experiment

    def _dropped_arm(*args: Any, **kwargs: Any) -> tuple[Any, Mapping[str, Any]]:
        results, provenances = original(*args, **kwargs)
        dropped = next(
            (record for record in results.records if record.strategy_id != ARMS[0]),
            None,
        )
        assert dropped is not None, (
            "the readiness probe could not isolate one arm to drop from "
            f"the real result records {tuple(r.strategy_id for r in results.records)}"
        )
        return (
            dataclasses.replace(results, records=(dropped,)),
            provenances,
        )

    with pytest.MonkeyPatch.context() as mp:
        _refuse_network(mp)
        mp.setattr(portlearn.research, "run_declared_experiment", _dropped_arm)
        mp.setattr(sys, "argv", ["classical_research_smoke.py", "--fixture"])
        with (
            redirect_stdout(io.StringIO()) as out,
            redirect_stderr(io.StringIO()) as err,
            pytest.raises(ValueError, match="readiness") as excinfo,
        ):
            runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert type(excinfo.value) is not AssertionError, (
        "the comparator-drop rejection must be the frozen named "
        "readiness rejection, not a test-side sentinel"
    )
    assert "ComparisonError" not in type(excinfo.value).__name__, (
        "the comparator-drop rejection must fire in the frozen readiness "
        "chain before the comparison surface, not as a comparison error"
    )
    captured = out.getvalue() + err.getvalue()
    comparison_lines = [
        line for line in captured.splitlines() if line.startswith("[c]")
    ]
    assert not comparison_lines, (
        "a comparison block was rendered after a comparator was dropped "
        "from the run results"
    )
    stamped = _MARKER_LINE.search(captured)
    assert stamped is None, (
        "a completed digest-stamped report was produced after a "
        "comparator was dropped from the run results"
    )


def test_offline_replay_executes_the_documented_entry_point_with_the_network_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _refuse_network(monkeypatch)
    output = _run_fixture_replay(monkeypatch)
    lines = output.splitlines()
    for tag in ("[s]", "[p]", "[d]", "[r]", "[m]", "[c]", "[f]"):
        assert any(line.startswith(tag) for line in lines), (
            f"report is missing the {tag} section of the public path"
        )
    assert len(_fence_lines(output)) >= 2, (
        "the boundary fence must open and close the report"
    )
    prefix, _created_at, marker = _split_report(output)
    stamped = _MARKER_LINE.fullmatch(marker)
    assert stamped is not None, "report must end digest-stamped"
    assert stamped.group("d") == hashlib.sha256(prefix.encode()).hexdigest(), (
        "the digest stamp does not authenticate the deterministic prefix"
    )
    assert output.count("ledger_digest") >= 2, (
        "per-arm ledger digests are missing from the run facts"
    )
    assert "final_wealth" in output, "per-arm final wealth is missing"

    # Comparator-drop readiness (classical-research F1/M10): a run result missing one
    # declared arm must surface the frozen named readiness rejection —
    # never a silently completed target-only report.
    _require_comparison_readiness(monkeypatch)


def test_bypassing_the_public_runtime_is_impossible() -> None:
    _require_example()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(portlearn.experiments.Experiment, "run", _sentinel_experiment_run)
        mp.setattr(sys, "argv", ["classical_research_smoke.py", "--fixture"])
        with (
            pytest.raises(_PublicPathSentinel, match=r"Experiment\.run"),
            redirect_stdout(io.StringIO()),
        ):
            runpy.run_path(str(EXAMPLE), run_name="__main__")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            portlearn.research,
            "resolve_temporal_protocols",
            _sentinel_resolve,
        )
        mp.setattr(sys, "argv", ["classical_research_smoke.py", "--fixture"])
        with (
            pytest.raises(_PublicPathSentinel, match="resolve_temporal_protocols"),
            redirect_stdout(io.StringIO()),
        ):
            runpy.run_path(str(EXAMPLE), run_name="__main__")


def test_importing_the_example_is_side_effect_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_example()
    before = {path: _snapshot(path) for path in _WATCHED_DIRS}
    _refuse_network(monkeypatch)
    monkeypatch.setattr(
        portlearn.experiments.Experiment, "run", _sentinel_experiment_run
    )
    monkeypatch.setattr(
        portlearn.research, "resolve_temporal_protocols", _sentinel_resolve
    )
    with (
        redirect_stdout(io.StringIO()) as out,
        redirect_stderr(io.StringIO()) as err,
    ):
        runpy.run_path(str(EXAMPLE), run_name="classical_research_smoke_module")
    assert out.getvalue() == "", "importing the example printed to stdout"
    assert err.getvalue() == "", "importing the example printed to stderr"
    after = {path: _snapshot(path) for path in _WATCHED_DIRS}
    assert after == before, "importing the example disturbed the filesystem"


def test_fixture_mode_declares_itself_synthetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _refuse_network(monkeypatch)
    output = _run_fixture_replay(monkeypatch)
    assert "synthetic=yes" in output, (
        "fixture mode must declare synthetic=yes in the source block"
    )
    assert _fence_lines(output), "fixture mode must print the synthetic fence"
    committed = sorted(
        path.name
        for path in FF_FIXTURES.glob("*.zip")
        if path.name not in SHORT_HISTORY_FIXTURES
    )
    assert committed, (
        "the committed long synthetic fixture is absent from "
        f"{FF_FIXTURES.relative_to(REPO)}"
    )
    manifest = MANIFEST.read_text()
    for name in committed:
        rows = [line for line in manifest.splitlines() if name in line]
        assert rows, f"MANIFEST row missing for committed fixture {name}"
        assert any("synthetic" in row.lower() for row in rows), (
            f"the MANIFEST row for {name} does not self-label the replica synthetic"
        )
        with zipfile.ZipFile(FF_FIXTURES / name) as archive:
            blob = "\n".join(
                archive.read(member).decode("utf-8", errors="replace")
                for member in archive.namelist()
            )
        members = blob.splitlines()
        copyright_index = next(
            (
                index
                for index, line in enumerate(members)
                if "Copyright" in line
                and "Eugene F. Fama and Kenneth R. French" in line
            ),
            None,
        )
        assert copyright_index is not None, (
            f"fixture {name} does not preserve the provider copyright trailer"
        )
        window = "\n".join(members[copyright_index : copyright_index + 4]).lower()
        assert "synthetic" in window, (
            f"the copyright trailer of {name} is not followed by an "
            "explicit synthetic marker"
        )


def test_future_information_is_invisible_at_every_decision_instant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_example()
    _refuse_network(monkeypatch)
    decisions: list[tuple[str, Any, Any]] = []
    with pytest.MonkeyPatch.context() as mp:
        _spy_decisions(mp, decisions)
        _run_fixture_replay(mp)
    assert decisions, "no live strategy decisions were observed on the seam"
    by_class: dict[str, list[tuple[Any, Any]]] = {}
    for name, context, result in decisions:
        by_class.setdefault(name, []).append((context, result))
    assert set(by_class) == {"EqualWeight", "InverseVolatility"}, (
        "both declared arms must decide through the frozen strategy seam"
    )
    assert len(by_class["EqualWeight"]) == len(by_class["InverseVolatility"]), (
        "both arms must ride the same declared decision schedule"
    )
    for name, pairs in by_class.items():
        universes = {tuple(context.universe) for context, _ in pairs}
        assert len(universes) == 1, f"{name} decided over a non-constant universe"
    assert tuple(by_class["EqualWeight"][0][0].universe) == tuple(
        by_class["InverseVolatility"][0][0].universe
    ), "the two arms must consume one common point-in-time universe"
    for name, pairs in by_class.items():
        for context, _ in pairs:
            records = list(context.information)
            assert records, f"{name} decided at an instant with no admitted records"
            for record in records:
                assert record.available_time <= context.decision_time, (
                    f"{name} was supplied a record not yet available at "
                    "its decision instant"
                )
    first_instant = month_end_instant(2019, 7, ZONE)
    every_context = [context for pairs in by_class.values() for context, _ in pairs]
    assert min(context.decision_time for context in every_context) == first_instant, (
        "the first decision instant must be the 2019-07 month end in the zone"
    )
    first_contexts = [
        context for context in every_context if context.decision_time == first_instant
    ]
    for context in first_contexts:
        latest_period = max(record.observation_time for record in context.information)
        assert (latest_period.year, latest_period.month) == (2019, 5), (
            "at the first decision instant the latest visible period must be 2019-05"
        )
    inverse_first = [
        context
        for context, _ in by_class["InverseVolatility"]
        if context.decision_time == first_instant
    ]
    assert inverse_first, "the inverse-volatility arm did not decide first"
    assert len(list(inverse_first[0].information)) >= 36, (
        "the declared 36-observation inverse-volatility window requires "
        "at least 36 admitted records at the first instant"
    )
    probe = list(first_contexts[0].information)
    latest_available = max(record.available_time for record in probe)
    with pytest.raises(FutureInformationError):
        InformationSet(probe, as_of=latest_available - timedelta(microseconds=1))
    with pytest.raises(NaiveTimestampError):
        # The naive instant is exactly the probed defect.
        InformationSet(probe, as_of=datetime(2019, 7, 31))  # noqa: DTZ001


def test_costs_turnover_and_accounting_ride_the_recorded_paths() -> None:
    _require_example()
    conditions_seen: list[Any] = []
    decisions: list[tuple[str, Any, Any]] = []
    with pytest.MonkeyPatch.context() as mp:
        _refuse_network(mp)
        _spy_conditions(mp, conditions_seen)
        _spy_decisions(mp, decisions)
        baseline = _run_fixture_replay(mp)
    assert len(conditions_seen) == 1, (
        "the declared design routes both arms through one common conditions object"
    )
    assert {name for name, _, _ in decisions} == {"EqualWeight", "InverseVolatility"}
    counts: dict[str, int] = {}
    for name, _, _ in decisions:
        counts[name] = counts.get(name, 0) + 1
    assert counts["EqualWeight"] == counts["InverseVolatility"], (
        "both arms must be accounted under the one recorded conditions object"
    )
    conditions = conditions_seen[0]
    assert isinstance(
        conditions.accounting_engine, portlearn.ledger.ExactFillAccounting
    ), "the declared accounting engine must be the frozen ExactFillAccounting"
    cost_model = conditions.cost_model
    assert isinstance(cost_model, portlearn.costs.Proportional), (
        "the declared cost model must ride the frozen public Proportional surface"
    )
    assert cost_model.rate == 0.0010, "the declared cost rate must be 0.0010"
    assert cost_model.convention == "one_way", "turnover must be declared one_way"
    opening = dict(conditions.initial_weights)
    assert opening, "the declared opening book is absent from the conditions wire"
    first_inverse = min(
        (
            (context, result)
            for name, context, result in decisions
            if name == "InverseVolatility"
        ),
        key=lambda pair: pair[0].decision_time,
    )
    first_equal = min(
        (
            (context, result)
            for name, context, result in decisions
            if name == "EqualWeight"
        ),
        key=lambda pair: pair[0].decision_time,
    )
    declared_cost = portlearn.costs.Proportional(rate=0.0010, turnover=portlearn.turnover.one_way)
    inverse_target = dict(first_inverse[1].decision.target_weights)
    equal_target = dict(first_equal[1].decision.target_weights)
    assert declared_cost.estimate_trade_cost(opening, inverse_target) > 0.0, (
        "the inverse-volatility first rebalance must carry strictly "
        "positive turnover against the opening 1/N book"
    )
    assert declared_cost.estimate_trade_cost(opening, equal_target) == pytest.approx(
        0.0, abs=1e-12
    ), "the equal-weight opening book rebalances to itself at zero turnover"

    # M4 mutation kill through the narrow zero-rate constructor seam.  The
    # frozen report surface declares no gross-basis MetricsBlock, so the
    # classical-research section-15 authorized narrowing applies: the kill is asserted on
    # the net ledger digest plus conditions-wire visibility.
    mutated_conditions: list[Any] = []
    with pytest.MonkeyPatch.context() as mp:
        _refuse_network(mp)
        mp.setattr(portlearn.costs, "Proportional", _ZeroRateProportional)
        _spy_conditions(mp, mutated_conditions)
        mp.setattr(sys, "argv", ["classical_research_smoke.py", "--fixture"])
        with redirect_stdout(io.StringIO()) as buffer:
            runpy.run_path(str(EXAMPLE), run_name="__main__")
        mutated = buffer.getvalue()
    assert len(mutated_conditions) == 1
    mutated_wire = mutated_conditions[0]
    assert isinstance(
        mutated_wire.accounting_engine, portlearn.ledger.ExactFillAccounting
    ), "the zero-rate mutation must leave the accounting engine in place"
    assert mutated_wire.cost_model.rate == 0.0, (
        "the zero-rate mutation did not ride the recorded conditions wire"
    )
    assert mutated_wire.cost_model.convention == "one_way"
    baseline_digests = _arm_ledger_digests(baseline, "inverse_volatility")
    mutated_digests = _arm_ledger_digests(mutated, "inverse_volatility")
    assert len(baseline_digests) == 1, (
        "the inverse-volatility arm must stamp exactly one net ledger digest"
    )
    assert len(mutated_digests) == 1
    assert mutated_digests[0] != baseline_digests[0], (
        "zeroing the cost rate left the net ledger digest unchanged"
    )


def test_snapshot_hash_mismatch_rejects_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _require_example()
    _refuse_network(monkeypatch)
    pinned = _pinned_content_sha256()
    pristine = (FF_FIXTURES / "ff_industry49_monthly_csv.zip").read_bytes()
    corrupted = pristine[:-1] + bytes([pristine[-1] ^ 0xFF])
    for label, blob in (
        ("one_corrupted_byte", corrupted),
        ("fixture_stand_in", pristine),
    ):
        snapshot = tmp_path / f"snapshot_{label}.zip"
        snapshot.write_bytes(blob)
        observed = hashlib.sha256(blob).hexdigest()
        assert observed != pinned, "the probe bytes accidentally match the pinned hash"
        # Zero decisions executed: the actual documented CLI path driven
        # in-process with the strategy seams instrumented; the catch is
        # exact (a snapshot identity mismatch is a declared-value
        # rejection, so ValueError) and anchored to the observed hash, so
        # a rejection that neither raises nor names the mismatched bytes
        # fails the floor.  The child-process leg is deliberately absent:
        # a separate process would escape this process's network
        # refusal, so the offline contract is only provable in-process.
        decisions: list[tuple[str, Any, Any]] = []
        with pytest.MonkeyPatch.context() as mp:
            _refuse_network(mp)
            _spy_decisions(mp, decisions)
            mp.setattr(
                sys,
                "argv",
                ["classical_research_smoke.py", "--snapshot", str(snapshot)],
            )
            with (
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
                pytest.raises(ValueError, match=observed) as excinfo,
            ):
                runpy.run_path(str(EXAMPLE), run_name="__main__")
        assert pinned in str(excinfo.value), (
            f"the {label} rejection must name the manifest-pinned SHA-256"
        )
        assert decisions == [], (
            f"strategy decisions executed before the {label} snapshot rejection"
        )


def test_identical_bytes_produce_deterministic_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_example()
    _refuse_network(monkeypatch)
    first = _run_fixture_replay(monkeypatch)
    second = _run_fixture_replay(monkeypatch)
    for arm in ARMS:
        first_digests = _arm_ledger_digests(first, arm)
        second_digests = _arm_ledger_digests(second, arm)
        assert len(first_digests) == 1, f"{arm} must stamp exactly one ledger digest"
        assert first_digests == second_digests, (
            f"the {arm} ledger digest is not deterministic over identical bytes"
        )
        first_count = _arm_period_count(first, arm)
        second_count = _arm_period_count(second, arm)
        assert first_count is not None and first_count == second_count, (
            f"the {arm} period count is missing or non-deterministic"
        )
        first_metrics = _arm_metric_values(first, arm)
        second_metrics = _arm_metric_values(second, arm)
        assert set(first_metrics) == set(METRIC_NAMES), (
            f"{arm} must restate all six frozen metric values"
        )
        assert first_metrics == second_metrics, (
            f"the {arm} metrics block is not deterministic over identical bytes"
        )
    assert "2019-07" in first and "2024-06" in first, (
        "the declared evaluation span bounds must appear in the report"
    )
    first_prefix, _first_created_at, first_marker = _split_report(first)
    second_prefix, _second_created_at, second_marker = _split_report(second)
    assert first_prefix == second_prefix, (
        "the deterministic report prefix is not identical across runs"
    )
    assert first_marker == second_marker, "the DIGEST_MARKER line is not identical"
    stamped = _DIGEST_LINE.search(first_marker)
    assert stamped is not None
    assert stamped.group("d") == hashlib.sha256(first_prefix.encode()).hexdigest(), (
        "the DIGEST_MARKER does not authenticate the deterministic prefix"
    )

    # Real-snapshot leg, transparent and substitution-free: exercised only
    # when the exact manifest-pinned bytes are locally possessed.  The
    # absent case is asserted explicitly and makes no real-data claim.
    pinned = _pinned_content_sha256()
    possessed = REAL_SNAPSHOT.is_file() and (
        hashlib.sha256(REAL_SNAPSHOT.read_bytes()).hexdigest() == pinned
    )
    if possessed:
        real_first = _run_snapshot_replay(monkeypatch, REAL_SNAPSHOT)
        real_second = _run_snapshot_replay(monkeypatch, REAL_SNAPSHOT)
        for arm in ARMS:
            assert _arm_ledger_digests(real_first, arm) == _arm_ledger_digests(
                real_second, arm
            ), f"the {arm} real-snapshot replay is not deterministic"
        real_first_prefix, _real_first_created_at, real_first_marker = _split_report(
            real_first
        )
        real_second_prefix, _real_second_created_at, real_second_marker = _split_report(
            real_second
        )
        assert real_first_prefix == real_second_prefix
        assert real_first_marker == real_second_marker
    else:
        assert not possessed, (
            "the manifest-pinned real snapshot is not locally possessed with "
            "a matching hash; this execution asserts fixture-mode determinism "
            "only and makes no real-data replay claim"
        )


# --------------------------------------------------------------------------- #
# Frozen floors F9-F11 (classical-research section 8, exact names and semantics)
# --------------------------------------------------------------------------- #


def test_real_adapter_decodes_genuine_retained_provider_bytes_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F9: genuine retained provider bytes decode via the adapter route.

    When the exact manifest-pinned retained snapshot is locally
    possessed, the actual ``--snapshot`` CLI replay decodes those bytes
    through the public ``fama_french.load`` adapter route with the
    network refused, the public asset-return fence admits the decoded
    portfolio-returns dataset, and the factor-kind rejection is
    demonstrated on the same public seam.  When the pinned bytes are
    absent, no real-data claim is made: the conditional custody
    boundary is asserted transparently, with no skip and no fixture
    substitution.
    """

    _require_example()
    _refuse_network(monkeypatch)
    pinned = _pinned_content_sha256()
    possessed = REAL_SNAPSHOT.is_file() and (
        hashlib.sha256(REAL_SNAPSHOT.read_bytes()).hexdigest() == pinned
    )
    if not possessed:
        assert not possessed, (
            "the manifest-pinned real snapshot is not locally possessed "
            "with a matching hash; this execution makes no real-data "
            "decode claim and asserts the custody boundary transparently"
        )
        return

    raw = REAL_SNAPSHOT.read_bytes()
    load_calls: list[tuple[dict[str, Any], Any]] = []
    original_load = fama_french.load

    def _recording_load(*args: Any, **kwargs: Any) -> Any:
        dataset = original_load(*args, **kwargs)
        load_calls.append((kwargs, dataset))
        return dataset

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fama_french, "load", _recording_load)
        output = _run_snapshot_replay(mp, REAL_SNAPSHOT)

    _split_report(output)  # the replay completed a digest-stamped report
    retained = [
        (kwargs, dataset) for kwargs, dataset in load_calls if kwargs.get("data") == raw
    ]
    assert retained, (
        "the replay never handed the exact retained provider bytes to the "
        "public fama_french.load adapter route"
    )
    assert all(kwargs.get("retrieval") is not None for kwargs, _ in retained), (
        "the adapter route decoded without hash-pinned retrieval facts"
    )
    dataset = retained[0][1]
    assert getattr(dataset, "records", ()), (
        "the retained provider bytes decoded to no period-key records"
    )
    provenance = dataset.retrieval_provenance
    assert provenance.dataset_kind == "portfolio_returns", (
        "the retained snapshot did not decode to a portfolio-returns "
        f"dataset (dataset_kind={provenance.dataset_kind!r})"
    )
    ff.require_candidate_asset_returns(provenance)  # admits; no exception
    factor_kind = dataclasses.replace(provenance, dataset_kind="factor")
    with pytest.raises(ValueError, match="factor"):
        ff.require_candidate_asset_returns(factor_kind)


def test_report_and_table_are_descriptive_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F10: the report and rendered table make no inferential claim.

    On an actual ``--fixture`` run, neither the rendered comparison
    table nor the report carries any of the frozen forbidden tokens;
    the table carries exactly the six frozen metrics in declared order
    with no derived contrast; and the example does not post-process
    the rendered table -- proven by call-through identity on the
    public ``render_table`` seam, never by reading example source.
    """

    _require_example()
    _refuse_network(monkeypatch)
    rendered_tables: list[str] = []
    original_render = portlearn.comparisons.render_table

    def _recording_render(table: Any) -> str:
        rendered = original_render(table)
        rendered_tables.append(rendered)
        return rendered

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(portlearn.comparisons, "render_table", _recording_render)
        output = _run_fixture_replay(mp)

    assert len(rendered_tables) == 1, (
        "the example must render exactly one comparison table through the "
        f"public render_table seam (saw {len(rendered_tables)})"
    )
    rendered = rendered_tables[0]
    assert rendered.rstrip("\n") in output, (
        "the report does not embed the render_table output verbatim; the "
        "example post-processes or re-renders the comparison table"
    )
    row_metrics = [
        match.group("metric")
        for match in re.finditer(r"^row (?P<metric>\S+)\b", rendered, re.MULTILINE)
    ]
    assert row_metrics == list(METRIC_NAMES), (
        "the rendered table rows are not exactly the six frozen metrics "
        f"in declared order: {row_metrics}"
    )
    output_rows = [
        match.group("metric")
        for match in re.finditer(r"^row (?P<metric>\S+)\b", output, re.MULTILINE)
    ]
    assert output_rows == list(METRIC_NAMES), (
        "the report carries table rows beyond the rendered six frozen "
        f"metrics in declared order: {output_rows}"
    )
    for token in FORBIDDEN_CLAIM_TOKENS:
        pattern = r"\b" + re.escape(token) + r"\b"
        assert re.search(pattern, output, re.IGNORECASE) is None, (
            f"the report carries the frozen forbidden token {token!r}"
        )
        assert re.search(pattern, rendered, re.IGNORECASE) is None, (
            f"the rendered table carries the frozen forbidden token {token!r}"
        )


def _refuse_network_recording(monkeypatch: pytest.MonkeyPatch, log: list[str]) -> None:
    """Refuse the network while recording every attempted touch.

    Unlike ``_refuse_network``, each attempted socket or ``urlopen``
    call is recorded before the refusal raises, so a caller that
    swallows the exception still cannot erase the evidence of the
    touch (mutation M8).
    """

    def _refuse(kind: str) -> Any:
        def _refusal(*args: Any, **kwargs: Any) -> Any:
            log.append(kind)
            raise _PublicPathSentinel(f"SENTINEL: network touched via {kind}")

        return _refusal

    monkeypatch.setattr(socket, "socket", _refuse("socket.socket"))
    monkeypatch.setattr(urllib.request, "urlopen", _refuse("urlopen"))


def _json_objects(text: str) -> list[dict[str, Any]]:
    """Every JSON object embedded in ``text`` (decode errors skipped)."""

    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def test_refresh_is_the_only_network_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """F11: the run path is offline; ``--refresh`` is the only network path.

    The plain run (or, without local custody of the retained bytes,
    the fixture run) completes offline with the network refused and
    every attempted touch recorded.  The documented ``--refresh`` CLI
    path then calls the public fetch exactly once over a synthetic
    mocked return payload, writes the selected snapshot, prints the
    derived manifest JSON, and runs no experiment -- and it cannot
    write the fixed repository scratch location.
    """

    _require_example()
    pinned = _pinned_content_sha256()
    possessed = REAL_SNAPSHOT.is_file() and (
        hashlib.sha256(REAL_SNAPSHOT.read_bytes()).hexdigest() == pinned
    )
    run_log: list[str] = []
    with pytest.MonkeyPatch.context() as mp:
        _refuse_network_recording(mp, run_log)
        if possessed:
            mp.setattr(sys, "argv", ["classical_research_smoke.py"])
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                runpy.run_path(str(EXAMPLE), run_name="__main__")
        else:
            mp.setattr(sys, "argv", ["classical_research_smoke.py", "--fixture"])
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                runpy.run_path(str(EXAMPLE), run_name="__main__")
        _split_report(buffer.getvalue())
    assert run_log == [], f"the run path attempted a network touch: {run_log}"

    # The refresh leg: a synthetic public-fetch return payload is the
    # mocked input, never real evidence.
    payload = (FF_FIXTURES / "ff_industry49_monthly_csv.zip").read_bytes()
    entry = ff.FF_DATASETS["industry49_monthly_csv"]
    synthetic_retrieval = ff.FFRetrievalProvenance(
        source_id=f"ff:{entry.dataset_id}",
        content_sha256=hashlib.sha256(payload).hexdigest(),
        units=entry.units,
        frequency=entry.frequency,
        url=entry.url,
        dataset_kind=entry.dataset_kind,
        dataset_id=entry.dataset_id,
    )
    fetch_calls: list[tuple[Any, ...]] = []

    def _mocked_fetch_raw(*args: Any, **kwargs: Any) -> Any:
        fetch_calls.append(tuple(args) + tuple(kwargs.items()))
        return payload, synthetic_retrieval

    decisions: list[tuple[str, Any, Any]] = []
    target = tmp_path / "industry49_monthly_csv.zip"
    scratch_before = REAL_SNAPSHOT.is_file()
    with pytest.MonkeyPatch.context() as mp:
        _refuse_network_recording(mp, run_log)
        _spy_decisions(mp, decisions)
        mp.setattr(ff, "fetch_raw", _mocked_fetch_raw)
        mp.setattr(portlearn.experiments.Experiment, "run", _sentinel_experiment_run)
        mp.setattr(
            sys,
            "argv",
            ["classical_research_smoke.py", "--refresh", "--snapshot", str(target)],
        )
        buffer = io.StringIO()
        with redirect_stdout(buffer), suppress(SystemExit):
            runpy.run_path(str(EXAMPLE), run_name="__main__")
        refresh_output = buffer.getvalue()

    assert len(fetch_calls) == 1, (
        f"the refresh path must call the public fetch exactly once "
        f"(saw {len(fetch_calls)})"
    )
    assert fetch_calls[0][0] == "industry49_monthly_csv", (
        "the refresh fetch did not request the declared dataset"
    )
    assert target.is_file() and target.read_bytes() == payload, (
        "the refresh path did not write the fetched bytes to the "
        "selected snapshot location"
    )
    manifests = [
        obj
        for obj in _json_objects(refresh_output)
        if obj.get("dataset_id") == "industry49_monthly_csv"
    ]
    assert manifests, (
        "the refresh path printed no derived manifest JSON for the declared dataset"
    )
    assert any(
        obj.get("content_sha256") == hashlib.sha256(payload).hexdigest()
        for obj in manifests
    ), "the printed manifest JSON does not derive from the fetched bytes"
    assert decisions == [], "the refresh path executed strategy decisions"
    assert REAL_SNAPSHOT.is_file() == scratch_before, (
        "the refresh path wrote the fixed repository scratch location "
        "instead of the explicit snapshot override"
    )
