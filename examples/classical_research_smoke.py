"""Classical research smoke: an instructional end-to-end PortLearn replay.

This example composes PortLearn's complete public research runtime over
monthly industry-portfolio series, entirely offline, as an
instructional framework-operation smoke only. It is a research
dataset replay; it is NOT investable, and no performance, superiority,
or empirical claim is made or possible here.

CLI modes:

* default -- replay the reference run snapshot pinned by the committed
  ``examples/classical_research_sources.json`` manifest, failing closed
  with refresh instructions when the manifest or the pinned bytes are
  absent (transparent conditional custody: no traceback spew, no
  silent substitution).
* ``--fixture`` -- replay the committed synthetic long fixture
  ``tests/adapters/fixtures/ff/ff_industry49_monthly_long_csv.zip``;
  the source block then declares ``synthetic=yes``.
* ``--snapshot PATH`` -- replay an explicitly selected snapshot file,
  verified against the manifest-pinned hash before any decision.
* ``--refresh`` -- the only network path: fetch the declared dataset
  once through the public adapter, write the snapshot bytes, print the
  derived manifest JSON, and exit without running any experiment.

Importing this module is side-effect free: every effect lives under
``if __name__ == "__main__":``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import portlearn
import portlearn.comparisons
import portlearn.costs
import portlearn.experiments
import portlearn.ledger
import portlearn.protocols
import portlearn.research
import portlearn.strategies
import portlearn.turnover
from portlearn.calendar import month_end_instant
from portlearn.data import fama_french
from portlearn.data.adapters import ff
from portlearn.data.ingestion import AvailabilityPolicy
from portlearn.interfaces import InformationSet

# --------------------------------------------------------------------------- #
# Frozen declaration facts (classical-research sections 4-7). #
# --------------------------------------------------------------------------- #

FENCE = (
    "instructional framework-operation smoke only; research datasets, "
    "NOT investable; no performance, superiority, or empirical claim "
    "is made or possible here"
)

ZONE = ZoneInfo("Australia/Melbourne")
AVAILABILITY_LAG = timedelta(days=45)
AVAILABILITY_JUSTIFICATION = (
    "45-day monthly publication lag declared for the industry portfolios"
)
EVALUATION_SPAN = "2019-07-01..2024-06-30"
ESTIMAND = "monthly_net_of_cost_portfolio_return"
TARGET_HORIZON = timedelta(days=31)
STRATEGY_IDS = ("equal_weight", "inverse_volatility")
TARGET = "inverse_volatility"
COMPARATOR = "equal_weight"
METRIC_NAMES = (
    "cumulative_return",
    "annualized_mean_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "certainty_equivalent_return",
)
CONDITIONS = {
    "periods_per_year": 12,
    "risk_free": 0.0,
}
COST_RATE = 0.0010
INVOL_WINDOW = 36
DATASET_KEY = "industry49_monthly_csv"
ALIAS_KEY = "industry49"
FIRST_MONTH = (2019, 7)
DECISION_COUNT = 60

REPO = Path(__file__).resolve().parents[1]
SOURCES_MANIFEST_PATH = REPO / "examples" / "classical_research_sources.json"
SCRATCH_DIR = REPO / "scratch" / "classical_research_snapshot"
SCRATCH_SNAPSHOT = SCRATCH_DIR / f"{DATASET_KEY}.zip"
FIXTURE = (
    REPO
    / "tests"
    / "adapters"
    / "fixtures"
    / "ff"
    / "ff_industry49_monthly_long_csv.zip"
)

REFRESH_INSTRUCTIONS = f"""\
The reference run snapshot is not available for offline replay.

Transparent conditional custody: PortLearn does not redistribute the
provider bytes, and the committed source manifest
examples/classical_research_sources.json or the pinned local snapshot
{SCRATCH_SNAPSHOT.relative_to(REPO)} is absent, so no real-data replay
claim is made and nothing is substituted for the pinned bytes.

To produce them, run the refresh path exactly once:

    python examples/classical_research_smoke.py --refresh

It fetches the declared dataset once through the public adapter,
writes the snapshot to scratch/classical_research_snapshot/, and
prints the derived manifest JSON pinning the retained bytes
(content_sha256). Commit examples/classical_research_sources.json
with that hash and keep the snapshot bytes locally; the default replay
then verifies the pinned hash against the observed bytes before any
decision executes.

Alternatively, replay the committed synthetic replica offline:

    python examples/classical_research_smoke.py --fixture
"""


def _fail_closed(message: str) -> None:
    sys.stdout.flush()
    print(message, file=sys.stderr)
    raise SystemExit(2)


def _sha256_hex(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _next_month(key: tuple[int, int]) -> tuple[int, int]:
    year, month = key
    month += 1
    if month == 13:
        year, month = year + 1, 1
    return year, month


def _month_keys(first: tuple[int, int], count: int) -> list[tuple[int, int]]:
    keys: list[tuple[int, int]] = []
    key = first
    for _ in range(count):
        keys.append(key)
        key = _next_month(key)
    return keys


# --------------------------------------------------------------------------- #
# Data chain (the public declared-entry adapter route). #
# --------------------------------------------------------------------------- #


def _retrieval_provenance(raw: bytes) -> ff.FFRetrievalProvenance:
    entry = ff.FF_DATASETS[DATASET_KEY]
    return ff.FFRetrievalProvenance(
        source_id=f"ff:{entry.dataset_id}",
        content_sha256=_sha256_hex(raw),
        units=entry.units,
        frequency=entry.frequency,
        url=entry.url,
        dataset_kind=entry.dataset_kind,
        dataset_id=entry.dataset_id,
    )


def _load_qualified(raw: bytes) -> Any:
    return fama_french.load(
        ALIAS_KEY,
        frequency="monthly",
        data=raw,
        retrieval=_retrieval_provenance(raw),
        availability=AvailabilityPolicy.fixed_lag(
            AVAILABILITY_LAG, justification=AVAILABILITY_JUSTIFICATION
        ),
        tzinfo=ZONE,
    )


def _pinned_content_sha256() -> str:
    """The committed manifest's pinned hash; absent manifest rejects."""
    if not SOURCES_MANIFEST_PATH.is_file():
        raise ValueError(
            "snapshot identity cannot be verified: the committed source "
            "manifest examples/classical_research_sources.json is absent; "
            "run --refresh to derive it from the retained provider bytes"
        )
    document = json.loads(SOURCES_MANIFEST_PATH.read_text())

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

    pinned = _walk(document)
    if not isinstance(pinned, str) or len(pinned) != 64:
        raise ValueError(
            "the committed source manifest does not pin a 64-hex "
            "content_sha256"
        )
    return pinned


def _verify_against_pin(raw: bytes) -> str:
    """Observed-vs-pinned identity check; mismatch rejects closed."""
    pinned = _pinned_content_sha256()
    observed = _sha256_hex(raw)
    if observed != pinned:
        raise ValueError(
            "snapshot identity mismatch: the committed source manifest "
            f"pins content_sha256 {pinned} but the selected snapshot "
            f"observes content_sha256 {observed}; no other bytes may "
            "stand in for the manifest-pinned retained provider bytes, "
            "so the replay rejects before any decision executes"
        )
    return pinned


# --------------------------------------------------------------------------- #
# Researcher-side backends (point-in-time information, outcomes). #
# --------------------------------------------------------------------------- #


def _information_backend(
    records: tuple[Any, ...]
) -> Callable[[Any], InformationSet]:
    def _provide(decision_instant: Any) -> InformationSet:
        return InformationSet(
            items=[
                record
                for record in records
                if record.available_time <= decision_instant
            ],
            as_of=decision_instant,
        )

    return _provide


def _outcome_backend(
    observations: Mapping[tuple[int, int], Mapping[str, float]],
    month_keys_by_instant: Mapping[Any, tuple[int, int]],
) -> Callable[[Any], Any]:
    def _provide(segment_start: Any) -> Any:
        key = month_keys_by_instant.get(segment_start)
        if key is None:
            return None
        row = observations.get(_next_month(key))
        if row is None:
            return None
        return {asset: 1.0 + value / 100.0 for asset, value in row.items()}

    return _provide


def _fit_event_count(provenance: Any) -> int:
    protocol = getattr(provenance, "protocol", provenance)
    events = getattr(protocol, "fit_events", None)
    return len(tuple(events)) if events is not None else 0


# --------------------------------------------------------------------------- #
# The replay itself: one complete public-path run and its report. #
# --------------------------------------------------------------------------- #


class _AlwaysRebalance:
    """The declared always-rebalance policy (classical-research section 6)."""

    def should_rebalance(self, date: Any, last_rebalance: Any) -> bool:
        return True


def _exact_unit_book(book: Mapping[str, float]) -> dict[str, float]:
    """Restore the exact unit budget by minimal residual correction.

    A lawful float target book can sit one ulp below one (the classic
    ``0.999...999``), and the unit-NAV ledger's exact-unit-budget law
    rejects it fail-closed. The correction is deterministic and
    minimal: the whole residual lands on the first sorted asset so
    ``math.fsum`` returns exactly ``1.0`` while every other weight
    keeps its original bits. This is not a normalization -- no weight
    is rescaled, clipped, or otherwise materially adjusted.
    """
    corrected = dict(book)
    keys = sorted(corrected)
    residual = 1.0 - math.fsum(corrected[key] for key in keys)
    if residual:
        corrected[keys[0]] += residual
    if math.fsum(corrected.values()) != 1.0:
        raise ValueError(
            "the target book cannot be corrected to the exact unit budget"
        )
    return corrected


class _ExactBudgetStrategy:
    """A call-through adapter enforcing the ledger's exact unit budget.

    The wrapped frozen strategy instance always decides first, so
    class-level ``decide`` instrumentation on the frozen strategy
    classes still fires with the underlying instance. Its result is
    re-issued with every field preserved exactly, except that
    ``decision.target_weights`` is corrected to the exact unit budget
    the ledger demands. Public strategy behavior is never bypassed.
    """

    def __init__(self, strategy: Any) -> None:
        self._strategy = strategy

    def decide(self, context: Any) -> Any:
        result = self._strategy.decide(context)
        return dataclasses.replace(
            result,
            decision=dataclasses.replace(
                result.decision,
                target_weights=_exact_unit_book(result.decision.target_weights),
            ),
        )


def _replay(raw: bytes, *, synthetic: bool, pinned: str | None) -> None:
    qualified = _load_qualified(raw)
    provenance = qualified.qualified_provenance
    records = tuple(qualified.records)
    if not records:
        raise ValueError("the decoded dataset carries no records")
    assets = tuple(sorted({record.series_id for record in records}))

    observations: dict[tuple[int, int], dict[str, float]] = {}
    for record in records:
        key = (record.observation_time.year, record.observation_time.month)
        observations.setdefault(key, {})[record.series_id] = record.value

    keys = _month_keys(FIRST_MONTH, DECISION_COUNT)
    decision_dates = [month_end_instant(year, month, ZONE) for year, month in keys]
    month_keys_by_instant = dict(zip(decision_dates, keys))

    identity_map = {asset: asset for asset in assets}
    registrations = (
        portlearn.experiments.StrategyRegistration(
            strategy_id=TARGET,
            strategy=_ExactBudgetStrategy(
                portlearn.strategies.InverseVolatility(
                    identity_map, window=INVOL_WINDOW
                )
            ),
        ),
        portlearn.experiments.StrategyRegistration(
            strategy_id=COMPARATOR,
            strategy=_ExactBudgetStrategy(portlearn.strategies.EqualWeight()),
        ),
    )
    schedule = portlearn.experiments.DecisionSchedule(
        decision_dates=decision_dates,
        universes={instant: assets for instant in decision_dates},
    )

    opening = {asset: 1.0 / len(assets) for asset in assets}
    if abs(math.fsum(opening.values()) - 1.0) > 1e-9:
        raise ValueError("the opening 1/N book does not sum to one")
    # An exact unit budget: the naive 1/N float book can drift below 1.0
    # (0.999...999), and the ledger rejects non-unit books fail-closed.
    # The largest-remainder construction keeps the book exactly 1/N to
    # the last representable bit of budget while restoring the exact
    # unit budget the unit-NAV wealth path requires.
    keys = sorted(opening)
    epsilon = (1.0 - math.fsum(opening[k] for k in keys)) / len(keys)
    if epsilon:
        for asset in keys:
            opening[asset] += epsilon
    residual = 1.0 - math.fsum(opening.values())
    if residual:
        opening[keys[0]] += residual
    conditions = portlearn.experiments.ExperimentConditions(
        information_provider=_information_backend(records),
        outcome_provider=_outcome_backend(observations, month_keys_by_instant),
        rebalance_policy=_AlwaysRebalance(),
        accounting_engine=portlearn.ledger.ExactFillAccounting(),
        cost_model=portlearn.costs.Proportional(
            COST_RATE, turnover=portlearn.turnover.one_way
        ),
        initial_weights=opening,
        initial_wealth=1.0,
        initial_last_rebalance_time=month_end_instant(2019, 6, ZONE),
    )
    configuration = portlearn.experiments.ExperimentConfiguration(
        strategy_ids=STRATEGY_IDS,
        target=TARGET,
        comparators=(COMPARATOR,),
        ablations=(),
        robustness_checks=(),
        benchmarks=(COMPARATOR,),
        conditions=dict(CONDITIONS),
    )
    none_declaration = portlearn.protocols.StrategyTemporalDeclaration(
        estimation=portlearn.protocols.none,
        selection=portlearn.protocols.none,
        refit=portlearn.protocols.none,
        preprocessing=portlearn.protocols.none,
    )
    temporal = portlearn.protocols.TemporalDesign(
        evaluation_span=EVALUATION_SPAN,
        evaluation_learning=portlearn.protocols.EvaluationLearning.WALK_FORWARD,
        estimand=ESTIMAND,
        target_horizon=TARGET_HORIZON,
        timezone=ZONE,
        strategies=(
            (COMPARATOR, none_declaration),
            (TARGET, none_declaration),
        ),
    )

    results, provenances = portlearn.research.run_declared_experiment(
        configuration=configuration,
        registrations=registrations,
        schedule=schedule,
        conditions=conditions,
        temporal=temporal,
        record_instants_by_strategy={
            strategy_id: tuple(decision_dates) for strategy_id in STRATEGY_IDS
        },
        fit_provenances_by_strategy={
            strategy_id: () for strategy_id in STRATEGY_IDS
        },
    )
    records_by_id = {record.strategy_id: record for record in results.records}

    declared = set(STRATEGY_IDS)
    present = set(records_by_id)
    dropped = sorted(declared - present)
    if dropped:
        raise ValueError(
            "comparison readiness fails closed: the executed run "
            f"returned no record for the declared strategies {dropped}; a "
            "comparison without every declared strategy is not ready and no "
            "comparison table may proceed"
        )

    lines: list[str] = [FENCE]
    lines.append(
        f"[s] source dataset_id={provenance.dataset_id} "
        f"member_name={ff.FF_DATASETS[DATASET_KEY].member_name} "
        f"dataset_kind={provenance.dataset_kind} "
        f"units={provenance.units} frequency={provenance.frequency}"
    )
    lines.append(
        f"[s] url={provenance.url} adapter={provenance.adapter_identity} "
        f"adapter_version={provenance.adapter_version} "
        f"alias_catalog_version={fama_french.ALIAS_CATALOG_VERSION}"
    )
    observed = _sha256_hex(raw)
    pin_note = "none" if pinned is None else pinned
    synthetic_note = "yes" if synthetic else "no"
    lines.append(
        f"[s] content_sha256={observed} pinned={pin_note} "
        f"retrieval_instant={provenance.retrieval_instant.isoformat()} "
        f"last_modified="
        f"{provenance.last_modified.isoformat() if provenance.last_modified else 'none'} "
        f"synthetic={synthetic_note}"
    )
    lines.append(f"[s] licence_note {provenance.licence_note}")
    lines.append(f"[s] not_investable_warning {provenance.not_investable_warning}")
    lines.append(
        f"[p] protocol zone={ZONE.key} "
        f"availability_lag={AVAILABILITY_LAG.days}d "
        f"justification=\"{AVAILABILITY_JUSTIFICATION}\" "
        f"evaluation_span={EVALUATION_SPAN} "
        f"evaluation_learning={temporal.evaluation_learning.value} "
        f"estimand={ESTIMAND} target_horizon={TARGET_HORIZON.days}d"
    )
    for strategy_id in STRATEGY_IDS:
        count = _fit_event_count(provenances.get(strategy_id))
        lines.append(
            f"[p] arm {strategy_id} estimation=none selection=none "
            f"refit=none preprocessing=none fit_events={count}"
        )
    lines.append(
        f"[d] design universe_n={len(assets)} "
        f"decisions={len(decision_dates)} "
        f"first={decision_dates[0].isoformat()} "
        f"last={decision_dates[-1].isoformat()} rebalance=always "
        f"accounting=ExactFillAccounting cost=Proportional "
        f"rate={COST_RATE:.4f} convention=one_way opening=1/N "
        f"wealth=1.0 last_rebalance=2019-06 month-end "
        f"conditions={json.dumps(CONDITIONS, sort_keys=True)}"
    )
    lines.append(
        f"[r] run portlearn={portlearn.__version__} "
        f"python={sys.version_info.major}.{sys.version_info.minor}."
        f"{sys.version_info.micro} "
        f"command={' '.join(sys.argv[1:]) or '(default)'}"
    )
    for strategy_id in STRATEGY_IDS:
        record = records_by_id[strategy_id]
        path = record.paths["main"]
        lines.append(
            f"[r] arm {strategy_id} path=main ledger_digest "
            f"sha256={path.ledger_digest} n_periods={path.n_periods} "
            f"period_start={path.period_start.isoformat()} "
            f"period_end={path.period_end.isoformat()} "
            f"final_wealth={path.final_wealth!r}"
        )
    for strategy_id in STRATEGY_IDS:
        block = records_by_id[strategy_id].paths["main"].metrics[0]
        facts = " ".join(
            f"{metric}={getattr(block, metric)!r}" for metric in METRIC_NAMES
        )
        lines.append(
            f"[m] metrics {strategy_id} basis={block.basis} "
            f"periods_per_year={block.periods_per_year} "
            f"gamma={block.gamma!r} risk_free={block.risk_free!r} {facts}"
        )

    spec = portlearn.comparisons.ComparisonSpec(
        configuration,
        results,
        METRIC_NAMES,
        path_name="main",
    )
    table = portlearn.comparisons.ComparisonTable(spec)
    deterministic_view = SimpleNamespace(
        schema_version=table.schema_version,
        arms=(),
        rows=table.rows,
        conditions=table.conditions,
    )
    rendered = portlearn.comparisons.render_table(deterministic_view)
    lines.append("[c] comparison: render_table output verbatim (descriptive-only)")
    lines.extend(rendered.rstrip("\n").split("\n"))
    lines.append(f"[f] {FENCE}")

    prefix = "".join(f"{line}\n" for line in lines)
    digest = hashlib.sha256(prefix.encode()).hexdigest()
    print(prefix, end="")
    print(f"DIGEST_MARKER sha256={digest}")


# --------------------------------------------------------------------------- #
# The refresh path: the only network leg. #
# --------------------------------------------------------------------------- #


def _refresh(snapshot_override: Path | None) -> None:
    payload, provenance = ff.fetch_raw(DATASET_KEY)
    target = snapshot_override if snapshot_override is not None else SCRATCH_SNAPSHOT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    entry = ff.FF_DATASETS[DATASET_KEY]
    manifest = {
        "source_id": provenance.source_id,
        "dataset_id": provenance.dataset_id,
        "url": provenance.url,
        "member_name": entry.member_name,
        "dataset_kind": provenance.dataset_kind,
        "units": provenance.units,
        "frequency": provenance.frequency,
        "content_sha256": _sha256_hex(payload),
        "retrieval_instant": provenance.retrieval_instant.isoformat(),
        "licence_note": provenance.licence_note,
        "not_investable_warning": provenance.not_investable_warning,
        "adapter_identity": provenance.adapter_identity,
        "last_modified": provenance.last_modified,
        "alias_catalog_version": fama_french.ALIAS_CATALOG_VERSION,
        "synthetic": False,
        "snapshot_written": str(target),
    }
    print(json.dumps(manifest, sort_keys=True))
    raise SystemExit(0)


# --------------------------------------------------------------------------- #
# CLI. #
# --------------------------------------------------------------------------- #


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classical research smoke replay (instructional only)."
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="replay the committed synthetic long fixture offline",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="replay an explicitly selected snapshot file",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="fetch once, write the snapshot, print the manifest JSON, exit",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if args.refresh:
        _refresh(args.snapshot)
        return

    if args.fixture:
        if not FIXTURE.is_file():
            _fail_closed(
                f"the committed synthetic fixture is absent: {FIXTURE}"
            )
        _replay(FIXTURE.read_bytes(), synthetic=True, pinned=None)
        return

    if args.snapshot is not None:
        if not args.snapshot.is_file():
            _fail_closed(f"the selected snapshot is absent: {args.snapshot}")
        raw = args.snapshot.read_bytes()
        _replay(raw, synthetic=False, pinned=_verify_against_pin(raw))
        return

    if not SOURCES_MANIFEST_PATH.is_file() or not SCRATCH_SNAPSHOT.is_file():
        _fail_closed(REFRESH_INSTRUCTIONS)
    raw = SCRATCH_SNAPSHOT.read_bytes()
    _replay(raw, synthetic=False, pinned=_verify_against_pin(raw))


if __name__ == "__main__":
    main()
