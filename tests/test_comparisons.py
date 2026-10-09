"""Comparison-output contract tests: the structured snapshot and its render.

The surface under test is the ``portlearn.comparisons`` public module: the
nine frozen value objects (``ComparatorClassification``, ``SupportRelation``,
``MetricObservation``, ``PathIdentity``, ``ComparisonSpec``, ``ArmColumn``,
``ComparisonCell``, ``ComparisonRow``, ``ComparisonTable``), the module-owned
``ComparisonError``, and the one pure renderer ``render_table`` — exactly as
frozen by the approved comparison-tables contract (the results/runtime new-module
RED-first precedent).

RED structure (the house idiom): production does not exist yet, so this
module never imports ``portlearn.comparisons`` at collection time.  A cached
helper imports it at test-execution time; every test calls that helper
first, so collection succeeds and each behavior floor individually reaches
the same causal production-absence failure rather than one module-level
import error masquerading as the whole suite.  When the module lands, the
same calls exercise the full behavior below.

Behavior floors pinned here (26 floors; the contract's T1-T24 plus T7a/T7b,
with the scalar/tuple floor split into its two legs):

* import side-effect freedom and the pinned eleven-name ``__all__`` census
  (T1); the module-owned ``ValueError`` arm distinct from every frozen
  error type (T2).
* exact field derivation from the spec — arms, rows, cells, verdicts, full
  five-field sanctioned deviations, provenance, path identity, and the
  verbatim shared-conditions snapshot (T3).
* fail-closed consumption: the frozen readiness chain propagates unwrapped
  on a zero-comparator declaration (T4); a missing target record fails
  closed (T5); a record for an undeclared id rejects (T7); a record for a
  registered-but-unassigned id rejects (T7a); the metric vocabulary is the
  six frozen names — unknown, duplicate, and empty each reject (T8); a
  blank path name rejects (Law 2); classification and support keys must be
  declared comparator ids (T18's rejection legs).
* explicit missing states: a missing comparator record renders a
  ``missing_record`` column with every record-derived field exactly ``None``
  (T6); a declared ablation or robustness check without a record renders
  the same explicit state (T7b); a record without the named path renders
  ``missing_path`` cells and a path without a requested metric renders
  ``missing_metric`` cells (T12).
* declaration visibility: multiple declaration sets all surface, none
  selected (T9); gamma/basis mismatch is visible, never bridged (T10); the
  ``risk_free`` scalar never conflates with its length-1 tuple (T11a) and
  the canonical observation order follows the wire encoding across mixed
  forms (T11b).
* canonical ordering: submission-order invariance is byte-exact (T13);
  row order is the declared metric order, never alphabetical or
  value-ordered (T14); no rank, difference, extrema, or value-based
  ordering exists anywhere (T15).
* verbatim carriage: conformance verdicts and every five-field deviation
  entry per arm, with the empty-tuple/``None`` distinction preserved
  (T16); parameter relations carried verbatim, identity and difference
  alike (T17); classifications, support relations, and their defaults
  render, with ``not_applicable`` on every non-comparator arm (T18);
  estimator provenance never enters the table (T24).
* serialization: ``to_json`` is canonical, sorted, compact, ASCII,
  deterministic, and fail-closed on non-finite values (T19);
  ``from_json`` round-trips exactly, including mismatch and missing
  states (T20); ``from_json`` is exact-schema — unknown keys at any
  level, malformed tokens, and non-finite wire numbers all reject (T21).
* rendering: pure, deterministic, plain text with the shared-conditions
  header and per-arm header blocks, explicit ``missing`` tokens, and the
  literal ``not_available`` for every record-derived field of a
  ``missing_record`` arm (T22).
* snapshot discipline: tables are immutable, deeply snapshotted from their
  consumed inputs, and unhashable by design (T23).

Mutation teeth (each discriminating mutation is killed by the named tests;
at RED the module is absent so the legs are encoded, not yet executed
against mutated builds):

* M1 (sort rows or arms by recorded value) — killed by the declared-order
  floor, the no-rank floor, and submission-order invariance.
* M2 (select one declaration set instead of surfacing all) — killed by the
  multiple-declaration-sets floor.
* M3 (silently bridge mismatched declarations) — killed by the
  declaration-mismatch floor and the scalar/tuple floors.
* M4 (drop or summarize deviations, or fill a missing record's ``None``
  fields with fabricated values) — killed by the deviations floor and both
  explicit-missing-state floors.
* M5 (add a computed difference/ratio/winner column) — killed by the
  no-rank floor and the render floor.
* M6 (accept unknown keys in ``from_json``) — killed by the exact-schema
  floor.
* M7 (conflate the ``risk_free`` scalar with its length-1 tuple) — killed
  by both scalar/tuple floors.
* M8 (reorder arms by role or verdict) — killed by submission-order
  invariance and the no-rank floor.
* M9 (admit an undeclared or unassigned record, or silently skip it) —
  killed by both undeclared-id floors.

Oracles are derived independently in this module from the law text using
the stdlib alone (``json`` canonical re-dumps, wire-key encodings computed
with ``json.dumps``) and from the frozen seams' own public outputs (the
record's ``to_json`` for the instant wire convention); no oracle reads any
production helper, and no local surrogate of the comparison behavior
exists here.

Duck-typed stand-in records and blocks appear only where the frozen
constructors cannot express the hostile input the laws demand (a block
missing one metric name; a non-finite metric value; a mutable consumed
mapping): the frozen collection law admits anything carrying a non-blank
``strategy_id``, so these ride the collection exactly as the contract's
records field assumes.  Every other fixture is rebuilt through the frozen
public constructors (``build_ledger``, ``core_metrics``, ``RunManifest``,
``MetricsBlock``, ``PathResult``, ``ConformanceRecord``,
``SanctionedDeviation``, ``ParameterRelation``, ``ForecastLink``,
``ResultRecord``, ``ExperimentConfiguration``, ``ExperimentResults``).

This suite introduces no inferential computation of its own: it asserts
the absence of ranks, differences, winners, significance marks, and value
ordering, and never computes any.
"""

import functools
import importlib
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone

import pytest

from portlearn.costs import Proportional
from portlearn.experiments import (
    ConfigurationError,
    ExecutionError,
    ExperimentConfiguration,
    ExperimentResults,
    comparison_readiness,
)
from portlearn.forecasting import DeterminismClass
from portlearn.interfaces import PortfolioDecision
from portlearn.ledger import build_ledger
from portlearn.manifest import RunManifest
from portlearn.metrics import core_metrics
from portlearn.results import (
    ConformanceRecord,
    ForecastLink,
    MetricsBlock,
    ParameterRelation,
    PathResult,
    ResultRecord,
    ResultStructureError,
    SanctionedDeviation,
)
from portlearn.turnover import two_sided

# --------------------------------------------------------------------------- #
# The production route (absent at RED; cached import at execution time)
# --------------------------------------------------------------------------- #

PINNED_ALL = [
    "ArmColumn",
    "ComparisonCell",
    "ComparisonError",
    "ComparisonRow",
    "ComparisonSpec",
    "ComparisonTable",
    "ComparatorClassification",
    "MetricObservation",
    "PathIdentity",
    "SupportRelation",
    "render_table",
]

_METRIC_NAMES = (
    "cumulative_return",
    "annualized_mean_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "certainty_equivalent_return",
)

_FORBIDDEN_RANK_TOKENS = (
    "rank",
    "winner",
    "best",
    "worst",
    "delta",
    "difference",
    "*",
    "†",
    "★",
    "±",
)


@functools.cache
def _comparisons():
    """Import the public comparison module at test-execution time."""
    return importlib.import_module("portlearn.comparisons")


# --------------------------------------------------------------------------- #
# Frozen-seam fixtures (every builder uses frozen public constructors only)
# --------------------------------------------------------------------------- #

_MONTHS = tuple(datetime(2026, m, 1, tzinfo=UTC) for m in range(1, 6))
_TZ_PLUS_TEN = timezone(timedelta(hours=10))
_REBALANCE_FEB = PortfolioDecision(
    decision_time=_MONTHS[1],
    execution_time=_MONTHS[1],
    target_weights={"A": 0.5, "B": 0.5},
)
_CONDITIONS = {"cost_rate": 0.001, "periods_per_year": 12, "universe": ("A", "B")}


@functools.cache
def _ledger(name: str):
    growth = {
        "a": {
            _MONTHS[0]: {"A": 0.995, "B": 1.010},
            _MONTHS[1]: {"A": 0.996, "B": 1.011},
            _MONTHS[2]: {"A": 0.994, "B": 1.012},
            _MONTHS[3]: {"A": 0.997, "B": 1.013},
        },
        "b": {
            _MONTHS[0]: {"A": 1.002, "B": 1.004},
            _MONTHS[1]: {"A": 1.001, "B": 1.003},
            _MONTHS[2]: {"A": 1.003, "B": 1.005},
            _MONTHS[3]: {"A": 1.000, "B": 1.002},
        },
        "c": {
            _MONTHS[0]: {"A": 1.006, "B": 0.999},
            _MONTHS[1]: {"A": 1.005, "B": 1.000},
            _MONTHS[2]: {"A": 1.004, "B": 1.001},
            _MONTHS[3]: {"A": 1.007, "B": 0.998},
        },
    }[name]
    return build_ledger(
        initial_weights={"A": 0.4, "B": 0.6},
        accounting_instants=_MONTHS,
        growth_factors=growth,
        decisions=[_REBALANCE_FEB],
        cost_model=Proportional(rate=0.001, turnover=two_sided),
        initial_wealth=100.0,
    )


@functools.cache
def _summary(name: str, basis: str, gamma: float, rf_form: str):
    risk_free = {
        "scalar": 0.001,
        "tuple": (0.001,) * 4,
        "tuple002": (0.002,) * 4,
    }[rf_form]
    return core_metrics(
        _ledger(name),
        periods_per_year=12,
        gamma=gamma,
        risk_free=risk_free,
        basis=basis,
    )


def _block(name, basis="net", gamma=2.0, rf_form="scalar"):
    risk_free = {
        "scalar": 0.001,
        "tuple": (0.001,) * 4,
        "tuple002": (0.002,) * 4,
    }[rf_form]
    return MetricsBlock(
        _summary(name, basis, gamma, rf_form),
        basis=basis,
        periods_per_year=12,
        gamma=gamma,
        risk_free=risk_free,
    )


def _path_result(name, blocks):
    return PathResult(_ledger(name), tuple(blocks))


def _consistent():
    return ConformanceRecord(
        declared={"cost_rate": 0.001, "max_weight": 0.5, "universe": ("A", "B")},
        effective={"universe": ("A", "B"), "max_weight": 0.5, "cost_rate": 0.001},
        deviations=(),
    )


def _sanctioned():
    def sanction(field):
        return SanctionedDeviation(
            field=field,
            authority="portfolio owner",
            reference="sanction-log-7",
            reason=f"raised for the {field} sensitivity check",
            approved_at=datetime(2026, 9, 27, 7, 0, tzinfo=_TZ_PLUS_TEN),
        )

    return ConformanceRecord(
        declared={"cost_rate": 0.001, "max_weight": 0.5, "universe": ("A", "B")},
        effective={"cost_rate": 0.002, "max_weight": 0.25, "universe": ("A", "B")},
        deviations=(sanction("cost_rate"), sanction("max_weight")),
    )


def _relations():
    return (
        ParameterRelation(
            role_a="evaluation_gamma",
            role_b="decision_risk_aversion",
            value_a=2.0,
            value_b=2.0,
        ),
        ParameterRelation(
            role_a="evaluation_gamma",
            role_b="decision_risk_aversion",
            value_a=5.0,
            value_b=2.0,
        ),
    )


def _link(model: str):
    return ForecastLink(
        model=model,
        config_hash="ab12cd34" * 2,
        cutoff=datetime(2026, 4, 30, tzinfo=UTC),
        seed=7,
        determinism=DeterminismClass.DETERMINISTIC,
        fit_window_start=datetime(2026, 1, 1, tzinfo=UTC),
        refit_instants=(),
        forecast_target="expected_return",
        forecast_horizon=timedelta(hours=6),
        information=(),
    )


def _record(
    arm_id: str,
    *,
    ledger: str = "a",
    blocks=None,
    forecasts=(),
    conformance=None,
    relations=(),
    path_name: str = "main",
    paths=None,
) -> ResultRecord:
    manifest = RunManifest(
        run_id=f"run-{arm_id}",
        created_at=datetime(2026, 9, 27, 8, 15, tzinfo=_TZ_PLUS_TEN),
        python_version="3.13.5",
        package_version="0.0.1.dev3",
        dependency_pins={"scipy": "1.14.0"},
        commands=[f"portlearn-run {arm_id}"],
    )
    if paths is None:
        blocks = blocks if blocks is not None else [_block(ledger)]
        paths = {path_name: _path_result(ledger, blocks)}
    return ResultRecord(
        manifest,
        strategy_id=arm_id,
        paths=paths,
        forecasts=tuple(forecasts),
        conformance=conformance if conformance is not None else _consistent(),
        parameter_relations=relations,
        created_at=datetime(2026, 9, 27, 9, 30, tzinfo=_TZ_PLUS_TEN),
    )


def _config(
    ids,
    target,
    comparators=(),
    ablations=(),
    robustness=(),
    benchmarks=(),
    conditions=None,
):
    return ExperimentConfiguration(
        strategy_ids=tuple(ids),
        target=target,
        comparators=tuple(comparators),
        ablations=tuple(ablations),
        robustness_checks=tuple(robustness),
        benchmarks=tuple(benchmarks),
        conditions=dict(conditions if conditions is not None else _CONDITIONS),
    )


def _collection(*records):
    return ExperimentResults(records=tuple(records))


# --------------------------------------------------------------------------- #
# Duck-typed stand-ins (the frozen collection law admits them; used only
# for hostile inputs the frozen constructors cannot express)
# --------------------------------------------------------------------------- #


class _DuckBlock:
    """A block carrying only the metric names and declarations given."""

    def __init__(self, metrics, *, basis, periods_per_year, gamma, risk_free):
        for name, value in metrics.items():
            setattr(self, name, value)
        self.basis = basis
        self.periods_per_year = periods_per_year
        self.gamma = gamma
        self.risk_free = risk_free


class _DuckPath:
    def __init__(self, metrics, *, ledger_digest, n_periods, final_wealth):
        self.metrics = tuple(metrics)
        self.ledger_digest = ledger_digest
        self.n_periods = n_periods
        self.period_start = _MONTHS[0]
        self.period_end = _MONTHS[-1]
        self.final_wealth = final_wealth


class _DuckRecord:
    def __init__(self, arm_id, paths):
        self.strategy_id = arm_id
        self.paths = paths
        self.forecasts = ()
        self.conformance = _consistent()
        self.parameter_relations = ()
        self.run_id = f"run-{arm_id}"
        self.manifest_digest = "0123456789abcdef" * 4
        self.created_at = datetime(2026, 9, 27, 9, 30, tzinfo=_TZ_PLUS_TEN)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _table(
    configuration,
    records,
    metrics,
    path_name="main",
    classifications=None,
    support_relations=None,
):
    mod = _comparisons()
    spec = mod.ComparisonSpec(
        configuration=configuration,
        records=records,
        metrics=metrics,
        path_name=path_name,
        classifications=classifications if classifications is not None else {},
        support_relations=support_relations if support_relations is not None else {},
    )
    return mod.ComparisonTable(spec)


def _arm(table, strategy_id):
    matches = [a for a in table.arms if a.strategy_id == strategy_id]
    assert len(matches) == 1, f"expected exactly one arm for {strategy_id!r}"
    return matches[0]


def _row(table, metric):
    matches = [r for r in table.rows if r.metric == metric]
    assert len(matches) == 1, f"expected exactly one row for {metric!r}"
    return matches[0]


def _cell(table, metric, strategy_id):
    matches = [c for c in _row(table, metric).cells if c.strategy_id == strategy_id]
    assert len(matches) == 1, f"expected exactly one cell for {strategy_id!r}"
    return matches[0]


def _same_deviation(left, right):
    return (
        left.field == right.field
        and left.authority == right.authority
        and left.reference == right.reference
        and left.reason == right.reason
        and left.approved_at == right.approved_at
    )


def _same_relation(left, right):
    return (
        left.role_a == right.role_a
        and left.role_b == right.role_b
        and left.value_a == right.value_a
        and left.value_b == right.value_b
        and left.relation == right.relation
    )


def _wire(value):
    """The canonical wire encoding of a risk_free form (the Law 8 grammar)."""
    return json.dumps(value, separators=(",", ":"))


def _first_float_slot(node):
    """Locate the first float in a parsed wire tree (for corruption legs)."""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, float):
                return node, key
            found = _first_float_slot(value)
            if found is not None:
                return found
    if isinstance(node, list):
        for item in node:
            found = _first_float_slot(item)
            if found is not None:
                return found
    return None


def _first_dict_list(node):
    """Locate the first list-of-dicts in a parsed wire tree."""
    if isinstance(node, dict):
        for value in node.values():
            found = _first_dict_list(value)
            if found is not None:
                return found
    if isinstance(node, list):
        if node and all(isinstance(item, dict) for item in node):
            return node
        for item in node:
            found = _first_dict_list(item)
            if found is not None:
                return found
    return None


_RECORD_DERIVED_FIELDS = (
    "conformance_verdict",
    "sanctioned_deviations",
    "parameter_relations",
    "run_id",
    "manifest_digest",
    "created_at",
    "path_identity",
)


# --------------------------------------------------------------------------- #
# T1 — module boundary and the pinned public census
# --------------------------------------------------------------------------- #


def test_import_is_side_effect_free_and_frozen_modules_untouched():
    import portlearn.experiments as experiments_module
    import portlearn.results as results_module

    # Snapshot each frozen module BEFORE the production import: the attribute
    # census, the pinned ``__all__``, and the object identity of every
    # pre-existing attribute.  Never re-execute a frozen module here — doing
    # so would rebind its classes and poison every later fixture at the
    # frozen seam.
    snapshots = {
        frozen.__name__: {
            "names": sorted(vars(frozen)),
            "all": list(frozen.__all__),
            "identities": dict(vars(frozen)),
        }
        for frozen in (results_module, experiments_module)
    }

    mod = _comparisons()

    for frozen in (results_module, experiments_module):
        snapshot = snapshots[frozen.__name__]
        assert sorted(vars(frozen)) == snapshot["names"], (
            f"importing portlearn.comparisons changed the attribute census "
            f"of {frozen.__name__}"
        )
        assert list(frozen.__all__) == snapshot["all"], (
            f"importing portlearn.comparisons mutated {frozen.__name__}.__all__"
        )
        for name, before_value in snapshot["identities"].items():
            assert vars(frozen)[name] is before_value, (
                f"importing portlearn.comparisons rebound {frozen.__name__}.{name}"
            )

    assert list(mod.__all__) == PINNED_ALL  # the pinned eleven, in order
    for name in PINNED_ALL:
        assert hasattr(mod, name), f"__all__ names {name!r} but it is absent"


# --------------------------------------------------------------------------- #
# T2 — the module-owned rejection arm
# --------------------------------------------------------------------------- #


def test_comparison_error_is_the_module_owned_valueerror_arm():
    mod = _comparisons()

    assert issubclass(mod.ComparisonError, ValueError)
    for frozen_error in (
        ResultStructureError,
        ConfigurationError,
        ExecutionError,
    ):
        assert not issubclass(mod.ComparisonError, frozen_error)
        assert not issubclass(frozen_error, mod.ComparisonError)


# --------------------------------------------------------------------------- #
# T3 — exact field derivation from the spec
# --------------------------------------------------------------------------- #


def test_table_derives_exact_fields_from_spec():
    _comparisons()

    target = _record(
        "target-mv",
        ledger="a",
        conformance=_sanctioned(),
        relations=_relations(),
    )
    cmp_ew = _record("cmp-ew", ledger="b")
    cmp_iv = _record("cmp-iv", ledger="c")
    config = _config(
        ("target-mv", "cmp-ew", "cmp-iv"),
        "target-mv",
        comparators=("cmp-ew", "cmp-iv"),
        benchmarks=("cmp-ew",),
    )
    table = _table(
        config,
        _collection(target, cmp_ew, cmp_iv),
        ("sharpe_ratio", "cumulative_return"),
    )

    assert table.schema_version == "1.0.0"
    assert [a.strategy_id for a in table.arms] == ["cmp-ew", "cmp-iv", "target-mv"]

    target_arm = _arm(table, "target-mv")
    target_path = target.paths["main"]
    assert target_arm.role == "target"
    assert target_arm.is_benchmark is False
    assert target_arm.record_status == "present"
    assert target_arm.conformance_verdict == target.conformance.verdict
    assert len(target_arm.sanctioned_deviations) == 2
    for derived, recorded in zip(
        target_arm.sanctioned_deviations, target.conformance.deviations
    ):
        assert _same_deviation(derived, recorded)
    assert len(target_arm.parameter_relations) == 2
    for derived, recorded in zip(
        target_arm.parameter_relations, target.parameter_relations
    ):
        assert _same_relation(derived, recorded)
    assert target_arm.run_id == target.run_id
    assert target_arm.manifest_digest == target.manifest_digest
    assert target_arm.created_at == target.created_at
    identity = target_arm.path_identity
    assert identity is not None
    assert identity.ledger_digest == target_path.ledger_digest
    assert identity.n_periods == target_path.n_periods
    assert identity.period_start == target_path.period_start
    assert identity.period_end == target_path.period_end
    assert identity.final_wealth == target_path.final_wealth

    ew_arm = _arm(table, "cmp-ew")
    assert ew_arm.role == "comparator"
    assert ew_arm.is_benchmark is True
    assert ew_arm.record_status == "present"
    consistent = cmp_ew.conformance
    assert ew_arm.conformance_verdict == consistent.verdict
    assert ew_arm.sanctioned_deviations == ()  # the recorded empty fact
    assert ew_arm.parameter_relations == ()
    for arm in (target_arm, ew_arm, _arm(table, "cmp-iv")):
        for field in _RECORD_DERIVED_FIELDS:
            assert getattr(arm, field) is not None, (
                f"a present record carries {field}; an empty tuple is lawful, "
                "None is not"
            )

    assert [r.metric for r in table.rows] == ["sharpe_ratio", "cumulative_return"]
    block = target_path.metrics[0]
    cell = _cell(table, "sharpe_ratio", "target-mv")
    assert cell.status == "ok"
    assert len(cell.observations) == 1
    observation = cell.observations[0]
    assert observation.value == block.sharpe_ratio
    assert observation.basis == block.basis
    assert observation.periods_per_year == block.periods_per_year
    assert observation.gamma == block.gamma
    assert observation.risk_free == block.risk_free

    conditions = dict(table.conditions)
    assert conditions == dict(config.conditions)
    assert conditions["universe"] == ("A", "B")  # the frozen snapshot, verbatim


# --------------------------------------------------------------------------- #
# T4 — the frozen readiness chain propagates unwrapped
# --------------------------------------------------------------------------- #


def test_zero_comparator_declaration_fails_closed_through_frozen_readiness():
    mod = _comparisons()

    zero = _config(("target-mv",), "target-mv")
    with pytest.raises(ExecutionError):
        comparison_readiness(zero)

    target = _record("target-mv", ledger="a")
    with pytest.raises(ExecutionError) as exc_info:
        _table(zero, _collection(target), ("sharpe_ratio",))
    assert type(exc_info.value) is ExecutionError  # unwrapped, unchanged
    assert not isinstance(exc_info.value, mod.ComparisonError)

    # A lawful zero-comparator declaration that still names an ablation
    # arm is zero-comparator all the same: the frozen readiness chain
    # fires before any derivation (Law 2) and its ExecutionError
    # propagates unwrapped (Law 10). The missing ablation record is a
    # rendering concern (D4; floor T7b), never a readiness bypass.
    zero_with_ablation = _config(
        ("target-mv", "abl-n"),
        "target-mv",
        ablations=("abl-n",),
    )
    with pytest.raises(ExecutionError) as sibling_info:
        comparison_readiness(zero_with_ablation)
    frozen_message = str(sibling_info.value)  # the frozen sibling's own words

    with pytest.raises(ExecutionError) as ablation_info:
        _table(zero_with_ablation, _collection(target), ("sharpe_ratio",))
    assert type(ablation_info.value) is ExecutionError  # unwrapped, unchanged
    assert not isinstance(ablation_info.value, mod.ComparisonError)
    assert str(ablation_info.value) == frozen_message  # the exact frozen message


# --------------------------------------------------------------------------- #
# T5 — the missing target record fails closed
# --------------------------------------------------------------------------- #


def test_missing_target_record_fails_closed():
    mod = _comparisons()

    config = _config(
        ("target-mv", "cmp-ew"),
        "target-mv",
        comparators=("cmp-ew",),
    )
    only_comparator = _collection(_record("cmp-ew", ledger="b"))
    with pytest.raises(mod.ComparisonError):
        _table(config, only_comparator, ("sharpe_ratio",))


# --------------------------------------------------------------------------- #
# T6 — a missing comparator record is an explicit state, never a hole
# --------------------------------------------------------------------------- #


def test_missing_comparator_record_renders_explicit_missing_state():
    mod = _comparisons()

    target = _record("target-mv", ledger="a")
    cmp_ew = _record("cmp-ew", ledger="b")
    config = _config(
        ("target-mv", "cmp-ew", "cmp-missing", "cmp-default"),
        "target-mv",
        comparators=("cmp-ew", "cmp-missing", "cmp-default"),
    )
    table = _table(
        config,
        _collection(target, cmp_ew),
        ("sharpe_ratio",),
        classifications={
            "cmp-missing": mod.ComparatorClassification(
                kind="direct_comparator",
                method_note="direct RL comparator, policy-gradient sweep",
            )
        },
        support_relations={
            "cmp-missing": mod.SupportRelation(
                kind="same_support_declared",
                evidence_reference="run sheet 12",
            )
        },
    )

    missing_arm = _arm(table, "cmp-missing")
    assert missing_arm.record_status == "missing_record"
    for field in _RECORD_DERIVED_FIELDS:
        value = getattr(missing_arm, field)
        assert value is None, (
            f"{field} must be exactly None for a missing record; got {value!r} "
            "— never a fabricated, sentinel, or empty-set stand-in"
        )
    cell = _cell(table, "sharpe_ratio", "cmp-missing")
    assert cell.status == "missing_record"
    assert cell.observations == ()

    # Law 11 / §4: a comparator's classification and support relation
    # are the researcher's declared claims, carried on the arm verbatim
    # — a missing record nulls only the seven record-derived facts,
    # never the declaration metadata (only non-comparator arms render
    # not_applicable).
    assert missing_arm.classification is not None, (
        "a comparator's declared classification survives its missing "
        "record: the declaration is the researcher's claim, not a "
        "record-derived fact (Law 11)"
    )
    assert missing_arm.classification.kind == "direct_comparator"
    assert missing_arm.support is not None, (
        "a comparator always carries a support relation — declared or "
        "the undeclared default — whatever its record status (Law 11)"
    )
    assert missing_arm.support.kind == "same_support_declared"

    # the undeclared defaults hold for a missing comparator exactly as
    # for a present one: classification renders unclassified (None) and
    # support carries the explicit undeclared default
    default_arm = _arm(table, "cmp-default")
    assert default_arm.record_status == "missing_record"
    for field in _RECORD_DERIVED_FIELDS:
        assert getattr(default_arm, field) is None
    assert default_arm.classification is None  # renders unclassified
    assert default_arm.support is not None
    assert default_arm.support.kind == "undeclared"
    assert _cell(table, "sharpe_ratio", "cmp-default").status == "missing_record"

    for present_id in ("target-mv", "cmp-ew"):
        arm = _arm(table, present_id)
        assert arm.record_status == "present"
        for field in _RECORD_DERIVED_FIELDS:
            assert getattr(arm, field) is not None
        assert _cell(table, "sharpe_ratio", present_id).status == "ok"

    # the render shows each comparator's declared or default metadata
    # and never renders a comparator's classification or support as
    # not_applicable (Law 11)
    text = mod.render_table(table)
    sections = {}
    current = None
    for line in text.splitlines():
        if line.startswith("arm "):
            current = line[len("arm ") :]
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    declared_lines = sections["cmp-missing"]
    assert "classification direct_comparator" in declared_lines
    assert "support same_support_declared" in declared_lines
    default_lines = sections["cmp-default"]
    assert "classification unclassified" in default_lines
    assert "support undeclared" in default_lines
    for comparator_id in ("cmp-ew", "cmp-missing", "cmp-default"):
        offender = [
            line
            for line in sections[comparator_id]
            if line in ("classification not_applicable", "support not_applicable")
        ]
        assert offender == [], (
            f"comparator {comparator_id!r} rendered {offender!r}: a "
            "comparator's metadata is a declaration, never "
            "not_applicable (Law 11)"
        )

    assert _row(table, "sharpe_ratio").comparability == "missing_values"
    assert mod.ComparisonError is not None  # the module stays the error owner


# --------------------------------------------------------------------------- #
# T7 / T7a — role-less records never enter a table
# --------------------------------------------------------------------------- #


def test_record_for_undeclared_id_rejects():
    mod = _comparisons()

    config = _config(
        ("target-mv", "cmp-ew"),
        "target-mv",
        comparators=("cmp-ew",),
    )
    ghost = _record("ghost-id", ledger="c")
    with pytest.raises(mod.ComparisonError):
        _table(config, _collection(_record("target-mv"), ghost), ("sharpe_ratio",))


def test_declared_but_unassigned_record_rejects():
    mod = _comparisons()

    config = _config(
        ("target-mv", "cmp-ew", "spare-id"),
        "target-mv",
        comparators=("cmp-ew",),
    )
    spare = _record("spare-id", ledger="b")
    with pytest.raises(mod.ComparisonError):
        _table(
            config,
            _collection(_record("target-mv"), _record("cmp-ew", ledger="b"), spare),
            ("sharpe_ratio",),
        )


# --------------------------------------------------------------------------- #
# T7b — missing ablation / robustness records render the explicit state
# --------------------------------------------------------------------------- #


def test_missing_ablation_or_robustness_record_renders_explicit_missing_state():
    target = _record("target-mv", ledger="a")
    cmp_ew = _record("cmp-ew", ledger="b")
    config = _config(
        ("target-mv", "cmp-ew", "abl-n", "rob-p"),
        "target-mv",
        comparators=("cmp-ew",),
        ablations=("abl-n",),
        robustness=("rob-p",),
    )
    table = _table(config, _collection(target, cmp_ew), ("sharpe_ratio",))

    # the named comparator carries its record: readiness affirms and the
    # table builds, so the missing arms below are a rendering fact (Law 2)
    comparator_arm = _arm(table, "cmp-ew")
    assert comparator_arm.record_status == "present"

    for missing_id in ("abl-n", "rob-p"):
        arm = _arm(table, missing_id)
        assert arm.record_status == "missing_record"
        for field in _RECORD_DERIVED_FIELDS:
            assert getattr(arm, field) is None
        cell = _cell(table, "sharpe_ratio", missing_id)
        assert cell.status == "missing_record"
        assert cell.observations == ()

    target_arm = _arm(table, "target-mv")
    assert target_arm.record_status == "present"
    for field in _RECORD_DERIVED_FIELDS:
        assert getattr(target_arm, field) is not None
    assert _row(table, "sharpe_ratio").comparability == "missing_values"


# --------------------------------------------------------------------------- #
# T8 — the metric vocabulary is exactly the six frozen names
# --------------------------------------------------------------------------- #


def test_metric_vocabulary_is_the_six_frozen_names():
    mod = _comparisons()

    config = _config(
        ("target-mv", "cmp-ew"),
        "target-mv",
        comparators=("cmp-ew",),
    )
    records = _collection(_record("target-mv"), _record("cmp-ew", ledger="b"))
    for metrics in (
        ("sharpe_ratio", "unknown_metric"),
        ("sharpe_ratio", "sharpe_ratio"),
        (),
    ):
        with pytest.raises(mod.ComparisonError):
            _table(config, records, metrics)


def test_blank_path_name_rejects_fail_closed():
    mod = _comparisons()

    config = _config(
        ("target-mv", "cmp-ew"),
        "target-mv",
        comparators=("cmp-ew",),
    )
    records = _collection(_record("target-mv"), _record("cmp-ew", ledger="b"))
    with pytest.raises(mod.ComparisonError):
        _table(config, records, ("sharpe_ratio",), path_name="   ")


# --------------------------------------------------------------------------- #
# T9 — multiple declaration sets all surface, none selected
# --------------------------------------------------------------------------- #


def test_multiple_declaration_sets_surface_all_and_select_none():
    mod = _comparisons()

    target = _record(
        "target-mv",
        blocks=[_block("a", gamma=2.0), _block("a", gamma=5.0)],
    )
    config = _config(
        ("target-mv", "cmp-ew"),
        "target-mv",
        comparators=("cmp-ew",),
    )
    table = _table(
        config,
        _collection(target, _record("cmp-ew", ledger="b")),
        ("sharpe_ratio",),
    )

    cell = _cell(table, "sharpe_ratio", "target-mv")
    assert cell.status == "ok"
    assert len(cell.observations) == 2  # both declaration sets, none selected
    assert [o.gamma for o in cell.observations] == [2.0, 5.0]  # canonical key
    recorded = {block.sharpe_ratio for block in target.paths["main"].metrics}
    assert {o.value for o in cell.observations} == recorded
    assert _row(table, "sharpe_ratio").comparability == "multiple_declaration_sets"
    assert mod.ComparisonError is not None


# --------------------------------------------------------------------------- #
# T10 — declaration mismatch is visible, never bridged
# --------------------------------------------------------------------------- #


def test_declaration_mismatch_is_visible_not_bridged():
    _comparisons()

    # gamma differs across arms
    config = _config(
        ("target-mv", "cmp-ew", "cmp-iv"),
        "target-mv",
        comparators=("cmp-ew", "cmp-iv"),
    )
    records = _collection(
        _record("target-mv", blocks=[_block("a", gamma=2.0)]),
        _record("cmp-ew", blocks=[_block("b", gamma=5.0)]),
        _record("cmp-iv", blocks=[_block("c", gamma=2.0)]),
    )
    table = _table(config, records, ("sharpe_ratio",))
    assert _row(table, "sharpe_ratio").comparability == "declaration_mismatch"
    gammas = {
        "target-mv": 2.0,
        "cmp-ew": 5.0,  # its own declaration survives — no bridging
        "cmp-iv": 2.0,
    }
    for arm_id, gamma in gammas.items():
        observation = _cell(table, "sharpe_ratio", arm_id).observations[0]
        assert observation.gamma == gamma

    # basis differs across arms
    records_gross = _collection(
        _record("target-mv", blocks=[_block("a", basis="net")]),
        _record("cmp-ew", blocks=[_block("b", basis="gross")]),
    )
    config_two = _config(
        ("target-mv", "cmp-ew"),
        "target-mv",
        comparators=("cmp-ew",),
    )
    table_gross = _table(config_two, records_gross, ("sharpe_ratio",))
    assert _row(table_gross, "sharpe_ratio").comparability == "declaration_mismatch"
    assert (
        _cell(table_gross, "sharpe_ratio", "target-mv").observations[0].basis == "net"
    )
    assert _cell(table_gross, "sharpe_ratio", "cmp-ew").observations[0].basis == "gross"


# --------------------------------------------------------------------------- #
# T11 — the risk_free scalar never conflates with its tuple forms
# --------------------------------------------------------------------------- #


def test_risk_free_scalar_and_tuple_never_conflate():
    _comparisons()

    config = _config(
        ("target-mv", "cmp-ew"),
        "target-mv",
        comparators=("cmp-ew",),
    )
    # scalar 0.001 vs the same value as a length-1 tuple
    records = _collection(
        _record("target-mv", blocks=[_block("a", rf_form="scalar")]),
        _record(
            "cmp-ew",
            blocks=[
                MetricsBlock(
                    _summary("b", "net", 2.0, "tuple"),
                    basis="net",
                    periods_per_year=12,
                    gamma=2.0,
                    risk_free=(0.001,),
                )
            ],
        ),
    )
    table = _table(config, records, ("sharpe_ratio",))
    assert _row(table, "sharpe_ratio").comparability == "declaration_mismatch"
    target_obs = _cell(table, "sharpe_ratio", "target-mv").observations[0]
    cmp_obs = _cell(table, "sharpe_ratio", "cmp-ew").observations[0]
    assert target_obs.risk_free == 0.001  # verbatim, type-preserved
    assert cmp_obs.risk_free == (0.001,)  # never flattened, never merged

    # scalar vs a differing tuple: mismatch, never equality
    records_differing = _collection(
        _record("target-mv", blocks=[_block("a", rf_form="scalar")]),
        _record("cmp-ew", blocks=[_block("b", rf_form="tuple002")]),
    )
    table_differing = _table(config, records_differing, ("sharpe_ratio",))
    assert _row(table_differing, "sharpe_ratio").comparability == "declaration_mismatch"


def test_risk_free_canonical_order_follows_the_wire_encoding():
    _comparisons()

    target = _record(
        "target-mv",
        blocks=[
            _block("a", rf_form="tuple002"),
            _block("a", rf_form="scalar"),
            _block("a", rf_form="tuple"),
        ],
    )
    config = _config(
        ("target-mv", "cmp-ew"),
        "target-mv",
        comparators=("cmp-ew",),
    )
    table = _table(
        config, _collection(target, _record("cmp-ew", ledger="b")), ("sharpe_ratio",)
    )

    cell = _cell(table, "sharpe_ratio", "target-mv")
    assert len(cell.observations) == 3
    encoded = [_wire(o.risk_free) for o in cell.observations]
    # Law 4: ascending by the wire encoding — forms are never structurally
    # compared, so the oracle here is the sorted wire strings themselves.
    assert encoded == sorted(encoded)
    assert set(encoded) == {
        "0.001",
        "[0.001,0.001,0.001,0.001]",
        "[0.002,0.002,0.002,0.002]",
    }
    assert _row(table, "sharpe_ratio").comparability == "multiple_declaration_sets"


# --------------------------------------------------------------------------- #
# T12 — missing path and missing metric are explicit cell states
# --------------------------------------------------------------------------- #


def test_missing_path_and_missing_metric_are_explicit_states():
    _comparisons()

    # the record exists but carries no path of the compared name
    alt_only = _record("cmp-alt", paths={"alt": _path_result("c", [_block("c")])})
    config = _config(
        ("target-mv", "cmp-alt"),
        "target-mv",
        comparators=("cmp-alt",),
    )
    table = _table(
        config, _collection(_record("target-mv"), alt_only), ("sharpe_ratio",)
    )
    cell = _cell(table, "sharpe_ratio", "cmp-alt")
    assert cell.status == "missing_path"
    assert cell.observations == ()
    arm = _arm(table, "cmp-alt")
    assert arm.record_status == "present"  # the record is present
    assert arm.path_identity is None  # the named path is not
    for field in _RECORD_DERIVED_FIELDS:
        if field != "path_identity":
            assert getattr(arm, field) is not None
    assert _row(table, "sharpe_ratio").comparability == "missing_values"

    # the path exists but the requested metric is absent from its blocks
    duck_block = _DuckBlock(
        {"cumulative_return": 0.05},
        basis="net",
        periods_per_year=12,
        gamma=2.0,
        risk_free=0.001,
    )
    duck_path = _DuckPath(
        [duck_block],
        ledger_digest="ab12cd34" * 4,
        n_periods=4,
        final_wealth=101.0,
    )
    duck = _DuckRecord("cmp-duck", {"main": duck_path})
    config_duck = _config(
        ("target-mv", "cmp-duck"),
        "target-mv",
        comparators=("cmp-duck",),
    )
    table_duck = _table(
        config_duck,
        _collection(_record("target-mv"), duck),
        ("cumulative_return", "sharpe_ratio"),
    )
    present_cell = _cell(table_duck, "cumulative_return", "cmp-duck")
    assert present_cell.status == "ok"
    assert present_cell.observations[0].value == 0.05
    absent_cell = _cell(table_duck, "sharpe_ratio", "cmp-duck")
    assert absent_cell.status == "missing_metric"
    assert absent_cell.observations == ()
    assert _row(table_duck, "sharpe_ratio").comparability == "missing_values"
    assert _row(table_duck, "cumulative_return").comparability == "comparable"


# --------------------------------------------------------------------------- #
# T13 — submission-order invariance is byte-exact
# --------------------------------------------------------------------------- #


def test_submission_order_invariance_is_byte_exact():
    _comparisons()

    target = _record("target-mv", ledger="a")
    cmp_ew = _record("cmp-ew", ledger="b")
    cmp_iv = _record("cmp-iv", ledger="c")
    config = _config(
        ("target-mv", "cmp-ew", "cmp-iv"),
        "target-mv",
        comparators=("cmp-ew", "cmp-iv"),
    )
    forward = _table(
        config,
        _collection(target, cmp_ew, cmp_iv),
        ("sharpe_ratio", "max_drawdown"),
    )
    reverse = _table(
        config,
        _collection(cmp_iv, target, cmp_ew),
        ("sharpe_ratio", "max_drawdown"),
    )
    assert forward.to_json() == reverse.to_json()  # byte-identical


# --------------------------------------------------------------------------- #
# T14 — row order is the declared metric order
# --------------------------------------------------------------------------- #


def test_row_order_is_the_declared_metric_order():
    _comparisons()

    config = _config(
        ("target-mv", "cmp-ew"),
        "target-mv",
        comparators=("cmp-ew",),
    )
    records = _collection(_record("target-mv"), _record("cmp-ew", ledger="b"))
    declared = ("sharpe_ratio", "cumulative_return", "max_drawdown")
    table = _table(config, records, declared)
    assert [r.metric for r in table.rows] == list(declared)
    alphabetical = sorted(declared)
    assert list(declared) != alphabetical  # the fixture is adversarial
    assert [r.metric for r in table.rows] != alphabetical  # neither alpha…')


# --------------------------------------------------------------------------- #
# T15 — no rank, no value ordering, anywhere
# --------------------------------------------------------------------------- #


def test_no_rank_or_value_ordering_anywhere():
    mod = _comparisons()

    config = _config(
        ("target-mv", "cmp-ew", "cmp-iv"),
        "target-mv",
        comparators=("cmp-ew", "cmp-iv"),
    )
    calm = _collection(
        _record("target-mv", ledger="a"),
        _record("cmp-ew", ledger="b"),
        _record("cmp-iv", ledger="c"),
    )
    scrambled = _collection(
        _record("target-mv", ledger="c"),  # values rotated across arms
        _record("cmp-ew", ledger="a"),
        _record("cmp-iv", ledger="b"),
    )
    canonical = ["cmp-ew", "cmp-iv", "target-mv"]
    tables = []
    for records in (calm, scrambled):
        table = _table(config, records, ("sharpe_ratio", "cumulative_return"))
        tables.append(table)
        assert [a.strategy_id for a in table.arms] == canonical
        for row in table.rows:
            assert [c.strategy_id for c in row.cells] == canonical
        text = mod.render_table(table)
        for token in _FORBIDDEN_RANK_TOKENS:
            assert token not in text, f"rendered text carries {token!r}"
        positions = [text.index(sid) for sid in canonical]
        assert positions == sorted(positions)  # arms render in one order
    assert tables[0].to_json() != tables[1].to_json()  # values really moved


# --------------------------------------------------------------------------- #
# T16 — verdicts and full deviations per arm
# --------------------------------------------------------------------------- #


def test_conformance_verdicts_and_full_deviations_render_per_arm():
    mod = _comparisons()

    sanctioned = _record(
        "target-mv",
        ledger="a",
        conformance=_sanctioned(),
    )
    consistent = _record("cmp-ew", ledger="b")
    config = _config(
        ("target-mv", "cmp-ew"),
        "target-mv",
        comparators=("cmp-ew",),
    )
    table = _table(config, _collection(sanctioned, consistent), ("sharpe_ratio",))

    sanctioned_arm = _arm(table, "target-mv")
    assert sanctioned_arm.conformance_verdict == sanctioned.conformance.verdict
    assert len(sanctioned_arm.sanctioned_deviations) == 2
    for derived, recorded in zip(
        sanctioned_arm.sanctioned_deviations, sanctioned.conformance.deviations
    ):
        assert _same_deviation(derived, recorded)  # five fields, verbatim

    consistent_arm = _arm(table, "cmp-ew")
    assert consistent_arm.conformance_verdict == consistent.conformance.verdict
    assert consistent_arm.sanctioned_deviations == ()  # the recorded fact
    assert consistent_arm.sanctioned_deviations is not None  # never None here

    text = mod.render_table(table)
    assert sanctioned.conformance.verdict in text
    assert consistent.conformance.verdict in text
    for deviation in sanctioned.conformance.deviations:
        assert deviation.authority in text
        assert deviation.reference in text
        assert deviation.reason in text


# --------------------------------------------------------------------------- #
# T17 — parameter relations carried verbatim
# --------------------------------------------------------------------------- #


def test_parameter_relations_carried_verbatim():
    mod = _comparisons()

    target = _record("target-mv", ledger="a", relations=_relations())
    config = _config(
        ("target-mv", "cmp-ew"),
        "target-mv",
        comparators=("cmp-ew",),
    )
    table = _table(
        config, _collection(target, _record("cmp-ew", ledger="b")), ("sharpe_ratio",)
    )

    arm = _arm(table, "target-mv")
    assert len(arm.parameter_relations) == 2
    for derived, recorded in zip(arm.parameter_relations, target.parameter_relations):
        assert _same_relation(derived, recorded)
    tokens = {relation.relation for relation in arm.parameter_relations}
    assert tokens == {"identical", "distinct"}  # both, exactly as recorded
    text = mod.render_table(table)
    assert "identical" in text
    assert "distinct" in text


# --------------------------------------------------------------------------- #
# T18 — classifications, support relations, and their defaults
# --------------------------------------------------------------------------- #


def test_classifications_support_and_defaults_render():
    mod = _comparisons()

    classification = mod.ComparatorClassification(
        kind="same_support_equal_weight",
        method_note="equal-weighted rerun on the shared support",
    )
    support = mod.SupportRelation(
        kind="same_support_declared",
        evidence_reference="run sheet 12",
    )
    target = _record("target-mv", ledger="a")
    cmp_ew = _record("cmp-ew", ledger="b")
    cmp_iv = _record("cmp-iv", ledger="c")
    ablation = _record("abl-n", ledger="a")
    config = _config(
        ("target-mv", "cmp-ew", "cmp-iv", "abl-n"),
        "target-mv",
        comparators=("cmp-ew", "cmp-iv"),
        ablations=("abl-n",),
    )
    table = _table(
        config,
        _collection(target, cmp_ew, cmp_iv, ablation),
        ("sharpe_ratio",),
        classifications={"cmp-ew": classification},
        support_relations={"cmp-ew": support},
    )

    classified = _arm(table, "cmp-ew")
    assert classified.classification is not None
    assert classified.classification.kind == "same_support_equal_weight"
    assert classified.classification.method_note == (
        "equal-weighted rerun on the shared support"
    )
    assert classified.support is not None
    assert classified.support.kind == "same_support_declared"

    unclassified = _arm(table, "cmp-iv")
    assert unclassified.classification is None  # None means unclassified
    assert unclassified.support is not None  # the default applies
    assert unclassified.support.kind == "undeclared"

    for non_comparator_id in ("target-mv", "abl-n"):
        arm = _arm(table, non_comparator_id)
        assert arm.classification is None  # renders not_applicable
        assert arm.support is None  # renders not_applicable

    text = mod.render_table(table)
    assert "unclassified" in text
    assert "not_applicable" in text
    assert "same_support_equal_weight" in text
    assert "same_support_declared" in text

    # classification keyed on a non-comparator rejects at spec construction
    with pytest.raises(mod.ComparisonError):
        mod.ComparisonSpec(
            configuration=config,
            records=_collection(target, cmp_ew, cmp_iv, ablation),
            metrics=("sharpe_ratio",),
            path_name="main",
            classifications={"target-mv": classification},
        )
    # support keyed on a non-comparator rejects the same way
    with pytest.raises(mod.ComparisonError):
        mod.ComparisonSpec(
            configuration=config,
            records=_collection(target, cmp_ew, cmp_iv, ablation),
            metrics=("sharpe_ratio",),
            path_name="main",
            classifications={},
            support_relations={"abl-n": support},
        )
    # Law 11: blank notes and references reject; unknown kinds reject
    with pytest.raises(mod.ComparisonError):
        blank_note = mod.ComparatorClassification(
            kind="direct_comparator",
            method_note="   ",
        )
        _table(
            config,
            _collection(target, cmp_ew, cmp_iv, ablation),
            ("sharpe_ratio",),
            classifications={"cmp-ew": blank_note},
        )
    with pytest.raises(mod.ComparisonError):
        blank_reference = mod.SupportRelation(
            kind="undeclared",
            evidence_reference="  ",
        )
        _table(
            config,
            _collection(target, cmp_ew, cmp_iv, ablation),
            ("sharpe_ratio",),
            support_relations={"cmp-ew": blank_reference},
        )
    with pytest.raises(mod.ComparisonError):
        unknown_kind = mod.ComparatorClassification(kind="mystery_kind")
        _table(
            config,
            _collection(target, cmp_ew, cmp_iv, ablation),
            ("sharpe_ratio",),
            classifications={"cmp-ew": unknown_kind},
        )


# --------------------------------------------------------------------------- #
# T19 — to_json is canonical and deterministic
# --------------------------------------------------------------------------- #


def test_to_json_is_canonical_and_deterministic():
    _comparisons()

    config = _config(
        ("target-mv", "cmp-ew", "cmp-iv"),
        "target-mv",
        comparators=("cmp-ew", "cmp-iv"),
    )
    target = _record("target-mv", ledger="a")

    def build():
        return _table(
            config,
            _collection(
                target, _record("cmp-ew", ledger="b"), _record("cmp-iv", ledger="c")
            ),
            ("sharpe_ratio",),
        )

    table = build()
    text = table.to_json()
    assert text.isascii()  # ASCII-only
    assert ", " not in text  # compact separators
    assert ": " not in text
    parsed = json.loads(text)
    # the sorted-keys oracle: a canonical re-dump reproduces the bytes
    assert json.dumps(parsed, sort_keys=True, separators=(",", ":")) == text
    assert build().to_json() == text  # equal inputs, byte-identical output

    # the instant wire convention is the record's own (Law 8)
    record_wire = json.loads(target.to_json())["created_at"]
    assert record_wire in text

    # non-finite metric values reject fail-closed (a duck block carries one)
    poisoned = _DuckBlock(
        {"sharpe_ratio": float("inf")},
        basis="net",
        periods_per_year=12,
        gamma=2.0,
        risk_free=0.001,
    )
    poisoned_path = _DuckPath(
        [poisoned],
        ledger_digest="ab12cd34" * 4,
        n_periods=4,
        final_wealth=101.0,
    )
    poisoned_record = _DuckRecord("cmp-duck", {"main": poisoned_path})
    poison_config = _config(
        ("target-mv", "cmp-duck"),
        "target-mv",
        comparators=("cmp-duck",),
    )
    mod = _comparisons()
    with pytest.raises(mod.ComparisonError):
        poisoned_table = _table(
            poison_config,
            _collection(_record("target-mv"), poisoned_record),
            ("sharpe_ratio",),
        )
        poisoned_table.to_json()


# --------------------------------------------------------------------------- #
# T20 — from_json round-trips exactly
# --------------------------------------------------------------------------- #


def test_from_json_roundtrips_exactly():
    mod = _comparisons()

    config = _config(
        ("target-mv", "cmp-ew", "cmp-missing"),
        "target-mv",
        comparators=("cmp-ew", "cmp-missing"),
    )
    mismatch_config = _config(
        ("target-mv", "cmp-ew"),
        "target-mv",
        comparators=("cmp-ew",),
    )

    happy = _table(
        config,
        _collection(_record("target-mv"), _record("cmp-ew", ledger="b")),
        ("sharpe_ratio", "max_drawdown"),
    )
    mismatch = _table(
        mismatch_config,
        _collection(
            _record("target-mv", blocks=[_block("a", gamma=2.0)]),
            _record("cmp-ew", blocks=[_block("b", gamma=5.0)]),
        ),
        ("sharpe_ratio",),
    )
    multi = _table(
        mismatch_config,
        _collection(
            _record(
                "target-mv", blocks=[_block("a", gamma=2.0), _block("a", gamma=5.0)]
            ),
            _record("cmp-ew", ledger="b"),
        ),
        ("sharpe_ratio",),
    )

    for table in (happy, mismatch, multi):
        roundtrip = mod.ComparisonTable.from_json(table.to_json())
        assert roundtrip == table
        assert roundtrip.to_json() == table.to_json()

    # the missing-record state survives the wire with its Nones intact
    missing_arm = _arm(happy, "cmp-missing")
    assert missing_arm.record_status == "missing_record"
    for field in _RECORD_DERIVED_FIELDS:
        assert getattr(missing_arm, field) is None


# --------------------------------------------------------------------------- #
# T21 — from_json is exact-schema
# --------------------------------------------------------------------------- #


def test_from_json_is_exact_schema():
    mod = _comparisons()

    config = _config(
        ("target-mv", "cmp-ew"),
        "target-mv",
        comparators=("cmp-ew",),
    )
    table = _table(
        config,
        _collection(_record("target-mv"), _record("cmp-ew", ledger="b")),
        ("sharpe_ratio",),
    )
    text = table.to_json()

    # an unknown top-level key rejects
    parsed = json.loads(text)
    parsed["zzz_unknown_top"] = 1
    with pytest.raises(mod.ComparisonError):
        mod.ComparisonTable.from_json(
            json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        )

    # an unknown nested key rejects (inside the first list of objects)
    parsed = json.loads(text)
    nested = _first_dict_list(parsed)
    assert nested is not None
    nested[0]["zzz_unknown_nested"] = 1
    with pytest.raises(mod.ComparisonError):
        mod.ComparisonTable.from_json(
            json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        )

    # a malformed token rejects
    with pytest.raises(mod.ComparisonError):
        mod.ComparisonTable.from_json(text + "{")

    # a non-finite wire number rejects
    parsed = json.loads(text)
    slot = _first_float_slot(parsed)
    assert slot is not None
    slot[0][slot[1]] = float("nan")
    with pytest.raises(mod.ComparisonError):
        mod.ComparisonTable.from_json(
            json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        )


# --------------------------------------------------------------------------- #
# T22 — the renderer is pure, deterministic, plain text
# --------------------------------------------------------------------------- #


def test_render_table_is_pure_deterministic_plain_text():
    mod = _comparisons()

    config = _config(
        ("target-mv", "cmp-ew", "cmp-missing"),
        "target-mv",
        comparators=("cmp-ew", "cmp-missing"),
        benchmarks=("cmp-ew",),
    )
    target = _record(
        "target-mv",
        ledger="a",
        conformance=_sanctioned(),
        relations=_relations(),
    )
    cmp_ew = _record("cmp-ew", ledger="b")

    def build():
        return _table(config, _collection(target, cmp_ew), ("sharpe_ratio",))

    table = build()
    text = mod.render_table(table)
    assert text == mod.render_table(table)  # pure: same table, same text
    assert text == mod.render_table(build())  # equal tables, equal text
    assert "\x1b" not in text  # no ANSI escapes, no color

    # the shared-conditions header carries the Law 8 wire form
    conditions_wire = json.dumps(
        {"cost_rate": 0.001, "periods_per_year": 12, "universe": ["A", "B"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert conditions_wire in text

    # per-arm header values are all present for a present arm
    assert target.run_id in text
    assert target.manifest_digest in text
    assert target.paths["main"].ledger_digest in text
    assert target.conformance.verdict in text
    assert (
        "sanctioned_deviation" in text
        or len(target.conformance.deviations) > 0
        and target.conformance.deviations[0].reason in text
    )

    # a missing_record arm renders the explicit token for every
    # record-derived field — never an empty set, zero, blank, or fabrication
    assert "missing_record" in text
    assert "not_available" in text
    missing_arm = _arm(table, "cmp-missing")
    assert missing_arm.record_status == "missing_record"
    for field in _RECORD_DERIVED_FIELDS:
        assert getattr(missing_arm, field) is None


# --------------------------------------------------------------------------- #
# T23 — tables are immutable, snapshotted, and unhashable
# --------------------------------------------------------------------------- #


def test_tables_are_immutable_and_unhashable():
    duck_path = _DuckPath(
        [
            _DuckBlock(
                {"sharpe_ratio": 1.25},
                basis="net",
                periods_per_year=12,
                gamma=2.0,
                risk_free=0.001,
            )
        ],
        ledger_digest="ab12cd34" * 4,
        n_periods=4,
        final_wealth=101.0,
    )
    duck = _DuckRecord("cmp-duck", {"main": duck_path})
    config = _config(
        ("target-mv", "cmp-duck"),
        "target-mv",
        comparators=("cmp-duck",),
    )
    table = _table(config, _collection(_record("target-mv"), duck), ("sharpe_ratio",))
    frozen = table.to_json()

    with pytest.raises((AttributeError, TypeError)):
        table.schema_version = "2.0.0"  # post-construction mutation rejects
    arm = table.arms[0]
    with pytest.raises((AttributeError, TypeError)):
        arm.strategy_id = "hijacked"
    with pytest.raises(TypeError):
        hash(table)  # unhashable by design

    # later mutation of a consumed input cannot change a built table
    duck.paths["sneaky"] = duck_path
    assert table.to_json() == frozen


# --------------------------------------------------------------------------- #
# T24 — estimator provenance never enters the table
# --------------------------------------------------------------------------- #


def test_estimator_provenance_does_not_enter_the_table():
    mod = _comparisons()

    config = _config(
        ("target-mv", "cmp-ew"),
        "target-mv",
        comparators=("cmp-ew",),
    )
    bare = _collection(
        _record("target-mv", ledger="a", forecasts=()),
        _record("cmp-ew", ledger="b", forecasts=()),
    )
    loaded = _collection(
        _record("target-mv", ledger="a", forecasts=(_link("mlp-estimator"),)),
        _record("cmp-ew", ledger="b", forecasts=(_link("window-estimator"),)),
    )
    bare_table = _table(config, bare, ("sharpe_ratio",))
    loaded_table = _table(config, loaded, ("sharpe_ratio",))

    assert bare_table.to_json() == loaded_table.to_json()
    assert bare_table == loaded_table
    text = mod.render_table(loaded_table)
    assert "mlp-estimator" not in text
    assert "window-estimator" not in text
    assert "ab12cd34" not in text  # the config hash never renders either


def test_package_facade_lazily_resolves_comparisons_without_widening_all():
    """The root package must advertise and lazily resolve ``comparisons``.

    Frozen facade contract (the results/runtime lazy-export precedent): importing
    ``portlearn`` alone imports no other PortLearn module; the root
    ``__all__`` stays untouched; ``dir(portlearn)`` advertises the facade;
    and attribute access lazily resolves to the exact module object
    installed at ``sys.modules['portlearn.comparisons']`` on every access.
    The proof runs in a fresh interpreter so the parent process's already
    populated ``sys.modules`` cannot mask a missing facade.
    """
    probe = (
        "import sys\n"
        "import portlearn\n"
        "assert 'portlearn.comparisons' not in sys.modules, (\n"
        "    'eager import: portlearn.comparisons reachable before access'\n"
        ")\n"
        "assert 'comparisons' not in portlearn.__all__, (\n"
        "    'facade widened __all__'\n"
        ")\n"
        "assert 'comparisons' in dir(portlearn), (\n"
        "    'comparisons missing from dir(portlearn)'\n"
        ")\n"
        "lazy = portlearn.comparisons\n"
        "import sys as _sys\n"
        "assert lazy is _sys.modules['portlearn.comparisons'], (\n"
        "    'facade did not resolve to the installed module object'\n"
        ")\n"
        "assert portlearn.comparisons is lazy, (\n"
        "    'repeated attribute access is not identical'\n"
        ")\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "the lazy comparisons facade is missing from the root package:\n"
        f"returncode={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
