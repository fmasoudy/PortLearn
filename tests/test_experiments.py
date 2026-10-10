"""Experiment-declaration contract tests: the experiment-configuration declaration (§8 floors).

The surface under test is the ``portlearn.experiments`` public module: the
module-owned ``ConfigurationError``, the process-local live binding
``StrategyRegistration``, the canonically serializable declaration
``ExperimentConfiguration``, the declarative ``ExperimentSelection`` and
its four selection constructors (``select_all`` / ``select_none`` /
``select_single`` / ``select_subset``), and the uniform results collection
``ExperimentResults`` — exactly as frozen by the specification approval
(specification §§4–8a at accepted candidate ``5108ac0``, accepted experiment-boundary decision
Accepted; the B1/B2/B3 splits of the 2026-09-28 HOLD cure).

Floors pinned here (22 behavior floors, declaration §8): the exact module
surface (T1); the lazy-facade exposure and unchanged root ``__all__``
(T2); error-type ownership (T3); registration identity with verbatim
declaration carriage (T4); no identity derivation (T5); the four-role
vocabulary with benchmark as comparator metadata only (T6); structural
cardinality validity including the lawful target-only declaration (T7);
benchmark metadata membership (T8); the single shared-conditions
declaration (T9); the results §5.8a settings grammar (T10); the top-level
``risk_aversion`` lookup-key shape (T11); the verbatim results
``ResultRecord.strategy_id`` linkage law (T12); the uniform
``ExperimentResults`` shape for zero/one/many stand-ins (T13);
substitution invariance of the declaration bytes (T14); the four
selection modes (T15); selection invariance over the configuration
(T16); the canonical deterministic byte law with no live-object token
(T17); strict canonical-only deserialization (T18); deep immutability
with caller-container isolation (T19); the static-validation boundary
census (T20); the declaration/registration split (T21); and the runtime
runtime isolated from the declarative shapes — the one reserved
``Experiment`` object owning ``run``, never the declaration or the
selection, never a free-function executor (T22, runtime successor).

Per §8 every run-output stand-in is a results ``ResultRecord`` rebuilt
through the frozen public constructors (``RunManifest``, ``build_ledger``,
``core_metrics``, ``ConformanceRecord``) — never hand-inlined literals.
Canonical-byte laws are asserted schema-agnostically (strict-JSON
re-serialization identity, sorted/compact/ASCII properties, substring
splices) and derived in this module from the §5 law text alone.

Constructor-route readings pinned here (the spec freezes the nine-name
module surface and the declaration laws, but not the serialization
method route): canonical serialization lives as ``ExperimentConfiguration``
instance/class methods — ``canonical_text()`` and ``from_canonical_text()``
— because the T1/T20 exact surface census forbids module-level
serialization functions (the §14a recovery-probe vocabulary, moved onto
the declaration object).  Binding declared ids to live registrations is
runtime composition (D11a) and has no declaration surface.

RED-first: the module is absent at this tranche, so every
``portlearn.experiments`` import happens INSIDE the tests and builders —
pytest reports the causal absence (``ModuleNotFoundError: No module
named 'portlearn.experiments'``) for every target node instead of
aborting collection.  The facade-message node is the one lawful second
causal class (the root facade does not yet name ``experiments``), and
the frozen-seam controls stay green throughout.  Floor names are
behavior names; the §8 floor each node pins is recorded in its
docstring.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from portlearn.costs import Proportional
from portlearn.interfaces import DecisionResult, InformationSet, PortfolioDecision
from portlearn.ledger import ExactFillAccounting, build_ledger
from portlearn.manifest import RunManifest
from portlearn.metrics import core_metrics
from portlearn.timing import FutureInformationError, NaiveTimestampError
from portlearn.turnover import two_sided
from portlearn.weights import (
    PortfolioWeights,
    WeightConstraints,
    WeightState,
    require_valid_target,
)

# --------------------------------------------------------------------------- #
# The module under test — imported inside every node so collection succeeds   #
# while the module is absent (the causal RED signal).                         #
# --------------------------------------------------------------------------- #


@functools.cache
def _experiments() -> Any:
    """Import ``portlearn.experiments`` on demand (RED: ModuleNotFoundError)."""
    return importlib.import_module("portlearn.experiments")


@functools.cache
def _results() -> Any:
    """Import the frozen ``portlearn.results`` seam on demand (green)."""
    return importlib.import_module("portlearn.results")


# --------------------------------------------------------------------------- #
# Generation isolation (harness plumbing — no assertion or floor changes).    #
# --------------------------------------------------------------------------- #
#
# ``tests/test_package_contract.py`` runs a registry-purge probe that deletes
# every ``portlearn``/``portlearn.*`` entry from ``sys.modules`` and re-imports
# the root package fresh.  When this module is collected before that probe
# runs, the frozen-seam names bound at import time above (``RunManifest``,
# ``build_ledger``, ...) belong to the collection-time generation, while the
# lazily imported modules bind post-purge generations.  The fixture below
# pins the collection-time generation back for the duration of this module's
# nodes and restores the ambient registry exactly afterwards, so the frozen
# seams, the results stand-in records, and the module under test all share one
# coherent generation.  Mirrors the ``tests/test_results.py`` precedent.


def _portlearn_generation() -> dict[str, Any]:
    """Snapshot the registered ``portlearn`` root/submodule generation."""
    return {
        name: module
        for name, module in sys.modules.items()
        if name == "portlearn" or name.startswith("portlearn.")
    }


_COLLECTION_GENERATION = _portlearn_generation()


@pytest.fixture(autouse=True, scope="module")
def _one_coherent_module_generation():
    """Run this module's nodes against one coherent module generation."""
    ambient = _portlearn_generation()
    sys.modules.update(_COLLECTION_GENERATION)
    try:
        yield
    finally:
        for name in sorted(_portlearn_generation()):
            del sys.modules[name]
        sys.modules.update(ambient)


# --------------------------------------------------------------------------- #
# Live component stand-ins — opaque finance components; never executed.       #
# --------------------------------------------------------------------------- #


class _LiveComponent:
    """A live strategy component exposing the Strategy surface (decide)."""

    def decide(self, context: Any) -> Any:  # pragma: no cover
        raise AssertionError("the declaration surface never executes a strategy")


class _OtherLiveComponent:
    """A second, structurally different live component under the same ids."""

    def decide(self, context: Any) -> Any:  # pragma: no cover
        raise AssertionError("the declaration surface never executes a strategy")


class _DecidelessComponent:
    """A component lacking the Strategy surface (the D6 negative probe)."""


# --------------------------------------------------------------------------- #
# Frozen-seam fixtures — rebuilt through the public results constructors (§8).    #
# --------------------------------------------------------------------------- #

_TZ_PLUS_TEN = timezone(timedelta(hours=10))
_MONTHS = tuple(datetime(2026, m, 1, tzinfo=UTC) for m in range(1, 6))
_CREATED_AT = datetime(2026, 9, 28, 9, 30, tzinfo=_TZ_PLUS_TEN)
_REBALANCE_FEB = PortfolioDecision(
    decision_time=_MONTHS[1],
    execution_time=_MONTHS[1],
    target_weights={"A": 0.5, "B": 0.5},
)


@functools.cache
def _manifest() -> RunManifest:
    """The consumed run identity (the frozen public constructor)."""
    return RunManifest(
        run_id="m43-red-run",
        created_at=datetime(2026, 9, 28, 8, 15, tzinfo=_TZ_PLUS_TEN),
        python_version="3.13.5",
        package_version="0.0.1.dev3",
        dependency_pins={"scipy": "1.14.0"},
        commands=["portlearn-run m43"],
    )


@functools.cache
def _path_two_assets():
    """A two-asset, four-period path with one boundary rebalance."""
    return build_ledger(
        initial_weights={"A": 0.4, "B": 0.6},
        accounting_instants=_MONTHS,
        growth_factors={
            _MONTHS[0]: {"A": 0.995, "B": 1.01},
            _MONTHS[1]: {"A": 0.995, "B": 1.011},
            _MONTHS[2]: {"A": 0.995, "B": 1.012},
            _MONTHS[3]: {"A": 0.995, "B": 1.013},
        },
        decisions=[_REBALANCE_FEB],
        cost_model=Proportional(rate=0.001, turnover=two_sided),
        initial_wealth=100.0,
    )


@functools.cache
def _path_single_asset():
    """A single-asset, three-period path with no executions."""
    return build_ledger(
        initial_weights={"S": 1.0},
        accounting_instants=_MONTHS[:4],
        growth_factors={
            _MONTHS[0]: {"S": 1.10},
            _MONTHS[1]: {"S": 0.50},
            _MONTHS[2]: {"S": 1.20},
        },
        decisions=[],
        cost_model=Proportional(rate=0.001, turnover=two_sided),
        initial_wealth=1.0,
    )


def _block_two_assets_net():
    results = _results()
    summary = core_metrics(
        _path_two_assets(), periods_per_year=12, gamma=2.0, risk_free=0.001
    )
    return results.MetricsBlock(
        summary, basis="net", periods_per_year=12, gamma=2.0, risk_free=0.001
    )


def _block_single_asset_net():
    results = _results()
    summary = core_metrics(
        _path_single_asset(), periods_per_year=12, gamma=5.0, risk_free=0.002
    )
    return results.MetricsBlock(
        summary, basis="net", periods_per_year=12, gamma=5.0, risk_free=0.002
    )


def _path_result_two_assets():
    results = _results()
    return results.PathResult(_path_two_assets(), (_block_two_assets_net(),))


def _path_result_single_asset():
    results = _results()
    return results.PathResult(_path_single_asset(), (_block_single_asset_net(),))


def _conformance():
    results = _results()
    return results.ConformanceRecord(
        declared={"cost_rate": 0.001, "universe": ("A", "B")},
        effective={"universe": ("A", "B"), "cost_rate": 0.001},
        deviations=(),
    )


def _record(strategy_id: str):
    """One results ``ResultRecord`` run-output stand-in (public constructors)."""
    results = _results()
    return results.ResultRecord(
        _manifest(),
        strategy_id=strategy_id,
        paths={"ew": _path_result_two_assets(), "iv": _path_result_single_asset()},
        forecasts=(),
        conformance=_conformance(),
        created_at=_CREATED_AT,
    )


# --------------------------------------------------------------------------- #
# Declaration helpers — every one imports the module on demand.               #
# --------------------------------------------------------------------------- #


def _registration(strategy_id: str, component: Any | None = None):
    experiments = _experiments()
    return experiments.StrategyRegistration(
        strategy_id=strategy_id,
        strategy=component if component is not None else _LiveComponent(),
    )


def _config(
    *,
    strategy_ids=("min-var", "equal-weight"),
    target="min-var",
    comparators=("equal-weight",),
    ablations=(),
    robustness_checks=(),
    benchmarks=("equal-weight",),
    conditions=None,
):
    """Build a declaration; ``conditions=None`` omits the keyword entirely
    (the documented empty default of D8a)."""
    experiments = _experiments()
    kwargs: dict[str, Any] = {
        "strategy_ids": strategy_ids,
        "target": target,
        "comparators": comparators,
        "ablations": ablations,
        "robustness_checks": robustness_checks,
        "benchmarks": benchmarks,
    }
    if conditions is not None:
        kwargs["conditions"] = conditions
    return experiments.ExperimentConfiguration(**kwargs)


def _config_default_conditions(**overrides):
    """A declaration carrying the default (empty) conditions container."""
    experiments = _experiments()
    kwargs: dict[str, Any] = {
        "strategy_ids": ("min-var", "equal-weight"),
        "target": "min-var",
        "comparators": ("equal-weight",),
    }
    kwargs.update(overrides)
    return experiments.ExperimentConfiguration(**kwargs)


def _canonical(config: Any) -> str:
    """The declaration's canonical text (the frozen serialization route)."""
    return config.canonical_text()


# Schema-agnostic canonical-byte laws (§5 Law 8), derived from stdlib alone.


def _reserialize(payload: Any, **dumps_kwargs: Any) -> str:
    dumps_kwargs.setdefault("sort_keys", True)
    dumps_kwargs.setdefault("ensure_ascii", True)
    dumps_kwargs.setdefault("allow_nan", False)
    return json.dumps(payload, **dumps_kwargs)


def _first_nested_mapping_text(payload: Any) -> str | None:
    """Canonical text of the first mapping nested inside a mapping value."""
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, dict) and value:
                return json.dumps(value, sort_keys=True, separators=(",", ":"))
            found = _first_nested_mapping_text(value)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _first_nested_mapping_text(value)
            if found is not None:
                return found
    return None


# --------------------------------------------------------------------------- #
# T1 — module surface (runtime successor: nine declarative + four runtime)       #
# --------------------------------------------------------------------------- #


def test_module_exports_exactly_the_pinned_surface() -> None:
    """Floor T1 (runtime successor) — the module exports exactly thirteen names.

    The nine declaration declarative names keep their frozen order and the four
    approved runtime names (``ExecutionError``, ``DecisionSchedule``,
    ``ExperimentConditions``, ``Experiment``) follow in the approved
    order; the census stays exact, so a surprise export, a missing
    export, or a reordered one still fails.
    """
    module = _experiments()
    assert module.__all__ == [
        "ConfigurationError",
        "StrategyRegistration",
        "ExperimentConfiguration",
        "ExperimentSelection",
        "ExperimentResults",
        "select_all",
        "select_none",
        "select_single",
        "select_subset",
        "ExecutionError",
        "DecisionSchedule",
        "ExperimentConditions",
        "Experiment",
    ]
    for name in module.__all__:
        assert getattr(module, name) is not None, f"{name} does not resolve"


# --------------------------------------------------------------------------- #
# T2 — lazy facade exposure                                                    #
# --------------------------------------------------------------------------- #


def test_facade_lazily_exposes_experiments_and_dir_lists_it() -> None:
    """Floor T2 — the lazy facade resolves the module; ``__dir__`` lists it."""
    direct = _experiments()
    import portlearn as pl
    from portlearn import experiments as via_facade

    assert pl.experiments is direct
    assert via_facade is direct
    assert "experiments" in dir(pl)


def test_facade_error_message_names_experiments_among_public_modules() -> None:
    """Floor T2 — the unknown-attribute error enumerates ``experiments``."""
    _experiments()
    import portlearn as pl

    with pytest.raises(AttributeError, match="experiments"):
        _discard = pl.Definitely_not_a_portlearn_surface  # noqa: B950


def test_root_package_all_is_unchanged_control() -> None:
    """Always-green control — root ``__all__`` untouched by the facade law."""
    import portlearn as pl

    assert pl.__all__ == ["__version__", "data", "weights"]


# --------------------------------------------------------------------------- #
# T3 — error-type ownership                                                    #
# --------------------------------------------------------------------------- #


def test_configuration_error_is_module_owned_value_error() -> None:
    """Floor T3 — ``ConfigurationError`` is module-owned, a ValueError."""
    experiments = _experiments()
    assert issubclass(experiments.ConfigurationError, ValueError)
    assert experiments.ConfigurationError.__module__ == "portlearn.experiments"


# --------------------------------------------------------------------------- #
# T4 — registration identity                                                   #
# --------------------------------------------------------------------------- #


def test_registration_admits_explicit_id_and_carries_it_verbatim() -> None:
    """Floor T4 — an explicit non-blank id admits and is carried verbatim."""
    experiments = _experiments()
    reg = experiments.StrategyRegistration(
        strategy_id="min-var", strategy=_LiveComponent()
    )
    assert reg.strategy_id == "min-var"
    assert reg.strategy is not None
    config = _config()
    assert "min-var" in config.strategy_ids
    assert "equal-weight" in config.strategy_ids


def test_registration_blank_or_whitespace_id_rejects() -> None:
    """Floor T4 — blank/whitespace registration ids reject fail-closed."""
    experiments = _experiments()
    for bad in ("", "   ", "\t\n"):
        with pytest.raises(experiments.ConfigurationError):
            experiments.StrategyRegistration(strategy_id=bad, strategy=_LiveComponent())


def test_registration_requires_strategy_id_argument() -> None:
    """Floor T4 — ``strategy_id`` has no default (missing is a TypeError)."""
    experiments = _experiments()
    with pytest.raises(TypeError):
        experiments.StrategyRegistration(strategy=_LiveComponent())


def test_registration_rejects_unknown_keywords() -> None:
    """Floors T4/T6 — the exact-field-set law: unknown kwargs are TypeErrors."""
    experiments = _experiments()
    with pytest.raises(TypeError):
        experiments.StrategyRegistration(
            strategy_id="min-var", strategy=_LiveComponent(), role="comparator"
        )
    with pytest.raises(TypeError):
        experiments.StrategyRegistration(
            strategy_id="min-var",
            strategy=_LiveComponent(),
            conditions={"risk_aversion": 1.0},
        )


def test_declared_duplicate_ids_reject() -> None:
    """Floor T4 — duplicate ids across registrations reject (no suffixing)."""
    experiments = _experiments()
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentConfiguration(
            strategy_ids=("min-var", "equal-weight", "min-var"),
            target="min-var",
        )


def test_declaration_id_survives_roundtrip_verbatim() -> None:
    """Floor T4 — the declaration id, not the component, survives the trip."""
    experiments = _experiments()
    _registration("min-var", _LiveComponent())
    config = _config()
    text = _canonical(config)
    reborn = experiments.ExperimentConfiguration.from_canonical_text(text)
    assert reborn.strategy_ids == config.strategy_ids
    assert "min-var" in reborn.strategy_ids
    for element in reborn.strategy_ids:
        assert isinstance(element, str)


# --------------------------------------------------------------------------- #
# T5 — no identity derivation                                                  #
# --------------------------------------------------------------------------- #


def test_same_component_under_two_ids_lawful_and_distinct() -> None:
    """Floor T5 — one component may occupy two ids; the ids stay distinct."""
    component = _LiveComponent()
    reg_min = _registration("min-var", component)
    reg_ew = _registration("equal-weight", component)
    assert reg_min.strategy_id == "min-var"
    assert reg_ew.strategy_id == "equal-weight"
    assert reg_min.strategy is reg_ew.strategy is component


def test_component_class_never_changes_admitted_ids() -> None:
    """Floor T5 — different component classes never drift an admitted id."""
    experiments = _experiments()
    first = experiments.StrategyRegistration(
        strategy_id="min-var", strategy=_LiveComponent()
    )
    second = experiments.StrategyRegistration(
        strategy_id="min-var", strategy=_OtherLiveComponent()
    )
    assert first.strategy_id == second.strategy_id == "min-var"
    text = _canonical(_config())
    for token in ("_LiveComponent", "_OtherLiveComponent", "object at 0x"):
        assert token not in text


# --------------------------------------------------------------------------- #
# T6 — role vocabulary                                                         #
# --------------------------------------------------------------------------- #


def test_role_assignment_uses_documented_fields_only() -> None:
    """Floor T6 — roles enter only through the four documented fields."""
    experiments = _experiments()
    with pytest.raises(TypeError):
        experiments.ExperimentConfiguration(
            strategy_ids=("min-var", "equal-weight"),
            target="min-var",
            comparators=("equal-weight",),
            roles={"equal-weight": "comparator"},
        )
    with pytest.raises(TypeError):
        experiments.ExperimentConfiguration(
            strategy_ids=("min-var",),
            target="min-var",
            role="target",
        )


def test_registration_carries_no_role_attribute() -> None:
    """Floor T6 — registrations never carry roles (attribute path, M1)."""
    reg = _registration("min-var")
    assert not hasattr(reg, "role")
    assert not hasattr(reg, "roles")
    with pytest.raises(dataclasses.FrozenInstanceError):
        reg.role = "comparator"  # type: ignore[misc]


def test_no_benchmark_role_token_admitted() -> None:
    """Floor T6 — ``benchmark`` is not a role token on any surface."""
    experiments = _experiments()
    with pytest.raises(TypeError):
        experiments.ExperimentConfiguration(
            strategy_ids=("min-var", "equal-weight"),
            target="min-var",
            comparators=("equal-weight",),
            benchmark="equal-weight",
        )
    with pytest.raises(TypeError):
        experiments.ExperimentConfiguration(
            strategy_ids=("min-var", "equal-weight"),
            target="min-var",
            comparators=("equal-weight",),
            role_benchmarks=("equal-weight",),
        )
    reg = _registration("equal-weight")
    assert not hasattr(reg, "benchmark")


def test_four_roles_assignable_through_documented_fields() -> None:
    """Floor T6 — all four roles are assignable; ids stay role-free."""
    config = _config(
        strategy_ids=("min-var", "equal-weight", "drop-cost", "alt-window", "spare"),
        target="min-var",
        comparators=("equal-weight",),
        ablations=("drop-cost",),
        robustness_checks=("alt-window",),
        benchmarks=("equal-weight",),
    )
    assert config.target == "min-var"
    assert config.comparators == ("equal-weight",)
    assert config.ablations == ("drop-cost",)
    assert config.robustness_checks == ("alt-window",)
    assert "spare" in config.strategy_ids


# --------------------------------------------------------------------------- #
# T7 — cardinality (structural validity)                                       #
# --------------------------------------------------------------------------- #


def test_sequence_target_rejects() -> None:
    """Floor T7 — a list-valued target rejects (exactly one target)."""
    experiments = _experiments()
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentConfiguration(
            strategy_ids=("min-var", "equal-weight"),
            target=["min-var"],  # type: ignore[arg-type]
            comparators=("equal-weight",),
        )


def test_missing_or_none_target_rejects() -> None:
    """Floor T7 — zero targets reject: omitted is a TypeError, None a config
    error."""
    experiments = _experiments()
    with pytest.raises(TypeError):
        experiments.ExperimentConfiguration(
            strategy_ids=("min-var",),
            comparators=(),
        )
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentConfiguration(
            strategy_ids=("min-var",),
            target=None,  # type: ignore[arg-type]
        )


def test_empty_registry_rejects() -> None:
    """Floor T7 — a declaration with zero registered ids rejects."""
    experiments = _experiments()
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentConfiguration(strategy_ids=(), target="min-var")


def test_target_only_zero_comparators_admits_and_canonicalizes() -> None:
    """Floor T7 (B1) — a target-only declaration is lawful structural state."""
    experiments = _experiments()
    config = experiments.ExperimentConfiguration(
        strategy_ids=("min-var",), target="min-var"
    )
    assert config.comparators == ()
    assert _canonical(config)


def test_unassigned_registered_ids_are_lawful() -> None:
    """Floor T7 — registered ids carrying no role admit (selection needs
    them)."""
    config = _config(
        strategy_ids=("min-var", "equal-weight", "spare-arm"),
        target="min-var",
        comparators=("equal-weight",),
        benchmarks=("equal-weight",),
    )
    assert "spare-arm" in config.strategy_ids


def test_target_id_repeated_in_comparators_rejects() -> None:
    """Floor T7 — one role per id: the target may not double as comparator."""
    experiments = _experiments()
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentConfiguration(
            strategy_ids=("min-var", "equal-weight"),
            target="min-var",
            comparators=("min-var",),
        )


def test_id_in_two_non_target_roles_rejects() -> None:
    """Floor T7 — one role per id across comparator/ablation/robustness."""
    experiments = _experiments()
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentConfiguration(
            strategy_ids=("min-var", "equal-weight", "third"),
            target="min-var",
            comparators=("equal-weight",),
            ablations=("equal-weight",),
        )
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentConfiguration(
            strategy_ids=("min-var", "equal-weight", "third"),
            target="min-var",
            robustness_checks=("third",),
            ablations=("third",),
        )


# --------------------------------------------------------------------------- #
# T8 — benchmark metadata                                                      #
# --------------------------------------------------------------------------- #


def test_benchmark_naming_declared_comparator_admits() -> None:
    """Floor T8 — a benchmark naming a declared comparator admits."""
    config = _config(benchmarks=("equal-weight",))
    assert config.benchmarks == ("equal-weight",)


def test_benchmark_naming_target_rejects() -> None:
    """Floor T8 — the target is not comparator metadata."""
    experiments = _experiments()
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentConfiguration(
            strategy_ids=("min-var", "equal-weight"),
            target="min-var",
            comparators=("equal-weight",),
            benchmarks=("min-var",),
        )


def test_benchmark_naming_ablation_or_unregistered_rejects() -> None:
    """Floor T8 — benchmarks are comparator members only."""
    experiments = _experiments()
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentConfiguration(
            strategy_ids=("min-var", "equal-weight", "drop-cost"),
            target="min-var",
            comparators=("equal-weight",),
            ablations=("drop-cost",),
            benchmarks=("drop-cost",),
        )
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentConfiguration(
            strategy_ids=("min-var", "equal-weight"),
            target="min-var",
            comparators=("equal-weight",),
            benchmarks=("not-registered",),
        )


# --------------------------------------------------------------------------- #
# T9 — single shared-conditions declaration                                    #
# --------------------------------------------------------------------------- #


def test_configuration_carries_one_conditions_container() -> None:
    """Floor T9 — one conditions mapping lives on the configuration."""
    config = _config(conditions={"risk_aversion": 2.5, "universe": ("A", "B")})
    assert config.conditions["risk_aversion"] == 2.5
    assert config.conditions["universe"] == ("A", "B")


def test_per_strategy_conditions_kwarg_rejects() -> None:
    """Floor T9 (D5) — no per-strategy conditions field exists."""
    experiments = _experiments()
    with pytest.raises(TypeError):
        experiments.ExperimentConfiguration(
            strategy_ids=("min-var", "equal-weight"),
            target="min-var",
            comparators=("equal-weight",),
            per_strategy_conditions={"min-var": {"risk_aversion": 1.0}},
        )


def test_condition_insertion_order_never_content() -> None:
    """Floor T9 — equal conditions in different orders: same snapshot, bytes."""
    experiments = _experiments()
    left = _config(conditions={"risk_aversion": 2.5, "universe": ("A", "B"), "n": 3})
    right = experiments.ExperimentConfiguration(
        strategy_ids=("min-var", "equal-weight"),
        target="min-var",
        comparators=("equal-weight",),
        benchmarks=("equal-weight",),
        conditions={"n": 3, "universe": ("A", "B"), "risk_aversion": 2.5},
    )
    assert left.conditions == right.conditions
    assert _canonical(left) == _canonical(right)


def test_conditions_none_rejects_and_empty_is_lawful() -> None:
    """Floor T9 (D8a) — ``conditions=None`` rejects; empty/default admit."""
    experiments = _experiments()
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentConfiguration(
            strategy_ids=("min-var",),
            target="min-var",
            conditions=None,
        )
    empty = _config_default_conditions(conditions={})
    default = _config_default_conditions()
    assert empty.conditions == default.conditions
    assert _canonical(empty) == _canonical(default)
    assert _canonical(empty).endswith("{}") or '"conditions":{}' in _canonical(empty)


# --------------------------------------------------------------------------- #
# T10 — the results §5.8a settings grammar                                        #
# --------------------------------------------------------------------------- #


def test_lawful_condition_kinds_admit_and_roundtrip() -> None:
    """Floor T10 — every §5.8a-lawful kind survives canonical round-trip."""
    experiments = _experiments()
    conditions = {
        "none_value": None,
        "flag": True,
        "count": 7,
        "rate": 0.25,
        "label": "base",
        "universe": ("A", "B"),
        "arbitrary nonblank key": {"depth": 2, "inner": (1, 2.5, "x", False, None)},
        "weights": (0.5, 0.25),
    }
    config = _config(conditions=conditions)
    text = _canonical(config)
    reborn = experiments.ExperimentConfiguration.from_canonical_text(text)
    assert reborn.conditions == config.conditions
    assert _canonical(reborn) == text


def test_unlawful_condition_values_reject() -> None:
    """Floor T10 — nonfinite floats, bytes, bad keys, objects reject."""
    experiments = _experiments()
    unlawful = (
        {"nan": float("nan")},
        {"inf": float("inf")},
        {"blob": b"bytes"},
        {"": 1},
        {"   ": 1},
        {"nested": {"": 1}},
        {"obj": object()},
    )
    for conditions in unlawful:
        with pytest.raises(experiments.ConfigurationError):
            experiments.ExperimentConfiguration(
                strategy_ids=("min-var",),
                target="min-var",
                conditions=conditions,
            )


def test_signed_zero_unifies_in_canonical_bytes() -> None:
    """Floor T10 — ``0.0`` and ``-0.0`` produce identical canonical bytes."""
    left = _config(conditions={"rate": 0.0, "n": 1})
    right = _config(conditions={"rate": -0.0, "n": 1})
    assert _canonical(left) == _canonical(right)


# --------------------------------------------------------------------------- #
# T11 — the risk-aversion lookup key                                           #
# --------------------------------------------------------------------------- #


def test_top_level_risk_aversion_finite_number_admits() -> None:
    """Floor T11 — a finite non-bool top-level ``risk_aversion`` admits."""
    for value in (2.5, 3, 0.0):
        config = _config(conditions={"risk_aversion": value})
        assert config.conditions["risk_aversion"] == value


def test_top_level_risk_aversion_bool_and_string_reject() -> None:
    """Floor T11 — booleans and strings reject at the top level."""
    experiments = _experiments()
    for bad in (True, False, "2.5"):
        with pytest.raises(experiments.ConfigurationError):
            experiments.ExperimentConfiguration(
                strategy_ids=("min-var",),
                target="min-var",
                conditions={"risk_aversion": bad},
            )


def test_nested_risk_aversion_admits_never_the_lookup() -> None:
    """Floor T11 — a nested occurrence is lawful, never the decision lookup."""
    config = _config(conditions={"family": {"risk_aversion": 3.0}})
    assert config.conditions["family"]["risk_aversion"] == 3.0


# --------------------------------------------------------------------------- #
# T12 — results identity linkage                                                  #
# --------------------------------------------------------------------------- #


def test_result_records_link_verbatim_to_declared_ids() -> None:
    """Floor T12 — record ids are verbatim registry ids (D4a linkage law)."""
    record_min = _record("min-var")
    record_ew = _record("equal-weight")
    config = _config()
    for record in (record_min, record_ew):
        assert record.strategy_id in config.strategy_ids
    assert "min-var" in config.strategy_ids


def test_linkage_violation_is_detectable_without_aliasing() -> None:
    """Floor T12 — a non-containing registry is a violation: no case-folding,
    no renaming."""
    experiments = _experiments()
    record = _record("min-var")
    other = experiments.ExperimentConfiguration(
        strategy_ids=("equal-weight", "inverse-vol"),
        target="inverse-vol",
        comparators=("equal-weight",),
    )
    assert record.strategy_id not in other.strategy_ids
    declared = _config().strategy_ids
    for alias in ("MIN-VAR", "Min-Var", "min_var", "min-var-2"):
        assert alias not in other.strategy_ids
        assert alias not in declared


# --------------------------------------------------------------------------- #
# T13 — uniform results shape                                                  #
# --------------------------------------------------------------------------- #


def test_results_admit_zero_one_many_same_type() -> None:
    """Floor T13 — zero, one, many records: one type, never bare/None."""
    experiments = _experiments()
    empty = experiments.ExperimentResults(records=())
    one = experiments.ExperimentResults(records=(_record("min-var"),))
    many = experiments.ExperimentResults(
        records=(_record("min-var"), _record("equal-weight"))
    )
    assert type(empty) is type(one) is type(many) is experiments.ExperimentResults
    assert len(empty) == 0
    assert len(one) == 1
    assert len(many) == 2


def test_results_order_canonically_by_strategy_id() -> None:
    """Floor T13 — canonical ``strategy_id`` ordering regardless of input."""
    experiments = _experiments()
    results = experiments.ExperimentResults(
        records=(_record("min-var"), _record("equal-weight"), _record("alpha-tilt"))
    )
    assert [record.strategy_id for record in results] == [
        "alpha-tilt",
        "equal-weight",
        "min-var",
    ]


def test_results_reject_duplicate_strategy_ids() -> None:
    """Floor T13 (M8) — duplicate result ids reject fail-closed."""
    experiments = _experiments()
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentResults(records=(_record("min-var"), _record("min-var")))


def test_results_len_iteration_and_record_identity() -> None:
    """Floor T13 — iteration and ``len`` work; records are the stand-ins."""
    experiments = _experiments()
    records = (_record("min-var"), _record("equal-weight"))
    results = experiments.ExperimentResults(records=records)
    collected = list(results)
    assert len(results) == len(collected) == 2
    assert collected[0].strategy_id == "equal-weight"
    assert collected[0] in records


def test_results_reject_blank_or_missing_strategy_ids() -> None:
    """Floor T13 — records without a non-blank ``strategy_id`` reject."""
    experiments = _experiments()

    class _Blank:
        strategy_id = "   "

    class _Missing:
        pass

    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentResults(records=(_Blank(),))
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentResults(records=(_Missing(),))


# --------------------------------------------------------------------------- #
# T14 — substitution invariance                                                #
# --------------------------------------------------------------------------- #


def test_substitution_leaves_declaration_byte_identical() -> None:
    """Floor T14 (D12a) — re-registration/swap never changes the declaration."""
    config = _config(conditions={"risk_aversion": 2.5})
    before = _canonical(config)
    _registration("min-var", _LiveComponent())
    _registration("equal-weight", _OtherLiveComponent())
    swapped = (
        _registration("min-var", _OtherLiveComponent()),
        _registration("equal-weight", _LiveComponent()),
    )
    assert swapped[0].strategy is not swapped[1].strategy
    assert _canonical(config) == before
    assert config.conditions["risk_aversion"] == 2.5
    for token in ("_LiveComponent", "_OtherLiveComponent", "object at 0x"):
        assert token not in before


# --------------------------------------------------------------------------- #
# T15 — selection modes                                                        #
# --------------------------------------------------------------------------- #


def test_four_selection_modes_produce_exact_ids() -> None:
    """Floor T15 — full/none/single/subset name registered ids verbatim."""
    experiments = _experiments()
    config = _config(
        strategy_ids=("min-var", "equal-weight", "spare-arm"),
        target="min-var",
        comparators=("equal-weight",),
        benchmarks=("equal-weight",),
    )
    full = experiments.select_all(config)
    none = experiments.select_none(config)
    single = experiments.select_single(config, "min-var")
    subset = experiments.select_subset(config, ("spare-arm", "equal-weight"))
    assert full.mode == "full"
    assert full.strategy_ids == ("equal-weight", "min-var", "spare-arm")
    assert none.mode == "none"
    assert none.strategy_ids == ()
    assert single.mode == "single"
    assert single.strategy_ids == ("min-var",)
    assert subset.mode == "subset"
    assert subset.strategy_ids == ("equal-weight", "spare-arm")


def test_subset_duplicates_and_empty_reject() -> None:
    """Floor T15 — duplicate subset names reject; empty is the none mode."""
    experiments = _experiments()
    config = _config()
    with pytest.raises(experiments.ConfigurationError):
        experiments.select_subset(config, ("min-var", "min-var"))
    with pytest.raises(experiments.ConfigurationError):
        experiments.select_subset(config, ())


def test_unknown_selection_names_reject() -> None:
    """Floor T15 — unknown names reject in single and subset modes."""
    experiments = _experiments()
    config = _config()
    with pytest.raises(experiments.ConfigurationError):
        experiments.select_single(config, "not-registered")
    with pytest.raises(experiments.ConfigurationError):
        experiments.select_subset(config, ("equal-weight", "not-registered"))


def test_single_select_of_unassigned_id_admits() -> None:
    """Floor T15 — roles are irrelevant to selection (unassigned id admits)."""
    experiments = _experiments()
    config = _config(
        strategy_ids=("min-var", "equal-weight", "spare-arm"),
        target="min-var",
        comparators=("equal-weight",),
        benchmarks=("equal-weight",),
    )
    selection = experiments.select_single(config, "spare-arm")
    assert selection.strategy_ids == ("spare-arm",)


def test_selection_object_validates_mode_and_ids() -> None:
    """Floor T15 — ``ExperimentSelection`` admits exactly the four tokens."""
    experiments = _experiments()
    selection = experiments.ExperimentSelection(
        mode="subset", strategy_ids=("spare-arm", "equal-weight")
    )
    assert selection.mode == "subset"
    assert selection.strategy_ids == ("equal-weight", "spare-arm")
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentSelection(mode="everything", strategy_ids=("a",))
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentSelection(mode="subset", strategy_ids=("a", "a"))
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentSelection(mode="single", strategy_ids=("  ",))


# --------------------------------------------------------------------------- #
# T16 — selection invariance                                                   #
# --------------------------------------------------------------------------- #


def test_selection_never_mutates_the_configuration() -> None:
    """Floor T16 — selections leave canonical bytes and conditions intact."""
    experiments = _experiments()
    config = _config(
        strategy_ids=("min-var", "equal-weight", "spare-arm"),
        target="min-var",
        comparators=("equal-weight",),
        benchmarks=("equal-weight",),
        conditions={"risk_aversion": 2.5},
    )
    before = _canonical(config)
    experiments.select_all(config)
    experiments.select_none(config)
    experiments.select_single(config, "spare-arm")
    experiments.select_subset(config, ("equal-weight", "spare-arm"))
    assert _canonical(config) == before
    assert config.conditions["risk_aversion"] == 2.5
    assert config.target == "min-var"
    assert config.comparators == ("equal-weight",)


def test_selection_returns_selection_never_bare_record_or_none() -> None:
    """Floor T16 — every constructor returns ``ExperimentSelection`` data."""
    experiments = _experiments()
    config = _config()
    for selection in (
        experiments.select_all(config),
        experiments.select_none(config),
        experiments.select_single(config, "min-var"),
        experiments.select_subset(config, ("equal-weight",)),
    ):
        assert type(selection) is experiments.ExperimentSelection
        assert selection is not None
        assert isinstance(selection.strategy_ids, tuple)


# --------------------------------------------------------------------------- #
# T17 — canonical form                                                         #
# --------------------------------------------------------------------------- #


def test_canonical_bytes_deterministic_order_blind() -> None:
    """Floor T17 — equal declarations from permuted inputs: identical bytes."""
    experiments = _experiments()
    left = _config(
        strategy_ids=("min-var", "equal-weight", "spare-arm"),
        target="min-var",
        comparators=("equal-weight",),
        benchmarks=("equal-weight",),
        conditions={"risk_aversion": 2.5, "universe": ("A", "B")},
    )
    right = experiments.ExperimentConfiguration(
        strategy_ids=["spare-arm", "equal-weight", "min-var"],
        target="min-var",
        comparators=["equal-weight"],
        benchmarks=["equal-weight"],
        conditions={"universe": ("A", "B"), "risk_aversion": 2.5},
    )
    assert left == right
    assert _canonical(left) == _canonical(right)


def test_one_declared_fact_difference_changes_bytes() -> None:
    """Floor T17 — declarations differing in one fact have different bytes."""
    experiments = _experiments()
    base = {
        "strategy_ids": ("min-var", "equal-weight", "third"),
        "target": "min-var",
        "comparators": ("equal-weight",),
        "benchmarks": ("equal-weight",),
        "conditions": {"risk_aversion": 2.5},
    }
    variations = (
        {
            "target": "equal-weight",
            "comparators": ("min-var",),
            "benchmarks": ("min-var",),
        },
        {"comparators": ("equal-weight", "third")},
        {"ablations": ("third",)},
        {"robustness_checks": ("third",)},
        {"benchmarks": ()},
        {"conditions": {"risk_aversion": 3.0}},
    )
    texts = set()
    for overrides in variations:
        kwargs = {**base, **overrides}
        texts.add(_canonical(experiments.ExperimentConfiguration(**kwargs)))
    assert len(texts) == len(variations)
    assert _canonical(experiments.ExperimentConfiguration(**base)) not in texts


def test_canonical_bytes_sorted_compact_ascii_strict() -> None:
    """Floor T17 — sorted keys, compact separators, ASCII, strict JSON."""
    config = _config(conditions={"rate": 0.25, "label": "café", "n": 3})
    text = _canonical(config)
    assert text.isascii()
    assert ", " not in text
    assert ": " not in text
    payload = json.loads(text)
    assert _reserialize(payload, separators=(",", ":")) == text


def test_no_live_object_token_ever_enters_bytes() -> None:
    """Floor T17 (B2) — no repr/class/import/pickle/factory token in bytes."""
    _registration("min-var", _LiveComponent())
    _registration("equal-weight", _OtherLiveComponent())
    text = _canonical(_config())
    for token in (
        "_LiveComponent",
        "_OtherLiveComponent",
        "object at 0x",
        "pickle",
        "__reduce__",
        "import",
        "module",
        "class",
        "factory",
    ):
        assert token not in text


# --------------------------------------------------------------------------- #
# T18 — deserialization                                                        #
# --------------------------------------------------------------------------- #


def test_canonical_roundtrip_equal_declaration_no_live_objects() -> None:
    """Floor T18 (B2) — round-trip reconstructs an equal declaration only."""
    experiments = _experiments()
    config = _config(conditions={"risk_aversion": 2.5, "universe": ("A", "B")})
    text = _canonical(config)
    reborn = experiments.ExperimentConfiguration.from_canonical_text(text)
    assert reborn == config
    assert _canonical(reborn) == text
    assert reborn.conditions == config.conditions
    assert reborn.benchmarks == config.benchmarks


def test_noncanonical_spellings_reject() -> None:
    """Floor T18 — only the canonical form admits; variants reject."""
    experiments = _experiments()
    text = _canonical(_config())
    payload = json.loads(text)
    indented = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
    spaced = _reserialize(payload, separators=(", ", ": "))
    unsorted = json.dumps(
        {key: payload[key] for key in reversed(list(payload))},
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    for variant in (indented, spaced, unsorted):
        assert variant != text
        with pytest.raises(experiments.ConfigurationError):
            experiments.ExperimentConfiguration.from_canonical_text(variant)
    assert experiments.ExperimentConfiguration.from_canonical_text(
        text
    ) == experiments.ExperimentConfiguration.from_canonical_text(text)


def test_non_object_roots_reject() -> None:
    """Floor T18 — non-JSON-object roots reject."""
    experiments = _experiments()
    for bad in ("[]", "5", '"text"', "null", "true"):
        with pytest.raises(experiments.ConfigurationError):
            experiments.ExperimentConfiguration.from_canonical_text(bad)


def test_duplicate_keys_reject_at_every_depth() -> None:
    """Floor T18 — duplicate keys reject at the top level and nested."""
    experiments = _experiments()
    text = _canonical(_config(conditions={"family": {"depth": 2, "n": 1}}))
    payload = json.loads(text)
    first_key = next(iter(payload))
    head = (
        json.dumps(first_key)
        + ":"
        + _reserialize(payload[first_key], separators=(",", ":"))
    )
    top_dup = "{" + head + "," + head + "}"
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentConfiguration.from_canonical_text(top_dup)
    nested = _first_nested_mapping_text(payload)
    assert nested is not None
    inner_key = next(iter(json.loads(nested)))
    inner_head = (
        json.dumps(inner_key)
        + ":"
        + _reserialize(json.loads(nested)[inner_key], separators=(",", ":"))
    )
    nested_dup = "{" + inner_head + "," + inner_head + "}"
    assert nested_dup in text or nested in text
    spliced = text.replace(nested, nested_dup)
    assert spliced != text
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentConfiguration.from_canonical_text(spliced)


def test_unknown_fields_reject_on_deserialization() -> None:
    """Floor T18 — unknown top-level fields reject."""
    experiments = _experiments()
    text = _canonical(_config())
    payload = json.loads(text)
    payload["unexpected_extra_field"] = {"role": "benchmark"}
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentConfiguration.from_canonical_text(
            _reserialize(payload, separators=(",", ":"))
        )


def test_wire_grammar_violations_reject() -> None:
    """Floor T18 — §5.8a violations through the wire reject."""
    experiments = _experiments()
    text = _canonical(_config(conditions={"risk_aversion": 2.5}))
    assert '"risk_aversion":2.5' in text
    for bad_value in ("true", '"2.5"'):
        spliced = text.replace('"risk_aversion":2.5', f'"risk_aversion":{bad_value}')
        assert spliced != text
        with pytest.raises(experiments.ConfigurationError):
            experiments.ExperimentConfiguration.from_canonical_text(spliced)


# --------------------------------------------------------------------------- #
# T19 — immutability                                                           #
# --------------------------------------------------------------------------- #


def test_configuration_frozen_and_caller_isolated() -> None:
    """Floor T19 — the declaration is deeply frozen against callers."""
    ids = ["min-var", "equal-weight"]
    conditions = {"risk_aversion": 2.5, "universe": ("A", "B")}
    config = _config(strategy_ids=ids, conditions=conditions)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.target = "equal-weight"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.conditions = {}  # type: ignore[misc]
    with pytest.raises(TypeError):
        config.conditions["risk_aversion"] = 99.0
    ids.append("late-addition")
    conditions["risk_aversion"] = 99.0
    conditions["universe"] = ("X",)
    assert sorted(config.strategy_ids) == ["equal-weight", "min-var"]
    assert config.conditions["risk_aversion"] == 2.5
    assert config.conditions["universe"] == ("A", "B")


def test_registration_selection_results_frozen() -> None:
    """Floor T19 — registration, selection, and results resist mutation."""
    experiments = _experiments()
    reg = _registration("min-var")
    with pytest.raises(dataclasses.FrozenInstanceError):
        reg.strategy_id = "renamed"  # type: ignore[misc]
    config = _config()
    selection = experiments.select_all(config)
    with pytest.raises(dataclasses.FrozenInstanceError):
        selection.strategy_ids = ()  # type: ignore[misc]
    results = experiments.ExperimentResults(records=(_record("min-var"),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        results.records = ()  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# T20 — static-validation boundary census                                      #
# --------------------------------------------------------------------------- #

_FORBIDDEN_SURFACE_TOKENS = (
    "run",
    "run_experiment",
    "execute",
    "apply_conditions",
    "apply",
    "validate_target",
    "validate",
    "enforce",
    "resolve_backend",
    "resolve",
)


def test_no_execution_entry_points_on_module_census() -> None:
    """Floor T20 (runtime successor) — exact census; no free-function executor.

    The census is the same thirteen approved names T1 pins (nine
    declarative declaration names plus ``ExecutionError``, ``DecisionSchedule``,
    ``ExperimentConditions``, ``Experiment``), so a surprise, missing, or
    renamed export still fails.  Execution lives only on the reserved
    ``Experiment`` class: every token below stays banned as a module-level
    attribute and in ``__all__`` (``Experiment.run`` is approved and
    unaffected); the declarative objects carry no execution methods (T22).
    """
    module = _experiments()
    assert set(module.__all__) == {
        "ConfigurationError",
        "StrategyRegistration",
        "ExperimentConfiguration",
        "ExperimentSelection",
        "ExperimentResults",
        "select_all",
        "select_none",
        "select_single",
        "select_subset",
        "ExecutionError",
        "DecisionSchedule",
        "ExperimentConditions",
        "Experiment",
    }
    for token in _FORBIDDEN_SURFACE_TOKENS:
        assert not hasattr(module, token), f"forbidden surface token: {token}"
        assert token not in module.__all__


def test_configuration_carries_no_execution_behavior() -> None:
    """Floors T20/T22 — the declaration object has no runtime surface."""
    config = _config()
    for token in ("run", "execute", "apply_conditions", "validate_target"):
        assert not hasattr(config, token)


# --------------------------------------------------------------------------- #
# T21 — declaration/registration split (B2)                                    #
# --------------------------------------------------------------------------- #


def test_no_constructor_field_accepts_a_live_component() -> None:
    """Floor T21 — components never enter the declaration constructor."""
    experiments = _experiments()
    component = _LiveComponent()
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentConfiguration(
            strategy_ids=(component,),  # type: ignore[arg-type]
            target="min-var",
        )
    with pytest.raises(experiments.ConfigurationError):
        experiments.ExperimentConfiguration(
            strategy_ids=("min-var",),
            target=component,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        experiments.ExperimentConfiguration(
            strategy_ids=("min-var",),
            target="min-var",
            registrations=(_registration("min-var"),),
        )
    with pytest.raises(TypeError):
        experiments.ExperimentConfiguration(
            strategy_ids=("min-var",),
            target="min-var",
            components={"min-var": component},
        )


def test_live_components_never_change_canonical_bytes() -> None:
    """Floor T21 — different live components under the same ids: same bytes."""
    config = _config()
    bytes_with_first = _canonical(config)
    _registration("min-var", _LiveComponent())
    _registration("equal-weight", _LiveComponent())
    bytes_with_second = _canonical(config)
    _registration("min-var", _OtherLiveComponent())
    _registration("equal-weight", _OtherLiveComponent())
    assert bytes_with_first == bytes_with_second == _canonical(config)


def test_registration_is_process_local_and_non_serializable() -> None:
    """Floor T21 — a registration never serializes (B2 process-local)."""
    reg = _registration("min-var", _LiveComponent())
    for route in ("canonical_text", "to_json", "from_canonical_text", "from_json"):
        assert not hasattr(reg, route)
    with pytest.raises(TypeError):
        json.dumps(reg)


def test_decide_check_fires_at_registration_not_deserialization() -> None:
    """Floor T21 (D6/D11a) — the structural check lives on registration."""
    experiments = _experiments()
    with pytest.raises(experiments.ConfigurationError):
        experiments.StrategyRegistration(
            strategy_id="undecidable", strategy=_DecidelessComponent()
        )
    text = _canonical(_config())
    assert "undecidable" not in text
    reborn = experiments.ExperimentConfiguration.from_canonical_text(text)
    assert reborn == _config()


# --------------------------------------------------------------------------- #
# T22 — runtime isolated from the declarative shapes (B3)                      #
# --------------------------------------------------------------------------- #


def test_runtime_isolated_from_declarative_shapes() -> None:
    """Floor T22 (runtime successor) — one reserved runtime object; ``run`` on it.

    The reserved ``Experiment`` export exists and is the only execution
    home: ``run`` lives on it and on no other runtime shape; the
    declarative ``ExperimentConfiguration`` and ``ExperimentSelection``
    instances expose no execution behavior; and no module-level
    free-function executor (``run``/``run_experiment``) exists.  A second
    runtime type would already fail the exact T1 census.
    """
    module = _experiments()
    assert "Experiment" in module.__all__
    experiment_type = module.Experiment
    assert isinstance(experiment_type, type)
    assert hasattr(experiment_type, "run")
    assert callable(experiment_type.run)
    assert not hasattr(module.DecisionSchedule, "run")
    assert not hasattr(module.ExperimentConditions, "run")
    config = _config()
    assert not hasattr(config, "run")
    assert not callable(getattr(config, "run", None))
    selection = module.select_all(config)
    assert not hasattr(selection, "run")
    assert not callable(getattr(selection, "run", None))
    for name in ("run", "run_experiment"):
        assert not hasattr(module, name), f"free-function executor: {name}"
        assert name not in module.__all__


# --------------------------------------------------------------------------- #
# Always-green controls (frozen results seams — harness-wiring proof).           #
# --------------------------------------------------------------------------- #


def test_control_stand_in_records_admit_through_public_constructors() -> None:
    """Green control — the results ``ResultRecord`` stand-ins really build."""
    record = _record("min-var")
    assert record.strategy_id == "min-var"
    assert set(record.paths) == {"ew", "iv"}


def test_control_record_rejects_blank_strategy_id() -> None:
    """Green control — the frozen seam behind the linkage law fails closed."""
    results = _results()
    with pytest.raises(results.ResultStructureError):
        _record("   ")


# --------------------------------------------------------------------------- #
# runtime RED section — runtime floors F1-F4 (micro-tranche 1).                  #
#                                                                             #
# The runtime surface (ExecutionError, DecisionSchedule,                #
# ExperimentConditions, Experiment) does not exist yet, so every node below  #
# is causally RED through the absent public surface: the lazy _experiments() #
# import resolves (the module and its nine declaration names are green above) and   #
# the first missing runtime name fails as a missing export inside the node — #
# never a collection, module-import, or fixture-plumbing failure.            #
#                                                                             #
# Constructor vocabulary follows the runtime §4 pinned attribute names exactly  #
# (DecisionSchedule decision_dates/universes; ExperimentConditions           #
# information_provider/outcome_provider/rebalance_policy/accounting_engine/  #
# cost_model/initial_weights/initial_wealth/initial_last_rebalance_time) and #
# the frozen positional composition Experiment(configuration, registrations, #
# schedule, conditions).  Run stand-ins use frozen seams only: synthetic     #
# strategies implementing the frozen decide surface, provider/policy         #
# stand-ins, the shipped ExactFillAccounting and Proportional cost model,    #
# and the results/declaration declarative helpers above.                               #
# --------------------------------------------------------------------------- #

_M44_DATES = _MONTHS[:3]
_M44_UNIVERSES = {date: ("A", "B") for date in _M44_DATES}
_M44_OPENING_LAST_REBALANCE = datetime(2025, 12, 1, tzinfo=UTC)


class _RuntimeStrategyStandIn:
    """A synthetic strategy implementing the frozen ``decide`` surface.

    Returns an equal-weight TARGET over the context's declared universe
    with same-instant execution (the frozen seam's permitted special
    case), threading no state — the minimal lawful runtime arm.
    """

    def decide(self, context: Any) -> Any:
        per_asset = 1.0 / len(context.universe)
        decision = PortfolioDecision(
            decision_time=context.decision_time,
            execution_time=context.decision_time,
            target_weights={asset: per_asset for asset in context.universe},
        )
        return DecisionResult(decision=decision)


class _InformationProviderStandIn:
    """The common point-in-time backend: one call per declared date."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def __call__(self, instant: Any) -> Any:
        self.calls.append(instant)
        return InformationSet(items=(), as_of=instant)


class _OutcomeProviderStandIn:
    """The common realized-outcome backend: flat gross factors per segment."""

    def __init__(self, universes: dict[Any, tuple[str, ...]]) -> None:
        self._universes = universes
        self.calls: list[Any] = []

    def __call__(self, instant: Any) -> Any:
        self.calls.append(instant)
        return {asset: 1.0 for asset in self._universes[instant]}


class _AlwaysRebalanceStandIn:
    """A rebalance policy that answers True at every query."""

    def should_rebalance(self, decision_time: Any, last_rebalance_time: Any) -> bool:
        return True


def _m44_schedule():
    """The declared decision/universe schedule (§4 pinned vocabulary)."""
    experiments = _experiments()
    return experiments.DecisionSchedule(
        decision_dates=_M44_DATES,
        universes=_M44_UNIVERSES,
    )


def _m44_conditions():
    """The injected common backends plus the explicit opening state."""
    experiments = _experiments()
    return experiments.ExperimentConditions(
        information_provider=_InformationProviderStandIn(),
        outcome_provider=_OutcomeProviderStandIn(_M44_UNIVERSES),
        rebalance_policy=_AlwaysRebalanceStandIn(),
        accounting_engine=ExactFillAccounting(),
        cost_model=Proportional(rate=0.001, turnover=two_sided),
        initial_weights={"A": 0.5, "B": 0.5},
        initial_wealth=100.0,
        initial_last_rebalance_time=_M44_OPENING_LAST_REBALANCE,
    )


def _m44_runnable_registration(strategy_id: str):
    """A registration whose component can lawfully execute (F3/F4 runs)."""
    return _registration(strategy_id, _RuntimeStrategyStandIn())


def test_experiments_module_exports_runtime_surface() -> None:
    """F1 — the runtime names export; the declaration nine remain reachable."""
    module = _experiments()
    for name in (
        "ExecutionError",
        "DecisionSchedule",
        "ExperimentConditions",
        "Experiment",
    ):
        assert hasattr(module, name), name
        assert name in module.__all__, name
    assert issubclass(module.ExecutionError, ValueError)
    for name in (
        "ConfigurationError",
        "StrategyRegistration",
        "ExperimentConfiguration",
        "ExperimentSelection",
        "ExperimentResults",
        "select_all",
        "select_none",
        "select_single",
        "select_subset",
    ):
        assert hasattr(module, name), name
        assert name in module.__all__, name


def test_experiment_binds_declared_ids_to_live_registrations_exactly() -> None:
    """F2 — binding is the exact-id entry act, fail-closed at construction."""
    module = _experiments()
    config = _config()
    schedule = _m44_schedule()
    conditions = _m44_conditions()
    experiment = module.Experiment(
        config,
        (_registration("min-var"), _registration("equal-weight")),
        schedule,
        conditions,
    )
    assert isinstance(experiment, module.Experiment)
    with pytest.raises(module.ConfigurationError):
        module.Experiment(config, (_registration("min-var"),), schedule, conditions)
    with pytest.raises(module.ConfigurationError):
        module.Experiment(
            config,
            (
                _registration("min-var"),
                _registration("equal-weight"),
                _registration("shrinkage"),
            ),
            schedule,
            conditions,
        )
    with pytest.raises(module.ConfigurationError):
        module.Experiment(
            config,
            (
                _registration("min-var"),
                _registration("min-var"),
                _registration("equal-weight"),
            ),
            schedule,
            conditions,
        )


def test_binding_discriminates_mixed_case_strategy_ids() -> None:
    """F3 — ids are exact strings: case variants never fold, never bind."""
    module = _experiments()
    schedule = _m44_schedule()
    conditions = _m44_conditions()
    declared_alpha = _config(
        strategy_ids=("Alpha",), target="Alpha", comparators=(), benchmarks=()
    )
    for registered_id in ("alpha", "ALPHA"):
        with pytest.raises(module.ConfigurationError):
            module.Experiment(
                declared_alpha,
                (_m44_runnable_registration(registered_id),),
                schedule,
                conditions,
            )
    declared_both = _config(
        strategy_ids=("Alpha", "alpha"),
        target="Alpha",
        comparators=("alpha",),
        benchmarks=("alpha",),
    )
    experiment = module.Experiment(
        declared_both,
        (_m44_runnable_registration("Alpha"), _m44_runnable_registration("alpha")),
        schedule,
        conditions,
    )
    results = experiment.run(module.select_all(declared_both))
    assert [record.strategy_id for record in results] == ["Alpha", "alpha"]


def test_run_returns_uniform_results_collection_for_every_mode() -> None:
    """F4 — every mode returns ExperimentResults; None is the full mode."""
    module = _experiments()
    config = _config(
        strategy_ids=("min-var", "equal-weight", "shrinkage"),
        target="min-var",
        comparators=("equal-weight", "shrinkage"),
        benchmarks=("equal-weight",),
    )
    experiment = module.Experiment(
        config,
        (
            _m44_runnable_registration("min-var"),
            _m44_runnable_registration("equal-weight"),
            _m44_runnable_registration("shrinkage"),
        ),
        _m44_schedule(),
        _m44_conditions(),
    )
    none_mode = experiment.run(module.select_none(config))
    assert isinstance(none_mode, module.ExperimentResults)
    assert none_mode is not None
    assert tuple(none_mode) == ()
    single = experiment.run(module.select_single(config, "min-var"))
    assert single is not None
    assert isinstance(single, module.ExperimentResults)
    assert [record.strategy_id for record in single] == ["min-var"]
    subset = experiment.run(module.select_subset(config, ("equal-weight", "shrinkage")))
    assert isinstance(subset, module.ExperimentResults)
    assert [record.strategy_id for record in subset] == [
        "equal-weight",
        "shrinkage",
    ]
    full = experiment.run(module.select_all(config))
    assert isinstance(full, module.ExperimentResults)
    assert [record.strategy_id for record in full] == [
        "equal-weight",
        "min-var",
        "shrinkage",
    ]
    default_mode = experiment.run(None)
    assert isinstance(default_mode, module.ExperimentResults)
    assert [record.strategy_id for record in default_mode] == [
        "equal-weight",
        "min-var",
        "shrinkage",
    ]


# --------------------------------------------------------------------------- #
# Runtime §8 floors F5–F8 (micro-tranche 2). The run-entry and condition-       #
# sameness laws: V1–V4 entry validation before the first decision date       #
# (Law 4/§4 D5–D5b, §8a M3/M5/M11), the declared schedule (Law 3/V5,        #
# §8a M4), the one common information/universe supply (Law 7 (i)/(iii),      #
# D5a, §8a M5), and point-in-time universe screening (Law 8/V3, §8a M4).     #
#                                                                            #
# RED shape: the runtime names are absent, so every node resolves them       #
# lazily through the accepted `_experiments()` helper — collection succeeds  #
# and each node fails causally on the absent approved surface.               #
#                                                                            #
# F8 screening seam (owner-resolved 2026-10-02; runtime §16 amendment            #
# record): the owner named the two optional direct DecisionSchedule           #
# fields — `screening_instants` and `membership_available_times` — as         #
# one optional-but-complete paired contract. Both absent consumes             #
# the declared universes unchanged; both present requires exact               #
# decision-date and universe-member coverage, aware datetimes, and            #
# membership_available_time <= screening_instant <= decision_date,            #
# failing closed under the frozen point-in-time error taxonomy                #
# (NaiveTimestampError / FutureInformationError / declaration-family          #
# ConfigurationError-ExecutionError). F6/F8 pin it below.                     #
# --------------------------------------------------------------------------- #


class _RecordingStrategyStandIn:
    """A synthetic arm that records every context it receives."""

    def __init__(self) -> None:
        self.contexts: list[Any] = []

    def decide(self, context: Any) -> Any:
        self.contexts.append(context)
        per_asset = 1.0 / len(context.universe)
        decision = PortfolioDecision(
            decision_time=context.decision_time,
            execution_time=context.decision_time,
            target_weights={asset: per_asset for asset in context.universe},
        )
        return DecisionResult(decision=decision)


class _RaisingInformationProvider:
    def __call__(self, instant: Any) -> Any:
        raise RuntimeError(f"information backend unavailable at {instant}")


class _RaisingOutcomeProvider:
    def __call__(self, instant: Any) -> Any:
        raise RuntimeError(f"outcome backend unavailable at {instant}")


def _outcome_book(instant: Any, factor: float) -> dict[str, float]:
    """A flat common factor book for ``instant``'s declared universe."""
    return {asset: factor for asset in _M44_UNIVERSES[instant]}


class _DuplicateAssetBook(Mapping):
    """The representable duplicate-id route: a mapping-view over id/factor pairs.

    A dict literal cannot carry a duplicate id; this ``Mapping`` yields the
    duplicate through its public iteration surface (``len`` 3, two distinct
    keys), which is the only representable way to hand the runtime a book
    whose key set deviates by duplication (D5b/Law 12b).
    """

    def __init__(self, pairs: tuple[tuple[str, float], ...]) -> None:
        self._pairs = pairs

    def __getitem__(self, key: str) -> float:
        for candidate, factor in self._pairs:
            if candidate == key:
                return factor
        raise KeyError(key)

    def __iter__(self) -> Any:
        return iter(key for key, _factor in self._pairs)

    def __len__(self) -> int:
        return len(self._pairs)


def _m44_conditions_with(**overrides: Any):
    """The accepted common conditions with named fields overridden."""
    fields: dict[str, Any] = {
        "information_provider": _InformationProviderStandIn(),
        "outcome_provider": _OutcomeProviderStandIn(_M44_UNIVERSES),
        "rebalance_policy": _AlwaysRebalanceStandIn(),
        "accounting_engine": ExactFillAccounting(),
        "cost_model": Proportional(rate=0.001, turnover=two_sided),
        "initial_weights": {"A": 0.5, "B": 0.5},
        "initial_wealth": 100.0,
        "initial_last_rebalance_time": _M44_OPENING_LAST_REBALANCE,
    }
    if set(overrides) - set(fields):
        raise AssertionError(
            f"unknown override fields: {sorted(set(overrides) - set(fields))}"
        )
    fields.update(overrides)
    return _experiments().ExperimentConditions(**fields)


def _m44_entry_experiment(conditions: Any, schedule: Any | None = None):
    """A minimal one-arm runnable entry composition (F5/F6 driver)."""
    module = _experiments()
    config = _config(
        strategy_ids=("min-var",),
        target="min-var",
        comparators=(),
        benchmarks=(),
    )
    strategy = _RecordingStrategyStandIn()
    experiment = module.Experiment(
        config,
        (_registration("min-var", strategy),),
        schedule if schedule is not None else _m44_schedule(),
        conditions,
    )
    return experiment, strategy


def test_run_entry_validates_backends_before_first_decision_date() -> None:
    """F5 — V1–V4 entry validation precedes the loop; decide stays at zero."""
    module = _experiments()
    legal_ids = _M44_UNIVERSES[_M44_DATES[0]]
    later = _M44_DATES[0] + timedelta(days=1)

    def reject(conditions: Any, label: str) -> None:
        experiment, strategy = _m44_entry_experiment(conditions)
        with pytest.raises(module.ExecutionError, match=label):
            experiment.run(
                module.select_all(
                    _config(
                        strategy_ids=("min-var",),
                        target="min-var",
                        comparators=(),
                        benchmarks=(),
                    )
                )
            )
        assert strategy.contexts == []

    # V1 — the information provider must resolve for every declared date.
    reject(
        _m44_conditions_with(information_provider=_RaisingInformationProvider()),
        "information",
    )
    reject(
        _m44_conditions_with(
            information_provider=lambda instant: {"not": "an information set"},
        ),
        "information",
    )
    reject(
        _m44_conditions_with(
            information_provider=lambda instant: InformationSet(
                items=(), as_of=instant + timedelta(days=1)
            ),
        ),
        "information",
    )
    # V2 — the outcome provider must resolve for every required segment start.
    reject(_m44_conditions_with(outcome_provider=_RaisingOutcomeProvider()), "outcome")
    reject(
        _m44_conditions_with(
            outcome_provider=lambda instant: [factor for factor in (1.0, 2.0)],
        ),
        "outcome",
    )
    reject(
        _m44_conditions_with(
            outcome_provider=lambda instant: _outcome_book(instant, 1.0) | {"C": 1.0},
        ),
        "outcome",
    )
    reject(
        _m44_conditions_with(
            outcome_provider=lambda instant: {
                asset: 1.0 for asset in _M44_UNIVERSES[instant] if asset != "B"
            },
        ),
        "outcome",
    )
    duplicate_pairs = tuple(
        (asset, 1.05)
        for asset in (*_M44_UNIVERSES[_M44_DATES[0]], _M44_UNIVERSES[_M44_DATES[0]][-1])
    )
    reject(
        _m44_conditions_with(
            outcome_provider=lambda instant: _DuplicateAssetBook(duplicate_pairs),
        ),
        "outcome",
    )
    reject(
        _m44_conditions_with(
            outcome_provider=lambda instant: _outcome_book(instant, float("nan")),
        ),
        "outcome",
    )
    # V4 — the opening state must be present and lawful.
    reject(
        _m44_conditions_with(initial_weights={"A": 0.6, "B": 0.5}),
        "opening|initial",
    )
    reject(
        _m44_conditions_with(
            initial_weights={
                asset: float("inf") / len(legal_ids) for asset in legal_ids
            },
        ),
        "opening|initial",
    )
    reject(
        _m44_conditions_with(initial_weights={"A": 0.5, "B": 0.5, "C": 0.0}),
        "opening|initial",
    )
    reject(_m44_conditions_with(initial_wealth=0.0), "opening|initial")
    reject(_m44_conditions_with(initial_wealth=float("nan")), "opening|initial")
    for bad_rebalance in (
        _M44_DATES[1],
        _M44_DATES[0] + timedelta(microseconds=1),
        later,
    ):
        reject(
            _m44_conditions_with(initial_last_rebalance_time=bad_rebalance),
            "opening|initial",
        )
    for absent_field in (
        "initial_weights",
        "initial_wealth",
        "initial_last_rebalance_time",
    ):
        fields: dict[str, Any] = {
            "information_provider": _InformationProviderStandIn(),
            "outcome_provider": _OutcomeProviderStandIn(_M44_UNIVERSES),
            "rebalance_policy": _AlwaysRebalanceStandIn(),
            "accounting_engine": ExactFillAccounting(),
            "cost_model": Proportional(rate=0.001, turnover=two_sided),
            "initial_weights": {"A": 0.5, "B": 0.5},
            "initial_wealth": 100.0,
            "initial_last_rebalance_time": _M44_OPENING_LAST_REBALANCE,
        }
        del fields[absent_field]
        reject(_experiments().ExperimentConditions(**fields), "opening|initial")


def test_run_entry_rejects_incoherent_schedule() -> None:
    """F6 — Law 3/V5: an incoherent schedule never reaches the loop.

    Each case is wrapped over the whole construction-plus-entry chain, so
    the node pins the lawful error family without over-pinning whether a
    listed case rejects at the ``DecisionSchedule`` constructor or at run
    entry (the spec permits either), and asserts the strategy's ``decide``
    is never called whenever an arm could have been reached.

    The screening cases ride the owner-approved paired screening contract
    (§4 amendment, 2026-10-02): a half-declared pair or a decision-date /
    universe-member coverage defect rejects with the declaration family,
    while the frozen point-in-time taxonomy pins the other classes exactly
    — a naive screening instant or naive membership availability rejects
    with ``NaiveTimestampError`` (reused from ``portlearn.timing``), and
    either inequality breach (membership evidence dated after the
    screening instant, a screening instant after its decision date)
    rejects with ``FutureInformationError``.  Nothing is dropped,
    defaulted, inferred, or silently re-dated.
    """
    module = _experiments()
    lawful = (module.ExecutionError, module.ConfigurationError, ValueError)
    declared = (module.ConfigurationError, module.ExecutionError)

    def lawful_reject(
        make_schedule: Any, family: tuple[type[BaseException], ...] = lawful
    ) -> None:
        armed: list[Any] = []
        with pytest.raises(family):
            schedule = make_schedule()
            experiment, strategy = _m44_entry_experiment(_m44_conditions(), schedule)
            armed.append(strategy)
            experiment.run(None)
        if armed:
            assert armed[0].contexts == []

    dates = list(_M44_DATES)
    lawful_reject(
        lambda: module.DecisionSchedule(
            decision_dates=[date.replace(tzinfo=None) for date in dates],
            universes=_M44_UNIVERSES,
        )
    )
    lawful_reject(
        lambda: module.DecisionSchedule(
            decision_dates=[dates[1], dates[0], dates[2]], universes=_M44_UNIVERSES
        )
    )
    lawful_reject(
        lambda: module.DecisionSchedule(
            decision_dates=[dates[0], dates[0], dates[2]], universes=_M44_UNIVERSES
        )
    )

    def short_universes() -> dict[Any, tuple[str, ...]]:
        universes = dict(_M44_UNIVERSES)
        universes.pop(dates[2])
        return universes

    lawful_reject(
        lambda: module.DecisionSchedule(
            decision_dates=dates, universes=short_universes()
        )
    )
    # The paired screening contract (§4 amendment, 2026-10-02): one
    # complete lawful pair for reference, then every declaration defect
    # over the same construction-plus-entry chain, and the frozen
    # point-in-time classes pinned to their exact reused types.
    screening = {date: date - timedelta(days=1) for date in dates}
    available = {
        date: {asset: date - timedelta(days=2) for asset in _M44_UNIVERSES[date]}
        for date in dates
    }

    def paired(instants: Any = None, books: Any = None) -> Any:
        fields: dict[str, Any] = {
            "decision_dates": list(dates),
            "universes": dict(_M44_UNIVERSES),
        }
        if instants is not None:
            fields["screening_instants"] = dict(instants)
        if books is not None:
            fields["membership_available_times"] = {
                date: dict(book) for date, book in books.items()
            }
        return module.DecisionSchedule(**fields)

    # Exactly one of the pair present — an incomplete declaration defect.
    lawful_reject(lambda: paired(instants=screening), declared)
    lawful_reject(lambda: paired(books=available), declared)
    # Outer coverage — all and only the decision dates, in either mapping.
    instants_missing_date = dict(screening)
    instants_missing_date.pop(dates[1])
    lawful_reject(
        lambda: paired(instants=instants_missing_date, books=available), declared
    )
    books_missing_date = {date: dict(available[date]) for date in dates}
    books_missing_date.pop(dates[2])
    lawful_reject(
        lambda: paired(instants=screening, books=books_missing_date), declared
    )
    instants_extra_date = dict(screening)
    instants_extra_date[_MONTHS[3]] = _MONTHS[3] - timedelta(days=1)
    lawful_reject(
        lambda: paired(instants=instants_extra_date, books=available), declared
    )
    books_extra_date = {date: dict(available[date]) for date in dates}
    books_extra_date[_MONTHS[3]] = dict(available[dates[0]])
    lawful_reject(lambda: paired(instants=screening, books=books_extra_date), declared)
    # Inner coverage — exactly the declared universe members, per date.
    books_missing_member = {date: dict(available[date]) for date in dates}
    del books_missing_member[dates[0]]["B"]
    lawful_reject(
        lambda: paired(instants=screening, books=books_missing_member), declared
    )
    books_extra_member = {date: dict(available[date]) for date in dates}
    books_extra_member[dates[2]]["C"] = dates[2] - timedelta(days=2)
    lawful_reject(
        lambda: paired(instants=screening, books=books_extra_member), declared
    )

    def pit_reject(make_schedule: Any, *expected: type[BaseException]) -> None:
        armed: list[Any] = []
        with pytest.raises(expected):
            schedule = make_schedule()
            experiment, strategy = _m44_entry_experiment(_m44_conditions(), schedule)
            armed.append(strategy)
            experiment.run(None)
        if armed:
            assert armed[0].contexts == []

    # Naive carried datetimes reject with the reused frozen type.
    naive_instant = dict(screening)
    naive_instant[dates[0]] = naive_instant[dates[0]].replace(tzinfo=None)
    pit_reject(
        lambda: paired(instants=naive_instant, books=available), NaiveTimestampError
    )
    naive_availability = {date: dict(available[date]) for date in dates}
    naive_availability[dates[1]]["A"] = naive_availability[dates[1]]["A"].replace(
        tzinfo=None
    )
    pit_reject(
        lambda: paired(instants=screening, books=naive_availability),
        NaiveTimestampError,
    )
    # A screening instant after its decision date rejects as future
    # information (the Law-3 clause now executable under the amendment).
    late_screening = dict(screening)
    late_screening[dates[2]] = dates[2] + timedelta(days=1)
    pit_reject(
        lambda: paired(instants=late_screening, books=available),
        FutureInformationError,
    )
    # Membership evidence dated after the screening instant likewise.
    late_availability = {date: dict(available[date]) for date in dates}
    late_availability[dates[2]]["B"] = screening[dates[2]] + timedelta(hours=1)
    pit_reject(
        lambda: paired(instants=screening, books=late_availability),
        FutureInformationError,
    )


def test_run_applies_identical_information_and_universe_to_every_selected_strategy() -> (
    None
):
    """F7 — Law 7 (i)/(iii), D5a: one common supply binds every arm."""
    module = _experiments()
    provider = _InformationProviderStandIn()
    conditions = _m44_conditions_with(information_provider=provider)
    strategy_alpha = _RecordingStrategyStandIn()
    strategy_beta = _RecordingStrategyStandIn()
    config = _config(
        strategy_ids=("alpha", "beta"),
        target="alpha",
        comparators=("beta",),
        benchmarks=("beta",),
    )
    experiment = module.Experiment(
        config,
        (
            _registration("alpha", strategy_alpha),
            _registration("beta", strategy_beta),
        ),
        _m44_schedule(),
        conditions,
    )
    results = experiment.run(module.select_all(config))
    assert [record.strategy_id for record in results] == ["alpha", "beta"]
    assert provider.calls == list(_M44_DATES)
    for strategy in (strategy_alpha, strategy_beta):
        assert [context.decision_time for context in strategy.contexts] == list(
            _M44_DATES
        )
        for context in strategy.contexts:
            assert context.universe == ("A", "B")
    for alpha_context, beta_context in zip(
        strategy_alpha.contexts, strategy_beta.contexts, strict=True
    ):
        assert alpha_context.information is beta_context.information


def test_universe_membership_resolves_point_in_time() -> None:
    """F8 — Law 8/V3: point-in-time screening over the declared universes.

    Absent screening metadata is lawful and means the explicitly declared
    universes are consumed unchanged: the run executes on them, the
    information provider is called exactly once per declared decision
    date, and nothing else is consulted — no extra decision-date call, no
    screening provider, no inferred membership.  A lawfully declared
    availability-evidenced pair — exact decision-date and member coverage,
    aware datetimes, with the equality boundaries
    ``membership_available_time == screening_instant == decision date``
    admitted alongside strict interiors — supplies exactly those
    explicitly declared universes.  Breaches reject fail-closed under the
    frozen taxonomy: membership evidence dated after the screening
    instant and a screening instant after its decision date reject with
    ``FutureInformationError``; a naive screening instant or naive
    membership availability rejects with ``NaiveTimestampError`` — never
    a filtered, dropped, or silently re-dated membership.
    """
    module = _experiments()
    dates = list(_M44_DATES)

    def paired_schedule(instants: Any, books: Any) -> Any:
        return module.DecisionSchedule(
            decision_dates=dates,
            universes=dict(_M44_UNIVERSES),
            screening_instants=dict(instants),
            membership_available_times={
                date: dict(book) for date, book in books.items()
            },
        )

    def run_pair_declared(schedule: Any):
        provider = _InformationProviderStandIn()
        experiment, strategy = _m44_entry_experiment(
            _m44_conditions_with(information_provider=provider), schedule
        )
        results = experiment.run(None)
        return provider, strategy, results

    def assert_declared_universes_unchanged(
        provider: Any, strategy: Any, results: Any
    ) -> None:
        assert [record.strategy_id for record in results] == ["min-var"]
        assert provider.calls == list(_M44_DATES)
        assert [context.decision_time for context in strategy.contexts] == list(
            _M44_DATES
        )
        for context in strategy.contexts:
            assert context.universe == _M44_UNIVERSES[context.decision_time]

    # Absent pair — the declared universes enter unchanged and the only
    # backend resolutions are the one-per-declared-date information calls.
    assert_declared_universes_unchanged(*run_pair_declared(_m44_schedule()))

    # Lawful availability-evidenced screening — every equality boundary
    # admitted (availability == screening == decision date on the first
    # date), strict interiors lawful elsewhere — supplies exactly the
    # declared universes with no additional backend consultation.
    instants = {
        dates[0]: dates[0],
        dates[1]: dates[1] - timedelta(days=1),
        dates[2]: dates[2] - timedelta(hours=1),
    }
    books = {
        dates[0]: {"A": dates[0], "B": dates[0]},
        dates[1]: {
            "A": dates[1] - timedelta(days=2),
            "B": dates[1] - timedelta(days=1, hours=12),
        },
        dates[2]: {
            "A": dates[2] - timedelta(days=2),
            "B": dates[2] - timedelta(days=1),
        },
    }
    assert_declared_universes_unchanged(
        *run_pair_declared(paired_schedule(instants, books))
    )

    def pit_reject(make_schedule: Any, *expected: type[BaseException]) -> None:
        armed: list[Any] = []
        with pytest.raises(expected):
            schedule = make_schedule()
            experiment, strategy = _m44_entry_experiment(_m44_conditions(), schedule)
            armed.append(strategy)
            experiment.run(None)
        if armed:
            assert armed[0].contexts == []

    # Membership evidence dated after its screening instant — future
    # information at the screening boundary — rejects fail-closed.
    late_availability = {date: dict(books[date]) for date in dates}
    late_availability[dates[1]]["A"] = instants[dates[1]] + timedelta(hours=1)
    pit_reject(
        lambda: paired_schedule(instants, late_availability), FutureInformationError
    )
    # A screening instant after its decision date rejects the same way.
    late_screening = dict(instants)
    late_screening[dates[2]] = dates[2] + timedelta(days=1)
    pit_reject(lambda: paired_schedule(late_screening, books), FutureInformationError)
    # Naive carried datetimes reject with the reused frozen type.
    naive_instant = dict(instants)
    naive_instant[dates[0]] = naive_instant[dates[0]].replace(tzinfo=None)
    pit_reject(lambda: paired_schedule(naive_instant, books), NaiveTimestampError)
    naive_availability = {date: dict(books[date]) for date in dates}
    naive_availability[dates[0]]["B"] = naive_availability[dates[0]]["B"].replace(
        tzinfo=None
    )
    pit_reject(
        lambda: paired_schedule(instants, naive_availability), NaiveTimestampError
    )


# --------------------------------------------------------------------------- #
# Runtime §8 floors F9-F14 — the decision loop, centralized TARGET validation,   #
# conformance verdicts, collection linkage, and comparison readiness.  One    #
# behavior-named node per floor, public paths only, real assertions, no       #
# skips: unimplemented floors surface as causal failures at the exact         #
# missing behavior; nodes covering already-correct behavior pass lawfully.    #
# --------------------------------------------------------------------------- #


class _DecideEventLedger:
    """The global decide-order record shared by the F10 synthetic arms."""

    def __init__(self) -> None:
        self.events: list[tuple[Any, str]] = []

    def record(self, instant: Any, strategy_id: str) -> None:
        self.events.append((instant, strategy_id))


class _StatefulSequencedStrategyStandIn:
    """A synthetic arm threading state solely via ``next_strategy_state``."""

    def __init__(self, strategy_id: str, ledger: _DecideEventLedger) -> None:
        self.strategy_id = strategy_id
        self.ledger = ledger
        self.states_received: list[Any] = []
        self.calls = 0

    def decide(self, context: Any) -> Any:
        self.calls += 1
        self.states_received.append(context.strategy_state)
        self.ledger.record(context.decision_time, self.strategy_id)
        per_asset = 1.0 / len(context.universe)
        decision = PortfolioDecision(
            decision_time=context.decision_time,
            execution_time=context.decision_time,
            target_weights={asset: per_asset for asset in context.universe},
        )
        return DecisionResult(
            decision=decision, next_strategy_state=("tick", self.calls)
        )


class _CountingRebalancePolicy:
    """A rebalance policy recording every query it receives."""

    def __init__(self) -> None:
        self.queries: list[tuple[Any, Any]] = []

    def should_rebalance(self, decision_time: Any, last_rebalance_time: Any) -> bool:
        self.queries.append((decision_time, last_rebalance_time))
        return True


class _FirstQueryOnlyRebalancePolicy:
    """A policy answering once per date; a second same-date query is a
    per-arm re-query defect and must not survive the run."""

    def __init__(self) -> None:
        self.queries: list[tuple[Any, Any]] = []
        self._seen: set[Any] = set()

    def should_rebalance(self, decision_time: Any, last_rebalance_time: Any) -> bool:
        self.queries.append((decision_time, last_rebalance_time))
        if decision_time in self._seen:
            raise RuntimeError(
                f"the rebalance policy was re-queried for {decision_time!r}: "
                "one query per declared date per run binds every arm"
            )
        self._seen.add(decision_time)
        return True


class _SkewedTargetStrategyStandIn:
    """A synthetic arm emitting one fixed lawful skewed TARGET book."""

    def decide(self, context: Any) -> Any:
        decision = PortfolioDecision(
            decision_time=context.decision_time,
            execution_time=context.decision_time,
            target_weights={"A": 0.25, "B": 0.75},
        )
        return DecisionResult(decision=decision)


def _m44_later_public_members(owner: Any, *needles: str) -> list[str]:
    """Public member names of ``owner`` mentioning any needle — behavior
    discovery for surfaces the milestone pins by behavior, not by symbol."""
    return sorted(
        name
        for name in dir(owner)
        if not name.startswith("_")
        and any(needle in name.lower() for needle in needles)
    )


def test_rebalance_decision_binds_every_arm_identically() -> None:
    """F9 — Law 7(ii): one rebalance-policy query per date per run; the
    resolved fact binds every arm; a policy is never re-queried per arm."""
    module = _experiments()
    dates = list(_M44_DATES)
    config = _config(
        strategy_ids=("alpha", "beta"),
        target="alpha",
        comparators=("beta",),
        benchmarks=("beta",),
    )

    def two_arm_experiment(policy: Any):
        strategy_alpha = _RecordingStrategyStandIn()
        strategy_beta = _RecordingStrategyStandIn()
        experiment = module.Experiment(
            config,
            (
                _registration("alpha", strategy_alpha),
                _registration("beta", strategy_beta),
            ),
            _m44_schedule(),
            _m44_conditions_with(rebalance_policy=policy),
        )
        return experiment, strategy_alpha, strategy_beta

    # Exactly one query per declared date per run — no arm-count term in
    # the query count — and the query carries the lawfully advancing
    # last-rebalance instant (the opening instant before the first
    # execution, then each prior executed decision instant).
    counting = _CountingRebalancePolicy()
    experiment, strategy_alpha, strategy_beta = two_arm_experiment(counting)
    results = experiment.run(module.select_all(config))
    assert [record.strategy_id for record in results] == ["alpha", "beta"]
    assert counting.queries == [
        (dates[0], _M44_OPENING_LAST_REBALANCE),
        (dates[1], dates[0]),
        (dates[2], dates[1]),
    ]
    assert len(strategy_alpha.contexts) == len(dates)
    assert len(strategy_beta.contexts) == len(dates)

    # The same law polices a violating runtime: a policy that treats a
    # second same-date query as a defect must see both arms complete —
    # a per-arm re-query cannot occur anywhere in the run.
    strict = _FirstQueryOnlyRebalancePolicy()
    experiment, strategy_alpha, strategy_beta = two_arm_experiment(strict)
    results = experiment.run(module.select_all(config))
    assert [record.strategy_id for record in results] == ["alpha", "beta"]
    assert len(strategy_alpha.contexts) == len(dates)
    assert len(strategy_beta.contexts) == len(dates)

    # Single-arm driver of the same law: one arm over the full schedule
    # still makes exactly one query per declared date.
    solo = _CountingRebalancePolicy()
    experiment, _strategy = _m44_entry_experiment(
        _m44_conditions_with(rebalance_policy=solo)
    )
    experiment.run(None)
    assert solo.queries == [
        (dates[0], _M44_OPENING_LAST_REBALANCE),
        (dates[1], dates[0]),
        (dates[2], dates[1]),
    ]


def test_decision_loop_advances_in_declared_order() -> None:
    """F10 — Law 12: contexts are constructed and arms evaluated in the
    declared date order and canonical id order; per-arm state threads
    only through ``next_strategy_state``."""
    module = _experiments()
    dates = list(_M44_DATES)
    ledger = _DecideEventLedger()
    strategy_alpha = _StatefulSequencedStrategyStandIn("alpha", ledger)
    strategy_bravo = _StatefulSequencedStrategyStandIn("bravo", ledger)
    strategy_upper = _StatefulSequencedStrategyStandIn("Alpha", ledger)
    config = _config(
        strategy_ids=("bravo", "alpha", "Alpha"),
        target="alpha",
        comparators=("bravo",),
        benchmarks=("bravo",),
    )
    experiment = module.Experiment(
        config,
        (
            _registration("alpha", strategy_alpha),
            _registration("bravo", strategy_bravo),
            _registration("Alpha", strategy_upper),
        ),
        _m44_schedule(),
        _m44_conditions(),
    )
    results = experiment.run(module.select_all(config))

    # The global decide order is date-major, canonical-id-minor: every
    # declared id decides at each date before any id advances — and
    # mixed-case ids stay distinct and canonically ordered within each
    # date (``Alpha`` sorts before ``alpha`` before ``bravo``).
    assert ledger.events == [
        (date, strategy_id)
        for date in dates
        for strategy_id in ("Alpha", "alpha", "bravo")
    ]
    assert [record.strategy_id for record in results] == ["Alpha", "alpha", "bravo"]

    # Each arm begins at the lawful absent state and threads its own
    # transition sequence: every context after the first carries exactly
    # that arm's previous ``next_strategy_state`` — arm-local, never
    # cross-threaded, never silently dropped.
    for strategy in (strategy_alpha, strategy_bravo, strategy_upper):
        assert strategy.states_received == [
            None,
            ("tick", 1),
            ("tick", 2),
        ]


def test_strategy_output_target_passes_centralized_validation() -> None:
    """F11 — Law 10/V5: a lawful TARGET book flows through centralized
    validation into the common execution and accounting engine, and the
    produced record's path summary is exactly the one that engine
    composes from the declared common conditions."""
    module = _experiments()

    # A skewed-but-lawful TARGET (net exposure exactly 1.0, long-only):
    # the frozen weights seam accepts it, so the runtime must too.
    skewed_book = PortfolioWeights({"A": 0.25, "B": 0.75}, WeightState.TARGET)
    require_valid_target(skewed_book, WeightConstraints())

    strategy = _SkewedTargetStrategyStandIn()
    config = _config(
        strategy_ids=("min-var",),
        target="min-var",
        comparators=(),
        benchmarks=(),
    )
    experiment = module.Experiment(
        config,
        (_registration("min-var", strategy),),
        _m44_schedule(),
        _m44_conditions(),
    )
    results = experiment.run(None)

    assert [record.strategy_id for record in results] == ["min-var"]
    path_result = next(iter(results)).paths["main"]

    # The reference path is what the common engine composes from the
    # identical opening state, common factor books, the arm's exact
    # TARGET books, the declared cost model, and the declared
    # accounting engine — the record's derived path summary must match
    # it field for field through the frozen results surface.
    reference = build_ledger(
        {"A": 0.5, "B": 0.5},
        list(_M44_DATES),
        {date: {"A": 1.0, "B": 1.0} for date in _M44_DATES[:-1]},
        [
            PortfolioDecision(
                decision_time=date,
                execution_time=date,
                target_weights={"A": 0.25, "B": 0.75},
            )
            for date in _M44_DATES[:-1]
        ],
        Proportional(rate=0.001, turnover=two_sided),
        ExactFillAccounting(),
        initial_wealth=100.0,
    )
    # Sanity: the skewed book actually turns over against the equal
    # opening (cost bites), so the equality below is informative.
    assert reference.final_wealth < 100.0
    assert path_result.n_periods == reference.n_periods
    assert path_result.period_start == reference.rows[0].period_start
    assert path_result.period_end == reference.rows[-1].period_end
    assert path_result.final_wealth == reference.final_wealth


def test_conformance_verdict_recorded_on_every_arm_record() -> None:
    """F12 — Law 11/V6: every produced record carries a results
    ``ConformanceRecord`` with the computed verdict; declared-equal
    effective settings yield ``consistent``; a covered difference yields
    ``sanctioned_deviation`` with the structured sanction on the
    record — a concealed difference is the one unlawful state."""
    module = _experiments()
    results_module = _results()

    # Declared-equal single arm: the run declares no conditions, so the
    # declared and effective settings are equal and the verdict is the
    # results-computed ``consistent`` token — never fabricated, never
    # missing, and the deviation ledger is empty.
    experiment, _strategy = _m44_entry_experiment(_m44_conditions())
    results = experiment.run(None)
    assert len(results) == 1
    conformance = next(iter(results)).conformance
    assert isinstance(conformance, results_module.ConformanceRecord)
    assert conformance.verdict == "consistent"
    assert tuple(conformance.deviations) == ()

    # Multi-arm: the computed verdict rides on EVERY produced record —
    # never only the target's.
    config = _config(
        strategy_ids=("alpha", "beta"),
        target="alpha",
        comparators=("beta",),
        benchmarks=("beta",),
    )
    experiment = module.Experiment(
        config,
        (
            _registration("alpha", _RuntimeStrategyStandIn()),
            _registration("beta", _RuntimeStrategyStandIn()),
        ),
        _m44_schedule(),
        _m44_conditions(),
    )
    results = experiment.run(module.select_all(config))
    assert [record.strategy_id for record in results] == ["alpha", "beta"]
    for arm_record in results:
        conformance = arm_record.conformance
        assert isinstance(conformance, results_module.ConformanceRecord)
        assert conformance.verdict == "consistent"
        assert tuple(conformance.deviations) == ()

    # A covered declared/effective difference: the declaration pins the
    # evaluation definition ``periods_per_year`` at 4, and what the run
    # actually executed is observable on the record's own metric
    # blocks.  If the run honored the declaration the verdict is
    # ``consistent``; if it ran anything else, the difference is real
    # and the record must disclose it — the results-computed
    # ``sanctioned_deviation`` with a structured sanction covering that
    # exact field.  Recording ``consistent`` over a run that executed
    # something else is the concealed-deviation defect (V6).
    declared_conditions = {"periods_per_year": 4}
    config = _config(
        strategy_ids=("min-var",),
        target="min-var",
        comparators=(),
        benchmarks=(),
        conditions=declared_conditions,
    )
    experiment = module.Experiment(
        config,
        (_registration("min-var", _RuntimeStrategyStandIn()),),
        _m44_schedule(),
        _m44_conditions(),
    )
    results = experiment.run(None)
    assert len(results) == 1
    record = next(iter(results))
    conformance = record.conformance
    assert isinstance(conformance, results_module.ConformanceRecord)
    assert dict(conformance.declared) == {"periods_per_year": 4}
    ran_periods = {block.periods_per_year for block in record.paths["main"].metrics}
    if ran_periods == {4}:
        assert conformance.verdict == "consistent"
    else:
        assert conformance.verdict == "sanctioned_deviation", (
            "the run executed periods_per_year "
            f"{sorted(ran_periods)} against the declared 4: a real "
            "declared/effective difference the record must disclose via "
            "the results-computed verdict — never conceal as 'consistent'"
        )
        assert any(
            deviation.field == "periods_per_year"
            for deviation in conformance.deviations
        )


def test_collection_builder_matches_selected_strategy_ids_exactly() -> None:
    """F13 — Law 11a/V7: for single/subset/full the collection's ids
    equal the selected ids exactly — mixed-case ids stay distinct, one
    record per id, canonical order, and one uniform collection type."""
    module = _experiments()
    config = _config(
        strategy_ids=("bravo", "alpha", "Alpha"),
        target="alpha",
        comparators=("bravo",),
        benchmarks=("bravo",),
    )
    experiment = module.Experiment(
        config,
        (
            _registration("alpha", _RuntimeStrategyStandIn()),
            _registration("bravo", _RuntimeStrategyStandIn()),
            _registration("Alpha", _RuntimeStrategyStandIn()),
        ),
        _m44_schedule(),
        _m44_conditions(),
    )

    # Full: every declared id exactly once — mixed-case ``Alpha`` and
    # ``alpha`` are two distinct ids — in canonical order, collected in
    # the one uniform ``ExperimentResults`` type.
    full = experiment.run(module.select_all(config))
    assert type(full) is module.ExperimentResults
    ids_full = [record.strategy_id for record in full]
    assert ids_full == ["Alpha", "alpha", "bravo"]
    assert len(set(ids_full)) == len(ids_full)
    assert len(full) == 3

    # Subset: exactly the selected ids, canonically ordered.
    subset = experiment.run(module.select_subset(config, ("bravo", "Alpha")))
    assert type(subset) is module.ExperimentResults
    assert [record.strategy_id for record in subset] == ["Alpha", "bravo"]
    assert len(subset) == 2

    # Single: the one selected id, and the single-arm collection is the
    # same uniform type carrying the same record type as the many-arm
    # collections — no special-cased singleton shape.
    single = experiment.run(module.select_single(config, "alpha"))
    assert type(single) is module.ExperimentResults
    assert [record.strategy_id for record in single] == ["alpha"]
    assert len(single) == 1
    assert type(next(iter(single))) is type(next(iter(full)))


def test_comparison_readiness_fails_closed_without_comparator() -> None:
    """F14 — D10/Law 6 V4: comparison readiness passes with at least one
    declared comparator and fails closed with zero, through an explicit
    runtime surface; a target-only run still executes lawfully because
    readiness gates comparison, not execution."""
    module = _experiments()

    def build_experiment(config: Any) -> Any:
        return module.Experiment(
            config,
            tuple(
                _registration(strategy_id, _RuntimeStrategyStandIn())
                for strategy_id in config.strategy_ids
            ),
            _m44_schedule(),
            _m44_conditions(),
        )

    with_comparator_config = _config(
        strategy_ids=("min-var", "equal-weight"),
        target="min-var",
        comparators=("equal-weight",),
        benchmarks=(),
    )
    without_comparator_config = _config(
        strategy_ids=("min-var",),
        target="min-var",
        comparators=(),
        benchmarks=(),
    )
    with_comparator = build_experiment(with_comparator_config)
    without_comparator = build_experiment(without_comparator_config)

    # The target-only RUN is lawful: readiness gates comparison, never
    # execution, so the zero-comparator experiment still produces its
    # record.
    results = without_comparator.run(None)
    assert [record.strategy_id for record in results] == ["min-var"]

    def readiness_surfaces(owner: Any) -> list[tuple[str, Any]]:
        return [
            (name, getattr(owner, name))
            for name in _m44_later_public_members(owner, "readiness", "comparison")
        ]

    surfaces = readiness_surfaces(with_comparator) or readiness_surfaces(module)
    if not surfaces:
        pytest.fail(
            "the comparison-readiness surface is absent: D10 requires an "
            "explicit runtime readiness check — a public name mentioning "
            "'readiness' or 'comparison' on the Experiment object or the "
            "module — and none exists on either"
        )

    def invoke(surface: Any, config: Any, experiment: Any) -> Any:
        for args in ((), (config,), (experiment,)):
            try:
                return surface(*args)
            except TypeError:
                continue
        raise AssertionError(
            "the readiness surface is not invocable with no arguments, "
            "the configuration, or the experiment"
        )

    without_surfaces = dict(
        readiness_surfaces(without_comparator) or readiness_surfaces(module)
    )
    for name, surface in surfaces:
        assert callable(surface), f"readiness surface {name!r} must be callable"
        # Passes with a declared comparator: no rejection, and a boolean
        # return must affirm readiness.
        outcome = invoke(surface, with_comparator_config, with_comparator)
        assert outcome is not False, (
            f"readiness surface {name!r} must pass with a declared comparator"
        )
        # Fails closed with zero comparators — an explicit rejection,
        # never a silent pass.
        surface_without = without_surfaces.get(name, surface)
        with pytest.raises(module.ExecutionError):
            invoke(surface_without, without_comparator_config, without_comparator)


# --------------------------------------------------------------------------- #
# Runtime §8 floors F15-F20 — serialization of the live composition, the          #
# centralized TARGET fail-closed paths (construction, declared bounds,         #
# tolerance identity), concealed-deviation blocking, and the forecast          #
# boundary (point-in-time admission, temporal-protocol provenance, decision-instant        #
# dating).  One behavior-named node per floor, public paths only, real         #
# assertions, no skips: each enumerated direction is asserted; nodes           #
# covering already-correct behavior pass lawfully (green controls).            #
# --------------------------------------------------------------------------- #


class _FixedBookStrategyStandIn:
    """A synthetic arm emitting one fixed TARGET mapping book at every decide."""

    def __init__(self, book: dict[str, float]) -> None:
        self.book = dict(book)
        self.contexts: list[Any] = []

    def decide(self, context: Any) -> Any:
        self.contexts.append(context)
        return DecisionResult(
            decision=PortfolioDecision(
                decision_time=context.decision_time,
                execution_time=context.decision_time,
                target_weights=dict(self.book),
            )
        )


class _SpyAccountingEngine:
    """A delegating accounting backend recording every book it is given."""

    def __init__(self) -> None:
        self.books: list[dict[str, float]] = []

    def account(
        self, decision: Any, pre_trade_weights: Any, realized_returns: Any
    ) -> Any:
        self.books.append(dict(decision.target_weights))
        return ExactFillAccounting().account(
            decision, pre_trade_weights, realized_returns
        )


class _ForecastSupplyStandIn:
    """The common information backend supplying forecast-shaped items.

    Per declared date the resolved ``InformationSet`` carries an optional
    forecast item (a ``Forecast`` value, availability governed by
    ``available_when``) and an optional temporal-protocol provenance item (a
    ``ForecastLink`` value) — the two lawful public channels through which
    a synthetic forecast arm's material reaches the runtime.
    """

    def __init__(self, *, forecast_fn: Any = None, link: Any = None) -> None:
        self._forecast_fn = forecast_fn
        self._link = link

    def __call__(self, instant: Any) -> Any:
        from portlearn.interfaces import TimedObservation

        items = []
        if self._forecast_fn is not None:
            items.append(
                TimedObservation(
                    series_id="m44b-forecast",
                    observation_time=instant,
                    available_time=instant
                    if self._forecast_fn.leak is False
                    else instant + timedelta(days=1),
                    value=self._forecast_fn(instant),
                )
            )
        if self._link is not None:
            items.append(
                TimedObservation(
                    series_id="m44b-provenance",
                    observation_time=instant,
                    available_time=instant,
                    value=self._link,
                )
            )
        return InformationSet(items=items, as_of=instant)


class _ForecastFn:
    """A forecast factory over dates, optionally leaky or off-dated."""

    def __init__(
        self,
        *,
        offset: timedelta | None = None,
        leak: bool = False,
    ) -> None:
        self.offset = offset
        self.leak = leak

    def __call__(self, instant: Any) -> Any:
        from portlearn.interfaces import Forecast

        return Forecast(
            values={"A": 0.001, "B": 0.001},
            target="expected_return",
            decision_time=instant if self.offset is None else instant + self.offset,
            produced_by="m44b-forecaster",
        )


class _ForecastSpyStrategy:
    """A synthetic arm recording the forecast each context carried."""

    def __init__(self) -> None:
        self.forecasts: list[Any] = []
        self.contexts: list[Any] = []

    def decide(self, context: Any) -> Any:
        self.contexts.append(context)
        self.forecasts.append(context.forecast)
        per_asset = 1.0 / len(context.universe)
        return DecisionResult(
            decision=PortfolioDecision(
                decision_time=context.decision_time,
                execution_time=context.decision_time,
                target_weights={asset: per_asset for asset in context.universe},
            )
        )


def _m44b_link() -> Any:
    """A lifecycle-incoherent ForecastLink (fit window after cutoff, refit
    before the window, seed on a nondeterministic class) — lawful to
    supply, and semantically NOT the runtime's to police (A5)."""
    results = _results()
    return results.ForecastLink(
        model="m44b-lstm",
        config_hash="m44b-hash",
        cutoff=_M44_DATES[0],
        seed=7,
        determinism=results.DeterminismClass.NONDETERMINISTIC,
        fit_window_start=_M44_DATES[1],
        refit_instants=(_M44_DATES[0] - timedelta(days=1),),
        forecast_target="expected_return",
        forecast_horizon=timedelta(days=31),
        information=(),
    )


def _m44_invalid_target_experiment(book: dict[str, float], conditions: Any):
    """A one-arm composition whose arm emits ``book`` (F16/F17 driver)."""
    module = _experiments()
    config = _config(
        strategy_ids=("min-var",),
        target="min-var",
        comparators=(),
        benchmarks=(),
    )
    strategy = _FixedBookStrategyStandIn(book)
    experiment = module.Experiment(
        config,
        (_registration("min-var", strategy),),
        _m44_schedule(),
        conditions,
    )
    return experiment, strategy


def test_experiment_live_composition_rejects_serialization() -> None:
    """Floor F15 — serializing the ``Experiment`` (or its conditions' live
    backends, or a live registration) rejects; the configuration's
    canonical declaration bytes remain the only serializable experiment
    artifact (B2: the live binding is process-local, never serialized)."""
    import pickle

    module = _experiments()
    conditions = _m44_conditions()
    registrations = (
        _registration("min-var", _LiveComponent()),
        _registration("equal-weight", _LiveComponent()),
    )
    experiment = module.Experiment(
        _config(),
        registrations,
        _m44_schedule(),
        conditions,
    )
    # Every live object of the composition rejects serialization — the
    # backends here are plain picklable class instances, so a pass is the
    # runtime failing to reject, not an artifact of unpicklable backends.
    for live in (experiment, conditions, *registrations):
        with pytest.raises((TypeError, pickle.PicklingError)):
            pickle.dumps(live)
    # The canonical declaration bytes stay the one serializable artifact:
    # the round trip is green throughout (control).
    text = _canonical(_config())
    reborn = module.ExperimentConfiguration.from_canonical_text(text)
    assert reborn == _config()
    with pytest.raises((TypeError, pickle.PicklingError)):
        pickle.dumps(reborn)


def test_invalid_strategy_target_fails_closed() -> None:
    """Floor F16 — every TARGET the frozen seam rejects raises
    ``ExecutionError`` at that date, before execution/accounting: a
    non-finite weight or blank asset id (``PortfolioWeights``
    construction), a net exposure outside the declared
    ``[budget − budget_tolerance, budget + budget_tolerance]`` interval,
    a negative weight under ``allow_short=False``, and a gross/|net|
    exposure beyond a declared inclusive bound.  The non-TARGET-state
    direction is unreachable through the lawful mapping surface
    (``PortfolioDecision`` carries mappings), so it is pinned by the
    frozen-seam control the centralized call must consume."""
    module = _experiments()

    # Frozen-seam controls (green): the exact validator directions the
    # runtime must evaluate centrally, and its TARGET-only state law.
    require_valid_target(
        PortfolioWeights({"A": 0.25, "B": 0.75}, WeightState.TARGET),
        WeightConstraints(),
    )
    with pytest.raises(ValueError, match="only WeightState.TARGET"):
        require_valid_target(
            PortfolioWeights({"A": 0.5, "B": 0.5}, WeightState.POST_TRADE),
            WeightConstraints(),
        )

    # Each enumerated direction: declared constraints ride the shared
    # conditions mapping; the arm emits the offending book; the run
    # rejects with ExecutionError carrying the frozen offending facts,
    # after exactly one decide, with nothing handed to accounting.
    directions: tuple[tuple[str, dict[str, float], dict[str, Any], str], ...] = (
        (
            "non-finite weight",
            {"A": float("nan"), "B": 0.5},
            {},
            "finite",
        ),
        (
            "blank asset id",
            {"A": 0.5, " ": 0.5},
            {},
            "non-blank",
        ),
        (
            "net outside declared interval",
            {"A": 0.51, "B": 0.51},
            {"budget_tolerance": 0.01},
            "budget_tolerance",
        ),
        (
            "negative weight under allow_short=False",
            {"A": -0.1, "B": 1.1},
            {},
            "allow_short",
        ),
        (
            "gross beyond declared inclusive bound",
            {"A": 1.1, "B": -0.1},
            {"allow_short": True, "max_gross_exposure": 1.1},
            "max_gross_exposure",
        ),
        (
            "|net| beyond declared inclusive bound",
            {"A": 1.4, "B": 0.0},
            {"budget_tolerance": 0.5, "max_abs_net_exposure": 1.2},
            "max_abs_net_exposure",
        ),
    )
    for label, book, declared_constraints, token in directions:
        spy = _SpyAccountingEngine()
        conditions = _m44_conditions_with(accounting_engine=spy)
        config = _config(
            strategy_ids=("min-var",),
            target="min-var",
            comparators=(),
            benchmarks=(),
            conditions=dict(declared_constraints) or None,
        )
        strategy = _FixedBookStrategyStandIn(book)
        experiment = module.Experiment(
            config,
            (_registration("min-var", strategy),),
            _m44_schedule(),
            conditions,
        )
        with pytest.raises(module.ExecutionError, match=token):
            experiment.run(None)
        # The rejection fires at that date, after exactly one decide and
        # before anything is executed or accounted for the arm.
        assert len(strategy.contexts) == 1, label
        assert spy.books == [], label

    # Green control: the lawful skewed book through the identical
    # composition (spy engine consumed) completes — the node is not a
    # blanket rejection of books.
    spy = _SpyAccountingEngine()
    experiment, strategy = _m44_invalid_target_experiment(
        {"A": 0.25, "B": 0.75}, _m44_conditions_with(accounting_engine=spy)
    )
    results = experiment.run(None)
    assert [record.strategy_id for record in results] == ["min-var"]
    assert len(strategy.contexts) == len(_M44_DATES)


def test_invalid_target_is_never_clipped_normalized_or_repaired() -> None:
    """Floor F17 — under a declared ``budget_tolerance`` of 0.01: a TARGET
    beyond the interval by the smallest margin rejects with
    ``ExecutionError`` at that date (the frozen validator's facts, never
    the ledger's), while a TARGET off budget *within* the declared
    tolerance is admitted as-is by the central validation — identical
    weights, byte-for-byte the emitted book — so any failure belongs to
    the frozen ledger's own unit-budget law, never to a tightened
    tolerance, and no renormalized, repaired, or hold-previous path can
    complete the run."""
    module = _experiments()
    tolerance_conditions = {"budget_tolerance": 0.01}

    # Direction 1 — beyond the declared interval by ~5e-11 (the floor's
    # "no matter how small the excess"): the centralized validator
    # rejects at the first date — one decide, nothing accounted, and the
    # declared-tolerance facts on the error (not the ledger's
    # exactly-1.0 message).
    beyond = {"A": 0.50500000005, "B": 0.505}
    spy = _SpyAccountingEngine()
    config = _config(
        strategy_ids=("min-var",),
        target="min-var",
        comparators=(),
        benchmarks=(),
        conditions=tolerance_conditions,
    )
    strategy = _FixedBookStrategyStandIn(beyond)
    experiment = module.Experiment(
        config,
        (_registration("min-var", strategy),),
        _m44_schedule(),
        _m44_conditions_with(accounting_engine=spy),
    )
    with pytest.raises(module.ExecutionError, match="budget_tolerance"):
        experiment.run(None)
    assert len(strategy.contexts) == 1
    assert spy.books == []

    # Direction 2 — off budget within the declared tolerance: the
    # central validation ADMITS the book (the frozen seam accepts it), so
    # the run can never complete on a renormalized or repaired book, the
    # rejection never speaks the validator's tolerance token, and the
    # exact emitted net is what the failure reports.
    within = {"A": 0.5025, "B": 0.5025}
    require_valid_target(
        PortfolioWeights(dict(within), WeightState.TARGET),
        WeightConstraints(budget_tolerance=0.01),
    )
    strategy = _FixedBookStrategyStandIn(within)
    experiment = module.Experiment(
        config,
        (_registration("min-var", strategy),),
        _m44_schedule(),
        _m44_conditions(),
    )
    with pytest.raises((module.ExecutionError, ValueError)) as excinfo:
        experiment.run(None)
    message = str(excinfo.value)
    assert "budget_tolerance" not in message, (
        "a TARGET within the declared tolerance was rejected by the "
        "central validation — the tightened-tolerance path"
    )
    assert "1.005" in message, (
        "the failure must report the exact emitted book (net 1.005) — "
        "a renormalized book would report 1.0 and complete"
    )


def test_unrecorded_deviation_blocks_run_at_record_construction() -> None:
    """Floor F18 — an arm whose effective settings differ from declared
    without a sanction fails the run at the results record-construction law
    (the uncovered difference rejecting ``ConformanceRecord``): the run
    ends fail-closed with no record and no partial collection.  The one
    concealed completion — verdict ``consistent`` over a run that
    executed something else — is the unlawful state."""
    module = _experiments()
    results_module = _results()
    declared_conditions = {"periods_per_year": 4}
    config = _config(
        strategy_ids=("alpha", "beta"),
        target="alpha",
        comparators=("beta",),
        benchmarks=("beta",),
        conditions=declared_conditions,
    )
    experiment = module.Experiment(
        config,
        (
            _registration("alpha", _RuntimeStrategyStandIn()),
            _registration("beta", _RuntimeStrategyStandIn()),
        ),
        _m44_schedule(),
        _m44_conditions(),
    )
    try:
        results = experiment.run(module.select_all(config))
    except (module.ExecutionError, results_module.ResultStructureError) as exc:
        # The fail-closed end: the results construction rejection surfacing
        # as the run's end, naming the uncovered field.
        assert "periods_per_year" in str(exc)
    else:
        # The honored end: the run executed the declared settings, so
        # every record's blocks carry them and the verdict is the
        # computed ``consistent`` — anything else is concealment.
        assert [record.strategy_id for record in results] == ["alpha", "beta"]
        for record in results:
            ran = {block.periods_per_year for block in record.paths["main"].metrics}
            assert ran == {4}, (
                "the run executed periods_per_year "
                f"{sorted(ran)} against the declared 4 while recording "
                f"verdict {record.conformance.verdict!r} — an unrecorded "
                "deviation concealed as a result"
            )
            assert record.conformance.verdict == "consistent"


def test_forecast_fit_boundary_violation_fails_closed() -> None:
    """Floor F19 — the runtime-side forecast boundary only: a forecast whose
    information is not available by the decision instant never enters the
    ``DecisionContext`` (the frozen point-in-time admission law firing
    through the runtime as ``ExecutionError``); a lawfully available
    forecast does enter the context dated exactly at its decision
    instant; and a present temporal-protocol provenance declaration is recorded
    as-is on the record — dropping it, or semantically interpreting it
    (re-running any forecasting-lifecycle fitting validation inside the runtime), has no
    lawful path: the lifecycle-incoherent link below must pass through
    untouched."""
    module = _experiments()

    def experiment_with(provider: Any, strategy: Any):
        config = _config(
            strategy_ids=("min-var",),
            target="min-var",
            comparators=(),
            benchmarks=(),
        )
        return module.Experiment(
            config,
            (_registration("min-var", strategy),),
            _m44_schedule(),
            _m44_conditions_with(information_provider=provider),
        )

    # Direction 1 — PIT leakage: an item unavailable at the decision
    # instant rejects with ExecutionError through the runtime (green
    # control: the frozen FutureInformationError firing at InformationSet
    # construction, surfaced by the backend-resolution law).
    leaky = _ForecastSupplyStandIn(forecast_fn=_ForecastFn(leak=True))
    with pytest.raises(module.ExecutionError, match="information"):
        experiment_with(leaky, _RecordingStrategyStandIn()).run(None)

    # Direction 2 — the lawful forecast enters the context: every decide
    # carries a forecast dated exactly at that context's decision
    # instant, with the producer's identity preserved.
    spy = _ForecastSpyStrategy()
    experiment_with(_ForecastSupplyStandIn(forecast_fn=_ForecastFn()), spy).run(None)
    assert len(spy.contexts) == len(_M44_DATES)
    for context, forecast in zip(spy.contexts, spy.forecasts, strict=True):
        assert forecast is not None, (
            "a lawfully supplied forecast was dropped before the strategy "
            "decided — the DecisionContext forecast channel is absent"
        )
        assert forecast.decision_time == context.decision_time
        assert forecast.produced_by == "m44b-forecaster"
        assert dict(forecast.values) == {"A": 0.001, "B": 0.001}

    # Direction 3 — present temporal-protocol provenance is recorded as-is: the link
    # is lifecycle-incoherent (fit window after cutoff, refit before it, seed
    # on a nondeterministic class), the run still completes (no fitting
    # validation exists inside the runtime), and the record carries the
    # link unchanged — never dropped, never interpreted.
    link = _m44b_link()
    record = next(
        iter(
            experiment_with(
                _ForecastSupplyStandIn(link=link), _RecordingStrategyStandIn()
            ).run(None)
        )
    )
    recorded = tuple(record.forecasts)
    assert recorded == (link,) or recorded == tuple(link for _ in _M44_DATES), (
        f"present temporal-protocol provenance must be recorded as-is, got {recorded!r}"
    )


def test_forecast_must_be_dated_at_its_decision_instant() -> None:
    """Floor F20 — a forecast dated off its context's ``decision_time``
    rejects: the frozen ``DecisionContext`` forecast-dating law surfaced
    through the runtime, in both the future-dated and past-dated
    directions.  The item is lawfully available at the instant (the
    admission law is silent on dating), so dropping the forecast and
    completing is the defect; the frozen ``InvalidChronologyError``
    surfaces unchanged (D13)."""
    from portlearn.timing import InvalidChronologyError

    module = _experiments()

    for label, offset in (
        ("future-dated", timedelta(days=31)),
        ("past-dated", -timedelta(days=31)),
    ):
        provider = _ForecastSupplyStandIn(forecast_fn=_ForecastFn(offset=offset))
        strategy = _RecordingStrategyStandIn()
        config = _config(
            strategy_ids=("min-var",),
            target="min-var",
            comparators=(),
            benchmarks=(),
        )
        experiment = module.Experiment(
            config,
            (_registration("min-var", strategy),),
            _m44_schedule(),
            _m44_conditions_with(information_provider=provider),
        )
        with pytest.raises(InvalidChronologyError):
            experiment.run(None)
        # The law fires while composing the decision context — before the
        # strategy ever decides on a mis-dated forecast.
        assert strategy.contexts == []


# --------------------------------------------------------------------------- #
# Runtime §8 floors F21-F26 (micro-tranche 3) — the collection-builder             #
# linkage law with no partial results, the in-loop fail-closed family,          #
# absent-injection rejection with no fallback, the unchanged declaration               #
# declaration surface, the one-runtime-type/one-run-entry census, and           #
# the frozen timing-error taxonomy surfacing unchanged.  One behavior-          #
# named node per floor, public paths only, real assertions, no skips:           #
# nodes covering already-correct behavior pass lawfully (green controls);       #
# the RED nodes fail causally on the missing fail-closed wrap.                  #
# --------------------------------------------------------------------------- #


class _RaisingDecideStrategy:
    """A synthetic arm whose ``decide`` fails inside the decision loop."""

    def __init__(self) -> None:
        self.contexts: list[Any] = []

    def decide(self, context: Any) -> Any:
        self.contexts.append(context)
        raise RuntimeError("strategy failed inside the decision loop")


class _LoopFailureAccountingEngine:
    """A delegating accounting backend that fails mid-loop.

    Carries the frozen three-argument ``account`` signature —
    ``(decision, pre_trade_weights, realized_returns)`` — and delegates
    to the shipped ``ExactFillAccounting``; it never re-declares the
    seam with a shape of its own.
    """

    def __init__(self, fail_on_call: int = 1) -> None:
        self._inner = ExactFillAccounting()
        self.calls: list[Any] = []
        self._fail_on_call = fail_on_call

    def account(
        self, decision: Any, pre_trade_weights: Any, realized_returns: Any
    ) -> Any:
        self.calls.append(decision)
        if len(self.calls) >= self._fail_on_call:
            raise RuntimeError(f"accounting engine failed on call {len(self.calls)}")
        return self._inner.account(decision, pre_trade_weights, realized_returns)


class _LoopFailureRebalancePolicy:
    """A rebalance policy whose query fails inside the decision loop."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def should_rebalance(self, decision_time: Any, last_rebalance_time: Any) -> bool:
        self.calls.append(decision_time)
        raise RuntimeError("rebalance policy failed inside the decision loop")


class _LoopFailureCostModel(Proportional):
    """The shipped proportional cost model failing at ledger build time."""

    def f_cost(self, trade: Any) -> float:
        raise RuntimeError("cost model failed while building the ledger")


class _MisdatedDecisionStrategy:
    """A lawful-shape arm whose decision is dated off its context instant."""

    def decide(self, context: Any) -> Any:
        earlier = context.decision_time - timedelta(days=1)
        return DecisionResult(
            decision=PortfolioDecision(
                decision_time=earlier,
                execution_time=earlier,
                target_weights={"A": 0.5, "B": 0.5},
            )
        )


def _m44_one_arm(strategy: Any, **overrides: Any) -> tuple[Any, Any]:
    """A minimal one-arm runnable composition over the accepted fixtures."""
    module = _experiments()
    config = _config(
        strategy_ids=("min-var",),
        target="min-var",
        comparators=(),
        benchmarks=(),
    )
    experiment = module.Experiment(
        config,
        (_registration("min-var", strategy),),
        _m44_schedule(),
        _m44_conditions_with(**overrides),
    )
    return experiment, config


def test_run_collection_linkage_exact_and_never_partial() -> None:
    """Floor F21 — the collection builder links records to the selected
    ids exactly — verbatim and case-exact (``Alpha``/``alpha`` never
    fold), no selected id missing, no unselected id added, no id
    duplicated — for every mode including ``none``; and a run whose arm
    fails mid-loop ends fail-closed with ``ExecutionError`` and no
    partial collection: a completed-arms prefix never escapes as a
    result."""
    module = _experiments()
    config = _config(
        strategy_ids=("alpha", "Alpha", "bravo"),
        target="alpha",
        comparators=("bravo",),
        benchmarks=("bravo",),
    )
    experiment = module.Experiment(
        config,
        tuple(
            _registration(strategy_id, _RuntimeStrategyStandIn())
            for strategy_id in ("Alpha", "alpha", "bravo")
        ),
        _m44_schedule(),
        _m44_conditions(),
    )

    # Linkage exactness for every mode: the collection's ids equal the
    # selected ids verbatim, one record per id, canonically ordered.
    for selection, expected in (
        (module.select_all(config), ["Alpha", "alpha", "bravo"]),
        (module.select_subset(config, ("alpha", "Alpha")), ["Alpha", "alpha"]),
        (module.select_single(config, "alpha"), ["alpha"]),
        (module.select_none(config), []),
    ):
        results = experiment.run(selection)
        assert type(results) is module.ExperimentResults
        collected = [record.strategy_id for record in results]
        assert collected == expected
        assert len(set(collected)) == len(collected)
        assert len(results) == len(expected)

    # The duplicate-record direction: two records for one id can never
    # be collected — the uniform collection rejects them fail-closed
    # through its own public constructor (green control).
    with pytest.raises(module.ConfigurationError):
        module.ExperimentResults(records=(_record("dup"), _record("dup")))

    # No partial result: the last arm fails mid-loop after two arms
    # completed, and the whole run rejects with ``ExecutionError`` —
    # never a collection carrying the completed arms only.
    failing = module.Experiment(
        config,
        (
            _registration("alpha", _RuntimeStrategyStandIn()),
            _registration("Alpha", _RuntimeStrategyStandIn()),
            _registration("bravo", _RaisingDecideStrategy()),
        ),
        _m44_schedule(),
        _m44_conditions(),
    )
    with pytest.raises(module.ExecutionError):
        failing.run(None)
    # The failed run leaves no partial state behind: the surviving arms
    # still run to their own complete, exact collection.
    survivors = failing.run(module.select_single(config, "alpha"))
    assert [record.strategy_id for record in survivors] == ["alpha"]


def test_in_loop_backend_failures_reject_execution_error_no_partial() -> None:
    """Floor F22 — a strategy, accounting-engine, rebalance-policy, or
    cost-model failure inside the run becomes one fail-closed
    ``ExecutionError`` with no partial result, and the common providers
    are resolved once per declared date and holding-segment start at
    entry — never re-called inside the decision loop."""
    module = _experiments()

    def rejects(strategy: Any, **overrides: Any) -> None:
        experiment, _used_config = _m44_one_arm(strategy, **overrides)
        with pytest.raises(module.ExecutionError):
            experiment.run(None)

    # Strategy failure inside the loop.
    rejects(_RaisingDecideStrategy())
    # Accounting failure inside the loop (frozen three-argument seam).
    rejects(
        _RecordingStrategyStandIn(),
        accounting_engine=_LoopFailureAccountingEngine(),
    )
    # Rebalance-policy failure inside the loop.
    rejects(
        _RecordingStrategyStandIn(),
        rebalance_policy=_LoopFailureRebalancePolicy(),
    )
    # Cost-model failure at ledger construction — after the loop but
    # still inside the run, so no partial result may escape there
    # either.
    rejects(
        _RecordingStrategyStandIn(),
        cost_model=_LoopFailureCostModel(rate=0.001, turnover=two_sided),
    )

    # Providers are not called in the loop: one information resolution
    # per declared decision date and one outcome resolution per holding
    # segment start, regardless of how many arms the run executes.
    info = _InformationProviderStandIn()
    outcome = _OutcomeProviderStandIn(_M44_UNIVERSES)
    config = _config(
        strategy_ids=("alpha", "beta"),
        target="alpha",
        comparators=("beta",),
        benchmarks=("beta",),
    )
    experiment = module.Experiment(
        config,
        (
            _registration("alpha", _RuntimeStrategyStandIn()),
            _registration("beta", _RuntimeStrategyStandIn()),
        ),
        _m44_schedule(),
        _m44_conditions_with(information_provider=info, outcome_provider=outcome),
    )
    results = experiment.run(None)
    assert [record.strategy_id for record in results] == ["alpha", "beta"]
    assert info.calls == list(_M44_DATES)
    assert outcome.calls == list(_M44_DATES[:-1])


_M44_CONDITIONS_FIELDS = (
    "information_provider",
    "outcome_provider",
    "rebalance_policy",
    "accounting_engine",
    "cost_model",
    "initial_weights",
    "initial_wealth",
    "initial_last_rebalance_time",
)


def test_absent_backend_or_opening_state_rejects_no_fallback() -> None:
    """Floor F23 — an absent backend or opening-state field rejects at
    run entry with ``ExecutionError`` naming the absent field, before
    the first decision and before any provider is resolved: nothing
    defaults, nothing is inferred, and no fallback calendar, provider,
    or opening state is invented.  An absent schedule rejects at
    composition with no default schedule."""
    module = _experiments()
    config = _config(
        strategy_ids=("min-var",),
        target="min-var",
        comparators=(),
        benchmarks=(),
    )

    # Every absent field rejects at entry, naming the field, with the
    # strategy never deciding (green control: V3 presence precedes V1).
    for absent_field in _M44_CONDITIONS_FIELDS:
        strategy = _RecordingStrategyStandIn()
        fields: dict[str, Any] = {
            "information_provider": _InformationProviderStandIn(),
            "outcome_provider": _OutcomeProviderStandIn(_M44_UNIVERSES),
            "rebalance_policy": _AlwaysRebalanceStandIn(),
            "accounting_engine": ExactFillAccounting(),
            "cost_model": Proportional(rate=0.001, turnover=two_sided),
            "initial_weights": {"A": 0.5, "B": 0.5},
            "initial_wealth": 100.0,
            "initial_last_rebalance_time": _M44_OPENING_LAST_REBALANCE,
        }
        del fields[absent_field]
        experiment = module.Experiment(
            config,
            (_registration("min-var", strategy),),
            _m44_schedule(),
            module.ExperimentConditions(**fields),
        )
        with pytest.raises(module.ExecutionError, match=absent_field):
            experiment.run(None)
        assert strategy.contexts == []

    # Presence precedes resolution: with an absent field the providers
    # are never even called — there is no resolution-then-reject and no
    # fallback supply behind an absent injection.
    provider = _InformationProviderStandIn()
    outcome = _OutcomeProviderStandIn(_M44_UNIVERSES)
    partial = {
        "information_provider": provider,
        "outcome_provider": outcome,
        "rebalance_policy": _AlwaysRebalanceStandIn(),
        "accounting_engine": ExactFillAccounting(),
        "initial_weights": {"A": 0.5, "B": 0.5},
        "initial_wealth": 100.0,
        "initial_last_rebalance_time": _M44_OPENING_LAST_REBALANCE,
    }
    experiment = module.Experiment(
        config,
        (_registration("min-var", _RecordingStrategyStandIn()),),
        _m44_schedule(),
        module.ExperimentConditions(**partial),
    )
    with pytest.raises(module.ExecutionError, match="cost_model"):
        experiment.run(None)
    assert provider.calls == []
    assert outcome.calls == []

    # An absent schedule rejects at composition: ``Experiment`` admits
    # only a declared ``DecisionSchedule`` — no default calendar is
    # invented behind a missing or unusable schedule.
    for bad_schedule in (None, "monthly", ("A", "B"), _M44_DATES):
        with pytest.raises(module.ConfigurationError):
            module.Experiment(
                config,
                (_registration("min-var", _RuntimeStrategyStandIn()),),
                bad_schedule,
                _m44_conditions(),
            )


def test_declaration_surface_unchanged() -> None:
    """Floor F24 — the declaration surface is unchanged by the runtime
    runtime: the canonical round-trip still reconstructs the equal
    declaration, the declaration and every declarative shape carry no
    execution entry point, and per-strategy conditions still have no
    field on the constructor — the one shared-conditions container is
    the whole vocabulary (``TypeError``, never a quiet merge)."""
    module = _experiments()
    config = _config(conditions={"risk_aversion": 2.5, "universe": ("A", "B")})

    # The canonical round-trip is unchanged.
    text = config.canonical_text()
    reborn = module.ExperimentConfiguration.from_canonical_text(text)
    assert reborn == config
    assert reborn.canonical_text() == text
    assert dict(reborn.conditions) == dict(config.conditions)

    # No declarative shape ever gains a run/execution entry point.
    selection = module.select_all(config)
    for owner in (config, selection):
        for token in ("run", "execute", "apply_conditions"):
            assert not hasattr(owner, token), token
    results = module.ExperimentResults(records=())
    assert not hasattr(results, "run")

    # Per-strategy conditions are absent from the declaration surface:
    # the constructor admits no such field (exact-field-set law), and
    # the canonical bytes carry the one shared container only.
    with pytest.raises(TypeError):
        module.ExperimentConfiguration(
            strategy_ids=("min-var", "equal-weight"),
            target="min-var",
            comparators=("equal-weight",),
            ablations=(),
            robustness_checks=(),
            benchmarks=("equal-weight",),
            per_strategy_conditions={"min-var": {"risk_aversion": 1.0}},
        )
    conditions_text = _canonical(
        _config(
            strategy_ids=("min-var", "equal-weight"),
            conditions={"risk_aversion": 2.5, "universe": ("A", "B")},
        )
    )
    assert "risk_aversion" in conditions_text
    assert "per_strategy" not in conditions_text


def test_one_experiment_runtime_type_one_run_entry() -> None:
    """Floor F25 — exactly one Experiment runtime type and exactly one
    run entry point: ``Experiment`` is the sole exported type owning a
    ``run`` (defined on the class itself, not inherited), no other
    exported runtime shape is executable, the four selection functions
    are the only exported plain functions, and a live experiment
    exposes exactly one public member — ``run``."""
    module = _experiments()

    runtime_types = [
        name
        for name in module.__all__
        if isinstance(getattr(module, name), type)
        and hasattr(getattr(module, name), "run")
    ]
    assert runtime_types == ["Experiment"]

    # The one runtime type defines run itself — never an inherited or
    # mixed-in execution surface.
    assert "run" in module.Experiment.__dict__

    # No second runtime type: every other exported type is
    # non-executable, and the exported plain functions are exactly the
    # four declaration selection constructors — no free-function executor.
    for name in module.__all__:
        exported = getattr(module, name)
        if isinstance(exported, type):
            if exported is not module.Experiment:
                assert not hasattr(exported, "run"), name
        elif callable(exported):
            assert name in {
                "select_all",
                "select_none",
                "select_single",
                "select_subset",
            }, name

    # A live experiment exposes exactly one public member: run.
    experiment, _config_used = _m44_one_arm(_RuntimeStrategyStandIn())
    public_members = [name for name in dir(experiment) if not name.startswith("_")]
    assert public_members == ["run"]
    assert callable(experiment.run)


def test_frozen_timing_errors_surface_unchanged() -> None:
    """Floor F26 — the frozen timing-error taxonomy surfaces unchanged
    through the experiment runtime: a strategy decision dated off its context
    rejects with the raw frozen ``InvalidChronologyError`` (never
    generically wrapped into ``ExecutionError``, never swallowed into a
    completed run), and the schedule's screening contract rejects with
    the raw frozen ``NaiveTimestampError`` and ``FutureInformationError``
    exactly as declared."""
    from portlearn.timing import InvalidChronologyError

    module = _experiments()

    # A mis-dated decision surfaces the frozen chronology error itself,
    # not a generic wrapper — and never a completed run.
    experiment, _config_used = _m44_one_arm(_MisdatedDecisionStrategy())
    with pytest.raises(InvalidChronologyError):
        experiment.run(None)

    # The screening contract surfaces the frozen point-in-time taxonomy
    # unchanged: a naive screening instant or naive membership
    # availability rejects with ``NaiveTimestampError``; a screening
    # instant postdating its decision date, or membership evidence
    # available only after the screening instant, rejects with
    # ``FutureInformationError``.
    ok_instants = {date: date for date in _M44_DATES}
    aware_books = {
        date: {asset: _M44_OPENING_LAST_REBALANCE for asset in _M44_UNIVERSES[date]}
        for date in _M44_DATES
    }
    naive_instants = {
        date: datetime(2026, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        for date in _M44_DATES
    }
    naive_books = {
        date: {
            asset: datetime(2025, 1, 1, tzinfo=UTC).replace(tzinfo=None)
            for asset in _M44_UNIVERSES[date]
        }
        for date in _M44_DATES
    }
    late_instants = {date: date + timedelta(days=1) for date in _M44_DATES}
    late_books = {
        date: {asset: date + timedelta(days=2) for asset in _M44_UNIVERSES[date]}
        for date in _M44_DATES
    }
    for label, instants, books, expected in (
        (
            "naive screening instant",
            naive_instants,
            aware_books,
            NaiveTimestampError,
        ),
        (
            "naive membership availability",
            ok_instants,
            naive_books,
            NaiveTimestampError,
        ),
        (
            "late screening instant",
            late_instants,
            aware_books,
            FutureInformationError,
        ),
        (
            "late membership availability",
            ok_instants,
            late_books,
            FutureInformationError,
        ),
    ):
        with pytest.raises(expected):
            module.DecisionSchedule(
                decision_dates=_M44_DATES,
                universes=_M44_UNIVERSES,
                screening_instants=instants,
                membership_available_times=books,
            )
        assert label

    # Each taxonomy member fires on its exact trigger (green control:
    # the direction-specific pins).
    with pytest.raises(NaiveTimestampError):
        module.DecisionSchedule(
            decision_dates=_M44_DATES,
            universes=_M44_UNIVERSES,
            screening_instants=naive_instants,
            membership_available_times=aware_books,
        )
    with pytest.raises(FutureInformationError):
        module.DecisionSchedule(
            decision_dates=_M44_DATES,
            universes=_M44_UNIVERSES,
            screening_instants=late_instants,
            membership_available_times=aware_books,
        )


# --------------------------------------------------------------------------- #
# Runtime §8 floors F27-F29 (tranche 4A append-only; the 99-node prefix above is  #
# byte-preserved).                                                            #
# --------------------------------------------------------------------------- #


def test_run_output_determinism_canonical_order_and_bytes() -> None:
    """Floor F27 - Law 14/D12: the identical declared configuration with
    the identical injected backends composes deterministically: full and
    subset modes execute every selected id exactly once in canonical id
    order (never declaration, registration, or selection order), and each
    arm's record bytes are identical run-over-run and mode-over-mode once
    the frozen per-run identity (the run link and the creation instant)
    is normalized away."""
    module = _experiments()
    config = _config(
        strategy_ids=("zeta", "alpha"),
        target="alpha",
        comparators=("zeta",),
        benchmarks=(),
    )

    def _fresh_experiment():
        return module.Experiment(
            config,
            (
                _registration("zeta", _RuntimeStrategyStandIn()),
                _registration("alpha", _RuntimeStrategyStandIn()),
            ),
            _m44_schedule(),
            _m44_conditions(),
        )

    def _normalized_bytes(record: Any) -> str:
        payload = json.loads(record.to_json())
        assert {"run_id", "manifest_digest", "created_at"} <= set(payload)
        for derived in ("run_id", "manifest_digest", "created_at"):
            del payload[derived]
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    # Declared zeta-first and registered zeta-first; every mode executes
    # in canonical id order, and a zeta-first selection never perturbs it.
    first = list(_fresh_experiment().run(None))
    second = list(_fresh_experiment().run(None))
    single = list(_fresh_experiment().run(module.select_subset(config, ("zeta",))))
    reordered = list(
        _fresh_experiment().run(module.select_subset(config, ("zeta", "alpha")))
    )
    assert [record.strategy_id for record in first] == ["alpha", "zeta"]
    assert [record.strategy_id for record in second] == ["alpha", "zeta"]
    assert [record.strategy_id for record in single] == ["zeta"]
    assert [record.strategy_id for record in reordered] == ["alpha", "zeta"]

    # Byte-identical per arm run-over-run and mode-over-mode.
    assert [_normalized_bytes(record) for record in first] == [
        _normalized_bytes(record) for record in second
    ]
    assert [_normalized_bytes(record) for record in reordered] == [
        _normalized_bytes(record) for record in first
    ]
    assert _normalized_bytes(first[1]) == _normalized_bytes(single[0])

    # One shared run link per run, fresh identity per run: the
    # normalization above removed real distinguishing content.
    assert first[0].run_id == first[1].run_id
    assert first[0].run_id != second[0].run_id

    # The derived path content is itself deterministic and identical
    # across identical arms (shared snapshot, shared engine).
    digest = first[0].paths["main"].ledger_digest
    assert digest == first[1].paths["main"].ledger_digest
    assert digest == second[0].paths["main"].ledger_digest
    assert digest == single[0].paths["main"].ledger_digest


def test_public_import_path_reaches_runtime_surface() -> None:
    """Floor F28 - D12: ``portlearn.experiments`` and its ``__all__``
    export the runtime names; the root ``portlearn.__all__`` remains
    unchanged; the root lazy facade (the PEP 562 ``__getattr__``)
    resolves the module; and the runtime names stay module-level
    exports of ``portlearn.experiments``, never root-level names."""
    import portlearn as pl

    module = _experiments()
    runtime_names = (
        "ExecutionError",
        "DecisionSchedule",
        "ExperimentConditions",
        "Experiment",
    )

    # The module's __all__ exports the runtime names alongside the
    # frozen declaration nine (one surface, thirteen names).
    assert isinstance(module.__all__, list)
    for name in runtime_names:
        assert name in module.__all__, name
    for name in (
        "ConfigurationError",
        "StrategyRegistration",
        "ExperimentConfiguration",
        "ExperimentSelection",
        "ExperimentResults",
        "select_all",
        "select_none",
        "select_single",
        "select_subset",
    ):
        assert name in module.__all__, name

    # The root __all__ remains exactly the frozen trio.
    assert pl.__all__ == ["__version__", "data", "weights"]

    # The lazy facade attribute resolves the module itself.
    from portlearn import experiments as via_facade

    assert via_facade is module
    assert pl.experiments is module
    assert "experiments" in dir(pl)

    # The runtime names are module-level exports, never root-level
    # names: the facade answers only with the module, and the root
    # package never grows the runtime surface directly.
    for name in runtime_names:
        assert hasattr(module, name), name
        assert not hasattr(pl, name), name
        assert name not in pl.__all__
        assert name not in dir(pl)


def test_outcome_supply_resolves_common_growth_factors_once_per_segment_start() -> (
    None,
):
    """Floor F29 - Law 7(ii)/Law 12b/V3/A1: the outcome provider is
    called exactly once per required segment-start instant - the first
    decision date itself plus every later declared date that begins a
    holding segment, none before the first decision and none at the
    horizon's last date - before the first decision and never inside
    the decision loop; every consumed book's key set equals its
    instant's declared-universe id set; every arm drifts over the
    identical snapshotted resolutions; and a none-mode run resolves no
    outcomes."""
    module = _experiments()
    events: list[tuple[str, Any]] = []

    class _TracingOutcomeProvider:
        """The counting stand-in: one flat book per requested instant."""

        def __init__(self) -> None:
            self.calls: list[Any] = []
            self.books: dict[Any, dict[str, float]] = {}

        def __call__(self, instant: Any) -> dict[str, float]:
            self.calls.append(instant)
            events.append(("outcome", instant))
            book = {asset: 1.0 for asset in _M44_UNIVERSES[instant]}
            self.books[instant] = book
            return book

    class _TracingStrategyStandIn:
        """An equal-weight arm tracing its decision instants."""

        def decide(self, context: Any) -> Any:
            events.append(("decide", context.decision_time))
            per_asset = 1.0 / len(context.universe)
            return DecisionResult(
                decision=PortfolioDecision(
                    decision_time=context.decision_time,
                    execution_time=context.decision_time,
                    target_weights={asset: per_asset for asset in context.universe},
                )
            )

    config = _config(
        strategy_ids=("alpha", "zeta"),
        target="alpha",
        comparators=("zeta",),
        benchmarks=(),
    )

    def _traced_experiment(provider: Any):
        conditions = module.ExperimentConditions(
            information_provider=_InformationProviderStandIn(),
            outcome_provider=provider,
            rebalance_policy=_AlwaysRebalanceStandIn(),
            accounting_engine=ExactFillAccounting(),
            cost_model=Proportional(rate=0.001, turnover=two_sided),
            initial_weights={"A": 0.5, "B": 0.5},
            initial_wealth=100.0,
            initial_last_rebalance_time=_M44_OPENING_LAST_REBALANCE,
        )
        return module.Experiment(
            config,
            (
                _registration("alpha", _TracingStrategyStandIn()),
                _registration("zeta", _TracingStrategyStandIn()),
            ),
            _m44_schedule(),
            conditions,
        )

    # Full mode with two arms: exactly the segment-start instants, once
    # each - never per arm, never before the first decision date, never
    # at the last declared date (it begins no holding segment).
    provider = _TracingOutcomeProvider()
    results = list(_traced_experiment(provider).run(None))
    assert provider.calls == list(_M44_DATES[:-1])
    assert _M44_DATES[-1] not in provider.calls
    assert len(provider.calls) == len(_M44_DATES) - 1

    # Every consumed book's key set equals its instant's declared
    # universe id set exactly (A1 - Law 12b).
    for instant, book in provider.books.items():
        assert set(book) == set(_M44_UNIVERSES[instant])

    # V3: every outcome resolution precedes the first decision - no
    # provider call ever occurs inside the decision loop.
    first_decision = next(
        index for index, event in enumerate(events) if event[0] == "decide"
    )
    assert all(event[0] == "outcome" for event in events[:first_decision])

    # Both arms drift over the identical snapshotted instant-to-book
    # resolutions: identical arms compose identical path content.
    assert len(results) == 2
    assert (
        results[0].paths["main"].ledger_digest == results[1].paths["main"].ledger_digest
    )

    # A none-mode run resolves no outcomes.
    silent_provider = _TracingOutcomeProvider()
    none_results = list(
        _traced_experiment(silent_provider).run(module.select_none(config))
    )
    assert len(none_results) == 0
    assert silent_provider.calls == [], (
        f"a none-mode run resolves no outcomes; resolved {silent_provider.calls}"
    )


def test_outcome_boundary_is_separate_from_information_set() -> None:
    """Floor F30 - B1/A1/Law 12b: the realized-outcome supply is a
    distinct injected ``outcome_provider`` callable - never the
    information provider, never an ``InformationSet`` - and its V3
    factor-book snapshot is execution/accounting-private: no outcome
    factor ever reaches a strategy-visible surface, an invalid book
    rejects at entry with ``ExecutionError``, and a frozen
    ledger-mandated rejection surfaces unchanged."""

    module = _experiments()
    config = _config(
        strategy_ids=("alpha",),
        target="alpha",
        comparators=(),
        benchmarks=(),
    )

    class _BookProvider:
        """A distinct outcome backend serving declared books."""

        def __init__(self, books: Mapping[Any, Mapping[str, float]]) -> None:
            self._books = books
            self.calls: list[Any] = []

        def __call__(self, instant: Any) -> Any:
            self.calls.append(instant)
            return dict(self._books[instant])

    class _DualRoleProvider:
        """One object serving both the information and outcome roles."""

        def __init__(self) -> None:
            self.calls: list[Any] = []

        def __call__(self, instant: Any) -> Any:
            self.calls.append(instant)
            return InformationSet(items=(), as_of=instant)

    class _InformationSetProvider:
        """An outcome backend resolving to an InformationSet."""

        def __call__(self, instant: Any) -> Any:
            return InformationSet(items=(), as_of=instant)

    def _experiment(
        information_provider: Any,
        outcome_provider: Any,
        strategy: Any,
    ):
        return module.Experiment(
            config,
            (_registration("alpha", strategy),),
            _m44_schedule(),
            module.ExperimentConditions(
                information_provider=information_provider,
                outcome_provider=outcome_provider,
                rebalance_policy=_AlwaysRebalanceStandIn(),
                accounting_engine=ExactFillAccounting(),
                cost_model=Proportional(rate=0.001, turnover=two_sided),
                initial_weights={"A": 0.5, "B": 0.5},
                initial_wealth=100.0,
                initial_last_rebalance_time=_M44_OPENING_LAST_REBALANCE,
            ),
        )

    # The realized-outcome supply is a distinct injected callable: the
    # same object serving both surfaces rejects at entry, before any
    # decision is taken.
    dual = _DualRoleProvider()
    strategy = _RecordingStrategyStandIn()
    with pytest.raises(module.ExecutionError):
        _experiment(dual, dual, strategy).run(None)
    assert strategy.contexts == []

    # An ``InformationSet`` (any non-mapping) is never the outcome
    # resolution - the boundary is closed at entry.
    strategy = _RecordingStrategyStandIn()
    with pytest.raises(module.ExecutionError):
        _experiment(
            _InformationProviderStandIn(), _InformationSetProvider(), strategy
        ).run(None)
    assert strategy.contexts == []

    # A backend exception rejects as the runtime's own ExecutionError.
    strategy = _RecordingStrategyStandIn()
    with pytest.raises(module.ExecutionError):
        _experiment(
            _InformationProviderStandIn(), _RaisingOutcomeProvider(), strategy
        ).run(None)
    assert strategy.contexts == []

    # A1/Law 12b: every book's key set equals its date's declared
    # universe id set exactly - a missing or extra id rejects at entry.
    for label, book in (
        ("missing-id", {"A": 1.0}),
        ("extra-id", {"A": 1.0, "B": 1.0, "C": 1.0}),
    ):
        strategy = _RecordingStrategyStandIn()
        books = {date: dict(book) for date in _M44_DATES[:-1]}
        with pytest.raises(module.ExecutionError):
            _experiment(
                _InformationProviderStandIn(), _BookProvider(books), strategy
            ).run(None)
        assert strategy.contexts == [], label

    # A non-finite factor value rejects at entry.
    for label, value in (
        ("non-finite", float("inf")),
        ("not-a-number", float("nan")),
    ):
        strategy = _RecordingStrategyStandIn()
        books = {date: {"A": value, "B": 1.0} for date in _M44_DATES[:-1]}
        with pytest.raises(module.ExecutionError):
            _experiment(
                _InformationProviderStandIn(), _BookProvider(books), strategy
            ).run(None)
        assert strategy.contexts == [], label

    # B1: the snapshotted books are execution/accounting-private - no
    # factor value ever appears anywhere on a recorded decision
    # context (the books are consumed only after the relevant
    # decision/execution has passed).
    def _carries(
        value: Any,
        needle: float,
        depth: int = 0,
        seen: set[int] | None = None,
    ) -> bool:
        if seen is None:
            seen = set()
        if depth > 8 or id(value) in seen:
            return False
        seen.add(id(value))
        if isinstance(value, (bool, int, str, bytes, datetime)):
            return False
        if isinstance(value, float):
            return value == needle
        if isinstance(value, Mapping):
            return any(
                _carries(item, needle, depth + 1, seen) for item in value.values()
            )
        if isinstance(value, (list, tuple, set, frozenset)):
            return any(_carries(item, needle, depth + 1, seen) for item in value)
        found = False
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                found = found or _carries(
                    getattr(value, field.name), needle, depth + 1, seen
                )
            return found
        for slot in getattr(value, "__slots__", ()):
            found = found or _carries(
                getattr(value, slot, None), needle, depth + 1, seen
            )
        if hasattr(value, "__dict__"):
            for member in vars(value).values():
                found = found or _carries(member, needle, depth + 1, seen)
        return found

    distinct_books = {date: {"A": 4.0, "B": 2.5} for date in _M44_DATES[:-1]}
    strategy = _RecordingStrategyStandIn()
    results = list(
        _experiment(
            _InformationProviderStandIn(), _BookProvider(distinct_books), strategy
        ).run(None)
    )
    assert len(results) == 1
    assert len(strategy.contexts) == len(_M44_DATES)
    for context in strategy.contexts:
        for needle in (4.0, 2.5):
            assert not _carries(context, needle), needle

    # A frozen ledger-mandated rejection surfaces unchanged: the
    # all-zero book's denominator failure is never shadowed by a
    # generic ExecutionError.
    zero_books = {date: {"A": 0.0, "B": 0.0} for date in _M44_DATES[:-1]}
    strategy = _RecordingStrategyStandIn()
    with pytest.raises(ValueError) as caught:
        _experiment(
            _InformationProviderStandIn(), _BookProvider(zero_books), strategy
        ).run(None)
    assert not isinstance(caught.value, module.ExecutionError)

    # A negative factor is as invalid as a non-finite one: it rejects
    # at entry with the runtime's own ExecutionError - never as a raw
    # ledger ValueError escaping only after the decisions have run.
    strategy = _RecordingStrategyStandIn()
    negative_books = {date: {"A": 1.0, "B": -0.5} for date in _M44_DATES[:-1]}
    with pytest.raises(module.ExecutionError):
        _experiment(
            _InformationProviderStandIn(), _BookProvider(negative_books), strategy
        ).run(None)
    assert strategy.contexts == []


def test_opening_state_is_explicit_and_seeds_first_context_lawfully() -> None:
    """Floor F31 - the explicit-opening-state rule: the opening state is explicit -
    absent ``initial_weights``/``initial_wealth`` reject with nothing
    invented, the declared opening book seeds the first decision
    context PRE_TRADE with ``current_weights_as_of`` = the first
    decision date and is never drifted before that first decision,
    every later PRE_TRADE book is derived by drift and accounting, the
    first ``should_rebalance`` query carries the caller-declared
    ``initial_last_rebalance_time`` (advancing only on a True answer),
    and a False answer is the lawful hold - no execution, no timestamp
    advance, no cost."""

    module = _experiments()
    config = _config(
        strategy_ids=("alpha",),
        target="alpha",
        comparators=(),
        benchmarks=(),
    )
    declared_opening = {"A": 0.25, "B": 0.75}

    class _AnsweringPolicy:
        """A rebalance policy recording every (decision, last) query."""

        def __init__(self, answer: bool) -> None:
            self._answer = answer
            self.queries: list[tuple[Any, Any]] = []

        def should_rebalance(
            self, decision_time: Any, last_rebalance_time: Any
        ) -> bool:
            self.queries.append((decision_time, last_rebalance_time))
            return self._answer

    class _BookProvider:
        """A distinct outcome backend serving declared books."""

        def __init__(self, books: Mapping[Any, Mapping[str, float]]) -> None:
            self._books = books

        def __call__(self, instant: Any) -> Any:
            return dict(self._books[instant])

    def _conditions_fields(**opening_overrides: Any) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "information_provider": _InformationProviderStandIn(),
            "outcome_provider": _OutcomeProviderStandIn(_M44_UNIVERSES),
            "rebalance_policy": _AnsweringPolicy(True),
            "accounting_engine": ExactFillAccounting(),
            "cost_model": Proportional(rate=0.001, turnover=two_sided),
            "initial_weights": dict(declared_opening),
            "initial_wealth": 100.0,
            "initial_last_rebalance_time": _M44_OPENING_LAST_REBALANCE,
        }
        fields.update(opening_overrides)
        return fields

    def _experiment(strategy: Any, **opening_overrides: Any):
        fields = _conditions_fields(**opening_overrides)
        return module.Experiment(
            config,
            (_registration("alpha", strategy),),
            _m44_schedule(),
            module.ExperimentConditions(**fields),
        )

    # Absent opening fields reject at entry, naming the field - no
    # equal-weight, cash, zero, costless, or synthesized-wealth
    # opening is ever invented behind them.
    for absent_field in ("initial_weights", "initial_wealth"):
        strategy = _RecordingStrategyStandIn()
        fields = _conditions_fields()
        del fields[absent_field]
        with pytest.raises(module.ExecutionError, match=absent_field):
            module.Experiment(
                config,
                (_registration("alpha", strategy),),
                _m44_schedule(),
                module.ExperimentConditions(**fields),
            ).run(None)
        assert strategy.contexts == []

    # A declared-but-unlawful opening rejects at entry: universe
    # mismatch (A1/Law 12b), a non-unit budget, a non-positive or
    # non-finite wealth, a naive opening instant, and an opening
    # last-rebalance instant after the first decision date.
    for label, overrides in (
        ("missing-id", {"initial_weights": {"A": 1.0}}),
        ("extra-id", {"initial_weights": {"A": 0.4, "B": 0.4, "C": 0.2}}),
        ("non-unit-budget", {"initial_weights": {"A": 0.4, "B": 0.4}}),
        ("zero-wealth", {"initial_wealth": 0.0}),
        ("negative-wealth", {"initial_wealth": -100.0}),
        ("non-finite-wealth", {"initial_wealth": float("inf")}),
        (
            "naive-lr",
            {
                "initial_last_rebalance_time": _M44_OPENING_LAST_REBALANCE.replace(
                    tzinfo=None
                )
            },
        ),
        (
            "late-lr",
            {"initial_last_rebalance_time": datetime(2026, 4, 1, tzinfo=UTC)},
        ),
    ):
        strategy = _RecordingStrategyStandIn()
        with pytest.raises(module.ExecutionError):
            _experiment(strategy, **overrides).run(None)
        assert strategy.contexts == [], label

    # The declared opening seeds the first context lawfully: PRE_TRADE,
    # the declared book unchanged - never drifted before the first
    # decision, proven by segment factors that would move it - and
    # ``current_weights_as_of`` = the first decision date itself (the
    # frozen chronology law satisfied).
    drifting_books = {date: {"A": 4.0, "B": 2.5} for date in _M44_DATES[:-1]}
    strategy = _RecordingStrategyStandIn()
    policy = _AnsweringPolicy(True)
    results = list(
        _experiment(
            strategy,
            outcome_provider=_BookProvider(drifting_books),
            rebalance_policy=policy,
        ).run(None)
    )
    assert len(results) == 1
    first = strategy.contexts[0]
    assert first.current_weights.state is WeightState.PRE_TRADE
    assert dict(first.current_weights.weights) == declared_opening
    assert first.current_weights_as_of == _M44_DATES[0]
    assert first.current_weights_as_of <= first.decision_time

    # Every later PRE_TRADE book is derived by drift and accounting -
    # never the carried opening, never an equal-weight default - and
    # is stamped at its own decision date.
    for later in strategy.contexts[1:]:
        assert later.current_weights.state is WeightState.PRE_TRADE
        assert dict(later.current_weights.weights) != declared_opening
        assert dict(later.current_weights.weights) != {"A": 0.5, "B": 0.5}
        assert later.current_weights_as_of == later.decision_time

    # A2: the first query carries the caller-declared opening instant
    # - never the first decision date - and a True answer advances the
    # carried instant to that decision date.
    assert policy.queries[0] == (_M44_DATES[0], _M44_OPENING_LAST_REBALANCE)
    assert policy.queries[1] == (_M44_DATES[1], _M44_DATES[0])

    # The A2 boundary is ``<=``: an opening instant equal to the first
    # decision date is lawful and runs.
    boundary_strategy = _RecordingStrategyStandIn()
    boundary_policy = _AnsweringPolicy(True)
    boundary = list(
        _experiment(
            boundary_strategy,
            rebalance_policy=boundary_policy,
            initial_last_rebalance_time=_M44_DATES[0],
        ).run(None)
    )
    assert len(boundary) == 1
    assert boundary_policy.queries[0] == (_M44_DATES[0], _M44_DATES[0])

    # The opening state is common, never per-arm: two arms' first
    # contexts carry the identical declared opening book (A2 identical
    # across arms).
    two_arm_config = _config(
        strategy_ids=("alpha", "zeta"),
        target="alpha",
        comparators=("zeta",),
        benchmarks=(),
    )
    arm_alpha = _RecordingStrategyStandIn()
    arm_zeta = _RecordingStrategyStandIn()
    two_arm = module.Experiment(
        two_arm_config,
        (_registration("alpha", arm_alpha), _registration("zeta", arm_zeta)),
        _m44_schedule(),
        module.ExperimentConditions(**_conditions_fields()),
    )
    assert len(list(two_arm.run(None))) == 2
    for arm in (arm_alpha, arm_zeta):
        assert dict(arm.contexts[0].current_weights.weights) == declared_opening
        assert arm.contexts[0].current_weights_as_of == _M44_DATES[0]

    # The True route costs the first allocation from the declared
    # opening book through the frozen ledger: the targets are
    # equal-weight, so an invented equal-weight opening would be
    # costless - the strictly realized drag is the declared book's own
    # turnover (0.001 x two-sided 0.5 on 100.0).
    flat_books = {date: {"A": 1.0, "B": 1.0} for date in _M44_DATES[:-1]}
    executed = list(
        _experiment(
            _RecordingStrategyStandIn(),
            outcome_provider=_BookProvider(flat_books),
        ).run(None)
    )
    assert executed[0].paths["main"].final_wealth == pytest.approx(99.95, rel=1e-9)

    # The False route is the lawful hold (explicit-hold rule): no execution,
    # no rebalance-timestamp advance - every query still carries the
    # declared opening instant - and no cost is ever charged.
    hold_policy = _AnsweringPolicy(False)
    held = list(
        _experiment(
            _RecordingStrategyStandIn(),
            outcome_provider=_BookProvider(flat_books),
            rebalance_policy=hold_policy,
        ).run(None)
    )
    assert hold_policy.queries == [
        (date, _M44_OPENING_LAST_REBALANCE) for date in _M44_DATES
    ]
    assert held[0].paths["main"].final_wealth == 100.0


def test_cost_surface_is_the_shipped_proportional_model_only() -> None:
    """Floor F32 - B3/M18: the runtime consumes the transaction-cost
    model as the shipped ``Proportional`` object and passes it to the
    frozen ledger route - a behaviorally identical duck-typed model, a
    wrapper around the genuine one, and a surfaceless object all
    reject at entry with ``ExecutionError`` and are never consumed -
    and no runtime-side generic cost-model abstraction or
    protocol-typed cost-injection path beyond ``Proportional``
    exists."""

    module = _experiments()
    config = _config(
        strategy_ids=("alpha",),
        target="alpha",
        comparators=(),
        benchmarks=(),
    )

    class _DuckCostModel:
        """A non-shipped cost object exposing the full shipped
        behavior - rejection must be by identity, never by probing."""

        def __init__(self) -> None:
            self._inner = Proportional(rate=0.001, turnover=two_sided)
            self.calls = 0

        @property
        def convention(self) -> str:
            return self._inner.convention

        def q(self, trade: Any) -> float:
            self.calls += 1
            return self._inner.q(trade)

        def f_cost(self, trade: Any) -> float:
            self.calls += 1
            return self._inner.f_cost(trade)

        def estimate_trade_cost(self, pre: Any, target: Any) -> float:
            self.calls += 1
            return self._inner.estimate_trade_cost(pre, target)

    class _WrappingCostModel:
        """A wrapper holding the genuine shipped model itself - still
        not the shipped object."""

        def __init__(self) -> None:
            self._inner = Proportional(rate=0.001, turnover=two_sided)
            self.calls = 0

        @property
        def convention(self) -> str:
            return self._inner.convention

        def q(self, trade: Any) -> float:
            self.calls += 1
            return self._inner.q(trade)

        def f_cost(self, trade: Any) -> float:
            self.calls += 1
            return self._inner.f_cost(trade)

        def estimate_trade_cost(self, pre: Any, target: Any) -> float:
            self.calls += 1
            return self._inner.estimate_trade_cost(pre, target)

    def _experiment(cost_model: Any, strategy: Any):
        return module.Experiment(
            config,
            (_registration("alpha", strategy),),
            _m44_schedule(),
            module.ExperimentConditions(
                information_provider=_InformationProviderStandIn(),
                outcome_provider=_OutcomeProviderStandIn(_M44_UNIVERSES),
                rebalance_policy=_AlwaysRebalanceStandIn(),
                accounting_engine=ExactFillAccounting(),
                cost_model=cost_model,
                initial_weights={"A": 0.25, "B": 0.75},
                initial_wealth=100.0,
                initial_last_rebalance_time=_M44_OPENING_LAST_REBALANCE,
            ),
        )

    # A duck-typed model with the full shipped behavior rejects at
    # entry and is never consumed.
    duck = _DuckCostModel()
    strategy = _RecordingStrategyStandIn()
    with pytest.raises(module.ExecutionError):
        _experiment(duck, strategy).run(None)
    assert strategy.contexts == []
    assert duck.calls == 0

    # A wrapper around the genuine model rejects just the same.
    wrapper = _WrappingCostModel()
    strategy = _RecordingStrategyStandIn()
    with pytest.raises(module.ExecutionError):
        _experiment(wrapper, strategy).run(None)
    assert strategy.contexts == []
    assert wrapper.calls == 0

    # A surfaceless object rejects too - the frozen ledger route's
    # bound-convention law standing behind the runtime's own check.
    strategy = _RecordingStrategyStandIn()
    with pytest.raises(module.ExecutionError):
        _experiment(object(), strategy).run(None)
    assert strategy.contexts == []

    # No runtime-side generic cost abstraction exists: no exported
    # name mentions cost, no ``CostModel`` protocol is re-promised,
    # and the conditions' only cost-related field is ``cost_model``.
    for name in module.__all__:
        assert "cost" not in name.lower(), name
    assert not hasattr(module, "CostModel")
    condition_fields = set(_M44_CONDITIONS_FIELDS)
    assert [name for name in condition_fields if "cost" in name] == ["cost_model"]

    # The shipped model itself runs the frozen route end to end and
    # its rate actually shapes the path: a zero-rate model leaves the
    # opening wealth untouched where the priced one drags it.
    priced = list(
        _experiment(
            Proportional(rate=0.001, turnover=two_sided),
            _RecordingStrategyStandIn(),
        ).run(None)
    )
    free = list(
        _experiment(
            Proportional(rate=0.0, turnover=two_sided),
            _RecordingStrategyStandIn(),
        ).run(None)
    )
    assert len(priced) == 1 and len(free) == 1
    assert priced[0].paths["main"].final_wealth < free[0].paths["main"].final_wealth
    assert free[0].paths["main"].final_wealth == 100.0


# --------------------------------------------------------------------------- #
# Temporal-protocol W3 Law 13 (post-approval owner clarification, 2026-10-03, temporal-protocol §12) —   #
# the runtime-side opaque-consumption seam: the entry/resolved layer maps the     #
# validated linkage into the frozen ForecastLink channel (D5); runtime receives  #
# and preserves those values opaquely.  Four behavior-named nodes: verbatim   #
# identity preservation with exactly-one-recorded-entry-per-distinct-link;    #
# the future-information prevention applied to the provenance channel; the    #
# no-resolution/no-fitting-validation law (source census plus a behavioral    #
# tripwire over a real link-carrying run); and the frozen-surface census     #
# (four-argument Experiment, eight-field ExperimentConditions, run(selection),#
# existing result schema — no parallel provenance carrier of any kind).       #
#                                                                              #
# Scope fence (temporal-protocol §12): missing or incomplete required protocol provenance  #
# fails at the temporal-protocol resolution/entry boundary, BEFORE runtime entry.  That      #
# boundary is owned by the temporal-protocol entry layer, which is outside this W3 runtime   #
# seam and not implemented by these nodes; no coverage of it is claimed here. #
# --------------------------------------------------------------------------- #


def _m44b_distinct_links() -> tuple[Any, ...]:
    """Three distinct per-date ``ForecastLink`` values, each lifecycle-incoherent
    (fit window after its cutoff) — lawful to supply, and semantically NOT
    the runtime's to police (Law 13: opaque consumption only)."""
    results = _results()
    return tuple(
        results.ForecastLink(
            model=f"m44b-w3-arm-{index}",
            config_hash=f"m44b-w3-hash-{index}",
            cutoff=instant,
            seed=index,
            determinism=results.DeterminismClass.NONDETERMINISTIC,
            fit_window_start=instant + timedelta(days=1),
            refit_instants=(instant - timedelta(days=1),),
            forecast_target="expected_return",
            forecast_horizon=timedelta(days=31),
            information=(),
        )
        for index, instant in enumerate(_M44_DATES)
    )


class _PerDateLinkSupply:
    """The common information backend carrying one distinct provenance link
    per declared date — each available exactly at its decision instant."""

    def __init__(self, links: Any) -> None:
        self._by_instant = dict(zip(_M44_DATES, links, strict=True))

    def __call__(self, instant: Any) -> Any:
        from portlearn.interfaces import TimedObservation

        return InformationSet(
            items=(
                TimedObservation(
                    series_id="m44b-provenance",
                    observation_time=instant,
                    available_time=instant,
                    value=self._by_instant[instant],
                ),
            ),
            as_of=instant,
        )


class _LeakedLinkSupply:
    """The common information backend whose ONLY item is a provenance link
    that is not available until after the decision instant."""

    def __init__(self, link: Any) -> None:
        self._link = link

    def __call__(self, instant: Any) -> Any:
        from portlearn.interfaces import TimedObservation

        return InformationSet(
            items=(
                TimedObservation(
                    series_id="m44b-provenance",
                    observation_time=instant,
                    available_time=instant + timedelta(days=1),
                    value=self._link,
                ),
            ),
            as_of=instant,
        )


def _w3_link_experiment(provider: Any, strategy: Any):
    """A one-arm composition whose information backend carries provenance
    links — the minimal lawful Law 13 carrier through the frozen seam."""
    module = _experiments()
    config = _config(
        strategy_ids=("min-var",),
        target="min-var",
        comparators=(),
        benchmarks=(),
    )
    return module.Experiment(
        config,
        (_registration("min-var", strategy),),
        _m44_schedule(),
        _m44_conditions_with(information_provider=provider),
    )


def test_w3_law13_forecast_links_preserved_opaquely_verbatim() -> None:
    """Law 13 (W3) — the linkage the entry layer mapped into the frozen
    ``ForecastLink`` channel is preserved opaquely: a link repeated across
    decision dates is recorded once, as the very object that was supplied
    (identity, not a rebuild), and distinct per-date links are each
    recorded exactly once.  The supplied links are lifecycle-incoherent; the
    run still completes end to end, because interpreting or re-validating
    them inside the runtime has no lawful path."""
    # One object repeated across every date: exactly one recorded entry,
    # and it IS the supplied object — never dropped, never rebuilt.
    repeated = _m44b_link()
    spy = _ForecastSpyStrategy()
    record = next(
        iter(
            _w3_link_experiment(
                _ForecastSupplyStandIn(link=repeated), spy
            ).run(None)
        )
    )
    assert len(spy.contexts) == len(_M44_DATES)
    assert record.forecasts == (repeated,)
    assert record.forecasts[0] is repeated

    # Distinct links per date: every distinct value recorded exactly once
    # (the record's canonical order is its own law; content and identity
    # are what opacity preserves here).
    distinct = _m44b_distinct_links()
    spy = _ForecastSpyStrategy()
    record = next(
        iter(
            _w3_link_experiment(_PerDateLinkSupply(distinct), spy).run(None)
        )
    )
    assert len(spy.contexts) == len(_M44_DATES)
    assert len(record.forecasts) == len(distinct)
    assert {id(link) for link in record.forecasts} == {
        id(link) for link in distinct
    }
    assert {link.model for link in record.forecasts} == {
        link.model for link in distinct
    }


def test_w3_law13_unavailable_link_item_never_enters_a_decision() -> None:
    """Law 13 (W3) — unavailable future information never enters a
    ``DecisionContext``/``InformationSet`` through the provenance channel
    either: a link carried on an item that is not available until after
    the decision instant rejects fail-closed through the runtime, before
    any strategy decides.  Availability is the frozen admission law; the
    link's semantic content is never consulted (opacity preserved)."""
    module = _experiments()

    strategy = _RecordingStrategyStandIn()
    with pytest.raises(module.ExecutionError, match="information"):
        _w3_link_experiment(_LeakedLinkSupply(_m44b_link()), strategy).run(
            None
        )
    assert strategy.contexts == []


def test_w3_law13_runtime_resolves_no_protocols_and_runs_no_fitting_validation(
    monkeypatch: Any,
) -> None:
    """Law 13 (W3) — runtime itself never resolves temporal protocols, never
    interprets ``ProtocolProvenance``, and never invokes forecasting-lifecycle fitting
    validation: the runtime source references none of the protocol
    resolution or forecasting-lifecycle validator entry points, the module surface exposes
    no protocol/provenance name, and a full link-carrying run completes
    with every one of those entry points replaced by recording sentinels
    that never fire.  (The sentinels intercept module-attribute calls;
    the source census intercepts direct-import calls — together, every
    call form is covered.)"""
    import pathlib

    from portlearn import forecasting, protocols

    module = _experiments()
    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    for token in (
        "resolve_temporal_protocols",
        "ProtocolProvenance",
        "TemporalResearchProtocol",
        "portlearn.protocols",
        "portlearn.forecasting",
        "fitting_provenance",
        "require_fit_inputs_admitted",
        "require_cutoff_from_fit_set",
        "require_selection_labels_available",
        "require_reproducible",
    ):
        assert token not in source, (
            f"the experiment runtime references the forbidden entry point {token!r}: "
            "protocol resolution and forecasting-lifecycle fitting validation live outside runtime"
        )
    assert [
        name
        for name in dir(module)
        if "protocol" in name.lower() or "provenance" in name.lower()
    ] == []

    fired: list[Any] = []

    def tripwire(*args: Any, **kwargs: Any) -> None:
        fired.append((args, kwargs))

    for name in (
        "fitting_provenance",
        "require_fit_inputs_admitted",
        "require_cutoff_from_fit_set",
        "require_selection_labels_available",
        "require_reproducible",
    ):
        monkeypatch.setattr(forecasting, name, tripwire)
    monkeypatch.setattr(protocols, "resolve_temporal_protocols", tripwire)

    spy = _ForecastSpyStrategy()
    record = next(
        iter(
            _w3_link_experiment(
                _PerDateLinkSupply(_m44b_distinct_links()), spy
            ).run(None)
        )
    )
    assert len(spy.contexts) == len(_M44_DATES)
    assert len(record.forecasts) == 3
    assert fired == []


def test_w3_law13_frozen_api_admits_no_parallel_provenance_carrier() -> None:
    """Law 13 (W3, §12) — the frozen runtime API/schema is preserved exactly:
    the four-argument ``Experiment`` composition, the eight-field
    ``ExperimentConditions``, ``run(selection)``, the thirteen-name module
    surface, and the existing nine-field ``ResultRecord`` whose
    ``forecasts`` channel is the one and only link surface.  No parallel
    provenance object, constructor argument, conditions field, or
    result-schema field exists.  (Scope fence: rejection of missing
    required protocol provenance is the temporal-protocol entry layer's act, before
    runtime entry — outside this seam, not claimed here.)"""
    import inspect

    module = _experiments()
    assert list(
        inspect.signature(module.Experiment.__init__).parameters
    ) == ["self", "config", "registrations", "schedule", "conditions"]
    assert list(module.ExperimentConditions.__slots__) == list(
        _M44_CONDITIONS_FIELDS
    )
    assert list(inspect.signature(module.Experiment.run).parameters) == [
        "self",
        "selection",
    ]
    assert list(module.__all__) == [
        "ConfigurationError",
        "StrategyRegistration",
        "ExperimentConfiguration",
        "ExperimentSelection",
        "ExperimentResults",
        "select_all",
        "select_none",
        "select_single",
        "select_subset",
        "ExecutionError",
        "DecisionSchedule",
        "ExperimentConditions",
        "Experiment",
    ]
    record_fields = [
        field.name for field in dataclasses.fields(_results().ResultRecord)
    ]
    assert record_fields == [
        "schema_version",
        "strategy_id",
        "created_at",
        "run_id",
        "manifest_digest",
        "paths",
        "forecasts",
        "conformance",
        "parameter_relations",
    ]
    assert [
        name
        for name in record_fields
        if "protocol" in name.lower() or "provenance" in name.lower()
    ] == []
