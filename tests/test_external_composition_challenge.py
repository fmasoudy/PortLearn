"""Prospective behavior contracts for the external composition challenge.

The example under specification composes two genuinely external component
ecosystems with PortLearn through frozen public contracts only:

* ``scikit-learn`` owns statistical estimation: a researcher-side
  ``SklearnShrinkageCovarianceFactory`` satisfies the public
  ``portlearn.forecasting.ForecasterFactory`` seam, and its fitted
  model emits the estimated covariance through the public
  ``portlearn.interfaces.Forecast`` contract using deterministic
  canonical asset-pair keys.
* ``cvxpy`` owns portfolio optimization: a researcher-side
  ``CvxpyMinimumVarianceStrategy`` satisfies the public ``Strategy``
  seam, consumes a covariance ``Forecast`` from
  ``DecisionContext.forecast``, and solves the long-only fully-invested
  minimum-variance allocation with ``cvxpy`` alone.
* ``SklearnCvxpyMinimumVarianceStrategy`` is the explicit end-to-end
  composition: sklearn alone estimates, cvxpy alone allocates, and the
  researcher-side adapter only translates and orchestrates.
* An allocation-only control arm consumes a deterministic covariance
  ``Forecast`` supplied through the same public context channel.
* Both arms are registered as live ``StrategyRegistration`` components
  under one canonical declaration and executed through the frozen
  ``portlearn.research.run_declared_experiment`` public entry path.

These nodes pin that architecture observably: external distribution
provenance, public-import-only discipline, no built-in laundering, no
example-local estimator or optimizer arithmetic, causal mutation
evidence on both external seams, deterministic run declaration, and
composition-only claim fencing with no empirical-performance claim.

Prospective RED discipline: the authorized example artifact is absent
by design at authoring time, so the expected example surface is loaded
inside tests and fixtures only -- never at collection time -- and every
behavior node fails with an ``EXPECTED_RED`` causal marker naming the
absent artifact while remaining executable once the implementation
lands.  No implementation stub is created here, and no inherited
contract is weakened.

All instrumentation is test-local and non-public: call-through spies on
the frozen public runtime seams and on the example's own module-level
adapter classes, controlled monkeypatch mutations of the external
estimator and solver results, and static AST/source censuses with
seeded-oracle self-checks.  No production or example surface is
altered, no network path is opened, and no wall-clock or empirical
market data enters any node.
"""

from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import io
import re
import socket
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy
import pytest

from portlearn.interfaces import Forecast

# --------------------------------------------------------------------------- #
# Frozen locations, names, and repo-state literals
# --------------------------------------------------------------------------- #

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / "examples" / "external_composition_challenge.py"

# The exact public names the authorized example module must expose. The
# names are minimal and architecture-revealing; everything else about the
# example is pinned behaviorally, not nominally.
FROZEN_PUBLIC_NAMES = (
    "SklearnShrinkageCovarianceFactory",
    "CvxpyMinimumVarianceStrategy",
    "SklearnCvxpyMinimumVarianceStrategy",
    "COVARIANCE_TARGET",
    "SKLEARN_COVARIANCE_ESTIMATOR",
    "covariance_pair_key",
    "main",
)

# The frozen release-preparation ``src/portlearn`` tree object: byte-equal
# tree proof that no PortLearn source edit accompanies the challenge.
FROZEN_SRC_TREE = "ebc2904a12722ff6476e142d5c64057e45633d0d"

# The public PortLearn module allowlist for challenge production and
# tests: any other portlearn import, underscore module, underscore
# symbol, or source-relative import fails the census.
PUBLIC_PORTLEARN_MODULES = frozenset(
    (
        "portlearn",
        "portlearn.calendar",
        "portlearn.comparisons",
        "portlearn.costs",
        "portlearn.data",
        "portlearn.experiments",
        "portlearn.forecasting",
        "portlearn.interfaces",
        "portlearn.ledger",
        "portlearn.protocols",
        "portlearn.rebalance",
        "portlearn.research",
        "portlearn.results",
        "portlearn.strategies",
        "portlearn.timing",
        "portlearn.turnover",
        "portlearn.weights",
    )
)

# Built-in PortLearn strategies and local optimizer arithmetic that the
# challenge adapters may never reference: built-in substitution and any
# example-local estimator/solver arithmetic are non-evidence.
FORBIDDEN_BUILTIN_TOKENS = (
    "portlearn.strategies",
    "from portlearn import strategies",
    "from portlearn.strategies",
)
FORBIDDEN_LOCAL_SOLVER_TOKENS = (
    "scipy.optimize",
    "scipy.linalg",
    "numpy.linalg",
    "np.linalg",
)

# The scikit-learn public shrinkage-covariance estimators admitted by the
# frozen architecture; anything else is not a shrinkage estimator.
SHRINKAGE_ESTIMATORS = frozenset(("LedoitWolf", "OAS"))

# Claim-fence vocabulary: required composition-only boundary tokens and
# forbidden empirical-performance claims.
REQUIRED_FENCE_TOKENS = ("synthetic", "composition", "non-empirical")
FORBIDDEN_CLAIM_PATTERNS = (
    "winner",
    "superior",
    "outperform",
    "out-of-sample",
    "investable",
    "sharpe",
    "risk-adjusted",
    "beats",
)

# Weight-validity bounds for the declared long-only fully-invested
# allocation: finite, keyed exactly to the universe, non-negative within
# solver noise, and summing to one within a solver-honest tolerance.
WEIGHT_SUM_TOLERANCE = 1e-6
LONG_ONLY_TOLERANCE = 1e-12

# Runtime-generated identity patterns excluded from determinism
# comparison, mirroring the accepted result-version laws.
_TIMESTAMP_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:?\d{2}|Z)?"
)
_LONG_HEX_PATTERN = re.compile(r"\b[0-9a-f]{16,}\b")

COMPOSED_ARM = "SklearnCvxpyMinimumVarianceStrategy"
ALLOCATION_ONLY_ARM = "CvxpyMinimumVarianceStrategy"

# The canonical external-component declaration required on the existing
# ``ExperimentConfiguration.conditions`` channel (§5.3): a non-empty
# ``external_components`` mapping whose §5.8a-lawful string values name
# each external distribution together with its responsibility family
# and its composed role.  The component keys are frozen test-side; the
# six semantic values are the contract.
CANONICAL_EXTERNAL_COMPONENTS: dict[str, dict[str, str]] = {
    "sklearn_covariance": {
        "distribution": "scikit-learn",
        "family": "estimation",
        "role": "covariance",
    },
    "cvxpy_allocator": {
        "distribution": "cvxpy",
        "family": "optimization",
        "role": "allocation",
    },
}


# --------------------------------------------------------------------------- #
# Example loading (inside tests only -- never at collection time)
# --------------------------------------------------------------------------- #


def _load_example() -> Any:
    """Load the authorized example module, failing RED while absent."""
    if not EXAMPLE.is_file():
        pytest.fail(
            "EXPECTED_RED absent authorized artifact: "
            f"{EXAMPLE.relative_to(REPO)}",
            pytrace=False,
        )
    spec = importlib.util.spec_from_file_location(
        "external_composition_example_under_test", EXAMPLE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _example_source() -> str:
    """The example's source text, failing RED while absent."""
    if not EXAMPLE.is_file():
        pytest.fail(
            "EXPECTED_RED absent authorized artifact: "
            f"{EXAMPLE.relative_to(REPO)}",
            pytrace=False,
        )
    return EXAMPLE.read_text(encoding="utf-8")


def _external_deps_installed() -> bool:
    """Whether both external distributions are importable here."""
    return (
        importlib.util.find_spec("sklearn") is not None
        and importlib.util.find_spec("cvxpy") is not None
    )


def _load_mutated_example(source: str) -> Any:
    """Load a controlled source mutation of the example from disk."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(source)
        mutated_path = handle.name
    spec = importlib.util.spec_from_file_location(
        "external_composition_example_mutated", mutated_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        Path(mutated_path).unlink(missing_ok=True)
    return module


# --------------------------------------------------------------------------- #
# Static censuses (with seeded-oracle teeth)
# --------------------------------------------------------------------------- #


def _portlearn_import_violations(source: str) -> list[str]:
    """Every portlearn import outside the public allowlist in ``source``."""
    violations: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] != "portlearn":
                    continue
                if alias.name not in PUBLIC_PORTLEARN_MODULES:
                    violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                violations.append(f"relative import (level={node.level})")
                continue
            module = node.module or ""
            if module.split(".")[0] != "portlearn":
                continue
            if module not in PUBLIC_PORTLEARN_MODULES:
                violations.append(f"from {module} import ...")
            for alias in node.names:
                if alias.name.startswith("_"):
                    violations.append(f"underscore symbol {module}.{alias.name}")
    return violations


def _forbidden_tokens(source: str, tokens: tuple[str, ...]) -> list[str]:
    """Every forbidden literal token present in ``source``."""
    return [token for token in tokens if token in source]


def _tree_object(revision: str) -> str:
    """The git tree object for ``src/portlearn`` at ``revision``."""
    result = subprocess.run(
        ("git", "rev-parse", f"{revision}:src/portlearn"),
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


# --------------------------------------------------------------------------- #
# Behavior instrumentation (test-local, call-through, non-public)
# --------------------------------------------------------------------------- #


class _DecisionSink:
    """Records every decide call on the example's two strategy classes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, Any]] = []

    def attach(self, monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
        for name in (COMPOSED_ARM, ALLOCATION_ONLY_ARM):
            strategy_class = getattr(module, name)

            def _wrapped(
                instance: Any,
                context: Any,
                _original: Any = strategy_class.decide,
                _name: str = name,
                _sink: _DecisionSink = self,
            ) -> Any:
                result = _original(instance, context)
                _sink.calls.append((_name, context, result))
                return result

            monkeypatch.setattr(strategy_class, "decide", _wrapped)

    def arm_names(self) -> set[str]:
        return {name for name, _, _ in self.calls}


class _RunnerSink:
    """Records the public research entry call, call-through."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def attach_before_example_load(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Patch the public entry so a subsequently loaded example binds it."""
        import portlearn.research as research_module

        original = research_module.run_declared_experiment

        def _recorded(**kwargs: Any) -> Any:
            self.calls.append(
                {
                    "configuration": kwargs.get("configuration"),
                    "registrations": kwargs.get("registrations"),
                    "conditions": kwargs.get("conditions"),
                    "fit_provenances": kwargs.get(
                        "fit_provenances_by_strategy"
                    ),
                }
            )
            return original(**kwargs)

        monkeypatch.setattr(
            research_module, "run_declared_experiment", _recorded
        )


def _run_main(module: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    """Execute the example entry with stdout captured and sockets refused."""
    def _refused(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "the offline challenge opened a network socket"
        )

    monkeypatch.setattr(socket, "socket", _refused)
    monkeypatch.setattr(socket, "create_connection", _refused)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        module.main()
    return buffer.getvalue()


def _normalize_report(report: str) -> str:
    """Erase runtime-generated identities/timestamps from a report."""
    normalized = _TIMESTAMP_PATTERN.sub("<TIMESTAMP>", report)
    return _LONG_HEX_PATTERN.sub("<IDENTITY>", normalized)


def _expected_pair_keys(module: Any, universe: tuple[str, ...]) -> set[str]:
    """The canonical asset-pair key set for one universe."""
    keys = set()
    for i, left in enumerate(universe):
        for right in universe[i:]:
            keys.add(module.covariance_pair_key(left, right))
    return keys


def _require_dependency_skip() -> None:
    """Skip honestly when an external distribution is not installed."""
    if not _external_deps_installed():
        pytest.skip(
            "external distributions scikit-learn and cvxpy are not "
            "installed in this environment; the external-call-evidence "
            "nodes require the locked optional extra"
        )


def _mutate_sklearn_covariance(
    monkeypatch: pytest.MonkeyPatch, module: Any, scale: float
) -> str:
    """Wrap the selected public sklearn estimator with a non-uniform
    covariance mutation.

    The first asset's variance is scaled by ``scale`` through a diagonal
    congruence (``D Σ D``), which preserves semidefiniteness.  A uniform
    scaling is provably insufficient evidence here: minimum-variance
    weights are invariant to a positive scalar multiple of the
    covariance, so the controlled mutation must be non-uniform to change
    the composed strategy's downstream weights.
    """
    import sklearn
    import sklearn.covariance

    estimator_name = module.SKLEARN_COVARIANCE_ESTIMATOR
    estimator = getattr(sklearn.covariance, estimator_name, None)
    assert estimator is not None, (
        f"SKLEARN_COVARIANCE_ESTIMATOR={estimator_name!r} does not name a "
        "public sklearn.covariance estimator"
    )
    original_fit = estimator.fit

    def _congruence_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_fit(self, *args, **kwargs)
        covariance = numpy.asarray(self.covariance_, dtype=float).copy()
        covariance[0, :] *= scale
        covariance[:, 0] *= scale
        self.covariance_ = covariance
        return result

    monkeypatch.setattr(estimator, "fit", _congruence_fit)
    return estimator_name


def _mutate_cvxpy_allocation(
    monkeypatch: pytest.MonkeyPatch, delta: float
) -> list[str]:
    """Wrap cvxpy's solve, perturbing solved variable values in place.

    Returns a recording list that receives one entry per invocation of
    the perturbed solve, so the calling node can fail on a zero-call /
    inert mutation and require the hook to reach every solve path.
    """
    import cvxpy

    original_solve = cvxpy.Problem.solve
    invocations: list[str] = []

    def _perturbed_solve(self: Any, *args: Any, **kwargs: Any) -> Any:
        status = original_solve(self, *args, **kwargs)
        invocations.append(getattr(self, "status", "solved"))
        for variable in self.variables():
            value = variable.value
            if value is None or getattr(value, "size", 0) < 2:
                continue
            perturbed = numpy.array(value, dtype=float)
            perturbed.flat[0] += delta
            perturbed.flat[1] -= delta
            variable.value = perturbed
        return status

    monkeypatch.setattr(cvxpy.Problem, "solve", _perturbed_solve)
    return invocations


def _weights_by_arm(
    sink: _DecisionSink,
) -> dict[str, list[dict[str, float]]]:
    """Snapshot recorded target weights by exact arm name."""
    snapshot: dict[str, list[dict[str, float]]] = {}
    for name, _, result in sink.calls:
        snapshot.setdefault(name, []).append(
            dict(result.decision.target_weights)
        )
    return snapshot


# --------------------------------------------------------------------------- #
# The frozen public surface
# --------------------------------------------------------------------------- #


class TestExampleSurfaceAndImportDiscipline:
    def test_authorized_example_artifact_exists_and_loads_offline(self) -> None:
        module = _load_example()
        assert module.__name__ == "external_composition_example_under_test"

    def test_example_defines_exactly_the_frozen_public_composition_names(
        self,
    ) -> None:
        module = _load_example()
        missing = [
            name for name in FROZEN_PUBLIC_NAMES if not hasattr(module, name)
        ]
        assert not missing, f"missing frozen public names: {missing}"
        assert callable(module.main)
        assert callable(module.covariance_pair_key)
        assert isinstance(module.COVARIANCE_TARGET, str)
        assert module.COVARIANCE_TARGET.strip() != ""
        assert isinstance(module.SKLEARN_COVARIANCE_ESTIMATOR, str)
        assert module.SKLEARN_COVARIANCE_ESTIMATOR in SHRINKAGE_ESTIMATORS

    def test_example_imports_portlearn_only_through_the_public_allowlist(
        self,
    ) -> None:
        violations = _portlearn_import_violations(_example_source())
        assert not violations, (
            f"non-allowlist portlearn imports in the example: {violations}"
        )

    def test_challenge_import_census_also_governs_this_test_module(
        self,
    ) -> None:
        own = _portlearn_import_violations(
            Path(__file__).read_text(encoding="utf-8")
        )
        assert not own, f"non-allowlist portlearn imports here: {own}"

    def test_import_census_detects_a_seeded_private_portlearn_import(
        self,
    ) -> None:
        seeded = (
            "from portlearn.interfaces import Forecast\n"
            "from portlearn._optimizer import solve\n"
            "import portlearn.forecasting\n"
            "from portlearn.forecasting import _fit_set_as_of\n"
            "from . import sibling\n"
        )
        violations = _portlearn_import_violations(seeded)
        assert "from portlearn._optimizer import ..." in violations
        assert (
            "underscore symbol portlearn.forecasting._fit_set_as_of"
            in violations
        )
        assert "relative import (level=1)" in violations
        assert "import portlearn.forecasting" not in violations

    def test_example_never_references_builtin_strategies_or_local_solvers(
        self,
    ) -> None:
        source = _example_source()
        builtins = _forbidden_tokens(source, FORBIDDEN_BUILTIN_TOKENS)
        assert not builtins, (
            f"built-in strategy laundering tokens present: {builtins}"
        )
        solvers = _forbidden_tokens(source, FORBIDDEN_LOCAL_SOLVER_TOKENS)
        assert not solvers, (
            f"example-local optimizer arithmetic tokens present: {solvers}"
        )

    def test_builtin_ban_detects_a_seeded_builtin_reference(self) -> None:
        seeded = (
            "import portlearn.strategies\n"
            "w = numpy.linalg.solve(a, b)\n"
        )
        assert _forbidden_tokens(seeded, FORBIDDEN_BUILTIN_TOKENS) == [
            "portlearn.strategies"
        ]
        assert _forbidden_tokens(
            seeded, FORBIDDEN_LOCAL_SOLVER_TOKENS
        ) == ["numpy.linalg"]

    def test_sklearn_factory_satisfies_the_public_forecaster_seam(
        self,
    ) -> None:
        module = _load_example()
        factory_class = module.SklearnShrinkageCovarianceFactory
        assert callable(getattr(factory_class, "fit", None)), (
            "the sklearn adapter must expose the ForecasterFactory fit seam"
        )

    def test_cvxpy_strategies_satisfy_the_public_strategy_seam(self) -> None:
        module = _load_example()
        for name in (COMPOSED_ARM, ALLOCATION_ONLY_ARM):
            strategy_class = getattr(module, name)
            assert callable(
                getattr(strategy_class, "decide", None)
            ), f"{name} must expose the Strategy decide seam"


# --------------------------------------------------------------------------- #
# The combined public-path run
# --------------------------------------------------------------------------- #


class TestCombinedPublicPathRun:
    def test_main_registers_both_arms_and_executes_the_public_runner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _require_dependency_skip()
        runner = _RunnerSink()
        runner.attach_before_example_load(monkeypatch)
        decisions = _DecisionSink()
        module = _load_example()
        decisions.attach(monkeypatch, module)
        report = _run_main(module, monkeypatch)
        assert len(runner.calls) == 1, (
            "the combined run must execute exactly one declared experiment"
        )
        registration_ids = [
            registration.strategy_id
            for registration in runner.calls[0]["registrations"]
        ]
        assert len(registration_ids) == 2, (
            "exactly two live strategies must be registered"
        )
        assert len(set(registration_ids)) == 2, (
            "the two registered arms must carry distinct strategy ids"
        )
        assert decisions.arm_names() == {COMPOSED_ARM, ALLOCATION_ONLY_ARM}, (
            "both the composed and allocation-only arms must decide"
        )
        for arm in registration_ids:
            assert arm in report, (
                f"registered arm {arm!r} must be visible in the run report"
            )
        conditions = runner.calls[0]["configuration"].conditions
        assert "external_components" in conditions, (
            "the declared configuration must carry the canonical "
            "'external_components' mapping on the existing conditions "
            "channel"
        )
        external = conditions["external_components"]
        assert isinstance(external, Mapping) and external, (
            "conditions['external_components'] must be a non-empty mapping"
        )
        for component_key, expected in CANONICAL_EXTERNAL_COMPONENTS.items():
            component = external.get(component_key)
            assert isinstance(component, Mapping), (
                f"external component {component_key!r} must be a mapping "
                "of scalar/string values"
            )
            for value_key, expected_value in expected.items():
                assert component.get(value_key) == expected_value, (
                    f"external component {component_key!r} must declare "
                    f"{value_key}={expected_value!r}; got "
                    f"{component.get(value_key)!r}"
                )

    def test_public_runner_patch_makes_the_combined_run_fail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _require_dependency_skip()
        import portlearn.research as research_module

        def _refused(**kwargs: Any) -> Any:
            raise AssertionError("public research entry executed")

        monkeypatch.setattr(
            research_module, "run_declared_experiment", _refused
        )
        module = _load_example()
        with pytest.raises(AssertionError, match="public research entry"):
            _run_main(module, monkeypatch)

    def test_both_arms_share_one_injected_common_conditions_container(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _require_dependency_skip()
        import portlearn.experiments as experiments_module

        injected: list[Any] = []
        original_init = experiments_module.Experiment.__init__

        def _recording_init(self: Any, *args: Any, **kwargs: Any) -> None:
            injected.append(
                kwargs.get(
                    "conditions",
                    args[3] if len(args) > 3 else None,
                )
            )
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(
            experiments_module.Experiment, "__init__", _recording_init
        )
        runner = _RunnerSink()
        runner.attach_before_example_load(monkeypatch)
        decisions = _DecisionSink()
        module = _load_example()
        decisions.attach(monkeypatch, module)
        _run_main(module, monkeypatch)
        assert len(injected) == 1, (
            "both arms must consume exactly one shared experiment "
            "construction with one injected conditions container"
        )
        assert injected[0] is runner.calls[0]["conditions"], (
            "the declared conditions and the injected conditions must be "
            "the same shared container"
        )
        assert decisions.arm_names() == {COMPOSED_ARM, ALLOCATION_ONLY_ARM}

    def test_every_decision_consumes_a_canonical_covariance_forecast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _require_dependency_skip()
        decisions = _DecisionSink()
        module = _load_example()
        decisions.attach(monkeypatch, module)
        _run_main(module, monkeypatch)
        assert decisions.calls, "no decisions were recorded"
        for name, context, _ in decisions.calls:
            forecast = context.forecast
            assert isinstance(forecast, Forecast), (
                f"{name} decided without a Forecast on the context channel"
            )
            assert forecast.target == module.COVARIANCE_TARGET, (
                f"{name} consumed a forecast with target {forecast.target!r}"
            )
            expected = _expected_pair_keys(module, tuple(context.universe))
            assert set(forecast.values) == expected, (
                f"{name} forecast keys are not the canonical asset pairs "
                f"of the decision universe {tuple(context.universe)}"
            )
            for value in forecast.values.values():
                assert numpy.isfinite(value)

    def test_decisions_use_only_information_available_at_decision_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _require_dependency_skip()
        decisions = _DecisionSink()
        module = _load_example()
        decisions.attach(monkeypatch, module)
        _run_main(module, monkeypatch)
        for name, context, _ in decisions.calls:
            for record in context.information:
                assert record.available_time <= context.decision_time, (
                    f"{name} consumed a record available after the decision "
                    "instant; point-in-time isolation is broken"
                )

    def test_allocations_are_finite_long_only_and_fully_invested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _require_dependency_skip()
        decisions = _DecisionSink()
        module = _load_example()
        decisions.attach(monkeypatch, module)
        _run_main(module, monkeypatch)
        for name, context, result in decisions.calls:
            weights = result.decision.target_weights
            assert set(weights) == set(context.universe), (
                f"{name} weights are not keyed exactly to the universe"
            )
            gross = 0.0
            for weight in weights.values():
                assert numpy.isfinite(weight), (
                    f"{name} produced a non-finite weight"
                )
                assert weight >= -LONG_ONLY_TOLERANCE, (
                    f"{name} produced a short weight {weight!r}"
                )
                gross += weight
            assert abs(gross - 1.0) <= WEIGHT_SUM_TOLERANCE, (
                f"{name} weights sum to {gross!r}, not one"
            )

    def test_allocation_only_arm_receives_a_deterministic_covariance_forecast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _require_dependency_skip()
        forecasts: list[dict[str, float]] = []
        decisions = _DecisionSink()
        module = _load_example()
        strategy_class = getattr(module, ALLOCATION_ONLY_ARM)
        original_decide = strategy_class.decide

        def _deciding(self: Any, context: Any) -> Any:
            forecast = context.forecast
            assert forecast is not None
            forecasts.append(dict(forecast.values))
            return original_decide(self, context)

        decisions.attach(monkeypatch, module)
        monkeypatch.setattr(strategy_class, "decide", _deciding)
        _run_main(module, monkeypatch)
        first_run = [dict(values) for values in forecasts]
        forecasts.clear()
        _run_main(module, monkeypatch)
        assert first_run, "allocation-only arm recorded no forecasts"
        assert first_run == [dict(values) for values in forecasts], (
            "the deterministic covariance producer is not deterministic "
            "across runs"
        )

    def test_report_declares_external_component_metadata_and_roles(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _require_dependency_skip()
        module = _load_example()
        report = _run_main(module, monkeypatch)
        lowered = report.lower()
        for distribution in ("scikit-learn", "cvxpy"):
            assert distribution in lowered, (
                f"the report must name the external distribution "
                f"{distribution!r}"
            )
        assert "estimation" in lowered, (
            "the report must declare the estimation component role"
        )
        assert "allocation" in lowered, (
            "the report must declare the allocation component role"
        )
        assert re.search(r"licen[cs]e", lowered), (
            "the report must record licence metadata for both components"
        )
        assert re.search(r"\d+\.\d+", report), (
            "the report must record resolved distribution versions"
        )

    def test_external_distributions_are_distinct_from_portlearn_and_each_other(
        self,
    ) -> None:
        _load_example()  # RED marker while the example is absent
        if not _external_deps_installed():
            pytest.skip(
                "distribution-distinctness requires both external "
                "distributions installed"
            )
        names = {
            dist.metadata["Name"]
            for dist in importlib.metadata.distributions()
            if dist.metadata["Name"] is not None
        }
        for distribution in ("scikit-learn", "cvxpy", "portlearn"):
            assert distribution in names, (
                f"distribution {distribution!r} is not installed"
            )
        assert len({"scikit-learn", "cvxpy", "portlearn"}) == 3

    def test_fit_provenance_carries_external_estimator_identity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _require_dependency_skip()
        runner = _RunnerSink()
        runner.attach_before_example_load(monkeypatch)
        module = _load_example()
        _run_main(module, monkeypatch)
        provenances = runner.calls[0]["fit_provenances"]
        assert provenances, "no fit provenance reached the public runner"
        wrapped = [
            getattr(image, "wrapped_identity", None)
            for sequence in provenances.values()
            for image in sequence
        ]
        assert any(
            identity is not None and "scikit" in identity.lower()
            for identity in wrapped
        ), (
            "no fit provenance image carries the sklearn estimator "
            "identity through wrapped_identity"
        )

    def test_run_output_respects_the_composition_claim_fence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _require_dependency_skip()
        module = _load_example()
        report = _run_main(module, monkeypatch).lower()
        for token in REQUIRED_FENCE_TOKENS:
            assert token in report, (
                f"the report must state the {token!r} boundary"
            )
        for pattern in FORBIDDEN_CLAIM_PATTERNS:
            assert pattern not in report, (
                f"the report makes the forbidden claim {pattern!r}"
            )

    def test_report_carries_a_deterministic_per_arm_weight_digest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _require_dependency_skip()
        runner = _RunnerSink()
        runner.attach_before_example_load(monkeypatch)
        module = _load_example()
        report = _run_main(module, monkeypatch)
        assert "digest" in report.lower(), (
            "the report must carry deterministic per-arm weight digests"
        )
        for arm in runner.calls[0]["configuration"].strategy_ids:
            assert arm in report

    def test_repeated_runs_produce_identical_normalized_reports(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _require_dependency_skip()
        module = _load_example()
        first = _normalize_report(_run_main(module, monkeypatch))
        second = _normalize_report(_run_main(module, monkeypatch))
        assert first == second, (
            "two runs of the deterministic fixture produced different "
            "normalized reports"
        )

    def test_main_entry_executes_under_the_documented_command_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _require_dependency_skip()
        _example_source()  # EXPECTED_RED marker while the artifact is absent
        import runpy

        def _refused(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("the offline challenge opened a socket")

        monkeypatch.setattr(socket, "socket", _refused)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            runpy.run_path(str(EXAMPLE), run_name="__main__")
        assert "synthetic" in buffer.getvalue().lower()

    def test_importing_the_example_module_prints_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _example_source()  # EXPECTED_RED marker while the artifact is absent
        import runpy

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            runpy.run_path(
                str(EXAMPLE), run_name="external_composition_module"
            )
        assert buffer.getvalue() == "", (
            "importing the example printed to stdout"
        )


# --------------------------------------------------------------------------- #
# Fail-closed dependency boundary
# --------------------------------------------------------------------------- #


class TestDependencyBoundaryFailsClosed:
    def _blocked_example_or_none(
        self, monkeypatch: pytest.MonkeyPatch, blocked: str
    ) -> Any:
        monkeypatch.setitem(sys.modules, blocked, None)
        try:
            return _load_example()
        except ImportError as error:
            message = str(error)
            assert "portlearn[external-composition]" in message, (
                f"blocking {blocked!r} must fail closed naming "
                f"portlearn[external-composition]; got: {message!r}"
            )
            assert blocked in message or blocked.replace("_", "-") in (
                message.lower()
            ), (
                f"the diagnostic must name the blocked distribution "
                f"{blocked!r}; got: {message!r}"
            )
            return None

    def test_blocked_sklearn_fails_closed_naming_the_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = self._blocked_example_or_none(monkeypatch, "sklearn")
        if module is not None:
            with pytest.raises(ImportError) as excinfo:
                module.main()
            assert "portlearn[external-composition]" in str(excinfo.value)

    def test_blocked_cvxpy_fails_closed_naming_the_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = self._blocked_example_or_none(monkeypatch, "cvxpy")
        if module is not None:
            with pytest.raises(ImportError) as excinfo:
                module.main()
            assert "portlearn[external-composition]" in str(excinfo.value)

    def test_blocked_cvxpy_produces_no_decision_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = self._blocked_example_or_none(monkeypatch, "cvxpy")
        if module is None:
            return
        decisions = _DecisionSink()
        decisions.attach(monkeypatch, module)
        with pytest.raises(ImportError):
            _run_main(module, monkeypatch)
        assert not decisions.calls, (
            "a decision executed although cvxpy was blocked; a built-in "
            "or local fallback allocated"
        )


# --------------------------------------------------------------------------- #
# Causal mutation floors
# --------------------------------------------------------------------------- #


class TestCausalMutationFloors:
    def test_sklearn_covariance_mutation_changes_forecast_and_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _require_dependency_skip()
        module = _load_example()

        baseline_decisions = _DecisionSink()
        baseline_decisions.attach(monkeypatch, module)
        baseline_report = _normalize_report(_run_main(module, monkeypatch))
        baseline_forecasts = [
            dict(context.forecast.values)
            for name, context, _ in baseline_decisions.calls
            if name == COMPOSED_ARM
        ]
        baseline_weights = [
            dict(result.decision.target_weights)
            for name, _, result in baseline_decisions.calls
            if name == COMPOSED_ARM
        ]
        monkeypatch.undo()
        assert baseline_forecasts, "baseline run recorded no forecasts"
        assert baseline_weights, "baseline run recorded no weights"

        decisions = _DecisionSink()
        decisions.attach(monkeypatch, module)
        _mutate_sklearn_covariance(monkeypatch, module, 0.5)
        mutated_report = _normalize_report(_run_main(module, monkeypatch))
        mutated_forecasts = [
            dict(context.forecast.values)
            for name, context, _ in decisions.calls
            if name == COMPOSED_ARM
        ]
        mutated_weights = [
            dict(result.decision.target_weights)
            for name, _, result in decisions.calls
            if name == COMPOSED_ARM
        ]
        assert mutated_forecasts != baseline_forecasts, (
            "a controlled non-uniform sklearn covariance mutation did not "
            "change the covariance delivered through the public Forecast "
            "seam"
        )
        assert mutated_weights != baseline_weights, (
            "a controlled non-uniform sklearn covariance mutation did not "
            "change the composed strategy's downstream weights: the "
            "composed arm does not consume the sklearn-estimated covariance"
        )
        assert mutated_report != baseline_report, (
            "a controlled sklearn covariance mutation did not change the "
            "reported run"
        )

    def test_cvxpy_allocation_mutation_changes_final_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _require_dependency_skip()
        module = _load_example()

        baseline_decisions = _DecisionSink()
        baseline_decisions.attach(monkeypatch, module)
        baseline_report = _normalize_report(_run_main(module, monkeypatch))
        baseline_weights = _weights_by_arm(baseline_decisions)
        monkeypatch.undo()
        assert baseline_weights, "baseline run recorded no weights"

        decisions = _DecisionSink()
        decisions.attach(monkeypatch, module)
        invocations = _mutate_cvxpy_allocation(monkeypatch, 0.05)
        mutated_report = _normalize_report(_run_main(module, monkeypatch))
        mutated_weights = _weights_by_arm(decisions)
        assert invocations, (
            "the cvxpy solve mutation hook was never invoked: the "
            "intended public cvxpy solve path did not execute"
        )
        for arm in (COMPOSED_ARM, ALLOCATION_ONLY_ARM):
            assert arm in baseline_weights, (
                f"the baseline run recorded no {arm} decisions to compare"
            )
            assert arm in mutated_weights, (
                f"the mutated run recorded no {arm} decisions to compare"
            )
            assert mutated_weights[arm] != baseline_weights[arm], (
                f"a controlled cvxpy solved-allocation mutation did not "
                f"change {arm}'s final target weights: that arm does not "
                "consume the cvxpy-solved allocation"
            )
        assert mutated_report != baseline_report, (
            "a controlled cvxpy solved-allocation mutation did not change "
            "the reported run"
        )

    def test_declared_parameter_mutation_changes_the_declared_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = _example_source()
        mutated_source, count = re.subn(
            r'COVARIANCE_TARGET\s*=\s*["\'][^"\']+["\']',
            'COVARIANCE_TARGET = "mutated.covariance.target"',
            source,
            count=1,
        )
        assert count == 1, (
            "COVARIANCE_TARGET must be declared as a module-level string "
            "constant in the example"
        )
        assert mutated_source != source
        if not _external_deps_installed():
            pytest.skip(
                "the declared-run report leg requires the external "
                "distributions installed"
            )
        module = _load_mutated_example(mutated_source)
        assert module.COVARIANCE_TARGET == "mutated.covariance.target"
        report = _run_main(module, monkeypatch)
        assert "mutated.covariance.target" in report, (
            "the declared forecast target is not wired into the declared run"
        )


# --------------------------------------------------------------------------- #
# Frozen source tree (byte-equality proof)
# --------------------------------------------------------------------------- #


class TestFrozenSourceTree:
    def test_src_portlearn_tree_object_matches_the_frozen_literal(
        self,
    ) -> None:
        assert _tree_object("HEAD") == FROZEN_SRC_TREE, (
            "the live src/portlearn tree object differs from the frozen "
            "release-preparation tree; PortLearn source was edited"
        )

    def test_tree_comparison_detects_a_controlled_drift(self) -> None:
        drifted = "0" * 40
        assert _tree_object("HEAD") != drifted, (
            "the tree-object comparison cannot detect drift"
        )


# --------------------------------------------------------------------------- #
# Adapter private-access contract (public-seam composition discipline)
# --------------------------------------------------------------------------- #

_ADAPTER_CLASS_NAME = "SklearnShrinkageCovarianceFactory"
_ADAPTER_METHOD_NAME = "fit"


def _adapter_method_private_access(source: str) -> list[str]:
    """Every private underscore attribute access in the adapter ``fit``.

    The census is deliberately narrow and locally scoped: it inspects
    only the body of ``SklearnShrinkageCovarianceFactory.fit`` in
    ``source`` and reports each attribute access whose name begins with
    an underscore.  The local scope is what lets this helper attribute a
    violation to the adapter method precisely; it is not permission: a
    private underscore read anywhere else in the example -- including a
    harness-side ``fit_set._records`` window computation -- is equally a
    defect, and it is rejected by the companion whole-example census in
    ``_whole_example_private_access``, which governs the complete
    shipped artifact.
    """
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.ClassDef) and node.name == _ADAPTER_CLASS_NAME):
            continue
        for child in node.body:
            if not (
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == _ADAPTER_METHOD_NAME
            ):
                continue
            return [
                f"{ast.unparse(access.value)}.{access.attr}"
                for access in ast.walk(child)
                if isinstance(access, ast.Attribute)
                and access.attr.startswith("_")
            ]
    return [f"<absent {_ADAPTER_CLASS_NAME}.{_ADAPTER_METHOD_NAME}>"]


class TestAdapterPublicSurfaceOnly:
    def test_private_access_census_rejects_a_seeded_private_read_and_permits_public_iteration(
        self,
    ) -> None:
        seeded_private = (
            f"class {_ADAPTER_CLASS_NAME}:\n"
            f"    def {_ADAPTER_METHOD_NAME}(self, plan, fit_set):\n"
            "        records = tuple(fit_set._records)\n"
            "        return records\n"
        )
        assert _adapter_method_private_access(seeded_private) == [
            "fit_set._records"
        ], "the census must reject a seeded fit_set._records read inside the adapter method"

        seeded_public = (
            f"class {_ADAPTER_CLASS_NAME}:\n"
            f"    def {_ADAPTER_METHOD_NAME}(self, plan, fit_set):\n"
            "        records = tuple(fit_set)\n"
            "        return records\n"
        )
        assert _adapter_method_private_access(seeded_public) == [], (
            "the census must permit the public tuple(fit_set) iteration form"
        )

        seeded_outside = (
            f"class {_ADAPTER_CLASS_NAME}:\n"
            f"    def {_ADAPTER_METHOD_NAME}(self, plan, fit_set):\n"
            "        records = tuple(fit_set)\n"
            "        return records\n"
            "\n"
            "\n"
            "def window_start(fit_set):\n"
            "    return min(r.observation_time for r in fit_set._records)\n"
        )
        assert _adapter_method_private_access(seeded_outside) == [], (
            "the adapter census must stay locally scoped to the adapter "
            "method: a fit_set._records read outside it is not this "
            "contract -- it is rejected by the companion whole-example "
            "census, which governs the complete shipped artifact"
        )

    def test_adapter_fit_reads_the_fit_window_through_public_surface_only(self) -> None:
        source = _example_source()
        violations = _adapter_method_private_access(source)
        assert not violations, (
            f"{_ADAPTER_CLASS_NAME}.{_ADAPTER_METHOD_NAME} must consume the "
            "fit window through the public InformationSet surface only: "
            f"private attribute access {violations} -- fit_set._records is "
            "private; iterate the fit window publicly instead"
        )


# --------------------------------------------------------------------------- #
# Whole-example private-access contract (complete-artifact discipline)
# --------------------------------------------------------------------------- #

#: The exact private attribute reads the complete example needs for
#: external-package report metadata and nothing else: reading the two
#: external distributions' versions from their module aliases.  Every
#: other underscore-prefixed attribute consumption anywhere in the
#: example -- module, class, or function body alike -- is a defect.
_WHOLE_EXAMPLE_ALLOWED_PRIVATE_READS = frozenset(
    (
        "_sklearn_module.__version__",
        "_cvxpy_module.__version__",
    )
)


def _whole_example_private_access(source: str) -> list[str]:
    """Every private underscore attribute read anywhere in ``source``.

    Unlike the locally scoped adapter census above, this helper parses
    the complete example artifact -- every statement in every module-,
    class-, and function-level body -- and reports each attribute access
    whose name begins with an underscore as ``line <n>: <expression>``,
    except the frozen external-package metadata reads in
    ``_WHOLE_EXAMPLE_ALLOWED_PRIVATE_READS``.  The census is AST-based:
    an underscore name appearing in a comment or string literal is not
    attribute consumption and is never reported.
    """
    violations: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Attribute) and node.attr.startswith("_")):
            continue
        expression = ast.unparse(node)
        if expression in _WHOLE_EXAMPLE_ALLOWED_PRIVATE_READS:
            continue
        violations.append(f"line {node.lineno}: {expression}")
    return violations


class TestWholeExamplePublicSurfaceOnly:
    def test_census_rejects_a_harness_private_read_and_permits_public_iteration(
        self,
    ) -> None:
        seeded_harness_private = (
            "def run_challenge(fit_set):\n"
            "    window_start = min("
            "r.observation_time for r in fit_set._records)\n"
            "    return window_start\n"
        )
        violations = _whole_example_private_access(seeded_harness_private)
        assert violations == ["line 2: fit_set._records"], (
            "the whole-example census must reject a fit_set._records read "
            "in a harness/run function outside the sklearn adapter, with "
            f"its source expression and location; got {violations!r}"
        )

        seeded_public = (
            "def run_challenge(fit_set):\n"
            "    window_start = min(r.observation_time for r in fit_set)\n"
            "    return window_start\n"
        )
        assert _whole_example_private_access(seeded_public) == [], (
            "the whole-example census must permit public iteration over "
            "the fit window: iterating fit_set itself is the public "
            "InformationSet surface"
        )

    def test_census_ignores_underscore_names_that_are_not_attribute_reads(
        self,
    ) -> None:
        seeded_non_code = (
            "# a comment naming fit_set._records is not attribute access\n"
            "def run_challenge(fit_set):\n"
            "    note = 'the private fit_set._records spelling in a literal'\n"
            "    return min(r.observation_time for r in fit_set), note\n"
        )
        assert _whole_example_private_access(seeded_non_code) == [], (
            "the census must parse attribute consumption with the AST, not "
            "match raw substrings: underscore names in comments and string "
            "literals are not reads"
        )

    def test_census_admits_exactly_the_external_metadata_reads(self) -> None:
        seeded_metadata = (
            "import sklearn as _sklearn_module\n"
            "import cvxpy as _cvxpy_module\n"
            "def report():\n"
            "    return (\n"
            "        _sklearn_module.__version__,\n"
            "        _cvxpy_module.__version__,\n"
            "    )\n"
        )
        assert _whole_example_private_access(seeded_metadata) == [], (
            "the census must admit exactly the two external-package "
            "metadata reads the deterministic report needs"
        )
        seeded_other_metadata = (
            "def report(fit_set):\n"
            "    return fit_set._as_of\n"
        )
        assert _whole_example_private_access(seeded_other_metadata) == [
            "line 2: fit_set._as_of"
        ], (
            "the metadata allowance must not extend to any other private "
            "attribute, external or PortLearn-side"
        )

    def test_complete_example_consumes_contracts_through_public_surface_only(
        self,
    ) -> None:
        source = _example_source()
        violations = _whole_example_private_access(source)
        assert not violations, (
            "the complete shipped example must consume PortLearn contracts "
            "through public surfaces only, everywhere from module level to "
            f"every function body: private attribute access {violations} -- "
            "fit_set._records is private; iterate the information set "
            "publicly (iterate fit_set itself) instead"
        )
