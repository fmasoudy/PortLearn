"""The minimum-variance strategy contract (test-first).

This module defines the fixed behavior floors for the third built-in
classical strategy, ``portlearn.strategies.MinimumVariance``, exactly
as specified in the documented design for this strategy.

Contract discipline, per the fixed section 8 preamble:

* every expected value is independently computed or hand-pinned from
  the section 14 design-preflight references - never derived from the
  implementation's own arithmetic;
* every floor exercises a public path only (public import, public
  constructor, public ``decide``, public contracts);
* three narrowly sanctioned instrumentation seams, each mandated by
  the fixed section 8 matrix itself:

  1. the solver seam (T6/T7): a fake ``scipy.optimize.minimize`` is
     injected through ``sys.modules`` (the adapter imports scipy
     inside the function body per the section 7 lazy-import law), so
     the floors can observe the fixed deterministic solver settings,
     feed pinned solver vectors through the acceptance rules, and
     probe the fixed analytic Jacobians the adapter registers;
  2. the estimator boundary (T4): the wrapper-to-engine call
     ``portlearn._allocation.solve_long_only_min_variance`` is
     instrumented through pytest's auto-undone ``monkeypatch`` solely
     to inject the section 8/T4 asymmetric-covariance case;
  3. the validator call-through floor instruments the PUBLIC
     ``portlearn.interfaces.require_decision_result_compatible``
     attribute, proving causally that ``decide`` invokes the public
     validator on its own inputs before returning;

* the class is absent at authoring time, so every target-dependent
  node resolves it lazily inside the test body and fails with an
  ``EXPECTED_REJECTION``-prefixed causal message instead of a module-level
  collection error - while nodes anchored purely on fixed live
  surfaces act as always-green controls.
"""

from __future__ import annotations

import importlib
import math
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from portlearn.interfaces import (
    DecisionContext,
    Forecast,
    InformationSet,
)
from portlearn.observations import AmbiguousObservationError, TimedObservation
from portlearn.timing import InvalidChronologyError
from portlearn.weights import (
    PortfolioWeights,
    WeightConstraints,
    WeightState,
    require_valid_target,
)

# --------------------------------------------------------------------------- #
# The absent production target, resolved lazily per node                       #
# --------------------------------------------------------------------------- #


def _require_strategies_module():
    """Import ``portlearn.strategies`` or fail with a causal rejection marker."""
    try:
        return importlib.import_module("portlearn.strategies")
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"EXPECTED_REJECTION absent public module 'portlearn.strategies': {exc}",
            pytrace=False,
        )


def _require_minimum_variance():
    """Resolve the public ``MinimumVariance`` class or fail causally."""
    module = _require_strategies_module()
    try:
        return module.MinimumVariance
    except AttributeError as exc:
        pytest.fail(
            "EXPECTED_REJECTION absent public class 'MinimumVariance' on module "
            f"'portlearn.strategies': {exc}",
            pytrace=False,
        )


def _require_adapter_module():
    """Import the private scipy adapter module or fail causally."""
    try:
        return importlib.import_module("portlearn._optimizer_adapters.scipy_adapter")
    except ModuleNotFoundError as exc:
        pytest.fail(
            "EXPECTED_REJECTION absent internal adapter module "
            f"'portlearn._optimizer_adapters.scipy_adapter': {exc}",
            pytrace=False,
        )


def _require_solver_exceptions():
    """Resolve the two fixed solver exception classes or fail causally."""
    adapter = _require_adapter_module()
    try:
        return adapter.OptimizationUnavailableError, (adapter.OptimizationFailureError)
    except AttributeError as exc:
        pytest.fail(
            "EXPECTED_REJECTION absent solver exception classes on the adapter "
            f"module: {exc}",
            pytrace=False,
        )


def _require_allocation_engine():
    """Import the private allocation engine module or fail causally."""
    try:
        module = importlib.import_module("portlearn._allocation")
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"EXPECTED_REJECTION absent internal engine module 'portlearn._allocation': {exc}",
            pytrace=False,
        )
    try:
        return module, module.solve_long_only_min_variance
    except AttributeError as exc:
        pytest.fail(
            "EXPECTED_REJECTION absent engine function "
            f"'solve_long_only_min_variance': {exc}",
            pytrace=False,
        )


@contextmanager
def _fresh_modules_context() -> Iterator[None]:
    """Import ``portlearn*`` fresh inside the block, then RESTORE the
    exact prior module registry on exit (the established pattern)."""
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "portlearn" or name.startswith("portlearn.")
    }
    for name in saved:
        del sys.modules[name]
    try:
        yield
    finally:
        for name in [
            name
            for name in sys.modules
            if name == "portlearn" or name.startswith("portlearn.")
        ]:
            del sys.modules[name]
        sys.modules.update(saved)


# --------------------------------------------------------------------------- #
# Hand-set world constants (section 14 reference fixtures; no invented numbers)  #
# --------------------------------------------------------------------------- #

#: The balanced common grid T1..T4 (aware UTC month ends).
_T0 = datetime(2025, 12, 31, tzinfo=UTC)

_T1 = datetime(2026, 1, 31, tzinfo=UTC)
_T2 = datetime(2026, 2, 28, tzinfo=UTC)
_T3 = datetime(2026, 3, 31, tzinfo=UTC)
_T4 = datetime(2026, 4, 30, tzinfo=UTC)

#: The default decision instant (after the four-point grid).
_T5 = datetime(2026, 5, 31, tzinfo=UTC)

#: A later decision instant (the trailing-window drift replay).
_T6 = datetime(2026, 6, 30, tzinfo=UTC)

_GRID = (_T1, _T2, _T3, _T4)

#: Fixture A (section 8/T1, section 14 P1): the interior two-asset
#: reference.  LOW alternates +/- sqrt(3e-4), HIGH alternates +/- 0.015
#: with a half-period offset, so over the window T1..T4:
#:   S11 = 4*(sqrt(3e-4))^2/(T-1) = 4e-4 exactly,
#:   S22 = 4*(0.015)^2/3          = 3e-4 exactly,
#:   S12 = 0 exactly (the deviation products cancel in pairs),
#: matching the pinned reference covariance diag(4e-4, 3e-4).  The
#: long-only fully-invested GMV of a diagonal covariance is the
#: inverse-variance book, so w = (3/7, 4/7) exactly in rationals.
_A = math.sqrt(3e-4)
_A_LOW = (_A, -_A, _A, -_A)
_A_HIGH = (0.015, 0.015, -0.015, -0.015)

#: Hand-derived fixture A GMV (exact rationals, asserted within 1e-9):
#:   w_low  = (3e-4)^-1 / ((3e-4)^-1 + (4e-4)^-1) = 3/7,
#:   w_high = 4/7.
_HAND_A_LOW = 3.0 / 7.0
_HAND_A_HIGH = 4.0 / 7.0

#: The section 14.2 scipy 1.18.1 SLSQP transcript vector for fixture A.
_A_SLSQP = (0.4285714285714286, 0.5714285714285715)

#: Fixture B (section 8/T1): the boundary three-asset reference over the
#: five-point grid T1..T5 with window=5 and s = 0.01:
#:   LOW  = ( s,  0,  0,  0, -s),
#:   HIGH = ( 0,  s,  0,  0, -s),
#:   MID  = (2s, 2s, -s, -s, -2s),
#: giving the pinned covariance
#:   S = [[5e-5, 2.5e-5, 1e-4], [2.5e-5, 5e-5, 1e-4], [1e-4, 1e-4, 3.5e-4]]
#: whose exact GMV is (1/2, 1/2, 0) with asset 3 exactly at the
#: boundary (its gradient exceeds the active multipliers by 1.25e-4).
_S = 0.01
_B_LOW = (_S, 0.0, 0.0, 0.0, -_S)
_B_HIGH = (0.0, _S, 0.0, 0.0, -_S)
_B_MID = (2.0 * _S, 2.0 * _S, -_S, -_S, -2.0 * _S)
_B_GRID = (_T1, _T2, _T3, _T4, _T5)

#: The section 14.2 scipy 1.18.1 SLSQP transcript vector for fixture B.
_B_SLSQP = (0.4999999999999997, 0.5000000000000003, 0.0)

#: The kappa fixture: fixture A's shape scaled by 20, so the pinned
#: covariance is diag(0.16, 0.12) and the fixed acceptance kappa of
#: 1e-8 is discriminated by construction - a weight pair shifted by
#: only 2.5e-7 from the optimum carries a KKT residual of 1.4e-7,
#: which the fixed kappa rejects but a kappa loosened to 1e-6 would
#: accept (the floors pin kappa = 1e-8 by construction).
_K_LOW = tuple(20.0 * value for value in _A_LOW)
_K_HIGH = tuple(20.0 * value for value in _A_HIGH)

#: The kappa-tampered vector: feasible, exactly on the budget, with
#: active-gradient residual 1.4e-7 (fixed kappa rejects, 1e-6 would
#: not), and domination-clean - so only the KKT rule can catch it.
_K_TAMPER = (0.4285711785714285, 0.5714288214285714)

#: The trailing-window drift fixture over T1..T6 (window=4): deciding
#: at T5 selects T2..T5, deciding at T6 selects T3..T6, and the two
#: selected windows pin different covariances and different GMV books:
#:   at T5: S = [[0.0011, 0.0003897114317029974],
#:               [0.0003897114317029974, 0.00050625]],
#:           w = (0.14094671442094456, 0.8590532855790555)  (interior),
#:   at T6: S = [[0.002, 0.0010392304845413265],
#:               [0.0010392304845413265, 0.0006749999999999999]],
#:           w = (0.0, 1.0)  (boundary).
_DRIFT_LOW = (_A, -_A, _A, -_A, 3.0 * _A, -3.0 * _A)
_DRIFT_HIGH = (0.015, 0.015, -0.015, -0.015, 0.03, -0.03)
_DRIFT_GRID = (_T1, _T2, _T3, _T4, _T5, _T6)
_DRIFT_W5_LOW = 0.14094671442094456
_DRIFT_W6_LOW = 0.0

#: The asset-to-return-series mappings of the fixtures.
_MAP_A = {"LOW": "low_ret", "HIGH": "high_ret"}
_MAP_B = {"LOW": "low_ret", "HIGH": "high_ret", "MID": "mid_ret"}
_MAP_C = {"LOW": "low_ret", "HIGH": "high_ret", "MIMIC": "mimic_ret"}

#: The canonical supported constraint declaration:
#: long-only, fully invested - and nothing else.
_CANONICAL = WeightConstraints()


def _obs(series: str, instant: datetime, value: object) -> TimedObservation:
    """One admitted record: same-instant availability, hand-set value."""
    return TimedObservation(
        series_id=series,
        observation_time=instant,
        available_time=instant,
        value=value,
    )


def _series_records(
    series: str,
    values: tuple[object, ...],
    instants: tuple[datetime, ...],
) -> tuple[TimedObservation, ...]:
    """The hand-set records of one series over its instant grid."""
    return tuple(
        _obs(series, instant, value)
        for instant, value in zip(instants, values, strict=True)
    )


def _a_records(
    extra: tuple[TimedObservation, ...] = (),
) -> tuple[TimedObservation, ...]:
    """The fixture A panel over the common grid T1..T4."""
    records = _series_records("low_ret", _A_LOW, _GRID)
    records += _series_records("high_ret", _A_HIGH, _GRID)
    return (*records, *extra)


def _b_records() -> tuple[TimedObservation, ...]:
    """The fixture B panel over the five-point grid T1..T5."""
    records = _series_records("low_ret", _B_LOW, _B_GRID)
    records += _series_records("high_ret", _B_HIGH, _B_GRID)
    records += _series_records("mid_ret", _B_MID, _B_GRID)
    return records


def _k_records() -> tuple[TimedObservation, ...]:
    """The kappa fixture panel over the common grid T1..T4."""
    records = _series_records("low_ret", _K_LOW, _GRID)
    records += _series_records("high_ret", _K_HIGH, _GRID)
    return records


def _drift_records() -> tuple[TimedObservation, ...]:
    """The drift fixture panel over the six-point grid T1..T6."""
    records = _series_records("low_ret", _DRIFT_LOW, _DRIFT_GRID)
    records += _series_records("high_ret", _DRIFT_HIGH, _DRIFT_GRID)
    return records


def _information(records, as_of=None):
    """The admitted information set at ``as_of`` (default the decision)."""
    return InformationSet(tuple(records), as_of=_T5 if as_of is None else as_of)


def _context(
    universe,
    *,
    decision_time=None,
    current_weights=None,
    current_weights_as_of=None,
    strategy_state=None,
    forecast=None,
    information=None,
):
    """One lawful ``DecisionContext`` over ``universe`` at an instant.

    Defaults deliberately exercise the holdings-vs-universe law: the
    pre-decision book holds ``CASH`` only, which is OUTSIDE every
    universe used below, so a strategy that reads the holdings keys
    instead of the caller-declared universe is falsified by default.
    """
    instant = _T5 if decision_time is None else decision_time
    holdings = (
        current_weights
        if current_weights is not None
        else PortfolioWeights({"CASH": 1.0}, WeightState.PRE_TRADE)
    )
    admitted = (
        information
        if information is not None
        else _information(_a_records(), as_of=instant)
    )
    return DecisionContext(
        decision_time=instant,
        information=admitted,
        universe=tuple(universe),
        current_weights=holdings,
        current_weights_as_of=(
            instant if current_weights_as_of is None else current_weights_as_of
        ),
        strategy_state=strategy_state,
        forecast=forecast,
    )


def _target_of(strategy, context):
    """Resolve the decision produced by ``strategy`` under ``context``
    so cross-context comparisons are plain equality on the decision
    (with ``.weights`` available for dict-compare exact equality)."""
    return strategy.decide(context).decision.target_weights


# --------------------------------------------------------------------------- #
# Public import route, facade identity, and declared module surface (T9)       #
# --------------------------------------------------------------------------- #


def test_public_import_route_exposes_minimum_variance() -> None:
    """The public route ``from portlearn.strategies import
    MinimumVariance`` works and the class is a class."""
    try:
        from portlearn.strategies import MinimumVariance
    except ImportError as exc:
        pytest.fail(
            f"EXPECTED_REJECTION the public import route fails because the class "
            f"is absent: {exc}",
            pytrace=False,
        )
    assert isinstance(MinimumVariance, type)


def test_module_all_declares_all_three_builtins_in_order() -> None:
    """The module ships exactly the evolved public surface:
    ``__all__ == ["EqualWeight", "InverseVolatility",
    "MinimumVariance", "MeanVariance"]`` (list equality, declaration order included)
    (T9)."""
    module = _require_strategies_module()
    try:
        minimum_variance = module.MinimumVariance
    except AttributeError as exc:
        pytest.fail(
            f"EXPECTED_REJECTION absent public class 'MinimumVariance': {exc}",
            pytrace=False,
        )
    assert isinstance(minimum_variance, type)
    assert module.__all__ == [
        "EqualWeight",
        "InverseVolatility",
        "MinimumVariance",
        "MeanVariance",
    ]


def test_root_all_is_unchanged_by_the_third_builtin() -> None:
    """The root package surface is not expanded by the third strategy:
    ``portlearn.__all__`` stays ``["__version__", "data", "weights"]``
    (the strategies facade rides the lazy attribute branch alone)."""
    import portlearn

    assert portlearn.__all__ == ["__version__", "data", "weights"]


def test_lazy_facade_import_works_with_scipy_absent() -> None:
    """The lazy-import law: ``import portlearn.strategies``
    succeeds in this scipy-absent process without importing scipy,
    through both the direct route and the root facade."""
    with _fresh_modules_context():
        import portlearn

        direct = importlib.import_module("portlearn.strategies")
        assert "scipy" not in sys.modules
        assert direct is portlearn.strategies


# --------------------------------------------------------------------------- #
# Constructor law: mapping, window, constraints (unconditional, ValueError only) #
# --------------------------------------------------------------------------- #


def test_constructor_rejects_a_bare_string_mapping() -> None:
    """F1: a bare string — a common mistake for a mapping of one —
    rejects with ``ValueError`` naming the law and the received type."""
    minimum_variance = _require_minimum_variance()
    with pytest.raises(ValueError, match="mapping"):
        minimum_variance("low_ret", 4)


def test_constructor_rejects_other_non_mapping_types() -> None:
    """F1: every other non-mapping type rejects the same way."""
    minimum_variance = _require_minimum_variance()
    for bad in (None, 0, 4.0, [("LOW", "low_ret")], ("LOW", "high_ret")):
        with pytest.raises(ValueError, match="mapping"):
            minimum_variance(bad, 4)


def test_constructor_rejects_an_empty_mapping() -> None:
    """F1: an empty mapping is not a portfolio mapping."""
    minimum_variance = _require_minimum_variance()
    with pytest.raises(ValueError, match="mapping"):
        minimum_variance({}, 4)


def test_constructor_rejects_a_blank_asset_key() -> None:
    """F2: a blank-string asset identifier rejects naming the
    offending key."""
    minimum_variance = _require_minimum_variance()
    with pytest.raises(ValueError) as record:
        blank_key = {"": "low_ret", "LOW": "low_ret2"}
        minimum_variance(blank_key, 4)
    assert "identifier" in str(record.value)


def test_constructor_rejects_a_blank_series_identifier() -> None:
    """F2: a blank-string series identifier rejects naming the
    offending value."""
    minimum_variance = _require_minimum_variance()
    with pytest.raises(ValueError) as record:
        blank_series = {"LOW": "   "}
        minimum_variance(blank_series, 4)
    assert "identifier" in str(record.value)


def test_constructor_rejects_a_non_string_series_identifier() -> None:
    """F2: a series identifier that is not a string at all rejects."""
    minimum_variance = _require_minimum_variance()
    with pytest.raises(ValueError) as record:
        non_string = {"LOW": 7}
        minimum_variance(non_string, 4)
    assert "series" in str(record.value)


def test_constructor_rejects_a_series_mapped_by_two_assets() -> None:
    """F3: the one-to-one law — a series identifier mapped by two
    different assets rejects naming the series and both assets."""
    minimum_variance = _require_minimum_variance()
    with pytest.raises(ValueError) as record:
        colliding = {"LOW": "low_ret", "HIGH": "low_ret"}
        minimum_variance(colliding, 4)
    message = str(record.value)
    assert "low_ret" in message
    assert "LOW" in message
    assert "HIGH" in message


def test_constructor_rejects_a_window_below_two() -> None:
    """F4: the estimation window floor — ``window < 2`` rejects naming
    the field, the value, and the floor."""
    minimum_variance = _require_minimum_variance()
    with pytest.raises(ValueError) as record:
        minimum_variance(dict(_MAP_A), 1)
    message = str(record.value)
    assert "window" in message
    assert "1" in message


def test_constructor_rejects_a_non_int_window() -> None:
    """F4: a non-``int`` window rejects (floats included)."""
    minimum_variance = _require_minimum_variance()
    with pytest.raises(ValueError, match="window"):
        minimum_variance(dict(_MAP_A), 4.0)


def test_constructor_rejects_a_bool_window() -> None:
    """F4: ``bool`` is explicitly rejected even though it is an int
    subclass (no accidental ``window=True`` acceptance)."""
    minimum_variance = _require_minimum_variance()
    with pytest.raises(ValueError, match="window"):
        minimum_variance(dict(_MAP_A), True)


def test_constructor_rejects_a_non_constraints_object() -> None:
    """F5: ``constraints`` must be a ``WeightConstraints``; anything
    else rejects naming the received type."""
    minimum_variance = _require_minimum_variance()
    for bad in (None, {"budget": 1.0}, 1.0):
        with pytest.raises(ValueError) as record:
            minimum_variance(dict(_MAP_A), 4, constraints=bad)
        message = str(record.value)
        assert "constraints" in message
        assert type(bad).__name__ in message


def test_constructor_snapshots_the_mapping_immutably() -> None:
    """The construction-time snapshot law: later mutation of a
    caller-supplied dict cannot change the exposed ``return_series``
    view or any future decision."""
    minimum_variance = _require_minimum_variance()
    source = {"LOW": "low_ret", "HIGH": "high_ret"}
    strategy = minimum_variance(source, 4)
    source["LOW"] = "tampered_ret"
    source["X"] = "x_ret"
    view = strategy.return_series
    assert dict(view) == {"LOW": "low_ret", "HIGH": "high_ret"}
    with pytest.raises(TypeError):
        view["LOW"] = "tampered_ret"  # type: ignore[index]


def test_constructor_exposes_the_three_fields_read_only() -> None:
    """The public semantic state is exactly ``return_series``,
    ``window``, and ``constraints``; each is exposed through a
    read-only property (assignment rejects)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    assert dict(strategy.return_series) == dict(_MAP_A)
    assert strategy.window == 4
    assert strategy.constraints == _CANONICAL
    for field in ("return_series", "window", "constraints"):
        with pytest.raises(AttributeError):
            setattr(strategy, field, None)


def test_construct_succeeds_with_scipy_absent() -> None:
    """The dependency law: construction is solver-independent —
    ``MinimumVariance(...)`` constructs with scipy absent from the
    process (the failure belongs to ``decide`` alone, F15)."""
    assert "scipy" not in sys.modules
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    assert strategy.window == 4


# --------------------------------------------------------------------------- #
# Decision-time constraint check (F6, T5) and support check (F7)                 #
# --------------------------------------------------------------------------- #


def test_decide_rejects_shorting_declaration() -> None:
    """``allow_short=True`` is outside the supported subset: ``decide``
    fails closed naming the field, its value, and the subset (F6)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4, WeightConstraints(allow_short=True))
    with pytest.raises(ValueError, match="allow_short") as caught:
        strategy.decide(_context(("LOW", "HIGH")))
    assert "True" in str(caught.value)


def test_decide_rejects_partial_investment_budget() -> None:
    """Any ``budget`` other than the default full-investment level is
    a different estimand: abort naming field+value+subset (F6)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4, WeightConstraints(budget=0.5))
    with pytest.raises(ValueError, match="budget") as caught:
        strategy.decide(_context(("LOW", "HIGH")))
    assert "0.5" in str(caught.value)


def test_decide_rejects_gross_exposure_cap() -> None:
    """A declared ``max_gross_exposure`` narrows the canonical program:
    abort naming field+value+subset (F6)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(
        dict(_MAP_A), 4, WeightConstraints(max_gross_exposure=1.2)
    )
    with pytest.raises(ValueError, match="max_gross_exposure") as caught:
        strategy.decide(_context(("LOW", "HIGH")))
    assert "1.2" in str(caught.value)


def test_decide_rejects_net_exposure_cap() -> None:
    """A declared ``max_abs_net_exposure`` narrows the canonical
    program: abort naming field+value+subset (F6, T5)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(
        dict(_MAP_A), 4, WeightConstraints(max_abs_net_exposure=1.2)
    )
    with pytest.raises(ValueError, match="max_abs_net_exposure") as caught:
        strategy.decide(_context(("LOW", "HIGH")))
    assert "1.2" in str(caught.value)


def test_decide_rejects_singleton_universe() -> None:
    """A one-asset minimum-variance portfolio is degenerate: the
    support check fails closed naming the size (F7)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    with pytest.raises(ValueError, match="universe") as caught:
        strategy.decide(_context(("LOW",)))
    assert "1" in str(caught.value)


def test_decide_rejects_empty_universe() -> None:
    """Zero assets is not a portfolio: the support check fails closed
    naming the size (F7)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    with pytest.raises(ValueError, match="universe"):
        strategy.decide(_context(()))


# --------------------------------------------------------------------------- #
# Decision-time domain check (F8) and mapping-completeness check (F9)            #
# --------------------------------------------------------------------------- #


def test_decide_rejects_window_below_domain_floor() -> None:
    """``window < N + 1`` cannot produce a full-column-rank demeaned
    panel: the domain check fires before any rank computation or solve,
    naming ``window``, ``N``, and the law (F8)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_B), 3)
    with pytest.raises(ValueError) as caught:
        strategy.decide(_context(("LOW", "HIGH", "MID")))
    message = str(caught.value)
    assert "window" in message
    assert "3" in message
    assert "N" in message or "universe" in message or "assets" in message


def test_domain_gate_fires_before_the_solve() -> None:
    """The F8 check is causal: it fires even when the optimizer extra
    is absent, proving no solve is attempted before the domain check."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_B), 3)
    with _fresh_modules_context():
        original = sys.modules.get("scipy")

        def _refuse(name, *args, **kwargs):
            raise AssertionError(
                "EXPECTED_REJECTION the domain check must fire before any "
                "solver import is attempted"
            )

        import builtins

        saved_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "scipy" or name.startswith("scipy."):
                _refuse(name, *args, **kwargs)
            return saved_import(name, *args, **kwargs)

        builtins.__import__ = guarded
        try:
            with pytest.raises(ValueError, match="window"):
                strategy.decide(_context(("LOW", "HIGH", "MID")))
        finally:
            builtins.__import__ = saved_import
            if original is not None:
                sys.modules["scipy"] = original


def test_decide_rejects_universe_asset_without_mapping_entry() -> None:
    """A universe asset with no ``return_series`` entry fails the
    mapping-completeness check naming the asset (F9) — never a bare
    ``KeyError`` from an internal index."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    with pytest.raises(ValueError, match="OUTSIDE") as caught:
        strategy.decide(_context(("LOW", "HIGH", "OUTSIDE")))
    assert "return_series" in str(caught.value)


def test_extra_mapped_series_outside_universe_is_inert() -> None:
    """Mappings for assets not in the current universe are harmless:
    the universe keys the mapping, never the converse."""
    minimum_variance = _require_minimum_variance()
    mapping = dict(_MAP_A)
    mapping["OTHER"] = "other_ret"
    strategy = minimum_variance(mapping, 4)
    baseline_strategy = minimum_variance(dict(_MAP_A), 4)
    with_extra = _target_of(strategy, _context(("LOW", "HIGH")))
    baseline = _target_of(baseline_strategy, _context(("LOW", "HIGH")))
    assert tuple(with_extra.weights) == ("LOW", "HIGH")
    for asset in ("LOW", "HIGH"):
        assert dict(with_extra.weights)[asset] == pytest.approx(
            dict(baseline.weights)[asset], abs=1e-9
        ), f"the extra mapping for OTHER must be inert to {asset}'s weight"
    assert "OTHER" not in tuple(with_extra.weights)
    assert "OTHER" not in str(with_extra)
    assert "other_ret" not in str(with_extra)


# --------------------------------------------------------------------------- #
# Trailing-window law (F10) and panel law (F11)                                #
# --------------------------------------------------------------------------- #


def test_decide_rejects_short_history_for_a_mapped_series() -> None:
    """Fewer than ``window`` admissible observations for one mapped
    series fails closed naming asset, series, available count, and the
    required window (F10)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    short = _series_records("low_ret", _A_LOW[:3], _GRID[:3])
    short += _series_records("high_ret", _A_HIGH, _GRID)
    with pytest.raises(ValueError) as caught:
        strategy.decide(_context(("LOW", "HIGH"), information=_information(short)))
    message = str(caught.value)
    assert "LOW" in message
    assert "low_ret" in message
    assert "3" in message
    assert "4" in message


def test_decide_rejects_zero_admitted_observations() -> None:
    """Zero admitted observations for a mapped series fails the same
    insufficient-history law (F10)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    only_high = _series_records("high_ret", _A_HIGH, _GRID)
    with pytest.raises(ValueError) as caught:
        strategy.decide(_context(("LOW", "HIGH"), information=_information(only_high)))
    message = str(caught.value)
    assert "LOW" in message
    assert "low_ret" in message


def test_decide_ignores_extra_older_history_beyond_the_window() -> None:
    """Extra older admissible history beyond the latest ``window``
    observations is accepted and leaves the selected window — hence
    the weights — unchanged (latest-window selection, T2)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    records = _a_records()
    stale = _series_records("low_ret", (5.0,), (_T0,))
    stale += _series_records("high_ret", (-5.0,), (_T0,))
    with_old = records + stale
    target_plain = _target_of(strategy, _context(("LOW", "HIGH")))
    target_with_old = _target_of(
        strategy,
        _context(
            ("LOW", "HIGH"),
            information=_information(with_old, as_of=_T5),
        ),
    )
    assert target_plain == target_with_old


#: Hand-derived weights when the EARLIEST four of five observations are
#: selected (both assets carry a T0 outlier): the demeaned panel is
#: (5.0 - 0.01, 0.01 - 0.01, ...) — computed independently in scratch
#: (exact Fractions): w_low = 0.5317477840637863, w_high = 1 - that.
_HAND_OUTLIER_W_LOW = 0.5317477840637863


def test_window_selects_latest_observations_only() -> None:
    """A pre-window outlier that WOULD change the weights if wrongly
    included is ignored: the latest-4 window pins the interior reference
    weights, proving the selection is genuinely trailing (T2)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    records = _a_records()
    stale = _series_records("low_ret", (5.0,), (_T0,))
    stale += _series_records("high_ret", (-5.0,), (_T0,))
    with_old = records + stale
    target = _target_of(
        strategy,
        _context(
            ("LOW", "HIGH"),
            information=_information(with_old, as_of=_T5),
        ),
    )
    weights = dict(target.weights)
    assert weights["LOW"] == pytest.approx(_HAND_A_LOW, abs=1e-9)
    assert weights["HIGH"] == pytest.approx(_HAND_A_HIGH, abs=1e-9)


def test_window_shifts_with_advanced_decision_time() -> None:
    """Advancing ``decision_time`` with correspondingly advanced
    information shifts the trailing window: no stale first-observation
    reuse (T10 drift)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    admitted_by5 = tuple(
        record for record in _drift_records() if record.available_time <= _T5
    )
    admitted_by6 = tuple(
        record for record in _drift_records() if record.available_time <= _T6
    )
    ctx5 = _context(
        ("LOW", "HIGH"),
        decision_time=_T5,
        information=_information(admitted_by5, as_of=_T5),
    )
    ctx6 = _context(
        ("LOW", "HIGH"),
        decision_time=_T6,
        information=_information(admitted_by6, as_of=_T6),
    )
    target5 = _target_of(strategy, ctx5)
    target6 = _target_of(strategy, ctx6)
    weights5 = dict(target5.weights)
    weights6 = dict(target6.weights)
    assert weights5["LOW"] == pytest.approx(_DRIFT_W5_LOW, abs=1e-9)
    assert weights6["LOW"] == pytest.approx(_DRIFT_W6_LOW, abs=1e-9)


# --------------------------------------------------------------------------- #
# Panel law (F11)                                                              #
# --------------------------------------------------------------------------- #


def test_decide_rejects_unbalanced_panel_key_set_mismatch() -> None:
    """One asset missing one timestamp inside the selected window
    fails closed naming the mismatched assets and the first divergent
    instant (F11)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    low_records = _series_records("low_ret", _A_LOW[:4], _GRID[:4])
    high_records = _series_records(
        "high_ret", _A_HIGH[:3] + (0.015,), (_T1, _T2, _T3, _T5)
    )
    records = low_records + high_records
    assert len(low_records) == len(high_records) == 4
    with pytest.raises(ValueError) as caught:
        strategy.decide(_context(("LOW", "HIGH"), information=_information(records)))
    message = str(caught.value)
    assert "LOW" in message
    assert "HIGH" in message
    assert "2026-04-30" in message


def test_decide_permuted_submission_order_yields_identical_weights() -> None:
    """The panel is built from the record SET: permuted submission
    order of identical records yields identical weights (T3)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    straight = _target_of(
        strategy,
        _context(("LOW", "HIGH"), information=_information(_a_records())),
    )
    permuted_records = tuple(reversed(_a_records()))
    permuted = _target_of(
        strategy,
        _context(("LOW", "HIGH"), information=_information(permuted_records)),
    )
    assert dict(straight.weights) == dict(permuted.weights)


def test_decide_universe_reordering_permutes_weights_correspondingly() -> None:
    """Universe reordering produces the correspondingly permuted TARGET
    (same asset weights keyed by identifier; presentation order is
    universe order) — and exact equality pins the map, not a sequence
    compare (T3)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    low_first = _target_of(strategy, _context(("LOW", "HIGH")))
    high_first = _target_of(strategy, _context(("HIGH", "LOW")))
    low_map = dict(low_first.weights)
    high_map = dict(high_first.weights)
    assert tuple(low_first.weights) == ("LOW", "HIGH")
    assert tuple(high_first.weights) == ("HIGH", "LOW")
    assert set(low_map) == set(high_map) == {"LOW", "HIGH"}
    for asset in ("LOW", "HIGH"):
        assert high_map[asset] == pytest.approx(low_map[asset], abs=1e-9)


# --------------------------------------------------------------------------- #
# Numerics check (F12) and covariance symmetry (F13)                            #
# --------------------------------------------------------------------------- #


def test_decide_rejects_non_finite_panel_value() -> None:
    """A NaN panel entry fails the numerics check naming the asset,
    timestamp, and value (F12) — never reaching the solver."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    records = _series_records("low_ret", (float("nan"), 0.0, 0.0, 0.0), _GRID)
    records += _series_records("high_ret", _A_HIGH, _GRID)
    with pytest.raises(ValueError) as caught:
        strategy.decide(_context(("LOW", "HIGH"), information=_information(records)))
    message = str(caught.value)
    assert "LOW" in message
    assert "2026-01-31" in message


def test_decide_rejects_asymmetric_covariance_from_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An asymmetric covariance is rejected by exact transpose equality
    before any solve (F13): the wrapper-to-engine call — the
    sanctioned estimator boundary — is spliced with pytest's
    auto-undone ``monkeypatch`` to perturb one off-diagonal by a
    material amount and hand the tampered matrix to the real engine."""
    minimum_variance = _require_minimum_variance()
    engine_module, engine = _require_allocation_engine()
    strategy = minimum_variance(dict(_MAP_A), 4)

    def splice(covariance, identifiers):
        tampered = [list(row) for row in covariance]
        tampered[0][1] = tampered[0][1] + 1e-5
        return engine(tampered, identifiers)

    monkeypatch.setattr(engine_module, "solve_long_only_min_variance", splice)
    with pytest.raises(ValueError, match="symmetr|transpose"):
        strategy.decide(_context(("LOW", "HIGH")))


# --------------------------------------------------------------------------- #
# Rank reference (F14)                                                            #
# --------------------------------------------------------------------------- #


#: Fixture C (section 8/T4): MIMIC = LOW + HIGH over the grid with a
#: q = 1/512 dyadic offset, making the demeaned deviation matrix rank 2
#: < N = 3 while satisfying window >= N + 1 (4 >= 4).
_Q = 1.0 / 512.0
_C_MIMIC = tuple(lv + hv + _Q for lv, hv in zip(_A_LOW, _A_HIGH, strict=True))


def _c_records() -> tuple[TimedObservation, ...]:
    records = _series_records("low_ret", _A_LOW, _GRID)
    records += _series_records("high_ret", _A_HIGH, _GRID)
    records += _series_records("mimic_ret", _C_MIMIC, _GRID)
    return records


def test_decide_rejects_rank_deficient_panel() -> None:
    """A panel whose demeaned deviation matrix has exact rank 2 < N=3
    fails closed naming the computed rank and N (F14) — the C-fixture
    satisfies window >= N + 1, so only the rank reference can catch it."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_C), 4)

    def _never_called(fun, x0, **kwargs):  # bounded cure: pre-solve observability
        raise AssertionError("EXPECTED_REJECTION the rank check must fire before any solve")

    with _scipy_fake(_never_called), pytest.raises(ValueError) as caught:
        strategy.decide(
            _context(("LOW", "HIGH", "MIMIC"), information=_information(_c_records()))
        )
    message = str(caught.value)
    assert "rank" in message
    assert "2" in message
    assert "3" in message


def test_decide_accepts_same_shape_without_the_mimic() -> None:
    """Control: dropping MIMIC from the same universe (N=2, same
    window) leaves a full-rank panel that solves to the fixture-A
    reference — proving F14 is the mimic's exact linear dependence, not
    the fixture shape."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    target = _target_of(
        strategy,
        _context(("LOW", "HIGH"), information=_information(_c_records())),
    )
    weights = dict(target.weights)
    assert weights["LOW"] == pytest.approx(_HAND_A_LOW, abs=1e-9)
    assert weights["HIGH"] == pytest.approx(_HAND_A_HIGH, abs=1e-9)


# --------------------------------------------------------------------------- #
# Solver seam instrumentation (the sanctioned T6/T7 solver boundary)           #
# --------------------------------------------------------------------------- #


def _fake_result(
    x: list[float], success: bool = True, status: int = 0
) -> SimpleNamespace:
    """A minimal stand-in for the scipy ``OptimizeResult`` surface the
    adapter is allowed to read: ``x``, ``success``, ``status``."""
    return SimpleNamespace(x=list(x), success=success, status=status)


@contextmanager
def _scipy_absent() -> Iterator[None]:
    """Make ``scipy`` unimportable for the block, restoring the exact
    prior registry on exit (simulating the missing optional extra)."""
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "scipy" or name.startswith("scipy.")
    }
    for name in saved:
        del sys.modules[name]
    import builtins

    real_import = builtins.__import__

    def refusing(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = refusing
    try:
        yield
    finally:
        builtins.__import__ = real_import
        sys.modules.update(saved)


@contextmanager
def _scipy_fake(minimize_fn) -> Iterator[None]:
    """Install a fake ``scipy.optimize.minimize`` through
    ``sys.modules`` (the adapter imports scipy inside its function
    body per the section 7 lazy-import law), restoring the exact prior
    registry on exit."""
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "scipy" or name.startswith("scipy.")
    }
    for name in saved:
        del sys.modules[name]
    scipy_module = types.ModuleType("scipy")
    optimize_module = types.ModuleType("scipy.optimize")
    scipy_module.optimize = optimize_module
    optimize_module.minimize = minimize_fn
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.optimize"] = optimize_module
    try:
        yield
    finally:
        for name in [
            name for name in sys.modules if name == "scipy" or name.startswith("scipy.")
        ]:
            del sys.modules[name]
        sys.modules.update(saved)


def test_decide_raises_unavailable_when_scipy_absent() -> None:
    """When scipy is absent (simulated import failure in a broken
    environment), ``decide`` fails closed with
    ``OptimizationUnavailableError`` — a ``ValueError`` subclass —
    naming the broken environment (F15)."""
    unavailable, _ = _require_solver_exceptions()
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    with _scipy_absent(), pytest.raises(ValueError, match="scipy") as caught:
        strategy.decide(_context(("LOW", "HIGH")))
    assert isinstance(caught.value, unavailable)
    message = str(caught.value)
    assert "pip install" in message


def test_unavailable_error_subclasses_value_error() -> None:
    """Both fixed solver exceptions live INSIDE the ``ValueError``
    family — additions inside it, never siblings beside it."""
    unavailable, failure = _require_solver_exceptions()
    assert issubclass(unavailable, ValueError)
    assert issubclass(failure, ValueError)


def test_decide_raises_failure_on_solver_non_success() -> None:
    """A solver result carrying a non-success status fails closed with
    ``OptimizationFailureError`` naming the status — no fallback solve,
    no retry (F16)."""
    _, failure = _require_solver_exceptions()
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    calls = []

    def failing_minimize(*args, **kwargs):
        calls.append((args, kwargs))
        return _fake_result([0.5, 0.5], success=False, status=8)

    with _scipy_fake(failing_minimize), pytest.raises(ValueError, match="8") as caught:
        strategy.decide(_context(("LOW", "HIGH")))
    assert isinstance(caught.value, failure)
    assert len(calls) == 1, "no fallback solve, no retry"


def test_decide_uses_pinned_deterministic_solver_settings() -> None:
    """The adapter owns the fixed deterministic configuration: method
    SLSQP, ``ftol=1e-12``, ``maxiter=1000``, initial point the uniform
    vector — and no caller override path exists."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    calls = []

    def recording_minimize(fun, x0, **kwargs):
        calls.append((fun, x0, kwargs))
        return _fake_result(list(_A_SLSQP))

    with _scipy_fake(recording_minimize):
        strategy.decide(_context(("LOW", "HIGH")))
    assert len(calls) == 1
    _, x0, kwargs = calls[0]
    assert kwargs.get("method") == "SLSQP"
    assert list(x0) == [0.5, 0.5]
    options = kwargs.get("options")
    assert isinstance(options, dict)
    assert options.get("ftol") == 1e-12
    assert options.get("maxiter") == 1000


def test_decide_registers_analytic_jacobians() -> None:
    """The fixed analytic gradients are part of the tested solver
    contract: the objective Jacobian is exactly ``2Σw`` and the
    equality-constraint Jacobian is the constant ones row (
    T7 jac floor) — pinned at the hand-computed probe values."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    calls = []

    def recording_minimize(fun, x0, **kwargs):
        calls.append((fun, x0, kwargs))
        return _fake_result(list(_A_SLSQP))

    with _scipy_fake(recording_minimize):
        strategy.decide(_context(("LOW", "HIGH")))
    _, _, kwargs = calls[0]
    jac = kwargs.get("jac")
    assert callable(jac), "analytic objective Jacobian must be registered"
    #: 2 Σ w at the uniform point of fixture A = (4e-4, 3e-4) exactly.
    assert list(jac([0.5, 0.5])) == [4e-4, 3e-4]
    constraints = kwargs.get("constraints")
    assert isinstance(constraints, (list, tuple)) and len(constraints) >= 1
    equality = constraints[0]
    assert callable(equality.get("fun"))
    eq_jac = equality.get("jac")
    assert callable(eq_jac), "constraint Jacobian of ones must be registered"
    assert list(eq_jac([0.5, 0.5])) == [1.0, 1.0]
    assert equality.get("type") == "eq"


# --------------------------------------------------------------------------- #
# Post-solve acceptance rules (F17): KKT, domination, budget, byte laws        #
# --------------------------------------------------------------------------- #


def test_decide_rejects_kkt_tampered_solver_vector() -> None:
    """A feasible-but-suboptimal vector with active-gradient residual
    1.4e-7 — inside a loosened kappa but OUTSIDE the fixed kappa of
    1e-8 — is rejected: the floors pin kappa = 1e-8 by construction
    (F17, T7)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    calls = []

    def tampered_minimize(fun, x0, **kwargs):
        calls.append(1)
        return _fake_result(list(_K_TAMPER))

    #: The kappa fixture: fixture A scaled by 20, so the pinned
    #: covariance is diag(0.16, 0.12) and _K_TAMPER's active-gradient
    #: residual of 1.4e-7 is discriminated by the fixed kappa = 1e-8
    #: by construction (a kappa loosened to 1e-6 would accept it).
    with _scipy_fake(tampered_minimize), pytest.raises(ValueError) as caught:
        strategy.decide(
            _context(("LOW", "HIGH"), information=_information(_k_records()))
        )
    message = str(caught.value)
    assert "KKT" in message or "kkt" in message or "marginal" in message
    #: 2 * 5e-10 = 1e-9 == budget_tolerance: no double counting.
    assert len(calls) == 1


def test_kkt_tamper_passes_every_rule_except_kkt() -> None:
    """Control proving the KKT rejection is causal, not incidental:
    the tampered vector is budget-exact, long-only, and
    domination-clean — only the active-set gradient equality can catch
    it (independently verified in scratch)."""
    #: Sum of _K_TAMPER is exactly 1.0; every entry positive; the
    #: domination law finds no strictly-better feasible neighbor.
    total = math.fsum(_K_TAMPER)
    assert total == 1.0
    assert all(w > 0 for w in _K_TAMPER)


def test_decide_rejects_domination_tampered_solver_vector() -> None:
    """A vector that satisfies budget, sign, and the KKT residual rule
    but is strictly dominated by a projected budget-transfer neighbor
    is rejected (F17, T7)."""
    #: (0.42856942857142855, 0.5714305714285715): KKT-clean (residual
    #: 2.8e-9 < 1e-8), budget-exact, long-only — but moving a 1e-6
    #: budget transfer toward the true optimum strictly improves the
    #: objective (independently verified in scratch).
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    calls = []

    def dominated_minimize(fun, x0, **kwargs):
        calls.append(1)
        return _fake_result([0.42856942857142855, 0.5714305714285715])

    with _scipy_fake(dominated_minimize), pytest.raises(ValueError) as caught:
        strategy.decide(_context(("LOW", "HIGH")))
    message = str(caught.value)
    assert "dominat" in message
    assert len(calls) == 1


def test_decide_rejects_budget_breaking_solver_vector() -> None:
    """A vector summing to 1 + 5e-9 — beyond ``budget_tolerance`` — is
    rejected by rule 2 even though its KKT residual is clean (F17)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)

    half = (1.0 + 5e-9) / 2.0

    def budget_breaking(fun, x0, **kwargs):
        return _fake_result([half, half])

    with _scipy_fake(budget_breaking), pytest.raises(ValueError) as caught:
        strategy.decide(_context(("LOW", "HIGH")))
    message = str(caught.value)
    assert "budget" in message


def test_decide_rejects_negative_weight_solver_vector() -> None:
    """A sign-dust vector (one entry at −1e-15, sum within tolerance,
    KKT-clean) is rejected by the exact sign rule alone (F17): the
    clipped variant (dust moved to +1e-15) is acceptance-clean in
    scratch, so clipping would change the emitted bytes — this floor
    kills it."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_B), 5)

    def sign_dust(fun, x0, **kwargs):
        return _fake_result([0.5 - 5e-13, 0.5 + 5e-13, -1e-15])

    with _scipy_fake(sign_dust), pytest.raises(ValueError) as caught:
        strategy.decide(
            _context(("LOW", "HIGH", "MID"), information=_information(_b_records()))
        )
    message = str(caught.value)
    assert "negative" in message or "sign" in message


def test_decide_does_not_renormalize_solver_vector_bytes() -> None:
    """No renormalization: the returned vector's exact bytes reach the
    emitted TARGET.  A vector summing to 1 − 5e-10 — INSIDE budget
    tolerance, KKT-clean, domination-clean — must be emitted
    identically; renormalizing it to sum exactly 1 would change
    the bytes (F17/T6 kill, T10 replay)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    #: (sum 0.9999999995): budget-residual 5e-10 <= 1e-9 passes.
    vector = (0.4285714285714286, 0.5714285709285715)

    def renorm_bait(fun, x0, **kwargs):
        return _fake_result(list(vector))

    with _scipy_fake(renorm_bait):
        target = _target_of(strategy, _context(("LOW", "HIGH")))
    weights = dict(target.weights)
    assert weights["LOW"] == vector[0]
    assert weights["HIGH"] == vector[1]
    assert math.fsum(weights.values()) != 1.0


def test_decide_does_not_clip_positive_rounding_dust() -> None:
    """No clipping: positive rounding dust on a boundary asset (1e-15
    on asset 3 of fixture B) passes acceptance (KKT classifies it
    boundary by the fixed tau_w law) and reaches the TARGET with the
    dust intact — clipping would change the bytes."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_B), 5)

    def with_dust(fun, x0, **kwargs):
        return _fake_result([0.5 - 5e-13, 0.5 + 5e-13, 1e-15])

    with _scipy_fake(with_dust):
        target = _target_of(
            strategy,
            _context(("LOW", "HIGH", "MID"), information=_information(_b_records())),
        )
    weights = dict(target.weights)
    assert weights["MID"] == 1e-15


# --------------------------------------------------------------------------- #
# Successor section 15.3 amendment floors (three nodes, 70 -> 73): the         #
# relative domination deadband, the ordered i != j direction set, and         #
# the scale-aware KKT tolerance                                               #
# --------------------------------------------------------------------------- #

#: The mu-scale fixture (section 15.3 item 3): fixture A's shape scaled
#: by 100, so the pinned covariance is diag(4, 3) - values scale by 100,
#: covariance by 100^2 - and the optimum stays (3/7, 4/7), but the
#: active-set multiplier grows to mu = 24/7 ~= 3.4286 > 1, so the fixed
#: scale-aware tolerance kappa * max(1, |mu|) ~= 3.4286e-8 strictly
#: exceeds the bare kappa = 1e-8 (every figure below hand-recomputed).
_M_LOW = tuple(100.0 * value for value in _A_LOW)
_M_HIGH = tuple(100.0 * value for value in _A_HIGH)


def _m_records() -> tuple[TimedObservation, ...]:
    """The mu-scale fixture panel over the common grid T1..T4."""
    records = _series_records("low_ret", _M_LOW, _GRID)
    records += _series_records("high_ret", _M_HIGH, _GRID)
    return records


def test_decide_honors_the_relative_domination_deadband_at_the_exact_tie_law() -> None:
    """The domination tie is exactly relative: tie = 1e-13 * |f(w)|.

    On fixture A (f(w*) ~= 1.714286e-4, so tie ~= 1.714e-17) the two
    hand-recomputed probes bracket the deadband: INSIDE, a vector whose
    best projected budget-transfer neighbor improves f by 8.538e-18 =
    0.50 x tie, is ACCEPTED identically (KKT residual 7.09e-10,
    budget exact - only the deadband law decides); BEYOND, a vector
    improving by 4.269e-17 = 2.49 x tie, is rejected.  A deadband of
    1e-12 would accept BEYOND and 1e-14 would reject INSIDE, so the
    pair pins the 1e-13 relative law (section 15.3 item 1; already-green
    behavior being fixed)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)

    #: w_low = 3/7 - 5.061e-7: improvement 7e-4 * (2*delta*1e-6 - 1e-12)
    #: = 0.50 x tie at the best neighbor (pair (0,1), eps=1e-6).
    def inside_deadband(fun, x0, **kwargs):
        return _fake_result([0.42857092247142853, 0.5714290775285715])

    with _scipy_fake(inside_deadband):
        target = _target_of(strategy, _context(("LOW", "HIGH")))
    weights = dict(target.weights)
    assert weights["LOW"] == 0.42857092247142853
    assert weights["HIGH"] == 0.5714290775285715

    #: w_low = 3/7 - 5.305e-7: improvement 2.49 x tie - beyond the law.
    def beyond_deadband(fun, x0, **kwargs):
        return _fake_result([0.42857089807142856, 0.5714291019285714])

    with _scipy_fake(beyond_deadband), pytest.raises(ValueError) as caught:
        strategy.decide(_context(("LOW", "HIGH")))
    assert "dominat" in str(caught.value)


def test_decide_rejects_reverse_direction_only_dominated_solver_vector() -> None:
    """Ordered i != j coverage: a vector dominated ONLY along the
    reverse direction is rejected.

    w_low = 3/7 + 2e-6 (LOW overweight) is KKT-clean (residual
    2.8e-9 < kappa) and budget-exact, and its strictly improving
    neighbor is reachable only along e_HIGH - e_LOW (pair (i=1, j=0),
    improvement 2.1e-15 = 122 x tie at eps=1e-6): the forward
    e_LOW - e_HIGH direction worsens f at every epsilon, so the
    superseded 1 <= i < j half-law would accept this vector.  The
    rejection message names the reverse direction (section 15.3 item
    2; already-green behavior being fixed)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    calls = []

    def reverse_only(fun, x0, **kwargs):
        calls.append(1)
        return _fake_result([0.42857342857142855, 0.5714265714285715])

    with _scipy_fake(reverse_only), pytest.raises(ValueError) as caught:
        strategy.decide(_context(("LOW", "HIGH")))
    message = str(caught.value)
    assert "dominat" in message
    assert "e_'HIGH'-e_'LOW'" in message
    assert len(calls) == 1


def test_decide_applies_the_scale_aware_kkt_tolerance_when_mu_exceeds_one() -> None:
    """The KKT tolerance scales with the multiplier: |g_i - mu| <=
    kappa * max(1, |mu|) (section 5.4 rule 3, section 15.3 item 3).

    On the mu-scale fixture (Sigma = diag(4, 3), mu = 24/7 at the
    optimum) the gradients are g_low = 8*w_low and g_high = 6*w_high,
    so a shift delta off the optimum carries residual 14*delta:

    * WITHIN: delta = 1.85e-8/14, residual = 1.85e-8 - beyond the bare
      kappa = 1e-8 but inside kappa * max(1, |mu|) = 3.4286e-8 - is
      budget-exact and domination-clean, so the fixed law REQUIRES its
      acceptance and identical emission.  This is the boundary case
      production must accept: the current residual check tests the
      residual against bare kappa and rejects it.
    * BEYOND: delta = 5e-9, residual = 7e-8 - outside even the
      scale-aware tolerance, rejected either way (control: the relative
      law is not a loosening beyond its scale)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    calls = []

    def within_scale(fun, x0, **kwargs):
        calls.append(1)
        return _fake_result([0.4285714298928571, 0.5714285701071429])

    with _scipy_fake(within_scale):
        target = _target_of(
            strategy, _context(("LOW", "HIGH"), information=_information(_m_records()))
        )
    weights = dict(target.weights)
    assert weights["LOW"] == 0.4285714298928571
    assert weights["HIGH"] == 0.5714285701071429
    assert math.fsum(weights.values()) == 1.0
    assert len(calls) == 1

    def beyond_scale(fun, x0, **kwargs):
        return _fake_result([0.4285714335714286, 0.5714285664285714])

    with _scipy_fake(beyond_scale), pytest.raises(ValueError) as caught:
        strategy.decide(
            _context(("LOW", "HIGH"), information=_information(_m_records()))
        )
    message = str(caught.value)
    assert "KKT" in message or "kkt" in message or "marginal" in message


# --------------------------------------------------------------------------- #
# Estimand references (T1)                                                        #
# --------------------------------------------------------------------------- #


def test_oracle_diagonal() -> None:
    """Fixture A: Σ = diag(4e-4, 3e-4) → GMV = (3/7, 4/7) exact within
    1e-9, budget and sign hold, and the scale-aware rules pass at the
    reference (T1)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    target = _target_of(strategy, _context(("LOW", "HIGH")))
    weights = dict(target.weights)
    assert weights["LOW"] == pytest.approx(_HAND_A_LOW, abs=1e-9)
    assert weights["HIGH"] == pytest.approx(_HAND_A_HIGH, abs=1e-9)
    assert math.fsum(weights.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(w >= 0.0 for w in weights.values())


def test_oracle_boundary() -> None:
    """Fixture B: 3-asset correlated Σ → GMV = (1/2, 1/2, 0) with
    asset 3 exactly at the boundary, within 1e-9; the scipy
    transcript vector is emitted within the same pin (T1)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_B), 5)
    target = _target_of(
        strategy,
        _context(("LOW", "HIGH", "MID"), information=_information(_b_records())),
    )
    weights = dict(target.weights)
    assert weights["LOW"] == pytest.approx(_B_SLSQP[0], abs=1e-9)
    assert weights["HIGH"] == pytest.approx(_B_SLSQP[1], abs=1e-9)
    assert weights["MID"] == pytest.approx(_B_SLSQP[2], abs=1e-9)
    assert math.fsum(weights.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(w >= 0.0 for w in weights.values())


# --------------------------------------------------------------------------- #
# Target independence (T8) and replay determinism (T10)                        #
# --------------------------------------------------------------------------- #


def test_target_ignores_current_weights() -> None:
    """Varying ``current_weights`` — including holdings outside the
    universe — leaves the TARGET identical."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    baseline = _target_of(strategy, _context(("LOW", "HIGH")))
    varied_book = _target_of(
        strategy,
        _context(
            ("LOW", "HIGH"),
            current_weights=PortfolioWeights(
                {"OUTSIDE": 0.7, "LOW": 0.3}, WeightState.PRE_TRADE
            ),
        ),
    )
    assert dict(baseline.weights) == dict(varied_book.weights)


def test_target_ignores_current_weights_for_warm_start() -> None:
    """The fixed solver settings have no warm start — in particular
    never from ``current_weights``: the exact ``x0`` handed to the
    solver is the uniform vector regardless of the book."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    calls = []

    def recording_minimize(fun, x0, **kwargs):
        calls.append(list(x0))
        return _fake_result(list(_A_SLSQP))

    with _scipy_fake(recording_minimize):
        strategy.decide(
            _context(
                ("LOW", "HIGH"),
                current_weights=PortfolioWeights(
                    {"LOW": 0.9, "HIGH": 0.1}, WeightState.PRE_TRADE
                ),
            )
        )
    assert calls == [[0.5, 0.5]]


def test_target_ignores_strategy_state_and_emits_none() -> None:
    """A carried ``strategy_state`` never changes the TARGET, and the
    emitted ``next_strategy_state`` is ``None``."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    plain = strategy.decide(_context(("LOW", "HIGH")))
    stated = strategy.decide(_context(("LOW", "HIGH"), strategy_state={"epoch": 41}))
    assert dict(plain.decision.target_weights.weights) == dict(
        stated.decision.target_weights.weights
    )
    assert stated.next_strategy_state is None


def test_target_ignores_present_forecast() -> None:
    """A present forecast is legal and ignored — the minimum-variance
    estimand is forecast-free."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    baseline = _target_of(strategy, _context(("LOW", "HIGH")))
    forecast = Forecast(
        values={"LOW": -0.4, "HIGH": 0.9},
        target="expected_return",
        decision_time=_T5,
        produced_by="scratch-probe",
    )
    forecasted = _target_of(strategy, _context(("LOW", "HIGH"), forecast=forecast))
    assert dict(baseline.weights) == dict(forecasted.weights)


# --------------------------------------------------------------------------- #
# Replay determinism (T10) and scale invariance                                #
# --------------------------------------------------------------------------- #


def test_replay_same_inputs_emits_byte_identical_weights() -> None:
    """Replaying the identical semantic input tuple in the same
    process and environment emits identical weights —
    no hidden RNG."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    first = dict(_target_of(strategy, _context(("LOW", "HIGH"))).weights)
    second = dict(_target_of(strategy, _context(("LOW", "HIGH"))).weights)
    assert first == second


def test_target_is_invariant_to_covariance_scale() -> None:
    """Scale equivalence: ×100 on every return leaves the argmin
    unchanged (the order law applied to the matrix; probe P-A6) — a
    per-period scaling knob would be allocation-inert."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    base_target = _target_of(strategy, _context(("LOW", "HIGH")))
    scaled_records = _series_records("low_ret", tuple(10.0 * v for v in _A_LOW), _GRID)
    scaled_records += _series_records(
        "high_ret", tuple(10.0 * v for v in _A_HIGH), _GRID
    )
    scaled_target = _target_of(
        strategy,
        _context(("LOW", "HIGH"), information=_information(scaled_records)),
    )
    assert dict(base_target.weights)["LOW"] == pytest.approx(
        dict(scaled_target.weights)["LOW"], abs=1e-9
    )


def test_universe_reordering_is_presentation_only() -> None:
    """Universe reordering is presentation-only: the per-asset weight
    mapping is identical (checked in the panel-law section too, from
    the set side); this floor pins it from the replay side."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    low_first = _target_of(strategy, _context(("LOW", "HIGH")))
    high_first = _target_of(strategy, _context(("HIGH", "LOW")))
    assert tuple(low_first.weights) == ("LOW", "HIGH")
    assert tuple(high_first.weights) == ("HIGH", "LOW")
    for asset in ("LOW", "HIGH"):
        assert dict(high_first.weights)[asset] == pytest.approx(
            dict(low_first.weights)[asset], abs=1e-9
        )


def test_validator_is_invoked_on_the_emitted_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public validator is the final check: ``decide``
    invokes ``require_decision_result_compatible`` on its own inputs —
    the very context it was given and the very result it returns —
    exactly once, before returning (the sanctioned validator
    call-through floor)."""
    minimum_variance = _require_minimum_variance()
    strategy = minimum_variance(dict(_MAP_A), 4)
    context = _context(("LOW", "HIGH"))

    import portlearn.interfaces as pl_iface
    import portlearn.strategies as pl_strategies

    #: keep the real result-validator so the F18 probe below is unmasked
    real_result_validator = pl_iface.require_decision_result_compatible

    class _ValidatorSentinel(ValueError):
        pass

    def _raising(context_arg, result_arg):
        raise _ValidatorSentinel

    monkeypatch.setattr(pl_iface, "require_decision_result_compatible", _raising)
    with pytest.raises(_ValidatorSentinel):
        strategy.decide(context)

    #: F18: the strategy-module validator seam — the
    #: bound name the wrapper resolves on an attribute lookup at the
    #: post-solve acceptance step — must reject unconditionally when the
    #: strict validator rejects the constructed TARGET (the fixed
    #: ValueError-family law, belt-and-braces).
    real_target_validator = pl_strategies.require_valid_target

    def _sentinel_target_validator(weights, constraints=None, *args, **kwargs):
        raise _ValidatorSentinel

    monkeypatch.setattr(
        pl_strategies, "require_valid_target", _sentinel_target_validator
    )
    monkeypatch.setattr(
        pl_iface, "require_decision_result_compatible", real_result_validator
    )
    try:
        with pytest.raises(_ValidatorSentinel):
            strategy.decide(context)
    finally:
        #: restore the true seam before the result-compatible checks.
        monkeypatch.setattr(
            pl_strategies, "require_valid_target", real_target_validator
        )
    assert pl_strategies.require_valid_target is real_target_validator

    recorded: list[tuple[object, object]] = []

    def _recording(context_arg, result_arg):
        recorded.append((context_arg, result_arg))

    monkeypatch.setattr(pl_iface, "require_decision_result_compatible", _recording)
    result = strategy.decide(context)
    assert len(recorded) == 1, (
        f"the public validator must be called exactly once; recorded {len(recorded)}"
    )
    recorded_context, recorded_result = recorded[0]
    assert recorded_context is context
    assert recorded_result is result


def test_replay_is_stable_across_strategy_instances() -> None:
    """Two independently constructed instances with identical fixed
    state emit identical weights (the same-input replay law extended
    to construction: no per-instance hidden RNG)."""
    minimum_variance = _require_minimum_variance()
    first = minimum_variance(dict(_MAP_A), 4)
    second = minimum_variance(dict(_MAP_A), 4)
    target_one = _target_of(first, _context(("LOW", "HIGH")))
    target_two = _target_of(second, _context(("LOW", "HIGH")))
    assert dict(target_one.weights) == dict(target_two.weights)


# --------------------------------------------------------------------------- #
# Inherited fixed contract controls (always passing, no strategy involvement)   #
# --------------------------------------------------------------------------- #


def test_frozen_validator_accepts_the_oracle_targets() -> None:
    """Independent control: the public validator accepts both
    hand-pinned reference books (a property of the strict validator and
    the reference values alone — no strategy involvement)."""
    book_a = PortfolioWeights(
        {"LOW": _HAND_A_LOW, "HIGH": _HAND_A_HIGH}, WeightState.TARGET
    )
    assert require_valid_target(book_a, _CANONICAL) is None
    book_b = PortfolioWeights(
        {"LOW": _B_SLSQP[0], "HIGH": _B_SLSQP[1], "MID": _B_SLSQP[2]},
        WeightState.TARGET,
    )
    assert require_valid_target(book_b, _CANONICAL) is None


def test_frozen_validator_rejects_a_negative_weight_target() -> None:
    """Independent control (validator teeth): a negative entry rejects
    under the default long-only declaration."""
    broken = PortfolioWeights({"LOW": 1.5, "HIGH": -0.5}, WeightState.TARGET)
    with pytest.raises(ValueError, match="negative"):
        require_valid_target(broken, _CANONICAL)


def test_frozen_validator_rejects_an_off_budget_target() -> None:
    """Independent control (validator teeth): the renormalization-bait
    vector (sum 1 − 5e-10) is INSIDE the default budget tolerance and
    must pass the strict validator — pinning that the renorm kill is
    the byte law's, not the validator's."""
    total = math.fsum((0.4285714285714286, 0.5714285709285715))
    assert abs(total - 1.0) <= 1e-9
    tolerated = PortfolioWeights(
        {"LOW": 0.4285714285714286, "HIGH": 0.5714285709285715},
        WeightState.TARGET,
    )
    assert require_valid_target(tolerated, _CANONICAL) is None


def test_frozen_context_rejects_misanchored_information() -> None:
    """Independent control: post-decision information cannot enter a
    lawful context (the fixed admission law — strategies never see
    it)."""
    with pytest.raises(InvalidChronologyError):
        _context(
            ("LOW", "HIGH"),
            decision_time=_T4,
            information=_information(_a_records(), as_of=_T5),
        )


def test_frozen_information_set_rejects_revision_pairs() -> None:
    """Independent control: revision pairs reject at InformationSet
    construction (upstream of any strategy)."""
    revised = _a_records() + (_obs("low_ret", _T1, 0.999),)
    with pytest.raises(AmbiguousObservationError):
        InformationSet(revised, as_of=_T5)


def test_frozen_information_set_admits_the_fixture_records() -> None:
    """Independent control: the fixture records are lawful inputs of
    the fixed admission law (the world fixtures are constructible)."""
    admitted = InformationSet(_a_records(), as_of=_T5)
    assert len(tuple(admitted)) == 8


def test_frozen_constraints_reject_a_non_positive_tolerance() -> None:
    """Independent control: ``WeightConstraints`` itself rejects a
    non-positive budget tolerance (the declaration surface is
    unconditional before any strategy exists)."""
    with pytest.raises(ValueError, match="budget_tolerance"):
        WeightConstraints(budget_tolerance=0.0)
