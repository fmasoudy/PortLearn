"""The mean-variance allocation strategy contract (test-first).

This module defines the fixed behavior floors for the fourth built-in
classical strategy, ``portlearn.strategies.MeanVariance``, exactly as
specified in the documented design for this strategy.

Contract discipline, per the fixed section 8 preamble and its rev2
decision-completeness directive: every floor T1-T27 - including the
inherited-law floors T19-T26 - is an explicit test in THIS file; the
minimum-variance suite is fixed regression protection, never substitute
coverage for this strategy.

* every expected value is independently computed or hand-pinned from
  the section 14 design-preflight references (exact ``Fraction`` algebra
  or the pinned probe transcripts) - never derived from the
  implementation's own arithmetic;
* every floor exercises a public path only (public import, public
  constructor, public ``decide``, public contracts), with the
  three narrowly sanctioned instrumentation seams of the house
  pattern: the solver seam (a fake ``scipy.optimize.minimize``
  injected through ``sys.modules``), the estimator boundary (the
  wrapper-to-engine call instrumented through pytest's auto-undone
  ``monkeypatch``), and the public validator attribute seam;
* the class is absent at authoring time, so every target-dependent
  node resolves it lazily inside the test body and fails with an
  ``EXPECTED_REJECTION``-prefixed causal message instead of a module-level
  collection error - while nodes anchored purely on fixed live
  surfaces act as always-green controls.
"""

from __future__ import annotations

import importlib
import math
import struct
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from fractions import Fraction
from types import SimpleNamespace

import pytest

from portlearn.interfaces import (
    DecisionContext,
    Forecast,
    InformationSet,
)
from portlearn.observations import TimedObservation
from portlearn.timing import InvalidChronologyError
from portlearn.weights import (
    PortfolioWeights,
    WeightConstraints,
    WeightState,
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


def _require_mean_variance():
    """Resolve the public ``MeanVariance`` class or fail causally."""
    module = _require_strategies_module()
    try:
        return module.MeanVariance
    except AttributeError as exc:
        pytest.fail(
            "EXPECTED_REJECTION absent public class 'MeanVariance' on module "
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


def _require_mean_variance_engine():
    """Import ``portlearn._allocation`` and resolve the mean-variance engine entry
    ``solve_long_only_mean_variance`` (fail causally when absent)."""
    try:
        module = importlib.import_module("portlearn._allocation")
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"EXPECTED_REJECTION absent internal engine module 'portlearn._allocation': {exc}",
            pytrace=False,
        )
    try:
        return module, module.solve_long_only_mean_variance
    except AttributeError as exc:
        pytest.fail(
            "EXPECTED_REJECTION absent mean-variance engine function "
            f"'solve_long_only_mean_variance': {exc}",
            pytrace=False,
        )


def _require_frozen_m33_engine():
    """Import ``portlearn._allocation`` and resolve the fixed minimum-variance entry
    ``solve_long_only_min_variance`` (an always-green control surface)."""
    module = importlib.import_module("portlearn._allocation")
    try:
        return module, module.solve_long_only_min_variance
    except AttributeError as exc:
        pytest.fail(
            "EXPECTED_REJECTION the fixed minimum-variance engine entry "
            f"'solve_long_only_min_variance' disappeared: {exc}",
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
# Solver seam instrumentation (the sanctioned solver boundary)                 #
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


def _double_bytes(value: float) -> bytes:
    """The IEEE-754 little-endian double bytes of ``value`` (the byte
    law's comparison primitive)."""
    return struct.pack("<d", value)


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

#: A post-decision instant (forecast chronology laws, T10).
_T6 = datetime(2026, 6, 30, tzinfo=UTC)

_GRID = (_T1, _T2, _T3, _T4)

#: The pinned interior fixture panel (window=4): LOW alternates
#: +/- sqrt(6) over T1,T2 and rests; HIGH rests then alternates
#: +/- sqrt(1.5); MID alternates +/- sqrt(0.75) at a half-period
#: offset.  With ddof=1 the pinned covariance is exactly diagonal,
#: S11 = 2*(sqrt(6))^2/3 = 4, S22 = 2*(sqrt(1.5))^2/3 = 1,
#: S33 = 4*(sqrt(0.75))^2/3 = 1, every off-diagonal an exact zero
#: (the deviation products cancel in disjoint sign patterns), so the
#: panel reproduces the section 14 reference Sigma = ((4, 1), (1, 2))'s
#: diagonal diag(4, 1) for the LOW/HIGH pair and diag(4, 1, 1) for
#: the boundary triple, with float noise of order 1e-16 only.
_SQ6 = math.sqrt(6.0)
_SQ15 = math.sqrt(1.5)
_SQ75 = math.sqrt(0.75)
_A_LOW = (_SQ6, -_SQ6, 0.0, 0.0)
_A_HIGH = (0.0, 0.0, _SQ15, -_SQ15)
_A_MID = (_SQ75, _SQ75, -_SQ75, -_SQ75)

#: The exact mathematical references of the pinned fixtures (section 14,
#: Fraction-exact): the interior two-asset optimum of
#: (sigma2, mu, lam) = ((4, 1), (1, 2), 2) is exactly (1/10, 9/10)
#: and the boundary three-asset optimum of
#: ((4, 1, 1), (1, 2, -1/2), 2) is exactly (1/10, 9/10, 0).
_EXACT_INTERIOR_ORACLE = (Fraction(1, 10), Fraction(9, 10))
_EXACT_BOUNDARY_ORACLE = (Fraction(1, 10), Fraction(9, 10), Fraction(0))

#: The named TEST-REFERENCE tolerance: the numerical solve (SLSQP on the
#: centered program) is compared to the exact reference only within this
#: explicitly named test-time constant - it is NOT a runtime
#: acceptance tolerance (runtime acceptance is section 5.4 rules 1-4
#: alone; T1's discriminating requirement).
_TEST_ORACLE_TOL = 1e-9

#: The pinned mutation optima (section 8 matrix, T4-T9; each verified
#: independently by exact Fraction active-set algebra): every value is
#: the optimum the WRONG program would produce, embedded so each
#: floor can assert the true solve lands near the reference and far from
#: the mutation - the executable form of the kill.
_MU_SWAP_ORACLE = (0.3, 0.7)  # T4: mu permuted (2, 1)
_SIG_SWAP_ORACLE = (0.7, 0.3)  # T5: Sigma rows/cols permuted diag(1, 4)
_MU_NEG_ORACLE = (0.3, 0.7)  # T6: mu negated (-1, -2)
_NO_HALF_ORACLE = (0.15, 0.85)  # T7: 1/2 dropped == effective lam 4
_LAM_LIN_ORACLE = (0.0, 1.0)  # T8: lam on the linear term
_LAM4_ORACLE = (0.15, 0.85)  # T9: the live lam channel at lam = 4

#: The T13 pinned no-repair stub: genuinely slightly non-unit
#: (math.fsum - 1 = 2^-40 + rounding ~ 9.09e-13 != 0, ~1000x inside
#: the default budget_tolerance 1e-9), budget-clean, KKT-clean and
#: domination-clean on the interior fixture - byte identity is the
#: only discriminator (probe P24a).
_STUB_NON_UNIT = (0.1, 0.9 + 2.0**-40)

#: The near-optimal dominated vectors of T11/T12: [0.101, 0.899]
#: (probe P14: dominated at slack ~9.5e-7 and KKT-rejected; at
#: c = 1e9 the raw tolerance law would accept it, the centered law
#: rejects it) and the feasible suboptimal [0.15, 0.85] (probe P13:
#: worst KKT residual 5e-1 vs tolerance ~1.2e-8).
_NEAR_OPTIMAL = (0.101, 0.899)
_SUBOPTIMAL = (0.15, 0.85)

#: The forecast-ignored GMV fallback optimum (probe P12): a silent
#: fallback to the minimum-variance program on the interior fixture
#: would return exactly (0.2, 0.8) - far from the mean-variance
#: reference (1/10, 9/10).
_GMV_ORACLE = (0.2, 0.8)

#: Fixture C (the rank mimic): MIMIC = LOW + HIGH + 1/512 over the
#: grid, so the demeaned deviation matrix has exact rank 2 < N = 3
#: while satisfying window >= N + 1 (4 >= 4) - only the exact-rank
#: reference can catch it (probe P20).
_Q = 1.0 / 512.0
_C_MIMIC = tuple(lv + hv + _Q for lv, hv in zip(_A_LOW, _A_HIGH, strict=True))

#: The asset-to-return-series mappings of the fixtures.
_MAP_2 = {"LOW": "low_ret", "HIGH": "high_ret"}
_MAP_3 = {"LOW": "low_ret", "HIGH": "high_ret", "MID": "mid_ret"}
_MAP_C = {"LOW": "low_ret", "HIGH": "high_ret", "MIMIC": "mimic_ret"}

#: The canonical supported constraint declaration: long-only, fully
#: invested - and nothing else.
_CANONICAL = WeightConstraints()

#: The pinned risk aversion of the reference fixtures.
_LAMBDA = 2.0


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


def _a_records() -> tuple[TimedObservation, ...]:
    """The interior fixture panel (LOW/HIGH) over the grid T1..T4."""
    records = _series_records("low_ret", _A_LOW, _GRID)
    records += _series_records("high_ret", _A_HIGH, _GRID)
    return records


def _b_records() -> tuple[TimedObservation, ...]:
    """The boundary fixture panel (LOW/HIGH/MID) over T1..T4."""
    records = _series_records("low_ret", _A_LOW, _GRID)
    records += _series_records("high_ret", _A_HIGH, _GRID)
    records += _series_records("mid_ret", _A_MID, _GRID)
    return records


def _c_records() -> tuple[TimedObservation, ...]:
    """The rank-mimic fixture panel over the grid T1..T4."""
    records = _series_records("low_ret", _A_LOW, _GRID)
    records += _series_records("high_ret", _A_HIGH, _GRID)
    records += _series_records("mimic_ret", _C_MIMIC, _GRID)
    return records


def _information(records, as_of=None):
    """The admitted information set at ``as_of`` (default the decision)."""
    return InformationSet(tuple(records), as_of=_T5 if as_of is None else as_of)


def _forecast(values, target="expected_return", decision_time=None):
    """A lawful ``Forecast`` over ``values`` at the decision instant."""
    return Forecast(
        values=dict(values),
        target=target,
        decision_time=_T5 if decision_time is None else decision_time,
        produced_by="m34-contract",
    )


#: The pinned reference forecasts: mu = (1, 2) on LOW/HIGH and
#: mu = (1, 2, -1/2) on LOW/HIGH/MID (section 14).
_F_2 = {"LOW": 1.0, "HIGH": 2.0}
_F_3 = {"LOW": 1.0, "HIGH": 2.0, "MID": -0.5}


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


def _mv_context(universe, **kwargs):
    """A lawful context whose forecast is the pinned reference mu vector
    over ``universe`` (the mean-variance default: G5 satisfied)."""
    forecast = kwargs.pop("forecast", None)
    if forecast is None:
        values = _F_2 if set(universe) == {"LOW", "HIGH"} else _F_3
        forecast = _forecast(values)
    return _context(universe, forecast=forecast, **kwargs)


def _target_of(strategy, context):
    """Resolve the decision produced by ``strategy`` under ``context``
    so cross-context comparisons are plain equality on the decision
    (with ``.weights`` available for dict-compare exact equality)."""
    return strategy.decide(context).decision.target_weights


# --------------------------------------------------------------------------- #
# Estimand references: exact Fraction mathematics vs the numerical solve (T1)     #
# --------------------------------------------------------------------------- #


def _solve_weights(strategy, universe, information=None, forecast=None):
    """Run a real-scipy decide and return the plain weight dict.

    Raw record tuples are wrapped into an admitted ``InformationSet``,
    and kwargs omitted when ``None`` let the lawful defaults inject
    (``_mv_context`` supplies the pinned reference forecast, ``_context``
    the default admitted information), so every caller hands the
    strategy a lawful mean-variance context (G5).
    """
    kwargs = {}
    if information is not None:
        kwargs["information"] = _information(information)
    if forecast is not None:
        kwargs["forecast"] = forecast
    return dict(_target_of(strategy, _mv_context(universe, **kwargs)).weights)


def test_interior_oracle_is_hit_within_the_named_test_tolerance() -> None:
    """T1: the mathematical reference of the interior fixture
    ((sigma^2, mu, lam) = ((4, 1), (1, 2), 2)) is EXACTLY (1/10, 9/10)
    (Fraction algebra); the numerical solve must land within the named
    TEST-REFERENCE tolerance 1e-9 - a test-time comparison constant, never
    a runtime acceptance tolerance (runtime acceptance is section 5.4
    rules 1-4 alone)."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    weights = _solve_weights(strategy, ("LOW", "HIGH"), information=_a_records())
    assert weights["LOW"] == pytest.approx(
        float(_EXACT_INTERIOR_ORACLE[0]), abs=_TEST_ORACLE_TOL
    )
    assert weights["HIGH"] == pytest.approx(
        float(_EXACT_INTERIOR_ORACLE[1]), abs=_TEST_ORACLE_TOL
    )
    assert math.fsum(weights.values()) == pytest.approx(1.0, abs=_TEST_ORACLE_TOL)
    assert all(w >= 0.0 for w in weights.values())


def test_boundary_oracle_is_hit_within_the_named_test_tolerance() -> None:
    """T1: the boundary fixture
    ((4, 1, 1), (1, 2, -1/2), 2) has the EXACT optimum (1/10, 9/10, 0)
    - asset MID sits exactly at the boundary - and the numerical solve
    lands within 1e-9 of it."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_3), 4, _LAMBDA)
    weights = _solve_weights(strategy, ("LOW", "HIGH", "MID"), information=_b_records())
    reference = _EXACT_BOUNDARY_ORACLE
    assert weights["LOW"] == pytest.approx(float(reference[0]), abs=_TEST_ORACLE_TOL)
    assert weights["HIGH"] == pytest.approx(float(reference[1]), abs=_TEST_ORACLE_TOL)
    assert weights["MID"] == pytest.approx(float(reference[2]), abs=_TEST_ORACLE_TOL)
    assert math.fsum(weights.values()) == pytest.approx(1.0, abs=_TEST_ORACLE_TOL)
    assert all(w >= 0.0 for w in weights.values())


def test_solve_is_far_from_every_objective_mutation_optimum() -> None:
    """T7/T8 discriminators: the wrong programs - the dropped-1/2
    objective (effective lam doubling, optimum (0.15, 0.85)) and the
    lambda-on-the-linear-term objective (optimum (0.0, 1.0)) - produce
    optima more than 0.02 away from the true solve at lam = 2, so the
    true solve's proximity to (1/10, 9/10) kills both assemblies."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    weights = _solve_weights(strategy, ("LOW", "HIGH"), information=_a_records())
    for mutation in (_NO_HALF_ORACLE, _LAM_LIN_ORACLE):
        distance = max(
            abs(weights["LOW"] - mutation[0]), abs(weights["HIGH"] - mutation[1])
        )
        assert distance > 0.02, (
            f"the solve must discriminate the mutation optimum {mutation}"
        )


def test_mu_order_permutation_moves_the_optimum() -> None:
    """T4: swapping the forecast entries between the two assets moves
    the optimum from (0.1, 0.9) to the independently verified (0.3,
    0.7) - mu is assembled in universe order, so a permuted forecast
    is a materially different program (kill: mis-association)."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    swapped = _forecast({"LOW": 2.0, "HIGH": 1.0})
    weights = _solve_weights(
        strategy, ("LOW", "HIGH"), information=_a_records(), forecast=swapped
    )
    assert weights["LOW"] == pytest.approx(_MU_SWAP_ORACLE[0], abs=_TEST_ORACLE_TOL)
    assert weights["HIGH"] == pytest.approx(_MU_SWAP_ORACLE[1], abs=_TEST_ORACLE_TOL)
    assert abs(weights["LOW"] - 0.1) > 0.02


def test_sigma_order_permutation_moves_the_optimum() -> None:
    """T5: swapping the SERIES mapping (not mu) swaps the Sigma
    rows/columns with respect to universe order - diag(1, 4) - and
    moves the optimum to (0.7, 0.3): the mirror defect of T4."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance({"LOW": "high_ret", "HIGH": "low_ret"}, 4, _LAMBDA)
    weights = _solve_weights(strategy, ("LOW", "HIGH"), information=_a_records())
    assert weights["LOW"] == pytest.approx(_SIG_SWAP_ORACLE[0], abs=_TEST_ORACLE_TOL)
    assert weights["HIGH"] == pytest.approx(_SIG_SWAP_ORACLE[1], abs=_TEST_ORACLE_TOL)


def test_mu_sign_flip_moves_the_optimum() -> None:
    """T6: negating mu moves the optimum to (0.3, 0.7) - a +mu^T w
    objective or a sign error in mu assembly lands on the mirrored
    allocation and is killed."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    negated = _forecast({"LOW": -1.0, "HIGH": -2.0})
    weights = _solve_weights(
        strategy, ("LOW", "HIGH"), information=_a_records(), forecast=negated
    )
    assert weights["LOW"] == pytest.approx(_MU_NEG_ORACLE[0], abs=_TEST_ORACLE_TOL)
    assert weights["HIGH"] == pytest.approx(_MU_NEG_ORACLE[1], abs=_TEST_ORACLE_TOL)


def test_lambda_channel_is_live_at_four() -> None:
    """T9: lam = 4 moves the interior optimum to (0.15, 0.85) - the
    risk-aversion channel is live (a minimum-variance solve in costume, an
    ignored lam, or a double-counted lam all miss this pin)."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, 4.0)
    weights = _solve_weights(strategy, ("LOW", "HIGH"), information=_a_records())
    assert weights["LOW"] == pytest.approx(_LAM4_ORACLE[0], abs=_TEST_ORACLE_TOL)
    assert weights["HIGH"] == pytest.approx(_LAM4_ORACLE[1], abs=_TEST_ORACLE_TOL)


# --------------------------------------------------------------------------- #
# Translation invariance (T27, rev2 G9a)                                       #
# --------------------------------------------------------------------------- #


def test_unit_shift_produces_byte_identical_decisions() -> None:
    """T27: with the exactly representable shift c = 1e9, mu and
    mu + c*1 produce identical centered vectors and hence
    identical solver inputs - the two returned decisions must be
    identical in-process (IEEE-754 exact equality per asset),
    while each lands within the named test tolerance of the exact
    reference (1/10, 9/10)."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    plain = _solve_weights(strategy, ("LOW", "HIGH"), information=_a_records())
    shifted = _solve_weights(
        strategy,
        ("LOW", "HIGH"),
        information=_a_records(),
        forecast=_forecast({"LOW": 1e9 + 1.0, "HIGH": 1e9 + 2.0}),
    )
    assert set(plain) == set(shifted) == {"LOW", "HIGH"}
    for asset in ("LOW", "HIGH"):
        assert _double_bytes(plain[asset]) == _double_bytes(shifted[asset])
    assert plain["LOW"] == pytest.approx(
        float(_EXACT_INTERIOR_ORACLE[0]), abs=_TEST_ORACLE_TOL
    )
    assert shifted["LOW"] == pytest.approx(
        float(_EXACT_INTERIOR_ORACLE[0]), abs=_TEST_ORACLE_TOL
    )


def test_rejections_are_identical_at_c_zero_and_c_one_billion() -> None:
    """T27: the near-optimal [0.101, 0.899] and the suboptimal
    [0.15, 0.85] stub vectors are rejected at BOTH c = 0 and
    c = 1e9 with the same accept/reject outcome - the centered
    tolerance law is translation-invariant (the raw law's tolerance
    would inflate to 1e+01 at c = 1e9 and accept [0.101, 0.899])."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    shifted = _forecast({"LOW": 1e9 + 1.0, "HIGH": 1e9 + 2.0})
    for vector in (_NEAR_OPTIMAL, _SUBOPTIMAL):

        def stub(fun, x0, vector=vector, **kwargs):
            return _fake_result(list(vector))

        with _scipy_fake(stub), pytest.raises(ValueError):
            strategy.decide(_mv_context(("LOW", "HIGH")))
        with _scipy_fake(stub), pytest.raises(ValueError):
            strategy.decide(_mv_context(("LOW", "HIGH"), forecast=shifted))


def test_optimum_stub_acceptance_is_identical_at_both_scales() -> None:
    """T27: the exact-reference stub [0.1, 0.9] is accepted exactly
    at c = 0 and at c = 1e9 alike (identical acceptance outcomes)."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    shifted = _forecast({"LOW": 1e9 + 1.0, "HIGH": 1e9 + 2.0})

    def reference_stub(fun, x0, **kwargs):
        return _fake_result([0.1, 0.9])

    with _scipy_fake(reference_stub):
        plain = dict(_target_of(strategy, _mv_context(("LOW", "HIGH"))).weights)
    with _scipy_fake(reference_stub):
        moved = dict(
            _target_of(strategy, _mv_context(("LOW", "HIGH"), forecast=shifted)).weights
        )
    assert plain == moved
    assert _double_bytes(plain["LOW"]) == _double_bytes(0.1)
    assert _double_bytes(plain["HIGH"]) == _double_bytes(0.9)


def test_centering_fails_closed_on_mu_bar_overflow() -> None:
    """T27/F9a: a finite raw mu whose compensated sum overflows
    (math.fsum raises OverflowError on 1.8e308 + 1.7e308) rejects
    ValueError BEFORE any solve - never an escaping OverflowError,
    never an inf entering the program."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    overflowing = _forecast({"LOW": 1.8e308, "HIGH": 1.7e308})
    with pytest.raises(ValueError, match="finite|overflow|mean"):
        strategy.decide(_mv_context(("LOW", "HIGH"), forecast=overflowing))


def test_centering_fails_closed_on_nonfinite_mu_tilde() -> None:
    """T27/F9a: every raw entry finite and mu_bar finite, yet one
    centered entry mu_i - mu_bar overflows to -inf (1.7e308 and
    1.6e308 against -1.7e308 with mu_bar = 1.6e308/3): rejects
    ValueError before any solve."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_3), 4, _LAMBDA)
    overflowing = _forecast({"LOW": 1.7e308, "HIGH": 1.6e308, "MID": -1.7e308})
    with pytest.raises(ValueError, match="finite|overflow|center"):
        strategy.decide(
            _context(
                ("LOW", "HIGH", "MID"),
                information=_information(_b_records()),
                forecast=overflowing,
            )
        )


# --------------------------------------------------------------------------- #
# Forecast check (F9, T10) and the G9a centering law (T11)                      #
# --------------------------------------------------------------------------- #


def test_missing_forecast_rejects() -> None:
    """F9/T10: with no forecast handed to the context, decide rejects
    ValueError naming the estimand - the mu leg is a first-class input,
    never fabricated (no shrinkage fallback, no zero-mu fallback)."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    #: ``_mv_context`` would inject the pinned reference forecast, so the
    #: missing-forecast path must be exercised through ``_context``
    #: with an explicit ``forecast=None``.
    with pytest.raises(ValueError, match="forecast"):
        strategy.decide(_context(("LOW", "HIGH"), forecast=None))


def test_missing_series_entry_rejects() -> None:
    """F9/T10: a forecast covering only HIGH must reject naming the
    missing asset LOW (consumption of exactly the universe keys)."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    partial = _forecast({"HIGH": 2.0})
    with pytest.raises(ValueError, match="LOW"):
        strategy.decide(_mv_context(("LOW", "HIGH"), forecast=partial))


def test_extra_forecast_key_is_inert_to_the_decision() -> None:
    """G7/T10: a forecast key outside the caller-declared universe is
    inert - consumption is keyed on exactly the universe keys, so a
    lawful forecast carrying one extra key yields the identical
    decision of the pinned reference forecast (no rejection, no drift)."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    plain = _target_of(strategy, _mv_context(("LOW", "HIGH")))
    stray = _forecast({"LOW": 1.0, "HIGH": 2.0, "OUTSIDE": 3.0})
    with_extra = _target_of(strategy, _mv_context(("LOW", "HIGH"), forecast=stray))
    assert dict(plain.weights) == dict(with_extra.weights)
    plain_bytes = [_double_bytes(w) for w in dict(plain.weights).values()]
    extra_bytes = [_double_bytes(w) for w in dict(with_extra.weights).values()]
    assert plain_bytes == extra_bytes


_ORACLE_FORECAST = {"LOW": 1.0, "HIGH": 2.0}


def _oracle_forecast():
    return _forecast(dict(_ORACLE_FORECAST))


def test_nonfinite_forecast_entry_rejects() -> None:
    """F9/T10: a NaN forecast value rejects before any solve (fail
    closed - never fabricated, never clamped)."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    for bad in (float("nan"), float("inf"), float("-inf")):
        poisoned = _forecast({"LOW": bad, "HIGH": 2.0})
        with pytest.raises(ValueError, match="finite|forecast"):
            strategy.decide(_mv_context(("LOW", "HIGH"), forecast=poisoned))


def test_bool_forecast_entry_rejects() -> None:
    """F9/T10: ``True`` is not a return forecast even though it is an
    int subclass: bool entries reject naming the type law."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    poisoned = _forecast({"LOW": True, "HIGH": 2.0})
    with pytest.raises(ValueError, match="bool|forecast"):
        strategy.decide(_mv_context(("LOW", "HIGH"), forecast=poisoned))


def test_non_string_forecast_key_rejects() -> None:
    """F9/T10: forecast keys are asset identifiers - an int key is not
    a silent hash miss: the unconditional ``Forecast`` law rejects the
    construction itself, so the poisoned forecast can never be handed
    to ``decide``."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    with pytest.raises(ValueError, match="string|identifier") as caught:
        poisoned = Forecast(
            values={7: 1.0, "HIGH": 2.0},
            target="expected_return",
            decision_time=_T5,
            produced_by="scratch-probe",
        )
        strategy.decide(_mv_context(("LOW", "HIGH"), forecast=poisoned))
    assert "int" in str(caught.value)


def test_forecast_target_must_be_the_return_estimand() -> None:
    """F9/T10: target='volatility' is a different estimand: reject
    naming the target and the required one."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    wrong = Forecast(
        values={"LOW": 0.1, "HIGH": 0.2},
        target="volatility",
        decision_time=_T5,
        produced_by="scratch-probe",
    )
    with pytest.raises(ValueError, match="volatility|expected_return|target"):
        strategy.decide(_mv_context(("LOW", "HIGH"), forecast=wrong))


def test_stale_forecast_rejects_on_decision_time_law() -> None:
    """F9/T10: a forecast produced after the decision time cannot be
    consumed (the forecast admission law, G9 pre-conditions)."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    late = Forecast(
        values={"LOW": 1.0, "HIGH": 2.0},
        target="expected_return",
        decision_time=_T6,
        produced_by="scratch-probe",
    )
    with pytest.raises((ValueError, InvalidChronologyError)):
        strategy.decide(_mv_context(("LOW", "HIGH"), forecast=late))


def test_forecast_values_are_centered_not_consumed_raw() -> None:
    """T11 (G9a): mu_tilde = mu - mu_bar * 1 with mu_bar computed by
    math.fsum(mu)/N: the CONSTANT component of the forecast is inert
    (only relative expected returns matter), pinned by the c = 1e9
    byte-identity law in the T27 block; this test floors the smaller
    statement - adding an exactly-representable constant 1.0 to every
    entry leaves the solved weights unchanged (approx-level, the byte
    law is pinned separately)."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    plain = _solve_weights(
        strategy,
        ("LOW", "HIGH"),
        information=_a_records(),
        forecast=_oracle_forecast(),
    )
    lifted = _solve_weights(
        strategy,
        ("LOW", "HIGH"),
        information=_a_records(),
        forecast=_forecast({"LOW": 2.0, "HIGH": 3.0}),
    )
    for asset in ("LOW", "HIGH"):
        assert plain[asset] == pytest.approx(lifted[asset], abs=_TEST_ORACLE_TOL)


def test_centered_mu_is_universe_ordered() -> None:
    """T11 (G9a): mu_tilde is assembled in UNIVERSE order, not
    forecast-dict iteration order - pinned by the T4/T5 permutation
    reference tests; this floor pins the wiring with a stub capture of
    the solver-visible program when the adapter seam is spliced."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    recorded = {}

    def capturing(fun, x0, **kwargs):
        recorded["fun"] = fun
        recorded["x0"] = list(x0)
        return _fake_result([0.9, 0.1])

    with _scipy_fake(capturing):
        target = _target_of(
            strategy, _mv_context(("HIGH", "LOW"), forecast=_oracle_forecast())
        )
    assert dict(target.weights)["HIGH"] == pytest.approx(0.9, abs=1e-9)
    assert dict(target.weights)["LOW"] == pytest.approx(0.1, abs=1e-9)


# --------------------------------------------------------------------------- #
# Acceptance machinery: no-repair byte laws (T13, P24a/P24b, F19/F20/F21)      #
# --------------------------------------------------------------------------- #

#: The pinned slightly non-unit fixture (T13b/G14/P24a): exactly
#: representable offset 2^-40 so the vector is genuinely non-unit
#: (fsum - 1 ~ 9.09e-13, ~1000x inside budget_tolerance 1e-9), KKT-
#: clean and domination-clean - byte identity is the only discriminator.
_NON_UNIT_STUB = (0.1, 0.9 + 2.0**-40)

#: The pinned negative-entry fixture for T13(a)/F20 byte evidence:
#: two-dimensional for the two-asset universe and exactly
#: budget-feasible (fsum == 1.0), so the only law it can break is the
#: negative-entry law itself (probe P24b).
_NEGATIVE_STUB = (1.01, -0.01)


def _returning(vector):
    """Build a stub minimize capturing the call once."""
    calls = []

    def stub(fun, x0, **kwargs):
        calls.append((fun, x0, kwargs))
        return _fake_result(list(vector))

    return stub, calls


def test_returns_valid_non_unit_vector_byte_for_byte() -> None:
    """T13(b)/G14/P24a: the pinned non-unit stub `[0.1, 0.9 + 2^-40]`
    (genuinely non-unit, ~1000x inside budget_tolerance, KKT- and
    domination-clean) is returned exactly - never renormalized:
    the returned target bytes equal the raw stub bytes and differ from
    the normalized image bytes in every component."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, 2.0)
    stub, calls = _returning(_NON_UNIT_STUB)
    with _scipy_fake(stub):
        target = _target_of(strategy, _mv_context(("LOW", "HIGH")))
    assert len(calls) == 1
    weights = dict(target.weights)
    stub_bytes = [struct.pack("<d", w) for w in _NON_UNIT_STUB]
    returned_bytes = [
        struct.pack("<d", weights["LOW"]),
        struct.pack("<d", weights["HIGH"]),
    ]
    assert returned_bytes == stub_bytes
    total = math.fsum(_NON_UNIT_STUB)
    normalized = [w / total for w in _NON_UNIT_STUB]
    norm_bytes = [struct.pack("<d", w) for w in normalized]
    assert returned_bytes != norm_bytes
    for got, rep in zip(returned_bytes, norm_bytes, strict=True):
        assert got != rep


def test_negative_entry_rejects_with_byte_evidence_of_no_repair() -> None:
    """T13(a)/F20/P17: a stubbed vector with a negative entry rejects,
    and the rejection is faithful - the vector shows no clipping and
    no renormalization: the raw stub (1.01, -0.01) sums to exactly 1
    under fsum (budget-clean), so any clip-then-renormalize artifact
    ((1.0, 0.0)) is a repaired image, never the raw bytes."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, 2.0)
    stub, calls = _returning(_NEGATIVE_STUB)
    with _scipy_fake(stub), pytest.raises(ValueError, match="negative") as caught:
        strategy.decide(_mv_context(("LOW", "HIGH")))
    assert len(calls) == 1, "no fallback solve, no retry"
    #: the raw stub sums to exactly 1.0 under fsum (budget-feasible),
    #: so the reject can only be the negative-entry law; a renormalized
    #: candidate would still sum to 1 but every repaired artifact
    #: (clip-then-renormalize) necessarily differs from the raw bytes
    #: in the negative component - pin the reject saw the raw vector.
    assert math.fsum(_NEGATIVE_STUB) == 1.0
    message = str(caught.value)
    assert "-0.01" in message or "-0.01" in repr(_NEGATIVE_STUB)


def test_off_budget_stub_vector_rejects() -> None:
    """F19/T25: a stubbed backend vector off budget beyond
    budget_tolerance (default 1e-9) rejects naming the budget law -
    never silently renormalized into acceptance."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, 2.0)
    stub, calls = _returning((0.1, 0.8))
    with _scipy_fake(stub), pytest.raises(ValueError, match="budget"):
        strategy.decide(_mv_context(("LOW", "HIGH")))
    assert len(calls) == 1


def test_nonfinite_stub_vector_rejects_before_all_acceptance() -> None:
    """F21/T25 (rev2 residual cure): a stubbed vector containing nan,
    +inf, or -inf rejects BEFORE any budget/KKT/domination evaluation
    and WITHOUT repair (no clipping of inf, no renormalization of
    nan) - the ordering floor is pinned by rejecting even when the
    budget check would pass on the raw magnitude."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, 2.0)
    for bad in (
        (float("nan"), 0.9),
        (float("inf"), 0.9),
        (0.1, float("-inf")),
    ):
        stub, calls = _returning(bad)
        with _scipy_fake(stub), pytest.raises(ValueError):
            strategy.decide(_mv_context(("LOW", "HIGH")))
        assert len(calls) == 1, f"no repair path for {bad}"


def test_absent_backend_decide_raises_unavailable() -> None:
    """F15/T24: with scipy absent, ``decide`` raises
    OptimizationUnavailableError; construction/import succeed. The
    eager-import defect is killed by the same block."""
    unavailable, _ = _require_solver_exceptions()
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, 2.0)
    assert "scipy" not in sys.modules
    with _scipy_absent(), pytest.raises(ValueError, match="scipy") as caught:
        strategy.decide(_mv_context(("LOW", "HIGH")))
    assert isinstance(caught.value, unavailable)


def test_solver_non_success_raises_failure_naming_status() -> None:
    """F16/T24: a solver result carrying a non-success status fails
    closed with OptimizationFailureError naming the status - no
    fallback solve, no retry, no GMV fallback."""
    _, failure = _require_solver_exceptions()
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, 2.0)
    calls = []

    def failing_minimize(*args, **kwargs):
        calls.append((args, kwargs))
        return _fake_result([0.5, 0.5], success=False, status=8)

    with _scipy_fake(failing_minimize), pytest.raises(ValueError, match="8") as caught:
        strategy.decide(_mv_context(("LOW", "HIGH")))
    assert isinstance(caught.value, failure)
    assert len(calls) == 1, "no fallback solve, no retry"


def test_silent_gmv_fallback_is_discriminated() -> None:
    """P12/T-estimand: the forecast must be consumed. A silent
    fallback to the minimum-variance program on the interior fixture
    would return exactly the GMV reference (0.2, 0.8) - far from the
    mean-variance reference (1/10, 9/10). A real solve pins the distance."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, 2.0)
    weights = _solve_weights(strategy, ("LOW", "HIGH"))
    for asset, expected in zip(("LOW", "HIGH"), _GMV_ORACLE, strict=True):
        assert weights[asset] != pytest.approx(expected, abs=1e-3), (
            f"a GMV fallback would sit at {expected} for {asset}"
        )
        assert weights[asset] == pytest.approx(
            dict(zip(("LOW", "HIGH"), _EXACT_INTERIOR_ORACLE))[asset], abs=1e-9
        )


# --------------------------------------------------------------------------- #
# Inherited constructor law floor (T19, C1-C8)                                 #
# --------------------------------------------------------------------------- #


def test_constructor_rejects_a_non_mapping() -> None:
    """C1/T19: a bare string - a common mistake for a mapping of one -
    rejects with ValueError naming the law and the received type."""
    mean_variance = _require_mean_variance()
    with pytest.raises(ValueError, match="mapping"):
        mean_variance("low_ret", 4, 2.0)


def test_constructor_rejects_every_non_mapping_type() -> None:
    """C1/T19: every other non-mapping type rejects the same way."""
    mean_variance = _require_mean_variance()
    for bad in (None, 0, 4.0, [("LOW", "low_ret")], ("LOW", "high_ret")):
        with pytest.raises(ValueError, match="mapping"):
            mean_variance(bad, 4, 2.0)


def test_constructor_rejects_an_empty_mapping() -> None:
    """C1/T19: an empty mapping is not a portfolio mapping."""
    mean_variance = _require_mean_variance()
    with pytest.raises(ValueError, match="mapping"):
        mean_variance({}, 4, 2.0)


def test_constructor_rejects_a_blank_asset_key() -> None:
    """C2/T19: a blank-string asset identifier rejects naming the
    offending key."""
    mean_variance = _require_mean_variance()
    with pytest.raises(ValueError) as record:
        mean_variance({"": "low_ret", "LOW": "low_ret2"}, 4, 2.0)
    assert "identifier" in str(record.value)


def test_constructor_rejects_a_blank_series_identifier() -> None:
    """C2/T19: a blank-string series identifier rejects naming the
    offending value."""
    mean_variance = _require_mean_variance()
    with pytest.raises(ValueError) as record:
        mean_variance({"LOW": "   "}, 4, 2.0)
    assert "identifier" in str(record.value)


def test_constructor_rejects_a_non_string_series_identifier() -> None:
    """C2/T19: a series identifier that is not a string at all
    rejects."""
    mean_variance = _require_mean_variance()
    with pytest.raises(ValueError) as record:
        mean_variance({"LOW": 7}, 4, 2.0)
    assert "series" in str(record.value)


def test_constructor_rejects_a_series_mapped_by_two_assets() -> None:
    """C3/T19: the one-to-one law - a series identifier mapped by two
    different assets rejects naming the series and both assets."""
    mean_variance = _require_mean_variance()
    with pytest.raises(ValueError) as record:
        mean_variance({"LOW": "low_ret", "HIGH": "low_ret"}, 4, 2.0)
    message = str(record.value)
    assert "low_ret" in message
    assert "LOW" in message
    assert "HIGH" in message


def test_constructor_rejects_a_window_below_two() -> None:
    """C4/T19: the estimation window floor - ``window < 2`` rejects
    naming the field, the value, and the floor."""
    mean_variance = _require_mean_variance()
    with pytest.raises(ValueError) as record:
        mean_variance(dict(_MAP_2), 1, 2.0)
    message = str(record.value)
    assert "window" in message
    assert "1" in message
    assert "2" in message


def test_constructor_rejects_a_non_int_window() -> None:
    """C4/T19: a non-``int`` window rejects (floats included)."""
    mean_variance = _require_mean_variance()
    with pytest.raises(ValueError, match="window"):
        mean_variance(dict(_MAP_2), 4.0, 2.0)


def test_constructor_rejects_a_bool_window() -> None:
    """C4/T19: ``bool`` is explicitly rejected even though it is an
    int subclass (no accidental ``window=True`` acceptance)."""
    mean_variance = _require_mean_variance()
    with pytest.raises(ValueError, match="window"):
        mean_variance(dict(_MAP_2), True, 2.0)


def test_constructor_rejects_a_non_constraints_object() -> None:
    """C5/T19: ``constraints`` must be a ``WeightConstraints``;
    anything else rejects naming the received type."""
    mean_variance = _require_mean_variance()
    for bad in (None, {"budget": 1.0}, 1.0):
        with pytest.raises(ValueError) as record:
            mean_variance(dict(_MAP_2), 4, 2.0, constraints=bad)
        message = str(record.value)
        assert "constraints" in message
        assert type(bad).__name__ in message


def test_constructor_rejects_a_non_real_risk_aversion() -> None:
    """C6/T19: a non-real ``risk_aversion`` (string, complex) rejects
    naming the field and the received type."""
    mean_variance = _require_mean_variance()
    for bad in ("2", 1 + 2j, None):
        with pytest.raises(ValueError) as record:
            mean_variance(dict(_MAP_2), 4, bad)
        message = str(record.value)
        assert "risk_aversion" in message
        assert type(bad).__name__ in message


def test_constructor_rejects_a_bool_risk_aversion() -> None:
    """C6/T19: ``bool`` is explicitly rejected for ``risk_aversion``
    even though it is an int subclass."""
    mean_variance = _require_mean_variance()
    with pytest.raises(ValueError) as record:
        mean_variance(dict(_MAP_2), 4, True)
    assert "risk_aversion" in str(record.value)


def test_constructor_rejects_a_non_positive_risk_aversion() -> None:
    """C6/T19: the positivity law - zero and negative lambda reject."""
    mean_variance = _require_mean_variance()
    for bad in (0.0, -2.0):
        with pytest.raises(ValueError) as record:
            mean_variance(dict(_MAP_2), 4, bad)
        assert "risk_aversion" in str(record.value)


def test_constructor_rejects_a_nonfinite_risk_aversion() -> None:
    """C6/T19: nonfinite ``risk_aversion`` (inf, nan) rejects."""
    mean_variance = _require_mean_variance()
    for bad in (float("inf"), float("nan")):
        with pytest.raises(ValueError) as record:
            mean_variance(dict(_MAP_2), 4, bad)
        assert "risk_aversion" in str(record.value)


# --------------------------------------------------------------------------- #
# Unsupported constraints (T20, F6-inherited)                                  #
# --------------------------------------------------------------------------- #


def test_decide_rejects_shorting_declaration() -> None:
    """T20: ``allow_short=True`` is outside the supported subset:
    ``decide`` fails closed naming the field, its value, and the
    subset (message law verbatim)."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, 2.0, WeightConstraints(allow_short=True))
    with pytest.raises(ValueError, match="allow_short") as caught:
        strategy.decide(_mv_context(("LOW", "HIGH")))
    assert "True" in str(caught.value)


def test_decide_rejects_partial_investment_budget() -> None:
    """T20: any ``budget`` other than the default full-investment
    level is a different estimand: abort naming
    field+value+subset."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, 2.0, WeightConstraints(budget=0.5))
    with pytest.raises(ValueError, match="budget") as caught:
        strategy.decide(_mv_context(("LOW", "HIGH")))
    assert "0.5" in str(caught.value)


def test_decide_rejects_gross_exposure_cap() -> None:
    """T20: a declared ``max_gross_exposure`` narrows the canonical
    program: abort naming field+value+subset."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(
        dict(_MAP_2), 4, 2.0, WeightConstraints(max_gross_exposure=1.2)
    )
    with pytest.raises(ValueError, match="max_gross_exposure") as caught:
        strategy.decide(_mv_context(("LOW", "HIGH")))
    assert "1.2" in str(caught.value)


def test_decide_rejects_net_exposure_cap() -> None:
    """T20: a declared ``max_abs_net_exposure`` narrows the canonical
    program: abort naming field+value+subset."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(
        dict(_MAP_2), 4, 2.0, WeightConstraints(max_abs_net_exposure=1.2)
    )
    with pytest.raises(ValueError, match="max_abs_net_exposure") as caught:
        strategy.decide(_mv_context(("LOW", "HIGH")))
    assert "1.2" in str(caught.value)


# --------------------------------------------------------------------------- #
# Universe/window/mapping coverage (T21, F2-F5 floor)                          #
# --------------------------------------------------------------------------- #


def test_decide_rejects_singleton_universe() -> None:
    """T21: a one-asset portfolio is degenerate: the support check
    fails closed naming the size."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, 2.0)
    with pytest.raises(ValueError, match="universe") as caught:
        strategy.decide(_mv_context(("LOW",)))
    assert "1" in str(caught.value)


def test_decide_rejects_empty_universe() -> None:
    """T21: zero assets is not a portfolio: the support check fails
    closed naming the size."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, 2.0)
    with pytest.raises(ValueError, match="universe"):
        strategy.decide(_mv_context(()))


def test_decide_rejects_window_below_domain_floor() -> None:
    """T21: ``window < N + 1`` cannot produce a full-column-rank
    demeaned panel: the domain check fires naming window and N."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_3), 3, 2.0)
    with pytest.raises(ValueError) as caught:
        strategy.decide(_mv_context(("LOW", "HIGH", "MID")))
    message = str(caught.value)
    assert "window" in message
    assert "3" in message
    assert "N" in message or "universe" in message or "assets" in message


def test_domain_gate_fires_before_the_solve() -> None:
    """T21: the F8 check is causal: it fires even when the optimizer
    extra is absent, proving no solve is attempted before the domain
    check."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_3), 3, 2.0)
    with _fresh_modules_context():
        import builtins

        saved_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "scipy" or name.startswith("scipy."):
                raise AssertionError(
                    "EXPECTED_REJECTION the domain check must fire before any "
                    "solver import is attempted"
                )
            return saved_import(name, *args, **kwargs)

        builtins.__import__ = guarded
        try:
            with pytest.raises(ValueError, match="window"):
                strategy.decide(_mv_context(("LOW", "HIGH", "MID")))
        finally:
            builtins.__import__ = saved_import


def test_decide_rejects_universe_asset_without_mapping_entry() -> None:
    """T21: a universe asset with no ``return_series`` entry fails the
    mapping-completeness check naming the asset - never a bare
    KeyError."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, 2.0)
    with pytest.raises(ValueError, match="OUTSIDE") as caught:
        strategy.decide(_mv_context(("LOW", "HIGH", "OUTSIDE")))
    assert "return_series" in str(caught.value)


# --------------------------------------------------------------------------- #
# Short/zero history and the balanced panel (T22, F10-F13)                     #
# --------------------------------------------------------------------------- #


def test_decide_rejects_short_history_for_a_mapped_series() -> None:
    """T22: fewer than ``window`` admissible observations for one
    mapped series fails closed naming asset, series, available count,
    and the required window."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, 2.0)
    short = _series_records("low_ret", _A_LOW[:3], _GRID[:3])
    records = short + _series_records("high_ret", _A_HIGH, _GRID)
    with pytest.raises(ValueError) as caught:
        strategy.decide(_mv_context(("LOW", "HIGH"), information=_information(records)))
    message = str(caught.value)
    assert "LOW" in message
    assert "low_ret" in message
    assert "3" in message
    assert "4" in message


def test_decide_rejects_zero_admitted_observations() -> None:
    """T22: zero admitted observations for a mapped series fails the
    same insufficient-history law."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, 2.0)
    only_high = _series_records("high_ret", _A_HIGH, _GRID)
    with pytest.raises(ValueError) as caught:
        strategy.decide(
            _mv_context(("LOW", "HIGH"), information=_information(only_high))
        )
    message = str(caught.value)
    assert "LOW" in message
    assert "low_ret" in message


def test_decide_rejects_unbalanced_panel_key_set_mismatch() -> None:
    """T22: one asset missing one timestamp inside the selected window
    fails closed naming the mismatched assets and the first divergent
    instant."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, 2.0)
    low_records = _series_records("low_ret", _A_LOW[:4], _GRID[:4])
    high_records = _series_records(
        "high_ret", _A_HIGH[:3] + (0.015,), (_T1, _T2, _T3, _T5)
    )
    records = low_records + high_records
    assert len(low_records) == len(high_records) == 4
    with pytest.raises(ValueError) as caught:
        strategy.decide(_mv_context(("LOW", "HIGH"), information=_information(records)))
    message = str(caught.value)
    assert "LOW" in message
    assert "HIGH" in message
    assert "2026-04-30" in message


def test_decide_rejects_non_finite_panel_value() -> None:
    """T22: a NaN panel entry fails the numerics check naming the
    asset, timestamp, and value - never reaching the solver."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, 2.0)
    records = _series_records("low_ret", (float("nan"), 0.0, 0.0, 0.0), _GRID)
    records += _series_records("high_ret", _A_HIGH, _GRID)
    with pytest.raises(ValueError) as caught:
        strategy.decide(_mv_context(("LOW", "HIGH"), information=_information(records)))
    message = str(caught.value)
    assert "LOW" in message
    assert "2026-01-31" in message


# --------------------------------------------------------------------------- #
# Rank reference (T23, F14)                                                       #
# --------------------------------------------------------------------------- #


def test_decide_rejects_rank_deficient_panel() -> None:
    """T23: a panel whose demeaned deviation matrix has exact rank
    2 < N = 3 fails closed naming the computed rank and N - the
    C-fixture satisfies window >= N + 1, so only the rank reference can
    catch it."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_C), 4, 2.0)

    def _never_called(fun, x0, **kwargs):
        raise AssertionError("EXPECTED_REJECTION the rank check must fire before any solve")

    with _scipy_fake(_never_called), pytest.raises(ValueError) as caught:
        strategy.decide(
            _mv_context(
                ("LOW", "HIGH", "MIMIC"),
                information=_information(_c_records()),
                # a complete expected_return forecast over the mimic
                # universe (the pinned _F_3 third-asset value) so the
                # G5-G9a forecast checks pass and only the rank check
                # can fire.
                forecast=_forecast({"LOW": 1.0, "HIGH": 2.0, "MIMIC": -0.5}),
            )
        )
    message = str(caught.value)
    assert "rank" in message
    assert "2" in message
    assert "3" in message


def test_control_same_shape_without_the_mimic_solves() -> None:
    """T23 control: dropping MIMIC from the same universe (N=2, same
    window) leaves a full-rank panel that solves to the interior
    reference - proving F14 is the mimic's exact linear dependence, not
    the fixture shape."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, 2.0)
    weights = _solve_weights(strategy, ("LOW", "HIGH"), information=_c_records())
    assert weights["LOW"] == pytest.approx(_EXACT_INTERIOR_ORACLE[0], abs=1e-9)
    assert weights["HIGH"] == pytest.approx(_EXACT_INTERIOR_ORACLE[1], abs=1e-9)


# --------------------------------------------------------------------------- #
# Public-path routing and decision-completeness (T26)                          #
# --------------------------------------------------------------------------- #


def test_public_route_and_four_entry_all() -> None:
    """T26: ``from portlearn.strategies import MeanVariance`` resolves;
    ``strategies.__all__`` equals exactly the four-entry list
    (declaration order, list equality); the root namespace does not
    grow; the lazy facade holds with scipy absent."""
    with _fresh_modules_context():
        from portlearn.strategies import MeanVariance

        module = _require_strategies_module()
        assert module.__all__ == [
            "EqualWeight",
            "InverseVolatility",
            "MinimumVariance",
            "MeanVariance",
        ]
        assert module.MeanVariance is MeanVariance
        assert isinstance(MeanVariance, type)


def test_root_all_is_unchanged_by_the_fourth_builtin() -> None:
    """T26/T17: the root package surface is not expanded by the
    fourth strategy: ``portlearn.__all__`` stays
    ``["__version__", "data", "weights"]``."""
    with _fresh_modules_context():
        root = importlib.import_module("portlearn")
        assert root.__all__ == ["__version__", "data", "weights"]


def test_root_facade_resolves_the_class_identically() -> None:
    """T26: ``pl.strategies is portlearn.strategies`` (lazy facade
    identity) and the facade route yields the identical class
    object."""
    with _fresh_modules_context():
        import portlearn as pl

        module = _require_strategies_module()
        assert pl.strategies is module
        assert pl.strategies.MeanVariance is module.MeanVariance


def test_lazy_facade_holds_with_scipy_absent() -> None:
    """T26: the lazy strategies facade survives a scipy-absent
    process: the attribute route still resolves the class without
    importing the optimizer extra (import-time scipy coupling is the
    killed defect)."""
    with _fresh_modules_context(), _scipy_absent():
        import portlearn as pl

        assert isinstance(pl.strategies.MeanVariance, type)
        #: inside the block, before either context manager restores
        #: the prior registry - the assertion is about THIS process
        #: state, not the restored one.
        assert "scipy" not in sys.modules


# --------------------------------------------------------------------------- #
# Determinism and state independence (T10/T14)                                 #
# --------------------------------------------------------------------------- #


def test_replay_is_byte_identical_in_process() -> None:
    """T10: identical (return_series, window, risk_aversion,
    constraints, information, universe, forecast) replayed in-process
    yields identical weights (no hidden state, no RNG, no warm
    start)."""
    mean_variance = _require_mean_variance()
    first = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    second = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    target_one = _target_of(first, _mv_context(("LOW", "HIGH")))
    target_two = _target_of(second, _mv_context(("LOW", "HIGH")))
    assert dict(target_one.weights) == dict(target_two.weights)
    one = [_double_bytes(w) for w in dict(target_one.weights).values()]
    two = [_double_bytes(w) for w in dict(target_two.weights).values()]
    assert one == two


def test_perturbed_holdings_leave_the_target_byte_identical() -> None:
    """T14: perturbed ``current_weights`` and non-None
    ``strategy_state`` inputs leave the target identical (no warm
    start from the book, no state leakage)."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    plain = _target_of(strategy, _mv_context(("LOW", "HIGH")))
    perturbed_book = PortfolioWeights({"CASH": 0.6, "LOW": 0.4}, WeightState.PRE_TRADE)
    with_book = _target_of(
        strategy,
        _mv_context(
            ("LOW", "HIGH"),
            current_weights=perturbed_book,
        ),
    )
    assert dict(plain.weights) == dict(with_book.weights)
    with_state = _target_of(
        strategy,
        _mv_context(("LOW", "HIGH"), strategy_state={"marker": 1}),
    )
    assert dict(plain.weights) == dict(with_state.weights)
    plain_bytes = [_double_bytes(w) for w in dict(plain.weights).values()]
    state_bytes = [_double_bytes(w) for w in dict(with_state.weights).values()]
    assert plain_bytes == state_bytes


# --------------------------------------------------------------------------- #
# Inherited constructor snapshot law (T19, C7/C8)                              #
# --------------------------------------------------------------------------- #


def test_constructor_snapshots_the_mapping_immutably() -> None:
    """T19/C7: construction-time snapshot - later mutation of a
    caller-supplied dict cannot change the exposed ``return_series``
    view or any future decision."""
    mean_variance = _require_mean_variance()
    source = {"LOW": "low_ret", "HIGH": "high_ret"}
    strategy = mean_variance(source, 4, _LAMBDA)
    source["LOW"] = "tampered_ret"
    source["X"] = "x_ret"
    view = strategy.return_series
    assert dict(view) == {"LOW": "low_ret", "HIGH": "high_ret"}
    with pytest.raises(TypeError):
        view["LOW"] = "tampered_ret"  # type: ignore[index]


def test_constructor_exposes_the_four_fields_read_only() -> None:
    """T19/C8: the public semantic state is exactly ``return_series``,
    ``window``, ``risk_aversion``, and ``constraints``, each exposed
    through a read-only property (assignment rejects)."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    assert dict(strategy.return_series) == dict(_MAP_2)
    assert strategy.window == 4
    assert strategy.risk_aversion == _LAMBDA
    assert strategy.constraints == _CANONICAL
    for field in ("return_series", "window", "risk_aversion", "constraints"):
        with pytest.raises(AttributeError):
            setattr(strategy, field, None)


def test_construction_succeeds_with_scipy_absent() -> None:
    """T19 (F15 support): construction is solver-independent -
    ``MeanVariance(...)`` constructs with scipy absent from the
    process (the failure belongs to ``decide`` alone)."""
    assert "scipy" not in sys.modules
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    assert strategy.window == 4


# --------------------------------------------------------------------------- #
# Public-path routing and decision-completeness (T26)                         #
# --------------------------------------------------------------------------- #


def test_public_import_and_all_route() -> None:
    """T26: ``from portlearn.strategies import MeanVariance`` resolves;
    ``strategies.__all__`` is exactly the four-entry list in
    declaration order; the root namespace does not grow; the lazy
    facade holds with scipy absent."""
    with _fresh_modules_context():
        from portlearn.strategies import MeanVariance

        module = _require_strategies_module()
        assert module.__all__ == [
            "EqualWeight",
            "InverseVolatility",
            "MinimumVariance",
            "MeanVariance",
        ]
        assert module.MeanVariance is MeanVariance
        assert isinstance(MeanVariance, type)
    import portlearn as pl

    assert pl.__all__ == ["__version__", "data", "weights"]
    assert not hasattr(pl, "MeanVariance")


def test_root_facade_is_lazy_under_scipy_absence() -> None:
    """T26: ``pl.strategies`` resolves the class without importing
    scipy - import-time optimizer coupling is the killed defect."""
    with _fresh_modules_context(), _scipy_absent():
        import portlearn as pl

        assert isinstance(pl.strategies.MeanVariance, type)
        #: inside the block, before either context manager restores
        #: the prior registry - the assertion is about THIS process
        #: state, not the restored one.
        assert "scipy" not in sys.modules


def test_strategy_entry_routes_the_m34_engine_by_module_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T26(a): the public ``MeanVariance.decide`` calls
    ``portlearn._allocation.solve_long_only_mean_variance`` BY MODULE
    ATTRIBUTE AT DECISION TIME - not bound at import or class
    definition time.  Splicing the module attribute after the class
    exists intercepts the call (import-time binding would bypass the
    splice and hit the real engine)."""
    mean_variance = _require_mean_variance()
    engine_module, _ = _require_mean_variance_engine()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    calls = []

    def splice(covariance, expected_returns, risk_aversion, identifiers):
        calls.append(
            {
                "covariance": [list(row) for row in covariance],
                "expected_returns": list(expected_returns),
                "risk_aversion": risk_aversion,
                "identifiers": list(identifiers),
            }
        )
        raise _SpliceSentinel()

    monkeypatch.setattr(engine_module, "solve_long_only_mean_variance", splice)
    with pytest.raises(_SpliceSentinel):
        strategy.decide(_mv_context(("LOW", "HIGH")))
    assert len(calls) == 1, "exactly one engine call, module-attribute routed"
    got = calls[0]
    assert got["identifiers"] == ["LOW", "HIGH"]
    assert got["risk_aversion"] == _LAMBDA
    #: the G9a centered representation: mean of (1, 2) is 1.5, so the
    #: centered vector handed downstream is exactly (-0.5, 0.5).
    assert got["expected_returns"] == pytest.approx([-0.5, 0.5], abs=1e-12)
    assert got["covariance"][0][0] == pytest.approx(4.0, abs=1e-9)
    assert got["covariance"][1][1] == pytest.approx(1.0, abs=1e-9)
    assert got["covariance"][0][1] == pytest.approx(0.0, abs=1e-9)


class _SpliceSentinel(Exception):
    """Marks a successful engine-splice interception."""


def test_adapter_entry_validates_before_the_lazy_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T26(b): ``solve_long_only_mean_variance`` validates its inputs
    BEFORE the lazy scipy import - an invalid input raises ValueError
    with scipy absent from the process."""
    _, engine = _require_mean_variance_engine()
    with _scipy_absent():
        asymmetric = [[4.0, 1.5], [1.0, 1.0]]
        with pytest.raises(ValueError, match="symmetr|transpose"):
            engine(
                asymmetric,
                [1.0, 2.0],
                _LAMBDA,
                ["LOW", "HIGH"],
            )
    #: control: the same call with scipy present succeeds (validation
    #: passes, the lazy import runs, no repair) - proving the raise
    #: above is the validator, not the absence itself.
    result = engine([[4.0, 0.0], [0.0, 1.0]], [1.0, 2.0], _LAMBDA, ["LOW", "HIGH"])
    #: the engine returns the adapter's ``ScipySolveResult``: compare
    #: the ordered weights-mapping values in identifier order, never
    #: iterate the result object itself.
    assert isinstance(result, _require_adapter_module().ScipySolveResult)
    assert list(result.weights.values()) == pytest.approx(
        _EXACT_INTERIOR_ORACLE, abs=_TEST_ORACLE_TOL
    )


def test_decision_path_avoids_both_frozen_m33_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T26(c): the decision path routes through NEITHER fixed
    minimum-variance entry - spiking ``solve_long_only_min_variance`` and the
    adapter's ``solve`` with sentinel raisers leaves a real decide
    untouched (that path would drop the mu/lambda terms)."""
    mean_variance = _require_mean_variance()
    engine_module, _ = _require_mean_variance_engine()
    adapter_module = _require_adapter_module()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)

    def refuse(*args, **kwargs):
        raise AssertionError(
            "EXPECTED_REJECTION the decision path must not route "
            "through the fixed minimum-variance entries"
        )

    monkeypatch.setattr(engine_module, "solve_long_only_min_variance", refuse)
    monkeypatch.setattr(adapter_module, "solve", refuse)
    target = _target_of(strategy, _mv_context(("LOW", "HIGH")))
    weights = dict(target.weights)
    assert weights["LOW"] == pytest.approx(
        _EXACT_INTERIOR_ORACLE[0], abs=_TEST_ORACLE_TOL
    )
    assert weights["HIGH"] == pytest.approx(
        _EXACT_INTERIOR_ORACLE[1], abs=_TEST_ORACLE_TOL
    )


# --------------------------------------------------------------------------- #
# Translation invariance (T27, G9a rev2)                                       #
# --------------------------------------------------------------------------- #


def test_c_equals_1e9_yields_byte_identical_decision_to_c_equals_0() -> None:
    """T27: on the interior fixture with the exactly representable
    shift c = 1e9, mu and mu + 1e9*1 produce identical centered
    vectors and hence identical returned decisions in-process."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    plain = _target_of(
        strategy, _mv_context(("LOW", "HIGH"), forecast=_oracle_forecast())
    )
    lifted = _target_of(
        strategy,
        _mv_context(
            ("LOW", "HIGH"),
            forecast=_forecast({"LOW": 1.0 + 1e9, "HIGH": 2.0 + 1e9}),
        ),
    )
    plain_map = dict(plain.weights)
    lifted_map = dict(lifted.weights)
    assert plain_map.keys() == lifted_map.keys()
    for asset, plain_weight in plain_map.items():
        assert _double_bytes(plain_weight) == _double_bytes(lifted_map[asset]), (
            f"identical law broken at {asset}: "
            f"{plain_weight!r} vs {lifted_map[asset]!r}"
        )
    for asset, exact in zip(("LOW", "HIGH"), _EXACT_INTERIOR_ORACLE):
        assert plain_map[asset] == pytest.approx(float(exact), abs=_TEST_ORACLE_TOL)
        assert lifted_map[asset] == pytest.approx(float(exact), abs=_TEST_ORACLE_TOL)


def test_rejection_outcomes_are_identical_at_c0_and_c1e9() -> None:
    """T27: the near-optimal [0.101, 0.899] rejection (KKT and
    domination) and the optimum acceptance are identical at c = 0 and
    c = 1e9 - the centered acceptance laws are translation-invariant
    (the raw law's tolerance would inflate by c and accept)."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)
    for values in ({"LOW": 1.0, "HIGH": 2.0}, {"LOW": 1.0 + 1e9, "HIGH": 2.0 + 1e9}):

        def rejecting(fun, x0, **kwargs):
            return _fake_result(list(_NEAR_OPTIMAL))

        with _scipy_fake(rejecting), pytest.raises(ValueError):
            _target_of(
                strategy, _mv_context(("LOW", "HIGH"), forecast=_forecast(values))
            )

        def optimal(fun, x0, **kwargs):
            return _fake_result([0.1, 0.9])

        with _scipy_fake(optimal):
            target = _target_of(
                strategy, _mv_context(("LOW", "HIGH"), forecast=_forecast(values))
            )
        assert dict(target.weights)["LOW"] == pytest.approx(0.1, abs=_TEST_ORACLE_TOL)


def test_centering_fail_closed_gates_fire() -> None:
    """T27: the G9a unconditional checks - nonfinite mu_bar (including
    fsum overflow) and nonfinite mu_tilde entries - reject rather than
    reaching the solver."""
    mean_variance = _require_mean_variance()
    strategy = mean_variance(dict(_MAP_2), 4, _LAMBDA)

    def never(fun, x0, **kwargs):
        raise AssertionError("the centering check must fire before the solve")

    with _scipy_fake(never):
        overflowing = _forecast({"LOW": 1.5e308, "HIGH": 1.5e308})
        with pytest.raises((ValueError, OverflowError)):
            _target_of(strategy, _mv_context(("LOW", "HIGH"), forecast=overflowing))
