"""The inverse-volatility strategy contract (test-first).

This module defines the fixed behavior floors for the second built-in
classical strategy, ``portlearn.strategies.InverseVolatility``, exactly
as specified in the documented design for this strategy.

Contract discipline, by contract:

* every expected value is independently computed or hand-pinned —
  never derived from the implementation's own arithmetic;
* every floor exercises a public path only (public import, public
  constructor, public ``decide``, public contracts) — no
  private attribute access, no monkey-patching of any private surface;
* one narrow, explicitly sanctioned exception: the validator
  call-through floor instruments the PUBLIC
  ``portlearn.interfaces.require_decision_result_compatible``
  attribute through pytest's auto-undone ``monkeypatch`` fixture,
  solely to prove causally that ``decide`` invokes the public
  validator on its own inputs before returning;
* the class is absent at authoring time, so every target-dependent
  node resolves it lazily inside the test body and fails with an
  ``EXPECTED_REJECTION``-prefixed causal message instead of a module-level
  collection error — while nodes anchored purely on fixed live
  surfaces (the admission contract's own rejection laws, the fixed
  validator's own rejection law, the decision-context role law) act as
  always-green controls.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from portlearn.interfaces import (
    DecisionContext,
    DecisionResult,
    Forecast,
    InformationSet,
    PortfolioDecision,
    require_decision_result_compatible,
)
from portlearn.observations import AmbiguousObservationError, TimedObservation
from portlearn.timing import FutureInformationError, InvalidChronologyError
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


def _require_inverse_volatility():
    """Resolve the public ``InverseVolatility`` class or fail causally."""
    module = _require_strategies_module()
    try:
        return module.InverseVolatility
    except AttributeError as exc:
        pytest.fail(
            "EXPECTED_REJECTION absent public class 'InverseVolatility' on module "
            f"'portlearn.strategies': {exc}",
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
# Hand-set world constants                                                     #
# --------------------------------------------------------------------------- #

#: The pre-window outlier instant (an older admissible observation).
_T0 = datetime(2025, 12, 31, tzinfo=UTC)

#: The balanced common grid T1..T4 (aware UTC month ends).
_T1 = datetime(2026, 1, 31, tzinfo=UTC)
_T2 = datetime(2026, 2, 28, tzinfo=UTC)
_T3 = datetime(2026, 3, 31, tzinfo=UTC)
_T4 = datetime(2026, 4, 30, tzinfo=UTC)

#: The default decision instant (after the whole grid).
_T5 = datetime(2026, 5, 31, tzinfo=UTC)

#: A later decision instant and the ledger horizon close.
_T6 = datetime(2026, 6, 30, tzinfo=UTC)
_T7 = datetime(2026, 7, 31, tzinfo=UTC)

_GRID = (_T1, _T2, _T3, _T4)

#: The base fixture returns over the common grid T1..T4 (hand-set).
_LOW_RETURNS = (0.01, 0.02, 0.01, 0.02)
_HIGH_RETURNS = (0.08, -0.02, 0.06, -0.04)

#: Hand-derived base-case weights:
#:
#: LOW : mean = 0.06/4 = 0.015, deviations +/-0.005,
#:       D_low = 4*(0.005)^2 = 0.0001, s_low = 1/sqrt(0.0001) = 100.0
#: HIGH: mean = 0.08/4 = 0.02, deviations (0.06,-0.04,0.04,-0.06),
#:       D_high = 0.0036+0.0016+0.0016+0.0036 = 0.0104,
#:       s_high = 1/sqrt(0.0104) ~= 9.80580675690920
#: w_low  = 100.0/(100.0+9.80580675690920)  ~= 0.9106986502214993
#: w_high = 9.80580675690920/(100.0+9.80580675690920) ~= 0.08930134977850068
_HAND_W_LOW = 0.9106986502214993
_HAND_W_HIGH = 0.08930134977850068

#: Hand-derived window=2 weights over the grid tail T3,T4:
#:
#: LOW : [0.01, 0.03] -> mean 0.02, deviations +/-0.01,
#:       D_low = 2*(0.01)^2 = 0.0002, s_low = 1/sqrt(0.0002)
#:       ~= 70.71067811865476
#: HIGH: [0.08, -0.02] -> mean 0.03, deviations +/-0.05,
#:       D_high = 2*(0.05)^2 = 0.005, s_high = 1/sqrt(0.005)
#:       ~= 14.142135623730951
#: score ratio = sqrt(0.005/0.0002) = sqrt(25) = 5 exactly, so
#: w_low = 5/6 ~= 0.8333333333333333 and w_high = 1/6.
_HAND_W2_LOW = 5.0 / 6.0
_HAND_W2_HIGH = 1.0 / 6.0

#: Hand-derived earliest-window weights for the >= window+1 outlier
#: fixture (both assets carry a T0 outlier; earliest 4 selected):
#: LOW : [5.0, 0.01, 0.02, 0.01] -> D = 18.650200000000002
#: HIGH: [-5.0, 0.08, -0.02, 0.06] -> D = 19.056800000000003
#: w_low ~= 0.50269586429682656, w_high ~= 0.49730413570317339 —
#: materially different from the base case (the selection-control law).
_HAND_EARLIEST_W_LOW = 0.50269586429682656

_MAPPING = {"LOW": "low_ret", "HIGH": "high_ret"}


def _obs(series: str, instant: datetime, value: object) -> TimedObservation:
    """One admitted record: same-instant availability, hand-set value."""
    return TimedObservation(
        series_id=series,
        observation_time=instant,
        available_time=instant,
        value=value,
    )


def _base_records(
    low: tuple[object, ...] = _LOW_RETURNS,
    high: tuple[object, ...] = _HIGH_RETURNS,
    low_instants: tuple[datetime, ...] = _GRID,
    high_instants: tuple[datetime, ...] = _GRID,
    extra: tuple[TimedObservation, ...] = (),
) -> tuple[TimedObservation, ...]:
    """The hand-set record fixture over per-series instant grids."""
    records = [_obs("low_ret", t, v) for t, v in zip(low_instants, low, strict=True)]
    records += [
        _obs("high_ret", t, v) for t, v in zip(high_instants, high, strict=True)
    ]
    return (*records, *extra)


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
        else _information(_base_records(), as_of=instant)
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


def _decide_target(strategy, **context_kwargs):
    """The decided target book of one ``decide`` call (public path)."""
    result = strategy.decide(_context(("LOW", "HIGH"), **context_kwargs))
    return result.decision.target_weights


# --------------------------------------------------------------------------- #
# Public import route, facade identity, and declared module surface            #
# --------------------------------------------------------------------------- #


def test_public_import_route_exposes_inverse_volatility() -> None:
    """The public route ``from portlearn.strategies import
    InverseVolatility`` works and the class is a class."""
    try:
        from portlearn.strategies import InverseVolatility
    except ImportError as exc:
        pytest.fail(
            f"EXPECTED_REJECTION the public import route fails because the class "
            f"is absent: {exc}",
            pytrace=False,
        )
    assert isinstance(InverseVolatility, type)


def test_module_all_declares_both_builtins_in_order() -> None:
    """The module ships exactly the evolved public surface:
    ``__all__ == ["EqualWeight", "InverseVolatility", "MinimumVariance", "MeanVariance"]`` (list equality,
    declaration order included)."""
    module = _require_strategies_module()
    try:
        inverse_volatility = module.InverseVolatility
    except AttributeError as exc:
        pytest.fail(
            f"EXPECTED_REJECTION absent public class 'InverseVolatility': {exc}",
            pytrace=False,
        )
    assert module.__all__ == [
        "EqualWeight",
        "InverseVolatility",
        "MinimumVariance",
        "MeanVariance",
    ]
    assert module.InverseVolatility is inverse_volatility


def test_root_facade_resolves_the_class_identically() -> None:
    """``pl.strategies is portlearn.strategies`` (lazy facade identity)
    and the facade route yields the identical class object."""
    with _fresh_modules_context():
        import portlearn

        strategies = _require_strategies_module()
        assert portlearn.strategies is strategies
        inverse_volatility = _require_inverse_volatility()
        assert portlearn.strategies.InverseVolatility is inverse_volatility


def test_decide_signature_accepts_exactly_one_context_positional() -> None:
    """The ``Strategy`` protocol is structural-only; this floor
    pins the behavior: ``decide`` accepts exactly one positional
    argument (a ``DecisionContext``)."""
    inverse_volatility = _require_inverse_volatility()
    import inspect

    strategy = inverse_volatility({"LOW": "low_ret"}, window=2)
    signature = inspect.signature(strategy.decide)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert len(positional) == 1, (
        "decide must accept exactly one positional DecisionContext; the "
        f"signature declares {len(positional)}"
    )


# --------------------------------------------------------------------------- #
# Constructor battery (unconditional, ValueError only)                           #
# --------------------------------------------------------------------------- #


def test_constructor_rejects_a_bare_string_mapping() -> None:
    """A single string is a common mistake for a mapping of one and
    rejects with ``ValueError`` naming the mapping law and the type."""
    inverse_volatility = _require_inverse_volatility()
    with pytest.raises(ValueError, match="must be a mapping") as caught:
        inverse_volatility("low_ret", window=4)
    assert "not str" in str(caught.value)


def test_constructor_rejects_non_mapping_types() -> None:
    """Any other non-mapping type rejects the same way."""
    inverse_volatility = _require_inverse_volatility()
    with pytest.raises(ValueError, match="must be a mapping"):
        inverse_volatility([("LOW", "low_ret")], window=4)
    with pytest.raises(ValueError, match="must be a mapping"):
        inverse_volatility(None, window=4)


def test_constructor_rejects_an_empty_mapping() -> None:
    """An inverse-volatility target over no mapped return series names
    no portfolio and rejects unconditionally."""
    inverse_volatility = _require_inverse_volatility()
    with pytest.raises(ValueError, match="must be a non-empty mapping"):
        inverse_volatility({}, window=4)


def test_constructor_rejects_non_string_asset_keys() -> None:
    """Every key (asset identifier) must be a non-blank exact string."""
    inverse_volatility = _require_inverse_volatility()
    with pytest.raises(ValueError, match="keys must be non-blank strings"):
        inverse_volatility({1: "low_ret"}, window=4)


def test_constructor_rejects_blank_asset_keys() -> None:
    """A blank key names no asset and rejects."""
    inverse_volatility = _require_inverse_volatility()
    with pytest.raises(ValueError, match="keys must be non-blank strings"):
        inverse_volatility({"   ": "low_ret"}, window=4)


def test_constructor_rejects_non_string_series_identifiers() -> None:
    """Every value (series identifier) must be a non-blank exact
    string."""
    inverse_volatility = _require_inverse_volatility()
    with pytest.raises(ValueError, match="values must be non-blank series identifiers"):
        inverse_volatility({"LOW": 3}, window=4)


def test_constructor_rejects_blank_series_identifiers() -> None:
    """A blank series identifier names no series and rejects."""
    inverse_volatility = _require_inverse_volatility()
    with pytest.raises(ValueError, match="values must be non-blank series identifiers"):
        inverse_volatility({"LOW": "  "}, window=4)


def test_constructor_rejects_one_series_serving_two_assets() -> None:
    """Two different asset keys mapping to the same series id reject,
    with the message naming the series id and both assets."""
    inverse_volatility = _require_inverse_volatility()
    with pytest.raises(ValueError, match="duplicate return-series mapping") as caught:
        inverse_volatility({"LOW": "s", "HIGH": "s"}, window=4)
    message = str(caught.value)
    assert "'s'" in message
    assert "'LOW'" in message
    assert "'HIGH'" in message


@pytest.mark.parametrize("window", [True, False])
def test_constructor_rejects_bool_windows(window) -> None:
    """``bool`` is an ``int`` subclass and rejects explicitly."""
    inverse_volatility = _require_inverse_volatility()
    with pytest.raises(ValueError, match="window must be an integer >= 2"):
        inverse_volatility({"LOW": "low_ret"}, window=window)


@pytest.mark.parametrize("window", [4.0, 2.5, "4", None, 4.0 + 0j])
def test_constructor_rejects_non_integer_windows(window) -> None:
    """Floats (including integral floats) and any other non-int type
    reject with the window law and the received type."""
    inverse_volatility = _require_inverse_volatility()
    with pytest.raises(ValueError, match="window must be an integer >= 2"):
        inverse_volatility({"LOW": "low_ret"}, window=window)


@pytest.mark.parametrize("window", [1, 0, -3])
def test_constructor_rejects_windows_below_two(window) -> None:
    """A dispersion estimate over fewer than two returns is undefined
    and rejects, the message carrying both law phrases."""
    inverse_volatility = _require_inverse_volatility()
    with pytest.raises(ValueError, match="window must be an integer >= 2") as caught:
        inverse_volatility({"LOW": "low_ret"}, window=window)
    assert "fewer than two returns" in str(caught.value)


def test_constructor_snapshots_the_mapping_immutably() -> None:
    """Later mutation of a caller-supplied dict cannot change any
    future decision, and the exposed snapshot is unchanged too."""
    inverse_volatility = _require_inverse_volatility()
    source = {"LOW": "low_ret", "HIGH": "high_ret"}
    strategy = inverse_volatility(source, window=4)
    source["LOW"] = "mutant_ret"
    source["X"] = "x_ret"
    del source["HIGH"]
    assert dict(strategy.return_series) == {
        "LOW": "low_ret",
        "HIGH": "high_ret",
    }
    target = dict(strategy.decide(_context(("LOW", "HIGH"))).decision.target_weights)
    assert math.isclose(target["LOW"], _HAND_W_LOW, rel_tol=1e-12)
    assert math.isclose(target["HIGH"], _HAND_W_HIGH, rel_tol=1e-12)


def test_public_properties_are_read_only_snapshots() -> None:
    """``return_series`` and ``window`` are inspectable read-only
    surfaces: view mutation raises ``TypeError`` and attribute
    reassignment raises ``AttributeError``."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility({"LOW": "low_ret"}, window=4)
    assert dict(strategy.return_series) == {"LOW": "low_ret"}
    assert strategy.window == 4
    with pytest.raises(TypeError):
        strategy.return_series["LOW"] = "mutant_ret"
    with pytest.raises(AttributeError):
        strategy.return_series = {"LOW": "other_ret"}
    with pytest.raises(AttributeError):
        strategy.window = 2


# --------------------------------------------------------------------------- #
# Hand-derived allocation laws                                                  #
# --------------------------------------------------------------------------- #


def test_two_asset_hand_derived_weights() -> None:
    """The base fixture decides the hand-derived unequal-dispersion
    weights (tight ``math.isclose`` tolerance, never the
    implementation's own arithmetic); key order follows universe
    order."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility(dict(_MAPPING), window=4)
    result = strategy.decide(_context(("LOW", "HIGH")))
    target = dict(result.decision.target_weights)
    assert list(target) == ["LOW", "HIGH"]
    assert math.isclose(target["LOW"], _HAND_W_LOW, rel_tol=1e-12), (
        f"w_low must be hand-derived {_HAND_W_LOW!r}; got {target['LOW']!r}"
    )
    assert math.isclose(target["HIGH"], _HAND_W_HIGH, rel_tol=1e-12), (
        f"w_high must be hand-derived {_HAND_W_HIGH!r}; got {target['HIGH']!r}"
    )


def test_minimum_window_hand_derived_weights() -> None:
    """``window=2`` over a dedicated two-observation common grid (T3, T4;
    LOW ``[0.01, 0.03]``, HIGH ``[0.08, -0.02]``) decides exactly 5/6 and
    1/6 (the score ratio is exactly 5 by hand)."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility(dict(_MAPPING), window=2)
    records = _base_records(
        low=(0.01, 0.03),
        high=(0.08, -0.02),
        low_instants=(_T3, _T4),
        high_instants=(_T3, _T4),
    )
    target = dict(
        strategy.decide(
            _context(("LOW", "HIGH"), information=_information(records))
        ).decision.target_weights
    )
    assert math.isclose(target["LOW"], _HAND_W2_LOW, rel_tol=1e-12), (
        f"w_low at window=2 must be 5/6; got {target['LOW']!r}"
    )
    assert math.isclose(target["HIGH"], _HAND_W2_HIGH, rel_tol=1e-12), (
        f"w_high at window=2 must be 1/6; got {target['HIGH']!r}"
    )


def test_record_submission_order_leaves_target_value_equal() -> None:
    """The same records submitted in reverse (and a third shuffled
    permutation) decide value-equal target books."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility(dict(_MAPPING), window=4)
    records = list(_base_records())
    reversed_records = tuple(reversed(records))
    shuffled_records = (
        records[6],
        records[0],
        records[7],
        records[2],
        records[4],
        records[1],
        records[5],
        records[3],
    )
    base_target = dict(
        strategy.decide(_context(("LOW", "HIGH"))).decision.target_weights
    )
    for permutation in (reversed_records, shuffled_records):
        information = _information(permutation)
        target = dict(
            strategy.decide(
                _context(("LOW", "HIGH"), information=information)
            ).decision.target_weights
        )
        assert target == base_target


def test_mapping_insertion_order_leaves_target_value_equal() -> None:
    """Equal mappings inserted in different orders decide value-equal
    targets; universe key order alone may differ as presentation."""
    inverse_volatility = _require_inverse_volatility()
    forward = inverse_volatility({"LOW": "low_ret", "HIGH": "high_ret"}, window=4)
    backward = inverse_volatility({"HIGH": "high_ret", "LOW": "low_ret"}, window=4)
    forward_target = dict(
        forward.decide(_context(("LOW", "HIGH"))).decision.target_weights
    )
    backward_target = dict(
        backward.decide(_context(("LOW", "HIGH"))).decision.target_weights
    )
    assert forward_target == backward_target


def test_records_outside_the_window_are_ignored() -> None:
    """An older admissible outlier return (well outside the window)
    leaves the decided weights unchanged from the no-outlier fixture."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility(dict(_MAPPING), window=4)
    outlier_records = _base_records(
        low=(5.0, *_LOW_RETURNS),
        high=(-5.0, *_HIGH_RETURNS),
        low_instants=(_T0, *_GRID),
        high_instants=(_T0, *_GRID),
    )
    target = dict(
        strategy.decide(
            _context(("LOW", "HIGH"), information=_information(outlier_records))
        ).decision.target_weights
    )
    assert math.isclose(target["LOW"], _HAND_W_LOW, rel_tol=1e-12), (
        f"the ancient +/-5.0 returns outside the window must leave the "
        f"weights unchanged; got {target['LOW']!r}"
    )
    assert math.isclose(target["HIGH"], _HAND_W_HIGH, rel_tol=1e-12)


def test_earliest_window_selection_would_change_the_target() -> None:
    """Selection control for the latest-window law: on the >= window+1
    outlier fixture, the hand-derived earliest-window weights differ
    materially from the decided latest-window weights, so a mutant that
    selects the earliest records (or all records) cannot survive this
    fixture."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility(dict(_MAPPING), window=4)
    outlier_records = _base_records(
        low=(5.0, *_LOW_RETURNS),
        high=(-5.0, *_HIGH_RETURNS),
        low_instants=(_T0, *_GRID),
        high_instants=(_T0, *_GRID),
    )
    target = dict(
        strategy.decide(
            _context(("LOW", "HIGH"), information=_information(outlier_records))
        ).decision.target_weights
    )
    # Earliest-4 hand derivation (in comments, independently):
    # D_low = 18.650200000000002, D_high = 19.056800000000003
    # w_low ~= 0.50269586429682656 — far from the base-case weight.
    assert abs(target["LOW"] - _HAND_EARLIEST_W_LOW) > 1e-3, (
        "the decided weights must be the latest-window weights; a value "
        "near the earliest-window hand weight means the wrong records "
        "were selected"
    )


def test_extra_older_records_do_not_unbalance_the_panel() -> None:
    """Five admissible records on one side and four on the other still
    succeed when the selected (latest window) grids match."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility(dict(_MAPPING), window=4)
    outlier_records = _base_records(
        low=(5.0, *_LOW_RETURNS),
        high=(-5.0, *_HIGH_RETURNS),
        low_instants=(_T0, *_GRID),
        high_instants=(_T0, *_GRID),
    )
    result = strategy.decide(
        _context(("LOW", "HIGH"), information=_information(outlier_records))
    )
    assert math.isclose(
        result.decision.target_weights["LOW"], _HAND_W_LOW, rel_tol=1e-12
    )


# --------------------------------------------------------------------------- #
# Unconditional decision laws (observable surface)                                #
# --------------------------------------------------------------------------- #


def test_shifted_observation_grids_reject_as_unbalanced() -> None:
    """Two assets observing same-count but shifted grids reject with
    the unbalanced-grid law naming both assets."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility(dict(_MAPPING), window=4)
    shifted = _base_records(
        high_instants=(_T2, _T3, _T4, _T5),
    )
    with pytest.raises(ValueError, match="unbalanced observation grid") as caught:
        strategy.decide(_context(("LOW", "HIGH"), information=_information(shifted)))
    message = str(caught.value)
    assert "'LOW'" in message
    assert "'HIGH'" in message
    assert "different observation instants" in message
    assert "no mismatched-date cross-section" in message


def test_insufficient_history_rejects_with_counts() -> None:
    """A mapped series with fewer admissible records than the window
    rejects naming the asset, the series, the counts, and the window —
    no partial target, no truncation."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility(dict(_MAPPING), window=4)
    short = _base_records(
        low=_LOW_RETURNS[:3],
        low_instants=_GRID[:3],
    )
    with pytest.raises(ValueError, match="insufficient history") as caught:
        strategy.decide(_context(("LOW", "HIGH"), information=_information(short)))
    message = str(caught.value)
    assert "'LOW'" in message
    assert "'low_ret'" in message
    assert "window=4" in message
    assert "3" in message
    assert "4" in message


def test_unmapped_universe_asset_rejects() -> None:
    """A universe asset with no ``return_series`` entry rejects naming
    the asset and the decision instant."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility(dict(_MAPPING), window=4)
    with pytest.raises(ValueError, match="missing return-series mapping") as caught:
        strategy.decide(_context(("LOW", "HIGH", "X")))
    message = str(caught.value)
    assert "'X'" in message
    assert "decision_time" in message


def test_empty_universe_rejects_fail_closed() -> None:
    """An empty effective support names no portfolio and rejects with
    the empty-support law naming the decision instant."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility({"LOW": "low_ret"}, window=4)
    with pytest.raises(
        ValueError, match="inverse-volatility effective support is empty"
    ) as caught:
        strategy.decide(_context(()))
    message = str(caught.value)
    assert "universe is empty" in message
    assert "decision_time" in message


def test_bool_return_value_rejects_as_non_real() -> None:
    """A selected ``True`` value rejects with the non-real law: bool is
    not a return."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility({"LOW": "low_ret"}, window=2)
    records = (
        _obs("low_ret", _T3, True),
        _obs("low_ret", _T4, 0.01),
    )
    with pytest.raises(ValueError, match="non-real return value") as caught:
        strategy.decide(_context(("LOW",), information=_information(records)))
    message = str(caught.value)
    assert "'LOW'" in message
    assert "bool" in message
    assert "bool is not a return" in message


def test_complex_return_value_rejects_as_non_real() -> None:
    """A complex selected value rejects with the non-real law naming
    the type."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility({"LOW": "low_ret"}, window=2)
    records = (
        _obs("low_ret", _T3, 1 + 2j),
        _obs("low_ret", _T4, 0.01),
    )
    with pytest.raises(ValueError, match="non-real return value") as caught:
        strategy.decide(_context(("LOW",), information=_information(records)))
    assert "'LOW'" in str(caught.value)
    assert "complex" in str(caught.value)


def test_nan_return_value_rejects_as_non_representable() -> None:
    """A NaN selected value rejects with the non-representable law
    naming the asset and the series."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility({"LOW": "low_ret"}, window=2)
    records = (
        _obs("low_ret", _T3, float("nan")),
        _obs("low_ret", _T4, 0.01),
    )
    with pytest.raises(
        ValueError, match="non-representable or nonfinite return value"
    ) as caught:
        strategy.decide(_context(("LOW",), information=_information(records)))
    message = str(caught.value)
    assert "'LOW'" in message
    assert "'low_ret'" in message


def test_infinite_return_value_rejects_as_non_representable() -> None:
    """An infinite selected value rejects the same way."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility({"LOW": "low_ret"}, window=2)
    records = (
        _obs("low_ret", _T3, float("inf")),
        _obs("low_ret", _T4, 0.01),
    )
    with pytest.raises(ValueError, match="non-representable or nonfinite return value"):
        strategy.decide(_context(("LOW",), information=_information(records)))


def test_conversion_overflow_return_value_rejects_as_value_error() -> None:
    """A real value too large for a finite float (the int ``10**1000``
    passes the Real check yet overflows ``float()``) rejects as the E6
    ``ValueError`` naming the asset and series — no raw
    ``OverflowError`` escapes ``decide``."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility({"LOW": "low_ret"}, window=2)
    records = (
        _obs("low_ret", _T3, 10**1000),
        _obs("low_ret", _T4, 0.01),
    )
    with pytest.raises(
        ValueError, match="non-representable or nonfinite return value"
    ) as caught:
        strategy.decide(_context(("LOW",), information=_information(records)))
    message = str(caught.value)
    assert "'LOW'" in message
    assert "'low_ret'" in message


def test_constant_window_rejects_zero_dispersion() -> None:
    """A window of constant returns has exactly zero dispersion and
    rejects with the zero-dispersion law — the message itself states
    that no epsilon or fallback exists, and no target is returned."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility({"LOW": "low_ret"}, window=4)
    records = tuple(_obs("low_ret", t, 0.01) for t in _GRID)
    with pytest.raises(
        ValueError, match="zero or nonfinite return dispersion"
    ) as caught:
        strategy.decide(_context(("LOW",), information=_information(records)))
    message = str(caught.value)
    assert "'LOW'" in message
    assert "undefined" in message
    assert "no epsilon" in message


def test_finite_returns_overflowing_the_square_reject() -> None:
    """Finite selected returns (magnitude 1e200) whose centered square
    overflows reject with the dispersion law, never a raw
    ``OverflowError``."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility({"LOW": "low_ret"}, window=2)
    records = (
        _obs("low_ret", _T3, 1e200),
        _obs("low_ret", _T4, -1e200),
    )
    with pytest.raises(
        ValueError, match="zero or nonfinite return dispersion"
    ) as caught:
        strategy.decide(_context(("LOW",), information=_information(records)))
    message = str(caught.value)
    assert "'LOW'" in message
    assert "finite" in message


def test_finite_returns_overflowing_the_dispersion_sum_reject() -> None:
    """Finite selected returns (+/-1e154) whose centered squares each
    fit but whose ``fsum`` overflows reject the same way."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility({"LOW": "low_ret"}, window=2)
    records = (
        _obs("low_ret", _T3, 1e154),
        _obs("low_ret", _T4, -1e154),
    )
    with pytest.raises(
        ValueError, match="zero or nonfinite return dispersion"
    ) as caught:
        strategy.decide(_context(("LOW",), information=_information(records)))
    assert "'LOW'" in str(caught.value)
    assert "finite" in str(caught.value)


def test_finite_returns_overflowing_the_mean_reject() -> None:
    """A near-maximal same-sign window (~1.5e308) overflows the mean
    accumulation itself and rejects the same way."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility({"LOW": "low_ret"}, window=2)
    records = (
        _obs("low_ret", _T3, 1.5e308),
        _obs("low_ret", _T4, 1.5e308),
    )
    with pytest.raises(
        ValueError, match="zero or nonfinite return dispersion"
    ) as caught:
        strategy.decide(_context(("LOW",), information=_information(records)))
    assert "finite" in str(caught.value)


def test_unrelated_series_records_leave_target_unchanged() -> None:
    """Admitted records for a series not in ``return_series.values()``
    are ignored: the target is value-equal to the fixture without
    them."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility(dict(_MAPPING), window=4)
    noisy = _base_records(
        extra=tuple(_obs("unrelated_ret", t, 999.0) for t in (_T0, *_GRID, _T5))
    )
    noisy_target = dict(
        strategy.decide(
            _context(("LOW", "HIGH"), information=_information(noisy))
        ).decision.target_weights
    )
    clean_target = dict(
        strategy.decide(_context(("LOW", "HIGH"))).decision.target_weights
    )
    assert noisy_target == clean_target


# --------------------------------------------------------------------------- #
# Full target independence and the state law                                    #
# --------------------------------------------------------------------------- #


def test_target_independent_of_forecast_state_and_holdings() -> None:
    """Paired contexts differing only in ``forecast``, in
    ``strategy_state``, in lawful PRE_TRADE ``current_weights``, and in
    ``current_weights_as_of`` decide exactly value-equal targets; the
    sentinel state is deep-equal and un-mutated afterwards; the result
    carries ``next_strategy_state is None``."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility(dict(_MAPPING), window=4)
    forecast = Forecast(
        values={"LOW": -0.9, "HIGH": 0.4},
        target="expected_return",
        decision_time=_T5,
        produced_by="test-forecaster",
    )
    sentinel = {"carried": [1, 2, 3]}
    other_book = PortfolioWeights({"LOW": 0.5, "HIGH": 0.5}, WeightState.PRE_TRADE)
    baseline = _decide_target(strategy)
    pairings = (
        _decide_target(strategy, forecast=forecast),
        _decide_target(strategy, strategy_state=sentinel),
        _decide_target(strategy, current_weights=other_book),
        _decide_target(strategy, current_weights_as_of=_T4),
    )
    for paired in pairings:
        assert paired == baseline
    assert sentinel == {"carried": [1, 2, 3]}
    result = strategy.decide(
        _context(("LOW", "HIGH"), strategy_state=sentinel, forecast=forecast)
    )
    assert result.next_strategy_state is None


# --------------------------------------------------------------------------- #
# Result anchoring and the mandatory public validator path                      #
# --------------------------------------------------------------------------- #


def test_result_is_dated_and_executed_at_the_decision_instant() -> None:
    """The returned decision is dated and executed at exactly the
    context's ``decision_time`` (the same-instant reference
    convention)."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility(dict(_MAPPING), window=4)
    context = _context(("LOW", "HIGH"))
    result = strategy.decide(context)
    assert result.decision.decision_time == context.decision_time
    assert result.decision.execution_time == context.decision_time
    assert require_decision_result_compatible(context, result) is None


def test_decide_calls_public_validator_before_returning(monkeypatch) -> None:
    """The sanctioned causal floor: ``decide`` invokes the PUBLIC
    ``portlearn.interfaces.require_decision_result_compatible`` exactly
    once, on its own ``(context, result)`` pair, before returning —
    proven by the sentinel-raise form and the recording form."""
    import portlearn.interfaces as pl_iface

    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility(dict(_MAPPING), window=4)
    context = _context(("LOW", "HIGH"))

    class _ValidatorSentinel(Exception):
        pass

    def _raising(context_arg, result_arg):
        raise _ValidatorSentinel("instrumented rejection")

    monkeypatch.setattr(pl_iface, "require_decision_result_compatible", _raising)
    with pytest.raises(_ValidatorSentinel):
        strategy.decide(context)

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


def test_frozen_validator_rejects_a_misanchored_result() -> None:
    """Independent validator teeth (a property of the strict validator
    alone): a hand-mis-anchored result rejects with the fixed
    ``InvalidChronologyError``."""
    context = _context(("LOW", "HIGH"))
    misanchored = DecisionResult(
        decision=PortfolioDecision(
            decision_time=_T4,
            execution_time=_T4,
            target_weights={"LOW": 0.5, "HIGH": 0.5},
        ),
        next_strategy_state=None,
    )
    with pytest.raises(InvalidChronologyError):
        require_decision_result_compatible(context, misanchored)


# --------------------------------------------------------------------------- #
# Inherited fixed contract controls (always passing, no strategy involvement)    #
# --------------------------------------------------------------------------- #


def test_exact_duplicate_records_reject_at_admission() -> None:
    """Duplicate-record ownership is the fixed admission contract's:
    submitting an exact duplicate rejects with
    ``AmbiguousObservationError`` before any strategy sees anything."""
    duplicate = _obs("low_ret", _T1, 0.01)
    with pytest.raises(AmbiguousObservationError):
        InformationSet((duplicate, duplicate), as_of=_T5)


def test_revision_pairs_reject_at_admission() -> None:
    """A revision pair (same series-observation identity, later
    availability, different value) likewise rejects with
    ``AmbiguousObservationError`` at admission."""
    original = _obs("low_ret", _T1, 0.01)
    revised = TimedObservation(
        series_id="low_ret",
        observation_time=_T1,
        available_time=_T2,
        value=0.02,
    )
    with pytest.raises(AmbiguousObservationError):
        InformationSet((original, revised), as_of=_T5)


def test_future_information_rejects_at_context_construction() -> None:
    """Look-ahead rejection is owned by the fixed admission contract:
    a record available only after the decision instant rejects with
    ``FutureInformationError`` at ``InformationSet`` construction."""
    lookahead = TimedObservation(
        series_id="low_ret",
        observation_time=_T5,
        available_time=_T6,
        value=0.01,
    )
    with pytest.raises(FutureInformationError):
        InformationSet((lookahead,), as_of=_T5)


def test_wrong_weights_role_rejects_at_context_construction() -> None:
    """The decision-context role law is inherited and fixed: a TARGET
    book as ``current_weights`` rejects at construction."""
    target_book = PortfolioWeights({"LOW": 1.0}, WeightState.TARGET)
    with pytest.raises(ValueError, match="PRE_TRADE"):
        DecisionContext(
            decision_time=_T5,
            information=_information(_base_records()),
            universe=("LOW",),
            current_weights=target_book,
            current_weights_as_of=_T5,
        )


# --------------------------------------------------------------------------- #
# Root package surface and fresh-interpreter import discipline                  #
# --------------------------------------------------------------------------- #


#: The fresh-interpreter probe source (the established pattern).
_PACKAGE_PROBE = """
import importlib
import sys

import portlearn

print('registered=' + ','.join(sorted(
    name
    for name in sys.modules
    if name == 'portlearn' or name.startswith('portlearn.')
)))
print('all=' + ','.join(sorted(portlearn.__all__)))
strategies = portlearn.strategies
import portlearn.strategies as direct

print('strategies_identity=' + str(strategies is direct))
print('inverse_volatility=' + str(getattr(direct, 'InverseVolatility', None)))
"""


def _fresh_interpreter_probe_facts() -> dict:
    """Run the probe in a fresh interpreter and return its printed
    ``key=value`` facts."""
    import os
    import subprocess

    import portlearn

    source_root = os.path.dirname(os.path.dirname(os.path.abspath(portlearn.__file__)))
    existing = os.environ.get("PYTHONPATH")
    pythonpath = source_root + (os.pathsep + existing if existing else "")
    completed = subprocess.run(
        [sys.executable, "-c", _PACKAGE_PROBE],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
        timeout=120,
        check=True,
    )
    facts: dict = {}
    for line in completed.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            facts[key] = value
    return facts


def test_fresh_interpreter_bare_import_registers_only_the_root_package() -> None:
    """In a fresh interpreter a bare ``import portlearn`` registers
    exactly the root package — no eager submodule."""
    facts = _fresh_interpreter_probe_facts()
    assert facts.get("registered") == "portlearn", facts


def test_root_all_unchanged_and_facade_exposes_the_class() -> None:
    """The root ``__all__`` is NOT expanded (the strategies facade
    rides the ``__getattr__ branch alone) and the facade route yields
    the identical class object in a fresh interpreter."""
    facts = _fresh_interpreter_probe_facts()
    assert facts.get("all") == "__version__,data,weights", facts
    assert facts.get("strategies_identity") == "True", facts
    resolved = facts.get("inverse_volatility")
    if resolved != "<class 'portlearn.strategies.InverseVolatility'>":
        pytest.fail(
            "EXPECTED_REJECTION the root facade does not yet expose the class "
            f"'InverseVolatility' on 'portlearn.strategies'; resolved: {resolved!r}",
            pytrace=False,
        )


# --------------------------------------------------------------------------- #
# Determinism and statelessness                                                 #
# --------------------------------------------------------------------------- #


def test_repeated_decides_on_equal_contexts_are_value_equal() -> None:
    """Same instance, fresh-but-equal context objects: repeated calls
    decide value-equal targets (no hidden per-call drift)."""
    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility(dict(_MAPPING), window=4)
    first = dict(strategy.decide(_context(("LOW", "HIGH"))).decision.target_weights)
    for _ in range(5):
        again = dict(strategy.decide(_context(("LOW", "HIGH"))).decision.target_weights)
        assert again == first


def test_independent_instances_decide_value_equal_results() -> None:
    """Two independently constructed instances with equal
    configuration decide value-equal results on value-equal contexts
    (statelessness; no per-instance residue)."""
    inverse_volatility = _require_inverse_volatility()
    first = inverse_volatility(dict(_MAPPING), window=4).decide(
        _context(("LOW", "HIGH"))
    )
    second = inverse_volatility(dict(_MAPPING), window=4).decide(
        _context(("LOW", "HIGH"))
    )
    assert first == second


# --------------------------------------------------------------------------- #
# Constraint-valid book (public weight contracts)                        #
# --------------------------------------------------------------------------- #


def test_decided_books_validate_under_default_constraints() -> None:
    """The base-case and window=2 targets lift to ``PortfolioWeights``
    TARGET books and pass the fixed tolerance-based budget validator
    under defaults (long-only, budget 1.0, tolerance 1e-9) — never an
    exact-total assertion."""
    inverse_volatility = _require_inverse_volatility()
    window_two_records = _base_records(
        low=(0.01, 0.03),
        high=(0.08, -0.02),
        low_instants=(_T3, _T4),
        high_instants=(_T3, _T4),
    )
    for window, expected, information in (
        (4, {"LOW": _HAND_W_LOW, "HIGH": _HAND_W_HIGH}, None),
        (
            2,
            {"LOW": _HAND_W2_LOW, "HIGH": _HAND_W2_HIGH},
            _information(window_two_records),
        ),
    ):
        strategy = inverse_volatility(dict(_MAPPING), window=window)
        target = dict(
            strategy.decide(
                _context(("LOW", "HIGH"), information=information)
            ).decision.target_weights
        )
        for asset, weight in expected.items():
            assert math.isclose(target[asset], weight, rel_tol=1e-12)
        book = PortfolioWeights(target, WeightState.TARGET)
        assert require_valid_target(book, WeightConstraints()) is None


def test_frozen_validator_rejects_a_hand_broken_book() -> None:
    """Negative control for the validator path: a hand-broken negative
    weight book rejects under the default long-only constraints."""
    broken = PortfolioWeights({"LOW": -0.5, "HIGH": 1.5}, WeightState.TARGET)
    with pytest.raises(ValueError):
        require_valid_target(broken, WeightConstraints())


# --------------------------------------------------------------------------- #
# Composition with the fixed qualified-data and accounting stacks              #
# --------------------------------------------------------------------------- #


def _qualified_fixture():
    """The base fixture carried by a real ``QualifiedDataset`` through
    the public data stack (records, provenance, policy)."""
    from portlearn.data.dataset import QualifiedDataset
    from portlearn.data.ingestion import AvailabilityPolicy

    source_bytes = b"m32-composition-fixture-bytes"
    digest = hashlib.sha256(source_bytes).hexdigest()
    policy = AvailabilityPolicy.same_instant()
    provenance = SimpleNamespace(
        content_sha256=digest, units="decimal_return", frequency="M"
    )
    qualified = SimpleNamespace(
        availability=policy,
        content_sha256=digest,
        units="decimal_return",
        frequency="M",
    )
    return QualifiedDataset(
        provider="synthetic",
        name="IVFIX",
        provider_dataset_id="IVFIX",
        adapter_version="synthetic-test-adapter/1.0.0",
        availability=policy,
        source_bytes=source_bytes,
        auxiliary_bytes=None,
        records=_base_records(),
        retrieval_provenance=provenance,
        qualified_provenance=qualified,
    )


def test_qualified_dataset_flows_through_to_a_hand_derived_target() -> None:
    """The composition chain ``QualifiedDataset → InformationSet →
    InverseVolatility → DecisionResult`` runs through public contracts
    only and lands on the hand-derived base-case weights."""
    inverse_volatility = _require_inverse_volatility()
    dataset = _qualified_fixture()
    information = dataset.to_information_set(as_of=_T5)
    strategy = inverse_volatility(dict(_MAPPING), window=4)
    result = strategy.decide(_context(("LOW", "HIGH"), information=information))
    target = dict(result.decision.target_weights)
    assert math.isclose(target["LOW"], _HAND_W_LOW, rel_tol=1e-12)
    assert math.isclose(target["HIGH"], _HAND_W_HIGH, rel_tol=1e-12)
    book = PortfolioWeights(target, WeightState.TARGET)
    assert require_valid_target(book, WeightConstraints()) is None


#: Dyadic-exact growth supplies for the accounting path (hand-set).
_GROWTH = {
    _T5: {"LOW": 1.0625, "HIGH": 1.1875},
    _T6: {"LOW": 1.25, "HIGH": 0.75},
}


def test_target_flows_through_public_accounting_into_hand_pinned_wealth() -> None:
    """The decided target flows through the public accounting
    stack end-to-end: decide twice (T5, T6) with same-instant
    execution, hand-drift the T5 target to the T6 pre-trade book via
    the public drift operation, then compose ``build_ledger`` with
    ``Proportional(0.001, one_way)`` under ``ExactFillAccounting``.

    Hand derivation (independently, w_low ~= 0.91069865022149932 and
    w_high ~= 0.089301349778500683 from the base case):

    Period [T5,T6):
      trade at T5 from {CASH:1.0} to {LOW:w_low, HIGH:w_high}:
        two_sided = 1 + w_low + w_high = 2.0 -> one_way 1.0
        q0 = 0.001*1.0 -> F_cost,0 = 0.999
      drift: D1 = w_low*1.0625 + w_high*1.1875 = 1.0736626687223125
      drifted LOW = w_low*1.0625/D1 = 0.9012302877325834
    Period [T6,T7):
      trade at T6 back to {LOW:w_low, HIGH:w_high}:
        |d_low| = |d_high| = 0.0094683624889159 -> one_way 0.0094683624889159
        q1 = 0.001*0.0094683624889159 = 9.4683624889160e-06
      drift: D2 = w_low*1.25 + w_high*0.75 = 1.2053493251107499
    final = D1 * 0.999 * D2 * (1 - q1) = 1.2928321934481568 (hand-computed).
    """
    from portlearn.costs import Proportional
    from portlearn.ledger import ExactFillAccounting, build_ledger
    from portlearn.rebalance import drift_weights
    from portlearn.turnover import one_way

    inverse_volatility = _require_inverse_volatility()
    strategy = inverse_volatility(dict(_MAPPING), window=4)
    holdings = PortfolioWeights({"CASH": 1.0}, WeightState.PRE_TRADE)
    result_t5 = strategy.decide(
        _context(("LOW", "HIGH"), decision_time=_T5, current_weights=holdings)
    )
    target_t5 = dict(result_t5.decision.target_weights)
    lifted = PortfolioWeights(dict(target_t5), WeightState.TARGET)
    assert require_valid_target(lifted, WeightConstraints()) is None
    drifted = drift_weights(
        PortfolioWeights(dict(target_t5), WeightState.POST_TRADE),
        _GROWTH[_T5],
    )
    result_t6 = strategy.decide(
        _context(
            ("LOW", "HIGH"),
            decision_time=_T6,
            current_weights=drifted,
            current_weights_as_of=_T6,
            information=_information(_base_records(), as_of=_T6),
        )
    )
    path = build_ledger(
        initial_weights={"CASH": 1.0},
        accounting_instants=(_T5, _T6, _T7),
        growth_factors=dict(_GROWTH),
        decisions=(result_t5.decision, result_t6.decision),
        cost_model=Proportional(0.001, turnover=one_way),
        accounting_engine=ExactFillAccounting(),
    )
    assert path.n_periods == 2
    assert len(path.rows) == 2
    assert math.isclose(path.final_wealth, 1.2928321934481568, rel_tol=1e-12), (
        f"final wealth must be the hand-derived 1.2928321934481568; "
        f"got {path.final_wealth!r}"
    )
    assert path.rows[0].turnover == 1.0
    assert math.isclose(path.rows[1].turnover, 0.0094683624889159, rel_tol=1e-12)
    assert math.isclose(
        path.rows[0].closing_weights["LOW"], 0.9012302877325834, rel_tol=1e-12
    )
    default_path = build_ledger(
        initial_weights={"CASH": 1.0},
        accounting_instants=(_T5, _T6, _T7),
        growth_factors=dict(_GROWTH),
        decisions=(result_t5.decision, result_t6.decision),
        cost_model=Proportional(0.001, turnover=one_way),
    )
    assert default_path.final_wealth == path.final_wealth


def test_same_support_as_equal_weight_over_the_identical_universe() -> None:
    """Same-support comparator fairness: the identical
    ``DecisionContext.universe`` supplied to both built-ins yields
    books over the identical key set — no selection, no ranking, no
    filtering (value difference is expected and not asserted)."""
    from portlearn.strategies import EqualWeight

    inverse_volatility = _require_inverse_volatility()
    universe = ("LOW", "HIGH", "X")
    mapping = {"LOW": "low_ret", "HIGH": "high_ret", "X": "x_ret"}
    records = _base_records(
        extra=tuple(
            _obs("x_ret", t, value)
            for t, value in zip(_GRID, (0.02, -0.01, 0.02, -0.01), strict=True)
        )
    )
    equal_book = dict(EqualWeight().decide(_context(universe)).decision.target_weights)
    inverse_book = dict(
        inverse_volatility(mapping, window=4)
        .decide(_context(universe, information=_information(records)))
        .decision.target_weights
    )
    assert set(equal_book) == set(inverse_book) == set(universe)
    assert len(inverse_book) == len(universe)
