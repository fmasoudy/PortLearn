"""The equal-weight strategy contract (test-first).

This module defines the fixed behavior floors for the first built-in
classical strategy, ``portlearn.strategies.EqualWeight``, exactly as
specified in the documented design for this strategy.

Contract discipline, by contract:

* every expected value is independently computed or hand-pinned —
  never derived from the implementation's own arithmetic;
* every floor exercises a public path only (public import, public
  constructor, public ``decide``) — no private attribute access, no
  monkey-patching of any private surface;
* one narrow, explicitly sanctioned exception: the validator
  call-through floor instruments the PUBLIC
  ``portlearn.interfaces.require_decision_result_compatible``
  attribute through pytest's auto-undone ``monkeypatch`` fixture,
  solely to prove causally that ``decide`` invokes the public
  validator on its own inputs before returning;
* the production module is absent at authoring time, so every
  target-dependent node resolves it lazily inside the test body and
  fails with an ``EXPECTED_REJECTION``-prefixed causal message instead of a
  module-level collection error — while nodes anchored purely on
  fixed live surfaces (the validator's own rejection laws, the
  default-constraint validator path) act as always-green controls.
"""

from __future__ import annotations

import importlib
import inspect
import math
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

import pytest

from portlearn.costs import Proportional
from portlearn.interfaces import (
    DecisionContext,
    DecisionResult,
    Forecast,
    InformationSet,
    PortfolioDecision,
    require_decision_result_compatible,
)
from portlearn.ledger import build_ledger
from portlearn.observations import TimedObservation
from portlearn.timing import InvalidChronologyError
from portlearn.turnover import one_way
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


def _require_equal_weight():
    """Resolve the public ``EqualWeight`` class or fail causally."""
    module = _require_strategies_module()
    try:
        return module.EqualWeight
    except AttributeError as exc:
        pytest.fail(
            "EXPECTED_REJECTION absent public class 'EqualWeight' on module "
            f"'portlearn.strategies': {exc}",
            pytrace=False,
        )


def _require_facade_exposes_strategies(root):
    """Resolve the root facade's lazy ``strategies`` attribute or fail
    causally (the facade branch is part of the absent production)."""
    try:
        return root.strategies
    except AttributeError as exc:
        pytest.fail(
            "EXPECTED_REJECTION the root package facade does not yet lazily "
            f"expose 'strategies': {exc}",
            pytrace=False,
        )


@contextmanager
def _fresh_modules_context() -> Iterator[None]:
    """Import ``portlearn*`` fresh inside the block, then RESTORE the
    exact prior module registry on exit.

    Mirrors the established pattern (test_package_contract /
    test_weight_contracts purge ``portlearn*`` to observe one fresh
    import). Restoring the registry afterwards keeps every LATER node
    in this module on the same generation as its collection-bound
    names (the file's own imports), so class identity and
    isinstance-checks stay meaningful — only the nodes inside the
    block ever see the fresh generation."""
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

#: The default decision instant (an aware UTC month end).
_T0 = datetime(2026, 6, 30, tzinfo=UTC)

#: A later aware UTC month end for cross-instant dynamics.
_T1 = datetime(2026, 7, 31, tzinfo=UTC)

#: The terminal accounting instant closing the second F14 segment.
_T2 = datetime(2026, 8, 31, tzinfo=UTC)

#: The hand-pinned exact binary doubles for 1/n (never computed here).
_RECIPROCAL = {
    2: 0.5,
    3: 0.3333333333333333,
    4: 0.25,
    5: 0.2,
    48: 0.020833333333333332,
}


def _context(
    universe,
    *,
    decision_time=None,
    current_weights=None,
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
    instant = _T0 if decision_time is None else decision_time
    holdings = (
        current_weights
        if current_weights is not None
        else PortfolioWeights({"CASH": 1.0}, WeightState.PRE_TRADE)
    )
    admitted = (
        information if information is not None else InformationSet((), as_of=instant)
    )
    return DecisionContext(
        decision_time=instant,
        information=admitted,
        universe=tuple(universe),
        current_weights=holdings,
        current_weights_as_of=instant,
        strategy_state=strategy_state,
        forecast=forecast,
    )


def _target_of(strategy, universe, **kwargs):
    """The decided target book of one ``decide`` call (public path)."""
    result = strategy.decide(_context(universe, **kwargs))
    return result.decision.target_weights


# --------------------------------------------------------------------------- #
# F1 — public import and lazy facade route                                     #
# --------------------------------------------------------------------------- #


def test_public_import_route_exposes_equal_weight() -> None:
    """The public route ``from portlearn.strategies import EqualWeight``
    works and the module ships exactly the public surface."""
    with _fresh_modules_context():
        from portlearn.strategies import EqualWeight

        module = _require_strategies_module()
        assert module.__all__ == [
            "EqualWeight",
            "InverseVolatility",
            "MinimumVariance",
            "MeanVariance",
        ]
        assert module.EqualWeight is EqualWeight
        # The class is a class (public constructor floor).
        assert isinstance(EqualWeight, type)


def test_root_package_lazily_exposes_strategies_module() -> None:
    """A bare ``import portlearn`` registers only the root package; the
    ``strategies`` attribute then resolves to exactly the module a
    direct import yields (lazy facade identity, mirroring the fixed
    weights/rebalance facade floors)."""
    with _fresh_modules_context():
        import portlearn

        names_after_bare_import = {
            name
            for name in sys.modules
            if name == "portlearn" or name.startswith("portlearn.")
        }
        assert names_after_bare_import == {"portlearn"}
        resolved = _require_facade_exposes_strategies(portlearn)
        import portlearn.strategies as direct

        assert resolved is direct
        assert direct.__name__ == "portlearn.strategies"
        # The two public routes give the identical class object.
        import portlearn as pl

        assert pl.strategies.EqualWeight is direct.EqualWeight


def test_root_package_all_and_error_message_name_strategies() -> None:
    """``__all__`` gains exactly the strategies entry and the
    unknown-attribute error names the public modules including
    ``strategies``."""
    with _fresh_modules_context():
        import portlearn as pl

        # The fixed design: the root __all__ is NOT modified —
        # strategies is exposed by the __getattr__ branch alone, exactly
        # like rebalance–ledger, and the protected weights pin
        # (test_weight_contracts) stays green.
        assert sorted(pl.__all__) == ["__version__", "data", "weights"]
        with pytest.raises(AttributeError) as caught:
            pl.definitely_not_a_portlearn_attribute  # noqa: B018 -- deliberate miss
        if "strategies" not in str(caught.value):
            pytest.fail(
                "EXPECTED_REJECTION the root facade's unknown-attribute error "
                "does not yet name 'strategies' among the lazy surfaces; "
                f"got: {caught.value}",
                pytrace=False,
            )


def test_default_construction_yields_deciding_strategy() -> None:
    """``EqualWeight()`` constructs with no arguments and ``decide`` on a
    plain universe returns a ``DecisionResult`` whose decision targets
    exactly the caller-declared universe (the F3 law at n=3)."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    result = strategy.decide(_context(("AAPL", "MSFT", "NVDA")))
    assert isinstance(result, DecisionResult)
    assert dict(result.decision.target_weights) == {
        "AAPL": 0.3333333333333333,
        "MSFT": 0.3333333333333333,
        "NVDA": 0.3333333333333333,
    }


# --------------------------------------------------------------------------- #
# F2 — protocol composition (behavioral Strategy-protocol satisfaction)        #
# --------------------------------------------------------------------------- #


def test_decide_signature_accepts_exactly_one_context_positional() -> None:
    """The ``Strategy`` protocol is structural-only; this floor
    pins the behavior: ``decide`` accepts exactly one positional
    argument (a ``DecisionContext``) — a zero-arg or extra-argument
    call is rejected by the signature itself."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    signature = inspect.signature(strategy.decide)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert len(positional) == 1, (
        "decide must accept exactly one positional DecisionContext; the "
        "forecast-only or multi-argument alternatives are not lawful"
    )
    # Behavioral: the one-argument call decides (already pinned at n=2).
    result = strategy.decide(_context(("AAPL", "MSFT")))
    assert isinstance(result, DecisionResult)


def test_decision_follows_context_chronology_laws() -> None:
    """The returned decision is anchored at the context's authoritative
    ``decision_time`` and never executes before it."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    result = strategy.decide(_context(("AAPL", "MSFT"), decision_time=_T1))
    assert result.decision.decision_time == _T1
    assert result.decision.execution_time >= result.decision.decision_time


# --------------------------------------------------------------------------- #
# F3 — dynamic full-universe target (hand-pinned exactness)                    #
# --------------------------------------------------------------------------- #


def test_dynamic_universe_allocates_exact_thirds() -> None:
    """Universe ``("AAPL","MSFT","NVDA")`` decides the exact hand-pinned
    1/3 book: key set exactly the universe, order preserved, every
    weight the binary double nearest 1/3 — even though the held book
    (CASH) shares no member with the universe (holdings-vs-universe)."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    target = _target_of(strategy, ("AAPL", "MSFT", "NVDA"))
    assert list(target) == ["AAPL", "MSFT", "NVDA"]
    assert set(target) == {"AAPL", "MSFT", "NVDA"}
    assert target["AAPL"] == 0.3333333333333333
    assert target["MSFT"] == 0.3333333333333333
    assert target["NVDA"] == 0.3333333333333333


def test_dynamic_support_ignores_holdings_outside_the_universe() -> None:
    """The dynamic support is the caller-declared universe, never the
    current-holdings keys: a pre-decision book holding ``BOND`` (with
    ``BOND`` outside the universe) must not leak into the target."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    holdings = PortfolioWeights({"BOND": 0.6, "AAPL": 0.4}, WeightState.PRE_TRADE)
    target = _target_of(strategy, ("AAPL", "MSFT", "NVDA"), current_weights=holdings)
    assert set(target) == {"AAPL", "MSFT", "NVDA"}
    assert "BOND" not in target
    assert target["AAPL"] == 0.3333333333333333
    assert target["MSFT"] == 0.3333333333333333
    assert target["NVDA"] == 0.3333333333333333


# --------------------------------------------------------------------------- #
# F4 — declared support                                                        #
# --------------------------------------------------------------------------- #


def test_declared_support_decides_half_over_two_names_in_order() -> None:
    """``EqualWeight(declared_support=("MSFT","AAPL"))`` over a 3-name
    universe targets exactly the declared two names, in declared
    order, each exactly 0.5; the third universe name is absent."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight(declared_support=("MSFT", "AAPL"))
    target = _target_of(strategy, ("AAPL", "MSFT", "NVDA"))
    assert list(target) == ["MSFT", "AAPL"]
    assert target["MSFT"] == 0.5
    assert target["AAPL"] == 0.5
    assert "NVDA" not in target


# --------------------------------------------------------------------------- #
# F5 — changed universe (value-bearing dynamics)                               #
# --------------------------------------------------------------------------- #


def test_same_instance_adapts_target_to_each_decision_universe() -> None:
    """One default instance decides 1/3 each on a 3-name universe at
    T0 and 1/4 each on a 4-name universe at T1; a declared-support
    sibling stays pinned to its declared 2-name book at both instants
    — the universe is carried by the context, not the instance."""
    EqualWeight = _require_equal_weight()
    dynamic = EqualWeight()
    declared = EqualWeight(declared_support=("MSFT", "AAPL"))
    three = _target_of(dynamic, ("AAPL", "MSFT", "NVDA"), decision_time=_T0)
    four = _target_of(
        dynamic,
        ("AAPL", "MSFT", "NVDA", "TSM"),
        decision_time=_T1,
    )
    assert dict(three) == {
        "AAPL": 0.3333333333333333,
        "MSFT": 0.3333333333333333,
        "NVDA": 0.3333333333333333,
    }
    assert dict(four) == {
        "AAPL": 0.25,
        "MSFT": 0.25,
        "NVDA": 0.25,
        "TSM": 0.25,
    }
    sibling_target = _target_of(declared, ("AAPL", "MSFT", "NVDA"))
    assert dict(sibling_target) == {"MSFT": 0.5, "AAPL": 0.5}


# --------------------------------------------------------------------------- #
# F6 — invalid declarations and the declared_support snapshot/readonly laws    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_declaration",
    [
        pytest.param("MSFT", id="bare-string"),
        pytest.param((), id="empty-tuple"),
        pytest.param(("MSFT", "  "), id="blank-entry"),
        pytest.param(("MSFT", 5), id="non-string-entry"),
        pytest.param(("AAPL", "AAPL"), id="duplicate-entry"),
    ],
)
def test_constructor_rejects_malformed_declared_support(bad_declaration) -> None:
    """Each constructor law its own floor: a bare string, an empty
    tuple, a blank entry, a non-string entry, and a duplicate entry
    each reject with a ``ValueError`` naming the law."""
    EqualWeight = _require_equal_weight()
    with pytest.raises(ValueError) as caught:
        EqualWeight(declared_support=bad_declaration)
    message = str(caught.value).lower()
    assert any(
        token in message
        for token in (
            "declared_support",
            "declared support",
            "identifier",
            "duplicate",
            "string",
        )
    ), f"the rejection must name the violated declaration law; got: {caught.value}"


def test_constructor_accepts_generic_iterable_and_snapshots_it() -> None:
    """A tuple (generic iterable) is accepted; the caller's list mutated
    after construction cannot change a later decision — the
    constructor snapshots the declaration."""
    EqualWeight = _require_equal_weight()
    caller_list = ["MSFT", "AAPL"]
    strategy = EqualWeight(declared_support=caller_list)
    caller_list.append("NVDA")
    target = _target_of(strategy, ("AAPL", "MSFT", "NVDA"))
    assert list(target) == ["MSFT", "AAPL"]
    assert target["MSFT"] == 0.5
    assert target["AAPL"] == 0.5
    assert "NVDA" not in target


def test_declared_support_attribute_is_read_only_snapshot() -> None:
    """``declared_support`` exposes the immutable snapshot tuple
    (``None`` in dynamic mode) and direct reassignment raises
    ``AttributeError`` because no setter exists."""
    EqualWeight = _require_equal_weight()
    declared = EqualWeight(declared_support=("MSFT", "AAPL"))
    dynamic = EqualWeight()
    assert declared.declared_support == ("MSFT", "AAPL")
    assert dynamic.declared_support is None
    with pytest.raises(AttributeError):
        declared.declared_support = ("AAPL",)


# --------------------------------------------------------------------------- #
# F7 — stale declared support                                                  #
# --------------------------------------------------------------------------- #


def test_stale_declared_member_absent_from_universe_rejects() -> None:
    """A declared member absent from the current universe is stale: the
    decision rejects with a ``ValueError`` naming the absent member and
    carrying the ``decision_time``."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight(declared_support=("AAPL", "TSLA"))
    with pytest.raises(ValueError) as caught:
        strategy.decide(_context(("AAPL", "MSFT")))
    assert "TSLA" in str(caught.value)
    assert "2026-06-30" in str(caught.value)


def test_staleness_is_per_decision_not_sticky() -> None:
    """A stale rejection is not a carried failure state: the same
    instance decides successfully on a sibling context whose universe
    re-admits the declared member."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight(declared_support=("AAPL", "TSLA"))
    with pytest.raises(ValueError):
        strategy.decide(_context(("AAPL", "MSFT")))
    target = _target_of(strategy, ("MSFT", "TSLA", "AAPL"))
    assert dict(target) == {"AAPL": 0.5, "TSLA": 0.5}


# --------------------------------------------------------------------------- #
# F8 — carried state                                                           #
# --------------------------------------------------------------------------- #


def test_next_strategy_state_is_none_for_every_decision() -> None:
    """The stateless law: ``next_strategy_state`` is exactly ``None``
    on every result — for a dynamic instance, a declared instance, and
    across consecutive decisions (any non-None state at any point is a
    violation)."""
    EqualWeight = _require_equal_weight()
    dynamic = EqualWeight()
    declared = EqualWeight(declared_support=("MSFT", "AAPL"))
    first_result = dynamic.decide(_context(("AAPL", "MSFT")))
    assert first_result.next_strategy_state is None
    assert declared.decide(_context(("MSFT", "AAPL"))).next_strategy_state is None
    second = dynamic.decide(
        _context(
            ("AAPL", "MSFT", "NVDA"),
            strategy_state=first_result.next_strategy_state,
        )
    )
    assert second.next_strategy_state is None


def test_context_strategy_state_is_ignored_not_mutated() -> None:
    """The carried ``strategy_state`` has no effect on the decision and
    is never mutated in place (the fixed context law: transitions are
    represented solely through ``next_strategy_state``)."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    sentinel_state = {"carry": 7}
    with_state = strategy.decide(
        _context(("AAPL", "MSFT"), strategy_state=sentinel_state)
    )
    without_state = strategy.decide(_context(("AAPL", "MSFT")))
    assert dict(with_state.decision.target_weights) == dict(
        without_state.decision.target_weights
    )
    assert sentinel_state == {"carry": 7}


# --------------------------------------------------------------------------- #
# F9 — forecast conditioning                                                   #
# --------------------------------------------------------------------------- #


def test_present_forecast_does_not_condition_the_decision() -> None:
    """A ``Forecast`` on the context never changes the target: same
    decision with and without a lawful forecast at ``decision_time``.
    The forecast must be dated exactly at the context's
    ``decision_time`` or the context itself rejects (fixed law)."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    plain = strategy.decide(_context(("AAPL", "MSFT")))
    forecast = Forecast(
        values={"AAPL": -0.9, "MSFT": 12.5},
        target="expected_return",
        decision_time=_T0,
        produced_by="test-forecaster",
    )
    conditioned = strategy.decide(_context(("AAPL", "MSFT"), forecast=forecast))
    assert dict(conditioned.decision.target_weights) == dict(
        plain.decision.target_weights
    )


# --------------------------------------------------------------------------- #
# F10 — composition with the fixed decision validator (public-path floor)     #
# --------------------------------------------------------------------------- #


def test_decide_calls_public_validator_before_returning(monkeypatch) -> None:
    """The sanctioned causal floor (spec cure): ``decide`` invokes the
    PUBLIC ``portlearn.interfaces.require_decision_result_compatible``
    exactly once, on its own ``(context, result)`` pair, before
    returning — proven causally by instrumenting the public attribute
    through pytest's auto-undone monkeypatch. Two controlled forms:
    the sentinel-raise form (a called validator's exception propagates
    out of ``decide``) and the recording form (exactly one recorded
    pair whose context/result are identical (``is``) to the caller's
    context and the returned result)."""
    import portlearn.interfaces as pl_iface

    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    context = _context(("AAPL", "MSFT"))

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


def test_decide_returns_validator_compatible_result_on_plain_path() -> None:
    """On the uninstrumented public path, the returned result satisfies
    the strict validator (the composite floor: compose, never restated)."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    context = _context(("AAPL", "MSFT", "NVDA"))
    result = strategy.decide(context)
    assert require_decision_result_compatible(context, result) is None


def test_result_with_present_forecast_is_forecast_compatible() -> None:
    """When a lawful forecast is present on the context, the decided
    result also satisfies the fixed forecast-compatibility validator
    (compose, never restated)."""
    from portlearn.interfaces import require_forecast_decision_compatible

    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    forecast = Forecast(
        values={"AAPL": 0.1, "MSFT": -0.2},
        target="expected_return",
        decision_time=_T0,
        produced_by="test-forecaster",
    )
    context = _context(("AAPL", "MSFT"), forecast=forecast)
    result = strategy.decide(context)
    assert require_forecast_decision_compatible(forecast, result.decision) is None


# --------------------------------------------------------------------------- #
# F11 — pure-function law (independent identical calls)                        #
# --------------------------------------------------------------------------- #


def test_repeated_decisions_are_independent_and_identical() -> None:
    """One instance called twice with a value-equal context decides
    value-equal results (independence, determinism), and no caller-
    visible object is mutated by deciding (input purity)."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    first = strategy.decide(_context(("AAPL", "MSFT", "NVDA")))
    second = strategy.decide(_context(("AAPL", "MSFT", "NVDA")))
    assert first == second
    assert first is not second
    assert dict(first.decision.target_weights) == dict(second.decision.target_weights)


def test_deciding_does_not_mutate_the_context_or_information_set() -> None:
    """Input purity at the object level: after deciding, the context's
    fields are unchanged and the admitted information set still
    iterates exactly its admitted records (length and as_of)."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    information = InformationSet(
        (
            TimedObservation(
                series_id="SYNTH/S",
                observation_time=_T0,
                available_time=_T0,
                value=1.0,
            ),
        ),
        as_of=_T0,
    )
    context = _context(("AAPL", "MSFT"), information=information)
    before = (context.decision_time, tuple(context.universe), context.strategy_state)
    strategy.decide(context)
    after = (context.decision_time, tuple(context.universe), context.strategy_state)
    assert before == after
    assert len(list(information)) == 1
    assert information.as_of == _T0


# --------------------------------------------------------------------------- #
# F12 — validator-passing TARGET (public weights-chain floor)                  #
# --------------------------------------------------------------------------- #


def test_decided_book_passes_default_constraint_validation_as_target() -> None:
    """The decided target book lifted into a ``PortfolioWeights``
    TARGET passes the fixed default-constraint validator
    (``WeightConstraints()``: long-only, budget 1.0, tolerance 1e-9) —
    the round-trip through the public weights chain holds for n=2..5
    and n=48 (the leap in the floor)."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    universes = [
        ("AAPL", "MSFT"),
        ("AAPL", "MSFT", "NVDA"),
        ("AAPL", "MSFT", "NVDA", "TSM"),
        ("AAPL", "MSFT", "NVDA", "TSM", "AVGO"),
        tuple(f"A{i:02d}" for i in range(1, 49)),
    ]
    for universe in universes:
        target = _target_of(strategy, universe)
        book = PortfolioWeights(dict(target), WeightState.TARGET)
        assert require_valid_target(book, WeightConstraints()) is None
        # Budget exactness: the hand-pinned reciprocals sum to 1 within
        # the fixed tolerance (independent arithmetic, not the
        # implementation's own).
        assert abs(math.fsum(target.values()) - 1.0) <= 1e-9


def test_decided_book_passes_validation_for_declared_support() -> None:
    """The declared-support target (0.5/0.5) passes the fixed default
    constraint validation as a TARGET, and dynamic-mode n=2 does too
    (both lawful modes sit inside the constraint envelope)."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight(declared_support=("MSFT", "AAPL"))
    target = _target_of(strategy, ("AAPL", "MSFT", "NVDA"))
    book = PortfolioWeights(dict(target), WeightState.TARGET)
    assert require_valid_target(book, WeightConstraints()) is None
    dynamic = EqualWeight()
    dynamic_target = _target_of(dynamic, ("AAPL", "MSFT"))
    dynamic_book = PortfolioWeights(dict(dynamic_target), WeightState.TARGET)
    assert require_valid_target(dynamic_book, WeightConstraints()) is None


# --------------------------------------------------------------------------- #
# F8 — unconditional empty effective support (dynamic mode)                     #
# --------------------------------------------------------------------------- #


def test_empty_universe_rejects_with_value_error_naming_decision_time() -> None:
    """Dynamic mode, ``context.universe == ()``: ``decide`` rejects with
    ``ValueError`` whose message carries the ``decision_time`` — no
    partial target, no empty-book return (unconditional)."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    with pytest.raises(ValueError) as caught:
        strategy.decide(_context(()))
    message = str(caught.value)
    assert "2026-06-30" in message, (
        f"the empty-support rejection must name the decision instant; got: {message}"
    )


def test_negative_control_empty_universe_context_is_itself_lawful() -> None:
    """PASS_CONTROL: the empty-universe context constructs fine under
    the fixed interface (empty is admissible there) — pinning that the
    F8 rejection is the strategy's own unconditional law, not smuggled
    from the context constructor."""
    context = _context(())
    assert tuple(context.universe) == ()
    assert len(context.universe) == 0
    assert context.decision_time == _T0


# --------------------------------------------------------------------------- #
# F9 — exact per-weight 1/n and default-constraint validity                    #
# --------------------------------------------------------------------------- #

#: Hand-pinned exact binary doubles nearest 1/n (never computed here).
_ONE_OVER_N = {
    2: 0.5,
    3: 0.3333333333333333,
    4: 0.25,
    5: 0.2,
    48: 0.020833333333333332,
}


def _universe_of_size(n: int) -> tuple[str, ...]:
    return tuple(f"ASSET_{index:02d}" for index in range(1, n + 1))


@pytest.mark.parametrize("n", [2, 3, 4, 5, 48])
def test_per_weight_is_exactly_one_over_n_and_validates(n) -> None:
    """The per-weight exactness contract: for n in {2,3,4,5,48} every
    weight equals the hand-pinned binary double nearest 1/n by ``==``,
    the key set is exactly the universe, and the lifted TARGET book
    passes the fixed default-constraint validator (long-only, budget
    1.0, tolerance 1e-9). No exact-total summation is asserted
    anywhere (the fixed floor law)."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    universe = _universe_of_size(n)
    target = _target_of(strategy, universe)
    assert set(target) == set(universe)
    expected_weight = _ONE_OVER_N[n]
    for identifier in universe:
        assert target[identifier] == expected_weight, (
            f"weight on {identifier!r} must be exactly {expected_weight!r} "
            f"at n={n}; got {target[identifier]!r}"
        )
    book = PortfolioWeights(dict(target), WeightState.TARGET)
    assert require_valid_target(book, WeightConstraints()) is None


# --------------------------------------------------------------------------- #
# F13 — default-constraint validation, negative controls                       #
# --------------------------------------------------------------------------- #


def test_negative_control_hand_broken_books_reject_the_validator() -> None:
    """PASS_CONTROL: the validator path has teeth — a hand-broken short
    book (budget 1.0 but a negative weight) and a hand-broken
    budget-drift book (long-only but net 0.9) each reject
    ``require_valid_target`` under the fixed default constraints.
    These are properties of the strict validator, pinning that F9's
    pass is meaningful; they are not evidence about the strategy."""
    short_book = PortfolioWeights({"AAPL": -0.25, "MSFT": 1.25}, WeightState.TARGET)
    with pytest.raises(ValueError):
        require_valid_target(short_book, WeightConstraints())
    drifted_book = PortfolioWeights({"AAPL": 0.5, "MSFT": 0.4}, WeightState.TARGET)
    with pytest.raises(ValueError):
        require_valid_target(drifted_book, WeightConstraints())


# --------------------------------------------------------------------------- #
# F10 — chronology and result validation                                       #
# --------------------------------------------------------------------------- #


def test_decision_is_anchored_and_executes_at_the_decision_instant() -> None:
    """The returned decision carries ``decision_time ==
    context.decision_time`` exactly and the fixed same-instant
    convention ``execution_time == decision_time`` (; both
    ``>`` and ``<`` deviations are violations — the validator only
    guarantees ``>=``, so the exact equality is this floor's own)."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    for instant in (_T0, _T1):
        result = strategy.decide(_context(("AAPL", "MSFT"), decision_time=instant))
        assert result.decision.decision_time == instant
        assert result.decision.execution_time == instant


def test_negative_control_misanchored_result_rejects_the_validator() -> None:
    """PASS_CONTROL (independent validator teeth, explicitly NOT
    evidence about ``decide``): a hand-mis-anchored result whose
    decision_time is one month earlier than the context's rejects with
    ``InvalidChronologyError`` when validated directly, and a
    hand-earlier execution_time rejects the same way."""
    context = _context(("AAPL", "MSFT"), decision_time=_T1)
    earlier = datetime(2026, 5, 31, tzinfo=UTC)
    misanchored = DecisionResult(
        decision=PortfolioDecision(
            decision_time=earlier,
            execution_time=earlier,
            target_weights={"AAPL": 0.5, "MSFT": 0.5},
        )
    )
    with pytest.raises(InvalidChronologyError):
        require_decision_result_compatible(context, misanchored)
    # A decision whose execution precedes its own decision instant
    # cannot even be CONSTRUCTED: the immutable PortfolioDecision checks
    # its own chronology law first (the validator's execution floor is
    # therefore unreachable by construction through public surfaces).
    with pytest.raises(InvalidChronologyError):
        PortfolioDecision(
            decision_time=_T1,
            execution_time=earlier,
            target_weights={"AAPL": 0.5, "MSFT": 0.5},
        )


# --------------------------------------------------------------------------- #
# F11 — forecast independence / no fake forecast                               #
# --------------------------------------------------------------------------- #


def test_no_forecast_context_decides_successfully() -> None:
    """``forecast=None`` decides successfully — the plain path is a
    first-class lawful mode (never a TypeError on missing forecast)."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    result = strategy.decide(_context(("AAPL", "MSFT")))
    assert isinstance(result, DecisionResult)
    assert dict(result.decision.target_weights) == {
        "AAPL": 0.5,
        "MSFT": 0.5,
    }


def test_forecast_carrying_context_decides_the_same_target() -> None:
    """The paired-context behavioral law: a real forecast carried by
    the context decides to the same target as the ``forecast=None``
    sibling; and the strategy creates no ``Forecast`` object of its
    own (nothing forecast-shaped is attached to the result)."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    plain = strategy.decide(_context(("AAPL", "MSFT")))
    skewed = Forecast(
        values={"AAPL": -0.9, "MSFT": 12.5},
        target="expected_return",
        decision_time=_T0,
        produced_by="test-forecaster",
    )
    conditioned = strategy.decide(_context(("AAPL", "MSFT"), forecast=skewed))
    assert dict(conditioned.decision.target_weights) == dict(
        plain.decision.target_weights
    )
    assert not isinstance(conditioned, Forecast)
    assert type(conditioned) is DecisionResult


# --------------------------------------------------------------------------- #
# F12 — stateless / non-mutation                                               #
# --------------------------------------------------------------------------- #


def test_next_strategy_state_is_none_on_every_decision() -> None:
    """``next_strategy_state is None`` on every decide — dynamic and
    declared modes, and across repeated decisions."""
    EqualWeight = _require_equal_weight()
    dynamic = EqualWeight()
    declared = EqualWeight(declared_support=("MSFT", "AAPL"))
    first = dynamic.decide(_context(("AAPL", "MSFT")))
    assert first.next_strategy_state is None
    assert declared.decide(_context(("MSFT", "AAPL"))).next_strategy_state is None
    second = dynamic.decide(
        _context(("AAPL", "MSFT", "NVDA"), strategy_state=first.next_strategy_state)
    )
    assert second.next_strategy_state is None


def test_carried_sentinel_state_is_unchanged_and_non_influencing() -> None:
    """A carried ``strategy_state`` sentinel dict is identical
    after decide and the decided target is identical with and without
    it (behavioral independence)."""
    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    sentinel = {"carry": 7, "nested": {"depth": 1}}
    without = strategy.decide(_context(("AAPL", "MSFT")))
    with_state = strategy.decide(_context(("AAPL", "MSFT"), strategy_state=sentinel))
    assert sentinel == {"carry": 7, "nested": {"depth": 1}}
    assert dict(with_state.decision.target_weights) == dict(
        without.decision.target_weights
    )


def test_two_independent_instances_decide_value_equal_results() -> None:
    """Two independently constructed instances decide value-equal
    results on value-equal contexts (no per-instance residue)."""
    EqualWeight = _require_equal_weight()
    first = EqualWeight().decide(_context(("AAPL", "MSFT", "NVDA")))
    second = EqualWeight().decide(_context(("AAPL", "MSFT", "NVDA")))
    assert first == second


# --------------------------------------------------------------------------- #
# F14 — accounting integration (hand-pinned wealth path)                       #
# --------------------------------------------------------------------------- #

#: Dyadic-exact growth supplies (exact-universe per segment, the drift law).
_F14_GROWTH = {
    _T0: {"AAPL": 1.0625, "MSFT": 1.1875},
    _T1: {"AAPL": 1.25, "MSFT": 0.75},
}


def test_target_flows_through_public_accounting_into_hand_pinned_wealth() -> None:
    """The decided target flows through the public accounting
    stack end-to-end: lift to ``PortfolioWeights(·, TARGET)``, pass
    ``require_valid_target`` under default constraints, decide twice
    (T0, T1) with same-instant execution, hand-drift the T0 target to
    the T1 pre-trade book via the public drift operation, then compose
    ``build_ledger`` with ``Proportional(0.001, one_way)``. Expected
    final wealth is hand-derived independently in the comments below
    (all dyadic-exact binary doubles):

    Period [T0,T1):
      trade at T0 from {CASH:1.0} to {AAPL:0.5, MSFT:0.5}:
        deltas CASH −1.0, A +0.5, M +0.5 → two_sided 2.0 → one_way 1.0
        q0 = 0.001·1.0 → F_cost,0 = 1 − 0.001 = 0.999
      drift over [T0,T1): D = 0.5·1.0625 + 0.5·1.1875 = 1.125
      net_1 = 1.125 · 0.999 = 1.123875
    Period [T1,T2):
      trade at T1 from drifted {17/36, 19/36} back to {0.5, 0.5}:
        |Δ| = 1/36 each → two_sided 1/18 → one_way 1/36
        q1 = 0.001/36 → F_cost,1 = 1 − 0.001/36
      drift over [T1,T2): D = 0.5·1.25 + 0.5·0.75 = 1.0
      net_2 = 1.0 · (1 − 0.001/36)
    final = 1 · 1.123875 · (1 − 0.001/36) = 1.12384378125 exactly.
    """
    from portlearn.rebalance import drift_weights

    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    holdings_t0 = PortfolioWeights({"CASH": 1.0}, WeightState.PRE_TRADE)
    result_t0 = strategy.decide(
        _context(("AAPL", "MSFT"), decision_time=_T0, current_weights=holdings_t0)
    )
    target_t0 = dict(result_t0.decision.target_weights)
    # F14's own requirement: the lifted book validates (public chain).
    lifted = PortfolioWeights(dict(target_t0), WeightState.TARGET)
    assert require_valid_target(lifted, WeightConstraints()) is None
    # The T1 pre-trade book: the public drift of the executed T0 target.
    drifted = drift_weights(
        PortfolioWeights(dict(target_t0), WeightState.POST_TRADE),
        _F14_GROWTH[_T0],
    )
    result_t1 = strategy.decide(
        _context(
            ("AAPL", "MSFT"),
            decision_time=_T1,
            current_weights=drifted,
        )
    )
    path = build_ledger(
        initial_weights={"CASH": 1.0},
        accounting_instants=(_T0, _T1, _T2),
        growth_factors=dict(_F14_GROWTH),
        decisions=(result_t0.decision, result_t1.decision),
        cost_model=Proportional(0.001, turnover=one_way),
    )
    assert path.n_periods == 2
    assert len(path.rows) == 2
    assert path.final_wealth == 1.12384378125, (
        f"final wealth must be the hand-derived 1.12384378125 exactly; "
        f"got {path.final_wealth!r}"
    )
    # Per-period hand-pinned identities (dyadic-exact).
    assert path.rows[0].gross_return == 0.125
    assert path.rows[1].gross_return == 0.0
    assert path.rows[0].turnover == 1.0
    assert path.rows[1].turnover == pytest.approx(1 / 36)
    assert path.rows[0].closing_weights["AAPL"] == 0.4722222222222222
    assert path.rows[0].closing_weights["MSFT"] == 0.5277777777777778


def test_equal_weight_decisions_compose_with_exact_fill_engine() -> None:
    """The default engine path also composes: the same two decisions
    under ``ExactFillAccounting()`` (the public default) produce the
    same hand-pinned wealth path — POST_TRADE = TARGET throughout."""
    from portlearn.ledger import ExactFillAccounting
    from portlearn.rebalance import drift_weights

    EqualWeight = _require_equal_weight()
    strategy = EqualWeight()
    holdings_t0 = PortfolioWeights({"CASH": 1.0}, WeightState.PRE_TRADE)
    result_t0 = strategy.decide(
        _context(("AAPL", "MSFT"), decision_time=_T0, current_weights=holdings_t0)
    )
    drifted = drift_weights(
        PortfolioWeights(
            dict(result_t0.decision.target_weights), WeightState.POST_TRADE
        ),
        _F14_GROWTH[_T0],
    )
    result_t1 = strategy.decide(
        _context(("AAPL", "MSFT"), decision_time=_T1, current_weights=drifted)
    )
    path = build_ledger(
        initial_weights={"CASH": 1.0},
        accounting_instants=(_T0, _T1, _T2),
        growth_factors=dict(_F14_GROWTH),
        decisions=(result_t0.decision, result_t1.decision),
        cost_model=Proportional(0.001, turnover=one_way),
        accounting_engine=ExactFillAccounting(),
    )
    assert path.final_wealth == 1.12384378125


# --------------------------------------------------------------------------- #
# F15 — determinism and constructor-input snapshot                             #
# --------------------------------------------------------------------------- #


def test_repeated_decide_on_same_context_is_stable() -> None:
    """Determinism: repeated ``decide`` on the same context instance
    decides value-equal results across repeated calls (no hidden
    per-call mutation drift), both modes."""
    EqualWeight = _require_equal_weight()
    for strategy in (EqualWeight(), EqualWeight(declared_support=("MSFT", "AAPL"))):
        context = _context(("AAPL", "MSFT", "NVDA"))
        first = strategy.decide(context)
        for _ in range(5):
            again = strategy.decide(context)
            assert again == first


def test_constructor_source_list_mutation_cannot_alter_behavior() -> None:
    """Snapshot law: constructing from a list then mutating the list
    afterwards cannot alter the decided target (the constructor
    snapshots its input)."""
    EqualWeight = _require_equal_weight()
    source = ["MSFT", "AAPL"]
    strategy = EqualWeight(declared_support=source)
    source[:] = ["NVDA", "TSM", "AVGO"]
    source.append("ORCL")
    result = strategy.decide(_context(("NVDA", "TSM", "AVGO", "ORCL", "MSFT", "AAPL")))
    assert dict(result.decision.target_weights) == {
        "MSFT": 0.5,
        "AAPL": 0.5,
    }


def test_constructor_source_list_mutation_cannot_alter_the_exposed_snapshot() -> None:
    """The exposed ``declared_support`` snapshot itself is immune to
    later mutation of the constructor source (F15 second sub-floor)."""
    EqualWeight = _require_equal_weight()
    source = ["MSFT", "AAPL"]
    strategy = EqualWeight(declared_support=source)
    source[:] = ["NVDA", "TSM"]
    assert strategy.declared_support == ("MSFT", "AAPL")


# --------------------------------------------------------------------------- #
# F16 — package import side-effect regression (fresh interpreter)              #
# --------------------------------------------------------------------------- #

#: The F16 probe source, run by a fresh interpreter whose import path
#: puts the source tree that provided THIS session's ``portlearn``
#: first; it prints ``key=value`` facts on stdout.
_PACKAGE_PROBE = """
import importlib
import sys

import portlearn

registered = sorted(
    name
    for name in sys.modules
    if name == 'portlearn' or name.startswith('portlearn.')
)
print('registered=' + ','.join(registered))
print(
    'weights_identity='
    + str(
        portlearn.weights is importlib.import_module('portlearn.weights')
    )
)
try:
    resolved = portlearn.strategies
except AttributeError as error:
    print('strategies_identity=False')
    print('strategies_error=' + str(error))
else:
    import portlearn.strategies as direct

    print('strategies_identity=' + str(resolved is direct))
print('all=' + ','.join(sorted(portlearn.__all__)))
try:
    portlearn.definitely_not_a_portlearn_attribute
except AttributeError as error:
    print(
        'error_names_strategies='
        + str('strategies' in str(error))
    )
"""


def _fresh_interpreter_probe_facts() -> dict:
    """Run the F16 probe in a fresh interpreter and return its printed
    ``key=value`` facts (the probe script mirrors the fixed
    fresh-process discipline: import side effects are only observable
    in a process that has not yet imported the package)."""
    import os
    import subprocess
    import sys

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
    """F16, lazy side: in a fresh interpreter a bare ``import portlearn``
    registers exactly the root package in ``sys.modules`` (no eager
    submodule — ``strategies`` included once it exists), and the fixed
    ``weights`` facade still resolves identically."""
    facts = _fresh_interpreter_probe_facts()
    assert facts.get("registered") == "portlearn", facts
    assert facts.get("weights_identity") == "True", facts


def test_fresh_interpreter_facade_exposes_strategies_identically() -> None:
    """F16, facade side: the same fresh interpreter resolves
    ``portlearn.strategies`` to exactly the module a direct import
    yields, ``pl.__all__`` carries the strategies entry, and the
    unknown-attribute error message names ``strategies``."""
    facts = _fresh_interpreter_probe_facts()
    assert facts.get("strategies_identity") == "True", (
        "the lazy facade must resolve portlearn.strategies identically "
        f"to a direct import; probe facts: {facts}"
    )
    assert facts.get("all") == "__version__,data,weights", (
        "the fixed root __all__ must stay unmodified (strategies rides "
        f"the __getattr__ branch alone); probe facts: {facts}"
    )
    assert facts.get("error_names_strategies") == "True", (
        "the facade's unknown-attribute error must name 'strategies'; "
        f"probe facts: {facts}"
    )
