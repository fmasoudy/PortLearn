"""Behavior-named conformance suite for the portfolio-weight contract.

These tests freeze the portfolio-weight behavioral contract —
the closed ``WeightState`` vocabulary, the
immutable snapshotted ``PortfolioWeights`` value object, the standing
``WeightConstraints`` declarations with unconditional well-formedness and
scalar declaration coherence, the pure exposure functions, and the
single TARGET-only validator ``require_valid_target``.

Every expected value is hand-computed on exact cases (see each test);
all identifier, numeric, snapshot, and boundary behavior is asserted
against the fixed laws L1–L14, never against the temporary absence or
presence of any unrelated successor implementation file.
"""

from __future__ import annotations

import math
import sys

import pytest

from portlearn.weights import (
    PortfolioWeights,
    WeightConstraints,
    WeightState,
    gross_exposure,
    net_exposure,
    require_valid_target,
)

TOL = 1e-9


# ---------------------------------------------------------------------------
# Construction and laws
# ---------------------------------------------------------------------------


def test_target_weights_round_trip_exact_weights() -> None:
    """A constructed TARGET round-trips its exact weights and state."""
    target = PortfolioWeights(
        {"AAPL": 0.60, "MSFT": 0.25, "CASH-LIKE": 0.15}, WeightState.TARGET
    )
    assert target.state is WeightState.TARGET
    assert dict(target.weights) == {"AAPL": 0.60, "MSFT": 0.25, "CASH-LIKE": 0.15}
    assert target.weight_of("AAPL") == 0.60
    assert target.weight_of("MSFT") == 0.25
    assert target.weight_of("CASH-LIKE") == 0.15
    assert target.assets == frozenset({"AAPL", "MSFT", "CASH-LIKE"})


def test_blank_and_whitespace_identifiers_rejected() -> None:
    """Blank and whitespace-only identifiers are rejected at construction."""
    for bad_identifier in ("", " ", "\t", "\n", " \t\n"):
        with pytest.raises(ValueError, match="identifier"):
            PortfolioWeights({bad_identifier: 0.5}, WeightState.TARGET)


def test_case_distinct_identifiers_both_retained() -> None:
    """Identifiers compare case-sensitively: "AAPL" and "aapl" both stay."""
    target = PortfolioWeights({"AAPL": 0.6, "aapl": 0.4}, WeightState.TARGET)
    assert target.assets == frozenset({"AAPL", "aapl"})
    assert target.weight_of("AAPL") == 0.6
    assert target.weight_of("aapl") == 0.4


def test_source_mapping_mutation_cannot_corrupt_snapshot() -> None:
    """The object snapshots its input; later source mutation is invisible."""
    source = {"AAPL": 0.6, "MSFT": 0.4}
    target = PortfolioWeights(source, WeightState.TARGET)
    source["AAPL"] = 99.0
    source["EVIL"] = -5.0
    del source["MSFT"]
    assert dict(target.weights) == {"AAPL": 0.6, "MSFT": 0.4}
    assert target.weight_of("AAPL") == 0.6
    assert target.weight_of("EVIL") == 0.0
    assert "EVIL" not in target.assets
    assert target.assets == frozenset({"AAPL", "MSFT"})


def test_absent_identifier_has_zero_effective_weight_but_assets_differ_from_explicit_zero() -> None:
    """Absent means effective weight 0.0, yet is distinct from explicit 0.0."""
    empty = PortfolioWeights({}, WeightState.TARGET)
    explicit_zero = PortfolioWeights({"AAPL": 0.0}, WeightState.TARGET)
    assert empty.assets == frozenset()
    assert explicit_zero.assets == frozenset({"AAPL"})
    assert empty.assets != explicit_zero.assets
    assert empty.weight_of("AAPL") == 0.0
    assert explicit_zero.weight_of("AAPL") == 0.0
    assert empty.weight_of("AAPL") == explicit_zero.weight_of("AAPL")


def test_empty_weights_admissible_with_zero_exposures() -> None:
    """The empty mapping is admissible; budget holds only near zero."""
    empty = PortfolioWeights({}, WeightState.TARGET)
    assert gross_exposure(empty.weights) == 0.0
    assert net_exposure(empty.weights) == 0.0
    require_valid_target(empty, WeightConstraints(budget=0.0))
    with pytest.raises(ValueError, match="budget"):
        require_valid_target(empty, WeightConstraints(budget=1.0))


def test_unknown_state_tag_rejected() -> None:
    """The state vocabulary is closed: unknown tags do not exist."""
    assert {member.name for member in WeightState} == {
        "TARGET",
        "PRE_TRADE",
        "POST_TRADE",
    }
    with pytest.raises(ValueError):
        WeightState("SIGNAL")  # unknown value tag is not a member
    with pytest.raises(ValueError, match="state"):
        PortfolioWeights({"AAPL": 0.5}, "SIGNAL")


def test_string_state_tag_rejected_no_coercion() -> None:
    """The string "TARGET", an int, and None are all rejected, no coercion."""
    for bad_state in ("TARGET", "PRE_TRADE", 1, None, True):
        with pytest.raises(ValueError, match="state"):
            PortfolioWeights({"AAPL": 0.5}, bad_state)


# ---------------------------------------------------------------------------
# Adversarial numeric inputs
# ---------------------------------------------------------------------------


def test_nan_inf_weights_rejected() -> None:
    """NaN and ±inf weights are rejected unconditionally at construction."""
    for bad_weight in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="weight"):
            PortfolioWeights({"AAPL": bad_weight}, WeightState.TARGET)
        with pytest.raises(ValueError, match="weight"):
            PortfolioWeights({"AAPL": bad_weight}, WeightState.PRE_TRADE)


def test_bool_weight_rejected_despite_int_subclassing() -> None:
    """A bool weight is rejected as a matter of law, not accident."""
    for bad_weight in (True, False):
        with pytest.raises(ValueError, match="weight"):
            PortfolioWeights({"AAPL": bad_weight}, WeightState.TARGET)


def test_string_weight_rejected() -> None:
    """A string weight is rejected: no numeric coercion of any kind."""
    with pytest.raises(ValueError, match="weight"):
        PortfolioWeights({"AAPL": "0.5"}, WeightState.TARGET)


def test_int_weight_admissible_and_equal_to_float() -> None:
    """An int weight is admissible and numerically equals its float value."""
    target = PortfolioWeights({"AAPL": 1, "MSFT": 0}, WeightState.TARGET)
    assert target.weight_of("AAPL") == 1.0
    assert isinstance(target.weight_of("AAPL"), float)
    as_float = PortfolioWeights({"AAPL": 1.0, "MSFT": 0.0}, WeightState.TARGET)
    assert target.weight_of("AAPL") == as_float.weight_of("AAPL")


# ---------------------------------------------------------------------------
# Huge-integer conversion and aggregation overflow (blocker cure: the
# module error surface is ValueError only, never OverflowError)
# ---------------------------------------------------------------------------


def test_huge_integer_weight_rejected_valueerror_not_overflowerror() -> None:
    """An int weight beyond the float range fails closed with
    ``ValueError`` at every weight-taking public entry — never
    ``OverflowError`` leaking from the int→float conversion."""
    for huge in (10**1000, -(10**1000)):
        with pytest.raises(ValueError, match="weight must be finite"):
            PortfolioWeights({"A": huge}, WeightState.TARGET)
        with pytest.raises(ValueError, match="weight must be finite"):
            gross_exposure({"A": huge})
        with pytest.raises(ValueError, match="weight must be finite"):
            net_exposure({"A": huge})


def test_huge_integer_constraint_scalars_rejected_valueerror() -> None:
    """Every ``WeightConstraints`` numeric declaration beyond the float
    range fails closed with ``ValueError`` naming its field."""
    huge = 10**1000
    with pytest.raises(ValueError, match="budget must be finite"):
        WeightConstraints(budget=huge)
    with pytest.raises(ValueError, match="budget_tolerance must be finite"):
        WeightConstraints(budget_tolerance=huge)
    with pytest.raises(ValueError, match="max_gross_exposure must be finite"):
        WeightConstraints(max_gross_exposure=huge)
    with pytest.raises(
        ValueError, match="max_abs_net_exposure must be finite"
    ):
        WeightConstraints(max_abs_net_exposure=huge)


def test_exposure_aggregation_overflow_rejected_valueerror_never_nonfinite() -> None:
    """Individually finite weights whose exposure sum overflows the
    float range raise ``ValueError`` unconditional — never
    ``OverflowError``, and never a non-finite return value."""
    with pytest.raises(ValueError, match="gross_exposure must be finite"):
        gross_exposure({"A": 1e308, "B": 1e308})
    with pytest.raises(ValueError, match="net_exposure must be finite"):
        net_exposure({"A": 1e308, "B": 1e308})


# ---------------------------------------------------------------------------
# Exposures (hand-computed) and mapping validity
# ---------------------------------------------------------------------------


def test_long_only_gross_equals_net_equals_one() -> None:
    """0.5 + 0.3 + 0.2: gross equals net equals exactly 1.0."""
    weights = {"AAPL": 0.5, "MSFT": 0.3, "BRK.B": 0.2}
    assert gross_exposure(weights) == 1.0
    assert net_exposure(weights) == 1.0


def test_long_short_exposures() -> None:
    """1.2 − 0.4 + 0.2: net 1.0, gross 1.8 (hand-computed)."""
    weights = {"AAPL": 1.2, "MSFT": -0.4, "X": 0.2}
    assert net_exposure(weights) == 1.0
    assert gross_exposure(weights) == 1.8
    assert net_exposure({"AAPL": 1.3, "MSFT": -0.3}) == 1.0


def test_exposure_functions_enforce_valid_weight_mapping() -> None:
    """Exposure functions enforce the same mapping validity as construction."""
    for bad_identifier in (" ", "\t"):
        mapping = {bad_identifier: 0.5}
        with pytest.raises(ValueError, match="identifier"):
            gross_exposure(mapping)
        with pytest.raises(ValueError, match="identifier"):
            net_exposure(mapping)
    for bad_weight in (float("nan"), float("inf"), True, "0.5"):
        mapping = {"AAPL": bad_weight}
        with pytest.raises(ValueError):
            gross_exposure(mapping)
        with pytest.raises(ValueError):
            net_exposure(mapping)


def test_negative_zero_is_not_a_short_position() -> None:
    """−0.0 passes allow_short=False and contributes 0.0 to gross."""
    target = PortfolioWeights(
        {"A": 0.5, "B": 0.5, "C": -0.0}, WeightState.TARGET
    )
    require_valid_target(target, WeightConstraints(allow_short=False, budget=1.0))
    assert gross_exposure(target.weights) == 1.0


# ---------------------------------------------------------------------------
# Constraint bounds on TARGET
# ---------------------------------------------------------------------------


def test_leverage_boundary_is_inclusive() -> None:
    """gross 2.0 passes max_gross_exposure=2.0; 2.0 + 1e-10 fails."""
    at_bound = PortfolioWeights({"A": 1.5, "B": -0.5}, WeightState.TARGET)
    require_valid_target(
        at_bound,
        WeightConstraints(allow_short=True, max_gross_exposure=2.0, budget=1.0),
    )
    over_bound = PortfolioWeights(
        {"A": 1.5 + 1e-10, "B": -0.5}, WeightState.TARGET
    )
    with pytest.raises(ValueError, match="max_gross_exposure"):
        require_valid_target(
            over_bound,
            WeightConstraints(
                allow_short=True, max_gross_exposure=2.0, budget=1.0
            ),
        )


def test_inclusive_max_abs_net_exposure_both_sides() -> None:
    """|net| == max_abs_net_exposure passes on both sides; just outside raises."""
    positive = PortfolioWeights({"A": 0.8}, WeightState.TARGET)
    require_valid_target(
        positive,
        WeightConstraints(
            allow_short=True, max_abs_net_exposure=0.8, budget=0.8
        ),
    )
    negative = PortfolioWeights({"B": -0.8}, WeightState.TARGET)
    require_valid_target(
        negative,
        WeightConstraints(
            allow_short=True, max_abs_net_exposure=0.8, budget=-0.8
        ),
    )
    over_positive = PortfolioWeights({"A": 0.8 + 1e-9}, WeightState.TARGET)
    with pytest.raises(ValueError, match="max_abs_net_exposure"):
        require_valid_target(
            over_positive,
            WeightConstraints(
                allow_short=True,
                max_abs_net_exposure=0.8,
                budget=0.8 + 1e-9,
            ),
        )
    over_negative = PortfolioWeights(
        {"B": -(0.8 + 1e-9)}, WeightState.TARGET
    )
    with pytest.raises(ValueError, match="max_abs_net_exposure"):
        require_valid_target(
            over_negative,
            WeightConstraints(
                allow_short=True,
                max_abs_net_exposure=0.8,
                budget=-(0.8 + 1e-9),
            ),
        )


def test_constraints_rejected_by_target_only_validator_pre_trade() -> None:
    """A PRE_TRADE object never enters the validator: the state check fires."""
    pre_trade = PortfolioWeights({"A": 2.5, "B": -1.0}, WeightState.PRE_TRADE)
    with pytest.raises(ValueError, match="TARGET"):
        require_valid_target(
            pre_trade,
            WeightConstraints(
                allow_short=False, max_gross_exposure=1.0, budget=1.0
            ),
        )


def test_constraints_rejected_by_target_only_validator_post_trade() -> None:
    """A POST_TRADE object never enters the validator: the state check fires."""
    post_trade = PortfolioWeights(
        {"A": 0.99, "B": 0.005}, WeightState.POST_TRADE
    )
    with pytest.raises(ValueError, match="TARGET"):
        require_valid_target(post_trade, WeightConstraints(budget=1.0))


# ---------------------------------------------------------------------------
# Cash (L13)
# ---------------------------------------------------------------------------


def test_cash_as_ordinary_instrument_participates_in_exposures() -> None:
    """A cash-like identifier is an ordinary instrument in the exposures."""
    target = PortfolioWeights(
        {"AAPL": 0.60, "MSFT": 0.25, "CASH-LIKE": 0.15}, WeightState.TARGET
    )
    assert net_exposure(target.weights) == 1.0
    assert gross_exposure(target.weights) == 1.0
    require_valid_target(
        target, WeightConstraints(allow_short=False, budget=1.0)
    )
    assert "CASH-LIKE" in target.assets


def test_off_budget_target_rejected_not_silently_completed_to_cash() -> None:
    """A 0.95 sum under budget=1.0 raises; no residual cash sleeve appears."""
    target = PortfolioWeights({"A": 0.6, "B": 0.35}, WeightState.TARGET)
    with pytest.raises(ValueError, match="budget"):
        require_valid_target(target, WeightConstraints(budget=1.0))
    assert target.assets == frozenset({"A", "B"})
    assert target.weight_of("CASH") == 0.0
    assert net_exposure(target.weights) == 0.95


# ---------------------------------------------------------------------------
# Tolerance (L10, hand-computed edges)
# ---------------------------------------------------------------------------


def test_thirds_target_inside_tolerance() -> None:
    """Three exact thirds net to exactly 1.0 under fsum."""
    target = PortfolioWeights(
        {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}, WeightState.TARGET
    )
    require_valid_target(target, WeightConstraints(budget=1.0))


def test_fsum_decimal_reordering_exactly_one_both_orders() -> None:
    """math.fsum is exact under reordering: both orders give exactly 1.0."""
    assert math.fsum([0.7, 0.2, 0.1]) == 1.0
    assert math.fsum([0.1, 0.2, 0.7]) == 1.0
    forward = {"A": 0.7, "B": 0.2, "C": 0.1}
    reverse = {"C": 0.1, "B": 0.2, "A": 0.7}
    assert net_exposure(forward) == 1.0
    assert net_exposure(reverse) == 1.0
    assert net_exposure(forward) == net_exposure(reverse)


def test_representation_gap_inside_tolerance() -> None:
    """fsum([0.1]*3) == 0.30000000000000004 passes budget=0.3 within 1e-9."""
    assert math.fsum([0.1, 0.1, 0.1]) == 0.30000000000000004
    target = PortfolioWeights(
        {"A": 0.1, "B": 0.1, "C": 0.1}, WeightState.TARGET
    )
    require_valid_target(target, WeightConstraints(budget=0.3))


def test_plus_1e_10_off_budget_inside_tolerance() -> None:
    """A net of 1.0 + 1e-10 passes budget=1.0 under the default tolerance."""
    target = PortfolioWeights(
        {"A": 0.5 + 1e-10, "B": 0.5}, WeightState.TARGET
    )
    require_valid_target(target, WeightConstraints(budget=1.0))


def test_plus_1e_8_off_budget_outside_tolerance() -> None:
    """A net of 1.0 + 1e-8 is outside the 1e-9 tolerance and raises."""
    target = PortfolioWeights({"A": 0.5 + 1e-8, "B": 0.5}, WeightState.TARGET)
    with pytest.raises(ValueError, match="budget"):
        require_valid_target(target, WeightConstraints(budget=1.0))


def test_zero_tolerance_rejected() -> None:
    """budget_tolerance=0.0 is rejected: strictly positive is the law."""
    with pytest.raises(ValueError, match="budget_tolerance"):
        WeightConstraints(budget_tolerance=0.0)


def test_negative_tolerance_rejected() -> None:
    """A negative budget_tolerance is rejected at construction."""
    with pytest.raises(ValueError, match="budget_tolerance"):
        WeightConstraints(budget_tolerance=-1e-9)


def test_non_finite_constraint_bounds_rejected() -> None:
    """NaN/inf exposure bounds are rejected for both bound fields."""
    for non_finite in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="max_gross_exposure"):
            WeightConstraints(max_gross_exposure=non_finite)
        with pytest.raises(ValueError, match="max_abs_net_exposure"):
            WeightConstraints(max_abs_net_exposure=non_finite)


# ---------------------------------------------------------------------------
# State semantics (L1/L9/L12)
# ---------------------------------------------------------------------------


def test_realized_states_rejected_by_target_only_validator() -> None:
    """Realized states are rejected on the state check alone; no wealth or
    normalization claim is made or tested either way."""
    post_trade = PortfolioWeights(
        {"A": 0.99, "B": 0.005}, WeightState.POST_TRADE
    )
    with pytest.raises(ValueError, match="TARGET"):
        require_valid_target(post_trade, WeightConstraints(budget=1.0))
    fully_compliant = PortfolioWeights(
        {"A": 0.6, "B": 0.4}, WeightState.POST_TRADE
    )
    with pytest.raises(ValueError, match="TARGET"):
        require_valid_target(
            fully_compliant,
            WeightConstraints(
                allow_short=False, max_gross_exposure=1.0, budget=1.0
            ),
        )


def test_same_mapping_as_target_violates_budget() -> None:
    """Only the desired allocation is budget-validated: the identical
    mapping as a TARGET under budget=1.0 raises on the budget law."""
    mapping = {"A": 0.99, "B": 0.005}  # net 0.995
    as_target = PortfolioWeights(mapping, WeightState.TARGET)
    with pytest.raises(ValueError, match="budget"):
        require_valid_target(as_target, WeightConstraints(budget=1.0))
    compliant = PortfolioWeights({"A": 1.0, "B": 0.0}, WeightState.TARGET)
    require_valid_target(compliant, WeightConstraints(budget=1.0))


def test_no_operation_derives_realized_states_from_target() -> None:
    """The documented API exposes no operation deriving realized states
    from a TARGET — asserted against the module's documented API, not
    against absent files."""
    import portlearn.weights as weights_module

    documented = [
        "PortfolioWeights",
        "WeightConstraints",
        "WeightState",
        "gross_exposure",
        "net_exposure",
        "require_valid_target",
    ]
    assert list(weights_module.__all__) == documented
    ceiling_imports = frozenset(
        {
            "annotations",
            "enum",
            "math",
            "Mapping",
            "dataclass",
            "MappingProxyType",
            "Any",
        }
    )
    public_names = {
        name for name in dir(weights_module) if not name.startswith("_")
    }
    module_owned = public_names - ceiling_imports
    assert module_owned == set(documented)
    for banned_fragment in (
        "derive",
        "realize",
        "realized",
        "execute",
        "settle",
        "post_trade_from",
        "pre_trade_from",
        "to_post_trade",
        "to_pre_trade",
    ):
        assert not any(
            banned_fragment in name.lower() for name in module_owned
        ), f"no public name may suggest {banned_fragment!r} machinery"


# ---------------------------------------------------------------------------
# Constraint well-formedness (L8, unconditional at construction)
# ---------------------------------------------------------------------------


def test_negative_bound_rejected() -> None:
    """Negative exposure bounds are rejected for both bound fields."""
    with pytest.raises(ValueError, match="max_gross_exposure"):
        WeightConstraints(max_gross_exposure=-0.1)
    with pytest.raises(ValueError, match="max_abs_net_exposure"):
        WeightConstraints(max_abs_net_exposure=-0.1)


def test_bool_constraints_rejected() -> None:
    """A bool is an int subclass and is rejected as a matter of law."""
    with pytest.raises(ValueError, match="allow_short"):
        WeightConstraints(allow_short=1)
    with pytest.raises(ValueError, match="allow_short"):
        WeightConstraints(allow_short="yes")
    with pytest.raises(ValueError, match="budget"):
        WeightConstraints(budget=True)
    with pytest.raises(ValueError, match="max_gross_exposure"):
        WeightConstraints(max_gross_exposure=True)
    with pytest.raises(ValueError, match="max_abs_net_exposure"):
        WeightConstraints(max_abs_net_exposure=True)
    with pytest.raises(ValueError, match="budget_tolerance"):
        WeightConstraints(budget_tolerance=True)


def test_non_finite_budget_rejected() -> None:
    """budget=nan and budget=inf each raise at construction."""
    with pytest.raises(ValueError, match="budget"):
        WeightConstraints(budget=float("nan"))
    with pytest.raises(ValueError, match="budget"):
        WeightConstraints(budget=float("inf"))


# ---------------------------------------------------------------------------
# Declaration coherence (L8, TARGET-law satisfiability)
# ---------------------------------------------------------------------------


def test_declaration_coherence_inclusive_boundary_admissible() -> None:
    """budget == L + 1e-9 constructs in both short modes; −1e-9 long-only."""
    for allow_short in (True, False):
        WeightConstraints(
            allow_short=allow_short,
            max_gross_exposure=1.0,
            budget=1.0 + TOL,
        )
    WeightConstraints(allow_short=False, max_gross_exposure=1.0, budget=-TOL)


def test_declaration_coherence_just_outside_rejected() -> None:
    """Just-outside budgets are contradictory in every mode they bound."""
    for allow_short in (True, False):
        with pytest.raises(ValueError):
            WeightConstraints(
                allow_short=allow_short,
                max_gross_exposure=1.0,
                budget=1.0 + 2 * TOL,
            )
    with pytest.raises(ValueError):
        WeightConstraints(
            allow_short=False, max_gross_exposure=1.0, budget=-2 * TOL
        )
    with pytest.raises(ValueError):
        WeightConstraints(
            allow_short=True,
            max_gross_exposure=1.0,
            budget=-(1.0 + 2 * TOL),
        )


def test_no_declared_exposure_limits_impose_no_ceiling() -> None:
    """No limits declared ⇒ no ceiling: budget=7.5 constructs both modes,
    while long-only sign coherence still applies."""
    WeightConstraints(allow_short=True, budget=7.5)
    WeightConstraints(allow_short=False, budget=7.5)
    with pytest.raises(ValueError):
        WeightConstraints(allow_short=False, budget=-2 * TOL)


def test_declaration_coherence_least_declared_limit_governs() -> None:
    """max_gross=2.0 with max_abs_net=0.8 ⇒ L=0.8 governs the boundary."""
    for allow_short in (True, False):
        WeightConstraints(
            allow_short=allow_short,
            max_gross_exposure=2.0,
            max_abs_net_exposure=0.8,
            budget=0.8,
        )
        WeightConstraints(
            allow_short=allow_short,
            max_gross_exposure=2.0,
            max_abs_net_exposure=0.8,
            budget=0.8 + TOL,
        )
        with pytest.raises(ValueError):
            WeightConstraints(
                allow_short=allow_short,
                max_gross_exposure=2.0,
                max_abs_net_exposure=0.8,
                budget=0.8 + 2 * TOL,
            )


def test_declaration_coherence_zero_ceiling() -> None:
    """max_gross_exposure=0.0 admits only budgets within tolerance of 0."""
    for allow_short in (True, False):
        WeightConstraints(
            allow_short=allow_short, max_gross_exposure=0.0, budget=0.0
        )
        WeightConstraints(
            allow_short=allow_short, max_gross_exposure=0.0, budget=5e-10
        )
        with pytest.raises(ValueError):
            WeightConstraints(
                allow_short=allow_short,
                max_gross_exposure=0.0,
                budget=1e-8,
            )


# ---------------------------------------------------------------------------
# Conformance stand-ins
# ---------------------------------------------------------------------------


def test_external_stand_in_target_weights_pass_construction_and_validation() -> None:
    """A dependency-free external stand-in composes through the public
    constructor and accessors alone: an equal-weight producer over a
    synthetic universe constructs and validates exactly."""

    def equal_weight_stand_in(universe: tuple[str, ...]) -> dict[str, float]:
        count = len(universe)
        return {identifier: 1.0 / count for identifier in universe}

    produced = equal_weight_stand_in(("EQUITY.ASX.WOW", "EQUITY.ASX.CBA"))
    target = PortfolioWeights(produced, WeightState.TARGET)
    mandate = WeightConstraints(allow_short=False, max_gross_exposure=1.0)
    require_valid_target(target, mandate)
    assert target.assets == frozenset({"EQUITY.ASX.WOW", "EQUITY.ASX.CBA"})
    assert target.weight_of("EQUITY.ASX.WOW") == 0.5
    assert net_exposure(target.weights) == 1.0
    assert gross_exposure(target.weights) == 1.0


def test_hostile_stand_in_nan_blank_and_string_state_all_rejected() -> None:
    """A hostile stand-in emitting NaN weights, blank identifiers, and
    string state tags is rejected unconditionally — each hostile emission
    is isolated at its entry point (the constructor's own identifier,
    numeric, and state checks), never silently coerced."""

    def hostile_weights() -> dict[str, float]:
        return {"GOOD": 0.5, " ": 0.25, "BROKEN": float("nan")}

    # Blank identifier and NaN weight emissions, isolated one per call so
    # each check is exercised on its own (dict order: " " first, NaN second).
    with pytest.raises(ValueError, match="identifier"):
        PortfolioWeights({" ": 0.25}, WeightState.TARGET)
    with pytest.raises(ValueError, match="weight"):
        PortfolioWeights({"BROKEN": float("nan")}, WeightState.TARGET)
    # The full hostile payload is rejected unconditionally regardless of which
    # check fires first.
    with pytest.raises(ValueError):
        PortfolioWeights(hostile_weights(), WeightState.TARGET)
    # A string state tag is rejected with no coercion into the vocabulary.
    with pytest.raises(ValueError, match="state"):
        PortfolioWeights({"GOOD": 0.5}, "TARGET")


# ---------------------------------------------------------------------------
# Supporting fixed-surface behaviors (immutability, mutation
# stand-in, and the additive lazy root exposure )
# ---------------------------------------------------------------------------


def test_value_objects_and_snapshot_view_reject_mutation() -> None:
    """Immutable value objects repel attribute and view
    mutation — a stand-in attempting to rewrite a constraints object or
    a snapshotted weights view fails closed."""
    target = PortfolioWeights({"AAPL": 0.6, "MSFT": 0.4}, WeightState.TARGET)
    with pytest.raises(AttributeError):
        target.state = WeightState.POST_TRADE
    with pytest.raises(AttributeError):
        target.weights = {"AAPL": 1.0}
    with pytest.raises(TypeError):
        target.weights["AAPL"] = 0.25
    mandate = WeightConstraints(allow_short=False, budget=1.0)
    with pytest.raises(AttributeError):
        mandate.allow_short = True
    with pytest.raises(AttributeError):
        mandate.budget = 0.5
    assert mandate.allow_short is False
    assert mandate.budget == 1.0


def test_root_package_lazily_exposes_weights_module() -> None:
    """portlearn lazily exposes weights by exactly one additional
    PEP 562 branch — additive only, no eager imports."""
    for name in [
        name for name in list(sys.modules) if name == "portlearn" or name.startswith("portlearn.")
    ]:
        del sys.modules[name]
    import portlearn

    assert sorted(portlearn.__all__) == ["__version__", "data", "weights"]
    portlearn_names = {
        name
        for name in sys.modules
        if name == "portlearn" or name.startswith("portlearn.")
    }
    assert portlearn_names == {"portlearn"}
    weights_module = portlearn.weights
    import portlearn.weights as resolved

    assert weights_module is resolved
    assert resolved.__name__ == "portlearn.weights"
    portlearn_names = {
        name
        for name in sys.modules
        if name == "portlearn" or name.startswith("portlearn.")
    }
    assert portlearn_names == {"portlearn", "portlearn.weights"}


# ---------------------------------------------------------------------------
# L4 immutability, blocker cure: no retained mutable backing alias
# ---------------------------------------------------------------------------


def test_no_mutable_backing_alias_can_alter_public_observable_state() -> None:
    """L4: the object retains no accessible mutable backing dict — the
    known backing-alias attack (write through a retained private snapshot
    slot, e.g. ``x._snapshot['A'] = 9.0``, and ``dict(x.weights)``
    changes) must fail: every public observable stays exactly as
    constructed."""
    target = PortfolioWeights({"A": 1.0, "B": 0.5}, WeightState.TARGET)
    # Attack every private slot the instance actually carries; an
    # implementation that retains a mutable backing dict is falsified here.
    for slot in getattr(type(target), "__slots__", ()):
        if slot.startswith("_"):
            backing = getattr(target, slot, None)
            if isinstance(backing, dict):
                backing["A"] = 9.0  # the known falsifying probe
                backing["EVIL"] = -5.0
                backing.pop("B", None)
    assert dict(target.weights) == {"A": 1.0, "B": 0.5}
    assert target.weight_of("A") == 1.0
    assert target.weight_of("B") == 0.5
    assert target.weight_of("EVIL") == 0.0
    assert "EVIL" not in target.weights
    assert target.assets == frozenset({"A", "B"})
