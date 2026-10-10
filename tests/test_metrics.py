"""Metrics contract tests: the core portfolio metrics (§8 floors T1–T20).

The surface under test is the ``portlearn.metrics`` public module: the
six single-metric functions plus ``core_metrics`` and the frozen
``MetricsSummary`` record, exactly as frozen by the specification approval
(specification §§4–5 with the owner amendments applied:
A1 ``annualized_mean_return``, A2 required no-default ``risk_free``,
A3 the CER strict-positivity domain guard, A4 the ``ledger`` parameter
name).  The laws floored here: declared-basis return routing with the
net default (D3) — pinned per metric on F1 (T5a) and at the §5.5
wealth-path seam on F3, whose cost lands at the drawdown trough
(T6a); declared positive-int ``periods_per_year`` with no
silent 252/12/52 (D5); arithmetic mean × p, σ × √p (ddof=1), Sharpe
excess-per-period-BEFORE-annualization (D4, probe P25 discriminates
the order); max drawdown as a non-positive fraction on the
declared-basis wealth path (D1); CRRA CE/CER with the γ=1 ln branch,
the closed form for γ ≠ 1, and the strict-positivity gross-factor
domain guard naming the offending zero-based period (D2/D8/A3); and
fail-closed input laws for empty paths, non-finite consumed values,
unknown basis tokens, bad ``periods_per_year``/``gamma``/``risk_free``
declarations, and wrong-length rf series (D7/§5.6).  The stdlib-only
core fence is an executable floor (T20).

Numerical literals are traced to the executed §14 probe transcript
(fixtures F1/F2 verbatim therein: P8–P10, P16–P25) and the zero-rf
Sharpe of T12a is derived from the P18/P19 primitives as recorded in
the RED evidence transcript.  The D3 close-out literals (T5a/T6a) are
the full-precision products of the rebuilt F1 fixture and of the F3
fixture: the transcript's P8 display is 6-digit-rounded and its
product sits ~1.3e-5 outside ``rel_tol=1e-12``, so the §13a rebuild —
never the display — is the oracle.  Per §8/§13a the fixtures are
rebuilt through ``portlearn.ledger.build_ledger`` — never
hand-inlined floats without provenance — and float comparisons use
``math.isclose`` with ``rel_tol=1e-12`` (exact-equality assertions
are wrong at this seam: the ledger's own composition yields e.g.
``0.10000000000000009`` for a 10% single-segment period, §14
history).

RED-first: the module is absent at this tranche, so every metrics
import happens INSIDE the tests — pytest reports the causal absence
(``ModuleNotFoundError: No module named 'portlearn.metrics'``)
instead of aborting collection.  Floor names follow metrics §8 verbatim
(base rows T1–T20 plus the owner-amendment suffix rows T12a/T15a).
"""

from __future__ import annotations

import dataclasses
import functools
import math
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from portlearn.costs import Proportional
from portlearn.interfaces import PortfolioDecision
from portlearn.ledger import LedgerPath, build_ledger
from portlearn.turnover import two_sided

# --------------------------------------------------------------------------- #
# Fixtures F1/F2 — rebuilt through build_ledger exactly as the §14 probe built #
# them; every literal below re-derives the transcript values (P8–P17).        #
# --------------------------------------------------------------------------- #

_MONTHS = tuple(datetime(2026, m, 1, tzinfo=UTC) for m in range(1, 6))

_F1_REBALANCE = PortfolioDecision(  # executed at the period-1 boundary
    decision_time=_MONTHS[1],
    execution_time=_MONTHS[1],
    target_weights={"A": 0.5, "B": 0.5},
)


@functools.cache
def _f1_path() -> LedgerPath:
    """F1 — two assets, four periods, one execution at period 1.

    The §14 probe fixture (P8–P11): per-period gross returns
    ``[0.004, 0.003, 0.003568, 0.004148]``, net
    ``[0.004, 0.002792, 0.003568, 0.004148]`` (period 1 carries the
    cost), ``W_0 = 100``, ``final_wealth = 101.4586573892536``.
    """
    return build_ledger(
        initial_weights={"A": 0.4, "B": 0.6},
        accounting_instants=_MONTHS,
        growth_factors={
            _MONTHS[0]: {"A": 0.995, "B": 1.01},
            _MONTHS[1]: {"A": 0.995, "B": 1.011},
            _MONTHS[2]: {"A": 0.995, "B": 1.012},
            _MONTHS[3]: {"A": 0.995, "B": 1.013},
        },
        decisions=[_F1_REBALANCE],
        cost_model=Proportional(rate=0.001, turnover=two_sided),
        initial_wealth=100.0,
    )


@functools.cache
def _f2_path() -> LedgerPath:
    """F2 — single asset, three periods, no executions, ``W_0 = 1``.

    The §14 probe fixture (P16–P18): per-period net returns
    ``[0.10000000000000009, -0.5, 0.19999999999999996]`` (gross ==
    net elementwise), wealth path ``[1.0, 1.1, 0.55, 0.66]``.
    """
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


@functools.cache
def _f3_path() -> LedgerPath:
    """F3 — cost lands at the drawdown trough: basis-routing fixture.

    Two flat assets over three periods; one execution at the period-1
    boundary (a 50/50 -> 40/60 rebalance, two-sided turnover 1.2) and
    a -30% period-2 growth on asset B.  Net wealth path
    ``[1.0, 1.0, 0.9998, 0.6998599999999999]`` (the 0.02% cost drags
    the trough); the gross-compounded path is ``[1.0, 1.0, 1.0, 0.7]``.
    The drawdowns differ (net ``-0.3001400000000001`` vs gross
    ``-0.30000000000000004``), so the two bases are discriminated at
    the wealth-path seam where max_drawdown consumes them (§5.5).
    """
    return build_ledger(
        initial_weights={"A": 0.5, "B": 0.5},
        accounting_instants=_MONTHS[:4],
        growth_factors={
            _MONTHS[0]: {"A": 1.0, "B": 1.0},
            _MONTHS[1]: {"A": 1.0, "B": 1.0},
            _MONTHS[2]: {"A": 1.0, "B": 0.5},
        },
        decisions=[
            PortfolioDecision(
                decision_time=_MONTHS[1],
                execution_time=_MONTHS[1],
                target_weights={"A": 0.4, "B": 0.6},
            )
        ],
        cost_model=Proportional(rate=0.001, turnover=two_sided),
        initial_wealth=1.0,
    )


def _metrics() -> Any:
    """Import ``portlearn.metrics`` inside the test body so pytest can
    report the causal module absence instead of aborting collection
    (the lazy-import house pattern for absent surfaces)."""
    from importlib import import_module

    return import_module("portlearn.metrics")


def _fn(name: str) -> Callable[..., Any]:
    return getattr(_metrics(), name)


def _close(actual: float, expected: float) -> bool:
    """The §8 comparison law: ``math.isclose`` at ``rel_tol=1e-12``."""
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0)


def _row_replaced(path: LedgerPath, index: int, **overrides: Any) -> LedgerPath:
    """A constructed path value with ONE row replaced by a constructed
    ``LedgerRow`` carrying the overridden fields (T4/T15a injection —
    the record is a plain frozen value, so injection is a dataclass
    replace, never a ``build_ledger`` call)."""
    row = dataclasses.replace(path.rows[index], **overrides)
    return LedgerPath(
        rows=path.rows[:index] + (row,) + path.rows[index + 1 :],
        final_wealth=path.final_wealth,
        n_periods=path.n_periods,
    )


# Per-metric invocation declarations (function name -> required
# keywords), used by the parametrized fail-closed floors.  Resolved
# lazily so the suite collects cleanly while the module is absent.
_METRIC_CALLS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("cumulative_return", {}),
    ("annualized_mean_return", {"periods_per_year": 12}),
    ("annualized_volatility", {"periods_per_year": 12}),
    ("sharpe_ratio", {"periods_per_year": 12, "risk_free": 0.001}),
    ("max_drawdown", {}),
    ("certainty_equivalent_return", {"gamma": 2.0}),
    (
        "core_metrics",
        {"periods_per_year": 12, "gamma": 2.0, "risk_free": 0.001},
    ),
)

# The metrics that consume per-period returns r_t on the declared
# basis (max_drawdown on the net basis consumes the wealth fields
# instead — §5.5 — and is guarded through its own consumed surface).
_RETURN_CONSUMING_CALLS = tuple(
    case for case in _METRIC_CALLS if case[0] != "max_drawdown"
)

# The four periods_per_year-consuming APIs of floor T12.
_P_CONSUMING_CALLS: tuple[tuple[str, dict[str, Any]], ...] = tuple(
    case
    for case in _METRIC_CALLS
    if case[0]
    in (
        "annualized_mean_return",
        "annualized_volatility",
        "sharpe_ratio",
        "core_metrics",
    )
)

# §14 transcript literals (fixtures F1/F2), cited per assertion.
F1_NET_CUM = 0.014586573892535926  # P10: prod(1+r_net) - 1 == W_T/W_0 - 1
# F1 gross literals are the full-precision products of the REBUILT F1
# fixture's gross series (§13a rebuild; P8's transcript display is
# 6-digit-rounded and its product 0.014797008434048742 sits ~1.3e-5
# away — outside rel_tol=1e-12, so the rebuild, not the display, is
# the oracle).  Derived under the §5 law each pins.
F1_GROSS_CUM = 0.014796810681999784  # prod(1+r_gross) - 1, rebuilt F1
F1_NET_MEAN_P = 0.04352403483591405  # mean(r_net) x 12
F1_GROSS_MEAN_P = 0.0441474133219697  # mean(r_gross) x 12
F1_NET_VOL_P = 0.0021080438966844394  # sd(r_net, ddof=1) x sqrt(12)
F1_GROSS_VOL_P = 0.0017848434434508613  # sd(r_gross, ddof=1) x sqrt(12)
F1_NET_SHARPE_RF001 = 14.954164325275903  # excess-first Sharpe, r_net, rf=0.001
F1_GROSS_SHARPE_RF001 = 18.01133507811479  # excess-first Sharpe, r_gross
F1_NET_CER2_ANN = 0.04439940096847672  # gamma=2 CER_g^12 - 1, net basis
F1_GROSS_CER2_ANN = 0.04504926888625249  # gamma=2 CER_g^12 - 1, gross basis
F3_NET_MDD = -0.3001400000000001  # ledger wealth path [1, 1, 0.9998, 0.69986]
F3_GROSS_MDD = -0.30000000000000004  # compounded gross path [1, 1, 1, 0.7]
F2_CUM = -0.34  # P17 wealth path [1.0, 1.1, 0.55, 0.66]: 0.66/1.0 - 1
F2_MEAN_P = -0.7999999999999998  # P18: mean(-0.06666666666666665) x 12
F2_VOL_P = 1.3114877048604001  # P18: sd(ddof=1)=0.37859388972001823 x sqrt(12)
F2_SHARPE_RF001 = -0.6191441955503748  # P19/P20: scalar rf == constant series
F2_SHARPE_VARYING_RF = -0.6381288390709251  # P25 correct: excess first
F2_SHARPE_WRONG_ORDER = -0.6465939382102437  # P25 wrong: annualize each first
F2_SHARPE_RF0 = -0.6099942813304186  # derived: (P18 mean - 0)/P18 sd x sqrt(12)
F2_CER1_NET = -0.12934123088263882  # P21: geometric-mean gross 0.8706587691173612 - 1
F2_CER1_ANN = -0.8102526400000001  # P21: annualized compound form (p=12)
F2_CER2_NET = (
    -0.19838056680161942
)  # P22: closed form == bisection, gross 0.8016194331983806
F2_CER2_ANN = -0.9295925129408775  # P22: annualized compound form (p=12)

# The decision-layer vocabulary that CER error messages must NOT
# mention or suggest (D8: the evaluation gamma is never identified
# with the decision risk-aversion/target convention).
_M34_FORBIDDEN_TOKENS = (
    "risk_aversion",
    "risk-aversion",
    "target_return",
    "meanvariance",
    "lambda",
    "λ",
    "m3.4",
)

# T20 — the stdlib-fence subprocess program: import the module, build
# a small path through the (stdlib-only) ledger, compute the combined
# summary, then prove numpy/pandas/scipy never entered sys.modules.
_T20_PROGRAM = """
import sys
from datetime import UTC, datetime

from portlearn.costs import Proportional
from portlearn.ledger import build_ledger
from portlearn.turnover import two_sided

import portlearn.metrics as metrics

months = tuple(datetime(2026, m, 1, tzinfo=UTC) for m in range(1, 5))
path = build_ledger(
    initial_weights={"S": 1.0},
    accounting_instants=months,
    growth_factors={
        months[0]: {"S": 1.10},
        months[1]: {"S": 0.50},
        months[2]: {"S": 1.20},
    },
    decisions=[],
    cost_model=Proportional(rate=0.001, turnover=two_sided),
    initial_wealth=1.0,
)
summary = metrics.core_metrics(
    path, periods_per_year=12, gamma=2.0, risk_free=0.001
)
assert summary is not None, "core_metrics returned nothing"
banned = {"numpy", "pandas", "scipy"}
loaded = banned.intersection(sys.modules)
assert not loaded, f"stdlib fence broken: {sorted(loaded)}"
print("METRICS_STDLIB_OK")
"""


# --------------------------------------------------------------------------- #
# T1 — module surface                                                         #
# --------------------------------------------------------------------------- #


def test_t01_module_all_is_exactly_the_seven_functions_and_summary() -> None:
    """T1 — ``portlearn.metrics.__all__`` is exactly the eight declared
    names in declaration order (``annualized_mean_return`` per A1),
    every name resolves, and ``MetricsSummary`` is a frozen dataclass."""
    module = _metrics()
    assert module.__all__ == [
        "cumulative_return",
        "annualized_mean_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "certainty_equivalent_return",
        "core_metrics",
        "MetricsSummary",
    ]
    for name in module.__all__:
        assert getattr(module, name) is not None, f"{name} does not resolve"
    summary_cls = module.MetricsSummary
    assert dataclasses.is_dataclass(summary_cls)
    # Frozen at the class level (the runtime dataclass parameter); T18
    # re-verifies frozen-ness behaviorally on a live record instance.
    params = getattr(summary_cls, "__dataclass_params__", None)
    assert params is not None and params.frozen is True


# --------------------------------------------------------------------------- #
# T2 — facade exposure                                                        #
# --------------------------------------------------------------------------- #


def test_t02_facade_exposes_the_metrics_module() -> None:
    """T2 — ``portlearn.metrics`` resolves through the lazy root
    facade to the very module object the direct import yields."""
    import portlearn

    direct = _metrics()
    assert direct.__name__ == "portlearn.metrics"
    assert portlearn.metrics is direct


def test_t02_facade_error_message_names_metrics_among_public_modules() -> None:
    """T2 — the facade's unknown-attribute error names ``metrics``
    among the public modules (the message update is part of the §10
    write set; the branch is absent at RED time, so this floor fails
    causally today)."""
    import portlearn

    with pytest.raises(AttributeError) as caught:
        portlearn.definitely_not_a_portlearn_attribute  # noqa: B018 -- deliberate miss
    message = str(caught.value)
    if "metrics" not in message:
        pytest.fail(
            "EXPECTED_REJECTION the root facade's unknown-attribute error "
            "does not yet name 'metrics' among the lazy public modules; "
            f"got: {message}",
            pytrace=False,
        )


# --------------------------------------------------------------------------- #
# T3 — empty path fails closed                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("function_name", "kwargs"),
    _METRIC_CALLS,
    ids=[name for name, _ in _METRIC_CALLS],
)
def test_t03_empty_path_fails_closed_from_every_metric(
    function_name: str, kwargs: dict[str, Any]
) -> None:
    """T3 — a zero-row ``LedgerPath`` (constructed value, not via
    ``build_ledger``) is a ``ValueError`` from every metric (§5.6:
    no metrics on an empty path)."""
    empty = LedgerPath(rows=(), final_wealth=1.0, n_periods=0)
    with pytest.raises(ValueError):
        _fn(function_name)(empty, **kwargs)


# --------------------------------------------------------------------------- #
# T4 — non-finite consumed values fail closed (D7)                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "pos-inf", "neg-inf"],
)
@pytest.mark.parametrize(
    ("function_name", "kwargs"),
    _RETURN_CONSUMING_CALLS,
    ids=[name for name, _ in _RETURN_CONSUMING_CALLS],
)
def test_t04_nonfinite_return_fails_closed_naming_metric_and_period(
    function_name: str, kwargs: dict[str, Any], bad_value: float
) -> None:
    """T4 — one non-finite declared-basis return injected as a
    constructed ``LedgerRow`` value at period index 1 is a
    ``ValueError`` naming the metric and the offending zero-based
    period index; no NaN skipping, no nan_policy knob (D7/§5.6)."""
    poisoned = _row_replaced(
        _f2_path(), 1, net_return=bad_value, gross_return=bad_value
    )
    with pytest.raises(ValueError) as caught:
        _fn(function_name)(poisoned, **kwargs)
    message = str(caught.value)
    assert function_name in message, (
        f"the rejection must name the metric '{function_name}'; got: {message}"
    )
    assert re.search(r"\b1\b", message), (
        "the rejection must identify the offending zero-based period "
        f"index 1; got: {message}"
    )


def test_t04_nonfinite_wealth_fails_closed_for_max_drawdown() -> None:
    """T4 — ``max_drawdown`` consumes the wealth fields on the net
    basis (§5.5), so its fail-closed surface is the wealth path: a
    NaN ``wealth_close`` at period index 1 is a ``ValueError`` naming
    the metric and the offending zero-based period index (D7)."""
    poisoned = _row_replaced(_f2_path(), 1, wealth_close=float("nan"))
    with pytest.raises(ValueError) as caught:
        _fn("max_drawdown")(poisoned)
    message = str(caught.value)
    assert "max_drawdown" in message
    assert re.search(r"\b1\b", message)


# --------------------------------------------------------------------------- #
# T5 — net basis is the default and consumes net_return                       #
# --------------------------------------------------------------------------- #


def test_t05_net_basis_is_the_default_and_consumes_net_return() -> None:
    """T5 — F1's default-basis cumulative return equals the P10 net
    literal, the default agrees with the explicit ``basis="net"``
    declaration, and it differs from the gross-basis value exactly
    because period 1 carries the transaction cost (P8)."""
    f1 = _f1_path()
    assert f1.rows[1].gross_return != f1.rows[1].net_return  # the cost carrier
    default_value = _fn("cumulative_return")(f1)
    explicit_net = _fn("cumulative_return")(f1, basis="net")
    assert _close(default_value, F1_NET_CUM)
    assert _close(explicit_net, F1_NET_CUM)
    gross_value = _fn("cumulative_return")(f1, basis="gross")
    assert not _close(default_value, gross_value)


# The all-metric basis-routing table (D3 close-out): every metric,
# both declared bases, full-precision rebuilt-F1 oracles.  Each row:
# (function name, metric kwargs, net literal, gross literal).  CER is
# pinned in its annualized compound form (the form core_metrics
# reports; the per-period form is T13/T14's law).
_BASIS_ROUTING_TABLE: tuple[tuple[str, dict[str, Any], float, float], ...] = (
    ("cumulative_return", {}, F1_NET_CUM, F1_GROSS_CUM),
    (
        "annualized_mean_return",
        {"periods_per_year": 12},
        F1_NET_MEAN_P,
        F1_GROSS_MEAN_P,
    ),
    (
        "annualized_volatility",
        {"periods_per_year": 12},
        F1_NET_VOL_P,
        F1_GROSS_VOL_P,
    ),
    (
        "sharpe_ratio",
        {"periods_per_year": 12, "risk_free": 0.001},
        F1_NET_SHARPE_RF001,
        F1_GROSS_SHARPE_RF001,
    ),
    (
        "certainty_equivalent_return",
        {"gamma": 2.0, "periods_per_year": 12},
        F1_NET_CER2_ANN,
        F1_GROSS_CER2_ANN,
    ),
)


@pytest.mark.parametrize(
    ("function_name", "kwargs", "net_literal", "gross_literal"),
    _BASIS_ROUTING_TABLE,
    ids=[name for name, _, _, _ in _BASIS_ROUTING_TABLE],
)
def test_t05a_every_metric_routes_its_declared_basis(
    function_name: str, kwargs: dict[str, Any], net_literal: float, gross_literal: float
) -> None:
    """T5a — every return-consuming metric reads the declared basis on
    every call: on the rebuilt F1 fixture each metric's ``basis="net"``
    output equals its full-precision net oracle and its
    ``basis="gross"`` output equals its gross oracle, and the two
    never agree (D3: one basis per call, never a mixed or implicit
    read — V1's silent estimand swap is impossible)."""
    f1 = _f1_path()
    net_value = _fn(function_name)(f1, basis="net", **kwargs)
    gross_value = _fn(function_name)(f1, basis="gross", **kwargs)
    assert _close(net_value, net_literal), (
        f"{function_name} on basis='net' must equal its full-precision "
        f"net oracle; got {net_value!r} vs {net_literal!r}"
    )
    assert _close(gross_value, gross_literal), (
        f"{function_name} on basis='gross' must equal its full-precision "
        f"gross oracle; got {gross_value!r} vs {gross_literal!r}"
    )
    assert not _close(net_value, gross_value), (
        f"{function_name} must discriminate the bases on F1 (period 1 "
        "carries the cost); identical values mean a mixed-basis read"
    )


def test_t05a_core_metrics_routes_its_declared_basis_to_every_field() -> None:
    """T5a — ``core_metrics`` carries the declaration to every field:
    on F1 the net-basis summary's six fields equal the six single
    metrics on ``basis="net"``, the gross-basis summary's fields equal
    the same metrics on ``basis="gross"``, and the two summaries
    differ in the fields the cost reaches (D3: no per-metric mixed
    bases inside one call)."""
    metrics = _metrics()
    f1 = _f1_path()
    kwargs = {"periods_per_year": 12, "gamma": 2.0, "risk_free": 0.001}
    net_summary = metrics.core_metrics(f1, basis="net", **kwargs)
    gross_summary = metrics.core_metrics(f1, basis="gross", **kwargs)

    per_metric: dict[str, dict[str, Any]] = {
        "cumulative_return": {},
        "annualized_mean_return": {"periods_per_year": 12},
        "annualized_volatility": {"periods_per_year": 12},
        "sharpe_ratio": {"periods_per_year": 12, "risk_free": 0.001},
        "max_drawdown": {},
        "certainty_equivalent_return": {"gamma": 2.0, "periods_per_year": 12},
    }
    differing_fields: list[str] = []
    for name, call_kwargs in per_metric.items():
        net_expected = getattr(metrics, name)(f1, basis="net", **call_kwargs)
        gross_expected = getattr(metrics, name)(f1, basis="gross", **call_kwargs)
        assert getattr(net_summary, name) == net_expected, (
            f"net-basis summary field {name} must equal the net-basis "
            "single metric on the same declarations"
        )
        assert getattr(gross_summary, name) == gross_expected, (
            f"gross-basis summary field {name} must equal the gross-basis "
            "single metric on the same declarations"
        )
        if net_expected != gross_expected:
            differing_fields.append(name)
    # The cost lives in F1 period 1, so every return-consuming metric
    # and the net wealth path must feel it; the summaries differ.
    assert {"cumulative_return", "annualized_mean_return", "sharpe_ratio"} <= set(
        differing_fields
    ), (
        "the F1 cost must separate the bases in the return-consuming "
        f"fields; only {differing_fields} differ"
    )
    assert net_summary != gross_summary


# --------------------------------------------------------------------------- #
# T6 — gross basis consumes gross_return                                      #
# --------------------------------------------------------------------------- #


def test_t06_gross_basis_consumes_gross_return() -> None:
    """T6 — on F2 (no executions: gross == net elementwise, P16) the
    ``basis="gross"`` cumulative equals the ``basis="net"`` value; on
    F1 the gross-basis cumulative differs from T5's net value and
    equals the full-precision product of the rebuilt F1 gross series
    (the display-rounded P8 product is ~1.3e-5 away — outside the
    §8 tolerance — so the §13a rebuild is the oracle)."""
    f2 = _f2_path()
    f2_net = _fn("cumulative_return")(f2, basis="net")
    f2_gross = _fn("cumulative_return")(f2, basis="gross")
    assert _close(f2_gross, f2_net)
    assert _close(f2_gross, F2_CUM)

    f1 = _f1_path()
    f1_gross = _fn("cumulative_return")(f1, basis="gross")
    assert _close(f1_gross, F1_GROSS_CUM)
    assert not _close(f1_gross, F1_NET_CUM)


def test_t06a_max_drawdown_routes_its_declared_basis() -> None:
    """T6a — ``max_drawdown`` consumes the declared basis at the
    wealth-path seam (§5.5): on F3 — where the cost lands at the
    drawdown trough — the default and ``basis="net"`` equal the
    ledger's own wealth-path drawdown ``-0.3001400000000001``, while
    ``basis="gross"`` reconstructs the cost-free counterfactual path
    and reports ``-0.30000000000000004``; the two bases are
    discriminated (D3 at 5.5's seam: the reconstruction law, not the
    ledger's net wealth fields, is what gross-basis drawdown reads)."""
    f3 = _f3_path()
    net_mdd = _fn("max_drawdown")(f3, basis="net")
    gross_mdd = _fn("max_drawdown")(f3, basis="gross")
    assert _close(net_mdd, F3_NET_MDD)
    assert _close(gross_mdd, F3_GROSS_MDD)
    assert not _close(net_mdd, gross_mdd)

    default_mdd = _fn("max_drawdown")(f3)
    assert _close(default_mdd, F3_NET_MDD)  # net is the declared default (D3)

    # Control (P17 law): on F2, where no execution ever fires, the
    # net wealth path and the compounded gross path coincide and both
    # bases report the same -0.5 drawdown.
    f2_mdd_net = _fn("max_drawdown")(_f2_path(), basis="net")
    f2_mdd_gross = _fn("max_drawdown")(_f2_path(), basis="gross")
    assert _close(f2_mdd_net, -0.5)
    assert _close(f2_mdd_gross, -0.5)


# --------------------------------------------------------------------------- #
# T7 — unknown basis token fails closed                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("function_name", "kwargs"),
    _METRIC_CALLS,
    ids=[name for name, _ in _METRIC_CALLS],
)
def test_t07_unknown_basis_token_fails_closed(
    function_name: str, kwargs: dict[str, Any]
) -> None:
    """T7 — ``basis="total"`` is not a declared basis token; every
    metric rejects it with ``ValueError`` (§5.6; no third basis
    token exists, D3)."""
    with pytest.raises(ValueError):
        _fn(function_name)(_f2_path(), basis="total", **kwargs)


# --------------------------------------------------------------------------- #
# T8 — cumulative-return law                                                  #
# --------------------------------------------------------------------------- #


def test_t08_cumulative_return_law_wealth_form_and_product_form() -> None:
    """T8 — F2's cumulative return equals ``W_T/W_0 − 1`` on the exact
    P17 wealth path ``[1.0, 1.1, 0.55, 0.66]`` (i.e. ``0.66 − 1``)
    and equals the product form ``Π(1+r_t) − 1`` within
    ``rel_tol=1e-12`` (§5.1 identity, probe P10)."""
    f2 = _f2_path()
    assert [1.0, 1.1, 0.55, 0.66] == [
        f2.rows[0].wealth_open,
        f2.rows[0].wealth_close,
        f2.rows[1].wealth_close,
        f2.rows[2].wealth_close,
    ]
    actual = _fn("cumulative_return")(f2)
    wealth_form = 0.66 / 1.0 - 1.0
    product_form = math.prod(1.0 + row.net_return for row in f2.rows) - 1.0
    assert _close(actual, wealth_form)
    assert _close(actual, product_form)
    assert _close(actual, F2_CUM)


# --------------------------------------------------------------------------- #
# T9 — annualized mean and volatility (D5/D6)                                 #
# --------------------------------------------------------------------------- #


def test_t09_annualized_mean_and_volatility_laws() -> None:
    """T9 — F2 net basis, ``periods_per_year=12``: the annualized mean
    return is the P18 arithmetic mean × 12 and the annualized
    volatility is the P18 ddof=1 stdev × √12."""
    f2 = _f2_path()
    annualized_mean = _fn("annualized_mean_return")(f2, periods_per_year=12)
    assert _close(annualized_mean, F2_MEAN_P)
    annualized_vol = _fn("annualized_volatility")(f2, periods_per_year=12)
    assert _close(annualized_vol, F2_VOL_P)


# --------------------------------------------------------------------------- #
# T10 — Sharpe: scalar rf equals constant series (D4)                         #
# --------------------------------------------------------------------------- #


def test_t10_sharpe_scalar_rf_equals_constant_series() -> None:
    """T10 — F2 with rf=0.001 as a scalar and as the constant series
    ``[0.001, 0.001, 0.001]`` give the identical Sharpe
    ``-0.6191441955503748`` (P19/P20 broadcast law)."""
    f2 = _f2_path()
    scalar_form = _fn("sharpe_ratio")(f2, periods_per_year=12, risk_free=0.001)
    series_form = _fn("sharpe_ratio")(
        f2, periods_per_year=12, risk_free=[0.001, 0.001, 0.001]
    )
    assert _close(scalar_form, F2_SHARPE_RF001)
    assert _close(series_form, F2_SHARPE_RF001)
    assert _close(scalar_form, series_form)


# --------------------------------------------------------------------------- #
# T11 — Sharpe: excess per period BEFORE annualization (D4)                   #
# --------------------------------------------------------------------------- #


def test_t11_sharpe_takes_excess_per_period_before_annualization() -> None:
    """T11 — F2 with the varying rf series ``[0.000, 0.010, 0.002]``
    (P25): the correct excess-first law yields ``-0.6381288390709251``;
    the wrong order (annualize each side first) yields
    ``-0.6465939382102437`` — asserted unequal, pinning the order."""
    f2 = _f2_path()
    actual = _fn("sharpe_ratio")(
        f2, periods_per_year=12, risk_free=[0.000, 0.010, 0.002]
    )
    assert _close(actual, F2_SHARPE_VARYING_RF)
    assert not _close(actual, F2_SHARPE_WRONG_ORDER)


# --------------------------------------------------------------------------- #
# T12 — no silent periods_per_year (D5)                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("function_name", "kwargs"),
    _P_CONSUMING_CALLS,
    ids=[name for name, _ in _P_CONSUMING_CALLS],
)
def test_t12_omitting_periods_per_year_is_a_type_error(
    function_name: str, kwargs: dict[str, Any]
) -> None:
    """T12 — omitting ``periods_per_year`` on every annualized metric
    and on ``core_metrics`` is a ``TypeError`` (missing required
    keyword; there is no silent 252/12/52 default, D5)."""
    stripped = {
        key: value for key, value in kwargs.items() if key != "periods_per_year"
    }
    with pytest.raises(TypeError):
        _fn(function_name)(_f2_path(), **stripped)


@pytest.mark.parametrize(
    "bad_p",
    [0, 12.0, True],
    ids=["zero", "float", "bool"],
)
@pytest.mark.parametrize(
    ("function_name", "kwargs"),
    _P_CONSUMING_CALLS,
    ids=[name for name, _ in _P_CONSUMING_CALLS],
)
def test_t12_invalid_periods_per_year_is_a_value_error(
    function_name: str, kwargs: dict[str, Any], bad_p: Any
) -> None:
    """T12 — ``periods_per_year`` must be a positive ``int`` with
    ``bool`` excluded: 0, a float 12.0, and ``True`` are each a
    ``ValueError`` on every p-consuming surface (D5 declaration law)."""
    declared = {
        key: value for key, value in kwargs.items() if key != "periods_per_year"
    }
    with pytest.raises(ValueError):
        _fn(function_name)(_f2_path(), periods_per_year=bad_p, **declared)


def test_t12_cer_optional_periods_per_year_is_validated_when_declared() -> None:
    """T12 — CER's optional ``periods_per_year`` follows the same §4
    declaration law: a declared invalid value (0) is a ``ValueError``
    (the optionality of the per-period form never relaxes the type
    law)."""
    with pytest.raises(ValueError):
        _fn("certainty_equivalent_return")(_f2_path(), gamma=2.0, periods_per_year=0)


# --------------------------------------------------------------------------- #
# T12a — risk_free is required (D4/A2)                                        #
# --------------------------------------------------------------------------- #


def test_t12a_omitting_risk_free_is_a_type_error_on_both_surfaces() -> None:
    """T12a — omitting ``risk_free`` on ``sharpe_ratio`` and on
    ``core_metrics`` is a ``TypeError`` (missing required keyword;
    there is no default and no silent zero — owner amendment A2)."""
    with pytest.raises(TypeError):
        _fn("sharpe_ratio")(_f2_path(), periods_per_year=12)
    with pytest.raises(TypeError):
        _fn("core_metrics")(_f2_path(), periods_per_year=12, gamma=2.0)


def test_t12a_explicit_zero_risk_free_is_a_valid_declaration() -> None:
    """T12a — an explicitly declared ``risk_free=0.0`` is valid and
    means e_t = r_t (§5.4): the zero-rf Sharpe on F2 stays reachable,
    now only by explicit declaration, and equals the value derived
    from the P18/P19 transcript primitives."""
    f2 = _f2_path()
    explicit_zero = _fn("sharpe_ratio")(f2, periods_per_year=12, risk_free=0.0)
    zero_series = _fn("sharpe_ratio")(
        f2, periods_per_year=12, risk_free=[0.0, 0.0, 0.0]
    )
    assert _close(explicit_zero, F2_SHARPE_RF0)
    assert _close(zero_series, F2_SHARPE_RF0)
    assert _close(explicit_zero, zero_series)


# --------------------------------------------------------------------------- #
# T13 — CER γ=1 is the ln branch (D2)                                         #
# --------------------------------------------------------------------------- #


def test_t13_cer_gamma_one_is_the_ln_branch() -> None:
    """T13 — F2 net basis, γ=1: the per-period CER is the geometric
    mean of gross factors minus one (P21: ``0.8706587691173612 − 1``)
    and the annualized compound form (p=12) matches P21."""
    f2 = _f2_path()
    per_period = _fn("certainty_equivalent_return")(f2, gamma=1)
    assert _close(per_period, F2_CER1_NET)
    annualized = _fn("certainty_equivalent_return")(f2, gamma=1, periods_per_year=12)
    assert _close(annualized, F2_CER1_ANN)


# --------------------------------------------------------------------------- #
# T14 — CER γ=2 closed form (D2)                                              #
# --------------------------------------------------------------------------- #


def test_t14_cer_gamma_two_closed_form() -> None:
    """T14 — F2 net basis, γ=2: the per-period CER is the P22 closed
    form (gross certainty equivalent ``0.8016194331983806``, verified
    equal to the bisection root by the probe) and the annualized
    compound form (p=12) matches P22."""
    f2 = _f2_path()
    per_period = _fn("certainty_equivalent_return")(f2, gamma=2.0)
    assert _close(per_period, F2_CER2_NET)
    annualized = _fn("certainty_equivalent_return")(f2, gamma=2.0, periods_per_year=12)
    assert _close(annualized, F2_CER2_ANN)


# --------------------------------------------------------------------------- #
# T15 — CER gamma is required and validated (D8)                              #
# --------------------------------------------------------------------------- #


def test_t15_omitting_gamma_is_a_type_error_on_both_surfaces() -> None:
    """T15 — omitting ``gamma`` on ``certainty_equivalent_return`` and
    on ``core_metrics`` is a ``TypeError`` (missing required keyword;
    γ is never defaulted and never a module constant, D8)."""
    with pytest.raises(TypeError):
        _fn("certainty_equivalent_return")(_f2_path())
    with pytest.raises(TypeError):
        _fn("core_metrics")(_f2_path(), periods_per_year=12, risk_free=0.001)


@pytest.mark.parametrize(
    "bad_gamma",
    [0, -1.0, True],
    ids=["zero", "negative", "bool"],
)
@pytest.mark.parametrize(
    "function_name",
    ["certainty_equivalent_return", "core_metrics"],
)
def test_t15_invalid_gamma_is_a_value_error(function_name: str, bad_gamma: Any) -> None:
    """T15 — γ must be a positive real with ``bool`` excluded: 0, a
    negative value, and ``True`` are each a ``ValueError`` whose
    message neither mentions nor suggests any decision-layer parameter (D8:
    the evaluation γ is not the decision risk-aversion)."""
    with pytest.raises(ValueError) as caught:
        if function_name == "core_metrics":
            _fn(function_name)(
                _f2_path(),
                periods_per_year=12,
                gamma=bad_gamma,
                risk_free=0.001,
            )
        else:
            _fn(function_name)(_f2_path(), gamma=bad_gamma)
    message = str(caught.value).lower()
    offenders = [tok for tok in _M34_FORBIDDEN_TOKENS if tok in message]
    assert not offenders, (
        "the gamma rejection must not mention or suggest any decision-layer "
        f"decision-layer parameter; found {offenders} in: {message}"
    )


# --------------------------------------------------------------------------- #
# T15a — CER domain guard: strictly positive gross factors (D2/D7/A3)         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_return", [-1.0, -1.5], ids=["total-loss", "below-minus-one"]
)
@pytest.mark.parametrize(
    "function_name",
    ["certainty_equivalent_return", "core_metrics"],
)
def test_t15a_cer_domain_guard_rejects_nonpositive_gross_factors(
    function_name: str, bad_return: float
) -> None:
    """T15a — a declared-basis return r_t ≤ −1 (gross factor ≤ 0),
    injected as a constructed ``LedgerRow`` value at zero-based period
    index 2, is a ``ValueError`` from ``certainty_equivalent_return``
    and from ``core_metrics`` identifying the offending period by its
    zero-based index; finiteness alone does not make r_t ≤ −1
    admissible (A3: −1.5 is finite yet rejected)."""
    poisoned = _row_replaced(
        _f2_path(), 2, net_return=bad_return, gross_return=bad_return
    )
    with pytest.raises(ValueError) as caught:
        if function_name == "core_metrics":
            _fn(function_name)(
                poisoned, periods_per_year=12, gamma=2.0, risk_free=0.001
            )
        else:
            _fn(function_name)(poisoned, gamma=2.0)
    message = str(caught.value)
    assert re.search(r"\b2\b", message), (
        "the domain-guard rejection must identify the offending "
        f"zero-based period index 2; got: {message}"
    )


def test_t15a_all_positive_gross_factors_raise_no_domain_error() -> None:
    """T15a control — a path whose declared-basis gross factors are
    all strictly positive (F2: 1.10/0.50/1.20) raises no domain error
    from either CER surface."""
    f2 = _f2_path()
    _fn("certainty_equivalent_return")(f2, gamma=2.0)
    _fn("core_metrics")(f2, periods_per_year=12, gamma=2.0, risk_free=0.001)


# --------------------------------------------------------------------------- #
# T16 — risk-free length and finiteness laws (D4/D7)                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_rf",
    [
        [0.001, 0.001],
        [0.001, 0.001, 0.001, 0.001],
        [0.001, float("nan"), 0.002],
        float("nan"),
        float("inf"),
    ],
    ids=["too-short", "too-long", "nan-entry", "nan-scalar", "inf-scalar"],
)
@pytest.mark.parametrize(
    "function_name",
    ["sharpe_ratio", "core_metrics"],
)
def test_t16_risk_free_length_and_finiteness_laws(
    function_name: str, bad_rf: Any
) -> None:
    """T16 — an rf series of length ≠ ``n_periods`` is a ``ValueError``
    (P26 declaration law) and a non-finite rf value is a ``ValueError``
    (D7); omitting ``risk_free`` entirely is T12a's ``TypeError``
    floor, not this law."""
    with pytest.raises(ValueError):
        if function_name == "core_metrics":
            _fn(function_name)(
                _f2_path(), periods_per_year=12, gamma=2.0, risk_free=bad_rf
            )
        else:
            _fn(function_name)(_f2_path(), periods_per_year=12, risk_free=bad_rf)


# --------------------------------------------------------------------------- #
# T17 — max-drawdown law (D1/§5.5)                                            #
# --------------------------------------------------------------------------- #


def test_t17_max_drawdown_law() -> None:
    """T17 — F2's max drawdown is exactly ``-0.5`` (P17: peak 1.1 at
    trough 0.55; running drawdowns ``[0.0, 0.0, -0.5, -0.4]``); on F1
    (an all-positive net path, P8/P9) it is ``0.0``; and the reported
    value is a non-positive fraction in every case (D1)."""
    f2_mdd = _fn("max_drawdown")(_f2_path())
    assert f2_mdd == -0.5
    assert f2_mdd <= 0.0

    f1_mdd = _fn("max_drawdown")(_f1_path())
    assert f1_mdd == 0.0
    assert f1_mdd <= 0.0


# --------------------------------------------------------------------------- #
# T18 — MetricsSummary is plain typed values                                  #
# --------------------------------------------------------------------------- #


def test_t18_metrics_summary_is_plain_typed_values_matching_the_functions() -> None:
    """T18 — ``core_metrics`` on F2 (p=12, γ=2, net, rf=0.001) returns
    a frozen record whose fields are exactly the six core metric
    names, each a plain ``float`` equal to the individual function's
    output on the same declarations; no schema/serialization surface
    exists on it (§11 fence: no results shapes)."""
    metrics = _metrics()
    f2 = _f2_path()
    summary = metrics.core_metrics(f2, periods_per_year=12, gamma=2.0, risk_free=0.001)
    assert dataclasses.is_dataclass(summary)
    params = getattr(type(summary), "__dataclass_params__", None)
    assert params is not None and params.frozen is True  # class-level law
    with pytest.raises(AttributeError):  # ...and the behavioral floor
        summary.cumulative_return = 0.0  # type: ignore[misc]

    field_names = {field.name for field in dataclasses.fields(summary)}
    assert field_names == {
        "cumulative_return",
        "annualized_mean_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "certainty_equivalent_return",
    }
    declarations: dict[str, dict[str, Any]] = {
        "cumulative_return": {},
        "annualized_mean_return": {"periods_per_year": 12},
        "annualized_volatility": {"periods_per_year": 12},
        "sharpe_ratio": {"periods_per_year": 12, "risk_free": 0.001},
        "max_drawdown": {},
        "certainty_equivalent_return": {"gamma": 2.0, "periods_per_year": 12},
    }
    for name, kwargs in declarations.items():
        value = getattr(summary, name)
        assert type(value) is float, f"{name} must be a plain float"
        expected = getattr(metrics, name)(f2, **kwargs)
        assert value == expected, (
            f"{name} field must equal the individual function's output "
            f"on the same declarations: {value!r} != {expected!r}"
        )

    forbidden_surface = {
        "to_dict",
        "to_json",
        "as_dict",
        "serialize",
        "schema",
        "manifest",
        "provenance",
        "version",
        "to_csv",
    }
    present = forbidden_surface.intersection(dir(summary))
    assert not present, (
        f"MetricsSummary must carry no schema/serialization surface; "
        f"found {sorted(present)}"
    )


# --------------------------------------------------------------------------- #
# T19 — determinism/replay                                                    #
# --------------------------------------------------------------------------- #


def test_t19_determinism_replay_equal_results() -> None:
    """T19 — two independent calls on the same path under the same
    declarations return ``==``-equal results with byte-identical
    reprs; no RNG, no global state (§4 record-not-process posture)."""
    metrics = _metrics()
    f2 = _f2_path()
    first = metrics.core_metrics(f2, periods_per_year=12, gamma=2.0, risk_free=0.001)
    second = metrics.core_metrics(f2, periods_per_year=12, gamma=2.0, risk_free=0.001)
    assert first == second
    assert repr(first) == repr(second)

    for name, kwargs in (
        ("cumulative_return", {}),
        ("sharpe_ratio", {"periods_per_year": 12, "risk_free": 0.001}),
        ("certainty_equivalent_return", {"gamma": 2.0, "periods_per_year": 12}),
    ):
        left = getattr(metrics, name)(f2, **kwargs)
        right = getattr(metrics, name)(f2, **kwargs)
        assert left == right
        assert repr(left) == repr(right)


# --------------------------------------------------------------------------- #
# T20 — stdlib-only core (fence)                                              #
# --------------------------------------------------------------------------- #


def test_t20_stdlib_only_core_fence() -> None:
    """T20 — in a clean subprocess, importing ``portlearn.metrics``
    and computing ``core_metrics`` on a small built path leaves
    ``numpy``, ``pandas``, and ``scipy`` absent from ``sys.modules``
    (§11 stdlib-only core fence as an executable floor)."""
    completed = subprocess.run(
        [sys.executable, "-c", _T20_PROGRAM],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, (
        "the stdlib-fence subprocess failed:\n"
        f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
    )
    assert "METRICS_STDLIB_OK" in completed.stdout
