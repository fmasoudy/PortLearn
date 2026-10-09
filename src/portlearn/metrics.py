"""Core portfolio metrics over one recorded wealth path (the metrics layer).

The first evaluation-family surface: six finance-standard point statistics
computed from one immutable :class:`portlearn.ledger.LedgerPath` record, plus one
combined summary record. Every convention is **declared per call** — the
return basis (net-of-cost default, gross counterfactual), the annualization
frequency ``periods_per_year``, the risk-free declaration ``risk_free``, and
the evaluation risk aversion ``gamma`` — and every input rule fails closed
(`ValueError`) instead of silently patching the domain.

Design implemented here (metrics, with amendments A1–A4):

* **Basis routing.** ``basis="net"`` reads ``LedgerRow.net_return`` (the
  investor's realized experience; the default), ``basis="gross"`` reads
  ``gross_return`` (pre-cost selection quality). One basis per call, never
  mixed; no third token exists. ``max_drawdown`` consumes the basis at the
  wealth path: the ledger's own wealth recursion on the net
  basis, the compounded gross-factor reconstruction from ``W_0`` on the gross
  basis.
* **Annualization.** Arithmetic mean × ``p``; sample standard
  deviation (``ddof=1``) × ``√p``; Sharpe × ``√p`` with the excess taken
  **per period before** annualization; CER annualized by compounding.
  No silent 252/12/52 default, no frequency inference from timestamps.
* **Fail-closed inputs.** Empty paths, non-finite consumed values,
  unknown basis tokens, invalid ``periods_per_year``/``gamma``/``risk_free``
  declarations, wrong-length risk-free series, undefined dispersion
  (``T < 2``), and zero excess volatility are all ``ValueError`` naming the
  offending metric and, where a period is implicated, its zero-based index.
* **CRRA certainty equivalent.** ``u(w) = ln w`` at ``γ = 1``,
  ``u(w) = w^(1−γ)/(1−γ)`` otherwise, applied to per-period gross factors on
  the declared basis. Every gross factor must be strictly positive
  (``1 + r_t > 0``) — finiteness alone is insufficient; a factor ``≤ 0`` is a
  ``ValueError`` naming the offending zero-based period. ``gamma`` is the
  caller's declared evaluation risk aversion, never defaulted and never
  identified with any decision-layer convention.
* **Stdlib-only core.** Only ``math``, ``statistics``, ``collections.abc``,
  ``dataclasses``, ``typing``, and ``portlearn.ledger`` — the metrics layer
  inherits the ledger's dependency diet (no numpy/pandas/scipy anywhere in
  the core path).
* **Determinism.** Pure functions over an immutable record: no RNG,
  no global state; identical path + identical declarations give identical
  results.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from portlearn.ledger import LedgerPath

__all__ = [  # noqa: RUF022 — the frozen metrics T1 contract pins this declaration order
    "cumulative_return",
    "annualized_mean_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "certainty_equivalent_return",
    "core_metrics",
    "MetricsSummary",
]

_BASIS_TOKENS = ("net", "gross")


# --------------------------------------------------------------------------- #
# Declaration-law validation helpers (§4 declaration law, §5.6 input laws)     #
# --------------------------------------------------------------------------- #


def _check_basis(basis: Any, metric_name: str) -> None:
    """The basis token must be exactly ``'net'`` or ``'gross'``."""
    if basis not in _BASIS_TOKENS:
        raise ValueError(
            f"{metric_name}: unknown basis token {basis!r}; the declared "
            "bases are exactly 'net' and 'gross' — one basis per call, "
            "never a mixed read."
        )


def _check_non_empty(ledger: LedgerPath, metric_name: str) -> None:
    """No metric is defined on a zero-row path."""
    if not ledger.rows:
        raise ValueError(
            f"{metric_name}: the ledger path carries no accounting rows; "
            "no core metric is defined on an empty wealth path."
        )


def _check_periods_per_year(periods_per_year: Any, metric_name: str) -> None:
    """``periods_per_year`` must be a positive ``int`` with ``bool`` excluded
    (the frequency is declared, never defaulted or inferred)."""
    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, int)
        or periods_per_year <= 0
    ):
        raise ValueError(
            f"{metric_name}: periods_per_year must be declared as a "
            f"positive int (bool excluded); got {periods_per_year!r}. "
            "There is no silent 252/12/52 default."
        )


def _check_gamma(gamma: Any, metric_name: str) -> None:
    """``gamma`` must be a positive real with ``bool`` excluded (the
    evaluation risk aversion is declared, never defaulted)."""
    if (
        isinstance(gamma, bool)
        or not isinstance(gamma, (int, float))
        or not math.isfinite(gamma)
        or gamma <= 0
    ):
        raise ValueError(
            f"{metric_name}: gamma must be declared as a positive real "
            f"number (bool excluded); got {gamma!r}."
        )


def _returns(ledger: LedgerPath, basis: str, metric_name: str) -> list[float]:
    """Declared-basis per-period simple returns with the finiteness guard."""
    attribute = "net_return" if basis == "net" else "gross_return"
    values: list[float] = []
    for index, row in enumerate(ledger.rows):
        value = getattr(row, attribute)
        if not math.isfinite(value):
            raise ValueError(
                f"{metric_name}: the {basis}-basis return at zero-based "
                f"period index {index} is not finite ({value!r}); core "
                "metrics fail closed on non-finite consumed values."
            )
        values.append(float(value))
    return values


def _risk_free_series(risk_free: Any, n_periods: int, metric_name: str) -> list[float]:
    """Normalize the declared ``risk_free`` to a per-period series:
    a real scalar broadcasts to every period; a sequence must have length
    exactly ``n_periods``; every entry must be finite."""
    if isinstance(risk_free, (bool, int, float)):
        if not math.isfinite(risk_free):
            raise ValueError(
                f"{metric_name}: the declared risk_free {risk_free!r} is "
                "not finite; core metrics fail closed on non-finite "
                "consumed values."
            )
        return [float(risk_free)] * n_periods
    if not isinstance(risk_free, Sequence):
        raise ValueError(  # noqa: TRY004 — the declaration surface is ValueError-only
            f"{metric_name}: risk_free must be declared as a real scalar or "
            f"a per-period sequence; got {type(risk_free).__name__}."
        )
    declared = list(risk_free)
    if len(declared) != n_periods:
        raise ValueError(
            f"{metric_name}: the declared risk_free series has length "
            f"{len(declared)}, but the path has {n_periods} periods; a "
            "per-period risk_free series must match n_periods exactly."
        )
    series: list[float] = []
    for index, value in enumerate(declared):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(
                f"{metric_name}: the risk_free series entry at zero-based "
                f"period index {index} is not a finite real ({value!r}); "
                "core metrics fail closed on non-finite consumed values."
            )
        series.append(float(value))
    return series


def _net_wealth_path(ledger: LedgerPath, metric_name: str) -> list[float]:
    """The ledger's own net-of-cost wealth path ``[W_0, W_1, ..., W_T]``
    with the finiteness guard on every point."""
    path: list[float] = []
    opening = ledger.rows[0].wealth_open
    if not math.isfinite(opening):
        raise ValueError(
            f"{metric_name}: the net-basis wealth_open at zero-based period "
            f"index 0 is not finite ({opening!r}); core metrics fail closed "
            "on non-finite consumed values."
        )
    path.append(float(opening))
    for index, row in enumerate(ledger.rows):
        closing = row.wealth_close
        if not math.isfinite(closing):
            raise ValueError(
                f"{metric_name}: the net-basis wealth_close at zero-based "
                f"period index {index} is not finite ({closing!r}); core "
                "metrics fail closed on non-finite consumed values."
            )
        path.append(float(closing))
    return path


def _gross_wealth_path(
    ledger: LedgerPath, gross_returns: Sequence[float], metric_name: str
) -> list[float]:
    """The cost-free counterfactual wealth path reconstructed by compounding
    gross factors from ``W_0`` (the ledger's wealth fields are
    net-of-cost by construction, so the gross basis is rebuilt)."""
    opening = ledger.rows[0].wealth_open
    if not math.isfinite(opening):
        raise ValueError(
            f"{metric_name}: the gross-basis reconstruction base "
            f"wealth_open at zero-based period index 0 is not finite "
            f"({opening!r}); core metrics fail closed on non-finite "
            "consumed values."
        )
    path: list[float] = [float(opening)]
    for value in gross_returns:
        path.append(path[-1] * (1.0 + value))
    return path


def _check_cer_domain(returns: Sequence[float], basis: str, metric_name: str) -> None:
    """The CRRA strict-positivity domain guard: every declared-basis
    gross factor ``1 + r_t`` must be strictly positive before any utility
    arithmetic — finiteness alone is insufficient."""
    for index, value in enumerate(returns):
        gross_factor = 1.0 + value
        if gross_factor <= 0.0:
            raise ValueError(
                f"{metric_name}: the CRRA certainty equivalent is defined "
                "only on strictly positive gross factors 1 + r > 0; the "
                f"{basis}-basis gross factor at zero-based period index "
                f"{index} is {gross_factor!r} (return {value!r})."
            )


def _cer_gross_factor(
    returns: Sequence[float], gamma: float, basis: str, metric_name: str
) -> float:
    """The per-period gross certainty equivalent ``CER_g`` solving
    ``u(CER_g) = E[u]`` for the declared CRRA utility: the ln branch
    at ``γ = 1`` (geometric mean of gross factors), the closed form
    ``((1−γ)·E[u])^(1/(1−γ))`` otherwise."""
    _check_cer_domain(returns, basis, metric_name)
    if gamma == 1:
        gross_factor = math.exp(
            statistics.fmean([math.log(1.0 + value) for value in returns])
        )
    else:
        exponent = 1.0 - gamma
        expected_utility = statistics.fmean(
            [(1.0 + value) ** exponent / exponent for value in returns]
        )
        gross_factor = (exponent * expected_utility) ** (1.0 / exponent)
    if not math.isfinite(gross_factor) or gross_factor <= 0.0:
        raise ValueError(
            f"{metric_name}: the CRRA certainty-equivalent computation "
            f"produced a non-admissible value ({gross_factor!r}) at gamma "
            f"= {gamma!r}; core metrics fail closed."
        )
    return gross_factor


# --------------------------------------------------------------------------- #
# Public metric functions (§4 public surface, §5 laws)                         #
# --------------------------------------------------------------------------- #


def cumulative_return(ledger: LedgerPath, *, basis: str = "net") -> float:
    """Cumulative return on the declared basis:
    ``R_cum = Π_{t=1..T} (1 + r_t) − 1``, identical to ``W_T/W_0 − 1`` on the
    net basis (the record's own linkage identity)."""
    _check_basis(basis, "cumulative_return")
    _check_non_empty(ledger, "cumulative_return")
    returns = _returns(ledger, basis, "cumulative_return")
    return math.prod(1.0 + value for value in returns) - 1.0


def annualized_mean_return(
    ledger: LedgerPath, *, periods_per_year: int, basis: str = "net"
) -> float:
    """Annualized arithmetic mean return (A1): the per-period
    arithmetic mean × the declared ``periods_per_year`` — a linear reporting
    convention, never CAGR/geometric annualization."""
    _check_basis(basis, "annualized_mean_return")
    _check_non_empty(ledger, "annualized_mean_return")
    _check_periods_per_year(periods_per_year, "annualized_mean_return")
    returns = _returns(ledger, basis, "annualized_mean_return")
    return statistics.fmean(returns) * periods_per_year


def annualized_volatility(
    ledger: LedgerPath, *, periods_per_year: int, basis: str = "net"
) -> float:
    """Annualized volatility: the sample standard deviation of the
    declared-basis per-period returns (``ddof=1``) × ``√p``."""
    _check_basis(basis, "annualized_volatility")
    _check_non_empty(ledger, "annualized_volatility")
    _check_periods_per_year(periods_per_year, "annualized_volatility")
    returns = _returns(ledger, basis, "annualized_volatility")
    if len(returns) < 2:
        raise ValueError(
            "annualized_volatility: the declared-basis path has "
            f"{len(returns)} period(s); dispersion is undefined below two "
            "observations (the ddof=1 estimator needs at least two)."
        )
    return statistics.stdev(returns) * math.sqrt(periods_per_year)


def sharpe_ratio(
    ledger: LedgerPath,
    *,
    periods_per_year: int,
    risk_free: float | Sequence[float],
    basis: str = "net",
) -> float:
    """Annualized Sharpe ratio (A2): excess per period FIRST —
    ``e_t = r_t − rf_t`` — then ``(mean(e) / stdev(e, ddof=1)) × √p``.
    ``risk_free`` is a required declaration with no default; a scalar
    broadcasts, a series must have length ``n_periods``; an explicitly
    declared ``risk_free=0.0`` is valid and means ``e_t = r_t``."""
    _check_basis(basis, "sharpe_ratio")
    _check_non_empty(ledger, "sharpe_ratio")
    _check_periods_per_year(periods_per_year, "sharpe_ratio")
    returns = _returns(ledger, basis, "sharpe_ratio")
    series = _risk_free_series(risk_free, len(returns), "sharpe_ratio")
    excess = [value - rf for value, rf in zip(returns, series)]
    if len(excess) < 2:
        raise ValueError(
            "sharpe_ratio: the declared-basis path has "
            f"{len(excess)} period(s); the excess-dispersion estimator is "
            "undefined below two observations."
        )
    excess_stdev = statistics.stdev(excess)
    if excess_stdev == 0.0:
        raise ValueError(
            "sharpe_ratio: the per-period excess return has zero "
            "dispersion, so the ratio is undefined; core metrics fail "
            "closed rather than report a silent ±inf."
        )
    return (statistics.fmean(excess) / excess_stdev) * math.sqrt(periods_per_year)


def max_drawdown(ledger: LedgerPath, *, basis: str = "net") -> float:
    """Maximum drawdown as a non-positive fraction: the deepest
    ``W_t / max(W_0..W_t) − 1`` on the declared-basis wealth path. The net
    basis reads the ledger's own wealth recursion; the gross basis
    reconstructs the cost-free counterfactual by compounding gross factors
    from ``W_0``."""
    _check_basis(basis, "max_drawdown")
    _check_non_empty(ledger, "max_drawdown")
    if basis == "net":
        wealth = _net_wealth_path(ledger, "max_drawdown")
    else:
        gross_returns = _returns(ledger, "gross", "max_drawdown")
        wealth = _gross_wealth_path(ledger, gross_returns, "max_drawdown")
    peak = wealth[0]
    worst = 0.0
    for value in wealth[1:]:
        if value > peak:
            peak = value
            continue
        drawdown = value / peak - 1.0
        worst = min(worst, drawdown)
    return worst


def certainty_equivalent_return(
    ledger: LedgerPath,
    *,
    gamma: float,
    basis: str = "net",
    periods_per_year: int | None = None,
) -> float:
    """CRRA certainty-equivalent return: the guaranteed
    per-period return making the declared-``gamma`` investor indifferent to
    the realized declared-basis gross distribution. Reports the per-period
    net simple-return form ``CER_g − 1`` by default and the annualized
    compound form ``CER_g^p − 1`` when ``periods_per_year`` is declared."""
    _check_basis(basis, "certainty_equivalent_return")
    _check_non_empty(ledger, "certainty_equivalent_return")
    _check_gamma(gamma, "certainty_equivalent_return")
    if periods_per_year is not None:
        _check_periods_per_year(periods_per_year, "certainty_equivalent_return")
    returns = _returns(ledger, basis, "certainty_equivalent_return")
    gross_factor = _cer_gross_factor(
        returns, gamma, basis, "certainty_equivalent_return"
    )
    if periods_per_year is None:
        return gross_factor - 1.0
    annualized = gross_factor**periods_per_year - 1.0
    if not math.isfinite(annualized):
        raise ValueError(
            "certainty_equivalent_return: the annualized compound form is "
            f"not finite ({annualized!r}) at gamma = {gamma!r} and "
            f"periods_per_year = {periods_per_year!r}; core metrics fail "
            "closed."
        )
    return annualized


@dataclass(frozen=True)
class MetricsSummary:
    """The six core metrics as plain immutable ``float`` fields (the module fence:
    named values only — no schema, serialization, provenance, or versioning
    surface exists on this record)."""

    cumulative_return: float
    annualized_mean_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    certainty_equivalent_return: float


def core_metrics(
    ledger: LedgerPath,
    *,
    periods_per_year: int,
    gamma: float,
    risk_free: float | Sequence[float],
    basis: str = "net",
) -> MetricsSummary:
    """Every core metric on one path under one declaration set: the six
    single-metric rules carried into one immutable record, each field exactly the
    individual function's output on the same declarations — one basis per
    call, never per-metric mixed bases."""
    _check_basis(basis, "core_metrics")
    _check_non_empty(ledger, "core_metrics")
    returns = _returns(ledger, basis, "core_metrics")
    _check_periods_per_year(periods_per_year, "core_metrics")
    _check_gamma(gamma, "core_metrics")
    _risk_free_series(risk_free, len(returns), "core_metrics")
    _check_cer_domain(returns, basis, "core_metrics")
    _net_wealth_path(ledger, "core_metrics")
    return MetricsSummary(
        cumulative_return=cumulative_return(ledger, basis=basis),
        annualized_mean_return=annualized_mean_return(
            ledger, periods_per_year=periods_per_year, basis=basis
        ),
        annualized_volatility=annualized_volatility(
            ledger, periods_per_year=periods_per_year, basis=basis
        ),
        sharpe_ratio=sharpe_ratio(
            ledger, periods_per_year=periods_per_year, risk_free=risk_free, basis=basis
        ),
        max_drawdown=max_drawdown(ledger, basis=basis),
        certainty_equivalent_return=certainty_equivalent_return(
            ledger, gamma=gamma, periods_per_year=periods_per_year, basis=basis
        ),
    )
