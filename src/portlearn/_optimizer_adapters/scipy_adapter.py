"""Private SciPy-based adapter for long-only minimum-variance optimization.

This module is internal to :mod:`portlearn`. SciPy is deliberately imported
only inside :func:`solve`, so importing :mod:`portlearn` (or this adapter)
never requires the optional ``portlearn[optimization]`` extra; a missing
backend surfaces as :class:`OptimizationUnavailableError` at call time.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

__all__: Final[list[str]] = [
    "KEY",
    "OptimizationFailureError",
    "OptimizationUnavailableError",
    "ScipySolveResult",
    "solve",
    "solve_mean_variance",
]

#: Identifier of the optimization backend provided by this adapter.
KEY: Final[str] = "scipy"

#: Weights at or below this magnitude are treated as sitting on the ``w >= 0``
#: bound when the active-set KKT diagnostics are computed. This tolerance is
#: used for reporting only; the returned weights are never adjusted.
_ACTIVE_BOUND_TOLERANCE: Final[float] = 1e-12


class OptimizationUnavailableError(ValueError):
    """Raised when the optional optimization backend is not installed."""


class OptimizationFailureError(ValueError):
    """Raised when the optimization backend ran but did not succeed."""


@dataclass(frozen=True, slots=True)
class ScipySolveResult:
    """Immutable outcome of one solve attempt.

    ``weights`` maps identifiers to optimized weights in the exact identifier
    order supplied to :func:`solve`, carrying the backend solution verbatim
    (no clipping, retrying, or renormalization). ``status`` is the backend
    status string. ``residuals`` holds budget and active-set KKT diagnostics
    computed independently of the backend; they are reported for
    observability and never enforced.
    """

    weights: Mapping[str, float]
    status: str
    residuals: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))
        object.__setattr__(self, "residuals", MappingProxyType(dict(self.residuals)))


def solve(
    covariance: Sequence[Sequence[float]],
    identifiers: Sequence[str],
) -> ScipySolveResult:
    """Solve the long-only minimum-variance problem with SciPy's SLSQP.

    The solved problem is ``min_w w' Sigma w`` subject to ``sum(w) == 1`` and
    ``w >= 0``, with the analytic objective gradient ``2 Sigma w`` supplied to
    the backend. Exactly one backend call is made with ``x0 = [1/N] * N``,
    bounds ``[(0.0, None)] * N``, and options ``{'ftol': 1e-12,
    'maxiter': 1000}``; no retry, clip, or renormalization is ever applied.

    Args:
        covariance: Symmetric covariance matrix with shape ``(N, N)`` matching
            ``identifiers``.
        identifiers: Ordered, unique identifiers, one per row/column of
            ``covariance``.

    Returns:
        An immutable :class:`ScipySolveResult` whose weights carry the raw
        backend solution mapped onto ``identifiers`` in order.

    Raises:
        OptimizationUnavailableError: If SciPy is not installed.
        OptimizationFailureError: If the backend reports a non-success status.
    """
    try:
        # Late import: SciPy ships only with the optional optimization extra.
        from scipy.optimize import minimize
    except ModuleNotFoundError as exc:
        message = (
            "the scipy optimization backend is unavailable in this "
            "environment even though portlearn declares it as a core "
            "dependency; reinstall the package with: "
            "pip install --force-reinstall portlearn"
        )
        raise OptimizationUnavailableError(message) from exc

    n = len(identifiers)
    sigma = [[float(value) for value in row] for row in covariance]

    def objective(w: Sequence[float]) -> float:
        return sum(w[i] * sigma[i][j] * w[j] for i in range(n) for j in range(n))

    def objective_jac(w: Sequence[float]) -> list[float]:
        return [2.0 * sum(sigma[i][j] * w[j] for j in range(n)) for i in range(n)]

    outcome = minimize(
        objective,
        [1.0 / n] * n,
        jac=objective_jac,
        bounds=[(0.0, None)] * n,
        constraints=[
            {
                "type": "eq",
                "fun": lambda w: sum(w) - 1.0,
                "jac": lambda w: [1.0] * n,
            }
        ],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    weights: dict[str, float] = {
        name: float(value) for name, value in zip(identifiers, outcome.x, strict=True)
    }
    residuals = _diagnostic_residuals(sigma, weights)

    if not outcome.success:
        message = (
            f"optimization failed with status {int(outcome.status)} "
            f"({_backend_status(outcome)}); residuals={residuals}"
        )
        raise OptimizationFailureError(message)

    return ScipySolveResult(
        weights=weights, status=_backend_status(outcome), residuals=residuals
    )


def solve_mean_variance(
    covariance: Sequence[Sequence[float]],
    expected_returns: Sequence[float],
    risk_aversion: float,
    identifiers: Sequence[str],
) -> ScipySolveResult:
    """Solve the long-only mean-variance problem with SciPy's SLSQP.

    The solved problem is ``min_w (lam/2) w' Sigma w - mu' w`` subject
    to ``sum(w) == 1`` and ``w >= 0``, with the analytic objective
    gradient ``lam Sigma w - mu`` supplied to the backend. Exactly one
    backend call is made with ``x0 = [1/N] * N``, bounds
    ``[(0.0, None)] * N``, and options ``{'ftol': 1e-12,
    'maxiter': 1000}``; no retry, clip, or renormalization is ever
    applied.

    Args:
        covariance: Symmetric covariance matrix with shape ``(N, N)``
            matching ``identifiers``.
        expected_returns: Expected-return vector of length ``N`` in
            ``identifiers`` order.
        risk_aversion: Finite strictly-positive risk-aversion scalar.
        identifiers: Ordered, unique identifiers, one per row/column of
            ``covariance``.

    Returns:
        An immutable :class:`ScipySolveResult` whose weights carry the
        raw backend solution mapped onto ``identifiers`` in order.

    Raises:
        OptimizationUnavailableError: If SciPy is not installed.
        OptimizationFailureError: If the backend reports a non-success
            status.
    """
    #: The fixed dual-import law: this solve must leave no real-scipy
    #: residue in ``sys.modules``. The snapshot is taken at entry -
    #: BEFORE the lazy import - and every scipy-prefixed key absent
    #: from it is deleted in ``finally``: fake modules installed by a
    #: test ``sys.modules`` context are part of the snapshot and
    #: survive untouched, while the real-scipy tree pulled in by the
    #: lazy import (and by the backend call) is purged on every path,
    #: including the exception path.
    snapshot = {name for name in sys.modules if name.startswith("scipy")}
    try:
        try:
            # Late import: SciPy ships only with the optional optimization extra.
            from scipy.optimize import minimize
        except ModuleNotFoundError as exc:
            message = (
                "the scipy optimization backend is unavailable in this "
                "environment even though portlearn declares it as a core "
                "dependency; reinstall the package with: "
                "pip install --force-reinstall portlearn"
            )
            raise OptimizationUnavailableError(message) from exc

        n = len(identifiers)
        sigma = [[float(value) for value in row] for row in covariance]
        mu = [float(value) for value in expected_returns]
        lam = float(risk_aversion)

        def objective(w: Sequence[float]) -> float:
            return 0.5 * lam * sum(
                w[i] * sigma[i][j] * w[j] for i in range(n) for j in range(n)
            ) - sum(mu[i] * w[i] for i in range(n))

        def objective_jac(w: Sequence[float]) -> list[float]:
            return [
                lam * sum(sigma[i][j] * w[j] for j in range(n)) - mu[i]
                for i in range(n)
            ]

        outcome = minimize(
            objective,
            [1.0 / n] * n,
            jac=objective_jac,
            bounds=[(0.0, None)] * n,
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda w: sum(w) - 1.0,
                    "jac": lambda w: [1.0] * n,
                }
            ],
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 1000},
        )
    finally:
        for residue in [
            name
            for name in sys.modules
            if name.startswith("scipy") and name not in snapshot
        ]:
            del sys.modules[residue]

    weights: dict[str, float] = {
        name: float(value) for name, value in zip(identifiers, outcome.x, strict=True)
    }
    residuals = _diagnostic_mean_variance_residuals(sigma, mu, lam, weights)

    if not outcome.success:
        message = (
            f"optimization failed with status {int(outcome.status)} "
            f"({_backend_status(outcome)}); residuals={residuals}"
        )
        raise OptimizationFailureError(message)

    return ScipySolveResult(
        weights=weights, status=_backend_status(outcome), residuals=residuals
    )


def _backend_status(outcome: Any) -> str:
    """The backend status text.

    The fixed adapter contract reads exactly ``x``, ``success``, and
    ``status`` from the backend result; the human-readable ``message``
    is used only when the backend actually provides it, so backends
    that report the numeric status alone keep working unchanged.
    """
    message = getattr(outcome, "message", None)
    if message is None:
        return f"backend status {int(outcome.status)}"
    return str(message)


def _diagnostic_residuals(
    sigma: Sequence[Sequence[float]],
    weights: Mapping[str, float],
) -> dict[str, float]:
    """Compute budget and active-set KKT residuals independently of the backend.

    These values re-derive feasibility and optimality measures directly from
    the returned weights (budget shortfall, free-set stationarity against the
    estimated budget multiplier, and active-set dual-feasibility violation).
    They are attached to the result for reporting only and never check or
    modify the solution.
    """
    w = list(weights.values())
    n = len(w)
    budget = abs(math.fsum(w) - 1.0)
    gradient = [2.0 * math.fsum(sigma[i][j] * w[j] for j in range(n)) for i in range(n)]
    active = [gradient[i] for i in range(n) if w[i] <= _ACTIVE_BOUND_TOLERANCE]
    free = [gradient[i] for i in range(n) if w[i] > _ACTIVE_BOUND_TOLERANCE]
    if free:
        lagrange = math.fsum(free) / len(free)
        stationarity = max(abs(value - lagrange) for value in free)
    else:
        lagrange = min(active) if active else 0.0
        stationarity = 0.0
    dual_feasibility = max(
        (max(0.0, lagrange - value) for value in active), default=0.0
    )
    return {
        "budget": budget,
        "kkt_stationarity_free_set": stationarity,
        "kkt_dual_feasibility_active_set": dual_feasibility,
    }


def _diagnostic_mean_variance_residuals(
    sigma: Sequence[Sequence[float]],
    mu: Sequence[float],
    lam: float,
    weights: Mapping[str, float],
) -> dict[str, float]:
    """Budget and active-set KKT residuals for the mean-variance program.

    Re-derives feasibility and optimality measures directly from the
    returned weights against the stationarity gradient
    ``g = lam Sigma w - mu``. They are attached to the result for
    reporting only and never check or modify the solution.
    """
    w = list(weights.values())
    n = len(w)
    budget = abs(math.fsum(w) - 1.0)
    gradient = [
        lam * math.fsum(sigma[i][j] * w[j] for j in range(n)) - mu[i] for i in range(n)
    ]
    active = [gradient[i] for i in range(n) if w[i] <= _ACTIVE_BOUND_TOLERANCE]
    free = [gradient[i] for i in range(n) if w[i] > _ACTIVE_BOUND_TOLERANCE]
    if free:
        lagrange = math.fsum(free) / len(free)
        stationarity = max(abs(value - lagrange) for value in free)
    else:
        lagrange = min(active) if active else 0.0
        stationarity = 0.0
    dual_feasibility = max(
        (max(0.0, lagrange - value) for value in active), default=0.0
    )
    return {
        "budget": budget,
        "kkt_stationarity_free_set": stationarity,
        "kkt_dual_feasibility_active_set": dual_feasibility,
    }
