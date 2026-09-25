"""Private allocation engine: long-only minimum- and mean-variance entry
points."""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from portlearn._optimizer_adapters.scipy_adapter import ScipySolveResult


def solve_long_only_min_variance(
    covariance: Sequence[Sequence[float]],
    identifiers: Sequence[str],
) -> ScipySolveResult:
    """Validate inputs and delegate the long-only minimum-variance solve.

    Validation performed here is exact (no numerical tolerance): the
    covariance matrix must be square with shape matching ``identifiers``,
    identifiers must be unique, and the matrix must be exactly symmetric.
    Only after validation passes is the optimizer adapter imported, so
    malformed input never depends on the optional optimization extra.

    Args:
        covariance: Symmetric covariance matrix with shape ``(N, N)`` matching
            ``identifiers``.
        identifiers: Ordered, unique identifiers, one per row/column of
            ``covariance``.

    Returns:
        The immutable solve result produced by the optimizer adapter.

    Raises:
        ValueError: If any validation check fails.
        OptimizationUnavailableError: If SciPy is not installed
            (propagated from the adapter).
        OptimizationFailureError: If the backend fails (propagated from the
            adapter).
    """
    n = len(identifiers)
    if n == 0:
        message = "identifiers must name at least one asset"
        raise ValueError(message)

    if len(covariance) != n:
        message = (
            f"covariance shape mismatch: expected {n} rows for {n} "
            f"identifiers, got {len(covariance)} rows"
        )
        raise ValueError(message)

    for index, row in enumerate(covariance):
        if len(row) != n:
            message = f"covariance row {index} has length {len(row)}; expected {n}"
            raise ValueError(message)

    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in identifiers:
        if name in seen:
            duplicates.add(name)
        else:
            seen.add(name)
    if duplicates:
        message = f"identifiers must be unique; duplicates: {sorted(duplicates)}"
        raise ValueError(message)

    for i in range(n):
        for j in range(i + 1, n):
            if covariance[i][j] != covariance[j][i]:
                message = (
                    "covariance must be exactly symmetric: "
                    f"covariance[{i}][{j}] == {covariance[i][j]!r} but "
                    f"covariance[{j}][{i}] == {covariance[j][i]!r}"
                )
                raise ValueError(message)

    # Late lookup: the adapter (and its SciPy dependency) is only
    # imported after every input validation check has passed.
    from portlearn._optimizer_adapters import scipy_adapter

    return scipy_adapter.solve(covariance, identifiers)


def solve_long_only_mean_variance(
    covariance: Sequence[Sequence[float]],
    expected_returns: Sequence[float],
    risk_aversion: float,
    identifiers: Sequence[str],
) -> ScipySolveResult:
    """Validate inputs and delegate the long-only mean-variance solve.

    Validation performed here is exact (no numerical tolerance): the
    covariance matrix must be square with shape matching
    ``identifiers``, identifiers must be unique, the matrix must be
    exactly symmetric, ``expected_returns`` must be a length-``N``
    sequence of real finite numbers, and ``risk_aversion`` must be a
    finite strictly-positive real number. Only after validation passes
    is the optimizer adapter imported, so malformed input never
    depends on the optional optimization extra.

    Args:
        covariance: Symmetric covariance matrix with shape ``(N, N)``
            matching ``identifiers``.
        expected_returns: Expected-return vector of length ``N`` in
            ``identifiers`` order.
        risk_aversion: Finite strictly-positive risk-aversion scalar.
        identifiers: Ordered, unique identifiers, one per row/column of
            ``covariance``.

    Returns:
        The immutable solve result produced by the optimizer adapter,
        carried verbatim (no reordering, no wrapping).

    Raises:
        ValueError: If any validation check fails.
        OptimizationUnavailableError: If SciPy is not installed
            (propagated from the adapter).
        OptimizationFailureError: If the backend fails (propagated from
            the adapter).
    """
    n = len(identifiers)
    if n == 0:
        message = "identifiers must name at least one asset"
        raise ValueError(message)

    if len(covariance) != n:
        message = (
            f"covariance shape mismatch: expected {n} rows for {n} "
            f"identifiers, got {len(covariance)} rows"
        )
        raise ValueError(message)

    for index, row in enumerate(covariance):
        if len(row) != n:
            message = f"covariance row {index} has length {len(row)}; expected {n}"
            raise ValueError(message)

    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in identifiers:
        if name in seen:
            duplicates.add(name)
        else:
            seen.add(name)
    if duplicates:
        message = f"identifiers must be unique; duplicates: {sorted(duplicates)}"
        raise ValueError(message)

    for i in range(n):
        for j in range(i + 1, n):
            if covariance[i][j] != covariance[j][i]:
                message = (
                    "covariance must be exactly symmetric: "
                    f"covariance[{i}][{j}] == {covariance[i][j]!r} but "
                    f"covariance[{j}][{i}] == {covariance[j][i]!r}"
                )
                raise ValueError(message)

    if len(expected_returns) != n:
        message = (
            f"expected_returns length mismatch: expected {n} entries for "
            f"{n} identifiers, got {len(expected_returns)} entries"
        )
        raise ValueError(message)
    for index, value in enumerate(expected_returns):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            message = (
                "expected_returns entries must be real numbers; entry "
                f"{index} is {type(value).__name__}"
            )
            raise ValueError(  # noqa: TRY004 — the rejection surface is ValueError-only, mirroring the solve law
                message
            )
        if not math.isfinite(float(value)):
            message = (
                "expected_returns entries must be finite; entry "
                f"{index} ({identifiers[index]}) is {value!r}"
            )
            raise ValueError(message)

    if isinstance(risk_aversion, bool) or not isinstance(risk_aversion, numbers.Real):
        message = (
            f"risk_aversion must be a real number; got {type(risk_aversion).__name__}"
        )
        raise ValueError(  # noqa: TRY004 — the rejection surface is ValueError-only, mirroring the solve law
            message
        )
    risk_aversion_float = float(risk_aversion)
    if not math.isfinite(risk_aversion_float):
        message = f"risk_aversion must be finite; got {risk_aversion!r}"
        raise ValueError(message)
    if risk_aversion_float <= 0.0:
        message = f"risk_aversion must be strictly positive; got {risk_aversion!r}"
        raise ValueError(message)

    # Late lookup: the adapter (and its SciPy dependency) is only
    # imported after every input validation check has passed.
    from portlearn._optimizer_adapters import scipy_adapter

    return scipy_adapter.solve_mean_variance(
        covariance, expected_returns, risk_aversion_float, identifiers
    )
