"""Portfolio-weight value objects, exposures, and TARGET validation.

This module defines the portfolio-weight contract: the closed
three-member ``WeightState`` vocabulary, the immutable snapshotted
``PortfolioWeights`` value object, the standing ``WeightConstraints``
declarations with strict well-formedness and scalar declaration
coherence, the pure exposure functions over any valid weight mapping,
and the single TARGET-only validator ``require_valid_target``.

The error surface is ``ValueError`` only, with precise strict
messages; this module owns no instants and reuses no timing or
observations errors. The module is stdlib-only by design.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

__all__ = [
    "PortfolioWeights",
    "WeightConstraints",
    "WeightState",
    "gross_exposure",
    "net_exposure",
    "require_valid_target",
]


class WeightState(enum.Enum):
    """The closed three-state vocabulary of a portfolio-weights object.

    ``TARGET`` is what a strategy desires before trading; ``PRE_TRADE``
    is what the portfolio holds at a decision instant before execution;
    ``POST_TRADE`` is what the portfolio holds after execution. The
    vocabulary is closed; introducing another state requires an explicit
    public contract change.
    """

    TARGET = "TARGET"
    PRE_TRADE = "PRE_TRADE"
    POST_TRADE = "POST_TRADE"


def _require_identifier(key: object) -> None:
    """Every identifier is a non-blank exact string, never normalized."""
    if not isinstance(key, str) or not key.strip():
        raise ValueError(
            f"identifier must be a non-blank exact string; got {key!r}"
        )


def _checked_float(value: float, name: str) -> float:
    """Checked real→float conversion: an int beyond the float range
    raises ``OverflowError`` from ``float()`` — mapped here onto the
    unconditional finiteness check so the module error surface stays
    ``ValueError`` only. No clipping, no normalization: an out-of-range
    real is rejected, never rescaled."""
    try:
        converted = float(value)
    except OverflowError:
        converted = math.inf if value > 0 else -math.inf
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return converted


def _require_finite_weight(value: object) -> None:
    """Every weight is a finite real number; ``bool`` is rejected."""
    if isinstance(value, bool):
        raise ValueError(  # noqa: TRY004 — bool rejection is contract, not type discipline
            f"bool is not a weight; got {value!r}"
        )
    if not isinstance(value, (int, float)):
        raise ValueError(  # noqa: TRY004 — the error surface is ValueError-only by contract
            f"weight must be a real number; got {type(value).__name__}"
        )
    _checked_float(value, "weight")


def _finite_fsum(values: list[float], name: str) -> float:
    """``math.fsum`` guarded so the aggregation itself can never leak
    ``OverflowError`` or return a non-finite total: individually finite
    values whose exact sum overflows the float range are rejected
    unconditionally with ``ValueError`` (ValueError-only error surface)."""
    try:
        total = math.fsum(values)
    except OverflowError:
        raise ValueError(f"{name} must be finite; got non-finite sum") from None
    if not math.isfinite(total):
        raise ValueError(f"{name} must be finite; got non-finite sum")
    return float(total)


def gross_exposure(weights: Mapping[str, float]) -> float:
    """``gross_exposure = Σ|w|`` over the declared support.

    A pure module-level function over any mapping, enforcing the same
    valid-weight-mapping definition as ``PortfolioWeights`` construction
    (identifier and weight checks) strict, accumulated with
    ``math.fsum`` for exact, order-independent summation. It implies no
    leverage limit by existing: permissible exposure limits exist only
    where declared (``WeightConstraints``).
    """
    for key, value in weights.items():
        _require_identifier(key)
        _require_finite_weight(value)
    return _finite_fsum(
        [abs(_checked_float(w, "weight")) for w in weights.values()],
        "gross_exposure",
    )


def net_exposure(weights: Mapping[str, float]) -> float:
    """``net_exposure = Σw`` over the declared support.

    A pure module-level function over any mapping (so a raw
    ``PortfolioDecision.target_weights`` can be measured without
    constructing a ``PortfolioWeights``), enforcing the same
    valid-weight-mapping definition strict, accumulated with
    ``math.fsum``.
    """
    for key, value in weights.items():
        _require_identifier(key)
        _require_finite_weight(value)
    return _finite_fsum(
        [_checked_float(w, "weight") for w in weights.values()],
        "net_exposure",
    )


class PortfolioWeights:
    """The immutable, defensively snapshotted portfolio-weights object.

    Public shape: constructor ``PortfolioWeights(weights, state)``;
    public read attribute ``weights`` — the immutable mapping view
    snapshotted at initialization, so later modify of the source
    mapping cannot change the object and modify of the view itself
    raises; public read attribute ``state: WeightState`` holding an
    actual enum member (a string, an int, ``None``, or a bool is
    rejected with no coercion); ``assets: immutableset[str]`` returning
    exactly the explicitly declared identifiers, explicit zeros
    included; and ``weight_of(identifier) -> float`` returning exactly
    ``0.0`` for an identifier absent from ``assets``. An immutable
    hand-written class realizes the snapshot guarantee; class mechanics
    are an implementation choice, the public shape above is the public
    contract.
    """

    __slots__ = ("state", "weights")

    def __init__(
        self, weights: Mapping[str, float], state: WeightState
    ) -> None:
        if not isinstance(state, WeightState):
            raise ValueError(  # noqa: TRY004 — the error surface is ValueError-only by contract
                "state must be an actual WeightState member "
                f"(TARGET/PRE_TRADE/POST_TRADE); got {state!r}"
            )
        snapshot: dict[str, float] = {}
        for key, value in weights.items():
            _require_identifier(key)
            _require_finite_weight(value)
            snapshot[key] = _checked_float(value, "weight")
        # Construction writes through object.__setattr__ only — the
        # public __setattr__/__delattr__ below unconditionally reject, so
        # the object is immutable from the instant it exists. The snapshot
        # dict is a local: only its immutable MappingProxyType view is
        # retained, so no mutable backing dict alias exists on the object.
        object.__setattr__(self, "weights", MappingProxyType(snapshot))
        object.__setattr__(self, "state", state)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"{type(self).__name__} is an immutable value object; "
            f"attribute {name!r} cannot be assigned"
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            f"{type(self).__name__} is an immutable value object; "
            f"attribute {name!r} cannot be deleted"
        )

    @property
    def assets(self) -> frozenset[str]:
        """The declared key set, explicit zeros included."""
        return frozenset(self.weights)

    def weight_of(self, identifier: str) -> float:
        """An absent identifier has effective weight exactly 0.0."""
        return float(self.weights.get(identifier, 0.0))


def _require_real(name: str, value: object) -> float:
    """Well-formedness helper: a finite real; ``bool`` is not a number."""
    if isinstance(value, bool):
        raise ValueError(  # noqa: TRY004 — bool rejection is contract, not type discipline
            f"{name} must be a real number, not bool; got {value!r}"
        )
    if not isinstance(value, (int, float)):
        raise ValueError(  # noqa: TRY004 — the error surface is ValueError-only by contract
            f"{name} must be a real number; got {type(value).__name__}"
        )
    return _checked_float(value, name)


def _require_bound(name: str, value: object) -> float:
    """An optional exposure bound — finite, nonnegative, not a bool."""
    bound = _require_real(name, value)
    if bound < 0.0:
        raise ValueError(f"{name} must be nonnegative; got {bound!r}")
    return bound


@dataclass(frozen=True)
class WeightConstraints:
    """The portfolio's standing TARGET-level weight declarations.

    Fields: ``allow_short`` (no negative weight may exist in a validated
    ``TARGET`` when ``False``), ``max_gross_exposure`` (inclusive),
    ``max_abs_net_exposure`` (inclusive, two-sided on the absolute net
    exposure), ``budget`` (the declared net-exposure level for a
    ``TARGET`` — a declaration, not proof of full-wealth accounting),
    and ``budget_tolerance`` (default ``1e-9``, caller-settable).

    Well-formedness is strict at construction, and declaration
    coherence rejects a declaration only when no possible target net
    exposure can lie simultaneously within the budget interval
    ``[budget − budget_tolerance, budget + budget_tolerance]`` and the
    declared exposure limits: with ``L`` the least declared nonnegative
    limit present, ``abs(budget) > L + budget_tolerance`` is
    contradictory with shorts permitted, and ``budget <
    −budget_tolerance`` or ``budget > L + budget_tolerance`` is
    contradictory long-only; when neither exposure limit is declared
    there is no finite exposure ceiling and hence no exposure-ceiling
    contradiction, though long-only sign coherence still applies. This
    is scalar declaration-level satisfiability only — no individual
    asset weight is constrained and no optimizer feasibility of any
    kind is performed (portfolio optimization is out of scope here).
    """

    allow_short: bool = False
    max_gross_exposure: float | None = None
    max_abs_net_exposure: float | None = None
    budget: float = 1.0
    budget_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if not isinstance(self.allow_short, bool):
            raise ValueError(  # noqa: TRY004 — the error surface is ValueError-only by contract
                "allow_short must be a bool; got "
                f"{type(self.allow_short).__name__}"
            )
        if self.max_gross_exposure is not None:
            object.__setattr__(
                self,
                "max_gross_exposure",
                _require_bound("max_gross_exposure", self.max_gross_exposure),
            )
        if self.max_abs_net_exposure is not None:
            object.__setattr__(
                self,
                "max_abs_net_exposure",
                _require_bound(
                    "max_abs_net_exposure", self.max_abs_net_exposure
                ),
            )
        object.__setattr__(
            self, "budget", _require_real("budget", self.budget)
        )
        tolerance = _require_real("budget_tolerance", self.budget_tolerance)
        if tolerance <= 0.0:
            raise ValueError(
                f"budget_tolerance must be strictly positive; got {tolerance!r}"
            )
        object.__setattr__(self, "budget_tolerance", tolerance)
        self._require_declaration_coherent()

    def _require_declaration_coherent(self) -> None:
        """Declaration coherence: scalar TARGET-level satisfiability."""
        limits = [
            limit
            for limit in (self.max_gross_exposure, self.max_abs_net_exposure)
            if limit is not None
        ]
        if limits:
            ceiling = min(limits)
            if self.allow_short:
                if abs(self.budget) > ceiling + self.budget_tolerance:
                    raise ValueError(
                        "incoherent declaration: no possible target net "
                        "exposure within [budget − budget_tolerance, budget "
                        "+ budget_tolerance] and the declared limits "
                        f"(|budget|={abs(self.budget)!r} exceeds "
                        f"L + budget_tolerance = "
                        f"{ceiling + self.budget_tolerance!r})"
                    )
            elif (
                self.budget < -self.budget_tolerance
                or self.budget > ceiling + self.budget_tolerance
            ):
                raise ValueError(
                    "incoherent declaration: no possible long-only target "
                    "net exposure within [budget − budget_tolerance, budget "
                    "+ budget_tolerance] and the declared limits (budget "
                    "outside [-budget_tolerance, L + budget_tolerance])"
                )
        elif not self.allow_short and self.budget < -self.budget_tolerance:
            # No declared ceiling => no exposure-ceiling contradiction is
            # possible; long-only sign coherence still applies.
            raise ValueError(
                "incoherent declaration: a long-only target net exposure "
                "cannot be negative beyond budget_tolerance (budget="
                f"{self.budget!r})"
            )


def require_valid_target(
    weights: PortfolioWeights, constraints: WeightConstraints
) -> None:
    """Validate a ``TARGET`` against the declared constraints.

    The single validation entry, TARGET-only by contract: an input whose
    state is not ``WeightState.TARGET`` raises ``ValueError`` on the
    state safety check — ``PRE_TRADE`` and ``POST_TRADE`` objects are never
    passed through this validator and receive construction-time
    representational validity only. For a ``TARGET`` this validates the
    budget requirement ``|net_exposure − budget| <= budget_tolerance``,
    the short prohibition under ``allow_short=False``, and the inclusive
    ``max_gross_exposure``/``max_abs_net_exposure`` bounds. No generic
    realized-state validator exists or is implied.
    """
    if weights.state is not WeightState.TARGET:
        raise ValueError(
            "require_valid_target validates only WeightState.TARGET "
            f"objects; got state {weights.state!r} (PRE_TRADE/POST_TRADE "
            "receive construction-time representational validity only and "
            "are never passed through this validator)"
        )
    net = net_exposure(weights.weights)
    if abs(net - constraints.budget) > constraints.budget_tolerance:
        raise ValueError(
            f"TARGET net exposure {net!r} violates declared budget "
            f"{constraints.budget!r} within budget_tolerance "
            f"{constraints.budget_tolerance!r}"
        )
    if not constraints.allow_short:
        for identifier, weight in weights.weights.items():
            if weight < 0.0:
                raise ValueError(
                    f"negative weight {weight!r} on {identifier!r} is "
                    "forbidden when allow_short=False"
                )
    gross = gross_exposure(weights.weights)
    if (
        constraints.max_gross_exposure is not None
        and gross > constraints.max_gross_exposure
    ):
        raise ValueError(
            f"TARGET gross exposure {gross!r} exceeds max_gross_exposure "
            f"{constraints.max_gross_exposure!r} (inclusive bound)"
        )
    if (
        constraints.max_abs_net_exposure is not None
        and abs(net) > constraints.max_abs_net_exposure
    ):
        raise ValueError(
            f"TARGET |net exposure| {abs(net)!r} exceeds "
            f"max_abs_net_exposure "
            f"{constraints.max_abs_net_exposure!r} (inclusive, two-sided)"
        )
