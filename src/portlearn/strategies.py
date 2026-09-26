"""The built-in classical strategies.

This module locates public strategy classes — nothing more. A strategy
is the decision-time primitive of PortLearn: it consumes
one :class:`~portlearn.interfaces.DecisionContext` (the decision-time
aggregate over the caller-declared universe, the admitted information,
the pre-trade book, and any carried state or forecast) and returns one
:class:`~portlearn.interfaces.DecisionResult` carrying a target weight
book dated exactly at the context's authoritative ``decision_time``.

The built-ins are :class:`EqualWeight` (the equal-weight allocation over the effective support), :class:`InverseVolatility` (the inverse-volatility risk-based allocation over the full caller-declared universe), :class:`MinimumVariance` (the long-only fully-invested minimum-variance allocation over the full caller-declared universe), and :class:`MeanVariance` (the long-only fully-invested mean-variance allocation over the full caller-declared universe).

The first built-in is :class:`EqualWeight`, the equal-weight allocation
over an effective support that is either declared at construction or
taken dynamically as each decision's caller-declared universe. It is
deliberately finance-first and minimal: long-only, fully invested to
the unit budget, stateless, with an allocation that reads neither the
carried strategy state nor any forecast. Turnover, costs, and accounting
are consumed downstream through their own interfaces — nothing
here computes or applies them.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Final

from portlearn import _allocation
from portlearn import interfaces as _interfaces
from portlearn.timing import instant_key
from portlearn.weights import (
    PortfolioWeights,
    WeightConstraints,
    WeightState,
    require_valid_target,
)

__all__ = [  # noqa: RUF022 — the fixed contract pins this declaration order
    "EqualWeight",
    "InverseVolatility",
    "MinimumVariance",
    "MeanVariance",
]

# The fixed post-solve acceptance floors: the
# active-set classification threshold tau_w and the marginal-gradient
# (KKT) tolerance kappa are pinned module constants, never
# caller-settable, and the domination probe epsilons are fixed too.
_TAU_W: float = 1e-12
_KAPPA: float = 1e-8
_DOMINATION_EPSILONS: tuple[float, ...] = (1e-6, 1e-4, 1e-2)

#: Deterministic relative-improvement floor separating a genuinely
#: dominated vector (whose projected neighbor improves the objective
#: by orders of magnitude) from the few-ulp evaluation noise at an
#: exact corner solution the projection reproduces.
_DOMINATION_TIE: float = 1e-13

# The canonical supported constraints declaration, built once at
# import time: long-only, fully invested, and nothing else.
_CANONICAL_CONSTRAINTS: Final[WeightConstraints] = WeightConstraints()


class EqualWeight:
    """Equal weight over the effective support (declared, or the
    current universe). Long-only, fully invested, stateless.

    ``EqualWeight()`` constructs the **dynamic full-universe** mode:
    the effective support at each decision is exactly that decision's
    ``context.universe``. ``EqualWeight(declared_support=S)`` constructs
    the **declared-support** mode: the effective support is exactly the
    declared, order-preserved instrument list ``S`` at every decision.

    Invalid inputs raise ``ValueError``: PortLearn does not silently
    repair or substitute them. A bare string rejects (a single string
    names one instrument, not a support), an empty sequence rejects,
    each entry must be a non-blank exact string (no case folding, no
    trimming, no normalization), and duplicates reject. Caller-declared
    order is preserved exactly. The declaration is snapshotted
    immutably at construction, so later modify of a caller-supplied
    list cannot change any future decision.

    At :meth:`decide` time the allocation law is exact: ``n`` being the
    size of the effective support, every member receives exactly the
    binary float ``1/n`` and the target contains exactly the effective
    support keys — no cash residual sleeve, no implicit ``CASH`` key,
    no non-support member. In declared-support mode every declared
    member must be present in the current ``context.universe``; any
    absent member raises an informative ``ValueError``. An empty
    effective support raises the same way: an equal-weight target over
    no instruments names no portfolio. The decided result executes at
    the same instant it is decided (``execution_time == decision_time``,
    the reference convention) and carries ``next_strategy_state=None``.
    """

    def __init__(self, declared_support: Sequence[str] | None = None) -> None:
        """Validate and snapshot the optional declared support.

        ``declared_support=None`` (the default) selects dynamic
        full-universe mode. Any other value must be a non-string
        sequence of non-blank, exact-string, duplicate-free instrument
        identifiers in caller-declared order; the constructor rejects a
        bare string, an empty sequence, a blank entry, a non-string
        entry, and a duplicate entry each with its own ``ValueError``
        naming the violated declaration law.
        """
        if declared_support is None:
            self._declared_support: tuple[str, ...] | None = None
            return
        if isinstance(declared_support, str):
            raise ValueError(  # noqa: TRY004 — the rejection surface is ValueError-only, mirroring the DecisionContext.universe law
                "declared_support must be a sequence of exact-string "
                "instrument identifiers in caller-declared order, not a "
                f"single string; got {declared_support!r}. A single "
                "string names one instrument, not a support, so the "
                "declaration is rejected unconditionally."
            )
        entries: list[str] = []
        seen: set[str] = set()
        for entry in declared_support:
            if not isinstance(entry, str):
                raise ValueError(  # noqa: TRY004 — the rejection surface is ValueError-only, mirroring the identifier law
                    "declared_support must be a sequence of exact-string "
                    "instrument identifiers; got a non-string entry "
                    f"{type(entry).__name__}: {entry!r}. A non-string "
                    "entry names no instrument, so the declaration is "
                    "rejected unconditionally."
                )
            if not entry.strip():
                raise ValueError(
                    "declared_support must be a sequence of non-blank "
                    "exact-string instrument identifiers; got the blank "
                    f"entry {entry!r}. A blank identifier names no "
                    "instrument, so the declaration is rejected "
                    "unconditional."
                )
            if entry in seen:
                raise ValueError(
                    f"duplicate declared_support entry: {entry!r} appears "
                    "more than once in the declared support - the "
                    "declaration is a sequence of distinct instrument "
                    "identifiers in caller-declared order, so a "
                    "duplicate names no additional instrument and is "
                    "rejected unconditionally."
                )
            seen.add(entry)
            entries.append(entry)
        if not entries:
            raise ValueError(
                "declared_support must name a non-empty support: an empty "
                "declaration equal-weights no instruments, so it names no "
                "portfolio and is rejected unconditionally. Omit "
                "declared_support (None) for dynamic full-universe mode."
            )
        self._declared_support = tuple(entries)

    @property
    def declared_support(self) -> tuple[str, ...] | None:
        """The immutable declared-support snapshot, or ``None``.

        Returns the tuple snapshotted at construction in
        declared-support mode, and ``None`` in dynamic full-universe
        mode. The property is read-only: no setter exists, so direct
        reassignment raises ``AttributeError`` and the declared support
        cannot be modify through the public surface.
        """
        return self._declared_support

    def decide(
        self, context: _interfaces.DecisionContext
    ) -> _interfaces.DecisionResult:
        """Decide the equal-weight target for one decision context.

        The effective support is the construction snapshot in
        declared-support mode — every declared member must be present
        in ``context.universe`` at this decision, else ``ValueError``
        naming the absent member(s) and the ``decision_time`` — and
        otherwise ``context.universe`` itself, which must be non-empty.
        Each of the ``n`` members receives exactly ``1/n``; the target
        keys are exactly the effective support in order. The result is
        dated and executed at ``context.decision_time`` (same-instant
        convention), carries ``next_strategy_state=None``, and is
        passed through the public validator
        ``require_decision_result_compatible`` before being returned.

        The allocation is independent of ``context.forecast`` and of
        ``context.strategy_state``: neither is required to produce the
        target, neither is modify, and no forecast is fabricated.
        """
        if self._declared_support is None:
            effective_support = tuple(context.universe)
            if not effective_support:
                raise ValueError(
                    "the equal-weight effective support is empty: the "
                    "decision context's universe is empty at "
                    f"decision_time={context.decision_time.isoformat()}, "
                    "and an equal-weight target over no instruments names "
                    "no portfolio, so the decision is rejected unconditionally."
                )
        else:
            absent = [
                member
                for member in self._declared_support
                if member not in context.universe
            ]
            if absent:
                raise ValueError(
                    "stale declared support: the declared member(s) "
                    f"{', '.join(repr(member) for member in absent)} are "
                    "absent from the decision context's universe at "
                    f"decision_time={context.decision_time.isoformat()}. "
                    "The declared support is honored exactly or not at "
                    "all - the strategy never silently intersects it "
                    "with the current universe."
                )
            effective_support = self._declared_support
        weight = 1 / len(effective_support)
        result = _interfaces.DecisionResult(
            decision=_interfaces.PortfolioDecision(
                decision_time=context.decision_time,
                execution_time=context.decision_time,
                target_weights={member: weight for member in effective_support},
            ),
            next_strategy_state=None,
        )
        _interfaces.require_decision_result_compatible(context, result)
        return result


class InverseVolatility:
    """Inverse-volatility allocation over the full universe.

    A classical **risk-based allocation baseline**: each universe asset
    receives a weight proportional to the inverse square root of its
    own return dispersion, estimated over the latest balanced window of
    admitted returns. Like equal weight it is a heuristic primitive —
    not an optimizer, forecaster, or selector — and it is finance-first
    and minimal: long-only, fully invested through the normalization
    itself, stateless, and reading neither the carried strategy state,
    any forecast, nor the pre-trade book. Holdings affect only
    downstream trade sizing, turnover, costs, and accounting — never
    target formation.

    ``InverseVolatility(return_series, window)`` maps each asset
    identifier to exactly one return-series identifier (one series
    cannot serve two assets) and fixes the estimation window length
    (an integer ``>= 2``). Invalid inputs raise ``ValueError``; the
    mapping is snapshotted immutably at construction, so later
    modify of a caller-supplied mapping cannot change any future
    decision.

    At :meth:`decide` time the effective support is exactly that
    decision's ``context.universe`` (the universe *is* the support — no
    declared-support mode, no selection, ranking, or filtering). For
    each support asset the latest ``window`` admitted records of its
    mapped series are selected on a common balanced observation grid;
    the estimand per asset is the centered sum of squares
    ``D_i = fsum((r - mean)**2)`` over the window, the score is
    ``s_i = 1/sqrt(D_i)``, and the target weight is ``s_i / fsum(s)``.
    Zero or nonfinite dispersion, non-real or non-representable return
    values, unbalanced grids, insufficient history, missing mappings,
    and an empty universe each raise a named ``ValueError`` — no
    epsilon, no clipping, no equal-weight fallback, no dropping an
    asset. The decided result executes at the same instant it is
    decided (``execution_time == decision_time``, the reference
    convention) and carries ``next_strategy_state=None``.
    """

    def __init__(self, return_series: Mapping[str, str], window: int) -> None:
        """Validate and snapshot the mapping and window (invalid
        inputs raise ``ValueError``)."""
        if isinstance(return_series, str):
            raise ValueError(  # noqa: TRY004 — the rejection surface is ValueError-only, mirroring the EqualWeight bare-string law
                "return_series must be a mapping of asset identifier to "
                "return-series identifier, not str: a single string names "
                "one return series, not an asset-to-series mapping; got "
                f"{return_series!r}."
            )
        if not isinstance(return_series, Mapping):
            raise ValueError(  # noqa: TRY004 — the rejection surface is ValueError-only, mirroring the EqualWeight constructor law
                "return_series must be a mapping of asset identifier to "
                "return-series identifier; got "
                f"{type(return_series).__name__}: {return_series!r}."
            )
        if not return_series:
            raise ValueError(
                "return_series must be a non-empty mapping of asset "
                "identifier to return-series identifier: an "
                "inverse-volatility target over no mapped return series "
                "names no portfolio, so the configuration is rejected "
                "unconditional."
            )
        series_by_asset: dict[str, str] = {}
        asset_by_series: dict[str, str] = {}
        for key, value in return_series.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    "return_series keys must be non-blank strings (exact "
                    "asset identifiers; no case folding, trimming, or "
                    "normalization); got the key "
                    f"{key!r} of type {type(key).__name__}."
                )
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "return_series values must be non-blank series "
                    "identifiers (exact series identities; no case "
                    "folding, trimming, or normalization); the value for "
                    f"asset {key!r} is {value!r} of type "
                    f"{type(value).__name__}."
                )
            if value in asset_by_series:
                raise ValueError(
                    "duplicate return-series mapping: the return series "
                    f"{value!r} is mapped to both asset "
                    f"{asset_by_series[value]!r} and asset {key!r}; one "
                    "return series cannot serve two assets - the mapping "
                    "is the explicit asset-to-series identity, and "
                    "sharing would double-count one series' information "
                    "as two assets' returns, so the configuration is "
                    "rejected unconditionally."
                )
            asset_by_series[value] = key
            series_by_asset[key] = value
        if isinstance(window, bool) or not isinstance(window, int):
            raise ValueError(  # noqa: TRY004 — the unconditional window law is ValueError-only by the fixed contract
                "window must be an integer >= 2 (the estimation window "
                "length); got "
                f"{type(window).__name__} (window={window!r}). The "
                "window is an exact observation count, not a flag or an "
                "approximation, so the configuration is rejected "
                "unconditional."
            )
        if window < 2:
            raise ValueError(
                f"window must be an integer >= 2; got window={window} - "
                "a dispersion estimate over fewer than two returns is "
                "undefined, so the configuration is rejected unconditionally."
            )
        self._return_series = series_by_asset
        self._window = window

    @property
    def return_series(self) -> Mapping[str, str]:
        """Read-only view of the immutable asset-to-series snapshot.

        Returns a read-only mapping over the snapshot taken at
        construction — modify the view raises ``TypeError``, and
        modify the original caller mapping after construction leaves
        the snapshot and all future targets unchanged. The property is
        read-only: no setter exists, so direct reassignment raises
        ``AttributeError``.
        """
        return MappingProxyType(self._return_series)

    @property
    def window(self) -> int:
        """The read-only immutable estimation window (integer >= 2)."""
        return self._window

    def decide(
        self, context: _interfaces.DecisionContext
    ) -> _interfaces.DecisionResult:
        """Decide the inverse-volatility target for one decision context.

        The effective support is exactly ``tuple(context.universe)``;
        an empty universe and any universe asset without a
        ``return_series`` entry raise a named
        ``ValueError``. For each support asset the latest ``window``
        admitted records of its mapped series are selected (sorted on
        normalized observation instants) and the ordered instants must
        form one common balanced grid across all support assets. Each
        selected value must be a real, finitely representable return;
        the per-asset dispersion is the centered sum of squares
        ``fsum((r - mean)**2)`` (strictly positive and finite, else
        ``ValueError``), the score is ``1/sqrt(D)``, and the weights are
        the scores normalized by their sum, in universe key order.

        The result is dated and executed at ``context.decision_time``
        (same-instant convention), carries ``next_strategy_state=None``,
        and is passed through the immutable public validator
        ``require_decision_result_compatible`` before being returned.
        The target is independent of ``context.forecast``,
        ``context.strategy_state``, ``context.current_weights``, and
        ``current_weights_as_of``: none is required to produce the
        target and none is modify.
        """
        support = tuple(context.universe)
        if not support:
            raise ValueError(
                "the inverse-volatility effective support is empty: the "
                "decision context's universe is empty at "
                f"decision_time={context.decision_time.isoformat()}, and "
                "an inverse-volatility target over no assets names no "
                "portfolio, so the decision is rejected unconditionally."
            )
        missing = [asset for asset in support if asset not in self._return_series]
        if missing:
            raise ValueError(
                "missing return-series mapping: the universe asset(s) "
                f"{', '.join(repr(asset) for asset in missing)} have no "
                "return_series entry at "
                f"decision_time={context.decision_time.isoformat()} - "
                "the universe is the effective support and is never "
                "selected, ranked, or filtered, so the decision is "
                "rejected unconditionally."
            )
        window = self._window
        mapped_series = {self._return_series[asset] for asset in support}
        buckets: dict[str, list] = {series: [] for series in mapped_series}
        for record in context.information:
            series_id = record.series_id
            if series_id in buckets:
                buckets[series_id].append(record)
        grids: dict[str, tuple] = {}
        selected: dict[str, tuple] = {}
        for asset in support:
            series = self._return_series[asset]
            ordered = sorted(
                buckets[series],
                key=lambda record: instant_key(record.observation_time),
            )
            available = len(ordered)
            if available < window:
                raise ValueError(
                    "insufficient history: asset "
                    f"{asset!r} series {series!r} has {available} "
                    f"admissible record(s) available but window={window} "
                    f"requires {window} - no partial target and no "
                    "truncation exist, so the decision is rejected "
                    "unconditional."
                )
            latest = tuple(ordered[-window:])
            grids[asset] = tuple(
                instant_key(record.observation_time) for record in latest
            )
            selected[asset] = latest
        reference = support[0]
        for asset in support[1:]:
            if grids[asset] != grids[reference]:
                raise ValueError(
                    "unbalanced observation grid: assets "
                    f"{reference!r} and {asset!r} selected different "
                    "observation instants over the window - the estimand "
                    "is defined on one balanced common grid of the same "
                    "observation periods, and no mismatched-date "
                    "cross-section is estimated, so the decision is "
                    "rejected unconditionally."
                )
        scores: dict[str, float] = {}
        for asset in support:
            series = self._return_series[asset]
            returns: list[float] = []
            for record in selected[asset]:
                value = record.value
                if isinstance(value, bool):
                    raise ValueError(  # noqa: TRY004 — the E5 non-real law is ValueError-only by the fixed contract
                        "non-real return value: asset "
                        f"{asset!r} series {series!r} carries the bool "
                        f"value {value!r} at "
                        f"observation_time="
                        f"{record.observation_time.isoformat()} - bool "
                        "is not a return, so the decision is rejected "
                        "unconditional."
                    )
                if not isinstance(value, numbers.Real):
                    raise ValueError(  # noqa: TRY004 — the E5 non-real law is ValueError-only by the fixed contract
                        "non-real return value: asset "
                        f"{asset!r} series {series!r} carries a "
                        f"{type(value).__name__} value {value!r} at "
                        f"observation_time="
                        f"{record.observation_time.isoformat()} - a "
                        "return must be a real scalar, so the decision "
                        "is rejected unconditionally."
                    )
                try:
                    converted = float(value)
                except OverflowError:
                    raise ValueError(
                        "non-representable or nonfinite return value: "
                        f"asset {asset!r} series {series!r} carries the "
                        f"real value {value!r} at observation_time="
                        f"{record.observation_time.isoformat()} whose "
                        "float conversion overflows - the value cannot "
                        "be represented as a finite float return, so "
                        "the decision is rejected unconditionally."
                    ) from None
                if not math.isfinite(converted):
                    raise ValueError(
                        "non-representable or nonfinite return value: "
                        f"asset {asset!r} series {series!r} carries the "
                        f"value {value!r} at observation_time="
                        f"{record.observation_time.isoformat()} that "
                        "converts to a nonfinite float - the value "
                        "cannot be represented as a finite float "
                        "return, so the decision is rejected unconditionally."
                    )
                returns.append(converted)
            try:
                mean = math.fsum(returns) / window
                dispersion = math.fsum((value - mean) ** 2 for value in returns)
            except OverflowError:
                raise ValueError(
                    "zero or nonfinite return dispersion: asset "
                    f"{asset!r} series {series!r} - the centered sum of "
                    "squared deviations over the window overflowed the "
                    "dispersion arithmetic; the estimand is undefined "
                    "here, no epsilon, no clipping, and no equal-weight "
                    "fallback exist, and finite inputs whose dispersion "
                    "arithmetic overflows reject the same way, so the "
                    "decision is rejected unconditionally."
                ) from None
            if not math.isfinite(dispersion) or dispersion <= 0.0:
                raise ValueError(
                    "zero or nonfinite return dispersion: asset "
                    f"{asset!r} series {series!r} - the centered sum of "
                    "squared deviations over the window is "
                    f"{dispersion!r}, so the inverse-dispersion score is "
                    "undefined; no epsilon, no clipping, and no "
                    "equal-weight fallback exist, and finite inputs "
                    "whose dispersion arithmetic overflows reject the "
                    "same way, so the decision is rejected unconditionally."
                )
            try:
                score = 1 / math.sqrt(dispersion)
            except OverflowError:
                raise ValueError(
                    "zero or nonfinite return dispersion: asset "
                    f"{asset!r} series {series!r} - the "
                    "inverse-square-root score overflows; the estimand "
                    "is undefined here, no epsilon, no clipping, and no "
                    "equal-weight fallback exist, and finite inputs "
                    "whose dispersion arithmetic overflows reject the "
                    "same way, so the decision is rejected unconditionally."
                ) from None
            if not math.isfinite(score):
                raise ValueError(
                    "zero or nonfinite return dispersion: asset "
                    f"{asset!r} series {series!r} - the "
                    "inverse-square-root score is nonfinite; the "
                    "estimand is undefined here, no epsilon, no "
                    "clipping, and no equal-weight fallback exist, and "
                    "finite inputs whose dispersion arithmetic "
                    "overflows reject the same way, so the decision is "
                    "rejected unconditionally."
                )
            scores[asset] = score
        normalizer = math.fsum(scores[asset] for asset in support)
        if not math.isfinite(normalizer) or normalizer <= 0.0:
            raise ValueError(
                "inverse-volatility normalizer is nonpositive or "
                f"nonfinite ({normalizer!r}); the normalized target is "
                "undefined, so the decision is rejected unconditionally."
            )
        target_weights: dict[str, float] = {}
        for asset in support:
            weight = scores[asset] / normalizer
            if not math.isfinite(weight):
                raise ValueError(
                    "inverse-volatility output weight for asset "
                    f"{asset!r} is nonfinite under normalizer "
                    f"{normalizer!r}; the normalized target is invalid, "
                    "so the decision is rejected unconditionally."
                )
            target_weights[asset] = weight
        result = _interfaces.DecisionResult(
            decision=_interfaces.PortfolioDecision(
                decision_time=context.decision_time,
                execution_time=context.decision_time,
                target_weights=target_weights,
            ),
            next_strategy_state=None,
        )
        _interfaces.require_decision_result_compatible(context, result)
        return result


class _TargetWeightBook(dict[str, float]):
    """The MinimumVariance TARGET presentation: a plain dict (so key
    order is universe order and value equality is exact exact equality)
    that additionally exposes a read-only ``.weights`` view over
    itself."""

    @property
    def weights(self) -> Mapping[str, float]:
        """A read-only mapping view over this exact dict."""
        return MappingProxyType(self)


def _require_canonical_constraints(constraints: WeightConstraints) -> None:
    """Reject every constraint declaration outside the canonical
    supported subset (long-only, fully invested, no exposure caps),
    naming the field, its value, and the subset."""
    if constraints.allow_short:
        raise ValueError(
            "unsupported constraints declaration: MinimumVariance "
            "supports exactly the canonical long-only fully-invested "
            "subset (WeightConstraints() with allow_short=False, "
            "budget=1.0, and no exposure caps), but "
            f"allow_short={constraints.allow_short!r} declares a "
            "shorting permission outside that subset, so the decision "
            "is rejected unconditionally."
        )
    if constraints.budget != 1.0:
        raise ValueError(
            "unsupported constraints declaration: MinimumVariance "
            "supports exactly the canonical long-only fully-invested "
            "subset (WeightConstraints() with allow_short=False, "
            "budget=1.0, and no exposure caps), but "
            f"budget={constraints.budget!r} declares a "
            "partial-investment level outside that subset, so the "
            "decision is rejected unconditionally."
        )
    if constraints.max_gross_exposure is not None:
        raise ValueError(
            "unsupported constraints declaration: MinimumVariance "
            "supports exactly the canonical long-only fully-invested "
            "subset (WeightConstraints() with allow_short=False, "
            "budget=1.0, and no exposure caps), but "
            f"max_gross_exposure={constraints.max_gross_exposure!r} "
            "declares a gross-exposure cap outside that subset, so the "
            "decision is rejected unconditionally."
        )
    if constraints.max_abs_net_exposure is not None:
        raise ValueError(
            "unsupported constraints declaration: MinimumVariance "
            "supports exactly the canonical long-only fully-invested "
            "subset (WeightConstraints() with allow_short=False, "
            "budget=1.0, and no exposure caps), but "
            "max_abs_net_exposure="
            f"{constraints.max_abs_net_exposure!r} declares a "
            "net-exposure cap outside that subset, so the decision is "
            "rejected unconditionally."
        )


def _project_to_simplex(vector: Sequence[float]) -> list[float]:
    """The Euclidean projection of one point onto the probability
    simplex by the standard sorting algorithm - deterministic, with no
    randomness anywhere."""
    size = len(vector)
    ordered = sorted(vector, reverse=True)
    cumulative = [ordered[0]]
    for value in ordered[1:]:
        cumulative.append(cumulative[-1] + value)
    rho = 0
    for index in range(size):
        if ordered[index] - (cumulative[index] - 1.0) / (index + 1) > 0.0:
            rho = index
    theta = (cumulative[rho] - 1.0) / (rho + 1)
    return [max(0.0, value - theta) for value in vector]


def _quadratic_form(sigma: Sequence[Sequence[float]], vector: Sequence[float]) -> float:
    """``w' Sigma w`` accumulated with ``math.fsum`` - deterministic
    and order-independent for a fixed matrix and vector."""
    size = len(vector)
    return math.fsum(
        sigma[i][j] * vector[i] * vector[j] for i in range(size) for j in range(size)
    )


def _exact_rank(matrix: list[list[Fraction]]) -> int:
    """The exact rank of a rational matrix by Gaussian elimination
    over the rationals - no floating tolerance anywhere."""
    work = [row[:] for row in matrix]
    row_count = len(work)
    column_count = len(work[0])
    rank = 0
    for column in range(column_count):
        pivot = next(
            (r for r in range(rank, row_count) if work[r][column] != 0),
            None,
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][column]
        for r in range(rank + 1, row_count):
            if work[r][column] != 0:
                factor = work[r][column] / pivot_value
                work[r] = [a - factor * b for a, b in zip(work[r], work[rank])]
        rank += 1
    return rank


def _demeaned_fraction_panel(
    columns: Sequence[Sequence[float]], window: int
) -> list[list[Fraction]]:
    """The demeaned observation-by-asset panel over the exact
    rationals, for the exact-rank reference."""
    deviations: list[list[Fraction]] = []
    for column in columns:
        fraction_column = [Fraction(value) for value in column]
        mean = sum(fraction_column, Fraction(0)) / window
        deviations.append([value - mean for value in fraction_column])
    return [[deviations[j][t] for j in range(len(columns))] for t in range(window)]


def _require_real_column(
    asset: str,
    series: str,
    selected: Sequence[Any],
) -> list[float]:
    """Convert one asset's selected records to finite real floats,
    rejecting every non-real or non-representable value."""
    column: list[float] = []
    for record in selected:
        value = record.value
        if isinstance(value, bool):
            raise ValueError(  # noqa: TRY004 — the non-real law is ValueError-only by the fixed contract
                "non-real return value: asset "
                f"{asset!r} series {series!r} carries the bool value "
                f"{value!r} at observation_time="
                f"{record.observation_time.isoformat()} - bool is not "
                "a return, so the decision is rejected unconditionally."
            )
        if not isinstance(value, numbers.Real):
            raise ValueError(  # noqa: TRY004 — the non-real law is ValueError-only by the fixed contract
                "non-real return value: asset "
                f"{asset!r} series {series!r} carries a "
                f"{type(value).__name__} value {value!r} at "
                f"observation_time="
                f"{record.observation_time.isoformat()} - a return "
                "must be a real scalar, so the decision is rejected "
                "unconditional."
            )
        try:
            converted = float(value)
        except OverflowError:
            raise ValueError(
                "non-representable or nonfinite return value: asset "
                f"{asset!r} series {series!r} carries the real value "
                f"{value!r} at observation_time="
                f"{record.observation_time.isoformat()} whose float "
                "conversion overflows - the value cannot be "
                "represented as a finite float return, so the decision "
                "is rejected unconditionally."
            ) from None
        if not math.isfinite(converted):
            raise ValueError(
                "non-representable or nonfinite return value: asset "
                f"{asset!r} series {series!r} carries the value "
                f"{value!r} at observation_time="
                f"{record.observation_time.isoformat()} that converts "
                "to a nonfinite float - the value cannot be "
                "represented as a finite float return, so the decision "
                "is rejected unconditionally."
            )
        column.append(converted)
    return column


def _require_accepted_solver_vector(
    sigma: Sequence[Sequence[float]],
    support: Sequence[str],
    raw: Sequence[float],
    budget: float,
    budget_tolerance: float,
) -> None:
    """Accept the raw solver vector exactly as returned (no clip, no
    renormalization) through the fixed post-solve rules: finite and
    exactly nonnegative signs, on-budget within the declared
    tolerance, KKT-clean at the fixed floors, and not strictly
    dominated by any deterministic projected budget-transfer
    neighbor."""
    size = len(raw)
    for asset, weight in zip(support, raw, strict=True):
        if not math.isfinite(weight):
            raise ValueError(
                "nonfinite solver weight: the optimizer returned the "
                f"nonfinite weight {weight!r} for asset {asset!r} - no "
                "clipping and no renormalization exist, so the "
                "decision is rejected unconditionally."
            )
        if weight < 0.0:
            raise ValueError(
                "negative solver weight: the optimizer returned the "
                f"negative weight {weight!r} for asset {asset!r} - the "
                "program is long-only, and no clipping and no "
                "renormalization exist, so the decision is rejected "
                "unconditional."
            )
    total = math.fsum(raw)
    if abs(total - budget) > budget_tolerance:
        raise ValueError(
            "solver vector breaks the declared budget: "
            f"fsum(weights)={total!r} differs from budget="
            f"{budget!r} by more than budget_tolerance="
            f"{budget_tolerance!r}, and no renormalization exists, so "
            "the decision is rejected unconditionally."
        )
    gradient = [
        2.0 * math.fsum(sigma[i][j] * raw[j] for j in range(size)) for i in range(size)
    ]
    multiplier = min(gradient)
    #: The fixed scale-aware tolerance: the KKT
    #: residual is tested against kappa scaled by max(1, |mu|), never
    #: against bare kappa (a multiplier of magnitude mu scales the
    #: residuals a legitimate solve can carry by the same factor).
    tolerance = _KAPPA * max(1.0, abs(multiplier))
    for i in range(size):
        residual = gradient[i] - multiplier
        if raw[i] > _TAU_W and residual > tolerance:
            raise ValueError(
                "KKT residual beyond the fixed tolerance: asset "
                f"{support[i]!r} carries weight {raw[i]!r} above "
                f"tau_w={_TAU_W!r} but its marginal gradient "
                f"g_i={gradient[i]!r} exceeds the active-set "
                f"multiplier mu=min(g)={multiplier!r} by "
                f"{residual!r} > kappa*max(1,|mu|)={tolerance!r} "
                f"(kappa={_KAPPA!r}), so the returned vector is not "
                "the minimum-variance optimum and the decision is "
                "rejected unconditionally."
            )
        if raw[i] <= _TAU_W and residual < -tolerance:
            #: The fixed rule 3 boundary inequality g_i >= mu -
            #: kappa*max(1, |mu|): with mu = min(g) the residual is
            #: nonnegative by construction, so this clause can only
            #: fire under a different multiplier law - it is
            #: implemented for exact fidelity to the fixed rule.
            raise ValueError(
                "KKT boundary violation: asset "
                f"{support[i]!r} carries boundary weight {raw[i]!r} "
                f"<= tau_w={_TAU_W!r} but its marginal gradient "
                f"g_i={gradient[i]!r} falls below the active-set "
                f"multiplier mu=min(g)={multiplier!r} by "
                f"{-residual!r} > kappa*max(1,|mu|)={tolerance!r} "
                f"(kappa={_KAPPA!r}), so the returned vector is not "
                "the minimum-variance optimum and the decision is "
                "rejected unconditionally."
            )
    objective = _quadratic_form(sigma, raw)
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            for epsilon in _DOMINATION_EPSILONS:
                shifted = [
                    weight
                    + (epsilon if index == i else -epsilon if index == j else 0.0)
                    for index, weight in enumerate(raw)
                ]
                neighbor = _project_to_simplex(shifted)
                neighbor_objective = _quadratic_form(sigma, neighbor)
                #: Deterministic strictness: an improvement is real only
                #: when it exceeds relative rounding noise of the
                #: quadratic-form evaluation (a projected neighbor that
                #: coincides with the returned corner point differs by
                #: at most a few ulps, never by the orders of magnitude
                #: a genuinely dominated vector improves by).
                tie = _DOMINATION_TIE * abs(objective)
                if neighbor_objective < objective - tie:
                    raise ValueError(
                        "dominated solver vector: the deterministic "
                        "budget-transfer neighbor along "
                        f"e_{support[i]!r}-e_{support[j]!r} at "
                        f"epsilon={epsilon!r}, Euclidean-projected onto "
                        "the probability simplex, strictly improves the "
                        f"objective ({neighbor_objective!r} < "
                        f"{objective!r}), so the returned vector is not "
                        "optimal and the decision is rejected "
                        "unconditional."
                    )


class MinimumVariance:
    """Long-only, fully-invested minimum-variance allocation over the
    full caller-declared universe.

    ``MinimumVariance(return_series, window, constraints)`` maps each
    asset identifier to exactly one return-series identifier (one
    series cannot serve two assets), fixes the estimation window (an
    integer ``>= 2``), and snapshots the standing TARGET-level
    constraints declaration (a ``WeightConstraints``, default
    ``WeightConstraints()``). Invalid inputs raise ``ValueError``;
    the declaration is snapshotted immutably at construction.

    At :meth:`decide` time the supported constraint subset is exactly
    the canonical one - long-only (``allow_short=False``), fully
    invested (``budget=1.0``), no exposure caps - and any other
    declaration rejects before anything is estimated. The universe
    must declare at least two assets and the window must satisfy
    ``window >= N + 1``. For each universe asset the latest ``window``
    admissible records of its mapped series are selected on one
    common balanced observation grid, every selected value must be a
    real finitely-representable return, the sample covariance
    (``ddof=1``, ``math.fsum``, symmetric assignment so the matrix
    equals its transpose exactly) is computed, and the exact
    ``Fraction`` rank of the demeaned panel must equal ``N`` before
    any solve. The program is solved with the internal solver
    ``portlearn._allocation.solve_long_only_min_variance``, and the
    raw solver vector is accepted in universe order without clipping
    and without renormalization before reaching the TARGET exactly.
    Invalid configurations, data conditions, or solve outcomes raise
    ``ValueError``.
    """

    def __init__(
        self,
        return_series: Mapping[str, str],
        window: int,
        constraints: WeightConstraints = _CANONICAL_CONSTRAINTS,
    ) -> None:
        """Validate and snapshot the mapping, window, and constraints
        (invalid inputs raise ``ValueError``)."""
        if isinstance(return_series, str) or not isinstance(return_series, Mapping):
            raise ValueError(  # noqa: TRY004 — the mapping law is ValueError-only by the fixed contract
                "return_series must be a mapping of asset identifier "
                "to return-series identifier (a bare string names one "
                "series, not an asset-to-series mapping); got "
                f"{type(return_series).__name__}: {return_series!r}."
            )
        if not return_series:
            raise ValueError(
                "return_series must be a non-empty mapping of asset "
                "identifier to return-series identifier: a "
                "minimum-variance target over no mapped return series "
                "names no portfolio, so the configuration is rejected "
                "unconditional."
            )
        series_by_asset: dict[str, str] = {}
        asset_by_series: dict[str, str] = {}
        for key, value in return_series.items():
            if not isinstance(key, str):
                raise ValueError(  # noqa: TRY004 — the identifier law is ValueError-only by the fixed contract
                    "return_series keys must be non-blank exact-string "
                    "asset identifiers (no case folding, trimming, or "
                    "normalization); got the key "
                    f"{key!r} of type {type(key).__name__}."
                )
            if not key.strip():
                raise ValueError(
                    "return_series keys must be non-blank exact-string "
                    "asset identifiers; got the blank identifier "
                    f"{key!r}."
                )
            if not isinstance(value, str):
                raise ValueError(  # noqa: TRY004 — the series-identifier law is ValueError-only by the fixed contract
                    "return_series values must be non-blank exact-string "
                    "series identifiers; the series identifier for "
                    f"asset {key!r} is {value!r} of type "
                    f"{type(value).__name__}."
                )
            if not value.strip():
                raise ValueError(
                    "return_series values must be non-blank exact-string "
                    "series identifiers; the series identifier for "
                    f"asset {key!r} is the blank identifier {value!r}."
                )
            if value in asset_by_series:
                raise ValueError(
                    "duplicate return-series mapping: the return series "
                    f"{value!r} is mapped to both asset "
                    f"{asset_by_series[value]!r} and asset {key!r}; one "
                    "return series cannot serve two assets - the "
                    "mapping is the explicit asset-to-series identity, "
                    "and sharing would double-count one series' "
                    "information as two assets' returns, so the "
                    "configuration is rejected unconditionally."
                )
            asset_by_series[value] = key
            series_by_asset[key] = value
        if isinstance(window, bool) or not isinstance(window, int):
            raise ValueError(  # noqa: TRY004 — the window law is ValueError-only by the fixed contract
                "window must be an integer >= 2 (the estimation window "
                "length); got type "
                f"{type(window).__name__} (window={window!r})."
            )
        if window < 2:
            raise ValueError(
                f"window must be an integer >= 2; got window={window} - "
                "a covariance estimate over fewer than two returns is "
                "undefined, so the configuration is rejected "
                "unconditional."
            )
        if not isinstance(constraints, WeightConstraints):
            raise ValueError(  # noqa: TRY004 — the constraints law is ValueError-only by the fixed contract
                "constraints must be a WeightConstraints instance (the "
                "standing TARGET-level declaration); got "
                f"{type(constraints).__name__}: {constraints!r}."
            )
        self._return_series = series_by_asset
        self._window = window
        self._constraints = constraints

    @property
    def return_series(self) -> Mapping[str, str]:
        """A read-only view of the immutable asset-to-series
        snapshot."""
        return MappingProxyType(self._return_series)

    @property
    def window(self) -> int:
        """The read-only immutable estimation window (integer
        ``>= 2``)."""
        return self._window

    @property
    def constraints(self) -> WeightConstraints:
        """The read-only immutable constraints snapshot."""
        return self._constraints

    def decide(
        self, context: _interfaces.DecisionContext
    ) -> _interfaces.DecisionResult:
        """Decide the long-only fully-invested minimum-variance TARGET
        for one decision context, following the module's fixed
        decide-time sequence."""
        constraints = self._constraints
        _require_canonical_constraints(constraints)
        support = tuple(context.universe)
        asset_count = len(support)
        if asset_count < 2:
            raise ValueError(
                "degenerate minimum-variance universe: the decision "
                "context's universe declares "
                f"{asset_count} asset(s) at decision_time="
                f"{context.decision_time.isoformat()}, and a "
                "minimum-variance target over fewer than 2 assets names "
                "no diversification, so the decision is rejected "
                "unconditional."
            )
        window = self._window
        if window < asset_count + 1:
            raise ValueError(
                "window below the estimation domain floor: "
                f"window={window} but the universe declares "
                f"N={asset_count} assets, and the demeaned panel needs "
                f"window >= N + 1 = {asset_count + 1} observations to "
                "reach full column rank, so the decision is rejected "
                "unconditional."
            )
        missing = [asset for asset in support if asset not in self._return_series]
        if missing:
            raise ValueError(
                "missing return-series mapping: the universe asset(s) "
                f"{', '.join(repr(asset) for asset in missing)} have no "
                "return_series entry at decision_time="
                f"{context.decision_time.isoformat()} - the universe is "
                "the effective support and is never selected, ranked, "
                "or filtered, so the decision is rejected unconditionally."
            )
        buckets: dict[str, list] = {self._return_series[asset]: [] for asset in support}
        for record in context.information:
            series_id = record.series_id
            if series_id in buckets:
                buckets[series_id].append(record)
        grids: dict[str, tuple] = {}
        selected: dict[str, tuple] = {}
        for asset in support:
            series = self._return_series[asset]
            ordered = sorted(
                buckets[series],
                key=lambda record: instant_key(record.observation_time),
            )
            available = len(ordered)
            if available < window:
                raise ValueError(
                    "insufficient history: asset "
                    f"{asset!r} series {series!r} has {available} "
                    f"admissible record(s) available but window={window} "
                    f"requires {window} - no partial estimate and no "
                    "truncation exist, so the decision is rejected "
                    "unconditional."
                )
            latest = tuple(ordered[-window:])
            grids[asset] = tuple(
                instant_key(record.observation_time) for record in latest
            )
            selected[asset] = latest
        reference = support[0]
        for asset in support[1:]:
            if grids[asset] != grids[reference]:
                divergence = next(
                    index
                    for index, (first, second) in enumerate(
                        zip(grids[reference], grids[asset])
                    )
                    if first != second
                )
                raise ValueError(
                    "unbalanced observation grid: assets "
                    f"{reference!r} and {asset!r} selected different "
                    "observation instants over the window - the first "
                    f"divergence is at observation index {divergence}: "
                    f"{reference!r} holds "
                    f"{selected[reference][divergence].observation_time.isoformat()} "
                    f"while {asset!r} holds "
                    f"{selected[asset][divergence].observation_time.isoformat()}; "
                    "the estimand is defined on one balanced common "
                    "grid of the same observation periods, and no "
                    "mismatched-date cross-section is estimated, so "
                    "the decision is rejected unconditionally."
                )
        columns = [
            _require_real_column(asset, self._return_series[asset], selected[asset])
            for asset in support
        ]
        deviations = [
            [value - math.fsum(column) / window for value in column]
            for column in columns
        ]
        sigma: list[list[float]] = [[0.0] * asset_count for _ in range(asset_count)]
        for i in range(asset_count):
            for j in range(i, asset_count):
                entry = math.fsum(
                    deviations[i][k] * deviations[j][k] for k in range(window)
                ) / (window - 1)
                sigma[i][j] = entry
                sigma[j][i] = entry
        rank = _exact_rank(_demeaned_fraction_panel(columns, window))
        if rank != asset_count:
            raise ValueError(
                "rank-deficient demeaned panel: the exact-rank reference "
                f"computed rank {rank} over N={asset_count} assets "
                f"(window={window}), so the sample covariance is "
                "singular and the minimum-variance program has no "
                "unique solution, so the decision is rejected "
                "unconditional."
            )
        solve_result = _allocation.solve_long_only_min_variance(sigma, list(support))
        raw = [float(solve_result.weights[asset]) for asset in support]
        _require_accepted_solver_vector(
            sigma,
            support,
            raw,
            constraints.budget,
            constraints.budget_tolerance,
        )
        require_valid_target(
            PortfolioWeights(dict(zip(support, raw, strict=True)), WeightState.TARGET),
            constraints,
        )
        result = _interfaces.DecisionResult(
            decision=_interfaces.PortfolioDecision(
                decision_time=context.decision_time,
                execution_time=context.decision_time,
                target_weights=_TargetWeightBook(zip(support, raw, strict=True)),
            ),
            next_strategy_state=None,
        )
        _interfaces.require_decision_result_compatible(context, result)
        return result


def _require_accepted_mean_variance_vector(
    sigma: Sequence[Sequence[float]],
    mu_tilde: Sequence[float],
    risk_aversion: float,
    support: Sequence[str],
    raw: Sequence[float],
    budget: float,
    budget_tolerance: float,
) -> None:
    """Accept the raw mean-variance solver vector exactly as returned
    (no clip, no renormalization) through the fixed post-solve rules
    evaluated on the CENTERED program ``min (lam/2) w' Sigma w -
    mu_tilde' w``: finite and exactly nonnegative signs, on-budget
    within the declared tolerance, KKT-clean at the fixed floors, and
    not strictly dominated by any deterministic projected
    budget-transfer neighbor.

    The ``(sigma, mu_tilde)`` program, the identifiers, and the
    returned vector ``raw`` are ALL assembled in UNIVERSE order
    (``context.universe`` - the fixed order law, the
    one shared order of the forecast channel and the covariance
    channel), so acceptance is positional on that universe order.
    """
    size = len(raw)
    for asset, weight in zip(support, raw, strict=True):
        if not math.isfinite(weight):
            raise ValueError(
                "nonfinite solver weight: the optimizer returned the "
                f"nonfinite weight {weight!r} for asset {asset!r} - no "
                "clipping and no renormalization exist, so the "
                "decision is rejected unconditionally."
            )
        if weight < 0.0:
            raise ValueError(
                "negative solver weight: the optimizer returned the "
                f"negative weight {weight!r} for asset {asset!r} - the "
                "program is long-only, and no clipping and no "
                "renormalization exist, so the decision is rejected "
                "unconditional."
            )
    total = math.fsum(raw)
    if abs(total - budget) > budget_tolerance:
        raise ValueError(
            "solver vector breaks the declared budget: "
            f"fsum(weights)={total!r} differs from budget="
            f"{budget!r} by more than budget_tolerance="
            f"{budget_tolerance!r}, and no renormalization exists, so "
            "the decision is rejected unconditionally."
        )
    gradient = [
        risk_aversion * math.fsum(sigma[i][j] * raw[j] for j in range(size))
        - mu_tilde[i]
        for i in range(size)
    ]
    #: The fixed multiplier: theta = min_i g_i
    #: over ALL components (the fixed multiplier rule verbatim) - at
    #: the optimum every asset holding weight has equal marginal
    #: objective g_i = theta and every asset at the boundary has
    #: g_i >= theta, so the minimum is attained on the active set;
    #: minimizing over the active components alone would silently
    #: re-define the multiplier whenever a boundary asset carries
    #: the smallest gradient.
    theta = min(gradient)
    #: The fixed scale-aware tolerance, mirrored
    #: from the minimum-variance law: the KKT residual is tested
    #: against kappa scaled by max(1, |theta|), never against bare
    #: kappa (a multiplier of magnitude theta scales the residuals a
    #: legitimate solve can carry by the same factor).
    tolerance = _KAPPA * max(1.0, abs(theta))
    for index in range(size):
        residual = gradient[index] - theta
        if raw[index] > _TAU_W and residual > tolerance:
            raise ValueError(
                "KKT residual beyond the fixed tolerance: asset "
                f"{support[index]!r} carries weight {raw[index]!r} "
                f"above tau_w={_TAU_W!r} but its centered marginal "
                f"gradient g_i={gradient[index]!r} exceeds the "
                f"active-set multiplier theta=min(g)={theta!r} by "
                f"{residual!r} > kappa*max(1,|theta|)={tolerance!r} "
                f"(kappa={_KAPPA!r}), so the returned vector is not "
                "the centered mean-variance optimum and the decision "
                "is rejected unconditionally."
            )
        if raw[index] <= _TAU_W and residual < -tolerance:
            #: The fixed rule 3 boundary clause g_i >= theta -
            #: kappa*max(1, |theta|): with theta = min_i g_i the
            #: residual is nonnegative by construction, so this
            #: clause can only fire under a different multiplier law
            #: - it is implemented for exact fidelity to the fixed
            #: rule (mirrored from the minimum-variance law).
            raise ValueError(
                "KKT boundary violation: asset "
                f"{support[index]!r} carries boundary weight "
                f"{raw[index]!r} <= tau_w={_TAU_W!r} but its centered "
                f"marginal gradient g_i={gradient[index]!r} falls "
                f"below the multiplier theta=min(g)={theta!r} by "
                f"{-residual!r} > kappa*max(1,|theta|)={tolerance!r} "
                f"(kappa={_KAPPA!r}), so the returned vector is not "
                "the centered mean-variance optimum and the decision "
                "is rejected unconditionally."
            )
    objective = 0.5 * risk_aversion * _quadratic_form(sigma, raw) - math.fsum(
        mu_tilde[index] * raw[index] for index in range(size)
    )
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            #: The fixed rule 4 closed list: D = {e_i - e_j : i != j}
            #: over ALL ordered pairs in universe order (N(N-1)
            #: directions, the fixed direction-set rule verbatim) - a
            #: boundary donor is probed through its Euclidean-projected
            #: neighbor like every other direction, never skipped.
            for epsilon in _DOMINATION_EPSILONS:
                shifted = [
                    weight
                    + (epsilon if index == i else -epsilon if index == j else 0.0)
                    for index, weight in enumerate(raw)
                ]
                neighbor = _project_to_simplex(shifted)
                neighbor_objective = 0.5 * risk_aversion * _quadratic_form(
                    sigma, neighbor
                ) - math.fsum(
                    mu_tilde[index] * neighbor[index] for index in range(size)
                )
                #: Deterministic strictness, mirrored from the fixed
                #: minimum-variance law: an improvement is real only
                #: when it exceeds relative rounding noise of the
                #: centered objective's evaluation (a projected
                #: neighbor that coincides with the returned vector
                #: differs by at most a few ulps, never by the orders
                #: of magnitude a genuinely dominated vector improves
                #: by).
                tie = _DOMINATION_TIE * abs(objective)
                if neighbor_objective < objective - tie:
                    raise ValueError(
                        "dominated solver vector: the deterministic "
                        "budget-transfer neighbor along "
                        f"e_{support[i]!r}-e_{support[j]!r} at "
                        f"epsilon={epsilon!r}, Euclidean-projected onto "
                        "the probability simplex, strictly improves the "
                        f"centered objective ({neighbor_objective!r} < "
                        f"{objective!r}), so the returned vector is not "
                        "the centered mean-variance optimum and the "
                        "decision is rejected unconditionally."
                    )


class MeanVariance:
    """Long-only, fully-invested mean-variance allocation over the
    full caller-declared universe.

    ``MeanVariance(return_series, window, risk_aversion,
    constraints)`` maps each asset identifier to exactly one
    return-series identifier (one series cannot serve two assets),
    fixes the estimation window (an integer ``>= 2``), the risk
    aversion (a real finite ``> 0`` multiplier on the quadratic risk
    term), and snapshots the standing TARGET-level constraints
    declaration (a ``WeightConstraints``, default
    ``WeightConstraints()``). Invalid inputs raise ``ValueError``;
    the declaration is snapshotted immutably at construction.

    At :meth:`decide` time the supported constraint subset is exactly
    the canonical one - long-only (``allow_short=False``), fully
    invested (``budget=1.0``), no exposure caps - and any other
    declaration rejects before anything is estimated. The universe
    must declare at least two assets and the window must satisfy
    ``window >= N + 1``. For each universe asset the latest ``window``
    admissible records of its mapped series are selected on one
    common balanced observation grid, every selected value must be a
    real finitely-representable return, and the sample covariance
    (``ddof=1``, ``math.fsum``, symmetric assignment so the matrix
    equals its transpose exactly) is computed. The context must carry
    a forecast whose target is exactly ``"expected_return"`` covering
    every universe asset with real finite values; the forecast vector
    is CENTERED (``mu_tilde = mu - mu_bar`` with ``mu_bar =
    math.fsum(mu) / N``) before any solve, so only relative expected
    returns matter, and the centered program is the solved one:
    ``min (lam/2) w' Sigma w - mu_tilde' w`` over the long-only
    unit-budget simplex. The exact ``Fraction`` rank of the demeaned
    panel must equal ``N`` before any solve. The program is solved
    with the internal solver
    ``portlearn._allocation.solve_long_only_mean_variance``, and the
    raw solver vector is accepted in universe order without clipping
    and without renormalization before reaching the TARGET exactly.
    Invalid configurations, data conditions, or solve outcomes raise
    ``ValueError``.
    """

    def __init__(
        self,
        return_series: Mapping[str, str],
        window: int,
        risk_aversion: float,
        constraints: WeightConstraints = _CANONICAL_CONSTRAINTS,
    ) -> None:
        """Validate and snapshot the mapping, window, risk aversion,
        and constraints (invalid inputs raise ``ValueError``)."""
        if isinstance(return_series, str) or not isinstance(return_series, Mapping):
            raise ValueError(  # noqa: TRY004 — the mapping law is ValueError-only by the fixed contract
                "return_series must be a mapping of asset identifier "
                "to return-series identifier (a bare string names one "
                "series, not an asset-to-series mapping); got "
                f"{type(return_series).__name__}: {return_series!r}."
            )
        if not return_series:
            raise ValueError(
                "return_series must be a non-empty mapping of asset "
                "identifier to return-series identifier: a "
                "mean-variance target over no mapped return series "
                "names no portfolio, so the configuration is rejected "
                "unconditional."
            )
        series_by_asset: dict[str, str] = {}
        asset_by_series: dict[str, str] = {}
        for key, value in return_series.items():
            if not isinstance(key, str):
                raise ValueError(  # noqa: TRY004 — the identifier law is ValueError-only by the fixed contract
                    "return_series keys must be non-blank exact-string "
                    "asset identifiers (no case folding, trimming, or "
                    "normalization); got the key "
                    f"{key!r} of type {type(key).__name__}."
                )
            if not key.strip():
                raise ValueError(
                    "return_series keys must be non-blank exact-string "
                    "asset identifiers; got the blank identifier "
                    f"{key!r}."
                )
            if not isinstance(value, str):
                raise ValueError(  # noqa: TRY004 — the series-identifier law is ValueError-only by the fixed contract
                    "return_series values must be non-blank exact-string "
                    "series identifiers; the series identifier for "
                    f"asset {key!r} is {value!r} of type "
                    f"{type(value).__name__}."
                )
            if not value.strip():
                raise ValueError(
                    "return_series values must be non-blank exact-string "
                    "series identifiers; the series identifier for "
                    f"asset {key!r} is the blank identifier {value!r}."
                )
            if value in asset_by_series:
                raise ValueError(
                    "duplicate return-series mapping: the return series "
                    f"{value!r} is mapped to both asset "
                    f"{asset_by_series[value]!r} and asset {key!r}; one "
                    "return series cannot serve two assets - the "
                    "mapping is the explicit asset-to-series identity, "
                    "and sharing would double-count one series' "
                    "information as two assets' returns, so the "
                    "configuration is rejected unconditionally."
                )
            asset_by_series[value] = key
            series_by_asset[key] = value
        if isinstance(window, bool) or not isinstance(window, int):
            raise ValueError(  # noqa: TRY004 — the window law is ValueError-only by the fixed contract
                "window must be an integer >= 2 (the estimation window "
                "length); got type "
                f"{type(window).__name__} (window={window!r})."
            )
        if window < 2:
            raise ValueError(
                f"window must be an integer >= 2; got window={window} - "
                "a covariance estimate over fewer than two returns is "
                "undefined, so the configuration is rejected "
                "unconditional."
            )
        if isinstance(risk_aversion, bool) or not isinstance(
            risk_aversion, numbers.Real
        ):
            raise ValueError(  # noqa: TRY004 — the risk-aversion law is ValueError-only by the fixed contract
                "risk_aversion must be a real number > 0 (the "
                "risk-aversion multiplier on the quadratic risk term); "
                f"got type {type(risk_aversion).__name__} "
                f"(risk_aversion={risk_aversion!r})."
            )
        risk_aversion_float = float(risk_aversion)
        if not math.isfinite(risk_aversion_float):
            raise ValueError(
                "risk_aversion must be a finite real number > 0; got "
                f"the nonfinite risk_aversion={risk_aversion!r} - a "
                "nonfinite risk aversion names no estimand, so the "
                "configuration is rejected unconditionally."
            )
        if risk_aversion_float <= 0.0:
            raise ValueError(
                "risk_aversion must be a finite real number > 0; got "
                f"risk_aversion={risk_aversion_float!r} - a "
                "nonpositive risk aversion names no risk-reward "
                "tradeoff, so the configuration is rejected "
                "unconditional."
            )
        if not isinstance(constraints, WeightConstraints):
            raise ValueError(  # noqa: TRY004 — the constraints law is ValueError-only by the fixed contract
                "constraints must be a WeightConstraints instance (the "
                "standing TARGET-level declaration); got "
                f"{type(constraints).__name__}: {constraints!r}."
            )
        self._return_series = series_by_asset
        self._window = window
        self._risk_aversion = risk_aversion_float
        self._constraints = constraints

    @property
    def return_series(self) -> Mapping[str, str]:
        """A read-only view of the immutable asset-to-series
        snapshot."""
        return MappingProxyType(self._return_series)

    @property
    def window(self) -> int:
        """The read-only immutable estimation window (integer
        ``>= 2``)."""
        return self._window

    @property
    def risk_aversion(self) -> float:
        """The read-only immutable risk aversion (finite real
        ``> 0``)."""
        return self._risk_aversion

    @property
    def constraints(self) -> WeightConstraints:
        """The read-only immutable constraints snapshot."""
        return self._constraints

    def decide(
        self, context: _interfaces.DecisionContext
    ) -> _interfaces.DecisionResult:
        """Decide the long-only fully-invested mean-variance TARGET
        for one decision context, following the module's fixed
        decide-time sequence."""
        constraints = self._constraints
        _require_canonical_constraints(constraints)
        support = tuple(context.universe)
        asset_count = len(support)
        if asset_count < 2:
            raise ValueError(
                "degenerate mean-variance universe: the decision "
                "context's universe declares "
                f"{asset_count} asset(s) at decision_time="
                f"{context.decision_time.isoformat()}, and a "
                "mean-variance target over fewer than 2 assets names "
                "no diversification, so the decision is rejected "
                "unconditional."
            )
        window = self._window
        if window < asset_count + 1:
            raise ValueError(
                "window below the estimation domain floor: "
                f"window={window} but the universe declares "
                f"N={asset_count} assets, and the demeaned panel needs "
                f"window >= N + 1 = {asset_count + 1} observations to "
                "reach full column rank, so the decision is rejected "
                "unconditional."
            )
        missing = [asset for asset in support if asset not in self._return_series]
        if missing:
            raise ValueError(
                "missing return-series mapping: the universe asset(s) "
                f"{', '.join(repr(asset) for asset in missing)} have no "
                "return_series entry at decision_time="
                f"{context.decision_time.isoformat()} - the universe is "
                "the effective support and is never selected, ranked, "
                "or filtered, so the decision is rejected unconditionally."
            )
        forecast = context.forecast
        if forecast is None:
            raise ValueError(
                "missing forecast: the decision context carries no "
                "forecast at decision_time="
                f"{context.decision_time.isoformat()} - the expected-"
                "return estimand is a first-class input of the "
                "mean-variance program (never fabricated, no "
                "shrinkage fallback, no zero-mu fallback), so the "
                "decision is rejected unconditionally."
            )
        if forecast.target != "expected_return":
            raise ValueError(
                "forecast target mismatch: the mean-variance "
                "estimand is exactly the expected return, but the "
                f"forecast declares target={forecast.target!r} - a "
                "forecast of another estimand (a risk measure, a "
                "volatility, or any other declared quantity) cannot "
                "stand in for the expected-return vector, so the "
                "decision is rejected unconditionally."
            )
        missing_forecast = [asset for asset in support if asset not in forecast.values]
        if missing_forecast:
            raise ValueError(
                "incomplete forecast: the universe asset(s) "
                f"{', '.join(repr(asset) for asset in missing_forecast)} "
                "carry no forecast entry - consumption is keyed on "
                "exactly the caller-declared universe keys (any other "
                "key is inert), so the expected-return vector is "
                "incomplete and the decision is rejected unconditionally."
            )
        mu_universe: list[float] = []
        for asset in support:
            value = forecast.values[asset]
            if isinstance(value, bool):
                raise ValueError(  # noqa: TRY004
                    "non-real forecast value: asset "
                    f"{asset!r} carries the bool value {value!r} - "
                    "bool is not an expected return, so the decision "
                    "is rejected unconditionally."
                )
            if not isinstance(value, numbers.Real):
                raise ValueError(  # noqa: TRY004
                    "non-real forecast value: asset "
                    f"{asset!r} carries a {type(value).__name__} "
                    f"value {value!r} - an expected return must be a "
                    "real scalar, so the decision is rejected "
                    "unconditional."
                )
            entry = float(value)
            if not math.isfinite(entry):
                raise ValueError(
                    "nonfinite forecast value: asset "
                    f"{asset!r} carries the value {value!r} that "
                    "converts to a nonfinite float - the expected-"
                    "return vector must be finite, so the decision "
                    "is rejected unconditionally."
                )
            mu_universe.append(entry)
        try:
            mu_bar = math.fsum(mu_universe) / asset_count
        except OverflowError:
            raise ValueError(
                "non-representable centered forecast: the compensated "
                "sum of the expected-return vector overflows under "
                "math.fsum, so the centered vector mu_tilde = mu - "
                "mu_bar is undefined - no value enters the program, "
                "so the decision is rejected unconditionally."
            ) from None
        mu_tilde: list[float] = []
        for asset, value in zip(support, mu_universe, strict=True):
            centered = value - mu_bar
            if not math.isfinite(centered):
                raise ValueError(
                    "non-representable centered forecast: the centered "
                    f"entry mu_tilde[{asset!r}] = {value!r} - "
                    f"{mu_bar!r} is nonfinite - the centered program "
                    "is undefined, so the decision is rejected "
                    "unconditional."
                )
            mu_tilde.append(centered)
        buckets: dict[str, list] = {self._return_series[asset]: [] for asset in support}
        for record in context.information:
            series_id = record.series_id
            if series_id in buckets:
                buckets[series_id].append(record)
        grids: dict[str, tuple] = {}
        selected: dict[str, tuple] = {}
        for asset in support:
            series = self._return_series[asset]
            ordered = sorted(
                buckets[series],
                key=lambda record: instant_key(record.observation_time),
            )
            available = len(ordered)
            if available < window:
                raise ValueError(
                    "insufficient history: asset "
                    f"{asset!r} series {series!r} has {available} "
                    f"admissible record(s) available but window={window} "
                    f"requires {window} - no partial estimate and no "
                    "truncation exist, so the decision is rejected "
                    "unconditional."
                )
            latest = tuple(ordered[-window:])
            grids[asset] = tuple(
                instant_key(record.observation_time) for record in latest
            )
            selected[asset] = latest
        reference = support[0]
        for asset in support[1:]:
            if grids[asset] != grids[reference]:
                divergence = next(
                    index
                    for index, (first, second) in enumerate(
                        zip(grids[reference], grids[asset])
                    )
                    if first != second
                )
                raise ValueError(
                    "unbalanced observation grid: assets "
                    f"{reference!r} and {asset!r} selected different "
                    "observation instants over the window - the first "
                    f"divergence is at observation index {divergence}: "
                    f"{reference!r} holds "
                    f"{selected[reference][divergence].observation_time.isoformat()} "
                    f"while {asset!r} holds "
                    f"{selected[asset][divergence].observation_time.isoformat()}; "
                    "the estimand is defined on one balanced common "
                    "grid of the same observation periods, and no "
                    "mismatched-date cross-section is estimated, so "
                    "the decision is rejected unconditionally."
                )
        columns = [
            _require_real_column(asset, self._return_series[asset], selected[asset])
            for asset in support
        ]
        deviations = [
            [value - math.fsum(column) / window for value in column]
            for column in columns
        ]
        sigma: list[list[float]] = [[0.0] * asset_count for _ in range(asset_count)]
        for i in range(asset_count):
            for j in range(i, asset_count):
                entry = math.fsum(
                    deviations[i][k] * deviations[j][k] for k in range(window)
                ) / (window - 1)
                sigma[i][j] = entry
                sigma[j][i] = entry
        rank = _exact_rank(_demeaned_fraction_panel(columns, window))
        if rank != asset_count:
            raise ValueError(
                "rank-deficient demeaned panel: the exact-rank reference "
                f"computed rank {rank} over N={asset_count} assets "
                f"(window={window}), so the sample covariance is "
                "singular and the mean-variance program has no "
                "unique solution, so the decision is rejected "
                "unconditional."
            )
        solve_result = _allocation.solve_long_only_mean_variance(
            sigma, mu_tilde, self._risk_aversion, list(support)
        )
        raw = [float(solve_result.weights[asset]) for asset in support]
        _require_accepted_mean_variance_vector(
            sigma,
            mu_tilde,
            self._risk_aversion,
            support,
            raw,
            constraints.budget,
            constraints.budget_tolerance,
        )
        require_valid_target(
            PortfolioWeights(dict(zip(support, raw, strict=True)), WeightState.TARGET),
            constraints,
        )
        result = _interfaces.DecisionResult(
            decision=_interfaces.PortfolioDecision(
                decision_time=context.decision_time,
                execution_time=context.decision_time,
                target_weights=_TargetWeightBook(zip(support, raw, strict=True)),
            ),
            next_strategy_state=None,
        )
        _interfaces.require_decision_result_compatible(context, result)
        return result
