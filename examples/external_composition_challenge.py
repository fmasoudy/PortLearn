"""External composition challenge: scikit-learn estimation x cvxpy allocation.

This module is the external composition challenge entry point. It composes two
genuinely external component ecosystems with the PortLearn public
contracts only:

* ``scikit-learn`` owns covariance **estimation** (shrinkage estimators):
  its estimators are wrapped behind the public forecaster interfaces so that
  fitted models emit canonical ``Forecast`` values on the
  ``COVARIANCE_TARGET`` book, with fit provenance stamped through the
  public ``fitting_provenance`` helper.
* ``cvxpy`` owns weight **optimization**: both research strategies
  translate the covariance forecast supplied through the public
  ``context.forecast`` interface into solved weights through cvxpy alone.

The researcher code in this file only translates and orchestrates. All
data are synthetic and generated locally; the composition is a
non-empirical demonstration of public contract interfaces, not evidence about
any real market, and it makes no comparative claim.

Run it with:

    python examples/external_composition_challenge.py

Running this module requires the optional extra
``portlearn[external-composition]`` (scikit-learn and cvxpy); a missing
external distribution fails closed with an explicit install hint.
Importing this module is side-effect free: every effect lives under
``main``.
"""

import argparse
import hashlib
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

# --------------------------------------------------------------------------- #
# Fail-closed external composition imports.
# --------------------------------------------------------------------------- #

_MISSING_EXTRA = "portlearn[external-composition]"


def _fail_closed(distribution: str, import_name: str | None = None) -> ImportError:
    detail = f" (import name {import_name!r})" if import_name else ""
    return ImportError(
        f"the external composition challenge requires the external "
        f"distribution {distribution!r}{detail}, which is not importable in "
        f"this environment; install the optional extra '{_MISSING_EXTRA}' "
        f"(no fallback exists by design: the challenge fails closed)"
    )


try:  # pragma: no cover - exercised only by dependency-blocking tests
    import sklearn as _sklearn_module
    from sklearn.covariance import OAS, LedoitWolf
except ImportError:  # pragma: no cover
    raise _fail_closed("scikit-learn", import_name="sklearn") from None

try:  # pragma: no cover - exercised only by dependency-blocking tests
    import cvxpy as _cvxpy_module
except ImportError:  # pragma: no cover
    raise _fail_closed("cvxpy") from None


# Public PortLearn contracts only; no built-in strategy modules, no privates.
from portlearn.calendar import month_end_instant
from portlearn.costs import Proportional
from portlearn.experiments import (
    DecisionSchedule,
    ExperimentConditions,
    ExperimentConfiguration,
    StrategyRegistration,
)
from portlearn.forecasting import (
    FittingPlan,
    FitWindow,
    ForecastProvenance,
    fitting_provenance,
)
from portlearn.interfaces import (
    DecisionContext,
    DecisionResult,
    Forecast,
    InformationSet,
    PortfolioDecision,
    TimedObservation,
)
from portlearn.ledger import ExactFillAccounting
from portlearn.protocols import (
    EvaluationLearning,
    StrategyTemporalDeclaration,
    TemporalDesign,
    WindowMode,
    monthly,
    none,
    resolve_temporal_protocols,
    window,
)
from portlearn.research import run_declared_experiment
from portlearn.turnover import one_way

# --------------------------------------------------------------------------- #
# Frozen challenge constants.
# --------------------------------------------------------------------------- #

#: The single forecast target used by every covariance forecast in the shared
#: information set: a full symmetric covariance book over the universe.
COVARIANCE_TARGET = "covariance"

#: The scikit-learn shrinkage estimator wrapped by the factory (the
#: challenge admits exactly ``LedoitWolf`` or ``OAS``).
SKLEARN_COVARIANCE_ESTIMATOR = "LedoitWolf"

#: Number of synthetic instruments in the challenge universe.
N_ASSETS = 6

#: Universe tickers (synthetic; no market data are read anywhere).
UNIVERSE: tuple[str, ...] = tuple(f"SYN{i:02d}" for i in range(N_ASSETS))

#: Deterministic synthetic-panel seed; the composition is non-empirical and
#: must be reproducible across runs.
PANEL_SEED = 20260927

#: Shared calendar zone for the whole challenge.
ZONE = ZoneInfo("UTC")

#: The declared temporal target horizon (monthly decisions).
TARGET_HORIZON = timedelta(days=31)

#: The canonical identity recorded in fit provenance for the wrapped
#: scikit-learn estimator.
SKLEARN_WRAPPED_IDENTITY = (
    f"scikit-learn:{SKLEARN_COVARIANCE_ESTIMATOR}:covariance-estimation"
)

#: The producer identity of the deterministic covariance book served to the
#: allocation-only strategy through the same forecast interface.
DETERMINISTIC_PRODUCER = "deterministic:constant-covariance"

#: Strategy names registered in the single declared experiment.
ALLOCATION_ONLY_ARM = "cvxpy_minimum_variance"
COMPOSED_ARM = "sklearn_cvxpy_minimum_variance"
STRATEGY_IDS = (ALLOCATION_ONLY_ARM, COMPOSED_ARM)

#: The declared cost rate and turnover convention (zero-cost challenge).
COST_RATE = 0.0

#: Synthetic panel: monthly observations per instrument.
PANEL_FIRST = (2023, 8)
PANEL_MONTHS = 24

#: Declared evaluation span and its monthly decision instants.
EVALUATION_SPAN = "2025-01-01..2025-06-30"
DECISION_MONTHS = (1, 2, 3, 4, 5, 6)
#: Decision months whose single forecast channel is produced by the wrapped
#: scikit-learn estimator; the remaining months carry the deterministic
#: book for the allocation-only strategy. Exactly one forecast observation is
#: admitted per decision instant (the runtime's one-channel law).
SKLEARN_MONTHS = (1, 3, 5)
DETERMINISTIC_MONTHS = (2, 4, 6)

#: Rolling calendar estimation window for both strategies.
ESTIMATION_WINDOW_MONTHS = 6

#: Deterministic diagonal/off-diagonal scale for the constant book.
DETERMINISTIC_VARIANCE = 0.0004
DETERMINISTIC_COVARIANCE = 0.0001

_ESTIMATOR_NAMES = ("LedoitWolf", "OAS")


def covariance_pair_key(left: str, right: str) -> str:
    """The canonical covariance-book key for one instrument pair.

    Canonical means order-invariant over the pair and stable across runs:
    the two symbols are sorted and joined with the frozen ``::`` separator,
    so ``covariance_pair_key(a, b) == covariance_pair_key(b, a)`` and the
    resulting book has exactly one entry per unordered pair (diagonal
    included).
    """
    first, second = sorted((left, right))
    return f"{first}::{second}"


def covariance_pair_keys(universe: Sequence[str] = UNIVERSE) -> tuple[str, ...]:
    """All canonical covariance-book keys over a universe, in canonical order."""
    return tuple(
        covariance_pair_key(universe[i], universe[j])
        for i in range(len(universe))
        for j in range(i, len(universe))
    )


def covariance_book(
    symbols: Sequence[str], covariance: Sequence[Sequence[float]]
) -> dict[str, float]:
    """Translate a covariance matrix (rows follow ``symbols``) into the book."""
    return {
        covariance_pair_key(symbols[i], symbols[j]): float(covariance[i][j])
        for i in range(len(symbols))
        for j in range(i, len(symbols))
    }


def _exact_unit_book(book: Mapping[str, float]) -> dict[str, float]:
    """Restore the exact float unit budget by minimal residual correction.

    A lawful normalized book can sit one representable bit below one, and
    the ledger's exact unit-budget law rejects that fail-closed. The
    correction is deterministic and minimal: the whole residual lands on
    the first sorted symbol while every other weight keeps its bits. No
    weight is rescaled, clipped, or otherwise materially adjusted.
    """
    corrected = dict(book)
    keys = sorted(corrected)
    residual = 1.0 - math.fsum(corrected[key] for key in keys)
    if residual:
        corrected[keys[0]] += residual
    if math.fsum(corrected.values()) != 1.0:
        raise ValueError(
            "the solved book cannot be corrected to the exact unit budget"
        )
    return corrected


# --------------------------------------------------------------------------- #
# Synthetic panel (local, deterministic, zero network access).
# --------------------------------------------------------------------------- #


def _month_keys(first: tuple[int, int], count: int) -> list[tuple[int, int]]:
    keys: list[tuple[int, int]] = []
    year, month = first
    for _ in range(count):
        keys.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return keys


def _month_of(instant: datetime) -> tuple[int, int]:
    return (instant.year, instant.month)


def synthetic_panel() -> tuple[tuple[datetime, dict[str, float]], ...]:
    """A deterministic synthetic monthly return panel for the universe.

    The population covariance is built from elementwise products and sums
    only ( exposures times their transpose plus a positive diagonal ), so
    it is symmetric positive definite by construction without any matrix
    decomposition. Draws are generated once from a fixed seed.
    """
    import numpy as np

    rng = np.random.default_rng(PANEL_SEED)
    exposures = rng.normal(size=(N_ASSETS, 2)) * 0.012
    diagonal = np.abs(rng.normal(size=N_ASSETS)) * 0.004 + 0.001
    covariance = exposures @ exposures.T + np.diag(diagonal)
    draws = rng.multivariate_normal(
        mean=np.zeros(N_ASSETS), cov=covariance, size=PANEL_MONTHS
    )
    return tuple(
        (
            month_end_instant(year, month, ZONE),
            {UNIVERSE[i]: float(draws[t][i]) for i in range(N_ASSETS)},
        )
        for t, (year, month) in enumerate(_month_keys(PANEL_FIRST, PANEL_MONTHS))
    )


def _deterministic_book() -> dict[str, float]:
    """The constant covariance book for the allocation-only strategy's dates."""
    return {
        covariance_pair_key(UNIVERSE[i], UNIVERSE[j]): (
            DETERMINISTIC_VARIANCE if i == j else DETERMINISTIC_COVARIANCE
        )
        for i in range(N_ASSETS)
        for j in range(i, N_ASSETS)
    }


# --------------------------------------------------------------------------- #
# Component 1: scikit-learn owns estimation (wrapped forecaster lifecycle).
# --------------------------------------------------------------------------- #


class _SklearnFittedModel:
    """A public ``FittedModel``-conforming wrapper over a fitted estimator.

    Its ``forecast`` method is pure translation: the scikit-learn estimator
    has already produced the covariance at fit time; this object only
    re-issues it as a canonical ``Forecast`` on the ``COVARIANCE_TARGET``
    book at the requesting information set's own as-of instant.
    """

    def __init__(
        self, book: Mapping[str, float], provenance: ForecastProvenance
    ) -> None:
        self.book = dict(book)
        self.provenance = provenance

    def forecast(self, information_set: Any) -> Forecast:
        decision_time = getattr(information_set, "as_of", None)
        return Forecast(
            values=dict(self.book),
            target=COVARIANCE_TARGET,
            decision_time=decision_time,
            produced_by=SKLEARN_WRAPPED_IDENTITY,
        )


@dataclass(frozen=True)
class SklearnShrinkageCovarianceFactory:
    """A forecaster factory whose fitted models wrap a scikit-learn estimator.

    The factory speaks only public PortLearn interfaces: it carries the public
    forecaster-factory ``fit`` lifecycle, its fits return
    ``FittedModel``-conforming objects, every emitted value is a canonical
    ``Forecast`` on the ``COVARIANCE_TARGET`` book, and fit provenance is
    stamped via the public ``fitting_provenance`` helper with the
    scikit-learn wrapped identity. scikit-learn alone estimates the
    covariance; nothing else in this module does.
    """

    estimator_name: str = SKLEARN_COVARIANCE_ESTIMATOR

    def __post_init__(self) -> None:
        if self.estimator_name not in _ESTIMATOR_NAMES:
            raise ValueError(
                "SKLEARN_COVARIANCE_ESTIMATOR must be one of "
                f"{list(_ESTIMATOR_NAMES)}; got {self.estimator_name!r}"
            )

    def fit(self, plan: Any, fit_set: Any) -> _SklearnFittedModel:
        """Fit the wrapped scikit-learn estimator on one declared fit window.

        ``fit_set`` is the fit window's admitted ``InformationSet`` (one
        scalar return observation per instrument per month); the estimator
        class is resolved from this module's namespace at fit time so the
        wrapped estimator stays the single point of estimation.
        """
        records = tuple(fit_set)
        times = sorted({record.observation_time for record in records})
        symbols = sorted({record.series_id for record in records})
        grid = {
            (record.series_id, record.observation_time): float(record.value)
            for record in records
        }
        matrix = [[grid[(symbol, time)] for symbol in symbols] for time in times]
        if self.estimator_name == "LedoitWolf":
            estimator = LedoitWolf()
        else:
            estimator = OAS()
        estimator.fit(matrix)
        covariance = estimator.covariance_
        provenance = fitting_provenance(
            plan, fit_set, wrapped_identity=SKLEARN_WRAPPED_IDENTITY
        )
        return _SklearnFittedModel(covariance_book(symbols, covariance), provenance)


# --------------------------------------------------------------------------- #
# Component 2: cvxpy owns optimization (both research strategies).
# --------------------------------------------------------------------------- #


def _solve_minimum_variance(book: Mapping[str, float]) -> dict[str, float]:
    """Solve the unit-budget minimum-variance program with cvxpy alone.

    The program: minimize ``w' C w`` subject to ``sum(w) == 1`` and
    ``w >= 0``. Only cvxpy models and solves it; this function merely
    translates the canonical covariance book into the program and the
    solved allocation back into a per-instrument book at the exact unit
    budget.
    """
    import cvxpy as cp
    import numpy as np

    required = covariance_pair_keys()
    missing = [key for key in required if key not in book]
    if missing:
        raise ValueError(
            "the covariance book is missing canonical pair keys: "
            f"{sorted(missing)}"
        )
    matrix = np.zeros((N_ASSETS, N_ASSETS))
    for i in range(N_ASSETS):
        for j in range(N_ASSETS):
            matrix[i, j] = float(book[covariance_pair_key(UNIVERSE[i], UNIVERSE[j])])
    weights = cp.Variable(N_ASSETS)
    objective = cp.Minimize(cp.quad_form(weights, cp.psd_wrap(matrix)))
    problem = cp.Problem(objective, [cp.sum(weights) == 1, weights >= 0])
    problem.solve()
    if weights.value is None:
        raise RuntimeError(
            "cvxpy did not return a solved allocation; the challenge has "
            "no fallback optimizer by design"
        )
    solved = [float(value) for value in weights.value]
    if not all(math.isfinite(value) and value >= 0.0 for value in solved):
        raise RuntimeError(
            "the cvxpy allocation is not finite and long-only; the "
            "challenge has no fallback optimizer by design"
        )
    total = math.fsum(solved)
    if total <= 0.0:
        raise RuntimeError(
            "the cvxpy allocation does not carry a positive budget; the "
            "challenge has no fallback optimizer by design"
        )
    return _exact_unit_book(
        {UNIVERSE[i]: solved[i] / total for i in range(N_ASSETS)}
    )


def _require_covariance_forecast(context: DecisionContext) -> Forecast:
    forecast = context.forecast
    if forecast is None or forecast.target != COVARIANCE_TARGET:
        raise ValueError(
            "a decision requires exactly one forecast valued on the "
            f"{COVARIANCE_TARGET!r} target through context.forecast"
        )
    missing = [key for key in covariance_pair_keys() if key not in forecast.values]
    if missing:
        raise ValueError(
            "the covariance forecast is missing canonical pair keys: "
            f"{sorted(missing)}"
        )
    return forecast


@dataclass(frozen=True)
class CvxpyMinimumVarianceStrategy:
    """Allocation-only strategy: deterministic covariance in, cvxpy weights out.

    This strategy performs no estimation at all. It consumes the covariance
    forecast supplied through ``context.forecast`` (the same interface the
    composed strategy uses) and delegates the entire optimization to cvxpy.
    """

    journal: Any = None

    def decide(self, context: DecisionContext) -> DecisionResult:
        forecast = _require_covariance_forecast(context)
        weights = _solve_minimum_variance(forecast.values)
        if self.journal is not None:
            self.journal.setdefault(ALLOCATION_ONLY_ARM, []).append(
                (context.decision_time.isoformat(), weights)
            )
        return DecisionResult(
            decision=PortfolioDecision(
                decision_time=context.decision_time,
                execution_time=context.decision_time,
                target_weights=weights,
            )
        )


@dataclass(frozen=True)
class SklearnCvxpyMinimumVarianceStrategy:
    """Composed strategy: scikit-learn covariance reaches it only via the forecast.

    This strategy's ``decide`` never estimates anything: the scikit-learn
    covariance arrives exclusively through ``context.forecast`` (estimated
    upstream by :class:`SklearnShrinkageCovarianceFactory`), and cvxpy
    alone turns that book into solved weights.
    """

    journal: Any = None

    def decide(self, context: DecisionContext) -> DecisionResult:
        forecast = _require_covariance_forecast(context)
        weights = _solve_minimum_variance(forecast.values)
        if self.journal is not None:
            self.journal.setdefault(COMPOSED_ARM, []).append(
                (context.decision_time.isoformat(), weights)
            )
        return DecisionResult(
            decision=PortfolioDecision(
                decision_time=context.decision_time,
                execution_time=context.decision_time,
                target_weights=weights,
            )
        )


# --------------------------------------------------------------------------- #
# The single declared experiment (temporal design, conditions, providers).
# --------------------------------------------------------------------------- #


class _AlwaysRebalance:
    """The declared always-rebalance policy."""

    def should_rebalance(self, date: Any, last_rebalance: Any) -> bool:
        return True


def _temporal_design() -> TemporalDesign:
    """One shared temporal design; the composed strategy declares monthly refits.

    Both strategies estimate on the same rolling calendar window. The composed
    strategy additionally declares a monthly refit cadence so the wrapped
    scikit-learn estimator's fits materialize as canonical fit events,
    keeping its fit provenance images aligned one-for-one with the
    resolved protocol.
    """
    estimation = window(
        months=ESTIMATION_WINDOW_MONTHS, mode=WindowMode.ROLLING
    )
    composed = StrategyTemporalDeclaration(
        estimation=estimation,
        selection=none,
        refit=monthly,
        preprocessing=none,
    )
    allocation_only = StrategyTemporalDeclaration(
        estimation=estimation,
        selection=none,
        refit=none,
        preprocessing=none,
    )
    return TemporalDesign(
        evaluation_span=EVALUATION_SPAN,
        evaluation_learning=EvaluationLearning.WALK_FORWARD,
        estimand="monthly_net_of_cost_portfolio_return",
        target_horizon=TARGET_HORIZON,
        timezone=ZONE,
        strategies=(
            (ALLOCATION_ONLY_ARM, allocation_only),
            (COMPOSED_ARM, composed),
        ),
    )


def external_components_record() -> Mapping[str, Mapping[str, str]]:
    """The frozen external-component record carried in the conditions."""
    return {
        "sklearn_covariance": {
            "distribution": "scikit-learn",
            "family": "estimation",
            "role": "covariance",
        },
        "cvxpy_allocator": {
            "distribution": "cvxpy",
            "family": "optimization",
            "role": "allocation",
        },
    }


def _conditions(
    information_provider: Callable[[Any], InformationSet],
    outcome_provider: Callable[[Any], Any],
    initial_weights: Mapping[str, float],
) -> ExperimentConditions:
    return ExperimentConditions(
        information_provider=information_provider,
        outcome_provider=outcome_provider,
        rebalance_policy=_AlwaysRebalance(),
        accounting_engine=ExactFillAccounting(),
        cost_model=Proportional(COST_RATE, turnover=one_way),
        initial_weights=initial_weights,
        initial_wealth=1.0,
        initial_last_rebalance_time=month_end_instant(2024, 12, ZONE),
    )


def _window_observations(
    by_month: Mapping[datetime, Mapping[str, float]],
    timeline: Sequence[datetime],
    cutoff: datetime,
) -> tuple[TimedObservation, ...]:
    """One scalar observation per instrument per month in the fit window."""
    instants = [
        instant
        for instant in timeline
        if instant <= cutoff
    ][-ESTIMATION_WINDOW_MONTHS:]
    if len(instants) < ESTIMATION_WINDOW_MONTHS:
        raise ValueError(
            "the fit window cannot cover the declared estimation window "
            f"ending at {cutoff.isoformat()}"
        )
    return tuple(
        TimedObservation(
            series_id=symbol,
            observation_time=instant,
            available_time=instant,
            value=float(by_month[instant][symbol]),
        )
        for instant in instants
        for symbol in UNIVERSE
    )


def _information_backend(
    forecast_models: Mapping[datetime, _SklearnFittedModel],
    decision_dates: Sequence[datetime],
    sklearn_dates: frozenset[datetime],
    by_month: Mapping[datetime, Mapping[str, float]],
) -> Callable[[Any], InformationSet]:
    """Point-in-time information: panel history plus one forecast channel.

    Every admitted set carries at most one ``Forecast``-valued observation
    — the runtime's one-channel-per-instant law — issued exactly at the
    decision instant it serves. scikit-learn dates receive the wrapped
    estimator's book through the fitted model's public ``forecast`` interface;
    the remaining dates receive the deterministic constant book through
    the same interface.
    """

    def _provide(instant: Any) -> InformationSet:
        items = [
            TimedObservation(
                series_id=symbol,
                observation_time=observation_time,
                available_time=observation_time,
                value=float(row[symbol]),
            )
            for observation_time, row in by_month.items()
            if observation_time <= instant
            for symbol in UNIVERSE
        ]
        if instant in sklearn_dates:
            cutoffs = [c for c in forecast_models if c <= instant]
            model = forecast_models[max(cutoffs)]
            forecast = model.forecast(InformationSet(items=[], as_of=instant))
        elif instant in decision_dates:
            forecast = Forecast(
                values=_deterministic_book(),
                target=COVARIANCE_TARGET,
                decision_time=instant,
                produced_by=DETERMINISTIC_PRODUCER,
            )
        else:
            forecast = None
        if forecast is not None:
            items.append(
                TimedObservation(
                    series_id="covariance_forecast",
                    observation_time=instant,
                    available_time=instant,
                    value=forecast,
                )
            )
        return InformationSet(items=items, as_of=instant)

    return _provide


def _outcome_backend(
    by_month: Mapping[datetime, Mapping[str, float]],
) -> Callable[[Any], Any]:
    """The realized next-month growth book for each decision segment."""

    def _provide(segment_start: Any) -> Any:
        year, month = _month_of(segment_start)
        following = (year + 1, 1) if month == 12 else (year, month + 1)
        row = by_month.get(month_end_instant(following[0], following[1], ZONE))
        if row is None:
            return None
        return {symbol: 1.0 + value for symbol, value in row.items()}

    return _provide


def run_challenge() -> Any:
    """Declare and run the single external-composition experiment."""
    from types import SimpleNamespace

    panel = synthetic_panel()
    by_month = {instant: row for instant, row in panel}
    timeline = tuple(instant for instant, _ in panel)
    decision_dates = [
        month_end_instant(2025, month, ZONE) for month in DECISION_MONTHS
    ]
    sklearn_dates = frozenset(
        month_end_instant(2025, month, ZONE) for month in SKLEARN_MONTHS
    )

    design = _temporal_design()
    resolved = resolve_temporal_protocols(
        design,
        STRATEGY_IDS,
        {arm: timeline for arm in STRATEGY_IDS},
    )

    # scikit-learn alone estimates: one real fit per resolved canonical
    # composed-strategy fit event, provenance stamped cutoff-for-cutoff.
    factory = SklearnShrinkageCovarianceFactory()
    composed_images: list[ForecastProvenance] = []
    forecast_models: dict[datetime, _SklearnFittedModel] = {}
    for event in tuple(resolved[COMPOSED_ARM].fit_events):
        fit_set = InformationSet(
            items=_window_observations(by_month, timeline, event.cutoff),
            as_of=event.cutoff,
        )
        window_start = min(record.observation_time for record in fit_set)
        plan = FittingPlan(
            model=f"scikit-learn:{SKLEARN_COVARIANCE_ESTIMATOR}",
            config={
                "estimator": SKLEARN_COVARIANCE_ESTIMATOR,
                "assets": N_ASSETS,
                "window_months": ESTIMATION_WINDOW_MONTHS,
            },
            fit_window=FitWindow(window_start=window_start),
        )
        model = factory.fit(plan, fit_set)
        composed_images.append(model.provenance)
        forecast_models[event.cutoff] = model

    # The allocation-only strategy declares no refits: exactly one canonical
    # initial-fit event, covered by one deterministic provenance image.
    allocation_events = tuple(resolved[ALLOCATION_ONLY_ARM].fit_events)
    allocation_images: list[ForecastProvenance] = []
    for event in allocation_events:
        fit_set = InformationSet(
            items=_window_observations(by_month, timeline, event.cutoff),
            as_of=event.cutoff,
        )
        window_start = min(record.observation_time for record in fit_set)
        plan = FittingPlan(
            model=DETERMINISTIC_PRODUCER,
            config={"assets": N_ASSETS, "book": "constant"},
            fit_window=FitWindow(window_start=window_start),
        )
        allocation_images.append(fitting_provenance(plan, fit_set))

    journal: dict[str, list[tuple[str, dict[str, float]]]] = {}
    registrations = (
        StrategyRegistration(
            strategy_id=ALLOCATION_ONLY_ARM,
            strategy=CvxpyMinimumVarianceStrategy(journal=journal),
        ),
        StrategyRegistration(
            strategy_id=COMPOSED_ARM,
            strategy=SklearnCvxpyMinimumVarianceStrategy(journal=journal),
        ),
    )
    schedule = DecisionSchedule(
        decision_dates=list(decision_dates),
        universes={instant: UNIVERSE for instant in decision_dates},
    )
    opening = _exact_unit_book(
        {symbol: 1.0 / N_ASSETS for symbol in UNIVERSE}
    )
    conditions = _conditions(
        _information_backend(
            forecast_models, decision_dates, sklearn_dates, by_month
        ),
        _outcome_backend(by_month),
        opening,
    )
    configuration = ExperimentConfiguration(
        strategy_ids=STRATEGY_IDS,
        target=COMPOSED_ARM,
        comparators=(ALLOCATION_ONLY_ARM,),
        ablations=(),
        robustness_checks=(),
        benchmarks=(ALLOCATION_ONLY_ARM,),
        conditions={
            "synthetic": True,
            "composition": "scikit-learn estimation x cvxpy allocation",
            "external_components": dict(external_components_record()),
        },
    )

    results, provenances = run_declared_experiment(
        configuration=configuration,
        registrations=registrations,
        schedule=schedule,
        conditions=conditions,
        temporal=design,
        record_instants_by_strategy={
            arm: timeline for arm in STRATEGY_IDS
        },
        fit_provenances_by_strategy={
            COMPOSED_ARM: tuple(composed_images),
            ALLOCATION_ONLY_ARM: tuple(allocation_images),
        },
    )
    records = {record.strategy_id: record for record in results.records}
    dropped = sorted(set(STRATEGY_IDS) - set(records))
    if dropped:
        raise ValueError(
            "the executed run returned no record for the declared strategies "
            f"{dropped}; the composition challenge fails closed"
        )
    return SimpleNamespace(
        results=results,
        provenances=dict(provenances),
        records=records,
        journal=journal,
        fit_provenances={
            COMPOSED_ARM: tuple(composed_images),
            ALLOCATION_ONLY_ARM: tuple(allocation_images),
        },
    )


# --------------------------------------------------------------------------- #
# Deterministic report.
# --------------------------------------------------------------------------- #


def _weight_digest(entries: Sequence[tuple[str, Mapping[str, float]]]) -> str:
    canonical = ";".join(
        f"{iso}:{symbol}={weights[symbol]:.6f}"
        for iso, weights in entries
        for symbol in sorted(weights)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _report(run: Any) -> str:
    lines = [
        "external composition challenge — deterministic report",
        "[f] synthetic data; composition demonstration; non-empirical",
        "",
        "[c] external components",
        (
            "[c] name=scikit-learn "
            f"version={_sklearn_module.__version__} "
            "family=estimation role=covariance licence=BSD-3-Clause"
        ),
        (
            "[c] name=cvxpy "
            f"version={_cvxpy_module.__version__} "
            "family=optimization role=allocation licence=Apache-2.0"
        ),
        "",
        (
            "[x] external_components sklearn_covariance: "
            "distribution=scikit-learn family=estimation role=covariance"
        ),
        (
            "[x] external_components cvxpy_allocator: "
            "distribution=cvxpy family=optimization role=allocation"
        ),
        "",
        (
            f"[t] forecast target={COVARIANCE_TARGET} "
            f"estimator={SKLEARN_COVARIANCE_ESTIMATOR} "
            "wrap_identity=fitting_provenance.wrapped_identity"
        ),
        (f"[t] fit provenance wrapped identity {SKLEARN_WRAPPED_IDENTITY} cutoffs="),
    ]
    for image in run.fit_provenances[COMPOSED_ARM]:
        lines[-1] += f"{image.cutoff.isoformat()},"
    for arm in sorted(run.journal):
        lines.append(f"[a] arm {arm} weight_digest={_weight_digest(run.journal[arm])}")
        for iso, weights in run.journal[arm]:
            rendered = " ".join(
                f"{symbol}={weights[symbol]:.6f}" for symbol in sorted(weights)
            )
            lines.append(f"[a] {arm} {iso} {rendered}")
    lines.extend(
        [
            "",
            (
                "[f] synthetic; composition; non-empirical — this document "
                "conveys no comparative claim"
            ),
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the challenge once and print the deterministic report.

    Unknown arguments are ignored so the entry point stays safe to invoke
    from harness contexts that pass their own argv.
    """
    parser = argparse.ArgumentParser(
        description="scikit-learn x cvxpy external composition challenge"
    )
    parser.parse_known_args(argv if argv is not None else sys.argv[1:])
    run = run_challenge()
    print(_report(run))
    return 0


if __name__ == "__main__":  # pragma: no cover
    main()
