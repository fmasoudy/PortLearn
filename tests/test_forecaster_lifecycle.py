"""Behavioral contract tests for the forecasting lifecycle module.

These nodes enforce the specified lifecycle laws adversarially: the
fitting plan is an immutable declared identity with exactly one clock
(no independent training-cutoff field exists or can be smuggled in),
the fitting cutoff is exactly the fit information set's own ``as_of``
enforced at the fitting boundary, refit grids reuse the fixed timing
errors, the fixed ``forecast(information_set)`` protocol is the only
prediction path (no side door), forecasts carry lifecycle provenance,
determinism classes are declared and honest, hyperparameter selection
labels must be realizable at the selection origin, the error taxonomy
reuses the fixed contract errors, and a dependency-free external
estimator stand-in behind a researcher-side wrapper binds every law
identically.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import random
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from portlearn.forecasting import (
    DeterminismClass,
    FittedModel,
    FittingPlan,
    FitWindow,
    ForecasterFactory,
    ForecastProvenance,
    PlanStructureError,
    ProvenanceStructureError,
    RefitSchedule,
    ScheduleStructureError,
    fitting_provenance,
    require_cutoff_from_fit_set,
    require_fit_inputs_admitted,
    require_reproducible,
    require_selection_labels_available,
)
from portlearn.interfaces import (
    Forecast,
    InformationSet,
    PortfolioDecision,
    require_forecast_decision_compatible,
)
from portlearn.observations import TimedObservation
from portlearn.timing import (
    FutureInformationError,
    InvalidChronologyError,
    NaiveTimestampError,
)

MELB = ZoneInfo("Australia/Melbourne")

EXPECTED_LIFECYCLE_SURFACE = {
    "DeterminismClass",
    "FittedModel",
    "FitWindow",
    "FittingPlan",
    "ForecasterFactory",
    "ForecastProvenance",
    "PlanStructureError",
    "ProvenanceStructureError",
    "RefitSchedule",
    "ScheduleStructureError",
    "fitting_provenance",
    "require_cutoff_from_fit_set",
    "require_fit_inputs_admitted",
    "require_reproducible",
    "require_selection_labels_available",
}


def day(number: int) -> datetime:
    """A UTC-midnight instant on a numbered day of January 2026."""
    return datetime(2026, 1, number, tzinfo=UTC)


def obs(
    observation_day: int, available_day: int, value: float, series: str = "px"
) -> TimedObservation:
    """One aware record; availability declared explicitly by day number."""
    return TimedObservation(
        series, day(observation_day), day(available_day), value
    )


def training_set(as_of: datetime, series: str = "px") -> InformationSet:
    """An admitted set of same-day-published records before ``as_of``."""
    return InformationSet(
        [
            obs(number, number, float(number), series)
            for number in range(1, as_of.day)
        ],
        as_of,
    )


def lifecycle_plan(**overrides: object) -> FittingPlan:
    """A well-formed declared plan; tests override single fields."""
    fields: dict[str, object] = {
        "model": "research-mean-model",
        "config": {"depth": {"layers": 3}, "shrinkage": 0.25},
        "seed": None,
        "fit_window": FitWindow(day(1)),
        "determinism": DeterminismClass.DETERMINISTIC,
    }
    fields.update(overrides)
    return FittingPlan(**fields)  # type: ignore[arg-type]


class _SeriesMeanForecaster:
    """A deterministic stand-in forecaster over one series."""

    def __init__(self, provenance: ForecastProvenance) -> None:
        self._provenance = provenance

    def forecast(self, information_set: InformationSet) -> Forecast:
        records = list(information_set)
        mean = sum(record.value for record in records) / len(records)
        return Forecast(
            {"asset": mean},
            "expected_return",
            information_set.as_of,
            self._provenance.produced_by,
        )


class _SeededRandomForecaster:
    """A seed-reproducible stand-in drawing from a seeded generator."""

    def __init__(
        self, provenance: ForecastProvenance, seed: int
    ) -> None:
        self._provenance = provenance
        self._generator = random.Random(seed)

    def forecast(self, information_set: InformationSet) -> Forecast:
        return Forecast(
            {"asset": self._generator.random()},
            "expected_return",
            information_set.as_of,
            self._provenance.produced_by,
        )


class _FactoryStandIn:
    """A researcher-side estimator factory honoring the lifecycle laws."""

    def __init__(self, plan: FittingPlan) -> None:
        self._plan = plan

    def fit(self, information_set: InformationSet) -> object:
        records = list(information_set)
        require_fit_inputs_admitted(records, information_set)
        provenance = fitting_provenance(self._plan, information_set)
        if self._plan.seed is None:
            return _SeriesMeanForecaster(provenance)
        return _SeededRandomForecaster(provenance, self._plan.seed)


class _ExternalEstimatorStandIn:
    """A dependency-free stand-in for an external estimator library.

    Its surface deliberately mimics a foreign library (an ``estimate``
    method plus its own identity and version) so the wrapper pattern is
    exercised against something that is not a PortLearn protocol.
    """

    library_name = "external-stand-in-library"
    library_version = "9.9.9-test"

    def __init__(self, claimed_training_end: datetime | None = None) -> None:
        self.claimed_training_end = claimed_training_end

    def estimate(self, records: list[TimedObservation]) -> dict[str, float]:
        return {
            "asset": sum(record.value for record in records) / len(records)
        }


class _WrappingForecaster:
    """A researcher-side wrapper adapting the external stand-in."""

    def __init__(self, plan: FittingPlan) -> None:
        self._plan = plan
        self._external = _ExternalEstimatorStandIn()
        self._provenance: ForecastProvenance | None = None

    def fit(self, information_set: InformationSet) -> _WrappingForecaster:
        records = list(information_set)
        require_fit_inputs_admitted(records, information_set)
        require_cutoff_from_fit_set(
            self._external.claimed_training_end
            if self._external.claimed_training_end is not None
            else information_set.as_of,
            information_set,
        )
        self._provenance = fitting_provenance(
            self._plan,
            information_set,
            wrapped_identity=(
                f"{_ExternalEstimatorStandIn.library_name}"
                f"@{_ExternalEstimatorStandIn.library_version}"
            ),
        )
        return self

    def forecast(self, information_set: InformationSet) -> Forecast:
        assert self._provenance is not None
        values = self._external.estimate(list(information_set))
        return Forecast(
            values,
            "expected_return",
            information_set.as_of,
            self._provenance.produced_by,
        )


def _assert_protocol_is_declaration_only(protocol: type) -> None:
    """Every method body in a lifecycle protocol is a bare ``...``."""
    tree = ast.parse(inspect.getsource(protocol))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            statements = [
                statement
                for statement in node.body
                if not (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                )
            ]
            assert len(statements) == 1, (
                f"protocol method {node.name!r} must be declaration-only"
            )
            only = statements[0]
            assert (
                isinstance(only, ast.Expr)
                and isinstance(only.value, ast.Constant)
                and only.value.value is Ellipsis
            ), f"protocol method {node.name!r} must be declaration-only"


# --------------------------------------------------------------------------- #
# Declared plan: one clock, immutable, canonicalized configuration
# --------------------------------------------------------------------------- #


def test_fitting_plan_declares_its_identity_without_any_cutoff_field() -> None:
    """The plan's fixed field set contains no training-cutoff field.

    The fitting cutoff is the fit information set's own ``as_of``; the
    plan is declared before any fit exists, so a cutoff-bearing plan
    construction is not representable at all.
    """
    plan = lifecycle_plan()

    assert {field.name for field in dataclasses.fields(FittingPlan)} == {
        "model",
        "config",
        "seed",
        "fit_window",
        "determinism",
        "nondeterminism_explanation",
    }
    for field in dataclasses.fields(FittingPlan):
        assert "cutoff" not in field.name.lower()
    assert {field.name for field in dataclasses.fields(FitWindow)} == {
        "window_start"
    }
    assert {field.name for field in dataclasses.fields(RefitSchedule)} == {
        "refit_instants"
    }

    with pytest.raises(TypeError):
        FittingPlan(  # type: ignore[call-arg]
            model="research-mean-model",
            config={},
            seed=None,
            fit_window=FitWindow(day(1)),
            determinism=DeterminismClass.DETERMINISTIC,
            training_cutoff=day(9),
        )
    with pytest.raises(TypeError):
        FittingPlan(  # type: ignore[call-arg]
            model="research-mean-model",
            config={},
            seed=None,
            fit_window=FitWindow(day(1)),
            determinism=DeterminismClass.DETERMINISTIC,
            cutoff=day(9),
        )
    with pytest.raises(TypeError):
        ForecastProvenance(
            model="research-mean-model",
            config_hash="0" * 64,
            cutoff=day(9),
            seed=None,
            determinism=DeterminismClass.DETERMINISTIC,
            fit_window=FitWindow(day(1)),
            training_cutoff=day(9),
        )

    assert plan.model == "research-mean-model"
    assert plan.seed is None
    assert plan.fit_window.window_start == day(1)
    assert plan.determinism is DeterminismClass.DETERMINISTIC


def test_fitting_plan_is_immutable_with_a_frozen_canonical_config() -> None:
    """Plan attributes reject mutation and the configuration cannot be
    mutated through nested caller containers; the config hash is a
    deterministic function of configuration content alone."""
    caller_config = {
        "shrinkage": 0.25,
        "depth": {"layers": 3, "rates": (0.1, 0.01)},
    }
    plan = lifecycle_plan(config=caller_config)

    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.model = "other-model"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.seed = 7  # type: ignore[misc]

    caller_config["depth"]["layers"] = 99
    caller_config["extra"] = True
    assert plan.config["depth"]["layers"] == 3
    assert "extra" not in plan.config
    nested = plan.config["depth"]
    with pytest.raises(TypeError):
        nested["layers"] = 99  # type: ignore[index]
    assert plan.config["depth"]["layers"] == 3

    reordered = lifecycle_plan(
        config={
            "depth": {"rates": (0.1, 0.01), "layers": 3},
            "shrinkage": 0.25,
        }
    )
    assert reordered.config_hash == plan.config_hash
    assert len(plan.config_hash) == 64

    different = lifecycle_plan(
        config={"shrinkage": 0.5, "depth": {"layers": 3}}
    )
    assert different.config_hash != plan.config_hash

    sequenced = lifecycle_plan(config={"rates": [1, 2, 3]})
    assert isinstance(sequenced.config["rates"], tuple)
    assert sequenced.config["rates"] == (1, 2, 3)


def test_fitting_plan_rejects_structural_failures_fail_closed() -> None:
    """Blank or non-string model identity, malformed configuration,
    wrong seed typing, a raw instant as the fit window, and a bare
    string in place of the determinism class all reject with the
    module-owned structural error (a ``ValueError`` subclass)."""
    for bad_model in ("", "   ", 42, None):
        with pytest.raises(PlanStructureError):
            lifecycle_plan(model=bad_model)
    with pytest.raises(PlanStructureError):
        lifecycle_plan(config=[("shrinkage", 0.25)])
    with pytest.raises(PlanStructureError):
        lifecycle_plan(config={1: 0.25})
    with pytest.raises(PlanStructureError):
        lifecycle_plan(config={"opts": {1, 2}})
    with pytest.raises(PlanStructureError):
        lifecycle_plan(config={"mystery": object()})
    for bad_seed in ("7", 3.5, True, [7]):
        with pytest.raises(PlanStructureError):
            lifecycle_plan(seed=bad_seed)
    with pytest.raises(PlanStructureError):
        lifecycle_plan(fit_window=day(1))
    with pytest.raises(PlanStructureError):
        lifecycle_plan(determinism="deterministic")
    with pytest.raises(NaiveTimestampError):
        FitWindow(datetime(2026, 1, 1))  # noqa: DTZ001  # intentional naive instant
    with pytest.raises(NaiveTimestampError):
        FitWindow("2026-01-01")

    assert issubclass(PlanStructureError, ValueError)


# --------------------------------------------------------------------------- #
# The cutoff law: one clock, the fit set's own as_of
# --------------------------------------------------------------------------- #


def test_fit_cutoff_is_exactly_the_fit_set_as_of_in_provenance() -> None:
    """Fitting provenance records the fit information set's own
    ``as_of`` as the cutoff — the same object, not a re-derived or
    caller-declared instant."""
    fit_set = training_set(day(10))
    provenance = fitting_provenance(lifecycle_plan(), fit_set)

    assert provenance.cutoff == fit_set.as_of
    assert provenance.cutoff is fit_set.as_of
    assert provenance.model == "research-mean-model"
    assert provenance.seed is None
    assert provenance.determinism is DeterminismClass.DETERMINISTIC
    assert provenance.fit_window.window_start == day(1)
    assert provenance.config_hash == lifecycle_plan().config_hash


def test_fitting_boundary_rejects_records_available_after_the_as_of() -> None:
    """Every record consumed by fitting is re-checked at the fitting
    boundary: a record whose availability follows the fit set's
    ``as_of`` rejects with the fixed ``FutureInformationError``."""
    fit_set = training_set(day(10))

    honest = [obs(1, 1, 1.0), obs(9, 9, 9.0)]
    assert require_fit_inputs_admitted(honest, fit_set) is None
    assert require_fit_inputs_admitted([], fit_set) is None

    hostile = obs(9, 11, 9.0)
    with pytest.raises(FutureInformationError):
        require_fit_inputs_admitted([obs(1, 1, 1.0), hostile], fit_set)

    naive_fit_set = SimpleNamespace(as_of=datetime(2026, 1, 10))  # noqa: DTZ001  # intentional naive instant
    with pytest.raises(NaiveTimestampError):
        require_fit_inputs_admitted(honest, naive_fit_set)
    with pytest.raises(PlanStructureError):
        require_fit_inputs_admitted(42, fit_set)  # type: ignore[arg-type]


def test_divergent_cutoff_assertions_are_rejected_at_one_clock() -> None:
    """A cutoff asserted anywhere other than the fit set's ``as_of`` —
    later or earlier — is rejected, and the provenance factory offers
    no cutoff parameter through which divergence could be smuggled."""
    fit_set = training_set(day(10))

    assert require_cutoff_from_fit_set(fit_set.as_of, fit_set) is None

    later = fit_set.as_of + timedelta(hours=1)
    earlier = fit_set.as_of - timedelta(hours=1)
    with pytest.raises(ProvenanceStructureError):
        require_cutoff_from_fit_set(later, fit_set)
    with pytest.raises(ProvenanceStructureError):
        require_cutoff_from_fit_set(earlier, fit_set)
    with pytest.raises(NaiveTimestampError):
        require_cutoff_from_fit_set(datetime(2026, 1, 10), fit_set)  # noqa: DTZ001  # intentional naive instant
    with pytest.raises(PlanStructureError):
        require_cutoff_from_fit_set(fit_set.as_of, SimpleNamespace())

    signature = inspect.signature(fitting_provenance)
    parameter_names = set(signature.parameters)
    assert parameter_names == {"plan", "fit_set", "wrapped_identity"}
    for name in parameter_names:
        assert "cutoff" not in name.lower()
    for helper in (
        require_cutoff_from_fit_set,
        require_fit_inputs_admitted,
        require_selection_labels_available,
    ):
        for name in inspect.signature(helper).parameters:
            assert "cutoff" not in name.lower()

    assert issubclass(ProvenanceStructureError, ValueError)


def test_declared_fit_window_start_must_not_follow_the_fit_cutoff() -> None:
    """A fit window that begins after the information the fit set
    admitted is chronologically impossible and rejects with the fixed
    ``InvalidChronologyError``; a window starting at or before the
    cutoff is admissible."""
    fit_set = training_set(day(10))

    with pytest.raises(InvalidChronologyError):
        fitting_provenance(lifecycle_plan(fit_window=FitWindow(day(11))), fit_set)

    boundary = fitting_provenance(
        lifecycle_plan(fit_window=FitWindow(day(10))), fit_set
    )
    assert boundary.cutoff == fit_set.as_of
    admissible = fitting_provenance(
        lifecycle_plan(fit_window=FitWindow(day(4))), fit_set
    )
    assert admissible.cutoff == fit_set.as_of


# --------------------------------------------------------------------------- #
# Refit schedules: declared grids under the fixed timing laws
# --------------------------------------------------------------------------- #


def test_refit_schedule_grid_validation_reuses_the_frozen_timing_errors() -> None:
    """Naive grid entries reject with ``NaiveTimestampError``; repeated,
    equal-in-another-zone, or reversed instants reject with
    ``InvalidChronologyError``; a non-sequence, a string, or an empty
    grid rejects with the module-owned structural error."""
    schedule = RefitSchedule([day(2), day(6), day(11)])
    assert schedule.refit_instants == (day(2), day(6), day(11))
    assert require_cutoff_from_fit_set(day(2), SimpleNamespace(as_of=day(2))) is None

    with pytest.raises(NaiveTimestampError):
        RefitSchedule([day(2), datetime(2026, 1, 6)])  # noqa: DTZ001  # intentional naive instant
    with pytest.raises(InvalidChronologyError):
        RefitSchedule([day(2), day(2), day(6)])
    with pytest.raises(InvalidChronologyError):
        RefitSchedule([day(6), day(2)])
    melbourne_same_instant_as_day3 = datetime(2026, 1, 3, 11, 0, tzinfo=MELB)
    with pytest.raises(InvalidChronologyError):
        RefitSchedule([day(3), melbourne_same_instant_as_day3])
    for malformed in ((), "2026-01-02", 42, None):
        with pytest.raises(ScheduleStructureError):
            RefitSchedule(malformed)  # type: ignore[arg-type]

    assert issubclass(ScheduleStructureError, ValueError)


def test_refit_schedule_is_immutable_and_copies_the_supplied_instants() -> None:
    """The schedule stores its own tuple of the original instants:
    caller-side mutation of the submitted container is invisible, the
    stored entries are the exact objects supplied, and attribute
    assignment rejects."""
    supplied = [day(2), day(5), day(9)]
    schedule = RefitSchedule(supplied)

    supplied.append(day(20))
    supplied[0] = day(30)
    assert schedule.refit_instants == (day(2), day(5), day(9))
    assert isinstance(schedule.refit_instants, tuple)

    melbourne_instant = datetime(2026, 1, 3, 11, 0, tzinfo=MELB)
    preserved = RefitSchedule([day(1), melbourne_instant])
    assert preserved.refit_instants[1] is melbourne_instant

    with pytest.raises(dataclasses.FrozenInstanceError):
        schedule.refit_instants = ()  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# The forecast origin: the fixed protocol is the only prediction path
# --------------------------------------------------------------------------- #


def test_forecast_origin_is_exactly_the_information_set_as_of() -> None:
    """A fitted stand-in produces a forecast whose ``decision_time`` is
    exactly the information set's ``as_of``, and that forecast is
    compatible with a same-instant portfolio decision under the fixed
    validator."""
    fit_set = training_set(day(10))
    fitted = _FactoryStandIn(lifecycle_plan()).fit(fit_set)

    decision_set = training_set(day(12), series="px")
    forecast = fitted.forecast(decision_set)

    assert isinstance(forecast, Forecast)
    assert forecast.decision_time == decision_set.as_of
    decision = PortfolioDecision(
        decision_time=decision_set.as_of,
        execution_time=decision_set.as_of + timedelta(days=1),
        target_weights={"asset": 1.0},
    )
    assert require_forecast_decision_compatible(forecast, decision) is None


def test_the_lifecycle_surface_offers_no_prediction_side_door() -> None:
    """The module defines no ``predict``-shaped path anywhere in its
    source, its export list is exactly the public contract surface,
    and the two lifecycle protocols are declaration-only and
    static-only."""
    from portlearn import forecasting

    source = Path(forecasting.__file__).read_text()
    tree = ast.parse(source)
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "predict" not in defined
    assert not hasattr(forecasting, "predict")

    assert set(forecasting.__all__) == EXPECTED_LIFECYCLE_SURFACE
    for name in sorted(EXPECTED_LIFECYCLE_SURFACE):
        assert getattr(forecasting, name, None) is not None

    _assert_protocol_is_declaration_only(FittedModel)
    _assert_protocol_is_declaration_only(ForecasterFactory)
    with pytest.raises(TypeError):
        isinstance(_SeriesMeanForecaster, FittedModel)
    with pytest.raises(TypeError):
        isinstance(_FactoryStandIn, ForecasterFactory)

    fit_members = [
        name
        for name, value in vars(ForecasterFactory).items()
        if not name.startswith("_") and callable(value)
    ]
    assert fit_members == ["fit"]
    forecast_members = [
        name
        for name, value in vars(FittedModel).items()
        if not name.startswith("_") and callable(value)
    ]
    assert forecast_members == ["forecast"]


def test_composed_lifecycle_walk_from_plan_through_fit_to_forecast() -> None:
    """The composed walk holds: a declared plan fits through the
    boundary checks on an admitted set, provenance records that set's
    ``as_of``, each scheduled refit instant owns decisions until the
    next, and the forecast origin equals the deciding set's
    ``as_of``."""
    schedule = RefitSchedule([day(4), day(8)])
    plan = lifecycle_plan()

    fitted_models = []
    for instant in schedule.refit_instants:
        fit_set = training_set(instant)
        model = _FactoryStandIn(plan).fit(fit_set)
        fitted_models.append((instant, model, fit_set.as_of))

    for instant, model, cutoff in fitted_models:
        assert cutoff == instant

    deciding = day(6)
    eligible = [
        (instant, model)
        for instant, model, _ in fitted_models
        if instant <= deciding
    ]
    chosen_instant, chosen_model = eligible[-1]
    assert chosen_instant == day(4)

    decision_set = training_set(deciding)
    forecast = chosen_model.forecast(decision_set)
    assert forecast.decision_time == decision_set.as_of
    assert forecast.produced_by.startswith("portlearn.forecasting:")


# --------------------------------------------------------------------------- #
# Provenance: the plan identity rides on every forecast
# --------------------------------------------------------------------------- #


def test_forecast_provenance_carries_plan_identity_cutoff_and_seed() -> None:
    """The deterministic provenance token carries the model identity,
    the configuration hash, the fit-set cutoff, the declared seed (or
    its explicit absence), and the determinism class."""
    fit_set = training_set(day(10))

    native = fitting_provenance(lifecycle_plan(), fit_set)
    token = native.produced_by
    assert native.model in token
    assert f"config_sha256={native.config_hash}" in token
    assert f"fit_cutoff={fit_set.as_of.isoformat()}" in token
    assert "seed=none" in token
    assert "determinism=deterministic" in token
    assert "wrapped=" not in token
    assert token == fitting_provenance(lifecycle_plan(), fit_set).produced_by

    seeded = fitting_provenance(
        lifecycle_plan(
            seed=7, determinism=DeterminismClass.SEED_REPRODUCIBLE
        ),
        fit_set,
    )
    assert "seed=7" in seeded.produced_by
    assert "determinism=seed-reproducible" in seeded.produced_by

    forecaster = _SeriesMeanForecaster(native)
    forecast = forecaster.forecast(fit_set)
    assert forecast.produced_by == token


def test_blank_provenance_identity_is_rejected_fail_closed() -> None:
    """A blank model identity rejects at plan construction, a blank
    wrapped-identity declaration rejects at the provenance factory, and
    the fixed forecast object rejects a blank ``produced_by``."""
    fit_set = training_set(day(10))

    with pytest.raises(PlanStructureError):
        lifecycle_plan(model="   ")
    with pytest.raises(PlanStructureError):
        fitting_provenance(lifecycle_plan(), fit_set, wrapped_identity="  ")
    with pytest.raises(PlanStructureError):
        fitting_provenance(lifecycle_plan(), fit_set, wrapped_identity=42)

    with pytest.raises(ValueError):
        Forecast(
            {"asset": 0.5},
            "expected_return",
            fit_set.as_of,
            " ",
        )


# --------------------------------------------------------------------------- #
# Determinism: declared classes, honest behavior
# --------------------------------------------------------------------------- #


def test_determinism_classes_are_exactly_the_three_declared_kinds() -> None:
    """The enum carries exactly DETERMINISTIC, SEED_REPRODUCIBLE, and
    NONDETERMINISTIC; a seed-reproducible declaration requires a
    declared seed; a nondeterministic declaration requires its
    explanation; and a contradictory explanation on a deterministic
    declaration rejects."""
    assert {member.name for member in DeterminismClass} == {
        "DETERMINISTIC",
        "SEED_REPRODUCIBLE",
        "NONDETERMINISTIC",
    }

    with pytest.raises(PlanStructureError):
        lifecycle_plan(determinism=DeterminismClass.SEED_REPRODUCIBLE)

    reproducible = lifecycle_plan(
        seed=11, determinism=DeterminismClass.SEED_REPRODUCIBLE
    )
    fit_set = training_set(day(10))
    provenance = fitting_provenance(reproducible, fit_set)
    assert provenance.determinism is DeterminismClass.SEED_REPRODUCIBLE
    assert provenance.seed == 11

    with pytest.raises(PlanStructureError):
        lifecycle_plan(
            determinism=DeterminismClass.NONDETERMINISTIC,
            nondeterminism_explanation="  ",
        )
    disclosed = lifecycle_plan(
        determinism=DeterminismClass.NONDETERMINISTIC,
        nondeterminism_explanation="gpu reduction order",
    )
    disclosed_provenance = fitting_provenance(disclosed, fit_set)
    assert (
        disclosed_provenance.nondeterminism_explanation
        == "gpu reduction order"
    )
    assert "nondeterminism_note=gpu reduction order" in (
        disclosed_provenance.produced_by
    )

    with pytest.raises(PlanStructureError):
        lifecycle_plan(
            determinism=DeterminismClass.DETERMINISTIC,
            nondeterminism_explanation="actually not",
        )


def test_deterministic_stand_in_produces_identical_forecasts_across_runs() -> None:
    """A deterministic stand-in run twice at one origin yields
    identical forecasts the contract helper accepts, a seeded
    reproducible stand-in reproduces across fresh instances, and a
    divergent pair at one origin is rejected by the helper."""
    fit_set = training_set(day(10))
    provenance = fitting_provenance(lifecycle_plan(), fit_set)

    stand_in = _SeriesMeanForecaster(provenance)
    first = stand_in.forecast(fit_set)
    second = stand_in.forecast(fit_set)
    assert first.values == second.values
    assert require_reproducible((first, second)) is None

    seeded_plan = lifecycle_plan(
        seed=11, determinism=DeterminismClass.SEED_REPRODUCIBLE
    )
    seeded_provenance = fitting_provenance(seeded_plan, fit_set)
    fresh_a = _SeededRandomForecaster(seeded_provenance, 11)
    fresh_b = _SeededRandomForecaster(seeded_provenance, 11)
    assert (
        require_reproducible(
            (fresh_a.forecast(fit_set), fresh_b.forecast(fit_set))
        )
        is None
    )

    divergent = dataclasses.replace(
        first, values={"asset": first.values["asset"] + 0.5}
    )
    with pytest.raises(ProvenanceStructureError):
        require_reproducible((first, divergent))

    relabeled = dataclasses.replace(first, produced_by="relabelled clone")
    with pytest.raises(ProvenanceStructureError):
        require_reproducible((first, relabeled))

    for malformed in ((), (first,), 42):
        with pytest.raises(ProvenanceStructureError):
            require_reproducible(malformed)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Tuning timing: selection labels realizable at the selection origin
# --------------------------------------------------------------------------- #


def test_selection_labels_available_after_the_origin_are_blocked() -> None:
    """Scoring a configuration on a label not yet available at the
    selection origin — the walk-forward leak — rejects with the fixed
    ``FutureInformationError``; one late label poisons the whole
    collection, and a naive origin rejects with the fixed naive
    error."""
    origin = day(10)
    realizable = obs(5, 9, 1.5)
    post_origin = obs(5, 11, 2.5)

    with pytest.raises(FutureInformationError):
        require_selection_labels_available([realizable, post_origin], origin)
    with pytest.raises(FutureInformationError):
        require_selection_labels_available([post_origin], origin)
    with pytest.raises(NaiveTimestampError):
        require_selection_labels_available([realizable], datetime(2026, 1, 10))  # noqa: DTZ001  # intentional naive instant
    with pytest.raises(PlanStructureError):
        require_selection_labels_available(42, origin)  # type: ignore[arg-type]


def test_selection_labels_honoring_the_origin_rule_are_admitted() -> None:
    """Labels whose availability is at or before the selection origin —
    including availability exactly at the origin — satisfy the rule."""
    origin = day(10)
    assert (
        require_selection_labels_available(
            [obs(5, 9, 1.5), obs(9, 10, 2.5)], origin
        )
        is None
    )
    assert require_selection_labels_available([], origin) is None
    melbourne_origin = datetime(2026, 1, 10, 11, 0, tzinfo=MELB)
    assert (
        require_selection_labels_available(
            [obs(9, 10, 2.5)], melbourne_origin
        )
        is None
    )


# --------------------------------------------------------------------------- #
# Package discipline: fixed errors reused, structural arm module-owned
# --------------------------------------------------------------------------- #


def test_error_taxonomy_reuses_frozen_errors_and_keeps_init_untouched() -> None:
    """The module defines none of the six fixed contract errors, the
    fixed battery tuple is unchanged, the structural arm is a
    ``ValueError`` arm disjoint from the fixed classes, and importing
    the package alone does not import this module."""
    from portlearn import forecasting
    from portlearn import observations as observations_module
    from portlearn import timing as timing_module
    from portlearn.leakage import FROZEN_CONTRACT_ERRORS

    frozen_names = {
        "NaiveTimestampError",
        "InvalidChronologyError",
        "FutureInformationError",
        "MissingAvailabilityError",
        "AmbiguousObservationError",
        "FeatureLineageError",
    }
    for name in frozen_names:
        assert name not in vars(forecasting), (
            f"the fixed error {name} must not be re-defined or shadowed"
        )

    assert FROZEN_CONTRACT_ERRORS == (
        timing_module.FutureInformationError,
        timing_module.InvalidChronologyError,
        timing_module.MissingAvailabilityError,
        timing_module.NaiveTimestampError,
        observations_module.AmbiguousObservationError,
        observations_module.FeatureLineageError,
    )

    for structural in (
        PlanStructureError,
        ScheduleStructureError,
        ProvenanceStructureError,
    ):
        assert issubclass(structural, ValueError)
        for fixed in FROZEN_CONTRACT_ERRORS:
            assert not issubclass(structural, fixed)

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import portlearn, sys;"
                "sys.exit(0 if 'portlearn.forecasting' not in sys.modules else 1)"
            ),
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 0, (
        "importing the package must not eagerly import the lifecycle module"
    )


# --------------------------------------------------------------------------- #
# External estimators: the wrapper pattern binds every law
# --------------------------------------------------------------------------- #


def test_external_stand_in_behind_a_wrapper_binds_every_law() -> None:
    """A dependency-free external-estimator stand-in behind a
    researcher-side wrapper conforms to the whole lifecycle: the fixed
    protocol shape, the one-clock cutoff law (including rejection of
    the wrapped library's divergent internal bookkeeping), and
    provenance recording the wrapped implementation's identity and
    version."""
    fit_set = training_set(day(10))
    plan = lifecycle_plan(seed=3, determinism=DeterminismClass.SEED_REPRODUCIBLE)

    wrapper = _WrappingForecaster(plan).fit(fit_set)
    assert isinstance(wrapper, _WrappingForecaster)

    decision_set = training_set(day(12))
    forecast = wrapper.forecast(decision_set)
    assert isinstance(forecast, Forecast)
    assert forecast.decision_time == decision_set.as_of
    decision = PortfolioDecision(
        decision_time=decision_set.as_of,
        execution_time=decision_set.as_of + timedelta(days=1),
        target_weights={"asset": 1.0},
    )
    assert require_forecast_decision_compatible(forecast, decision) is None

    assert wrapper._provenance is not None
    assert wrapper._provenance.cutoff is fit_set.as_of
    assert wrapper._provenance.determinism is (
        DeterminismClass.SEED_REPRODUCIBLE
    )
    assert _ExternalEstimatorStandIn.library_name in (
        wrapper._provenance.produced_by
    )
    assert _ExternalEstimatorStandIn.library_version in (
        wrapper._provenance.produced_by
    )
    assert wrapper._provenance.wrapped_identity == (
        f"{_ExternalEstimatorStandIn.library_name}"
        f"@{_ExternalEstimatorStandIn.library_version}"
    )

    hostile = _WrappingForecaster(plan)
    hostile._external = _ExternalEstimatorStandIn(
        claimed_training_end=fit_set.as_of + timedelta(days=1)
    )
    with pytest.raises(ProvenanceStructureError):
        hostile.fit(fit_set)

    late_records = [obs(9, 11, 9.0)]
    with pytest.raises(FutureInformationError):
        require_fit_inputs_admitted(late_records, fit_set)
