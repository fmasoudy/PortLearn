"""Behavioral contract tests for the feature-transform module.

These tests enforce the specified transform laws adversarially: window
membership keys on availability (a record late to be available never
joins a window timestamped before its availability), outputs are
lineage-monotone under the fixed exported validator, insufficient
windows unconditional (never NaN, never partial output, never
imputation), scaler statistics come only from the declared fit window
and are fixed after fit, lag shifts positions with availability
floored at the shifted instant, and carry-forward is bounded, declared,
and absent beyond its bound. A dependency-free foreign-transform
stand-in exercises the researcher-side wrapper pattern behind the
fixed protocol.
"""

from __future__ import annotations

import inspect
import statistics
from datetime import UTC, datetime, timedelta

import pytest

from portlearn.interfaces import require_feature_lineage
from portlearn.leakage import FROZEN_CONTRACT_ERRORS
from portlearn.observations import (
    AmbiguousObservationError,
    FeatureLineageError,
    TimedObservation,
)
from portlearn.timing import NaiveTimestampError
from portlearn.transforms import (
    CarryForward,
    InsufficientWindowError,
    Lag,
    MinMaxScaler,
    RollingMean,
    RollingVolatility,
    StandardScaler,
    WindowDeclarationError,
)


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


def late_published_series() -> list[TimedObservation]:
    """A revised-macro shape: a day-3 observation published on day 9.

    Availability order differs from observation order, so any window
    keyed on observation time silently admits information that did not
    exist at the output's timestamp — the named leakage pattern.
    """
    return [
        obs(1, 1, 1.0, series="macro.gdp"),
        obs(2, 2, 2.0, series="macro.gdp"),
        obs(3, 9, 9.0, series="macro.gdp"),
        obs(4, 4, 4.0, series="macro.gdp"),
        obs(5, 5, 5.0, series="macro.gdp"),
    ]


# --------------------------------------------------------------------------- #
# Window membership keys on availability
# --------------------------------------------------------------------------- #


def test_rolling_mean_window_excludes_records_late_to_be_available() -> None:
    outputs = RollingMean(2).transform(late_published_series())
    assert outputs == [
        TimedObservation("macro.gdp|rolling_mean[2]", day(2), day(2), 1.5),
        TimedObservation("macro.gdp|rolling_mean[2]", day(4), day(4), 3.0),
        TimedObservation("macro.gdp|rolling_mean[2]", day(5), day(5), 4.5),
        TimedObservation("macro.gdp|rolling_mean[2]", day(9), day(9), 7.0),
    ]
    # Independent re-derivation: no window timestamped before day 9 may
    # derive from the day-9-published value.
    for output in outputs:
        if output.available_time < day(9):
            admitted = [
                record
                for record in late_published_series()
                if record.available_time <= output.available_time
            ]
            members = admitted[-2:]
            assert 9.0 not in [member.value for member in members]
            assert output.value == statistics.fmean(
                [member.value for member in members]
            )


def test_rolling_volatility_window_excludes_records_late_to_be_available() -> None:
    outputs = RollingVolatility(2).transform(late_published_series())
    member_pairs = [(1.0, 2.0), (2.0, 4.0), (4.0, 5.0), (5.0, 9.0)]
    assert [output.observation_time for output in outputs] == [
        day(2),
        day(4),
        day(5),
        day(9),
    ]
    assert [output.available_time for output in outputs] == [
        day(2),
        day(4),
        day(5),
        day(9),
    ]
    assert [output.value for output in outputs] == pytest.approx(
        [statistics.pstdev(pair) for pair in member_pairs]
    )
    assert all(
        output.series_id == "macro.gdp|rolling_volatility[2]"
        for output in outputs
    )


# --------------------------------------------------------------------------- #
# Output timing is lineage-monotone under the strict validator
# --------------------------------------------------------------------------- #


def test_rolling_outputs_pass_the_frozen_lineage_validator() -> None:
    records = late_published_series()
    outputs = RollingMean(3).transform(records)
    assert len(outputs) == 3
    for output in outputs:
        # Each output derives from the records admitted at its own
        # reference instant — the fixed law's own scope ("the latest
        # input it derives from").
        admitted = [
            record
            for record in records
            if record.available_time <= output.available_time
        ]
        require_feature_lineage(admitted, [output])


def test_scaler_outputs_pass_the_frozen_lineage_validator() -> None:
    fit = [obs(1, 1, 1.0), obs(2, 2, 2.0), obs(3, 3, 6.0)]
    scaler = StandardScaler(fit)
    records = [obs(4, 4, 4.0), obs(5, 5, 5.0)]
    outputs = scaler.transform(records)
    # Each output derives from the fit window and its own record — the
    # derivation set the fixed law scopes itself to.
    for record, output in zip(records, outputs):
        require_feature_lineage(fit + [record], [output])
    # The strict validator itself — not a restatement — rejects a
    # hand-built output declared available before its inputs.
    hand_built = TimedObservation("px|standard_scaled", day(1), day(2), 0.0)
    with pytest.raises(FeatureLineageError):
        require_feature_lineage(fit + records, [hand_built])


# --------------------------------------------------------------------------- #
# Explicit fitting windows for stateful transforms
# --------------------------------------------------------------------------- #


def test_scaler_statistics_come_only_from_the_declared_fit_window() -> None:
    fit = [obs(1, 1, 1.0), obs(2, 2, 2.0), obs(3, 3, 6.0)]
    scaler = StandardScaler(fit)
    provenance = scaler.provenance
    assert provenance["transform"] == "StandardScaler"
    assert provenance["fit_record_count"] == 3
    assert provenance["fit_window_first_available"] == day(1)
    assert provenance["fit_window_last_available"] == day(3)
    assert provenance["mean"] == pytest.approx(3.0)
    assert provenance["standard_deviation"] == pytest.approx(
        statistics.pstdev([1.0, 2.0, 6.0])
    )
    # A record far beyond the declared fit window never moves the
    # fixed statistics: it is scaled by them, never absorbed into them.
    outputs = scaler.transform([obs(9, 9, 1000.0)])
    assert outputs[0].value == pytest.approx(
        (1000.0 - 3.0) / statistics.pstdev([1.0, 2.0, 6.0])
    )


def test_scaler_statistics_are_frozen_after_fit() -> None:
    fit = [obs(1, 1, 1.0), obs(2, 2, 2.0), obs(3, 3, 6.0)]
    scaler = StandardScaler(fit)
    frozen_mean = scaler.provenance["mean"]
    frozen_spread = scaler.provenance["standard_deviation"]
    first = scaler.transform([obs(4, 4, 4.0)])
    second = scaler.transform([obs(6, 6, 50.0), obs(7, 7, -20.0)])
    assert scaler.provenance["mean"] == frozen_mean
    assert scaler.provenance["standard_deviation"] == frozen_spread
    assert first[0].value == pytest.approx(
        (4.0 - frozen_mean) / frozen_spread
    )
    later_records = [obs(6, 6, 50.0), obs(7, 7, -20.0)]
    for record, output in zip(later_records, second):
        assert output.value == pytest.approx(
            (record.value - frozen_mean) / frozen_spread
        )


# --------------------------------------------------------------------------- #
# Insufficient windows unconditional
# --------------------------------------------------------------------------- #


def test_rolling_mean_fails_closed_on_insufficient_window_support() -> None:
    with pytest.raises(InsufficientWindowError) as raised:
        RollingMean(3).transform([obs(1, 1, 1.0), obs(2, 2, 2.0)])
    assert isinstance(raised.value, ValueError)
    with pytest.raises(InsufficientWindowError):
        RollingMean(2).transform([])


def test_scaler_fails_closed_on_insufficient_fit_window_support() -> None:
    with pytest.raises(InsufficientWindowError):
        StandardScaler([])
    with pytest.raises(InsufficientWindowError):
        StandardScaler([obs(1, 1, 1.0)])
    with pytest.raises(InsufficientWindowError):
        StandardScaler([obs(1, 1, 5.0), obs(2, 2, 5.0)])
    with pytest.raises(InsufficientWindowError):
        MinMaxScaler([obs(1, 1, 5.0), obs(2, 2, 5.0)])


# --------------------------------------------------------------------------- #
# No future normalization
# --------------------------------------------------------------------------- #


def test_scaler_fit_window_excludes_records_beyond_the_declared_window() -> None:
    fit = [obs(1, 1, 1.0), obs(2, 2, 2.0), obs(3, 3, 6.0)]
    scaler = StandardScaler(fit)
    late = obs(4, 9, -500.0)
    outputs = scaler.transform([obs(5, 5, 2.0), late])
    # The late record is scaled by the declared statistics like any
    # other record — it never joins them.
    assert outputs[1].value == pytest.approx(
        (-500.0 - 3.0) / statistics.pstdev([1.0, 2.0, 6.0])
    )
    # Per-output timing: a record available by day 5 yields an output
    # available by day 5; only the late record's own output waits for
    # day 9.
    assert outputs[0].available_time == day(5)
    assert outputs[1].available_time == day(9)


def test_future_fit_scaler_outputs_are_timing_killed_by_the_frozen_validator() -> None:
    # The declared fit window contains a record not available until
    # day 9, so the fixed statistics are knowable only then.
    fit = [obs(1, 1, 1.0), obs(2, 2, 2.0), obs(3, 9, 6.0)]
    scaler = StandardScaler(fit)
    records = [obs(4, 4, 4.0), obs(5, 5, 5.0)]
    outputs = scaler.transform(records)
    for output in outputs:
        assert output.available_time == day(9)
    require_feature_lineage(fit + records, outputs)
    # A hand-stamped output dated at transform-input timing is future
    # normalization; the strict validator kills it.
    leaked = TimedObservation(
        "px|standard_scaled", day(4), day(5), outputs[0].value
    )
    with pytest.raises(FeatureLineageError):
        require_feature_lineage(fit + records, [leaked])


# --------------------------------------------------------------------------- #
# Lag: position shift with an availability floor
# --------------------------------------------------------------------------- #


def test_lag_shifts_positions_and_floors_availability_at_the_shifted_instant() -> None:
    daily = timedelta(days=1)
    records = [
        obs(3, 3, 30.0),
        obs(4, 8, 40.0),
    ]
    outputs = Lag(2, daily).transform(records)
    assert outputs == [
        # Shifted observation instant day 5; availability floored up
        # to the shifted instant.
        TimedObservation("px|lag[2]", day(5), day(5), 30.0),
        # Shifted observation instant day 6; the original day-8
        # availability is later and is preserved — lagging can delay
        # availability, never advance it.
        TimedObservation("px|lag[2]", day(6), day(8), 40.0),
    ]
    for record, output in zip(records, outputs):
        require_feature_lineage([record], [output])


def test_lag_of_same_instant_records_constructs_valid_observations() -> None:
    records = [obs(3, 3, 7.0), obs(4, 4, 9.0)]
    outputs = Lag(1, timedelta(days=1)).transform(records)
    assert outputs == [
        TimedObservation("px|lag[1]", day(4), day(4), 7.0),
        TimedObservation("px|lag[1]", day(5), day(5), 9.0),
    ]


# --------------------------------------------------------------------------- #
# Carry-forward: bounded and declared
# --------------------------------------------------------------------------- #


def test_carry_forward_emits_within_staleness_bound() -> None:
    records = [obs(1, 2, 7.5, series="macro.q")]
    transform = CarryForward(
        timedelta(days=3), [day(1), day(3), day(5), day(6)]
    )
    outputs = transform.transform(records)
    assert outputs == [
        # Day 3: staleness one day, within the three-day bound.
        TimedObservation("macro.q|carry_forward", day(1), day(3), 7.5),
        # Day 5: staleness exactly three days — the bound is inclusive.
        TimedObservation("macro.q|carry_forward", day(1), day(5), 7.5),
    ]
    require_feature_lineage(records, outputs)
    # The declared real-time surface re-stamps the carried value at the
    # reference instant instead of the phenomenon date.
    real_time = CarryForward(
        timedelta(days=1), [day(3)], real_time_surface=True
    ).transform(records)
    assert real_time == [
        TimedObservation("macro.q|carry_forward", day(3), day(3), 7.5)
    ]


def test_carry_forward_is_absent_beyond_bound_and_before_series_start() -> None:
    records = [obs(1, 2, 7.5, series="macro.q")]
    transform = CarryForward(timedelta(days=3), [day(1), day(6), day(20)])
    outputs = transform.transform(records)
    # Day 1: nothing is available yet (series start); days 6 and 20:
    # staleness four and eighteen days, beyond the bound. Absent means
    # absent — no record, no placeholder, no imputation.
    assert outputs == []


def test_carry_forward_provenance_records_the_declared_bound() -> None:
    transform = CarryForward(timedelta(days=3), [day(5), day(3)])
    provenance = transform.provenance
    assert provenance["transform"] == "CarryForward"
    assert provenance["max_staleness"] == timedelta(days=3)
    assert list(provenance["reference_instants"]) == [day(3), day(5)]
    assert provenance["real_time_surface"] is False


# --------------------------------------------------------------------------- #
# Declared lookbacks only
# --------------------------------------------------------------------------- #


def test_lookback_lengths_must_be_declared_explicitly() -> None:
    with pytest.raises(TypeError):
        RollingMean()
    with pytest.raises(TypeError):
        Lag(1)
    with pytest.raises(TypeError):
        CarryForward(timedelta(days=1))
    declared_parameters = [
        (RollingMean, "window"),
        (RollingVolatility, "window"),
        (Lag, "periods"),
        (Lag, "frequency"),
        (CarryForward, "max_staleness"),
        (CarryForward, "reference_instants"),
        (StandardScaler, "fit_records"),
        (MinMaxScaler, "fit_records"),
    ]
    for owner, name in declared_parameters:
        parameter = inspect.signature(owner.__init__).parameters[name]
        assert parameter.default is inspect.Parameter.empty
    for malformed in (0, -1, 2.5, "3", True):
        with pytest.raises(WindowDeclarationError):
            RollingMean(malformed)
        with pytest.raises(WindowDeclarationError):
            RollingVolatility(malformed)
        with pytest.raises(WindowDeclarationError):
            Lag(malformed, timedelta(days=1))
    for bad_frequency in (timedelta(0), timedelta(days=-1)):
        with pytest.raises(WindowDeclarationError):
            Lag(1, bad_frequency)
    with pytest.raises(WindowDeclarationError):
        CarryForward(timedelta(days=-1), [day(3)])


# --------------------------------------------------------------------------- #
# Foreign-transform stand-in behind the fixed protocol
# --------------------------------------------------------------------------- #


class _ForeignDeviationEngine:
    """A dependency-free stand-in for a third-party transform."""

    version = "0.0.0-standin"

    def __init__(self) -> None:
        self._center = 0.0

    def fit(self, values: list[float]) -> None:
        self._center = statistics.fmean(values)

    def apply(self, values: list[float]) -> list[float]:
        return [value - self._center for value in values]


class _ForeignDeviationWrapper:
    """A researcher-side wrapper adapting the stand-in to the protocol."""

    def __init__(self, engine: _ForeignDeviationEngine, fit_records) -> None:
        series = {record.series_id for record in fit_records}
        if len(series) > 1:
            raise AmbiguousObservationError(
                "the wrapper's declared fit window spans more than one "
                f"series — {sorted(series)!r} — so no single-series fit "
                "window is defined."
            )
        self._engine = engine
        self._fit_records = tuple(fit_records)
        self._fit_latest_available = max(
            record.available_time for record in self._fit_records
        )
        self._engine.fit([record.value for record in self._fit_records])
        self.provenance = {
            "transform": "foreign_deviation_wrapper",
            "wrapped_implementation": type(engine).__name__,
            "wrapped_version": engine.version,
            "fit_record_count": len(self._fit_records),
        }

    def transform(self, observations):
        records = tuple(observations)
        values = self._engine.apply([record.value for record in records])
        outputs = []
        for record, value in zip(records, values):
            if record.available_time >= self._fit_latest_available:
                available = record.available_time
            else:
                available = self._fit_latest_available
            outputs.append(
                TimedObservation(
                    f"{record.series_id}|deviation",
                    record.observation_time,
                    available,
                    value,
                )
            )
        return outputs


def test_dependency_free_foreign_transform_stand_in_wrapped_behind_the_protocol() -> None:
    fit = [obs(1, 1, 2.0), obs(2, 9, 4.0)]
    wrapper = _ForeignDeviationWrapper(_ForeignDeviationEngine(), fit)
    records = [obs(3, 3, 10.0), obs(4, 4, 6.0)]
    outputs = wrapper.transform(records)
    assert callable(wrapper.transform)
    # Values come from the wrapped implementation fitted on the
    # declared window (center 3.0 over the fit values).
    assert [output.value for output in outputs] == pytest.approx(
        [7.0, 3.0]
    )
    # The fit-window law binds the wrapper: its statistics are knowable
    # only at the fit window's latest availability (day 9).
    for output in outputs:
        assert output.available_time == day(9)
    require_feature_lineage(fit + records, outputs)
    # Wrapper provenance records the wrapped implementation's identity
    # and version.
    assert wrapper.provenance["wrapped_implementation"] == (
        "_ForeignDeviationEngine"
    )
    assert wrapper.provenance["wrapped_version"] == "0.0.0-standin"


# --------------------------------------------------------------------------- #
# Provenance round-trip and error taxonomy
# --------------------------------------------------------------------------- #


def test_transform_provenance_round_trips_declared_parameters() -> None:
    assert dict(RollingMean(4).provenance) == {
        "transform": "RollingMean",
        "window": 4,
    }
    assert dict(RollingVolatility(5).provenance) == {
        "transform": "RollingVolatility",
        "window": 5,
    }
    assert dict(Lag(2, timedelta(days=1)).provenance) == {
        "transform": "Lag",
        "periods": 2,
        "frequency": timedelta(days=1),
    }
    fit = [obs(1, 1, 1.0), obs(2, 2, 3.0)]
    assert dict(MinMaxScaler(fit).provenance) == {
        "transform": "MinMaxScaler",
        "fit_record_count": 2,
        "fit_window_first_available": day(1),
        "fit_window_last_available": day(2),
        "minimum": 1.0,
        "maximum": 3.0,
    }
    provenance = RollingMean(3).provenance
    with pytest.raises(TypeError):
        provenance["window"] = 5


def test_error_taxonomy_reuses_frozen_errors_and_owns_valueerror_subclasses() -> None:
    assert issubclass(WindowDeclarationError, ValueError)
    assert issubclass(InsufficientWindowError, ValueError)
    for error in (WindowDeclarationError, InsufficientWindowError):
        assert error not in FROZEN_CONTRACT_ERRORS
    assert len(FROZEN_CONTRACT_ERRORS) == 6
    # Mixed-series input is an identity failure: the fixed
    # observations error, reused — never re-defined here.
    mixed = [obs(1, 1, 1.0, series="a"), obs(2, 2, 2.0, series="b")]
    single_series_transforms = [
        RollingMean(1),
        RollingVolatility(1),
        Lag(1, timedelta(days=1)),
        CarryForward(timedelta(days=1), [day(3)]),
    ]
    for transform in single_series_transforms:
        with pytest.raises(AmbiguousObservationError):
            transform.transform(mixed)
    with pytest.raises(AmbiguousObservationError):
        StandardScaler(mixed)
    scaler = StandardScaler([obs(1, 1, 1.0), obs(2, 2, 3.0)])
    with pytest.raises(AmbiguousObservationError):
        scaler.transform(mixed)
    # Naive reference instants are rejected by the fixed timing error,
    # reused at this surface.
    with pytest.raises(NaiveTimestampError):
        CarryForward(
            timedelta(days=1),
            [datetime(2026, 1, 3)],  # noqa: DTZ001 — deliberately naive
        )
