"""Chronologically valid feature transforms over declared observations.

This module provides the feature transforms as plain callables behind
the ``FeatureTransform`` protocol (the protocol is static-only with a
single ``transform`` method; there is no registry and no feature
library — this module ships concrete transforms and never widens the
protocol surface):

- **L1 — window membership keys on availability.** Every windowed
  transform declares its window over records' ``available_time``; a
  record whose ``available_time`` is after the output's reference
  instant is never a window member, no matter when the underlying
  phenomenon was observed. Window definition over ``observation_time``
  alone would admit revised values that did not exist when the output
  feature is timestamped — the named leakage pattern this module
  exists to make structurally absent.
- **L2 — output timing is lineage-monotone by construction.** Every
  output's ``available_time`` is at or after the latest
  ``available_time`` among the records that output derives from, and
  the output's ``observation_time`` is the transform's declared
  reference instant (for rolling windows, the window end in
  availability order). Outputs are ``TimedObservation`` records that
  re-enter every observation law. This module never re-implements the
  lineage law: validation is the frozen exported
  ``require_feature_lineage`` validator's job, composed by users and
  tests.
- **L3 — explicit fitting windows.** Scaling transforms declare their
  fitting window as the exact sequence of admitted records the
  statistics are computed over; the window's extent is recorded in
  provenance, and ``transform`` applies the frozen statistics — never
  recomputing them, never absorbing transform-time records into them.
  The fit-window cutoff is declared and recorded, in the same
  admission vocabulary later milestones compose (one fit-cutoff
  concept, no transform-local second clock).
- **L4 — insufficient windows fail closed.** Fewer records than a
  declared window requires — or a scaler fit window with no spread —
  raises ``InsufficientWindowError``, this module's own
  ``ValueError``. Never NaN emission, never partial-window output,
  never silent imputation: a feature either has full, knowable
  support or does not exist.
- **L5 — no future normalization.** Scaler statistics are computed
  only over the declared fitting window's past-available records, and
  every scaled output's ``available_time`` is floored at the fit
  window's latest availability: statistics that include a record not
  available until some instant are themselves knowable only then, and
  no output dated earlier can carry them.
- **L6 — lag is a position shift with an availability floor.** The
  lagged value keeps the input value; the output
  ``observation_time`` is the input's shifted by the declared number
  of periods of the series' own declared frequency; the output
  ``available_time`` is the later of the original availability and
  the shifted observation instant — lagging can delay availability,
  never advance it, and the floor keeps every output constructible
  under the frozen chronology law
  (``available_time >= observation_time``).
- **L7 — carry-forward is bounded and declared.** At each declared
  reference instant the carried value is the most recent record
  available by then, provided its staleness is within the declared
  bound; beyond the bound — and before the series starts — the
  outcome is absence (no record), never a placeholder. An optional
  real-time surface re-stamps the carried value at the reference
  instant instead of the phenomenon date; the choice is declared and
  recorded in provenance.
- **L9 — no default lookbacks.** Every lookback length, lag period,
  frequency, staleness bound, and fitting window is an explicit
  required constructor parameter recorded in provenance; none is
  ever tuned, defaulted, or selected by validation performance, and
  no indicator is named for a paper.
- **L10 — error taxonomy.** Structural declaration failures
  (malformed window, bad frequency, negative staleness) raise this
  module's own ``ValueError`` subclasses; timing, identity, and
  lineage failures reuse the frozen contract errors with their fixed
  module ownership (``NaiveTimestampError`` through the frozen
  normalizer, ``AmbiguousObservationError`` for mixed-series input).
  ``portlearn/__init__.py`` and ``FROZEN_CONTRACT_ERRORS`` are
  untouched.
- **L11 — wrappers are researcher-side.** This module ships no
  third-party wrapper: a researcher-side callable adapting a foreign
  implementation to the frozen protocol grants no exemption from any
  law, and the contract tests exercise the pattern with a
  dependency-free stand-in. Return/change transforms are deferred by
  decision and are not present here.

Every transform is single-series (cross-series features are
researcher compositions) and stdlib-only: pure functions over record
sequences — same records and declared parameters, same outputs.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from statistics import fmean, pstdev
from types import MappingProxyType
from typing import Any

from .observations import AmbiguousObservationError, TimedObservation
from .timing import instant_key, to_instant

__all__ = [
    "CarryForward",
    "InsufficientWindowError",
    "Lag",
    "MinMaxScaler",
    "RollingMean",
    "RollingVolatility",
    "StandardScaler",
    "WindowDeclarationError",
]


# --------------------------------------------------------------------------- #
# Fail-closed error taxonomy — the module-owned structural arm (L10)
# --------------------------------------------------------------------------- #


class WindowDeclarationError(ValueError):
    """A transform's declared window, lag, or bound is malformed.

    Raised for a non-integral or non-positive lookback length, a
    non-positive lag period, a lag frequency without strictly positive
    length, or a negative carry-forward staleness bound. Declaration
    parameters are explicit by law (no default lookbacks), so a value
    that names no usable window fails closed rather than defaulting.
    """


class InsufficientWindowError(ValueError):
    """A declared window lacks the records it requires (fail closed).

    Fewer records than the window requires — or a scaler fit window
    carrying no spread — rejects with this error rather than emitting
    NaN, a partial-window statistic, or an imputed value: a feature
    either has full, knowable support or does not exist.
    """


# --------------------------------------------------------------------------- #
# Shared declaration and admission helpers (private)
# --------------------------------------------------------------------------- #


def _require_count(value: Any, name: str) -> int:
    """Validate an explicit integral count parameter (lookback/periods)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise WindowDeclarationError(
            f"{name} must be an explicit positive integer count of "
            f"positions or records — a lookback is never defaulted or "
            f"inferred; got {type(value).__name__}: {value!r}. Declare "
            "the exact length so the window is auditable."
        )
    if value < 1:
        raise WindowDeclarationError(
            f"{name} must be a positive integer — a window of {value!r} "
            "members is no window at all; got "
            f"{type(value).__name__}: {value!r}. Declare the exact "
            "length so the window is auditable."
        )
    return value


def _require_positive_timedelta(value: Any, name: str) -> timedelta:
    """Validate a strictly positive duration (the lag frequency)."""
    if not isinstance(value, timedelta):
        raise WindowDeclarationError(
            f"{name} must be a datetime.timedelta declaring the series' "
            f"own frequency — the position-shift unit; got "
            f"{type(value).__name__}: {value!r}."
        )
    if value <= timedelta(0):
        raise WindowDeclarationError(
            f"{name} must have strictly positive length, but got "
            f"{value!r}: a zero or negative frequency shifts positions "
            "nowhere, so no lagged observation can be constructed."
        )
    return value


def _require_non_negative_timedelta(value: Any, name: str) -> timedelta:
    """Validate a non-negative duration (the staleness bound)."""
    if not isinstance(value, timedelta):
        raise WindowDeclarationError(
            f"{name} must be a datetime.timedelta declaring the maximum "
            f"staleness bound; got {type(value).__name__}: {value!r}."
        )
    if value < timedelta(0):
        raise WindowDeclarationError(
            f"{name} must not be negative, but got {value!r}: a bound "
            "admitting records from before they existed names no "
            "carry-forward policy."
        )
    return value


def _materialize(observations: Iterable[TimedObservation]) -> list[TimedObservation]:
    """Materialize input and enforce the single-series law (fail closed)."""
    records = list(observations)
    series = {record.series_id for record in records}
    if len(series) > 1:
        named = sorted(series)
        raise AmbiguousObservationError(
            "ambiguous observation input: the records span more than one "
            f"series — {named!r} — but transforms are single-series "
            "operations, so no window, lag, or statistic is defined over "
            "mixed input. Compose cross-series features researcher-side."
        )
    return records


def _availability_sorted(records: list[TimedObservation]) -> list[TimedObservation]:
    """Order records by availability instant (the L1 comparison basis).

    The sort key is the frozen UTC normalizer, so fold ambiguity and
    mixed zones never reorder records; the sort is stable, so records
    sharing one availability instant keep submission order.
    """
    return sorted(
        records, key=lambda record: instant_key(record.available_time)
    )


def _require_window_support(
    records: list[TimedObservation], window: int, transform_name: str
) -> None:
    """Fail closed when fewer records exist than the window requires."""
    if len(records) < window:
        raise InsufficientWindowError(
            f"insufficient window support: {transform_name} declares a "
            f"window of {window} records but the input carries "
            f"{len(records)} — a windowed feature either has full, "
            "knowable support or does not exist, so no partial-window "
            "statistic, NaN, or imputed value is emitted."
        )


def _scaled_output(
    series_id: str,
    suffix: str,
    record: TimedObservation,
    value: float,
    fit_latest_available: datetime,
    fit_latest_instant: datetime,
) -> TimedObservation:
    """One scaled record: value by frozen statistics, timing floored at fit.

    The output keeps the record's own ``observation_time``; its
    ``available_time`` is the later of the record's availability and
    the fit window's latest availability (L5): statistics that saw a
    record not available until some instant are knowable only then.
    """
    if instant_key(record.available_time) >= fit_latest_instant:
        available = record.available_time
    else:
        available = fit_latest_available
    return TimedObservation(
        f"{series_id}{suffix}", record.observation_time, available, value
    )


def _fit_window_extent(
    records: list[TimedObservation],
) -> tuple[datetime, datetime]:
    """The declared fit window's availability extent (original datetimes)."""
    ordered = _availability_sorted(records)
    return (
        ordered[0].available_time,
        ordered[-1].available_time,
    )


def _require_spread(values: list[float], transform_name: str) -> None:
    """Fail closed when the fit window carries no spread (fail-closed numerics)."""
    if len(values) < 2:
        raise InsufficientWindowError(
            f"insufficient window support: {transform_name} declares a "
            f"fitting window of at least two records, but the declared "
            f"window carries {len(values)} — a statistic over a single "
            "record names no spread, so scaling is undefined and the "
            "declaration rejects fail-closed rather than emitting "
            "degenerate output."
        )
    if transform_name == "StandardScaler" and pstdev(values) == 0.0:
        raise InsufficientWindowError(
            "insufficient window support: StandardScaler's declared "
            "fitting window carries no spread (every value equal), so "
            "the standard deviation is zero and scaling would divide "
            "by it — the declaration rejects fail-closed rather than "
            "emitting NaN or infinite values."
        )
    if transform_name == "MinMaxScaler" and min(values) == max(values):
        raise InsufficientWindowError(
            "insufficient window support: MinMaxScaler's declared "
            "fitting window carries no range (every value equal), so "
            "scaling would divide by a zero range — the declaration "
            "rejects fail-closed rather than emitting NaN or infinite "
            "values."
        )


# --------------------------------------------------------------------------- #
# Rolling window transforms (L1/L2/L4)
# --------------------------------------------------------------------------- #


class RollingMean:
    """The arithmetic mean over the last ``window`` available records.

    Window membership keys on ``available_time`` (L1): records are
    ordered by availability, and each output derives from the ``window``
    records latest available at the output's reference instant — the
    window end in availability order, which is also the output's
    ``observation_time`` and ``available_time`` (L2). Fewer records
    than the window requires fail closed (L4).
    """

    def __init__(self, window: int) -> None:
        self._window = _require_count(window, "window")
        self._provenance = {
            "transform": "RollingMean",
            "window": self._window,
        }

    @property
    def provenance(self) -> MappingProxyType:
        """The declared parameters reconstructing every output."""
        return MappingProxyType(self._provenance)

    def transform(
        self, observations: Iterable[TimedObservation]
    ) -> list[TimedObservation]:
        """Derive rolling-mean feature observations from input records."""
        records = _materialize(observations)
        _require_window_support(records, self._window, "RollingMean")
        ordered = _availability_sorted(records)
        outputs: list[TimedObservation] = []
        series_id = records[0].series_id
        output_series = f"{series_id}|rolling_mean[{self._window}]"
        for end in range(self._window - 1, len(ordered)):
            members = ordered[end - self._window + 1 : end + 1]
            reference = members[-1].available_time
            outputs.append(
                TimedObservation(
                    output_series,
                    reference,
                    reference,
                    fmean([member.value for member in members]),
                )
            )
        return outputs


class RollingVolatility:
    """The population standard deviation over the last ``window`` available records.

    The same availability-keyed window law as ``RollingMean`` (L1/L2),
    with the population standard deviation as the declared volatility
    estimator over available members; alternative estimators are
    deferred by decision, each requiring its own leakage audit.
    """

    def __init__(self, window: int) -> None:
        self._window = _require_count(window, "window")
        self._provenance = {
            "transform": "RollingVolatility",
            "window": self._window,
        }

    @property
    def provenance(self) -> MappingProxyType:
        """The declared parameters reconstructing every output."""
        return MappingProxyType(self._provenance)

    def transform(
        self, observations: Iterable[TimedObservation]
    ) -> list[TimedObservation]:
        """Derive rolling-volatility feature observations from input records."""
        records = _materialize(observations)
        _require_window_support(records, self._window, "RollingVolatility")
        ordered = _availability_sorted(records)
        outputs: list[TimedObservation] = []
        series_id = records[0].series_id
        output_series = f"{series_id}|rolling_volatility[{self._window}]"
        for end in range(self._window - 1, len(ordered)):
            members = ordered[end - self._window + 1 : end + 1]
            reference = members[-1].available_time
            outputs.append(
                TimedObservation(
                    output_series,
                    reference,
                    reference,
                    pstdev([member.value for member in members]),
                )
            )
        return outputs


# --------------------------------------------------------------------------- #
# Lag (L6)
# --------------------------------------------------------------------------- #


class Lag:
    """A position shift within the series' own declared frequency.

    Each output keeps its input's value; ``observation_time`` is the
    input's shifted by ``periods`` positions of ``frequency``; and
    ``available_time`` is the later of the original availability and
    the shifted observation instant — lagging can delay availability,
    never advance it, and the floor keeps every output constructible
    under the frozen chronology law
    (``available_time >= observation_time``).
    """

    def __init__(self, periods: int, frequency: timedelta) -> None:
        self._periods = _require_count(periods, "periods")
        self._frequency = _require_positive_timedelta(frequency, "frequency")
        self._provenance = {
            "transform": "Lag",
            "periods": self._periods,
            "frequency": self._frequency,
        }

    @property
    def provenance(self) -> MappingProxyType:
        """The declared parameters reconstructing every output."""
        return MappingProxyType(self._provenance)

    def transform(
        self, observations: Iterable[TimedObservation]
    ) -> list[TimedObservation]:
        """Derive lagged feature observations from input records."""
        records = _materialize(observations)
        shift = self._periods * self._frequency
        outputs: list[TimedObservation] = []
        for record in records:
            shifted_observation = record.observation_time + shift
            # The availability floor: the later of the original
            # availability and the shifted observation instant,
            # compared as normalized instants, stored as the original
            # datetime objects.
            if (
                instant_key(record.available_time)
                >= instant_key(shifted_observation)
            ):
                available = record.available_time
            else:
                available = shifted_observation
            outputs.append(
                TimedObservation(
                    f"{record.series_id}|lag[{self._periods}]",
                    shifted_observation,
                    available,
                    record.value,
                )
            )
        return outputs


# --------------------------------------------------------------------------- #
# Carry-forward (L7)
# --------------------------------------------------------------------------- #


class CarryForward:
    """The most recent available value, bounded by declared staleness.

    At each declared reference instant, the carried value is the most
    recent record with ``available_time`` at or before the instant,
    provided the record's staleness (reference instant minus its
    availability) is within the declared bound; otherwise the outcome
    is absence — no record, no placeholder, no imputation. By default
    the output keeps the carried record's own ``observation_time``
    (the phenomenon date); with ``real_time_surface`` declared, the
    output is re-stamped at the reference instant instead. Every
    output's ``available_time`` is the reference instant, so lineage
    monotonicity holds by construction (L2).
    """

    def __init__(
        self,
        max_staleness: timedelta,
        reference_instants: Iterable[datetime],
        real_time_surface: bool = False,
    ) -> None:
        self._max_staleness = _require_non_negative_timedelta(
            max_staleness, "max_staleness"
        )
        # Reference instants are declared, validated by the frozen
        # normalizer (naive inputs reject fail-closed), and recorded in
        # instant order — the declared surface the transform answers at.
        self._reference_instants = tuple(
            sorted(
                reference_instants,
                key=lambda instant: instant_key(instant, "reference instant"),
            )
        )
        for instant in self._reference_instants:
            to_instant(instant, "reference instant")
        self._real_time_surface = real_time_surface
        self._provenance = {
            "transform": "CarryForward",
            "max_staleness": self._max_staleness,
            "reference_instants": self._reference_instants,
            "real_time_surface": self._real_time_surface,
        }

    @property
    def provenance(self) -> MappingProxyType:
        """The declared parameters reconstructing every output."""
        return MappingProxyType(self._provenance)

    def transform(
        self, observations: Iterable[TimedObservation]
    ) -> list[TimedObservation]:
        """Derive carried-forward feature observations at the declared instants."""
        records = _materialize(observations)
        ordered = _availability_sorted(records)
        outputs: list[TimedObservation] = []
        series_id = records[0].series_id if records else ""
        output_series = f"{series_id}|carry_forward"
        for reference in self._reference_instants:
            reference_instant = to_instant(reference, "reference instant")
            visible = [
                record
                for record in ordered
                if instant_key(record.available_time) <= reference_instant
            ]
            if not visible:
                # Before the series starts, absence is the outcome —
                # never an error, never a placeholder.
                continue
            latest = visible[-1]
            staleness = reference_instant - instant_key(latest.available_time)
            if staleness > self._max_staleness:
                # Beyond the declared bound, absence again — no record.
                continue
            observation_time = (
                reference if self._real_time_surface else latest.observation_time
            )
            outputs.append(
                TimedObservation(
                    output_series,
                    observation_time,
                    reference,
                    latest.value,
                )
            )
        return outputs


# --------------------------------------------------------------------------- #
# Scaling transforms (L3/L5)
# --------------------------------------------------------------------------- #


class StandardScaler:
    """Standardization by statistics frozen over a declared fit window.

    The fitting window is the exact sequence of admitted records given
    at construction; the mean and population standard deviation are
    computed over it once, recorded in provenance, and frozen (L3).
    ``transform`` applies the frozen statistics record by record and
    floors every output's ``available_time`` at the fit window's
    latest availability (L5), so no output can carry statistics
    before they are knowable. A fit window with fewer than two
    records or no spread rejects fail-closed (L4).
    """

    def __init__(self, fit_records: Iterable[TimedObservation]) -> None:
        records = _materialize(fit_records)
        values = [record.value for record in records]
        _require_spread(values, "StandardScaler")
        self._fit_records = tuple(records)
        self._mean = fmean(values)
        self._standard_deviation = pstdev(values)
        first, last = _fit_window_extent(records)
        self._fit_latest_instant = instant_key(last)
        self._provenance = {
            "transform": "StandardScaler",
            "fit_record_count": len(records),
            "fit_window_first_available": first,
            "fit_window_last_available": last,
            "mean": self._mean,
            "standard_deviation": self._standard_deviation,
        }

    @property
    def provenance(self) -> MappingProxyType:
        """The declared fit window and the frozen statistics."""
        return MappingProxyType(self._provenance)

    def transform(
        self, observations: Iterable[TimedObservation]
    ) -> list[TimedObservation]:
        """Scale input records by the statistics frozen over the fit window."""
        records = _materialize(observations)
        last = self._provenance["fit_window_last_available"]
        return [
            _scaled_output(
                record.series_id,
                "|standard_scaled",
                record,
                (record.value - self._mean) / self._standard_deviation,
                last,
                self._fit_latest_instant,
            )
            for record in records
        ]


class MinMaxScaler:
    """Min-max normalization by statistics frozen over a declared fit window.

    The same fit-window law as ``StandardScaler`` (L3/L5/L4) with the
    minimum and maximum as the frozen statistics; a fit window with no
    range (every value equal) rejects fail-closed rather than dividing
    by zero.
    """

    def __init__(self, fit_records: Iterable[TimedObservation]) -> None:
        records = _materialize(fit_records)
        values = [record.value for record in records]
        _require_spread(values, "MinMaxScaler")
        self._fit_records = tuple(records)
        self._minimum = min(values)
        self._maximum = max(values)
        first, last = _fit_window_extent(records)
        self._fit_latest_instant = instant_key(last)
        self._provenance = {
            "transform": "MinMaxScaler",
            "fit_record_count": len(records),
            "fit_window_first_available": first,
            "fit_window_last_available": last,
            "minimum": self._minimum,
            "maximum": self._maximum,
        }

    @property
    def provenance(self) -> MappingProxyType:
        """The declared fit window and the frozen statistics."""
        return MappingProxyType(self._provenance)

    def transform(
        self, observations: Iterable[TimedObservation]
    ) -> list[TimedObservation]:
        """Scale input records by the statistics frozen over the fit window."""
        records = _materialize(observations)
        last = self._provenance["fit_window_last_available"]
        span = self._maximum - self._minimum
        return [
            _scaled_output(
                record.series_id,
                "|minmax_scaled",
                record,
                (record.value - self._minimum) / span,
                last,
                self._fit_latest_instant,
            )
            for record in records
        ]
