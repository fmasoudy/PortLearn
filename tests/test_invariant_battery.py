"""Contract-level financial invariant battery.

Permanent scenario nodes probing the contract timing, observation, and
interface predicates with finance-language scenarios — a decision
attempted on information not yet visible at its instant, a revision
invisible before its own availability, a derived feature declared
available before the data it derives from, a decision dated before
the forecast it consumes — plus the contract-blocking leakage
battery floors and the import law.

Discipline: nodes assert error **classes and laws only, never
error-message tails**; scenario inputs are fixed synthetic aware
instants on UTC (deterministic — no randomness, no real or
out-of-sample data, no performance claims); the battery imports
predicates only from the three contract modules — ``portlearn.timing``,
``portlearn.observations``, and ``portlearn.interfaces``.  The battery
does not duplicate the per-type unit suites: those pin structural
contracts per type, this file probes the same laws as cross-contract
finance-language scenarios.
"""

from __future__ import annotations

import ast
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from portlearn.interfaces import (
    Forecast,
    InformationSet,
    PortfolioDecision,
    require_feature_lineage,
    require_forecast_decision_compatible,
)
from portlearn.observations import (
    AmbiguousObservationError,
    FeatureLineageError,
    TimedObservation,
    require_lineage_monotone,
    vintage_as_of,
)
from portlearn.timing import (
    DecisionTiming,
    FutureInformationError,
    InvalidChronologyError,
    MissingAvailabilityError,
    NaiveTimestampError,
    ReturnRealizationPeriod,
    is_available_for_decision,
    require_available_for_decision,
)

BATTERY_PATH = Path(__file__).resolve()

# Fixed synthetic scenario instants (UTC): one quarter's macro reporting
# walk.  Deterministic constants — no seed, no randomness, no real data.
Q1_OBSERVATION = datetime(2026, 3, 31, 9, 0, tzinfo=UTC)  # phenomenon dated
Q1_FIRST_RELEASE = datetime(2026, 4, 1, 9, 0, tzinfo=UTC)  # first could be known
Q1_REVISION_RELEASE = datetime(2026, 5, 6, 9, 0, tzinfo=UTC)  # revised vintage
APRIL_DECISION = datetime(2026, 4, 30, 9, 0, tzinfo=UTC)
APRIL_EXECUTION = datetime(2026, 4, 30, 10, 0, tzinfo=UTC)
MAY_DECISION = datetime(2026, 5, 30, 9, 0, tzinfo=UTC)


def _cpi_first_release() -> TimedObservation:
    """The quarter's CPI figure as first published (available 2026-04-01)."""
    return TimedObservation(
        series_id="AUD_CPI_QOQ",
        observation_time=Q1_OBSERVATION,
        available_time=Q1_FIRST_RELEASE,
        value=0.6,
    )


def _cpi_revision(value: float = 0.8) -> TimedObservation:
    """The revised CPI vintage: same phenomenon date, later availability."""
    return TimedObservation(
        series_id="AUD_CPI_QOQ",
        observation_time=Q1_OBSERVATION,
        available_time=Q1_REVISION_RELEASE,
        value=value,
    )


def _unemployment_first_release() -> TimedObservation:
    """A second series published alongside CPI (distinct series-observation)."""
    return TimedObservation(
        series_id="AUD_UNEMP_QOQ",
        observation_time=Q1_OBSERVATION,
        available_time=Q1_FIRST_RELEASE,
        value=4.2,
    )


class _AvailabilityWithheld:
    """A synthetic source that cannot declare when its figure could first
    have been known (``available_time`` is ``None``)."""

    available_time: datetime | None = None


class _NaiveDatedForecast:
    """A duck-typed forecast carrying a naive origin instant: it cannot be
    placed on the single UTC timeline, so compatibility checking must reject
    it before any comparison is attempted."""

    decision_time: datetime = (
        datetime(2026, 5, 30, 9, 0)  # noqa: DTZ001 — naive instant by design
    )


# ---------------------------------------------------------------------------
# Admission law — lookahead in decisions (portlearn.timing)
# ---------------------------------------------------------------------------


def test_decision_on_information_not_yet_published_is_blocked() -> None:
    """A decision attempted on a revision that does not exist yet at the
    decision instant is look-ahead leakage and must reject with
    ``FutureInformationError``."""
    with pytest.raises(FutureInformationError):
        require_available_for_decision(_cpi_revision(), APRIL_DECISION)


def test_information_arriving_inside_the_decision_execution_gap_is_leakage() -> None:
    """Information first available after the decision but before execution
    satisfies no admission verdict: the gate is the decision instant only,
    so admitting it would be look-ahead leakage."""
    intraday_revision = TimedObservation(
        series_id="AUD_CPI_QOQ",
        observation_time=Q1_OBSERVATION,
        available_time=datetime(2026, 4, 30, 9, 30, tzinfo=UTC),
        value=0.9,
    )
    with pytest.raises(FutureInformationError):
        require_available_for_decision(intraday_revision, APRIL_DECISION)
    assert is_available_for_decision(intraday_revision, APRIL_DECISION) is False


def test_information_set_refuses_lookahead_records() -> None:
    """The admitted information set rejects a record whose availability is
    after the set's ``as_of`` — the set is exactly what could have been
    known at the decision, never gap information."""
    with pytest.raises(FutureInformationError):
        InformationSet([_cpi_revision()], APRIL_DECISION)


def test_naive_decision_instant_is_rejected_fail_closed() -> None:
    """A naive decision instant carries no UTC offset, so it cannot be
    placed on the single timeline: rejected with ``NaiveTimestampError``,
    never defaulted to a zone."""
    with pytest.raises(NaiveTimestampError):
        require_available_for_decision(
            _cpi_first_release(),
            datetime(  # noqa: DTZ001 — naive instant by design
                2026, 4, 30, 9, 0
            ),
        )


def test_calendar_date_decision_instant_is_rejected_fail_closed() -> None:
    """A bare calendar date is not an instant: no date-to-midnight coercion
    is ever performed, because that coercion is the classic daily-data
    look-ahead vector."""
    with pytest.raises(NaiveTimestampError):
        require_available_for_decision(_cpi_first_release(), date(2026, 4, 30))


def test_information_without_declared_availability_is_rejected() -> None:
    """A source unable to declare when its information first could have
    been known must fail closed with ``MissingAvailabilityError`` — never
    default to the observation instant or "immediately available"."""
    with pytest.raises(MissingAvailabilityError):
        require_available_for_decision(object(), APRIL_DECISION)


def test_none_availability_is_rejected_as_missing() -> None:
    """``available_time=None`` is a withheld availability declaration, not
    an absent one: the same mandatory-availability rejection applies."""
    with pytest.raises(MissingAvailabilityError):
        require_available_for_decision(_AvailabilityWithheld(), APRIL_DECISION)


# ---------------------------------------------------------------------------
# Chronology law (portlearn.timing)
# ---------------------------------------------------------------------------


def test_publication_may_not_precede_the_phenomenon_it_reports() -> None:
    """Information cannot be published before the phenomenon it reports:
    an observation declaring availability before its observation instant
    rejects with ``InvalidChronologyError``."""
    with pytest.raises(InvalidChronologyError):
        TimedObservation(
            series_id="AUD_CPI_QOQ",
            observation_time=Q1_OBSERVATION,
            available_time=Q1_OBSERVATION - timedelta(days=1),
            value=0.6,
        )


def test_execution_may_not_precede_decision() -> None:
    """A trade cannot execute before its decision is made: reversed
    decision/execution ordering rejects with ``InvalidChronologyError``."""
    with pytest.raises(InvalidChronologyError):
        DecisionTiming(
            decision_time=APRIL_DECISION,
            execution_time=APRIL_DECISION - timedelta(minutes=1),
            return_realization_period=ReturnRealizationPeriod(
                APRIL_EXECUTION, APRIL_EXECUTION + timedelta(hours=1)
            ),
        )


def test_outcomes_may_not_realize_before_execution() -> None:
    """Outcomes cannot start realizing before the trade executes:
    a realization period starting before the execution instant rejects
    with ``InvalidChronologyError``."""
    with pytest.raises(InvalidChronologyError):
        DecisionTiming(
            decision_time=APRIL_DECISION,
            execution_time=APRIL_EXECUTION,
            return_realization_period=ReturnRealizationPeriod(
                APRIL_EXECUTION - timedelta(minutes=1),
                APRIL_EXECUTION + timedelta(hours=1),
            ),
        )


def test_zero_length_or_reversed_realization_period_is_malformed() -> None:
    """The half-open realization period must have strictly positive
    length: zero-length and reversed bounds both reject with
    ``InvalidChronologyError``."""
    with pytest.raises(InvalidChronologyError):
        ReturnRealizationPeriod(APRIL_EXECUTION, APRIL_EXECUTION)
    with pytest.raises(InvalidChronologyError):
        ReturnRealizationPeriod(
            APRIL_EXECUTION + timedelta(hours=1), APRIL_EXECUTION
        )


def test_same_instant_decide_and_execute_is_admissible() -> None:
    """Equal adjacent instants are admissible at every ``≤`` boundary of
    the chronology law: a same-instant decide-and-execute event is a
    legitimate timing, not a violation."""
    timing = DecisionTiming(
        decision_time=APRIL_DECISION,
        execution_time=APRIL_DECISION,
        return_realization_period=ReturnRealizationPeriod(
            APRIL_DECISION, APRIL_DECISION + timedelta(hours=1)
        ),
    )
    assert timing.execution_time == timing.decision_time


# ---------------------------------------------------------------------------
# Vintage selection — ambiguous vintages (portlearn.observations)
# ---------------------------------------------------------------------------


def test_revised_value_is_invisible_before_its_own_availability() -> None:
    """At the April decision the visible CPI vintage is the first release:
    the May revision is a separate record that does not exist yet, so the
    point-in-time walk sees 0.6, never 0.8."""
    visible = vintage_as_of(
        [_cpi_first_release(), _cpi_revision()], APRIL_DECISION
    )
    assert visible is not None
    assert visible.available_time == Q1_FIRST_RELEASE
    assert visible.value == 0.6


def test_revision_is_visible_exactly_at_its_availability() -> None:
    """Availability exactly at the decision admits the revision: the
    inclusive boundary of the vintage law (latest availability at or
    before the decision wins)."""
    visible = vintage_as_of(
        [_cpi_first_release(), _cpi_revision()], Q1_REVISION_RELEASE
    )
    assert visible is not None
    assert visible.available_time == Q1_REVISION_RELEASE
    assert visible.value == 0.8


def test_series_with_nothing_visible_yet_yields_no_vintage() -> None:
    """Before any release the query returns "no vintage" — an explicit
    ``None`` verdict, never an error and never a silent coercion to the
    unreleased figure."""
    assert (
        vintage_as_of(
            [_cpi_first_release()], Q1_FIRST_RELEASE - timedelta(days=1)
        )
        is None
    )


def test_empty_vintage_query_is_not_an_error() -> None:
    """Empty input returns ``None``: no vintage is an explicit outcome,
    not ambiguity."""
    assert vintage_as_of([], APRIL_DECISION) is None


def test_duplicate_identity_records_reject_in_vintage_query() -> None:
    """Two records sharing the full identity triple reject with
    ``AmbiguousObservationError`` — the identity triple is a key with no
    last-write-wins and no value-equality exception."""
    with pytest.raises(AmbiguousObservationError):
        vintage_as_of([_cpi_first_release(), _cpi_first_release()], APRIL_DECISION)


def test_mixed_group_vintage_query_rejects() -> None:
    """A vintage query spanning two series-observations cannot select one
    without an illegitimate cross-group preference rule: it rejects with
    ``AmbiguousObservationError``."""
    with pytest.raises(AmbiguousObservationError):
        vintage_as_of(
            [_cpi_first_release(), _unemployment_first_release()], APRIL_DECISION
        )


# ---------------------------------------------------------------------------
# Information set composition — ambiguous vintages at admission
# ---------------------------------------------------------------------------


def test_information_set_refuses_two_vintages_of_one_observation() -> None:
    """The admitted set is given at most one vintage per
    series-observation: submitting a revision pair rejects with
    ``AmbiguousObservationError`` — the set never selects among vintages
    itself."""
    with pytest.raises(AmbiguousObservationError):
        InformationSet([_cpi_first_release(), _cpi_revision()], APRIL_DECISION)


def test_information_set_refuses_duplicate_identity_records() -> None:
    """Duplicate full-identity records reject at admission exactly as at
    vintage selection: no deduplication, no value-equality exception."""
    with pytest.raises(AmbiguousObservationError):
        InformationSet([_cpi_first_release(), _cpi_first_release()], APRIL_DECISION)


def test_empty_information_set_is_admissible() -> None:
    """When nothing is available at the decision instant the admitted set
    is empty — and empty is not ambiguous."""
    empty_set = InformationSet([], MAY_DECISION)
    assert tuple(empty_set) == ()
    assert empty_set.as_of == MAY_DECISION


def test_admitted_set_is_exactly_the_visible_submitted_history() -> None:
    """The set preserves the submitted visible records in submission order
    with their exact identities and values, pinned to its ``as_of``."""
    admitted = InformationSet(
        [_cpi_first_release(), _unemployment_first_release()], APRIL_DECISION
    )
    assert [(record.series_id, record.value) for record in admitted] == [
        ("AUD_CPI_QOQ", 0.6),
        ("AUD_UNEMP_QOQ", 4.2),
    ]
    assert admitted.as_of == APRIL_DECISION


# ---------------------------------------------------------------------------
# Feature lineage — non-monotone lineage
# ---------------------------------------------------------------------------


def test_feature_available_before_its_latest_input_is_leakage() -> None:
    """A derived feature declared available before the latest input it
    derives from would be visible before the data it derives from:
    ``FeatureLineageError``."""
    with pytest.raises(FeatureLineageError):
        require_lineage_monotone(Q1_FIRST_RELEASE, [Q1_REVISION_RELEASE])


def test_feature_with_empty_input_collection_has_no_availability() -> None:
    """A feature declaring no inputs has no defensible availability at
    all: ``FeatureLineageError``, never a default."""
    with pytest.raises(FeatureLineageError):
        require_lineage_monotone(Q1_FIRST_RELEASE, [])


def test_derived_features_may_not_outrun_their_inputs() -> None:
    """The interface-level lineage validator enforces the same law over
    whole derived observations: a feature observation available at its own
    observation instant, fed by inputs published later, rejects with
    ``FeatureLineageError``."""
    premature_feature = TimedObservation(
        series_id="FEAT_CPI_MOM",
        observation_time=Q1_OBSERVATION,
        available_time=Q1_OBSERVATION,
        value=0.7,
    )
    inputs = [_cpi_first_release(), _cpi_revision()]
    with pytest.raises(FeatureLineageError):
        require_feature_lineage(inputs, [premature_feature])


# ---------------------------------------------------------------------------
# Forecast/decision compatibility
# ---------------------------------------------------------------------------


def test_decision_may_not_precede_the_forecast_it_consumes() -> None:
    """A decision dated before the information it consumed is look-ahead
    leakage: ``require_forecast_decision_compatible`` rejects with
    ``InvalidChronologyError``."""
    forecast = Forecast(
        values={"AUD_CPI_QOQ": 0.7},
        target="expected_return",
        decision_time=MAY_DECISION,
        produced_by="battery.probe",
    )
    earlier_decision = PortfolioDecision(
        decision_time=APRIL_DECISION,
        execution_time=MAY_DECISION,
        target_weights={"AUD_CPI": 1.0},
    )
    with pytest.raises(InvalidChronologyError):
        require_forecast_decision_compatible(forecast, earlier_decision)


def test_deciding_at_or_after_the_forecast_origin_is_admissible() -> None:
    """Deciding exactly at the forecast origin, or strictly later on
    information admitted at the later instant, is admissible: the
    impossible ordering alone is forbidden."""
    forecast = Forecast(
        values={"AUD_CPI_QOQ": 0.7},
        target="expected_return",
        decision_time=APRIL_DECISION,
        produced_by="battery.probe",
    )
    same_instant = PortfolioDecision(
        decision_time=APRIL_DECISION,
        execution_time=APRIL_EXECUTION,
        target_weights={"AUD_CPI": 1.0},
    )
    later = PortfolioDecision(
        decision_time=MAY_DECISION,
        execution_time=MAY_DECISION + timedelta(hours=1),
        target_weights={"AUD_CPI": 1.0},
    )
    assert require_forecast_decision_compatible(forecast, same_instant) is None
    assert require_forecast_decision_compatible(forecast, later) is None


def test_naive_forecast_origin_rejects_before_any_comparison() -> None:
    """A naive forecast origin cannot be placed on the timeline, so the
    compatibility gate rejects it with ``NaiveTimestampError`` before any
    ordering comparison is attempted."""
    naive_forecast = _NaiveDatedForecast()
    decision = PortfolioDecision(
        decision_time=APRIL_DECISION,
        execution_time=APRIL_EXECUTION,
        target_weights={"AUD_CPI": 1.0},
    )
    with pytest.raises(NaiveTimestampError):
        require_forecast_decision_compatible(naive_forecast, decision)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Leakage battery floors — portlearn.leakage
# ---------------------------------------------------------------------------


def test_frozen_contract_error_closure_and_public_surfaces_are_exact() -> None:
    """Closure floor: ``FROZEN_CONTRACT_ERRORS`` equals exactly the six
    module-qualified classes imported from ``portlearn.timing`` /
    ``portlearn.observations`` — a battery can never expect a foreign
    error — and both ``portlearn.leakage`` and ``portlearn.manifest``
    expose exactly their pinned ``__all__`` public surfaces."""
    from portlearn import leakage, manifest

    expected_six = {
        FutureInformationError,
        AmbiguousObservationError,
        NaiveTimestampError,
        InvalidChronologyError,
        FeatureLineageError,
        MissingAvailabilityError,
    }
    assert len(leakage.FROZEN_CONTRACT_ERRORS) == 6
    assert set(leakage.FROZEN_CONTRACT_ERRORS) == expected_six
    for error_class in leakage.FROZEN_CONTRACT_ERRORS:
        assert error_class.__module__ in {
            "portlearn.timing",
            "portlearn.observations",
        }
    assert leakage.__all__ == [
        "LeakageCase",
        "LeakageFinding",
        "LeakageReport",
        "run_leakage_cases",
        "FROZEN_CONTRACT_ERRORS",
    ]
    assert manifest.__all__ == ["RunManifest"]


def test_leakage_case_admission_is_fail_closed() -> None:
    """Admission floor: ``LeakageCase`` rejects blank/non-string names,
    non-callable attempts, foreign expected errors, and non-class
    expected errors — all with the pinned built-in ``ValueError``."""

    def _naive_instant_probe() -> None:
        raise NaiveTimestampError

    from portlearn.leakage import LeakageCase

    frozen_error = NaiveTimestampError
    with pytest.raises(ValueError):
        LeakageCase("   ", _naive_instant_probe, frozen_error)
    with pytest.raises(ValueError):
        LeakageCase(7, _naive_instant_probe, frozen_error)
    with pytest.raises(ValueError):
        LeakageCase("non-callable-attempt", "not-callable", frozen_error)
    with pytest.raises(ValueError):
        LeakageCase("foreign-error", _naive_instant_probe, RuntimeError)
    with pytest.raises(ValueError):
        LeakageCase("non-class-error", _naive_instant_probe, "NaiveTimestampError")
    admitted = LeakageCase("naive-instant-probe", _naive_instant_probe, frozen_error)
    assert admitted.name == "naive-instant-probe"


def test_leakage_foreign_expected_error_message_is_public_wording() -> None:
    """A foreign expected error rejects with the public wording.

    The runtime message names PortLearn's supported contract error types;
    it must not carry internal process vocabulary that a public user
    cannot act on.
    """

    def _probe() -> None:
        raise NaiveTimestampError

    from portlearn.leakage import LeakageCase

    with pytest.raises(ValueError, match="supported contract error types") as caught:
        LeakageCase("foreign-error", _probe, RuntimeError)
    message = str(caught.value)
    assert (
        "expected_error must be one of PortLearn's supported contract "
        "error types"
    ) in message
    assert "closure" not in message.lower(), (
        "the runtime message must not carry closure-membership vocabulary"
    )


def test_manifest_empty_commands_message_is_public_wording() -> None:
    """An empty commands sequence rejects with the public wording.

    The runtime message states the schema requirement plainly; it must
    not editorialize about what an empty run proves.
    """
    from portlearn.manifest import RunManifest

    with pytest.raises(
        ValueError, match=r"must contain at least one command"
    ) as caught:
        RunManifest(
            run_id="empty-commands-probe",
            created_at=APRIL_DECISION,
            python_version="3.13.5",
            package_version="0.0.1.dev0",
            dependency_pins={"portlearn": "0.0.1.dev0"},
            commands=[],
        )
    message = str(caught.value)
    assert "RunManifest.commands must contain at least one command." in message
    assert "proves" not in message.lower(), (
        "the runtime message must not editorialize beyond the schema rule"
    )


def test_leakage_battery_is_total_and_reports_every_outcome() -> None:
    """Totality floor: ``run_leakage_cases`` executes every case, rejects
    an empty battery, and lands every outcome in the report as a finding —
    a wrong error class reports ``blocked is False`` with
    ``detail == "wrong error class"``; an unblocked leak reports
    ``error_type is None`` with ``detail == "leak was NOT blocked"``.
    The report is the evidence, never an exception."""

    def _blocked_lookahead() -> None:
        # The gate form blocks the revision leak; the boolean form below does not.
        require_available_for_decision(_cpi_revision(), APRIL_DECISION)

    def _deliberately_unblocked_leak() -> None:
        # A bare boolean query raises nothing, so this leak is NOT blocked.
        is_available_for_decision(_cpi_revision(), APRIL_DECISION)

    from portlearn.leakage import LeakageCase, run_leakage_cases

    cases = [
        LeakageCase("lookahead-admission", _blocked_lookahead, FutureInformationError),
        LeakageCase(
            "wrong-class-chronology", _blocked_lookahead, InvalidChronologyError
        ),
        LeakageCase(
            "deliberately-unblocked-leak",
            _deliberately_unblocked_leak,
            FutureInformationError,
        ),
    ]
    with pytest.raises(ValueError):
        run_leakage_cases([])
    report = run_leakage_cases(cases)
    by_name = {finding.case.name: finding for finding in report.findings}
    assert set(by_name) == {
        "lookahead-admission",
        "wrong-class-chronology",
        "deliberately-unblocked-leak",
    }
    blocked = by_name["lookahead-admission"]
    assert blocked.blocked is True
    assert blocked.error_type is FutureInformationError
    wrong_class = by_name["wrong-class-chronology"]
    assert wrong_class.blocked is False
    assert wrong_class.error_type is FutureInformationError
    assert wrong_class.detail == "wrong error class"
    unblocked = by_name["deliberately-unblocked-leak"]
    assert unblocked.blocked is False
    assert unblocked.error_type is None
    assert unblocked.detail == "leak was NOT blocked"


def test_leakage_report_summary_derives_only_from_findings() -> None:
    """Summary floor: ``all_blocked`` and ``summary`` derive only from the
    findings, as ``"{n}/{total} attempted leaks blocked"``."""
    from portlearn.leakage import LeakageCase, LeakageFinding, LeakageReport

    case = LeakageCase(
        "summary-proof", lambda: None, FutureInformationError
    )
    mixed = LeakageReport(
        findings=(
            LeakageFinding(case, True, FutureInformationError, "blocked"),
            LeakageFinding(case, False, None, "leak was NOT blocked"),
        )
    )
    assert mixed.all_blocked is False
    assert mixed.summary == "1/2 attempted leaks blocked"
    fully_blocked = LeakageReport(
        findings=(LeakageFinding(case, True, FutureInformationError, "blocked"),)
    )
    assert fully_blocked.all_blocked is True
    assert fully_blocked.summary == "1/1 attempted leaks blocked"


# ---------------------------------------------------------------------------
# Import law
# ---------------------------------------------------------------------------


def _module_level_portlearn_targets(path: Path) -> set[str]:
    """Every ``portlearn`` module imported at module level by ``path``."""
    targets: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            if node.module:
                targets.add(prefix + node.module)
            else:
                targets.update(prefix + alias.name for alias in node.names)
    return {
        target
        for target in targets
        if target == "portlearn" or target.startswith("portlearn.")
    }


def test_battery_imports_predicates_only_from_the_frozen_modules() -> None:
    """Import law: the battery imports predicates only from the three
    contract modules — ``portlearn.timing``, ``portlearn.observations``,
    ``portlearn.interfaces`` — and from all three of them; the leakage
    battery under test is exercised through function-local imports of
    ``portlearn.leakage`` and ``portlearn.manifest``, which import no
    predicates here."""
    allowed = {
        "portlearn.timing",
        "portlearn.observations",
        "portlearn.interfaces",
    }
    targets = _module_level_portlearn_targets(BATTERY_PATH)
    assert targets == allowed
