"""Regression guard: the sole aware-instant validator lives in portlearn.timing.

``_require_aware_instant`` must be defined exactly once — in
``src/portlearn/timing.py`` — and both ``observations.py`` and
``interfaces.py`` must reuse that exact function by import, never
re-implement it.  The architectural law: shared invariant validation
has a single implementation, reuse is by import/composition, and
executable tests prevent copy-pasted divergent validators.

Non-vacuous: with three copy-pasted definitions the sole-ownership,
import-reuse, and runtime-identity assertions fail.  timing.py's
rejection wording is the single canonical wording, which these tests
freeze.
"""

from __future__ import annotations

import ast
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from portlearn.interfaces import (
    Forecast,
    InformationSet,
    PortfolioDecision,
    require_forecast_decision_compatible,
)
from portlearn.observations import (
    TimedObservation,
    require_lineage_monotone,
    vintage_as_of,
)
from portlearn.timing import (
    DecisionTiming,
    NaiveTimestampError,
    ReturnRealizationPeriod,
    is_available_for_decision,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "portlearn"
TIMING_PATH = PACKAGE_ROOT / "timing.py"
REUSING_MODULES = (
    ("observations.py", PACKAGE_ROOT / "observations.py"),
    ("interfaces.py", PACKAGE_ROOT / "interfaces.py"),
)

VALIDATOR_NAME = "_require_aware_instant"

AWARE_INSTANT = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)
AWARE_LATER = datetime(2026, 3, 2, 9, 30, tzinfo=UTC)
NAIVE_INSTANT = datetime(2026, 3, 2, 9, 0)  # noqa: DTZ001  # intentional naive instant
CALENDAR_DATE = date(2026, 3, 2)
NON_INSTANT = 7

# timing.py's invalid-input wording is the single canonical wording:
# every calendar-date rejection across the package carries this tail.
CANONICAL_DATE_TAIL = (
    "declaring when the information first could have been known."
)

MALFORMED_INSTANTS = [
    pytest.param(NAIVE_INSTANT, id="naive-datetime"),
    pytest.param(CALENDAR_DATE, id="calendar-date"),
    pytest.param(NON_INSTANT, id="non-instant-input"),
]


def _bindings() -> tuple:
    """The validator as bound in each of the three modules."""
    import portlearn.interfaces as interfaces_module
    import portlearn.observations as observations_module
    import portlearn.timing as timing_module

    return (
        timing_module._require_aware_instant,
        observations_module._require_aware_instant,
        interfaces_module._require_aware_instant,
    )


# ---------------------------------------------------------------------------
# Sole implementation and reuse-by-import
# ---------------------------------------------------------------------------


def test_require_aware_instant_has_sole_definition_in_timing() -> None:
    """Exactly one function definition, and it is timing.py's."""
    definitions: list[str] = []
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                )
                and node.name == VALIDATOR_NAME
            ):
                definitions.append(path.name)
    assert definitions == ["timing.py"], (
        f"{VALIDATOR_NAME} must have exactly one function definition "
        "under src/portlearn, in timing.py — a single implementation "
        "for shared invariant validation; "
        f"found definitions in {definitions!r}"
    )


def test_reusing_modules_import_the_timing_validator() -> None:
    """observations.py and interfaces.py reuse by import, not re-impl."""
    for filename, path in REUSING_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = any(
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module == "timing"
            and any(alias.name == VALIDATOR_NAME for alias in node.names)
            for node in ast.walk(tree)
        )
        assert imported, (
            f"{filename} must reuse the timing-owned aware-instant "
            f"validator by importing {VALIDATOR_NAME} from .timing; "
            "re-implementing the shared invariant is forbidden"
        )


def test_validator_runtime_identity_across_all_three_modules() -> None:
    """``timing._require_aware_instant is`` each module's binding."""
    timing_validator, observations_validator, interfaces_validator = (
        _bindings()
    )
    assert TIMING_PATH.is_file()
    assert timing_validator is observations_validator, (
        "portlearn.observations must bind the timing-owned validator "
        "object itself, not a module-local copy"
    )
    assert timing_validator is interfaces_validator, (
        "portlearn.interfaces must bind the timing-owned validator "
        "object itself, not a module-local copy"
    )


def test_aware_instant_round_trips_from_every_binding() -> None:
    """An aware datetime is returned as the same object, everywhere."""
    for validator in _bindings():
        assert validator(AWARE_INSTANT, "checked_field") is AWARE_INSTANT


# ---------------------------------------------------------------------------
# Rejection behavior preserved through every module's public surfaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("malformed", MALFORMED_INSTANTS)
def test_timing_public_surfaces_reject_malformed_instants(
    malformed: object,
) -> None:
    period = ReturnRealizationPeriod(AWARE_INSTANT, AWARE_LATER)
    item = SimpleNamespace(available_time=AWARE_INSTANT)
    with pytest.raises(NaiveTimestampError):
        ReturnRealizationPeriod(malformed, AWARE_LATER)
    with pytest.raises(NaiveTimestampError):
        ReturnRealizationPeriod(AWARE_INSTANT, malformed)
    with pytest.raises(NaiveTimestampError):
        DecisionTiming(malformed, AWARE_LATER, period)
    with pytest.raises(NaiveTimestampError):
        DecisionTiming(AWARE_INSTANT, malformed, period)
    with pytest.raises(NaiveTimestampError):
        is_available_for_decision(item, malformed)


@pytest.mark.parametrize("malformed", MALFORMED_INSTANTS)
def test_observations_public_surfaces_reject_malformed_instants(
    malformed: object,
) -> None:
    record = TimedObservation(
        "MACRO.GDPQ.REAL", AWARE_INSTANT, AWARE_INSTANT, 1.0
    )
    with pytest.raises(NaiveTimestampError):
        TimedObservation("MACRO.GDPQ.REAL", malformed, AWARE_INSTANT, 1.0)
    with pytest.raises(NaiveTimestampError):
        TimedObservation("MACRO.GDPQ.REAL", AWARE_INSTANT, malformed, 1.0)
    with pytest.raises(NaiveTimestampError):
        vintage_as_of([record], malformed)
    with pytest.raises(NaiveTimestampError):
        require_lineage_monotone(malformed, [AWARE_INSTANT])
    with pytest.raises(NaiveTimestampError):
        require_lineage_monotone(AWARE_INSTANT, [malformed])


@pytest.mark.parametrize("malformed", MALFORMED_INSTANTS)
def test_interfaces_public_surfaces_reject_malformed_instants(
    malformed: object,
) -> None:
    weights = {"EQUITY.ASX.WOW": 0.5, "EQUITY.ASX.CBA": 0.5}
    dated_forecast = SimpleNamespace(decision_time=AWARE_INSTANT)
    dated_decision = SimpleNamespace(decision_time=AWARE_INSTANT)
    with pytest.raises(NaiveTimestampError):
        InformationSet([], malformed)
    with pytest.raises(NaiveTimestampError):
        Forecast(
            dict(weights), "expected_return", malformed, "synthetic-mean"
        )
    with pytest.raises(NaiveTimestampError):
        PortfolioDecision(malformed, AWARE_LATER, dict(weights))
    with pytest.raises(NaiveTimestampError):
        PortfolioDecision(AWARE_INSTANT, malformed, dict(weights))
    with pytest.raises(NaiveTimestampError):
        require_forecast_decision_compatible(
            SimpleNamespace(decision_time=malformed), dated_decision
        )
    with pytest.raises(NaiveTimestampError):
        require_forecast_decision_compatible(
            dated_forecast, SimpleNamespace(decision_time=malformed)
        )


# ---------------------------------------------------------------------------
# Canonical invalid-input wording (timing.py's), identical on every path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "malformed",
    MALFORMED_INSTANTS,
)
def test_rejection_messages_are_identical_across_all_three_bindings(
    malformed: object,
) -> None:
    messages = set()
    for validator in _bindings():
        with pytest.raises(NaiveTimestampError) as excinfo:
            validator(malformed, "checked_field")
        messages.add(str(excinfo.value))
    assert len(messages) == 1, (
        "the aware-instant validator's rejection wording must be "
        "identical across timing, observations, and interfaces — "
        f"one canonical wording owned by timing.py; got {len(messages)} "
        "distinct messages"
    )


def test_calendar_date_rejection_carries_the_canonical_timing_tail() -> None:
    """Every calendar-date rejection carries timing.py's canonical tail."""
    for validator in _bindings():
        with pytest.raises(NaiveTimestampError) as excinfo:
            validator(CALENDAR_DATE, "checked_field")
        assert CANONICAL_DATE_TAIL in str(excinfo.value), (
            "every calendar-date rejection must carry timing.py's "
            "canonical tail '... declaring when the information first "
            "could have been known.'"
        )


@pytest.mark.parametrize(
    "malformed, marker",
    [
        pytest.param(
            NAIVE_INSTANT, "is a naive timestamp", id="naive-datetime"
        ),
        pytest.param(CALENDAR_DATE, "is a calendar date", id="calendar-date"),
        pytest.param(
            NON_INSTANT,
            "must be a timezone-aware datetime instant",
            id="non-instant-input",
        ),
    ],
)
def test_canonical_wording_names_the_malformed_class(
    malformed: object, marker: str
) -> None:
    validator = _bindings()[0]
    with pytest.raises(NaiveTimestampError) as excinfo:
        validator(malformed, "checked_field")
    assert marker in str(excinfo.value)
