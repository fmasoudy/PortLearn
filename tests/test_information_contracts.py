"""Information-surface contract tests for portlearn.

These tests freeze the documented information semantics: the Data/Feature
separation laws including feature lineage monotonicity, point-in-time
vintage selection over a single ``(series_id, observation_time)`` group,
exact series identity with no normalization of any kind, the
``AmbiguousObservationError``/``FeatureLineageError`` arms of the
fail-closed taxonomy, and the frozen module ownership, import-cycle,
stdlib-only, and file-scope invariants of the package.  Sections below
cover the valid-admission matrix (vintage selection and determinism on
the observations surface), the rejection matrix (malformed instants,
lineage violations, ambiguous identity, and mixed groups), the pinned
non-error behavior matrix, and the module-level structural invariants.
"""

from __future__ import annotations

import ast
import sys
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from portlearn.observations import (
    AmbiguousObservationError,
    FeatureLineageError,
    TimedObservation,
    require_lineage_monotone,
    vintage_as_of,
)
from portlearn.timing import (
    FutureInformationError,
    InvalidChronologyError,
    MissingAvailabilityError,
    NaiveTimestampError,
)

MELBOURNE_SUMMER = timezone(timedelta(hours=11))
IST = timezone(timedelta(hours=5, minutes=30))  # half-hour offset zone

SERIES_MACRO = "MACRO.GDPQ.REAL"
SERIES_CPI = "MACRO.CPI.YOY"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "portlearn"
TIMING_PATH = PACKAGE_ROOT / "timing.py"
OBSERVATIONS_PATH = PACKAGE_ROOT / "observations.py"

# Point-in-time revision history for one (series_id, observation_time)
# group: the same quarterly observation released, then revised twice.
VINTAGE_OBSERVATION = datetime(2026, 2, 28, 9, 0, tzinfo=UTC)
FIRST_RELEASE = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)
SECOND_RELEASE = datetime(2026, 3, 16, 9, 0, tzinfo=UTC)
THIRD_RELEASE = datetime(2026, 4, 2, 9, 0, tzinfo=UTC)
DECISION_END_OF_APRIL = datetime(2026, 4, 30, 23, 0, tzinfo=UTC)


def revision(
    available_time: datetime,
    value: float = 1.0,
    series_id: str = SERIES_MACRO,
    observation_time: datetime = VINTAGE_OBSERVATION,
) -> TimedObservation:
    """One point-in-time record of the synthetic revision history."""
    return TimedObservation(
        series_id=series_id,
        observation_time=observation_time,
        available_time=available_time,
        value=value,
    )


def revision_history() -> list[TimedObservation]:
    return [
        revision(FIRST_RELEASE, 1.0),
        revision(SECOND_RELEASE, 2.0),
        revision(THIRD_RELEASE, 3.0),
    ]


# ---------------------------------------------------------------------------
# Valid-admission matrix (observations surface)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "decision,expected_available,expected_value",
    [
        pytest.param(
            datetime(2026, 3, 10, 9, 0, tzinfo=UTC),
            FIRST_RELEASE,
            1.0,
            id="after-first-release",
        ),
        pytest.param(
            FIRST_RELEASE,
            FIRST_RELEASE,
            1.0,
            id="exactly-at-first-release",
        ),
        pytest.param(
            datetime(2026, 3, 20, 9, 0, tzinfo=UTC),
            SECOND_RELEASE,
            2.0,
            id="after-second-release",
        ),
        pytest.param(
            THIRD_RELEASE,
            THIRD_RELEASE,
            3.0,
            id="exactly-at-third-release",
        ),
        pytest.param(
            DECISION_END_OF_APRIL,
            THIRD_RELEASE,
            3.0,
            id="after-third-release",
        ),
    ],
)
def test_latest_visible_vintage_is_selected(
    decision: datetime, expected_available: datetime, expected_value: float
) -> None:
    """The visible vintage is the latest available at-or-before the
    decision, never the earliest."""
    vintage = vintage_as_of(revision_history(), decision)
    assert vintage is not None
    assert vintage.series_id == SERIES_MACRO
    assert vintage.observation_time == VINTAGE_OBSERVATION
    assert vintage.available_time == expected_available
    assert vintage.value == expected_value


def test_revised_value_is_invisible_before_its_own_availability() -> None:
    """A revised value cannot be seen before its own release."""
    before_any_revision = datetime(2026, 3, 10, 9, 0, tzinfo=UTC)
    vintage = vintage_as_of(revision_history(), before_any_revision)
    assert vintage is not None
    assert vintage.available_time == FIRST_RELEASE
    assert vintage.value == 1.0  # the first estimate still stands

    just_before_third = THIRD_RELEASE - timedelta(microseconds=1)
    vintage = vintage_as_of(revision_history(), just_before_third)
    assert vintage is not None
    assert vintage.available_time == SECOND_RELEASE
    assert vintage.value == 2.0


def test_vintage_selection_is_input_order_independent() -> None:
    """Selection is by availability instant, not list position."""
    history = revision_history()
    for permutation in (
        [history[2], history[0], history[1]],
        [history[1], history[2], history[0]],
    ):
        vintage = vintage_as_of(permutation, DECISION_END_OF_APRIL)
        assert vintage is not None
        assert vintage.available_time == THIRD_RELEASE
        assert vintage.value == 3.0


def test_vintage_selection_compares_zones_as_instants() -> None:
    """Availability declared in a half-hour zone, decided in UTC."""
    history = [
        revision(FIRST_RELEASE.astimezone(IST), 1.0),
        revision(SECOND_RELEASE.astimezone(MELBOURNE_SUMMER), 2.0),
    ]
    vintage = vintage_as_of(history, SECOND_RELEASE)  # UTC form
    assert vintage is not None
    assert vintage.available_time == SECOND_RELEASE
    assert vintage.value == 2.0


def test_determinism_observations_surface() -> None:
    """Identical inputs yield identical vintages and messages."""
    history = revision_history()
    decision = datetime(2026, 3, 20, 9, 0, tzinfo=UTC)
    first = vintage_as_of(history, decision)
    assert first is not None
    fingerprint = (
        first.series_id,
        first.observation_time,
        first.available_time,
        first.value,
    )
    for _ in range(4):
        again = vintage_as_of(history, decision)
        assert again is not None
        assert (
            again.series_id,
            again.observation_time,
            again.available_time,
            again.value,
        ) == fingerprint

    def duplicate_message() -> str:
        with pytest.raises(AmbiguousObservationError) as excinfo:
            vintage_as_of(
                [revision(SECOND_RELEASE, 1.0), revision(SECOND_RELEASE, 2.0)],
                decision,
            )
        return str(excinfo.value)

    assert duplicate_message() == duplicate_message()

    def blank_identifier_message() -> str:
        with pytest.raises(ValueError) as excinfo:
            revision(FIRST_RELEASE, 1.0, series_id="   ")
        return str(excinfo.value)

    assert blank_identifier_message() == blank_identifier_message()

    def lineage_message() -> str:
        with pytest.raises(FeatureLineageError) as excinfo:
            require_lineage_monotone(FIRST_RELEASE, [SECOND_RELEASE])
        return str(excinfo.value)

    assert lineage_message() == lineage_message()


def test_vintage_as_of_validates_decision_time_before_any_branching() -> None:
    """decision_time is validated before any input branching."""
    with pytest.raises(NaiveTimestampError):
        vintage_as_of(revision_history(), datetime(2026, 3, 10, 9, 0))  # noqa: DTZ001  # intentional naive instant
    with pytest.raises(NaiveTimestampError):
        vintage_as_of(revision_history(), date(2026, 3, 10))
    with pytest.raises(NaiveTimestampError):
        vintage_as_of([], datetime(2026, 3, 10, 9, 0))  # naive wins over empty  # noqa: DTZ001  # intentional naive instant


# ---------------------------------------------------------------------------
# Rejection matrix (observations surface)
# ---------------------------------------------------------------------------


def test_constructor_rejects_naive_instants() -> None:
    """Naive observation or availability instants never construct."""
    naive = datetime(2026, 2, 28, 9, 0)  # noqa: DTZ001  # intentional naive instant
    with pytest.raises(NaiveTimestampError):
        revision(naive)  # naive available_time
    with pytest.raises(NaiveTimestampError):
        TimedObservation(
            series_id=SERIES_MACRO,
            observation_time=naive,  # naive observation_time
            available_time=FIRST_RELEASE,
            value=1.0,
        )


def test_constructor_rejects_date_inputs_without_coercion() -> None:
    """A dated observation never silently becomes midnight-available."""
    with pytest.raises(NaiveTimestampError):
        revision(date(2026, 3, 2))  # date as available_time
    with pytest.raises(NaiveTimestampError):
        TimedObservation(
            series_id=SERIES_MACRO,
            observation_time=VINTAGE_OBSERVATION,
            available_time=date(2026, 3, 2),  # date, not instant
            value=1.0,
        )


def test_constructor_rejects_absent_availability() -> None:
    """``available_time`` is mandatory on every contract surface."""
    with pytest.raises(MissingAvailabilityError):
        TimedObservation(
            series_id=SERIES_MACRO,
            observation_time=VINTAGE_OBSERVATION,
            available_time=None,
            value=1.0,
        )


def test_feature_availability_before_latest_input_is_rejected() -> None:
    """A feature cannot be available before its latest input."""
    with pytest.raises(FeatureLineageError):
        require_lineage_monotone(
            feature_available_time=SECOND_RELEASE - timedelta(minutes=1),
            input_available_times=[FIRST_RELEASE, SECOND_RELEASE],
        )


def test_lineage_cutoff_is_latest_input_not_earliest() -> None:
    """The lineage cutoff is the *latest* input, not the earliest.

    A feature available after the earliest input but before the latest
    is still rejected.
    """
    with pytest.raises(FeatureLineageError):
        require_lineage_monotone(
            feature_available_time=datetime(2026, 3, 2, 12, 0, tzinfo=UTC),
            input_available_times=[FIRST_RELEASE, SECOND_RELEASE],
        )


def test_lineage_boundary_at_the_latest_input_is_admissible() -> None:
    """A feature available exactly at, or after, its latest input."""
    assert (
        require_lineage_monotone(
            SECOND_RELEASE, [FIRST_RELEASE, SECOND_RELEASE]
        )
        is None
    )
    assert (
        require_lineage_monotone(
            SECOND_RELEASE + timedelta(minutes=1), [FIRST_RELEASE]
        )
        is None
    )


def test_empty_input_collection_is_rejected() -> None:
    """A feature with no declared inputs has no defensible
    availability."""
    with pytest.raises(FeatureLineageError):
        require_lineage_monotone(
            feature_available_time=SECOND_RELEASE,
            input_available_times=[],
        )


def test_lineage_surface_rejects_naive_and_date_inputs() -> None:
    """Malformed time inputs fail closed on the lineage surface too."""
    with pytest.raises(NaiveTimestampError):
        require_lineage_monotone(
            feature_available_time=datetime(2026, 3, 16, 9, 0),  # noqa: DTZ001  # intentional naive instant
            input_available_times=[FIRST_RELEASE],
        )
    with pytest.raises(NaiveTimestampError):
        require_lineage_monotone(
            feature_available_time=date(2026, 3, 16),
            input_available_times=[FIRST_RELEASE],
        )
    with pytest.raises(NaiveTimestampError):
        require_lineage_monotone(
            feature_available_time=SECOND_RELEASE,
            input_available_times=[datetime(2026, 3, 2, 9, 0)],  # noqa: DTZ001  # intentional naive instant
        )
    with pytest.raises(NaiveTimestampError):
        require_lineage_monotone(
            feature_available_time=SECOND_RELEASE,
            input_available_times=[date(2026, 3, 2)],
        )


def test_duplicate_identity_conflicting_values_is_rejected() -> None:
    """Two records with the same identity triple are ambiguous."""
    with pytest.raises(AmbiguousObservationError) as excinfo:
        vintage_as_of(
            [revision(SECOND_RELEASE, 1.0), revision(SECOND_RELEASE, 2.0)],
            DECISION_END_OF_APRIL,
        )
    assert SERIES_MACRO in str(excinfo.value)  # names the series_id


def test_duplicate_identity_identical_values_is_still_rejected() -> None:
    """No value-equality exception: the identity triple is a key, so even
    identical values are still ambiguous."""
    with pytest.raises(AmbiguousObservationError):
        vintage_as_of(
            [revision(SECOND_RELEASE, 1.0), revision(SECOND_RELEASE, 1.0)],
            DECISION_END_OF_APRIL,
        )


# ---------------------------------------------------------------------------
# Behavior matrix (pinned non-error outcomes)
# ---------------------------------------------------------------------------


def test_empty_input_returns_no_vintage() -> None:
    """Empty input is no vintage, explicitly not an error."""
    assert vintage_as_of([], DECISION_END_OF_APRIL) is None


def test_no_visible_vintage_returns_no_vintage() -> None:
    """Every release after the decision: no vintage, not an error."""
    history = revision_history()
    before_every_release = FIRST_RELEASE - timedelta(microseconds=1)
    assert vintage_as_of(history, before_every_release) is None
    assert vintage_as_of(history, VINTAGE_OBSERVATION) is None


@pytest.mark.parametrize(
    "identifier",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="spaces"),
        pytest.param("\t", id="tab"),
        pytest.param(" \n\r ", id="mixed-whitespace"),
        pytest.param("\u00a0", id="non-breaking-space"),
        pytest.param(123, id="non-string-integer"),
        pytest.param(None, id="non-string-none"),
    ],
)
def test_blank_or_non_string_identifier_is_rejected(
    identifier: object,
) -> None:
    """Blank or non-string series identifiers fail closed at
    construction with the built-in ``ValueError``."""
    with pytest.raises(ValueError) as excinfo:
        revision(FIRST_RELEASE, 1.0, series_id=identifier)  # type: ignore[arg-type]
    assert "series" in str(excinfo.value).lower()


def test_identifier_case_variants_remain_distinct_series() -> None:
    """No case folding.

    The two records share an observation_time but carry identifiers that
    differ only by case, so they form two groups and the mixed-group
    input must reject.  A case-folding mutation collapses them into one
    selectable group (their availability instants differ, so no duplicate
    arises) and flips this test to a non-error.
    """
    upper = revision(FIRST_RELEASE, 1.0, series_id=SERIES_CPI)
    lower = revision(SECOND_RELEASE, 2.0, series_id="macro.cpi.yoy")
    with pytest.raises(AmbiguousObservationError):
        vintage_as_of([upper, lower], DECISION_END_OF_APRIL)


def test_identifier_whitespace_variants_remain_distinct_series() -> None:
    """No whitespace stripping or trimming."""
    plain = revision(FIRST_RELEASE, 1.0, series_id=SERIES_CPI)
    leading = revision(SECOND_RELEASE, 2.0, series_id=" MACRO.CPI.YOY")
    with pytest.raises(AmbiguousObservationError):
        vintage_as_of([plain, leading], DECISION_END_OF_APRIL)
    trailing = revision(SECOND_RELEASE, 2.0, series_id="MACRO.CPI.YOY ")
    with pytest.raises(AmbiguousObservationError):
        vintage_as_of([plain, trailing], DECISION_END_OF_APRIL)


def test_unicode_normalization_variants_remain_distinct_series() -> None:
    """No Unicode normalization."""
    composed = revision(FIRST_RELEASE, 1.0, series_id="MACRO.CAF\u00c9.YOY")
    decomposed = revision(
        SECOND_RELEASE, 2.0, series_id="MACRO.CAFE\u0301.YOY"
    )
    with pytest.raises(AmbiguousObservationError):
        vintage_as_of([composed, decomposed], DECISION_END_OF_APRIL)


def test_identifier_is_preserved_exactly() -> None:
    """Identifiers round-trip byte-for-byte through construction
    and selection; no normalization on the way out either."""
    padded = revision(FIRST_RELEASE, 1.0, series_id=" MACRO.CPI.YOY ")
    assert padded.series_id == " MACRO.CPI.YOY "
    vintage = vintage_as_of([padded], DECISION_END_OF_APRIL)
    assert vintage is not None
    assert vintage.series_id == " MACRO.CPI.YOY "


def test_mixed_series_groups_are_rejected() -> None:
    """Input spanning more than one group rejects."""
    gdp = revision(FIRST_RELEASE, 1.0, series_id=SERIES_MACRO)
    cpi = revision(SECOND_RELEASE, 2.0, series_id=SERIES_CPI)
    with pytest.raises(AmbiguousObservationError) as excinfo:
        vintage_as_of([gdp, cpi], DECISION_END_OF_APRIL)
    message = str(excinfo.value)
    assert SERIES_MACRO in message or SERIES_CPI in message


def test_mixed_observation_times_are_rejected() -> None:
    """Same series, different observation_time: still mixed groups."""
    february = revision(
        FIRST_RELEASE,
        1.0,
        observation_time=datetime(2026, 2, 28, 9, 0, tzinfo=UTC),
    )
    march = revision(
        datetime(2026, 4, 1, 9, 0, tzinfo=UTC),
        2.0,
        observation_time=datetime(2026, 3, 31, 9, 0, tzinfo=UTC),
    )
    with pytest.raises(AmbiguousObservationError):
        vintage_as_of([february, march], DECISION_END_OF_APRIL)


# ---------------------------------------------------------------------------
# Module ownership, import-cycle, dependency, and file-scope invariants
# ---------------------------------------------------------------------------


def _import_targets(path: Path) -> set[str]:
    """Every module imported by ``path``, with relative imports prefixed
    by their leading dots."""
    targets: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            if node.module:
                targets.add(prefix + node.module)
            else:  # ``from . import timing`` — one target per alias
                targets.update(prefix + alias.name for alias in node.names)
    return targets


def test_error_module_ownership_is_frozen() -> None:
    """The six errors live in exactly their frozen home modules."""
    assert NaiveTimestampError.__module__ == "portlearn.timing"
    assert InvalidChronologyError.__module__ == "portlearn.timing"
    assert FutureInformationError.__module__ == "portlearn.timing"
    assert MissingAvailabilityError.__module__ == "portlearn.timing"
    assert AmbiguousObservationError.__module__ == "portlearn.observations"
    assert FeatureLineageError.__module__ == "portlearn.observations"


def test_timing_module_imports_stdlib_only() -> None:
    """No import cycle: ``portlearn.timing`` imports nothing from
    ``portlearn.observations`` — or from ``portlearn`` at all — and no
    non-stdlib module."""
    non_stdlib = {
        target
        for target in _import_targets(TIMING_PATH)
        if target.split(".")[0] not in sys.stdlib_module_names
    }
    assert non_stdlib == set(), (
        "portlearn.timing must import stdlib only, importing nothing from "
        "portlearn.observations so no import cycle exists on the frozen "
        f"surface; found {sorted(non_stdlib)}"
    )


def test_observations_module_imports_stdlib_and_timing_only() -> None:
    """``portlearn.observations`` may import the timing errors it raises
    and nothing else outside the stdlib."""
    allowed = {"portlearn", "portlearn.timing", ".timing"}
    illegal = {
        target
        for target in _import_targets(OBSERVATIONS_PATH)
        if target.split(".")[0] not in sys.stdlib_module_names
        and target not in allowed
    }
    assert illegal == set(), (
        "portlearn.observations may import only the stdlib and the timing "
        f"errors it raises; found {sorted(illegal)}"
    )


def test_package_file_scope_is_exactly_the_frozen_write_set() -> None:
    """Required-file floor semantics — the core package and test files
    must remain present under their exact names; none may be removed or
    renamed.  Additional files are admitted above this floor."""
    frozen_package_files = {
        "__init__.py",
        "py.typed",
        "timing.py",
        "observations.py",
    }
    package_files = {p.name for p in PACKAGE_ROOT.iterdir() if p.is_file()}
    assert frozen_package_files <= package_files, (
        "src/portlearn must still contain the package identity files plus "
        "the timing and observations contract modules (required-file "
        "floor; later files are admitted above it); missing "
        f"{sorted(frozen_package_files - package_files)}"
    )
    frozen_test_files = {
        "test_information_contracts.py",
        "test_package_contract.py",
        "test_timing_contracts.py",
    }
    test_files = {
        p.name for p in (REPOSITORY_ROOT / "tests").iterdir() if p.is_file()
    }
    assert frozen_test_files <= test_files, (
        "tests/ must still contain the base suite plus the timing and "
        "information contract files (required-file floor; later files "
        "are admitted above it); missing "
        f"{sorted(frozen_test_files - test_files)}"
    )


def test_package_init_stays_identity_only_without_eager_imports() -> None:
    """``src/portlearn/__init__.py`` never imports the contract modules
    (checked at source level)."""
    portlearn_imports = {
        target
        for target in _import_targets(PACKAGE_ROOT / "__init__.py")
        if target == "portlearn" or target.startswith(("portlearn.", "."))
    }
    assert portlearn_imports == set(), (
        "the package identity module must not import portlearn.timing or "
        "portlearn.observations; found "
        f"{sorted(portlearn_imports)}"
    )
