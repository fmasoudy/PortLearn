"""Behavioral contract tests for the alignment module.

These tests enforce the approved alignment laws adversarially: the
store indexes ``(series_id, observation_time)`` groups under the frozen
record discipline, visibility is decided solely by each record's
declared availability through the frozen vintage operation (never by a
second admission rule, never by observation-date matching), absent
slots are explicit ``None`` with no silent carry, admission is
inclusive at the decision instant, instants compare as normalized
instants across timezones, the monthly decision-calendar builder
composes the shared period-calendar helper (imported, never restated)
and emits UTC month-end instants including leap February, decision
grids are strictly increasing aware instants or reject with the frozen
chronology/timestamp errors, the module aggregates nothing and adds no
publication lag, and its documentation carries the researcher warning
against observation-date and naive ``merge_asof``-style alignment.
"""

from __future__ import annotations

import ast
import inspect
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import portlearn.alignment
from portlearn.calendar import month_end_instant
from portlearn.interfaces import InformationSet
from portlearn.leakage import FROZEN_CONTRACT_ERRORS
from portlearn.observations import (
    AmbiguousObservationError,
    TimedObservation,
)
from portlearn.timing import (
    InvalidChronologyError,
    NaiveTimestampError,
)

MELBOURNE = ZoneInfo("Australia/Melbourne")
PLUS_11 = timezone(timedelta(hours=11))
MINUS_5 = timezone(timedelta(hours=-5))
LAST_MICROSECOND = 999999


def month_end(year: int, month: int, zone: object = UTC) -> datetime:
    """The shared period-end instant, via the imported helper."""
    return month_end_instant(year, month, zone)


def obs(
    series_id: str,
    observation_time: datetime,
    available_time: datetime,
    value: float,
) -> TimedObservation:
    """One point-in-time record with declared availability."""
    return TimedObservation(series_id, observation_time, available_time, value)


def cpi_series() -> list[TimedObservation]:
    """A monthly macro series whose April print publishes two weeks
    after the April month-end decision instant — the calendar-join
    leak shape: observation before the decision, availability after.
    """
    return [
        obs(
            "macro.cpi",
            month_end(2026, 2),
            datetime(2026, 3, 3, tzinfo=UTC),
            99.0,
        ),
        obs(
            "macro.cpi",
            month_end(2026, 3),
            datetime(2026, 4, 2, tzinfo=UTC),
            100.0,
        ),
        obs(
            "macro.cpi",
            month_end(2026, 4, MELBOURNE),
            datetime(2026, 5, 15, tzinfo=UTC),
            101.5,
        ),
    ]


def gdp_series() -> list[TimedObservation]:
    """A quarterly macro series with a revised vintage: the first
    print publishes 2026-04-25, the revision 2026-05-20.
    """
    return [
        obs(
            "macro.gdp",
            month_end(2026, 3),
            datetime(2026, 4, 25, tzinfo=UTC),
            1.0,
        ),
        obs(
            "macro.gdp",
            month_end(2026, 3),
            datetime(2026, 5, 20, tzinfo=UTC),
            1.5,
        ),
    ]


def mixed_store() -> portlearn.alignment.ObservationStore:
    return portlearn.alignment.ObservationStore(
        cpi_series() + gdp_series()
    )


# --------------------------------------------------------------------------- #
# Store build and record discipline (L1)
# --------------------------------------------------------------------------- #


class TestStoreBuild:
    def test_build_indexes_groups_records_and_freezes_provenance(
        self,
    ) -> None:
        store = mixed_store()
        provenance = store.provenance
        assert dict(provenance) == {
            "record_count": 5,
            "series_count": 2,
            "observation_group_count": 4,
            "availability_basis": (
                "record available_time declared at ingestion; "
                "alignment adds no publication lag"
            ),
        }
        with pytest.raises(TypeError):
            provenance["record_count"] = 6

    def test_duplicate_identity_rejects_and_revision_pairs_admit(
        self,
    ) -> None:
        first = gdp_series()[0]
        duplicate = obs(
            "macro.gdp",
            month_end(2026, 3),
            datetime(2026, 4, 25, tzinfo=UTC),
            1.0,
        )
        with pytest.raises(AmbiguousObservationError):
            portlearn.alignment.ObservationStore([first, duplicate])
        # A revision pair — same observation, later availability — is
        # exactly what the store holds; vintage choice is a query-time
        # matter for the frozen operation, never a build-time reject.
        store = portlearn.alignment.ObservationStore(gdp_series())
        assert store.provenance["observation_group_count"] == 1
        assert store.provenance["record_count"] == 2

    def test_store_inputs_are_frozen_and_no_mutation_surface_exists(
        self,
    ) -> None:
        records = cpi_series()
        store = portlearn.alignment.ObservationStore(records)
        # The constructor copies: later caller-list mutation cannot
        # change what the store indexed.
        records.append(
            obs(
                "macro.cpi",
                month_end(2026, 5),
                datetime(2026, 6, 2, tzinfo=UTC),
                102.0,
            )
        )
        assert store.provenance["record_count"] == 3
        decision = datetime(2026, 6, 30, tzinfo=UTC)
        assert all(
            slot is not None and slot.value != 102.0
            for slot in store.visible_at(decision, ["macro.cpi"])
        )
        # No public mutator ships on the store.
        mutators = {
            name
            for name in dir(store)
            if not name.startswith("_")
            and name
            in {
                "add",
                "append",
                "insert",
                "put",
                "update",
                "remove",
                "pop",
                "clear",
                "discard",
                "extend",
                "setdefault",
            }
        }
        assert mutators == set()


# --------------------------------------------------------------------------- #
# Visibility: availability-aware admission through the frozen vintage law
# --------------------------------------------------------------------------- #


class TestVisibility:
    def test_visible_at_returns_latest_visible_observation_per_series(
        self,
    ) -> None:
        store = mixed_store()
        decision = datetime(2026, 5, 1, tzinfo=UTC)
        cpi, gdp = store.visible_at(decision, ["macro.cpi", "macro.gdp"])
        assert cpi is not None and cpi.value == 100.0
        assert gdp is not None and gdp.value == 1.0

    def test_visible_at_selects_the_visible_vintage_under_revision(
        self,
    ) -> None:
        store = portlearn.alignment.ObservationStore(gdp_series())
        before_revision = datetime(2026, 4, 30, tzinfo=UTC)
        after_revision = datetime(2026, 5, 21, tzinfo=UTC)
        first = store.visible_at(before_revision, ["macro.gdp"])[0]
        second = store.visible_at(after_revision, ["macro.gdp"])[0]
        assert first is not None and first.value == 1.0
        assert second is not None and second.value == 1.5

    def test_absent_slot_is_explicit_none(self) -> None:
        store = mixed_store()
        before_anything = datetime(2026, 1, 1, tzinfo=UTC)
        assert store.visible_at(before_anything, ["macro.cpi"]) == [None]
        # Absence is not an error and never a stale value: the honest
        # answer for "not yet knowable at this instant".
        assert store.visible_at(
            before_anything, ["macro.cpi", "macro.gdp"]
        ) == [None, None]

    def test_absence_propagates_to_alignment_and_information_sets(
        self,
    ) -> None:
        store = mixed_store()
        before_anything = datetime(2026, 1, 1, tzinfo=UTC)
        assert store.visible_at(before_anything, ["macro.cpi"]) == [None]
        requested = [
            ("macro.cpi", month_end(2026, 3)),
            ("macro.gdp", month_end(2026, 3)),
        ]
        assert (
            portlearn.alignment.align(store, before_anything, requested)
            == []
        )
        # The caller composes the empty admitted set without error.
        empty = InformationSet(
            portlearn.alignment.align(store, before_anything, requested),
            as_of=before_anything,
        )
        assert list(empty) == []

    def test_macro_leak_observation_before_decision_availability_after(
        self,
    ) -> None:
        store = portlearn.alignment.ObservationStore(cpi_series())
        decision = month_end(2026, 4, MELBOURNE)
        # The April print is dated before the decision but publishes
        # 2026-05-15 — calendar-date matching would admit it; the
        # availability law excludes it and yields March's vintage.
        slot = store.visible_at(decision, ["macro.cpi"])[0]
        assert slot is not None and slot.value == 100.0
        assert slot.observation_time == month_end(2026, 3)
        assert (
            portlearn.alignment.align(
                store, decision, [("macro.cpi", month_end(2026, 4, MELBOURNE))]
            )
            == []
        )
        admitted = portlearn.alignment.align(
            store,
            decision,
            [
                ("macro.cpi", month_end(2026, 2)),
                ("macro.cpi", month_end(2026, 3)),
                ("macro.cpi", month_end(2026, 4, MELBOURNE)),
            ],
        )
        assert [record.value for record in admitted] == [99.0, 100.0]

    def test_availability_exactly_at_the_decision_instant_admits(
        self,
    ) -> None:
        decision = month_end(2026, 4, MELBOURNE)
        store = portlearn.alignment.ObservationStore(
            [
                obs(
                    "macro.cpi",
                    month_end(2026, 4, MELBOURNE),
                    decision,
                    101.5,
                )
            ]
        )
        slot = store.visible_at(decision, ["macro.cpi"])[0]
        assert slot is not None and slot.value == 101.5
        assert (
            portlearn.alignment.align(
                store,
                decision,
                [("macro.cpi", month_end(2026, 4, MELBOURNE))],
            )[0].value
            == 101.5
        )

    def test_instants_compare_normalized_across_timezones(self) -> None:
        # 2026-05-15 00:00 at UTC+11 is 2026-05-14 13:00 UTC: the same
        # instant expressed in two zones must admit identically, and
        # one microsecond earlier in any zone must not.
        published_plus_11 = datetime(2026, 5, 15, tzinfo=PLUS_11)
        store = portlearn.alignment.ObservationStore(
            [
                obs(
                    "macro.cpi",
                    month_end(2026, 4, MELBOURNE),
                    published_plus_11,
                    101.5,
                )
            ]
        )
        same_instant_utc = datetime(2026, 5, 14, 13, 0, tzinfo=UTC)
        same_instant_minus_5 = datetime(
            2026, 5, 14, 8, 0, tzinfo=MINUS_5
        )
        slot = store.visible_at(same_instant_utc, ["macro.cpi"])[0]
        assert slot is not None and slot.value == 101.5
        slot = store.visible_at(same_instant_minus_5, ["macro.cpi"])[0]
        assert slot is not None and slot.value == 101.5
        earlier = same_instant_utc - timedelta(microseconds=1)
        assert store.visible_at(earlier, ["macro.cpi"]) == [None]


# --------------------------------------------------------------------------- #
# Alignment output (L2/L3)
# --------------------------------------------------------------------------- #


class TestAlign:
    def test_align_returns_admissible_vintages_in_requested_order(
        self,
    ) -> None:
        store = mixed_store()
        decision = datetime(2026, 5, 1, tzinfo=UTC)
        admitted = portlearn.alignment.align(
            store,
            decision,
            [
                ("macro.gdp", month_end(2026, 3)),
                ("macro.cpi", month_end(2026, 2)),
                ("macro.cpi", month_end(2026, 3)),
            ],
        )
        assert [record.series_id for record in admitted] == [
            "macro.gdp",
            "macro.cpi",
            "macro.cpi",
        ]
        assert [record.value for record in admitted] == [1.0, 99.0, 100.0]

    def test_align_output_submits_to_information_set_at_the_decision(
        self,
    ) -> None:
        store = mixed_store()
        decision = datetime(2026, 5, 1, tzinfo=UTC)
        admitted = portlearn.alignment.align(
            store,
            decision,
            [
                ("macro.cpi", month_end(2026, 3)),
                ("macro.gdp", month_end(2026, 3)),
            ],
        )
        information = InformationSet(admitted, as_of=decision)
        assert len(list(information)) == 2
        # Alignment returns the caller's vintage records themselves —
        # the store's own frozen record objects, by identity.
        assert set(admitted) <= set(cpi_series() + gdp_series())

    def test_unknown_series_or_group_rejects_fail_closed(self) -> None:
        store = mixed_store()
        decision = datetime(2026, 5, 1, tzinfo=UTC)
        unknown_series = portlearn.alignment.UnknownSeriesError
        with pytest.raises(unknown_series):
            store.visible_at(decision, ["macro.unemployment"])
        with pytest.raises(unknown_series):
            portlearn.alignment.align(
                store,
                decision,
                [("macro.unemployment", month_end(2026, 3))],
            )
        # A known series without the requested observation group is a
        # structural failure, not silent absence: the request names an
        # observation the store does not hold.
        with pytest.raises(unknown_series):
            portlearn.alignment.align(
                store,
                decision,
                [("macro.cpi", month_end(2026, 6))],
            )
        declaration = portlearn.alignment.GridDeclarationError
        with pytest.raises(declaration):
            portlearn.alignment.align(store, decision, ["macro.cpi"])
        with pytest.raises(declaration):
            portlearn.alignment.align(
                store, decision, [("macro.cpi", month_end(2026, 3), "extra")]
            )


# --------------------------------------------------------------------------- #
# Decision calendar composition (L4/L5)
# --------------------------------------------------------------------------- #


class TestDecisionCalendar:
    def test_monthly_calendar_builds_month_ends_including_leap_february(
        self,
    ) -> None:
        calendar = portlearn.alignment.monthly_decision_calendar(
            (2023, 12), 3, UTC
        )
        assert calendar == [
            datetime(
                2023, 12, 31, 23, 59, 59, LAST_MICROSECOND, tzinfo=UTC
            ),
            datetime(
                2024, 1, 31, 23, 59, 59, LAST_MICROSECOND, tzinfo=UTC
            ),
            datetime(
                2024, 2, 29, 23, 59, 59, LAST_MICROSECOND, tzinfo=UTC
            ),
        ]
        across_year = portlearn.alignment.monthly_decision_calendar(
            (2025, 11), 4, UTC
        )
        assert across_year[-1] == datetime(
            2026, 2, 28, 23, 59, 59, LAST_MICROSECOND, tzinfo=UTC
        )

    def test_monthly_calendar_is_utc_normalized_and_equals_imported_helper(
        self,
    ) -> None:
        # Southern-hemisphere zone across the autumn DST boundary:
        # March ends at UTC+11, April and May at UTC+10 — the builder
        # composes the shared helper and UTC-normalizes at storage.
        calendar = portlearn.alignment.monthly_decision_calendar(
            (2026, 3), 3, MELBOURNE
        )
        assert calendar == [
            month_end(2026, 3, MELBOURNE),
            month_end(2026, 4, MELBOURNE),
            month_end(2026, 5, MELBOURNE),
        ]
        assert [instant.isoformat() for instant in calendar] == [
            "2026-03-31T12:59:59.999999+00:00",
            "2026-04-30T13:59:59.999999+00:00",
            "2026-05-31T13:59:59.999999+00:00",
        ]
        for instant in calendar:
            assert instant.tzinfo is UTC

    def test_monthly_calendar_declaration_rejects_bad_shape(self) -> None:
        declaration = portlearn.alignment.GridDeclarationError
        for bad_start in [
            (2026, 0),
            (2026, 13),
            (2026, 1, 1),
            ("2026-01",),
            2026,
        ]:
            with pytest.raises(declaration):
                portlearn.alignment.monthly_decision_calendar(
                    bad_start, 2, UTC
                )
        for bad_count in [0, -1, True, 2.5, "3"]:
            with pytest.raises(declaration):
                portlearn.alignment.monthly_decision_calendar(
                    (2026, 1), bad_count, UTC
                )
        # A naive declared timezone rejects through the frozen law the
        # shared helper itself enforces — never a local default zone.
        with pytest.raises(NaiveTimestampError):
            portlearn.alignment.monthly_decision_calendar(
                (2026, 1), 2, None
            )


# --------------------------------------------------------------------------- #
# Grid chronology (L8)
# --------------------------------------------------------------------------- #


class TestGridChronology:
    def test_non_monotone_grid_rejects_with_frozen_chronology_error(
        self,
    ) -> None:
        decreasing = [
            datetime(2026, 3, 31, tzinfo=UTC),
            datetime(2026, 2, 28, tzinfo=UTC),
        ]
        with pytest.raises(InvalidChronologyError):
            portlearn.alignment.require_increasing_instants(decreasing)
        # Strictly increasing: equal adjacent instants reject too.
        equal = [
            datetime(2026, 3, 31, tzinfo=UTC),
            datetime(2026, 3, 31, tzinfo=UTC),
        ]
        with pytest.raises(InvalidChronologyError):
            portlearn.alignment.require_increasing_instants(equal)
        increasing = [
            datetime(2026, 2, 28, tzinfo=UTC),
            datetime(2026, 3, 31, tzinfo=UTC),
        ]
        assert (
            portlearn.alignment.require_increasing_instants(increasing)
            is None
        )
        single = [datetime(2026, 3, 31, tzinfo=UTC)]
        assert (
            portlearn.alignment.require_increasing_instants(single) is None
        )

    def test_naive_instants_reject_at_every_entry_point(self) -> None:
        naive = datetime(2026, 5, 15)  # noqa: DTZ001 — deliberately naive
        store = mixed_store()
        with pytest.raises(NaiveTimestampError):
            portlearn.alignment.ObservationStore(
                [obs("macro.cpi", naive, naive, 1.0)]
            )
        with pytest.raises(NaiveTimestampError):
            store.visible_at(naive, ["macro.cpi"])
        with pytest.raises(NaiveTimestampError):
            portlearn.alignment.align(
                store,
                naive,
                [("macro.cpi", month_end(2026, 3))],
            )
        with pytest.raises(NaiveTimestampError):
            portlearn.alignment.align(
                store,
                datetime(2026, 5, 1, tzinfo=UTC),
                [("macro.cpi", naive)],
            )
        with pytest.raises(NaiveTimestampError):
            portlearn.alignment.require_increasing_instants(
                [datetime(2026, 3, 31, tzinfo=UTC), naive]
            )

    def test_empty_or_non_iterable_grid_rejects_declaration_error(
        self,
    ) -> None:
        declaration = portlearn.alignment.GridDeclarationError
        with pytest.raises(declaration):
            portlearn.alignment.require_increasing_instants([])
        with pytest.raises(declaration):
            portlearn.alignment.require_increasing_instants(
                "2026-03-31"  # a string is one label, not a grid
            )
        with pytest.raises(declaration):
            portlearn.alignment.require_increasing_instants(20260331)


# --------------------------------------------------------------------------- #
# Surface, composition, and documentation laws (L5/L6/L10, no second rule)
# --------------------------------------------------------------------------- #


def module_tree() -> ast.Module:
    return ast.parse(inspect.getsource(portlearn.alignment))


class TestModuleSurface:
    def test_surface_is_exact_and_has_no_aggregation_operations(
        self,
    ) -> None:
        assert portlearn.alignment.__all__ == [
            "GridDeclarationError",
            "ObservationStore",
            "UnknownSeriesError",
            "align",
            "monthly_decision_calendar",
            "require_increasing_instants",
        ]
        public = {
            name
            for name in dir(portlearn.alignment)
            if not name.startswith("_")
        }
        assert public == set(portlearn.alignment.__all__) | {
            "Any",
            "AmbiguousObservationError",
            "Iterable",
            "Mapping",
            "MappingProxyType",
            "Sequence",
            "TimedObservation",
            "annotations",
            "datetime",
            "instant_key",
            "InvalidChronologyError",
            "month_end_instant",
            "pairwise",
            "to_instant",
            "vintage_as_of",
        }
        # No aggregation, resampling, interpolation, filling, or
        # joining operation is defined or called anywhere: alignment
        # aligns; frequency transformation belongs to transforms.
        tree = module_tree()
        banned_calls = {
            "sum",
            "mean",
            "fmean",
            "median",
            "mode",
            "aggregate",
            "resample",
            "interpolate",
            "ffill",
            "bfill",
            "fillna",
            "merge",
            "merge_asof",
            "join_asof",
            "cumsum",
            "cumprod",
            "cummax",
            "cummin",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(
                node.func, ast.Name
            ):
                assert node.func.id not in banned_calls
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert not any(
                    fragment in node.name.lower()
                    for fragment in (
                        "resampl",
                        "aggregat",
                        "interpolat",
                        "ffill",
                        "bfill",
                        "merge",
                        "cumsum",
                    )
                ), node.name

    def test_no_second_admission_predicate_and_no_information_sets(
        self,
    ) -> None:
        tree = module_tree()
        # The frozen vintage operation is the module's only vintage
        # path: it is imported from the observations module and called.
        vintage_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "vintage_as_of"
        ]
        assert vintage_calls
        # No comparison anywhere in the module may have an availability
        # operand: the inclusive admission law lives only in the frozen
        # modules, never re-implemented here.
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                operands = [node.left, *node.comparators]
                for operand in operands:
                    for sub in ast.walk(operand):
                        if isinstance(sub, ast.Attribute) and (
                            "available" in sub.attr.lower()
                        ):
                            raise AssertionError(
                                f"local admission predicate at line "
                                f"{node.lineno}: {ast.unparse(node)}"
                            )
                        if isinstance(sub, ast.Name) and (
                            "available" in sub.id.lower()
                        ):
                            raise AssertionError(
                                f"local admission predicate at line "
                                f"{node.lineno}: {ast.unparse(node)}"
                            )
        # The module never constructs information sets: admission of
        # the aligned records is the caller's, through the frozen
        # constructor laws.
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "InformationSet":
                raise AssertionError("alignment references InformationSet")
            if isinstance(node, ast.Attribute) and (
                node.attr == "InformationSet"
            ):
                raise AssertionError("alignment references InformationSet")

    def test_period_mapping_is_imported_never_restated(self) -> None:
        tree = module_tree()
        # The shared period-end helper is imported from the calendar
        # substrate and called inside the builder, whose output is
        # validated by the same grid law researchers use.
        source_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "portlearn.calendar"
        ]
        assert source_imports
        assert any(
            alias.name == "month_end_instant"
            for node in source_imports
            for alias in node.names
        )
        builder = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "monthly_decision_calendar"
        )
        called = {
            node.func.id
            for node in ast.walk(builder)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert {"month_end_instant", "require_increasing_instants"} <= called
        # No restated mapping, no local instant construction, and no
        # duration arithmetic (which could smuggle a publication lag):
        # every produced instant comes from the shared helper.
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(
                node.func, ast.Name
            ):
                assert node.func.id != "datetime", (
                    "local instant construction restates the mapping"
                )
            if isinstance(node, ast.Name):
                assert node.id not in {
                    "timedelta",
                    "monthrange",
                }, node.id
            if isinstance(node, ast.Attribute):
                assert node.attr != "monthrange", node.attr
            if isinstance(node, ast.Import) and any(
                alias.name == "calendar" for alias in node.names
            ):
                raise AssertionError("stdlib calendar re-imported")

    def test_researcher_warning_present_in_module_documentation(
        self,
    ) -> None:
        documentation = (portlearn.alignment.__doc__ or "").lower()
        for phrase in (
            "observation date",
            "merge_asof",
            "leak",
            "availability",
            "authoritative",
        ):
            assert phrase in documentation, phrase

    def test_error_taxonomy_reuses_frozen_and_owns_valueerror_subclasses(
        self,
    ) -> None:
        assert issubclass(
            portlearn.alignment.UnknownSeriesError, ValueError
        )
        assert issubclass(
            portlearn.alignment.GridDeclarationError, ValueError
        )
        for error in (
            portlearn.alignment.UnknownSeriesError,
            portlearn.alignment.GridDeclarationError,
        ):
            assert error not in FROZEN_CONTRACT_ERRORS
        assert len(FROZEN_CONTRACT_ERRORS) == 6
        # Frozen errors are reused by identity, never re-defined.
        assert (
            portlearn.alignment.AmbiguousObservationError
            is AmbiguousObservationError
        )
        assert (
            portlearn.alignment.InvalidChronologyError
            is InvalidChronologyError
        )


# --------------------------------------------------------------------------- #
# Repository hygiene: the module ships in the built wheel surface
# --------------------------------------------------------------------------- #


def test_alignment_module_lives_in_the_package_source_tree() -> None:
    module_path = Path(portlearn.alignment.__file__).resolve()
    assert module_path.name == "alignment.py"
    assert module_path.parent.name == "portlearn"
    assert module_path.parents[1].name == "src"
