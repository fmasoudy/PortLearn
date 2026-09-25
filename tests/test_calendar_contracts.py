"""Period-calendar contract tests for ``portlearn.calendar``.

These tests freeze the period-end → aware-instant convention
fixed in the substrate: a month-end or
quarter-end calendar day maps to the last instant of that period in a
declared timezone — 23:59:59.999999 on the last calendar day,
UTC-normalized at storage.  The mapping is defined exactly once in this
substrate module and imported — never restated — by later consumers.

Substrate discipline (C2/C3): the module is stdlib-only, pure, builds
every produced instant through the fixed ``portlearn.timing.to_instant``
law, defines no contract errors, and exposes no provider, ingestion,
alignment, or schedule machinery.
"""

from __future__ import annotations

import ast
import inspect
from datetime import UTC, datetime, timedelta, timezone

import pytest

from portlearn.calendar import month_end_instant, quarter_end_instant
from portlearn.timing import NaiveTimestampError, instant_key

# Deterministic synthetic zones (fixed offsets — no tzdata dependency).
ZONE_UTC = UTC
ZONE_PLUS_11 = timezone(timedelta(hours=11))
ZONE_MINUS_5 = timezone(timedelta(hours=-5))

LAST_MICROSECOND = 999999


class TestMonthEndInstant:
    """C1 — a month maps to the last instant of that month in the
    declared timezone, UTC-normalized at storage."""

    def test_regular_month_end_is_last_instant_of_month(self) -> None:
        last = month_end_instant(2026, 1, ZONE_UTC)
        assert last == datetime(2026, 1, 31, 23, 59, 59, LAST_MICROSECOND, tzinfo=UTC)

    def test_leap_february_month_end_is_the_twenty_ninth(self) -> None:
        last = month_end_instant(2024, 2, ZONE_UTC)
        assert last == datetime(2024, 2, 29, 23, 59, 59, LAST_MICROSECOND, tzinfo=UTC)

    def test_non_leap_february_month_end_is_the_twenty_eighth(self) -> None:
        last = month_end_instant(2023, 2, ZONE_UTC)
        assert last == datetime(2023, 2, 28, 23, 59, 59, LAST_MICROSECOND, tzinfo=UTC)

    def test_thirty_day_month_end(self) -> None:
        last = month_end_instant(2026, 4, ZONE_UTC)
        assert last == datetime(2026, 4, 30, 23, 59, 59, LAST_MICROSECOND, tzinfo=UTC)

    def test_december_month_end_does_not_rollover(self) -> None:
        last = month_end_instant(2026, 12, ZONE_UTC)
        assert last == datetime(2026, 12, 31, 23, 59, 59, LAST_MICROSECOND, tzinfo=UTC)

    def test_positive_offset_zone_shifts_the_utc_instant_earlier(self) -> None:
        # 2026-01-31 23:59:59.999999 at UTC+11 is 12:59:59.999999 UTC.
        last = month_end_instant(2026, 1, ZONE_PLUS_11)
        assert last == datetime(2026, 1, 31, 12, 59, 59, LAST_MICROSECOND, tzinfo=UTC)

    def test_negative_offset_zone_shifts_the_utc_instant_later(self) -> None:
        # 2026-01-31 23:59:59.999999 at UTC-5 is 2026-02-01 04:59:59.999999 UTC.
        last = month_end_instant(2026, 1, ZONE_MINUS_5)
        assert last == datetime(2026, 2, 1, 4, 59, 59, LAST_MICROSECOND, tzinfo=UTC)

    def test_returned_instant_is_utc_normalized_at_storage(self) -> None:
        last = month_end_instant(2026, 1, ZONE_PLUS_11)
        assert last.tzinfo is UTC

    def test_naive_zone_input_rejects_fail_closed(self) -> None:
        with pytest.raises(NaiveTimestampError):
            month_end_instant(2026, 1, None)

    def test_repeated_calls_return_equal_instants(self) -> None:
        first = month_end_instant(2026, 6, ZONE_UTC)
        second = month_end_instant(2026, 6, ZONE_UTC)
        assert first == second
        assert instant_key(first) == instant_key(second)


class TestQuarterEndInstant:
    """C1 — a quarter maps to the last instant of its closing month."""

    def test_first_quarter_end_is_march_thirty_one(self) -> None:
        last = quarter_end_instant(2026, 1, ZONE_UTC)
        assert last == datetime(2026, 3, 31, 23, 59, 59, LAST_MICROSECOND, tzinfo=UTC)

    def test_second_quarter_end_is_june_thirty(self) -> None:
        last = quarter_end_instant(2026, 2, ZONE_UTC)
        assert last == datetime(2026, 6, 30, 23, 59, 59, LAST_MICROSECOND, tzinfo=UTC)

    def test_third_quarter_end_is_september_thirty(self) -> None:
        last = quarter_end_instant(2026, 3, ZONE_UTC)
        assert last == datetime(2026, 9, 30, 23, 59, 59, LAST_MICROSECOND, tzinfo=UTC)

    def test_fourth_quarter_end_is_december_thirty_one(self) -> None:
        last = quarter_end_instant(2026, 4, ZONE_UTC)
        assert last == datetime(2026, 12, 31, 23, 59, 59, LAST_MICROSECOND, tzinfo=UTC)

    def test_quarter_end_agrees_with_closing_month_end(self) -> None:
        assert quarter_end_instant(2024, 1, ZONE_UTC) == month_end_instant(2024, 3, ZONE_UTC)

    def test_non_utc_declared_tz_is_utc_normalized(self) -> None:
        last = quarter_end_instant(2026, 1, ZONE_PLUS_11)
        assert last == datetime(2026, 3, 31, 12, 59, 59, LAST_MICROSECOND, tzinfo=UTC)
        assert last.tzinfo is UTC

    def test_naive_zone_input_rejects_fail_closed(self) -> None:
        with pytest.raises(NaiveTimestampError):
            quarter_end_instant(2026, 1, None)


class TestCalendarSubstrateDiscipline:
    """C2/C3 — purity and surface discipline: no provider, ingestion,
    alignment, or schedule machinery; imports from timing only."""

    def test_module_public_surface_is_exactly_the_two_helpers(self) -> None:
        import portlearn.calendar

        assert portlearn.calendar.__all__ == [
            "month_end_instant",
            "quarter_end_instant",
        ]

    def test_module_surface_has_no_foreign_machinery(self) -> None:
        import portlearn.calendar

        forbidden = (
            "ingest",
            "load",
            "align",
            "provider",
            "fetch",
            "schedule",
            "adapter",
            "frequency",
            "resample",
        )
        public_names = [n for n in dir(portlearn.calendar) if not n.startswith("_")]
        for name in public_names:
            assert not any(word in name.lower() for word in forbidden), (
                f"the calendar substrate exposes foreign machinery: {name!r}"
            )

    def test_module_defines_no_contract_errors(self) -> None:
        import portlearn.calendar

        for name in dir(portlearn.calendar):
            if name.startswith("_"):
                continue
            attribute = getattr(portlearn.calendar, name)
            assert not (isinstance(attribute, type) and issubclass(attribute, Exception)), (
                f"the calendar substrate defines the error {name!r}; "
                "contract errors are owned by their fixed modules"
            )

    def test_module_imports_portlearn_timing_only(self) -> None:
        import portlearn.calendar

        tree = ast.parse(inspect.getsource(portlearn.calendar))
        portlearn_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
                "portlearn"
            ):
                portlearn_modules.add(node.module)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("portlearn"):
                        portlearn_modules.add(alias.name)
        assert portlearn_modules <= {"portlearn.timing", "portlearn"}, (
            "the calendar substrate imports only from portlearn.timing; "
            f"found {sorted(portlearn_modules)}"
        )
