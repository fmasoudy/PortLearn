"""Behavior floors for ``describe``: sample statistics over retained
observations only, the pinned quantile algorithm, honest undefined
handling with value-domain accounting, and mandatory state/mode/
provenance/zero-record disclosures."""

from __future__ import annotations

import math
import statistics

import pytest
from _synthetic import (
    availability_policy,
    field_names,
    find_strings,
    fred_retained_values,
    fred_snapshot_dataset,
    has_marker,
    import_diagnostics,
    instant,
    month_key,
    period_record,
    qualified_dataset,
    timed_record,
    unqualified_dataset,
    walk_values,
)


def _series_entry(report, series_id):
    entries = getattr(report, "series", None) or getattr(report, "per_series", None)
    if entries is None:
        found = [
            item
            for item in walk_values(report)
            if getattr(item, "series_id", None) == series_id
        ]
        assert found, f"no per-series entry for {series_id!r} in the report"
        return found[0]
    for entry in entries:
        if getattr(entry, "series_id", None) == series_id:
            return entry
    raise AssertionError(f"no per-series entry named {series_id!r}")


def test_describe_computes_sample_statistics_over_retained_observations_only() -> None:
    diagnostics = import_diagnostics()

    dataset = unqualified_dataset(
        tuple(
            period_record("alpha", month_key(2023, month), value)
            for month, value in enumerate((2.0, 4.0, 9.0, 16.0), start=1)
        )
    )
    report = diagnostics.describe(dataset)
    entry = _series_entry(report, "alpha")

    assert entry.n == 4, "n must be the explicit retained numeric count"
    assert entry.mean == 7.75
    assert entry.min == 2.0
    assert entry.max == 16.0
    expected_std = math.sqrt(
        ((2.0 - 7.75) ** 2 + (4.0 - 7.75) ** 2 + (9.0 - 7.75) ** 2 + (16.0 - 7.75) ** 2)
        / 3.0
    )
    assert entry.std == pytest.approx(expected_std, rel=1e-12, abs=1e-12), (
        "std must be the sample standard deviation with ddof=1"
    )
    assert entry.std == pytest.approx(statistics.stdev([2.0, 4.0, 9.0, 16.0]))

    # Provider fixture whose table contained missing sentinels: describe
    # sees only retained rows (34 of 36), never recovered sentinels.
    fred = diagnostics.describe(fred_snapshot_dataset())
    fred_entry = _series_entry(fred, "SYNTHCPIM")
    retained = fred_retained_values()
    assert len(retained) == 34
    assert fred_entry.n == 34, (
        "n must count retained observations only — the two provider "
        "sentinel rows are absent records, invisible to describe"
    )
    assert fred_entry.min == pytest.approx(min(retained))
    assert fred_entry.max == pytest.approx(max(retained))
    assert fred_entry.mean == pytest.approx(statistics.fmean(retained))
    assert fred_entry.std == pytest.approx(statistics.stdev(retained))


def test_describe_quantile_algorithm_is_pinned_exactly() -> None:
    diagnostics = import_diagnostics()

    def report_for(values):
        return diagnostics.describe(
            unqualified_dataset(
                tuple(
                    period_record("alpha", month_key(2023, i + 1), value)
                    for i, value in enumerate(values)
                )
            )
        )

    # Interpolation case: h = (n-1)p between order statistics.
    entry = _series_entry(report_for((2.0, 4.0, 9.0, 16.0)), "alpha")
    assert entry.q25 == 3.5, "q25 must be x_(1) + 0.75*(x_(2)-x_(1)) = 3.5"
    assert entry.median == 6.5, "median must be 4 + 0.5*(9-4) = 6.5"
    assert entry.q75 == 10.75, "q75 must be 9 + 0.25*(16-9) = 10.75"

    # Tie case: repeated values pin exact order statistics.
    entry = _series_entry(report_for((5.0, 5.0, 7.0, 9.0, 9.0)), "alpha")
    assert entry.q25 == 5.0, "with h = 1 the quartile lands exactly on x_(2)"
    assert entry.median == 7.0, "with h = 2 the median is exactly x_(3)"
    assert entry.q75 == 9.0, "with h = 3 the quartile lands exactly on x_(4)"

    # n = 1 identity case: every quantile is the single observation.
    entry = _series_entry(report_for((42.0,)), "alpha")
    assert entry.q25 == 42.0
    assert entry.median == 42.0
    assert entry.q75 == 42.0


def test_describe_reports_undefined_for_insufficient_n_and_never_coerces_value_domains() -> (
    None
):
    diagnostics = import_diagnostics()

    dataset = unqualified_dataset((period_record("alpha", month_key(2023, 1), 5.0),))
    entry = _series_entry(diagnostics.describe(dataset), "alpha")
    assert entry.n == 1
    assert entry.std is None, "std is undefined for n < 2 and must be None"
    assert has_marker(entry, "insufficient_n"), (
        "the undefined std must carry the machine-readable reason 'insufficient_n'"
    )

    # Non-numeric value domains are counted and disclosed per series,
    # never coerced into statistics (qualified value domain admits them).
    qualified = diagnostics.describe(
        qualified_dataset(
            [
                timed_record("mixed", instant(2023, m, day), value)
                for m, day, value in (
                    (1, 1, 2.5),
                    (2, 1, True),
                    (3, 1, "n/a"),
                    (4, 1, None),
                    (5, 1, 3.5),
                )
            ],
            availability=availability_policy(),
        )
    )
    mixed = _series_entry(qualified, "mixed")
    assert mixed.n == 2, (
        "only int/float values count toward n; bool-as-0/1 and str never do"
    )
    assert mixed.mean == pytest.approx(3.0)
    assert mixed.min == 2.5
    assert mixed.max == 3.5
    assert mixed.std == pytest.approx(statistics.stdev([2.5, 3.5])), (
        "with exactly n==2 retained numeric values, std is the sample "
        "standard deviation (ddof=1); std is None only when n < 2"
    )
    names = field_names(mixed)
    lowered = {name.lower() for name in names}
    assert any("bool" in name for name in lowered), (
        "bool values must be counted and disclosed per series"
    )
    assert any(("str" in name) or ("string" in name) for name in lowered), (
        "str values must be counted and disclosed per series"
    )
    assert any(("none" in name) or ("null" in name) for name in lowered), (
        "present null values must be counted and disclosed per series"
    )
    for name in names:
        if "bool" in name.lower():
            assert getattr(mixed, name) == 1
        if "str" in name.lower() or "string" in name.lower():
            assert getattr(mixed, name) == 1
        if "none" in name.lower() or "null" in name.lower():
            assert getattr(mixed, name) == 1

    text_only = _series_entry(
        diagnostics.describe(
            __import__("_synthetic", fromlist=["qualified_dataset"]).qualified_dataset(
                [
                    timed_record("textonly", instant(2023, m, 1), value)
                    for m, value in ((1, "a"), (2, True), (3, None))
                ],
                availability=availability_policy(),
            )
        ),
        "textonly",
    )
    assert text_only.n == 0
    for statistic in ("mean", "std", "min", "max", "q25", "median", "q75"):
        assert getattr(text_only, statistic) is None, (
            f"{statistic} must be None for a series with zero numeric "
            "observations — never a fabricated zero"
        )
    assert find_strings(text_only), (
        "the all-undefined series entry must still carry a machine-readable reason"
    )


def test_describe_discloses_state_data_mode_provenance_and_zero_records() -> None:
    diagnostics = import_diagnostics()
    from _synthetic import qualified_dataset

    unqualified = unqualified_dataset(
        (period_record("alpha", month_key(2023, 1), 2.0),)
    )
    report = diagnostics.describe(unqualified)
    assert report.availability_state == "UNQUALIFIED"
    assert report.source_sha256 == unqualified.source_sha256
    assert report.provider == "synthetic"
    assert "retained observations" in " ".join(sorted(find_strings(report))), (
        "the report must carry the basis statement that statistics "
        "describe the retained observations only"
    )
    assert getattr(report, "data_mode", None) is None, (
        "a provider whose provenance declares no data mode must expose "
        "none on the report"
    )

    qualified = diagnostics.describe(
        qualified_dataset(
            [timed_record("alpha", instant(2023, 1, 1), 2.0)],
            availability=availability_policy(),
        )
    )
    assert qualified.availability_state == "QUALIFIED"

    fred = diagnostics.describe(fred_snapshot_dataset())
    assert fred.availability_state == "UNQUALIFIED"
    assert getattr(fred, "data_mode", None) == "CURRENT_SNAPSHOT", (
        "the FRED snapshot data mode must be disclosed on the report"
    )

    empty = diagnostics.describe(unqualified_dataset(()))
    assert getattr(empty, "zero_records", False) is True, (
        "a zero-record dataset yields a valid report with zero_records=True"
    )
    assert empty.availability_state == "UNQUALIFIED"
    assert has_marker(empty, "zero_records"), (
        "undefined aggregates must carry the reason 'zero_records'"
    )


# availability_policy is re-exported for the sibling floors' style; keep
# the module import surface honest.
_ = availability_policy
