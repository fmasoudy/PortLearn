"""Behavior floors for ``coverage``: observed-support/accounting only —
the fixed per-series record/series/distinct-exact-key/duplicate-cell
counts and value-domain presence accounting, duplicate disclosure, no
completeness claim, no chronology on UNQUALIFIED labels, and mandatory
state disclosure."""

from __future__ import annotations

from _synthetic import (
    availability_policy,
    field_names,
    find_strings,
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

#: The fixed per-series coverage accounting fields (S107-109).
COVERAGE_SERIES_FIELDS = (
    "n_records",
    "n_series",
    "n_distinct_exact_keys",
    "n_duplicate_cells",
)


def test_coverage_counts_records_series_and_distinct_keys_per_series() -> None:
    diagnostics = import_diagnostics()

    dataset = unqualified_dataset(
        (
            period_record("alpha", month_key(2023, 1), 1.0),
            period_record("alpha", month_key(2023, 2), 2.0),
            period_record("alpha", month_key(2023, 3), 3.0),
            period_record("alpha", month_key(2023, 3), 3.5),  # duplicate key
            period_record("beta", month_key(2023, 1), 10.0),
        )
    )
    report = diagnostics.coverage(dataset)

    assert report.availability_state == "UNQUALIFIED"
    assert report.n_records == 5, (
        "n_records is the retained-record accounting count (5, duplicates included)"
    )
    assert report.n_series == 2

    per_series = {
        entry.series_id: entry
        for entry in walk_values(report)
        if hasattr(entry, "series_id") and hasattr(entry, "n_records")
    }
    alpha = per_series["alpha"]
    assert alpha.n_records == 4
    assert alpha.n_distinct_exact_keys == 3, (
        "alpha retains 4 records over 3 distinct exact keys"
    )
    assert alpha.n_duplicate_cells == 1, (
        "exactly one (series_id, key) cell holds two retained records"
    )
    beta = per_series["beta"]
    assert beta.n_records == 1
    assert beta.n_distinct_exact_keys == 1
    assert beta.n_duplicate_cells == 0

    # The fixed accounting field names appear on the per-series
    # structure — no invented spelling shadows the fixed schema.
    names = field_names(report)
    for fixed in COVERAGE_SERIES_FIELDS:
        assert fixed in names, (
            f"the coverage report must carry the fixed accounting field "
            f"{fixed!r}; fields present: {sorted(names)}"
        )
    for invented in (
        "record_count",
        "series_count",
        "distinct_keys",
        "duplicate_count",
    ):
        assert invented not in names, (
            f"the invented field {invented!r} must not appear: the fixed "
            "S107-109 accounting names are the contract"
        )


def test_coverage_per_series_value_domain_accounting() -> None:
    """Per-series value-domain presence accounting — counts of the
    numeric / bool / str / None values actually retained, where the
    state's value domain admits them. Values are counted and disclosed,
    never coerced into statistics."""
    diagnostics = import_diagnostics()

    # UNQUALIFIED domain is floats only: each retained record is numeric.
    unq = unqualified_dataset(
        (
            period_record("alpha", month_key(2023, 1), 1.0),
            period_record("alpha", month_key(2023, 2), 2.0),
            period_record("beta", month_key(2023, 1), 10.0),
        )
    )
    unq_report = diagnostics.coverage(unq)
    per_series = {
        entry.series_id: entry
        for entry in walk_values(unq_report)
        if hasattr(entry, "series_id") and hasattr(entry, "n_records")
    }
    assert per_series["alpha"].n_records == 2
    assert per_series["beta"].n_records == 1
    # The per-series structure discloses value-domain presence counts:
    # whatever spelling carries them, alpha discloses two numeric
    # values and beta one, and no non-numeric presence is invented.
    alpha_strings = find_strings(per_series["alpha"])
    assert any("numeric" in leaf.lower() for leaf in alpha_strings) or any(
        "value" in leaf.lower() for leaf in alpha_strings
    ), (
        "the per-series accounting must disclose value-domain presence "
        f"counting; fields: {sorted(field_names(per_series['alpha']))}"
    )

    # QUALIFIED domain admits bool/str/None values: they are counted as
    # present non-numeric observations, never coerced and never dropped
    # from n_records.
    q = qualified_dataset(
        [timed_record("alpha", instant(2023, m), float(m)) for m in (1, 2)]
        + [timed_record("alpha", instant(2023, 3), True)]
        + [timed_record("alpha", instant(2023, 4), "n/a")]
        + [timed_record("alpha", instant(2023, 5), None)],
        availability=availability_policy(),
    )
    q_report = diagnostics.coverage(q)
    assert q_report.n_records == 5
    q_series = {
        entry.series_id: entry
        for entry in walk_values(q_report)
        if hasattr(entry, "series_id") and hasattr(entry, "n_records")
    }
    assert q_series["alpha"].n_records == 5, (
        "every retained record counts in n_records regardless of its "
        "value type — no silent dropping of non-numeric values"
    )
    q_names = field_names(q_series["alpha"])
    assert any(
        "numeric" in name or "bool" in name or "value" in name for name in q_names
    ) or any(
        "numeric" in leaf.lower() or "bool" in leaf.lower()
        for leaf in find_strings(q_series["alpha"])
    ), (
        "the QUALIFIED per-series accounting must disclose the "
        f"non-numeric value-domain presence; fields: {sorted(q_names)}"
    )


def test_coverage_discloses_duplicates_fail_closed() -> None:
    diagnostics = import_diagnostics()

    dataset = unqualified_dataset(
        (
            period_record("alpha", month_key(2023, 1), 1.0),
            period_record("alpha", month_key(2023, 1), 1.0),
            period_record("alpha", month_key(2023, 2), 2.0),
            period_record("beta", month_key(2023, 1), 10.0),
        )
    )
    report = diagnostics.coverage(dataset)
    assert has_marker(report, "duplicate"), (
        "duplicate keys must be disclosed on the report (machine-readable)"
    )
    names = field_names(report)
    lowered = {name.lower() for name in names}
    assert any("duplicate" in name for name in lowered), (
        f"coverage must carry a duplicate disclosure field; fields: {lowered}"
    )

    # FRED replica: 34 retained records, no duplicates expected.
    fred_report = diagnostics.coverage(fred_snapshot_dataset())
    assert fred_report.n_records == 34


def test_coverage_makes_no_completeness_claim_and_no_chronology_on_labels() -> None:
    diagnostics = import_diagnostics()

    dataset = unqualified_dataset(
        tuple(
            period_record("alpha", month_key(2023, m), float(m)) for m in (1, 2, 4, 7)
        )
    )
    report = diagnostics.coverage(dataset)
    strings = find_strings(report)
    joined = " \n".join(strings).lower()
    for banned in (
        "complete",
        "100%",
        "100 %",
        "completeness",
        "first",
        "last",
        "span",
        "gap",
        "frequency",
        "calendar",
        "missing periods",
    ):
        assert banned not in joined, (
            f"coverage must not state {banned!r}: observed-support only, "
            "no completeness claim, no chronology on unqualified labels"
        )
    for item in walk_values(report):
        assert not hasattr(item, "completeness"), (
            "no completeness field may exist on any coverage structure"
        )
        assert not hasattr(item, "first_key")
        assert not hasattr(item, "last_key")
        assert not hasattr(item, "span")
        assert not hasattr(item, "gaps")

    # No denominator inference: coverage accepts no expected-grid kwarg.
    try:
        diagnostics.coverage(
            dataset,
            expected_keys=[("alpha", month_key(2023, m)) for m in (1, 2)],
        )
    except TypeError:
        pass
    else:
        raise AssertionError(
            "coverage must accept no expected_keys argument: the expected "
            "grid belongs to missingness alone; coverage is "
            "observed-support accounting only"
        )


def test_coverage_groups_qualified_keys_on_observation_time_never_available() -> None:
    """QUALIFIED law: exact-key grouping/accounting uses the
    normalized observation-time identity; ``available_time`` is never
    read for grouping. Two records sharing an observation_time with
    DIFFERENT available_times form one duplicate cell (1 distinct key,
    1 duplicate) — an implementation grouping on the availability
    triple would instead report 2 distinct keys and 0 duplicates, so
    this reference kills available_time-grouping (W6). The distinct-
    instant support summary describes existing observation_time
    identity only."""
    diagnostics = import_diagnostics()

    revised = qualified_dataset(
        [
            timed_record(
                "alpha",
                instant(2023, 1),
                1.0,
                available=instant(2023, 1, 2),
            ),
            timed_record(
                "alpha",
                instant(2023, 1),
                1.5,  # revised value, later availability, same observation
                available=instant(2023, 1, 9),
            ),
            timed_record("alpha", instant(2023, 2), 2.0, available=instant(2023, 2, 2)),
        ],
        availability=availability_policy(),
    )
    report = diagnostics.coverage(revised)
    assert report.availability_state == "QUALIFIED"
    assert report.n_records == 3, "every retained record counts"
    per_series = {
        entry.series_id: entry
        for entry in walk_values(report)
        if hasattr(entry, "series_id") and hasattr(entry, "n_records")
    }
    alpha = per_series["alpha"]
    assert alpha.n_records == 3
    assert alpha.n_distinct_exact_keys == 1 + 1, (
        "exact keys group on observation_time identity only: the two "
        "same-instant records with different available_times form ONE "
        "cell, so distinct keys = {2023-01-01, 2023-02-01} = 2 — never "
        "3 (available_time is not a grouping key)"
    )
    assert alpha.n_duplicate_cells == 1, (
        "the same-instant pair is a duplicate cell under "
        "observation_time identity: an implementation grouping on the "
        "availability triple would report 0 duplicates here"
    )


def test_coverage_discloses_state_and_qualifies_cleanly() -> None:
    diagnostics = import_diagnostics()

    qualified = qualified_dataset(
        [timed_record("alpha", instant(2023, m), float(m)) for m in (1, 2, 3)],
        availability=availability_policy(),
    )
    q_report = diagnostics.coverage(qualified)
    assert q_report.availability_state == "QUALIFIED"
    assert q_report.n_records == 3

    empty = diagnostics.coverage(unqualified_dataset(()))
    assert empty.n_records == 0
    assert empty.n_series == 0

    unq = diagnostics.coverage(
        unqualified_dataset((period_record("alpha", month_key(2023, 1), 1.0),))
    )
    assert unq.availability_state == "UNQUALIFIED"
    assert unq.source_sha256 is not None
    assert getattr(unq, "data_mode", None) is None


# Keep import surface honest.
_ = (fred_snapshot_dataset, has_marker, find_strings, unqualified_dataset)
