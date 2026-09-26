"""Behavior floors for ``missingness``: the explicit caller-supplied
expected observation grid (required, exact state-native keys), exact-key
comparison only, duplicate observed/expected keys failing closed, the
fixed identifiability field names, and disclosed denominator/basis."""

from __future__ import annotations

import pytest
from _synthetic import (
    QUALIFIED_KEY_BASIS,
    UNQUALIFIED_KEY_BASIS,
    availability_policy,
    field_names,
    find_strings,
    has_marker,
    import_diagnostics,
    instant,
    month_key,
    package_py_files,
    period_record,
    qualified_dataset,
    timed_record,
    unqualified_dataset,
)

MONTH_GRID = [("alpha", month_key(2023, m)) for m in range(1, 13)]

#: The fixed missingness accounting fields (S130-131).
MISSINGNESS_FIELDS = (
    "expected_n",
    "observed_n",
    "absent_expected_n",
    "null_value_n",
    "unexpected_observed_n",
    "observed_fraction_of_expected",
)


def test_missingness_requires_the_explicit_expected_grid() -> None:
    """No call without ``expected_keys``; no inference kwarg spelling
    exists; the comparison is exact-key against the supplied grid and
    the fixed accounting fields carry the right values."""
    diagnostics = import_diagnostics()

    observed = unqualified_dataset(
        tuple(period_record("alpha", month_key(2023, m), float(m)) for m in (1, 5, 9))
    )
    with pytest.raises(TypeError):
        diagnostics.missingness(observed)
    with pytest.raises(TypeError):
        diagnostics.missingness(observed, expected_keys=None)

    for inference_spelling in (
        {"infer_frequency": "monthly"},
        {"infer_calendar": True},
        {"frequency": "monthly"},
        {"records_union": True},
        {"use_union_denominator": True},
        {"expected_frequency": "M"},
    ):
        with pytest.raises(TypeError):
            diagnostics.missingness(observed, **inference_spelling)

    report = diagnostics.missingness(observed, expected_keys=MONTH_GRID)
    assert report.expected_n == 12, (
        "expected_n is exactly the caller-supplied grid size — never an "
        "inferred calendar or records union"
    )
    assert report.observed_n == 3, "three exact-key matches were observed"
    assert report.absent_expected_n == 9, (
        "the 9 expected keys with no observed record are absent_expected_n"
    )
    assert report.null_value_n == 0, "no observed record carries a null value"
    assert report.unexpected_observed_n == 0, (
        "no observed record falls outside the expected grid"
    )
    assert report.observed_fraction_of_expected == pytest.approx(3 / 12)

    # A fully observed grid reports zero missing — exact-key equality.
    full = unqualified_dataset(
        tuple(period_record("alpha", key, 1.0) for _, key in MONTH_GRID)
    )
    full_report = diagnostics.missingness(full, expected_keys=MONTH_GRID)
    assert full_report.expected_n == 12
    assert full_report.observed_n == 12
    assert full_report.absent_expected_n == 0
    assert full_report.observed_fraction_of_expected == pytest.approx(1.0)


def test_missingness_absent_null_and_unexpected_accounting_is_identifiable() -> None:
    """The three distinct missingness mechanisms remain separately
    identifiable: expected keys with no record (absent), observed null
    values at expected keys (null), and observed records at keys outside
    the grid (unexpected). A schema collapsing them into one
    'missing_count' launders completeness."""
    diagnostics = import_diagnostics()

    # UNQUALIFIED float domain admits no null values: null_value_n is
    # structurally zero there, and absent/unexpected remain identifiable.
    unq_grid = [("alpha", month_key(2023, m)) for m in range(1, 7)]
    unq = unqualified_dataset(
        (
            period_record("alpha", month_key(2023, 1), 1.0),
            period_record("alpha", month_key(2023, 2), 2.0),
            period_record("alpha", month_key(2023, 3), 3.0),
            period_record("alpha", month_key(2024, 7), 99.0),
        )
    )
    unq_report = diagnostics.missingness(unq, expected_keys=unq_grid)
    assert unq_report.expected_n == 6
    assert unq_report.observed_n == 3
    assert unq_report.absent_expected_n == 3
    assert unq_report.null_value_n == 0, (
        "the UNQUALIFIED float value domain admits no null values: "
        "null_value_n is structurally zero"
    )
    assert unq_report.unexpected_observed_n == 1
    assert unq_report.observed_fraction_of_expected == pytest.approx(3 / 6)

    # QUALIFIED value domain admits None: the present-null record at an
    # expected key counts observed AND null — never silently dropped
    # from the denominator nor coerced into an absent cell.
    q = qualified_dataset(
        [timed_record("alpha", instant(2023, m), float(m)) for m in (1, 2, 3)]
        + [timed_record("alpha", instant(2023, 4), None)]
        + [timed_record("alpha", instant(2025, 7), 99.0)],
        availability=availability_policy(),
    )
    q_grid = [("alpha", instant(2023, m)) for m in range(1, 7)]
    report = diagnostics.missingness(q, expected_keys=q_grid)
    assert report.expected_n == 6
    assert report.observed_n == 4, (
        "observed_n counts exact-key observed records, including the null-valued one"
    )
    assert report.absent_expected_n == 2, (
        "expected keys 2023-05 and 2023-06 carry no observed record"
    )
    assert report.null_value_n == 1, (
        "exactly one observed record carries a null value at an expected key"
    )
    assert report.unexpected_observed_n == 1, (
        "exactly one observed record sits outside the expected grid"
    )
    assert report.observed_fraction_of_expected == pytest.approx(4 / 6)


def test_missingness_expected_series_with_no_records_counts_as_absent() -> None:
    """An expected key naming a series with no retained records at all
    still counts in expected_n and its cells count as absent — the
    expected grid is caller authority, not the dataset's series set."""
    diagnostics = import_diagnostics()

    dataset = unqualified_dataset(
        tuple(period_record("alpha", month_key(2023, m), float(m)) for m in (1, 2, 3))
    )
    grid = [("alpha", month_key(2023, m)) for m in (1, 2, 3)] + [
        ("ghost", month_key(2023, m)) for m in (1, 2)
    ]
    report = diagnostics.missingness(dataset, expected_keys=grid)
    assert report.expected_n == 5
    assert report.observed_n == 3
    assert report.absent_expected_n == 2, (
        "the ghost series' two expected cells are absent: an expected "
        "series with no records is missing, never dropped from the "
        "denominator"
    )
    assert report.unexpected_observed_n == 0
    assert report.observed_fraction_of_expected == pytest.approx(3 / 5)


def test_missingness_compares_on_exact_state_native_keys() -> None:
    """UNQUALIFIED pairs on the verbatim period-key string; QUALIFIED
    pairs on the normalized observation-time instant; keys are compared
    exactly — a label never converts into an instant or vice versa, and
    near-miss labels never match or coerce."""
    diagnostics = import_diagnostics()

    unq = unqualified_dataset(
        (
            period_record("alpha", month_key(2023, 1), 1.0),
            period_record("alpha", month_key(2023, 2), 2.0),
        )
    )
    report = diagnostics.missingness(
        unq, expected_keys=[("alpha", month_key(2023, m)) for m in (1, 2, 3)]
    )
    assert report.availability_state == "UNQUALIFIED"
    strings = find_strings(report)
    assert any("period_key" in leaf for leaf in strings), (
        "the UNQUALIFIED report must disclose the state-native key basis "
        "(period_key, verbatim provider label)"
    )
    assert has_marker(report, UNQUALIFIED_KEY_BASIS), (
        "the report must carry the exact UNQUALIFIED key-basis name"
    )

    q = qualified_dataset(
        [timed_record("alpha", instant(2023, m), float(m)) for m in (1, 2)],
        availability=availability_policy(),
    )
    q_report = diagnostics.missingness(
        q,
        expected_keys=[("alpha", instant(2023, m)) for m in (1, 2, 3)],
    )
    assert q_report.availability_state == "QUALIFIED"
    q_strings = find_strings(q_report)
    assert any("observation_time" in leaf for leaf in q_strings), (
        "the QUALIFIED report must disclose the state-native key basis "
        "(observation_time, the normalized identity)"
    )
    assert has_marker(q_report, QUALIFIED_KEY_BASIS), (
        "the report must carry the exact QUALIFIED key-basis name"
    )

    # Exact labels: "202301" never matches a January-2023 instant grid —
    # the string grid pairs with the string-keyed state only.
    mixed = diagnostics.missingness(unq, expected_keys=[("alpha", month_key(2023, 1))])
    assert mixed.expected_n == 1 and mixed.observed_n == 1

    # Near-miss labels never match: "2023-1" is not "2023-01".
    near_miss = unqualified_dataset((period_record("alpha", "2023-1", 5.0),))
    near_report = diagnostics.missingness(
        near_miss, expected_keys=[("alpha", "2023-01")]
    )
    assert near_report.observed_n == 0, (
        "the near-miss label '2023-1' must never match '2023-01': no "
        "date parsing, no coercion, no fuzzy matching"
    )
    assert near_report.unexpected_observed_n == 1, (
        "the unmatched observed record is unexpected, not silently matched"
    )
    near_padded = diagnostics.missingness(
        unqualified_dataset((period_record("alpha", "202301", 5.0),)),
        expected_keys=[("alpha", "202301")],
    )
    assert near_padded.observed_n == 1, "the exact label matches exactly"


def test_missingness_duplicate_keys_fail_closed() -> None:
    """Duplicate expected keys reject and duplicate observed cells reject
    (ValueError): duplicates are never averaged, deduplicated, or
    otherwise silently resolved into the accounting."""
    diagnostics = import_diagnostics()

    observed = unqualified_dataset(
        (
            period_record("alpha", month_key(2023, 1), 1.0),
            period_record("alpha", month_key(2023, 2), 2.0),
        )
    )

    with pytest.raises(ValueError):
        diagnostics.missingness(
            observed,
            expected_keys=[
                ("alpha", month_key(2023, 1)),
                ("alpha", month_key(2023, 1)),
            ],
        )

    duplicate_observed = unqualified_dataset(
        (
            period_record("alpha", month_key(2023, 1), 1.0),
            period_record("alpha", month_key(2023, 1), 1.5),
            period_record("alpha", month_key(2023, 2), 2.0),
        )
    )
    with pytest.raises(ValueError):
        diagnostics.missingness(
            duplicate_observed,
            expected_keys=[
                ("alpha", month_key(2023, 1)),
                ("alpha", month_key(2023, 2)),
            ],
        )


def test_missingness_wrong_type_keys_reject_fail_closed() -> None:
    """Expected keys of the wrong shape/type reject TypeError rather
    than being coerced into matches — the exact-key law must not be
    escapable by handing the comparison an int/bytes/object key."""
    diagnostics = import_diagnostics()

    observed = unqualified_dataset((period_record("alpha", month_key(2023, 1), 1.0),))
    for bad in (
        202301,  # bare int where a label string belongs
        b"202301",  # bytes are never str
        202301.0,  # float is never an int-like label
        ("alpha",),  # wrong tuple arity
        ("alpha", month_key(2023, 1), "extra"),  # wrong tuple arity
    ):
        with pytest.raises(TypeError):
            diagnostics.missingness(observed, expected_keys=[bad])

    # Non-iterable/non-sequence containers reject too.
    for bad in (202301, None):
        with pytest.raises(TypeError):
            diagnostics.missingness(observed, expected_keys=bad)


def test_missingness_discloses_state_provenance_and_basis() -> None:
    diagnostics = import_diagnostics()

    observed = unqualified_dataset(
        tuple(period_record("alpha", month_key(2023, m), float(m)) for m in (1, 2, 3))
    )
    report = diagnostics.missingness(observed, expected_keys=MONTH_GRID[:3])
    assert report.availability_state == "UNQUALIFIED"
    assert report.source_sha256 == observed.source_sha256
    names = field_names(report)
    lowered = {name.lower() for name in names}
    assert any("basis" in name for name in lowered) or any(
        "expected" in name for name in lowered
    ), (
        f"the report must disclose the comparison basis (caller-supplied "
        f"expected grid); fields: {lowered}"
    )
    assert has_marker(report, "expected_keys"), (
        "the report must name 'expected_keys' as the denominator basis"
    )

    empty = diagnostics.missingness(unqualified_dataset(()), expected_keys=[])
    assert empty.expected_n == 0
    assert empty.observed_n == 0
    assert empty.observed_fraction_of_expected is None, (
        "observed_fraction_of_expected is None when expected_n == 0 — "
        "a 0/0 fraction is never fabricated as 0.0"
    )


def test_missingness_never_infers_a_calendar_or_reparses_sentinels() -> None:
    """No grid inference for any input: a dataset with an obvious
    monthly pattern plus a grid whose interior labels are missing
    reports those cells absent — never extrapolated. Source scan
    (semantic, AST-based — docstring tokens cannot false-positive): no
    sentinel-constant comparison logic, no date/calendar construction
    or parsing call, and no presence-shaped hook on ResearchDataset."""
    import ast

    diagnostics = import_diagnostics()

    # Behavior: an obviously monthly sequence (Jan/Feb/Mar observed)
    # with a grid covering Jan..Jun reports Apr/May/Jun absent — the
    # interior gap is never filled by extrapolating the pattern.
    observed = unqualified_dataset(
        tuple(period_record("alpha", month_key(2023, m), float(m)) for m in (1, 2, 3))
    )
    grid = [("alpha", month_key(2023, m)) for m in range(1, 7)]
    report = diagnostics.missingness(observed, expected_keys=grid)
    assert report.absent_expected_n == 3, (
        "the three unobserved interior months are absent expected cells: "
        "no calendar extrapolation from the monthly pattern"
    )
    assert report.observed_n == 3

    # Source scan 1: no sentinel-constant comparison anywhere — the
    # provider sentinels (FF -99.99/-999, FRED ".") were dropped at
    # decode time by the adapters; diagnostics never re-parse them.
    sentinel_numbers = {-99.99, -999.0, -999, 99.99, -9999.0, -9999}
    sentinel_strings = {"."}
    for path in package_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            for comparator in (node.left, *node.comparators):
                if isinstance(comparator, ast.Constant) and (
                    (
                        isinstance(comparator.value, (int, float))
                        and not isinstance(comparator.value, bool)
                        and float(comparator.value) in sentinel_numbers
                    )
                    or (
                        isinstance(comparator.value, str)
                        and comparator.value in sentinel_strings
                    )
                ):
                    raise AssertionError(
                        f"{path.name} compares against the sentinel "
                        f"constant {comparator.value!r}: provider "
                        "sentinel reparsing is forbidden in diagnostics"
                    )

    # Source scan 2: no date/calendar construction or parsing call.
    banned_calls = (
        "date_range",
        "date",
        "datetime",
        "Timestamp",
        "to_datetime",
        "strptime",
        "fromisoformat",
    )
    for path in package_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else (
                        node.func.attr if isinstance(node.func, ast.Attribute) else None
                    )
                )
                assert name not in banned_calls, (
                    f"{path.name} constructs or parses {name!r}: "
                    "diagnostics never infer a calendar or date range"
                )

    # No presence-shaped hook on ResearchDataset: the dataset classes
    # expose no injected presence callable of any spelling.
    from portlearn.data.dataset import (
        QualifiedDataset,
        ResearchDataset,
        UnqualifiedDataset,
    )

    for state_cls in (ResearchDataset, UnqualifiedDataset, QualifiedDataset):
        for name in dir(state_cls):
            assert "presence" not in name.lower(), (
                f"{state_cls.__name__} exposes the presence-shaped "
                f"attribute {name!r}: no _presence hook may exist"
            )


def test_missingness_report_carries_the_frozen_accounting_fields() -> None:
    """The report dataclass carries exactly the fixed accounting field
    names (S130-131) — no invented spellings shadow the fixed schema."""
    diagnostics = import_diagnostics()

    report = diagnostics.missingness(
        unqualified_dataset((period_record("alpha", month_key(2023, 1), 1.0),)),
        expected_keys=[("alpha", month_key(2023, 1))],
    )
    names = field_names(report)
    missing = [name for name in MISSINGNESS_FIELDS if name not in names]
    assert not missing, (
        f"the missingness report must carry the fixed accounting fields "
        f"{MISSINGNESS_FIELDS}; missing {missing}; fields present: "
        f"{sorted(names)}"
    )
    # The invented names from the vetoed suite must not shadow the law.
    for invented in ("denominator", "present_count", "missing_count", "missing_keys"):
        assert invented not in names or invented in MISSINGNESS_FIELDS, (
            f"the invented field {invented!r} must not appear alongside "
            "the fixed accounting schema"
        )


# Keep import surface honest.
_ = (find_strings, has_marker, field_names)
