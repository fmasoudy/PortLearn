"""Behavior floors for ``correlation``: exact state-native-key alignment,
the ``method``/``deletion`` keyword laws (Pearson/Spearman as separately
selected methods, listwise default with pairwise as the explicit opt-in),
the frozen ``min_overlap=3`` floor, duplicate-cell and ``series=``
fail-closed laws, Spearman average ranks, and permutation invariance."""

from __future__ import annotations

import math

import pytest
from _synthetic import (
    QUALIFIED_KEY_BASIS,
    UNQUALIFIED_KEY_BASIS,
    availability_policy,
    carries_value,
    field_names,
    find_strings,
    has_marker,
    import_diagnostics,
    instant,
    month_key,
    period_record,
    permuted,
    qualified_dataset,
    timed_record,
    unqualified_dataset,
    walk_values,
)


def _two_series_dataset(alpha_values, beta_values, keys=(1, 2, 3, 4)):
    return unqualified_dataset(
        tuple(
            period_record("alpha", month_key(2023, k), a)
            for k, a in zip(keys, alpha_values)
        )
        + tuple(
            period_record("beta", month_key(2023, k), b)
            for k, b in zip(keys, beta_values)
        )
    )


def _pair_entries(report):
    entries = getattr(report, "pairs", None)
    if entries is None:
        entries = [item for item in walk_values(report) if hasattr(item, "n")]
    assert entries, "the correlation report carries no per-pair entries"
    return entries


def _pair_for(report, first, second):
    for entry in _pair_entries(report):
        if (
            (
                getattr(entry, "first", None) == first
                and getattr(entry, "second", None) == second
            )
            or (
                getattr(entry, "first", None) == second
                and getattr(entry, "second", None) == first
            )
            or (
                getattr(entry, "series_1", None) == first
                and getattr(entry, "series_2", None) == second
            )
            or (
                getattr(entry, "series_1", None) == second
                and getattr(entry, "series_2", None) == first
            )
        ):
            return entry
    raise AssertionError(f"no pair entry for ({first!r}, {second!r})")


def _coefficient(entry, method):
    """The pair coefficient under the selected method.

    The per-pair disclosure is ``coefficient or None+reason``; the exact
    attribute spelling of the coefficient is implementation detail, so
    ``coefficient`` and the selected method name are both accepted
    spellings — the value oracle, not the field name, is the contract.
    """
    for spelling in ("coefficient", method):
        if hasattr(entry, spelling):
            return getattr(entry, spelling)
    raise AssertionError(
        f"the pair entry discloses no coefficient under method {method!r}"
    )


def _discloses_no_availability_read(strings) -> bool:
    return any(
        "availab" in leaf.lower()
        and any(word in leaf.lower() for word in ("not", "never", "no "))
        for leaf in strings
    )


def _pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy)


def _spearman(xs, ys):
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks_ = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks_[order[k]] = average
            i = j + 1
        return ranks_

    return _pearson(ranks(xs), ranks(ys))


def _three_series_partial_overlap_dataset():
    """alpha {1,2,3,4}, beta {2,3,4,5}, gamma {3,4,5,6} — the listwise
    intersection of all three is {3,4} (n=2, below the floor) while the
    pairwise intersections alpha/beta and beta/gamma hold three keys."""
    return unqualified_dataset(
        tuple(
            period_record("alpha", month_key(2023, k), float(k)) for k in (1, 2, 3, 4)
        )
        + tuple(
            period_record("beta", month_key(2023, k), 10.0 * k) for k in (2, 3, 4, 5)
        )
        + tuple(
            period_record("gamma", month_key(2023, k), -3.0 * (k - 2))
            for k in (3, 4, 5, 6)
        )
    )


def test_correlation_defaults_to_listwise_deletion() -> None:
    """With records missing at different keys, listwise deletion keeps
    only keys shared by BOTH series (n=3 here), and the default report
    is the listwise one — never pandas-style pairwise-by-default."""
    diagnostics = import_diagnostics()

    # alpha on {1,2,3,4}, beta on {2,3,4} (no beta record at key 1) —
    # the listwise-intersection semantics probe.
    dataset = unqualified_dataset(
        (
            period_record("alpha", month_key(2023, 1), 1.0),
            period_record("alpha", month_key(2023, 2), 2.0),
            period_record("alpha", month_key(2023, 3), 3.0),
            period_record("alpha", month_key(2023, 4), 10.0),
            period_record("beta", month_key(2023, 2), 20.0),
            period_record("beta", month_key(2023, 3), 30.0),
            period_record("beta", month_key(2023, 4), 40.0),
        )
    )
    report = diagnostics.correlation(dataset)
    entry = _pair_for(report, "alpha", "beta")
    assert entry.n == 3, (
        f"listwise deletion must keep only the shared keys {{2,3,4}}; got n={entry.n}"
    )
    # Over shared keys alpha=(2,3,10) vs beta=(20,30,40): exact Pearson.
    expected = _pearson([2.0, 3.0, 10.0], [20.0, 30.0, 40.0])
    assert _coefficient(entry, "pearson") == pytest.approx(
        expected, rel=1e-12, abs=1e-12
    )
    assert getattr(entry, "method", None) == "pearson", (
        "the default selected method is pearson and must be disclosed on the pair"
    )
    assert getattr(entry, "deletion", None) == "listwise", (
        "the default deletion is listwise and must be disclosed on the pair"
    )


def test_correlation_pairwise_deletion_is_explicit_opt_in_only() -> None:
    """``deletion="pairwise"`` is the only spelling of the opt-in; the
    default is listwise. Under listwise the aligned set is the
    intersection across ALL selected series (here {3,4}, n=2 for every
    pair, below the floor); under pairwise each pair aligns on its own
    intersection with per-pair n of 3, 3, and 2 — different disclosed
    n's for identical data. Undeclared deletion values reject
    ValueError; unknown keyword spellings reject TypeError."""
    diagnostics = import_diagnostics()
    dataset = _three_series_partial_overlap_dataset()

    default_report = diagnostics.correlation(dataset)
    for first, second in (("alpha", "beta"), ("alpha", "gamma"), ("beta", "gamma")):
        entry = _pair_for(default_report, first, second)
        assert entry.n == 2, (
            "default listwise deletion intersects the key sets of ALL "
            f"series: alpha{{1..4}} ∩ beta{{2..5}} ∩ gamma{{3..6}} = {{3,4}}; "
            f"pair ({first},{second}) got n={entry.n}"
        )
        assert getattr(entry, "deletion", None) == "listwise"
        assert _coefficient(entry, "pearson") is None, (
            "n=2 sits below the frozen min_overlap=3 floor: the "
            "coefficient is undefined, never fabricated"
        )
        assert has_marker(entry, "insufficient_overlap")

    pairwise_report = diagnostics.correlation(dataset, deletion="pairwise")
    ab = _pair_for(pairwise_report, "alpha", "beta")
    assert ab.n == 3, "pairwise alpha-beta uses their shared keys {2,3,4}"
    assert _coefficient(ab, "pearson") == pytest.approx(1.0, abs=1e-12), (
        "alpha (2,3,4) vs beta (20,30,40) is exactly linear: Pearson 1.0"
    )
    assert getattr(ab, "deletion", None) == "pairwise"
    bg = _pair_for(pairwise_report, "beta", "gamma")
    assert bg.n == 3, "pairwise beta-gamma uses their shared keys {3,4,5}"
    assert _coefficient(bg, "pearson") == pytest.approx(-1.0, abs=1e-12), (
        "beta (30,40,50) vs gamma (-3,-6,-9) is exactly inverse: Pearson -1.0"
    )
    ag = _pair_for(pairwise_report, "alpha", "gamma")
    assert ag.n == 2, "pairwise alpha-gamma shares only {3,4}"
    assert _coefficient(ag, "pearson") is None and has_marker(
        ag, "insufficient_overlap"
    ), "the pairwise n=2 pair is still below the min_overlap floor"

    # Spearman selects identically under pairwise deletion.
    spearman_report = diagnostics.correlation(
        dataset, method="spearman", deletion="pairwise"
    )
    assert _coefficient(
        _pair_for(spearman_report, "alpha", "beta"), "spearman"
    ) == pytest.approx(1.0, abs=1e-12)

    # An undeclared deletion value rejects; so does an undeclared method.
    for bad_deletion in ("rows", "pair", None, "", "PAIRWISE"):
        with pytest.raises(ValueError):
            diagnostics.correlation(dataset, deletion=bad_deletion)
    for bad_method in ("kendall", "pairwise", None, "", "spearman_only"):
        with pytest.raises(ValueError):
            diagnostics.correlation(dataset, method=bad_method)

    # Unknown keyword spellings must never be silently accepted.
    for spelling in (
        {"deletions": "pairwise"},
        {"methods": "pearson"},
        {"min_n": 3},
        {"threshold": 3},
        {"overlap_min": 3},
    ):
        with pytest.raises(TypeError):
            diagnostics.correlation(dataset, **spelling)


def test_correlation_reports_every_pair_overlap_method_and_deletion() -> None:
    """Pearson and Spearman are separately selected methods: each call
    computes one coefficient per pair and discloses the selected
    ``method`` and ``deletion`` on every entry, with the report-level
    key-basis name, state, provenance, and the statement that no
    availability information was read."""
    diagnostics = import_diagnostics()

    dataset = _two_series_dataset(
        (1.0, 2.0, 3.0, 4.0),
        (1.0, 4.0, 9.0, 16.0),
    )

    pearson_report = diagnostics.correlation(dataset, method="pearson")
    for entry in _pair_entries(pearson_report):
        assert getattr(entry, "method", None) == "pearson"
        assert getattr(entry, "deletion", None) == "listwise"
        assert isinstance(getattr(entry, "n", None), int)
        # One coefficient per pair under the SELECTED method: the frozen
        # per-pair schema carries `method` + `coefficient`, never a
        # simultaneous pearson-and-spearman field pair from one call.
        assert not hasattr(entry, "spearman"), (
            "a pearson-selected report must not simultaneously carry a "
            "spearman coefficient: Pearson and Spearman are separately "
            "selected methods"
        )
    entry = _pair_for(pearson_report, "alpha", "beta")
    assert entry.n == 4
    assert _coefficient(entry, "pearson") == pytest.approx(
        _pearson([1, 2, 3, 4], [1, 4, 9, 16]), rel=1e-12, abs=1e-12
    )
    assert _coefficient(entry, "pearson") < 1.0

    # Spearman as its own selected method on the same monotone data.
    spearman_report = diagnostics.correlation(dataset, method="spearman")
    for entry in _pair_entries(spearman_report):
        assert getattr(entry, "method", None) == "spearman"
        assert getattr(entry, "deletion", None) == "listwise"
    spearman_entry = _pair_for(spearman_report, "alpha", "beta")
    assert spearman_entry.n == 4
    assert _coefficient(spearman_entry, "spearman") == pytest.approx(
        _spearman([1, 2, 3, 4], [1, 4, 9, 16]), rel=1e-12, abs=1e-12
    )
    assert _coefficient(spearman_entry, "spearman") == pytest.approx(1.0, abs=1e-12), (
        "monotone data: Spearman is exactly 1 while Pearson is below it"
    )

    # Pearson -1 exactly on a perfectly inverse linear pair (both methods).
    inverse = _two_series_dataset((1.0, 2.0, 3.0, 4.0), (4.0, 3.0, 2.0, 1.0))
    assert _coefficient(
        _pair_for(diagnostics.correlation(inverse, method="pearson"), "alpha", "beta"),
        "pearson",
    ) == pytest.approx(-1.0, abs=1e-12)
    assert _coefficient(
        _pair_for(diagnostics.correlation(inverse, method="spearman"), "alpha", "beta"),
        "spearman",
    ) == pytest.approx(-1.0, abs=1e-12)

    # Report-level disclosures: key basis, state, provenance, and the
    # statement that no availability information was read.
    assert pearson_report.availability_state == "UNQUALIFIED"
    assert pearson_report.source_sha256 == dataset.source_sha256
    assert has_marker(pearson_report, UNQUALIFIED_KEY_BASIS), (
        "the report must disclose the alignment-key basis name for the "
        f"UNQUALIFIED state ({UNQUALIFIED_KEY_BASIS!r})"
    )
    assert _discloses_no_availability_read(find_strings(pearson_report)), (
        "the report must state that no availability information was read"
    )

    qualified = qualified_dataset(
        [timed_record("alpha", instant(2023, m), float(m)) for m in (1, 2, 3, 4)]
        + [timed_record("beta", instant(2023, m), float(m) ** 2) for m in (1, 2, 3, 4)],
        availability=availability_policy(),
    )
    q_report = diagnostics.correlation(qualified, method="spearman")
    assert q_report.availability_state == "QUALIFIED"
    assert has_marker(q_report, QUALIFIED_KEY_BASIS), (
        "the report must disclose the alignment-key basis name for the "
        f"QUALIFIED state ({QUALIFIED_KEY_BASIS!r})"
    )
    assert _discloses_no_availability_read(find_strings(q_report))


def test_correlation_undefined_for_insufficient_overlap_or_constant_series() -> None:
    """A pair below the frozen ``min_overlap=3`` floor yields ``None``
    with the machine-readable reason ``insufficient_overlap`` carrying
    ``n`` and ``min_overlap``; a constant series yields ``None`` with
    ``constant_series`` naming it; ``min_overlap < 3`` rejects
    ValueError and unknown spellings reject TypeError."""
    diagnostics = import_diagnostics()

    dataset = unqualified_dataset(
        (
            period_record("alpha", month_key(2023, 1), 1.0),
            period_record("alpha", month_key(2023, 2), 2.0),
            period_record("beta", month_key(2023, 1), 10.0),
            period_record("beta", month_key(2023, 2), 20.0),
        )
    )
    for method in ("pearson", "spearman"):
        report = diagnostics.correlation(dataset, method=method)
        entry = _pair_for(report, "alpha", "beta")
        assert entry.n == 2
        assert _coefficient(entry, method) is None, (
            "overlap below the min_overlap=3 floor leaves coefficients "
            "undefined (None), never a fabricated value"
        )
        assert has_marker(entry, "insufficient_overlap"), (
            "the undefined pair must carry the machine-readable reason "
            "'insufficient_overlap'"
        )
        assert (getattr(entry, "min_overlap", None) == 3) or carries_value(entry, 3), (
            "the insufficient_overlap reason must carry min_overlap"
        )

    # The floor cannot be lowered by the caller: below 3 rejects
    # ValueError (never TypeError — the spelling exists, the value is
    # refused), and no alternative spelling accepts a lower threshold.
    for attempt in (2, 1, 0, -5):
        with pytest.raises(ValueError):
            diagnostics.correlation(dataset, min_overlap=attempt)
    for spelling in ({"min_n": 2}, {"threshold": 2}, {"minoverlap": 3}):
        with pytest.raises(TypeError):
            diagnostics.correlation(dataset, **spelling)

    # A series constant on its aligned support: zero denominator, None
    # coefficient with the reason naming the series — never NaN, never
    # a silent 0 or ±1.
    constant = _two_series_dataset((5.0, 5.0, 5.0, 5.0), (1.0, 4.0, 9.0, 16.0))
    for method in ("pearson", "spearman"):
        entry = _pair_for(
            diagnostics.correlation(constant, method=method), "alpha", "beta"
        )
        assert _coefficient(entry, method) is None
        assert has_marker(entry, "constant_series"), (
            "the undefined pair must carry the machine-readable reason "
            "'constant_series'"
        )
        assert "alpha" in find_strings(entry), (
            "the constant_series reason must name the constant series"
        )


def test_correlation_aligns_on_exact_state_native_keys() -> None:
    """Keyed-vs-positional misalignment killer: two series sharing
    identical value sequences on disjoint keys produce NO coefficient
    (overlap 0) where a positional join would fabricate one; UNQUALIFIED
    joins on the verbatim period_key, QUALIFIED on the normalized
    observation-time identity; hand-computed r values match."""
    diagnostics = import_diagnostics()

    dataset = unqualified_dataset(
        (
            period_record("alpha", month_key(2023, 1), 1.0),
            period_record("alpha", month_key(2023, 2), 2.0),
            period_record("alpha", month_key(2023, 3), 3.0),
            period_record("beta", month_key(2023, 1), 10.0),
            period_record("beta", month_key(2023, 2), 20.0),
            period_record("beta", month_key(2023, 3), 30.0),
            period_record("beta", month_key(2024, 1), 40.0),
        )
    )
    report = diagnostics.correlation(dataset)
    entry = _pair_for(report, "alpha", "beta")
    assert entry.n == 3, "pairing keys on the exact period_key string"
    assert _coefficient(entry, "pearson") == pytest.approx(
        _pearson([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]), rel=1e-12, abs=1e-12
    )

    # Disjoint keys, identical value sequences: exact-key alignment
    # leaves overlap 0 — a positional/pandas-style join would fabricate
    # a perfect coefficient here.
    mismatched = unqualified_dataset(
        tuple(period_record("alpha", month_key(2023, m), float(m)) for m in (1, 2, 3))
        + tuple(period_record("beta", month_key(2024, m), float(m)) for m in (1, 2, 3))
    )
    mismatched_entry = _pair_for(diagnostics.correlation(mismatched), "alpha", "beta")
    assert mismatched_entry.n == 0, (
        "disjoint exact keys must leave zero aligned pairs — values are "
        "never positionally joined"
    )
    assert _coefficient(mismatched_entry, "pearson") is None
    assert has_marker(mismatched_entry, "insufficient_overlap")

    # QUALIFIED state: pairing keys on the normalized observation_time.
    qualified = qualified_dataset(
        [timed_record("alpha", instant(2023, m), float(m)) for m in (1, 2, 3)]
        + [timed_record("beta", instant(2023, m), 10.0 * m) for m in (1, 2, 3)],
        availability=availability_policy(),
    )
    q_report = diagnostics.correlation(qualified)
    q_entry = _pair_for(q_report, "alpha", "beta")
    assert q_entry.n == 3
    assert _coefficient(q_entry, "pearson") == pytest.approx(1.0, abs=1e-12)

    # Different labels must never cross-pair: UNQUALIFIED "202401" and
    # QUALIFIED 2024-01 instants are distinct state-native keys.
    assert report.availability_state == "UNQUALIFIED"
    assert q_report.availability_state == "QUALIFIED"


def test_correlation_self_pairs_and_single_series_reject_or_undefined() -> None:
    diagnostics = import_diagnostics()

    dataset = _two_series_dataset((1.0, 2.0, 3.0, 4.0), (1.0, 4.0, 9.0, 16.0))
    report = diagnostics.correlation(dataset)
    self_entry = _pair_for(report, "alpha", "alpha")
    if _coefficient(self_entry, "pearson") is not None:
        assert _coefficient(self_entry, "pearson") == pytest.approx(1.0, abs=1e-12)
        spearman_self = diagnostics.correlation(dataset, method="spearman")
        assert _coefficient(
            _pair_for(spearman_self, "alpha", "alpha"), "spearman"
        ) == pytest.approx(1.0, abs=1e-12)
    assert _pair_for(report, "beta", "beta").n == 4

    single = unqualified_dataset(
        tuple(period_record("only", month_key(2023, m), float(m)) for m in (1, 2, 3, 4))
    )
    single_report = diagnostics.correlation(single)
    entry = _pair_for(single_report, "only", "only")
    assert entry.n == 4
    if _coefficient(entry, "pearson") is not None:
        assert _coefficient(entry, "pearson") == pytest.approx(1.0, abs=1e-12)

    assert find_strings(report), "the report carries disclosure strings"
    names = field_names(report)
    assert (
        any("deletion" in name.lower() for name in names)
        or has_marker(report, "listwise")
        or has_marker(report, "pairwise")
    ), "the report must disclose the deletion method actually used"


def test_correlation_duplicate_series_or_key_fails_closed() -> None:
    """A ``(series_id, key)`` cell appearing more than once within a
    series rejects ValueError naming the cell — duplicates are never
    silently averaged or set-collapsed into an estimand; duplicate ids
    in ``series=`` reject; an unknown series id rejects naming it; the
    ``series=`` selector restricts the report to the selected pairs."""
    diagnostics = import_diagnostics()

    duplicate_cell = unqualified_dataset(
        (
            period_record("alpha", month_key(2023, 1), 1.0),
            period_record("alpha", month_key(2023, 2), 2.0),
            period_record("alpha", month_key(2023, 3), 3.0),
            period_record("alpha", month_key(2023, 3), 3.5),
            period_record("beta", month_key(2023, 1), 10.0),
            period_record("beta", month_key(2023, 2), 20.0),
            period_record("beta", month_key(2023, 3), 30.0),
        )
    )
    with pytest.raises(ValueError) as excinfo:
        diagnostics.correlation(duplicate_cell)
    message = str(excinfo.value)
    assert "alpha" in message and month_key(2023, 3) in message, (
        "the duplicate-cell rejection must name the series and the exact "
        f"key of the duplicated cell; got {message!r}"
    )

    # Identical values at a duplicated cell still fail closed: the
    # ambiguity is the duplicated key, not the value difference.
    identical_values = unqualified_dataset(
        (
            period_record("alpha", month_key(2023, 1), 1.0),
            period_record("alpha", month_key(2023, 1), 1.0),
            period_record("alpha", month_key(2023, 2), 2.0),
            period_record("beta", month_key(2023, 1), 10.0),
            period_record("beta", month_key(2023, 2), 20.0),
        )
    )
    with pytest.raises(ValueError):
        diagnostics.correlation(identical_values)

    dataset = _three_series_partial_overlap_dataset()

    # The selector restricts alignment to the selected series: with
    # {alpha, beta} the listwise common set is {2,3,4} (n=3, Pearson 1)
    # and no gamma pair exists on the report.
    selected = diagnostics.correlation(dataset, series=["alpha", "beta"])
    selected_entry = _pair_for(selected, "alpha", "beta")
    assert selected_entry.n == 3, (
        "the listwise intersection is computed over the SELECTED series "
        "only: alpha{1..4} ∩ beta{2..5} = {2,3,4}"
    )
    assert _coefficient(selected_entry, "pearson") == pytest.approx(1.0, abs=1e-12)
    assert has_marker(selected, "alpha") and has_marker(selected, "beta"), (
        "the report must disclose the selected series list"
    )
    with pytest.raises(AssertionError):
        (
            _pair_for(selected, "beta", "gamma"),
            ("no pair involving an unselected series may appear on the report"),
        )

    # Duplicate ids in the selector reject: the selection is ambiguous.
    with pytest.raises(ValueError):
        diagnostics.correlation(dataset, series=["alpha", "alpha", "beta"])

    # An unknown id rejects ValueError naming it.
    with pytest.raises(ValueError) as unknown:
        diagnostics.correlation(dataset, series=["alpha", "ghost"])
    assert "ghost" in str(unknown.value), (
        "the unknown-series rejection must name the unknown id"
    )


def test_correlation_spearman_uses_average_ranks() -> None:
    """Hand-computed Spearman with ties in both series: tied values
    share the mean of their rank positions, then the Pearson formula
    applies to the ranks. The oracle kills ordinal (distinct/competition)
    ranking, which would yield exactly 1.0 here."""
    diagnostics = import_diagnostics()

    # x = (1, 2, 2, 3, 4) -> ranks (1, 2.5, 2.5, 4, 5)
    # y = (10, 10, 20, 30, 30) -> ranks (1.5, 1.5, 3, 4.5, 4.5)
    # cov(ranks) = 8.25, var(rx) = 9.5, var(ry) = 9.0
    # r = 8.25 / sqrt(9.5 * 9.0)  (hand-computed, not a library call)
    expected = 8.25 / math.sqrt(9.5 * 9.0)
    dataset = unqualified_dataset(
        tuple(
            period_record("alpha", month_key(2023, k), x)
            for k, x in enumerate((1.0, 2.0, 2.0, 3.0, 4.0), start=1)
        )
        + tuple(
            period_record("beta", month_key(2023, k), y)
            for k, y in enumerate((10.0, 10.0, 20.0, 30.0, 30.0), start=1)
        )
    )

    entry = _pair_for(
        diagnostics.correlation(dataset, method="spearman"), "alpha", "beta"
    )
    assert entry.n == 5
    assert _coefficient(entry, "spearman") == pytest.approx(expected, rel=1e-12), (
        "tied values must share the average of their rank positions"
    )
    assert abs(_coefficient(entry, "spearman") - 1.0) > 1e-9, (
        "an ordinal (distinct-rank) implementation yields exactly 1.0 "
        "here; average ranks yield strictly less"
    )
    # Cross-check the hand computation against the test-local reference.
    assert _coefficient(entry, "spearman") == pytest.approx(
        _spearman([1.0, 2.0, 2.0, 3.0, 4.0], [10.0, 10.0, 20.0, 30.0, 30.0]),
        rel=1e-12,
    )


def test_unqualified_diagnostics_outputs_are_permutation_invariant() -> None:
    """Permuting the record order of an UNQUALIFIED dataset yields
    value-equal describe/coverage/correlation reports: no order-dependent
    statistic exists on opaque labels (the same retained bytes, the same
    statistics)."""
    diagnostics = import_diagnostics()

    dataset = _two_series_dataset((1.0, 2.0, 3.0, 4.0), (10.0, 20.0, 30.0, 40.0))
    baseline = {
        "describe": diagnostics.describe(dataset),
        "coverage": diagnostics.coverage(dataset),
        "correlation": diagnostics.correlation(dataset),
    }
    for seed in (7, 13):
        shuffled = permuted(dataset, seed)
        assert diagnostics.describe(shuffled) == baseline["describe"], (
            f"describe must be invariant to record order (seed {seed})"
        )
        assert diagnostics.coverage(shuffled) == baseline["coverage"], (
            f"coverage must be invariant to record order (seed {seed})"
        )
        assert diagnostics.correlation(shuffled) == baseline["correlation"], (
            f"correlation must be invariant to record order (seed {seed})"
        )


# availability_policy/timed_record exercised above; keep imports honest.
_ = (availability_policy, timed_record, field_names)
