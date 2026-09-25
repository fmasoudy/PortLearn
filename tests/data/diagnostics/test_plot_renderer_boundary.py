"""Behavior floors for ``plot``: the renderer-neutral immutable
``PlotSpec``, the exact source/kind matrix, the fixed footnote and
disclosure fields, verbatim report-value reuse (no second statistics
implementation), UNQUALIFIED axes never datetime, and
``PlotSpec.render()`` as the sole public matplotlib boundary behind the
optional ``[plot]`` extra."""

from __future__ import annotations

import ast
import subprocess
import sys
from datetime import date, datetime

import pytest
from _synthetic import (
    PLOT_FOOTNOTE,
    REPOSITORY_ROOT,
    any_scope_runtime_imports,
    availability_policy,
    find_strings,
    function_scope_imports,
    import_diagnostics,
    instant,
    month_key,
    package_py_files,
    period_record,
    qualified_dataset,
    timed_record,
    unqualified_dataset,
    walk_values,
)

KINDS = ("describe", "correlation", "coverage", "missingness", "series")

#: The four report-producing kinds (``series`` is dataset-only).
REPORT_KINDS = ("describe", "correlation", "coverage", "missingness")

#: The statistics blocks plot must route a dataset source through.
_STATISTICS_BLOCKS = ("describe", "correlation", "coverage", "missingness")

#: Trivial numeric constants exempt from the verbatim-reuse subset law
#: (axis flags, identity constants and the like carry no information).
_TRIVIAL_NUMBERS = (0.0, 1.0)


def _dataset():
    return unqualified_dataset(
        (
            period_record("alpha", month_key(2023, 1), 1.0),
            period_record("alpha", month_key(2023, 2), 2.0),
            period_record("alpha", month_key(2023, 3), 3.0),
            period_record("beta", month_key(2023, 1), 10.0),
            period_record("beta", month_key(2023, 2), 20.0),
            period_record("beta", month_key(2023, 3), 25.0),
        )
    )


def _qualified():
    return qualified_dataset(
        [timed_record("alpha", instant(2023, m), float(m)) for m in (1, 2, 3)],
        availability=availability_policy(),
    )


def _report_grid():
    return [("alpha", month_key(2023, m)) for m in (1, 2, 3)] + [
        ("beta", month_key(2023, m)) for m in (1, 2, 3)
    ]


def _numeric_leaves(value):
    return {
        float(item)
        for item in walk_values(value)
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    }


def _nontrivial_numeric_leaves(value):
    return {item for item in _numeric_leaves(value) if item not in _TRIVIAL_NUMBERS}


def test_plot_returns_renderer_neutral_immutable_plot_spec() -> None:
    diagnostics = import_diagnostics()
    import dataclasses

    spec = diagnostics.plot(_dataset())
    assert dataclasses.is_dataclass(spec), (
        f"plot must return a typed PlotSpec value; got {type(spec).__name__}"
    )
    attribute = next(field.name for field in dataclasses.fields(spec))
    with pytest.raises((AttributeError, TypeError)):
        setattr(spec, attribute, "x")

    kind = getattr(spec, "kind", None)
    assert kind in KINDS, f"default kind must be one of {KINDS}; got {kind!r}"
    assert find_strings(spec), "the spec must carry disclosure strings"

    equal = diagnostics.plot(_dataset())
    assert spec == equal, "value-equal datasets yield value-equal specs"
    assert spec != diagnostics.plot(_dataset(), kind="correlation")


def test_plot_kind_and_source_matrix_is_exact() -> None:
    diagnostics = import_diagnostics()

    dataset = _dataset()
    describe_report = diagnostics.describe(dataset)
    correlation_report = diagnostics.correlation(dataset)
    coverage_report = diagnostics.coverage(dataset)
    missingness_report = diagnostics.missingness(dataset, expected_keys=_report_grid())

    # Dataset source: all five kinds valid. missingness forwards its
    # required expected_keys to the public missingness function.
    for kind in KINDS:
        if kind == "missingness":
            spec = diagnostics.plot(dataset, kind=kind, expected_keys=_report_grid())
        else:
            spec = diagnostics.plot(dataset, kind=kind)
        assert getattr(spec, "kind", None) == kind

    # missingness from a dataset without expected_keys is the §4.5
    # TypeError, forwarded unchanged — plot adds no default grid.
    with pytest.raises(TypeError):
        diagnostics.plot(dataset, kind="missingness")

    # Report source: each report plots its own kind only.
    for report, kind in (
        (describe_report, "describe"),
        (correlation_report, "correlation"),
        (coverage_report, "coverage"),
        (missingness_report, "missingness"),
    ):
        spec = diagnostics.plot(report, kind=kind)
        assert getattr(spec, "kind", None) == kind
        mismatched = next(k for k in KINDS if k != kind)
        with pytest.raises(ValueError):
            diagnostics.plot(report, kind=mismatched)

    # 'series' is dataset-only: no report source may plot as 'series'.
    with pytest.raises(ValueError):
        diagnostics.plot(describe_report, kind="series")

    # Unknown kinds reject unconditionally.
    for bad in ("box", "scatter", "hist", "", "DESCRIBE", None, 5):
        with pytest.raises((TypeError, ValueError)):
            diagnostics.plot(dataset, kind=bad)

    # Foreign sources reject TypeError — the accepted sources are
    # explicitly enumerated (one dataset or one report).
    for foreign in ("dataset", [dataset], 5, {"kind": "describe"}, None):
        with pytest.raises(TypeError):
            diagnostics.plot(foreign)


def test_plot_series_kind_carries_retained_series_labels_without_recompute() -> None:
    diagnostics = import_diagnostics()

    spec = diagnostics.plot(_dataset(), kind="series")
    strings = find_strings(spec)
    assert "alpha" in strings and "beta" in strings, (
        "the series spec must carry the retained series identifiers"
    )
    for item in walk_values(spec):
        assert not hasattr(item, "pearson"), (
            "the series spec carries no recomputed correlation facts"
        )


def test_plot_spec_carries_the_exact_fixed_footnote_and_disclosure() -> None:
    diagnostics = import_diagnostics()

    spec = diagnostics.plot(_dataset())
    strings = find_strings(spec)
    assert PLOT_FOOTNOTE in strings, (
        f"every spec must carry the exact fixed footnote {PLOT_FOOTNOTE!r}"
    )
    joined = " ".join(strings)
    assert "not" in joined.lower() and ("eligib" in joined.lower()), (
        "the footnote must state the not-decision-time-eligibility limit"
    )
    assert getattr(spec, "availability_state", None) == "UNQUALIFIED"
    qualified_spec = diagnostics.plot(_qualified())
    assert getattr(qualified_spec, "availability_state", None) == "QUALIFIED", (
        "the QUALIFIED spec must visibly display its state"
    )


def test_plot_of_report_reuses_report_values_verbatim() -> None:
    """``plot(report)`` consumes the report's already-computed values
    verbatim and never recalculates: the dataset path routes through
    exactly the public diagnostic functions (a monkeypatched block
    changes the dataset path), the report path is untouched by the
    monkeypatch, ``plot(dataset)`` is PlotSpec-value-equal to
    ``plot(report)`` for the four report-producing kinds, and every
    non-trivial numeric leaf the spec displays appears in the report —
    a second statistics implementation cannot invent values the report
    never computed. W8 killer."""
    diagnostics = import_diagnostics()

    dataset = _dataset()
    reports = {
        "describe": diagnostics.describe(dataset),
        "correlation": diagnostics.correlation(dataset),
        "coverage": diagnostics.coverage(dataset),
        "missingness": diagnostics.missingness(dataset, expected_keys=_report_grid()),
    }

    for kind, report in reports.items():
        # (a) plot(dataset) == plot(report) by PlotSpec value equality.
        # missingness on a dataset requires expected_keys, same
        # conditional forwarding as the kind matrix; other kinds stay bare.
        if kind == "missingness":
            from_dataset = diagnostics.plot(
                dataset, kind=kind, expected_keys=_report_grid()
            )
        else:
            from_dataset = diagnostics.plot(dataset, kind=kind)
        from_report = diagnostics.plot(report, kind=kind)
        assert from_dataset == from_report, (
            f"plot(dataset, kind={kind!r}) must equal plot(report) by "
            "PlotSpec value equality: the report source is a memo of the "
            "same statistics, not a trigger for recomputation"
        )

        # (b) Verbatim reuse: every non-trivial number the spec displays
        # is a number the report already carries — plot never revises
        # or aggregates a value the report did not compute.
        spec_numbers = _nontrivial_numeric_leaves(from_report)
        report_numbers = _numeric_leaves(report)
        fabricated = {
            number
            for number in spec_numbers
            if not any(
                pytest.approx(number, rel=1e-12, abs=1e-12) == known
                for known in report_numbers
            )
        }
        assert not fabricated, (
            f"the {kind} spec displays numbers absent from its report "
            f"(plot fabricated or recomputed them): {sorted(fabricated)}"
        )

    # The correlation spec really carries the report's coefficient: the
    # fixture pair is deliberately non-degenerate (Pearson is strictly
    # between -1 and 1), so the carried value is a fingerprint of the
    # report, not an identity constant.
    correlation_entry = next(
        item
        for item in walk_values(reports["correlation"])
        if hasattr(item, "n") and getattr(item, "coefficient", None) is not None
    )
    coefficient = correlation_entry.coefficient
    assert -1.0 < coefficient < 1.0 and coefficient != 0.0, (
        "the fixture correlation must be a non-degenerate fingerprint"
    )
    assert _numeric_leaves(diagnostics.plot(reports["correlation"], kind="correlation"))
    spec_coefficients = {
        number
        for number in _numeric_leaves(
            diagnostics.plot(reports["correlation"], kind="correlation")
        )
        if pytest.approx(coefficient, rel=1e-9, abs=1e-9) == number
    }
    assert spec_coefficients, (
        "the correlation spec must carry the report's coefficient value "
        f"{coefficient!r} verbatim (the correlation matrix)"
    )

    # (c) Monkeypatch kill: replacing the module's public statistics
    # blocks with booby traps leaves plot(report) value-identical —
    # the report path never recomputes — while plot(dataset) hits the
    # trap, proving the dataset path routes through exactly the public
    # diagnostic functions (no second statistics implementation).
    originals = {name: getattr(diagnostics, name) for name in _STATISTICS_BLOCKS}

    def _booby_trap(name):
        def trap(*args, **kwargs):
            raise AssertionError(
                f"plot() must route the dataset path through the public "
                f"diagnostics.{name}(): a second statistics "
                "implementation inside plot is forbidden"
            )

        return trap

    try:
        for name in _STATISTICS_BLOCKS:
            setattr(diagnostics, name, _booby_trap(name))
        for kind, report in reports.items():
            spec = diagnostics.plot(report, kind=kind)
            assert getattr(spec, "kind", None) == kind, (
                "the report path must succeed with the statistics blocks "
                "booby-trapped: it consumes report values verbatim"
            )
        with pytest.raises(AssertionError):
            diagnostics.plot(dataset, kind="describe")
    finally:
        for name, original in originals.items():
            setattr(diagnostics, name, original)
    for kind, report in reports.items():
        assert diagnostics.plot(report, kind=kind) == diagnostics.plot(
            report, kind=kind
        )


def test_plot_of_unqualified_never_constructs_datetime_axes() -> None:
    """On UNQUALIFIED data every plot spec carries the verbatim period
    labels only — never a datetime/date value on any path — and no
    date-range/calendar construction exists anywhere in the package
    (semantic AST scan of constructor and parsing calls, not docstring
    tokens). W7 killer."""
    diagnostics = import_diagnostics()

    spec = diagnostics.plot(_dataset(), kind="describe")
    strings = find_strings(spec)
    assert month_key(2023, 1) in strings, (
        "the UNQUALIFIED spec must carry the verbatim provider labels"
    )

    for kind in KINDS:
        payload = (
            diagnostics.plot(_dataset(), kind=kind, expected_keys=_report_grid())
            if kind == "missingness"
            else diagnostics.plot(_dataset(), kind=kind)
        )
        for item in walk_values(payload):
            assert not isinstance(item, (datetime, date)), (
                f"an UNQUALIFIED plot axis may never become a datetime "
                f"(kind {kind!r}): the labels are opaque strings and no "
                "chronology exists to plot"
            )

    # Source scan: no date/datetime construction or parsing call
    # anywhere in the diagnostics package — chronology is
    # qualification's job, never the plotter's.
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
                name = _attr_name(node.func)
                assert name not in banned_calls, (
                    f"{path.name} constructs or parses {name!r}: "
                    "diagnostics never construct or parse dates"
                )


def _attr_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def test_plotspec_render_is_the_sole_public_matplotlib_boundary() -> None:
    """``render()`` is the single public rendering entry; every other
    public name on the spec and the module is renderer-neutral; the
    module public surface is the fixed five blocks plus the ``PlotSpec``
    type; and the matplotlib import lives in exactly one function scope
    — the private renderer leaf — with exactly one module importing it
    anywhere, never at module scope."""
    diagnostics = import_diagnostics()

    spec = diagnostics.plot(_dataset())
    assert callable(getattr(spec, "render", None)), (
        "PlotSpec must expose exactly one public render() boundary"
    )
    public_spec_names = {name for name in dir(spec) if not name.startswith("_")}
    assert public_spec_names - {"render"} and all(
        not name.lower().startswith(("draw", "fig", "ax", "show", "save"))
        for name in public_spec_names
    ), f"no second rendering surface may exist on the spec: {sorted(public_spec_names)}"

    module_public = {name for name in dir(diagnostics) if not name.startswith("_")}
    # Real reference derived only from the public surface: the
    # public API is the five functions and their typed outputs — every
    # public callable other than the five blocks and the PlotSpec type
    # is a public-surface breach.
    for block in REPORT_KINDS + ("plot",):
        assert callable(getattr(diagnostics, block, None)), (
            f"the module must expose the public block {block!r}"
        )
    stray_callables = {
        name
        for name in module_public
        if callable(getattr(diagnostics, name))
        and name not in set(REPORT_KINDS) | {"plot", "PlotSpec"}
    }
    assert not stray_callables, (
        "the module public surface must be exactly the fixed five "
        "blocks plus the PlotSpec type; stray public callables found: "
        f"{sorted(stray_callables)}"
    )
    for name in module_public:
        assert not name.lower().startswith(("render", "draw", "fig", "show", "save")), (
            f"the module must expose no rendering helper {name!r}: "
            "PlotSpec.render() is the sole boundary"
        )

    # Import isolation: exactly one call-time matplotlib import, at the
    # private renderer leaf, and exactly one module touches matplotlib.
    total_function_scope_hits = sum(
        len(function_scope_imports(path, "matplotlib")) for path in package_py_files()
    )
    assert total_function_scope_hits == 1, (
        "the matplotlib import must exist exactly once, inside the "
        "renderer function body (the lazy private renderer leaf); found "
        f"{total_function_scope_hits} function-scope imports"
    )
    matplotlib_modules = [
        path
        for path in package_py_files()
        if any(fact.root == "matplotlib" for fact in any_scope_runtime_imports(path))
    ]
    assert len(matplotlib_modules) == 1, (
        "exactly one module in the package may import matplotlib (the "
        f"private renderer leaf); found {len(matplotlib_modules)}"
    )


RENDER_ISOLATION_PROBE = """\
import sys

import portlearn.data.diagnostics as diagnostics

assert "matplotlib" not in sys.modules, (
    "importing the diagnostics package must not import matplotlib"
)

records_cls = __import__(
    "portlearn.data._records", fromlist=["PeriodKeyObservation"]
).PeriodKeyObservation
records = (
    records_cls("alpha", "202301", 1.0),
    records_cls("alpha", "202302", 2.0),
    records_cls("alpha", "202303", 3.0),
)
from portlearn.data.dataset import UnqualifiedDataset
dataset = UnqualifiedDataset(
    provider="synthetic",
    name="SYNTH",
    provider_dataset_id="SYNTH",
    adapter_version="synthetic-test-adapter/1.0.0",
    source_bytes=b"plot-isolation-probe",
    auxiliary_bytes=None,
    records=records,
    retrieval_provenance=type(
        "P",
        (),
        {"content_sha256": "2f2acca14af6bd3d59c6fa3204dd6b9050e0a464db2ddd19bc42b014cc397c3b", "units": "index", "frequency": "Monthly"},
    )(),
    decoder=lambda *a, **k: (_ for _ in ()).throw(AssertionError()),
)
spec = diagnostics.plot(dataset, kind="describe")
assert "matplotlib" not in sys.modules, (
    "constructing a PlotSpec must not import matplotlib"
)

try:
    spec.render()
except Exception as error:  # optional extra absent in this environment
    message = str(error)
    assert "portlearn[plot]" in message, (
        f"render() without the [plot] extra must fail naming the extra "
        f"spelling 'portlearn[plot]'; got {error!r}"
    )
else:
    raise AssertionError(
        "render() must require the optional [plot] extra; it cannot "
        "succeed with matplotlib absent"
    )
print("render-isolation: ok")
"""


def test_render_requires_optional_plot_extra_and_imports_matplotlib_lazily() -> None:
    """In this environment the ``[plot]`` extra is not installed: render
    must fail naming the ``portlearn[plot]`` extra, and neither
    importing the package nor constructing the spec may import
    matplotlib."""
    import importlib.util

    if importlib.util.find_spec("matplotlib") is not None:
        pytest.skip("matplotlib present: the absent-extra path untestable")

    completed = subprocess.run(
        [sys.executable, "-c", RENDER_ISOLATION_PROBE],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        "the render optional-dependency isolation law failed in a fresh "
        f"interpreter:\n{completed.stdout}\n{completed.stderr}"
    )


# Keep import surface honest.
_ = (PLOT_FOOTNOTE, find_strings)
