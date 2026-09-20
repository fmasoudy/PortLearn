"""Plot specifications over the diagnostics reports (stdlib-only block).

``plot`` builds :class:`PlotSpec` values — frozen, stdlib-only
descriptions of one visualization — and never touches a plotting
stack: the optional renderer lives in the private leaf module and
imports it at call time only, so the core package and every spec stay
importable without the ``plot`` extra.

``plot`` accepts either one sealed dataset or one already-built
report.  The dataset path dispatches to the package's own public
report callables at call time (an explicit dynamic attribute read, so
the dispatch follows the live public surface), then builds the spec
from the resulting report — which is exactly why ``plot(dataset,
kind=...)`` and ``plot(report)`` are value-equal for the four report
kinds.  The report path consumes the report's already-computed values
verbatim and never recomputes a statistic.

Research datasets only; NOT investable.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from portlearn.data.dataset import ResearchDataset

__all__ = ["PlotSeries", "PlotSpec", "plot"]

#: The exact fixed footnote every plot spec carries.
PLOT_FOOTNOTE = (
    "Descriptive summary of retained records; not decision-time eligibility."
)

#: The exact key-basis disclosure names for the two sealed states.
UNQUALIFIED_KEY_BASIS = "period_key (verbatim provider label)"
QUALIFIED_KEY_BASIS = "observation_time (normalized identity)"

#: The exact five plot kinds.
PLOT_KINDS = ("describe", "correlation", "coverage", "missingness", "series")

#: The four report-producing kinds (``series`` is dataset-only).
_REPORT_KINDS = ("describe", "correlation", "coverage", "missingness")

#: Advisory renderer hint — never a rendering instruction.
RENDERER_HINT = "categorical lines with markers"


@dataclass(frozen=True)
class PlotSeries:
    """One named run of y values aligned to the spec's label positions."""

    name: str
    y: tuple[Any, ...]


@dataclass(frozen=True)
class PlotSpec:
    """A frozen, stdlib-only description of one visualization.

    The same spec type serves every kind.  ``labels`` carries the
    exact keys verbatim (never a derived calendar value), every
    numeric leaf of a spec drawn from a report exists in that report,
    and ``render`` is the sole public rendering boundary.
    """

    kind: str
    title: str
    subtitle: str
    key_basis: str
    labels: tuple[Any, ...]
    series: tuple[PlotSeries, ...]
    footnote: str
    availability_state: str
    data_mode: str | None
    source_sha256: str
    renderer_hint: str

    def render(self) -> Any:
        """Render through the private leaf; needs the ``plot`` extra."""
        from portlearn.data.diagnostics._renderer import render

        return render(self)


def _report_kind(value: Any) -> str | None:
    """Classify an already-built report value by its ``report_kind`` marker."""
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        return None
    detected = getattr(value, "report_kind", None)
    if detected in _REPORT_KINDS:
        return detected
    return None


def _require_dataset(source: Any) -> None:
    """Accept exactly one sealed dataset, structurally."""
    if not (
        hasattr(source, "records")
        and hasattr(source, "availability_state")
        and hasattr(source, "source_sha256")
    ):
        raise TypeError(
            "plot() accepts exactly one sealed ResearchDataset or one "
            "diagnostics report; got "
            f"{type(source).__name__!r}, fail-closed"
        )


def _local_id(series_id: str) -> str:
    """The series identity diagnostics reports under (pure strings)."""
    if "/" in series_id:
        return series_id.rsplit("/", 1)[-1]
    return series_id


def _state_key(record: Any, state: str) -> Any:
    if state == "UNQUALIFIED":
        return record.period_key
    return record.observation_time


def _key_basis(state: str) -> str:
    return UNQUALIFIED_KEY_BASIS if state == "UNQUALIFIED" else QUALIFIED_KEY_BASIS


def _subtitle(state: str, data_mode: Any) -> str:
    if data_mode is None:
        return f"availability state: {state}"
    return f"availability state: {state}; data mode: {data_mode}"


def _describe_spec(report: Any) -> PlotSpec:
    rows = report.per_series
    labels = tuple(sorted({key for row in rows for key in row.keys}))
    runs = tuple(
        PlotSeries(
            name=row.series_id,
            y=tuple(row.mean if key in row.keys else None for key in labels),
        )
        for row in rows
    )
    data_mode = getattr(report, "data_mode", None)
    return PlotSpec(
        kind="describe",
        title="Descriptive summary by series (mean of retained numeric values)",
        subtitle=_subtitle(report.availability_state, data_mode),
        key_basis=_key_basis(report.availability_state),
        labels=labels,
        series=runs,
        footnote=PLOT_FOOTNOTE,
        availability_state=report.availability_state,
        data_mode=data_mode,
        source_sha256=report.source_sha256,
        renderer_hint=RENDERER_HINT,
    )


def _correlation_spec(report: Any) -> PlotSpec:
    names = tuple(report.series)
    coefficients: dict[tuple[str, str], Any] = {}
    for pair in report.pairs:
        coefficients[(pair.first, pair.second)] = pair.coefficient
        coefficients[(pair.second, pair.first)] = pair.coefficient
    runs = tuple(
        PlotSeries(
            name=name,
            y=tuple(coefficients.get((name, other)) for other in names),
        )
        for name in names
    )
    data_mode = getattr(report, "data_mode", None)
    return PlotSpec(
        kind="correlation",
        title=f"{report.method} correlation, {report.deletion} deletion",
        subtitle=_subtitle(report.availability_state, data_mode),
        key_basis=report.key_basis,
        labels=names,
        series=runs,
        footnote=PLOT_FOOTNOTE,
        availability_state=report.availability_state,
        data_mode=data_mode,
        source_sha256=report.source_sha256,
        renderer_hint=RENDERER_HINT,
    )


def _coverage_spec(report: Any) -> PlotSpec:
    rows = report.per_series
    labels = tuple(row.series_id for row in rows)
    runs = (
        PlotSeries(
            name="n_records",
            y=tuple(row.n_records for row in rows),
        ),
        PlotSeries(
            name="n_distinct_exact_keys",
            y=tuple(row.n_distinct_exact_keys for row in rows),
        ),
        PlotSeries(
            name="n_duplicate_cells",
            y=tuple(row.n_duplicate_cells for row in rows),
        ),
    )
    data_mode = getattr(report, "data_mode", None)
    return PlotSpec(
        kind="coverage",
        title="Observed-support accounting by series",
        subtitle=_subtitle(report.availability_state, data_mode),
        key_basis=report.key_basis,
        labels=labels,
        series=runs,
        footnote=PLOT_FOOTNOTE,
        availability_state=report.availability_state,
        data_mode=data_mode,
        source_sha256=report.source_sha256,
        renderer_hint=RENDERER_HINT,
    )


def _missingness_spec(report: Any) -> PlotSpec:
    labels = (
        "expected",
        "observed",
        "absent_expected",
        "null_value",
        "unexpected_observed",
    )
    counts = (
        report.expected_n,
        report.observed_n,
        report.absent_expected_n,
        report.null_value_n,
        report.unexpected_observed_n,
    )
    runs = (
        PlotSeries(name="cells", y=counts),
        PlotSeries(
            name="observed_fraction_of_expected",
            y=(None, report.observed_fraction_of_expected, None, None, None),
        ),
    )
    data_mode = getattr(report, "data_mode", None)
    return PlotSpec(
        kind="missingness",
        title="Expected-grid missingness accounting",
        subtitle=_subtitle(report.availability_state, data_mode),
        key_basis=report.key_basis,
        labels=labels,
        series=runs,
        footnote=PLOT_FOOTNOTE,
        availability_state=report.availability_state,
        data_mode=data_mode,
        source_sha256=report.source_sha256,
        renderer_hint=RENDERER_HINT,
    )


_SPEC_BUILDERS = {
    "describe": _describe_spec,
    "correlation": _correlation_spec,
    "coverage": _coverage_spec,
    "missingness": _missingness_spec,
}


def _series_spec(dataset: Any) -> PlotSpec:
    """The raw-series spec: values aligned on the exact keys, verbatim."""
    state = dataset.availability_state
    keys: set[Any] = set()
    by_series: dict[str, dict[Any, Any]] = {}
    for record in dataset.records:
        key = _state_key(record, state)
        keys.add(key)
        by_series.setdefault(_local_id(record.series_id), {})[key] = record.value
    labels = tuple(sorted(keys))
    runs = tuple(
        PlotSeries(
            name=series_id,
            y=tuple(by_series[series_id].get(key) for key in labels),
        )
        for series_id in sorted(by_series)
    )
    data_mode = getattr(dataset, "data_mode", None)
    return PlotSpec(
        kind="series",
        title="Retained values by exact key",
        subtitle=_subtitle(state, data_mode),
        key_basis=_key_basis(state),
        labels=labels,
        series=runs,
        footnote=PLOT_FOOTNOTE,
        availability_state=state,
        data_mode=data_mode,
        source_sha256=dataset.source_sha256,
        renderer_hint=RENDERER_HINT,
    )


def plot(
    source: ResearchDataset, /, *, kind: str = "series", **kwargs: Any
) -> PlotSpec:
    """Build one :class:`PlotSpec` from a dataset or a report.

    For the four report kinds over a dataset, the report is computed
    through the package's live public callables — forwarding the exact
    keyword arguments the caller passed (``missingness`` requires and
    forwards ``expected_keys``) — and the spec is built from that
    report, so ``plot(dataset, kind=...)`` is value-equal to
    ``plot(report)`` for the same report.  A report source accepts only
    the kind matching its own type and is consumed verbatim.
    """
    if kind not in PLOT_KINDS:
        raise ValueError(
            f"kind must be one of {PLOT_KINDS!r}; got {kind!r} — the "
            "plot kind set is closed"
        )
    detected = _report_kind(source)
    if detected is not None:
        if kind != detected:
            raise ValueError(
                f"a {type(source).__name__} plots only as kind "
                f"{detected!r}; got {kind!r}, and an incompatible "
                "source/kind pair is never coerced"
            )
        return _SPEC_BUILDERS[detected](source)
    _require_dataset(source)
    if kind == "series":
        return _series_spec(source)

    from importlib import import_module

    diagnostics = import_module("portlearn.data.diagnostics")
    report_callable = getattr(diagnostics, kind)
    report = report_callable(source, **kwargs)
    return _SPEC_BUILDERS[kind](report)
