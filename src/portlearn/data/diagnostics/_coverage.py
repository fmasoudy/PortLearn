"""Exact-key coverage accounting (stdlib-only block).

``coverage`` is a standalone report callable with a plain signature:
it counts, per series, the retained records, the distinct exact keys
they occupy, and the duplicate cells — observed-support accounting
only.  It makes no claim beyond the observed support: no expectation
denominator, no chronology over opaque labels, and no extrapolated
regularity.  Exact keys group on the record's own state-native
identity; the availability record never enters the grouping.

Research datasets only; NOT investable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_KEY_BASIS = {
    "UNQUALIFIED": "period_key (verbatim provider label)",
    "QUALIFIED": "observation_time (normalized identity)",
}


def _display_id(series_id: str) -> str:
    """The per-series display identifier (leading namespaced segment
    dropped); generic string handling, no provider branch."""
    return series_id.split("/", 1)[-1]


def _record_key(record: Any, state: str) -> Any:
    if state == "QUALIFIED":
        return record.observation_time
    return record.period_key


@dataclass(frozen=True)
class SeriesCoverage:
    """One series' observed-support accounting."""

    series_id: str
    n_records: int
    n_distinct_exact_keys: int
    n_duplicate_cells: int
    n_numeric: int
    n_bool: int
    n_str: int
    n_null: int
    value_note: str = ""


@dataclass(frozen=True)
class CoverageReport:
    """The fixed coverage report: observed-support counts only."""

    report_kind: str
    availability_state: str
    provider: str
    source_sha256: str
    data_mode: str | None
    n_records: int
    n_series: int
    n_duplicate_cells: int
    duplicate_note: str
    basis: str
    key_basis: str
    per_series: tuple[SeriesCoverage, ...]
    reasons: tuple[str, ...] = field(default=())


def coverage(dataset: Any) -> CoverageReport:
    """Account the observed support of one sealed dataset: per-series
    retained-record, distinct-exact-key, and duplicate-cell counts with
    value-domain presence accounting.  Observed support only — the
    expectation grid belongs to ``missingness`` alone."""
    state = getattr(dataset, "availability_state", None)
    if state not in ("UNQUALIFIED", "QUALIFIED"):
        raise TypeError(
            "coverage() accepts exactly one sealed ResearchDataset; got "
            f"{type(dataset).__name__!r}."
        )

    grouped: dict[str, list[Any]] = {}
    for record in dataset.records:
        grouped.setdefault(_display_id(record.series_id), []).append(record)

    entries: list[SeriesCoverage] = []
    total_duplicates = 0
    for display in sorted(grouped):
        records = grouped[display]
        keys: dict[Any, int] = {}
        n_numeric = n_bool = n_str = n_null = 0
        for record in records:
            key = _record_key(record, state)
            keys[key] = keys.get(key, 0) + 1
            value = record.value
            if value is None:
                n_null += 1
            elif isinstance(value, bool):
                n_bool += 1
            elif isinstance(value, str):
                n_str += 1
            else:
                n_numeric += 1
        duplicates = sum(1 for count in keys.values() if count > 1)
        total_duplicates += duplicates
        entries.append(
            SeriesCoverage(
                series_id=display,
                n_records=len(records),
                n_distinct_exact_keys=len(keys),
                n_duplicate_cells=duplicates,
                n_numeric=n_numeric,
                n_bool=n_bool,
                n_str=n_str,
                n_null=n_null,
                value_note="value-domain presence counts by kind",
            )
        )

    try:
        data_mode = dataset.data_mode
    except AttributeError:
        data_mode = None

    return CoverageReport(
        report_kind="coverage",
        availability_state=state,
        provider=dataset.provider,
        source_sha256=dataset.source_sha256,
        data_mode=data_mode,
        n_records=len(dataset.records),
        n_series=len(entries),
        n_duplicate_cells=total_duplicates,
        duplicate_note=(
            "duplicate exact-key cells are disclosed per series; an "
            "ambiguous cell is never merged"
        ),
        basis="Observed-support accounting over the retained records.",
        key_basis=_KEY_BASIS[state],
        per_series=tuple(entries),
    )
