"""Descriptive summaries over retained records (stdlib-only block).

``describe`` is a standalone report callable: it reads one sealed
dataset and returns a frozen, stdlib-only report of per-series sample
statistics over the actually retained observations — no sentinel
recovery, no imputation, no coercion of non-numeric value domains.

Research datasets only; NOT investable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from portlearn.data.dataset import ResearchDataset


_BASIS = (
    "descriptive statistics over retained observations only; "
    "provider-marked absences dropped before retention are invisible to "
    "this report (see missingness for expected-grid absence accounting)"
)


def _local_id(series_id: str) -> str:
    """The series identity diagnostics reports under.

    Provider adapters namespace their series ids by documented
    convention (``PREFIX/name``); the namespaced prefix is provider
    bookkeeping, so diagnostics reports the local name. An
    unnamespaced id reports under itself. Pure string handling — no
    provider branch exists here.
    """
    if "/" in series_id:
        return series_id.rsplit("/", 1)[-1]
    return series_id


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _quantile(sorted_values: list[float], p: float) -> float:
    """Linear interpolation on order statistics at h = (n-1)p."""
    count = len(sorted_values)
    if count == 1:
        return sorted_values[0]
    h = (count - 1) * p
    index = math.floor(h)
    g = h - index
    if g == 0.0:
        return sorted_values[index]
    return sorted_values[index] + g * (sorted_values[index + 1] - sorted_values[index])


def _state_native_key(record: Any, availability_state: str) -> Any:
    if availability_state == "UNQUALIFIED":
        return record.period_key
    return record.observation_time


def _require_dataset(dataset: Any) -> None:
    if not (
        hasattr(dataset, "records")
        and hasattr(dataset, "availability_state")
        and hasattr(dataset, "source_sha256")
    ):
        raise TypeError(
            "diagnostics accept exactly one sealed ResearchDataset; got "
            f"{type(dataset).__name__!r}"
        )


def _optional(dataset: Any, name: str) -> Any:
    try:
        return getattr(dataset, name)
    except AttributeError:
        return None


@dataclass(frozen=True)
class SeriesSummary:
    """One series' value summary over its retained numeric observations."""

    series_id: str
    n: int
    mean: float | None
    std: float | None
    min: float | None
    max: float | None
    q25: float | None
    median: float | None
    q75: float | None
    n_bool: int
    n_str: int
    n_null: int
    keys: tuple[Any, ...]
    undefined: tuple[str, ...]


@dataclass(frozen=True)
class DescribeReport:
    """The typed describe report over retained observations."""

    report_kind: ClassVar[str] = "describe"

    availability_state: str
    provider: str
    name: str
    units: str | None
    frequency: str | None
    data_mode: str | None
    source_sha256: str
    n_records: int
    n_series: int
    zero_records: bool
    basis: str
    undefined: tuple[str, ...]
    per_series: tuple[SeriesSummary, ...]


def describe(dataset: ResearchDataset, /) -> DescribeReport:
    """Summarize retained observations per series, sample statistics only.

    ``std`` is the sample standard deviation (ddof=1), ``None`` with
    reason ``insufficient_n`` when a series retains fewer than two
    numeric observations; quartiles follow the pinned linear
    interpolation at ``h=(n-1)p``; bool/str/None values are counted per
    series, never coerced into statistics.
    """
    _require_dataset(dataset)
    state = dataset.availability_state
    records = tuple(dataset.records)

    grouped: dict[str, list[Any]] = {}
    key_sets: dict[str, set[Any]] = {}
    for record in records:
        local = _local_id(record.series_id)
        grouped.setdefault(local, []).append(record.value)
        key_sets.setdefault(local, set()).add(_state_native_key(record, state))

    per_series: list[SeriesSummary] = []
    for series_id in sorted(grouped):
        values = grouped[series_id]
        numeric = sorted(float(v) for v in values if _is_numeric(v))
        n_bool = sum(1 for v in values if isinstance(v, bool))
        n_str = sum(1 for v in values if isinstance(v, str))
        n_null = sum(1 for v in values if v is None)
        count = len(numeric)
        if count >= 1:
            mean: float | None = sum(numeric) / count
            minimum: float | None = numeric[0]
            maximum: float | None = numeric[-1]
            q25: float | None = _quantile(numeric, 0.25)
            median: float | None = _quantile(numeric, 0.5)
            q75: float | None = _quantile(numeric, 0.75)
        else:
            mean = None
            minimum = None
            maximum = None
            q25 = None
            median = None
            q75 = None
        if count >= 2:
            variance = sum((x - mean) ** 2 for x in numeric) / (count - 1)
            std: float | None = math.sqrt(variance)
            undefined: tuple[str, ...] = ()
        elif count == 1:
            std = None
            undefined = ("insufficient_n",)
        else:
            std = None
            undefined = ("insufficient_n", "no_numeric_values")
        per_series.append(
            SeriesSummary(
                series_id=series_id,
                n=count,
                mean=mean,
                std=std,
                min=minimum,
                max=maximum,
                q25=q25,
                median=median,
                q75=q75,
                n_bool=n_bool,
                n_str=n_str,
                n_null=n_null,
                keys=tuple(sorted(key_sets[series_id])),
                undefined=undefined,
            )
        )

    zero = not records
    return DescribeReport(
        availability_state=state,
        provider=dataset.provider,
        name=dataset.name,
        units=_optional(dataset, "units"),
        frequency=_optional(dataset, "frequency"),
        data_mode=_optional(dataset, "data_mode"),
        source_sha256=dataset.source_sha256,
        n_records=len(records),
        n_series=len(grouped),
        zero_records=zero,
        basis=_BASIS,
        undefined=("zero_records",) if zero else (),
        per_series=tuple(per_series),
    )
