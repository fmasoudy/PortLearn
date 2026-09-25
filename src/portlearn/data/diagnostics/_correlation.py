"""Pairwise correlation over shared exact keys (stdlib-only block).

``correlation`` is a standalone report callable: it aligns the numeric
values of the selected series on the dataset's state-native exact keys
(verbatim period labels when UNQUALIFIED, normalized observation-time
identity when QUALIFIED) and reports one Pearson or Spearman
coefficient per pair, with listwise deletion by default.

Research datasets only; NOT investable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from portlearn.data.dataset import ResearchDataset


UNQUALIFIED_KEY_BASIS = "period_key (verbatim provider label)"
QUALIFIED_KEY_BASIS = "observation_time (normalized identity)"

_DISCLOSURE = (
    "correlation is computed on retained records only; no availability "
    "information was read on any state"
)


def _local_id(series_id: str) -> str:
    if "/" in series_id:
        return series_id.rsplit("/", 1)[-1]
    return series_id


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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


def _average_ranks(values: list[float]) -> list[float]:
    """Competition-free ranks: tied values share the mean position."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


@dataclass(frozen=True)
class PairEntry:
    """One ordered series pair under the selected method and deletion."""

    first: str
    second: str
    method: str
    deletion: str
    n: int
    coefficient: float | None
    min_overlap: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CorrelationReport:
    """The typed correlation report over exact-key aligned pairs."""

    report_kind: ClassVar[str] = "correlation"

    availability_state: str
    provider: str
    name: str
    data_mode: str | None
    source_sha256: str
    key_basis: str
    method: str
    deletion: str
    min_overlap: int
    series: tuple[str, ...]
    pairs: tuple[PairEntry, ...]
    basis: str


def correlation(
    dataset: ResearchDataset,
    /,
    *,
    method: str = "pearson",
    deletion: str = "listwise",
    min_overlap: int = 3,
    series: Any = None,
) -> CorrelationReport:
    """Correlate the dataset's series pairwise on exact state-native keys.

    ``method`` is ``"pearson"`` or ``"spearman"`` (average-rank ties);
    ``deletion`` defaults to listwise (intersection across all selected
    series) with pairwise as explicit opt-in; a pair with fewer than
    ``min_overlap`` aligned observations, or a constant series on the
    aligned support, yields a ``None`` coefficient carrying a
    machine-readable reason — never a fabricated value.
    """
    _require_dataset(dataset)
    if method not in ("pearson", "spearman"):
        raise ValueError(
            f"method must be exactly 'pearson' or 'spearman'; got {method!r}"
        )
    if deletion not in ("listwise", "pairwise"):
        raise ValueError(
            f"deletion must be exactly 'listwise' or 'pairwise'; got {deletion!r}"
        )
    if isinstance(min_overlap, bool) or not isinstance(min_overlap, int):
        raise TypeError(
            f"min_overlap must be an int; got {type(min_overlap).__name__!r}"
        )
    if min_overlap < 3:
        raise ValueError(
            f"min_overlap must be >= 3 (the floor is not caller-lowerable); "
            f"got {min_overlap!r}"
        )

    state = dataset.availability_state
    keyed: dict[str, dict[Any, Any]] = {}
    for record in tuple(dataset.records):
        local = _local_id(record.series_id)
        cells = keyed.setdefault(local, {})
        key = _state_native_key(record, state)
        if key in cells:
            raise ValueError(
                f"duplicate (series_id, exact key) cell ({local!r}, {key!r}) "
                "within one series: duplicates are never averaged into an "
                "estimand; the dataset rejects unconditionally"
            )
        cells[key] = record.value

    all_ids = sorted(keyed)
    if series is None:
        selected = all_ids
    else:
        if isinstance(series, (str, bytes)) or not hasattr(series, "__iter__"):
            raise TypeError(
                f"series must be an iterable of series ids; got {type(series).__name__!r}"
            )
        chosen: list[str] = []
        for sid in series:
            if not isinstance(sid, str):
                raise TypeError(f"a series id must be str; got {sid!r}")
            if sid in chosen:
                raise ValueError(
                    f"duplicate series id {sid!r} in the selection: the "
                    "selection is ambiguous and rejects"
                )
            chosen.append(sid)
        unknown = [sid for sid in chosen if sid not in keyed]
        if unknown:
            raise ValueError(
                f"unknown series id(s) {unknown!r}: the dataset carries {all_ids!r}"
            )
        selected = sorted(chosen)

    numeric_keys = {
        sid: {key for key, value in keyed[sid].items() if _is_numeric(value)}
        for sid in selected
    }
    if deletion == "listwise" and numeric_keys:
        common: frozenset[Any] = frozenset.intersection(
            *(frozenset(keys) for keys in numeric_keys.values())
        )
    else:
        common = frozenset()

    # Cross-pairs first (i < j): the report's leading coefficient-bearing
    # entry is then a genuine two-series estimand, never the trivial
    # self-identity +1.0; self-pair entries follow, carrying their own
    # overlap accounting (a self pair is its own alignment).
    pair_order = [
        (first, second)
        for i, first in enumerate(selected)
        for second in selected[i + 1 :]
    ] + [(sid, sid) for sid in selected]

    entries: list[PairEntry] = []
    for first, second in pair_order:
        if deletion == "pairwise":
            aligned = numeric_keys[first] & numeric_keys[second]
        else:
            aligned = common
        n = len(aligned)
        coefficient: float | None = None
        reasons: tuple[str, ...] = ()
        if n < min_overlap:
            reasons = ("insufficient_overlap",)
        else:
            ordered = sorted(aligned)
            xs = [float(keyed[first][key]) for key in ordered]
            ys = [float(keyed[second][key]) for key in ordered]
            if method == "spearman":
                xs = _average_ranks(xs)
                ys = _average_ranks(ys)
            mean_x = sum(xs) / n
            mean_y = sum(ys) / n
            sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            sxx = sum((x - mean_x) ** 2 for x in xs)
            syy = sum((y - mean_y) ** 2 for y in ys)
            if sxx == 0.0 or syy == 0.0:
                reasons = ("constant_series",)
            else:
                coefficient = sxy / math.sqrt(sxx * syy)
        entries.append(
            PairEntry(
                first=first,
                second=second,
                method=method,
                deletion=deletion,
                n=n,
                coefficient=coefficient,
                min_overlap=min_overlap,
                reasons=reasons,
            )
        )

    return CorrelationReport(
        availability_state=state,
        provider=dataset.provider,
        name=dataset.name,
        data_mode=_optional(dataset, "data_mode"),
        source_sha256=dataset.source_sha256,
        key_basis=(
            UNQUALIFIED_KEY_BASIS if state == "UNQUALIFIED" else QUALIFIED_KEY_BASIS
        ),
        method=method,
        deletion=deletion,
        min_overlap=min_overlap,
        series=tuple(selected),
        pairs=tuple(entries),
        basis=_DISCLOSURE,
    )
