"""Expected-grid missingness accounting (stdlib-only block).

``missingness`` is a standalone report callable: it accounts the
dataset's observed exact ``(series, key)`` cells against a
caller-declared expected grid and returns fixed counts — how much of
the grid is observed, how much is absent, how many observed records
lie outside the grid, and how many observed records at expected cells
carry a null value.  The grid is explicit and required: the callable
never invents, derives, or interpolates expected keys, and it accepts
no other route to one.  Validation runs before any accounting: a grid
with a repeated cell and a dataset with a repeated ``(series, exact
key)`` cell each unconditional, and a grid entry of the wrong shape or
key type fails with a ``TypeError``.

Research datasets only; NOT investable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from portlearn.data.dataset import ResearchDataset

__all__ = ["MissingnessReport", "missingness"]


_UNQUALIFIED_KEY_BASIS = "period_key (verbatim provider label)"
_QUALIFIED_KEY_BASIS = "observation_time (normalized identity)"

_BASIS = (
    "Accounting of observed exact (series, key) cells against the "
    "caller-declared expected_keys grid; the grid is caller authority "
    "and is never inferred, extrapolated, or unioned with the records."
)


def _local_id(series_id: str) -> str:
    """The series identity diagnostics reports under (pure strings)."""
    if "/" in series_id:
        return series_id.rsplit("/", 1)[-1]
    return series_id


def _state_native_key(record: Any, availability_state: str) -> Any:
    if availability_state == "UNQUALIFIED":
        return record.period_key
    return record.observation_time


@dataclass(frozen=True)
class MissingnessReport:
    """The fixed missingness report against the declared grid."""

    report_kind: ClassVar[str] = "missingness"

    availability_state: str
    provider: str
    name: str
    source_sha256: str
    key_basis: str
    expected_n: int
    observed_n: int
    absent_expected_n: int
    null_value_n: int
    unexpected_observed_n: int
    observed_fraction_of_expected: float | None
    basis: str


def missingness(
    dataset: ResearchDataset, /, *, expected_keys: Any
) -> MissingnessReport:
    """Account observed exact cells against the declared expected grid.

    ``expected_keys`` is required: an explicit, duplicate-free iterable
    of ``(series_id, key)`` pairs where ``key`` is the sealed state's
    exact key type — the verbatim period label while UNQUALIFIED, the
    normalized observation identity while QUALIFIED.  Comparison is
    exact-key equality only: no label ever converts into an instant or
    vice versa, and near-miss labels never match.
    """
    state = getattr(dataset, "availability_state", None)
    if state not in ("UNQUALIFIED", "QUALIFIED"):
        raise TypeError(
            "missingness() accepts exactly one sealed ResearchDataset; "
            f"got {type(dataset).__name__!r}."
        )

    try:
        entries = list(expected_keys)
    except TypeError:
        raise TypeError(
            "expected_keys must be an iterable of (series_id, key) "
            "pairs declaring the expected observation grid; got "
            f"{type(expected_keys).__name__!r}, unconditional"
        ) from None

    expected_cells: list[tuple[str, Any]] = []
    seen_expected: set[tuple[str, Any]] = set()
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(
                "each expected_keys entry must be a (series_id, key) "
                f"pair; got {entry!r}, unconditional"
            )
        series_id, key = entry
        if not isinstance(series_id, str) or isinstance(series_id, bool):
            raise TypeError(
                "the expected_keys series identity must be a str; got "
                f"{type(series_id).__name__}: {series_id!r}, unconditional"
            )
        if state == "UNQUALIFIED":
            key_ok = isinstance(key, str) and not isinstance(key, bool)
        else:
            key_ok = isinstance(key, datetime) and not isinstance(key, bool)
        if not key_ok:
            raise TypeError(
                "each expected_keys key must be the sealed state's exact "
                f"key type ({'str' if state == 'UNQUALIFIED' else 'datetime'}"
                f" while {state}); got {type(key).__name__}: {key!r}, "
                "unconditional"
            )
        cell = (series_id, key)
        if cell in seen_expected:
            raise ValueError(
                f"expected_keys carries the exact cell {cell!r} more than "
                "once; the declared grid is a set of distinct cells, "
                "unconditional"
            )
        seen_expected.add(cell)
        expected_cells.append(cell)

    observed_cells: set[tuple[str, Any]] = set()
    null_value_n = 0
    for record in dataset.records:
        cell = (
            _local_id(record.series_id),
            _state_native_key(record, state),
        )
        if cell in observed_cells:
            raise ValueError(
                f"duplicate retained cell: series {cell[0]!r} carries the "
                f"exact key {cell[1]!r} more than once; the accounting "
                "requires at most one record per (series, exact key) cell, "
                "unconditional"
            )
        observed_cells.add(cell)
        if record.value is None and cell in seen_expected:
            null_value_n += 1

    expected_set = seen_expected
    observed_n = len(observed_cells & expected_set)
    absent_expected_n = len(expected_set) - observed_n
    unexpected_observed_n = len(observed_cells - expected_set)
    fraction: float | None = None
    if expected_set:
        fraction = observed_n / len(expected_set)

    key_basis = (
        _UNQUALIFIED_KEY_BASIS if state == "UNQUALIFIED" else _QUALIFIED_KEY_BASIS
    )
    return MissingnessReport(
        availability_state=state,
        provider=dataset.provider,
        name=dataset.name,
        source_sha256=dataset.source_sha256,
        key_basis=key_basis,
        expected_n=len(expected_set),
        observed_n=observed_n,
        absent_expected_n=absent_expected_n,
        null_value_n=null_value_n,
        unexpected_observed_n=unexpected_observed_n,
        observed_fraction_of_expected=fraction,
        basis=_BASIS,
    )
