"""Period-calendar substrate: the period-end → aware-instant mapping.

This module implements, exactly once, the period-calendar convention: a month-end or
quarter-end calendar day maps to the **last instant of that period in a
declared timezone** — ``23:59:59.999999`` on the last calendar day of
the period, expressed in the declared zone, then UTC-normalized at
storage.

It is a neutral substrate:

- **stdlib-only** (``calendar``/``datetime`` arithmetic, no providers);
- **pure** — same inputs, equal instants, no state;
- **disciplined** — it defines no contract errors, imports from
  ``portlearn.timing`` only, and knows nothing about ingestion,
  adapters, alignment, or schedules.  Consumers import these helpers;
  they never restate the mapping.
"""

from __future__ import annotations

import calendar as _calendar
from datetime import datetime
from typing import Any

from portlearn.timing import to_instant

__all__ = [
    "month_end_instant",
    "quarter_end_instant",
]

_LAST_TIME_OF_DAY = (23, 59, 59, 999999)


def _period_end_instant(year: int, month: int, tzinfo: Any) -> datetime:
    """The last instant of ``month`` in ``year`` under declared ``tzinfo``.

    Builds ``23:59:59.999999`` on the month's last calendar day in the
    declared zone and UTC-normalizes it through
    ``to_instant``, which also rejects a naive/invalid zone
    fail-closed.
    """
    last_day = _calendar.monthrange(year, month)[1]
    # A deliberately naive wall-clock reading: the declared zone is attached
    # on the next line and to_instant then UTC-normalizes the result.
    wall = datetime(  # noqa: DTZ001 — zone attached immediately below
        year, month, last_day, *_LAST_TIME_OF_DAY
    )
    return to_instant(wall.replace(tzinfo=tzinfo), "period_end")


def month_end_instant(year: int, month: int, tzinfo: Any) -> datetime:
    """The last instant of a month in a declared timezone.

    Returns the UTC-normalized instant of ``23:59:59.999999`` on the
    month's last calendar day (leap February included) in ``tzinfo``.
    ``tzinfo`` must be an aware timezone object; ``None`` or a naive
    substitute rejects with ``NaiveTimestampError`` — no default zone
    is ever assumed.
    """
    return _period_end_instant(year, month, tzinfo)


def quarter_end_instant(year: int, quarter: int, tzinfo: Any) -> datetime:
    """The last instant of a calendar quarter in a declared timezone.

    ``quarter`` is 1–4; the quarter's closing month is March, June,
    September, or December.  Returns the UTC-normalized instant of
    ``23:59:59.999999`` on that month's last calendar day in ``tzinfo``,
    with the same fail-closed timezone check as
    :func:`month_end_instant`.
    """
    if not isinstance(quarter, int) or isinstance(quarter, bool):
        raise ValueError(  # noqa: TRY004 — admission rejects with ValueError
            "quarter must be an integer 1-4 naming a calendar quarter; "
            f"got {type(quarter).__name__}: {quarter!r}"
        )
    if not 1 <= quarter <= 4:
        raise ValueError(
            "quarter must be an integer 1-4 naming a calendar quarter; "
            f"got {quarter!r}"
        )
    closing_month = quarter * 3
    return _period_end_instant(year, closing_month, tzinfo)
