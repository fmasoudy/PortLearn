"""Leakage battery for the contract error classes.

This module implements the invariant battery layer of the public
contract: a battery is a list of :class:`LeakageCase` scenarios — each a
named finance-language leak attempt paired with the single
contract error class expected to block it — executed totally by
:func:`run_leakage_cases`, which reports every outcome as a
:class:`LeakageFinding` and never lets an attempted leak fail silently.

Design laws honoured here:

* ``FROZEN_CONTRACT_ERRORS`` is exactly the six module-qualified
  contract error classes — a battery can never expect a foreign error
  class, and every expected error must be one of these six.
* Admission is strict: a case with a blank or non-string name, a
  non-callable attempt, or an expected error outside the supported
  contract error classes is rejected before anything runs.
* Totality: every admitted case produces exactly one finding, whatever
  it does — a wrong error class reports ``blocked is False`` with
  ``detail == "wrong error class"``, and a leak that raises nothing at
  all reports ``error_type is None`` with
  ``detail == "leak was NOT blocked"``.
* The summary derives only from the findings:
  ``"{blocked}/{total} attempted leaks blocked"``.

The module imports stdlib only at module level plus the contract error
classes by import (aware-instant validation remains implemented solely
in ``portlearn.timing``; nothing here re-implements it). Importing this
module performs no I/O and executes no case.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from portlearn.observations import (
    AmbiguousObservationError,
    FeatureLineageError,
)
from portlearn.timing import (
    FutureInformationError,
    InvalidChronologyError,
    MissingAvailabilityError,
    NaiveTimestampError,
)

__all__ = [  # noqa: RUF022 — deliberate order (types, function, constant)
    "LeakageCase",
    "LeakageFinding",
    "LeakageReport",
    "run_leakage_cases",
    "FROZEN_CONTRACT_ERRORS",
]


#: The exact set of contract error classes a battery case may
#: expect — the six module-qualified contract error classes. A tuple,
#: so the order is itself fixed and import-stable.
FROZEN_CONTRACT_ERRORS: tuple[type[Exception], ...] = (
    FutureInformationError,
    InvalidChronologyError,
    MissingAvailabilityError,
    NaiveTimestampError,
    AmbiguousObservationError,
    FeatureLineageError,
)

_FROZEN_ERROR_SET: frozenset[type[Exception]] = frozenset(FROZEN_CONTRACT_ERRORS)

#: Detail for a case whose attempt raised an error of the wrong
#: class — the leak was blocked, but by the wrong error class.
_WRONG_ERROR_CLASS_DETAIL = "wrong error class"

#: Detail for a case whose attempt raised nothing at all — the
#: attempted leak was not blocked by any contract.
_UNBLOCKED_LEAK_DETAIL = "leak was NOT blocked"

#: Detail for a case blocked by exactly its expected error.
_BLOCKED_DETAIL = "blocked"


class LeakageCase:
    """One named leak attempt paired with its expected contract error.

    The attempt is a zero-argument callable: running it is the leak
    scenario (for example, admitting a forecast whose feature vintage
    postdates the decision instant). ``expected_error`` must be exactly
    one of :data:`FROZEN_CONTRACT_ERRORS` — never a foreign class — and
    admission is strict with ``ValueError`` for a blank or
    non-string ``name``, a non-callable ``attempt``, or an expected
    error outside the supported contract error classes (including
    non-class inputs).
    """

    __slots__ = ("attempt", "expected_error", "name")

    def __init__(
        self,
        name: str,
        attempt: Callable[[], Any],
        expected_error: type[Exception],
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "LeakageCase.name must be a non-blank string identifying "
                f"the leak scenario; got {name!r}."
            )
        if not callable(attempt):
            raise ValueError(  # noqa: TRY004 — admission rejects with ValueError
                "LeakageCase.attempt must be callable — the zero-argument "
                "leak scenario to execute; got "
                f"{type(attempt).__name__}: {attempt!r}."
            )
        if not isinstance(expected_error, type) or not issubclass(
            expected_error, Exception
        ):
            raise ValueError(  # noqa: TRY004 — admission rejects with ValueError
                "expected_error must be one of PortLearn's supported contract "
                "error types (portlearn.leakage.FROZEN_CONTRACT_ERRORS); "
                f"got {expected_error!r}."
            )
        if expected_error not in _FROZEN_ERROR_SET:
            raise ValueError(
                "expected_error must be one of PortLearn's supported contract "
                "error types (portlearn.leakage.FROZEN_CONTRACT_ERRORS); "
                f"{expected_error.__module__}.{expected_error.__qualname__} "
                "is not one of them."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "attempt", attempt)
        object.__setattr__(self, "expected_error", expected_error)

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError(
            f"LeakageCase is immutable: field {key!r} cannot be reassigned."
        )

    def __delattr__(self, key: str) -> None:
        raise AttributeError(
            f"LeakageCase is immutable: field {key!r} cannot be deleted."
        )

    def __repr__(self) -> str:
        return (
            f"LeakageCase(name={self.name!r}, "
            f"expected_error={self.expected_error.__qualname__})"
        )


class LeakageFinding:
    """The total outcome of executing one :class:`LeakageCase`.

    ``blocked`` is true only when the attempt raised exactly the case's
    expected error class; ``error_type`` is the class that was actually
    raised (``None`` when nothing was raised); ``detail`` is one of the
    three fixed outcome phrases, so reports carry fixed verdicts, never message tails.
    """

    __slots__ = ("blocked", "case", "detail", "error_type")

    def __init__(
        self,
        case: LeakageCase,
        blocked: bool,
        error_type: type[Exception] | None,
        detail: str,
    ) -> None:
        object.__setattr__(self, "case", case)
        object.__setattr__(self, "blocked", blocked)
        object.__setattr__(self, "error_type", error_type)
        object.__setattr__(self, "detail", detail)

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError(
            f"LeakageFinding is immutable: field {key!r} cannot be reassigned."
        )

    def __delattr__(self, key: str) -> None:
        raise AttributeError(
            f"LeakageFinding is immutable: field {key!r} cannot be deleted."
        )

    def __repr__(self) -> str:
        error_name = (
            self.error_type.__qualname__
            if self.error_type is not None
            else "None"
        )
        return (
            f"LeakageFinding(case={self.case.name!r}, "
            f"blocked={self.blocked!r}, error_type={error_name}, "
            f"detail={self.detail!r})"
        )


class LeakageReport:
    """Findings from one total battery run, in admission order.

    ``all_blocked`` and ``summary`` derive only from ``findings``; the
    empty report is admitted and summarises as ``"0/0 attempted leaks
    blocked"`` with ``all_blocked is True`` (vacuous truth: nothing
    leaked).
    """

    __slots__ = ("findings",)

    def __init__(self, findings: Sequence[LeakageFinding]) -> None:
        object.__setattr__(self, "findings", tuple(findings))

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError(
            f"LeakageReport is immutable: field {key!r} cannot be reassigned."
        )

    def __delattr__(self, key: str) -> None:
        raise AttributeError(
            f"LeakageReport is immutable: field {key!r} cannot be deleted."
        )

    @property
    def all_blocked(self) -> bool:
        """True when every finding reports its leak as blocked."""
        return all(finding.blocked for finding in self.findings)

    @property
    def summary(self) -> str:
        """``"{blocked}/{total} attempted leaks blocked"``."""
        blocked = sum(1 for finding in self.findings if finding.blocked)
        return f"{blocked}/{len(self.findings)} attempted leaks blocked"

    def __repr__(self) -> str:
        return f"LeakageReport(summary={self.summary!r})"


def run_leakage_cases(cases: Sequence[LeakageCase]) -> LeakageReport:
    """Execute ``cases`` totally, reporting every outcome.

    The battery is strict on admission — it must be a non-empty
    sequence of :class:`LeakageCase` (an empty battery proves nothing
    and is rejected with ``ValueError``). Every admitted case then
    produces exactly one finding:

    * the attempt raises the case's expected error class → blocked,
      ``error_type`` = the expected class, ``detail == "blocked"``;
    * the attempt raises a different exception class → not blocked,
      ``error_type`` = the actually-raised class,
      ``detail == "wrong error class"``;
    * the attempt raises nothing → not blocked, ``error_type is None``,
      ``detail == "leak was NOT blocked"``.

    A raised exception is never allowed to abort the run: outcomes are
    converted to findings, so an attempted leak can never disappear
    silently.
    """
    if not cases:
        raise ValueError(
            "run_leakage_cases requires a non-empty sequence of LeakageCase "
            "scenarios — an empty battery executes nothing and proves "
            "nothing."
        )
    findings: list[LeakageFinding] = []
    for case in cases:
        if not isinstance(case, LeakageCase):
            raise ValueError(  # noqa: TRY004 — admission is uniformly ValueError
                "run_leakage_cases requires LeakageCase scenarios; got "
                f"{type(case).__name__}: {case!r}."
            )
        try:
            case.attempt()
        except Exception as raised:  # noqa: BLE001 — totality is the law
            actual = type(raised)
            if actual is case.expected_error:
                findings.append(
                    LeakageFinding(
                        case, True, actual, _BLOCKED_DETAIL
                    )
                )
            else:
                findings.append(
                    LeakageFinding(
                        case, False, actual, _WRONG_ERROR_CLASS_DETAIL
                    )
                )
        else:
            findings.append(
                LeakageFinding(case, False, None, _UNBLOCKED_LEAK_DETAIL)
            )
    return LeakageReport(findings)
