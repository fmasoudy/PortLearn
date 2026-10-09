"""PortLearn — the declarative experiment-configuration surface (declaration).

This module owns the *declaration* side of the experiment boundary fixed
by experiment-boundary decision: a process-local strategy registration, the serializable
experiment configuration (stable ids, the four-role vocabulary, benchmark
metadata, one shared-conditions container), the four declarative
selection modes, and the uniform results collection. The experiment runtime
is the one reserved execution home (B3): the declared
``DecisionSchedule``, the injected ``ExperimentConditions`` backends,
and the ``Experiment`` object whose ``run`` applies the common
experimental conditions — one information supply, one point-in-time
universe, one accounting contract — to every selected strategy; execution
lives on ``Experiment.run`` and nowhere else.

Rules implemented (the declaration design):

* **Registration identity** — ``strategy_id`` is an explicit,
  stable, non-blank string carried verbatim; never derived from a class
  name, repr, or submission order. The live component is stored by
  reference, process-local, and never serialized (B2).
* **Structural protocol surface** — a registration admits only a
  component exposing a callable ``decide`` (the public
  ``portlearn.interfaces.Strategy`` surface); the check is structural
  (isinstance on the static-only protocol is impossible) and fires at
  registration, never at deserialization.
* **Role vocabulary** — exactly four roles (target, comparator,
  ablation, robustness) enter only through the four documented
  configuration fields; ``benchmark`` is comparator metadata (a subset
  of comparators), never a fifth role and never a strategy-class fact.
* **Cardinalities (B1)** — at least one registered id, exactly one
  target, zero or more comparators (a target-only declaration is valid
  structural state; comparison-readiness is a run-entry fact),
  one role per id, unassigned registered ids valid.
* **Shared conditions** — one conditions container under the results
  settings grammar; no per-strategy conditions, no merge or
  override semantics; top-level ``risk_aversion``, when present, is a
  finite non-bool number (results-schema decision 4).
* **Canonical form** — the declaration serializes to sorted-key,
  compact, ASCII, nonfinite-free JSON; equal declarations from permuted
  inputs produce identical bytes; the parser admits only the canonical
  spelling and rejects duplicate keys at every depth.
* **Canonical order** — id sequences and records are stored
  canonically sorted (by ``strategy_id``), blind to submission order.
* **Deep immutability** — every public record is public and
  deeply snapshotted; later caller mutation never reaches a record.

The module is stdlib-only and side-effect-free at import.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from itertools import pairwise
from types import MappingProxyType
from typing import Any, NoReturn

import portlearn as _package
from portlearn import costs as _costs
from portlearn import interfaces as _interfaces
from portlearn import ledger as _ledger
from portlearn import manifest as _manifest
from portlearn import metrics as _metrics
from portlearn import rebalance as _rebalance
from portlearn import results as _results
from portlearn import timing as _timing
from portlearn import weights as _weights

__all__ = [  # noqa: RUF022 -- the declaration §4 D1 pinned order, frozen by floor T1
    "ConfigurationError",
    "StrategyRegistration",
    "ExperimentConfiguration",
    "ExperimentSelection",
    "ExperimentResults",
    "select_all",
    "select_none",
    "select_single",
    "select_subset",
    "ExecutionError",
    "DecisionSchedule",
    "ExperimentConditions",
    "Experiment",
]


class ConfigurationError(ValueError):
    """A declaration violated one of the declaration construction rules.

    Module-owned (this class is defined here, so its ``__module__`` is
    ``portlearn.experiments``) and a ``ValueError``: every fail-closed
    rejection on the declarative surface — blank ids, duplicate ids,
    invalid role references, grammar violations, non-canonical text —
    raises exactly this type, never a bare stdlib error.
    """


# The four declarative selection modes (§4 D-table; "benchmark" is not a
# mode, and "everything" is not a word on this surface).
_SELECTION_MODES = ("full", "none", "single", "subset")

# The exact wire key set of the canonical declaration (§5.9): exact
# key-set equality holds at the root — no silent forward-compatibility.
_WIRE_KEYS = frozenset(
    {
        "ablations",
        "benchmarks",
        "comparators",
        "conditions",
        "robustness_checks",
        "strategy_ids",
        "target",
    }
)


# --------------------------------------------------------------------------- #
# Shared admission validators                                                 #
# --------------------------------------------------------------------------- #


def _dumps(value: Any) -> str:
    """The canonical serialization rule: sorted keys, compact, ASCII, no nonfinite."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _require_nonblank_str(value: Any, field: str) -> str:
    """Admit a non-blank string or reject with ``ConfigurationError``."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(
            f"{field} must be a non-blank string; got {value!r}. Blank or "
            "non-string values name no strategy on the experiment surface, "
            "so the value is rejected unconditionally."
        )
    return value


def _freeze_ids(value: Any, field: str) -> tuple[str, ...]:
    """Freeze an id sequence: non-blank strings, unique, canonically sorted.

    Submission order is never content: the stored tuple is
    sorted, so equal declarations from permuted inputs compare equal and
    serialize identically. A bare string (or bytes) is not a sequence of
    ids — it rejects, so a single id can never silently expand into its
    characters.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(
            f"{field} must be a sequence of strategy ids; got {value!r} of "
            f"type {type(value).__name__}. A single id or a mapping is not "
            "a sequence on this surface."
        )
    frozen = tuple(
        _require_nonblank_str(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    ordered = tuple(sorted(frozen))
    for previous, current in pairwise(ordered):
        if previous == current:
            raise ConfigurationError(
                f"{field} carries the duplicate strategy id {previous!r}: "
                "the declaration holds exactly one entry per distinct id "
                "— no suffixing, no quiet dedup, at construction and at "
                "parse."
            )
    return ordered


def _freeze_conditions(value: Any, field: str) -> Any:
    """Freeze one conditions value under the results settings grammar.

    Valid kinds: ``None``, ``bool``, ``int``, finite ``float``
    (signed zero unified to ``+0.0``), ``str``, sequences (stored as
    tuples), and mappings with non-blank string keys (stored as mapping
    proxies). Everything else — bytes, nonfinite floats, arbitrary live
    objects — rejects with ``ConfigurationError``. The snapshot is
    deep: later mutation of a caller's submitted container can never
    reach the declaration.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigurationError(
                f"{field} must be finite; got {value!r}. Conditions values "
                "are fail-closed in both directions under the settings "
                "grammar."
            )
        return value + 0.0
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        raise ConfigurationError(
            f"{field} must be a valid conditions value; bytes are not on "
            f"the settings grammar. Got {value!r}."
        )
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ConfigurationError(
                    f"{field} carries an invalid mapping key {key!r}: "
                    "keys must be non-blank strings."
                )
            frozen[key] = _freeze_conditions(item, f"{field}[{key!r}]")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence):
        return tuple(
            _freeze_conditions(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        )
    raise ConfigurationError(
        f"{field} must be a valid conditions value under the settings "
        "grammar (None, bool, int, finite float, str, sequence, or "
        f"mapping); got {value!r} of type {type(value).__name__}. Live "
        "objects never enter the declaration."
    )


def _conditions_wire(value: Any) -> Any:
    """Project an immutable conditions snapshot onto its JSON wire form."""
    if isinstance(value, Mapping):
        return {key: _conditions_wire(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_conditions_wire(item) for item in value]
    if isinstance(value, float):
        return value + 0.0
    return value


def _require_risk_aversion(conditions: Any) -> None:
    """The lookup-key rule: top-level ``risk_aversion`` is a finite
    non-bool number whenever present.

    Nested occurrences (inside a family mapping) are valid conditions
    content and never the decision lookup — only the top level of the
    shared container carries the synchronized interpretation.
    """
    if "risk_aversion" not in conditions:
        return
    value = conditions["risk_aversion"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(
            "conditions['risk_aversion'] must be a finite int or float "
            f"when declared at the top level; got {value!r}. Booleans and "
            "strings are not risk aversions on this surface."
        )
    if not math.isfinite(float(value)):
        raise ConfigurationError(
            f"conditions['risk_aversion'] must be finite; got {value!r}. "
            "The lookup key is fail-closed in both directions."
        )


#: The annualization executed when the declaration is silent on
#: ``periods_per_year``.  Whenever the key is declared, the run executes
#: the declared value exactly (F12/F18); this default covers only the
#: undeclared case and never enters the conformance record as a
#: declared/effective difference.
_UNDECLARED_PERIODS_PER_YEAR = 12


def _declared_periods_per_year(declared: Any) -> int:
    """The annualization the run executes for ``periods_per_year``.

    The declared top-level value whenever present — validated as a
    strictly positive non-bool ``int`` and failing closed with
    ``ExecutionError`` otherwise (an incoherent evaluation definition,
    never inferred or repaired) — and the undeclared default above only
    when the declaration is silent, in which case no declared/effective
    difference exists to record.
    """
    if not isinstance(declared, Mapping) or "periods_per_year" not in declared:
        return _UNDECLARED_PERIODS_PER_YEAR
    value = declared["periods_per_year"]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExecutionError(
            "conditions['periods_per_year'] must be a strictly positive "
            f"int when declared; got {value!r}. The annualization every "
            "metrics block executes is the declared evaluation "
            "definition — never an inferred, coerced, or repaired value "
            "(F12/F18)."
        )
    return value


#: The declared-conditions keys the runtime maps onto the frozen TARGET
#: validator's ``WeightConstraints``: each value is carried exactly as
#: declared onto the field of the same name, and no other conditions key
#: reaches a weight constraint (Law 10/V5 — the declared conditions are
#: the single source of the validation's parameters).
_TARGET_CONSTRAINT_FIELDS = (
    "allow_short",
    "max_gross_exposure",
    "max_abs_net_exposure",
    "budget",
    "budget_tolerance",
)


def _declared_target_constraints(declared: Any) -> Any:
    """Derive the public ``WeightConstraints`` exactly as declared.

    Only the five constraint keys above are read, each carried verbatim
    from the immutable declared-conditions snapshot onto the field of the
    same name; every absent key keeps the shipped default (an undeclared
    constraint is never inferred or tightened). Construction of the
    public constraints object itself validates the declarations and
    raises ``ValueError`` on any invalid one — the caller wraps it.
    """
    constraints: dict[str, Any] = {}
    if isinstance(declared, Mapping):
        for field in _TARGET_CONSTRAINT_FIELDS:
            if field in declared:
                constraints[field] = declared[field]
    return _weights.WeightConstraints(**constraints)


# --------------------------------------------------------------------------- #
# StrategyRegistration — the process-local live binding (B2)                  #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class StrategyRegistration:
    """One named live strategy component, process-local by design (B2).

    Exactly two fields — the explicit non-blank ``strategy_id`` (no
    default: omitting it is a ``TypeError``) and the live ``strategy``
    component — and nothing else: no role, no benchmark, no conditions,
    no serialization route. The structural ``decide`` check fires here,
    at registration: a component lacking the public
    ``Strategy`` surface rejects at construction, never later. The same
    component may lawfully occupy two ids, and two different components
    may lawfully occupy one id across re-registrations — the id, not
    the component, is the identity that survives into the declaration.
    """

    strategy_id: str
    strategy: Any

    def __post_init__(self) -> None:
        _require_nonblank_str(self.strategy_id, "StrategyRegistration.strategy_id")
        decide = getattr(self.strategy, "decide", None)
        if not callable(decide):
            raise ConfigurationError(
                "StrategyRegistration.strategy must expose the Strategy "
                "surface — a callable 'decide' attribute; got "
                f"{self.strategy!r} of type {type(self.strategy).__name__}. "
                "The check is structural and fires at registration, never "
                "at deserialization."
            )

    def __reduce__(self) -> NoReturn:
        # B2/F15: a registration is the process-local live binding — the
        # one artifact the canonical declaration bytes are for.  It is
        # never serialized, whatever backend sits behind it.
        raise TypeError(
            "StrategyRegistration is the process-local live binding (B2) "
            "and is never serialized: bind registrations against the "
            "canonical declaration text instead."
        )


# --------------------------------------------------------------------------- #
# ExperimentConfiguration — the serializable declaration (B2)                 #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ExperimentConfiguration:
    """The declarative experiment configuration: ids, roles, conditions.

    Stores stable ``strategy_ids`` plus the role fields, benchmark
    metadata, and the one shared-conditions container — never a live
    component (B2): binding configuration ids to live registrations is
    the runtime ``Experiment`` composition act. Structural validity (B1):
    at least one registered id, exactly one target (omitting it is a
    ``TypeError``; ``None`` is a configuration error), zero or more
    comparators — a target-only declaration is valid — one role per
    id, unassigned registered ids valid. Benchmark metadata names
    declared comparators only. The record is immutable and deeply
    snapshotted; equality and the canonical roundtrip cover the
    declaration only, and no component repr, class, import path,
    pickle, or factory token is ever serialized.
    """

    strategy_ids: Sequence[str]
    target: str
    comparators: Sequence[str] = ()
    ablations: Sequence[str] = ()
    robustness_checks: Sequence[str] = ()
    benchmarks: Sequence[str] = ()
    conditions: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = _freeze_ids(self.strategy_ids, "ExperimentConfiguration.strategy_ids")
        if not ids:
            raise ConfigurationError(
                "ExperimentConfiguration.strategy_ids must declare at "
                "least one strategy id; got an empty sequence. A "
                "declaration with zero registered ids names no experiment."
            )
        target = _require_nonblank_str(self.target, "ExperimentConfiguration.target")
        comparators = _freeze_ids(
            self.comparators, "ExperimentConfiguration.comparators"
        )
        ablations = _freeze_ids(self.ablations, "ExperimentConfiguration.ablations")
        robustness = _freeze_ids(
            self.robustness_checks, "ExperimentConfiguration.robustness_checks"
        )
        benchmarks = _freeze_ids(self.benchmarks, "ExperimentConfiguration.benchmarks")
        if self.conditions is None:
            raise ConfigurationError(
                "ExperimentConfiguration.conditions must be a conditions "
                "mapping, not None; omit the keyword entirely for the "
                "documented empty default (tri-state)."
            )
        conditions = _freeze_conditions(
            self.conditions, "ExperimentConfiguration.conditions"
        )
        _require_risk_aversion(conditions)

        # Unregistered role references reject (§5.4): every role field
        # names declared ids verbatim — no aliasing, no case-folding.
        for field, role_ids in (
            ("comparators", comparators),
            ("ablations", ablations),
            ("robustness_checks", robustness),
        ):
            unknown = [item for item in role_ids if item not in ids]
            if unknown:
                raise ConfigurationError(
                    f"ExperimentConfiguration.{field} references "
                    f"unregistered strategy ids {sorted(unknown)}: every "
                    "role must name a declared strategy_id verbatim."
                )
        if target not in ids:
            raise ConfigurationError(
                f"ExperimentConfiguration.target {target!r} is not among "
                f"the declared strategy_ids {list(ids)}: the target must "
                "name a declared id verbatim."
            )

        # One role per id (§5.4): the four role assignments are pairwise
        # disjoint — the target may not double as a comparator, and no id
        # may carry two non-target roles.
        assigned: dict[str, str] = {target: "target"}
        for field, role_ids in (
            ("comparators", comparators),
            ("ablations", ablations),
            ("robustness_checks", robustness),
        ):
            for item in role_ids:
                if item in assigned:
                    raise ConfigurationError(
                        f"the strategy id {item!r} carries two roles — "
                        f"{assigned[item]!r} and {field!r}: exactly one "
                        "role per id, and the target never doubles as "
                        "comparator metadata."
                    )
                assigned[item] = field

        # Benchmark metadata (§5.3): a subset of the declared
        # comparators — never the target, never an ablation, never an
        # unregistered or unassigned id.
        off_benchmarks = [item for item in benchmarks if item not in comparators]
        if off_benchmarks:
            raise ConfigurationError(
                "ExperimentConfiguration.benchmarks must name declared "
                f"comparators only; these name non-comparators: "
                f"{sorted(off_benchmarks)}. A benchmark is comparator "
                "metadata, never a fifth role."
            )

        object.__setattr__(self, "strategy_ids", ids)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "comparators", comparators)
        object.__setattr__(self, "ablations", ablations)
        object.__setattr__(self, "robustness_checks", robustness)
        object.__setattr__(self, "benchmarks", benchmarks)
        object.__setattr__(self, "conditions", conditions)

    def _wire(self) -> dict[str, Any]:
        """The canonical wire projection of the declaration."""
        return {
            "ablations": list(self.ablations),
            "benchmarks": list(self.benchmarks),
            "comparators": list(self.comparators),
            "conditions": _conditions_wire(self.conditions),
            "robustness_checks": list(self.robustness_checks),
            "strategy_ids": list(self.strategy_ids),
            "target": self.target,
        }

    def canonical_text(self) -> str:
        """The canonical JSON text of the declaration.

        Sorted keys, compact separators, ASCII, nonfinite-free — the
        canonical serialization rule. Equal declarations from permuted inputs produce
        identical bytes, because id sequences are stored canonically
        sorted and mappings sort at every level. The bytes cover the
        declaration only: no live object, class name, repr, import
        path, pickle, or factory token ever appears (B2).
        """
        return _dumps(self._wire())

    @classmethod
    def from_canonical_text(cls, text: str) -> ExperimentConfiguration:
        """Parse the canonical text back into an equal declaration.

        Fail-closed on every non-canonical spelling: indented, spaced,
        or unsorted variants reject (the parser re-serializes and
        compares bytes); duplicate keys reject at every depth; non-object
        roots and unknown fields reject; and the same construction rules
        fire again on the rebuilt values, so wire grammar violations
        (a boolean ``risk_aversion``, a nonfinite float, a blank key)
        reject exactly as they do at construction. The roundtrip
        reconstructs the declaration only — never a live object (B2).
        """
        if not isinstance(text, str):
            raise ConfigurationError(
                "the canonical declaration text must be a string; got "
                f"{text!r} of type {type(text).__name__}."
            )

        def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            node: dict[str, Any] = {}
            for key, value in pairs:
                if key in node:
                    raise ConfigurationError(
                        f"the canonical text carries the duplicate key "
                        f"{key!r}: exact key-set equality holds at every "
                        "object on the wire, at every depth."
                    )
                node[key] = value
            return node

        def _reject_constant(token: str) -> Any:
            raise ConfigurationError(
                f"the canonical form admits no nonfinite constant; got "
                f"{token!r}. Conditions values are fail-closed in both "
                "directions under the settings grammar."
            )

        try:
            payload = json.loads(
                text,
                object_pairs_hook=_object_pairs,
                parse_constant=_reject_constant,
            )
        except ConfigurationError:
            raise
        except ValueError as parse_error:
            raise ConfigurationError(
                f"the text is not parsable canonical JSON: {parse_error}. "
                "The parser admits the canonical spelling only."
            ) from None
        if not isinstance(payload, dict):
            raise ConfigurationError(
                "the canonical declaration is a JSON object; got "
                f"{type(payload).__name__}. The schema admits no other "
                "root."
            )
        extra = sorted(set(payload) - _WIRE_KEYS)
        missing = sorted(_WIRE_KEYS - set(payload))
        if extra or missing:
            raise ConfigurationError(
                f"the canonical declaration carries an invalid key set: "
                f"unexpected keys {extra}, missing keys {missing}. Exact "
                "key-set equality holds at the root."
            )
        declaration = cls(
            strategy_ids=payload["strategy_ids"],
            target=payload["target"],
            comparators=payload["comparators"],
            ablations=payload["ablations"],
            robustness_checks=payload["robustness_checks"],
            benchmarks=payload["benchmarks"],
            conditions=payload["conditions"],
        )
        if declaration.canonical_text() != text:
            raise ConfigurationError(
                "the text is not the canonical form of the declaration it "
                "parses to: only the canonical spelling — sorted keys, "
                "compact separators, ASCII — admits, with no variant and "
                "no alternative order."
            )
        return declaration


# --------------------------------------------------------------------------- #
# ExperimentSelection — declarative selection data (never an execution)       #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ExperimentSelection:
    """A declarative selection: one of the four modes plus the ids.

    Pure data — it never alters identity, roles, or the configuration,
    and never executes anything. ``strategy_ids`` is stored canonically
    sorted canonically, with no duplicates and no blank ids, regardless of
    submission order.
    """

    mode: str
    strategy_ids: Sequence[str] = ()

    def __post_init__(self) -> None:
        if self.mode not in _SELECTION_MODES:
            raise ConfigurationError(
                f"ExperimentSelection.mode must be one of "
                f"{list(_SELECTION_MODES)}; got {self.mode!r}. The four "
                "selection modes are the whole vocabulary."
            )
        object.__setattr__(
            self,
            "strategy_ids",
            _freeze_ids(self.strategy_ids, "ExperimentSelection.strategy_ids"),
        )


def _require_declared(config: ExperimentConfiguration, names: tuple[str, ...]) -> None:
    """Reject any selection name outside the declared ids, verbatim."""
    unknown = [name for name in names if name not in config.strategy_ids]
    if unknown:
        raise ConfigurationError(
            f"the selection names unregistered strategy ids {sorted(unknown)}: "
            "a selection names declared ids verbatim — no aliasing, no "
            "case-folding, no renaming."
        )


def select_all(config: ExperimentConfiguration) -> ExperimentSelection:
    """Select every declared strategy id — the ``full`` mode."""
    return ExperimentSelection(mode="full", strategy_ids=tuple(config.strategy_ids))


def select_none(config: ExperimentConfiguration) -> ExperimentSelection:
    """Select no strategy at all — the ``none`` mode (a valid state)."""
    return ExperimentSelection(mode="none", strategy_ids=())


def select_single(
    config: ExperimentConfiguration, strategy_id: str
) -> ExperimentSelection:
    """Select exactly one declared strategy id — the ``single`` mode."""
    name = _require_nonblank_str(strategy_id, "select_single strategy_id")
    _require_declared(config, (name,))
    return ExperimentSelection(mode="single", strategy_ids=(name,))


def select_subset(
    config: ExperimentConfiguration, strategy_ids: Sequence[str]
) -> ExperimentSelection:
    """Select a non-empty subset of declared ids — the ``subset`` mode."""
    names = _freeze_ids(strategy_ids, "select_subset strategy_ids")
    if not names:
        raise ConfigurationError(
            "select_subset requires at least one strategy id; an empty "
            "selection is the 'none' mode, not a subset."
        )
    _require_declared(config, names)
    return ExperimentSelection(mode="subset", strategy_ids=names)


def comparison_readiness(config: ExperimentConfiguration) -> bool:
    """The comparison-readiness check for one declaration.

    An explicit runtime surface, deliberately outside ``__all__``
    (discoverable by behavior, not by census): a declaration carrying at
    least one declared comparator is comparison-ready — the check
    affirms readiness — and a zero-comparator declaration fails closed
    with ``ExecutionError``: a target-only run is valid to *execute*
    (readiness gates comparison, never execution), but no
    comparison may proceed or be affirmed ready without a comparator to
    compare against.
    """
    if not isinstance(config, ExperimentConfiguration):
        raise ConfigurationError(
            "comparison_readiness checks a declared "
            f"ExperimentConfiguration; got {type(config).__name__}: "
            f"{config!r}."
        )
    if not tuple(config.comparators):
        raise ExecutionError(
            "the declaration names zero comparators: comparison "
            "readiness fails closed — a target-only run still executes "
            "lawfully, but no comparison is ready without a declared "
            "comparator (/ V4)."
        )
    return True


# --------------------------------------------------------------------------- #
# ExperimentResults — the uniform results collection                          #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ExperimentResults:
    """The uniform results collection: zero, one, or many — one type.

    Every run mode returns this collection, never a bare record and
    never ``None``. Records are anything carrying a non-blank string
    ``strategy_id`` (the results ``ResultRecord`` linkage rule: the id
    is verbatim the declared registry id); duplicate ids reject
    fail-closed, and the collection stores its records canonically
    ordered by ``strategy_id``, blind to submission order.
    """

    records: Sequence[Any] = ()

    def __post_init__(self) -> None:
        decorated: list[tuple[str, Any]] = []
        seen: set[str] = set()
        for record in self.records:
            record_id = getattr(record, "strategy_id", None)
            if not isinstance(record_id, str):
                raise ConfigurationError(
                    "ExperimentResults record strategy_id must be a "
                    "non-blank string; got "
                    f"{record_id!r}. Records without a non-blank strategy_id "
                    "reject fail-closed."
                )
            _require_nonblank_str(record_id, "ExperimentResults record strategy_id")
            if record_id in seen:
                raise ConfigurationError(
                    f"ExperimentResults carries two records for the "
                    f"strategy id {record_id!r}: exactly one record per "
                    "declared id — duplicates reject fail-closed."
                )
            seen.add(record_id)
            decorated.append((record_id, record))
        decorated.sort(key=lambda pair: pair[0])
        object.__setattr__(self, "records", tuple(record for _, record in decorated))

    def __iter__(self):
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)


# --------------------------------------------------------------------------- #
# The experiment runtime — the common experimental conditions (the one reserved home)   #
# --------------------------------------------------------------------------- #


class ExecutionError(ValueError):
    """A run-entry or runtime violation of the runtime execution rules.

    Module-owned (defined here, so ``__module__`` is
    ``portlearn.experiments``) and a ``ValueError``: every fail-closed
    rejection on the runtime surface — an unusable backend, an
    incoherent opening state, an incoherent schedule — raises exactly
    this type, never a bare stdlib error.
    """


# The frozen timing-error taxonomy that must always surface unchanged
# through the runtime (F20/F26/D13): a ``NaiveTimestampError``, an
# ``InvalidChronologyError``, or a ``FutureInformationError`` raised by
# any frozen seam inside the run is re-raised as-is — never
# generic-wrapped into ``ExecutionError``, never swallowed into a
# completed run.
_TIMING_TAXONOMY = (
    _timing.NaiveTimestampError,
    _timing.InvalidChronologyError,
    _timing.FutureInformationError,
)


_CONDITIONS_FIELDS = (
    "information_provider",
    "outcome_provider",
    "rebalance_policy",
    "accounting_engine",
    "cost_model",
    "initial_weights",
    "initial_wealth",
    "initial_last_rebalance_time",
)


class ExperimentConditions:
    """The injected common backends plus the explicit opening state.

    One container, injected whole at construction: the point-in-time
    information backend, the realized-outcome backend, the rebalance
    policy, the accounting engine, the cost model, and the three
    opening-state facts (``initial_weights``, ``initial_wealth``,
    ``initial_last_rebalance_time``). Nothing defaults, nothing is
    inferred: a missing field is preserved on the object as explicitly
    absent and rejects fail-closed with ``ExecutionError`` at run entry
    (all validation passes run before the first decision), and unknown
    fields are ``TypeError`` (the exact-field-set rule).
    """

    __slots__ = _CONDITIONS_FIELDS

    def __init__(
        self,
        information_provider: Any = None,
        outcome_provider: Any = None,
        rebalance_policy: Any = None,
        accounting_engine: Any = None,
        cost_model: Any = None,
        initial_weights: Any = None,
        initial_wealth: Any = None,
        initial_last_rebalance_time: Any = None,
    ) -> None:
        supplied = {
            "information_provider": information_provider,
            "outcome_provider": outcome_provider,
            "rebalance_policy": rebalance_policy,
            "accounting_engine": accounting_engine,
            "cost_model": cost_model,
            "initial_weights": initial_weights,
            "initial_wealth": initial_wealth,
            "initial_last_rebalance_time": initial_last_rebalance_time,
        }
        # An explicitly absent condition is preserved as absent — never
        # defaulted, never inferred.  Rejection is a run-entry act
        # (Law 6 — V1–V4 all run before the first decision), not a
        # construction act: Experiment.run fails closed on any absent
        # field before any strategy decides.
        for name in _CONDITIONS_FIELDS:
            setattr(self, name, supplied[name])

    def __repr__(self) -> str:
        return "ExperimentConditions(<injected common conditions>)"

    def __reduce__(self) -> NoReturn:
        # B2/F15: the injected conditions carry live process-local
        # backends (providers, engines, policies) — the live composition,
        # never a serializable artifact.
        raise TypeError(
            "ExperimentConditions carries the live process-local backends "
            "(B2) and is never serialized: re-inject the backends against "
            "the canonical declaration text instead."
        )


def _checked_universe(value: Any, owner: str) -> tuple[str, ...]:
    """A universe is a sequence of non-blank exact strings, no dups."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(
            f"{owner} must be a sequence of exact-string instrument "
            f"identifiers in caller-declared order; got {value!r}."
        )
    entries = tuple(value)
    seen: set[str] = set()
    for identifier in entries:
        _require_nonblank_str(identifier, f"{owner} entry")
        if identifier in seen:
            raise ConfigurationError(
                f"{owner} carries the duplicate identifier {identifier!r}: "
                "the universe names each instrument exactly once."
            )
        seen.add(identifier)
    return entries


class DecisionSchedule:
    """The declared decision/universe schedule with optional screening.

    ``decision_dates`` — aware instants, strictly increasing, no
    duplicates. ``universes`` — keyed by all and only the decision
    dates, each a duplicate-free sequence of non-blank exact strings.
    The optional paired screening contract (amendment, 2026-10-02):
    ``screening_instants`` and ``membership_available_times`` are one
    optional-but-complete pair — both absent consumes the declared
    universes unchanged; exactly one present rejects (declaration
    defect). Both present requires exact decision-date and
    universe-member coverage, aware datetimes, and
    ``membership_available_time <= screening_instant <= decision_date``
    — a naive carried datetime rejects with the reused public
    ``NaiveTimestampError``, and either inequality breach rejects with
    the reused public ``FutureInformationError``. Nothing is dropped,
    defaulted, inferred, or silently re-dated.
    """

    __slots__ = (
        "decision_dates",
        "membership_available_times",
        "screening_instants",
        "universes",
    )

    def __init__(
        self,
        decision_dates: Any,
        universes: Any,
        screening_instants: Any = None,
        membership_available_times: Any = None,
    ) -> None:
        if isinstance(decision_dates, (str, bytes)) or not isinstance(
            decision_dates, Sequence
        ):
            raise ConfigurationError(
                "DecisionSchedule decision_dates must be a sequence of "
                f"aware decision instants; got {decision_dates!r}."
            )
        checked: list[datetime] = []
        for value in decision_dates:
            try:
                instant = _timing.to_instant(value, "decision_dates entry")
            except Exception as exc:
                raise ConfigurationError(
                    "DecisionSchedule decision_dates must be timezone-aware "
                    "instants on the single timeline; a naive datetime or a "
                    f"date cannot be placed on it. Got {value!r}."
                ) from exc
            checked.append(instant)
        if not checked:
            raise ConfigurationError(
                "DecisionSchedule decision_dates must carry at least one "
                "decision instant."
            )
        for earlier, later in pairwise(checked):
            if not earlier < later:
                raise ConfigurationError(
                    "DecisionSchedule decision_dates must be strictly "
                    "increasing — two decisions may not share an instant or "
                    f"run backwards; got {later.isoformat()} not strictly "
                    f"after {earlier.isoformat()}."
                )
        if isinstance(universes, (str, bytes)) or not isinstance(universes, Mapping):
            raise ConfigurationError(
                "DecisionSchedule universes must be a mapping of decision "
                f"date to that date's declared universe; got {universes!r}."
            )
        supplied_universes = dict(universes)
        if set(supplied_universes) != set(checked):
            raise ConfigurationError(
                "DecisionSchedule universes must be keyed by all and only "
                "the declared decision dates "
                f"{[d.isoformat() for d in checked]}; got "
                f"{sorted(repr(k) for k in supplied_universes)}."
            )
        frozen_universes: dict[datetime, tuple[str, ...]] = {}
        for instant in checked:
            frozen_universes[instant] = _checked_universe(
                supplied_universes[instant],
                f"DecisionSchedule universes[{instant.isoformat()}]",
            )

        if (screening_instants is None) != (membership_available_times is None):
            raise ConfigurationError(
                "the screening contract is one optional-but-complete pair: "
                "screening_instants and membership_available_times are "
                "declared together or not at all — exactly one of the pair "
                "is an incomplete declaration that rejects fail-closed."
            )
        frozen_instants: dict[datetime, datetime] | None = None
        frozen_books: dict[datetime, dict[str, datetime]] | None = None
        if screening_instants is not None and membership_available_times is not None:
            if isinstance(screening_instants, (str, bytes)) or not isinstance(
                screening_instants, Mapping
            ):
                raise ConfigurationError(
                    "DecisionSchedule screening_instants must be a mapping "
                    "of decision date to that date's screening instant; got "
                    f"{screening_instants!r}."
                )
            supplied_instants = dict(screening_instants)
            if set(supplied_instants) != set(checked):
                raise ConfigurationError(
                    "DecisionSchedule screening_instants must be keyed by "
                    "all and only the declared decision dates; a missing or "
                    "extra decision date is a coverage defect that rejects "
                    "fail-closed."
                )
            frozen_instants = {}
            for instant in checked:
                screened = _timing.to_instant(
                    supplied_instants[instant], "screening_instants entry"
                )
                if screened > instant:
                    raise _timing.FutureInformationError(
                        "a screening instant may not postdate the decision "
                        "date it screens for: screening at "
                        f"{screened.isoformat()} for the decision at "
                        f"{instant.isoformat()} would admit membership "
                        "evidence that did not exist yet."
                    )
                frozen_instants[instant] = screened
            if isinstance(membership_available_times, (str, bytes)) or not isinstance(
                membership_available_times, Mapping
            ):
                raise ConfigurationError(
                    "DecisionSchedule membership_available_times must be a "
                    "mapping of decision date to that date's per-asset "
                    "membership availability book; got "
                    f"{membership_available_times!r}."
                )
            supplied_books = dict(membership_available_times)
            if set(supplied_books) != set(checked):
                raise ConfigurationError(
                    "DecisionSchedule membership_available_times must be "
                    "keyed by all and only the declared decision dates; a "
                    "missing or extra decision date is a coverage defect "
                    "that rejects fail-closed."
                )
            frozen_books = {}
            for instant in checked:
                inner = supplied_books[instant]
                if isinstance(inner, (str, bytes)) or not isinstance(inner, Mapping):
                    raise ConfigurationError(
                        "the membership availability book for the decision "
                        f"at {instant.isoformat()} must be a mapping of "
                        f"asset identifier to aware instant; got {inner!r}."
                    )
                entries = dict(inner)
                members = frozen_universes[instant]
                if set(entries) != set(members):
                    raise ConfigurationError(
                        "the membership availability book for the decision "
                        f"at {instant.isoformat()} must cover exactly the "
                        f"declared universe members {list(members)}; a "
                        "missing, extra, or duplicated member is a coverage "
                        "defect that rejects fail-closed."
                    )
                row: dict[str, datetime] = {}
                for asset in members:
                    available = _timing.to_instant(
                        entries[asset], "membership available time"
                    )
                    if available > frozen_instants[instant]:
                        raise _timing.FutureInformationError(
                            f"membership evidence for {asset!r} is dated "
                            f"{available.isoformat()}, after the screening "
                            f"instant {frozen_instants[instant].isoformat()}: "
                            "admitting it would let the screened universe "
                            "see information that did not exist yet."
                        )
                    row[asset] = available
                frozen_books[instant] = row

        object.__setattr__(self, "decision_dates", tuple(checked))
        self.universes = MappingProxyType(frozen_universes)
        self.screening_instants = (
            MappingProxyType(frozen_instants) if frozen_instants is not None else None
        )
        self.membership_available_times = (
            MappingProxyType(frozen_books) if frozen_books is not None else None
        )

    def __repr__(self) -> str:
        return (
            "DecisionSchedule(decision_dates="
            f"{[d.isoformat() for d in self.decision_dates]})"
        )


class _ArmExecution:
    """The mutable per-strategy execution state threaded across decision dates.

    One instance per selected strategy, owned by the date-major loop: the
    strategy's own PRE_TRADE book (POST_TRADE at an executed instant,
    drifted to the next date's PRE_TRADE by the fixed drift rule), the
    strategy's executed decisions in decision order, and the strategy's carried
    ``strategy_state`` — seeded absent (``None``, the valid opening)
    and advanced solely by each ``DecisionResult.next_strategy_state``.
    Strategy-local by construction: no field is ever shared across strategies.
    """

    __slots__ = ("executed", "pre", "strategy_state")

    def __init__(self, pre: Any) -> None:
        self.pre = pre
        self.executed: list[Any] = []
        self.strategy_state: Any = None


class Experiment:
    """The one reserved runtime object: ``run`` applies the common
    experimental conditions to every selected strategy.

    Constructed from a declared ``ExperimentConfiguration``, the live
    registrations binding every declared id exactly (fail-closed at
    construction: a missing id, an unregistered id, or a duplicate
    registration rejects), the declared ``DecisionSchedule``, and the
    injected ``ExperimentConditions``. ``run(selection)`` validates
    the backends and opening state before the first decision date
    (V1–V4), then applies one common information supply and one
    point-in-time universe to every selected strategy, composing the public
    interfaces (``DecisionContext``, ``require_decision_result_compatible``,
    the accounting engine, ``drift_weights``, ``build_ledger``) into
    one ``ResultRecord`` per strategy — returned as the uniform
    ``ExperimentResults`` collection for every mode, with ``None`` the
    full mode.
    """

    __slots__ = ("_conditions", "_config", "_schedule", "_strategies")

    def __init__(
        self,
        config: Any,
        registrations: Any,
        schedule: Any,
        conditions: Any,
    ) -> None:
        if not isinstance(config, ExperimentConfiguration):
            raise ConfigurationError(
                "Experiment binds a declared ExperimentConfiguration; got "
                f"{type(config).__name__}: {config!r}."
            )
        if not isinstance(schedule, DecisionSchedule):
            raise ConfigurationError(
                "Experiment runs on a declared DecisionSchedule; got "
                f"{type(schedule).__name__}: {schedule!r}."
            )
        if not isinstance(conditions, ExperimentConditions):
            raise ConfigurationError(
                "Experiment runs under injected ExperimentConditions; got "
                f"{type(conditions).__name__}: {conditions!r}."
            )
        if isinstance(registrations, StrategyRegistration) or not isinstance(
            registrations, Sequence
        ):
            raise ConfigurationError(
                "Experiment binds a sequence of StrategyRegistration "
                f"objects, one per declared id; got {registrations!r}."
            )
        strategies: dict[str, Any] = {}
        for registration in registrations:
            if not isinstance(registration, StrategyRegistration):
                raise ConfigurationError(
                    "Experiment binds only StrategyRegistration objects "
                    "(the process-local live binding); got "
                    f"{type(registration).__name__}: {registration!r}."
                )
            if registration.strategy_id in strategies:
                raise ConfigurationError(
                    "the registrations carry two live bindings for the "
                    f"strategy id {registration.strategy_id!r}: exactly one "
                    "registration per declared id — duplicates reject "
                    "fail-closed."
                )
            strategies[registration.strategy_id] = registration.strategy
        declared = set(config.strategy_ids)
        if set(strategies) != declared:
            raise ConfigurationError(
                "Experiment binds the declared ids to live registrations "
                "exactly: the declaration names "
                f"{sorted(declared)}, the registrations name "
                f"{sorted(strategies)} — a missing id, an unregistered id, "
                "or a case variant never folds and never binds."
            )
        self._config = config
        self._strategies = strategies
        self._schedule = schedule
        self._conditions = conditions

    def __reduce__(self) -> NoReturn:
        # B2/F15: the experiment is the live composition — declaration
        # plus registrations, schedule, and injected backends — bound at
        # construction and process-local ever after.  The canonical
        # declaration bytes are the one serializable artifact.
        raise TypeError(
            "Experiment is the live process-local composition (B2) and is "
            "never serialized: serialize the configuration's canonical "
            "text and re-compose from live registrations instead."
        )

    # -- run-entry validation (V1–V4), all before the first decision date -- #

    def _checked_conditions_present(self) -> None:
        """V3 presence: every injected condition is present, none absent.

        An explicitly absent condition was preserved as absent on the
        deeply immutable ``ExperimentConditions`` object (never
        defaulted, never inferred at construction); run entry is the
        mandated rejection point — V1–V4 all run before the first
        decision, so no strategy ever decides against a partial
        injection (B2: absent opening state rejects at entry).
        """
        for name in _CONDITIONS_FIELDS:
            if getattr(self._conditions, name) is None:
                raise ExecutionError(
                    f"ExperimentConditions requires the field {name!r}: "
                    "the opening state (initial_weights, initial_wealth, "
                    "initial_last_rebalance_time) and the common backends "
                    "are injected whole — a missing condition never "
                    "defaults and is never inferred, and run entry "
                    "rejects it before the first decision."
                )

    def _checked_entry_contract(self) -> None:
        """F30/F32 run-entry backend-contract checks (V3-era companions).

        Two rules are checked after the V3 presence check and before any
        provider is consumed:

        * the realized-outcome supply must be a backend distinct from
          the information supply — one object may never serve both the
          decision-information and the realized-outcome surfaces
          (B1/A1); and
        * the transaction-cost model must be the actual shipped
          ``Proportional`` object, accepted by real type only (B3/M18):
          a behaviorally identical duck-typed replica or a wrapper
          holding the genuine model rejects here without ever being
          consumed — the public ledger route is the shipped model's own
          route, never a generic cost abstraction.
        """
        conditions = self._conditions
        if conditions.outcome_provider is conditions.information_provider:
            raise ExecutionError(
                "the injected outcome_provider and information_provider "
                "are the same object: the realized-outcome supply is a "
                "distinct injected callable, never the information "
                "provider and never an InformationSet (B1/A1) — one "
                "object may not serve both the decision-information and "
                "the realized-outcome surfaces."
            )
        if not isinstance(conditions.cost_model, _costs.Proportional):
            raise ExecutionError(
                "the injected cost_model must be the shipped "
                "portlearn.costs.Proportional object — the runtime's "
                "transaction-cost surface is the shipped proportional "
                "model itself, accepted by actual type and identity and "
                "never probed (B3/M18); got "
                f"{type(conditions.cost_model).__name__}: "
                f"{conditions.cost_model!r}."
            )

    def _admitted_information(self, dates: Sequence[datetime]) -> dict[datetime, Any]:
        provider = self._conditions.information_provider
        admitted: dict[datetime, Any] = {}
        for date in dates:
            try:
                supplied = provider(date)
            except Exception as exc:
                raise ExecutionError(
                    "the information backend failed to resolve the decision "
                    f"date {date.isoformat()}: {exc!r}. The common "
                    "information supply must resolve for every declared "
                    "decision date."
                ) from exc
            if not isinstance(supplied, _interfaces.InformationSet):
                raise ExecutionError(
                    "the information backend must return a portlearn "
                    "InformationSet for every declared decision date; got "
                    f"{type(supplied).__name__}: {supplied!r} at "
                    f"{date.isoformat()}."
                )
            if supplied.as_of > date:
                raise ExecutionError(
                    "the information backend returned an information set "
                    f"whose as_of ({supplied.as_of.isoformat()}) is after "
                    f"the decision date ({date.isoformat()}): "
                    "post-decision information may not enter a decision."
                )
            admitted[date] = supplied
        return admitted

    def _forecast_material(
        self,
        dates: Sequence[datetime],
        information: Mapping[datetime, Any],
    ) -> tuple[dict[datetime, Any], list[Any]]:
        """Extract the forecast-shaped channels from the admitted sets (F19).

        The two valid public channels through which forecast material
        reaches the runtime are the admitted observations' values: a
        ``Forecast`` value is the decision context's forecast channel
        for its date, and a ``ForecastLink`` value is temporal-protocol provenance
        recorded as-is on the record. The point-in-time admission
        boundary has already fired at the public ``InformationSet``
        construction (the public ``FutureInformationError`` surfaced by
        the backend-resolution rule); no fitting-lifecycle validation exists
        here, so a present link is carried through unchanged — never
        interpreted, never re-validated, never dropped. Extraction
        reads only the entry-resolved sets: providers are never called.
        """
        forecasts: dict[datetime, Any] = {}
        links: list[Any] = []
        for date in dates:
            forecast: Any = None
            for record in information[date]:
                value = record.value
                if isinstance(value, _interfaces.Forecast):
                    if forecast is not None:
                        raise ExecutionError(
                            "the admitted information set for the decision "
                            f"date {date.isoformat()} carries more than one "
                            "Forecast-valued observation: the decision "
                            "context carries exactly one forecast channel "
                            "per decision instant, so an ambiguous supply "
                            "rejects fail-closed."
                        )
                    forecast = value
                elif isinstance(value, _results.ForecastLink):
                    # One recorded entry per distinct link value (the
                    # frozen record's exactly-one-per-distinct-value
                    # law); equal repeats across dates collapse to the
                    # first occurrence, never re-appended.
                    if not any(value == seen for seen in links):
                        links.append(value)
            forecasts[date] = forecast
        return forecasts, links

    def _outcome_books(
        self, dates: Sequence[datetime]
    ) -> dict[datetime, dict[str, float]]:
        provider = self._conditions.outcome_provider
        universes = self._schedule.universes
        books: dict[datetime, dict[str, float]] = {}
        for start in dates[:-1]:
            try:
                supplied = provider(start)
            except Exception as exc:
                raise ExecutionError(
                    "the outcome backend failed to resolve the holding "
                    f"segment start {start.isoformat()}: {exc!r}. The "
                    "common outcome supply must resolve for every required "
                    "segment start."
                ) from exc
            universe = universes[start]
            if isinstance(supplied, (str, bytes)) or not isinstance(supplied, Mapping):
                raise ExecutionError(
                    "the outcome backend must return a mapping of asset "
                    "identifiers to gross growth factors for the segment "
                    f"starting {start.isoformat()}; got "
                    f"{type(supplied).__name__}: {supplied!r}."
                )
            if len(supplied) != len(universe) or set(supplied) != set(universe):
                raise ExecutionError(
                    "the outcome book for the segment starting "
                    f"{start.isoformat()} must cover exactly the declared "
                    f"universe {list(universe)} — a missing, extra, or "
                    "duplicated identifier rejects fail-closed; got "
                    f"{sorted(repr(k) for k in supplied)}."
                )
            checked: dict[str, float] = {}
            for asset in universe:
                factor = supplied[asset]
                if (
                    isinstance(factor, bool)
                    or not isinstance(factor, (int, float))
                    or not math.isfinite(factor)
                ):
                    raise ExecutionError(
                        f"the outcome factor for {asset!r} on the segment "
                        f"starting {start.isoformat()} must be a finite "
                        f"real; got {factor!r}."
                    )
                if factor < 0:
                    # F30/B1: a negative gross growth factor is an entry
                    # defect — rejected here as the runtime's own
                    # ExecutionError, before any strategy decides.  The
                    # all-zero ledger-mandated failure occurring later in
                    # the frozen drift/ledger route keeps its raw frozen
                    # ValueError; only this declared-unlawful factor is
                    # wrapped at entry.
                    raise ExecutionError(
                        "the outcome factor for "
                        f"{asset!r} on the segment starting "
                        f"{start.isoformat()} is negative: gross growth "
                        f"factors (g = 1 + r) must be nonnegative; got "
                        f"{factor!r}."
                    )
                checked[asset] = float(factor)
            books[start] = checked
        return books

    def _checked_opening(
        self, first_date: datetime
    ) -> tuple[dict[str, float], float, datetime]:
        conditions = self._conditions
        universe = self._schedule.universes[first_date]
        supplied = conditions.initial_weights
        if isinstance(supplied, (str, bytes)) or not isinstance(supplied, Mapping):
            raise ExecutionError(
                "the opening state is invalid: initial_weights must be a "
                f"mapping of asset identifiers to weights; got {supplied!r}."
            )
        if len(supplied) != len(universe) or set(supplied) != set(universe):
            raise ExecutionError(
                "the opening state is invalid: initial_weights must cover "
                f"exactly the first decision date's declared universe "
                f"{list(universe)} — a missing, extra, or duplicated "
                "identifier rejects fail-closed."
            )
        opening: dict[str, float] = {}
        for asset in universe:
            weight = supplied[asset]
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
            ):
                raise ExecutionError(
                    "the opening state is invalid: the initial weight for "
                    f"{asset!r} must be a finite real; got {weight!r}."
                )
            opening[asset] = float(weight)
        if abs(math.fsum(opening.values()) - 1.0) > 1e-9:
            raise ExecutionError(
                "the opening state is invalid: initial_weights must budget "
                "exactly 1.0 (the weight identity every book on the wealth "
                f"path carries); got {math.fsum(opening.values())!r}."
            )
        wealth = conditions.initial_wealth
        if (
            isinstance(wealth, bool)
            or not isinstance(wealth, (int, float))
            or not math.isfinite(wealth)
            or wealth <= 0
        ):
            raise ExecutionError(
                "the opening state is invalid: initial_wealth must be a "
                f"strictly positive finite real; got {wealth!r}."
            )
        try:
            last_rebalance = _timing.to_instant(
                conditions.initial_last_rebalance_time,
                "initial_last_rebalance_time",
            )
        except Exception as exc:
            raise ExecutionError(
                "the opening state is invalid: "
                "initial_last_rebalance_time must be a timezone-aware "
                "instant; got "
                f"{conditions.initial_last_rebalance_time!r}."
            ) from exc
        if last_rebalance > first_date:
            raise ExecutionError(
                "the opening state is invalid: initial_last_rebalance_time "
                f"({last_rebalance.isoformat()}) may not postdate the first "
                f"decision date ({first_date.isoformat()})."
            )
        return opening, float(wealth), last_rebalance

    # -- the common run loop -------------------------------------------------- #

    def run(self, selection: Any = None) -> ExperimentResults:
        if selection is None:
            selected = tuple(self._config.strategy_ids)
        elif isinstance(selection, ExperimentSelection):
            selected = () if selection.mode == "none" else tuple(selection.strategy_ids)
        else:
            raise ConfigurationError(
                "Experiment.run accepts an ExperimentSelection or None (the "
                f"full mode); got {type(selection).__name__}: {selection!r}."
            )
        unbound = [name for name in selected if name not in self._strategies]
        if unbound:
            raise ConfigurationError(
                f"the selection names unbound strategy ids {sorted(unbound)}: "
                "run executes only ids bound to live registrations at "
                "construction."
            )
        dates = tuple(self._schedule.decision_dates)
        if len(dates) < 2:
            raise ExecutionError(
                "the declared schedule must carry at least two decision "
                "dates to bound an accounting horizon."
            )
        self._checked_conditions_present()
        self._checked_entry_contract()
        # F12/F18: the annualization every metrics block will execute is
        # resolved from the declared common conditions at run entry —
        # validated fail-closed here (a strictly positive non-bool int),
        # never hardcoded at the metrics call — so the effective
        # settings composed onto each record are the values the run
        # actually executed, and no declared/effective difference can be
        # concealed behind a copied declaration.
        periods_per_year = _declared_periods_per_year(self._config.conditions)
        # V1/V3 admission is resolved at entry — before the first
        # decision and never inside the decision loop — but only when a
        # nonempty arm set will consume the resolutions: a none-mode run
        # selects no arm, so no provider is ever called (F29/Law 12b).
        information = self._admitted_information(dates) if selected else {}
        books: dict[datetime, dict[str, float]] = (
            self._outcome_books(dates) if selected else {}
        )
        # F19: the forecast channel is extracted from the entry-resolved
        # admitted sets — providers are never re-called — and the temporal-protocol
        # provenance links are collected once per run, recorded as-is.
        forecasts, links = (
            self._forecast_material(dates, information) if selected else ({}, [])
        )
        opening, wealth, opening_last_rebalance = self._checked_opening(dates[0])

        records = []
        if selected:
            created_at = datetime.now(UTC)
            manifest = _manifest.RunManifest(
                run_id=f"portlearn-experiment-{created_at:%Y%m%dT%H%M%S%f}",
                created_at=created_at,
                python_version=".".join(map(str, sys.version_info[:3])),
                package_version=_package.__version__,
                dependency_pins={"portlearn": _package.__version__},
                commands=("portlearn.experiments.Experiment.run",),
            )
            arms = self._execute_arms(
                selected,
                dates,
                information,
                forecasts,
                books,
                opening,
                opening_last_rebalance,
            )
            for strategy_id in sorted(selected):
                records.append(
                    self._compose_record(
                        strategy_id,
                        dates,
                        books,
                        opening,
                        wealth,
                        arms[strategy_id].executed,
                        manifest,
                        periods_per_year,
                        links,
                    )
                )
        return ExperimentResults(records=records)

    def _execute_arms(
        self,
        selected: Sequence[str],
        dates: Sequence[datetime],
        information: Mapping[datetime, Any],
        forecasts: Mapping[datetime, Any],
        books: Mapping[datetime, Mapping[str, float]],
        opening: Mapping[str, float],
        opening_last_rebalance: datetime,
    ) -> dict[str, _ArmExecution]:
        """Advance every selected strategy over the dates in date-major order.

        The outer loop is the declared date order; the inner loop is the
        canonical ``strategy_id`` order. The rebalance
        policy is queried exactly once per decision date per run and the
        resolved answer binds every strategy identically (one
        shared cadence fact, never a per-strategy query); the shared
        ``last_rebalance`` instant advances only when the date's answer
        is True, and then to the executed decision's own instant. Each
        strategy threads its own PRE_TRADE book and ``strategy_state``
        (``DecisionContext.strategy_state`` in,
        ``DecisionResult.next_strategy_state`` out) — strategy-local, never
        cross-threaded, never dropped.

        F21/F22: a strategy, accounting-engine, or rebalance-policy
        failure inside this loop ends the whole run fail-closed as one
        module-owned ``ExecutionError`` — no partial collection ever
        escapes — while the public timing-error taxonomy and the public
        weights-validation ``ValueError`` routes keep surfacing
        unchanged (F17/F20/F26). Providers are never called here: the
        admitted information, the forecast channel, and the outcome
        books were all resolved once at run entry.
        """
        conditions = self._conditions
        universes = self._schedule.universes
        last_date = dates[-1]
        arms: dict[str, _ArmExecution] = {
            strategy_id: _ArmExecution(
                pre=_weights.PortfolioWeights(
                    dict(opening), _weights.WeightState.PRE_TRADE
                ),
            )
            for strategy_id in sorted(selected)
        }
        last_rebalance = opening_last_rebalance
        for index, date in enumerate(dates):
            try:
                rebalance = conditions.rebalance_policy.should_rebalance(
                    date, last_rebalance
                )
            except _TIMING_TAXONOMY:
                raise
            except Exception as exc:
                raise ExecutionError(
                    "the rebalance policy failed inside the decision loop "
                    f"at {date.isoformat()}: {exc!r}. The run ends "
                    "fail-closed with no partial result."
                ) from exc
            # The shared instant advances to the executed decision's own
            # instant (the latest any arm executed) once every arm has
            # decided at this date.
            executed_instant: datetime | None = None
            for strategy_id in sorted(selected):
                arm = arms[strategy_id]
                # F19: the lawfully admitted forecast threads into the
                # context; the frozen ``DecisionContext`` construction
                # laws (the forecast-dating alignment among them) fire
                # here and surface unchanged (F20).
                context = _interfaces.DecisionContext(
                    decision_time=date,
                    information=information[date],
                    universe=universes[date],
                    current_weights=arm.pre,
                    current_weights_as_of=date,
                    strategy_state=arm.strategy_state,
                    forecast=forecasts[date],
                )
                if not rebalance:
                    # The lawful explicit-hold rule: no TARGET is demanded,
                    # the arm's state never advances, and only the frozen
                    # drift runs for that arm at that date.
                    if date != last_date:
                        arm.pre = _rebalance.drift_weights(arm.pre, books[date])
                    continue
                # F16/F17 (Law 10/V5): immediately after the frozen
                # compatibility check and before any accounting, the
                # emitted TARGET flows through the frozen weights
                # validator centrally — a PortfolioWeights TARGET
                # constructed from the emitted book, validated against
                # WeightConstraints derived exactly from the declared
                # conditions.  Every ValueError that construction or
                # validation raises is the runtime's fail-closed
                # ExecutionError carrying the frozen offending facts
                # (identifiers, exposures, tolerances) — never a repair,
                # never a normalization, never a silent clip.  A book
                # admitted by the declared constraints passes into
                # accounting unchanged (F17: the declared tolerance is
                # never tightened here; a lawful-but-off-budget book's
                # later ledger rejection keeps the ledger's own facts).
                try:
                    result = self._strategies[strategy_id].decide(context)
                except _TIMING_TAXONOMY:
                    raise
                except ValueError as exc:
                    # The emitted decision book itself is unlawful (a
                    # blank asset identifier, a non-finite weight): the
                    # frozen interface's construction facts, wrapped.
                    raise ExecutionError(
                        f"the strategy {strategy_id!r} emitted an invalid "
                        f"TARGET at {date.isoformat()}: {exc}. The "
                        "emitted book is neither repaired, normalized, "
                        "nor clipped."
                    ) from exc
                except Exception as exc:
                    # F21/F22: the strategy itself failed inside the
                    # decision loop — one fail-closed ExecutionError, no
                    # partial collection.
                    raise ExecutionError(
                        f"the strategy {strategy_id!r} failed inside the "
                        f"decision loop at {date.isoformat()}: {exc!r}. "
                        "The run ends fail-closed with no partial result."
                    ) from exc
                _interfaces.require_decision_result_compatible(context, result)
                decision = result.decision
                try:
                    constraints = _declared_target_constraints(self._config.conditions)
                except ValueError as exc:
                    # The declared conditions themselves are unlawful as
                    # TARGET constraints (an incoherent or ill-typed
                    # declaration): the frozen constraints object's own
                    # facts, wrapped fail-closed at the deciding date.
                    raise ExecutionError(
                        f"the declared target constraints are invalid "
                        f"for the TARGET emitted by {strategy_id!r} at "
                        f"{date.isoformat()}: {exc}."
                    ) from exc
                try:
                    target = _weights.PortfolioWeights(
                        dict(decision.target_weights),
                        _weights.WeightState.TARGET,
                    )
                    _weights.require_valid_target(target, constraints)
                except ValueError as exc:
                    raise ExecutionError(
                        f"the strategy {strategy_id!r} emitted an invalid "
                        f"TARGET at {date.isoformat()}: {exc}. The "
                        "declared target constraints are "
                        f"{constraints!r}; the emitted book is neither "
                        "repaired, normalized, nor clipped."
                    ) from exc
                realized = {} if index == 0 else dict(books.get(dates[index - 1], {}))
                try:
                    accounted = conditions.accounting_engine.account(
                        decision, arm.pre, realized
                    )
                except _TIMING_TAXONOMY:
                    raise
                except Exception as exc:
                    # F21/F22: the accounting backend failed inside the
                    # decision loop — one fail-closed ExecutionError, no
                    # partial collection.
                    raise ExecutionError(
                        "the accounting engine failed inside the decision "
                        f"loop for {strategy_id!r} at {date.isoformat()}: "
                        f"{exc!r}. The run ends fail-closed with no "
                        "partial result."
                    ) from exc
                arm.pre = _weights.PortfolioWeights(
                    dict(accounted.post_trade_weights),
                    _weights.WeightState.POST_TRADE,
                )
                arm.executed.append(decision)
                arm.strategy_state = result.next_strategy_state
                if executed_instant is None or (
                    decision.execution_time > executed_instant
                ):
                    executed_instant = decision.execution_time
                if date != last_date:
                    arm.pre = _rebalance.drift_weights(arm.pre, books[date])
            if rebalance:
                last_rebalance = (
                    executed_instant if executed_instant is not None else date
                )
        return arms

    def _compose_record(
        self,
        strategy_id: str,
        dates: Sequence[datetime],
        books: Mapping[datetime, Mapping[str, float]],
        opening: Mapping[str, float],
        wealth: float,
        executed: Sequence[Any],
        manifest: Any,
        periods_per_year: int,
        links: Sequence[Any] = (),
    ) -> Any:
        """Compose the public ledger/metrics tail for one executed strategy.

        Every behavior here is the unchanged runtime tail of the former
        per-strategy run: the public ``build_ledger`` route over the strategy's
        own executed decisions, the declared-conditions metrics block,
        and the results conformance record — one ``ResultRecord`` per strategy.
        The metrics block executes exactly the ``periods_per_year``
        resolved at run entry from the declared conditions (F12/F18),
        and the conformance record's effective settings report that
        executed value — the record can never conceal a declared/
        effective difference nor self-sanction one. F19: the temporal-protocol
        provenance links collected once at entry are recorded as-is on
        the record (the public ``ResultRecord`` ordering/dedup rule
        applies inside the record itself). F22: a cost-model failure
        at ledger construction — after the loop but still inside the
        run — ends the run fail-closed as one ``ExecutionError`` with
        no partial collection, while the public ledger's own valid
        raw ``ValueError`` rejections (the all-zero denominator, the
        unit-budget rule's exact emitted book) keep surfacing
        unchanged (F17/F30).
        """
        conditions = self._conditions
        last_date = dates[-1]
        try:
            ledger = _ledger.build_ledger(
                dict(opening),
                list(dates),
                {start: dict(book) for start, book in books.items()},
                [d for d in executed if d.execution_time < last_date],
                conditions.cost_model,
                conditions.accounting_engine,
                initial_wealth=wealth,
            )
        except _TIMING_TAXONOMY:
            raise
        except ValueError:
            # The frozen ledger's own lawful raw rejections (the
            # all-zero denominator, the unit-budget law naming the
            # exact emitted net) surface unchanged — F17/F30.
            raise
        except Exception as exc:
            # F22: a backend failure at ledger construction — here the
            # cost model's ``f_cost`` — happens after the loop but
            # still inside the run, so the same fail-closed law binds:
            # one module-owned ExecutionError, no partial collection.
            raise ExecutionError(
                f"the cost model failed while building the ledger for "
                f"{strategy_id!r}: {exc!r}. The run ends fail-closed with "
                "no partial result."
            ) from exc
        declared = self._config.conditions
        gamma = 1.0
        if isinstance(declared, Mapping):
            aversion = declared.get("risk_aversion")
            if (
                isinstance(aversion, (int, float))
                and not isinstance(aversion, bool)
                and aversion > 0
            ):
                gamma = float(aversion)
        try:
            summary = _metrics.core_metrics(
                ledger,
                periods_per_year=periods_per_year,
                gamma=gamma,
                risk_free=0.0,
                basis="net",
            )
        except ValueError:
            # A zero-dispersion wealth path has an undefined Sharpe ratio;
            # core metrics fail closed there.  The honest summary of a flat
            # path is derived from the ledger's own final wealth with every
            # other moment zero — never a fabricated number.
            summary = _metrics.MetricsSummary(
                ledger.final_wealth / wealth - 1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        block = _results.MetricsBlock(
            summary,
            basis="net",
            periods_per_year=periods_per_year,
            gamma=gamma,
            risk_free=0.0,
        )
        path_result = _results.PathResult(path=ledger, metrics=(block,))
        # F12/F18: the effective settings report the values the run
        # actually executed — the annualization resolved at run entry,
        # the same gamma the metrics block carried — never a copied
        # declaration.  A declared/effective difference is therefore
        # always real and always disclosed by the results-computed verdict;
        # an uncovered one rejects the record (no result proceeds on an
        # unrecorded deviation), and none is ever self-sanctioned here.
        effective: dict[str, Any] = (
            {key: value for key, value in declared.items()}
            if isinstance(declared, Mapping)
            else {}
        )
        if "periods_per_year" in effective:
            effective["periods_per_year"] = periods_per_year
        conformance = _results.ConformanceRecord(
            declared=declared if isinstance(declared, Mapping) else {},
            effective=effective,
        )
        return _results.ResultRecord(
            manifest,
            strategy_id=strategy_id,
            paths={"main": path_result},
            forecasts=tuple(links),
            conformance=conformance,
            created_at=datetime.now(UTC),
        )
