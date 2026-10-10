"""Comparison outputs: structured comparison tables and plain-text render.

This module is the comparison surface: nine public value objects
(``ComparatorClassification``, ``SupportRelation``,
``MetricObservation``, ``PathIdentity``, ``ComparisonSpec``,
``ArmColumn``, ``ComparisonCell``, ``ComparisonRow``,
``ComparisonTable``), the module-owned ``ComparisonError``, and the one
pure renderer ``render_table``.

The module is a pure consumption layer over the public results/declaration/runtime
interfaces: it reads one ``ExperimentConfiguration`` plus one
``ExperimentResults`` collection and derives an immutable, canonically
ordered, byte-deterministic ``ComparisonTable`` together with a
plain-text rendering. It computes nothing: recorded metric values,
declarations, verdicts, deviations, relations, roles, and provenance
are copied verbatim or shown as explicit missing states -- never
ranked, bridged, converted, or summarized.

No inferential evidence -- statistics, estimands, uncertainty
representations, hypothesis tests, confidence intervals, or their
provenance -- is computed, accepted, normalized, serialized, or
rendered here. Every comparison-side rejection raises
``ComparisonError`` with a finance-language message naming the
violated rule, while the public pre-comparison ``ExecutionError`` from
``comparison_readiness`` propagates unwrapped.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import Any

from portlearn.experiments import (
    ExperimentConfiguration,
    ExperimentResults,
    comparison_readiness,
)
from portlearn.results import ParameterRelation, SanctionedDeviation

__all__ = [  # noqa: RUF022 -- alphabetical -- the pinned order, frozen at specification approval
    "ArmColumn",
    "ComparisonCell",
    "ComparisonError",
    "ComparisonRow",
    "ComparisonSpec",
    "ComparisonTable",
    "ComparatorClassification",
    "MetricObservation",
    "PathIdentity",
    "SupportRelation",
    "render_table",
]


class ComparisonError(ValueError):
    """The module-owned rejection path.

    Every comparison-side fail-closed rejection raises exactly this type with
    a finance-language message naming the violated rule. Public-module
    errors -- the runtime ``ExecutionError`` of the pre-comparison
    readiness chain among them -- propagate unchanged: never wrapped,
    never swallowed, never re-raised as this class.
    """

    __slots__ = ()


# The comparison vocabulary is exactly the metrics/results frozen six; the
# renderer owns no metric family (the metrics surface's successors own those).
_METRIC_NAMES = (
    "cumulative_return",
    "annualized_mean_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "certainty_equivalent_return",
)

# The declared comparator-classification vocabulary (D6) and the
# declared support-relation vocabulary: exact tokens, nothing else.
_CLASSIFICATION_KINDS = frozenset(
    {
        "full_universe_equal_weight",
        "same_support_equal_weight",
        "same_support_simplified_weighting",
        "direct_comparator",
    }
)
_SUPPORT_KINDS = frozenset(
    {"same_support_declared", "different_support_declared", "undeclared"}
)

# The exact status and ordering tokens of the structured surface.
_ROLE_TOKENS = frozenset({"target", "comparator", "ablation", "robustness_check"})
_CELL_STATUS_TOKENS = frozenset(
    {"ok", "missing_record", "missing_path", "missing_metric"}
)
_COMPARABILITY_TOKENS = frozenset(
    {
        "comparable",
        "declaration_mismatch",
        "multiple_declaration_sets",
        "missing_values",
    }
)

_SCHEMA_VERSION = "1.0.0"
_MISSING = object()


# --------------------------------------------------------------------------- #
# Canonical JSON (Law 8) and the shared admission validators                   #
# --------------------------------------------------------------------------- #


def _dumps(value: Any) -> str:
    """The canonical serialization rule: sorted keys, compact, ASCII, no nonfinite."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except ValueError as nonserializable:
        raise ComparisonError(
            "the comparison table is not representable in canonical JSON "
            f"(sorted, compact, ASCII, nonfinite-free): {nonserializable}. "
            "Every serialized value is finite."
        ) from None


def _require_nonblank_str(value: Any, field: str) -> str:
    """Admit a non-blank string or reject with ``ComparisonError``."""
    if not isinstance(value, str) or not value.strip():
        raise ComparisonError(
            f"{field} must be a non-blank string; got {value!r}. Blank or "
            "non-string values name nothing on the comparison surface, so "
            "the value is rejected unconditionally."
        )
    return value


def _require_finite_number(value: Any, field: str) -> float:
    """Admit a finite exact ``int``/``float`` (never ``bool``), zero-signed."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonError(
            f"{field} must be a finite int or float; got {value!r}. "
            "Booleans are not numbers on the comparison surface."
        )
    number = float(value)
    if not math.isfinite(number):
        raise ComparisonError(
            f"{field} must be finite; got {value!r}. A recorded value the "
            "table cannot serialize lawfully never enters it -- fail-closed "
            "in both directions."
        )
    return number + 0.0


def _require_exact_keys(node: Any, required: frozenset[str], what: str) -> None:
    """Exact key-set equality at one fixed-shape wire object."""
    if not isinstance(node, dict):
        raise ComparisonError(
            f"{what} must be a JSON object; got {type(node).__name__}. The "
            "comparison schema admits no silent forward-compatibility."
        )
    extra = sorted(set(node) - required)
    missing = sorted(required - set(node))
    if extra or missing:
        raise ComparisonError(
            f"{what} carries an invalid key set: unexpected keys {extra}, "
            f"missing keys {missing}. Exact key-set equality holds at every "
            "object on the comparison wire."
        )


def _require_wire_sequence(node: Any, what: str) -> list[Any]:
    """Admit a JSON array (never an object or scalar) as a sequence."""
    if not isinstance(node, list):
        raise ComparisonError(
            f"{what} must be a JSON array; got {type(node).__name__}."
        )
    return node


def _require_wire_order(keys: list[str], what: str) -> None:
    """Reject any sequence not in strict canonical-byte order."""
    for previous, current in itertools.pairwise(keys):
        if not previous < current:
            raise ComparisonError(
                f"{what} is not in canonical order (or carries a duplicate): "
                "the canonical form is total -- sorted by each element's own "
                "canonical JSON bytes -- and the parser accepts no other "
                "order and no repeated element."
            )


def _require_aware(value: Any, field: str) -> datetime:
    """Admit an aware datetime or reject with ``ComparisonError``."""
    if not isinstance(value, datetime):
        raise ComparisonError(
            f"{field} must be an aware datetime; got {type(value).__name__}."
        )
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ComparisonError(
            f"{field} must carry an explicit UTC offset; got {value!r}. "
            "Naive instants anchor no financial timeline."
        )
    return value


def _parse_instant(text: Any, field: str) -> datetime:
    """Parse the record wire convention: ISO-8601 text, aware instants."""
    if not isinstance(text, str):
        raise ComparisonError(
            f"{field} must be an ISO-8601 instant string; got {type(text).__name__}."
        )
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as unparsable:
        raise ComparisonError(
            f"{field} must be an ISO-8601 instant string; {text!r} does not "
            f"parse ({unparsable})."
        ) from None
    return _require_aware(moment, field)


# --------------------------------------------------------------------------- #
# Canonical JSON (Law 8) and the shared admission validators                   #
# --------------------------------------------------------------------------- #


def _canonical_json(value: Any) -> str:
    """The canonical wire encoding: ASCII, sorted keys, compact separators."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _wire_risk_free(risk_free: Any) -> str:
    """The canonical wire form of one risk_free declaration."""
    return _canonical_json(risk_free)


def _render_number(value: float) -> str:
    """Render one float in the house wire format (the canonical number grammar)."""
    if not math.isfinite(value):
        raise ComparisonError(
            "non-finite metric values cannot enter a comparison table "
            "(the wire encoding admits finite numbers only)"
        )
    return _canonical_json(value)


def _require_text(value: Any, what: str) -> str:
    """Admit one non-blank trimmed printable-ASCII identifier."""
    if not isinstance(value, str):
        raise ComparisonError(
            f"{what} must be text; got {type(value).__name__} (identifiers are text)"
        )
    if not value.isascii() or not value.isprintable():
        raise ComparisonError(f"{what} must be printable-ASCII text; got {value!r}")
    if not value.strip():
        raise ComparisonError(
            f"{what} is blank; a blank name is not a name "
            "(blank identifiers reject fail-closed)"
        )
    if value != value.strip():
        raise ComparisonError(
            f"{what} carries leading or trailing whitespace: {value!r} "
            "(identifiers are trimmed)"
        )
    return value


def _validate_classification_kind(kind: str) -> None:
    """Admit one comparator-classification kind against the supported set."""
    if kind not in _CLASSIFICATION_KINDS:
        raise ComparisonError(
            f"unknown comparator-classification kind {kind!r}; the public "
            "vocabulary is full_universe_equal_weight, "
            "same_support_equal_weight, "
            "same_support_simplified_weighting, direct_comparator "
            ""
        )


def _validate_support_kind(kind: str) -> None:
    """Admit one support-relation kind against the supported set."""
    if kind not in _SUPPORT_KINDS:
        raise ComparisonError(
            f"unknown support-relation kind {kind!r}; the supported values "
            "is same_support_declared, different_support_declared, "
            "undeclared"
        )


# --------------------------------------------------------------------------- #
# The frozen value objects (D6; Law 11)                                       #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ComparatorClassification:
    """One declared comparator classification — a claim, never a proof.

    ``kind`` is one of the four supported values tokens;
    ``method_note`` carries the researcher's own words when given and
    is absent (``None``) otherwise — blank or whitespace-only notes
    reject fail-closed.
    """

    kind: str
    method_note: str | None = None

    def __post_init__(self) -> None:
        _validate_classification_kind(self.kind)
        if self.method_note is not None:
            _require_text(self.method_note, "ComparatorClassification.method_note")


@dataclasses.dataclass(frozen=True)
class SupportRelation:
    """One declared support relation between comparator and target.

    ``kind`` is one of the three supported tokens; the explicit default
    for a comparator without a declared entry is ``undeclared``.
    ``evidence_reference`` is absent (``None``) or non-blank.
    """

    kind: str
    evidence_reference: str | None = None

    def __post_init__(self) -> None:
        _validate_support_kind(self.kind)
        if self.evidence_reference is not None:
            _require_text(self.evidence_reference, "SupportRelation.evidence_reference")


@dataclasses.dataclass(frozen=True)
class MetricObservation:
    """One recorded metric value with its own four declarations.

    Copied verbatim from a ``MetricsBlock``: no rounding, no
    re-annualization, no basis conversion, no recomputation. The
    ``risk_free`` declaration is stored exactly as recorded — a scalar
    stays a scalar and a sequence stays a sequence; the two forms are
    never conflated in either direction.
    """

    value: float
    basis: str
    periods_per_year: int
    gamma: float
    risk_free: Any


@dataclasses.dataclass(frozen=True)
class PathIdentity:
    """The recorded identity of one compared path.

    Carried verbatim from the record's named path: the ledger digest,
    the period count, the period bounds, and final wealth. Final
    wealth rides as path identity, never as a comparison metric.
    """

    ledger_digest: str
    n_periods: int
    period_start: datetime
    period_end: datetime
    final_wealth: float


# --------------------------------------------------------------------------- #
# ComparisonSpec — the consumed declaration of one comparison (Law 2)         #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ComparisonSpec:
    """What one comparison consumes: the public declaration, the
    uniform record collection, the declared metric question, the
    compared path name, and the comparator-only annotations.

    Construction is fail-closed consumption: the public
    ``comparison_readiness`` gate fires before any derivation, every
    record id must be role-assigned by the configuration, the target
    must carry a record, and the metric vocabulary is the supported six.
    """

    configuration: ExperimentConfiguration
    records: ExperimentResults
    metrics: Sequence[str]
    path_name: str = "main"
    classifications: Mapping[str, ComparatorClassification] = dataclasses.field(
        default_factory=dict
    )
    support_relations: Mapping[str, SupportRelation] = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(self.configuration, ExperimentConfiguration):
            raise ComparisonError(
                "ComparisonSpec.configuration must be the public "
                "ExperimentConfiguration; got "
                f"{type(self.configuration).__name__}."
            )
        if not isinstance(self.records, ExperimentResults):
            raise ComparisonError(
                "ComparisonSpec.records must be the public "
                "ExperimentResults collection; got "
                f"{type(self.records).__name__}."
            )
        # The frozen runtime pre-comparison gate fires first, for every
        # declaration, and its ExecutionError propagates unwrapped
        # (Law 10).
        comparison_readiness(self.configuration)
        metrics = self._require_metrics()
        path_name = _require_text(self.path_name, "ComparisonSpec.path_name")
        comparators = tuple(self.configuration.comparators)
        classifications = self._require_annotations(
            self.classifications,
            ComparatorClassification,
            "classifications",
            comparators,
        )
        support_relations = self._require_annotations(
            self.support_relations,
            SupportRelation,
            "support_relations",
            comparators,
        )
        arm_ids = self._arm_universe()
        self._require_lawful_records(arm_ids)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "path_name", path_name)
        object.__setattr__(self, "classifications", classifications)
        object.__setattr__(self, "support_relations", support_relations)

    def _require_metrics(self) -> tuple[str, ...]:
        raw = self.metrics
        if isinstance(raw, str) or not isinstance(raw, Sequence):
            raise ComparisonError(
                "ComparisonSpec.metrics must be a sequence of the six "
                "supported metric names."
            )
        names = tuple(raw)
        if not names:
            raise ComparisonError(
                "ComparisonSpec.metrics is empty: a comparison without a "
                "declared metric question is not a comparison."
            )
        for name in names:
            if name not in _METRIC_NAMES:
                raise ComparisonError(
                    f"the metric name {name!r} is outside the public "
                    "six-name vocabulary of metrics/results."
                )
        if len(set(names)) != len(names):
            raise ComparisonError(
                "ComparisonSpec.metrics declares one metric twice: each "
                "metric is compared exactly once."
            )
        return names

    def _arm_universe(self) -> dict[str, str]:
        """The strategy universe: exactly the role-assigned declared ids."""
        config = self.configuration
        roles: dict[str, str] = {config.target: "target"}
        roles.update({cid: "comparator" for cid in config.comparators})
        roles.update({aid: "ablation" for aid in config.ablations})
        roles.update({rid: "robustness_check" for rid in config.robustness_checks})
        return roles

    def _require_annotations(
        self,
        raw: Any,
        expected_type: Any,
        what: str,
        comparators: tuple[str, ...],
    ) -> MappingProxyType:
        if not isinstance(raw, Mapping):
            raise ComparisonError(
                f"ComparisonSpec.{what} must be a mapping keyed by "
                "declared comparator id."
            )
        frozen: dict[str, Any] = {}
        for key, value in raw.items():
            _require_text(key, f"ComparisonSpec.{what} key")
            if key not in comparators:
                raise ComparisonError(
                    f"ComparisonSpec.{what} keys on {key!r}, which the "
                    "configuration does not declare as a comparator: "
                    "classifications and support relations attach to "
                    "comparators only."
                )
            if not isinstance(value, expected_type):
                raise ComparisonError(
                    f"ComparisonSpec.{what}[{key!r}] must be a "
                    f"{expected_type.__name__}; got "
                    f"{type(value).__name__}."
                )
            frozen[key] = value
        return MappingProxyType(frozen)

    def _require_lawful_records(self, arm_ids: dict[str, str]) -> None:
        for record in self.records:
            record_id = getattr(record, "strategy_id", None)
            if record_id not in arm_ids:
                raise ComparisonError(
                    f"a record carries the strategy id {record_id!r}, "
                    "which the configuration assigns to no role: "
                    "role-less strategies never enter a comparison table — "
                    "neither admitted nor silently dropped."
                )
        target_id = self.configuration.target
        if target_id not in {
            getattr(record, "strategy_id", None) for record in self.records
        }:
            raise ComparisonError(
                f"the declared target {target_id!r} carries no record: a "
                "comparison without its target fails closed."
            )


# --------------------------------------------------------------------------- #
# The structured snapshot: arm columns, cells, rows (Laws 3-6)                 #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ArmColumn:
    """One strategy's column header: role, flags, and verbatim record facts.

    Every record-derived field is exactly ``None`` iff the strategy's
    record is missing (``record_status == "missing_record"``) — never
    a fabricated value, sentinel, or empty-set stand-in.
    """

    strategy_id: str
    role: str
    is_benchmark: bool
    classification: ComparatorClassification | None
    support: SupportRelation | None
    record_status: str
    conformance_verdict: str | None
    sanctioned_deviations: tuple[SanctionedDeviation, ...] | None
    parameter_relations: tuple[ParameterRelation, ...] | None
    run_id: str | None
    manifest_digest: str | None
    created_at: datetime | None
    path_identity: PathIdentity | None


@dataclasses.dataclass(frozen=True)
class ComparisonCell:
    """One strategy's cell for one requested metric.

    ``observations`` is non-empty iff ``status == "ok"``; multiple
    declaration sets are all carried in canonical order — none
    selected.
    """

    strategy_id: str
    status: str
    observations: tuple[MetricObservation, ...]


@dataclasses.dataclass(frozen=True)
class ComparisonRow:
    """One requested metric across every strategy column."""

    metric: str
    comparability: str
    cells: tuple[ComparisonCell, ...]


# --------------------------------------------------------------------------- #
# ComparisonTable — the immutable structured snapshot (Laws 3-9)              #
# --------------------------------------------------------------------------- #


def _observation_wire_key(observation: MetricObservation) -> tuple[Any, ...]:
    """The canonical observation key.

    ``risk_free`` enters as its canonical wire encoding, so a scalar is
    never structurally compared with a sequence and a scalar never
    conflates with its length-1 tuple (floor T11).
    """
    try:
        risk_free_wire = _wire_risk_free(observation.risk_free)
    except (TypeError, ValueError) as unmappable:
        raise ComparisonError(
            "a risk_free declaration has no canonical wire form; "
            f"got {observation.risk_free!r}: {unmappable} "
            "(declarations order and compare by "
            "their wire encoding)"
        ) from None
    return (
        observation.basis,
        observation.periods_per_year,
        observation.gamma,
        risk_free_wire,
    )


def _derive_cell(
    strategy_id: str,
    metric: str,
    record: Any,
    path_name: str,
) -> ComparisonCell:
    """Derive one strategy's cell: an explicit state or canonically
    ordered verbatim observations."""
    if record is None:
        return ComparisonCell(strategy_id, "missing_record", ())
    paths = getattr(record, "paths", None)
    if not isinstance(paths, Mapping) or path_name not in paths:
        return ComparisonCell(strategy_id, "missing_path", ())
    path = paths[path_name]
    observations: list[MetricObservation] = []
    for block in tuple(path.metrics):
        value = getattr(block, metric, _MISSING)
        if value is _MISSING:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ComparisonError(
                f"the metric {metric!r} of strategy {strategy_id!r} carries a "
                f"non-numeric recorded value {value!r}: recorded metric "
                "values are numbers."
            )
        if not math.isfinite(float(value)):
            raise ComparisonError(
                f"the metric {metric!r} of strategy {strategy_id!r} carries the "
                f"non-finite value {value!r}: non-finite values cannot "
                "enter a comparison table."
            )
        observations.append(
            MetricObservation(
                value=float(value),
                basis=block.basis,
                periods_per_year=block.periods_per_year,
                gamma=block.gamma,
                risk_free=block.risk_free,
            )
        )
    if not observations:
        return ComparisonCell(strategy_id, "missing_metric", ())
    observations.sort(key=_observation_wire_key)
    return ComparisonCell(strategy_id, "ok", tuple(observations))


def _derive_row(metric: str, cells: tuple[ComparisonCell, ...]) -> ComparisonRow:
    """Derive one row's comparability, exactly per the documented rule."""
    if any(cell.status != "ok" for cell in cells):
        comparability = "missing_values"
    elif any(len(cell.observations) != 1 for cell in cells):
        comparability = "multiple_declaration_sets"
    else:
        quadruples = {_observation_wire_key(cell.observations[0]) for cell in cells}
        comparability = "comparable" if len(quadruples) == 1 else "declaration_mismatch"
    return ComparisonRow(metric, comparability, cells)


@dataclasses.dataclass(frozen=True, init=False)
class ComparisonTable:
    """The immutable structured snapshot of one comparison.

    Built exclusively from a validated ``ComparisonSpec``; accepts
    nothing derived. Strategies follow canonical ``strategy_id`` ascending,
    rows follow the declared metric order, cells follow strategy order,
    and observations follow their canonical key. Deeply
    snapshotted from its consumed inputs and unhashable by design
    .
    """

    schema_version: str
    arms: tuple[ArmColumn, ...]
    rows: tuple[ComparisonRow, ...]
    conditions: Mapping[str, Any]

    __hash__ = None

    def __init__(self, spec: ComparisonSpec) -> None:
        config = spec.configuration
        records_by_id = {
            getattr(record, "strategy_id", None): record for record in spec.records
        }
        benchmarks = tuple(config.benchmarks)
        comparators = tuple(config.comparators)
        for benchmark_id in benchmarks:
            if benchmark_id not in comparators:
                raise ComparisonError(
                    f"the benchmark {benchmark_id!r} is not a declared "
                    "comparator: a benchmark names a comparator "
                    "."
                )
        roles = spec._arm_universe()
        arms: list[ArmColumn] = []
        for strategy_id in sorted(roles):
            role = roles[strategy_id]
            record = records_by_id.get(strategy_id)
            if record is None:
                arm = ArmColumn(
                    strategy_id=strategy_id,
                    role=role,
                    is_benchmark=strategy_id in benchmarks,
                    classification=(
                        spec.classifications.get(strategy_id)
                        if role == "comparator"
                        else None
                    ),
                    support=(
                        spec.support_relations.get(
                            strategy_id,
                            SupportRelation(kind="undeclared"),
                        )
                        if role == "comparator"
                        else None
                    ),
                    record_status="missing_record",
                    conformance_verdict=None,
                    sanctioned_deviations=None,
                    parameter_relations=None,
                    run_id=None,
                    manifest_digest=None,
                    created_at=None,
                    path_identity=None,
                )
            else:
                paths = getattr(record, "paths", None)
                path = paths.get(spec.path_name) if isinstance(paths, Mapping) else None
                path_identity = (
                    PathIdentity(
                        ledger_digest=path.ledger_digest,
                        n_periods=path.n_periods,
                        period_start=path.period_start,
                        period_end=path.period_end,
                        final_wealth=path.final_wealth,
                    )
                    if path is not None
                    else None
                )
                arm = ArmColumn(
                    strategy_id=strategy_id,
                    role=role,
                    is_benchmark=strategy_id in benchmarks,
                    classification=(
                        spec.classifications.get(strategy_id)
                        if role == "comparator"
                        else None
                    ),
                    support=(
                        spec.support_relations.get(
                            strategy_id,
                            SupportRelation(kind="undeclared"),
                        )
                        if role == "comparator"
                        else None
                    ),
                    record_status="present",
                    conformance_verdict=record.conformance.verdict,
                    sanctioned_deviations=tuple(record.conformance.deviations),
                    parameter_relations=tuple(record.parameter_relations),
                    run_id=record.run_id,
                    manifest_digest=record.manifest_digest,
                    created_at=record.created_at,
                    path_identity=path_identity,
                )
            arms.append(arm)
        arm_ids = tuple(arm.strategy_id for arm in arms)
        rows: list[ComparisonRow] = []
        for metric in spec.metrics:
            cells = tuple(
                _derive_cell(
                    strategy_id,
                    metric,
                    records_by_id.get(strategy_id),
                    spec.path_name,
                )
                for strategy_id in arm_ids
            )
            rows.append(_derive_row(metric, cells))
        conditions = MappingProxyType(dict(config.conditions))
        object.__setattr__(self, "schema_version", _SCHEMA_VERSION)
        object.__setattr__(self, "arms", tuple(arms))
        object.__setattr__(self, "rows", tuple(rows))
        object.__setattr__(self, "conditions", conditions)


# --------------------------------------------------------------------------- #
# The Law 8 wire: exact key sets and the canonical wire projection            #
# --------------------------------------------------------------------------- #

_TABLE_KEYS = frozenset({"schema_version", "arms", "rows", "conditions"})
_ARM_KEYS = frozenset(
    {
        "strategy_id",
        "role",
        "is_benchmark",
        "classification",
        "support",
        "record_status",
        "conformance_verdict",
        "sanctioned_deviations",
        "parameter_relations",
        "run_id",
        "manifest_digest",
        "created_at",
        "path_identity",
    }
)
_CLASSIFICATION_KEYS = frozenset({"kind", "method_note"})
_SUPPORT_KEYS = frozenset({"kind", "evidence_reference"})
_DEVIATION_KEYS = frozenset(
    {"field", "authority", "reference", "reason", "approved_at"}
)
_RELATION_KEYS = frozenset({"role_a", "role_b", "value_a", "value_b", "relation"})
_PATH_KEYS = frozenset(
    {
        "ledger_digest",
        "n_periods",
        "period_start",
        "period_end",
        "final_wealth",
    }
)
_ROW_KEYS = frozenset({"metric", "comparability", "cells"})
_CELL_KEYS = frozenset({"strategy_id", "status", "observations"})
_OBSERVATION_KEYS = frozenset(
    {"value", "basis", "periods_per_year", "gamma", "risk_free"}
)
_RECORD_STATUS_TOKENS = frozenset({"present", "missing_record"})


def _classification_wire(value: ComparatorClassification) -> dict[str, Any]:
    return {"kind": value.kind, "method_note": value.method_note}


def _support_wire(value: SupportRelation) -> dict[str, Any]:
    return {"kind": value.kind, "evidence_reference": value.evidence_reference}


def _deviation_wire(value: SanctionedDeviation) -> dict[str, Any]:
    return {
        "field": value.field,
        "authority": value.authority,
        "reference": value.reference,
        "reason": value.reason,
        "approved_at": value.approved_at.isoformat(),
    }


def _relation_wire(value: ParameterRelation) -> dict[str, Any]:
    return {
        "role_a": value.role_a,
        "role_b": value.role_b,
        "value_a": value.value_a,
        "value_b": value.value_b,
        "relation": value.relation,
    }


def _identity_wire(value: PathIdentity) -> dict[str, Any]:
    return {
        "ledger_digest": value.ledger_digest,
        "n_periods": value.n_periods,
        "period_start": value.period_start.isoformat(),
        "period_end": value.period_end.isoformat(),
        "final_wealth": value.final_wealth,
    }


def _arm_wire(arm: ArmColumn) -> dict[str, Any]:
    return {
        "strategy_id": arm.strategy_id,
        "role": arm.role,
        "is_benchmark": arm.is_benchmark,
        "classification": (
            None
            if arm.classification is None
            else _classification_wire(arm.classification)
        ),
        "support": None if arm.support is None else _support_wire(arm.support),
        "record_status": arm.record_status,
        "conformance_verdict": arm.conformance_verdict,
        "sanctioned_deviations": (
            None
            if arm.sanctioned_deviations is None
            else [_deviation_wire(item) for item in arm.sanctioned_deviations]
        ),
        "parameter_relations": (
            None
            if arm.parameter_relations is None
            else [_relation_wire(item) for item in arm.parameter_relations]
        ),
        "run_id": arm.run_id,
        "manifest_digest": arm.manifest_digest,
        "created_at": (None if arm.created_at is None else arm.created_at.isoformat()),
        "path_identity": (
            None if arm.path_identity is None else _identity_wire(arm.path_identity)
        ),
    }


def _observation_wire(observation: MetricObservation) -> dict[str, Any]:
    return {
        "value": observation.value,
        "basis": observation.basis,
        "periods_per_year": observation.periods_per_year,
        "gamma": observation.gamma,
        "risk_free": _conditions_wire(observation.risk_free),
    }


def _cell_wire(cell: ComparisonCell) -> dict[str, Any]:
    return {
        "strategy_id": cell.strategy_id,
        "status": cell.status,
        "observations": [
            _observation_wire(observation) for observation in cell.observations
        ],
    }


def _row_wire(row: ComparisonRow) -> dict[str, Any]:
    return {
        "metric": row.metric,
        "comparability": row.comparability,
        "cells": [_cell_wire(cell) for cell in row.cells],
    }


def _table_wire(table: ComparisonTable) -> str:
    return _canonical_json(
        {
            "schema_version": table.schema_version,
            "arms": [_arm_wire(arm) for arm in table.arms],
            "rows": [_row_wire(row) for row in table.rows],
            "conditions": _conditions_wire(table.conditions),
        }
    )


# --------------------------------------------------------------------------- #
# The Law 8 wire: local projections (mirror the frozen house grammar)        #
# --------------------------------------------------------------------------- #


def _conditions_wire(value: Any) -> Any:
    """Project one immutable conditions/risk_free snapshot to wire form."""
    if isinstance(value, Mapping):
        return {key: _conditions_wire(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_conditions_wire(item) for item in value]
    if isinstance(value, float):
        return value + 0.0
    return value


def _refreeze_conditions(node: Any, what: str) -> Any:
    """Refreeze one parsed JSON tree under the house conditions grammar."""
    if node is None or isinstance(node, (bool, int)):
        return node
    if isinstance(node, float):
        if not math.isfinite(node):
            raise ComparisonError(
                f"{what} carries the non-finite number {node!r}: the wire "
                "admits finite numbers only."
            )
        return node + 0.0
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return tuple(_refreeze_conditions(item, what) for item in node)
    if isinstance(node, dict):
        frozen: dict[str, Any] = {}
        for key, item in node.items():
            if not isinstance(key, str) or not key.strip():
                raise ComparisonError(
                    f"{what} carries an invalid mapping key {key!r}: "
                    "keys must be non-blank strings."
                )
            frozen[key] = _refreeze_conditions(item, f"{what}[{key!r}]")
        return MappingProxyType(frozen)
    raise ComparisonError(
        f"{what} carries an invalid wire value of type {type(node).__name__}."
    )


def _reject_constant(token: str) -> Any:
    raise ComparisonError(
        f"the wire admits finite numbers only; got the non-finite literal {token}."
    )


def _reject_duplicate_keys(pairs: Any) -> dict[str, Any]:
    seen: set[str] = set()
    frozen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ComparisonError(
                f"the wire carries the object key {key!r} twice: "
                "duplicate keys reject at every depth."
            )
        seen.add(key)
        frozen[key] = value
    return frozen


def _parse_tree(text: Any, what: str) -> Any:
    if not isinstance(text, str):
        raise ComparisonError(f"{what} consumes JSON text; got {type(text).__name__}.")
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as unparsable:
        raise ComparisonError(
            f"{what} requires well-formed JSON text; got "
            f"{unparsable.msg} at offset {unparsable.pos} "
            "."
        ) from None


def _require_exact_keys(node: Any, keys: frozenset[str], what: str) -> None:
    if not isinstance(node, dict):
        raise ComparisonError(
            f"{what} must be a JSON object; got {type(node).__name__}."
        )
    got = frozenset(node)
    unknown = sorted(got - keys)
    if unknown:
        raise ComparisonError(
            f"{what} carries the unknown key(s) {unknown}: the schema is "
            "exact — unknown keys reject at every depth."
        )
    missing = sorted(keys - got)
    if missing:
        raise ComparisonError(f"{what} is missing the required key(s) {missing}.")


def _require_wire_list(node: Any, what: str) -> list[Any]:
    if not isinstance(node, list):
        raise ComparisonError(
            f"{what} must be a JSON array; got {type(node).__name__}."
        )
    return node


def _require_wire_str(node: Any, what: str) -> str:
    if not isinstance(node, str) or not node.strip():
        raise ComparisonError(f"{what} must be a non-blank string; got {node!r}.")
    return node


def _require_wire_finite(node: Any, what: str) -> float:
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise ComparisonError(f"{what} must be a number; got {node!r}.")
    if not math.isfinite(float(node)):
        raise ComparisonError(f"{what} must be finite; got {node!r}.")
    return float(node)


def _require_wire_int(node: Any, what: str) -> int:
    if isinstance(node, bool) or not isinstance(node, int):
        raise ComparisonError(f"{what} must be an int; got {node!r}.")
    return node


def _parse_wire_instant(node: Any, what: str) -> datetime:
    text = _require_wire_str(node, what)
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        raise ComparisonError(
            f"{what} must be an ISO-8601 instant; got {text!r}."
        ) from None
    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        raise ComparisonError(
            f"{what} must be timezone-aware; got {text!r}: naive instants "
            "reject fail-closed."
        )
    return moment


# --------------------------------------------------------------------------- #
# ComparisonTable — wire in: value-object parsers (Law 8)                     #
# --------------------------------------------------------------------------- #


def _wire_in_classification(node: Any) -> ComparatorClassification:
    _require_exact_keys(node, _CLASSIFICATION_KEYS, "a classification")
    return ComparatorClassification(
        kind=_require_wire_str(node["kind"], "classification kind"),
        method_note=(
            None
            if node["method_note"] is None
            else _require_wire_str(node["method_note"], "method_note")
        ),
    )


def _wire_in_support(node: Any) -> SupportRelation:
    _require_exact_keys(node, _SUPPORT_KEYS, "a support relation")
    return SupportRelation(
        kind=_require_wire_str(node["kind"], "support kind"),
        evidence_reference=(
            None
            if node["evidence_reference"] is None
            else _require_wire_str(node["evidence_reference"], "evidence_reference")
        ),
    )


def _wire_in_deviation(node: Any) -> SanctionedDeviation:
    _require_exact_keys(node, _DEVIATION_KEYS, "a sanctioned deviation")
    return SanctionedDeviation(
        field=_require_wire_str(node["field"], "deviation field"),
        authority=_require_wire_str(node["authority"], "deviation authority"),
        reference=_require_wire_str(node["reference"], "deviation reference"),
        reason=_require_wire_str(node["reason"], "deviation reason"),
        approved_at=_parse_wire_instant(node["approved_at"], "deviation approved_at"),
    )


def _wire_in_relation(node: Any) -> ParameterRelation:
    _require_exact_keys(node, _RELATION_KEYS, "a parameter relation")
    built = ParameterRelation(
        role_a=_require_wire_str(node["role_a"], "relation role_a"),
        role_b=_require_wire_str(node["role_b"], "relation role_b"),
        value_a=_require_wire_finite(node["value_a"], "relation value_a"),
        value_b=_require_wire_finite(node["value_b"], "relation value_b"),
    )
    stored = _require_wire_str(node["relation"], "relation token")
    if stored != built.relation:
        raise ComparisonError(
            f"a parameter relation stores the token {stored!r} while its "
            f"own values recompute {built.relation!r}: the token is "
            "computed, never trusted."
        )
    return built


# --------------------------------------------------------------------------- #
# ComparisonTable — wire in: the snapshot parsers (Law 8)                     #
# --------------------------------------------------------------------------- #


def _wire_in_identity(node: Any) -> PathIdentity:
    """Parse one path identity object from the wire."""
    _require_exact_keys(node, _PATH_KEYS, "a path identity")
    return PathIdentity(
        ledger_digest=_require_wire_str(node["ledger_digest"], "a ledger_digest"),
        n_periods=_require_wire_int(node["n_periods"], "an n_periods"),
        period_start=_parse_wire_instant(node["period_start"], "a period_start"),
        period_end=_parse_wire_instant(node["period_end"], "a period_end"),
        final_wealth=_require_wire_finite(node["final_wealth"], "a final_wealth"),
    )


def _wire_in_observation(node: Any) -> MetricObservation:
    """Parse one metric observation object from the wire.

    ``risk_free`` refreezes under the house conditions grammar, so a
    wire list returns as a tuple and a scalar stays a scalar — the
    two forms never conflate in either direction.
    """
    _require_exact_keys(node, _OBSERVATION_KEYS, "a metric observation")
    return MetricObservation(
        value=_require_wire_finite(node["value"], "an observation value"),
        basis=_require_wire_str(node["basis"], "an observation basis"),
        periods_per_year=_require_wire_int(
            node["periods_per_year"], "an observation periods_per_year"
        ),
        gamma=_require_wire_finite(node["gamma"], "an observation gamma"),
        risk_free=_refreeze_conditions(node["risk_free"], "an observation risk_free"),
    )


def _wire_in_cell(node: Any) -> ComparisonCell:
    """Parse one comparison cell from the wire.

    The status token must belong to the supported values, the
    observations/status coupling of must hold, and the
    observations must already sit in their canonical key order
     — the wire is consumed exactly as it is emitted.
    """
    _require_exact_keys(node, _CELL_KEYS, "a comparison cell")
    strategy_id = _require_wire_str(node["strategy_id"], "a cell strategy_id")
    status = _require_wire_str(node["status"], "a cell status")
    if status not in _CELL_STATUS_TOKENS:
        raise ComparisonError(
            f"a comparison cell carries the unknown status {status!r}: "
            "cell statuses come from the supported values "
            "."
        )
    observations = tuple(
        _wire_in_observation(item)
        for item in _require_wire_list(node["observations"], "cell observations")
    )
    if status == "ok" and not observations:
        raise ComparisonError(
            f"the cell of {strategy_id!r} claims the status ok while "
            "carrying no observations: an ok cell is non-empty "
            "."
        )
    if status != "ok" and observations:
        raise ComparisonError(
            f"the cell of {strategy_id!r} carries the status {status!r} "
            "with observations: an explicit missing state carries none "
            "."
        )
    keys = [_observation_wire_key(item) for item in observations]
    if keys != sorted(keys):
        raise ComparisonError(
            f"the cell of {strategy_id!r} carries observations outside "
            "their canonical key order: the wire order is the canonical "
            "order."
        )
    return ComparisonCell(strategy_id, status, observations)


def _wire_in_row(node: Any) -> ComparisonRow:
    """Parse one comparison row from the wire.

    The comparability token is re-derived from the parsed cells and
    compared against the stored token — computed tokens are
    re-derived, never trusted.
    """
    _require_exact_keys(node, _ROW_KEYS, "a comparison row")
    metric = _require_wire_str(node["metric"], "a row metric")
    comparability = _require_wire_str(node["comparability"], "a row comparability")
    if comparability not in _COMPARABILITY_TOKENS:
        raise ComparisonError(
            f"the row of {metric!r} carries the unknown comparability "
            f"{comparability!r}: comparability tokens come from the "
            "supported values."
        )
    cells = tuple(
        _wire_in_cell(item) for item in _require_wire_list(node["cells"], "row cells")
    )
    derived = _derive_row(metric, cells)
    if derived.comparability != comparability:
        raise ComparisonError(
            f"the row of {metric!r} stores the comparability "
            f"{comparability!r} while its own cells recompute "
            f"{derived.comparability!r}: the token is computed, never "
            "trusted."
        )
    return ComparisonRow(metric, comparability, cells)


def _wire_in_arm(node: Any) -> ArmColumn:
    """Parse one strategy column from the wire.

    Every token guard is fail-closed, and the record-derived fields
    carry exactly the None/value coupling of : a
    ``missing_record`` strategy carries ``None`` in every record-derived
    field while keeping its declared comparator metadata, a present
    strategy carries its verbatim record facts.
    """
    _require_exact_keys(node, _ARM_KEYS, "a comparison strategy")
    strategy_id = _require_wire_str(node["strategy_id"], "a strategy strategy_id")
    role = _require_wire_str(node["role"], "a strategy role")
    if role not in _ROLE_TOKENS:
        raise ComparisonError(
            f"the strategy {strategy_id!r} carries the unknown role "
            f"{role!r}: roles come from the supported values "
            "."
        )
    if not isinstance(node["is_benchmark"], bool):
        raise ComparisonError(
            f"the strategy {strategy_id!r} carries the non-boolean "
            f"is_benchmark {node['is_benchmark']!r}."
        )
    is_benchmark = node["is_benchmark"]
    if is_benchmark and role != "comparator":
        raise ComparisonError(
            f"the strategy {strategy_id!r} is a benchmark while carrying the "
            f"role {role!r}: a benchmark names a comparator "
            "."
        )
    record_status = _require_wire_str(node["record_status"], "a strategy record_status")
    if record_status not in _RECORD_STATUS_TOKENS:
        raise ComparisonError(
            f"the strategy {strategy_id!r} carries the unknown record_status "
            f"{record_status!r}: record statuses come from the public "
            "vocabulary."
        )
    classification = (
        None
        if node["classification"] is None
        else _wire_in_classification(node["classification"])
    )
    support = None if node["support"] is None else _wire_in_support(node["support"])
    if role != "comparator":
        if classification is not None or support is not None:
            raise ComparisonError(
                f"the {role!r} strategy {strategy_id!r} carries a classification "
                "or a support relation: only a comparator strategy carries them "
                "."
            )
    elif support is None:
        raise ComparisonError(
            f"the comparator strategy {strategy_id!r} carries no support "
            "relation: a comparator carries at least the undeclared "
            "default, whatever its record status."
        )
    conformance_verdict = (
        None
        if node["conformance_verdict"] is None
        else _require_wire_str(node["conformance_verdict"], "a conformance_verdict")
    )
    sanctioned_deviations = (
        None
        if node["sanctioned_deviations"] is None
        else tuple(
            _wire_in_deviation(item)
            for item in _require_wire_list(
                node["sanctioned_deviations"], "sanctioned deviations"
            )
        )
    )
    parameter_relations = (
        None
        if node["parameter_relations"] is None
        else tuple(
            _wire_in_relation(item)
            for item in _require_wire_list(
                node["parameter_relations"], "parameter relations"
            )
        )
    )
    run_id = (
        None
        if node["run_id"] is None
        else _require_wire_str(node["run_id"], "a strategy run_id")
    )
    manifest_digest = (
        None
        if node["manifest_digest"] is None
        else _require_wire_str(node["manifest_digest"], "a strategy manifest_digest")
    )
    created_at = (
        None
        if node["created_at"] is None
        else _parse_wire_instant(node["created_at"], "a strategy created_at")
    )
    path_identity = (
        None
        if node["path_identity"] is None
        else _wire_in_identity(node["path_identity"])
    )
    if record_status == "missing_record":
        carried = (
            conformance_verdict,
            sanctioned_deviations,
            parameter_relations,
            run_id,
            manifest_digest,
            created_at,
            path_identity,
        )
        if any(value is not None for value in carried):
            raise ComparisonError(
                f"the missing_record strategy {strategy_id!r} carries a "
                "record-derived value: every record-derived field is "
                "exactly None on a missing record."
            )
    else:
        required = (
            conformance_verdict,
            sanctioned_deviations,
            parameter_relations,
            run_id,
            manifest_digest,
            created_at,
        )
        if any(value is None for value in required):
            raise ComparisonError(
                f"the present strategy {strategy_id!r} misses a "
                "record-derived value: a present record carries its "
                "verdict, deviations, relations, run identity, and "
                "instant."
            )
    return ArmColumn(
        strategy_id=strategy_id,
        role=role,
        is_benchmark=is_benchmark,
        classification=classification,
        support=support,
        record_status=record_status,
        conformance_verdict=conformance_verdict,
        sanctioned_deviations=sanctioned_deviations,
        parameter_relations=parameter_relations,
        run_id=run_id,
        manifest_digest=manifest_digest,
        created_at=created_at,
        path_identity=path_identity,
    )


# --------------------------------------------------------------------------- #
# ComparisonTable — wire in: the strict reconstruction entrypoint (Law 8)     #
# --------------------------------------------------------------------------- #


def _wire_in_table(text: Any) -> ComparisonTable:
    """Rebuild one table from its canonical wire text.

    Exact-schema at every depth, pinned to the public schema
    version, canonical in strategy and row order, and immutable on
    arrival: the rebuilt table is the table that was emitted.
    """
    node = _parse_tree(text, "ComparisonTable.from_json")
    _require_exact_keys(node, _TABLE_KEYS, "a comparison table")
    version = _require_wire_str(node["schema_version"], "a schema_version")
    if version != _SCHEMA_VERSION:
        raise ComparisonError(
            f"the wire schema_version is {version!r} while this module "
            f"freezes {_SCHEMA_VERSION!r}: the version is pinned, never "
            "negotiated."
        )
    arms = tuple(
        _wire_in_arm(item)
        for item in _require_wire_list(node["arms"], "the table strategies")
    )
    rows = tuple(
        _wire_in_row(item)
        for item in _require_wire_list(node["rows"], "the table rows")
    )
    if not arms:
        raise ComparisonError(
            "the wire carries no strategies: a table carries at least the "
            "target strategy."
        )
    if not rows:
        raise ComparisonError(
            "the wire carries no rows: a table carries at least one requested metric."
        )
    arm_ids = [arm.strategy_id for arm in arms]
    if len(set(arm_ids)) != len(arm_ids):
        raise ComparisonError(
            f"the wire carries a strategy twice: {arm_ids!r} — a strategy "
            "appears exactly once."
        )
    if arm_ids != sorted(arm_ids):
        raise ComparisonError(
            f"the wire strategies are not in canonical ascending order: {arm_ids!r}."
        )
    metrics = [row.metric for row in rows]
    if len(set(metrics)) != len(metrics):
        raise ComparisonError(
            f"the wire carries a metric twice: {metrics!r} — rows "
            "follow the declared metric order, each metric once "
            "."
        )
    ordered_ids = tuple(arm_ids)
    for row in rows:
        if tuple(cell.strategy_id for cell in row.cells) != ordered_ids:
            raise ComparisonError(
                f"the row {row.metric!r} does not carry one cell per strategy "
                "in the strategy order: cells follow the strategies exactly "
                "."
            )
    if not isinstance(node["conditions"], dict):
        raise ComparisonError(
            "the shared conditions must be a JSON object; got "
            f"{type(node['conditions']).__name__}."
        )
    conditions = _refreeze_conditions(node["conditions"], "the shared conditions")
    table = object.__new__(ComparisonTable)
    object.__setattr__(table, "schema_version", _SCHEMA_VERSION)
    object.__setattr__(table, "arms", arms)
    object.__setattr__(table, "rows", rows)
    object.__setattr__(table, "conditions", conditions)
    return table


def _table_to_json(self: ComparisonTable) -> str:
    """Emit the canonical wire text of this table."""
    return _table_wire(self)


def _table_from_json(cls: type, text: Any) -> ComparisonTable:
    """Rebuild a table from its canonical wire text."""
    return _wire_in_table(text)


# The class body closed in an earlier part, so the wire methods attach
# here after the fact — the only sanctioned way to add them without
# touching the frozen class definition or widening the public surface.
ComparisonTable.to_json = _table_to_json
ComparisonTable.from_json = classmethod(_table_from_json)
# --------------------------------------------------------------------------- #
# render_table — the deterministic plain-text rendering                       #
# --------------------------------------------------------------------------- #


def render_table(table: ComparisonTable) -> str:
    """Render one comparison table as deterministic plain text.

    Strictly descriptive: every line restates a carried field — the
    shared conditions in their wire form, each strategy's header
    facts verbatim, and each requested metric's carried observations
    in the canonical order. The render computes nothing between strategies:
    it emits no ordering by recorded value, no selection among
    declaration sets, and no derived contrast; a missing strategy or cell
    states its explicit token, never a blank or a fabricated value.
    """
    lines: list[str] = []
    lines.append("portlearn comparison table")
    lines.append(f"schema_version {table.schema_version}")
    lines.append("conditions " + _canonical_json(_conditions_wire(table.conditions)))
    for arm in table.arms:
        lines.append("")
        lines.append(f"arm {arm.strategy_id}")
        lines.append(f"role {arm.role}")
        lines.append(f"is_benchmark {arm.is_benchmark}")
        if arm.role == "comparator":
            if arm.classification is None:
                lines.append("classification unclassified")
            else:
                lines.append(f"classification {arm.classification.kind}")
                if arm.classification.method_note is not None:
                    lines.append("method_note " + arm.classification.method_note)
            if arm.support is None:
                lines.append("support not_applicable")
            else:
                lines.append(f"support {arm.support.kind}")
                if arm.support.evidence_reference is not None:
                    lines.append("evidence_reference " + arm.support.evidence_reference)
        else:
            lines.append("classification not_applicable")
            lines.append("support not_applicable")
        lines.append(f"record_status {arm.record_status}")
        if arm.record_status == "missing_record":
            for label in (
                "conformance_verdict",
                "sanctioned_deviations",
                "parameter_relations",
                "run_id",
                "manifest_digest",
                "created_at",
                "path_identity",
            ):
                lines.append(f"{label} not_available")
            continue
        lines.append(f"conformance_verdict {arm.conformance_verdict}")
        if arm.sanctioned_deviations:
            for deviation in arm.sanctioned_deviations:
                lines.append(
                    "sanctioned_deviation field="
                    + deviation.field
                    + " authority="
                    + deviation.authority
                    + " reference="
                    + deviation.reference
                    + " reason="
                    + deviation.reason
                    + " approved_at="
                    + deviation.approved_at.isoformat()
                )
        else:
            lines.append("sanctioned_deviations none")
        if arm.parameter_relations:
            for relation in arm.parameter_relations:
                lines.append(
                    "parameter_relation "
                    + relation.role_a
                    + " "
                    + relation.role_b
                    + " value_a="
                    + _render_number(relation.value_a)
                    + " value_b="
                    + _render_number(relation.value_b)
                    + " relation="
                    + relation.relation
                )
        else:
            lines.append("parameter_relations none")
        lines.append(f"run_id {arm.run_id}")
        lines.append(f"manifest_digest {arm.manifest_digest}")
        lines.append(f"created_at {arm.created_at.isoformat()}")
        if arm.path_identity is None:
            lines.append("path_identity not_available")
        else:
            identity = arm.path_identity
            lines.append("path_identity ledger_digest=" + identity.ledger_digest)
            lines.append(f"path_identity n_periods={identity.n_periods}")
            lines.append(
                "path_identity period_start=" + identity.period_start.isoformat()
            )
            lines.append("path_identity period_end=" + identity.period_end.isoformat())
            lines.append(
                "path_identity final_wealth=" + _render_number(identity.final_wealth)
            )
    for row in table.rows:
        lines.append("")
        lines.append(f"row {row.metric} comparability={row.comparability}")
        for cell in row.cells:
            if cell.status != "ok":
                lines.append(f"cell {cell.strategy_id} status={cell.status}")
                continue
            for observation in cell.observations:
                lines.append(
                    "cell "
                    + cell.strategy_id
                    + " status=ok value="
                    + _render_number(observation.value)
                    + " basis="
                    + observation.basis
                    + " periods_per_year="
                    + str(observation.periods_per_year)
                    + " gamma="
                    + _render_number(observation.gamma)
                    + " risk_free="
                    + _canonical_json(_conditions_wire(observation.risk_free))
                )
    return "\n".join(lines) + "\n"
