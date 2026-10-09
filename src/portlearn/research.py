"""The public research-entry adapter: one declared run, provenance first.

Composes the public surfaces — the temporal protocol resolver and
the runtime ``Experiment`` runtime — into the single documented entry
point of the classical research route.
  The adapter owns exactly the orchestration the temporal-protocol acceptance
record carried forward — rejecting a missing or incomplete fit
provenance at run entry — and nothing else: no condition, no schema
field, no parallel provenance object, no strategy-specific logic, and
no network, filesystem, or clock read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import TypeVar

from portlearn.experiments import (
    DecisionSchedule,
    ExecutionError,
    Experiment,
    ExperimentConditions,
    ExperimentConfiguration,
    ExperimentResults,
    ExperimentSelection,
    StrategyRegistration,
)
from portlearn.forecasting import ForecastProvenance
from portlearn.protocols import (
    ExperimentTemporalDeclaration,
    ObservationRefitCadence,
    ProtocolProvenance,
    TemporalDesign,
    resolve_temporal_protocols,
)

__all__ = ["run_declared_experiment"]

_FitImage = TypeVar("_FitImage", bound=ForecastProvenance)


def run_declared_experiment(
    *,
    configuration: ExperimentConfiguration,
    registrations: Sequence[StrategyRegistration],
    schedule: DecisionSchedule,
    conditions: ExperimentConditions,
    temporal: TemporalDesign | ExperimentTemporalDeclaration,
    record_instants_by_strategy: Mapping[str, Sequence[datetime]],
    fit_provenances_by_strategy: Mapping[str, Sequence[_FitImage]],
    selection: ExperimentSelection | None = None,
) -> tuple[ExperimentResults, Mapping[str, ProtocolProvenance]]:
    """Resolve the declared temporal protocols, admit only complete
    per-strategy fit provenance, then delegate the run to the public
    ``Experiment`` unchanged.

    First, ``temporal`` and ``fit_provenances_by_strategy`` are
    required: explicit ``None`` (or omission) rejects ``TypeError``
    before any strategy decides; absence is never inferred and never
    defaulted. Second, resolution is exactly one
    ``resolve_temporal_protocols`` call over the declared strategy ids
    and the admitted record instants, with ``ProtocolStructureError``
    and ``ProtocolVocabularyError`` propagating unwrapped. Third,
    the fit-provenance mapping must cover the configured strategy
    names exactly, and every strategy's supplied image sequence must match
    the resolved canonical fit events one-for-one, in canonical event
    order, cutoff for cutoff; any missing, duplicate, extra, reordered,
    or cutoff-mismatched image rejects ``ExecutionError`` before any
    decision, while an explicitly empty sequence is complete exactly
    when the canonical sequence is empty. Fourth, execution then
    delegates to ``Experiment(..).run(selection)`` untouched. Finally,
    a post-run tripwire fails closed if an executed strategy is absent from
    the returned provenance mapping (unreachable by construction).
    """
    if temporal is None:
        raise TypeError(
            "run_declared_experiment requires a temporal declaration: "
            "temporal must be a TemporalDesign or "
            "ExperimentTemporalDeclaration, not None. Absence is never "
            "inferred and never defaulted (the adapter contract)."
        )
    if fit_provenances_by_strategy is None:
        raise TypeError(
            "run_declared_experiment requires per-strategy fit "
            "provenance: fit_provenances_by_strategy must be a mapping "
            "over the configured strategy names, not None. Absence is "
            "never inferred and never defaulted, and every no-fit strategy "
            "supplies an explicit empty sequence (the adapter contract)."
        )

    names = tuple(configuration.strategy_ids)

    resolved = resolve_temporal_protocols(temporal, names, record_instants_by_strategy)

    missing = [name for name in names if name not in fit_provenances_by_strategy]
    extra = [key for key in fit_provenances_by_strategy if key not in names]
    if missing or extra:
        raise ExecutionError(
            "fit_provenances_by_strategy must cover the configured "
            f"strategy names {list(names)} exactly: missing strategies "
            f"{missing}, unexpected strategies {extra}. Every strategy — including "
            "a no-fit strategy — supplies an explicit sequence, so omitted "
            "and empty are never conflated."
        )

    provenances: dict[str, ProtocolProvenance] = {}
    for name in names:
        events = resolved[name].fit_events
        if isinstance(events, ObservationRefitCadence):
            raise ExecutionError(
                f"strategy {name!r} resolved to an unmaterialized "
                "observation refit cadence: resolve_temporal_protocols "
                "materializes every cadence before returning, so this "
                "is an adapter defect — failing closed (adapter "
                "rule 3)."
            )
        canonical = tuple(events)
        supplied = tuple(fit_provenances_by_strategy[name])
        if len(supplied) != len(canonical):
            raise ExecutionError(
                f"strategy {name!r} supplies {len(supplied)} fit "
                f"provenance image(s) against {len(canonical)} resolved "
                "canonical fit event(s): the images must correspond "
                "one-for-one and in canonical event order. Zero, "
                "missing, duplicate, and extra images all reject "
                "before any decision."
            )
        for image, event in zip(supplied, canonical, strict=True):
            if image.cutoff != event.cutoff:
                raise ExecutionError(
                    f"strategy {name!r} supplies a fit provenance image "
                    f"with cutoff {image.cutoff.isoformat()} where the "
                    "canonical fit event at this position has cutoff "
                    f"{event.cutoff.isoformat()}: reordered, duplicated, "
                    "and cutoff-mismatched images all reject before any "
                    "decision."
                )
        provenances[name] = ProtocolProvenance(
            protocol=resolved[name],
            fit_provenances=supplied,
        )

    results = Experiment(configuration, registrations, schedule, conditions).run(
        selection
    )

    for record in results.records:
        if record.strategy_id not in provenances:
            raise ExecutionError(
                f"the executed strategy {record.strategy_id!r} is absent "
                "from the protocol provenance mapping: the adapter "
                "validates every strategy before the first decision, so "
                "this is an adapter defect — failing closed (adapter "
                "adapter rule 5)."
            )

    return results, MappingProxyType(provenances)
