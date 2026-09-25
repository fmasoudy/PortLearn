"""Synthetic foundation-contract wiring example for PortLearn.

Foundation contracts and composition only: PortLearn cannot yet run a real
portfolio experiment, so nothing here is a performance or capability claim.
Every input is a fixed invented constant set in a clearly historical 1999
era; no market or macro data is read, and nothing external is touched.
"""

from __future__ import annotations

import platform
from datetime import UTC, datetime

import portlearn
from portlearn.interfaces import InformationSet, require_feature_lineage
from portlearn.leakage import LeakageCase, run_leakage_cases
from portlearn.manifest import RunManifest
from portlearn.observations import (
    FeatureLineageError,
    TimedObservation,
    require_lineage_monotone,
    vintage_as_of,
)
from portlearn.timing import (
    DecisionTiming,
    FutureInformationError,
    ReturnRealizationPeriod,
    is_available_for_decision,
)

#: Boundary statement kept on one source line so the fence wording is
#: stated verbatim in the example itself.
FENCE = "This example demonstrates foundation contracts and composition, not an implemented end-to-end portfolio research workflow."

#: Synthetic 1999-era instants (UTC) invented for this demonstration.
OBSERVED_AT = datetime(1999, 1, 5, 14, 30, tzinfo=UTC)
DECISION_AT = datetime(1999, 1, 5, 15, 0, tzinfo=UTC)
EXECUTION_AT = datetime(1999, 1, 5, 15, 5, tzinfo=UTC)
REVISED_AT = datetime(1999, 1, 6, 9, 0, tzinfo=UTC)
LATER_AT = datetime(1999, 1, 7, 0, 0, tzinfo=UTC)
REALIZED_FROM = datetime(1999, 2, 1, 0, 0, tzinfo=UTC)
REALIZED_TO = datetime(1999, 3, 1, 0, 0, tzinfo=UTC)


def main() -> None:
    """Execute the synthetic foundation-contract wiring demonstration."""
    print(FENCE)
    print("Every input is synthetic; no real data is used.\n")

    # [1] Chronology: one decision instant, its execution instant, and the
    # half-open return realization window that follows execution.
    timing = DecisionTiming(
        decision_time=DECISION_AT,
        execution_time=EXECUTION_AT,
        return_realization_period=ReturnRealizationPeriod(
            start_time=REALIZED_FROM, end_time=REALIZED_TO
        ),
    )
    print(f"[1] timing: decide {timing.decision_time.isoformat()}, execute {timing.execution_time.isoformat()}")

    # [2] Point-in-time vintages: two synthetic prints of one observation;
    # the visible vintage depends on the instant it is queried at.
    first_print = TimedObservation(
        series_id="synthetic.series.example",
        observation_time=OBSERVED_AT,
        available_time=OBSERVED_AT,
        value=0.0124,
    )
    revision = TimedObservation(
        series_id="synthetic.series.example",
        observation_time=OBSERVED_AT,
        available_time=REVISED_AT,
        value=0.0131,
    )
    at_decision = vintage_as_of([first_print, revision], DECISION_AT)
    later = vintage_as_of([first_print, revision], LATER_AT)
    print(f"[2] vintages: at decision value={at_decision.value}, later value={later.value}")

    # [3] Admission: only records already available at the decision instant
    # may enter its information set.
    visible = is_available_for_decision(first_print, DECISION_AT)
    visible_later = is_available_for_decision(revision, DECISION_AT)
    information_set = InformationSet(items=[first_print], as_of=DECISION_AT)
    members = ", ".join(r.series_id for r in information_set)
    print(f"[3] admission: first print visible {visible}, revision visible {visible_later}; "
          f"admitted as of {information_set.as_of.isoformat()}: {members}")

    # [4] Derivation lineage: a derived feature may not be declared
    # available before its input.
    derived_feature = TimedObservation(
        series_id="synthetic.feature.example",
        observation_time=OBSERVED_AT,
        available_time=OBSERVED_AT,
        value=0.40,
    )
    require_feature_lineage([first_print], [derived_feature])
    print(f"[4] lineage: {derived_feature.series_id} accepted (not available before its input)")

    # [5] Information-leakage battery: each synthetic leak attempt must be
    # blocked by its fixed contract error.
    report = run_leakage_cases([
        LeakageCase(
            name="admit an observation not yet available at the decision",
            attempt=lambda: InformationSet(items=[revision], as_of=DECISION_AT),
            expected_error=FutureInformationError,
        ),
        LeakageCase(
            name="declare a feature available before its input",
            attempt=lambda: require_lineage_monotone(
                datetime(1999, 1, 5, 14, 0, tzinfo=UTC), [OBSERVED_AT]
            ),
            expected_error=FeatureLineageError,
        ),
    ])
    print(f"[5] leakage battery: {report.summary}")
    if not report.all_blocked:
        raise AssertionError("synthetic leakage battery did not block every attempt")

    # [6] Run manifest: canonical JSON that round-trips through from_json.
    manifest = RunManifest(
        run_id="example-foundation-contract-wiring",
        created_at=datetime(1999, 1, 5, 16, 0, tzinfo=UTC),
        python_version=platform.python_version(),
        package_version=portlearn.__version__,
        dependency_pins={"portlearn": portlearn.__version__},
        commands=("python examples/foundation_contract_wiring.py",),
    )
    canonical = manifest.to_json()
    print(f"[6] manifest: {canonical}")
    print(f"    round-trips unchanged: {RunManifest.from_json(canonical) == manifest}")
    print("\nComposition complete: the foundation contracts compose, and the")
    print("synthetic leak attempts were blocked by the fixed contract.")


if __name__ == "__main__":
    main()
