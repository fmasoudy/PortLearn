"""Offline information-set composition smoke over committed synthetic fixtures.

An availability-and-provenance plumbing check: this script composes
PortLearn's public decoder, transform, alignment, information-set,
leakage, and manifest surfaces over committed synthetic provider-format
fixtures (invented values replicating provider formats; no
provider-owned data redistributed) and prints availability and
provenance facts alone. Research datasets only; NOT investable — no
factor or industry-portfolio series here is a tradable asset. No other
capability claim is made, and none is possible from this output.

Offline and side-effect free: importing this module executes nothing;
every read is a committed fixture under ``tests/adapters/fixtures``,
no network is touched, no file is written, and no clock feeds the data
path — every retrieval instant is a fixed fixture-derived constant, so
repeated replays print identical summaries.
"""

from __future__ import annotations

import hashlib
import platform
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import portlearn
from portlearn.alignment import (
    ObservationStore,
    align,
    monthly_decision_calendar,
)
from portlearn.calendar import month_end_instant
from portlearn.data.adapters import ff, fred
from portlearn.data.ingestion import AvailabilityPolicy, SourceProvenance
from portlearn.interfaces import InformationSet, require_feature_lineage
from portlearn.leakage import LeakageCase, run_leakage_cases
from portlearn.manifest import RunManifest
from portlearn.observations import (
    AmbiguousObservationError,
    FeatureLineageError,
    TimedObservation,
    vintage_as_of,
)
from portlearn.timing import FutureInformationError, NaiveTimestampError
from portlearn.transforms import RollingMean

#: Boundary statement kept verbatim on one source line.
FENCE = (
    "Availability and provenance smoke only: synthetic committed "
    "fixtures, offline replay; research datasets only, NOT investable; "
    "no other capability claim is made or possible from this output."
)

#: The declared research timezone for every period-end mapping here.
ZONE = ZoneInfo("Australia/Melbourne")

#: Committed synthetic fixtures (invented values in the provider formats).
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "adapters" / "fixtures"
FF_MONTHLY_ZIP = FIXTURES / "ff" / "ff_factors_monthly_csv.zip"
FF_DAILY_ZIP = FIXTURES / "ff" / "ff_factors_daily_csv.zip"
FF49_ZIP = FIXTURES / "ff" / "ff_industry49_monthly_csv.zip"
FRED_CPIM_OBS = FIXTURES / "fred" / "obs_synthcpim_monthly.json"
FRED_CPIM_META = FIXTURES / "fred" / "meta_synthcpim.json"
FRED_DFFD_OBS = FIXTURES / "fred" / "obs_synthdffd_daily.json"
FRED_DFFD_META = FIXTURES / "fred" / "meta_synthdffd.json"

#: Declared availability policies (researcher declarations, never defaults).
FF_POLICY = AvailabilityPolicy.fixed_lag(
    timedelta(days=45),
    justification=(
        "Researcher declaration: the provider regenerates the monthly "
        "research files during the middle of the following month, so a "
        "45-day declared lag approximates first knowability of a "
        "revised snapshot."
    ),
)
FF_DAILY_POLICY = AvailabilityPolicy.fixed_lag(
    timedelta(days=15),
    justification=(
        "Researcher declaration: the daily research file is republished "
        "on a roughly two-week cycle, so a 15-day declared lag "
        "approximates first knowability of a revised snapshot."
    ),
)
CPIM_POLICY = AvailabilityPolicy.fixed_lag(
    timedelta(days=28),
    justification=(
        "Researcher declaration: a synthetic CPI-like release schedule "
        "about four weeks after the reference month."
    ),
)
DFFD_POLICY = AvailabilityPolicy.fixed_lag(
    timedelta(days=2),
    justification=(
        "Researcher declaration: a synthetic daily-rate publication "
        "convention of two days."
    ),
)

#: Fixed synthetic retrieval instant for the FF legs: adopting it through
#: the decoder's hash-pinned ``retrieval`` parameter keeps every printed
#: provenance fact fixture-derived (no clock feeds the data path).
FF_RETRIEVAL_AT = datetime(2026, 9, 12, 0, 0, tzinfo=UTC)

#: The synthetic FRED revision vintage used in the availability facts.
REVISION_AVAILABLE_AT = datetime(2026, 9, 25, 23, 59, 59, 999999, tzinfo=ZONE)

#: A fixed manifest instant (provenance of the run, not a data timestamp).
MANIFEST_CREATED_AT = datetime(2026, 9, 13, 0, 0, tzinfo=ZONE)

#: The pinned synthetic FF49 subset: three named industry portfolios of
#: the forty-nine, at the fixture's final monthly section. Candidate
#: asset-input semantics only — never an investability claim.
FF49_SUBSET = ("Banks", "Oil", "Util")
FF49_SUBSET_MONTH = (2023, 6)

#: The monthly factor groups requested at the composition decision.
FACTOR_SERIES = ("FF/HML", "FF/Mkt-RF", "FF/RF", "FF/SMB")
FACTOR_MONTH = (2025, 12)

#: The macro series and the reference month requested in the facts.
MACRO_SERIES = "FRED/SYNTHCPIM"
MACRO_MONTH = (2025, 12)

#: The daily-derived monthly feature window over the daily factor leg.
FEATURE_INPUT_SERIES = "FF/Mkt-RF"
FEATURE_WINDOW = 5

DIGEST_MARKER = "[h] replay output sha256: "


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ff_retrieval(dataset_id: str, data: bytes) -> ff.FFProvenance:
    """Hash-pinned synthetic retrieval provenance for one FF decode."""
    entry = ff.FF_DATASETS[dataset_id]
    return ff.FFProvenance(
        source_id=f"ff:{entry.dataset_id}",
        retrieval_instant=FF_RETRIEVAL_AT,
        content_sha256=_sha256(data),
        licence_note=entry.licence_note,
        availability=FF_POLICY,
        adapter_identity=ff.ADAPTER_IDENTITY,
        adapter_version=portlearn.__version__,
        row_count=0,
        header_order=[],
        units=entry.units,
        frequency=entry.frequency,
        url=entry.url,
        last_modified=None,
        dataset_kind=entry.dataset_kind,
        dataset_id=entry.dataset_id,
    )


def _family_line(
    label: str, records: int, provenance: SourceProvenance
) -> str:
    """One provenance summary line: availability and provenance facts only."""
    availability = provenance.availability
    lag = (
        f"FIXED_LAG {availability.delta}"
        if availability.mode == "FIXED_LAG"
        else availability.mode
    )
    mode = getattr(provenance, "data_mode", None)
    mode_text = f" data_mode={mode}" if mode is not None else ""
    return (
        f"[s] {label}: records={records}"
        f" content_sha256={provenance.content_sha256}"
        f" retrieval_instant={provenance.retrieval_instant.isoformat()}"
        f" availability={lag}"
        f" adapter={provenance.adapter_identity}"
        f"/{provenance.adapter_version}{mode_text}"
    )


def compose_replay() -> types.SimpleNamespace:
    """Compose the offline information-set smoke and return its facts.

    Pure composition over committed fixtures: reads fixtures, decodes
    through the public adapters, derives the feature, aligns, admits,
    and assembles the leakage battery — printing nothing and writing
    nothing. ``main`` renders these facts; tests re-derive them.
    """
    # FF research factors: monthly and daily legs through the
    # provider-format decoder with hash-pinned synthetic retrieval facts.
    monthly_bytes = FF_MONTHLY_ZIP.read_bytes()
    ff_monthly, ff_monthly_prov = ff.decode(
        monthly_bytes,
        "factors_monthly_csv",
        FF_POLICY,
        tz=ZONE,
        retrieval=_ff_retrieval("factors_monthly_csv", monthly_bytes),
    )
    daily_bytes = FF_DAILY_ZIP.read_bytes()
    ff_daily, ff_daily_prov = ff.decode(
        daily_bytes,
        "factors_daily_csv",
        FF_DAILY_POLICY,
        tz=ZONE,
        retrieval=_ff_retrieval("factors_daily_csv", daily_bytes),
    )

    # FRED macro legs: a monthly synthetic index and a daily synthetic
    # rate decoded from provider-format JSON under the snapshot data
    # mode, availability floored at the payload's realtime-end instant.
    cpim_bytes = FRED_CPIM_OBS.read_bytes()
    cpim, cpim_prov = fred.decode(
        cpim_bytes,
        "SYNTHCPIM",
        CPIM_POLICY,
        data_mode=fred.CURRENT_SNAPSHOT,
        tzinfo=ZONE,
        frequency="Monthly",
        metadata_bytes=FRED_CPIM_META.read_bytes(),
    )
    dffd_bytes = FRED_DFFD_OBS.read_bytes()
    dffd, dffd_prov = fred.decode(
        dffd_bytes,
        "SYNTHDFFD",
        DFFD_POLICY,
        data_mode=fred.CURRENT_SNAPSHOT,
        tzinfo=ZONE,
        frequency="Daily",
        metadata_bytes=FRED_DFFD_META.read_bytes(),
    )

    # Daily-derived monthly feature: the fixed rolling transform over
    # the daily factor leg, lineage-checked at each output's own window.
    feature_inputs = sorted(
        (
            record
            for record in ff_daily
            if record.series_id == FEATURE_INPUT_SERIES
        ),
        key=lambda record: record.available_time,
    )
    feature_outputs = RollingMean(FEATURE_WINDOW).transform(feature_inputs)
    for output in feature_outputs:
        admitted_at_output = [
            record
            for record in feature_inputs
            if record.available_time <= output.available_time
        ]
        require_feature_lineage(admitted_at_output, [output])

    # FF49 research portfolio series: the asset-input fence and a pinned
    # three-industry subset — never an investability claim.
    ff49_bytes = FF49_ZIP.read_bytes()
    ff49, ff49_prov = ff.decode(
        ff49_bytes,
        "industry49_monthly_csv",
        FF_POLICY,
        tz=ZONE,
        retrieval=_ff_retrieval("industry49_monthly_csv", ff49_bytes),
    )
    ff.require_candidate_asset_returns(ff49_prov)
    factor_fence_blocked = False
    try:
        ff.require_candidate_asset_returns(ff_monthly_prov)
    except ValueError:
        factor_fence_blocked = True

    # Alignment: one store over every decoded family, one month-end
    # decision instant from the calendar builder, and the requested
    # observation groups admitted through the strict vintage operation.
    store = ObservationStore(
        [*ff_monthly, *ff_daily, *feature_outputs, *cpim, *dffd, *ff49]
    )
    decision_instant = monthly_decision_calendar((2026, 9), 1, ZONE)[0]
    subset_month = month_end_instant(*FF49_SUBSET_MONTH, ZONE)
    macro_month = month_end_instant(*MACRO_MONTH, ZONE)
    requested_groups = [
        *(
            (series, month_end_instant(*FACTOR_MONTH, ZONE))
            for series in FACTOR_SERIES
        ),
        (feature_outputs[-1].series_id, feature_outputs[-1].observation_time),
        (MACRO_SERIES, macro_month),
        *((f"FF/{name}", subset_month) for name in FF49_SUBSET),
    ]
    aligned_records = align(store, decision_instant, requested_groups)

    # Availability facts at the publication boundary of the macro leg:
    # invisible one microsecond before the retrieval floor, admitted
    # exactly at it — explicit absence, never an imputed value.
    floor = cpim_prov.retrieval_instant
    pre_publication = floor - timedelta(microseconds=1)
    macro_groups = [(MACRO_SERIES, macro_month)]
    before_floor = align(store, pre_publication, macro_groups)
    at_floor = align(store, floor, macro_groups)

    # A synthetic revision pair over the decoded December macro record:
    # the first print is visible at the floor; the revision is invisible
    # until its own availability, then becomes the visible vintage.
    first_print = next(
        record
        for record in cpim
        if record.observation_time == macro_month
    )
    revision = TimedObservation(
        first_print.series_id,
        first_print.observation_time,
        REVISION_AVAILABLE_AT,
        first_print.value + 0.25,
    )
    vintage_at_floor = vintage_as_of([first_print, revision], floor)
    vintage_after = vintage_as_of(
        [first_print, revision], REVISION_AVAILABLE_AT
    )

    # The leakage battery: future, revision, lineage, and undated-
    # decision leak attempts, each expected to be blocked by its fixed
    # contract error.
    naive_decision = datetime(2026, 9, 30)  # noqa: DTZ001 — deliberately naive
    early_feature = TimedObservation(
        feature_outputs[-1].series_id,
        feature_inputs[0].observation_time,
        feature_inputs[0].available_time,
        0.0,
    )
    leakage_cases = [
        LeakageCase(
            name="admit a pre-publication macro record at an earlier decision",
            attempt=lambda: InformationSet(items=cpim, as_of=pre_publication),
            expected_error=FutureInformationError,
        ),
        LeakageCase(
            name="admit two vintages of one macro observation",
            attempt=lambda: InformationSet(
                items=[first_print, revision],
                as_of=REVISION_AVAILABLE_AT,
            ),
            expected_error=AmbiguousObservationError,
        ),
        LeakageCase(
            name="declare a feature available before its latest input",
            attempt=lambda: require_feature_lineage(
                feature_inputs, [early_feature]
            ),
            expected_error=FeatureLineageError,
        ),
        LeakageCase(
            name="query the alignment surface with an undated decision instant",
            attempt=lambda: align(store, naive_decision, macro_groups),
            expected_error=NaiveTimestampError,
        ),
    ]

    manifest = RunManifest(
        run_id="information-set-smoke-offline-replay",
        created_at=MANIFEST_CREATED_AT,
        python_version=platform.python_version(),
        package_version=portlearn.__version__,
        dependency_pins={"portlearn": portlearn.__version__},
        commands=("python examples/information_set_smoke.py",),
    )

    return types.SimpleNamespace(
        ff_monthly=ff_monthly,
        ff_monthly_prov=ff_monthly_prov,
        ff_daily=ff_daily,
        ff_daily_prov=ff_daily_prov,
        cpim=cpim,
        cpim_prov=cpim_prov,
        dffd=dffd,
        dffd_prov=dffd_prov,
        feature_inputs=feature_inputs,
        feature_outputs=feature_outputs,
        ff49=ff49,
        ff49_prov=ff49_prov,
        factor_provenance=ff_monthly_prov,
        portfolio_provenance=ff49_prov,
        factor_fence_blocked=factor_fence_blocked,
        store=store,
        decision_instant=decision_instant,
        requested_groups=requested_groups,
        aligned_records=aligned_records,
        floor=floor,
        pre_publication=pre_publication,
        before_floor=before_floor,
        at_floor=at_floor,
        first_print=first_print,
        revision=revision,
        vintage_at_floor=vintage_at_floor,
        vintage_after=vintage_after,
        leakage_cases=leakage_cases,
        manifest=manifest,
    )


def main() -> None:
    """Replay the offline information-set composition smoke."""
    replay = compose_replay()

    lines = [FENCE, "families:"]
    lines.append(
        _family_line("ff_factors_monthly", len(replay.ff_monthly),
                     replay.ff_monthly_prov)
    )
    lines.append(
        _family_line("ff_factors_daily", len(replay.ff_daily),
                     replay.ff_daily_prov)
    )
    lines.append(
        _family_line("fred_macro_monthly", len(replay.cpim),
                     replay.cpim_prov)
    )
    lines.append(
        _family_line("fred_macro_daily", len(replay.dffd),
                     replay.dffd_prov)
    )
    latest_output = replay.feature_outputs[-1]
    lines.append(
        f"[f] feature: series={latest_output.series_id}"
        f" transform=RollingMean window={FEATURE_WINDOW}"
        f" inputs={len(replay.feature_inputs)}"
        f" outputs={len(replay.feature_outputs)}"
        f" lineage_monotone={len(replay.feature_outputs)}"
        f"/{len(replay.feature_outputs)}"
    )
    subset_month = month_end_instant(*FF49_SUBSET_MONTH, ZONE)
    subset_values = sorted(
        record.value
        for record in replay.ff49
        if record.observation_time == subset_month
        and record.series_id in {f"FF/{name}" for name in FF49_SUBSET}
    )
    lines.append(
        f"[s] ff49_industry_portfolios:"
        f" content_sha256={replay.ff49_prov.content_sha256}"
        f" kind={replay.ff49_prov.dataset_kind}"
        f" records={len(replay.ff49)}"
        f" subset={'+'.join(FF49_SUBSET)}"
        f" subset_month={FF49_SUBSET_MONTH[0]}-{FF49_SUBSET_MONTH[1]:02d}"
        f" subset_values_present={len(subset_values)}/{len(FF49_SUBSET)}"
        f" asset_input_fence=admitted"
        f" factor_kind_fence_rejected={replay.factor_fence_blocked}"
        f" not_investable=yes"
    )

    lines.append("availability:")
    lines.append(
        f"[a] macro pre-publication ({replay.pre_publication.isoformat()}):"
        f" {len(replay.before_floor)} of 1 requested groups admitted"
    )
    lines.append(
        f"[a] macro retrieval floor ({replay.floor.isoformat()}):"
        f" {len(replay.at_floor)} of 1 requested groups admitted"
    )
    assert replay.vintage_at_floor is replay.first_print
    assert replay.vintage_after is replay.revision
    lines.append(
        f"[a] revision visibility: floor vintage is the first print;"
        f" the revised vintage becomes visible from"
        f" {REVISION_AVAILABLE_AT.isoformat()}"
    )
    lines.append(
        f"[1] decision {replay.decision_instant.isoformat()}:"
        f" {len(replay.aligned_records)} records admitted from"
        f" {len(replay.requested_groups)} requested observation groups"
    )

    report = run_leakage_cases(replay.leakage_cases)
    lines.append(f"[b] leakage battery: {report.summary}")
    for finding in report.findings:
        blocked_as = (
            finding.error_type.__qualname__
            if finding.error_type is not None
            else "unblocked"
        )
        lines.append(f"    blocked: {finding.case.name} -> {blocked_as}")
    if not report.all_blocked:
        raise AssertionError(
            "the leakage battery did not block every attempted leak"
        )

    lines.append(f"[m] manifest: {replay.manifest.to_json()}")
    lines.append(
        "composition complete: three data families composed, every leak "
        "attempt blocked, provenance re-derivable from committed fixtures"
    )

    text = "\n".join(lines) + "\n"
    digest = _sha256(text.encode())
    print(text + DIGEST_MARKER + digest, end="")


if __name__ == "__main__":
    main()
