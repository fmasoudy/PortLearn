"""The forecasting and estimation lifecycle contract.

This module defines the lifecycle laws around the
``Forecaster`` protocol — fitting, refitting, forecast timing,
tuning, seeds, determinism, and provenance — as declared value
objects and data-admission helpers beside that protocol, never inside it.
It ships contracts only: no estimator, no wrapper, no model zoo, no
registry, and no engine. A research repository implements a forecaster satisfying the ``Forecaster`` protocol and consumes these
lifecycle objects.

Five laws govern the whole module:

- **One clock for the cutoff.** The training-information cutoff of a
  fit is the ``as_of`` of the ``InformationSet`` consumed by
  ``fit`` — exactly that field, never re-derived, never
  caller-declared, and no cutoff parameter, field, or keyword exists
  on this module's surface through which a second clock could be
  smuggled. Admission at that ``as_of`` has already excluded every
  record whose ``available_time`` follows it, so the law holds by
  construction; the exported boundary checks re-assert it detectably
  so a hostile or wrapped estimator that keeps future-available
  records fails closed instead of warning.
- **No second time vocabulary.** Every chronology or admission
  failure at this surface raises one of the timing or observations errors, re-used by qualified import from
  ``portlearn.timing`` and never re-defined here; the only errors
  this module defines are the structural ``ValueError`` subclasses
  (``PlanStructureError``, ``ScheduleStructureError``,
  ``ProvenanceStructureError``) for malformed lifecycle objects.
- **One prediction path.** The ``forecast(information_set) -> Forecast`` signature is the entire
  prediction contract; this module defines no other callable that
  produces a ``Forecast`` and no path that bypasses
  ``InformationSet`` admission.
- **Declarations over machinery.** ``FittingPlan`` and
  ``RefitSchedule`` are immutable declared value objects; the
  experiment loop — the caller — owns every fit and refit decision.
  No scheduler, engine, registry, or hyperparameter-search
  implementation exists here.
- **Declared determinism.** A plan carries exactly one determinism
  class — ``DETERMINISTIC``, ``SEED_REPRODUCIBLE`` (which requires a
  declared seed), or ``NONDETERMINISTIC`` (which requires its
  explanation) — and the class rides through provenance onto every
  forecast. The lifecycle never silently assumes determinism.

Configuration carried by a plan is deep-copied and canonicalized at
construction into nested immutable containers (dict →
``MappingProxyType``, list/tuple → tuple, string keys only), so a
caller can never modify a plan's configuration through a nested
container after construction, and the configuration hash is a
deterministic SHA-256 over the canonicalized value: key order and
list/tuple distinction never affect it, while any value difference
does.

The module imports stdlib only at module level plus ``portlearn``
timing by qualified module import — aware-instant validation and those errors remain implemented solely in ``portlearn.timing``;
nothing here re-implements or re-exports them. Importing this module
performs no I/O and modifies nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from hashlib import sha256
from itertools import pairwise
from types import MappingProxyType
from typing import Any, Protocol

from . import timing as _timing

__all__ = [
    "DeterminismClass",
    "FitWindow",
    "FittedModel",
    "FittingPlan",
    "ForecastProvenance",
    "ForecasterFactory",
    "PlanStructureError",
    "ProvenanceStructureError",
    "RefitSchedule",
    "ScheduleStructureError",
    "fitting_provenance",
    "require_cutoff_from_fit_set",
    "require_fit_inputs_admitted",
    "require_reproducible",
    "require_selection_labels_available",
]


# --------------------------------------------------------------------------- #
# Structural error arm — module-owned, disjoint from the timing and observations errors
# --------------------------------------------------------------------------- #


class PlanStructureError(ValueError):
    """A fitting plan, fit window, or helper input is malformed.

    The module-owned structural arm: a blank or non-string model
    identity, an unsupported configuration shape, a mis-typed seed, a
    raw instant in place of a fit window, a wrong determinism
    declaration, or a malformed record collection. Distinct on purpose from the timing module's chronology and admission errors, which remain the only errors for timing and admission violations.
    """


class ScheduleStructureError(ValueError):
    """A refit schedule declaration is structurally malformed.

    Raised for a schedule that is not a non-empty sequence of aware
    instants (a non-iterable, a string, or an empty grid). Naive or
    non-monotone entries raise the timing module's errors instead —
    this class never shadows them.
    """


class ProvenanceStructureError(ValueError):
    """Lifecycle provenance is malformed or dishonest.

    Raised when a cutoff is asserted somewhere other than the fit
    set's own ``as_of`` (the one-clock law), when a provenance or
    reproducibility input is structurally malformed, or when a
    declared determinism class is contradicted by the evidence.
    """


# --------------------------------------------------------------------------- #
# Declared determinism classes
# --------------------------------------------------------------------------- #


class DeterminismClass(Enum):
    """The three declared determinism kinds of a fitting method.

    ``DETERMINISTIC`` — same inputs (including any seed) produce
    identical outputs, always. ``SEED_REPRODUCIBLE`` — identical
    outputs given an identical declared seed; the class requires
    one. ``NONDETERMINISTIC`` — residual nondeterminism disclosed by
    a mandatory explanation. Exactly one class is carried on every
    plan and rides through provenance; none is ever assumed.
    """

    DETERMINISTIC = "deterministic"
    SEED_REPRODUCIBLE = "seed-reproducible"
    NONDETERMINISTIC = "nondeterministic"


# --------------------------------------------------------------------------- #
# Declared value objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FitWindow:
    """The declared start of the fitting window.

    A single declared aware instant — the earliest observation time
    the fitting method will consume — with no end field: the window's
    end is the fit information set's own ``as_of`` (the one-clock
    law), so a second end clock cannot exist either. Naive
    timestamps and calendar dates are rejected with ``NaiveTimestampError``; the stored instant is the exact object
    supplied, never coerced or normalized.
    """

    window_start: datetime

    def __post_init__(self) -> None:
        _timing.to_instant(self.window_start, "window_start")


def _require_model_identity(model: Any) -> str:
    """Reject a blank or non-string model identity ."""
    if not isinstance(model, str):
        raise PlanStructureError(
            "a fitting plan's model must be an exact string identity "
            "naming the fitting method it declares; got "
            f"{type(model).__name__}: {model!r}. A non-string "
            "identity names no method, so the plan is rejected "
            "unconditional."
        )
    if not model.strip():
        raise PlanStructureError(
            "a fitting plan's model must be a non-empty, non-blank "
            "string identity naming the fitting method it declares; "
            f"got {model!r}. A blank identity names no method, so "
            "the plan is rejected unconditionally."
        )
    return model


def _require_seed(seed: Any) -> int | None:
    """Accept a declared integer seed or a declared ``None`` only."""
    if isinstance(seed, bool):
        raise PlanStructureError(
            "a fitting plan's seed must be an integer or declared "
            f"None; got {seed!r} of type bool. A boolean is not a "
            "seed, so the plan is rejected unconditionally."
        )
    if seed is None or isinstance(seed, int):
        return seed
    raise PlanStructureError(
        "a fitting plan's seed must be an integer or declared None; "
        f"got {type(seed).__name__}: {seed!r}. Any other seed type "
        "carries no declared reproducibility meaning, so the plan "
        "is rejected unconditionally."
    )


_SUPPORTED_SCALAR_TYPES = (str, int, float, bool, type(None))


def _canonicalize_config(value: Any, path: str) -> Any:
    """Canonicalize a configuration value into immutable containers.

    Mappings become ``MappingProxyType`` over canonicalized keys and
    values with string keys only, sequences become tuples, and only
    the scalar types a configuration may legitimately carry pass
    through. Any other type — a set, a custom object, a non-string
    key — rejects with ``PlanStructureError``: configuration is
    declared data, not an execution surface, and its content hash
    must be a pure function of content.
    """
    if isinstance(value, _SUPPORTED_SCALAR_TYPES):
        return value
    if isinstance(value, Mapping):
        canonical: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise PlanStructureError(
                    "a fitting plan's configuration must map non-blank "
                    f"string names to values; got key {key!r} at "
                    f"{path}. Configuration is declared data with "
                    "named fields, so a non-string or blank key names "
                    "no setting and the plan is rejected unconditionally."
                )
            canonical[key] = _canonicalize_config(item, f"{path}[{key!r}]")
        return MappingProxyType(canonical)
    if isinstance(value, (list, tuple)):
        return tuple(
            _canonicalize_config(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise PlanStructureError(
        "a fitting plan's configuration must contain only scalar, "
        f"mapping, and sequence values; got {type(value).__name__} at "
        f"{path}: {value!r}. Configuration is declared data, not an "
        "execution surface, so the plan is rejected unconditionally."
    )


def _config_digest(canonical: Any) -> str:
    """A deterministic SHA-256 over the canonicalized configuration.

    Key order never affects the digest (mapping entries are sorted by
    key), float values hash by their exact ``repr``, and
    ``True``/``1`` stay distinct because a type tag travels with
    every value. The digest is a pure function of content: two
    canonicalizations of equal content always agree and any content
    difference separates.
    """

    def encode(node: Any) -> str:
        if isinstance(node, Mapping):
            entries = sorted(
                (key, encode(item)) for key, item in node.items()
            )
            body = ";".join(f"{key}:{item}" for key, item in entries)
            return "{" + body + "}"
        if isinstance(node, tuple):
            return "[" + ",".join(encode(item) for item in node) + "]"
        if isinstance(node, bool):
            return f"b:{node}"
        if node is None:
            return "n:"
        if isinstance(node, str):
            return f"s:{node}"
        if isinstance(node, int):
            return f"i:{node}"
        return f"f:{node!r}"

    return sha256(encode(canonical).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FittingPlan:
    """The declared unit of fitting provenance.

    A ``FittingPlan`` declares what will be fitted, and nothing
    else: the model identity (an exact string), the configuration
    (deep-copied and canonicalized into nested immutable containers
    at construction — a caller can never modify it afterwards, not
    even through a nested container that was passed in), the random
    seed or its declared absence, the fit window (a ``FitWindow``
    value object), and exactly one determinism class.

    There is deliberately no training-cutoff field and none can be
    added: the cutoff of any fit under this plan is the ``as_of`` of
    the ``InformationSet`` that fit consumes — one clock, recorded
    in provenance by ``fitting_provenance``, and re-asserted by the
    exported boundary checks. A cutoff-bearing construction is not
    representable: unexpected keyword arguments reject with
    ``TypeError``.

    ``SEED_REPRODUCIBLE`` requires a declared integer seed.
    ``NONDETERMINISTIC`` requires a non-blank explanation carried
    verbatim into provenance. A deterministic or seed-reproducible
    declaration with an explanation rejects — the declaration must
    stand alone. Immutable; the configuration's content hash is
    exposed as ``config_hash`` and is deterministic over content.
    """

    model: str
    config: Mapping[str, Any] = field(default_factory=dict)
    seed: int | None = None
    fit_window: FitWindow = field(default=None)  # type: ignore[assignment]
    determinism: DeterminismClass = DeterminismClass.DETERMINISTIC
    nondeterminism_explanation: str | None = None

    def __post_init__(self) -> None:
        model = _require_model_identity(self.model)
        canonical = _canonicalize_config(self.config, "config")
        if not isinstance(canonical, Mapping):
            raise PlanStructureError(
                "a fitting plan's configuration must be a mapping from "
                f"names to values; got {type(self.config).__name__}: "
                f"{self.config!r}. Configuration is declared data, so "
                "the plan is rejected unconditionally."
            )
        seed = _require_seed(self.seed)
        if not isinstance(self.fit_window, FitWindow):
            raise PlanStructureError(
                "a fitting plan's fit_window must be a FitWindow value "
                f"object; got {type(self.fit_window).__name__}: "
                f"{self.fit_window!r}. The window is declared data, "
                "never a bare instant, so the plan is rejected "
                "unconditional."
            )
        if not isinstance(self.determinism, DeterminismClass):
            raise PlanStructureError(
                "a fitting plan's determinism must be a "
                "DeterminismClass member — DETERMINISTIC, "
                "SEED_REPRODUCIBLE, or NONDETERMINISTIC; got "
                f"{type(self.determinism).__name__}: "
                f"{self.determinism!r}. The class is a declared fact, "
                "never a string or an inference, so the plan is "
                "rejected unconditionally."
            )
        explanation = self.nondeterminism_explanation
        if self.determinism is DeterminismClass.NONDETERMINISTIC:
            if not isinstance(explanation, str) or not explanation.strip():
                raise PlanStructureError(
                    "a NONDETERMINISTIC fitting plan must disclose its "
                    "residual nondeterminism in a non-blank "
                    "nondeterminism_explanation; the lifecycle never "
                    "silently assumes determinism and never accepts an "
                    "undisclosed leak of it."
                )
        elif explanation is not None:
            raise PlanStructureError(
                "a fitting plan carries nondeterminism_explanation only "
                "under a NONDETERMINISTIC declaration; got "
                f"{self.determinism.name} with explanation "
                f"{explanation!r}. A determinism-respecting declaration "
                "must stand alone."
            )
        if (
            self.determinism is DeterminismClass.SEED_REPRODUCIBLE
            and seed is None
        ):
            raise PlanStructureError(
                "a SEED_REPRODUCIBLE fitting plan requires a declared "
                "integer seed; got None. Reproducibility that depends "
                "on a seed cannot be declared without one, so the plan "
                "is rejected unconditionally."
            )
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "config", canonical)
        object.__setattr__(self, "seed", seed)

    @property
    def config_hash(self) -> str:
        """The deterministic SHA-256 content hash of the config."""
        return _config_digest(self.config)


@dataclass(frozen=True)
class RefitSchedule:
    """The declared grid of instants when re-estimation is permitted.

    An immutable value object over researcher-supplied plain aware
    instants — no calendar machinery of any kind. Construction
    validates the grid with the timing module's rules: a naive entry
    rejects with ``NaiveTimestampError``; repeated, reversed, or
    zone-aliased-equal entries reject with
    ``InvalidChronologyError`` (strict increase, compared on
    normalized instants so equal instants in different zones still
    reject); a non-sequence, a string, or an empty grid rejects with
    ``ScheduleStructureError``. The stored tuple holds the exact
    datetime objects supplied, never normalized copies.

    A schedule declares when re-estimation is permitted — never when
    prediction is. The experiment loop owns every fit; estimators
    gain no ``refit`` method, no internal clock, and no scheduler
    from this object.
    """

    refit_instants: tuple[datetime, ...]

    def __init__(self, refit_instants: Sequence[datetime]) -> None:
        if isinstance(refit_instants, (str, bytes)):
            raise ScheduleStructureError(
                "a refit schedule must be a sequence of aware instants, "
                f"not a single {type(refit_instants).__name__}: "
                f"{refit_instants!r}."
            )
        try:
            entries = tuple(refit_instants)
        except TypeError as error:
            raise ScheduleStructureError(
                "a refit schedule must be an iterable of aware instants; "
                f"got {type(refit_instants).__name__}: "
                f"{refit_instants!r}."
            ) from error
        if not entries:
            raise ScheduleStructureError(
                "a refit schedule must contain at least one aware "
                "instant; an empty grid names no re-estimations."
            )
        normalized = [
            _timing.instant_key(entry, "refit instant") for entry in entries
        ]
        for previous, current in pairwise(normalized):
            if not previous < current:
                raise _timing.InvalidChronologyError(
                    "chronology violation: a refit schedule must be "
                    "strictly increasing aware instants, but "
                    f"{current.isoformat()} does not strictly follow "
                    f"{previous.isoformat()}. A re-estimation grid "
                    "that repeats or reverses an instant names no "
                    "valid sequence of fits."
                )
        object.__setattr__(self, "refit_instants", entries)


@dataclass(frozen=True)
class ForecastProvenance:
    """The provenance of one fit under a declared plan.

    Records exactly the lifecycle facts: the model identity, the
    configuration content hash, the fit cutoff — always the fit
    information set's own ``as_of``, never an independently
    supplied instant — the declared seed or its explicit absence,
    the declared determinism class (and its explanation under
    ``NONDETERMINISTIC``), the declared fit window, and, where a
    researcher-side wrapper adapted an external estimator, that
    wrapped implementation's recorded identity. ``produced_by`` is
    the deterministic provenance token that rides on the ``Forecast`` value object without changing its schema.

    Immutable. Use ``fitting_provenance`` to construct the object with
    the cutoff taken from the fit set itself.
    """

    model: str
    config_hash: str
    cutoff: datetime
    seed: int | None
    determinism: DeterminismClass
    fit_window: FitWindow
    wrapped_identity: str | None = None
    nondeterminism_explanation: str | None = None
    produced_by: str = ""

    def __post_init__(self) -> None:
        token = "portlearn.forecasting:"
        token += f"{self.model}"
        token += f"/config_sha256={self.config_hash}"
        token += f"/fit_cutoff={self.cutoff.isoformat()}"
        token += f"/seed={self.seed if self.seed is not None else 'none'}"
        token += f"/determinism={self.determinism.value}"
        if self.determinism is DeterminismClass.NONDETERMINISTIC:
            token += (
                f"/nondeterminism_note={self.nondeterminism_explanation}"
            )
        if self.wrapped_identity is not None:
            token += f"/wrapped={self.wrapped_identity}"
        object.__setattr__(self, "produced_by", token)


# --------------------------------------------------------------------------- #
# The lifecycle protocols — declaration-only, static-only
# --------------------------------------------------------------------------- #


class FittedModel(Protocol):
    """A fitted model: a forecaster produced by one fit.

    Declaration-only and static-only, exactly like the ``Forecaster`` protocol it satisfies: the single ``forecast``
    method consumes one admitted ``InformationSet``, treats its
    ``as_of`` as the forecast origin, and returns a ``Forecast``
    whose ``decision_time`` equals that ``as_of`` — the implementer's
    obligation, verified behaviorally at this surface, never by
    ``isinstance``. No ``refit``, no state accessor, and no second
    prediction path exists on this protocol.
    """

    def forecast(self, information_set: Any) -> Any:
        """Produce the forecast for one admitted information set."""
        ...


class ForecasterFactory(Protocol):
    """An estimator: fits exactly one admitted information set.

    Declaration-only and static-only. ``fit`` consumes exactly one
    ``InformationSet`` — the training set, admission already enforced
    at its ``as_of`` — and returns a ``FittedModel``. The training
    cutoff is that set's ``as_of`` (one clock); there is no cutoff
    parameter, and a fit that cannot honor the fit-set cutoff
    rejects at the fitting boundary with ``FutureInformationError`` rather than warning. No registry, no
    ``get_params``, no hyperparameter search exists on this
    protocol: selection timing is governed by
    ``require_selection_labels_available`` on the caller's side.
    """

    def fit(self, information_set: Any) -> FittedModel:
        """Fit on one admitted information set and return the model."""
        ...


# --------------------------------------------------------------------------- #
# Boundary checks and provenance helpers
# --------------------------------------------------------------------------- #


def _fit_set_as_of(fit_set: Any) -> datetime:
    """The fit set's own ``as_of``, validated as an aware instant."""
    as_of = getattr(fit_set, "as_of", None)
    if as_of is None and fit_set is not None:
        raise PlanStructureError(
            "a fit information set must expose its own ``as_of`` "
            "instant; got "
            f"{type(fit_set).__name__}: {fit_set!r}. The cutoff of a "
            "fit is exactly that instant (one clock), so a fit set "
            "without one cannot be checked."
        )
    return _timing.to_instant(as_of, "the fit set's as_of")


def require_fit_inputs_admitted(records: Iterable[Any], fit_set: Any) -> None:
    """Re-assert admission for every record fitting consumes.

    The fitting boundary rule: each record's ``available_time`` must
    be at or before the fit set's own ``as_of``. Any record whose
    availability follows that instant rejects with ``FutureInformationError`` — fitting that cannot honor the
    fit-set cutoff fails closed, never warns — and a malformed
    record or a naive ``as_of`` raises the corresponding typed error. A
    structurally malformed records collection rejects with
    ``PlanStructureError``. Returns ``None`` when every record is
    admitted.
    """
    if isinstance(records, (str, bytes)) or not isinstance(records, Iterable):
        raise PlanStructureError(
            "the records consumed by fitting must be an iterable of "
            f"admitted observations; got {type(records).__name__}: "
            f"{records!r}. Fitting checks what it can enumerate, so a "
            "non-iterable input is rejected unconditionally."
        )
    cutoff = _fit_set_as_of(fit_set)
    for record in records:
        available = getattr(record, "available_time", None)
        if available is None:
            raise PlanStructureError(
                "a record consumed by fitting must expose its "
                f"available_time; got {type(record).__name__}: "
                f"{record!r}. The fitting boundary cannot check a "
                "record that declares no availability."
            )
        available_instant = _timing.to_instant(
            available, "the record's available_time"
        )
        if available_instant > cutoff:
            series_id = getattr(record, "series_id", None)
            raise _timing.FutureInformationError(
                "look-ahead rejection at the fitting boundary: the "
                f"record {series_id!r} consumed by fitting is not "
                "available at the fit set's own as_of — available_time="
                f"{available_instant.isoformat()} is after as_of="
                f"{cutoff.isoformat()}. The training cutoff is exactly "
                "the fit set's as_of (one clock), so fitting that "
                "consumes future-available records violates the "
                "lifecycle contract and fails closed."
            )


def require_cutoff_from_fit_set(asserted: Any, fit_set: Any) -> None:
    """Reject any cutoff assertion that diverges from the fit set's.

    The one-clock consistency rule: an instant asserted as the
    training cutoff anywhere in the lifecycle must be exactly the
    fit set's own ``as_of`` — later or earlier both reject with
    ``ProvenanceStructureError``, a naive assertion rejects with ``NaiveTimestampError``, and a fit set with no ``as_of``
    rejects with ``PlanStructureError``. Comparison is on normalized
    instants, so equal instants expressed in different zones agree.
    Returns ``None`` when the assertion agrees.
    """
    asserted_instant = _timing.to_instant(asserted, "the asserted cutoff")
    actual = _fit_set_as_of(fit_set)
    if _timing.instant_key(asserted_instant) != _timing.instant_key(actual):
        raise ProvenanceStructureError(
            "one-clock violation: a training cutoff was asserted as "
            f"{asserted_instant.isoformat()}, but the cutoff of a fit "
            "is exactly the fit information set's own as_of="
            f"{actual.isoformat()}. There is no second clock — any "
            "divergent assertion (the wrapped library's internal "
            "bookkeeping included) is rejected unconditionally."
        )


def require_selection_labels_available(
    labels: Iterable[Any], selection_origin: Any
) -> None:
    """Require hyperparameter-selection labels at the selection origin.

    The tuning-timing law: every label a selection scores on must be
    realizable at the selection origin — its ``available_time`` at
    or before that instant. A label available after the origin is
    the walk-forward leak made structural: it rejects with ``FutureInformationError`` at the selection boundary. A naive origin rejects with ``NaiveTimestampError``; a
    structurally malformed labels collection rejects with
    ``PlanStructureError``. Returns ``None`` when every label is
    realizable.
    """
    if isinstance(labels, (str, bytes)) or not isinstance(labels, Iterable):
        raise PlanStructureError(
            "the labels consumed by a selection must be an iterable of "
            f"observations; got {type(labels).__name__}: {labels!r}. "
            "The selection boundary checks what it can enumerate, so a "
            "non-iterable input is rejected unconditionally."
        )
    origin = _timing.to_instant(selection_origin, "the selection origin")
    for label in labels:
        available = getattr(label, "available_time", None)
        if available is None:
            raise PlanStructureError(
                "a selection label must expose its available_time; got "
                f"{type(label).__name__}: {label!r}. The selection "
                "boundary cannot check a label that declares no "
                "availability."
            )
        available_instant = _timing.to_instant(
            available, "the label's available_time"
        )
        if available_instant > origin:
            series_id = getattr(label, "series_id", None)
            raise _timing.FutureInformationError(
                "look-ahead rejection at the selection boundary: the "
                f"selection label {series_id!r} is not realizable at "
                "the selection origin — available_time="
                f"{available_instant.isoformat()} is after the "
                f"selection origin {origin.isoformat()}. Tuning that "
                "scores configurations on labels not yet available is "
                "the walk-forward leak, and it is rejected unconditionally."
            )


def fitting_provenance(
    plan: FittingPlan,
    fit_set: Any,
    wrapped_identity: str | None = None,
) -> ForecastProvenance:
    """Build the provenance of one fit under a declared plan.

    The one constructor of ``ForecastProvenance``: the cutoff is
    taken from the fit information set's own ``as_of`` — there is no
    cutoff parameter through which divergence could be smuggled —
    and the plan's declared identity, configuration hash, seed,
    determinism class, and fit window ride through unchanged. The
    declared fit window's start must not follow the cutoff (``InvalidChronologyError`` otherwise). Where a
    researcher-side wrapper adapted an external estimator,
    ``wrapped_identity`` records that implementation's identity and
    version (non-blank string); it is provenance only and grants no
    exemption from any lifecycle law.
    """
    if not isinstance(plan, FittingPlan):
        raise PlanStructureError(
            "fitting provenance requires a FittingPlan value object; "
            f"got {type(plan).__name__}: {plan!r}. Provenance records "
            "a declared plan, never loose arguments."
        )
    cutoff = _fit_set_as_of(fit_set)
    window_start = _timing.instant_key(
        plan.fit_window.window_start, "the fit window's window_start"
    )
    if window_start > _timing.instant_key(cutoff):
        raise _timing.InvalidChronologyError(
            "chronology violation: the declared fit window starts at "
            f"{plan.fit_window.window_start.isoformat()}, after the "
            f"fit cutoff {cutoff.isoformat()} (the fit set's own "
            "as_of). A fitting window cannot open after the "
            "information the fit consumed."
        )
    if wrapped_identity is not None and (
        not isinstance(wrapped_identity, str) or not wrapped_identity.strip()
    ):
        raise PlanStructureError(
            "a wrapped estimator identity must be a non-blank "
            "string recording the external implementation's "
            f"identity and version; got {wrapped_identity!r}."
        )
    return ForecastProvenance(
        model=plan.model,
        config_hash=plan.config_hash,
        cutoff=cutoff,
        seed=plan.seed,
        determinism=plan.determinism,
        fit_window=plan.fit_window,
        wrapped_identity=wrapped_identity,
        nondeterminism_explanation=plan.nondeterminism_explanation,
    )


def require_reproducible(forecasts: Any) -> None:
    """Assert declared determinism is honored by actual output.

    Given two or more ``Forecast`` results that a single declared
    determinism class claims are identical — two runs of the same
    deterministic stand-in, or two fresh instances of the same
    seeded declaration — every forecast must agree on values,
    target, decision time, and produced-by provenance. Any
    divergence is a dishonest declaration and rejects with
    ``ProvenanceStructureError``; a structurally malformed input
    (fewer than two forecasts, a non-sequence, or a non-forecast
    entry) rejects with the same structural error. Returns ``None``
    when the declaration is honored.
    """
    if isinstance(forecasts, (str, bytes)) or not isinstance(
        forecasts, Iterable
    ):
        raise ProvenanceStructureError(
            "reproducibility evidence must be an iterable of forecast "
            f"results; got {type(forecasts).__name__}: {forecasts!r}."
        )
    entries = list(forecasts)
    if len(entries) < 2:
        raise ProvenanceStructureError(
            "reproducibility evidence requires at least two forecasts "
            f"to compare; got {len(entries)}. A single result proves "
            "nothing about determinism."
        )
    first = entries[0]
    reference_fields = ("values", "target", "decision_time", "produced_by")
    for name in reference_fields:
        if not hasattr(first, name):
            raise ProvenanceStructureError(
                "reproducibility evidence entries must be forecast "
                f"results exposing {reference_fields}; got "
                f"{type(first).__name__}: {first!r}."
            )
    for other in entries[1:]:
        if any(
            getattr(other, name, None) != getattr(first, name)
            for name in reference_fields
        ):
            raise ProvenanceStructureError(
                "determinism violation: forecasts claimed identical "
                "under one declared determinism class diverge — the "
                "results differ in values, target, decision_time, or "
                "produced-by provenance. A declared class the evidence "
                "contradicts is rejected unconditionally."
            )
