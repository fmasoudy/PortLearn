"""The ``ResearchDataset`` container: one dataset, two sealed states.

This module is the sole pandas home of the package:
the module-scope pandas import lives here and nowhere else, pandas is a
normal core dependency (never an optional extra, never a guarded
import), and no other data-subsystem module touches it.

The container law: one ``ResearchDataset`` value object
with exactly two sealed states — UNQUALIFIED (retrieval only: period
labels, no availability anywhere) and QUALIFIED (decision-time eligible
records under an explicit caller policy).  The two-state model makes
every forbidden mixed state unrepresentable at the public constructors:
an UNQUALIFIED dataset carries ``availability is None`` and
``qualified_provenance is None`` by construction, a QUALIFIED dataset
carries both populated by construction, and the canonical constructors
validate every cross-field invariant fail-closed (there is no
privileged internal construction path; the facades build datasets
through exactly these constructors).

Both state subclasses declare no storage of their own (``__slots__``
stays the base tuple alone), so every attribute write resolves to a
read-only property descriptor and fails — including the low-level
assignment bypass.

Decoders are injected: this module never imports the
provider adapter modules, so the dependency direction is one-way and
the import graph stays acyclic.  A facade hands the frozen qualified
decoder of its provider in as an ordinary callable when constructing a
dataset; ``qualify`` routes through it, which is what hash-pins the
re-decode to the exact retained bytes.

Equality is value equality over the full frozen field set with exactly
one normalization: the wall-clock ``retrieval_instant`` inside the
typed provenances is excluded (two live retrievals of the same bytes
are the same dataset).  The hash derives solely from
the frozen identity tuple — provenances and records join equality but
never the hash, so no unhashable provenance field (the qualified FRED
record's dict ``request_params``) can break hashing.

Research datasets only; NOT investable.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable
from typing import Any

import pandas as pd

from portlearn.interfaces import InformationSet
from portlearn.observations import TimedObservation
from portlearn.timing import MissingAvailabilityError

from ._records import PeriodKeyObservation

__all__ = [
    "QUALIFIED",
    "UNQUALIFIED",
    "QualifiedDataset",
    "ResearchDataset",
    "UnqualifiedDataError",
    "UnqualifiedDataset",
]

#: The two sealed availability states.
UNQUALIFIED = "UNQUALIFIED"
QUALIFIED = "QUALIFIED"

_STATES = (UNQUALIFIED, QUALIFIED)


class UnqualifiedDataError(TypeError):
    """An UNQUALIFIED dataset was pushed onto a decision-time surface.

    Raised by :meth:`ResearchDataset.to_information_set` when the
    dataset is still UNQUALIFIED: its records carry no availability
    facts at all, so admitting them at a decision instant is not a
    validation question but a category error — qualification (an
    explicit caller policy plus zone, hash-pinned to the retained
    bytes) is the only route in.
    """


class _SealedInstantAttribute(AttributeError, MissingAvailabilityError):
    """Read of a decision-time attribute off a sealed UNQUALIFIED record.

    Dual-classified on purpose: as an ``AttributeError``
    the attribute genuinely does not exist (``hasattr`` is ``False``,
    and no decision-time surface can mistake the record for a qualified
    one), and as a ``MissingAvailabilityError`` the frozen admission
    law reads the very same absence as missing availability — attribute
    absence is missing availability, the law the frozen surfaces
    already enforce.
    """

    def __init__(self, record: Any, name: str) -> None:
        super().__init__(
            f"the UNQUALIFIED record {record.series_id!r}@"
            f"{record.period_key!r} declares no {name}: the period-key "
            "record carries the provider's own period label only, and "
            "placing it on the single timeline is exactly the "
            "qualification step it omits — availability is a declaration "
            "the dataset's qualify(availability=..., tzinfo=...) owns, "
            "never an attribute an unqualified load could fabricate."
        )


class _SealedPeriodKeyObservation(PeriodKeyObservation):
    """A period-key record sealed against decision-time misreads.

    Same value, same frozen field triple, same equality as the
    underlying :class:`PeriodKeyObservation` — but a read of the two
    decision-time attribute names raises the dual-classified sealed
    refusal instead of a bare ``AttributeError``, so the frozen
    admission surfaces report missing availability exactly as
    pinned.  Constructed only by the dataset container, from records the
    frozen unqualified decoder produced.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        if name in ("observation_time", "available_time"):
            raise _SealedInstantAttribute(self, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_non_blank_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{label} must be a non-empty, non-blank string; got "
            f"{value!r}. A blank identifier names nothing, so the "
            "dataset rejects fail-closed."
        )
    return value


def _normalized(provenance: Any) -> Any:
    """Provenance value with the wall-clock retrieval instant excluded.

    The retrieval instant is a retrieval fact of the
    fetch, not a fact of the dataset's content, so dataset equality
    compares provenances field-for-field with exactly this one field
    removed.  ``None`` stays ``None`` (no qualified provenance exists
    in the UNQUALIFIED state).
    """
    if provenance is None:
        return None
    return (
        type(provenance).__qualname__,
        tuple(
            (field.name, getattr(provenance, field.name))
            for field in dataclasses.fields(provenance)
            if field.name != "retrieval_instant"
        ),
    )


class ResearchDataset:
    """One provider dataset in exactly one of the two sealed states.

    Read-only surfaces: every public attribute is a property without a
    setter, so mutation is impossible in both states
    — including through the low-level assignment bypass, which hits the
    property descriptor and fails.  ``units``, ``frequency``, and (where
    the active typed provenance declares one) ``data_mode`` delegate to
    that provenance (the facade never restates them);
    ``source_sha256`` is computed from the retained bytes at
    construction, never trusted from a caller.  ``provenance`` routes
    read-only to the active state's provenance — the retrieval
    provenance while UNQUALIFIED, the qualified provenance once
    QUALIFIED — with ``retrieval_provenance`` and
    ``qualified_provenance`` remaining the explicit named facts.
    """

    __slots__ = (
        "_adapter_version",
        "_auxiliary_bytes",
        "_availability",
        "_availability_state",
        "_decoder",
        "_name",
        "_provider",
        "_provider_dataset_id",
        "_qualified_provenance",
        "_records",
        "_retrieval_provenance",
        "_source_bytes",
        "_source_sha256",
    )

    def __init__(
        self,
        *,
        provider: str,
        name: str,
        provider_dataset_id: str,
        adapter_version: str,
        availability_state: str,
        availability: Any,
        source_bytes: bytes,
        auxiliary_bytes: bytes | None,
        records: tuple[Any, ...],
        retrieval_provenance: Any,
        qualified_provenance: Any = None,
        decoder: Callable[..., Any] | None = None,
    ) -> None:
        """Validate every cross-field invariant fail-closed, or reject."""
        if type(self) is ResearchDataset:
            raise TypeError(
                "ResearchDataset is the sealed two-state container; "
                "construct the state type (UnqualifiedDataset or "
                "QualifiedDataset) instead — a stateless dataset is "
                "unrepresentable."
            )
        # The provider id is an opaque, structurally validated non-blank
        # string (provider-neutral kernel): any non-blank string —
        # built-in (``"fama_french"``, ``"fred"``) or integrator-supplied
        # (``"crsp"``, ``"my_private_source"``) — composes into the same
        # sealed container with no kernel edit, no registry, and no
        # provider subclass.
        self._provider = _require_non_blank_str(provider, "provider")
        self._name = _require_non_blank_str(name, "name")
        self._provider_dataset_id = _require_non_blank_str(
            provider_dataset_id, "provider_dataset_id"
        )
        self._adapter_version = _require_non_blank_str(
            adapter_version, "adapter_version"
        )
        if availability_state not in _STATES:
            raise ValueError(
                f"availability_state must be one of {_STATES!r}; got "
                f"{availability_state!r} — the two-state model is closed."
            )
        self._availability_state = availability_state
        if not isinstance(source_bytes, bytes):
            raise TypeError(
                "source_bytes must be the exact retained provider bytes "
                f"(bytes); got {type(source_bytes).__name__}."
            )
        self._source_bytes = source_bytes
        self._source_sha256 = _sha256(source_bytes)
        if auxiliary_bytes is not None and not isinstance(
            auxiliary_bytes, bytes
        ):
            raise TypeError(
                "auxiliary_bytes must be retained provider metadata bytes "
                "or None; got "
                f"{type(auxiliary_bytes).__name__}."
            )
        self._auxiliary_bytes = auxiliary_bytes
        if not isinstance(retrieval_provenance, object) or (
            retrieval_provenance is None
        ):
            raise ValueError(
                "retrieval_provenance is mandatory: the typed retrieval "
                "facts are retained from load and never rebuilt."
            )
        if hasattr(retrieval_provenance, "availability"):
            raise ValueError(
                "retrieval_provenance carries an availability field — a "
                "retrieval-only record must never satisfy the qualified "
                "provenance contract, fail-closed."
            )
        for fact in ("content_sha256", "units", "frequency"):
            if not hasattr(retrieval_provenance, fact):
                raise ValueError(
                    "retrieval_provenance is not a typed retrieval record: "
                    f"the {fact!r} retrieval fact is absent."
                )
        if retrieval_provenance.content_sha256 != self._source_sha256:
            raise ValueError(
                "retrieval provenance hash pin mismatch: the retained "
                "bytes are not the bytes the retrieval recorded "
                f"({retrieval_provenance.content_sha256!r} pinned, "
                f"{self._source_sha256!r} retained); the dataset rejects "
                "fail-closed."
            )
        self._retrieval_provenance = retrieval_provenance
        if availability_state == UNQUALIFIED:
            if availability is not None:
                raise ValueError(
                    "an UNQUALIFIED dataset carries no availability: the "
                    "state is sealed, availability enters only through "
                    "qualify(...), fail-closed."
                )
            if qualified_provenance is not None:
                raise ValueError(
                    "an UNQUALIFIED dataset carries no qualified "
                    "provenance: the state is sealed, fail-closed."
                )
            if not isinstance(decoder, Callable):
                raise TypeError(
                    "an UNQUALIFIED dataset requires the injected provider "
                    "decoder callable so qualify(...) "
                    "routes through the frozen qualified decoder."
                )
            self._availability = None
            self._qualified_provenance = None
            self._decoder = decoder
        else:
            if availability is None:
                raise ValueError(
                    "a QUALIFIED dataset carries the caller-supplied "
                    "availability policy; a policy-less qualified state "
                    "is unrepresentable, fail-closed."
                )
            if qualified_provenance is None:
                raise ValueError(
                    "a QUALIFIED dataset carries its typed qualified "
                    "provenance; the state is sealed, fail-closed."
                )
            if not hasattr(qualified_provenance, "availability"):
                raise ValueError(
                    "qualified_provenance must satisfy the qualified "
                    "provenance contract (it carries the availability "
                    "declaration), fail-closed."
                )
            if qualified_provenance.availability != availability:
                raise ValueError(
                    "the qualified provenance availability declaration "
                    "disagrees with the dataset availability policy — the "
                    "two must be the same declaration, fail-closed."
                )
            if (
                qualified_provenance.content_sha256
                != self._source_sha256
            ):
                raise ValueError(
                    "qualified provenance hash pin mismatch: the decode "
                    "was not over the retained bytes, fail-closed."
                )
            if decoder is not None:
                raise TypeError(
                    "a QUALIFIED dataset carries no decoder: "
                    "qualification is one-way, fail-closed."
                )
            self._availability = availability
            self._qualified_provenance = qualified_provenance
            self._decoder = None
        if not isinstance(records, tuple):
            raise TypeError(
                "records must be a tuple (the dataset is immutable); got "
                f"{type(records).__name__}."
            )
        if availability_state == UNQUALIFIED:
            # Seal each period-key record: the value triple is unchanged,
            # but a decision-time attribute read reports the frozen
            # missing-availability law instead of a bare AttributeError
            # (attribute absence is missing availability).
            records = tuple(
                _SealedPeriodKeyObservation(
                    series_id=record.series_id,
                    period_key=record.period_key,
                    value=record.value,
                )
                for record in records
            )
        self._records = records
        self._validate_records()

    def _validate_records(self) -> None:
        """Records must match the sealed state exactly, or reject."""
        if self._availability_state == UNQUALIFIED:
            for record in self._records:
                if not isinstance(record, PeriodKeyObservation):
                    raise TypeError(
                        "an UNQUALIFIED dataset holds period-key records "
                        "only; a non-period-key record "
                        f"({type(record).__name__}) is unrepresentable, "
                        "fail-closed."
                    )
        else:
            for record in self._records:
                if not isinstance(record, TimedObservation):
                    raise TypeError(
                        "a QUALIFIED dataset holds timed observation "
                        "records only; a record without decision-time "
                        f"facts ({type(record).__name__}) is "
                        "unrepresentable, fail-closed."
                    )

    # -- read-only surfaces (properties without setters; mutation is
    #    impossible in both states) --------------------------------------

    @property
    def provider(self) -> str:
        """The opaque provider id (any non-blank string)."""
        return self._provider

    @property
    def name(self) -> str:
        """The research alias (FF) or series id (FRED) the load named."""
        return self._name

    @property
    def provider_dataset_id(self) -> str:
        """The provider artifact id the load named/resolver selected."""
        return self._provider_dataset_id

    @property
    def adapter_version(self) -> str:
        """The adapter distribution version stamped at retrieval."""
        return self._adapter_version

    @property
    def availability_state(self) -> str:
        """The sealed state: ``"UNQUALIFIED"`` or ``"QUALIFIED"``."""
        return self._availability_state

    @property
    def availability(self) -> Any:
        """The caller-supplied policy (QUALIFIED), else ``None``."""
        return self._availability

    @property
    def units(self) -> str:
        """Units, delegated to the typed provenance of this state."""
        return self._state_provenance.units

    @property
    def frequency(self) -> str:
        """Frequency, delegated to the typed provenance of this state."""
        return self._state_provenance.frequency

    @property
    def data_mode(self) -> str:
        """Data mode, delegated to the typed provenance of this state.

        Carried only by providers whose typed provenance declares one
        (FRED's closed mode set); a provider with no data
        mode fact exposes no ``data_mode`` at all — the kernel stores
        and validates nothing here.
        """
        return self._state_provenance.data_mode

    @property
    def source_bytes(self) -> bytes:
        """The exact retained provider bytes."""
        return self._source_bytes

    @property
    def source_sha256(self) -> str:
        """The sha256 of the retained bytes, computed at construction."""
        return self._source_sha256

    @property
    def auxiliary_bytes(self) -> bytes | None:
        """Retained provider metadata bytes (FRED); ``None`` for FF."""
        return self._auxiliary_bytes

    @property
    def records(self) -> tuple[Any, ...]:
        """The sealed record tuple of this state."""
        return self._records

    @property
    def retrieval_provenance(self) -> Any:
        """The typed retrieval facts, retained from load, never rebuilt."""
        return self._retrieval_provenance

    @property
    def qualified_provenance(self) -> Any:
        """The typed qualified provenance; ``None`` unless QUALIFIED."""
        return self._qualified_provenance

    @property
    def provenance(self) -> Any:
        """The typed provenance of the active sealed state.

        The provider-neutral routing surface: the retrieval provenance
        while UNQUALIFIED, the qualified provenance once QUALIFIED.
        Read-only; the advanced named surfaces
        (:attr:`retrieval_provenance`, :attr:`qualified_provenance`)
        remain the explicit facts.
        """
        return self._state_provenance

    @property
    def _state_provenance(self) -> Any:
        if self._availability_state == UNQUALIFIED:
            return self._retrieval_provenance
        return self._qualified_provenance

    # -- equality, hash, representation ---------------------------------

    def _identity_scalars(self) -> tuple[Any, ...]:
        return (
            self._provider,
            self._name,
            self._provider_dataset_id,
            self._adapter_version,
            self._availability_state,
            self._availability,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResearchDataset):
            return NotImplemented
        return (
            self._identity_scalars() == other._identity_scalars()
            and self._records == other._records
            and self._source_bytes == other._source_bytes
            and self._auxiliary_bytes == other._auxiliary_bytes
            and _normalized(self._retrieval_provenance)
            == _normalized(other._retrieval_provenance)
            and _normalized(self._qualified_provenance)
            == _normalized(other._qualified_provenance)
        )

    def __hash__(self) -> int:
        """Hash from the frozen identity tuple only.

        Provenances and records join equality but never the hash, so no
        unhashable provenance field can break hashing; value-equal
        datasets always hash equal, and unequal datasets may collide.
        """
        return hash(
            (
                self._provider,
                self._name,
                self._provider_dataset_id,
                self._adapter_version,
                self._availability_state,
                self._availability,
                self.units,
                self.frequency,
                self._source_sha256,
            )
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(provider={self._provider!r}, "
            f"name={self._name!r}, "
            f"availability_state={self._availability_state!r}, "
            f"records={len(self._records)}, "
            f"source_sha256={self._source_sha256!r})"
        )

    # -- state transitions and bridges -----------------------------------

    def qualify(
        self, availability: Any = None, *, tzinfo: Any = None
    ) -> QualifiedDataset:
        """Qualify through the injected frozen decoder, or refuse.

        Overridden by :class:`UnqualifiedDataset`; the base (and the
        QUALIFIED state) refuse: qualification is one-way.
        """
        raise TypeError(
            "qualification is one-way: this dataset admits no "
            "qualify(...) — an UNQUALIFIED dataset qualifies exactly "
            "once, a QUALIFIED dataset never again."
        )

    def to_information_set(self, as_of: Any = None) -> InformationSet:
        """Bridge to the decision-time surface; overridden per state."""
        raise UnqualifiedDataError(
            "an UNQUALIFIED dataset cannot enter decision time: its "
            "records carry no availability facts at all. Qualify it "
            "first — dataset.qualify(availability=<AvailabilityPolicy>, "
            "tzinfo=<aware zone>) — then admit the qualified records; no "
            "default AvailabilityPolicy exists and none is assumed."
        )

    def to_pandas(self) -> pd.DataFrame:
        """Convert to a DataFrame with the frozen per-state columns.

        UNQUALIFIED: ``["series_id", "period_key", "value"]``.
        QUALIFIED: ``["series_id", "observation_time", "value",
        "available_time"]``.  Pandas is a normal core dependency here —
        this is the module-scope pandas home, and the conversion is
        plain interoperability, never a decision-time surface.
        """
        if self._availability_state == UNQUALIFIED:
            return pd.DataFrame(
                {
                    "series_id": [
                        record.series_id for record in self._records
                    ],
                    "period_key": [
                        record.period_key for record in self._records
                    ],
                    "value": [record.value for record in self._records],
                }
            )
        return pd.DataFrame(
            {
                "series_id": [record.series_id for record in self._records],
                "observation_time": [
                    record.observation_time for record in self._records
                ],
                "value": [record.value for record in self._records],
                "available_time": [
                    record.available_time for record in self._records
                ],
            }
        )


class UnqualifiedDataset(ResearchDataset):
    """The retrieval-only sealed state: period labels, no availability.

    Constructed by the provider facades through exactly the public
    canonical constructor (no privileged internal path exists),
    holding the provider's period-key records, the retained
    bytes, and the typed retrieval facts.  The provider's frozen
    qualified decoder travels in as an injected callable so
    :meth:`qualify` re-decodes the retained bytes under the caller's
    policy — hash-pinned, never re-fetched, never restated.
    """

    __slots__ = ()

    def __init__(
        self,
        *,
        provider: str,
        name: str,
        provider_dataset_id: str,
        adapter_version: str,
        source_bytes: bytes,
        auxiliary_bytes: bytes | None,
        records: tuple[PeriodKeyObservation, ...],
        retrieval_provenance: Any,
        decoder: Callable[..., Any],
    ) -> None:
        super().__init__(
            provider=provider,
            name=name,
            provider_dataset_id=provider_dataset_id,
            adapter_version=adapter_version,
            availability_state=UNQUALIFIED,
            availability=None,
            source_bytes=source_bytes,
            auxiliary_bytes=auxiliary_bytes,
            records=records,
            retrieval_provenance=retrieval_provenance,
            qualified_provenance=None,
            decoder=decoder,
        )

    def qualify(
        self, availability: Any = None, *, tzinfo: Any = None
    ) -> QualifiedDataset:
        """Qualify with the indivisible evidence pair, or refuse.

        ``availability`` and ``tzinfo`` are an indivisible pair: both or
        neither, one-sided or empty refusal is fail-closed, and no
        default :class:`~portlearn.data.ingestion.AvailabilityPolicy` and no
        default zone is ever supplied.  The re-decode runs
        through the injected frozen provider decoder over the retained
        bytes with the retained retrieval facts, so it is hash-pinned
        to exactly the bytes the retrieval recorded.
        """
        if availability is None or tzinfo is None:
            raise ValueError(
                "qualification takes the indivisible evidence pair: pass "
                "BOTH availability= (an explicit AvailabilityPolicy "
                "declaration) and tzinfo= (an aware zone) together. "
                "One-sided or empty qualification is refused fail-closed "
                "— no default AvailabilityPolicy and no default zone is "
                "ever supplied, because each alone smuggles a silent "
                "decision-time assumption."
            )
        timed, qualified_provenance = self._decoder(  # type: ignore[misc]
            self._source_bytes,
            self._provider_dataset_id,
            availability,
            tzinfo=tzinfo,
            retrieval=self._retrieval_provenance,
        )
        return QualifiedDataset(
            provider=self._provider,
            name=self._name,
            provider_dataset_id=self._provider_dataset_id,
            adapter_version=qualified_provenance.adapter_version,
            availability=availability,
            source_bytes=self._source_bytes,
            auxiliary_bytes=self._auxiliary_bytes,
            records=tuple(timed),
            retrieval_provenance=self._retrieval_provenance,
            qualified_provenance=qualified_provenance,
        )

    def to_information_set(self, as_of: Any = None) -> InformationSet:
        """Refuse: UNQUALIFIED records cannot enter decision time."""
        raise UnqualifiedDataError(
            "an UNQUALIFIED dataset cannot enter decision time: its "
            "records carry no availability facts at all, so there is "
            "nothing admissible at any decision instant. Qualify it "
            "first — dataset.qualify(availability=<AvailabilityPolicy>, "
            "tzinfo=<aware zone>) — and admit the qualified records; no "
            "default AvailabilityPolicy exists and none is assumed, "
            "because availability is a declaration, never an inference."
        )


class QualifiedDataset(ResearchDataset):
    """The decision-time eligible sealed state.

    Reached only through :meth:`UnqualifiedDataset.qualify` (and the
    facades' one-step qualified load, which is that same path): the
    records are the frozen decoder's timed observations stamped under
    exactly the caller's policy, the qualified provenance is the frozen
    decoder's own record, and the retrieval facts of the load are
    retained unchanged.  The state is terminal: no further
    qualification exists.
    """

    __slots__ = ()

    def __init__(
        self,
        *,
        provider: str,
        name: str,
        provider_dataset_id: str,
        adapter_version: str,
        availability: Any,
        source_bytes: bytes,
        auxiliary_bytes: bytes | None,
        records: tuple[TimedObservation, ...],
        retrieval_provenance: Any,
        qualified_provenance: Any,
    ) -> None:
        super().__init__(
            provider=provider,
            name=name,
            provider_dataset_id=provider_dataset_id,
            adapter_version=adapter_version,
            availability_state=QUALIFIED,
            availability=availability,
            source_bytes=source_bytes,
            auxiliary_bytes=auxiliary_bytes,
            records=records,
            retrieval_provenance=retrieval_provenance,
            qualified_provenance=qualified_provenance,
            decoder=None,
        )

    def to_information_set(self, as_of: Any = None) -> InformationSet:
        """Admit the qualified records at ``as_of``, fail-closed.

        The bridge is the frozen :class:`~portlearn.interfaces.\
InformationSet` admission law itself: every record's timing
        (``available_time``) and lineage (the decoder's declared table
        provenance) is carried unchanged, and admission at ``as_of`` is
        decided by that law, never by this container.
        """
        return InformationSet(self._records, as_of=as_of)
