"""Shared leaf primitives for retrieval-only (UNQUALIFIED) data surfaces.

This module is a leaf of the ``portlearn.data`` package:
it defines the provider-agnostic value objects that the
retrieval-only sibling surfaces reduce provider bytes to, and **nothing
else**.  It imports no sibling provider module and no other part of
``portlearn.data`` (a leaf stays a leaf, a leaf stays a leaf,
so the package import graph stays acyclic).

Two value objects live here:

* :class:`PeriodKeyObservation` — one provider-labelled observation row
  carrying the provider's own temporal label verbatim (``period_key``),
  never a fabricated instant.  A record of this type declares no
  ``observation_time`` and no ``available_time`` at all: the attributes
  do not exist, so no decision-time surface can mistake it for a
  qualified :class:`~portlearn.observations.TimedObservation`.  This is
  the beginner surface — first-class historical availability is
  *absent by construction*, never defaulted.
* :class:`RetrievalProvenance` — the base class for the retrieval-only
  typed provenance records (``FFRetrievalProvenance``,
  ``FREDRetrievalProvenance``): retrieval facts only, **no availability
  field**, and deliberately *not* a
  :class:`~portlearn.data.ingestion.SourceProvenance` subclass (a
  retrieval-only record must never satisfy the qualified provenance
  contract).

Decoding laws enforced by the provider
sibling decoders, not here: identical bytes decode to identical
records; provider temporal labels are carried exactly, never
normalized; missing-data sentinels decode to absent records; and when a
caller supplies retrieval provenance the decode is hash-pinned to the
exact bytes the fetch recorded, failing closed on any mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PeriodKeyObservation", "RetrievalProvenance"]


@dataclass(frozen=True)
class PeriodKeyObservation:
    """One observation row carrying the provider's own period label.

    The triple ``(series_id, period_key, value)`` is the whole record:
    no instant attribute exists to fabricate.  ``period_key`` is the
    provider's temporal label carried verbatim (for example ``"202301"``
    for a Fama-French monthly row or ``"2023-01"`` for a FRED monthly
    observation date) — it is never normalized to an instant, because
    placing a period on the single timeline is exactly the
    qualification step this record deliberately omits.

    Construction is fail-closed: ``series_id`` and ``period_key`` must
    be non-blank strings (a blank identifier names no series; a blank
    period label names no period), and ``value`` must be a float in the
    closed ingestion value domain — no arbitrary object may pose as an
    observation value.  The record is frozen and hashable; identity and
    equality are field-wise over the whole triple.
    """

    series_id: str
    period_key: str
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.series_id, str) or not self.series_id.strip():
            raise ValueError(
                "series_id must be a non-empty, non-blank string identifier "
                "naming the series the observation belongs to; got "
                f"{self.series_id!r}. A blank identifier names no series, "
                "so the period-key observation is rejected fail-closed."
            )
        if not isinstance(self.period_key, str) or not self.period_key.strip():
            raise ValueError(
                "period_key must be a non-empty, non-blank provider period "
                f"label; got {self.period_key!r}. A blank label names no "
                "period, so the period-key observation is rejected "
                "fail-closed."
            )
        if not isinstance(self.value, float):
            raise TypeError(
                "value must be a float in the closed observation value "
                f"domain; got {type(self.value).__name__}: {self.value!r}. "
                "An untyped value cannot pose as an observation, so the "
                "period-key observation is rejected fail-closed."
            )


@dataclass(frozen=True)
class RetrievalProvenance:
    """Base for retrieval-only provenance: retrieval facts, no policy.

    The concrete provider records (``FFRetrievalProvenance``,
    ``FREDRetrievalProvenance``) extend this base with their pinned
    retrieval field sets.  The contract every subclass carries by
    construction:

    * retrieval facts only — the field set holds no ``availability``
      field at all, so an unqualified load can never smuggle a policy;
    * *not* a :class:`~portlearn.data.ingestion.SourceProvenance` — a
      retrieval-only record never satisfies the qualified provenance
      contract, so the qualified decoders' provenance fences cannot be
      satisfied by a retrieval-only artifact;
    * frozen and hashable (concrete classes keep every field hashable;
      the FRED record's ``request_params`` is a canonical immutable
      tuple of key-sorted pairs, so it remains hashable end-to-end).
    """

    def __post_init__(self) -> None:
        if type(self) is RetrievalProvenance:
            raise TypeError(
                "RetrievalProvenance is an abstract base for the provider "
                "retrieval-only provenance records; construct the provider "
                "record (FFRetrievalProvenance or FREDRetrievalProvenance) "
                "instead."
            )
