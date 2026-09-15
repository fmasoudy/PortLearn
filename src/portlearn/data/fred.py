"""The FRED public data facade (progressive disclosure).

The first level of the progressive-disclosure surface: one call —
``load(...)`` — returning a sealed two-state dataset, constructed
through exactly the public canonical constructors (no privileged
internal path exists).

The fetch path always begins at the frozen retrieval-only sibling
(:func:`~portlearn.data.adapters.fred.fetch_raw`): the provider's exact
bytes plus typed retrieval facts with the API key validated before any
socket opens, sent to the provider inside the request URL only, and
never recorded anywhere.  The series metadata document is fetched
through the same provider seam (``adapters.fred.urlopen``) and retained
verbatim as ``auxiliary_bytes`` so a later qualification re-decodes
with identical units and frequency facts.

The raw load is UNQUALIFIED — no availability anywhere, the
``CURRENT_SNAPSHOT`` limitation carried as the data mode.  The
qualified one-step load requires the indivisible evidence pair
``(availability, tzinfo)`` — both or neither — and is exactly the raw
load plus an immediate ``qualify(...)`` over the same retained bytes,
so one-step and two-step are byte-identical by construction.  No
default availability policy and no default zone is ever supplied.

Import graph: this facade imports the dataset container and its own
provider adapter only — never the sibling facade, never anything below
``portlearn.data`` importing upward.

Research datasets only; NOT investable.
"""

from __future__ import annotations

from typing import Any

from portlearn.data.adapters import fred
from portlearn.data.dataset import QualifiedDataset, UnqualifiedDataset

__all__ = ["load"]


def _canonical_frequency(frequency: Any) -> str:
    """Map the facade's lowercase declaration to the provider label."""
    if not isinstance(frequency, str) or not frequency.strip():
        raise ValueError(
            "frequency is mandatory and this facade supplies no default: "
            "declare the series frequency (monthly, quarterly, or daily) "
            f"so the period mapping is never inferred; got {frequency!r}."
        )
    canonical = frequency.strip().capitalize()
    if canonical not in fred.SUPPORTED_FREQUENCIES:
        raise ValueError(
            f"frequency {frequency!r} is outside the closed provider set "
            f"{fred.SUPPORTED_FREQUENCIES!r}; unsupported frequencies "
            "reject fail-closed — they have no frozen period mapping."
        )
    return canonical


def _fetch_metadata(series_id: str, api_key: str) -> bytes:
    """Fetch the series metadata document through the provider seam.

    The key rides the request URL to the provider only — the same
    fetch-time-only law the adapter enforces — and the returned bytes
    are never parsed here: they are retained verbatim as auxiliary
    facts for the frozen decoders, which own every parse rule.
    """
    from urllib.parse import urlencode

    # One URL home: the base is derived from the adapter's single _BASE_URL.
    base = fred._BASE_URL.rsplit("/", 1)[0]
    query = urlencode(
        {"series_id": str(series_id), "file_type": "json", "api_key": api_key}
    )
    with fred.urlopen(f"{base}?{query}", timeout=60) as response:
        return response.read()


def load(
    series_id: str,
    api_key: str,
    *,
    frequency: str,
    data: bytes | None = None,
    retrieval: Any = None,
    metadata_bytes: bytes | None = None,
    availability: Any = None,
    tzinfo: Any = None,
) -> UnqualifiedDataset | QualifiedDataset:
    """Load one FRED series into a sealed dataset.

    ``api_key`` is a required argument: this facade never reads
    credentials from the environment or any other silent source.  The
    key is validated before any socket opens, sent to the provider
    inside the request URL only, and never recorded in any provenance
    or error message.

    Progressive disclosure in one signature:

    * **Simple (raw path):** ``load(series_id, api_key,
      frequency=...)`` fetches the observations bytes plus the series
      metadata document and returns the UNQUALIFIED dataset — period
      labels, the ``CURRENT_SNAPSHOT`` limitation, no availability.
    * **Explicit (qualified one-step):** add the indivisible pair
      ``availability=`` (an explicit policy declaration) **and**
      ``tzinfo=`` (an aware zone) together; one-sided evidence rejects
      ``ValueError``.
    * **Full control (Level 2):** callers needing byte-level control
      fetch through :mod:`portlearn.data.adapters.fred` and hand the
      bytes and retrieval facts here.
    """
    canonical = _canonical_frequency(frequency)
    if (availability is not None or tzinfo is not None) and (
        availability is None or tzinfo is None
    ):
        raise ValueError(
            "qualification takes the indivisible evidence pair: pass "
            "BOTH availability= (an explicit AvailabilityPolicy "
            "declaration) and tzinfo= (an aware zone) together — "
            "one-sided evidence is refused fail-closed, because a "
            "lone policy or a lone zone each smuggle a silent "
            "decision-time assumption. No default availability and "
            "no default zone is ever supplied."
        )
    if data is None:
        if retrieval is not None or metadata_bytes is not None:
            raise ValueError(
                "retrieval provenance and metadata bytes may be supplied "
                "only together with the exact observation bytes they "
                "describe (data=...): retrieval facts without bytes name "
                "a retrieval that cannot be verified, fail-closed."
            )
        data, retrieval = fred.fetch_raw(series_id, api_key)
        metadata_bytes = _fetch_metadata(series_id, api_key)
    else:
        if not isinstance(data, bytes):
            raise TypeError(
                "data must be the exact retained provider bytes (bytes); "
                f"got {type(data).__name__}."
            )
        if retrieval is not None and (
            not hasattr(retrieval, "content_sha256")
            or retrieval.content_sha256 != fred._sha256(data)
        ):
            raise ValueError(
                "retrieval provenance hash pin mismatch: the supplied "
                "bytes are not the bytes the retrieval recorded "
                f"({retrieval.content_sha256!r} pinned); the load "
                "rejects fail-closed."
            )
    records, provenance = fred.decode_unqualified(
        data,
        series_id,
        frequency=canonical,
        metadata_bytes=metadata_bytes,
        retrieval=retrieval,
    )

    def _qualified_decoder(
        retained: bytes,
        resolved_id: str,
        policy: Any,
        *,
        tzinfo: Any = None,
        retrieval: Any = None,
    ) -> Any:
        return fred.decode(
            retained,
            resolved_id,
            policy,
            data_mode=fred.CURRENT_SNAPSHOT,
            tzinfo=tzinfo,
            frequency=canonical,
            metadata_bytes=metadata_bytes,
            retrieval=retrieval,
        )

    dataset = UnqualifiedDataset(
        provider="fred",
        name=series_id,
        provider_dataset_id=series_id,
        adapter_version=provenance.adapter_version,
        source_bytes=data,
        auxiliary_bytes=metadata_bytes,
        records=tuple(records),
        retrieval_provenance=provenance,
        decoder=_qualified_decoder,
    )
    if availability is None:
        return dataset
    return dataset.qualify(availability=availability, tzinfo=tzinfo)
