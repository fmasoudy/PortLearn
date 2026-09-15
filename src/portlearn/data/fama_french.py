"""The Fama-French public data facade (progressive disclosure).

The first level of the progressive-disclosure surface: the complete
beginner surface is one call — ``load(...)`` — returning a sealed
two-state :class:`~portlearn.data.dataset.ResearchDataset`.  Nothing
below this module is required knowledge for that surface, and this
facade holds no privileged dataset-construction access: it
obtains datasets through exactly the public canonical constructors any
external integrator would use.

The fetch path always begins at the frozen retrieval-only sibling
(:func:`~portlearn.data.adapters.ff.fetch_raw`): the provider's exact
bytes plus typed retrieval facts, hash-pinned end-to-end.  The raw load
is UNQUALIFIED — no availability anywhere.  The qualified one-step load
requires the indivisible evidence pair ``(availability, tzinfo)`` —
both or neither — and is exactly the raw load plus an immediate
``qualify(...)`` over the same retained bytes, so one-step and two-step
are byte-identical by construction.  No default
availability policy and no default zone is ever supplied.

Import graph: this facade imports the alias catalog, the dataset
container, and its own provider adapter only — never the sibling
facade, never anything below ``portlearn.data`` importing upward.

Research datasets only; NOT investable.
"""

from __future__ import annotations

from typing import Any, Final

from portlearn.data.adapters import ff
from portlearn.data.dataset import QualifiedDataset, UnqualifiedDataset

__all__ = ["ALIAS_CATALOG_VERSION", "load", "resolve_ff_dataset_id"]

# --------------------------------------------------------------------------- #
# The provider alias catalog: research names to provider artifacts
# --------------------------------------------------------------------------- #

#: The catalog revision: versions this closed alias catalog's
#: resolution rules.  Datasets do not carry it: identity carries
#: provider-native provenance, not the catalog's own version.
ALIAS_CATALOG_VERSION: Final[int] = 1


#: The closed alias table: research name -> frequency -> provider
#: dataset id.  The frequencies are the catalog's own selector keys
#: (the FF catalog's lowercase ``monthly``/``daily``); every value is an
#: existing provider artifact — the selector never invents one.
_FF_ARTIFACTS: Final[dict[str, dict[str, str]]] = {
    "industry49": {
        "monthly": "industry49_monthly_csv",
    },
    "factors": {
        "monthly": "factors_monthly_csv",
        "daily": "factors_daily_csv",
    },
}


def resolve_ff_dataset_id(name: Any, frequency: Any = None) -> str:
    """Resolve one research alias to a provider dataset id, or reject.

    Fail-closed in exactly three ways, each before any network is
    touched:

    * an unknown or blank name rejects — the catalog is closed and no
      fuzzy or permissive matching exists;
    * a name whose alias maps to several artifacts with no ``frequency``
      rejects, listing the artifacts — the researcher, not the catalog,
      performs the selection;
    * a ``frequency`` that is not among the alias's artifacts rejects,
      listing the artifacts — a selector picks an existing artifact or
      rejects; it never resamples or converts one.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            "a research alias must be a non-empty, non-blank string; got "
            f"{name!r}. The closed alias catalog is "
            f"{sorted(_FF_ARTIFACTS)} — no permissive or fuzzy alias "
            "matching exists, fail-closed."
        )
    artifacts = _FF_ARTIFACTS.get(name)
    if artifacts is None:
        raise ValueError(
            f"the research alias {name!r} is not in the closed "
            f"catalog {sorted(_FF_ARTIFACTS)}; an unknown name rejects "
            "before any network is touched, fail-closed."
        )
    if frequency is None:
        if len(artifacts) > 1:
            raise ValueError(
                f"the research alias {name!r} maps to multiple provider "
                f"artifacts {sorted(artifacts.values())}; declare "
                "frequency= to select one — the catalog is a selector "
                "only and never picks an artifact on the researcher's "
                "behalf."
            )
        selected_frequency, dataset_id = next(iter(artifacts.items()))
        del selected_frequency
        return dataset_id
    if not isinstance(frequency, str) or frequency not in artifacts:
        raise ValueError(
            f"frequency {frequency!r} is not among the artifacts of the "
            f"research alias {name!r}: {sorted(artifacts.values())}. The "
            "catalog is a selector only — an existing artifact is picked "
            "or the request rejects; no artifact is ever resampled or "
            "converted to another frequency, fail-closed."
        )
    return artifacts[frequency]


def load(
    name: str,
    *,
    frequency: str | None = None,
    data: bytes | None = None,
    retrieval: Any = None,
    availability: Any = None,
    tzinfo: Any = None,
) -> UnqualifiedDataset | QualifiedDataset:
    """Load one Fama-French research dataset into a sealed dataset.

    Progressive disclosure in one signature:

    * **Simple (raw path):** ``load(name, frequency=...)`` fetches the
      provider's exact bytes (or adopts caller-supplied ``data`` plus
      ``retrieval`` facts) and returns the UNQUALIFIED dataset — period
      labels, no availability anywhere, nothing decision-time.
    * **Explicit (qualified one-step):** pass the indivisible pair
      ``availability=`` (an explicit
      :class:`~portlearn.data.ingestion.AvailabilityPolicy`) **and**
      ``tzinfo=`` (an aware zone) together; the load is then exactly
      the raw load plus an immediate ``qualify(...)`` over the same
      retained bytes and retrieval facts.  One-sided evidence rejects
      ``ValueError`` — a lone policy or a lone zone each smuggle a
      silent decision-time assumption.
    * **Full control (Level 2):** callers needing byte-level control
      fetch through :mod:`portlearn.data.adapters.ff` and hand the
      bytes and retrieval facts here.

    The alias catalog is a selector only: an unknown name, an ambiguous
    artifact, or a frequency not among the alias's artifacts rejects
    before any network is touched, and no artifact is ever resampled.
    """
    dataset_id = resolve_ff_dataset_id(name, frequency)
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
        if retrieval is not None:
            raise ValueError(
                "retrieval provenance may be supplied only together with "
                "the exact data bytes it pinned (data=...): retrieval "
                "facts without bytes name a retrieval that cannot be "
                "verified, fail-closed."
            )
        data, retrieval = ff.fetch_raw(dataset_id)
    else:
        if not isinstance(data, bytes):
            raise TypeError(
                "data must be the exact retained provider bytes (bytes); "
                f"got {type(data).__name__}."
            )
        if retrieval is not None and (
            not hasattr(retrieval, "content_sha256")
            or retrieval.content_sha256 != ff._sha256(data)
        ):
            raise ValueError(
                "retrieval provenance hash pin mismatch: the supplied "
                "bytes are not the bytes the retrieval recorded "
                f"({retrieval.content_sha256!r} pinned); the load "
                "rejects fail-closed."
            )
    records, provenance = ff.decode_unqualified(
        data, dataset_id, retrieval=retrieval
    )
    dataset = UnqualifiedDataset(
        provider="fama_french",
        name=name,
        provider_dataset_id=dataset_id,
        adapter_version=provenance.adapter_version,
        source_bytes=data,
        auxiliary_bytes=None,
        records=tuple(records),
        retrieval_provenance=provenance,
        decoder=ff.decode,
    )
    if availability is None:
        return dataset
    return dataset.qualify(availability=availability, tzinfo=tzinfo)
