"""Run manifest: minimal, aware-only, canonical-JSON run identity.

This module implements the run-manifest layer of the public
contract: every synthetic wiring run records one :class:`RunManifest`
carrying exactly six fields — ``run_id``, ``created_at``,
``python_version``, ``package_version``, ``dependency_pins``,
``commands`` — and nothing else. The schema is deliberately minimal:
**no ``seed`` field exists** (seeds live in run parameters,
not in run identity), so a manifest can never silently assert
randomness control it does not provide.

Design laws honoured here:

* **Identity is fail-closed** — blank ``run_id`` /
  ``python_version`` / ``package_version``, an empty or blank-valued
  ``dependency_pins`` mapping, or an empty ``commands`` sequence
  rejects with ``ValueError``.
* **Aware-ness law** — ``created_at`` must be a timezone-aware
  instant; naive datetimes and calendar dates reject with
  ``NaiveTimestampError`` (reused by import from ``portlearn.timing``,
  where it is solely implemented).
* **Canonical serialization** — ``to_json()`` emits deterministic
  canonical JSON: sorted keys, the compact separators ``","`` and
  ``":"``, and ISO-8601 datetime text carrying the original offset.
  ``from_json`` inverts it exactly and rejects offset-free serialized
  instants with ``NaiveTimestampError``.
* **Offset preservation** — the round trip preserves the original
  UTC offset exactly (a ``+10:00`` instant stays ``+10:00``), because
  the offset is part of the recorded fact.
* **Equality** — two manifests are equal iff their canonical JSON
  bytes are equal, so manifest equality is byte-stable identity.

The module imports the standard library plus ``NaiveTimestampError``
and the aware-instant validator from ``portlearn.timing`` by import;
importing it performs no I/O and writes nothing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import Any, ClassVar

from portlearn.timing import NaiveTimestampError, _require_aware_instant

__all__ = ["RunManifest"]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``object_pairs_hook`` that fails closed on duplicate object keys.

    ``json.loads`` silently keeps the last value of a duplicated key;
    a manifest must never accept an ambiguous payload, so any
    duplicate — at the top level or in any nested object — rejects.
    Raises ``_DuplicateKeyError`` (a ``ValueError``) naming the key.
    """

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(
                f"duplicate JSON object key {key!r} — the payload is "
                "ambiguous and no last-write-wins reading is allowed"
            )
        result[key] = value
    return result


class _DuplicateKeyError(ValueError):
    """Internal marker: the JSON text contained a duplicate object key."""


class RunManifest:
    """Minimal six-field run identity with canonical JSON serialization.

    The public schema is exactly the six fields — there is no ``seed``
    field: seeds live in run parameters, not run identity.

    ``created_at`` keeps its original UTC offset through every
    serialization round trip; equality is canonical-JSON byte equality.
    """

    __slots__ = (
        "commands",
        "created_at",
        "dependency_pins",
        "package_version",
        "python_version",
        "run_id",
    )

    #: The exact public schema keys, in canonical (sorted) order.
    _SCHEMA_KEYS: ClassVar[tuple[str, ...]] = (
        "commands",
        "created_at",
        "dependency_pins",
        "package_version",
        "python_version",
        "run_id",
    )

    def __init__(
        self,
        run_id: str,
        created_at: datetime,
        python_version: str,
        package_version: str,
        dependency_pins: Mapping[str, str],
        commands: Sequence[str],
    ) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(
                "RunManifest.run_id must be a non-blank string identifying "
                f"the run; got {run_id!r}."
            )
        if not isinstance(python_version, str) or not python_version.strip():
            raise ValueError(
                "RunManifest.python_version must be a non-blank string; "
                f"got {python_version!r}."
            )
        if not isinstance(package_version, str) or not package_version.strip():
            raise ValueError(
                "RunManifest.package_version must be a non-blank string; "
                f"got {package_version!r}."
            )
        # Aware-ness law: reuse the single timing-owned validator by
        # import — never re-implement it here.
        aware_created_at = _require_aware_instant(
            created_at, "RunManifest.created_at"
        )
        if not isinstance(dependency_pins, Mapping):
            raise ValueError(  # noqa: TRY004 — admission rejects with ValueError
                "RunManifest.dependency_pins must be a mapping of "
                "distribution name to version pin; got "
                f"{type(dependency_pins).__name__}: {dependency_pins!r}."
            )
        if not dependency_pins:
            raise ValueError(
                "RunManifest.dependency_pins must contain at least one "
                "(distribution, pin) pair — a manifest with no dependency "
                "provenance is not fail-closed."
            )
        for distribution, pin in dependency_pins.items():
            if not isinstance(distribution, str) or not distribution.strip():
                raise ValueError(
                    "RunManifest.dependency_pins keys must be non-blank "
                    f"distribution names; got {distribution!r}."
                )
            if not isinstance(pin, str) or not pin.strip():
                raise ValueError(
                    "RunManifest.dependency_pins values must be non-blank "
                    f"version pins; got {pin!r} for {distribution!r}."
                )
        if isinstance(commands, (str, bytes)) or not isinstance(
            commands, Sequence
        ):
            raise ValueError(  # noqa: TRY004 — admission rejects with ValueError
                "RunManifest.commands must be a sequence of command "
                "strings; got "
                f"{type(commands).__name__}: {commands!r}."
            )
        commands_tuple = tuple(commands)
        if not commands_tuple:
            raise ValueError(
                "RunManifest.commands must contain at least one command."
            )
        for command in commands_tuple:
            if not isinstance(command, str) or not command.strip():
                raise ValueError(
                    "RunManifest.commands entries must be non-blank "
                    f"strings; got {command!r}."
                )
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "created_at", aware_created_at)
        object.__setattr__(self, "python_version", python_version)
        object.__setattr__(self, "package_version", package_version)
        # Genuine immutability: the pins are stored as a read-only
        # mapping over a private copied dict — never the caller's
        # mapping and never a mutable dict — so no in-place mutation
        # can change the manifest, its hash, or its serialization.
        object.__setattr__(
            self, "dependency_pins", MappingProxyType(dict(dependency_pins))
        )
        object.__setattr__(self, "commands", commands_tuple)

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError(
            f"RunManifest is immutable: field {key!r} cannot be reassigned."
        )

    def __delattr__(self, key: str) -> None:
        raise AttributeError(
            f"RunManifest is immutable: field {key!r} cannot be deleted."
        )

    def to_json(self) -> str:
        """Canonical JSON text: sorted keys, compact separators, the
        original UTC offset preserved in the ISO-8601 instant."""
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "python_version": self.python_version,
            "package_version": self.package_version,
            "dependency_pins": dict(self.dependency_pins),
            "commands": list(self.commands),
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @classmethod
    def from_json(cls, text: str) -> RunManifest:
        """Reconstruct a manifest from canonical JSON text.

        Malformed JSON, a non-object payload, a payload whose keys
        are not exactly the six schema keys, or any duplicated object
        key (top-level or nested) rejects with ``ValueError`` —
        duplicate keys never resolve last-write-wins; a serialized
        ``created_at`` that carries no UTC offset rejects with
        ``NaiveTimestampError`` — never is a default timezone
        silently assumed.
        """
        if not isinstance(text, str):
            raise ValueError(  # noqa: TRY004 — parsing rejects with ValueError
                "RunManifest.from_json requires JSON text; got "
                f"{type(text).__name__}: {text!r}."
            )
        try:
            payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        except _DuplicateKeyError as duplicate:
            raise ValueError(
                "RunManifest.from_json rejects JSON text with duplicate "
                f"object keys — fail-closed: {duplicate}."
            ) from None
        except ValueError as malformed:
            raise ValueError(
                "RunManifest.from_json requires well-formed JSON text; "
                f"parsing failed: {malformed!r}."
            ) from None
        if not isinstance(payload, dict):
            raise ValueError(  # noqa: TRY004 — parsing rejects with ValueError
                "RunManifest.from_json requires a JSON object with exactly "
                f"the six schema keys; got {type(payload).__name__}."
            )
        if set(payload) != set(cls._SCHEMA_KEYS):
            raise ValueError(
                "RunManifest.from_json requires exactly the six schema "
                f"keys {list(cls._SCHEMA_KEYS)}; got {sorted(payload)}."
            )
        raw_created_at = payload["created_at"]
        if not isinstance(raw_created_at, str):
            raise ValueError(  # noqa: TRY004 — parsing rejects with ValueError
                "RunManifest.from_json requires created_at as ISO-8601 "
                f"text; got {raw_created_at!r}."
            )
        try:
            parsed_created_at = datetime.fromisoformat(raw_created_at)
        except ValueError as unparsable:
            raise ValueError(
                "RunManifest.from_json requires created_at as ISO-8601 "
                f"text; got {raw_created_at!r} ({unparsable!r})."
            ) from None
        if parsed_created_at.tzinfo is None or (
            parsed_created_at.utcoffset() is None
        ):
            raise NaiveTimestampError(
                "RunManifest.from_json: created_at is a naive timestamp — "
                "it carries no explicit UTC offset, so it cannot be placed "
                f"on the single timeline; got {raw_created_at!r}. Supply "
                "an offset-carrying ISO-8601 instant instead."
            )
        return cls(
            run_id=payload["run_id"],
            created_at=parsed_created_at,
            python_version=payload["python_version"],
            package_version=payload["package_version"],
            dependency_pins=payload["dependency_pins"],
            commands=payload["commands"],
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RunManifest):
            return NotImplemented
        return self.to_json() == other.to_json()

    def __hash__(self) -> int:
        return hash(self.to_json())

    def __repr__(self) -> str:
        created = self.created_at.isoformat()
        return (
            f"RunManifest(run_id={self.run_id!r}, created_at={created!r})"
        )
