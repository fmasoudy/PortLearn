"""Shared fixture builders and probe helpers for the diagnostics behavior suites.

Every dataset here is built through the public canonical constructors
(``UnqualifiedDataset`` / ``QualifiedDataset``) over either synthetic
records or the committed synthetic provider replicas under
``tests/adapters/fixtures`` — network-free and deterministic.  The
helpers in this module deliberately contain **no** import of
``portlearn.data.diagnostics``: each behavior floor imports that module
inside the test body so a missing implementation reports as the causal
failure of that floor instead of aborting collection.

Research datasets only; NOT investable.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from portlearn.data._records import PeriodKeyObservation
from portlearn.data.dataset import QualifiedDataset, UnqualifiedDataset
from portlearn.observations import TimedObservation

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "portlearn"
DATA_PACKAGE_ROOT = PACKAGE_ROOT / "data"
DIAGNOSTICS_PACKAGE_ROOT = DATA_PACKAGE_ROOT / "diagnostics"
FIXTURES = REPOSITORY_ROOT / "tests" / "adapters" / "fixtures"

FRED_MONTHLY_JSON = FIXTURES / "fred" / "obs_synthcpim_monthly.json"
FRED_META_JSON = FIXTURES / "fred" / "meta_synthcpim.json"

SYNTH_SOURCE_BYTES = b"portlearn-diagnostics-synthetic-retained-bytes"
SYNTH_UNITS = "index"
SYNTH_FREQUENCY = "Monthly"

#: The exact fixed footnote every plot spec must carry.
PLOT_FOOTNOTE = (
    "Descriptive summary of retained records; not decision-time eligibility."
)

#: The exact key-basis disclosure names for the two sealed states.
UNQUALIFIED_KEY_BASIS = "period_key (verbatim provider label)"
QUALIFIED_KEY_BASIS = "observation_time (normalized identity)"

#: The exact five plot kinds.
PLOT_KINDS = ("describe", "correlation", "coverage", "missingness", "series")

#: Imports a diagnostics module may never contain at any runtime scope.
FORBIDDEN_RUNTIME_IMPORT_ROOTS = ("pandas", "numpy")

#: Imports a diagnostics module may never contain anywhere (network,
#: clock, randomness, filesystem, subprocess surfaces).
FORBIDDEN_IMPORT_ROOTS_ANYWHERE = (
    "socket",
    "urllib",
    "requests",
    "http",
    "random",
    "time",
    "secrets",
    "uuid",
    "subprocess",
    "pathlib",
)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def import_diagnostics() -> Any:
    """Import the diagnostics package inside the calling test (causal RED)."""
    return importlib.import_module("portlearn.data.diagnostics")


def require_diagnostics_package() -> Path:
    """Fail with the causal absence message unless the package exists."""
    if not DIAGNOSTICS_PACKAGE_ROOT.is_dir():
        raise AssertionError(
            "the diagnostics package does not exist at "
            f"{DIAGNOSTICS_PACKAGE_ROOT}: the five-block public surface "
            "(describe/correlation/coverage/missingness/plot) is absent, so "
            "no diagnostics behavior contract can execute"
        )
    return DIAGNOSTICS_PACKAGE_ROOT


def package_py_files() -> list[Path]:
    return sorted(
        path for path in require_diagnostics_package().rglob("*.py") if path.is_file()
    )


def _refuse_decoder(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError(
        "qualification is outside the diagnostics contract; the synthetic "
        "decoder must never be invoked by any diagnostics behavior test"
    )


@dataclass(frozen=True)
class SyntheticRetrievalProvenance:
    """Typed retrieval facts: retrieval-only, no availability field."""

    content_sha256: str
    units: str
    frequency: str


@dataclass(frozen=True)
class SyntheticQualifiedProvenance:
    """Typed qualified facts: carries the availability declaration."""

    availability: Any
    content_sha256: str
    units: str
    frequency: str


def period_record(
    series_id: str, period_key: str, value: float
) -> PeriodKeyObservation:
    return PeriodKeyObservation(
        series_id=series_id, period_key=period_key, value=float(value)
    )


def month_key(year: int, month: int) -> str:
    """A Fama-French-style verbatim provider period label."""
    return f"{year:04d}{month:02d}"


def fred_label(year: int, month: int) -> str:
    """A FRED-style verbatim provider period label."""
    return f"{year:04d}-{month:02d}"


def instant(year: int, month: int, day: int = 1, hour: int = 12) -> datetime:
    """A normalized aware observation-time identity (UTC)."""
    return datetime(year, month, day, hour, 0, 0, tzinfo=UTC)


def timed_record(
    series_id: str,
    observation: datetime,
    value: Any,
    available: datetime | None = None,
) -> TimedObservation:
    """One qualified record; ``available`` defaults to the observation."""
    return TimedObservation(
        series_id=series_id,
        observation_time=observation,
        available_time=observation if available is None else available,
        value=value,
    )


def unqualified_dataset(
    records: tuple[PeriodKeyObservation, ...] | list[PeriodKeyObservation],
    *,
    provider: str = "synthetic",
    name: str = "SYNTH",
    provider_dataset_id: str = "SYNTH",
    units: str = SYNTH_UNITS,
    frequency: str = SYNTH_FREQUENCY,
    source_bytes: bytes = SYNTH_SOURCE_BYTES,
    adapter_version: str = "synthetic-test-adapter/1.0.0",
) -> UnqualifiedDataset:
    """An UNQUALIFIED dataset over synthetic period-key records."""
    return UnqualifiedDataset(
        provider=provider,
        name=name,
        provider_dataset_id=provider_dataset_id,
        adapter_version=adapter_version,
        source_bytes=source_bytes,
        auxiliary_bytes=None,
        records=tuple(records),
        retrieval_provenance=SyntheticRetrievalProvenance(
            content_sha256=sha256_hex(source_bytes),
            units=units,
            frequency=frequency,
        ),
        decoder=_refuse_decoder,
    )


def qualified_dataset(
    records: tuple[TimedObservation, ...] | list[TimedObservation],
    *,
    availability: Any,
    provider: str = "synthetic",
    name: str = "SYNTHQ",
    provider_dataset_id: str = "SYNTHQ",
    units: str = SYNTH_UNITS,
    frequency: str = SYNTH_FREQUENCY,
    source_bytes: bytes = SYNTH_SOURCE_BYTES,
    adapter_version: str = "synthetic-test-adapter/1.0.0",
) -> QualifiedDataset:
    """A QUALIFIED dataset over synthetic timed observations."""
    return QualifiedDataset(
        provider=provider,
        name=name,
        provider_dataset_id=provider_dataset_id,
        adapter_version=adapter_version,
        availability=availability,
        source_bytes=source_bytes,
        auxiliary_bytes=None,
        records=tuple(records),
        retrieval_provenance=SyntheticRetrievalProvenance(
            content_sha256=sha256_hex(source_bytes),
            units=units,
            frequency=frequency,
        ),
        qualified_provenance=SyntheticQualifiedProvenance(
            availability=availability,
            content_sha256=sha256_hex(source_bytes),
            units=units,
            frequency=frequency,
        ),
    )


def availability_policy() -> Any:
    """A declared availability policy (same-instant), never defaulted."""
    from portlearn.data.ingestion import AvailabilityPolicy

    return AvailabilityPolicy.same_instant()


def fred_snapshot_dataset() -> UnqualifiedDataset:
    """The committed synthetic FRED replica, decoded through the frozen
    provider decoder: 36 table rows of which two carry the provider
    missing sentinel (``"."``), so exactly 34 records are retained."""
    from portlearn.data.adapters import fred as fred_adapter

    data = FRED_MONTHLY_JSON.read_bytes()
    metadata = FRED_META_JSON.read_bytes()
    records, provenance = fred_adapter.decode_unqualified(
        data,
        "SYNTHCPIM",
        frequency="Monthly",
        metadata_bytes=metadata,
    )
    return UnqualifiedDataset(
        provider="fred",
        name="SYNTHCPIM",
        provider_dataset_id="SYNTHCPIM",
        adapter_version=provenance.adapter_version,
        source_bytes=data,
        auxiliary_bytes=metadata,
        records=tuple(records),
        retrieval_provenance=provenance,
        decoder=fred_adapter.decode,
    )


def fred_retained_values() -> list[float]:
    """The retained numeric values of the FRED replica, derived from the
    provider payload itself (an expectation independent of the dataset):
    every table row whose value cell is not the missing sentinel."""
    import json

    payload = json.loads(FRED_MONTHLY_JSON.read_text(encoding="utf-8"))
    return [
        float(row["value"]) for row in payload["observations"] if row["value"] != "."
    ]


def permuted(dataset: Any, seed: int) -> Any:
    """The same dataset content with record order permuted (same bytes)."""
    import random

    records = list(dataset.records)
    random.Random(seed).shuffle(records)
    if dataset.availability_state == "UNQUALIFIED":
        return unqualified_dataset(
            records,
            provider=dataset.provider,
            name=dataset.name,
            provider_dataset_id=dataset.provider_dataset_id,
            units=dataset.units,
            frequency=dataset.frequency,
            source_bytes=dataset.source_bytes,
            adapter_version=dataset.adapter_version,
        )
    return qualified_dataset(
        records,
        availability=dataset.availability,
        provider=dataset.provider,
        name=dataset.name,
        provider_dataset_id=dataset.provider_dataset_id,
        units=dataset.units,
        frequency=dataset.frequency,
        source_bytes=dataset.source_bytes,
        adapter_version=dataset.adapter_version,
    )


# --------------------------------------------------------------------------
# Structural probes over stdlib report values
# --------------------------------------------------------------------------


def walk_values(value: Any, seen: set[int] | None = None) -> Iterator[Any]:
    """Yield every nested value of a stdlib report structure."""
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    yield value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from walk_values(getattr(value, field.name), seen)
    elif isinstance(value, (MappingProxyType, dict)):
        for item in value.values():
            yield from walk_values(item, seen)
    elif isinstance(value, (tuple, list, frozenset, set)):
        for item in value:
            yield from walk_values(item, seen)


def find_strings(value: Any) -> set[str]:
    return {item for item in walk_values(value) if isinstance(item, str)}


def carries_value(value: Any, target: float) -> bool:
    """Whether any numeric leaf equals ``target`` (bools never match ints)."""
    for item in walk_values(value):
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)) and item == target:
            return True
    return False


def field_names(value: Any) -> set[str]:
    """Every reachable field/attribute name of a report structure."""
    names: set[str] = set()
    for item in walk_values(value):
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            names.update(field.name for field in dataclasses.fields(item))
        elif isinstance(item, (MappingProxyType, dict)):
            names.update(str(key) for key in item)
    return names


def first_field_name(report: Any) -> str:
    if dataclasses.is_dataclass(report):
        return dataclasses.fields(report)[0].name
    public = [name for name in dir(report) if not name.startswith("_")]
    return public[0]


def assert_no_foreign_types(report: Any, forbidden_prefixes: tuple[str, ...]) -> None:
    for item in walk_values(report):
        module = type(item).__module__
        assert not module.startswith(forbidden_prefixes), (
            f"a {type(item).__name__} from {module!r} crossed the "
            f"diagnostics boundary: reports are stdlib types only"
        )


def assert_no_instances(report: Any, banned_types: tuple[type, ...]) -> None:
    for item in walk_values(report):
        assert not isinstance(item, banned_types), (
            f"a mutable {type(item).__name__} appears in a report: "
            "reports expose no mutation surface"
        )


def has_marker(report: Any, marker: str) -> bool:
    """Whether any string leaf equals or contains the marker."""
    return any(marker in leaf for leaf in find_strings(report))


# --------------------------------------------------------------------------
# Source-scan probes (AST)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportFact:
    root: str
    scope: str  # "module" | "branch" | "function"
    type_checking_guarded: bool


def scan_imports(path: Path) -> list[ImportFact]:
    """Classify every import in one module by scope and TYPE_CHECKING guard."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    facts: list[ImportFact] = []

    def roots(node: ast.Import | ast.ImportFrom) -> list[str]:
        if isinstance(node, ast.Import):
            return [alias.name.split(".")[0] for alias in node.names]
        module = node.module or ""
        return (
            [module.split(".")[0]]
            if module
            else ([alias.name.split(".")[0] for alias in node.names])
        )

    def visit(nodes: list[ast.stmt], scope: str, guarded: bool) -> None:
        for node in nodes:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for root in roots(node):
                    facts.append(
                        ImportFact(
                            root=root,
                            scope=scope,
                            type_checking_guarded=guarded,
                        )
                    )
            elif isinstance(node, ast.If):
                test = node.test
                is_type_checking = (
                    isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"
                ) or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")
                nested_scope = (
                    scope
                    if scope == "function"
                    else ("module" if isinstance(node, ast.Module) else scope)
                )
                # An `if TYPE_CHECKING:` block at module level keeps module
                # scope but marks every import inside it guarded; any other
                # branch keeps plain branch scope.
                visit(
                    node.body,
                    "module" if scope == "module" else nested_scope,
                    guarded or is_type_checking,
                )
                visit(
                    node.orelse,
                    "module" if scope == "module" else nested_scope,
                    guarded,
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(node.body, "function", guarded)
            elif isinstance(node, ast.ClassDef):
                visit(node.body, "class", guarded)
            elif isinstance(node, ast.Try):
                visit(node.body, scope, guarded)
                for handler in node.handlers:
                    visit(handler.body, scope, guarded)
                visit(node.orelse, scope, guarded)
                visit(node.finalbody, scope, guarded)

    visit(tree.body, "module", False)
    return facts


def module_scope_runtime_imports(path: Path) -> list[ImportFact]:
    return [
        fact
        for fact in scan_imports(path)
        if fact.scope == "module" and not fact.type_checking_guarded
    ]


def any_scope_runtime_imports(path: Path) -> list[ImportFact]:
    return [fact for fact in scan_imports(path) if not fact.type_checking_guarded]


def function_scope_imports(path: Path, root: str) -> list[ast.AST]:
    """Every import node for ``root`` that sits inside a function body."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[ast.AST] = []

    def matches(node: ast.Import | ast.ImportFrom) -> bool:
        if isinstance(node, ast.Import):
            return any(alias.name.split(".")[0] == root for alias in node.names)
        return bool(node.module) and node.module.split(".")[0] == root

    def visit(nodes: list[ast.stmt], in_function: bool) -> None:
        for node in nodes:
            if isinstance(node, (ast.Import, ast.ImportFrom)) and in_function:
                if matches(node):
                    hits.append(node)
            elif isinstance(node, ast.If):
                visit(node.body, in_function)
                visit(node.orelse, in_function)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(node.body, True)
            elif isinstance(node, ast.ClassDef):
                visit(node.body, in_function)
            elif isinstance(node, ast.Try):
                visit(node.body, in_function)
                for handler in node.handlers:
                    visit(handler.body, in_function)
                visit(node.orelse, in_function)
                visit(node.finalbody, in_function)

    visit(tree.body, False)
    return hits


def source_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def stdlib_modules() -> frozenset[str]:
    return sys.stdlib_module_names
