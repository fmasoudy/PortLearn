"""Structural public-surface contracts for the data-subsystem namespace.

These nodes pin the cleaned public layout of the data subsystem: the
public dataset module beside its package re-export, the ingestion
chokepoint's home inside the data package, the provider-neutral record
primitives in one shared records module, the FF-only alias surface
living on the FF provider facade, the lazy package-import law, the exact
on-disk file layout, and the absence of internal development terminology
in the affected production sources (public names, constants, messages,
docstrings, and comments).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "portlearn"
DATA_ROOT = PACKAGE_ROOT / "data"
ADAPTERS_ROOT = DATA_ROOT / "adapters"

#: Modules the cleaned namespace must no longer expose (pre-v0.1.0, no
#: compatibility re-exports).
REMOVED_MODULES = (
    "portlearn.data._dataset",
    "portlearn.data._aliases",
    "portlearn.data.adapters._unqualified",
    "portlearn.ingestion",
)

#: Modules the cleaned namespace must expose.
REQUIRED_MODULES = (
    "portlearn.data.dataset",
    "portlearn.data._records",
    "portlearn.data.ingestion",
)

#: Files that must not exist after the cleanup.
REMOVED_FILES = (
    DATA_ROOT / "_dataset.py",
    DATA_ROOT / "_aliases.py",
    ADAPTERS_ROOT / "_unqualified.py",
    PACKAGE_ROOT / "ingestion.py",
)

#: Files that must exist after the cleanup.
REQUIRED_FILES = (
    DATA_ROOT / "dataset.py",
    DATA_ROOT / "_records.py",
    DATA_ROOT / "ingestion.py",
    DATA_ROOT / "__init__.py",
    DATA_ROOT / "fama_french.py",
    DATA_ROOT / "fred.py",
    ADAPTERS_ROOT / "__init__.py",
    ADAPTERS_ROOT / "ff.py",
    ADAPTERS_ROOT / "fred.py",
)

#: The affected production sources (post-cleanup paths, plus their
#: pre-cleanup predecessors so the terminology scan has teeth before the
#: move lands).
AFFECTED_SOURCES = (
    DATA_ROOT / "__init__.py",
    DATA_ROOT / "dataset.py",
    DATA_ROOT / "_records.py",
    DATA_ROOT / "ingestion.py",
    DATA_ROOT / "fama_french.py",
    DATA_ROOT / "fred.py",
    ADAPTERS_ROOT / "__init__.py",
    ADAPTERS_ROOT / "ff.py",
    ADAPTERS_ROOT / "fred.py",
    DATA_ROOT / "_dataset.py",
    DATA_ROOT / "_aliases.py",
    ADAPTERS_ROOT / "_unqualified.py",
    PACKAGE_ROOT / "ingestion.py",
)

#: Internal governance terminology that must not appear in the affected
#: production sources (researcher-facing surfaces use domain/software
#: terms instead).  Careful word boundaries keep the scan contextual:
#: only whole-word matches count.
PROHIBITED_TERMS = re.compile(
    r"ADR-\d+"
    r"|\bGate[- ]?[12]\b"
    r"|\breview gate\b"
    r"|\bfloor\w*\b"
    r"|\bRED\b"
    r"|\bGREEN\b"
    r"|\bgovernance\b"
    r"|\bowner\b"
    r"|\bauthorization\b"
    r"|\bmilestones?\b"
    r"|\bM\d+\.\d+\b"
    r"|\bsuccessor revision\b"
    r"|\bapproved\b"
    r"|\bfrozen contract\b"
    r"|\bL\d+\b"
    r"|§"
)

_NAMESPACE_PROBE = f"""\
import importlib
import json
import sys

MODULES = {list(REMOVED_MODULES + REQUIRED_MODULES)!r}


def probe(name):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        return "absent"
    return "present"


import portlearn.data

state = {{
    "lazy": {{
        "pandas": "pandas" in sys.modules,
        "registered": sorted(
            name for name in sys.modules if name.startswith("portlearn")
        ),
    }},
    "modules": {{name: probe(name) for name in MODULES}},
}}
print(json.dumps(state))
"""


def _namespace_state() -> dict:
    """Probe module presence and the lazy-import law in a fresh process."""
    completed = subprocess.run(
        [sys.executable, "-c", _NAMESPACE_PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"namespace probe failed: {completed.stderr.strip()}"
    )
    state = json.loads(completed.stdout)
    assert isinstance(state, dict)
    return state


def test_module_layout_is_exact() -> None:
    """The cleaned namespace: new modules present, old modules absent."""
    state = _namespace_state()
    modules = state["modules"]
    missing = [name for name in REQUIRED_MODULES if modules[name] != "present"]
    assert not missing, f"required modules absent: {missing}"
    lingering = [name for name in REMOVED_MODULES if modules[name] != "absent"]
    assert not lingering, (
        f"superseded modules still importable (no compatibility path is "
        f"permitted pre-v0.1.0): {lingering}"
    )


def test_lazy_package_import_law_holds() -> None:
    """``import portlearn.data`` registers the package alone — no pandas,
    no provider facade, no dataset module."""
    lazy = _namespace_state()["lazy"]
    assert lazy["pandas"] is False, (
        "importing portlearn.data must never import pandas eagerly"
    )
    assert lazy["registered"] == ["portlearn", "portlearn.data"], (
        f"import portlearn.data registered extra modules: "
        f"{lazy['registered']}"
    )


def test_research_dataset_imports_through_both_public_paths() -> None:
    """``from portlearn.data import ResearchDataset`` and
    ``from portlearn.data.dataset import ResearchDataset`` resolve to the
    same public class."""
    from portlearn.data import ResearchDataset as via_package
    from portlearn.data.dataset import ResearchDataset as via_module

    assert via_package is via_module
    assert via_module.__module__ == "portlearn.data.dataset"
    assert hasattr(via_module, "to_pandas")


def test_ingestion_chokepoint_lives_in_the_data_package() -> None:
    """The declared-schema ingestion surface is importable from its sole
    clean namespace, ``portlearn.data.ingestion``."""
    from portlearn.data.ingestion import (
        AvailabilityPolicy,
        DeclaredTableSchema,
        MissingParquetExtraError,
        SourceProvenance,
        from_csv,
        from_dataframe,
        from_parquet,
        from_records,
    )

    assert callable(from_csv)
    assert callable(from_dataframe)
    assert callable(from_records)
    assert callable(from_parquet)
    assert issubclass(MissingParquetExtraError, Exception)
    assert AvailabilityPolicy.same_instant() is not None
    assert DeclaredTableSchema is not None
    assert SourceProvenance is not None


def test_provider_neutral_records_live_in_the_shared_records_module() -> None:
    """``PeriodKeyObservation`` and ``RetrievalProvenance`` are defined in
    ``portlearn.data._records`` and re-exported unchanged by both provider
    adapters."""
    from portlearn.data._records import (
        PeriodKeyObservation,
        RetrievalProvenance,
    )
    from portlearn.data.adapters import ff, fred

    assert PeriodKeyObservation.__module__ == "portlearn.data._records"
    assert RetrievalProvenance.__module__ == "portlearn.data._records"
    assert ff.PeriodKeyObservation is PeriodKeyObservation
    assert fred.PeriodKeyObservation is PeriodKeyObservation
    assert issubclass(ff.FFRetrievalProvenance, RetrievalProvenance)
    assert issubclass(fred.FREDRetrievalProvenance, RetrievalProvenance)


def test_ff_alias_surface_still_selects_before_any_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The FF alias rules stay available on the FF provider surface and
    keep rejecting before any network is touched."""
    import portlearn as pl

    ff = import_module("portlearn.data.adapters.ff")

    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("no network may be touched on rejection")

    monkeypatch.setattr(ff, "urlopen", _explode)
    with pytest.raises(ValueError):
        pl.data.fama_french.load("not_a_research_alias", frequency="monthly")
    with pytest.raises(ValueError) as ambiguity:
        pl.data.fama_french.load("factors")
    message = str(ambiguity.value)
    assert "factors_monthly_csv" in message
    assert "factors_daily_csv" in message


def test_file_layout_is_exact() -> None:
    """The moved/removed files on disk: new homes present, old homes and
    the generic alias module gone; the adapters package holds exactly the
    three provider-surface files."""
    missing = [path for path in REQUIRED_FILES if not path.is_file()]
    assert not missing, f"required files absent: {missing}"
    lingering = [path for path in REMOVED_FILES if path.is_file()]
    assert not lingering, f"superseded files still on disk: {lingering}"
    adapter_files = {
        path.name
        for path in ADAPTERS_ROOT.iterdir()
        if path.suffix == ".py"
    }
    assert adapter_files == {"__init__.py", "ff.py", "fred.py"}, (
        f"the adapters package must hold exactly the provider surface; "
        f"found {sorted(adapter_files)}"
    )


def test_affected_sources_carry_no_internal_governance_terms() -> None:
    """The affected production sources speak domain/software language
    only: no internal development terminology in public names, constants,
    messages, docstrings, or comments."""
    sources = [path for path in AFFECTED_SOURCES if path.is_file()]
    assert sources, "no affected sources found to scan"
    offenders: dict[str, list[str]] = {}
    for path in sources:
        hits = PROHIBITED_TERMS.findall(path.read_text(encoding="utf-8"))
        if hits:
            offenders[path.relative_to(REPOSITORY_ROOT).as_posix()] = hits
    assert not offenders, (
        f"internal development terminology found in affected production "
        f"sources: {offenders}"
    )
