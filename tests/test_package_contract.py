"""Package-contract tests for the PortLearn distribution.

These tests enforce the public package and dependency contract:
distribution/import identity, the frozen development version, empty runtime
and optional dependencies, the bounded hatchling build backend, and the
exact development dependency group.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"

# Frozen package-contract values.
DISTRIBUTION_NAME = "portlearn"
FROZEN_DEVELOPMENT_VERSION = "0.0.1.dev0"
REQUIRES_PYTHON = ">=3.11"
BUILD_REQUIRES = ["hatchling>=1.32.0,<2"]
BUILD_BACKEND = "hatchling.build"
DEVELOPMENT_DEPENDENCIES = {
    "pytest>=9.1.1,<10",
    "ruff>=0.16.6,<0.17",
}
PUBLIC_REPOSITORY_URL = "https://github.com/fmasoudy/PortLearn"
PUBLIC_ISSUES_URL = "https://github.com/fmasoudy/PortLearn/issues"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    if not PYPROJECT_PATH.is_file():
        pytest.fail(
            "pyproject.toml does not exist at the repository root; it is the "
            "single package and tool-configuration authority"
        )
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)


@pytest.fixture(scope="module")
def project_table(pyproject: dict) -> dict:
    return pyproject.get("project", {})


def test_pyproject_declares_distribution_identity(project_table: dict) -> None:
    assert project_table.get("name") == DISTRIBUTION_NAME, (
        "[project] name must be the frozen distribution name "
        f"{DISTRIBUTION_NAME!r}"
    )


def test_pyproject_declares_frozen_development_version(project_table: dict) -> None:
    assert project_table.get("version") == FROZEN_DEVELOPMENT_VERSION, (
        "[project] version must be the frozen PEP 440 development release "
        f"{FROZEN_DEVELOPMENT_VERSION!r}"
    )


def test_pyproject_python_floor_has_no_untested_upper_exclusion(
    project_table: dict,
) -> None:
    declared = project_table.get("requires-python")
    assert declared == REQUIRES_PYTHON, (
        "requires-python must be exactly "
        f"{REQUIRES_PYTHON!r} — minimum 3.11 with no untested upper-version "
        "exclusion"
    )


def test_no_runtime_dependencies(project_table: dict) -> None:
    runtime = project_table.get("dependencies", [])
    assert runtime == [], (
        f"no runtime dependency is declared; found {runtime!r}"
    )


def test_no_optional_capability_groups(project_table: dict) -> None:
    optional = project_table.get("optional-dependencies", {})
    assert optional == {}, (
        f"no optional capability group is declared; found {optional!r}"
    )


def test_project_urls_declare_the_canonical_repository_and_issues(
    project_table: dict,
) -> None:
    """``[project.urls]`` carries the canonical public project links.

    Repository and Issues are the two links a public package needs so
    users can reach source and report defects; they point at the
    canonical public repository.
    """
    urls = project_table.get("urls", {})
    assert urls.get("Repository") == PUBLIC_REPOSITORY_URL, (
        "[project.urls] Repository must be the canonical public "
        f"repository URL {PUBLIC_REPOSITORY_URL!r}; found {urls!r}"
    )
    assert urls.get("Issues") == PUBLIC_ISSUES_URL, (
        "[project.urls] Issues must be the canonical public issue "
        f"tracker URL {PUBLIC_ISSUES_URL!r}; found {urls!r}"
    )


def test_build_backend_is_bounded_hatchling(pyproject: dict) -> None:
    build_system = pyproject.get("build-system", {})
    assert build_system.get("build-backend") == BUILD_BACKEND, (
        f"build-backend must be {BUILD_BACKEND!r}"
    )
    assert sorted(build_system.get("requires", [])) == sorted(BUILD_REQUIRES), (
        "build-system requires must be exactly the bounded hatchling "
        f"requirement {BUILD_REQUIRES!r}"
    )


def test_development_group_is_exactly_pytest_and_ruff(pyproject: dict) -> None:
    groups = pyproject.get("dependency-groups", {})
    declared = {spec for spec in groups.get("dev", [])}
    assert declared == DEVELOPMENT_DEPENDENCIES, (
        "the dev dependency group must be exactly pytest and ruff with their "
        f"frozen bounds; found {sorted(declared)!r}"
    )


def test_package_imports_under_sole_supported_name() -> None:
    import portlearn

    assert portlearn.__name__ == DISTRIBUTION_NAME, (
        "the sole supported import name is 'portlearn'; no 'src' or "
        "'portlearn.src' import path is supported"
    )
    assert sys.modules[DISTRIBUTION_NAME] is portlearn


def test_import_triggers_no_eager_portlearn_submodules() -> None:
    """A fresh import must register exactly the root package module.

    Every ``portlearn`` and ``portlearn.*`` entry is first purged from
    ``sys.modules`` so the import cannot be satisfied from a cached entry.
    The registry is then snapshotted, ``portlearn`` is imported fresh, and
    the import-time registry delta must be exactly ``{'portlearn'}`` — the
    root package alone. Any additional entry would mean the package eagerly
    imports future modules at import time.

    The metadata lookup that derives ``__version__`` is resolved once
    before the snapshot so the delta measures only what the fresh import
    itself registers: CPython loads the stdlib parsing machinery behind
    ``importlib.metadata`` lazily on first use, which would otherwise leak
    unrelated interpreter modules into the delta and make the assertion
    depend on which tests ran earlier rather than on the package contract.
    """
    metadata.version(DISTRIBUTION_NAME)  # settle lazy stdlib imports first

    purged = sorted(
        name
        for name in list(sys.modules)
        if name == DISTRIBUTION_NAME
        or name.startswith(f"{DISTRIBUTION_NAME}.")
    )
    for name in purged:
        del sys.modules[name]

    snapshot = set(sys.modules)
    import portlearn  # noqa: F401 — importing fresh is the act under test

    delta = set(sys.modules) - snapshot
    assert delta == {DISTRIBUTION_NAME}, (
        f"a fresh import of {DISTRIBUTION_NAME!r} must register only the root "
        f"package module in sys.modules; registry delta was {sorted(delta)}"
    )


def test_distribution_metadata_reports_identity_and_version() -> None:
    assert metadata.metadata(DISTRIBUTION_NAME)["Name"] == DISTRIBUTION_NAME
    assert metadata.version(DISTRIBUTION_NAME) == FROZEN_DEVELOPMENT_VERSION, (
        "installed distribution metadata must report the frozen development "
        f"version {FROZEN_DEVELOPMENT_VERSION!r}"
    )


def test_package_dunder_version_matches_distribution_metadata() -> None:
    import portlearn

    declared = metadata.version(DISTRIBUTION_NAME)
    assert portlearn.__version__ == declared, (
        "portlearn.__version__ must reflect installed distribution metadata — "
        "pyproject.toml is the single version authority, so the dunder must "
        "never drift from it"
    )


def test_dunder_version_is_derived_not_hard_coded() -> None:
    """The ``__version__`` dunder must be a derivation, not a second authority.

    pyproject.toml is the single version authority. A hard-coded literal in
    ``src/portlearn/__init__.py`` — even one whose value currently equals
    the frozen ``0.0.1.dev0`` — is an independent duplicate authority that
    silently drifts on the first version bump. The frozen version must
    therefore never appear as a literal anywhere in the module, and the
    ``__version__`` assignment must be a call expression deriving the value
    at import time from installed distribution metadata. The sole literal
    permitted inside that call is the distribution name, the frozen lookup
    key — never a version.
    """
    init_path = REPOSITORY_ROOT / "src" / "portlearn" / "__init__.py"
    assert init_path.is_file(), (
        f"package identity module missing: {init_path}"
    )
    tree = ast.parse(init_path.read_text(encoding="utf-8"))

    frozen_literals = [
        part.value
        for part in ast.walk(tree)
        if isinstance(part, ast.Constant)
        and isinstance(part.value, str)
        and part.value == FROZEN_DEVELOPMENT_VERSION
    ]
    assert not frozen_literals, (
        "the frozen development version must not be hard-coded in "
        "src/portlearn/__init__.py: pyproject.toml is the single version "
        "authority, so a literal duplicate drifts on the first version bump"
    )

    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1, (
        "src/portlearn/__init__.py must assign __version__ exactly once; "
        f"found {len(assignments)} assignments"
    )

    value = assignments[0].value
    assert isinstance(value, ast.Call), (
        "__version__ must be assigned a call expression deriving the value "
        "from installed distribution metadata (e.g. "
        "importlib.metadata.version('portlearn')); a hard-coded literal is a "
        "second, drifting version authority"
    )
    foreign_literals = [
        part.value
        for part in ast.walk(value)
        if isinstance(part, ast.Constant)
        and isinstance(part.value, str)
        and part.value != DISTRIBUTION_NAME
    ]
    assert not foreign_literals, (
        "the __version__ derivation may carry only the distribution name as "
        "its literal (the frozen lookup key); found non-name literals "
        f"{foreign_literals!r}"
    )


def test_distribution_declares_no_runtime_requirements() -> None:
    requires = metadata.requires(DISTRIBUTION_NAME)
    assert requires is None, (
        f"installed distribution must declare no runtime requirement; found "
        f"{requires!r}"
    )


# --- Wheel-content verification: the complete package surface ---

VERIFICATION_SCRIPT = REPOSITORY_ROOT / "scripts" / "verify_built_wheel.py"
PACKAGE_SOURCE = REPOSITORY_ROOT / "src" / DISTRIBUTION_NAME


def _verification_module() -> dict:
    """Execute ``scripts/verify_built_wheel.py`` offline and return it.

    The script only acts under ``if __name__ == "__main__"``, so loading
    it under a different module name runs no verification and creates
    nothing; its constants and helpers become directly assertable.
    """
    import importlib.util

    assert VERIFICATION_SCRIPT.is_file(), (
        f"wheel verification script missing: {VERIFICATION_SCRIPT}"
    )
    spec = importlib.util.spec_from_file_location(
        "verify_built_wheel_under_test", VERIFICATION_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return vars(module)


def _complete_wheel_surface() -> set[str]:
    """The complete package surface a built wheel must carry.

    Derived from the source tree itself — every Python module plus the
    ``py.typed`` marker — so the expectation cannot silently drift when
    a module is added.
    """
    assert PACKAGE_SOURCE.is_dir(), (
        f"package source directory missing: {PACKAGE_SOURCE}"
    )
    surface = {
        f"{DISTRIBUTION_NAME}/{path.name}"
        for path in PACKAGE_SOURCE.iterdir()
        if path.suffix == ".py" or path.name == "py.typed"
    }
    assert surface, "the package source tree unexpectedly declares no files"
    return surface


def test_required_wheel_members_cover_the_complete_package_surface() -> None:
    """The wheel check must require the whole package, not two files.

    A check requiring only ``__init__.py`` and ``py.typed`` lets a wheel
    silently missing any contract module (interfaces, leakage, manifest,
    observations, timing) pass verification. The required-member tuple
    must equal the complete source surface.
    """
    module = _verification_module()
    required = set(module["REQUIRED_WHEEL_MEMBERS"])
    expected = _complete_wheel_surface()
    assert required == expected, (
        "REQUIRED_WHEEL_MEMBERS must require exactly the complete package "
        "surface (every src module plus py.typed); missing "
        f"{sorted(expected - required)}, unexpected "
        f"{sorted(required - expected)}"
    )


def test_wheel_missing_a_contract_module_fails_verification(
    tmp_path: Path,
) -> None:
    """A wheel that omits one contract module must fail closed.

    A wheel carrying ``__init__.py`` and ``py.typed`` but no
    ``interfaces.py`` — the gap a two-file check cannot see — must make
    ``assert_wheel_members`` exit nonzero naming the missing member.
    """
    module = _verification_module()
    incomplete = tmp_path / "portlearn-0.0.1-py3-none-any.whl"
    import zipfile

    with zipfile.ZipFile(incomplete, "w") as archive:
        archive.writestr(
            f"{DISTRIBUTION_NAME}/__init__.py",
            '__version__ = "0.0.1.dev0"\n',
        )
        archive.writestr(f"{DISTRIBUTION_NAME}/py.typed", "")
        for member in (
            "leakage.py",
            "manifest.py",
            "observations.py",
            "timing.py",
        ):
            archive.writestr(f"{DISTRIBUTION_NAME}/{member}", "")
    with pytest.raises(SystemExit) as caught:
        module["assert_wheel_members"](incomplete)
    message = str(caught.value)
    assert "missing required members" in message, (
        f"the failure must name the missing members; got {message!r}"
    )
    assert f"{DISTRIBUTION_NAME}/interfaces.py" in message, (
        "the omitted contract module must be named in the failure"
    )


def test_complete_wheel_passes_member_verification(tmp_path: Path) -> None:
    """Control: a wheel carrying the full surface still passes.

    The strengthened check must not over-reject: a wheel containing
    every required member validates exactly as before.
    """
    module = _verification_module()
    complete = tmp_path / "portlearn-0.0.1-py3-none-any.whl"
    import zipfile

    with zipfile.ZipFile(complete, "w") as archive:
        for member in module["REQUIRED_WHEEL_MEMBERS"]:
            archive.writestr(member, "")
    module["assert_wheel_members"](complete)  # must not raise
