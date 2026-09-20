"""Package-contract tests for the PortLearn distribution.

These tests enforce the public package and dependency contract:
distribution/import identity, the frozen development version, the
pandas core dependency with exactly the one guarded optional capability
group, the bounded
hatchling build backend, the exact development dependency group, and
the absence of the undeclared top-level ``portlearn.adapters``
namespace from the public package surface.
"""

from __future__ import annotations

import ast
import inspect
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"

# Frozen package-contract values.
DISTRIBUTION_NAME = "portlearn"
FROZEN_DEVELOPMENT_VERSION = "0.0.1.dev2"
REQUIRES_PYTHON = ">=3.11"
BUILD_REQUIRES = ["hatchling>=1.32.0,<2"]
BUILD_BACKEND = "hatchling.build"
DEVELOPMENT_DEPENDENCIES = {
    "pytest>=9.1.1,<10",
    "ruff>=0.16.6,<0.17",
    "matplotlib>=3.9.2",
}
PUBLIC_REPOSITORY_URL = "https://github.com/fmasoudy/PortLearn"
PUBLIC_ISSUES_URL = "https://github.com/fmasoudy/PortLearn/issues"

# The sole core runtime dependency: pandas with an evidence-supported
# floor and no upper cap — the first and only core runtime dependency
# first and only core runtime dependency the distribution declares.
PANDAS_CORE_REQUIREMENT = "pandas>=2.2.3"


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
        f"[project] name must be the frozen distribution name {DISTRIBUTION_NAME!r}"
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


def test_package_contract_pins_core_pandas_and_frozen_parquet_extra(
    project_table: dict,
) -> None:
    """Core runtime dependencies are exactly ``["pandas>=2.2.3"]``
    (evidence-supported floor, no upper cap) and the parquet extra
    remains the frozen guarded pin, unchanged by the plot-extra
    amendment; the exact whole-table shape — including the ``plot``
    extra — is owned by
    ``test_pyproject_declares_plot_extra_with_matplotlib_floor_not_core``
    (floor 29) so the two oracles can never contradict each other."""
    runtime = project_table.get("dependencies", [])
    assert runtime == [PANDAS_CORE_REQUIREMENT], (
        "the sole core runtime dependency is pandas with the "
        f"evidence-supported floor and no upper cap; expected "
        f"['{PANDAS_CORE_REQUIREMENT}'], found {runtime!r}"
    )

    parquet_pin = "pyarrow>=21.0.0,<26"
    optional = project_table.get("optional-dependencies", {})
    assert optional.get("parquet") == [parquet_pin], (
        "the guarded parquet extra must remain the frozen pin "
        f"['{parquet_pin}'] through later amendments; found "
        f"{optional.get('parquet')!r}"
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


def test_pyproject_declares_plot_extra_with_matplotlib_floor_not_core(
    project_table: dict,
) -> None:
    """The optional-dependency table carries exactly the two guarded
    capability groups — the frozen parquet extra and the new ``plot``
    extra with the ``matplotlib>=3.9.2`` floor — while the core runtime
    dependency set remains pandas alone: matplotlib is never core
    (rendering is an optional capability behind ``PlotSpec.render()``,
    the sole public matplotlib boundary)."""
    runtime = project_table.get("dependencies", [])
    assert runtime == [PANDAS_CORE_REQUIREMENT], (
        "matplotlib must never join the core runtime dependencies: the "
        "sole core dependency remains pandas; found "
        f"{runtime!r}"
    )

    parquet_pin = "pyarrow>=21.0.0,<26"
    plot_pin = "matplotlib>=3.9.2"
    optional = project_table.get("optional-dependencies", {})
    assert optional == {"parquet": [parquet_pin], "plot": [plot_pin]}, (
        "the optional-dependency table must be exactly the guarded "
        f"parquet extra {{'parquet': ['{parquet_pin}']}} and the plot "
        f"extra {{'plot': ['{plot_pin}']}} with the evidence-supported "
        f"matplotlib floor; found {optional!r}"
    )


PUBLIC_DIAGNOSTICS_PACKAGE_MEMBER = f"{DISTRIBUTION_NAME}/data/diagnostics/__init__.py"

DIAGNOSTICS_PUBLIC_BLOCKS = (
    "describe",
    "correlation",
    "coverage",
    "missingness",
    "plot",
)


def test_wheel_verifier_pins_the_diagnostics_subpackage_floor() -> None:
    """The wheel-verification floor pins the public diagnostics surface
    only: the subpackage marker a wheel silently omitting
    ``portlearn.data.diagnostics`` would lack, and never a private
    per-block filename — exact private filenames are implementation
    detail and expressly not frozen, so private modules join through
    the recursive derivation alone."""
    module = _verification_module()
    floor = set(module["REQUIRED_FLOOR_MEMBERS"])
    assert PUBLIC_DIAGNOSTICS_PACKAGE_MEMBER in floor, (
        "REQUIRED_FLOOR_MEMBERS must pin the public diagnostics package "
        f"marker {PUBLIC_DIAGNOSTICS_PACKAGE_MEMBER!r} so a wheel silently "
        "omitting the diagnostics surface fails verification; floor "
        f"currently holds {sorted(floor)}"
    )

    diagnostics_entries = {
        member
        for member in floor
        if member.startswith(f"{DISTRIBUTION_NAME}/data/diagnostics/")
    }
    assert diagnostics_entries == {PUBLIC_DIAGNOSTICS_PACKAGE_MEMBER}, (
        "the frozen floor must not pin private diagnostics filenames "
        "(exact private filenames are implementation detail); "
        f"only the public package marker may be frozen, found "
        f"{sorted(diagnostics_entries)}"
    )

    # The installed-wheel behavior probe verifies the public import
    # surface only: lazy diagnostics exposure, exactly the five callable
    # blocks, matplotlib absent after the import, the pandas proof
    # preserved, and a missing-extra render refusal naming portlearn[plot].
    probe = module["DATA_SUBSYSTEM_PROBE"]
    assert '"portlearn.data.diagnostics" not in sys.modules' in probe, (
        "the data-subsystem probe must prove a bare import of "
        "portlearn.data does not register the diagnostics subpackage "
        "(lazy exposure)"
    )
    assert '"matplotlib" not in sys.modules' in probe, (
        "the data-subsystem probe must assert matplotlib is absent after "
        "importing the installed diagnostics surface (the "
        "plot extra is genuinely optional)"
    )
    for block in DIAGNOSTICS_PUBLIC_BLOCKS:
        assert f'"{block}"' in probe, (
            f"the data-subsystem probe must verify the public block "
            f"{block!r} is callable on the installed diagnostics surface"
        )
    assert "callable(getattr(diagnostics" in probe, (
        "the probe must verify the five blocks through public attribute "
        "access on the diagnostics module, never through private module "
        "paths"
    )
    assert f"{DISTRIBUTION_NAME}.data.diagnostics." not in probe, (
        "the probe must make no private diagnostics module-path "
        "assumption (private filenames are not frozen)"
    )
    assert '"pandas" in sys.modules' in probe, (
        "the pandas proof (importing portlearn.data.dataset "
        "triggers the core pandas import) must be preserved in the probe"
    )
    assert "to_pandas" in probe, (
        "the ResearchDataset.to_pandas presence proof must be "
        "preserved in the probe"
    )
    assert "portlearn[plot]" in probe, (
        "the probe's missing-extra render refusal must identify the "
        "portlearn[plot] extra spelling"
    )


def test_wheel_verifier_installs_locked_dependencies_through_lock_aware_offline_sync() -> None:
    """The verifier's dependency stage must be lock-aware, not pip-style.

    A bare ``uv pip install --offline <pins>`` resolves requirements on its
    own, and resolution is exactly what an offline uv cannot do against a
    fresh cache: the cache a lock-populating ``uv sync --locked`` leaves
    behind holds the lock's exact distribution entries, which the
    pip-style resolver cannot find, so the stage fails with "no solution
    found" even though every needed byte is cached. The dependency stage
    therefore runs a lock-aware offline ``uv sync`` from the
    repository/lock context, targeting the isolated verification
    environment through ``UV_PROJECT_ENVIRONMENT``.

    Every flag is load-bearing: ``--locked`` makes ``uv.lock`` the
    version authority and fails on a stale lock; ``--offline`` guarantees
    no index access and no network resolution at verification time;
    ``--no-dev`` keeps development dependencies out of the verification
    environment; ``--no-install-project`` keeps the project itself from
    being installed from source; ``--inexact`` is the documented
    no-removal option that keeps the sync from uninstalling the wheel the
    first stage installed.
    """
    module = _verification_module()
    source = inspect.getsource(module["install_locked_runtime_dependencies"])

    assert '"pip"' not in source, (
        "regression guard: the dependency stage must never resolve "
        "through a bare `uv pip install --offline` — its ad-hoc resolution "
        "fails against a fresh lock-populated cache, which is the exact "
        "defect this contract exists to prevent"
    )
    assert '"sync"' in source, (
        "the dependency stage must drive uv's lock-aware project 'sync' "
        "interface so uv.lock — not an ad-hoc resolver — selects every "
        "dependency version"
    )
    for flag in (
        "--locked",
        "--offline",
        "--no-dev",
        "--no-install-project",
        "--inexact",
    ):
        assert f'"{flag}"' in source, (
            f"the lock-aware offline sync must pass {flag!r}: each flag "
            "carries one guarantee of the dependency-installation contract"
        )
    assert "UV_PROJECT_ENVIRONMENT" in source, (
        "the sync must target the isolated verification environment via "
        "UV_PROJECT_ENVIRONMENT, never the checkout's own environment"
    )
    assert "REPOSITORY_ROOT" in source, (
        "the sync must run from the repository/lock context "
        "(cwd REPOSITORY_ROOT) so uv.lock governs the installation"
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


def test_development_group_is_exactly_pytest_ruff_and_matplotlib(
    pyproject: dict,
) -> None:
    groups = pyproject.get("dependency-groups", {})
    declared = {spec for spec in groups.get("dev", [])}
    assert declared == DEVELOPMENT_DEPENDENCIES, (
        "the dev dependency group must be exactly pytest, ruff, and "
        "matplotlib (the bounded dev addition for headless Agg "
        f"render tests) with their frozen bounds; found {sorted(declared)!r}"
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
        if name == DISTRIBUTION_NAME or name.startswith(f"{DISTRIBUTION_NAME}.")
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
    assert init_path.is_file(), f"package identity module missing: {init_path}"
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


def _requirement_name(entry: str) -> str:
    """The bare distribution name of one ``Requires-Dist`` entry."""
    head = entry.split(";", 1)[0]
    for marker in ("<", ">", "=", "!", "[", " "):
        head = head.split(marker, 1)[0]
    return head.strip().lower()


def test_distribution_declares_unconditional_core_pandas() -> None:
    """The installed distribution carries pandas as an unconditional
    ``Requires-Dist`` entry.

    The core dependency surface is exactly one unconditional pandas
    requirement (no ``extra ==`` marker — a plain ``pip install
    portlearn`` resolves it) plus the marker-guarded parquet-extra
    pyarrow entry and nothing else. A pandas entry guarded by a marker,
    or any second unconditional requirement, changes the installed
    contract and fails here.
    """
    requires = list(metadata.requires(DISTRIBUTION_NAME) or [])
    pandas_entries = [
        entry for entry in requires if _requirement_name(entry) == "pandas"
    ]
    assert pandas_entries, (
        "the installed distribution must declare the core pandas "
        f"requirement unconditionally; found Requires-Dist {requires!r}"
    )
    for entry in pandas_entries:
        assert "extra ==" not in entry, (
            "the pandas requirement must be unconditional — no extra "
            f"marker; found {entry!r}"
        )
    unconditioned = [
        entry
        for entry in requires
        if "extra ==" not in entry and _requirement_name(entry) != "pandas"
    ]
    assert not unconditioned, (
        "no unconditional requirement other than pandas may exist "
        f"(only marker-guarded parquet-extra entries); found "
        f"{unconditioned!r}"
    )


def test_undeclared_top_level_adapter_namespace_is_absent() -> None:
    """The undeclared top-level ``portlearn.adapters`` namespace is
    absent from the public package surface.

    RED-first: importing ``portlearn.adapters`` (and its ``ff``/``fred``
    submodules) raises ``ModuleNotFoundError``. Provider adapters are
    introduced only under ``portlearn.data.adapters``; the top-level
    ``portlearn.adapters`` namespace is not part of the public package
    surface, and no attribute or submodule of that name may exist on
    ``portlearn``.
    """
    import importlib

    for name in (
        "portlearn.adapters",
        "portlearn.adapters.ff",
        "portlearn.adapters.fred",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)

    import portlearn

    assert not hasattr(portlearn, "adapters"), (
        "portlearn.adapters is not part of the public package surface: "
        "no attribute or submodule of that name may exist on portlearn"
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
    ``py.typed`` marker, recursively so subpackages are included — so the
    expectation cannot silently drift when a module or subpackage is
    added, and cannot silently omit an entire subpackage the way a
    non-recursive listing could.
    """
    assert PACKAGE_SOURCE.is_dir(), (
        f"package source directory missing: {PACKAGE_SOURCE}"
    )
    surface = {
        f"{DISTRIBUTION_NAME}/{path.relative_to(PACKAGE_SOURCE).as_posix()}"
        for path in PACKAGE_SOURCE.rglob("*")
        if path.is_file() and (path.suffix == ".py" or path.name == "py.typed")
    }
    assert surface, "the package source tree unexpectedly declares no files"
    return surface


def test_required_wheel_members_cover_the_complete_package_surface() -> None:
    """The wheel check must require the whole package — floor plus derivation.

    A check requiring only ``__init__.py`` and ``py.typed`` lets a wheel
    silently missing any contract module (interfaces, leakage, manifest,
    observations, timing) pass verification. The required-member tuple
    must equal the complete source surface derived recursively (so future
    subpackages join automatically), while the static frozen floor keeps
    deletion of a frozen contract module detected even if the source tree
    itself ever lost it.
    """
    module = _verification_module()
    floor = set(module["REQUIRED_FLOOR_MEMBERS"])
    frozen_floor = {
        f"{DISTRIBUTION_NAME}/{name}"
        for name in (
            "__init__.py",
            "interfaces.py",
            "leakage.py",
            "manifest.py",
            "observations.py",
            "timing.py",
            "py.typed",
        )
    } | {
        # The public diagnostics package marker joins the
        # frozen floor; private per-block filenames are expressly NOT
        # frozen (implementation detail) and join through the recursive
        # derivation alone.
        f"{DISTRIBUTION_NAME}/data/diagnostics/__init__.py",
    }
    assert floor == frozen_floor, (
        "REQUIRED_FLOOR_MEMBERS must be exactly the frozen contract floor "
        f"{sorted(frozen_floor)}; found {sorted(floor)}"
    )
    required = set(module["REQUIRED_WHEEL_MEMBERS"])
    expected = _complete_wheel_surface()
    assert required == expected, (
        "REQUIRED_WHEEL_MEMBERS must require exactly the complete package "
        "surface (every src module plus py.typed, recursively); missing "
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
