"""Strict verification of the PortLearn distribution built by ``uv build``.

Executed verbatim after the build::

    uv build
    uv run python scripts/verify_built_wheel.py

This script verifies the artifact that would be distributed, not the editable
working copy. It is stdlib-only and exits non-zero on any failed assertion or
ambiguity:

1. exactly one wheel and one source distribution produced by ``uv build``
   exist under ``dist/``;
2. the wheel contains the complete package surface — the static fixed
   floor of contract modules plus every shippable file currently under
   ``src/portlearn/`` (``.py`` modules and the ``py.typed`` marker,
   derived recursively so subpackages are included) — so a wheel missing
   any module or subpackage fails;
3. a fresh virtual environment is created under a system temporary directory
   outside the repository checkout;
4. only the built wheel is installed into it — no project source, no
   development dependencies, no editable path, no dependency resolution, no
   package index access;
5. with the working directory outside the checkout, ``import portlearn``
   succeeds from the virtual environment (never from the checkout), and the
   installed distribution metadata reports name ``portlearn`` and exactly the
   version declared in the repository's ``pyproject.toml`` (the single
   version authority), which is read with the standard-library ``tomllib``
   parser rather than hard-coded here — this first stage installs the wheel
   with ``--no-deps --no-index``, so the root import is proven to succeed
   *without* pandas: the core dependency does not leak into the root import;
6. second stage: the *declared*
   runtime dependency set (``[project].dependencies``), each requirement
   pinned to the exact version the repository's ``uv.lock`` already
   resolved, is installed into the same verification environment from the
   local uv cache in offline mode — no package-index access and no network
   resolution at verification time;
7. with the working directory still outside the checkout, the installed
   data subsystem is probed: ``import portlearn.data`` and
   ``import portlearn.data.dataset`` succeed and resolve inside the
   environment (never the checkout), the ``dataset`` import triggers the
   core ``pandas`` import, and ``ResearchDataset.to_pandas`` — the public
   interoperability surface — is present, so the verifier covers the
   pandas-bearing surface the wheel now ships.

Beyond its own temporary environment, this script deletes or alters nothing.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DIST_DIRECTORY = REPOSITORY_ROOT / "dist"
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"
LOCK_PATH = REPOSITORY_ROOT / "uv.lock"
DISTRIBUTION_NAME = "portlearn"
PACKAGE_SOURCE_ROOT = REPOSITORY_ROOT / "src" / DISTRIBUTION_NAME

#: The fixed contract floor a built wheel must always carry, whatever the
#: source tree later adds: the seven foundation members plus the public
#: diagnostics subpackage marker. A source-tree deletion cannot shrink this
#: floor, so a wheel missing any fixed contract module — or silently
#: omitting the entire diagnostics subpackage — fails verification even
#: when the derived surface has moved on. Only the public package marker
#: is fixed: exact private per-block filenames are implementation detail
#: and join through the recursive derivation alone. Pinned by
#: ``tests/test_package_contract.py::test_required_wheel_members_cover_the_complete_package_surface``.
REQUIRED_FLOOR_MEMBERS = (
    "portlearn/__init__.py",
    "portlearn/interfaces.py",
    "portlearn/leakage.py",
    "portlearn/manifest.py",
    "portlearn/observations.py",
    "portlearn/timing.py",
    "portlearn/py.typed",
    "portlearn/data/diagnostics/__init__.py",
)


def _derived_package_surface() -> tuple[str, ...]:
    """Every shippable file under ``src/portlearn/``, recursively.

    Derivation (not enumeration) keeps this the single source of truth:
    a module or subpackage added to the source tree joins
    the required wheel surface automatically, so the check can never again
    go stale — a new module colliding with a pinned static list is the exact
    defect class this hybrid cures. Recursion (``rglob``) includes
    subpackages, so a wheel silently omitting an entire subpackage fails.
    Strict rejection: an unreadable or empty source tree is an error, never a
    vacuous pass.
    """
    if not PACKAGE_SOURCE_ROOT.is_dir():
        fail(
            "package source directory does not exist; cannot derive the "
            f"required wheel surface: {PACKAGE_SOURCE_ROOT}"
        )
    members = {
        f"{DISTRIBUTION_NAME}/{path.relative_to(PACKAGE_SOURCE_ROOT).as_posix()}"
        for path in PACKAGE_SOURCE_ROOT.rglob("*")
        if path.is_file() and (path.suffix == ".py" or path.name == "py.typed")
    }
    if not members:
        fail(
            "derived package surface is empty; refusing to verify against a "
            "vacuous member set"
        )
    return tuple(sorted(members))


#: The complete surface a built wheel must carry: the fixed floor UNION the
#: recursively derived source surface — additions auto-join (derivation),
#: deletions of fixed contract modules stay detected (floor). Pinned by
#: ``tests/test_package_contract.py::test_required_wheel_members_cover_the_complete_package_surface``.
REQUIRED_WHEEL_MEMBERS = tuple(
    sorted(set(REQUIRED_FLOOR_MEMBERS) | set(_derived_package_surface()))
)

# Executed inside the fresh virtual environment with the working directory
# outside the checkout: imports the installed distribution (asserting it
# resolves inside the environment, never inside the checkout) and checks the
# installed metadata name and the pyproject-declared version.
IMPORT_PROBE = """\
from pathlib import Path
from importlib import metadata

import portlearn

module_file = Path(portlearn.__file__).resolve()
venv_root = Path({venv_root!r}).resolve()
checkout_root = Path({checkout_root!r}).resolve()

assert module_file.is_relative_to(venv_root), (
    f"import leaked outside the verification environment: {{module_file}}"
)
assert not module_file.is_relative_to(checkout_root), (
    f"import resolved to the repository checkout, not the installed wheel: "
    f"{{module_file}}"
)
assert metadata.metadata({distribution_name!r})["Name"] == {distribution_name!r}
assert metadata.version({distribution_name!r}) == {version!r}, (
    f"unexpected distribution version: {{metadata.version({distribution_name!r})!r}}"
)
print(f"imported-from: {{module_file}}")
print(
    "metadata: "
    f"{{metadata.metadata({distribution_name!r})['Name']}} "
    f"{{metadata.version({distribution_name!r})}}"
)
"""


def fail(message: str) -> None:
    """Exit non-zero with an explicit, actionable failure reason."""
    raise SystemExit(f"verify_built_wheel: FAIL: {message}")


def declared_version() -> str:
    """The version declared by the repository's ``pyproject.toml``.

    ``pyproject.toml`` is the single version authority, so the expected
    version is derived from it at run time with the standard-library
    ``tomllib`` parser instead of being duplicated as a hard-coded literal
    here.
    """
    if not PYPROJECT_PATH.is_file():
        fail(f"pyproject.toml does not exist at the repository root: {PYPROJECT_PATH}")
    try:
        with PYPROJECT_PATH.open("rb") as handle:
            data = tomllib.load(handle)
        version = data["project"]["version"]
    except (tomllib.TOMLDecodeError, KeyError) as error:
        fail(
            "pyproject.toml does not declare a readable [project] version "
            f"to verify the build against: {error!r}"
        )
    if not isinstance(version, str) or not version:
        fail(
            "pyproject.toml [project] version must be a non-empty string; "
            f"found {version!r}"
        )
    return version


def locate_distribution_artifacts() -> tuple[Path, Path]:
    """Return (wheel, sdist) — exactly one of each, else abort."""
    if not DIST_DIRECTORY.is_dir():
        fail("dist/ does not exist; run `uv build` first")

    wheels = sorted(DIST_DIRECTORY.glob("*.whl"))
    sdists = sorted(DIST_DIRECTORY.glob("*.tar.gz"))

    if len(wheels) != 1:
        fail(
            "expected exactly one wheel under dist/, found "
            f"{len(wheels)}: {[w.name for w in wheels]}"
        )
    if len(sdists) != 1:
        fail(
            "expected exactly one source distribution under dist/, found "
            f"{len(sdists)}: {[s.name for s in sdists]}"
        )

    wheel, sdist = wheels[0], sdists[0]
    for artifact in (wheel, sdist):
        if not artifact.name.startswith(f"{DISTRIBUTION_NAME}-"):
            fail(
                f"artifact {artifact.name!r} is not a {DISTRIBUTION_NAME!r} "
                "distribution"
            )
    return wheel, sdist


def assert_wheel_members(wheel: Path) -> None:
    """The wheel must carry the complete package surface, not two files.

    Requiring only ``__init__.py`` and ``py.typed`` let a wheel missing
    any contract module pass verification; the check now requires every
    required member.
    """
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    missing = [member for member in REQUIRED_WHEEL_MEMBERS if member not in names]
    if missing:
        fail(f"wheel {wheel.name} is missing required members: {missing}")


def create_isolated_environment(root: Path) -> Path:
    """Create a fresh virtual environment under a system temporary directory."""
    if root.is_relative_to(REPOSITORY_ROOT):
        fail(
            "the verification environment must live outside the repository "
            f"checkout; refusing {root}"
        )
    venv_path = root / "verify-venv"
    completed = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        fail(
            "creating the verification virtual environment failed "
            f"(exit {completed.returncode}): {completed.stderr.strip()}"
        )
    return venv_path


def install_wheel_only(venv_path: Path, wheel: Path) -> None:
    """Install exactly the built wheel: no deps, no index, no editable path."""
    venv_python = venv_path / "bin" / "python"
    completed = subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            str(wheel.resolve()),
        ],
        cwd=venv_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        fail(
            f"installing {wheel.name} into the verification environment failed "
            f"(exit {completed.returncode}): {completed.stderr.strip()}"
        )


def import_from_outside_checkout(
    venv_path: Path, root: Path, expected_version: str
) -> None:
    """Import the installed package with cwd outside the checkout."""
    probe_directory = root / "probe"
    probe_directory.mkdir()
    venv_python = venv_path / "bin" / "python"
    environment = {
        key: value for key, value in os.environ.items() if key.upper() != "PYTHONPATH"
    }
    completed = subprocess.run(
        [
            str(venv_python),
            "-c",
            IMPORT_PROBE.format(
                venv_root=str(venv_path),
                checkout_root=str(REPOSITORY_ROOT),
                distribution_name=DISTRIBUTION_NAME,
                version=expected_version,
            ),
        ],
        cwd=probe_directory,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        fail(
            "importing the installed distribution from outside the checkout "
            f"failed (exit {completed.returncode}): "
            f"{(completed.stderr or completed.stdout).strip()}"
        )
    for line in completed.stdout.strip().splitlines():
        print(f"  {line}")


def declared_runtime_dependencies() -> list[str]:
    """The declared runtime dependency set, each pinned to the locked version.

    The requirement *names* come from ``pyproject.toml``'s
    ``[project].dependencies`` (the declared contract surface); each name
    is then pinned to the exact version ``uv.lock`` already resolved for
    this checkout (the local resolution the build environment carries).
    Pinning to the locked version keeps the second stage offline: uv can
    satisfy every pin from its local cache without index access.
    """
    if not PYPROJECT_PATH.is_file():
        fail(f"pyproject.toml does not exist at the repository root: {PYPROJECT_PATH}")
    try:
        with PYPROJECT_PATH.open("rb") as handle:
            data = tomllib.load(handle)
        declared = data["project"]["dependencies"]
    except (tomllib.TOMLDecodeError, KeyError) as error:
        fail(
            "pyproject.toml does not declare a readable [project] "
            f"dependencies row: {error!r}"
        )
    if not isinstance(declared, list) or not all(
        isinstance(item, str) and item.strip() for item in declared
    ):
        fail(
            "pyproject.toml [project] dependencies must be a list of "
            f"non-empty requirement strings; found {declared!r}"
        )

    if not LOCK_PATH.is_file():
        fail(f"uv.lock does not exist at the repository root: {LOCK_PATH}")
    try:
        with LOCK_PATH.open("rb") as handle:
            lock = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        fail(f"uv.lock is not readable TOML: {error!r}")
    locked_versions: dict[str, str] = {}
    for package in lock.get("package", []):
        name = package.get("name")
        version = package.get("version")
        if isinstance(name, str) and isinstance(version, str):
            locked_versions[name] = version

    pinned: list[str] = []
    for requirement in declared:
        name = re.split(r"[\s\[>=<!;~]", requirement.strip(), maxsplit=1)[0]
        version = locked_versions.get(name)
        if version is None:
            fail(
                f"declared runtime dependency {name!r} has no resolved "
                "version in uv.lock; cannot pin the second-stage install "
                "offline"
            )
        pinned.append(f"{name}=={version}")
    if not pinned:
        fail(
            "the declared runtime dependency set is empty; the second "
            "stage has nothing to install"
        )
    return pinned


def install_locked_runtime_dependencies(
    venv_path: Path, requirements: list[str]
) -> None:
    """Install the locked runtime dependency set from the local uv cache.

    The install is lock-aware: a bare ad-hoc offline resolution cannot
    use a fresh cache — the cache a lock-populating ``uv sync --locked``
    leaves behind holds the lock's exact distribution entries, which an
    independent resolver does not find — so this stage drives uv's
    project sync from the repository/lock context instead, with every
    flag carrying one guarantee:

    * ``--locked``: ``uv.lock`` is the sole version authority, and a
      stale lock fails closed instead of being silently re-resolved;
    * ``--offline``: no index access and no network resolution at
      verification time — every wheel must come from the local cache;
    * ``--no-dev``: development dependencies stay out of the
      verification environment;
    * ``--no-install-project``: the project itself is never installed
      from source — only the exact built wheel may provide it;
    * ``--inexact``: uv's documented no-removal option — without it the
      sync would treat the wheel installed by the previous stage as an
      extraneous package and uninstall it.

    ``UV_PROJECT_ENVIRONMENT`` targets the isolated verification
    environment, never the checkout's own environment (a stale
    ``VIRTUAL_ENV`` from the calling shell is dropped so uv cannot warn
    about or target it).
    """
    environment = {
        key: value
        for key, value in os.environ.items()
        if key != "VIRTUAL_ENV"
    }
    environment["UV_PROJECT_ENVIRONMENT"] = str(venv_path)
    completed = subprocess.run(
        [
            "uv",
            "sync",
            "--locked",
            "--no-dev",
            "--no-install-project",
            "--inexact",
            "--offline",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        fail(
            "installing the locked runtime dependency set "
            f"{requirements} from the local cache failed (exit "
            f"{completed.returncode}): {completed.stderr.strip()}"
        )


# Executed inside the verification environment with the working directory
# outside the checkout: imports the installed data subsystem and proves the
# core pandas dependency is exercised by it,
# then proves the diagnostics subpackage ships in the wheel with its public
# surface intact — lazy exposure, exactly the five callable public blocks,
# and no matplotlib on the import path — with a missing-extra render
# refusal that names the ``portlearn[plot]`` extra. No private
# diagnostics module path is assumed anywhere: exact
# private filenames are implementation detail, so the probe
# verifies behavior through the public import surface alone. Literal
# braces are doubled because the probe is ``str.format``-ed with the
# environment roots below.
DATA_SUBSYSTEM_PROBE = """\
import sys

import portlearn.data
import portlearn.data.dataset

assert "pandas" in sys.modules, (
    "importing portlearn.data.dataset must trigger the core pandas import"
)

dataset_cls = portlearn.data.dataset.ResearchDataset
assert hasattr(dataset_cls, "to_pandas"), (
    "ResearchDataset must expose the public to_pandas interoperability "
    "surface"
)

assert "portlearn.data.diagnostics" not in sys.modules, (
    "importing portlearn.data must not eagerly register the diagnostics "
    "subpackage (lazy exposure)"
)

diagnostics = portlearn.data.diagnostics
assert "matplotlib" not in sys.modules, (
    "resolving and importing the diagnostics surface must not import "
    "matplotlib: the [plot] extra is genuinely optional"
)

expected_blocks = (
    "describe",
    "correlation",
    "coverage",
    "missingness",
    "plot",
)
from portlearn.data.diagnostics import (  # noqa: E402
    describe,
    correlation,
    coverage,
    missingness,
    plot,
)
for block in expected_blocks:
    assert callable(getattr(diagnostics, block, None)), (
        f"the installed diagnostics surface must expose {{block!r}}"
    )

records_cls = __import__(
    "portlearn.data._records", fromlist=["PeriodKeyObservation"]
).PeriodKeyObservation
records = (
    records_cls("alpha", "202301", 1.0),
    records_cls("alpha", "202302", 2.0),
    records_cls("alpha", "202303", 3.0),
)
dataset = portlearn.data.dataset.UnqualifiedDataset(
    provider="synthetic",
    name="SYNTH",
    provider_dataset_id="SYNTH",
    adapter_version="synthetic-test-adapter/1.0.0",
    source_bytes=b"wheel-verify-probe",
    auxiliary_bytes=None,
    records=records,
    retrieval_provenance=type(
        "P",
        (),
        {{"content_sha256": "2191658ae1074398b34c47eb057d1785f966f85d68a7c76b7af58ccd5011231f", "units": "index", "frequency": "Monthly"}},
    )(),
    decoder=lambda *a, **k: (_ for _ in ()).throw(AssertionError()),
)
spec = diagnostics.plot(dataset, kind="describe")
assert "matplotlib" not in sys.modules, (
    "constructing a PlotSpec must not import matplotlib"
)
try:
    spec.render()
except Exception as error:
    message = str(error)
    assert "portlearn[plot]" in message, (
        "render() without the [plot] extra must fail naming the extra "
        f"spelling 'portlearn[plot]'; got {{error!r}}"
    )
else:
    raise AssertionError(
        "render() must require the optional [plot] extra; it cannot "
        "succeed with matplotlib absent"
    )

from pathlib import Path

module_file = Path(portlearn.data.dataset.__file__).resolve()
venv_root = Path({venv_root!r}).resolve()
checkout_root = Path({checkout_root!r}).resolve()

assert module_file.is_relative_to(venv_root), (
    f"data-subsystem import leaked outside the verification environment: "
    f"{{module_file}}"
)
assert not module_file.is_relative_to(checkout_root), (
    f"data-subsystem import resolved to the repository checkout, not the "
    f"installed wheel: {{module_file}}"
)
assert "matplotlib" not in sys.modules, (
    "the full data-subsystem probe must complete without matplotlib ever "
    "being imported (the [plot] extra is optional)"
)
print(f"data-subsystem imported-from: {{module_file}}")
print("to_pandas: present on ResearchDataset")
print("diagnostics: lazy exposure, five callable blocks, no matplotlib")
print("render without [plot] extra: refused naming portlearn[plot]")
"""


def probe_data_subsystem(venv_path: Path, root: Path) -> None:
    """Import the installed data subsystem with cwd outside the checkout."""
    probe_directory = root / "probe"
    venv_python = venv_path / "bin" / "python"
    environment = {
        key: value for key, value in os.environ.items() if key.upper() != "PYTHONPATH"
    }
    completed = subprocess.run(
        [
            str(venv_python),
            "-c",
            DATA_SUBSYSTEM_PROBE.format(
                venv_root=str(venv_path),
                checkout_root=str(REPOSITORY_ROOT),
            ),
        ],
        cwd=probe_directory,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        fail(
            "probing the installed data subsystem from outside the "
            f"checkout failed (exit {completed.returncode}): "
            f"{(completed.stderr or completed.stdout).strip()}"
        )
    for line in completed.stdout.strip().splitlines():
        print(f"  {line}")


def main() -> int:
    expected_version = declared_version()
    wheel, sdist = locate_distribution_artifacts()
    print(f"wheel: {wheel}")
    print(f"sdist: {sdist}")
    print(f"expected version (from pyproject.toml): {expected_version}")

    assert_wheel_members(wheel)
    print(f"wheel members: {list(REQUIRED_WHEEL_MEMBERS)} present")

    with tempfile.TemporaryDirectory(prefix="portlearn-verify-") as temporary:
        root = Path(temporary).resolve()
        venv_path = create_isolated_environment(root)
        print(f"verification environment: {venv_path}")
        install_wheel_only(venv_path, wheel)
        import_from_outside_checkout(venv_path, root, expected_version)

        requirements = declared_runtime_dependencies()
        print(f"locked runtime dependencies: {requirements}")
        install_locked_runtime_dependencies(venv_path, requirements)
        probe_data_subsystem(venv_path, root)

    print(
        "verify_built_wheel: PASS: "
        f"{DISTRIBUTION_NAME} {expected_version} imports from the "
        "built wheel in an isolated environment outside the checkout "
        "(root import without pandas; data subsystem with the locked "
        "runtime dependency set)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
