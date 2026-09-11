"""Fail-closed verification of the PortLearn distribution built by ``uv build``.

Executed verbatim after the build::

    uv build
    uv run python scripts/verify_built_wheel.py

This script verifies the artifact that would be distributed, not the editable
working copy. It is stdlib-only and exits non-zero on any failed assertion or
ambiguity:

1. exactly one wheel and one source distribution produced by ``uv build``
   exist under ``dist/``;
2. the wheel contains the complete package surface — every module under
   ``portlearn/`` (``__init__.py``, ``interfaces.py``, ``leakage.py``,
   ``manifest.py``, ``observations.py``, ``timing.py``) plus the
   ``py.typed`` marker — so a wheel missing any contract module fails;
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
   parser rather than hard-coded here.

Beyond its own temporary environment, this script deletes or alters nothing.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DIST_DIRECTORY = REPOSITORY_ROOT / "dist"
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"
DISTRIBUTION_NAME = "portlearn"
#: The complete package surface a built wheel must carry: every source
#: module plus the ``py.typed`` marker — not just the package identity
#: files — so a wheel silently missing a contract module fails
#: verification. Kept in lockstep with ``src/portlearn/`` and pinned by
#: ``tests/test_package_contract.py::test_required_wheel_members_cover_the_complete_package_surface``.
REQUIRED_WHEEL_MEMBERS = (
    "portlearn/__init__.py",
    "portlearn/interfaces.py",
    "portlearn/leakage.py",
    "portlearn/manifest.py",
    "portlearn/observations.py",
    "portlearn/timing.py",
    "portlearn/py.typed",
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
    """Return (wheel, sdist) — exactly one of each, else fail closed."""
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
        key: value
        for key, value in os.environ.items()
        if key.upper() != "PYTHONPATH"
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

    print(
        "verify_built_wheel: PASS: "
        f"{DISTRIBUTION_NAME} {expected_version} imports from the "
        "built wheel in an isolated environment outside the checkout"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
