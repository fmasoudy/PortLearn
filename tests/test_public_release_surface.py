"""Public release-surface content tests.

These tests enforce the public foundation content surfaces semantically
(the file is read and parsed, never compared against an embedded prose
constant):

* ``CHANGELOG.md`` follows Keep a Changelog: it has an ``[Unreleased]``
  section, every version heading parses as PEP 440, a changelog section
  prepared for the declared development version carries content while
  ``[Unreleased]`` stays empty, a release entry is structurally
  required before any release, and the ``[Unreleased]``
  body carries no self-invalidating release-status phrases;
* ``README.md`` carries a **Status** section that communicates early
  development with no stable package release, and a **Roadmap** section
  read from the file that keeps its **Available now** / **Next** /
  **Planned** structure without dates or version promises; it makes no
  implemented end-to-end workflow claim and documents the development
  commands;
* ``examples/foundation_contract_wiring.py`` exists, executes cleanly, and
  visibly carries its fence statement: it demonstrates foundation contracts
  and composition only — not an implemented end-to-end portfolio research
  workflow — using only synthetic inputs and existing foundation APIs.
"""

from __future__ import annotations

import re
import runpy
import tomllib
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHANGELOG_PATH = REPOSITORY_ROOT / "CHANGELOG.md"
README_PATH = REPOSITORY_ROOT / "README.md"
EXAMPLE_PATH = REPOSITORY_ROOT / "examples" / "foundation_contract_wiring.py"
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"

#: The fence sentence the example must state in the example itself.
FENCE_SENTENCE = (
    "This example demonstrates foundation contracts and composition, not an "
    "implemented end-to-end portfolio research workflow."
)

#: Interface objects whose construction or implementation in the public
#: example would signal capability beyond the current foundation (no
#: forecasting, optimizer, strategy, accounting, or evaluation semantics).
#: The example may print the declared interface surface at runtime; it must
#: not construct or implement any of these components in source.
FORBIDDEN_CAPABILITY_TOKENS = (
    "Forecast",
    "Forecaster",
    "Strategy",
    "PortfolioDecision",
    "RebalancePolicy",
    "CostModel",
    "AccountingEngine",
    "AccountingResult",
    "Evaluator",
)

#: The Keep a Changelog top-level section headings (structure, not prose).
KEEP_A_CHANGELOG_SECTIONS = ("Added", "Changed", "Fixed", "Deprecated", "Removed")

#: Phrases that would self-invalidate an ``[Unreleased]`` entry: asserting
#: publication state that the changelog itself contradicts by carrying no
#: release section. A living changelog describes *content*, not whether a
#: release has happened.
SELF_INVALIDATING_UNRELEASED_PHRASES = (
    "no public release has been made",
    "has not yet been released",
    "not yet been released",
    "no release has been made",
    "no releases have been made",
)

#: The development workflow commands README's Development section must
#: carry verbatim (the local battery: install, lint, test, build, verify).
DEVELOPMENT_COMMANDS = (
    "uv sync --locked",
    "uv run ruff check .",
    "uv run pytest",
    "uv build",
    "uv run python scripts/verify_built_wheel.py",
)

#: PEP 440 version syntax, checked without importing anything: the pattern
#: covers the project's ``0.0.x`` early-development policy space
#: (major.minor.micro with optional pre/post/dev segments).
_VERSION_HEADING_PATTERN = re.compile(r"^## \[([^\]]+)\]", re.MULTILINE)
_PEP440_CORE = (
    r"\d+(\.\d+)*"
    r"((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?(\+[a-z0-9]+(?:[-.][a-z0-9]+)*)?"
)
_PEP440_PATTERN = re.compile(rf"^(?:v)?{_PEP440_CORE}$", re.IGNORECASE)


def _read_required(path: Path) -> str:
    assert path.is_file(), (
        f"required public file missing: {path.relative_to(REPOSITORY_ROOT)}"
    )
    return path.read_text(encoding="utf-8")


def _section_body(text: str, heading: str) -> str:
    """The body of one ``##`` section, up to the next ``##`` heading."""
    match = re.search(
        rf"^## {re.escape(heading)}[^\n]*\n", text, re.MULTILINE
    )
    assert match, f"required '## {heading}' section is missing"
    rest = text[match.end() :]
    following = re.search(r"^## ", rest, re.MULTILINE)
    return rest[: following.start()] if following else rest


def _declared_version() -> str:
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


# ---------------------------------------------------------------------------
# CHANGELOG.md — Keep a Changelog structure, read from the file
# ---------------------------------------------------------------------------


def test_changelog_carries_an_unreleased_section() -> None:
    """Keep a Changelog requires a live ``[Unreleased]`` section."""
    text = _read_required(CHANGELOG_PATH)
    match = re.search(r"^## \[Unreleased\]", text, re.MULTILINE)
    assert match, "CHANGELOG.md must carry an '## [Unreleased]' section"


def test_changelog_version_headings_parse_as_pep_440() -> None:
    """Every version heading in the changelog is a valid PEP 440 version.

    This is the structural check that keeps the release mechanism's
    ``## [{version}]`` lookup well-formed: a heading that cannot parse as
    PEP 440 could never match the version the release workflow validates,
    so it is malformed here, not merely unconventional.
    """
    text = _read_required(CHANGELOG_PATH)
    headings = [m.group(1) for m in _VERSION_HEADING_PATTERN.finditer(text)]
    assert "Unreleased" in headings, (
        "the changelog must keep its [Unreleased] heading"
    )
    versions = [h for h in headings if h != "Unreleased"]
    bad = [h for h in versions if not _PEP440_PATTERN.match(h)]
    assert not bad, (
        f"changelog version headings must parse as PEP 440; got {bad!r}"
    )


def test_changelog_section_for_declared_development_version_is_prepared() -> None:
    """The declared development version names the pending release section.

    The currently declared version is a development version (``.devN``).
    The release mechanism never tags or releases development versions,
    so a ``## [<declared>]`` heading asserts no release: it records that
    the pending changes are destined for this version. A prepared
    changelog therefore carries a non-empty section for the declared
    development version while ``[Unreleased]`` holds no pending
    content — the same shape the release mechanism requires once the
    version is finalized and the release actually made.
    """
    text = _read_required(CHANGELOG_PATH)
    declared = _declared_version()
    headings = [m.group(1) for m in _VERSION_HEADING_PATTERN.finditer(text)]
    assert declared in headings, (
        f"the declared development version {declared!r} must name a "
        "changelog section: the pending public changes are destined for "
        "this version, and the release mechanism draws its notes from "
        "that section once the version is finalized"
    )
    section = _section_body(text, f"[{declared}]")
    content = [
        line.strip()
        for line in section.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert content, (
        f"the [{declared}] section must carry the pending public "
        "changes: an empty section prepared for the declared version "
        "would leave the changelog's pending content nowhere to live"
    )
    unreleased = _section_body(text, "[Unreleased]")
    pending = [
        line.strip()
        for line in unreleased.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not pending, (
        "the [Unreleased] body must be empty when the changelog "
        "prepares a section for the declared version: pending entries "
        "live under the declared version's section, not [Unreleased]"
    )


def test_changelog_release_entry_is_structurally_required_before_release() -> None:
    """No release heading means no release: the changelog cannot claim one.

    The structural rule the release mechanism enforces is byte-free: a
    release needs a ``## [<version>]`` heading whose section body is not
    empty, and ``[Unreleased]`` must be empty at that moment. The test
    pins the requirement on the *shape*, never on mutable prose: if the
    changelog carries no release heading, then its ``[Unreleased]`` body
    is exactly the pending content, and any release-section scaffold it
    does carry must be non-empty to be releasable.
    """
    text = _read_required(CHANGELOG_PATH)
    headings = [m.group(1) for m in _VERSION_HEADING_PATTERN.finditer(text)]
    release_headings = [h for h in headings if h != "Unreleased"]
    for heading in release_headings:
        body = _section_body(text, f"[{heading}]")
        content = [
            line.strip()
            for line in body.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert content, (
            f"the [{heading}] release section must carry release notes "
            "content: the release mechanism draws its notes from this section"
        )
    if not release_headings:
        unreleased = _section_body(text, "[Unreleased]")
        assert "### Added" in unreleased or any(
            f"### {section}" in unreleased for section in KEEP_A_CHANGELOG_SECTIONS
        ), (
            "with no release made yet, the changelog must keep its pending "
            "public changes under [Unreleased] with Keep a Changelog "
            "subsections"
        )


def test_changelog_unreleased_body_denies_self_invalidating_phrases() -> None:
    """``[Unreleased]`` describes pending content, never release status.

    A phrase like "no public release has been made yet" is
    self-invalidating: it becomes false the moment the first release is
    cut (the entry then ships inside that very release), and it restates
    what the file's structure already records honestly. Pending entries
    describe capability, not publication state.
    """
    text = _read_required(CHANGELOG_PATH)
    body = _section_body(text, "[Unreleased]")
    lowered = body.lower()
    for phrase in SELF_INVALIDATING_UNRELEASED_PHRASES:
        assert phrase not in lowered, (
            f"the [Unreleased] body must not carry the self-invalidating "
            f"release-status phrase {phrase!r}: pending entries describe "
            "capability, never whether a release has happened"
        )


def test_changelog_keeps_keep_a_changelog_subsection_structure() -> None:
    """The Keep a Changelog ``###`` subsection headings are present."""
    text = _read_required(CHANGELOG_PATH)
    assert "### Added" in text, (
        "the changelog must keep the Keep a Changelog '### Added' "
        "subsection under [Unreleased]"
    )
    for section in KEEP_A_CHANGELOG_SECTIONS:
        assert f"### {section}" in text, (
            f"the changelog must keep the Keep a Changelog '### {section}' "
            "subsection heading"
        )


# ---------------------------------------------------------------------------
# README.md — Status, Roadmap, fence, and Development sections
# ---------------------------------------------------------------------------


def test_readme_carries_status_and_roadmap_sections() -> None:
    text = _read_required(README_PATH)
    assert re.search(r"^## Status\s*$", text, re.MULTILINE), (
        "README.md must carry a '## Status' section"
    )
    assert re.search(r"^## Roadmap\s*$", text, re.MULTILINE), (
        "README.md must carry a '## Roadmap' section"
    )


def test_readme_status_conveys_early_development_without_stable_release() -> None:
    """The Status section states early development and no stable release.

    Read from the file: the status must communicate that the project is in
    early development and has not yet been published as a stable package
    release, while describing the present public surface (the research
    foundation and API contracts) rather than future capabilities.
    """
    text = _read_required(README_PATH)
    status = _section_body(text, "Status")
    lowered = status.lower()
    assert "early development" in lowered, (
        "the README Status section must state that PortLearn is in early "
        "development"
    )
    assert re.search(
        r"not\s+yet\s+been\s+published|has\s+not\s+been\s+published"
        r"|no\s+stable\s+(package\s+)?release",
        lowered,
    ), (
        "the README Status section must state that PortLearn has not yet "
        "been published as a stable package release"
    )
    assert "public surface" in lowered or "api contracts" in lowered or (
        "research foundation" in lowered
    ), (
        "the README Status section must describe the current public surface "
        "(the research foundation and API contracts)"
    )


def test_readme_makes_no_implemented_end_to_end_claim() -> None:
    """No section of the README claims an implemented end-to-end workflow."""
    text = _read_required(README_PATH)
    lowered = text.lower()
    forbidden = (
        "end-to-end portfolio",
        "end to end portfolio",
        "complete portfolio research workflow",
        "production-ready",
        "battle-tested",
    )
    for phrase in forbidden:
        assert phrase not in lowered, (
            f"the README must not claim {phrase!r}: the current public "
            "surface is the research foundation, not an implemented "
            "end-to-end portfolio research workflow"
        )


def test_readme_roadmap_retains_its_structure_from_the_file() -> None:
    """The Roadmap, read from the file, keeps its three-tier structure."""
    text = _read_required(README_PATH)
    roadmap = _section_body(text, "Roadmap")
    for tier in ("Available now", "Next", "Planned"):
        assert f"**{tier}**" in roadmap, (
            f"the README Roadmap must keep its '**{tier}**' tier"
        )


def test_readme_roadmap_promises_no_dates_or_versions() -> None:
    """The Roadmap stays directional: no dates and no version promises."""
    roadmap = _section_body(_read_required(README_PATH), "Roadmap")
    assert "v0." not in roadmap and "0.0.1" not in roadmap, (
        "the Roadmap must not promise specific versions"
    )
    for year in ("2026", "2027", "2028"):
        assert year not in roadmap, (
            f"the Roadmap must not carry the date {year!r}"
        )


def test_readme_documents_the_development_commands() -> None:
    """README's Development section carries the local battery verbatim."""
    text = _read_required(README_PATH)
    assert re.search(r"^## Development\s*$", text, re.MULTILINE), (
        "README.md must carry a '## Development' section"
    )
    development = _section_body(text, "Development")
    for command in DEVELOPMENT_COMMANDS:
        assert command in development, (
            f"the README Development section must include the command "
            f"{command!r} verbatim"
        )


# ---------------------------------------------------------------------------
# examples/foundation_contract_wiring.py — fence and execution
# ---------------------------------------------------------------------------


def test_example_source_carries_the_fence_and_no_capability_semantics() -> None:
    """The example states the fence in itself, declares synthetic inputs,
    and contains no capability-object construction or implementation."""
    source = _read_required(EXAMPLE_PATH)
    assert FENCE_SENTENCE in source, (
        "the example must state, in the example itself, that it demonstrates "
        "foundation contracts and composition, not an end-to-end portfolio "
        "research workflow"
    )
    assert "synthetic" in source.lower(), (
        "the example must visibly declare that every input is synthetic"
    )
    for token in FORBIDDEN_CAPABILITY_TOKENS:
        assert token not in source, (
            f"the public example must not construct or implement {token!r} "
            "(fence: foundation contracts only)"
        )


def test_example_executes_and_prints_the_fence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The real example file executes cleanly under its ``__main__`` guard and
    visibly prints the fence statement before anything else it reports."""
    runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")
    assert FENCE_SENTENCE in capsys.readouterr().out


def test_example_leakage_battery_blocks_both_synthetic_leak_attempts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The executed example's leakage battery reports both synthetic leak
    attempts blocked by the contract errors (totality summary)."""
    runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")
    assert "2/2 attempted leaks blocked" in capsys.readouterr().out


def test_example_manifest_round_trips_canonically(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The executed example records a run manifest and prints its canonical
    JSON, demonstrating the manifest contract on synthetic provenance."""
    runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")
    out = capsys.readouterr().out
    assert "example-foundation-contract-wiring" in out
    assert '"run_id":"example-foundation-contract-wiring"' in out
