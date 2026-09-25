"""Controlled public release mechanism tests.

These tests enforce the unconditional manual-trigger public release
workflow (``.github/workflows/release.yml``):

* the workflow triggers on ``workflow_dispatch`` only — never on any
  push, merge, or milestone event;
* it is guarded to the public ``fmasoudy/PortLearn`` repository, holds
  least-privilege ``contents: write`` only, pins every action to an
  immutable commit SHA, and serializes runs in a ``public-release``
  concurrency group without cancel-in-progress;
* the checkout does not persist credentials, and every later step
  performs read-only repository operations except the single final
  release creation, which is the only step that mutates and
  which names its repository and token explicitly;
* the exact target commit is checked out by full 40-hex SHA with full
  history and verified to be the current ``origin/main`` head (never a
  stale or off-main commit);
* tag/Release existence is probed so that only a confirmed answer
  permits the run to proceed: a remote-tag query that fails is never
  read as tag-absence, and only an explicit HTTP 404 means "no release
  yet" — every other status aborts;
* the version comes exclusively from ``pyproject.toml`` (exact equality
  with the requested version; PEP 440 validated with the real
  ``packaging`` implementation, never re-invented), ``tag_name`` must
  equal ``v`` plus that version, the changelog must carry the exact
  versioned release section with non-empty notes and an empty/absent
  ``[Unreleased]`` section (empty — not merely bullet-free), and the
  version must be monotone-forward against existing public version
  tags;
* a version carrying a PEP 440 development segment (``.devN``, as in
  the currently declared ``0.0.1.dev0``) is rejected — development
  releases are never tagged publicly — and pre-release versions
  (``a``/``b``/``rc``) are also rejected: the mechanism does not create
  prerelease-marked GitHub Releases, so RC support is deferred until
  that support exists;
* every validation step precedes the single ``gh release create``
  action — no manual tag creation, no push, no delete, no move, no
  overwrite, and no alternative creation vector (``gh api`` POST,
  mutating curl) anywhere in the workflow;
* the embedded validation module (extractable between explicit sentinel
  markers) is offline and side-effect free, its command-line entry
  point wires argv and the notes file exactly as the workflow invokes
  it, so a deliberately mismatched dry input is provable to unconditional
  *before any creation* without network access and without creating any
  tag or release.

No private path is consulted: the workflow and these tests are part of
the public surface.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "release.yml"

PUBLIC_REPOSITORY = "fmasoudy/PortLearn"
REQUIRED_INPUTS = ("version", "tag_name", "target_sha")
CREATION_STEP_NAME = "Create the GitHub Release and its tag in one action"
CREATION_COMMAND = (
    'gh release create "$PL_TAG_NAME" --target "$PL_TARGET_SHA" '
    '--title "$PL_TAG_NAME" --notes-file "${RUNNER_TEMP}/release-notes.md"'
)
SENTINEL_BEGIN = "# --- portlearn-release-validation: begin (pure, offline) ---"
SENTINEL_END = "# --- portlearn-release-validation: end ---"
VALIDATION_MODULE_NAME = "portlearn_release_validation"
#: The complete set of callables the embedded module may define: pure
#: validation only — no creation entry point may exist, so a failing
#: validation is provably incapable of creating anything.
ALLOWED_MODULE_CALLABLES = {
    "ReleaseValidationError",
    "check_monotone_forward",
    "changelog_section",
    "declared_version",
    "heading_pattern",
    "main",
    "meaningful_content",
    "reject_development_release",
    "reject_prerelease_version",
    "validate_release",
}

FORBIDDEN_TRIGGER_KEYS = (
    "push",
    "pull_request",
    "pull_request_target",
    "schedule",
    "workflow_run",
    "release",
    "create",
    "delete",
    "issues",
    "milestone",
    "status",
    "discussion",
    "watch",
    "fork",
)

FORBIDDEN_PERMISSION_SCOPES = (
    "actions",
    "attestations",
    "checks",
    "deployments",
    "discussions",
    "id-token",
    "issues",
    "packages",
    "pages",
    "pull-requests",
    "security-events",
    "statuses",
)

#: Repository-mutating command patterns: no step outside the creation
#: step may run any of these, and no alternative creation vector
#: (API-level POST/PATCH/PUT/DELETE) may exist anywhere.
MUTATING_COMMAND_PATTERN = (
    r"\bgit push\b"
    r"|\bgit tag\b(?! --list)"
    r"|\bgit remote add\b"
    r"|\bgh release create\b"
    r"|\bgh api\b.*--method (POST|PATCH|PUT|DELETE)"
    r"|\bcurl\b.*\s-(X|--request)\s+(POST|PATCH|PUT|DELETE)"
)


def _release_workflow_text() -> str:
    if not WORKFLOW_PATH.is_file():
        pytest.fail(
            "the controlled public release workflow is missing: "
            ".github/workflows/release.yml (the unconditional "
            "workflow_dispatch-only public release surface)"
        )
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def _trigger_block(text: str) -> str:
    """The body of the ``on:`` mapping, up to the next top-level key."""
    match = re.search(
        r"^on:\n(.*?)(?=^[A-Za-z][A-Za-z_-]*:)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match, "the workflow must define an on: trigger mapping"
    return match.group(1)


def _step_names(text: str) -> list[str]:
    return re.findall(r"^      - name: (.+)$", text, re.MULTILINE)


def _step_run_block(text: str, step_name: str) -> str:
    """The ``run:`` script of one named step (empty for action steps)."""
    match = re.search(
        rf"^      - name: {re.escape(step_name)}\n"
        r"(?:        [^\n]*\n)*?"
        r"        run: \|\n"
        r"((?:          .*\n?)+)",
        text,
        re.MULTILINE,
    )
    if match is None:
        return ""
    return match.group(1)


def _required_run_block(text: str, step_name: str) -> str:
    run = _step_run_block(text, step_name)
    assert run, f"the workflow step {step_name!r} must have a run: block"
    return run


def _step_yaml(text: str, step_name: str) -> str:
    """The full YAML of one named step (name, env, run)."""
    match = re.search(
        rf"^      - name: {re.escape(step_name)}\n"
        r"(?:        [^\n]*\n)+"
        r"(?=^      - name: |\Z)",
        text,
        re.MULTILINE,
    )
    assert match, f"the workflow must define step {step_name!r}"
    return match.group(0)


def _extract_validation_block(text: str) -> str:
    """The sentinel-delimited pure Python embedded in the workflow."""
    assert text.count(SENTINEL_BEGIN) == 1, (
        "the workflow must embed its validation module between the "
        f"sentinel {SENTINEL_BEGIN!r} exactly once"
    )
    assert text.count(SENTINEL_END) == 1, (
        "the workflow must close its validation module with the sentinel "
        f"{SENTINEL_END!r} exactly once"
    )
    _, begin_marker, after_begin = text.partition(SENTINEL_BEGIN)
    assert begin_marker
    body, end_marker, _ = after_begin.partition(SENTINEL_END)
    assert end_marker
    return dedent(body).strip() + "\n"


def _validation_module() -> dict:
    """Execute the embedded validation module offline and return it."""
    block = _extract_validation_block(_release_workflow_text())
    namespace: dict = {"__name__": VALIDATION_MODULE_NAME}
    exec(compile(block, "<release-validation>", "exec"), namespace)  # noqa: S102
    return namespace


def _write_release_fixture(
    root: Path,
    *,
    declared: str = "0.0.1",
    section_version: str | None = None,
    release_entries: tuple[str, ...] = (
        "Initial public release of the research foundation contracts.",
    ),
    unreleased_entries: tuple[str, ...] = (),
    unreleased_prose: str | None = None,
) -> tuple[Path, Path]:
    """A self-contained release fixture tree (no repository state)."""
    versioned = section_version if section_version is not None else declared
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        f'[project]\nname = "portlearn"\nversion = "{declared}"\n',
        encoding="utf-8",
    )
    parts = [
        "# Changelog",
        "",
        "## [Unreleased]",
        "",
        "### Added",
        "",
    ]
    parts.extend(f"- {entry}" for entry in unreleased_entries)
    if unreleased_prose is not None:
        parts.append(unreleased_prose)
    parts.extend(
        [
            "",
            f"## [{versioned}] - 2026-09-09",
            "",
            "### Added",
            "",
        ]
    )
    parts.extend(f"- {entry}" for entry in release_entries)
    parts.append("")
    changelog = root / "CHANGELOG.md"
    changelog.write_text("\n".join(parts), encoding="utf-8")
    return pyproject, changelog


def _run_module_cli(
    script: Path, argv: list[str]
) -> subprocess.CompletedProcess[str]:
    """Run the embedded module offline as a real subprocess."""
    return subprocess.run(
        [sys.executable, str(script), *argv],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


# ---------------------------------------------------------------------------
# Workflow surface contract
# ---------------------------------------------------------------------------


def test_release_workflow_exists() -> None:
    assert WORKFLOW_PATH.is_file(), (
        "the controlled public release workflow is the required "
        "public surface: .github/workflows/release.yml"
    )


def test_workflow_declares_a_clear_name() -> None:
    assert re.search(
        r"^name: public-release$", _release_workflow_text(), re.MULTILINE
    ), "the workflow must declare the clear name 'public-release'"


def test_workflow_triggers_on_workflow_dispatch_only() -> None:
    block = _trigger_block(_release_workflow_text())
    keys = [
        key
        for key in re.findall(r"^  ([A-Za-z][A-Za-z_-]*):$", block, re.MULTILINE)
        if key != "inputs"  # the input declarations, not a trigger
    ]
    assert keys == ["workflow_dispatch"], (
        "the sole supported trigger is a manual workflow_dispatch; found "
        f"trigger keys {keys} (workflow_dispatch-only trigger rule)"
    )
    for forbidden in FORBIDDEN_TRIGGER_KEYS:
        assert forbidden not in keys, (
            f"a {forbidden} trigger would make the release mechanism "
            "automatic rather than manually dispatched"
        )


def test_workflow_declares_exactly_the_three_required_string_inputs() -> None:
    block = _trigger_block(_release_workflow_text())
    declared_inputs = re.findall(r"^      ([a-z_]+):$", block, re.MULTILINE)
    assert sorted(declared_inputs) == sorted(REQUIRED_INPUTS), (
        "the workflow must take exactly the required string inputs "
        f"{sorted(REQUIRED_INPUTS)}; found {sorted(declared_inputs)}"
    )
    for name in REQUIRED_INPUTS:
        match = re.search(
            rf"^      {name}:\n((?:        .*\n)+)", block, re.MULTILINE
        )
        assert match, f"input {name!r} must declare its properties"
        properties = match.group(1)
        assert "required: true" in properties, (
            f"input {name!r} must be required: a release never proceeds "
            "on an omitted value"
        )
        assert "type: string" in properties, (
            f"input {name!r} must be a plain string input"
        )


def test_workflow_is_guarded_to_the_public_repository() -> None:
    guard = f"if: github.repository == '{PUBLIC_REPOSITORY}'"
    assert guard in _release_workflow_text(), (
        "the release job must be guarded to the public repository "
        f"{PUBLIC_REPOSITORY!r}; it must never run anywhere else"
    )


def test_workflow_holds_least_privilege_contents_write_only() -> None:
    text = _release_workflow_text()
    assert re.search(
        r"^permissions:\n  contents: write$", text, re.MULTILINE
    ), (
        "creating a release tag needs exactly contents: write and nothing "
        "broader (least permissions)"
    )
    for scope in FORBIDDEN_PERMISSION_SCOPES:
        assert f"  {scope}:" not in text, (
            f"the workflow must not request the {scope} permission; "
            "contents: write is the only permission the mechanism needs"
        )


def test_workflow_serializes_runs_in_a_public_release_concurrency_group() -> None:
    assert re.search(
        r"^concurrency:\n  group: public-release\n  cancel-in-progress: false$",
        _release_workflow_text(),
        re.MULTILINE,
    ), (
        "the workflow must serialize dispatches in a 'public-release' "
        "concurrency group with cancel-in-progress: false, so two "
        "overlapping runs cannot race between validation and creation"
    )


def test_workflow_pins_every_action_to_an_immutable_commit_sha() -> None:
    uses = re.findall(r"uses: (\S+)@(\S+)", _release_workflow_text())
    assert uses, "the workflow must reference its actions via uses:"
    for action, pin in uses:
        sha = pin.split()[0]
        assert re.fullmatch(r"[0-9a-f]{40}", sha), (
            f"action {action!r} must be pinned to a full 40-hex immutable "
            f"commit SHA; found {pin!r}"
        )


def test_checkout_does_not_persist_credentials() -> None:
    text = _release_workflow_text()
    assert "persist-credentials: false" in text, (
        "the checkout must not persist the runtime token into the git "
        "credential store: every later step stays least-authority"
    )
    assert not re.search(r"persist-credentials:\s*true", text), (
        "no step may re-enable credential persistence"
    )


def test_workflow_checks_out_the_exact_target_commit_with_full_history() -> None:
    text = _release_workflow_text()
    assert "ref: ${{ inputs.target_sha }}" in text, (
        "the checkout must pin the exact requested target commit, not a "
        "mutable branch head"
    )
    assert "fetch-depth: 0" in text, (
        "full history is required to verify main-head equality and to "
        "compare against existing public version tags"
    )
    assert (
        '"+refs/heads/main:refs/remotes/origin/main"' in text
        and "git fetch --tags origin" in text
    ), (
        "the workflow must refresh origin/main and the existing public "
        "tags (read-only fetches) before validating anything"
    )


def test_workflow_verifies_the_target_is_the_current_main_head() -> None:
    text = _release_workflow_text()
    assert (
        "abort unless the target commit is the current origin main head"
        in text
    ), "the main-head equality check must be an explicit named step"
    assert 'git cat-file -e "${PL_TARGET_SHA}^{commit}"' in text, (
        "the target SHA must be verified to exist as a commit"
    )
    assert "git rev-parse refs/remotes/origin/main" in text, (
        "the current origin/main head must be resolved for the equality "
        "check (no stale and no off-main target may be tagged)"
    )


def test_main_head_guard_keeps_its_direction_and_exit() -> None:
    """The main-head guard compares in the safe direction and exits.

    An inverted comparison (``!=`` → ``==``) or an error echo whose
    ``exit 1`` was removed must fail this test instead of shipping.
    """
    run = _required_run_block(
        _release_workflow_text(),
        "abort unless the target commit is the current origin main head",
    )
    assert 'if [ "$main_head" != "$PL_TARGET_SHA" ]; then' in run, (
        "the main-head guard must unconditional whenever the resolved head "
        "differs from the requested target; an inverted comparison lets "
        "a stale or off-main commit be tagged"
    )
    assert re.search(r"^\s*exit 1\s*$", run, re.MULTILINE), (
        "the main-head guard must terminate with exit 1 on mismatch; an "
        "error echo alone would let the run continue"
    )


def test_remote_tag_guard_fails_closed_on_query_error() -> None:
    """ls-remote status is captured; only success may mean absence.

    The guard must capture the ``git ls-remote`` exit status separately
    from its output and abort on any nonzero status: a transport, auth,
    or rate-limit failure yields empty stdout, which must never be read
    as "the tag is absent" (a fail-open defect).
    """
    run = _required_run_block(
        _release_workflow_text(),
        "Abort if the tag already exists locally or on the remote",
    )
    assert re.search(
        r'^\s*remote_tag_query="\(?\$\(git ls-remote --tags origin '
        r"\"refs/tags/\$PL_TAG_NAME\"\)\)?\"\s*$",
        run,
        re.MULTILINE,
    ) or (
        'remote_tag_query="$(git ls-remote --tags origin '
        '"refs/tags/$PL_TAG_NAME")"' in run
    ), (
        "the remote-tag guard must capture the ls-remote query output in "
        "a plain assignment so its exit status becomes observable "
        "(set -e never fires inside if-conditions or ||-chains)"
    )
    assert 'remote_tag_query_status="$?"' in run, (
        "the ls-remote exit status must be captured immediately after "
        "the query, separately from its output"
    )
    assert 'if [ "$remote_tag_query_status" -ne 0 ]; then' in run, (
        "the guard must abort whenever the ls-remote query itself fails: "
        "query failure is never tag-absence"
    )
    assert re.search(r"::error::", run), (
        "the query-failure path must emit a workflow error annotation"
    )
    assert re.search(r"^\s*exit 1\s*$", run, re.MULTILINE), (
        "the query-failure path must terminate the step"
    )
    assert 'if [ -n "$remote_tag_query" ]; then' in run, (
        "after a successful query, any nonempty output means the remote "
        "tag exists and the guard must unconditional"
    )
    assert not re.search(r"if \[ -n \"\$\(git ls-remote", run), (
        "the remote-tag guard must not test ls-remote output inside an "
        "if-condition: that form discards the query's exit status and "
        "treats transport/auth failures as tag-absence"
    )


def test_release_existence_probe_distinguishes_404_from_other_statuses() -> None:
    """Only an explicit HTTP 404 means "no release yet".

    A generic ``gh release view`` exit 1 conflates 404 with 401/403/5xx
    and network failure; the probe must discriminate: 200 → exists →
    fail; 404 → absent → proceed; anything else → unconditional.
    """
    run = _required_run_block(
        _release_workflow_text(),
        "Abort if the GitHub Release already exists",
    )
    assert not re.search(r"\bgh release view\b", run), (
        "the probe must not use `gh release view`: its generic exit 1 "
        "cannot distinguish 404 (absent) from could-not-determine"
    )
    assert re.search(r'release_status="\$\(curl', run), (
        "the probe must capture the HTTP status of the release-by-tag "
        "query so the status can be discriminated explicitly"
    )
    assert re.search(r'"\$\{?GITHUB_API_URL\}?/repos/', run), (
        "the probe must target the runner-provided GITHUB_API_URL inside "
        "a double-quoted URL"
    )
    assert re.search(r'if \[ "\$release_status" = "200" \]; then', run), (
        "an explicit 200 means the release exists; the guard must fail "
        "closed rather than proceed"
    )
    assert re.search(r'elif \[ "\$release_status" = "404" \]; then', run), (
        "release-absence may be concluded only from an explicit 404; "
        "string-exact branch matching keeps a non-numeric or unexpected "
        "status out of the absent branch"
    )
    assert re.search(
        r"^\s*else\s*$\n\s+echo \"::error::.*\n\s+exit 1\s*$",
        run,
        re.MULTILINE,
    ), (
        "every status other than 200 and 404 (auth failure, rate limit, "
        "server error, transport garbage) must land in an else branch "
        "that fails closed"
    )
    assert run.count("exit 1") >= 2, (
        "both the exists (200) path and the could-not-determine path "
        "must terminate the step"
    )


def test_release_probe_env_names_token_and_repository_explicitly() -> None:
    step = _step_yaml(
        _release_workflow_text(),
        "Abort if the GitHub Release already exists",
    )
    assert re.search(r"GH_TOKEN: \$\{\{ github\.token \}\}", step), (
        "the release-existence probe must authenticate with the job "
        "token via GH_TOKEN"
    )
    assert re.search(r"GH_REPO: \$\{\{ github\.repository \}\}", step), (
        "the release-existence probe must name its repository explicitly "
        "via GH_REPO; persist-credentials: false removed any implicit "
        "origin-remote inference"
    )
    run = _required_run_block(
        _release_workflow_text(),
        "Abort if the GitHub Release already exists",
    )
    assert '-H "Authorization: Bearer ${GH_TOKEN}"' in run, (
        "the probe must authenticate with the job token through the "
        "GH_TOKEN environment reference in its Authorization header"
    )
    assert 'Authorization: Bearer ***' not in run, (
        "the probe must never carry a literal placeholder in its "
        "Authorization header: only the shell environment reference "
        "resolves to the real token at run time"
    )


def test_every_validation_step_precedes_the_single_creation_step() -> None:
    text = _release_workflow_text()
    names = _step_names(text)
    assert len(names) >= 6, (
        "the workflow must separate validation steps from creation: "
        f"found only {len(names)} named steps"
    )
    assert CREATION_STEP_NAME in names, (
        f"the workflow must have a single final step named {CREATION_STEP_NAME!r}"
    )
    creation_index = names.index(CREATION_STEP_NAME)
    assert creation_index == len(names) - 1, (
        "creation must be the last step: every validation runs before "
        "anything is created (unconditional ordering)"
    )
    abort_steps = [name for name in names if name.startswith(("Abort", "abort"))]
    assert len(abort_steps) >= 4, (
        "the unconditional checks (main-head equality, existing tag, "
        "existing release, consistency) must each be explicit steps"
    )
    for name in abort_steps:
        assert names.index(name) < creation_index, (
            f"validation step {name!r} must precede the creation step"
        )
    assert text.count("gh release create") == 1, (
        "exactly one creation action exists: tag and release are "
        "created together, never manually split"
    )


def test_creation_is_one_gh_release_create_with_target_title_and_notes() -> None:
    assert CREATION_COMMAND in _release_workflow_text(), (
        "creation must be the single command "
        f"{CREATION_COMMAND!r} so the tag and its release are never "
        "manually split"
    )


def test_create_step_holds_pinpoint_authority_and_one_command() -> None:
    """Creation is one command naming its repo and token explicitly.

    With persist-credentials: false the persisted origin remote and
    credential are gone, so the creation step must not rely on implicit
    repository inference: it sets GH_TOKEN and GH_REPO explicitly, and
    its run block is exactly one command — the pinned
    ``gh release create``. An appended second command (an alternative
    creation vector) fails this test.
    """
    text = _release_workflow_text()
    step = _step_yaml(text, CREATION_STEP_NAME)
    assert re.search(r"GH_TOKEN: \$\{\{ github\.token \}\}", step), (
        "the creation step must explicitly set GH_TOKEN: ${{ github.token }}"
    )
    assert re.search(r"GH_REPO: \$\{\{ github\.repository \}\}", step) or (
        "--repo" in step
    ), (
        "the creation step must name its repository explicitly "
        "(GH_REPO: ${{ github.repository }} or --repo), never implicit "
        "origin-remote inference"
    )
    run = _required_run_block(text, CREATION_STEP_NAME)
    commands = [line.strip() for line in run.splitlines() if line.strip()]
    mutating_commands = [c for c in commands if c != "set -euo pipefail"]
    assert mutating_commands == [CREATION_COMMAND], (
        "the creation step must run exactly one command — "
        f"{CREATION_COMMAND!r}; found {commands!r}"
    )


def test_only_the_creation_step_runs_a_mutating_command() -> None:
    """Every step before creation is read-only; creation alone mutates.

    Kills the "gutted release-exists guard" and "alternative creation
    vector" mutation classes: no step other than creation may push,
    create tags, add remotes, or mutate via gh api/curl.
    """
    text = _release_workflow_text()
    mutating_steps = [
        name
        for name in _step_names(text)
        if re.search(MUTATING_COMMAND_PATTERN, _step_run_block(text, name))
    ]
    assert mutating_steps == [CREATION_STEP_NAME], (
        "the creation step must be the only step running a repository "
        f"mutating command; found mutating steps {mutating_steps}"
    )
    for name in _step_names(text):
        if name == CREATION_STEP_NAME:
            continue
        run = _step_run_block(text, name)
        assert not re.search(r"\bgit config\b.*credential", run), (
            f"step {name!r} must not re-enable stored credentials"
        )


def test_workflow_defines_no_alternative_creation_vector() -> None:
    """No mutation route other than the single release creation exists.

    ``gh api`` is forbidden anywhere (an API-level creation route would
    bypass the single-release-action guarantee), and curl may appear
    only as the read-only status probe — never with request bodies or
    mutating methods.
    """
    text = _release_workflow_text()
    assert not re.search(r"\bgh api\b", text), (
        "gh api is not permitted anywhere in the release workflow: it "
        "would open an alternative creation vector"
    )
    assert not re.search(r"\bcurl\b.*\s(-X\s+(POST|PATCH|PUT|DELETE)|--request|--data\b|-d\b)", text), (
        "curl may only serve as the read-only release-existence status "
        "probe; no mutating methods or request bodies are permitted"
    )
    assert not re.search(r"\bgit remote add\b", text), (
        "the workflow must not widen repository access by adding remotes"
    )


def test_workflow_never_pushes_deletes_moves_or_overwrites_anything() -> None:
    text = _release_workflow_text()
    assert not re.search(r"\bgit push\b", text), (
        "the workflow must never push refs; creation happens only through "
        "the single release-creation action"
    )
    assert not re.search(r"\bgit tag\b(?! --list)", text), (
        "no manual tag creation is permitted; tags may only be listed "
        "(read-only) before the single creation action"
    )
    for forbidden in (
        "--force",
        "--clobber",
        "--overwrite",
        "--delete",
        ":refs/tags",
        "gh release delete",
        "gh release edit",
    ):
        assert forbidden not in text, (
            f"{forbidden!r} is forbidden: the mechanism never deletes, "
            "moves, or overwrites an existing tag or release"
        )


def test_shell_steps_set_strict_error_handling() -> None:
    """Guard and query steps begin with set -euo pipefail.

    ``set -e`` alone does not fire inside if-conditions or ||-chains;
    pipefail is required so a failed pipeline member (for example the
    tag-collection pipe) can never be read as success.
    """
    text = _release_workflow_text()
    for step_name in (
        "abort unless the target commit is the current origin main head",
        "Abort if the tag already exists locally or on the remote",
        "Abort if the GitHub Release already exists",
        "Collect existing public version tags",
        "Abort on any release-consistency mismatch",
    ):
        run = _required_run_block(text, step_name)
        assert run.splitlines()[0].strip() == "set -euo pipefail", (
            f"step {step_name!r} must begin with 'set -euo pipefail' "
            "(strict error handling for guard and query steps)"
        )


def test_collect_tags_step_stays_read_only() -> None:
    run = _required_run_block(
        _release_workflow_text(), "Collect existing public version tags"
    )
    assert 'git tag --list "v*"' in run, (
        "the tag collection must list existing tags read-only"
    )
    assert not re.search(r"\bgit tag\b(?! --list)", run), (
        "the tag collection step must never create tags"
    )
    assert "existing_tags.txt" in run, (
        "the collected tags must land in the file the validation module reads"
    )


def test_workflow_documents_the_lightweight_tag_design_decision() -> None:
    comments = "\n".join(
        line
        for line in _release_workflow_text().splitlines()
        if line.lstrip().startswith("#")
    )
    for keyword in ("lightweight", "annotated", "partial"):
        assert keyword in comments, (
            "the workflow must document (in a comment) that it "
            f"intentionally selects the release-created lightweight tag "
            f"over a separate annotated tag, and why ({keyword!r} is "
            "missing from the workflow comments)"
        )


def test_workflow_comments_state_the_intended_path_and_real_tradeoff() -> None:
    """The design comments describe intent and risk, not authority.

    The header describes the workflow as the *intended* release path —
    not as an authorization ruling — and the tag-design comment
    describes the single ``gh release create --target`` operation as a
    partial-state risk reduction, not as an atomic API transaction
    (no such transactional guarantee exists across the REST calls).
    """
    comments = "\n".join(
        line
        for line in _release_workflow_text().splitlines()
        if line.lstrip().startswith("#")
    )
    assert re.search(r"intended path|intended release path", comments), (
        "the workflow comments must present the mechanism as the intended "
        "release path"
    )
    assert "only authorized way" not in comments, (
        "the workflow comments must describe the intended path, not an "
        "authorization ruling"
    )
    assert "atomic API" not in comments, (
        "the workflow comments must not claim an atomic API transaction: "
        "a single gh release create --target operation reduces "
        "partial-state risk but is not transactional"
    )
    assert re.search(r"reduces? partial-state risk|reduce partial-state risk", comments), (
        "the tag-design comment must state that the single "
        "gh release create --target operation reduces partial-state risk"
    )


# ---------------------------------------------------------------------------
# Embedded validation module: offline, pure, unconditional
# ---------------------------------------------------------------------------


def test_embedded_validation_module_is_offline_and_pure() -> None:
    text = _release_workflow_text()
    block = _extract_validation_block(text)
    ast.parse(block)  # the embedded module must be standalone-valid Python
    assert not re.search(
        r"\b(subprocess|socket|urllib|requests|http|ftp|git|gh)\b", block
    ), (
        "the embedded validation module must not touch the network or any "
        "repository command: it is executed offline by these tests"
    )
    module = _validation_module()
    for name in ALLOWED_MODULE_CALLABLES:
        assert name in module, f"the validation module must define {name!r}"


def test_embedded_module_defines_no_creation_entry_point() -> None:
    module = _validation_module()
    defined = {
        name
        for name, value in module.items()
        if callable(value)
        and getattr(value, "__module__", None) == VALIDATION_MODULE_NAME
    }
    assert defined <= ALLOWED_MODULE_CALLABLES, (
        "the embedded module may define pure validation callables only; "
        f"found unexpected callables {sorted(defined - ALLOWED_MODULE_CALLABLES)}"
    )


def test_module_cli_wires_argv_and_notes_file_behaviorally(
    tmp_path: Path,
) -> None:
    """The embedded module's CLI behaves as the workflow invokes it.

    The workflow execs the module with argparse flags and reads back the
    notes file; success, the failure exit code, and the notes content
    are exercised offline in a real subprocess — the same contract the
    runner relies on.
    """
    script = tmp_path / "release_validation.py"
    script.write_text(
        _extract_validation_block(_release_workflow_text()), encoding="utf-8"
    )
    pyproject, changelog = _write_release_fixture(tmp_path)
    existing = tmp_path / "existing_tags.txt"
    existing.write_text("v0.0.0\n", encoding="utf-8")
    notes = tmp_path / "release-notes.md"

    passed = _run_module_cli(
        script,
        [
            "--version", "0.0.1",
            "--tag-name", "v0.0.1",
            "--target-sha", "a" * 40,
            "--pyproject", str(pyproject),
            "--changelog", str(changelog),
            "--existing-tags-file", str(existing),
            "--notes-out", str(notes),
        ],
    )
    assert passed.returncode == 0, (
        f"a consistent CLI invocation must exit 0; stderr: {passed.stderr!r}"
    )
    assert "release validation passed" in passed.stdout
    assert notes.is_file(), "a passing CLI run must write the notes file"
    assert "Initial public release" in notes.read_text(encoding="utf-8"), (
        "the notes file must carry the changelog release section"
    )

    failed = _run_module_cli(
        script,
        [
            "--version", "9.9.9",  # deliberately mismatched dry input
            "--tag-name", "v9.9.9",
            "--target-sha", "b" * 40,
            "--pyproject", str(pyproject),
            "--changelog", str(changelog),
            "--existing-tags-file", str(existing),
            "--notes-out", str(notes),
        ],
    )
    assert failed.returncode == 1, (
        "a mismatched CLI invocation must exit nonzero (unconditional)"
    )
    assert "release validation failed" in failed.stderr, (
        f"the CLI must report the reason on stderr; got {failed.stderr!r}"
    )
    assert "9.9.9" in failed.stderr and "0.0.1" in failed.stderr


def test_module_cli_rejects_incomplete_invocations(tmp_path: Path) -> None:
    """Argparse wiring: an incomplete invocation fails closed."""
    script = tmp_path / "release_validation.py"
    script.write_text(
        _extract_validation_block(_release_workflow_text()), encoding="utf-8"
    )
    missing = _run_module_cli(script, ["--version", "0.0.1"])
    assert missing.returncode != 0, (
        "an incomplete CLI invocation must exit nonzero (argparse error)"
    )
    assert "required" in missing.stderr, (
        f"argparse must name the missing required flags; got {missing.stderr!r}"
    )


def test_consistent_dry_input_validates_and_returns_the_changelog_notes(
    tmp_path: Path,
) -> None:
    module = _validation_module()
    pyproject, changelog = _write_release_fixture(tmp_path)
    notes = module["validate_release"](
        version="0.0.1",
        tag_name="v0.0.1",
        target_sha="a" * 40,
        pyproject_path=pyproject,
        changelog_path=changelog,
        existing_version_tags=["v0.0.0"],
    )
    assert "Initial public release" in notes, (
        "validation must return the changelog release section as the "
        "release notes"
    )


def test_mismatched_dry_input_fails_closed_before_any_creation(
    tmp_path: Path,
) -> None:
    """The deliberately-mismatched dry run.

    A requested version that does not match the version authority at the
    target commit must abort with the reason, offline, before anything
    is created — and the validation module is provably incapable of
    creating anything (no creation entry point exists).
    """
    module = _validation_module()
    pyproject, changelog = _write_release_fixture(tmp_path)  # declares 0.0.1
    with pytest.raises(module["ReleaseValidationError"]) as caught:
        module["validate_release"](
            version="9.9.9",  # deliberately mismatched dry input
            tag_name="v9.9.9",
            target_sha="b" * 40,
            pyproject_path=pyproject,
            changelog_path=changelog,
            existing_version_tags=["v0.0.1"],
        )
    message = str(caught.value)
    assert "does not equal" in message, (
        f"the failure must report the version mismatch; got {message!r}"
    )
    assert "9.9.9" in message and "0.0.1" in message
    assert not (tmp_path / "release-notes.md").exists(), (
        "a failed validation must not even write the release-notes file"
    )


@pytest.mark.parametrize(
    ("declared", "version", "tag_name", "target_sha", "existing", "reason"),
    [
        ("0.0.1", "0.0.1", "v0.0.2", "a" * 40, [], "must equal"),
        ("0.0.1", "0.0.1", "v0.0.1", "abc123", [], "40-character"),
        ("0.0.1", "0.0.2", "v0.0.2", "a" * 40, [], "does not equal"),
        ("0.0.1", "0.0.1", "v0.0.1", "a" * 40, ["v0.0.1"], "monotone-forward"),
        ("0.0.0", "0.0.0", "v0.0.0", "a" * 40, ["v0.0.1"], "monotone-forward"),
        ("0.0.1.hotdog", "0.0.1.hotdog", "v0.0.1.hotdog", "a" * 40, [], "PEP 440"),
    ],
)
def test_each_rule_violation_fails_closed_with_its_reason(
    tmp_path: Path,
    declared: str,
    version: str,
    tag_name: str,
    target_sha: str,
    existing: list[str],
    reason: str,
) -> None:
    module = _validation_module()
    pyproject, changelog = _write_release_fixture(tmp_path, declared=declared)
    with pytest.raises(module["ReleaseValidationError"]) as caught:
        module["validate_release"](
            version=version,
            tag_name=tag_name,
            target_sha=target_sha,
            pyproject_path=pyproject,
            changelog_path=changelog,
            existing_version_tags=existing,
        )
    assert reason in str(caught.value), (
        f"expected the {reason!r} rule to unconditional; got {caught.value!r}"
    )


def test_current_development_version_is_rejected_fail_closed(
    tmp_path: Path,
) -> None:
    """The version the package currently declares (0.0.1.dev0) is rejected.

    ``0.0.1.dev0`` is the development state the repository declares
    today; requesting it as a release must unconditionally with the
    development-release reason, offline, before anything is created.
    """
    module = _validation_module()
    pyproject, changelog = _write_release_fixture(
        tmp_path, declared="0.0.1.dev0"
    )
    with pytest.raises(module["ReleaseValidationError"]) as caught:
        module["validate_release"](
            version="0.0.1.dev0",
            tag_name="v0.0.1.dev0",
            target_sha="a" * 40,
            pyproject_path=pyproject,
            changelog_path=changelog,
        )
    message = str(caught.value)
    assert "development" in message, (
        f"the dev-segment rule must name development releases; got {message!r}"
    )
    assert "0.0.1.dev0" in message
    assert not (tmp_path / "release-notes.md").exists(), (
        "a failed validation must not even write the release-notes file"
    )


def test_valid_non_dev_prerelease_is_not_rejected(tmp_path: Path) -> None:
    """A pre-release version such as ``0.0.1rc1`` fails validation.

    The mechanism creates GitHub Releases without the prerelease flag,
    so a pre-release version must unconditionally with the
    prerelease-deferred reason rather than be mislabelled as a full
    release. ``test_prerelease_versions_are_deferred_fail_closed``
    below covers every PEP 440 pre-release phase.
    """
    module = _validation_module()
    pyproject, changelog = _write_release_fixture(
        tmp_path, declared="0.0.1rc1"
    )
    with pytest.raises(module["ReleaseValidationError"]) as caught:
        module["validate_release"](
            version="0.0.1rc1",
            tag_name="v0.0.1rc1",
            target_sha="a" * 40,
            pyproject_path=pyproject,
            changelog_path=changelog,
            existing_version_tags=["v0.0.0"],
        )
    message = str(caught.value)
    assert "prerelease" in message, (
        "a pre-release version must be rejected with the prerelease-"
        f"deferred reason; got {message!r}"
    )
    assert "0.0.1rc1" in message
    assert not (tmp_path / "release-notes.md").exists(), (
        "a failed validation must not even write the release-notes file"
    )


def test_missing_changelog_release_section_fails_closed(tmp_path: Path) -> None:
    module = _validation_module()
    pyproject, changelog = _write_release_fixture(
        tmp_path, declared="0.0.1", section_version="0.0.7"
    )
    with pytest.raises(module["ReleaseValidationError"], match="no release section"):
        module["validate_release"](
            version="0.0.1",
            tag_name="v0.0.1",
            target_sha="a" * 40,
            pyproject_path=pyproject,
            changelog_path=changelog,
        )


def test_empty_release_notes_fail_closed(tmp_path: Path) -> None:
    module = _validation_module()
    pyproject, changelog = _write_release_fixture(
        tmp_path, release_entries=()
    )
    with pytest.raises(module["ReleaseValidationError"], match="must not be empty"):
        module["validate_release"](
            version="0.0.1",
            tag_name="v0.0.1",
            target_sha="a" * 40,
            pyproject_path=pyproject,
            changelog_path=changelog,
        )


def test_non_empty_unreleased_section_fails_closed(tmp_path: Path) -> None:
    module = _validation_module()
    pyproject, changelog = _write_release_fixture(
        tmp_path,
        unreleased_entries=("An unfinished entry that must move first.",),
    )
    with pytest.raises(
        module["ReleaseValidationError"], match=r"\[Unreleased\].*not empty"
    ):
        module["validate_release"](
            version="0.0.1",
            tag_name="v0.0.1",
            target_sha="a" * 40,
            pyproject_path=pyproject,
            changelog_path=changelog,
        )


def test_prose_under_unreleased_fails_closed(tmp_path: Path) -> None:
    """[Unreleased] must be empty, not merely bullet-free.

    Non-heading prose under [Unreleased] is pending material exactly
    like a bullet; only empty Keep-a-Changelog subsection headings are
    allowed to remain.
    """
    module = _validation_module()
    pyproject, changelog = _write_release_fixture(
        tmp_path,
        unreleased_prose="Draft narrative that has not moved into a release section yet.",
    )
    with pytest.raises(
        module["ReleaseValidationError"], match=r"\[Unreleased\].*not empty"
    ):
        module["validate_release"](
            version="0.0.1",
            tag_name="v0.0.1",
            target_sha="a" * 40,
            pyproject_path=pyproject,
            changelog_path=changelog,
        )


def test_empty_unreleased_subsection_headings_remain_allowed(
    tmp_path: Path,
) -> None:
    """Empty '###' subsection headings under [Unreleased] are allowed.

    Keep-a-Changelog files conventionally carry empty subsection
    headings under [Unreleased]; they are structure, not pending
    content, and must not trip the emptiness rule.
    """
    module = _validation_module()
    pyproject, changelog = _write_release_fixture(tmp_path)
    notes = module["validate_release"](
        version="0.0.1",
        tag_name="v0.0.1",
        target_sha="a" * 40,
        pyproject_path=pyproject,
        changelog_path=changelog,
        existing_version_tags=["v0.0.0"],
    )
    assert "Initial public release" in notes


@pytest.mark.parametrize(
    "prerelease_version",
    ["0.0.1rc1", "0.0.1rc2", "0.0.1a1", "0.0.1b2", "1.0.0rc1"],
)
def test_prerelease_versions_are_deferred_fail_closed(
    tmp_path: Path, prerelease_version: str
) -> None:
    """Pre-release versions are rejected unconditionally.

    The controlled release mechanism does not create prerelease-marked
    GitHub Releases, so ``a``/``b``/``rc`` segments must unconditional
    exactly like ``.devN`` does: prerelease support is deferred until
    the creation step can label releases correctly. Every representative
    PEP 440 pre-release phase is covered.
    """
    module = _validation_module()
    pyproject, changelog = _write_release_fixture(
        tmp_path, declared=prerelease_version
    )
    with pytest.raises(module["ReleaseValidationError"]) as caught:
        module["validate_release"](
            version=prerelease_version,
            tag_name=f"v{prerelease_version}",
            target_sha="a" * 40,
            pyproject_path=pyproject,
            changelog_path=changelog,
            existing_version_tags=["v0.0.0"],
        )
    message = str(caught.value)
    assert "prerelease" in message, (
        "the pre-release deferral rule must name prereleases; got "
        f"{message!r}"
    )
    assert prerelease_version in message
    assert not (tmp_path / "release-notes.md").exists(), (
        "a failed validation must not even write the release-notes file"
    )


def test_final_versions_are_not_swept_into_prerelease_rejection(
    tmp_path: Path,
) -> None:
    """Control: final versions must still validate (no over-rejection).

    The deferral must reject exactly pre-release and development
    versions. A final release such as ``0.0.1`` carries no ``a``/``b``/
    ``rc``/``.devN`` segment and must validate exactly as before, so
    the rejection rule cannot have swept final releases in.
    """
    module = _validation_module()
    pyproject, changelog = _write_release_fixture(tmp_path)
    notes = module["validate_release"](
        version="0.0.1",
        tag_name="v0.0.1",
        target_sha="a" * 40,
        pyproject_path=pyproject,
        changelog_path=changelog,
        existing_version_tags=["v0.0.0"],
    )
    assert "Initial public release" in notes, (
        "a final version must validate and return the changelog release "
        "section as the release notes"
    )
