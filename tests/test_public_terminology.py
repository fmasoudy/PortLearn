"""Terminology floor: public-facing surfaces carry no internal labels.

Static regression floor pinning the public-facing text surfaces of the
package -- README/CHANGELOG prose, shipped example comments and prose,
module docstrings/literals, and user-facing runtime message literals --
to the public vocabulary ratified for the 0.0.1.dev4 release surface.

Module surfaces are scanned for docstrings and non-docstring string
literals (comments excluded); example surfaces are scanned for
docstrings, string literals, and comments. Output-token literals that
the wire format freezes (``[a]/[p]/[r]/[s] arm``) are exempt.
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

README = REPO / "README.md"
CHANGELOG = REPO / "CHANGELOG.md"
MODULE_SURFACES = sorted((REPO / "src" / "portlearn").glob("*.py"))
EXAMPLE_SURFACES = sorted(
    p for p in (REPO / "examples").glob("*.py") if p.name != "__init__.py"
)

# Public-interface prose bans: internal specification labels (decision
# ids, law numbers, section markers, frozen-spec vocabulary) and the
# internal-only metaphors (seam, arm as prose, weight book).
_BANNED_PATTERNS = [
    ("decision label", re.compile(r"\bD\d+[a-z]?(?:/(?:D\d+[a-z]?|[A-Z]\d+))*\b")),
    ("law label", re.compile(r"\bLaw-[0-9]|Laws? \d|\bLaws 3-7\b")),
    ("section marker", re.compile(r"§§?[0-9]")),
    (
        "frozen-spec vocabulary",
        re.compile(
            r"specification[ -]approval|frozen challenge vocabulary|frozen PortLearn|challenge vocabulary"
        ),
    ),
    ("internal metaphor", re.compile(r"\bseam\b")),
    ("arm prose", re.compile(r"\barm\b|\barms\b")),
    ("weight book", re.compile(r"\bweight books?\b")),
    ("run-of-record", re.compile(r"run-of-record")),
]

_PRESERVED_LITERAL_MARKERS = (
    "[a] arm",
    "[p] arm",
    "[r] arm",
    "[s] arm",
)


def _matches(text, patterns):
    hits = []
    for name, pattern in patterns:
        found = pattern.findall(text)
        if found:
            hits.append(f"{name}: {found[:6]}")
    return hits


def _docstrings(source):
    import ast

    tree = ast.parse(source)
    out = []
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                out.append(doc)
    return out


def _string_literals(source):
    """Non-docstring literals as (text, token_type) pairs
    (tokenize STRING/FSTRING_MIDDLE)."""
    lits = []
    for t in tokenize.generate_tokens(io.StringIO(source).readline):
        if t.type in (tokenize.STRING, tokenize.FSTRING_MIDDLE):
            lits.append((t.string, t.type))
    return lits, len(lits)


def _comments(source):
    return [
        t.string
        for t in tokenize.generate_tokens(io.StringIO(source).readline)
        if t.type == tokenize.COMMENT
    ]


# Wire/API vocabulary exemption (owner adjudication): the public field
# name and serialized comparison-table key is the exact string "arms".
# A standalone string literal whose evaluated value is exactly "arms"
# is wire vocabulary, not prose, and is exempt from the arm-prose ban
# only. Anything more than the bare key -- substrings, embedded prose,
# f-string middles, non-string constants -- remains fully banned.
_ARM_PROSE_PATTERN_NAME = "arm prose"
_WIRE_KEY_VALUE = "arms"

# Internal specification vocabulary (final public-terminology amendment):
# the standalone words "frozen", "law"/"laws", and "lawful"/"unlawful" are
# internal contract-specification labels. Public module docstrings and
# user-visible runtime string literals must state the same facts as direct
# researcher-facing prose (``supported values``, ``rule``, ``validation
# rule``, ``valid``, ...). The floor is objective: standalone means
# whole-word, case-insensitive occurrences only -- substring carriers such
# as "lawfully", "flaws", "bylaws", or "refrozen" are different words, and
# Python syntax/metadata (for example a ``@dataclass(frozen=True)`` flag)
# is not a string and is never scanned. This floor never broadens the
# arm/arms wire-key or frozen render-token exemptions: those strip only
# the arm-prose pattern and this vocabulary adds no arm-related pattern.
_MODULE_INTERNAL_VOCABULARY = [
    ("internal vocabulary: frozen", re.compile(r"\bfrozen\b", re.IGNORECASE)),
    ("internal vocabulary: law", re.compile(r"\blaws?\b", re.IGNORECASE)),
    (
        "internal vocabulary: lawful",
        re.compile(r"\b(?:un)?lawful\b", re.IGNORECASE),
    ),
]

_MODULE_BANNED_PATTERNS = _BANNED_PATTERNS + _MODULE_INTERNAL_VOCABULARY


def _is_exact_wire_key_literal(literal):
    """True only for a standalone str literal evaluating to "arms"."""
    import ast

    try:
        node = ast.parse(literal, mode="eval").body
    except (SyntaxError, ValueError):
        return False
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return False
    return node.value == _WIRE_KEY_VALUE


# Frozen render-token exemption (owner adjudication): the deterministic
# comparison-table renderer emits the per-strategy section header
# f"arm {strategy_id}", whose f-string middle token is exactly "arm ".
# That output token is frozen behavior, not prose: it is exempt from the
# arm-prose ban only, and only when the scanned path relative to the
# repository root is the single module that renders it. Substrings,
# grown variants, other paths, docstrings, comments, and ordinary string
# literals remain fully banned.
_RENDER_TOKEN_VALUE = "arm "
_RENDER_TOKEN_PATH = "src/portlearn/comparisons.py"


def _is_frozen_render_middle(literal, token_type, rel_path):
    """True only for the FSTRING_MIDDLE token "arm " in comparisons.py."""
    return (
        token_type == tokenize.FSTRING_MIDDLE
        and literal == _RENDER_TOKEN_VALUE
        and rel_path == _RENDER_TOKEN_PATH
    )


def _scan_python_file(path, prose_patterns, include_comments):
    """Scan docstrings, string literals, and (for examples) comments."""
    source = path.read_text(encoding="utf-8")
    try:
        rel_path = path.relative_to(REPO).as_posix()
    except ValueError:
        rel_path = ""
    docstring_list = _docstrings(source)
    literals, literal_count = _string_literals(source)

    violations = []
    for hit in _matches("\n".join(docstring_list), prose_patterns):
        violations.append(f"{path}: docstrings: {hit}")
    comment_count = 0
    if include_comments:
        comment_list = _comments(source)
        comment_count = len(comment_list)
        for hit in _matches("\n".join(comment_list), prose_patterns):
            violations.append(f"{path}: comments: {hit}")
    for literal, token_type in literals:
        if any(marker in literal for marker in _PRESERVED_LITERAL_MARKERS):
            continue
        patterns = prose_patterns
        if _is_exact_wire_key_literal(literal) or _is_frozen_render_middle(
            literal, token_type, rel_path
        ):
            patterns = [
                (name, pattern)
                for name, pattern in prose_patterns
                if name != _ARM_PROSE_PATTERN_NAME
            ]
        for hit in _matches(literal, patterns):
            violations.append(f"{path}: string literal {literal!r}: {hit}")
    return violations, len(docstring_list), literal_count, comment_count


def test_scanned_surfaces_exist_and_are_pinned():
    """Non-vacuity: every Python surface has docstrings and literals;
    comments are required only for example surfaces."""
    for path in MODULE_SURFACES + EXAMPLE_SURFACES:
        source = path.read_text(encoding="utf-8")
        docstring_list = _docstrings(source)
        literals, _ = _string_literals(source)
        assert docstring_list, f"{path}: no docstrings to scan"
        assert literals, f"{path}: no non-docstring literals to scan"
    for path in EXAMPLE_SURFACES:
        source = path.read_text(encoding="utf-8")
        assert _comments(source), f"{path}: no comments to scan"


def test_readme_and_changelog_prose_is_public():
    violations = []
    for path in (README, CHANGELOG):
        text = path.read_text(encoding="utf-8")
        for hit in _matches(text, _BANNED_PATTERNS):
            violations.append(f"{path.name}: {hit}")
    assert not violations, "README/CHANGELOG prose carries internal labels"


def test_example_prose_is_public():
    violations = []
    for path in EXAMPLE_SURFACES:
        file_violations, _, _, _ = _scan_python_file(
            path, _BANNED_PATTERNS, include_comments=True
        )
        violations.extend(file_violations)
    assert not violations, "example prose carries internal labels"


def test_module_prose_and_runtime_literals_are_public():
    violations = []
    for path in MODULE_SURFACES:
        file_violations, _, _, _ = _scan_python_file(
            path, _MODULE_BANNED_PATTERNS, include_comments=False
        )
        violations.extend(file_violations)
    assert not violations, "module prose/runtime literals carry internal labels"


def test_module_internal_specification_vocabulary_is_absent():
    """The module surface carries no internal specification vocabulary.

    Beyond the long-standing public-interface prose bans, public module
    docstrings and non-comment runtime string literals must not use the
    internal specification words ``frozen``, ``law``/``laws``, or
    ``lawful``/``unlawful`` as standalone (whole-word, case-insensitive)
    terms. Comments stay excluded, Python syntax/metadata is never a
    string, and the arm/arms wire-key and frozen render-token exemptions
    are untouched: this floor adds vocabulary patterns only and never
    widens any exemption.
    """
    violations = []
    for path in MODULE_SURFACES:
        file_violations, _, _, _ = _scan_python_file(
            path, _MODULE_INTERNAL_VOCABULARY, include_comments=False
        )
        violations.extend(file_violations)
    assert not violations, (
        "module prose/runtime literals carry internal specification "
        "vocabulary (frozen/law/laws/lawful/unlawful)"
    )


def test_frozen_render_token_exemption_is_exact():
    """The render-token exemption accepts exactly one path/value pair.

    The frozen comparison-table section header f-string middle ``arm `` is
    exempt from the arm-prose ban only in the module that renders it. The
    same value on any other path, grown variants, other tokens, and
    ordinary string literals stay banned, and the normal floor keeps
    scanning every public module.
    """
    render_path = "src/portlearn/comparisons.py"
    # Accepts exactly the one path/value pair.
    assert _is_frozen_render_middle(
        _RENDER_TOKEN_VALUE, tokenize.FSTRING_MIDDLE, render_path
    )
    # Rejects the same value on any other path.
    assert not _is_frozen_render_middle(
        _RENDER_TOKEN_VALUE, tokenize.FSTRING_MIDDLE, "src/portlearn/metrics.py"
    )
    # Rejects grown or different f-string middles.
    assert not _is_frozen_render_middle(
        "arm extra", tokenize.FSTRING_MIDDLE, render_path
    )
    assert not _is_frozen_render_middle(
        "strategy ", tokenize.FSTRING_MIDDLE, render_path
    )
    # Rejects an ordinary STRING literal carrying the same text.
    assert not _is_frozen_render_middle('"arm "', tokenize.STRING, render_path)
    # End-to-end non-vacuity: the rendering module itself scans clean only
    # because the exemption covers its frozen header token.
    violations, _, literal_count, _ = _scan_python_file(
        REPO / render_path, _BANNED_PATTERNS, include_comments=False
    )
    assert literal_count > 0
    assert not violations
    # The normal floor still scans all public modules.
    assert MODULE_SURFACES and all(path.is_file() for path in MODULE_SURFACES)
