"""The wiring table must be DERIVED from the code, not hand-maintained.

A hand-maintained checklist is itself a declaration, and a
declaration-vs-declaration artifact is what this entire audit has been about.
So the required control set is DERIVED by walking the production entry points
and is asserted to be a SUBSET of the table's rows: add a control call to a
production path with no table row and this goes RED.

That is the auditor's pre-registered attack, and it is asserted directly in
``test_a_new_production_control_without_a_row_is_rejected`` rather than left as
a property someone has to trust.

NOTE FOR ANYONE MUTATION-TESTING THIS FILE: it reads source by ABSOLUTE REPO
PATH (``parents[2] / "src" / "helix"``), NOT through an import. A harness that
copies ``src/`` to a temp tree and points ``PYTHONPATH`` at the copy will not
exercise it -- the test keeps reading the real repo and keeps passing, which
looks like a decorative test rather than a working one. Mutate in a FULL-REPO
sandbox instead. (An auditor nearly filed a P1 against this file for exactly
that reason, and caught it only because three greens were implausible against a
file containing ``test_a_new_production_control_without_a_row_is_rejected``.)

STATED LIMITATION, because the derivation is a heuristic and pretending
otherwise would be the same error one level up: a control is recognised by
NAME (``assert_*`` / ``_assert_*``) or by being imported from one of the
control modules below. A control that is neither -- say a bare
``validate_thing()`` defined inline in ``sandbox.py`` -- would not be derived.
The heuristic is checked in both directions (it must find the controls we know
about, and it must reject an added one), but it cannot be complete.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "helix"
TABLE = REPO_ROOT / "docs" / "design" / "sandbox-home-isolation.md"

# Production entry points whose bodies constitute "the run path".
#
# LIMITATION, stated because this list is itself a hand-maintained declaration
# and pretending otherwise would repeat the defect one level up: a control
# invoked from a function NOT listed here is invisible to the derivation. That
# is not hypothetical -- ``assert_volume_subpath_supported`` was reported as an
# unwired stale row purely because ``preflight_auth`` was missing from this
# dict. The two directional tests below (derived-has-row, row-is-called) catch
# a control added to a LISTED path; they cannot catch a whole new entry point.
# Adding one is a deliberate act that should include adding it here.
_ENTRY_POINTS = {
    "sandbox.py": ("_docker_args", "run_sandboxed_commands"),
    "evolution.py": ("run_evolution",),
    "cli.py": ("_ensure_auth_subpath",),
    "authpreflight.py": ("preflight_auth",),
}

# Modules whose exported functions count as release controls.
_CONTROL_MODULES = (
    "helix.transcripts",
    "helix.sandbox_home",
    "helix.subpath_bootstrap",
    "helix.backend_layout",
    "helix.authpreflight",
)


def _imported_control_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _CONTROL_MODULES:
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def _called_names(func: ast.FunctionDef) -> set[str]:
    called: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                called.add(target.id)
            elif isinstance(target, ast.Attribute):
                called.add(target.attr)
    return called


def derive_required_controls() -> set[str]:
    """Every control invoked on a production entry point."""
    required: set[str] = set()
    for filename, functions in _ENTRY_POINTS.items():
        path = SRC / filename
        tree = ast.parse(path.read_text())
        control_imports = _imported_control_names(tree)
        wanted = {
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in functions
        }
        for func in wanted:
            for name in _called_names(func):
                if name.startswith(("assert_", "_assert_")) or name in control_imports:
                    required.add(name)
    return required


def table_rows() -> set[str]:
    """Control names named in the first column of the wiring table."""
    text = TABLE.read_text()
    section = text[text.index("## 8d.") :]
    section = section[: section.index("\n## ", 10)]
    names: set[str] = set()
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        first = line.split("|")[1]
        names.update(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", first))
    return names


def test_the_table_exists_and_has_rows() -> None:
    """Non-vacuity: an empty or moved table must not silently pass everything."""
    rows = table_rows()
    assert len(rows) >= 5, f"wiring table looks empty or moved: {rows}"


def test_derivation_finds_the_controls_we_know_about() -> None:
    """Non-vacuity for the derivation itself.

    If ``derive_required_controls`` returned an empty set, the subset assertion
    below would pass trivially -- the exact empty-set trap that certified
    opencode. So the derivation must positively find known controls.
    """
    derived = derive_required_controls()
    assert derived, "derivation found no controls at all"
    for expected in (
        "_assert_no_shared_home_mount",
        "_assert_agent_argv_uses_only_known_flags",
        "capture_claude_transcript",
        "ensure_transcript_host_dir",
    ):
        assert expected in derived, f"derivation missed {expected}"


def test_every_derived_control_has_a_table_row() -> None:
    """The checklist cannot rot: a new production control needs a row.

    Catches: adding a release control to a production path and leaving the
    wiring table describing the previous state.
    """
    missing = derive_required_controls() - table_rows()
    assert not missing, (
        f"controls invoked on production paths with NO wiring-table row: "
        f"{sorted(missing)}. Add a row naming the production call site and the "
        f"test that fails when the CALL is deleted -- an empty third column is "
        f"an unprotected control, not a documentation gap."
    )


def test_a_new_production_control_without_a_row_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The auditor's pre-registered attack, asserted rather than trusted.

    Simulates adding ``_assert_a_brand_new_control()`` to ``_docker_args`` with
    no table row, and requires the derivation to surface it. Without this, the
    subset check could pass forever simply because the derived set never grows.
    """
    original = (SRC / "sandbox.py").read_text()
    mutated = original.replace(
        '    if scope == "agent":\n        # Order matters: refuse unknown flags FIRST',
        '    if scope == "agent":\n'
        "        _assert_a_brand_new_control(args)\n"
        "        # Order matters: refuse unknown flags FIRST",
        1,
    )
    assert mutated != original, "mutation anchor not found; update this test"

    # Mirror EVERY entry-point file, so adding one to _ENTRY_POINTS cannot make
    # this test fail for an unrelated reason.
    fake_src = tmp_path / "helix"
    fake_src.mkdir()
    for name in _ENTRY_POINTS:
        (fake_src / name).write_text(
            mutated if name == "sandbox.py" else (SRC / name).read_text()
        )
    monkeypatch.setattr(
        "tests.unit.test_wiring_table_is_current.SRC", fake_src, raising=False
    )
    import tests.unit.test_wiring_table_is_current as module

    monkeypatch.setattr(module, "SRC", fake_src)
    derived = module.derive_required_controls()
    assert "_assert_a_brand_new_control" in derived, (
        "the derivation did not notice a newly added production control -- the "
        "checklist would rot silently"
    )
    assert "_assert_a_brand_new_control" not in table_rows()


# ---------------------------------------------------------------------------
# F-23: call sites are LOCATED, never typed
# ---------------------------------------------------------------------------


def locate_call_sites(control: str) -> list[tuple[str, int]]:
    """Every production call site of *control*, found by AST."""
    sites: list[tuple[str, int]] = []
    for filename, functions in _ENTRY_POINTS.items():
        tree = ast.parse((SRC / filename).read_text())
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef) or func.name not in functions:
                continue
            for node in ast.walk(func):
                if not isinstance(node, ast.Call):
                    continue
                target = node.func
                name = (
                    target.id
                    if isinstance(target, ast.Name)
                    else target.attr
                    if isinstance(target, ast.Attribute)
                    else None
                )
                if name == control:
                    sites.append((filename, node.lineno))
    return sites


def test_every_derived_control_is_actually_called_and_guards_are_singular() -> None:
    """Catches a MOVED or DELETED call -- not just a missing row.

    The wiring table previously carried TYPED ``file:line`` references, and
    every ``sandbox.py`` row was wrong in the very commit that introduced them
    -- off by 40 lines, because the table described a pre-commit state of the
    file. One row sent a reader into the retired, unreachable volume-mode
    block.

    So locations are DERIVED here rather than written down. Requiring exactly
    one call site also catches a control being invoked twice, which is how a
    guard silently becomes order-dependent.
    """
    uncalled: dict[str, list[tuple[str, int]]] = {}
    duplicated_guards: dict[str, list[tuple[str, int]]] = {}
    for control in sorted(derive_required_controls()):
        sites = locate_call_sites(control)
        if not sites:
            uncalled[control] = sites
        elif control.startswith(("assert_", "_assert_")) and len(sites) > 1:
            duplicated_guards[control] = sites

    assert not uncalled, (
        f"controls with NO production call site: {sorted(uncalled)} -- derived "
        f"as required but never invoked, which is the unasserted-call defect "
        f"in its purest form."
    )
    # GUARDS must be singular; BUILDERS legitimately are not. `layout_for`
    # serves two distinct paths (the mount guard and the login bootstrap), and
    # `transcript_host_dir` is called once to BUILD the bind argument and once
    # to ENSURE the directory exists before any container starts. Requiring
    # "exactly one" everywhere would have forced one of those to be deleted or
    # hidden behind an alias, which is worse than the property it protects.
    assert not duplicated_guards, (
        f"validation guards invoked more than once: {duplicated_guards}. A "
        f"guard called twice is order-dependent, and the table cannot name a "
        f"single location for it."
    )


def test_the_table_carries_no_typed_line_numbers() -> None:
    """Line numbers must be GENERATED, never typed.

    A typed ``file:line`` in a checklist is a declaration about the artifact,
    and this project's whole finding list is declarations that disagreed with
    artifacts. A table with no line numbers is honest; one with wrong numbers
    actively misleads.

    Catches: someone helpfully re-adding them.
    """
    text = TABLE.read_text()
    section = text[text.index("## 8d.") :]
    section = section[: section.index("\n## ", 10)]
    rows = [line for line in section.splitlines() if line.startswith("|")]
    offenders = [line for line in rows if re.search(r"\.py:\d+", line)]
    assert not offenders, (
        f"wiring table rows carry typed line numbers: {offenders}. "
        f"Locations are derived by locate_call_sites(); state function names "
        f"in the table instead."
    )


def test_the_table_has_no_rows_for_controls_that_are_no_longer_called() -> None:
    """Staleness in the INVERSE direction: a row for a control nothing invokes.

    Found by this test on the commit that introduced it. Deleting the retired
    volume-mode branch removed the only caller of
    ``assert_layout_is_isolatable``, leaving a table row describing it as a
    live control with a wiring assertion. A reader would have believed a
    retired control was still guarding something.

    Rows for retired controls are permitted ONLY when marked as such -- struck
    through and stating they have no call site -- so the table can record what
    was withdrawn without implying it still runs.
    """
    text = TABLE.read_text()
    section = text[text.index("## 8d.") :]
    section = section[: section.index("\n## ", 10)]

    live: set[str] = set()
    for line in section.splitlines():
        if not line.startswith("|") or "~~" in line or "---" in line:
            continue
        cells = line.split("|")
        if len(cells) < 3:
            continue
        names = re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", cells[1])
        if names and "NO CALL SITE" not in cells[2].upper():
            live.update(names)

    unwired = {name for name in live if name and not locate_call_sites(name)}
    assert not unwired, (
        f"wiring table lists these as LIVE controls, but nothing on a "
        f"production path calls them: {sorted(unwired)}. Either wire them, or "
        f"mark the row retired (~~struck~~, 'NO CALL SITE')."
    )
