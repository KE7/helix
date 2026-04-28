"""Unit tests for helix.mutator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from helix.population import Candidate, EvalResult
from helix.config import AgentConfig, HelixConfig, EvaluatorConfig, SandboxConfig
from helix.mutator import (
    MutationError,
    PromptArtifactCollisionError,
    BACKEND_RESULT_ARTIFACT_NAME,
    BACKEND_STDERR_ARTIFACT_NAME,
    BACKEND_STDOUT_ARTIFACT_NAME,
    build_mutation_prompt,
    invoke_claude_code,
    mutate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_eval_result(
    candidate_id: str = "g0-s0",
    scores: dict | None = None,
    asi: dict | None = None,
    instance_scores: dict | None = None,
) -> EvalResult:
    return EvalResult(
        candidate_id=candidate_id,
        scores=scores if scores is not None else {"pass_rate": 0.5},
        asi=asi or {"stdout": "2 passed, 2 failed", "stderr": ""},
        instance_scores=instance_scores or {"test_a": 1.0, "test_b": 0.0},
    )


def make_candidate(
    cid: str = "g0-s0",
    worktree_path: str = "/tmp/fake-wt",
) -> Candidate:
    return Candidate(
        id=cid,
        worktree_path=worktree_path,
        branch_name=f"helix/{cid}",
        generation=0,
        parent_id=None,
        parent_ids=[],
        operation="seed",
    )


def make_config(objective: str = "Pass all tests") -> HelixConfig:
    return HelixConfig(
        objective=objective,
        evaluator=EvaluatorConfig(command="pytest -q"),
    )


# ---------------------------------------------------------------------------
# Tests: build_mutation_prompt
# ---------------------------------------------------------------------------


class TestBuildMutationPrompt:
    def test_contains_objective(self):
        er = make_eval_result()
        prompt = build_mutation_prompt("Optimise sorting", er)
        assert "Optimise sorting" in prompt

    def test_contains_scores(self):
        er = make_eval_result(scores={"pass_rate": 0.75})
        prompt = build_mutation_prompt("goal", er)
        assert "pass_rate" in prompt
        assert "0.75" in prompt

    def test_contains_stdout(self):
        er = make_eval_result(asi={"stdout": "unique_stdout_xyz", "stderr": ""})
        prompt = build_mutation_prompt("goal", er)
        assert "unique_stdout_xyz" in prompt

    def test_contains_stderr(self):
        er = make_eval_result(asi={"stdout": "", "stderr": "unique_stderr_abc"})
        prompt = build_mutation_prompt("goal", er)
        assert "unique_stderr_abc" in prompt

    def test_contains_extra_asi(self):
        er = make_eval_result(asi={"stdout": "", "stderr": "", "extra_0": "coverage: 80%"})
        prompt = build_mutation_prompt("goal", er)
        assert "coverage: 80%" in prompt

    def test_contains_background(self):
        er = make_eval_result()
        prompt = build_mutation_prompt("goal", er, background="special context here")
        assert "special context here" in prompt

    def test_default_background_when_none(self):
        er = make_eval_result()
        prompt = build_mutation_prompt("goal", er, background=None)
        assert "no additional background" in prompt

    def test_contains_mutation_complete_marker(self):
        er = make_eval_result()
        prompt = build_mutation_prompt("goal", er)
        assert "[MUTATION COMPLETE]" in prompt

    def test_contains_execution_instructions(self):
        er = make_eval_result()
        prompt = build_mutation_prompt("goal", er)
        assert "Task instructions:" in prompt

    def test_no_scores_fallback(self):
        er = make_eval_result(scores={})
        prompt = build_mutation_prompt("goal", er)
        assert "no scores recorded" in prompt


class TestPerExampleDiagnostics:
    """``build_mutation_prompt`` renders ``eval_result.per_example_side_info``
    as the Diagnostics section under the new GEPA O.A. contract
    (``optimize_anything_adapter.py:524-553`` + ``format_samples``
    at ``gepa/strategies/instruction_proposal.py:54-95``).  The legacy
    batch-level ``side_info`` rendering is used only when
    ``per_example_side_info`` is absent.
    """

    def _make(
        self,
        per_example_side_info: list[dict] | None,
        side_info: dict | None = None,
        instance_scores: dict | None = None,
    ) -> EvalResult:
        return EvalResult(
            candidate_id="g0-s0",
            scores={"pass_rate": 0.5},
            asi={"stdout": "", "stderr": ""},
            instance_scores=instance_scores or {"cube_lifting__0": 1.0, "cube_lifting__1": 0.0},
            side_info=side_info,
            per_example_side_info=per_example_side_info,
        )

    def test_renders_per_example_headers_from_instance_scores(self):
        er = self._make(
            per_example_side_info=[
                {"trajectory": "fell over", "scores": {"success": 1.0, "affordance": 0.8}},
                {"trajectory": "stuck", "scores": {"success": 0.0, "affordance": 0.2}},
            ],
            instance_scores={"cube_lifting__0": 1.0, "cube_lifting__1": 0.0},
        )
        prompt = build_mutation_prompt("goal", er)
        # Per-example ids render as ``### Example <id>`` (h3 under the
        # surrounding ``## Diagnostics`` h2).  Line-anchored check —
        # a substring match would also accept ``# Example`` or
        # ``#### Example`` and miss the header-level change.
        assert "\n### Example cube_lifting__0\n" in prompt
        assert "\n### Example cube_lifting__1\n" in prompt

    def test_reserved_scores_key_renamed_for_mutator(self):
        er = self._make(
            per_example_side_info=[
                {"scores": {"success": 1.0, "affordance": 0.8}},
            ],
            instance_scores={"ex_0": 1.0},
        )
        prompt = build_mutation_prompt("goal", er)
        # GEPA parity: the reserved "scores" key renders under a
        # friendly header at h4 (one level below the example).
        assert "\n#### Scores (Higher is Better)\n" in prompt
        # Values render via GEPA's recursive format_samples — the
        # objective-axis names become their own h5 markdown sub-headers
        # with the primitive scalar on the next line.
        assert "\n##### success\n" in prompt
        assert "1.0" in prompt
        assert "\n##### affordance\n" in prompt
        assert "0.8" in prompt

    def test_other_side_info_keys_render_verbatim(self):
        er = self._make(
            per_example_side_info=[
                {"trajectory": "unique_traj_marker", "loss": 0.42},
            ],
            instance_scores={"ex_0": 1.0},
        )
        prompt = build_mutation_prompt("goal", er)
        # ``### Example`` (h3) → ``#### {key}`` (h4) → scalar body.
        assert "\n#### trajectory\n" in prompt
        assert "unique_traj_marker" in prompt
        assert "\n#### loss\n" in prompt
        assert "0.42" in prompt

    def test_diagnostics_section_header_present(self):
        er = self._make(
            per_example_side_info=[{"scores": {"s": 1.0}}],
            instance_scores={"ex_0": 1.0},
        )
        prompt = build_mutation_prompt("goal", er)
        assert "## Diagnostics" in prompt

    def test_empty_per_example_slots_still_render_headers(self):
        """When every per-example side_info is ``{}`` (bare-score
        evaluator), the section still shows per-example headers so the
        mutator knows the data was present but empty — rather than
        silently dropping the section."""
        er = self._make(
            per_example_side_info=[{}, {}],
            instance_scores={"ex_0": 1.0, "ex_1": 0.0},
        )
        prompt = build_mutation_prompt("goal", er)
        assert "# Example ex_0" in prompt
        assert "# Example ex_1" in prompt
        assert "no per-example side_info" in prompt

    def test_legacy_side_info_used_when_per_example_absent(self):
        """Fallback: ``per_example_side_info=None`` + batch-level
        ``side_info={...}`` keeps the legacy rendering (non-helix_result
        paths that still populate the batch-level field)."""
        er = self._make(
            per_example_side_info=None,
            side_info={"legacy_key": "legacy_val"},
        )
        prompt = build_mutation_prompt("goal", er)
        assert "## Diagnostics" in prompt
        assert "legacy_key" in prompt
        assert "legacy_val" in prompt
        # Per-example markers must NOT appear on the legacy path.
        assert "# Example" not in prompt

    def test_no_diagnostics_when_both_absent(self):
        er = self._make(per_example_side_info=None, side_info=None)
        prompt = build_mutation_prompt("goal", er)
        assert "## Diagnostics" not in prompt

    def test_nested_dict_renders_via_recursive_headers(self):
        """GEPA ``render_value`` parity: a dict value inside side_info
        renders its keys as bumped-level sub-headers, recursively.

        Header depths under the monotonic hierarchy:
          ## Diagnostics (h2) → ### Example (h3) → #### key (h4)
          → ##### subkey (h5) → ###### subsubkey (h6, capped).
        """
        er = self._make(
            per_example_side_info=[
                {
                    "trajectory": {
                        "grasp": {"success": True, "pose": "top-down"},
                        "place": "centered",
                    },
                },
            ],
            instance_scores={"ex_0": 1.0},
        )
        prompt = build_mutation_prompt("goal", er)
        # Top-level key at h4 (one below ### Example).
        assert "\n#### trajectory\n" in prompt
        # First-level nested keys at h5.
        assert "\n##### grasp\n" in prompt
        assert "\n##### place\n" in prompt
        # Second-level nested keys at h6 (cap).
        assert "\n###### success\n" in prompt
        assert "\n###### pose\n" in prompt
        # Scalar leaves render as plain text (no further headers).
        assert "top-down" in prompt
        assert "centered" in prompt

    def test_list_values_render_as_item_headers(self):
        """GEPA ``render_value`` parity: a list value renders each
        element under ``##### Item N`` sub-headers (h5 under the h4
        key header)."""
        er = self._make(
            per_example_side_info=[
                {"attempts": ["first_try_failed", "retry_succeeded"]},
            ],
            instance_scores={"ex_0": 1.0},
        )
        prompt = build_mutation_prompt("goal", er)
        assert "\n#### attempts\n" in prompt
        assert "\n##### Item 1\n" in prompt
        assert "\n##### Item 2\n" in prompt
        assert "first_try_failed" in prompt
        assert "retry_succeeded" in prompt

    def test_mixed_nested_structure(self):
        """List of dicts + dicts-of-lists render recursively, each
        level bumping the header depth."""
        er = self._make(
            per_example_side_info=[
                {
                    "steps": [
                        {"action": "grasp", "outcome": "ok"},
                        {"action": "lift", "outcome": "dropped"},
                    ],
                },
            ],
            instance_scores={"ex_0": 1.0},
        )
        prompt = build_mutation_prompt("goal", er)
        assert "\n#### steps\n" in prompt
        assert "\n##### Item 1\n" in prompt
        assert "\n##### Item 2\n" in prompt
        # Dict keys inside each list element bump to h6 (capped).
        assert "\n###### action\n" in prompt
        assert "\n###### outcome\n" in prompt
        assert "grasp" in prompt
        assert "dropped" in prompt

    def test_primitive_values_no_json_dumps(self):
        """Port of GEPA's ``render_value`` scalar case: primitives
        render as plain stripped ``str(value)``, NOT ``json.dumps``.
        Check that a bare float doesn't get quoted or wrapped."""
        er = self._make(
            per_example_side_info=[{"loss": 0.42}],
            instance_scores={"ex_0": 1.0},
        )
        prompt = build_mutation_prompt("goal", er)
        assert "\n#### loss\n" in prompt
        # The scalar should be rendered as just "0.42" on its own
        # line, not '"0.42"' or "0.42\n" wrapped in JSON.
        assert "0.42" in prompt
        assert '"0.42"' not in prompt

    def test_diagnostics_header_hierarchy_is_monotonic(self):
        """The Diagnostics block's header levels never jump backwards
        as the reader descends the tree.  Concretely: under
        ``## Diagnostics`` (h2) the next header is at h3 or deeper, the
        next is h3-or-deeper again, etc.  Bumps of +1 and decreases
        (coming back up from a sub-branch) are fine; a jump from h2
        straight to h1 would inverted the hierarchy and confuse
        markdown tooling / the LLM's markdown parser."""
        er = self._make(
            per_example_side_info=[
                {"trajectory": {"grasp": "ok"}, "scores": {"success": 1.0}},
                {"trajectory": {"grasp": "dropped"}, "scores": {"success": 0.0}},
            ],
            instance_scores={"ex_0": 1.0, "ex_1": 0.0},
        )
        prompt = build_mutation_prompt("goal", er)

        # Find the Diagnostics block.
        diag_idx = prompt.index("## Diagnostics")
        diag = prompt[diag_idx:]
        # Next section (## ...) ends the block — slice up to there.
        next_h2 = diag.find("\n## ", 1)
        if next_h2 != -1:
            diag = diag[:next_h2]

        # Extract header lines + depths.
        headers = []
        for line in diag.splitlines():
            s = line.lstrip("#")
            depth = len(line) - len(s)
            if depth >= 1 and s.startswith(" "):
                headers.append((depth, line))
        assert headers, "Diagnostics block has no headers at all"

        # The first header must be ``## Diagnostics`` itself (h2).
        assert headers[0][0] == 2

        # No header is at depth 1 (h1) — they'd visually outrank the
        # surrounding ``## Diagnostics``.
        h1s = [h for d, h in headers if d == 1]
        assert not h1s, f"Unexpected h1 headers inside Diagnostics: {h1s}"

        # Every depth is <= 6 (markdown cap).
        for d, h in headers:
            assert d <= 6, f"{d}-hash header exceeds markdown h6 cap: {h!r}"

    def test_max_markdown_header_level_caps_at_six(self):
        """GEPA caps nested header depth at ``######`` (h6) — deeper
        nesting folds to the same level rather than emitting h7+
        which is not valid markdown."""
        deeply_nested: Any = "leaf"
        # Build ~10 levels of nesting.
        for i in range(10):
            deeply_nested = {f"k{i}": deeply_nested}
        er = self._make(
            per_example_side_info=[{"root": deeply_nested}],
            instance_scores={"ex_0": 1.0},
        )
        prompt = build_mutation_prompt("goal", er)
        # No header with 7+ ``#``.
        for line in prompt.splitlines():
            stripped = line.lstrip("#")
            prefix_len = len(line) - len(stripped)
            assert prefix_len <= 6, (
                f"Found a {prefix_len}-hash markdown header (>h6): {line!r}"
            )

    def test_per_example_takes_precedence_over_legacy(self):
        """When both fields are populated (hybrid evaluator path),
        per-example wins — the legacy batch-level dict is dropped
        from the rendered prompt."""
        er = self._make(
            per_example_side_info=[{"per_example_marker": "YES"}],
            side_info={"legacy_marker": "NO"},
            instance_scores={"ex_0": 1.0},
        )
        prompt = build_mutation_prompt("goal", er)
        assert "per_example_marker" in prompt
        assert "YES" in prompt
        assert "legacy_marker" not in prompt
        assert "NO" not in prompt


# ---------------------------------------------------------------------------
# Tests: mutation-prompt artifact persistence
# ---------------------------------------------------------------------------


class TestMutationPromptArtifact:
    """``mutate()`` writes the rendered prompt to
    ``<worktree>/.agent_task_prompt.md`` (or reserved fallback)
    before invoking Claude Code
    and adds the artifact + ``helix_batch.json`` to the worktree's
    ``.gitignore`` so neither leaks into the candidate git tree."""

    def _build(self, tmp_path, mocker):
        parent = make_candidate("g0-s0", str(tmp_path / "parent"))
        er = make_eval_result("g0-s0")
        config = make_config()
        wt = tmp_path / "g1-s0"
        wt.mkdir()
        child = make_candidate("g1-s0", str(wt))
        mocker.patch("helix.mutator.clone_candidate", return_value=child)
        mocker.patch("helix.mutator.invoke_claude_code", return_value={"result": "ok"})
        mocker.patch("helix.mutator.snapshot_candidate", return_value="abc123")
        mocker.patch("helix.mutator.remove_worktree")
        return parent, er, config, wt

    def test_artifact_written_before_claude_invocation(
        self, tmp_path: Path, mocker
    ):
        """The prompt file must exist at the time ``invoke_claude_code``
        is called so Claude can in principle read it during the session
        and so post-hoc inspection works even on a crashed mutation."""
        parent, er, config, wt = self._build(tmp_path, mocker)

        # Capture whether the artifact exists at the moment
        # invoke_claude_code is entered.
        exists_at_invoke: list[bool] = []

        def fake_invoke(*_a, **_kw):
            exists_at_invoke.append(
                (wt / ".agent_task_prompt.md").exists()
                or (wt / ".agent_internal" / "task_prompt.md").exists()
            )
            return {"result": "ok"}

        mocker.patch("helix.mutator.invoke_claude_code", side_effect=fake_invoke)
        mutate(parent, er, "g1-s0", config, tmp_path)
        assert exists_at_invoke == [True]

    def test_artifact_contains_rendered_prompt(
        self, tmp_path: Path, mocker
    ):
        parent, er, config, wt = self._build(tmp_path, mocker)
        mutate(parent, er, "g1-s0", config, tmp_path)

        if (wt / ".agent_task_prompt.md").exists():
            artifact_path = wt / ".agent_task_prompt.md"
        else:
            artifact_path = wt / ".agent_internal" / "task_prompt.md"
        content = artifact_path.read_text()
        # Objective + autonomous-rules block are both in the rendered
        # prompt via build_mutation_prompt.
        assert config.objective in content
        assert "[MUTATION COMPLETE]" in content

    def test_gitignore_excludes_helix_artifacts(
        self, tmp_path: Path, mocker
    ):
        """``.gitignore`` in the worktree gains entries for the prompt
        file AND ``helix_batch.json`` so neither file enters the
        candidate diff on the next generation."""
        parent, er, config, wt = self._build(tmp_path, mocker)
        mutate(parent, er, "g1-s0", config, tmp_path)

        gi = (wt / ".gitignore").read_text()
        assert ".agent_task_prompt.md" in gi
        assert ".agent_internal/" in gi
        assert "helix_batch.json" in gi

    def test_gitignore_append_is_idempotent(
        self, tmp_path: Path, mocker
    ):
        """A pre-existing ``.gitignore`` with HELIX patterns already
        present must not grow duplicate entries on subsequent mutate()
        calls — otherwise long runs balloon the gitignore."""
        parent, er, config, wt = self._build(tmp_path, mocker)
        # Seed a gitignore with the patterns already present + other lines.
        (wt / ".gitignore").write_text(
            "*.pyc\n"
            "# HELIX per-invocation artifacts (never commit to candidate tree)\n"
            ".agent_task_prompt.md\n"
            ".agent_internal/\n"
            "helix_batch.json\n"
        )
        mutate(parent, er, "g1-s0", config, tmp_path)

        gi = (wt / ".gitignore").read_text()
        assert gi.count(".agent_task_prompt.md") == 1
        assert gi.count(".agent_internal/") == 1
        assert gi.count("helix_batch.json") == 1
        assert "*.pyc" in gi  # existing content preserved

    def test_gitignore_created_when_absent(
        self, tmp_path: Path, mocker
    ):
        parent, er, config, wt = self._build(tmp_path, mocker)
        # Ensure no pre-existing gitignore.
        assert not (wt / ".gitignore").exists()
        mutate(parent, er, "g1-s0", config, tmp_path)
        assert (wt / ".gitignore").exists()

    def test_collision_uses_fallback_without_overwriting_user_file(
        self, tmp_path: Path, mocker
    ):
        parent, er, config, wt = self._build(tmp_path, mocker)
        user_content = "user-owned prompt file\n"
        (wt / ".agent_task_prompt.md").write_text(user_content)

        mutate(parent, er, "g1-s0", config, tmp_path)

        assert (wt / ".agent_task_prompt.md").read_text() == user_content
        fallback = wt / ".agent_internal" / "task_prompt.md"
        assert fallback.exists()
        assert "[MUTATION COMPLETE]" in fallback.read_text()

    def test_collision_on_both_reserved_paths_fails_closed(
        self, tmp_path: Path, mocker
    ):
        parent, er, config, wt = self._build(tmp_path, mocker)
        (wt / ".agent_task_prompt.md").write_text("user primary\n")
        fallback = wt / ".agent_internal" / "task_prompt.md"
        fallback.parent.mkdir()
        fallback.write_text("user fallback\n")

        with pytest.raises(PromptArtifactCollisionError, match="prompt artifact"):
            mutate(parent, er, "g1-s0", config, tmp_path)

        assert (wt / ".agent_task_prompt.md").read_text() == "user primary\n"
        assert fallback.read_text() == "user fallback\n"


# ---------------------------------------------------------------------------
# Tests: invoke_claude_code
# ---------------------------------------------------------------------------


class TestInvokeClaudeCode:
    def test_returns_parsed_json_on_success(self, mocker):
        payload = {"result": "ok", "turns": 3}
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout=json.dumps(payload),
            stderr="",
            returncode=0,
        )
        config = AgentConfig()
        result = invoke_claude_code("/tmp/wt", "do something", config)
        assert result == payload

    def test_raises_on_nonzero_returncode(self, mocker):
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="fatal error",
            returncode=1,
        )
        config = AgentConfig()
        with pytest.raises(MutationError, match="exited with code 1"):
            invoke_claude_code("/tmp/wt", "do something", config)

    def test_raises_on_invalid_json(self, mocker):
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout="not valid json {{{{",
            stderr="",
            returncode=0,
        )
        config = AgentConfig()
        with pytest.raises(MutationError, match="Failed to parse"):
            invoke_claude_code("/tmp/wt", "do something", config)

    def test_cli_args_include_required_flags(self, mocker):
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(stdout="{}", stderr="", returncode=0)
        config = AgentConfig(allowed_tools=["Read", "Edit"])
        invoke_claude_code("/tmp/wt", "the prompt", config)

        call_args = mock_run.call_args
        args_list = call_args[0][0]

        assert "claude" in args_list
        assert "--dangerously-skip-permissions" in args_list
        assert "--print" in args_list
        assert "--output-format" in args_list
        assert "json" in args_list
        assert "--allowedTools" in args_list
        assert "Read,Edit" in args_list
        assert "--model" not in args_list
        assert "--max-turns" not in args_list
        assert "the prompt" not in args_list
        assert ".agent_task_prompt.md" in args_list[-1]
        assert "input" not in call_args[1]

    def test_uses_correct_cwd(self, mocker):
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(stdout="{}", stderr="", returncode=0)
        config = AgentConfig()
        invoke_claude_code("/specific/path", "prompt", config)
        assert mock_run.call_args[1]["cwd"] == "/specific/path"

    def test_no_timeout_in_subprocess(self, mocker):
        """subprocess.run should not have a timeout — let Claude run forever."""
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(stdout="{}", stderr="", returncode=0)
        config = AgentConfig()
        invoke_claude_code("/tmp/wt", "prompt", config)
        assert "timeout" not in mock_run.call_args[1]

    def test_codex_cli_args_include_required_flags(self, mocker):
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout='{"type":"thread.started","thread_id":"thr_123"}\n',
            stderr="",
            returncode=0,
        )
        config = AgentConfig(backend="codex", model="gpt-5")

        result = invoke_claude_code("/tmp/wt", "the prompt", config)

        args_list = mock_run.call_args[0][0]
        assert args_list[:2] == ["codex", "exec"]
        assert "--json" in args_list
        assert "--dangerously-bypass-approvals-and-sandbox" in args_list
        assert "--model" in args_list
        assert "gpt-5" in args_list
        assert "the prompt" not in args_list
        assert ".agent_task_prompt.md" in args_list[-1]
        assert "input" not in mock_run.call_args[1]
        assert result["events"][0]["thread_id"] == "thr_123"

    def test_cursor_cli_args_use_stream_json(self, mocker):
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout='{"type":"system","subtype":"init","session_id":"sess_123"}\n',
            stderr="",
            returncode=0,
        )
        config = AgentConfig(backend="cursor", model="gpt-5")

        result = invoke_claude_code("/tmp/wt", "the prompt", config)

        args_list = mock_run.call_args[0][0]
        assert args_list[:2] == ["cursor", "agent"]
        assert "--print" in args_list
        assert "--output-format" in args_list
        assert "stream-json" in args_list
        assert "--yolo" in args_list
        assert "--workspace" in args_list
        assert "/tmp/wt" in args_list
        assert "the prompt" not in args_list
        assert ".agent_task_prompt.md" in args_list[-1]
        assert "input" not in mock_run.call_args[1]
        assert result["events"][0]["session_id"] == "sess_123"

    def test_backend_artifacts_written_for_structured_backends(self, tmp_path: Path, mocker):
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout=(
                '{"type":"system","subtype":"init","session_id":"sess_123"}\n'
                '{"type":"tool.call","name":"read_file"}\n'
            ),
            stderr="warning text",
            returncode=0,
        )
        config = AgentConfig(backend="cursor")

        invoke_claude_code(str(tmp_path), "prompt", config)

        assert (tmp_path / BACKEND_STDOUT_ARTIFACT_NAME).read_text().startswith('{"type":"system"')
        assert (tmp_path / BACKEND_STDERR_ARTIFACT_NAME).read_text() == "warning text"
        payload = json.loads((tmp_path / BACKEND_RESULT_ARTIFACT_NAME).read_text())
        assert payload["backend"] == "cursor"
        assert payload["usage"]["session_id"] == "sess_123"
        assert payload["usage"]["tool_event_count"] == 1

    def test_gemini_tolerates_text_preamble_before_json_stream(self, mocker):
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout=(
                "MCP issues detected. Run /mcp list for status.\n"
                '{"type":"init","session_id":"sess_123"}\n'
                '{"type":"result","status":"success","stats":{"input_tokens":10,"output_tokens":2}}\n'
            ),
            stderr="",
            returncode=0,
        )
        config = AgentConfig(backend="gemini")

        result = invoke_claude_code("/tmp/wt", "prompt", config)

        assert result["events"][0]["type"] == "init"
        assert result["unparsable_lines"] == ["MCP issues detected. Run /mcp list for status."]

    @pytest.mark.parametrize(
        ("backend", "expected_prefix", "expected_flags"),
        [
            ("gemini", ["gemini"], ["--yolo", "--prompt", "--output-format", "stream-json"]),
            (
                "opencode",
                ["opencode", "run"],
                ["--format", "json", "--dangerously-skip-permissions", "--file"],
            ),
        ],
    )
    def test_gemini_and_opencode_cli_args(self, mocker, backend, expected_prefix, expected_flags):
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout='{"type":"system","session_id":"sess_123"}\n',
            stderr="",
            returncode=0,
        )
        config = AgentConfig(backend=backend, model="test-model")

        result = invoke_claude_code("/tmp/wt", "the prompt", config)

        args_list = mock_run.call_args[0][0]
        assert args_list[: len(expected_prefix)] == expected_prefix
        for flag in expected_flags:
            assert flag in args_list
        assert "test-model" in args_list
        assert "the prompt" not in args_list
        assert any(".agent_task_prompt.md" in arg for arg in args_list)
        assert "input" not in mock_run.call_args[1]
        assert result["events"][0]["session_id"] == "sess_123"

    def test_opencode_usage_normalization_captures_session_id_and_cost(self, tmp_path: Path, mocker):
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout=(
                '{"type":"step_start","sessionID":"ses_123"}\n'
                '{"type":"step_finish","part":{"tokens":{"input":10,"output":2},"cost":0.25}}\n'
            ),
            stderr="",
            returncode=0,
        )
        config = AgentConfig(backend="opencode")

        invoke_claude_code(str(tmp_path), "prompt", config)

        payload = json.loads((tmp_path / BACKEND_RESULT_ARTIFACT_NAME).read_text())
        assert payload["usage"]["session_id"] == "ses_123"
        assert payload["usage"]["cost_usd"] == 0.25

    def test_uses_sandbox_when_enabled(self, tmp_path: Path, mocker):
        mock_run = mocker.patch("helix.mutator.run_sandboxed_command")
        mock_run.return_value = MagicMock(stdout="{}", stderr="", returncode=0)

        result = invoke_claude_code(
            str(tmp_path),
            "prompt",
            AgentConfig(),
            sandbox=SandboxConfig(enabled=True),
        )

        assert result == {}
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["scope"] == "agent"
        assert mock_run.call_args.kwargs["sync_back"] is True
        assert mock_run.call_args.kwargs["image"] == "ghcr.io/ke7/helix-evo-runner-claude:latest"

    def test_backend_auth_env_is_passed_automatically(self, mocker, monkeypatch):
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout='{"type":"system","subtype":"init","session_id":"sess_123"}\n',
            stderr="",
            returncode=0,
        )
        monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")

        invoke_claude_code("/tmp/wt", "prompt", AgentConfig(backend="cursor"))

        assert mock_run.call_args.kwargs["env"]["CURSOR_API_KEY"] == "cursor-key"


# ---------------------------------------------------------------------------
# Tests: mutate
# ---------------------------------------------------------------------------


class TestMutate:
    def test_returns_candidate_on_success(self, tmp_path: Path, mocker):
        parent = make_candidate("g0-s0", str(tmp_path / "parent"))
        er = make_eval_result("g0-s0")
        config = make_config()

        child_path = tmp_path / "g1-s0"
        child_path.mkdir()
        child = make_candidate("g1-s0", str(child_path))
        mocker.patch("helix.mutator.clone_candidate", return_value=child)
        mocker.patch("helix.mutator.invoke_claude_code", return_value={"result": "ok"})
        mocker.patch("helix.mutator.snapshot_candidate", return_value="abc123")
        mocker.patch("helix.mutator.remove_worktree")

        result = mutate(parent, er, "g1-s0", config, Path("/tmp"))

        assert result is child

    def test_sets_operation_to_mutate(self, tmp_path: Path, mocker):
        parent = make_candidate("g0-s0")
        er = make_eval_result()
        config = make_config()

        child_path = tmp_path / "g1-s0"
        child_path.mkdir()
        child = make_candidate("g1-s0", str(child_path))
        mocker.patch("helix.mutator.clone_candidate", return_value=child)
        mocker.patch("helix.mutator.invoke_claude_code", return_value={})
        mocker.patch("helix.mutator.snapshot_candidate", return_value="sha")
        mocker.patch("helix.mutator.remove_worktree")

        result = mutate(parent, er, "g1-s0", config, Path("/tmp"))
        assert result.operation == "mutate"

    def test_returns_none_on_mutation_error(self, tmp_path: Path, mocker):
        parent = make_candidate("g0-s0")
        er = make_eval_result()
        config = make_config()

        child_path = tmp_path / "g1-s0"
        child_path.mkdir()
        child = make_candidate("g1-s0", str(child_path))
        mocker.patch("helix.mutator.clone_candidate", return_value=child)
        mocker.patch(
            "helix.mutator.invoke_claude_code",
            side_effect=MutationError("timeout"),
        )
        mock_remove = mocker.patch("helix.mutator.remove_worktree")
        mocker.patch("helix.mutator.snapshot_candidate")

        result = mutate(parent, er, "g1-s0", config, Path("/tmp"))

        assert result is None
        mock_remove.assert_called_once_with(child)

    def test_removes_worktree_on_failure(self, tmp_path: Path, mocker):
        parent = make_candidate("g0-s0")
        er = make_eval_result()
        config = make_config()

        child_path = tmp_path / "g1-s0"
        child_path.mkdir()
        child = make_candidate("g1-s0", str(child_path))
        mocker.patch("helix.mutator.clone_candidate", return_value=child)
        mocker.patch(
            "helix.mutator.invoke_claude_code",
            side_effect=MutationError("bad json"),
        )
        mock_remove = mocker.patch("helix.mutator.remove_worktree")
        mocker.patch("helix.mutator.snapshot_candidate")

        mutate(parent, er, "g1-s0", config, Path("/tmp"))

        mock_remove.assert_called_once_with(child)

    def test_snapshot_not_called_by_mutate_on_success(self, tmp_path: Path, mocker):
        """mutate() must NOT call snapshot_candidate — the caller owns that step.

        Callers (evolution.py) must call save_state() BEFORE snapshot_candidate()
        so that state is persisted even if the commit step crashes.
        """
        parent = make_candidate("g0-s0")
        er = make_eval_result()
        config = make_config()

        child_path = tmp_path / "g1-s0"
        child_path.mkdir()
        child = make_candidate("g1-s0", str(child_path))
        mocker.patch("helix.mutator.clone_candidate", return_value=child)
        mocker.patch("helix.mutator.invoke_claude_code", return_value={})
        mock_snapshot = mocker.patch("helix.mutator.snapshot_candidate", return_value="sha")
        mocker.patch("helix.mutator.remove_worktree")

        result = mutate(parent, er, "g1-s0", config, Path("/tmp"))

        # mutate() returns the child but does NOT snapshot internally
        assert result is child
        mock_snapshot.assert_not_called()

    def test_snapshot_not_called_on_failure(self, tmp_path: Path, mocker):
        parent = make_candidate("g0-s0")
        er = make_eval_result()
        config = make_config()

        child_path = tmp_path / "g1-s0"
        child_path.mkdir()
        child = make_candidate("g1-s0", str(child_path))
        mocker.patch("helix.mutator.clone_candidate", return_value=child)
        mocker.patch(
            "helix.mutator.invoke_claude_code",
            side_effect=MutationError("timeout"),
        )
        mock_snapshot = mocker.patch("helix.mutator.snapshot_candidate")
        mocker.patch("helix.mutator.remove_worktree")

        mutate(parent, er, "g1-s0", config, Path("/tmp"))

        mock_snapshot.assert_not_called()

    def test_passes_background_to_prompt(self, tmp_path: Path, mocker):
        parent = make_candidate("g0-s0")
        er = make_eval_result()
        config = make_config()

        child_path = tmp_path / "g1-s0"
        child_path.mkdir()
        child = make_candidate("g1-s0", str(child_path))
        mocker.patch("helix.mutator.clone_candidate", return_value=child)
        mock_invoke = mocker.patch("helix.mutator.invoke_claude_code", return_value={})
        mocker.patch("helix.mutator.snapshot_candidate", return_value="sha")
        mocker.patch("helix.mutator.remove_worktree")

        mutate(parent, er, "g1-s0", config, Path("/tmp"), background="special context")

        prompt_arg = mock_invoke.call_args[0][1]
        assert "special context" in prompt_arg
