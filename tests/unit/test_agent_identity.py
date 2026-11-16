"""What ties an agent's session and a rejected proposal back to their candidate."""

from __future__ import annotations

from pathlib import Path

import pytest

from helix import timing
from helix.config import AgentConfig
from helix.mutator import _build_backend_args, _prompt_file_instruction
from helix.timing import recording, span

BACKENDS = ["agy", "claude", "codex", "cursor", "opencode", "pi"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_launch_line_names_the_candidate(backend: str) -> None:
    # In the sandbox the agent's workspace is /workspace; the name comes from the host worktree.
    args = _build_backend_args("/workspace", AgentConfig(backend=backend), ".agent_task_prompt.md", candidate_id="g3-s12")
    line = next(a for a in args if ".agent_task_prompt.md" in a)
    assert line.endswith("This is HELIX candidate g3-s12.")
    assert line.startswith("Read .agent_task_prompt.md")


@pytest.mark.parametrize("name", [None, "", "wt", "workspace", "g3-s12/../x"])
def test_no_candidate_no_marker(name: str | None) -> None:
    assert "HELIX candidate" not in _prompt_file_instruction(".agent_task_prompt.md", name)


def test_a_merge_candidate_is_named_too() -> None:
    assert _prompt_file_instruction("p.md", "g12-m1").endswith("This is HELIX candidate g12-m1.")


def test_a_decision_records_paths_only_when_given(tmp_path: Path) -> None:
    path = tmp_path / "timing.jsonl"
    with recording(path), span("generation", generation=2):
        timing.decision("g2-s5", accepted=False, reason="tamper-rejected mutation candidate",
                        paths=["val/b.json", "evaluate.py"])
        timing.decision("g2-s6", accepted=False, reason="minibatch-gate-rejected candidate")
    decisions = [r for r in timing.read_records(path) if r.get("event") == "decision"]
    assert decisions[0]["paths"] == ["evaluate.py", "val/b.json"]
    assert "paths" not in decisions[1]
