"""Tests for helix.ui.model.build_static."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import helix
from helix.config import HelixConfig, load_config
from helix.ui.model import build_static, dataset_sizes, evaluator_modules

EXAMPLES = sorted((Path(__file__).parents[2] / "examples").glob("*/helix.toml"))


def _config(**sections: Any) -> HelixConfig:
    data: dict[str, Any] = {"objective": "test objective", "evaluator": {"command": "python evaluate.py"}}
    data.update(sections)
    return HelixConfig.model_validate(data)


def _warning_ids(config: HelixConfig, root: Path) -> list[str]:
    static = build_static(config, root, root / "helix.toml")
    return [w["id"] for w in static["isolation"]["warnings"]]


@pytest.mark.parametrize("config_path", EXAMPLES, ids=lambda path: path.parent.name)
def test_example_configs_build_a_well_formed_model(config_path: Path) -> None:
    config = load_config(config_path)
    static = build_static(config, config_path.parent, config_path)

    json.dumps(static)
    assert set(static) == {
        "project", "config", "loop", "isolation", "repro", "evaluator_modules", "sizes",
    }
    assert static["project"]["name"] == config_path.parent.name
    assert static["project"]["helix_version"] == helix.__version__
    ids = [node["id"] for node in static["loop"]["nodes"]]
    assert ids[:4] == ["frontier", "mutation", "gate", "validation"]
    assert ("merge" in ids) == config.evolution.merge_enabled
    assert static["isolation"]["agent"] in {"host", "sandbox"}
    assert static["isolation"]["evaluator"] in {"host", "sidecar", "sandbox"}
    for warning in static["isolation"]["warnings"]:
        assert set(warning) == {"id", "text", "detail", "keys"}
    assert all(len(row) == 2 and all(isinstance(v, str) for v in row) for row in static["repro"])
    assert set(static["sizes"]) == {"train", "val"}
    for module in static["evaluator_modules"]:
        assert set(module) == {"name", "source"}


def test_web_researcher_example_finds_its_timed_names() -> None:
    path = next(p for p in EXAMPLES if p.parent.name == "web_researcher")
    static = build_static(load_config(path), path.parent, path)
    names = [m["name"] for m in static["evaluator_modules"]]
    assert names[:2] == ["ground_truth", "solve"]
    assert static["evaluator_modules"][0]["source"].startswith("evaluate.py:")
    assert static["sizes"] == {"train": 5, "val": 5}


def test_config_facts_and_repro(tmp_path: Path) -> None:
    config = _config(
        passthrough_env=["MUJOCO_GL", "OPENAI_API_KEY"],
        env={"OPENAI_API_KEY": "sk-secret", "HF_HOME": "/x"},
        evolution={"max_evaluations": -1, "val_stage_size": 0, "num_parallel_proposals": 2},
        agent={"backend": "codex", "model": "m", "effort": "high", "max_turns": 80},
        sandbox={"enabled": True, "auth_volume": "run-vol", "omit_from_agent": ["a", "b"]},
    )
    static = build_static(config, tmp_path, tmp_path / "helix.toml")
    facts = static["config"]
    assert facts["max_evaluations"] is None
    assert facts["val_stage_size"] is None
    assert facts["proposals_per_generation"] == 2
    assert facts["env_names"] == ["MUJOCO_GL", "OPENAI_API_KEY", "HF_HOME"]
    assert "sk-secret" not in json.dumps(static)
    assert facts["sandbox"]["auth_volume"] == "run-vol"
    assert facts["sandbox"]["auth_volume_shared"] is False
    assert facts["sandbox"]["omit_from_agent"] == 2
    assert static["loop"]["planned"]["mutation"] == "2 proposals · codex high"

    repro = dict(static["repro"])
    digest = hashlib.sha256(config.model_dump_json().encode()).hexdigest()[:16]
    assert repro["config"] == f"helix.toml · {digest}"
    assert repro["agent"] == "codex · m · high · 80 turns"
    assert repro["sandbox"] == "on · evaluator on host · auth volume run-vol"
    assert repro["seed"] == "repository"
    assert repro["helix"] == f"{helix.__version__} (this viewer)"


def test_zones_follow_sandbox_settings(tmp_path: Path) -> None:
    host = build_static(_config(), tmp_path, tmp_path / "helix.toml")
    assert host["isolation"] == {"agent": "host", "evaluator": "host", "warnings": []}

    sidecar = _config(
        evaluator={
            "command": "python evaluate.py",
            "sidecar": {"image": "eval:1", "command": "serve", "endpoint": "http://e:1"},
        },
        sandbox={"enabled": True, "evaluator": True, "auth_volume": "v"},
        evolution={"merge_enabled": True},
    )
    static = build_static(sidecar, tmp_path, tmp_path / "helix.toml")
    zones = {node["id"]: node["zone"] for node in static["loop"]["nodes"]}
    assert zones == {
        "frontier": "host", "mutation": "sandbox", "gate": "sidecar",
        "validation": "sidecar", "merge": "sandbox",
    }


@pytest.mark.parametrize(
    ("sections", "fires"),
    [
        ({"agent": {"web_access": False}}, True),
        ({"agent": {"web_access": False}, "sandbox": {"enabled": True, "auth_volume": "v"}}, True),
        (
            {"agent": {"web_access": False},
             "sandbox": {"enabled": True, "network": "none", "auth_volume": "v"}},
            False,
        ),
        (
            {"agent": {"web_access": False},
             "sandbox": {"enabled": True, "network": "my-internal", "auth_volume": "v"}},
            False,
        ),
        ({}, False),
    ],
)
def test_web_access_open_network(tmp_path: Path, sections: dict[str, Any], fires: bool) -> None:
    assert ("web-access-open-network" in _warning_ids(_config(**sections), tmp_path)) == fires


def test_the_auth_volume_is_a_fact_not_a_warning(tmp_path: Path) -> None:
    shared = build_static(_config(sandbox={"enabled": True}), tmp_path, tmp_path / "helix.toml")
    assert shared["isolation"]["warnings"] == []
    assert shared["config"]["sandbox"]["auth_volume"] == "helix-auth-claude"
    assert shared["config"]["sandbox"]["auth_volume_shared"] is True


def test_omit_without_sandbox(tmp_path: Path) -> None:
    assert "omit-without-sandbox" in _warning_ids(
        _config(sandbox={"omit_from_agent": ["evaluate.py"]}), tmp_path
    )
    assert "omit-without-sandbox" not in _warning_ids(
        _config(sandbox={"enabled": True, "auth_volume": "v", "omit_from_agent": ["e.py"]}),
        tmp_path,
    )


def test_minibatch_repeats_with_train_size(tmp_path: Path) -> None:
    # 10 // 3 = 3 disjoint minibatches.
    repeats = _config(
        dataset={"train_size": 10}, evolution={"num_parallel_proposals": 2, "mutations_per_parent": 2}
    )
    assert "minibatch-repeats" in _warning_ids(repeats, tmp_path)
    fits = _config(dataset={"train_size": 10}, evolution={"num_parallel_proposals": 3})
    assert "minibatch-repeats" not in _warning_ids(fits, tmp_path)
    unknown = _config(evolution={"num_parallel_proposals": 8})
    assert "minibatch-repeats" not in _warning_ids(unknown, tmp_path)


def test_minibatch_repeats_with_train_dir(tmp_path: Path) -> None:
    train = tmp_path / "splits" / "train"
    train.mkdir(parents=True)
    for name in ("a__0", "a__1", "b__0", "b__1"):
        (train / f"{name}.json").write_text("{}")
    (train / "notes.txt").write_text("ignored")
    config = _config(
        seedless={"train_path": "splits/train"},
        evolution={"num_parallel_proposals": 3, "minibatch_size": 2},
    )
    assert "minibatch-repeats" in _warning_ids(config, tmp_path)
    assert dataset_sizes(config, tmp_path) == {"train": 4, "val": 4}


def test_dataset_sizes(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text('{"x": 1}\n\n{"x": 2}\n{"x": 3}\n')
    (tmp_path / "val.json").write_text("[1, 2]")
    (tmp_path / "empty.json").write_text("")
    assert dataset_sizes(_config(), tmp_path) == {"train": None, "val": None}
    assert dataset_sizes(
        _config(seedless={"train_path": "train.jsonl", "val_path": "val.json"}), tmp_path
    ) == {"train": 3, "val": 2}
    # dataset.val_size wins over val_path; train_path wins over train_size.
    assert dataset_sizes(
        _config(
            seedless={"train_path": "train.jsonl", "val_path": "val.json"},
            dataset={"train_size": 9, "val_size": 7},
        ),
        tmp_path,
    ) == {"train": 3, "val": 7}
    # An empty or missing train_path falls back to train_size; an empty
    # val_path does not fall back to the train ids.
    assert dataset_sizes(
        _config(seedless={"train_path": "empty.json", "val_path": "empty.json"},
                dataset={"train_size": 4}),
        tmp_path,
    ) == {"train": 4, "val": None}
    assert dataset_sizes(_config(seedless={"train_path": "missing"}), tmp_path) == {
        "train": None, "val": None,
    }


def test_planned_values(tmp_path: Path) -> None:
    config = _config(
        dataset={"train_size": 40, "val_size": 150},
        evolution={
            "batch_sampler": "stratified", "num_sampled_groups": 4,
            "num_examples_per_group": 2, "minibatch_size": 8, "num_parallel_proposals": 2,
        },
        agent={"backend": "codex", "effort": "high"},
    )
    planned = build_static(config, tmp_path, tmp_path / "helix.toml")["loop"]["planned"]
    assert planned == {"mutation": "2 proposals · codex high", "gate": "8 ids", "validation": "150 ids"}
    single = build_static(_config(), tmp_path, tmp_path / "helix.toml")["loop"]["planned"]
    assert single == {"mutation": "1 proposal · claude", "gate": None, "validation": None}


def test_evaluator_module_scan(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "ev").mkdir(parents=True)
    (project / "ev" / "run.sh").write_text(
        "#!/bin/sh\n# python ev/commented.py\nset -e\nexec ./.venv/bin/python -I ev/main.py\n"
    )
    (project / "ev" / "main.py").write_text(
        "from helix import timed\n"
        "with timed('Scene kernel'):\n"
        "    pass\n"
        "timed.record(\"Planner\", 0.4)\n"
        "with timed('Scene kernel'):\n"
        "    pass\n"
        "timed(name)\n"
    )
    (project / "ev" / "commented.py").write_text("timed('Commented')\n")
    (project / "extra.py").write_text("helix.timed(\"Extra\")\n")
    (project / "helper.py").write_text("timed.record('Helper', 1.0)\n")
    (project / "big.py").write_text("timed('Big')\n" + "#" * (2 * 1024 * 1024))
    outside = tmp_path / "outside.py"
    outside.write_text("timed('Outside')\n")
    (project / "link.py").symlink_to(outside)

    config = _config(
        evaluator={
            "command": "bash ev/run.sh",
            "extra_commands": ["python3 extra.py", "python ../outside.py", f"python {outside}"],
            "protected_files": ["helper.py", "link.py", "big.py", "ev", "../outside.py"],
        }
    )
    modules = evaluator_modules(config, project)
    assert modules == [
        {"name": "Scene kernel", "source": "ev/main.py:2"},
        {"name": "Planner", "source": "ev/main.py:4"},
        {"name": "Extra", "source": "extra.py:1"},
        {"name": "Helper", "source": "helper.py:1"},
    ]


def test_evaluator_module_scan_tolerates_missing_and_unparseable(tmp_path: Path) -> None:
    config = _config(evaluator={"command": "bash 'missing.sh", "extra_commands": ["python gone.py"]})
    assert evaluator_modules(config, tmp_path) == []


def test_group_key_separator_is_in_static(tmp_path: Path) -> None:
    static = build_static(_config(evolution={"group_key_separator": "::"}), tmp_path, tmp_path / "helix.toml")
    assert static["config"]["group_key_separator"] == "::"
