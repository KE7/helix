"""Unit tests for GEPA-aligned state persistence.

Covers the additions called out in an internal divergence audit
(schema thin vs GEPA):

* C1 — per-(candidate, example) eval cache survives save → load round-trip
  (GEPA's ``evaluation_cache`` field plus ``GEPAState.save``/``GEPAState.load``
  and the cache-sync logic in ``initialize_gepa_state``, all in
  ``core/state.py``).
* C/§3 — per-program discovery budget (``num_metric_calls_by_discovery``)
  is persisted on every accept site (GEPA's ``num_metric_calls_by_discovery``
  field, appended to in ``update_state_with_new_program``, ``core/state.py``).
* D1 — schema_version is written and a missing version is treated as the
  unversioned predecessor with defaulted new fields (GEPA's
  ``_VALIDATION_SCHEMA_VERSION`` and ``_upgrade_state_dict``, ``core/state.py``).
* Backward compatibility — pre-migration state.json files load cleanly
  with new fields populated to defaults.
* Resume fidelity — when val_stage_size is None the new state additions
  must not change behaviour: GEPA-identical save points, GEPA-identical
  cache key semantics on resume.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helix.eval_cache import EvaluationCache as MinibatchEvalCache
from helix.state import (
    SCHEMA_VERSION,
    BudgetState,
    EvolutionState,
    clear_eval_cache,
    load_eval_cache,
    load_state,
    save_eval_cache,
    save_state,
)


# ---------------------------------------------------------------------------
# Round-trip with all new fields populated
# ---------------------------------------------------------------------------


def _make_full_state() -> EvolutionState:
    """Build an EvolutionState with every field populated."""
    return EvolutionState(
        generation=4,
        frontier=["g0-s0", "g1-s1", "g2-m1"],
        instance_scores={
            "g0-s0": {"task_a": 0.5, "task_b": 0.7},
            "g1-s1": {"task_a": 0.6, "task_b": 0.7},
            "g2-m1": {"task_a": 0.65, "task_b": 0.75},
        },
        budget=BudgetState(
            evaluations=42,
            input_tokens=100,
            output_tokens=40,
            cached_input_tokens=30,
            cache_creation_input_tokens=20,
            cache_read_input_tokens=10,
            reasoning_tokens=5,
            cost_usd=1.23,
        ),
        config_hash="cfg-deadbeef",
        mutation_counter=2,
        merge_counter=1,
        total_merge_invocations=1,
        merge_attempted_pairs=[["g0-s0", "g1-s1"]],
        i=3,
        num_metric_calls_by_discovery={
            "g0-s0": 0,
            "g1-s1": 14,
            "g2-m1": 28,
        },
        active_frontier={"task_a": ["g2-m1"], "task_b": ["g2-m1"]},
    )


def test_state_roundtrip_preserves_all_fields(tmp_path: Path) -> None:
    """save_state → load_state must deep-equal the original on every field.

    Audit ref: C/§3 + D1.
    """
    state = _make_full_state()
    save_state(state, tmp_path)

    loaded = load_state(tmp_path)
    assert loaded is not None
    # Field-by-field deep equality — dataclass equality is structural.
    assert loaded == state
    # Sanity-check that the new fields specifically survived.
    assert loaded.num_metric_calls_by_discovery == state.num_metric_calls_by_discovery
    assert loaded.active_frontier == state.active_frontier
    assert loaded.schema_version == SCHEMA_VERSION


def test_state_json_writes_schema_version(tmp_path: Path) -> None:
    """state.json on disk must contain ``schema_version``.

    Audit ref: D1 — without a written version, future migrations would have
    no way to recognize the legacy/v0 predecessor.
    """
    save_state(_make_full_state(), tmp_path)

    raw = json.loads((tmp_path / ".helix" / "state.json").read_text())
    assert raw["schema_version"] == SCHEMA_VERSION
    assert "num_metric_calls_by_discovery" in raw


# ---------------------------------------------------------------------------
# Backward compatibility — load a pre-migration state.json
# ---------------------------------------------------------------------------


def _write_legacy_state_json(base_dir: Path) -> None:
    """Write a minimal pre-migration state.json (no schema_version, no
    num_metric_calls_by_discovery) — i.e. the v0 schema HELIX shipped with.
    """
    helix_dir = base_dir / ".helix"
    helix_dir.mkdir(parents=True, exist_ok=True)
    legacy = {
        # Note: deliberately omits "schema_version" and
        # "num_metric_calls_by_discovery".  Mirrors the on-disk format that
        # existed before rng-state-persist audit D1 was addressed.
        "generation": 1,
        "frontier": ["g0-s0"],
        "instance_scores": {"g0-s0": {"task_a": 0.5}},
        "budget": {"evaluations": 1},
        "config_hash": "legacy-hash",
        "mutation_counter": 0,
        "merge_counter": 0,
        "total_merge_invocations": 0,
        "merge_attempted_pairs": [],
        "i": -1,
    }
    (helix_dir / "state.json").write_text(json.dumps(legacy))


def test_load_legacy_state_populates_defaults(tmp_path: Path) -> None:
    """A v0 state.json (no schema_version) must load with new fields defaulted.

    Audit ref: D1 — migration path for unversioned state files.
    """
    _write_legacy_state_json(tmp_path)

    loaded = load_state(tmp_path)
    assert loaded is not None
    # The new fields default — empty dict for discovery, current schema_version.
    assert loaded.num_metric_calls_by_discovery == {}
    assert loaded.schema_version == SCHEMA_VERSION
    # Pre-existing fields still load correctly.
    assert loaded.generation == 1
    assert loaded.frontier == ["g0-s0"]
    assert loaded.budget.evaluations == 1
    assert loaded.budget.cached_input_tokens == 0
    assert loaded.budget.cache_creation_input_tokens == 0
    assert loaded.budget.cache_read_input_tokens == 0
    assert loaded.budget.reasoning_tokens == 0


def test_load_state_rejects_newer_schema_version(tmp_path: Path) -> None:
    """A future-version state.json must be rejected with a clear error.

    Audit ref: D1 — schema migration path (analogous to GEPA's schema-version
    check in ``GEPAState.load``, ``core/state.py``).
    """
    helix_dir = tmp_path / ".helix"
    helix_dir.mkdir(parents=True)
    payload = {
        "schema_version": SCHEMA_VERSION + 999,
        "generation": 0,
        "frontier": [],
        "instance_scores": {},
        "budget": {"evaluations": 0},
        "config_hash": "h",
    }
    (helix_dir / "state.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="newer than supported"):
        load_state(tmp_path)


# ---------------------------------------------------------------------------
# Eval cache persistence — companion pickle round-trip
# ---------------------------------------------------------------------------


def test_eval_cache_pickle_roundtrip(tmp_path: Path) -> None:
    """save_eval_cache → load_eval_cache must round-trip the tuple-keyed dict.

    Audit ref: C1 — JSON cannot encode tuple keys; we use a sibling pickle
    so the per-(candidate_hash, example_id) cache survives crash/resume the
    same way GEPA's pickled state does (``GEPAState.save``/``GEPAState.load``,
    ``core/state.py``).
    """
    cache: MinibatchEvalCache[object, str] = MinibatchEvalCache[object, str]()
    cache.put({"prompt.md": "v1"}, "task_a", output="out-a", score=0.4)
    cache.put({"prompt.md": "v1"}, "task_b", output="out-b", score=0.7)
    cache.put({"prompt.md": "v2"}, "task_a", output="out-a2", score=0.9)

    save_eval_cache(cache._cache, tmp_path)
    loaded = load_eval_cache(tmp_path)

    assert loaded == cache._cache
    # Spot-check tuple keys to make sure pickle preserved them as tuples (a
    # JSON path would have coerced them to strings or failed outright).
    sample_key = next(iter(loaded.keys()))
    assert isinstance(sample_key, tuple)
    assert len(sample_key) == 2


def test_eval_cache_load_returns_none_when_missing(tmp_path: Path) -> None:
    """load_eval_cache must return None when no companion pickle exists.

    Audit ref: C1 — fresh runs (no prior state) should not raise.
    """
    assert load_eval_cache(tmp_path) is None


def test_eval_cache_load_tolerates_non_dict(tmp_path: Path) -> None:
    """A corrupt eval_cache.pkl warns, quarantines, and resumes with an empty cache."""
    import pickle as _pickle

    helix_dir = tmp_path / ".helix"
    helix_dir.mkdir(parents=True)
    target = helix_dir / "eval_cache.pkl"
    target.write_bytes(_pickle.dumps(["not", "a", "dict"]))

    # The message deliberately does NOT name the quarantine path (it would
    # leak the machine's directory layout); the quarantine itself is asserted
    # below, so the behaviour is still pinned.
    with pytest.warns(RuntimeWarning, match="A diagnostic copy was retained"):
        assert load_eval_cache(tmp_path) is None

    # The corrupt file must be moved aside (not deleted) so the user can
    # post-mortem, and must not still occupy the canonical path (so the
    # next save doesn't have to overwrite it implicitly).
    assert not target.exists()
    quarantined = list(helix_dir.glob("eval_cache.pkl.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == _pickle.dumps(["not", "a", "dict"])


def test_eval_cache_load_tolerates_unreadable_pickle(tmp_path: Path) -> None:
    """Unreadable pickle bytes should not abort resume; file is quarantined."""
    helix_dir = tmp_path / ".helix"
    helix_dir.mkdir(parents=True)
    target = helix_dir / "eval_cache.pkl"
    target.write_bytes(b"not a pickle")

    with pytest.warns(RuntimeWarning, match="Ignoring unreadable eval cache"):
        assert load_eval_cache(tmp_path) is None

    assert not target.exists()
    quarantined = list(helix_dir.glob("eval_cache.pkl.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"not a pickle"


# ---------------------------------------------------------------------------
# Resume fidelity — N saves followed by load preserves cumulative state
# ---------------------------------------------------------------------------


def test_resume_preserves_cache_and_discovery_across_saves(tmp_path: Path) -> None:
    """Simulate N save→load cycles: cache and discovery budget must accumulate.

    Audit ref: C1 + C/§3.  This is the "resume fidelity" test from the task
    brief in compressed form — instead of running a full evolution we drive
    the persistence API directly so the test stays deterministic.

    The invariant under check: when val_stage_size is None (i.e. the
    GEPA-parity path), every save→load cycle leaves the state and cache
    bit-identical to what was last committed.  Any drift here would mean
    HELIX's resume diverges from GEPA's pickled-state behaviour.
    """
    # First "run": seed + 2 mutations.
    state = EvolutionState(
        generation=0,
        frontier=[],
        instance_scores={},
        budget=BudgetState(evaluations=0),
        config_hash="h",
    )
    cache: MinibatchEvalCache[object, str] = MinibatchEvalCache[object, str]()

    state.budget.evaluations = 5
    state.frontier.append("g0-s0")
    state.num_metric_calls_by_discovery["g0-s0"] = state.budget.evaluations
    cache.put({"prompt.md": "seed"}, "task_a", output="o", score=0.5)
    save_state(state, tmp_path)
    save_eval_cache(cache._cache, tmp_path)

    state.budget.evaluations = 12
    state.generation = 1
    state.mutation_counter = 1
    state.frontier.append("g1-s1")
    state.num_metric_calls_by_discovery["g1-s1"] = state.budget.evaluations
    cache.put({"prompt.md": "v1"}, "task_a", output="o", score=0.7)
    save_state(state, tmp_path)
    save_eval_cache(cache._cache, tmp_path)

    # Crash + resume — load both pieces from disk.
    loaded_state = load_state(tmp_path)
    loaded_cache_dict = load_eval_cache(tmp_path)
    assert loaded_state is not None
    assert loaded_cache_dict is not None

    # Discovery budget must show both candidates with their respective totals.
    assert loaded_state.num_metric_calls_by_discovery == {"g0-s0": 5, "g1-s1": 12}
    # Cache must contain both seed-era and gen-1 entries.
    assert len(loaded_cache_dict) == 2

    # Continue: do one more accept after resume, save again, reload.
    fresh_cache: MinibatchEvalCache[object, str] = MinibatchEvalCache[object, str]()
    fresh_cache._cache.update(loaded_cache_dict)
    loaded_state.budget.evaluations = 20
    loaded_state.generation = 2
    loaded_state.mutation_counter = 2
    loaded_state.frontier.append("g2-s2")
    loaded_state.num_metric_calls_by_discovery["g2-s2"] = loaded_state.budget.evaluations
    fresh_cache.put({"prompt.md": "v2"}, "task_b", output="o2", score=0.9)
    save_state(loaded_state, tmp_path)
    save_eval_cache(fresh_cache._cache, tmp_path)

    final_state = load_state(tmp_path)
    final_cache_dict = load_eval_cache(tmp_path)
    assert final_state is not None
    assert final_cache_dict is not None
    assert final_state.num_metric_calls_by_discovery == {
        "g0-s0": 5,
        "g1-s1": 12,
        "g2-s2": 20,
    }
    assert len(final_cache_dict) == 3
    # Cumulative cache: every (hash, example_id) entry written above must be present.
    assert final_cache_dict == fresh_cache._cache


# ---------------------------------------------------------------------------
# val_stage_size disabled → save_state on-disk format is GEPA-aligned
# ---------------------------------------------------------------------------


def test_state_json_keys_when_val_stage_disabled(tmp_path: Path) -> None:
    """The CORE INVARIANT: when val_stage_size is None/0 the persisted state
    contains the same fields as a baseline GEPA-aligned save.

    Audit ref: D1 — the on-disk schema must be stable so a GEPA-baseline
    state file produced by the previous HELIX (no val_stage_size) still
    round-trips after the schema additions land.  This test pins the exact
    set of keys we write so any future divergence shows up loudly.
    """
    state = _make_full_state()
    save_state(state, tmp_path)
    raw = json.loads((tmp_path / ".helix" / "state.json").read_text())
    expected_keys = {
        "schema_version",
        "generation",
        "frontier",
        "instance_scores",
        "budget",
        "config_hash",
        "mutation_counter",
        "merge_counter",
        "total_merge_invocations",
        "merge_attempted_pairs",
        "merge_description_triplets",
        "i",
        "num_metric_calls_by_discovery",
        "active_frontier",
        "frontier_type",
        "resume_semantics",
    }
    assert set(raw.keys()) == expected_keys, (
        f"state.json keys diverged from schema_version={SCHEMA_VERSION} "
        f"contract: {set(raw.keys()) ^ expected_keys}"
    )


def test_legacy_state_load_followed_by_save_writes_versioned_payload(tmp_path: Path) -> None:
    """Loading a legacy v0 state.json and re-saving must promote it to the
    current schema_version with new-field defaults present.

    Audit ref: D1 — completes the migration loop (load defaults → save with
    version bumped), so a one-time resume after the upgrade is sufficient
    to leave on-disk state at the current schema.
    """
    _write_legacy_state_json(tmp_path)
    loaded = load_state(tmp_path)
    assert loaded is not None
    save_state(loaded, tmp_path)

    raw = json.loads((tmp_path / ".helix" / "state.json").read_text())
    assert raw["schema_version"] == SCHEMA_VERSION
    assert raw["num_metric_calls_by_discovery"] == {}


def test_clear_eval_cache_removes_stale_pickle(tmp_path: Path) -> None:
    """Disabling cache must not leave a stale pickle for later resumes."""
    cache: MinibatchEvalCache[object, str] = MinibatchEvalCache[object, str]()
    cache.put({"prompt.md": "v1"}, "task_a", output="out-a", score=0.4)
    save_eval_cache(cache._cache, tmp_path)

    clear_eval_cache(tmp_path)

    assert load_eval_cache(tmp_path) is None


def test_eval_cache_warnings_do_not_leak_filesystem_paths(tmp_path):
    """Corrupt-cache warnings must not print absolute paths.

    These RuntimeWarnings reach the user directly.  Naming the on-disk
    location leaks the machine's directory layout for no diagnostic gain:
    the caller already knows which project it ran.  The quarantine copy is
    still written; only its path is withheld from the message.
    """
    import pickle
    import warnings as _warnings

    from helix.state import _eval_cache_path, load_eval_cache

    for payload in (b"not-a-pickle", pickle.dumps(["not", "a", "dict"])):
        base = tmp_path / f"case-{len(payload)}"
        (base / ".helix").mkdir(parents=True, exist_ok=True)
        _eval_cache_path(base).write_bytes(payload)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            assert load_eval_cache(base) is None

        assert caught, "expected a RuntimeWarning"
        for entry in caught:
            message = str(entry.message)
            assert str(base) not in message, message
            assert str(tmp_path) not in message, message
            assert ".corrupt-" not in message, message
