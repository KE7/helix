"""Unit tests for the ``helix_result`` score parser.

BREAKING (pre-1.0): ``helix_result`` is now **per-example** — the
evaluator emits a list of ``[score_i, side_info_i]`` pairs positional
to the ids in ``helix_batch.json``.  HELIX zips them into the id-keyed
``instance_scores`` the minibatch gate needs and stores the
per-example side_info list on :class:`helix.population.EvalResult` for
the reflection prompt.  See :mod:`helix.parsers.helix_result` and
``/tmp/gepa_audit_report.md`` for the GEPA parity rationale.

The parser is strict by design: any deviation from the contract raises
:class:`EvaluatorError` rather than silently falling back.  The footgun
it exists to eliminate is the silent zero-fill that hid id-mismatch
bugs for 113 generations in a real run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helix.exceptions import EvaluatorError
from helix.parsers import get_parser
from helix.parsers.helix_result import parse as helix_result_parse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_batch(worktree: Path, ids: list[str]) -> None:
    (worktree / "helix_batch.json").write_text(json.dumps(ids))


def _pairs(
    scores_list: list[float], side_infos: list[dict] | None = None
) -> list[list]:
    """Build a per-example ``[score, side_info]`` payload."""
    if side_infos is None:
        side_infos = [{} for _ in scores_list]
    assert len(scores_list) == len(side_infos)
    return [[s, si] for s, si in zip(scores_list, side_infos)]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHelixResultHappyPath:
    def test_zips_ids_with_per_example_scores(self, tmp_path: Path) -> None:
        ids = ["cube_lifting__0", "cube_stack__1", "nut_assembly__2"]
        _write_batch(tmp_path, ids)
        payload = _pairs([1.0, 0.0, 0.5])
        line = f"HELIX_RESULT={json.dumps(payload)}"
        stdout = f"preamble\n{line}\ntrailing\n"

        scores, instance_scores, pesi, obj = helix_result_parse(
            0, stdout, "", tmp_path
        )

        assert instance_scores == {
            "cube_lifting__0": 1.0,
            "cube_stack__1": 0.0,
            "nut_assembly__2": 0.5,
        }
        # mean of [1.0, 0.0, 0.5] == 0.5
        assert scores["success"] == pytest.approx(0.5)
        # Per-example side_info captured in id order (empty dicts here).
        assert pesi == [{}, {}, {}]
        # No side_info["scores"] key → empty objective harvest per example.
        assert obj == [{}, {}, {}]

    def test_success_is_mean_of_scores(self, tmp_path: Path) -> None:
        _write_batch(tmp_path, ["0", "1", "2", "3"])
        payload = _pairs([0.25, 0.5, 0.75, 1.0])
        line = f"HELIX_RESULT={json.dumps(payload)}"

        scores, _is, _pesi, _obj = helix_result_parse(0, line + "\n", "", tmp_path)

        assert scores["success"] == pytest.approx(0.625)

    def test_integer_and_bool_scores_coerced_to_float(self, tmp_path: Path) -> None:
        _write_batch(tmp_path, ["a", "b", "c", "d"])
        # bool is a subclass of int; the parser accepts both.
        payload = [[1, {}], [0, {}], [True, {}], [False, {}]]
        line = f"HELIX_RESULT={json.dumps(payload)}"

        scores, instance_scores, _pesi, _obj = helix_result_parse(
            0, line + "\n", "", tmp_path
        )

        assert instance_scores == {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0}
        assert scores["success"] == pytest.approx(0.5)

    def test_last_helix_result_line_wins(self, tmp_path: Path) -> None:
        # Mirrors executor's reverse-scan behaviour: the last line wins.
        _write_batch(tmp_path, ["x", "y"])
        first = f"HELIX_RESULT={json.dumps(_pairs([0.1, 0.1]))}"
        last = f"HELIX_RESULT={json.dumps(_pairs([0.9, 0.9]))}"
        stdout = f"{first}\n{last}\n"

        scores, instance_scores, _pesi, _obj = helix_result_parse(
            0, stdout, "", tmp_path
        )

        assert instance_scores == {"x": 0.9, "y": 0.9}
        assert scores["success"] == pytest.approx(0.9)

    def test_nonzero_returncode_zeros_success_keeps_instance_scores(
        self, tmp_path: Path
    ) -> None:
        _write_batch(tmp_path, ["a", "b"])
        line = f"HELIX_RESULT={json.dumps(_pairs([1.0, 1.0]))}"

        scores, instance_scores, _pesi, _obj = helix_result_parse(
            1, line + "\n", "", tmp_path
        )

        assert scores["success"] == 0.0
        # Per-instance scores are still surfaced — only the aggregate is zeroed.
        assert instance_scores == {"a": 1.0, "b": 1.0}

    def test_empty_batch_and_empty_payload(self, tmp_path: Path) -> None:
        # Degenerate but well-formed: HELIX asked for zero examples
        # (e.g. a len-0 minibatch) and the evaluator emitted an empty list.
        _write_batch(tmp_path, [])
        line = f"HELIX_RESULT={json.dumps([])}"

        scores, instance_scores, pesi, obj = helix_result_parse(
            0, line + "\n", "", tmp_path
        )

        assert scores == {"success": 0.0}
        assert instance_scores == {}
        assert pesi == []
        assert obj == []

    def test_per_example_side_info_passthrough(self, tmp_path: Path) -> None:
        """Each per-example ``side_info`` dict flows through verbatim,
        positional to the ids, for the reflection prompt."""
        ids = ["a", "b"]
        _write_batch(tmp_path, ids)
        side_a = {"trajectory": "tried X; fell over", "loss": 0.12}
        side_b = {"trajectory": "tried Y; stuck", "loss": 0.9, "note": "retry?"}
        payload = [[1.0, side_a], [0.0, side_b]]
        line = f"HELIX_RESULT={json.dumps(payload)}"

        _s, _is, pesi, _obj = helix_result_parse(0, line + "\n", "", tmp_path)

        assert pesi == [side_a, side_b]

    def test_null_per_example_side_info_becomes_empty_dict(
        self, tmp_path: Path
    ) -> None:
        """``null`` in the side_info slot is lenient → stored as ``{}``."""
        _write_batch(tmp_path, ["a", "b"])
        # JSON null parses to Python None.
        line = 'HELIX_RESULT=[[1.0, null], [0.5, null]]'

        _s, _is, pesi, obj = helix_result_parse(0, line + "\n", "", tmp_path)

        assert pesi == [{}, {}]
        assert obj == [{}, {}]


# ---------------------------------------------------------------------------
# Reserved side_info["scores"] → per-example objective_scores harvest
# ---------------------------------------------------------------------------


class TestObjectiveScoresHarvest:
    """Mirrors GEPA's ``OptimizeAnythingAdapter._process_side_info``
    (``optimize_anything_adapter.py:260-272``): each per-example
    ``side_info["scores"]`` dict is harvested into the corresponding
    slot of ``objective_scores``.  No frontier wiring yet (that's a
    later commit in this branch); this test just pins the harvest.
    """

    def test_scores_key_harvested_per_example(self, tmp_path: Path) -> None:
        ids = ["a", "b", "c"]
        _write_batch(tmp_path, ids)
        payload = [
            [1.0, {"scores": {"obj_alpha": 0.9, "obj_beta": 0.1}}],
            [0.5, {"scores": {"obj_alpha": 0.5}}],
            [0.0, {"no": "scores key here"}],
        ]
        line = f"HELIX_RESULT={json.dumps(payload)}"

        _s, _is, _pesi, obj = helix_result_parse(0, line + "\n", "", tmp_path)

        assert obj == [
            {"obj_alpha": 0.9, "obj_beta": 0.1},
            {"obj_alpha": 0.5},
            {},
        ]

    def test_scores_key_non_dict_ignored(self, tmp_path: Path) -> None:
        _write_batch(tmp_path, ["a"])
        payload = [[1.0, {"scores": "not-a-dict"}]]
        line = f"HELIX_RESULT={json.dumps(payload)}"

        _s, _is, _pesi, obj = helix_result_parse(0, line + "\n", "", tmp_path)

        assert obj == [{}]

    def test_scores_drops_non_numeric_and_non_finite(self, tmp_path: Path) -> None:
        _write_batch(tmp_path, ["a"])
        # Hand-rolled JSON to smuggle NaN / Infinity past json.dumps.
        stdout = (
            'HELIX_RESULT=[[1.0, {"scores": '
            '{"good": 0.5, "bad_str": "nope", "nan": NaN, "inf": Infinity}}]]\n'
        )

        _s, _is, _pesi, obj = helix_result_parse(0, stdout, "", tmp_path)

        assert obj == [{"good": 0.5}]

    def test_scores_harvest_coerces_bool_and_int(self, tmp_path: Path) -> None:
        _write_batch(tmp_path, ["a"])
        payload = [
            [1.0, {"scores": {"bool_true": True, "bool_false": False, "int_one": 1}}]
        ]
        line = f"HELIX_RESULT={json.dumps(payload)}"

        _s, _is, _pesi, obj = helix_result_parse(0, line + "\n", "", tmp_path)

        assert obj == [{"bool_true": 1.0, "bool_false": 0.0, "int_one": 1.0}]

    def test_side_info_scores_preserved_alongside_harvest(
        self, tmp_path: Path
    ) -> None:
        """The harvest does not strip ``side_info["scores"]`` from the
        per-example side_info — it still flows through for reflection."""
        _write_batch(tmp_path, ["a"])
        side_info = {"trajectory": "ok", "scores": {"latency_ms": 42.0}}
        payload = [[1.0, side_info]]
        line = f"HELIX_RESULT={json.dumps(payload)}"

        _s, _is, pesi, obj = helix_result_parse(0, line + "\n", "", tmp_path)

        assert pesi == [side_info]
        assert obj == [{"latency_ms": 42.0}]


# ---------------------------------------------------------------------------
# Strict contract — mismatch raises EvaluatorError
# ---------------------------------------------------------------------------


class TestHelixResultStrictness:
    def test_length_mismatch_too_short_raises(self, tmp_path: Path) -> None:
        _write_batch(tmp_path, ["a", "b", "c"])
        line = f"HELIX_RESULT={json.dumps(_pairs([1.0, 0.0]))}"  # len 2, ids len 3

        with pytest.raises(EvaluatorError, match="length mismatch"):
            helix_result_parse(0, line + "\n", "", tmp_path)

    def test_length_mismatch_too_long_raises(self, tmp_path: Path) -> None:
        _write_batch(tmp_path, ["a"])
        line = f"HELIX_RESULT={json.dumps(_pairs([1.0, 0.0, 0.5]))}"

        with pytest.raises(EvaluatorError, match="length mismatch"):
            helix_result_parse(0, line + "\n", "", tmp_path)

    def test_missing_helix_batch_raises(self, tmp_path: Path) -> None:
        # helix_batch.json intentionally absent.
        line = f"HELIX_RESULT={json.dumps(_pairs([1.0]))}"

        with pytest.raises(EvaluatorError, match="helix_batch.json"):
            helix_result_parse(0, line + "\n", "", tmp_path)

    def test_missing_helix_result_line_raises(self, tmp_path: Path) -> None:
        _write_batch(tmp_path, ["a", "b"])

        with pytest.raises(EvaluatorError, match="no HELIX_RESULT"):
            helix_result_parse(0, "boring stdout\nno result line\n", "", tmp_path)

    def test_malformed_helix_result_json_raises(self, tmp_path: Path) -> None:
        _write_batch(tmp_path, ["a"])

        with pytest.raises(EvaluatorError, match="JSON-decode"):
            helix_result_parse(
                0, "HELIX_RESULT=not-valid-json\n", "", tmp_path
            )

    def test_non_list_payload_raises(self, tmp_path: Path) -> None:
        _write_batch(tmp_path, ["a"])
        line = f"HELIX_RESULT={json.dumps({'score': 1.0})}"  # dict, not list

        with pytest.raises(EvaluatorError, match="list of per-example"):
            helix_result_parse(0, line + "\n", "", tmp_path)

    def test_legacy_scalar_plus_dict_shape_rejected_with_migration_hint(
        self, tmp_path: Path
    ) -> None:
        """Old (removed) contract: ``[float, {"scores": {...}}]``.  The
        new parser recognises that shape and rejects with a pointer to
        the migration guide rather than silently misbehaving."""
        _write_batch(tmp_path, ["a"])
        line = f"HELIX_RESULT={json.dumps([0.5, {'scores': {'a': 0.5}}])}"

        with pytest.raises(EvaluatorError, match="legacy shape"):
            helix_result_parse(0, line + "\n", "", tmp_path)

    def test_intermediate_batch_level_shape_rejected_with_hint(
        self, tmp_path: Path
    ) -> None:
        """Intermediate shape ``[scores_list, side_info_dict]`` (one
        batch-level side_info) is also rejected with a pointer to the
        per-example fix."""
        _write_batch(tmp_path, ["a", "b"])
        line = f"HELIX_RESULT={json.dumps([[1.0, 0.5], {'note': 'batch'}])}"

        with pytest.raises(EvaluatorError, match="per-example"):
            helix_result_parse(0, line + "\n", "", tmp_path)

    def test_per_example_entry_not_a_pair_raises(self, tmp_path: Path) -> None:
        """Each per-example entry must be a 2-element list."""
        _write_batch(tmp_path, ["a", "b"])
        # Second entry is a bare scalar instead of [score, side_info].
        line = f"HELIX_RESULT={json.dumps([[1.0, {}], 0.5])}"

        with pytest.raises(EvaluatorError, match="2-element"):
            helix_result_parse(0, line + "\n", "", tmp_path)

    def test_per_example_side_info_non_dict_raises(self, tmp_path: Path) -> None:
        """``side_info`` must be a dict (or null for leniency); a
        non-dict, non-null payload is rejected."""
        _write_batch(tmp_path, ["a"])
        line = f"HELIX_RESULT={json.dumps([[1.0, 'not-a-dict']])}"

        with pytest.raises(EvaluatorError, match="side_info"):
            helix_result_parse(0, line + "\n", "", tmp_path)

    def test_non_numeric_score_entry_raises(self, tmp_path: Path) -> None:
        _write_batch(tmp_path, ["a", "b"])
        line = f"HELIX_RESULT={json.dumps([[1.0, {}], ['oops', {}]])}"

        with pytest.raises(EvaluatorError, match="not numeric"):
            helix_result_parse(0, line + "\n", "", tmp_path)

    def test_non_finite_score_entry_raises(self, tmp_path: Path) -> None:
        _write_batch(tmp_path, ["a", "b"])
        # ``json.dumps`` rejects NaN unless we write it raw.
        stdout = 'HELIX_RESULT=[[1.0, {}], [NaN, {}]]\n'

        with pytest.raises(EvaluatorError, match="non-finite"):
            helix_result_parse(0, stdout, "", tmp_path)

    def test_helix_batch_not_a_list_raises(self, tmp_path: Path) -> None:
        (tmp_path / "helix_batch.json").write_text('{"not": "a list"}')
        line = f"HELIX_RESULT={json.dumps(_pairs([1.0]))}"

        with pytest.raises(EvaluatorError, match="list\\[str\\]"):
            helix_result_parse(0, line + "\n", "", tmp_path)


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestHelixResultRegistry:
    def test_get_parser_returns_helix_result(self) -> None:
        assert get_parser("helix_result") is helix_result_parse
