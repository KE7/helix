"""Unit tests for the leak-free circle-packing run (examples/circle_packing).

Covers the validity rules and certified score, the candidate-side validity
checker, the independent exact-arithmetic checker, the evaluator service's
input handling, ledger, archives and execution cap, the evaluator client's
archive, the egress gateway policy, the run configuration template and the
summary's accounting helpers.
The Docker-level isolation checks live in
``examples/circle_packing/clean_run/scripts/isolation_probe.py``.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import random

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "circle_packing"
CLEAN = EXAMPLE / "clean_run"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rules = _load("packing_rules", CLEAN / "evaluator" / "packing_rules.py")
checker = _load("cp_check_packing", EXAMPLE / "check_packing.py")
exact = _load("cp_independent_check", CLEAN / "scripts" / "independent_check.py")
harness = _load("cp_candidate_harness", CLEAN / "evaluator" / "candidate_harness.py")
server = _load("cp_evaluator_server", CLEAN / "evaluator" / "evaluator_server.py")
client = _load("cp_evaluator_client", CLEAN / "evaluator" / "evaluator_client.py")
gateway = _load("cp_anthropic_gateway", CLEAN / "gateway" / "anthropic_gateway.py")
summarize = _load("cp_summarize", CLEAN / "scripts" / "summarize.py")


R = 1.0 / 16.0


def grid_packing() -> list[list[float]]:
    """A naive valid packing: 26 cells of an 8x8 grid, radius 1/16 each.

    Dyadic coordinates are exact in binary floating point, so neighbouring
    circles touch exactly and the packing is valid even at tolerance 0.
    """
    return [[R + (i % 8) * 2 * R, R + (i // 8) * 2 * R, R] for i in range(26)]


def with_row(packing: list[list[Any]], index: int, row: Any) -> list[Any]:
    changed = [list(p) for p in packing]
    changed[index] = row
    return changed


# ---------------------------------------------------------------------------
# Strict rules
# ---------------------------------------------------------------------------


class TestPackingRules:
    def test_valid_grid_scores_its_exact_radius_sum(self) -> None:
        packing = grid_packing()
        verdict = rules.evaluate_packing(packing)
        assert verdict["valid"] is True
        assert verdict["errors"] == []
        assert verdict["score"] == math.fsum(p[2] for p in packing)

    def test_tuple_rows_and_int_values_are_accepted(self) -> None:
        packing = [tuple(p) for p in grid_packing()]
        packing[0] = (0, 0, 0)  # a zero-radius point at the square's corner
        verdict = rules.evaluate_packing(tuple(packing))
        assert verdict["valid"] is True

    def test_touching_circles_are_valid(self) -> None:
        # The grid circles touch their neighbours exactly.
        assert rules.evaluate_packing(grid_packing())["details"]["max_pair_overlap"] == 0.0

    def test_tolerance_matches_optimize_anything(self) -> None:
        assert rules.TOLERANCE == 1e-6

    def test_overlap_beyond_tolerance_scores_zero(self) -> None:
        packing = grid_packing()
        packing[1][0] -= 2e-6  # move circle 1 into circle 0
        verdict = rules.evaluate_packing(packing)
        assert verdict["valid"] is False
        assert verdict["score"] == 0.0
        assert verdict["details"]["overlap_violations"] >= 1
        assert "overlap" in " ".join(verdict["errors"])

    def test_overlap_within_tolerance_is_valid_and_shrunk(self) -> None:
        packing = grid_packing()
        packing[1][0] -= 5e-7
        verdict = rules.evaluate_packing(packing)
        assert verdict["valid"] is True
        shrink = verdict["details"]["certification_shrink"]
        assert shrink == pytest.approx(2.5e-7, rel=1e-6)
        assert verdict["score"] == math.fsum(max(0.0, p[2] - shrink) for p in packing)
        assert verdict["score"] < math.fsum(p[2] for p in packing)

    @pytest.mark.parametrize(
        "row",
        [
            [R - 2e-6, R, R],  # left edge
            [R, R - 2e-6, R],  # bottom edge
            [0.95, 0.95, 0.1],  # right/top edges
            [2.0, 2.0, 0.01],  # entirely outside
            [-0.5, 0.5, 0.0],  # a point outside
        ],
    )
    def test_out_of_bounds_scores_zero(self, row: list[float]) -> None:
        verdict = rules.evaluate_packing(with_row(grid_packing(), 0, row))
        assert verdict["valid"] is False
        assert verdict["score"] == 0.0

    def test_boundary_within_tolerance_is_valid_and_shrunk(self) -> None:
        verdict = rules.evaluate_packing(with_row(grid_packing(), 0, [R - 5e-7, R, R]))
        assert verdict["valid"] is True
        assert verdict["details"]["certification_shrink"] == pytest.approx(5e-7, rel=1e-6)
        assert verdict["score"] == pytest.approx(26 / 16 - 26 * 5e-7, abs=1e-15)

    @pytest.mark.parametrize("count", [0, 1, 25, 27, 100])
    def test_wrong_count_scores_zero(self, count: int) -> None:
        packing = (grid_packing() * 5)[:count]
        verdict = rules.evaluate_packing(packing)
        assert verdict["valid"] is False and verdict["score"] == 0.0
        assert f"got {count}" in verdict["errors"][0]

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    @pytest.mark.parametrize("column", [0, 1, 2])
    def test_non_finite_values_score_zero(self, bad: float, column: int) -> None:
        row = list(grid_packing()[3])
        row[column] = bad
        verdict = rules.evaluate_packing(with_row(grid_packing(), 3, row))
        assert verdict["valid"] is False and verdict["score"] == 0.0
        assert "NaN or infinite" in verdict["errors"][0]

    def test_negative_radius_scores_zero(self) -> None:
        verdict = rules.evaluate_packing(with_row(grid_packing(), 2, [0.5, 0.5, -0.01]))
        assert verdict["valid"] is False and verdict["score"] == 0.0
        assert verdict["details"]["negative_radii"] == 1

    def test_negative_radii_cannot_offset_huge_ones(self) -> None:
        # A negative radius next to an oversized one keeps each pair's radius
        # sum small; the per-circle rules still reject both.
        packing = with_row(grid_packing(), 0, [0.5, 0.5, 3.0])
        packing = with_row(packing, 1, [0.5, 0.5, -2.9])
        verdict = rules.evaluate_packing(packing)
        assert verdict["valid"] is False and verdict["score"] == 0.0

    @pytest.mark.parametrize("huge", [1e3, 1e300, 1e308, 10**400])
    def test_huge_radii_score_zero(self, huge: float) -> None:
        verdict = rules.evaluate_packing(with_row(grid_packing(), 0, [0.5, 0.5, huge]))
        assert verdict["valid"] is False and verdict["score"] == 0.0
        json.dumps(server._json_safe(verdict), allow_nan=False)

    def test_all_huge_radii_do_not_overflow(self) -> None:
        packing = [[0.5, 0.5, 1e308] for _ in range(26)]
        verdict = rules.evaluate_packing(packing)
        assert verdict["valid"] is False and verdict["score"] == 0.0
        json.dumps(server._json_safe(verdict), allow_nan=False)

    @pytest.mark.parametrize(
        "row",
        [
            [True, 0.5, 0.1],
            [0.5, 0.5, "0.1"],
            [0.5, None, 0.1],
            [0.5, 0.5, [0.1]],
            [0.5, 0.5, {"r": 0.1}],
            [0.5, 0.5, complex(0.1, 0)],
            [0.5, 0.5],
            [0.5, 0.5, 0.1, 0.0],
            "0.5,0.5,0.1",
            None,
            {"x": 0.5, "y": 0.5, "r": 0.1},
        ],
    )
    def test_malformed_rows_score_zero(self, row: Any) -> None:
        verdict = rules.evaluate_packing(with_row(grid_packing(), 5, row))
        assert verdict["valid"] is False and verdict["score"] == 0.0

    @pytest.mark.parametrize("packing", [None, "circles", 26, {"circles": []}, b"xyz"])
    def test_non_list_packings_score_zero(self, packing: Any) -> None:
        verdict = rules.evaluate_packing(packing)
        assert verdict["valid"] is False and verdict["score"] == 0.0

    def test_all_zero_radii_is_valid_with_score_zero(self) -> None:
        packing = [[0.5, 0.5, 0.0] for _ in range(26)]
        verdict = rules.evaluate_packing(packing)
        assert verdict["valid"] is True and verdict["score"] == 0.0

    def test_duplicate_circles_overlap(self) -> None:
        packing = with_row(grid_packing(), 1, list(grid_packing()[0]))
        assert rules.evaluate_packing(packing)["valid"] is False

    def test_inflating_radii_within_tolerance_gains_nothing(self) -> None:
        # Growing every radius of an exactly valid packing by just under
        # tol/2 stays valid, but the certification shrink takes it back.
        base = grid_packing()
        inflated = [[x, y, r + 0.49e-6] for x, y, r in base]
        v_base = rules.evaluate_packing(base)
        v_inf = rules.evaluate_packing(inflated)
        assert v_inf["valid"] is True
        assert v_inf["details"]["radius_sum_before_shrink"] > v_base["score"]
        assert v_inf["score"] <= v_base["score"] + 1e-14
        too_much = [[x, y, r + 2e-6] for x, y, r in base]
        assert rules.evaluate_packing(too_much)["valid"] is False

    def test_deterministic_and_order_independent(self) -> None:
        packing = grid_packing()
        first = rules.evaluate_packing(packing)
        assert rules.evaluate_packing(packing) == first
        assert rules.evaluate_packing(list(reversed(packing)))["score"] == first["score"]

    def test_score_never_exceeds_a_certified_exactly_valid_sum(self) -> None:
        # Random perturbations of an exactly valid packing, many of them
        # valid only up to the tolerance: every score must be attainable by a
        # packing that satisfies the constraints exactly (checked in exact
        # rational arithmetic by the independent checker).
        rng = random.Random(20260916)
        checked = 0
        for _ in range(60):
            packing = [
                [x + rng.uniform(-2e-7, 2e-7), y + rng.uniform(-2e-7, 2e-7), r + rng.uniform(-1e-7, 2e-7)]
                for x, y, r in grid_packing()
            ]
            verdict = rules.evaluate_packing(packing)
            if not verdict["valid"]:
                assert verdict["score"] == 0.0
                continue
            checked += 1
            result = exact.check(packing, claimed_score=verdict["score"])
            assert result["claimed_score_within_certified_sum"] is True, result
            raw = exact.check(packing, claimed_score=verdict["details"]["radius_sum_before_shrink"])
            if verdict["details"]["certification_shrink"] > 1e-12:
                assert raw["claimed_score_within_certified_sum"] is False
        assert checked >= 10

    def test_score_never_exceeds_the_radius_sum(self) -> None:
        cases = [grid_packing(), with_row(grid_packing(), 0, [0.5, 0.5, 0.4])]
        for packing in cases:
            verdict = rules.evaluate_packing(packing)
            radius_sum = math.fsum(p[2] for p in packing)
            assert verdict["score"] <= radius_sum
            if not verdict["valid"]:
                assert verdict["score"] == 0.0


# ---------------------------------------------------------------------------
# Candidate-side checker and exact checker agree with the rules
# ---------------------------------------------------------------------------


def _cases() -> list[Any]:
    grid = grid_packing()
    overlap = [list(p) for p in grid]
    overlap[1][0] -= 2e-6
    within = [list(p) for p in grid]
    within[1][0] -= 5e-7
    return [
        within,
        grid,
        overlap,
        with_row(grid, 0, [0.5, 0.5, 0.4]),
        with_row(grid, 0, [2.0, 2.0, 0.01]),
        with_row(grid, 3, [math.nan, 0.5, 0.1]),
        with_row(grid, 3, [True, 0.5, 0.1]),
        with_row(grid, 2, [0.5, 0.5, -0.01]),
        grid[:25],
        None,
        [[0.5, 0.5, 0.0] for _ in range(26)],
    ]


@pytest.mark.parametrize("case", _cases())
def test_candidate_checker_agrees_with_rules(case: Any) -> None:
    assert (checker.violations(case) == []) is rules.evaluate_packing(case)["valid"]


def test_candidate_checker_reports_no_score() -> None:
    source = (EXAMPLE / "check_packing.py").read_text()
    assert "fsum" not in source and "sum(" not in source


@pytest.mark.parametrize("case", [c for c in _cases() if rules.evaluate_packing(c)["valid"]])
def test_candidate_checker_residual_matches_the_evaluator_shrink(case: Any) -> None:
    assert checker.residual_violation(case) == rules.evaluate_packing(case)["details"]["certification_shrink"]


class TestIndependentCheck:
    def test_exactly_valid_grid(self) -> None:
        result = exact.check(grid_packing())
        assert result["exactly_valid"] is True
        assert result["valid_at_tolerance"] is True
        assert result["uniform_shrink_for_exact_validity"] == 0.0
        assert result["certified_exactly_valid_radius_sum"] == 26 / 16

    def test_overlap_within_tolerance_is_not_exactly_valid(self) -> None:
        packing = grid_packing()
        packing[1][0] -= 5e-10
        result = exact.check(packing)
        assert result["valid_at_tolerance"] is True
        assert result["exactly_valid"] is False
        assert result["overlapping_pairs_exact"] >= 1
        shrink = result["uniform_shrink_for_exact_validity"]
        assert 0 < shrink <= 5e-10
        certified = result["certified_exactly_valid_radius_sum"]
        assert certified < result["radius_sum"]

    def test_default_tolerance_is_the_evaluators(self) -> None:
        assert exact.check(grid_packing())["tolerance"] == rules.TOLERANCE

    def test_overlap_beyond_tolerance(self) -> None:
        packing = grid_packing()
        packing[1][0] -= 2e-6
        result = exact.check(packing)
        assert result["valid_at_tolerance"] is False
        assert result["overlapping_pairs_over_tolerance"] >= 1

    @pytest.mark.parametrize("bad", [grid_packing()[:25], [[0.5, 0.5, math.nan]] * 26])
    def test_rejects_malformed(self, bad: Any) -> None:
        with pytest.raises(ValueError):
            exact.check(bad)


# ---------------------------------------------------------------------------
# Candidate harness conversion
# ---------------------------------------------------------------------------


class TestHarness:
    def test_rows_convert_reals_and_mark_everything_else(self) -> None:
        rows = harness._rows([(0.1, 1, True), ("a", None, 2.5), 7])
        assert rows[0][:2] == [0.1, 1.0]
        assert rows[0][2] == {"__non_numeric__": "bool"}
        assert rows[1][0] == {"__non_numeric__": "str"}
        assert rows[1][2] == 2.5
        assert rows[2] == {"__non_row__": "int"}
        verdict = rules.evaluate_packing(rows)
        assert verdict["valid"] is False

    def test_too_many_rows_is_an_error(self) -> None:
        with pytest.raises(ValueError):
            harness._rows([[0.0, 0.0, 0.0]] * (harness.MAX_ROWS + 1))

    def test_harness_end_to_end(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "solve.py").write_text(
            "def pack_circles(n):\n"
            "    r = 1.0 / 12.0\n"
            "    return [(r + (i % 6) * 2 * r, r + (i // 6) * 2 * r, r) for i in range(n)]\n"
        )
        out = tmp_path / "out.json"
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["candidate_harness.py", str(out)])
        monkeypatch.setattr(harness, "_limit_resources", lambda: None)
        sys.modules.pop("solve", None)
        try:
            assert harness.main() == 0
        finally:
            sys.modules.pop("solve", None)
            if str(tmp_path) in sys.path:
                sys.path.remove(str(tmp_path))
        payload = json.loads(out.read_text())
        assert payload["status"] == "ok"
        assert rules.evaluate_packing(payload["circles"])["valid"] is True


# ---------------------------------------------------------------------------
# Evaluator service
# ---------------------------------------------------------------------------


def _targz(entries: list[tarfile.TarInfo], data: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for info in entries:
            payload = data.get(info.name)
            tar.addfile(info, io.BytesIO(payload) if payload is not None else None)
    return buffer.getvalue()


class TestEvaluatorService:
    def test_safe_extract_keeps_regular_files_only(self, tmp_path: Path) -> None:
        good = tarfile.TarInfo("pkg/solve.py")
        good.size = 5
        link = tarfile.TarInfo("evil_link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/ledger/evaluations.jsonl"
        archive = _targz([good, link], {"pkg/solve.py": b"x = 1"})
        files = server.safe_extract(archive, tmp_path)
        assert files == [{"path": "pkg/solve.py", "bytes": 5}]
        assert (tmp_path / "pkg" / "solve.py").read_bytes() == b"x = 1"
        assert not (tmp_path / "evil_link").exists()

    @pytest.mark.parametrize("name", ["../escape.py", "/abs/solve.py", "a/../../b.py"])
    def test_safe_extract_refuses_escaping_paths(self, tmp_path: Path, name: str) -> None:
        info = tarfile.TarInfo(name)
        info.size = 1
        archive = _targz([info], {name: b"x"})
        with pytest.raises(server.ArchiveError):
            server.safe_extract(archive, tmp_path / "dest")

    def test_safe_extract_refuses_garbage(self, tmp_path: Path) -> None:
        with pytest.raises(server.ArchiveError):
            server.safe_extract(b"not a tarball", tmp_path)

    @pytest.mark.parametrize(
        "peer,local",
        [
            ("127.0.0.1", True),
            ("127.3.4.5", True),
            ("::ffff:127.0.0.1", True),
            ("172.20.0.2", True),
            ("172.20.0.3", False),
            ("::1", True),
        ],
    )
    def test_local_peer_detection(self, peer: str, local: bool) -> None:
        assert server.is_local_peer(peer, {"127.0.0.1", "::1", "172.20.0.2"}) is local

    def test_candidate_output_symlink_is_refused(self, tmp_path: Path) -> None:
        secret = tmp_path / "secret.jsonl"
        secret.write_text('{"status": "ok", "circles": []}')
        link = tmp_path / "out.json"
        link.symlink_to(secret)
        payload, error = server._read_candidate_output(link)
        assert payload is None and error

    def test_candidate_output_nan_tokens_become_invalid_packing(self, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        rows = grid_packing()
        text = json.dumps({"status": "ok", "circles": rows})
        text = text.replace(str(rows[0][0]), "NaN", 1)
        out.write_text(text)
        payload, error = server._read_candidate_output(out)
        assert error is None
        verdict = rules.evaluate_packing(payload["circles"])
        assert verdict["valid"] is False and verdict["score"] == 0.0

    def test_side_info_scores_zero_for_invalid_and_is_strict_json(self) -> None:
        bad = with_row(grid_packing(), 0, [0.5, 0.5, 1e308])
        verdict = rules.evaluate_packing(bad)
        side_info = server.build_side_info(verdict, {"wall_seconds": 1.0}, bad)
        assert side_info["scores"] == {"sum_radii": 0.0}
        assert side_info["valid"] is False
        json.dumps(side_info, allow_nan=False)
        assert "circles_json" in side_info

    def _evaluate(
        self, tmp_path: Path, monkeypatch, body: dict[str, Any], run_result: dict[str, Any]
    ) -> tuple[dict[str, Any], Any]:
        import pwd

        me = pwd.getpwuid(os.getuid())
        monkeypatch.setattr(server, "LEDGER_DIR", tmp_path / "ledger")
        monkeypatch.setattr(server, "WORK_ROOT", tmp_path / "work")
        (tmp_path / "work").mkdir()
        monkeypatch.setattr(server.pwd, "getpwnam", lambda name: me)
        monkeypatch.setattr(server, "chown_tree", lambda *a: None)
        monkeypatch.setattr(server, "kill_user_processes", lambda *a, **k: 0)
        monkeypatch.setattr(server, "remove_owned_ipc", lambda uid: [])
        monkeypatch.setattr(server, "run_candidate", lambda *a, **k: dict(run_result))
        ledger = server.Ledger(tmp_path / "ledger")
        response = server.evaluate_request(body, ledger, "172.20.0.9")
        return response, ledger

    def _body(self, tree: dict[str, bytes], split: str = "train") -> dict[str, Any]:
        infos = []
        for name, data in tree.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            infos.append(info)
        archive = _targz(infos, tree)
        return {
            "protocol": 1,
            "split": split,
            "ids": ["0"],
            "candidate_id": "g1-s1",
            "archive_b64": base64.b64encode(archive).decode(),
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "files_sha256": "f" * 64,
        }

    def test_every_execution_is_ledgered_in_order(self, tmp_path: Path, monkeypatch) -> None:
        body = self._body({"solve.py": b"# candidate"})
        body2 = self._body({"solve.py": b"# another candidate"})
        ok = {
            "payload": {"status": "ok", "circles": grid_packing()},
            "timed_out": False,
            "exit_code": 0,
            "wall_seconds": 0.5,
        }
        response, ledger = self._evaluate(tmp_path, monkeypatch, body, ok)
        [[score, side_info]] = response["results"]
        assert score == 26 / 16 and side_info["valid"] is True
        timeout = {"payload": None, "timed_out": True, "output_error": "timed out"}
        monkeypatch.setattr(server, "run_candidate", lambda *a, **k: dict(timeout))
        response2 = server.evaluate_request(body2, ledger, "172.20.0.9")
        assert response2["results"][0][0] == 0.0
        assert "timed out" in response2["results"][0][1]["feedback"]
        lines = [
            json.loads(line)
            for line in (tmp_path / "ledger" / "evaluations.jsonl").read_text().splitlines()
        ]
        assert [line["seq"] for line in lines] == [1, 2]
        assert [line["valid"] for line in lines] == [True, False]
        assert lines[0]["circles"] == grid_packing()
        # Each scored tree is kept, byte for byte.
        archive = tmp_path / "ledger" / lines[0]["archive"]
        assert archive.read_bytes() == base64.b64decode(body["archive_b64"])
        assert lines[0]["rules"]["tolerance"] == rules.TOLERANCE
        assert lines[0]["candidate_id"] == "g1-s1"
        assert not (tmp_path / "work" / response["evaluation_id"]).exists()
        # A restarted service continues the count.
        assert server.Ledger(tmp_path / "ledger").seq == 2
        starts = (tmp_path / "ledger" / "execution_starts.jsonl").read_text().splitlines()
        assert [json.loads(line)["seq"] for line in starts] == [1, 2]

    @pytest.mark.parametrize("valid", [False, True])
    def test_candidate_claimed_scores_cannot_replace_geometry(
        self, tmp_path: Path, monkeypatch, valid: bool
    ) -> None:
        packing = grid_packing() if valid else [[0.5, 0.5, 0.5]] * 26
        run = {
            "payload": {"status": "ok", "circles": packing, "score": 999999},
            "stdout_tail": 'HELIX_RESULT=[[999999, {}]]',
            "exit_code": 0,
        }
        response, ledger = self._evaluate(
            tmp_path, monkeypatch, self._body({"solve.py": b"# claimed result"}), run
        )
        [[score, side_info]] = response["results"]
        assert score == (26 / 16 if valid else 0.0)
        assert side_info["valid"] is valid
        assert ledger.executions == 1

    def _ok(self, score_rows: list[list[float]] | None = None) -> dict[str, Any]:
        return {"payload": {"status": "ok", "circles": score_rows or grid_packing()},
                "timed_out": False, "exit_code": 0, "wall_seconds": 0.5}

    def test_identical_tree_is_answered_from_cache_without_executing(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        body = self._body({"solve.py": b"# candidate"})
        first, ledger = self._evaluate(tmp_path, monkeypatch, body, self._ok())
        calls = {"n": 0}

        def must_not_run(*a: Any, **k: Any) -> dict[str, Any]:
            calls["n"] += 1
            raise AssertionError("a cached tree must not be executed again")

        monkeypatch.setattr(server, "run_candidate", must_not_run)
        # The same tree as a parent re-evaluation, then as a validation call.
        again = server.evaluate_request(body, ledger, "172.20.0.9")
        val = server.evaluate_request({**body, "split": "val"}, ledger, "172.20.0.9")
        assert calls["n"] == 0
        assert again["results"] == first["results"] == val["results"]
        assert again["cached_from_seq"] == 1 and ledger.seq == 1
        records = (tmp_path / "ledger" / "evaluations.jsonl").read_text().splitlines()
        assert len(records) == 1
        requests = [json.loads(t) for t in (tmp_path / "ledger" / "requests.jsonl").read_text().splitlines()]
        assert [(r["kind"], r["seq"], r["split"]) for r in requests] == [
            ("execution", 1, "train"), ("cache_hit", 1, "train"), ("cache_hit", 1, "val")]
        # The cache survives a restart of the service.
        assert server.Ledger(tmp_path / "ledger").cached(body["archive_sha256"]) is not None

    def test_a_tampered_archive_is_never_served_from_cache(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        body = self._body({"solve.py": b"# candidate"})
        _, ledger = self._evaluate(tmp_path, monkeypatch, body, self._ok())
        other = self._body({"solve.py": b"# other"})
        forged = {**other, "archive_sha256": body["archive_sha256"]}
        response = server.evaluate_request(forged, ledger, "172.20.0.9")
        assert ledger.seq == 2 and response["results"][0][0] == 0.0
        assert "checksum mismatch" in response["results"][0][1]["feedback"]

    def test_spent_budget_refuses_new_rounds_but_finishes_the_last_validation(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(server, "MAX_EXECUTIONS", 2)
        seed = self._body({"solve.py": b"# seed"}, split="val")
        _, ledger = self._evaluate(tmp_path, monkeypatch, seed, self._ok())
        child = self._body({"solve.py": b"# child"})
        server.evaluate_request(child, ledger, "172.20.0.9")
        assert ledger.exhausted() and (tmp_path / "ledger" / server.BUDGET_MARKER).exists()
        # The child that used the last execution still gets its validation.
        val = server.evaluate_request({**child, "split": "val"}, ledger, "172.20.0.9")
        assert val["cached_from_seq"] == 2
        # A new round (a parent re-evaluation, even a cached one) is refused,
        # and so is anything that would need another execution.
        with pytest.raises(server.BudgetExhausted):
            server.evaluate_request(child, ledger, "172.20.0.9")
        with pytest.raises(server.BudgetExhausted):
            server.evaluate_request(self._body({"solve.py": b"# new"}, split="val"), ledger, "172.20.0.9")
        assert ledger.seq == 2

    def test_an_unfinished_execution_number_is_never_reused(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        ledger = server.Ledger(tmp_path / "ledger")
        assert ledger.start_execution({"split": "train"}) == 1
        # The service dies before recording seq 1.
        restarted = server.Ledger(tmp_path / "ledger")
        assert restarted.seq == 1 and restarted.executions == 0
        assert restarted.start_execution({"split": "train"}) == 2

    def test_a_raising_evaluation_is_still_recorded(self, tmp_path: Path, monkeypatch) -> None:
        def boom(*a: Any, **k: Any) -> dict[str, Any]:
            raise RuntimeError("disk full")

        body = self._body({"solve.py": b"#"})
        monkeypatch.setattr(server, "run_candidate", boom)
        monkeypatch.setattr(server, "LEDGER_DIR", tmp_path / "ledger")
        monkeypatch.setattr(server, "WORK_ROOT", tmp_path / "work")
        (tmp_path / "work").mkdir()
        import pwd

        me = pwd.getpwuid(os.getuid())
        monkeypatch.setattr(server.pwd, "getpwnam", lambda name: me)
        monkeypatch.setattr(server, "chown_tree", lambda *a: None)
        monkeypatch.setattr(server, "kill_user_processes", lambda *a, **k: 0)
        monkeypatch.setattr(server, "remove_owned_ipc", lambda uid: [])
        ledger = server.Ledger(tmp_path / "ledger")
        response = server.evaluate_request(body, ledger, "172.20.0.9")
        assert response["results"][0][0] == 0.0
        [record] = [json.loads(t) for t in (tmp_path / "ledger" / "evaluations.jsonl").read_text().splitlines()]
        assert record["seq"] == 1 and "RuntimeError: disk full" in record["internal_error"]
        # An internal error is not cached: the tree is executed again next time.
        assert ledger.cached(body["archive_sha256"]) is None

    def test_deadline_monitor_records_the_overdue_execution_and_exits(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        ledger = server.Ledger(tmp_path / "ledger")
        seq = ledger.start_execution({"split": "train"})
        in_flight = server.InFlight()
        in_flight.begin({"seq": seq, "split": "train", "evaluation_id": "e"})
        in_flight.started -= 1000
        monkeypatch.setattr(server, "IN_FLIGHT", in_flight)

        class Exited(Exception):
            pass

        def fake_exit(code: int) -> None:
            raise Exited(code)

        monkeypatch.setattr(server.os, "_exit", fake_exit)
        with pytest.raises(Exited):
            server.deadline_monitor(ledger, limit=10.0, poll=0.01)
        [record] = [json.loads(t) for t in (tmp_path / "ledger" / "evaluations.jsonl").read_text().splitlines()]
        assert record["seq"] == seq and record["score"] == 0.0
        assert "deadline" in record["errors"][0]
        events = (tmp_path / "ledger" / "service_events.jsonl").read_text()
        assert "evaluation_aborted" in events

    @pytest.mark.parametrize("kind", ["fifo", "directory", "missing"])
    def test_candidate_output_that_is_not_a_regular_file_never_blocks(
        self, tmp_path: Path, kind: str
    ) -> None:
        import threading

        out = tmp_path / "out.json"
        if kind == "fifo":
            os.mkfifo(out)
        elif kind == "directory":
            out.mkdir()
        result: dict[str, Any] = {}
        thread = threading.Thread(
            target=lambda: result.update(value=server._read_candidate_output(out)), daemon=True
        )
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive(), "reading the candidate output blocked"
        payload, error = result["value"]
        assert payload is None and error

    def test_stored_stream_is_capped_to_its_tail(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(server, "STORED_STREAM_BYTES", 10)
        full = tmp_path / "stdout.txt"
        full.write_bytes(b"0123456789abcdefghij")
        stored = tmp_path / "stored.txt"
        assert server._store_stream(full, stored) == 20
        assert stored.read_bytes() == b"abcdefghij"

    def test_stale_output_file_in_tree_is_removed(self, tmp_path: Path, monkeypatch) -> None:
        body = self._body({"solve.py": b"#", server.OUTPUT_NAME: b'{"status":"ok"}'})
        seen: dict[str, bool] = {}

        def fake_run(workdir: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
            seen["stale_present"] = (workdir / server.OUTPUT_NAME).exists()
            return {"payload": None, "output_error": "no output"}

        self._evaluate(tmp_path, monkeypatch, self._body({"solve.py": b"# first"}), {})
        monkeypatch.setattr(server, "run_candidate", fake_run)
        server.evaluate_request(body, server.Ledger(tmp_path / "ledger"), "172.20.0.9")
        assert seen == {"stale_present": False}

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda b: b.update(archive_sha256="0" * 64),
            lambda b: b.update(archive_b64="!!!"),
            lambda b: b.pop("archive_b64"),
            lambda b: b.update(client_error="tree too large"),
        ],
    )
    def test_bad_trees_still_count_and_score_zero(
        self, tmp_path: Path, monkeypatch, mutate: Any
    ) -> None:
        body = self._body({"solve.py": b"#"})
        mutate(body)
        response, ledger = self._evaluate(tmp_path, monkeypatch, body, {})
        assert response["results"][0][0] == 0.0
        assert ledger.seq == 1

    @pytest.mark.parametrize(
        "change",
        [{"ids": ["0", "1"]}, {"ids": []}, {"protocol": 2}, {"ids": [0]}],
    )
    def test_malformed_requests_are_refused_without_a_ledger_entry(
        self, tmp_path: Path, monkeypatch, change: dict[str, Any]
    ) -> None:
        body = {**self._body({"solve.py": b"#"}), **change}
        with pytest.raises(ValueError):
            self._evaluate(tmp_path, monkeypatch, body, {})
        assert not (tmp_path / "ledger" / "evaluations.jsonl").exists()


class TestEvaluatorHttp:
    """The HTTP front: local-peer refusal and the execution budget."""

    def _serve(self, tmp_path: Path, monkeypatch, *, cap: int, seq: int) -> tuple[Any, Any]:
        import pwd
        import threading
        from http.server import ThreadingHTTPServer

        me = pwd.getpwuid(os.getuid())
        monkeypatch.setattr(server, "LEDGER_DIR", tmp_path / "ledger")
        monkeypatch.setattr(server, "WORK_ROOT", tmp_path / "work")
        (tmp_path / "work").mkdir(exist_ok=True)
        monkeypatch.setattr(server.pwd, "getpwnam", lambda name: me)
        monkeypatch.setattr(server, "chown_tree", lambda *a: None)
        monkeypatch.setattr(server, "kill_user_processes", lambda *a, **k: 0)
        monkeypatch.setattr(server, "remove_owned_ipc", lambda uid: [])
        monkeypatch.setattr(server, "run_candidate", lambda *a, **k: {
            "payload": {"status": "ok", "circles": grid_packing()}, "timed_out": False})
        ledger = server.Ledger(tmp_path / "ledger")
        ledger.seq = seq
        monkeypatch.setattr(server, "MAX_EXECUTIONS", cap)
        server.Handler.ledger = ledger
        server.Handler.local = {"127.0.0.2"}
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd, ledger

    def _post(self, httpd: Any, body: dict[str, Any] | None = None) -> int:
        import http.client

        data = json.dumps(body if body is not None else {"protocol": 1}).encode()
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
        conn.request("POST", "/evaluate", body=data, headers={"Content-Type": "application/json"})
        return int(conn.getresponse().status)

    def test_local_peers_are_refused(self, tmp_path: Path, monkeypatch) -> None:
        httpd, _ = self._serve(tmp_path, monkeypatch, cap=0, seq=0)
        try:
            assert self._post(httpd) == 403  # 127.0.0.1 is always local
        finally:
            httpd.shutdown()
        rejections = (tmp_path / "ledger" / "rejected_requests.jsonl").read_text()
        assert "inside the evaluator" in rejections

    def test_spent_budget_is_refused_with_429(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(server, "is_local_peer", lambda peer, local: False)
        httpd, ledger = self._serve(tmp_path, monkeypatch, cap=3, seq=2)
        make = TestEvaluatorService()._body
        try:
            assert self._post(httpd, make({"solve.py": b"# a"})) == 200
            assert ledger.seq == 3
            assert self._post(httpd, make({"solve.py": b"# b"})) == 429
        finally:
            httpd.shutdown()
        rejections = (tmp_path / "ledger" / "rejected_requests.jsonl").read_text()
        assert "budget of 3 scoring executions is spent" in rejections
        assert ledger.seq == 3

    def test_cap_zero_is_unlimited(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(server, "is_local_peer", lambda peer, local: False)
        httpd, _ = self._serve(tmp_path, monkeypatch, cap=0, seq=10_000)
        try:
            body = TestEvaluatorService()._body({"solve.py": b"# a"})
            assert self._post(httpd, body) == 200
        finally:
            httpd.shutdown()


class TestCandidateStateScrub:
    """Nothing a candidate writes outside its working directory survives."""

    def test_writable_directory_scan_finds_world_writable_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "shm").mkdir()
        (tmp_path / "shm").chmod(0o1777)
        (tmp_path / "private").mkdir()
        (tmp_path / "private").chmod(0o755)
        (tmp_path / "excluded").mkdir()
        (tmp_path / "excluded").chmod(0o777)
        found = server.candidate_writable_dirs(
            99999, 99999, exclude=[tmp_path / "excluded"], root=str(tmp_path)
        )
        assert str(tmp_path / "shm") in found
        assert str(tmp_path / "private") not in found
        assert str(tmp_path / "excluded") not in found

    def test_removes_owned_entries_including_very_deep_trees(self, tmp_path: Path) -> None:
        scratch = tmp_path / "tmp"
        scratch.mkdir()
        (scratch / "warm_start.npy").write_bytes(b"state")
        deep = scratch / "deep"
        deep.mkdir()
        # Deeper than PATH_MAX and than Python's recursion limit.
        fd = os.open(deep, os.O_RDONLY)
        for _ in range(1500):
            os.mkdir("d", dir_fd=fd)
            child = os.open("d", os.O_RDONLY, dir_fd=fd)
            os.close(fd)
            fd = child
        os.close(os.open("leaf", os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=fd))
        os.close(fd)
        (scratch / "link").symlink_to(tmp_path / "outside")
        (tmp_path / "outside").mkdir()
        (tmp_path / "outside" / "keep.txt").write_text("keep")
        removed, remaining = server.remove_owned_entries([str(scratch)], os.getuid())
        assert remaining == []
        assert sorted(Path(p).name for p in removed) == ["deep", "link", "warm_start.npy"]
        assert list(scratch.iterdir()) == []
        assert (tmp_path / "outside" / "keep.txt").read_text() == "keep"

    def test_entries_of_other_owners_are_left_alone(self, tmp_path: Path) -> None:
        (tmp_path / "mine.txt").write_text("x")
        removed, remaining = server.remove_owned_entries([str(tmp_path)], os.getuid() + 12345)
        assert removed == [] and remaining == [] and (tmp_path / "mine.txt").exists()


# ---------------------------------------------------------------------------
# Evaluator client
# ---------------------------------------------------------------------------


class TestEvaluatorClient:
    def test_archive_is_deterministic_and_skips_private_paths(self, tmp_path: Path) -> None:
        (tmp_path / "solve.py").write_text("x = 1\n")
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "util.py").write_text("y = 2\n")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "HEAD").write_text("ref")
        (tmp_path / ".helix_artifacts").mkdir()
        (tmp_path / ".helix_artifacts" / "t.jsonl").write_text("{}")
        (tmp_path / "helix_batch.json").write_text('["0"]')
        (tmp_path / ".agent_task_prompt.md").write_text("prompt")
        (tmp_path / "link.py").symlink_to("/etc/passwd")
        first, digest1 = client.build_archive(tmp_path)
        second, digest2 = client.build_archive(tmp_path)
        assert first == second and digest1 == digest2
        with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as tar:
            assert sorted(tar.getnames()) == ["lib/util.py", "solve.py"]
        extracted = server.safe_extract(first, tmp_path / "out")
        assert {f["path"] for f in extracted} == {"lib/util.py", "solve.py"}


# ---------------------------------------------------------------------------
# Egress gateway policy
# ---------------------------------------------------------------------------


MODELS = frozenset({"claude-sonnet-5"})


def _msg(**extra: Any) -> bytes:
    body = {"model": "claude-sonnet-5", "max_tokens": 10, "messages": []}
    body.update(extra)
    return json.dumps(body).encode()


class TestGatewayPolicy:
    def test_allows_messages_with_client_tools(self) -> None:
        body = _msg(
            tools=[{"name": "Bash", "input_schema": {}}, {"type": "custom", "name": "Read"}],
            output_config={"effort": "xhigh"},
            thinking={"type": "adaptive"},
        )
        ok, reason, meta = gateway.check_request("POST", "/v1/messages?beta=true", body, MODELS)
        assert ok, reason
        assert meta["effort"] == "xhigh" and meta["thinking"] == "adaptive"
        assert meta["tool_names"] == ["Bash", "Read"]

    def test_allows_count_tokens(self) -> None:
        ok, _, _ = gateway.check_request("POST", "/v1/messages/count_tokens", _msg(), MODELS)
        assert ok

    @pytest.mark.parametrize(
        "method,path",
        [
            ("GET", "/v1/models"),
            ("POST", "/v1/complete"),
            ("POST", "/v1/files"),
            ("CONNECT", "example.com:443"),
            ("GET", "http://example.com/"),
            ("POST", "http://example.com/v1/messages"),
            ("GET", "/v1/messages"),
        ],
    )
    def test_denies_other_routes(self, method: str, path: str) -> None:
        ok, _, _ = gateway.check_request(method, path, _msg(), MODELS)
        assert not ok

    @pytest.mark.parametrize(
        "body",
        [
            _msg(model="claude-opus-5"),
            _msg(tools=[{"type": "web_search_20260209", "name": "web_search"}]),
            _msg(tools=[{"type": "web_fetch_20260209", "name": "web_fetch"}]),
            _msg(tools=[{"type": "code_execution_20260521", "name": "code_execution"}]),
            _msg(tools=[{"type": "mcp_toolset", "mcp_server_name": "x"}]),
            _msg(tools=[{"name": "WebSearch", "input_schema": {}}]),
            _msg(tools=[{"name": "WebFetch", "input_schema": {}}]),
            _msg(mcp_servers=[{"type": "url", "url": "https://example.com", "name": "x"}]),
            _msg(container={"skills": []}),
            _msg(tools="web_search"),
            b"not json",
            b"[]",
        ],
    )
    def test_denies_web_capable_or_malformed_requests(self, body: bytes) -> None:
        ok, _, _ = gateway.check_request("POST", "/v1/messages", body, MODELS)
        assert not ok

    def test_usage_parser_reads_sse_usage(self) -> None:
        parser = gateway.UsageParser(event_stream=True)
        events = [
            {"type": "message_start", "message": {"model": "claude-sonnet-5",
             "usage": {"input_tokens": 12, "cache_read_input_tokens": 100}}},
            {"type": "content_block_delta", "delta": {"text": "hi"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
             "usage": {"output_tokens": 7}},
        ]
        stream = "".join(f"event: x\ndata: {json.dumps(e)}\n\n" for e in events).encode()
        for i in range(0, len(stream), 13):  # arbitrary chunk boundaries
            parser.feed(stream[i : i + 13])
        parser.finish()
        assert parser.usage == {
            "input_tokens": 12, "cache_read_input_tokens": 100, "output_tokens": 7
        }
        assert parser.response_model == "claude-sonnet-5"
        assert parser.stop_reason == "end_turn"

    def test_usage_parser_flattens_cache_creation(self) -> None:
        parser = gateway.UsageParser(event_stream=False)
        parser.feed(json.dumps({"model": "claude-sonnet-5", "usage": {
            "input_tokens": 3, "cache_creation_input_tokens": 50,
            "cache_creation": {"ephemeral_5m_input_tokens": 20, "ephemeral_1h_input_tokens": 30},
        }}).encode())
        parser.finish()
        assert parser.usage["cache_creation.ephemeral_1h_input_tokens"] == 30
        assert parser.usage["cache_creation_input_tokens"] == 50

    def test_records_top_level_body_keys(self) -> None:
        ok, _, meta = gateway.check_request("POST", "/v1/messages", _msg(system="s"), MODELS)
        assert ok and meta["body_keys"] == ["max_tokens", "messages", "model", "system"]


def _user(*blocks: Any) -> dict[str, Any]:
    return {"role": "user", "content": list(blocks)}


PDF_URL = "https://arxiv.org/pdf/2605.19633"
IMAGE_URL = "http://hydra.nat.uni-magdeburg.de/packing/csq/csq26.png"
B64_PNG = {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}


def _tool_result(*content: Any) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": "toolu_1", "content": list(content)}


class TestGatewayContentPolicy:
    """Anthropic's servers must never fetch anything on the agent's behalf."""

    @pytest.mark.parametrize(
        "messages,where",
        [
            ([_user({"type": "document", "source": {"type": "url", "url": PDF_URL}})],
             "messages[0].content[0].source.type='url'"),
            ([_user({"type": "image", "source": {"type": "url", "url": IMAGE_URL}})],
             "messages[0].content[0].source.type='url'"),
            ([_user({"type": "text", "text": "hi"},
                    _tool_result({"type": "document", "source": {"type": "url", "url": PDF_URL}}))],
             "messages[0].content[1].content[0].source.type='url'"),
            ([_user(_tool_result({"type": "image", "source": {"type": "url", "url": IMAGE_URL}}))],
             "messages[0].content[0].content[0].source.type='url'"),
            ([_user({"type": "document", "source": {"type": "file", "file_id": "file_1"}})],
             "source.type='file'"),
            ([_user({"type": "image", "source": {"type": "file", "file_id": "file_1"}})],
             "source.type='file'"),
            ([_user({"type": "document", "source": {"type": "content", "content": [
                {"type": "image", "source": {"type": "url", "url": IMAGE_URL}}]}})],
             "source.content[0].source.type='url'"),
            ([_user({"type": "container_upload", "file_id": "file_1"})], "type='container_upload'"),
            ([{"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_fetch",
                 "input": {"url": PDF_URL}}]}], "type='server_tool_use'"),
            ([_user({"type": "web_fetch_tool_result", "tool_use_id": "srvtoolu_1", "content": {}})],
             "type='web_fetch_tool_result'"),
            ([_user({"type": "mcp_tool_use", "id": "x", "name": "fetch", "server_name": "s", "input": {}})],
             "type='mcp_tool_use'"),
            ([{"role": "system", "content": [{"type": "document", "source": {"type": "url", "url": PDF_URL}}]}],
             "type='document'"),
            ([{"role": "tool", "content": "x"}], "role='tool'"),
            ([_user({"source": {"type": "url", "url": PDF_URL}})], "type=None"),
            ([_user("not a block")], "is not an object"),
        ],
    )
    @pytest.mark.parametrize("path", ["/v1/messages?beta=true", "/v1/messages/count_tokens?beta=true"])
    def test_refuses_remote_content(self, messages: list[Any], where: str, path: str) -> None:
        ok, reason, meta = gateway.check_request("POST", path, _msg(messages=messages), MODELS)
        assert not ok
        assert reason.startswith("content not allowed") and where in reason, reason
        if "source.type" in where:
            # The refused block's source type is logged for auditing.
            assert any(k.endswith(":url") or k.endswith(":file") for k in meta["source_types"])

    def test_refuses_remote_content_in_system(self) -> None:
        body = _msg(system=[{"type": "document", "source": {"type": "url", "url": PDF_URL}}])
        ok, reason, _ = gateway.check_request("POST", "/v1/messages", body, MODELS)
        assert not ok and "system[0].type='document'" in reason

    def test_allows_inline_content_claude_code_sends(self) -> None:
        messages = [
            {"role": "user", "content": "Read .agent_task_prompt.md"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "...", "signature": "sig"},
                {"type": "redacted_thinking", "data": "abc"},
                {"type": "text", "text": "Reading."},
                {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "plot.png"}},
            ]},
            _user(
                _tool_result({"type": "text", "text": "image follows"},
                             {"type": "image", "source": B64_PNG},
                             {"type": "tool_reference", "tool_name": "Bash"}),
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="}},
                {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "notes"}},
                {"type": "document", "source": {"type": "content", "content": [
                    {"type": "text", "text": "a"}, {"type": "image", "source": B64_PNG}]}},
                {"type": "search_result", "source": "local notes", "title": "t",
                 "content": [{"type": "text", "text": "x"}]},
                {"type": "text", "text": "continue", "cache_control": {"type": "ephemeral"}},
            ),
            {"role": "system", "content": "mid-conversation reminder"},
            {"role": "assistant", "content": [{"type": "compaction", "content": "summary"}]},
        ]
        body = _msg(messages=messages, system=[{"type": "text", "text": "You are Claude Code."}],
                    context_management={"edits": [{"type": "clear_thinking_20251015"}]})
        ok, reason, meta = gateway.check_request("POST", "/v1/messages?beta=true", body, MODELS)
        assert ok, reason
        assert meta["block_types"]["image"] == 2
        assert meta["source_types"] == {"image:base64": 2, "document:base64": 1,
                                        "document:text": 1, "document:content": 1}
        assert meta["context_edits"] == ["clear_thinking_20251015"]

    @pytest.mark.parametrize(
        "header",
        ["files-api-2025-04-14", "mcp-client-2025-11-20", "skills-2025-10-02",
         "code-execution-2025-08-25", "web-fetch-2025-09-10",
         "claude-code-20250219,web-search-2025-03-05"],
    )
    def test_refuses_remote_content_betas(self, header: str) -> None:
        ok, reason, meta = gateway.check_request("POST", "/v1/messages", _msg(), MODELS, beta_header=header)
        assert not ok and reason.startswith("anthropic-beta not allowed")
        assert meta["model"] == "claude-sonnet-5"

    def test_allows_the_betas_claude_code_sends(self) -> None:
        header = ("claude-code-20250219,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,"
                  "context-management-2025-06-27,prompt-caching-scope-2026-01-05,"
                  "mid-conversation-system-2026-04-07,effort-2025-11-24")
        ok, reason, _ = gateway.check_request("POST", "/v1/messages", _msg(), MODELS, beta_header=header)
        assert ok, reason

    def test_joins_repeated_beta_headers(self) -> None:
        assert gateway.joined_header(["claude-code-20250219", "files-api-2025-04-14"]) == (
            "claude-code-20250219,files-api-2025-04-14"
        )
        assert gateway.joined_header(None) is None

    @pytest.mark.parametrize("key", ["fallbacks", "fallback_credit_token", "user_profile_id", "betas"])
    def test_refuses_unreviewed_top_level_parameters(self, key: str) -> None:
        ok, reason, _ = gateway.check_request("POST", "/v1/messages", _msg(**{key: []}), MODELS)
        assert not ok and key in reason

    def test_deeply_nested_content_is_refused_not_crashing(self) -> None:
        block: dict[str, Any] = {"type": "text", "text": "x"}
        for _ in range(3000):
            block = {"type": "tool_result", "tool_use_id": "t", "content": [block]}
        body = json.dumps({"model": "claude-sonnet-5", "max_tokens": 1,
                           "messages": [{"role": "user", "content": [block]}]}).encode()
        ok, _, _ = gateway.check_request("POST", "/v1/messages", body, MODELS)
        assert not ok

    def test_repeated_beta_header_reaches_the_policy(self, tmp_path: Path, monkeypatch) -> None:
        import http.client
        import threading
        from http.server import ThreadingHTTPServer

        monkeypatch.setenv("GATEWAY_CAPTURE_ONLY", "1")
        monkeypatch.setenv("GATEWAY_ALLOWED_MODELS", "claude-sonnet-5")
        monkeypatch.setattr(gateway, "LOG_PATH", tmp_path / "requests.jsonl")
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), gateway.Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
            body = _msg()
            conn.putrequest("POST", "/v1/messages")
            conn.putheader("content-type", "application/json")
            conn.putheader("content-length", str(len(body)))
            conn.putheader("anthropic-beta", "claude-code-20250219")
            conn.putheader("anthropic-beta", "files-api-2025-04-14")
            conn.endheaders(body)
            status = conn.getresponse().status
        finally:
            httpd.shutdown()
        assert status == 403
        [record] = [json.loads(t) for t in (tmp_path / "requests.jsonl").read_text().splitlines()]
        assert record["reason"] == "anthropic-beta not allowed: files-api-2025-04-14"


# ---------------------------------------------------------------------------
# Leak checks on what the agent can read
# ---------------------------------------------------------------------------

FORBIDDEN = [
    "2.63", "2.635", "2.6358", "2.636", "2.64", "packomania", "alphaevolve",
    "friedman", "optimize_anything", "optimize anything", "gepa", "shinka",
    "openevolve", "known_best", "record", "best known", "best-known",
    "state of the art", "http://", "https://", "www.", "slsqp", "l-bfgs",
    "lbfgs", "hexagonal", "lattice", "cma-es", "bilevel", "linear program",
]

PLACEHOLDERS = {
    "@LEDGER_DIR@": "/tmp/ledger",
    "@IMAGE_TAG@": "test-revision",
    "@EVAL_CPUS@": "14",
    "@MAX_EVALUATIONS@": "200",
    "@MAX_EXECUTIONS@": "200",
    "@MAX_GENERATIONS@": "1000",
    "@AGENT_CPUS@": "2",
    "@MODEL@": "claude-sonnet-5",
    "@EFFORT@": "xhigh",
    "@MAX_TURNS@": "40",
}


def _rendered_template(tmp_path: Path | None = None) -> str:
    text = (CLEAN / "helix.toml.template").read_text()
    for key, value in PLACEHOLDERS.items():
        text = text.replace(key, value)
    assert "@" not in "".join(line.split("#")[0] for line in text.splitlines())
    return text


def _background() -> str:
    import tomllib

    data = tomllib.loads(_rendered_template())
    return data["objective"] + "\n" + data["agent"]["background"]


def test_template_loads_with_helix_and_matches_the_protocol(tmp_path: Path) -> None:
    from helix.config import load_config

    path = tmp_path / "helix.toml"
    path.write_text(_rendered_template())
    config = load_config(path)
    assert config.dataset.train_size == 1 and config.dataset.val_size == 1
    assert config.evolution.minibatch_size == 1
    assert config.evolution.max_evaluations == 0
    assert config.evolution.perfect_score_threshold is None
    assert config.evolution.num_parallel_proposals == 1
    assert config.evolution.mutations_per_parent == 1
    assert config.evolution.merge_enabled is False
    assert config.agent.backend == "claude" and config.agent.effort == "xhigh"
    assert config.agent.disallowed_tools == ["WebSearch", "WebFetch"]
    assert config.sandbox.enabled and config.sandbox.evaluator
    assert config.sandbox.network == "helix-circle-clean-agent"
    assert config.sandbox.auth_volume == "helix-circle-clean-claude-home"
    assert config.evaluator.sidecar is not None
    assert config.evaluator.sidecar.internal_network is True
    assert "CP_MAX_EXECUTIONS=200" in config.evaluator.sidecar.command
    # Nothing agent-visible carries the budget or credentials.
    assert all("CP_MAX" not in value for value in config.env.values())
    assert config.env["CLAUDE_CODE_OAUTH_TOKEN"].startswith("placeholder")
    assert "rules" not in config.agent.background.lower() or "1e-6" in config.agent.background


@pytest.mark.parametrize(
    "name,text",
    [
        ("solve.py", (EXAMPLE / "solve.py").read_text()),
        ("check_packing.py", (EXAMPLE / "check_packing.py").read_text()),
        ("objective+background", _background()),
    ],
)
def test_agent_visible_text_has_no_targets_or_pointers(name: str, text: str) -> None:
    lowered = text.lower()
    found = [token for token in FORBIDDEN if token in lowered]
    assert found == [], f"{name} mentions {found}"


def test_seed_is_valid_under_the_strict_rules() -> None:
    numpy = pytest.importorskip("numpy")
    del numpy
    sys.modules.pop("solve", None)
    seed = _load("solve", EXAMPLE / "solve.py")
    try:
        packing = [list(row) for row in seed.pack_circles(26)]
    finally:
        sys.modules.pop("solve", None)
    verdict = rules.evaluate_packing(packing)
    assert verdict["valid"] is True
    assert round(verdict["score"], 6) == 0.979764
    independent = exact.check(packing, claimed_score=verdict["score"])
    assert independent["valid_at_tolerance"] is True
    assert independent["claimed_score_within_certified_sum"] is True
    # The greedy radius shrink leaves floating-point contacts overlapping by
    # ~1e-17, which is exactly why the rules carry a tolerance.
    assert independent["max_pair_overlap_approx"] < 1e-15


# ---------------------------------------------------------------------------
# The local (unsandboxed) example evaluator applies the same scoring rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body,expected",
    [
        (f"return {grid_packing()!r}", 26 / 16),
        (
            f"return {with_row(grid_packing(), 0, [R - 5e-7, R, R])!r}",
            rules.evaluate_packing(with_row(grid_packing(), 0, [R - 5e-7, R, R]))["score"],
        ),
        (f"return {with_row(grid_packing(), 1, [3 * R - 2e-6, R, R])!r}", 0.0),  # overlap
        ("return [(0.5, 0.5, 0.5)] * n", 0.0),  # overlapping
        (f"return {grid_packing()[:25]!r}", 0.0),  # wrong count
        (f"return {with_row(grid_packing(), 0, [2.0, 2.0, 0.01])!r}", 0.0),  # outside
        ("return [(float('nan'), 0.5, 0.1)] * n", 0.0),
        ("return [(0.5, 0.5, -0.1)] * n", 0.0),
        ("return [(True, 0.5, 0.0)] * n", 0.0),
        ("return [('0.5', 0.5, 0.0)] * n", 0.0),
        ("raise RuntimeError('boom')", 0.0),
    ],
)
def test_local_example_evaluator_scores_invalid_packings_zero(
    tmp_path: Path, body: str, expected: float
) -> None:
    import subprocess

    (tmp_path / "solve.py").write_text(f"def pack_circles(n):\n    {body}\n")
    (tmp_path / "evaluate.py").write_text((EXAMPLE / "evaluate.py").read_text())
    (tmp_path / "helix_batch.json").write_text('["0"]')
    out = subprocess.run(
        [sys.executable, "evaluate.py"], cwd=tmp_path, capture_output=True, text=True,
        check=True,
    ).stdout
    line = next(text for text in out.splitlines() if text.startswith("HELIX_RESULT="))
    [[score, side_info]] = json.loads(line[len("HELIX_RESULT="):])
    assert score == expected
    assert side_info["scores"]["sum_radii"] == expected


# ---------------------------------------------------------------------------
# Summary accounting helpers
# ---------------------------------------------------------------------------


def test_best_record_prefers_higher_valid_score_then_earlier_execution() -> None:
    records = [
        {"seq": 1, "valid": True, "score": 1.0},
        {"seq": 2, "valid": False, "score": 0.0},
        {"seq": 3, "valid": True, "score": 2.0},
        {"seq": 4, "valid": True, "score": 2.0},
    ]
    assert summarize.best_record(records)["seq"] == 3
    assert summarize.best_record([records[1]]) is None


def test_subscription_usage_counts_cache_writes_and_reads() -> None:
    requests = [
        {"event": "start"},
        {"decision": "deny", "reason": "server tool not allowed"},
        {"decision": "allow", "status": 200, "model": "claude-sonnet-5",
         "response_model": "claude-sonnet-5", "effort": "xhigh",
         "usage": {"input_tokens": 1_000_000, "output_tokens": 100_000,
                   "cache_creation_input_tokens": 400_000,
                   "cache_creation.ephemeral_1h_input_tokens": 100_000,
                   "cache_read_input_tokens": 2_000_000}},
    ]
    out = summarize.usage_cost(requests)
    assert out["tokens_by_model"]["claude-sonnet-5"]["input_tokens"] == 1_000_000
    assert out["tokens_by_model"]["claude-sonnet-5"]["cache_creation_input_tokens"] == 400_000
    assert out["subscription_usage"]["dollars_charged"] is None
    assert out["denied_requests_by_reason"] == {"server tool not allowed": 1}
    assert out["requests_by_model_and_effort"] == {"claude-sonnet-5:xhigh": 1}


def test_gateway_capture_mode_records_and_refuses_without_upstream(tmp_path: Path, monkeypatch) -> None:
    import http.client
    import threading
    from http.server import ThreadingHTTPServer

    monkeypatch.setenv("GATEWAY_CAPTURE_ONLY", "1")
    monkeypatch.setenv("GATEWAY_ALLOWED_MODELS", "claude-sonnet-5")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(gateway, "LOG_PATH", tmp_path / "requests.jsonl")

    def refuse_upstream(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("capture mode must not contact the upstream")

    monkeypatch.setattr(gateway.http.client, "HTTPSConnection", refuse_upstream)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), gateway.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    body = _msg(output_config={"effort": "xhigh"}, tools=[{"name": "Bash", "input_schema": {}}])
    try:
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
        conn.request("POST", "/v1/messages?beta=true", body=body, headers={
            "content-type": "application/json", "anthropic-beta": "x-2026-01-01",
            "x-api-key": "placeholder"})
        status = conn.getresponse().status
    finally:
        httpd.shutdown()
    assert status == 403
    [record] = [json.loads(text) for text in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert record["decision"] == "deny" and record["reason"].startswith("capture only")
    assert record["model"] == "claude-sonnet-5" and record["effort"] == "xhigh"
    assert record["tool_names"] == ["Bash"] and record["anthropic_beta"] == "x-2026-01-01"
    assert "placeholder" not in json.dumps(record)


# ---------------------------------------------------------------------------
# Credential watchdog and gateway error capture
# ---------------------------------------------------------------------------

watchdog = _load("cp_watchdog", CLEAN / "scripts" / "watchdog.py")


def _fwd(status: int, error_type: str | None = None, message: str | None = None) -> dict[str, Any]:
    return {"decision": "allow", "status": status, "error_type": error_type, "error_message": message}


def test_watchdog_stops_after_consecutive_credential_failures() -> None:
    billing = _fwd(400, "invalid_request_error", "Your credit balance is too low to access the Anthropic API.")
    assert watchdog.should_stop([_fwd(200), billing, billing, billing]) is True
    assert watchdog.should_stop([billing, billing, _fwd(200)]) is False
    assert watchdog.should_stop([_fwd(401, "authentication_error")] * 3) is True
    # Denied-by-policy requests never count; transient errors never stop the run.
    denied = {"decision": "deny", "status": 403}
    assert watchdog.should_stop([billing, denied, billing, denied, billing]) is True
    assert watchdog.should_stop([_fwd(429, "rate_limit_error")] * 5) is False
    assert watchdog.should_stop([_fwd(529, "overloaded_error")] * 5) is False
    assert watchdog.should_stop([_fwd(400, "invalid_request_error", "prompt is too long")] * 5) is False


def test_gateway_records_upstream_error_type() -> None:
    parser = gateway.UsageParser(event_stream=False)
    parser.feed(json.dumps({"type": "error", "error": {
        "type": "invalid_request_error", "message": "Your credit balance is too low"}}).encode())
    parser.finish()
    assert parser.error_type == "invalid_request_error"
    assert parser.error_message == "Your credit balance is too low"


def test_subscription_refresh_stays_in_auth_volume_and_never_logs_tokens(tmp_path, monkeypatch):
    from io import BytesIO
    path = tmp_path / '.credentials.json'
    path.write_text(json.dumps({'claudeAiOauth': {
        'accessToken': 'test-expired', 'refreshToken': 'test-refresh',
        'expiresAt': 0, 'scopes': ['user:inference'], 'subscriptionType': 'max'}}))
    monkeypatch.setattr(gateway, 'CREDENTIAL_PATH', path)
    monkeypatch.setattr(gateway, 'LOG_PATH', tmp_path / 'events.jsonl')
    calls = []
    def renew(request, **kwargs):
        calls.append(request)
        assert request.full_url == 'https://platform.claude.com/v1/oauth/token'
        assert json.loads(request.data)['refresh_token'] == 'test-refresh'
        return BytesIO(json.dumps({'access_token': 'test-new-access',
            'refresh_token': 'test-new-refresh', 'expires_in': 3600}).encode())
    monkeypatch.setattr(gateway.urllib.request, 'urlopen', renew)
    assert gateway.subscription_token() == 'test-new-access'
    assert gateway.subscription_token() == 'test-new-access'
    assert len(calls) == 1
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())['claudeAiOauth']['subscriptionType'] == 'max'
    assert 'test-' not in (tmp_path / 'events.jsonl').read_text()


def test_subscription_refresh_error_never_exposes_response_or_token(tmp_path, monkeypatch):
    path = tmp_path / '.credentials.json'
    path.write_text(json.dumps({'claudeAiOauth': {
        'accessToken': 'test-expired', 'refreshToken': 'test-refresh',
        'expiresAt': 0, 'scopes': ['user:inference']}}))
    monkeypatch.setattr(gateway, 'CREDENTIAL_PATH', path)
    def fail(*args, **kwargs):
        raise ValueError('test-refresh secret echoed by upstream')
    monkeypatch.setattr(gateway.urllib.request, 'urlopen', fail)
    with pytest.raises(gateway.CredentialError) as caught:
        gateway.subscription_token()
    assert 'test-refresh' not in str(caught.value)
    assert caught.value.__suppress_context__


def test_gateway_forwards_subscription_auth_without_logging_it(tmp_path, monkeypatch):
    import http.client
    import threading
    from http.server import ThreadingHTTPServer
    monkeypatch.delenv('GATEWAY_CAPTURE_ONLY', raising=False)
    monkeypatch.setenv('GATEWAY_ALLOWED_MODELS', 'claude-sonnet-5')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'unused-test-api-key')
    monkeypatch.setattr(gateway, 'LOG_PATH', tmp_path / 'requests.jsonl')
    monkeypatch.setattr(gateway, 'subscription_token', lambda: 'test-private-subscription-token')
    sent = {}
    class Response:
        status = 200
        def getheader(self, name, default=None):
            return 'application/json' if name == 'Content-Type' else default
        def getheaders(self):
            return [('Content-Type', 'application/json'), ('anthropic-ratelimit-unified-status', 'allowed')]
        def read1(self, size):
            return b''
    class Upstream:
        def __init__(self, host, *args, **kwargs):
            assert host == 'api.anthropic.com'
        def request(self, method, path, body, headers):
            sent.update(headers)
        def getresponse(self):
            return Response()
        def close(self):
            pass
    monkeypatch.setattr(gateway.http.client, 'HTTPSConnection', Upstream)
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), gateway.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection('127.0.0.1', httpd.server_address[1], timeout=10)
        conn.request('POST', '/v1/messages', body=_msg(), headers={
            'Authorization': 'Bearer placeholder', 'X-Api-Key': 'client-must-not-forward'})
        response = conn.getresponse()
        assert response.status == 200
        response.read()
    finally:
        httpd.shutdown()
    assert sent['authorization'] == 'Bearer test-private-subscription-token'
    assert not any(k.lower() == 'x-api-key' for k in sent)
    assert 'oauth-2025-04-20' in sent['anthropic-beta']
    text = (tmp_path / 'requests.jsonl').read_text()
    assert 'test-private-subscription-token' not in text
    assert 'unused-test-api-key' not in text
    assert json.loads(text)['auth_mode'] == 'claude_subscription'


def test_summary_reports_unreached_checkpoints_and_every_evaluation():
    records = [{'seq': i, 'score': i / 100, 'valid': i != 3,
                'files_sha256': str(i)} for i in range(1, 134)]
    points = summarize.checkpoints(records, (63, 133, 200))
    assert points['63']['best_score'] == 0.63
    assert points['133']['best_score'] == 1.33
    assert points['200']['best_score'] is None and not points['200']['reached']
    curve = summarize.execution_curve(records)
    assert len(curve) == 133 and curve[2]['best_score'] == 0.02
    assert summarize.distinct_tree_curve(records)['best_score_by_distinct_trees']['200'] is None


def test_summary_detects_unfinished_missing_and_duplicate_executions():
    budget = summarize.budget_accounting(
        [{'seq': 1}, {'seq': 1}, {'seq': 4}], [{'seq': 1}, {'seq': 2}, {'seq': 4}],
        [], [], 200, 0, False)
    assert budget['unfinished_executions'] == [2]
    assert budget['missing_execution_numbers'] == [3]
    assert budget['duplicate_execution_records'] == [1]
    assert not budget['ledger_consistent'] and budget['run_incomplete']


def test_summary_attributes_removed_candidate_from_ledger():
    roles = summarize.execution_roles([
        {'seq': 1, 'kind': 'execution', 'split': 'val', 'candidate_id': 'g0-s0'},
        {'seq': 1, 'kind': 'cache_hit', 'split': 'train', 'candidate_id': 'g0-s0'},
        {'seq': 2, 'kind': 'execution', 'split': 'train', 'candidate_id': 'g12-s13', 'phase': 'gate'}])
    assert roles[2] == {'candidate_id': 'g12-s13', 'generation': 12,
                        'role': 'gate evaluation of a new child', 'phase': 'gate'}


def test_watchdog_detects_budget_refusal_lost_calls_and_abort():
    assert watchdog.evaluator_stop_reason([{'status': 429}], [])[0] == 'budget_spent'
    assert watchdog.evaluator_stop_reason([{'status': 409}], [])[0] == 'evaluator_refusal'
    assert watchdog.evaluator_stop_reason([], [{'event': 'evaluation_aborted'}])[0] == 'evaluator_aborted'
    assert watchdog.diverged(watchdog.counter_gap(10, 14), 0)
    assert not watchdog.diverged(watchdog.counter_gap(10, 12), 0)


def test_compute_summary_measures_bash_python_and_coordinate_literals(tmp_path):
    path = tmp_path / 'session.jsonl'
    path.write_text('\n'.join(json.dumps(e) for e in [
        {'timestamp': '2026-09-17T00:00:00Z', 'message': {'content': [
            {'type': 'tool_use', 'id': 'x', 'name': 'Bash', 'input': {'command': 'python3 solve.py'}}]}},
        {'timestamp': '2026-09-17T00:00:03Z', 'message': {'content': [
            {'type': 'tool_result', 'tool_use_id': 'x', 'content': 'ok'}]}}
    ]))
    bash = summarize.transcript_tool_use([path])['bash']
    assert bash['invocations'] == bash['running_python'] == 1
    assert bash['python_seconds_total'] == 3
    assert summarize.hardcoded_evidence({'solve.py': 'x=0.123456789'}, [[0.123456789]*3]*26)['appears_hard_coded']


def test_service_enforces_exact_200_execution_cap_with_cache_hits(tmp_path, monkeypatch):
    case = TestEvaluatorService()
    monkeypatch.setattr(server, 'MAX_EXECUTIONS', 200)
    body = case._body({'solve.py': b'# 0'})
    _, ledger = case._evaluate(tmp_path, monkeypatch, body, case._ok())
    for index in range(1, 200):
        # Parent/validation cache hits must not reduce the new-tree budget.
        assert server.evaluate_request(body, ledger, 'runner')['cached_from_seq'] == index
        body = case._body({'solve.py': f'# {index}'.encode()})
        assert server.evaluate_request(body, ledger, 'runner')['results'][0][0] == 26 / 16
    with pytest.raises(server.BudgetExhausted):
        server.evaluate_request(case._body({'solve.py': b'# 200'}), ledger, 'runner')
    assert server.evaluate_request({**body, 'split': 'val'}, ledger, 'runner')['cached_from_seq'] == 200
    assert ledger.seq == ledger.executions == 200
    assert len(ledger.starts.read_text().splitlines()) == 200
    assert len(ledger.requests.read_text().splitlines()) == 400


def test_invalid_summary_returns_nonzero_and_labels_headline(tmp_path, monkeypatch):
    private = tmp_path / '.helix-run'
    (private / 'ledger').mkdir(parents=True)
    (private / 'protocol.env').write_text(f"RUN_DIR='{tmp_path}'\nMAX_EVALUATIONS='200'\n")
    (private / 'ledger/evaluations.jsonl').write_text(json.dumps({
        'seq': 1, 'valid': True, 'score': 3.0, 'candidate_id': 'g7-s8',
        'circles': grid_packing()}) + '\n')
    monkeypatch.setattr(summarize, 'PRIVATE', private)
    monkeypatch.setattr(sys, 'argv', ['summarize.py', '--no-rerun'])
    assert summarize.main() == 2
    result = json.loads((private / 'summary/summary.json').read_text())
    assert any(label.startswith('INVALID') for label in result['headline'])
    assert result['best_valid_within_budget']['generation'] == 7


def test_watchdog_requires_consecutive_subscription_refresh_failures():
    failure = {'decision': 'deny', 'status': 503,
               'reason': 'subscription login unavailable'}
    assert not watchdog.should_stop([failure])
    assert watchdog.should_stop([failure] * 3)
    assert not watchdog.should_stop([failure, _fwd(200), failure, failure])


def test_watchdog_ignores_historical_subscription_failure_after_recovery():
    # Full-run regression: stale preflight failures preceded successful inference.
    records = [{'decision': 'deny', 'status': 503,
                'reason': 'subscription login unavailable; run helix sandbox login claude'}] * 10
    records += [_fwd(200)] * 3
    assert not watchdog.should_stop(records)


def test_watchdog_success_and_unrelated_denials_are_not_auth_failures():
    assert not watchdog.should_stop([_fwd(200, 'authentication_error')] * 3)
    assert not watchdog.should_stop([{'decision': 'deny', 'status': 503,
                                      'reason': 'temporary gateway failure'}] * 3)


def test_summary_does_not_treat_a_smaller_budget_marker_as_full_completion():
    result = summarize.budget_accounting([{'seq': 1}], [{'seq': 1}], [], [], 200, 0, True)
    assert result['sidecar_budget_marker_present']
    assert not result['budget_exhausted'] and result['run_incomplete']


def test_watchdog_ignores_gateway_history_before_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gateway_log = tmp_path / "gateway.jsonl"
    gateway_log.write_text(json.dumps({"decision": "deny", "status": 503,
                                      "reason": "subscription login unavailable"}) + "\n")
    gateway_log.write_text(gateway_log.read_text() * 3)
    pid_file = tmp_path / "pid"
    pid_file.write_text("123")
    monkeypatch.setattr(sys, "argv", [
        "watchdog.py", "--pid-file", str(pid_file),
        "--stop-record", str(tmp_path / "stop.json"),
        "--gateway-log", str(gateway_log), "--ledger-dir", str(tmp_path),
        "--helix-state", str(tmp_path / "state.json"),
        "--helix-log", str(tmp_path / "helix.log"),
    ])
    alive = iter([True, False])
    monkeypatch.setattr(watchdog, "_alive", lambda pid: next(alive))
    monkeypatch.setattr(watchdog.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(watchdog, "stop", lambda *args: pytest.fail("stale failure stopped run"))
    assert watchdog.main() == 0
    assert not (tmp_path / "stop.json").exists()


def test_scoring_has_no_fixed_reference_ceiling() -> None:
    # Larger instances let us test scores above published n=26 values using
    # elementary, exactly feasible packings, without claiming a new n=26 record.
    for width in (8, 16):
        radius = 1.0 / (2 * width)
        packing = [[(2*x+1)*radius, (2*y+1)*radius, radius]
                   for x in range(width) for y in range(width)]
        verdict = rules.evaluate_packing(packing, n=width**2)
        assert verdict["valid"]
        assert verdict["score"] == math.fsum(row[2] for row in packing)
        assert verdict["score"] > 2.64


def test_reference_is_a_static_fixture_not_an_optimizer() -> None:
    reference = _load("cp_reference_fixture", EXAMPLE.parents[1] / "docs/experiments/circle-packing/reference_known_best.py")
    packing = reference.pack_circles(26)
    verdict = rules.evaluate_packing(packing)
    assert verdict["valid"]
    assert verdict["score"] > 2.6
    # The fixture is independent of search time, optimizer packages and seeds.
    assert not hasattr(reference, "TIME_BUDGET")
    assert not hasattr(reference, "minimize")
    packing[0] = (0, 0, 0)
    assert reference.pack_circles(26)[0] != packing[0]
    with pytest.raises(ValueError, match="26"):
        reference.pack_circles(25)


def test_setup_uses_selected_image_tag(tmp_path: Path) -> None:
    import subprocess
    from helix.config import load_config

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text('#!/bin/sh\nif [ "$1" = run ]; then echo 2; fi\n')
    docker.chmod(0o755)
    run_dir = tmp_path / "experiment"
    subprocess.run(["bash", str(CLEAN / "scripts/setup_run_dir.sh"), str(run_dir)],
                   env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                        "HELIX_CIRCLE_CLEAN_TAG": "test-revision", "EFFORT": ""},
                   check=True, capture_output=True, text=True)
    config = load_config(run_dir / ".helix-run/helix.toml")
    assert config.sandbox.image == "helix-circle-clean/agent-claude:test-revision"
    assert config.evaluator.sidecar.image == "helix-circle-clean/evaluator:test-revision"
    assert config.evolution.perfect_score_threshold is None


@pytest.mark.parametrize("has_caffeinate", [False, True])
def test_run_wrapper_starts_with_optional_caffeinate(tmp_path: Path, has_caffeinate: bool) -> None:
    import shutil
    import subprocess

    private = tmp_path / ".helix-run"
    (private / "bin").mkdir(parents=True)
    (private / "logs/gateway").mkdir(parents=True)
    wrapper = private / "bin/run_wrapper.sh"
    shutil.copyfile(CLEAN / "scripts/run_wrapper.sh", wrapper)
    for name in ("watchdog.py", "agent_monitor.py"):
        (private / "bin" / name).write_text("")
    (private / "protocol.env").write_text(
        f"RUN_DIR='{tmp_path}'\nHELIX_SRC='{tmp_path}'\nAGENT_NETWORK='test'\n"
        "GATEWAY_CONTAINER='test'\nARCHIVER_CONTAINER='test'\n"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("env", "date", "dirname", "rm"):
        (bin_dir / name).symlink_to(shutil.which(name))
    marker = tmp_path / "started"
    commands = {"git": "echo revision", "docker": "exit 0", "sleep": "exit 0",
                "uv": f"echo started > '{marker}'"}
    if has_caffeinate:
        commands["caffeinate"] = 'shift; exec "$@"'
    for name, body in commands.items():
        path = bin_dir / name
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    subprocess.run(["/bin/bash", str(wrapper), "evolve"],
                   env={"PATH": str(bin_dir), "HOME": str(tmp_path)},
                   check=True, capture_output=True, text=True, timeout=10)
    assert marker.read_text().strip() == "started"
