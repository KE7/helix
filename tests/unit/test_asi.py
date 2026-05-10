"""Unit tests for :mod:`helix.asi` — the evaluator-facing log channel.

The contract under test:

* ``helix.log(...)`` is a no-op when ``HELIX_ASI_LOG`` is unset
  (evaluator code can keep the same imports in local debug runs).
* When the env var points at a writable path, calls append one
  JSONL record per call.  The record + trailing newline land in a
  single ``write(2)`` so concurrent appends from a multi-worker
  evaluator cannot interleave a single record across two writes.
* Filesystem failures are swallowed: logging is best-effort and
  must never turn an evaluator observability call into a hard
  crash.
* :func:`helix.asi.read_text` renders the captured records with
  scalars formatted bare and nested values formatted as JSON
  literals (so the mutation prompt shows readable structured data
  instead of Python repr).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from helix import asi


# ---------------------------------------------------------------------------
# log(): no-op outside HELIX
# ---------------------------------------------------------------------------


class TestLogNoOpOutsideHelix:
    def test_no_env_var_no_raise(self, monkeypatch):
        """Without ``HELIX_ASI_LOG`` set, ``helix.log`` must silently
        succeed — evaluator scripts often run locally for debug and the
        same imports must keep working."""
        monkeypatch.delenv(asi.HELIX_ASI_LOG_ENV, raising=False)
        # No assertion — just must not raise.
        asi.log("hello", score=1.0)

    def test_empty_env_var_no_raise(self, monkeypatch):
        """An empty-string env var is treated identically to unset."""
        monkeypatch.setenv(asi.HELIX_ASI_LOG_ENV, "")
        asi.log("hello", score=1.0)

    def test_no_file_created_when_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv(asi.HELIX_ASI_LOG_ENV, raising=False)
        asi.log("hello")
        # Nothing should have been written anywhere — including the
        # default tmp_path location.
        assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# log(): happy path + atomicity
# ---------------------------------------------------------------------------


class TestLogWritesRecord:
    def test_writes_jsonl_record_with_message(self, tmp_path, monkeypatch):
        target = tmp_path / "asi.jsonl"
        monkeypatch.setenv(asi.HELIX_ASI_LOG_ENV, str(target))
        asi.log("hello world", score=0.5)
        lines = target.read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["message"] == "hello world"
        assert record["score"] == 0.5

    def test_appends_one_line_per_call(self, tmp_path, monkeypatch):
        target = tmp_path / "asi.jsonl"
        monkeypatch.setenv(asi.HELIX_ASI_LOG_ENV, str(target))
        asi.log("first")
        asi.log("second")
        asi.log("third")
        assert target.read_text().splitlines() == [
            json.dumps({"message": "first"}, sort_keys=True),
            json.dumps({"message": "second"}, sort_keys=True),
            json.dumps({"message": "third"}, sort_keys=True),
        ]

    def test_record_emits_in_single_write(self, tmp_path, monkeypatch):
        """Atomicity invariant: the record + ``"\\n"`` go out in a
        single ``write`` call so POSIX ``O_APPEND`` semantics keep a
        single record from being split by an interleaved append.

        We patch the file handle's ``write`` method and assert it was
        called exactly once for the body (the trailing newline must
        ride along, not be a second write).
        """
        target = tmp_path / "asi.jsonl"
        monkeypatch.setenv(asi.HELIX_ASI_LOG_ENV, str(target))

        original_open = Path.open
        write_calls: list[str] = []

        class _SpyHandle:
            def __init__(self, real_handle):
                self._real = real_handle

            def __enter__(self):
                self._real.__enter__()
                return self

            def __exit__(self, *exc):
                return self._real.__exit__(*exc)

            def write(self, data):
                write_calls.append(data)
                return self._real.write(data)

        def _spy_open(self, *args, **kwargs):
            return _SpyHandle(original_open(self, *args, **kwargs))

        with patch.object(Path, "open", _spy_open):
            asi.log("atomic", marker=1)

        assert len(write_calls) == 1, f"expected one write call, got {write_calls!r}"
        assert write_calls[0].endswith("\n")
        assert json.loads(write_calls[0])["message"] == "atomic"

    def test_empty_call_skips_write(self, tmp_path, monkeypatch):
        """``log()`` with neither values nor fields is a no-op even
        when the env var is set — there's nothing to record."""
        target = tmp_path / "asi.jsonl"
        monkeypatch.setenv(asi.HELIX_ASI_LOG_ENV, str(target))
        asi.log()
        assert not target.exists()


# ---------------------------------------------------------------------------
# log(): OSError / unwritable path is best-effort
# ---------------------------------------------------------------------------


class TestLogOSErrorSwallowed:
    def test_unwritable_directory_does_not_raise(self, tmp_path, monkeypatch):
        """Pointing the env var at a non-existent directory makes the
        ``open("a")`` call fail with ``OSError``.  ``helix.log`` must
        swallow it: observability is best-effort and an evaluator
        should never crash because its log file disappeared."""
        bad_path = tmp_path / "does" / "not" / "exist" / "asi.jsonl"
        monkeypatch.setenv(asi.HELIX_ASI_LOG_ENV, str(bad_path))
        # Must not raise.
        asi.log("hello")
        assert not bad_path.exists()

    def test_open_oserror_is_swallowed(self, tmp_path, monkeypatch):
        """Belt-and-suspenders: explicitly inject an ``OSError`` from
        the ``open`` call (e.g. read-only filesystem)."""
        target = tmp_path / "asi.jsonl"
        monkeypatch.setenv(asi.HELIX_ASI_LOG_ENV, str(target))

        def _boom(self, *_args, **_kwargs):
            raise OSError("read-only filesystem")

        with patch.object(Path, "open", _boom):
            asi.log("should not crash")  # must not raise


# ---------------------------------------------------------------------------
# read_text(): rendering
# ---------------------------------------------------------------------------


class TestReadTextRendering:
    def test_message_renders_as_bare_line(self):
        raw = json.dumps({"message": "the note"}) + "\n"
        assert asi.read_text(raw) == "the note"

    def test_scalar_field_renders_unquoted(self):
        raw = json.dumps({"score": 0.42}) + "\n"
        assert "score: 0.42" in asi.read_text(raw)

    def test_nested_dict_renders_as_json_literal(self):
        """Nested values must render via ``json.dumps`` so the mutation
        prompt sees a readable JSON literal (e.g.
        ``trajectory: {"step": 1}``) instead of Python's repr-style
        single-quoted ``{'step': 1}``."""
        raw = json.dumps({"trajectory": {"step": 1, "ok": True}}) + "\n"
        rendered = asi.read_text(raw)
        # JSON serialisation: double-quoted keys + ``true`` (not Python ``True``).
        assert "trajectory: " in rendered
        assert '"step": 1' in rendered
        assert '"ok": true' in rendered
        # Must NOT be Python repr.
        assert "'step': 1" not in rendered

    def test_list_field_renders_as_json_literal(self):
        raw = json.dumps({"attempts": ["fail", "ok"]}) + "\n"
        rendered = asi.read_text(raw)
        assert 'attempts: ["fail", "ok"]' in rendered

    def test_non_json_line_passes_through(self):
        """A garbage line (e.g. evaluator wrote freeform text into the
        same file) must not crash the renderer; pass it through
        verbatim."""
        rendered = asi.read_text("not json at all\n")
        assert rendered == "not json at all"

    def test_message_then_fields_render_in_order(self):
        raw = json.dumps({"message": "header", "score": 0.7, "ok": True}) + "\n"
        rendered = asi.read_text(raw)
        # Message first; then sorted fields.  Sorted is deterministic
        # for prompt reproducibility.  Scalars (incl. booleans) round-trip
        # via ``str(...)`` so a Python ``True`` renders as ``True`` —
        # ``json.dumps`` is reserved for nested containers, where the
        # repr-vs-JSON-literal distinction actually matters.
        lines = rendered.splitlines()
        assert lines[0] == "header"
        assert "ok: True" in lines
        assert "score: 0.7" in lines


# ---------------------------------------------------------------------------
# read() / clear()
# ---------------------------------------------------------------------------


class TestReadAndClear:
    def test_read_missing_file_returns_empty(self, tmp_path):
        assert asi.read(tmp_path / "does-not-exist.jsonl") == ""

    def test_clear_removes_file(self, tmp_path):
        target = tmp_path / "asi.jsonl"
        target.write_text("anything\n")
        asi.clear(target)
        assert not target.exists()

    def test_clear_missing_file_no_raise(self, tmp_path):
        # Idempotent: clearing a non-existent path is fine.
        asi.clear(tmp_path / "never-existed.jsonl")
