# Source and scoring record

This demo is pinned to the following primary sources:

- Harness: `microsoft/SWE-bench-Live` commit
  `70ec57e852e3f2d195790fe71f553e272c691833` (MIT).
- Harness runtime submodule: `microsoft/RepoLaunch` commit
  `7735b1e7363dd3bbc69bd0ef80db646a2ae391fd` (MIT).
- Dataset: `SWE-bench-Live/MultiLang` revision
  `608f7ae9ab8ea1f9f0d030fe04562cf6bd1a0c8b`.
- Dataset object: `data/c-00000-of-00001.parquet`, 8,872,150 bytes,
  SHA-256 `0d3b31cc38c807160e3fef132ed0f86b1e33890a842372894c2340ad08794674`.
- Task: `capstone-engine__capstone-2743`, repository base commit
  `56db8c2b690eb6372c91f8d76621f43a33c4dbe4`.
- Official image:
  `docker.io/starryzhang/sweb.eval.x86_64.capstone-engine_1776_capstone-2743@sha256:c3d6222106db9afce1eaf6036f67d540011e46ea8e59419097c32d0555032ed9`,
  platform `linux/amd64`.

The official upstream gold smoke was executed from the pinned harness checkout:

```bash
uv run --no-project --with datasets --with docker --with typing-extensions \
  python -m evaluation.evaluation \
  --dataset /tmp/swebench-live-capstone-2743.jsonl \
  --platform linux \
  --patch_dir gold \
  --output_dir /tmp/swebench-live-gold-1 \
  --workers 1 --overwrite 1 \
  --instance_ids capstone-engine__capstone-2743
```

Result: `Success: 1`, `Failure: 0`, `Error: 0`. Two more independent gold
runs, including two concurrent amd64-emulated containers, returned the same
official resolution. Gold proves the evaluator path only; it is never reported
as a HELIX candidate score.

During adapter development, the first `python3 evaluate.py --gold-smoke` used
`git clean -fdx` and deleted the image's prebuilt ignored build tree. The smoke
became an unnecessary full emulated rebuild and was terminated; the adapter
reported `docker start --attach ... returned non-zero exit status 137`. This was
an adapter fidelity defect, not an amd64-emulation failure. Candidate execution
now happens in a disposable local Git clone while the official `/testbed` is
left exactly as shipped. The identical command then resolved all 13 parsed
tests in 5.206 seconds (`rebuild_exit_code=0`, `test_exit_code=0`).

`official_runner.py` preserves the pinned upstream sequence and decision rule:
apply the official test patch, apply the prediction patch best-effort, execute
the row's exact rebuild/test/print commands, execute the row's parser, and mark
resolved only when every FAIL_TO_PASS test passes and no parsed PASS_TO_PASS or
FAIL_TO_PASS test fails. Extra hashes, byte counts, exit codes, and timings are
explicitly auxiliary diagnostics.
