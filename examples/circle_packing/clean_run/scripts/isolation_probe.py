"""Isolation checks for the leak-free circle-packing run, through HELIX's own code.

Run with HELIX's environment and WITHOUT an Anthropic credential in the
environment (the probe refuses otherwise, because HELIX forwards one into
agent containers)::

    env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \\
      uv run --project <helix checkout> python isolation_probe.py --run-dir <run dir>

launch.sh runs it before every launch.  It needs the run's agent network,
egress gateway and agent home volume to exist (launch.sh creates them).

It loads ``<run dir>/.helix-run/helix.toml`` with ``helix.config.load_config``
and starts containers exactly as ``helix evolve`` would
(``helix.sandbox.run_sandboxed_command`` / ``start_evaluator_sidecar``,
``helix.executor.run_evaluator``, ``helix.mutator._build_backend_args``), then
checks:

A. the candidate tree holds only the seed files; no target values, sources or
   strategy names appear in agent-visible text, including the fully rendered
   first mutation prompt; the claude argv denies WebSearch and WebFetch;
B. an agent container sees only the candidate workspace copy and its
   dedicated home volume (no host paths, no evaluator code, no ledger, no
   HELIX state, no credential), cannot reach the web or the model API
   directly, and can reach only the egress gateway, which refuses everything
   but the Messages API without web or server tools;
C. the evaluator sidecar scores the seed, valid, within-tolerance, invalid,
   crashing and timed-out candidates as the rules say, records every scoring
   execution and the tree it scored, refuses evaluation requests made by
   candidate code, keeps candidate code away from the ledger and the scoring
   code, cleans up candidate processes, and cannot be reached from an agent
   container;
D. (no API spend) Claude Code, run by HELIX's argv builder in an agent
   container pointed at a capture-only gateway, sends requests for the
   configured model with ``output_config.effort`` equal to the configured
   effort, offers no WebSearch/WebFetch tool, no server tool and no MCP server;
E. (``--api-auth-check`` only; one Messages request with at most 64 output tokens) the
   model API answers through the real gateway from inside an agent container
   with the configured model and effort.

The report is written as JSON; the exit code is 0 only if every check passed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from helix.config import HelixConfig, load_config
from helix.executor import _scrub_environment, run_evaluator
from helix.mutator import _add_backend_auth_env, _build_backend_args, build_mutation_prompt
from helix.population import Candidate
from helix.sandbox import (
    current_evaluator_sidecar_runtime,
    resolve_sandbox_image,
    run_sandboxed_command,
    start_evaluator_sidecar,
)

LABEL = "helix-circle-clean"
EXPECTED_TREE = {".gitignore", "check_packing.py", "solve.py"}
FORBIDDEN_TOKENS = [
    "2.63", "2.635", "2.636", "2.64", "packomania", "alphaevolve", "friedman",
    "optimize_anything", "optimize anything", "gepa", "shinka", "openevolve",
    "known_best", "best known", "best-known", "record", "state of the art",
    "http://", "https://", "www.", "slsqp", "l-bfgs", "lbfgs", "hexagonal",
    "lattice", "cma-es", "bilevel", "linear program",
]
BLOCKED_URLS = [
    "https://example.com/",
    "http://example.com/",
    "https://www.packomania.com/",
    "https://packomania.com/",
    "http://hydra.nat.uni-magdeburg.de/packing/csq/csq.html",
    "https://erich-friedman.github.io/packing/cirRsqu/",
    "https://gepa-ai.github.io/gepa/",
    "https://github.com/",
    "https://pypi.org/simple/",
    "https://api.anthropic.com/v1/messages",
    "https://1.1.1.1/",
]

AGENT_PROBE = r"""
import json, os, subprocess, sys, urllib.request
out = {}
def run(cmd, timeout=40):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"rc": p.returncode, "out": p.stdout[-400:], "err": p.stderr[-400:]}
    except Exception as e:
        return {"rc": None, "error": type(e).__name__}
out["workspace"] = sorted(os.listdir("/workspace"))
mounts = []
with open("/proc/self/mountinfo") as f:
    for line in f:
        parts = line.split()
        mounts.append({"mount_point": parts[4], "root": parts[3], "source": parts[-2]})
out["mounts"] = mounts
out["home_files"] = sorted(
    os.path.relpath(os.path.join(d, n), "/home/node")
    for d, _, files in os.walk("/home/node") for n in files
)[:200]
hits = []
for root, dirs, files in os.walk("/"):
    if root.startswith(("/proc", "/sys", "/dev")):
        dirs[:] = []
        continue
    for name in files + dirs:
        low = name.lower()
        if any(k in low for k in ("known_best", "packing_rules", "evaluator_server", "cpeval",
                                  "helix.toml", "evaluations.jsonl", "reference_known",
                                  "circle-packing", "circle_packing", "state.json", "ledger",
                                  "packomania", "alphaevolve")):
            hits.append(os.path.join(root, name))
out["suspicious_paths"] = hits[:50]
env = dict(os.environ)
out["env_keys"] = sorted(env)
out["env_has_real_key"] = any("sk-ant" in v for v in env.values())
out["anthropic_base_url"] = env.get("ANTHROPIC_BASE_URL")
out["api_key_is_placeholder"] = env.get("CLAUDE_CODE_OAUTH_TOKEN", "").startswith("placeholder")
out["docker_socket"] = os.path.exists("/var/run/docker.sock")
out["curl"] = {u: run(["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "-m", "15", u])
               for u in json.loads(sys.argv[1])}
def py_fetch(url):
    try:
        urllib.request.urlopen(url, timeout=15)
        return "reachable"
    except Exception as e:
        return type(e).__name__
out["python_fetch"] = {u: py_fetch(u) for u in ["https://example.com/", "https://www.packomania.com/"]}
out["dns"] = {h: run(["getent", "hosts", h]) for h in ["example.com", "www.packomania.com", "packomania.com", "github.com", "pypi.org", "api.anthropic.com", "anthropic-gateway", "helix-evaluator"]}
gw = "http://anthropic-gateway:8080"
out["gateway_as_proxy"] = {
    u: run(["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "-m", "15", "-x", gw, u])
    for u in ["https://example.com/", "http://example.com/", "https://api.anthropic.com/v1/models"]
}
out["gateway_get_models"] = run(["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "-m", "15", gw + "/v1/models"])
def post(body, route="/v1/messages"):
    return run(["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "-m", "15", "-X", "POST",
                "-H", "content-type: application/json", "-d", json.dumps(body), gw + route])
base = {"model": sys.argv[2], "max_tokens": 1, "messages": [{"role": "user", "content": "x"}]}
out["gateway_web_search_tool"] = post({**base, "tools": [{"type": "web_search_20260209", "name": "web_search"}]})
out["gateway_web_fetch_tool"] = post({**base, "tools": [{"type": "web_fetch_20260209", "name": "web_fetch"}]})
out["gateway_code_execution_tool"] = post({**base, "tools": [{"type": "code_execution_20260521", "name": "code_execution"}]})
out["gateway_websearch_named_tool"] = post({**base, "tools": [{"name": "WebSearch", "input_schema": {"type": "object"}}]})
out["gateway_webfetch_named_tool"] = post({**base, "tools": [{"name": "WebFetch", "input_schema": {"type": "object"}}]})
url_image = {"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}}
url_doc = {"type": "document", "source": {"type": "url", "url": "https://example.com/x.pdf"}}
file_doc = {"type": "document", "source": {"type": "file", "file_id": "file_test"}}
blocks = [url_image, url_doc, file_doc,
          {"type": "tool_result", "tool_use_id": "x", "content": [url_doc]},
          {"type": "document", "source": {"type": "content", "content": [url_image]}}]
out["gateway_remote_sources"] = [post({**base, "messages": [{"role": "user", "content": [block]}]}, route)
                                 for route in ["/v1/messages", "/v1/messages/count_tokens"] for block in blocks]
out["gateway_mcp_servers"] = post({**base, "mcp_servers": [{"type": "url", "url": "https://example.com/mcp", "name": "x"}]})
out["gateway_other_model"] = post({**base, "model": "claude-opus-5"})
out["evaluator_by_name"] = run(["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "-m", "8", "http://helix-evaluator:8080/evaluate"])
if len(sys.argv) > 3 and sys.argv[3]:
    out["evaluator_by_ip"] = run(["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "-m", "8", "http://" + sys.argv[3] + ":8080/evaluate"])
out["check_packing"] = run(["python3", "check_packing.py"], timeout=120)
out["versions"] = run(["python3", "-c", "import sys, numpy, scipy; print(sys.version.split()[0], numpy.__version__, scipy.__version__)"])
out["claude_version"] = run(["claude", "--version"])
print("PROBE_JSON=" + json.dumps(out))
"""

SNOOP_SOLVE = r'''
import json, os, subprocess, sys, urllib.request

def _try(label, fn):
    try:
        value = fn()
        print(label, "OK", repr(value)[:120])
    except Exception as exc:
        print(label, "FAIL", type(exc).__name__)

def _post(url):
    body = json.dumps({"protocol": 1, "split": "val", "ids": ["0"], "client_error": "x"}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=5).status

def pack_circles(n):
    _try("read_ledger", lambda: open("/srv/cpeval/ledger/evaluations.jsonl").read(50))
    _try("list_ledger", lambda: os.listdir("/srv/cpeval/ledger"))
    _try("list_archives", lambda: os.listdir("/srv/cpeval/ledger/candidate_archives"))
    _try("list_ledger_parent", lambda: os.listdir("/srv/cpeval"))
    _try("write_ledger", lambda: open("/srv/cpeval/ledger/evaluations.jsonl", "a").write("{}"))
    _try("read_rules", lambda: open("/opt/cpeval/private/packing_rules.py").read(50))
    _try("list_private", lambda: os.listdir("/opt/cpeval/private"))
    _try("server_environ", lambda: open("/proc/1/environ").read(50))
    _try("write_server_stdout", lambda: open("/proc/1/fd/1", "w").write("HELIX_RESULT=[[9.0, {}]]\n"))
    _try("list_work_root", lambda: os.listdir("/work"))
    _try("fetch_web", lambda: urllib.request.urlopen("https://example.com/", timeout=5).status)
    _try("post_localhost", lambda: _post("http://127.0.0.1:8080/evaluate"))
    _try("post_by_service_name", lambda: _post("http://helix-evaluator:8080/evaluate"))
    _try("spawn_daemon", lambda: subprocess.Popen(["sleep", "1000"], start_new_session=True).pid)
    r = 1.0 / 16.0
    return [(r + (i % 8) * 2 * r, r + (i // 8) * 2 * r, r) for i in range(n)]
'''

GRID = "r = 1 / 16\nrows = [[r + (i % 8) * 2 * r, r + (i // 8) * 2 * r, r] for i in range(26)]\n"


def _certified(rows: list[list[float]], tol: float = 1e-6) -> float:
    """Independent float re-statement of the scoring rule, for expected values."""
    boundary = max(max(r - x, r - y, x + r - 1, y + r - 1) for x, y, r in rows)
    overlap = max(
        rows[i][2] + rows[j][2] - math.hypot(rows[i][0] - rows[j][0], rows[i][1] - rows[j][1])
        for i in range(len(rows)) for j in range(i + 1, len(rows))
    )
    assert boundary <= tol and overlap <= tol
    shrink = max(0.0, boundary, overlap / 2)
    return math.fsum(max(0.0, r - shrink) for _, _, r in rows)


def _grid(bump: float = 0.0) -> list[list[float]]:
    r = 1 / 16
    rows = [[r + (i % 8) * 2 * r, r + (i // 8) * 2 * r, r] for i in range(26)]
    rows[9][2] += bump
    return rows


EVAL_CASES: dict[str, tuple[str, float, dict[str, Any]]] = {
    # name: (solve.py source, expected score, extra checks)
    "valid_grid": (GRID + "def pack_circles(n):\n    return rows\n", 26 / 16, {"valid": True}),
    "numpy_rows": (
        "import numpy as np\n" + GRID + "def pack_circles(n):\n    return np.array(rows)\n",
        26 / 16, {"valid": True},
    ),
    "overlap_within_tolerance_is_shrunk": (
        GRID + "rows[9][2] += 5e-7\ndef pack_circles(n):\n    return rows\n",
        _certified(_grid(5e-7)), {"valid": True},
    ),
    "overlap_beyond_tolerance": (
        GRID + "rows[9][2] += 5e-6\ndef pack_circles(n):\n    return rows\n", 0.0, {"valid": False},
    ),
    "overlapping": ("def pack_circles(n):\n    return [(0.5, 0.5, 0.5)] * n\n", 0.0, {"valid": False}),
    "out_of_bounds": (
        GRID + "rows[0][0] = -0.01\ndef pack_circles(n):\n    return rows\n", 0.0, {"valid": False},
    ),
    "wrong_count": (GRID + "def pack_circles(n):\n    return rows[:25]\n", 0.0, {"valid": False}),
    "nan": ("def pack_circles(n):\n    return [(float('nan'), 0.5, 0.1)] * n\n", 0.0, {"valid": False}),
    "negative_and_huge": (
        "def pack_circles(n):\n    return [(0.5, 0.5, 1e308)] + [(0.5, 0.5, -1.0)] * (n - 1)\n",
        0.0, {"valid": False},
    ),
    "raises": ("def pack_circles(n):\n    raise RuntimeError('boom')\n", 0.0, {"valid": False}),
    "exit_early": ("import os\ndef pack_circles(n):\n    os._exit(0)\n", 0.0, {"valid": False}),
    "timeout": (
        "import time\ndef pack_circles(n):\n    time.sleep(3600)\n", 0.0,
        {"valid": False, "timed_out": True},
    ),
    "snoop": (SNOOP_SOLVE, 26 / 16, {"valid": True}),
}

EVAL_CASES["fifo"] = ("import os, sys\ndef pack_circles(n):\n    os.mkfifo(sys.argv[1]); os._exit(0)\n", 0.0, {"valid": False})
for index in range(3):
    EVAL_CASES[f"persistent_state_{index}"] = (
        GRID + f"# unique tree {index}\n" +
        "def pack_circles(n):\n    from pathlib import Path\n"
        "    for directory in ['/tmp', '/var/tmp', '/dev/shm']:\n"
        "        p = Path(directory) / 'cp-persistence-canary'\n"
        "        if p.exists(): return []\n"
        "        p.write_text('marker')\n"
        "    return rows\n", 26 / 16, {"valid": True})


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, passed: bool, evidence: Any = None) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "evidence": evidence})
        print(("PASS " if passed else "FAIL ") + name, flush=True)

    @property
    def ok(self) -> bool:
        return all(c["passed"] for c in self.checks)


def _docker(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check)


def _agent_env(config: HelixConfig) -> dict[str, str]:
    # Exactly what invoke_claude_code hands the sandbox.
    env = _scrub_environment(passthrough_env=config.passthrough_env, fixed_env=config.env)
    _add_backend_auth_env(env, config.agent.backend)
    return env


def _seed_copy(run_dir: Path, dest: Path) -> None:
    archive = subprocess.run(
        ["git", "-C", str(run_dir), "archive", "HEAD"], capture_output=True, check=True
    ).stdout
    dest.mkdir(parents=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)


def _scan(text: str) -> list[str]:
    low = text.lower()
    return [t for t in FORBIDDEN_TOKENS if t in low]


def check_run_dir(run_dir: Path, config: HelixConfig, report: Report) -> None:
    tracked = set(
        subprocess.run(
            ["git", "-C", str(run_dir), "ls-files"], capture_output=True, text=True, check=True
        ).stdout.split()
    )
    report.check("A1 candidate tree holds only the seed files", tracked == EXPECTED_TREE, sorted(tracked))
    status = subprocess.run(
        ["git", "-C", str(run_dir), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True, text=True, check=True,
    ).stdout
    report.check("A2 no untracked or modified files would be snapshotted into the seed", status == "", status)
    top_level = sorted(p.name for p in run_dir.iterdir())
    report.check(
        "A3 run directory holds only the seed files, .git and the excluded .helix-run (plus .helix once running)",
        set(top_level) <= EXPECTED_TREE | {".git", ".helix-run", ".helix"},
        top_level,
    )
    found: dict[str, list[str]] = {}
    for name in sorted(tracked):
        hits = _scan((run_dir / name).read_text(errors="replace"))
        if hits:
            found[name] = hits
    hits = _scan(config.objective + "\n" + (config.agent.background or ""))
    if hits:
        found["objective+background"] = hits
    report.check("A4 no targets, known values, sources or strategy names in the seed files, objective or background",
                 not found, found)
    exclude = (run_dir / ".git" / "info" / "exclude").read_text()
    report.check("A5 .helix-run is excluded from the candidate tree", "/.helix-run/" in exclude)
    agent = config.agent
    report.check(
        "A6 agent settings",
        agent.backend == "claude"
        and set(agent.disallowed_tools) >= {"WebSearch", "WebFetch"}
        and config.sandbox.enabled
        and config.sandbox.evaluator
        and config.sandbox.auth_volume is not None
        and config.evolution.minibatch_size == 1
        and config.dataset.train_size == 1
        and config.dataset.val_size == 1,
        {"model": agent.model, "effort": agent.effort, "max_turns": agent.max_turns,
         "disallowed_tools": agent.disallowed_tools, "network": config.sandbox.network,
         "auth_volume": config.sandbox.auth_volume, "max_evaluations": config.evolution.max_evaluations,
         "minibatch_size": config.evolution.minibatch_size},
    )
    argv = _build_backend_args("/workspace", agent, ".agent_task_prompt.md")
    report.check(
        "A7 claude argv denies WebSearch and WebFetch and sets model, effort and max turns",
        "--disallowedTools=WebSearch,WebFetch" in argv
        and argv[argv.index("--effort") + 1] == agent.effort
        and argv[argv.index("--model") + 1] == agent.model
        and argv[argv.index("--max-turns") + 1] == str(agent.max_turns),
        argv,
    )


def check_prompt(config: HelixConfig, seed_result: Any, report: Report, out_dir: Path) -> None:
    prompt = build_mutation_prompt(
        config.objective, seed_result, config.agent.background, config.agent.max_turns
    )
    (out_dir / "first_mutation_prompt.md").write_text(prompt)
    report.check("A8 the rendered first mutation prompt (seed feedback included) has no targets, sources or strategy names",
                 not _scan(prompt), {"hits": _scan(prompt), "chars": len(prompt)})


def check_infrastructure(config: HelixConfig, report: Report) -> None:
    network = config.sandbox.network
    info = _docker("network", "inspect", network, "--format", "{{json .}}")
    data = json.loads(info.stdout) if info.returncode == 0 else {}
    containers = [c.get("Name") for c in (data.get("Containers") or {}).values()]
    report.check(
        "B0 agent network is --internal, labeled, and holds only the gateway",
        data.get("Internal") is True
        and (data.get("Labels") or {}).get(LABEL) == "1"
        and containers == ["helix-circle-clean-gateway"],
        {"internal": data.get("Internal"), "labels": data.get("Labels"), "containers": containers},
    )


def run_agent_probe(config: HelixConfig, run_dir: Path, report: Report, sidecar_ip: str, label: str) -> dict[str, Any]:
    workdir = Path(tempfile.mkdtemp(prefix="cp-probe-agent-"))
    try:
        source = workdir / "tree"
        _seed_copy(run_dir, source)
        image = resolve_sandbox_image(config.sandbox, config.agent.backend)
        result = run_sandboxed_command(
            ["python3", "-c", AGENT_PROBE, json.dumps(BLOCKED_URLS), str(config.agent.model), sidecar_ip],
            cwd=source,
            env=_agent_env(config),
            sandbox=config.sandbox,
            scope="agent",
            sync_back=False,
            image=image,
            agent_backend=config.agent.backend,
        )
        line = next((rec for rec in result.stdout.splitlines() if rec.startswith("PROBE_JSON=")), None)
        if line is None:
            report.check(f"{label} agent probe ran", False,
                         {"stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]})
            return {}
        return json.loads(line[len("PROBE_JSON="):])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def assess_agent_probe(probe: dict[str, Any], report: Report, *, with_sidecar: bool) -> None:
    if not probe:
        return
    if not with_sidecar:
        report.check(
            "B1 agent workspace holds only the candidate files (+ HELIX's synthetic .git)",
            set(probe["workspace"]) == EXPECTED_TREE | {".git"},
            probe["workspace"],
        )
        allowed_mounts = {"/", "/workspace", "/home/node", "/etc/resolv.conf", "/etc/hostname", "/etc/hosts"}
        extra = [
            m for m in probe["mounts"]
            if m["mount_point"] not in allowed_mounts
            and not m["mount_point"].startswith(("/proc", "/sys", "/dev"))
        ]
        host_like = [m for m in probe["mounts"] if m["mount_point"] not in ("/workspace", "/etc/resolv.conf", "/etc/hostname", "/etc/hosts")
                     and ("/Users" in m["root"] or "/Users" in m["mount_point"] or "host_mnt" in m["root"])]
        report.check("B2 no host filesystem mounts beyond the workspace copy and home volume; no Docker socket",
                     not extra and not host_like and not probe["docker_socket"],
                     {"extra": extra, "host_like": host_like, "all": probe["mounts"]})
        report.check("B3 no evaluator code, references, ledgers, HELIX state or configs anywhere on the agent filesystem",
                     not probe["suspicious_paths"], probe["suspicious_paths"])
        report.check("B4 no credential in the agent environment or home volume",
                     not probe["env_has_real_key"] and probe["api_key_is_placeholder"]
                     and not any(".credentials" in f for f in probe["home_files"]),
                     {"env_keys": probe["env_keys"], "home_files": probe["home_files"]})
        curl = probe["curl"]
        report.check("B5 direct web and API egress fails (curl: example.com, packomania, github, pypi, api.anthropic.com, 1.1.1.1)",
                     all(v.get("rc") not in (0, None) or v.get("out") in ("000", "") for v in curl.values()), curl)
        report.check("B6 direct web egress fails (python urllib)",
                     all(v != "reachable" for v in probe["python_fetch"].values()), probe["python_fetch"])
        dns = probe["dns"]
        report.check("B7 external names do not resolve; the gateway does",
                     all(dns[h].get("rc") != 0 for h in ("example.com", "www.packomania.com", "packomania.com", "github.com", "pypi.org", "api.anthropic.com"))
                     and dns["anthropic-gateway"].get("rc") == 0, dns)
        proxy = probe["gateway_as_proxy"]
        report.check("B8 the gateway refuses to act as a web proxy",
                     all(v.get("out") in ("403", "000", "") for v in proxy.values()), proxy)
        report.check("B9 the gateway refuses non-Messages routes", probe["gateway_get_models"].get("out") == "403",
                     probe["gateway_get_models"])
        denied = {k: probe[k] for k in ("gateway_web_search_tool", "gateway_web_fetch_tool",
                                        "gateway_code_execution_tool", "gateway_websearch_named_tool",
                                        "gateway_mcp_servers", "gateway_other_model", "gateway_webfetch_named_tool")}
        report.check("B10 the gateway refuses web and server tools, MCP servers and other models",
                     all(v.get("out") == "403" for v in denied.values()), denied)
        report.check("B14 URL/file/nested sources denied on Messages and count_tokens",
                     all(v.get("out") == "403" for v in probe["gateway_remote_sources"]),
                     probe["gateway_remote_sources"])
        report.check("B11 check_packing.py reports VALID for the seed without printing a score",
                     probe["check_packing"].get("rc") == 0
                     and probe["check_packing"].get("out", "").startswith("VALID")
                     and "0.97" not in probe["check_packing"].get("out", ""),
                     probe["check_packing"])
        report.check("B12 agent Python matches the evaluator (3.11.2 / numpy 2.3.3 / scipy 1.16.2)",
                     probe["versions"].get("out", "").split() == ["3.11.2", "2.3.3", "1.16.2"], probe["versions"])
    evaluator_hits = {k: v for k, v in probe.items() if k.startswith("evaluator_by")}
    report.check(
        ("C9" if with_sidecar else "B13") + " agent container cannot reach the evaluator service",
        all(v.get("out") in ("000", "") for v in evaluator_hits.values()),
        evaluator_hits,
    )


def _ledger_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(rec) for rec in path.read_text().splitlines() if rec.strip()]


def check_evaluator(config: HelixConfig, run_dir: Path, report: Report, out_dir: Path) -> Any:
    ledger_dir = out_dir / "ledger"
    if ledger_dir.exists():
        shutil.rmtree(ledger_dir)
    ledger_dir.mkdir(parents=True)
    ledger_dir.chmod(0o700)
    assert config.evaluator.sidecar is not None
    # The probe's sidecar gets its own ledger, a short candidate time limit and
    # an execution cap of exactly the number of evaluations planned below, so
    # the cap itself is exercised too.
    planned = 1 + len(EVAL_CASES)
    command = re.sub(
        r"CP_MAX_EXECUTIONS=\d+", f"CP_MAX_EXECUTIONS={planned}", config.evaluator.sidecar.command
    ).replace("python3 ", "env CP_CANDIDATE_TIMEOUT=20 python3 ", 1)
    assert f"CP_MAX_EXECUTIONS={planned}" in command
    ledger_target = config.evaluator.sidecar.volumes[0].split(":")[1]
    sidecar = config.evaluator.sidecar.model_copy(
        update={"volumes": [f"{ledger_dir}:{ledger_target}:rw"], "command": command}
    )
    test_config = config.model_copy(
        update={"evaluator": config.evaluator.model_copy(update={"sidecar": sidecar})}
    )
    work = Path(tempfile.mkdtemp(prefix="cp-probe-eval-"))
    seed = None
    try:
        with start_evaluator_sidecar(
            sidecar, passthrough_env=test_config.passthrough_env, fixed_env=test_config.env,
            extra_hosts=test_config.sandbox.extra_hosts,
        ):
            runtime = current_evaluator_sidecar_runtime()
            assert runtime is not None
            ip = _docker("inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                         runtime.container_name).stdout.strip()

            def evaluate(name: str, prepare: Callable[[Path], None], split: str = "val") -> Any:
                tree = work / name
                prepare(tree)
                (tree / "helix_batch.json").write_text('["0"]')
                candidate = Candidate(id=f"probe-{name}", worktree_path=str(tree), branch_name="probe",
                                      generation=0, parent_id=None, parent_ids=[], operation="probe")
                return run_evaluator(candidate, test_config, split=split)

            seed = evaluate("seed", lambda t: _seed_copy(run_dir, t))
            seed_score = seed.instance_scores.get("0")
            report.check("C1 seed scores 0.979764 through HELIX's sandboxed evaluator path",
                         seed_score is not None and round(seed_score, 6) == 0.979764,
                         {"score": seed_score, "side_info_keys": sorted((seed.per_example_side_info or [{}])[0])})
            results: dict[str, Any] = {}
            for name, (source, expected, extra) in EVAL_CASES.items():
                def prepare(t: Path, source: str = source) -> None:
                    _seed_copy(run_dir, t)
                    (t / "solve.py").write_text(source)
                started = time.monotonic()
                res = evaluate(name, prepare)
                side = (res.per_example_side_info or [{}])[0]
                results[name] = {
                    "score": res.instance_scores.get("0"),
                    "expected": expected,
                    "valid": side.get("valid"),
                    "feedback": str(side.get("feedback"))[:300],
                    "timed_out": (side.get("candidate_run") or {}).get("timed_out"),
                    "stdout_tail": side.get("stdout_tail"),
                    "seconds": round(time.monotonic() - started, 1),
                }
                score = res.instance_scores.get("0")
                ok = score is not None and abs(score - expected) <= 1e-12 and side.get("valid") is extra["valid"]
                if extra.get("timed_out"):
                    ok = ok and results[name]["timed_out"] is True
                report.check(f"C2 {name}: valid={extra['valid']} score={expected!r}", ok, results[name])
            try:
                over = evaluate("over_cap", lambda t: _seed_copy(run_dir, t), split="train")
                over_result: Any = {"unexpected_score": over.instance_scores.get("0")}
            except Exception as exc:  # noqa: BLE001 - the refusal surfaces as an evaluator error
                over_result = {"refused": type(exc).__name__}
            report.check("C10 the evaluator refuses scoring requests beyond its execution cap",
                         "refused" in over_result, over_result)
            snoop_out = results["snoop"]["stdout_tail"] or ""
            must_fail = ["read_ledger", "list_ledger", "list_archives", "list_ledger_parent",
                         "write_ledger", "read_rules", "list_private",
                         "server_environ", "write_server_stdout", "list_work_root", "fetch_web",
                         "post_localhost", "post_by_service_name"]
            outcomes = {line.split()[0]: line.split()[1] for line in snoop_out.splitlines() if len(line.split()) >= 2}
            report.check("C3 candidate code cannot read the ledger or scoring code, reach the web, or request evaluations",
                         all(outcomes.get(k) == "FAIL" for k in must_fail), outcomes)
            listing = _docker("exec", runtime.container_name, "ps", "-eo", "uid=,stat=,comm=").stdout
            leftovers = [rec.split() for rec in listing.splitlines() if rec.split() and rec.split()[0] == "2000"]
            report.check("C4 no candidate processes (running or zombie) survive an evaluation",
                         not leftovers,
                         {"candidate_processes": leftovers, "daemon_spawned": outcomes.get("spawn_daemon")})
            probe = run_agent_probe(config, run_dir, report, ip, "C")
            assess_agent_probe(probe, report, with_sidecar=True)
        evaluations = _ledger_lines(ledger_dir / "evaluations.jsonl")
        rejections = _ledger_lines(ledger_dir / "rejected_requests.jsonl")
        expected_count = 1 + len(EVAL_CASES)
        report.check("C5 ledger holds exactly one record per HELIX evaluator call, in order",
                     [e["seq"] for e in evaluations] == list(range(1, expected_count + 1)),
                     {"count": len(evaluations), "expected": expected_count})
        report.check("C6 candidate-originated requests (403) and the over-cap request (429) were refused and logged, not counted",
                     any(r["status"] == 403 for r in rejections)
                     and [r["status"] for r in rejections].count(429) == 1
                     and all(r["status"] in (403, 429) for r in rejections), rejections)
        report.check("C7 ledger scores match what HELIX received",
                     evaluations and evaluations[0]["score"] == seed_score, None)
        archived = [e for e in evaluations if e.get("archive") and (ledger_dir / e["archive"]).is_file()]
        report.check("C8 ledger keeps each scored packing and candidate tree",
                     bool(evaluations) and len(evaluations[0].get("circles") or []) == 26
                     and len(archived) == len(evaluations),
                     {"archived": len(archived), "records": len(evaluations)})
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return seed


def production_prompt(config: HelixConfig, run_dir: Path, report: Report, out_dir: Path) -> None:
    """Score the seed with the production settings for the exact rendered prompt."""
    assert config.evaluator.sidecar is not None
    ledger = out_dir / ("prompt-ledger-" + uuid.uuid4().hex[:8])
    ledger.mkdir()
    sidecar = config.evaluator.sidecar.model_copy(update={
        "volumes": [f"{ledger}:/srv/cpeval/ledger:rw"]})
    probe_config = config.model_copy(update={
        "evaluator": config.evaluator.model_copy(update={"sidecar": sidecar})})
    with tempfile.TemporaryDirectory(prefix="cp-prompt-") as tmp:
        tree = Path(tmp) / "tree"
        _seed_copy(run_dir, tree)
        (tree / "helix_batch.json").write_text('["0"]')
        candidate = Candidate(id="g0-s0", worktree_path=str(tree), branch_name="probe",
                              generation=0, parent_id=None, parent_ids=[], operation="seed")
        with start_evaluator_sidecar(sidecar, passthrough_env=config.passthrough_env,
                                     fixed_env=config.env, extra_hosts=config.sandbox.extra_hosts):
            result = run_evaluator(candidate, probe_config, split="val")
        check_prompt(config, result, report, out_dir)
        report.check("A9 prompt seed uses production 600-second timeout",
                     result.per_example_side_info[0]["candidate_run"]["time_limit_seconds"] == 600)


def symlink_probe(config: HelixConfig, run_dir: Path, report: Report) -> None:
    from helix.mutator import _ignore_helix_artifacts
    with tempfile.TemporaryDirectory(prefix="cp-links-") as tmp:
        root = Path(tmp)
        victim = root / "host-only.txt"
        victim.write_text("private host canary\n")
        tree = root / "tree"
        _seed_copy(run_dir, tree)
        code = ("import os,pathlib; p=pathlib.Path('.helix_artifacts/backend_transcripts'); "
                "p.mkdir(parents=True); "
                f"os.symlink({str(victim)!r},p/'host.jsonl'); "
                "pathlib.Path('.gitignore').unlink(); "
                f"os.symlink({str(victim)!r},'.gitignore'); "
                f"os.symlink({str(victim)!r},'escape')")
        result = run_sandboxed_command(["python3", "-c", code], cwd=tree, env=_agent_env(config),
            sandbox=config.sandbox, scope="agent", sync_back=True,
            image=resolve_sandbox_image(config.sandbox, config.agent.backend), agent_backend="claude")
        _ignore_helix_artifacts(tree)
        files = [p for p in tree.rglob('*') if p.is_file() and not p.is_symlink()]
        report.check("B15 agent-planted symlinks neither copy nor modify host files",
                     result.returncode == 0 and victim.read_text() == "private host canary\n"
                     and not (tree / "escape").is_symlink()
                     and all(b"private host canary" not in p.read_bytes() for p in files))


def capture_agent_requests(config: HelixConfig, run_dir: Path, report: Report, out_dir: Path) -> None:
    """Run Claude Code as HELIX would, against a gateway that records and refuses."""
    suffix = uuid.uuid4().hex[:8]
    network = f"{LABEL}-probe-{suffix}"
    gateway = f"{LABEL}-probe-gateway-{suffix}"
    # A throwaway home volume, so the capture session leaves no state or
    # transcript in the run's own agent home volume.
    home = f"{LABEL}-probe-home-{suffix}"
    capture = out_dir / f"capture-{suffix}"
    capture.mkdir(parents=True)
    gateway_image = config.sandbox.image.replace("agent-claude", "gateway") if config.sandbox.image else ""
    workdir = Path(tempfile.mkdtemp(prefix="cp-probe-capture-"))
    try:
        _docker("network", "create", "--internal", "--label", f"{LABEL}=1", network, check=True)
        _docker("volume", "create", "--label", f"{LABEL}=1", home, check=True)
        _docker("run", "--rm", "--label", f"{LABEL}=probe", "--network", "none", "-v", f"{home}:/home/node",
                str(config.sandbox.image), "chown", "node:node", "/home/node", check=True)
        _docker("run", "-d", "--name", gateway, "--label", f"{LABEL}=1", "--network", network,
                "--network-alias", "anthropic-gateway", "-e", "GATEWAY_CAPTURE_ONLY=1",
                "-e", f"GATEWAY_ALLOWED_MODELS={config.agent.model}",
                "-v", f"{capture}:/logs", gateway_image, check=True)
        time.sleep(2)
        source = workdir / "tree"
        _seed_copy(run_dir, source)
        (source / ".agent_task_prompt.md").write_text("Reply with the word OK.\n")
        sandbox = config.sandbox.model_copy(
            update={"network": network, "auth_volume": home, "timeout_seconds": 180}
        )
        argv = _build_backend_args("/workspace", config.agent, ".agent_task_prompt.md")
        started = time.monotonic()
        try:
            result = run_sandboxed_command(
                argv, cwd=source, env=_agent_env(config), sandbox=sandbox, scope="agent",
                sync_back=False, image=resolve_sandbox_image(sandbox, config.agent.backend),
                agent_backend=config.agent.backend,
            )
            outcome = {"exit_code": result.returncode, "stdout_tail": result.stdout[-600:],
                       "stderr_tail": result.stderr[-600:]}
        except subprocess.TimeoutExpired:
            outcome = {"timed_out": True}
        outcome["seconds"] = round(time.monotonic() - started, 1)
        time.sleep(1)
        lines = _ledger_lines(capture / "gateway_requests.jsonl")
        requests = [rec for rec in lines if rec.get("event") != "start"]
        messages = [rec for rec in requests if rec.get("path") == "/v1/messages"]
        main = [rec for rec in messages if rec.get("model") == config.agent.model]
        summary = {
            "claude_outcome": outcome,
            "requests": [
                {k: rec.get(k) for k in ("method", "path", "decision", "reason", "model", "effort", "thinking",
                                       "tool_names", "body_keys", "anthropic_beta", "max_tokens", "stream")}
                for rec in requests
            ],
        }
        (out_dir / f"capture-{suffix}.json").write_text(json.dumps(summary, indent=2))
        tools = sorted({t for rec in messages for t in (rec.get("tool_names") or [])})
        report.check("D1 first Claude Code request uses the configured model (later turns rely on gateway policy)",
                     bool(main), {"models": sorted({str(rec.get('model')) for rec in messages}), "requests": len(requests)})
        report.check(f"D2 every main-model request carries output_config.effort = {config.agent.effort!r}",
                     bool(main) and all(rec.get("effort") == config.agent.effort for rec in main),
                     [{k: rec.get(k) for k in ("model", "effort", "thinking")} for rec in main])
        report.check("D3 no request offered WebSearch/WebFetch, a server tool or an MCP server",
                     bool(messages)
                     and not ({"WebSearch", "WebFetch", "web_search", "web_fetch"} & set(tools))
                     and all(rec.get("reason") != "mcp_servers is not allowed" for rec in requests)
                     and not any(str(rec.get("reason", "")).startswith(("server tool", "tool not allowed")) for rec in requests)
                     and all("mcp_servers" not in (rec.get("body_keys") or []) for rec in messages),
                     {"tools_offered": tools})
        other = [rec for rec in requests if rec.get("path") not in ("/v1/messages", "/v1/messages/count_tokens")]
        report.check("D4 every request other than the Messages API was refused by policy",
                     all(str(rec.get("reason", "")).startswith("route not allowed") for rec in other),
                     sorted({f"{rec.get('method')} {rec.get('path')} -> {rec.get('reason')}" for rec in requests}))
    finally:
        _docker("rm", "-f", gateway)
        _docker("network", "rm", network)
        _docker("volume", "rm", home)
        shutil.rmtree(workdir, ignore_errors=True)


def api_auth_check(config: HelixConfig, run_dir: Path, report: Report) -> None:
    """A tiny real Claude Code subscription request, with production model/effort."""
    with tempfile.TemporaryDirectory(prefix="cp-probe-auth-") as tmp:
        source = Path(tmp) / "tree"
        _seed_copy(run_dir, source)
        argv = _build_backend_args("/workspace", config.agent, ".agent_task_prompt.md")
        argv[-1] = "Reply with the word OK. Do not use tools."
        env = _agent_env(config)
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = "64"
        try:
            result = run_sandboxed_command(
                argv, cwd=source, env=env,
                sandbox=config.sandbox.model_copy(update={"timeout_seconds": 180}),
                scope="agent", sync_back=False,
                image=resolve_sandbox_image(config.sandbox, config.agent.backend),
                agent_backend=config.agent.backend,
            )
            data = json.loads(result.stdout)
            evidence = {"exit_code": result.returncode, "is_error": data.get("is_error"),
                        "subtype": data.get("subtype"), "usage": data.get("usage"),
                        "model_usage": data.get("modelUsage"),
                        "result": str(data.get("result", ""))[:300]}
            passed = result.returncode == 0 and data.get("is_error") is False and any(
                model.startswith(str(config.agent.model)) for model in (data.get("modelUsage") or {}))
        except (ValueError, subprocess.TimeoutExpired) as exc:
            evidence = {"error": type(exc).__name__}
            passed = False
        report.check(f"E1 Claude Code subscription answers {config.agent.model} at effort {config.agent.effort!r} "
                     "through the gateway from inside an agent container", passed, evidence)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--api-auth-check", action="store_true")
    parser.add_argument("--skip-evaluator", action="store_true")
    parser.add_argument("--skip-capture", action="store_true")
    args = parser.parse_args()
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        if os.environ.get(key):
            print(f"refusing to run with {key} set: HELIX would forward it into agent containers", file=sys.stderr)
            return 2
    run_dir = args.run_dir.resolve()
    config = load_config(run_dir / ".helix-run" / "helix.toml")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = run_dir / ".helix-run" / "isolation" / ("preflight-" + stamp)
    out_dir.mkdir(parents=True, exist_ok=False)
    out = args.out or out_dir / f"isolation-{stamp}.json"
    report = Report()
    check_run_dir(run_dir, config, report)
    check_infrastructure(config, report)
    probe = run_agent_probe(config, run_dir, report, "", "B")
    assess_agent_probe(probe, report, with_sidecar=False)
    if not args.skip_evaluator:
        seed = check_evaluator(config, run_dir, report, out_dir)
        if seed is not None:
            production_prompt(config, run_dir, report, out_dir)
    symlink_probe(config, run_dir, report)
    if not args.skip_capture:
        capture_agent_requests(config, run_dir, report, out_dir)
    if args.api_auth_check:
        api_auth_check(config, run_dir, report)
    out.write_text(json.dumps({"utc": stamp, "run_dir": str(run_dir), "passed": report.ok,
                               "checks": report.checks}, indent=2))
    print(f"{'ALL PASSED' if report.ok else 'SOME CHECKS FAILED'}; report: {out}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
