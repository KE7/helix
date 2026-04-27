"""HELIX command-line interface."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, NoReturn

import click
from rich.table import Table
from rich.tree import Tree

from helix import __version__
from helix.backends import BACKENDS
from helix.config import load_config
from helix.logging_config import setup_file_logging
from helix.sandbox import run_sandbox_auth_command, sandbox_auth_volume_name
from helix.display import (
    console,
    print_error,
    print_info,
    print_success,
    print_warning,
    render_frontier_table,
)
from helix.exceptions import RateLimitError
from helix.lineage import load_lineage
from helix.population import EvalResult, FrontierType, ParetoFrontier, Candidate
from helix.state import load_state, save_state
from helix.worktree import remove_worktree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HELIX_TOML_TEMPLATE = """\
# helix.toml — HELIX evolutionary configuration
# Run `helix evolve` to start optimising.

# HELIX evolves your whole repo. Each mutation may read, edit, create, or delete
# any file in the project tree. There is no target_file setting.
#
# To restrict what the mutation backend touches, describe constraints in
# agent.background.

# Describe what you want the code to do better.
objective = "Describe the optimisation objective"

[evaluator]
# Command to evaluate a candidate (run from worktree root).
# Use the SAME environment your evaluator dependencies live in, e.g.
#   command = "uv run python evaluate.py"
# or
#   command = "bash run_eval.sh"
# Avoid bare `python3 evaluate.py` unless that interpreter already has every
# evaluator dependency installed.
command = "uv run python evaluate.py"
# score_parser = "pytest"   # or "exitcode", "json_accuracy", "json_score"
# protected_files = ["evaluate.py"]  # optional extra evaluator-immutable files

[evolution]
max_generations = 20
merge_enabled = false

[agent]
backend = "claude"
# model = "sonnet"  # optional backend-specific model
max_turns = 20
# background = "Only modify files under src/. Do not touch tests/ or config/."

[sandbox]
# Recommended for new projects: set enabled=true after Docker is running and
# after `helix sandbox login <backend>`. HELIX keeps this opt-in so machines
# without Docker can still use local workflows.
enabled = false
# image = "ghcr.io/ke7/helix-evo-runner-claude:latest"  # optional; defaults from agent.backend
# network = "bridge"
# skip_special_files = true  # skip FIFOs/sockets/devices during workspace sync
# Agent containers mount a persistent Docker auth volume named
# helix-auth-<backend>. Run `helix sandbox login <backend>` once per backend.
"""

_HELIX_DIR = ".helix"


def _helix_dir(project_root: Path) -> Path:
    return project_root / _HELIX_DIR


def _print_cleanup_hint() -> None:
    print_info(
        "HELIX worktrees and saved state remain on disk after the run. "
        "Run [cyan]helix clean[/cyan] when you want to discard them."
    )


def _handle_keyboard_interrupt(project_root: Path) -> NoReturn:
    state = load_state(project_root)
    if state is None:
        print_warning(
            "Interrupted before HELIX saved resumable state. "
            "You may need to run [cyan]helix clean[/cyan] if worktrees were created."
        )
    else:
        print_warning(
            "Interrupted. Evolution state has been preserved. "
            "Run [cyan]helix resume[/cyan] to continue."
        )
    _print_cleanup_hint()
    raise SystemExit(130)


def _create_initial_gitignore(project_root: Path) -> None:
    """Create a .gitignore with common noise patterns if it doesn't exist."""
    gitignore_path = project_root / ".gitignore"

    # Common noise patterns to exclude
    noise_patterns = [
        "# Python",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        ".pytest_cache/",
        ".mypy_cache/",
        ".hypothesis/",
        "*.egg-info/",
        "",
        "# Build artifacts",
        "build/",
        "dist/",
        "",
        "# Coverage",
        ".coverage",
        "htmlcov/",
        "",
        "# Environment",
        ".env",
        ".venv/",
        "venv/",
        "",
        "# Other",
        "node_modules/",
        ".DS_Store",
    ]

    if not gitignore_path.exists():
        gitignore_path.write_text("\n".join(noise_patterns) + "\n")


def _ensure_git_repo(project_root: Path) -> None:
    """Initialise git repo if not already present."""
    result = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print_info("No git repository found — initialising one…")
        subprocess.run(["git", "init"], cwd=project_root, check=True)
        # Create .gitignore before git add to filter noise files
        _create_initial_gitignore(project_root)
        subprocess.run(["git", "add", "-A"], cwd=project_root, check=True)
        subprocess.run(
            ["git", "commit", "-m", "helix: initial seed"],
            cwd=project_root,
            check=True,
        )
        print_success("Initialised git repository and committed initial seed.")


def _update_gitignore(project_root: Path) -> None:
    """Ensure ``.helix/`` is listed in .gitignore (creates file if absent)."""
    gitignore = project_root / ".gitignore"
    entry = ".helix/"

    if gitignore.exists():
        content = gitignore.read_text()
        # Check each line to avoid duplicates
        if entry in content.splitlines():
            return
        # Append with a newline separator if file doesn't end with one
        if content and not content.endswith("\n"):
            content += "\n"
        gitignore.write_text(content + entry + "\n")
    else:
        gitignore.write_text(entry + "\n")


def _load_all_evaluations(base_dir: Path) -> tuple[dict[str, EvalResult], dict[str, str]]:
    """Load saved evaluation results and parse errors from base_dir/evaluations/."""
    results: dict[str, EvalResult] = {}
    errors: dict[str, str] = {}
    eval_dir = base_dir / "evaluations"
    if not eval_dir.exists():
        return results, errors
    for path in eval_dir.glob("*.json"):
        cid = path.stem
        try:
            data = json.loads(path.read_text())
            cid = data["candidate_id"]
            results[cid] = EvalResult(
                candidate_id=cid,
                scores=data.get("scores", {}),
                instance_scores=data.get("instance_scores", {}),
                asi=data.get("asi", {}),
            )
        except Exception as exc:
            errors[cid] = f"evaluation file unreadable: {exc}"
    return results, errors


def _reconstruct_frontier(
    base_dir: Path, state: Any
) -> tuple[ParetoFrontier, dict[str, Candidate], list[str]]:
    """Rebuild in-memory ``ParetoFrontier`` from persisted state + evaluations.

    The frontier is constructed with ``state.frontier_type`` rather
    than the class default ``"instance"`` — so read-only display
    commands (``helix frontier``, ``helix best``) show the frontier
    with the SAME axis the evolution run used.  Legacy states (no
    persisted ``frontier_type``) default to ``"instance"`` via
    ``EvolutionState.frontier_type``'s field default, preserving the
    pre-multi-axis display behaviour.
    """
    # ``frontier_type`` is part of ``EvolutionState`` after commit Q;
    # getattr + whitelist narrowing keeps us defensive against an
    # ad-hoc test that passes a mock state without the attribute
    # (or a hand-crafted state.json with a bogus literal).
    _raw_ft = getattr(state, "frontier_type", "instance")
    _valid: tuple[FrontierType, ...] = (
        "instance", "objective", "hybrid", "cartesian",
    )
    frontier_type: FrontierType = _raw_ft if _raw_ft in _valid else "instance"
    frontier = ParetoFrontier(frontier_type=frontier_type)
    candidates: dict[str, Candidate] = {}
    skipped: list[str] = []
    worktrees_dir = base_dir / "worktrees"
    all_results, eval_errors = _load_all_evaluations(base_dir)

    for cid in state.frontier:
        result = all_results.get(cid)
        wt_path = worktrees_dir / cid
        if not wt_path.exists():
            skipped.append(f"{cid}: missing worktree")
            continue
        if result is None:
            skipped.append(f"{cid}: {eval_errors.get(cid, 'missing evaluation result')}")
            continue
        try:
            gen = int(cid.split("-")[0].lstrip("g"))
        except (IndexError, ValueError):
            gen = 0
        cand = Candidate(
            id=cid,
            worktree_path=str(wt_path),
            branch_name=f"helix/{cid}",
            generation=gen,
            parent_id=None,
            parent_ids=[],
            operation="restored",
        )
        candidates[cid] = cand
        frontier.add(cand, result)

    return frontier, candidates, skipped


def _load_log_entries(base_dir: Path) -> list[dict[str, Any]]:
    """Load all log entries from base_dir/log/*.json, sorted by generation."""
    log_dir = base_dir / "log"
    if not log_dir.exists():
        return []
    entries: list[dict[str, Any]] = []
    for path in log_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            entries.append(data)
        except Exception:
            pass
    entries.sort(key=lambda e: (e.get("generation", 0), e.get("candidate_id", "")))
    return entries


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version=__version__, prog_name="helix")
def cli() -> None:
    """HELIX: Hierarchical Evolution via LLM-Informed eXploration."""


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@cli.command()
def init() -> None:
    """Initialise HELIX in the current directory."""
    project_root = Path.cwd()
    _ensure_git_repo(project_root)
    _update_gitignore(project_root)

    base_dir = _helix_dir(project_root)
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "worktrees").mkdir(exist_ok=True)
    (base_dir / "evaluations").mkdir(exist_ok=True)

    toml_path = project_root / "helix.toml"
    if toml_path.exists():
        print_warning("helix.toml already exists — not overwriting.")
    else:
        toml_path.write_text(_HELIX_TOML_TEMPLATE)
        print_success("Created helix.toml — edit it before running `helix evolve`.")

    print_success("HELIX initialised.")
    console.print(
        "\n[bold]Next steps:[/bold]\n"
        "  1. Edit [cyan]helix.toml[/cyan] — set your objective and evaluator command.\n"
        "     Make sure [cyan]evaluator.command[/cyan] uses the same environment as your\n"
        "     evaluator dependencies (for example [cyan]uv run python evaluate.py[/cyan]\n"
        "     or [cyan]bash run_eval.sh[/cyan], not whichever [cyan]python3[/cyan] happens\n"
        "     to be on PATH).\n"
        "  2. Run [cyan]helix evolve[/cyan] to start evolution.\n"
    )


# ---------------------------------------------------------------------------
# sandbox
# ---------------------------------------------------------------------------


@cli.group(name="sandbox")
def sandbox_cli() -> None:
    """Manage HELIX Docker sandbox helpers."""


@sandbox_cli.command(name="login")
@click.argument("backend", type=click.Choice(BACKENDS))
@click.option("--image", default=None, help="Override the backend runner image.")
@click.option("--network", default="bridge", show_default=True, help="Docker network mode.")
@click.option("--add-host-gateway", is_flag=True, default=False,
              help="Add host.docker.internal for Linux Docker hosts.")
def sandbox_login(
    backend: str,
    image: str | None,
    network: str,
    add_host_gateway: bool,
) -> None:
    """Log into a backend inside its persistent sandbox auth volume."""
    volume = sandbox_auth_volume_name(backend)
    print_info(f"Using Docker auth volume [cyan]{volume}[/cyan].")
    if backend == "gemini":
        print_warning(
            "Gemini CLI does not expose a dedicated login subcommand; HELIX "
            "starts the interactive Gemini CLI so you can complete its auth flow."
        )
    result = run_sandbox_auth_command(
        backend,
        action="login",
        image=image,
        network=network,
        add_host_gateway=add_host_gateway,
        interactive=True,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


@sandbox_cli.command(name="status")
@click.argument("backend", required=False, type=click.Choice(BACKENDS))
@click.option("--image", default=None, help="Override the backend runner image.")
@click.option("--network", default="bridge", show_default=True, help="Docker network mode.")
@click.option("--add-host-gateway", is_flag=True, default=False,
              help="Add host.docker.internal for Linux Docker hosts.")
def sandbox_status(
    backend: str | None,
    image: str | None,
    network: str,
    add_host_gateway: bool,
) -> None:
    """Show login status for one backend or all backends."""
    backends = [backend] if backend is not None else BACKENDS
    exit_code = 0
    for item in backends:
        console.print(f"[bold]{item}[/bold] ({sandbox_auth_volume_name(item)})")
        result = run_sandbox_auth_command(
            item,
            action="status",
            image=image,
            network=network,
            add_host_gateway=add_host_gateway,
        )
        output = (result.stdout or "").strip()
        error = (result.stderr or "").strip()
        if output:
            console.print(output)
        if error:
            console.print(error, style="red")
        if result.returncode != 0:
            exit_code = result.returncode
    if exit_code:
        raise SystemExit(exit_code)


@sandbox_cli.command(name="logout")
@click.argument("backend", type=click.Choice(BACKENDS))
@click.option("--image", default=None, help="Override the backend runner image.")
@click.option("--network", default="bridge", show_default=True, help="Docker network mode.")
@click.option("--add-host-gateway", is_flag=True, default=False,
              help="Add host.docker.internal for Linux Docker hosts.")
def sandbox_logout(
    backend: str,
    image: str | None,
    network: str,
    add_host_gateway: bool,
) -> None:
    """Log out a backend from its persistent sandbox auth volume."""
    result = run_sandbox_auth_command(
        backend,
        action="logout",
        image=image,
        network=network,
        add_host_gateway=add_host_gateway,
    )
    if result.stdout:
        console.print(result.stdout.strip())
    if result.stderr:
        console.print(result.stderr.strip(), style="red")
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print_success(f"Logged out sandbox backend {backend!r}.")


# ---------------------------------------------------------------------------
# evolve
# ---------------------------------------------------------------------------


@cli.command(
    help=(
        "Run the HELIX evolutionary loop on the current project. "
        "HELIX evolves your WHOLE REPO — the configured agent backend may read, edit, create, or delete "
        "any file in the project tree in each mutation. "
        "There is no target_file; the candidate is your entire working tree at HEAD.\n\n"
        "Requires a helix.toml file in the project directory. "
        "Run `helix init` first to create one, or see README for the minimal schema."
    ),
)
@click.option("--config", "config_path", default="helix.toml", show_default=True,
              help="Path to helix.toml config file.")
@click.option("--dir", "project_dir", default=None,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Project root directory (defaults to current working directory).")
@click.option("--objective", default=None, help="Override the objective string.")
@click.option("--evaluator", default=None, help="Override the evaluator command.")
@click.option("--generations", default=None, type=int,
              help="Override max_generations.")
@click.option("--no-merge", "no_merge", is_flag=True, default=False,
              help="Disable merge operations.")
@click.option("--backend", default=None,
              type=click.Choice(["claude", "codex", "cursor", "gemini", "opencode"]),
              help="Override the mutation backend.")
@click.option("--model", default=None,
              help="Override the backend model.")
@click.option("--effort", default=None,
              help="Override backend effort / reasoning level when supported.")
def evolve(
    config_path: str,
    project_dir: Path | None,
    objective: str | None,
    evaluator: str | None,
    generations: int | None,
    no_merge: bool,
    backend: str | None,
    model: str | None,
    effort: str | None,
) -> None:
    from helix.evolution import run_evolution

    project_root = project_dir if project_dir is not None else Path.cwd()

    # Fix 2: ensure .helix/ is gitignored before anything else.
    # Git repo creation is handled by create_seed_worktree → worktree._ensure_git_repo.
    _update_gitignore(project_root)

    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = project_root / cfg_file

    try:
        config = load_config(cfg_file)
    except FileNotFoundError:
        # Fix 5b: clear actionable error message when helix.toml is missing.
        print_error(
            f"❌ No helix.toml found in {project_root}\n"
            "Run [cyan]helix init[/cyan] first, or cd into a directory that contains one."
        )
        raise SystemExit(1)

    # Apply CLI overrides
    if objective:
        config = config.model_copy(update={"objective": objective})
    if evaluator:
        config = config.model_copy(
            update={"evaluator": config.evaluator.model_copy(update={"command": evaluator})}
        )
    if generations is not None:
        config = config.model_copy(
            update={
                "evolution": config.evolution.model_copy(
                    update={"max_generations": generations}
                )
            }
        )
    if no_merge:
        config = config.model_copy(
            update={
                "evolution": config.evolution.model_copy(update={"merge_enabled": False})
            }
        )
    if backend is not None or model is not None or effort is not None:
        agent_updates: dict[str, Any] = {}
        if backend is not None:
            agent_updates["backend"] = backend
        if model is not None:
            agent_updates["model"] = model
        if effort is not None:
            agent_updates["effort"] = effort
        config = config.model_copy(
            update={"agent": config.agent.model_copy(update=agent_updates)}
        )

    base_dir = _helix_dir(project_root)
    setup_file_logging(base_dir)
    try:
        run_evolution(config, project_root, base_dir)
    except RateLimitError as exc:
        # Rate limits exhausted retries and bubbled all the way to the CLI.
        # State has been saved during the run; tell the user how to resume.
        logger.error("Rate limit reached during evolution: %s", exc)
        print_error(
            f"Rate limit reached: {exc}\n"
            "Evolution state has been saved. "
            "Run [cyan]helix resume[/cyan] to continue when rate limits clear."
        )
        raise SystemExit(2)
    except KeyboardInterrupt:
        _handle_keyboard_interrupt(project_root)
    else:
        _print_cleanup_hint()


# ---------------------------------------------------------------------------
# frontier
# ---------------------------------------------------------------------------


@cli.command("frontier")
@click.option("--dir", "project_dir", default=None,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Project root directory (defaults to current working directory).")
def frontier_cmd(project_dir: Path | None) -> None:
    """Show the current Pareto frontier."""
    project_root = project_dir if project_dir is not None else Path.cwd()
    base_dir = _helix_dir(project_root)

    state = load_state(project_root)
    if state is None:
        print_warning("No evolution state found. Run `helix evolve` first.")
        return

    all_results, _ = _load_all_evaluations(base_dir)
    frontier, _, skipped = _reconstruct_frontier(base_dir, state)
    for skipped_entry in skipped:
        print_warning(f"Skipping frontier entry: {skipped_entry}")
    render_frontier_table(frontier, all_results)


# ---------------------------------------------------------------------------
# best
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--dir", "project_dir", default=None,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Project root directory (defaults to current working directory).")
@click.option("--export", "export_path", default=None, type=click.Path(),
              help="Export best candidate to this path.")
def best(project_dir: Path | None, export_path: str | None) -> None:
    """Show the best candidate and optionally export it."""
    project_root = project_dir if project_dir is not None else Path.cwd()
    base_dir = _helix_dir(project_root)

    state = load_state(project_root)
    if state is None:
        print_warning("No evolution state found. Run `helix evolve` first.")
        return

    frontier, candidates, skipped = _reconstruct_frontier(base_dir, state)
    for skipped_entry in skipped:
        print_warning(f"Skipping frontier entry: {skipped_entry}")
    if not frontier._candidates:
        print_warning("Frontier is empty.")
        return

    best_cand = frontier.best()
    best_result = frontier._results.get(best_cand.id)

    console.print(f"\n[bold green]Best candidate:[/bold green] [cyan]{best_cand.id}[/cyan]")
    if best_result:
        console.print(f"  Aggregate score: {best_result.aggregate_score():.4f}")
        for k, v in sorted(best_result.scores.items()):
            console.print(f"  {k}: {v:.4f}")

    if export_path:
        dest = Path(export_path)
        if dest.exists():
            print_error(f"Export path already exists: {dest}")
            raise SystemExit(1)
        shutil.copytree(best_cand.worktree_path, str(dest))
        print_success(f"Exported best candidate to: {dest}")


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--dir", "project_dir", default=None,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Project root directory (defaults to current working directory).")
def history(project_dir: Path | None) -> None:
    """Show the candidate lineage as a tree."""
    project_root = project_dir if project_dir is not None else Path.cwd()
    base_dir = _helix_dir(project_root)
    lineage_path = base_dir / "lineage.json"

    lineage = load_lineage(lineage_path)
    if not lineage:
        print_warning("No lineage data found. Run `helix evolve` first.")
        return

    # Build tree starting from roots (entries with no parent)
    tree = Tree("[bold blue]HELIX lineage[/bold blue]")
    nodes: dict[str, Tree] = {}

    # Process roots first
    for entry in lineage.values():
        if entry.parent is None:
            node = tree.add(f"[cyan]{entry.id}[/cyan]  gen={entry.generation}  op={entry.operation}")
            nodes[entry.id] = node

    # Iteratively attach children (handle arbitrary ordering)
    remaining = [e for e in lineage.values() if e.parent is not None]
    max_iterations = len(remaining) + 1
    iteration = 0
    while remaining and iteration < max_iterations:
        iteration += 1
        still_remaining = []
        for entry in remaining:
            parent_node = nodes.get(entry.parent)  # type: ignore[arg-type]
            if parent_node is not None:
                node = parent_node.add(
                    f"[cyan]{entry.id}[/cyan]  gen={entry.generation}  op={entry.operation}"
                )
                nodes[entry.id] = node
            else:
                still_remaining.append(entry)
        remaining = still_remaining

    # Attach any orphans to the root tree directly
    for entry in remaining:
        node = tree.add(
            f"[cyan]{entry.id}[/cyan]  gen={entry.generation}  op={entry.operation}  "
            f"[dim](orphan)[/dim]"
        )
        nodes[entry.id] = node

    console.print(tree)


# ---------------------------------------------------------------------------
# log
# ---------------------------------------------------------------------------


@cli.command("log")
@click.option("--dir", "project_dir", default=None,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Project root directory (defaults to current working directory).")
def log_cmd(project_dir: Path | None) -> None:
    """Show the evolution trajectory: per-generation parent lineage and mutation log."""
    project_root = project_dir if project_dir is not None else Path.cwd()
    base_dir = _helix_dir(project_root)
    setup_file_logging(base_dir)

    entries = _load_log_entries(base_dir)
    if not entries:
        print_warning("No log entries found. Run `helix evolve` first.")
        return

    table = Table(title="Evolution Log", show_lines=True)
    table.add_column("Gen", style="bold cyan", justify="right")
    table.add_column("Candidate", style="cyan")
    table.add_column("Operation", style="yellow")
    table.add_column("Timestamp")
    table.add_column("Summary")

    for entry in entries:
        summary = entry.get("summary", {})
        summary_str = (
            ", ".join(f"{k}={v}" for k, v in summary.items())
            if summary
            else ""
        )
        table.add_row(
            str(entry.get("generation", "")),
            str(entry.get("candidate_id", "")),
            str(entry.get("operation", "")),
            str(entry.get("timestamp", "")),
            summary_str,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--config", "config_path", default="helix.toml", show_default=True,
              help="Path to helix.toml config file.")
@click.option("--dir", "project_dir", default=None,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Project root directory (defaults to current working directory).")
def resume(config_path: str, project_dir: Path | None) -> None:
    """Resume a previously started evolution run."""
    from helix.evolution import run_evolution

    project_root = project_dir if project_dir is not None else Path.cwd()

    # Fix 2: ensure .helix/ is gitignored (idempotent, safe to call on resume too).
    _update_gitignore(project_root)

    base_dir = _helix_dir(project_root)
    setup_file_logging(base_dir)

    state = load_state(project_root)
    if state is None:
        print_warning("No evolution state found. Starting a fresh run instead.")
    else:
        # Check for missing worktrees and drop them from the frontier
        worktrees_dir = base_dir / "worktrees"
        missing: list[str] = []
        for cid in state.frontier:
            wt_path = worktrees_dir / cid
            if not wt_path.exists():
                print_warning(
                    f"Worktree missing for candidate {cid!r} — dropping from frontier."
                )
                missing.append(cid)

        if missing:
            state.frontier = [cid for cid in state.frontier if cid not in missing]
            save_state(state, project_root)

    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = project_root / cfg_file

    try:
        config = load_config(cfg_file)
    except FileNotFoundError:
        print_error(f"Config file not found: {cfg_file}")
        raise SystemExit(1)

    print_info(f"Resuming from generation {state.generation if state else 0}…")
    try:
        run_evolution(config, project_root, base_dir)
    except KeyboardInterrupt:
        _handle_keyboard_interrupt(project_root)
    else:
        _print_cleanup_hint()


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--dir", "project_dir", default=None,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Project root directory (defaults to current working directory).")
def clean(project_dir: Path | None) -> None:
    """Remove all HELIX worktrees and delete the .helix/ directory."""
    project_root = project_dir if project_dir is not None else Path.cwd()
    base_dir = _helix_dir(project_root)
    worktrees_dir = base_dir / "worktrees"
    base_dir_exists = base_dir.exists()

    if base_dir_exists:
        click.confirm(
            "This will remove all worktrees and delete .helix/. Continue?",
            abort=True,
        )
    else:
        print_warning(
            ".helix/ directory not found — pruning stale HELIX git state only."
        )

    # Collect all candidate IDs from state + any orphaned worktree directories
    candidate_ids: list[str] = []
    if base_dir_exists:
        state = load_state(project_root)
        if state is not None:
            candidate_ids.extend(state.frontier)

    # Also scan worktrees directory for any directories not in state
    if base_dir_exists and worktrees_dir.exists():
        for wt_path in worktrees_dir.iterdir():
            if wt_path.is_dir() and wt_path.name not in candidate_ids:
                candidate_ids.append(wt_path.name)

    for cid in candidate_ids:
        wt_path = worktrees_dir / cid
        if wt_path.exists():
            try:
                gen = int(cid.split("-")[0].lstrip("g"))
            except (IndexError, ValueError):
                gen = 0
            cand = Candidate(
                id=cid,
                worktree_path=str(wt_path),
                branch_name=f"helix/{cid}",
                generation=gen,
                parent_id=None,
                parent_ids=[],
                operation="restored",
            )
            try:
                remove_worktree(cand)
                print_info(f"Removed worktree: {cid}")
            except Exception as exc:
                print_warning(f"Could not remove worktree {cid}: {exc}")

    prune = subprocess.run(
        ["git", "worktree", "prune"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if prune.returncode != 0 and prune.stderr.strip():
        print_warning(f"Could not prune git worktrees: {prune.stderr.strip()}")

    branch_result = subprocess.run(
        ["git", "branch", "--list", "helix/*"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if branch_result.returncode == 0:
        branches = [
            b.strip().lstrip("* ").strip()
            for b in branch_result.stdout.splitlines()
            if b.strip()
        ]
        for branch in branches:
            delete = subprocess.run(
                ["git", "branch", "-D", branch],
                cwd=project_root,
                check=False,
                capture_output=True,
                text=True,
            )
            if delete.returncode == 0:
                print_info(f"Removed branch: {branch}")
            elif delete.stderr.strip():
                print_warning(
                    f"Could not remove stale branch {branch}: {delete.stderr.strip()}"
                )

    if base_dir_exists:
        shutil.rmtree(base_dir, ignore_errors=True)
        print_success("Cleaned up .helix/ directory.")
    elif branch_result.returncode == 0 and branch_result.stdout.strip():
        print_success("Pruned stale HELIX git state.")
