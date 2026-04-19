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
from helix.config import load_config, HelixConfig
from helix.logging_config import setup_file_logging
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
from helix.population import EvalResult, ParetoFrontier, Candidate
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
# To restrict what Claude Code touches, describe constraints in claude.background.

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

[claude]
model = "sonnet"
max_turns = 20
# background = "Only modify files under src/. Do not touch tests/ or config/."
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
    """Rebuild in-memory ParetoFrontier from persisted state and evaluations."""
    frontier = ParetoFrontier()
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
# evolve
# ---------------------------------------------------------------------------


@cli.command(
    help=(
        "Run the HELIX evolutionary loop on the current project. "
        "HELIX evolves your WHOLE REPO — Claude Code may read, edit, create, or delete "
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
@click.option("--model", default=None,
              help="Override the Claude model (e.g. claude-haiku-4-5-20250514).")
@click.option("--effort", default=None,
              help="Override Claude effort level (e.g. low, medium, high, xhigh, max).")
def evolve(
    config_path: str,
    project_dir: Path | None,
    objective: str | None,
    evaluator: str | None,
    generations: int | None,
    no_merge: bool,
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
    if model is not None or effort is not None:
        claude_updates: dict[str, Any] = {}
        if model is not None:
            claude_updates["model"] = model
        if effort is not None:
            claude_updates["effort"] = effort
        config = config.model_copy(
            update={"claude": config.claude.model_copy(update=claude_updates)}
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
