"""HELIX display utilities using Rich."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table

if TYPE_CHECKING:
    from helix.population import EvalResult, ParetoFrontier
    from helix.state import BudgetState
    from helix.config import EvolutionConfig


class HelixPhase(Enum):
    """Enumeration of evolution phases for progress display.

    Each phase corresponds to a major step in the HELIX evolution loop
    and is displayed to the user via set_phase().
    """
    SEED_GENERATION = "Generating seed candidate"
    SEED_EVAL = "Evaluating seed"
    TRAIN_EVALUATION = "Running train evaluation"
    VAL_EVALUATION = "Running validation evaluation"
    MUTATION = "Applying mutation"
    MUTATION_GATING = "Gating mutation"
    MERGE = "Merging candidates"
    PARETO_UPDATE = "Updating Pareto frontier"
    CLEANUP = "Cleaning up dominated candidates"


# Module-level console that persists across calls
console = Console(highlight=False)


def set_phase(phase: HelixPhase) -> None:
    """Update the displayed phase using a Rich spinner-style status line."""
    console.print(f"[bold dim]⟳  {phase.value}…[/bold dim]")


def render_generation(
    gen: int,
    total: int,
    frontier: "ParetoFrontier",
    result: "EvalResult | None",
    *,
    mutations_attempted: int = 0,
    mutations_accepted: int = 0,
) -> None:
    """Print a Rich Panel showing generation summary.

    Displays the current generation number, candidate scores, acceptance
    status, and the current Pareto frontier composition.
    """
    lines: list[str] = [f"[bold]Generation {gen} / {total}[/bold]"]

    if result is not None:
        cid = result.candidate_id
        agg = result.aggregate_score()

        # Try to find parent info
        parent_id: str | None = None
        if cid in frontier._candidates:
            parent_id = frontier._candidates[cid].parent_id

        if parent_id and parent_id in frontier._results:
            p_agg = frontier._results[parent_id].aggregate_score()
            lines.append(f"  Parent : [cyan]{parent_id}[/cyan]")
            lines.append(f"  Score  : {p_agg:.4f} → {agg:.4f}")
        else:
            lines.append(f"  Score  : {agg:.4f}")

        for k, v in sorted(result.scores.items()):
            lines.append(f"  {k}: {v:.4f}")

        # Acceptance: candidate is accepted when it is (or was added to) the frontier
        accepted = cid in frontier._candidates
        status_str = (
            "[green bold]ACCEPTED[/green bold]"
            if accepted
            else "[red bold]REJECTED[/red bold]"
        )
        lines.append(f"  Status : {status_str}")

    # Current frontier summary
    if frontier._candidates:
        lines.append("")
        lines.append("[bold]Current frontier:[/bold]")
        for cid, _cand in frontier._candidates.items():
            r = frontier._results.get(cid)
            score_str = f"{r.aggregate_score():.4f}" if r else "?"
            lines.append(f"  [cyan]{cid}[/cyan]  {score_str}")

    panel = Panel(
        "\n".join(lines),
        title=f"[bold blue]Generation {gen}[/bold blue]",
        border_style="blue",
    )
    console.print(panel)


def render_frontier_table(
    frontier: "ParetoFrontier",
    results: "dict[str, EvalResult]",
) -> None:
    """Render a Rich Table with columns: Candidate | Scores | Instance Wins | Generation | Operation."""
    table = Table(
        title="Pareto Frontier",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Candidate", style="cyan", no_wrap=True)
    table.add_column("Scores")
    table.add_column("Instance Wins", justify="right")
    table.add_column("Generation", justify="right")
    table.add_column("Operation")

    wins = frontier._instance_wins() if frontier._candidates else {}

    for cid, cand in frontier._candidates.items():
        result = results.get(cid)
        if result:
            score_parts = [f"agg={result.aggregate_score():.4f}"]
            for k, v in sorted(result.scores.items()):
                score_parts.append(f"{k}={v:.4f}")
            scores_str = "\n".join(score_parts)
        else:
            scores_str = "(no data)"

        win_count = str(len(wins.get(cid, set())))
        table.add_row(cid, scores_str, win_count, str(cand.generation), cand.operation)

    console.print(table)


def render_budget(budget: "BudgetState", config_evolution: "EvolutionConfig") -> None:
    """Render Rich Progress bars for evaluation and Claude call budgets."""
    with Progress(
        TextColumn("[bold blue]{task.description:<20}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        console=console,
        transient=False,
    ) as progress:
        progress.add_task(
            "Evaluations",
            total=config_evolution.max_metric_calls,
            completed=budget.evaluations,
        )


def print_info(msg: str) -> None:
    """Print an informational message in cyan."""
    console.print(f"[cyan]{msg}[/cyan]")


def print_success(msg: str) -> None:
    """Print a success message in bold green."""
    console.print(f"[green bold]{msg}[/green bold]")


def print_warning(msg: str) -> None:
    """Print a warning message in yellow."""
    console.print(f"[yellow]{msg}[/yellow]")


def print_error(msg: str) -> None:
    """Print an error message in bold red."""
    console.print(f"[red bold]{msg}[/red bold]")
