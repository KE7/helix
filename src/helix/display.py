"""HELIX display utilities using Rich."""

from __future__ import annotations

from enum import Enum
import threading
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table

if TYPE_CHECKING:
    from helix.population import EvalResult, ParetoFrontier
    from helix.state import BudgetState
    from helix.config import EvolutionConfig


from dataclasses import dataclass, field
from rich.live import Live


@dataclass
class UsageStats:
    """Tracks resource usage for a single candidate or generation."""

    input_tokens: int = 0
    output_tokens: int = 0
    num_turns: int = 0
    tool_event_count: int = 0
    tool_names: list[str] = field(default_factory=list)
    cost_usd: float = 0.0

    def add(self, other: UsageStats | dict[str, Any]) -> None:
        if isinstance(other, dict):
            self.input_tokens += int(other.get("input_tokens", 0))
            self.output_tokens += int(other.get("output_tokens", 0))
            self.num_turns += int(other.get("num_turns", 0))
            self.tool_event_count += int(other.get("tool_event_count", 0))
            self.cost_usd += float(other.get("cost_usd", 0.0))
            new_tools = other.get("tool_names", [])
            if isinstance(new_tools, list):
                for t in new_tools:
                    self.tool_names.append(t)
        else:
            self.input_tokens += other.input_tokens
            self.output_tokens += other.output_tokens
            self.num_turns += other.num_turns
            self.tool_event_count += other.tool_event_count
            self.cost_usd += other.cost_usd
            for t in other.tool_names:
                self.tool_names.append(t)


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


def render_status_panel(
    gen: int,
    total: int,
    phase: HelixPhase | str,
    current_usage: UsageStats,
    cumulative_budget: "BudgetState",
    frontier: "ParetoFrontier | None" = None,
    config_evolution: "EvolutionConfig | None" = None,
    mutations_attempted: int = 0,
    mutations_accepted: int = 0,
) -> Panel:
    """Render a live status panel with current progress and usage stats."""
    phase_str = phase.value if isinstance(phase, HelixPhase) else str(phase)
    lines = [
        f"[bold]Generation {gen} / {total}[/bold]",
        f"Status: [cyan]{phase_str}[/cyan]",
        "",
    ]

    # Evolutionary Progress
    if frontier:
        # Clone dictionaries safely to avoid RuntimeError if changed by a worker thread
        frontier_candidates = {}
        frontier_results = {}
        for _ in range(5):
            try:
                frontier_candidates = dict(frontier._candidates)
                frontier_results = dict(frontier._results)
                break
            except RuntimeError:
                pass

        if frontier_candidates:
            try:
                best_cand = frontier.best()
                r = frontier_results.get(best_cand.id)
                score = r.aggregate_score() if r is not None else 0.0
                lines.append(
                    f"Best Score: [bold green]{score:.4f}[/bold green] ({best_cand.id})"
                )
            except (ValueError, KeyError, RuntimeError):
                pass

            f_ids = list(frontier_candidates.keys())
            if len(f_ids) > 1:
                lines.append(
                    f"Frontier  : [dim]{', '.join(f_ids[:5])}{'...' if len(f_ids) > 5 else ''}[/dim]"
                )

            if mutations_attempted > 0:
                rate = (mutations_accepted / mutations_attempted) * 100
                lines.append(
                    f"Acceptance: {mutations_accepted}/{mutations_attempted} ({rate:.1f}%)"
                )
            lines.append("")

    # Current Generation Usage
    lines.extend(
        [
            "[bold]Current Generation Usage:[/bold]",
            f"  Tokens: {current_usage.input_tokens:,} in / {current_usage.output_tokens:,} out",
            f"  Turns : {current_usage.num_turns}",
        ]
    )

    tools_str = f"{current_usage.tool_event_count}"
    if current_usage.tool_names:
        from collections import Counter

        counts = Counter(current_usage.tool_names)
        tool_details = ", ".join(
            f"{name} x {count}" for name, count in counts.most_common()
        )
        tools_str += f" ({tool_details})"
    lines.append(f"  Tools : {tools_str}")

    # Cumulative Usage & Budget
    lines.extend(
        [
            "",
            "[bold]Cumulative Evolution Total:[/bold]",
            f"  Tokens: {cumulative_budget.input_tokens:,} in / {cumulative_budget.output_tokens:,} out",
            f"  Cost  : [green]${cumulative_budget.cost_usd:.4f}[/green]",
        ]
    )

    if config_evolution:
        cap = config_evolution.max_evaluations
        if cap > 0:
            pct = min(100, (cumulative_budget.evaluations / cap) * 100)
            bar_width = 20
            filled = int(bar_width * pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            lines.append(
                f"  Budget: |{bar}| {cumulative_budget.evaluations}/{cap} evals"
            )
        else:
            lines.append(f"  Budget: {cumulative_budget.evaluations} evaluations")

    return Panel(
        "\n".join(lines),
        title="[bold blue]HELIX Evolution[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    )


_local_display = threading.local()


def _get_active_live() -> HelixLiveDisplay | None:
    return getattr(_local_display, "active_live", None)


def _set_active_live(live: HelixLiveDisplay | None) -> None:
    _local_display.active_live = live


class HelixLiveDisplay:
    """Context manager for the HELIX live status display."""

    def __init__(
        self,
        gen: int,
        total: int,
        cumulative_budget: "BudgetState",
        frontier: "ParetoFrontier | None" = None,
        config_evolution: "EvolutionConfig | None" = None,
    ):
        self.gen = gen
        self.total = total
        self.cumulative_budget = cumulative_budget
        self.frontier = frontier
        self.config_evolution = config_evolution
        self.mutations_attempted = 0
        self.mutations_accepted = 0
        self.current_usage = UsageStats()
        self.phase: HelixPhase | str = "Initializing"
        self._live = Live(
            self._render(),
            console=console,
            refresh_per_second=4,
            transient=False,
        )

    def _render(self) -> Panel:
        return render_status_panel(
            self.gen,
            self.total,
            self.phase,
            self.current_usage,
            self.cumulative_budget,
            frontier=self.frontier,
            config_evolution=self.config_evolution,
            mutations_attempted=self.mutations_attempted,
            mutations_accepted=self.mutations_accepted,
        )

    def update(
        self,
        phase: HelixPhase | str | None = None,
        usage: UsageStats | dict[str, Any] | None = None,
        mutations_attempted: int | None = None,
        mutations_accepted: int | None = None,
    ) -> None:
        """Update the live display with new phase, usage, or mutation info."""
        if phase is not None:
            self.phase = phase
        if usage is not None:
            self.current_usage.add(usage)
        if mutations_attempted is not None:
            self.mutations_attempted = mutations_attempted
        if mutations_accepted is not None:
            self.mutations_accepted = mutations_accepted
        self._live.update(self._render())

    def __enter__(self) -> "HelixLiveDisplay":
        self._previous_live = _get_active_live()
        _set_active_live(self)
        self._live.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._live.stop()
        _set_active_live(self._previous_live)


def set_phase(phase: HelixPhase | str) -> None:
    """Update the displayed phase. Updates live display if active, else prints."""
    active = _get_active_live()
    if active:
        active.update(phase=phase)
    else:
        phase_str = phase.value if isinstance(phase, HelixPhase) else str(phase)
        console.print(f"[bold dim]⟳  {phase_str}…[/bold dim]")


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
    """Render Rich Progress bars for evaluation budget usage."""
    with Progress(
        TextColumn("[bold blue]{task.description:<20}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        console=console,
        transient=False,
    ) as progress:
        cap = config_evolution.max_evaluations
        progress.add_task(
            "Evaluations",
            total=cap if cap > 0 else None,
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
