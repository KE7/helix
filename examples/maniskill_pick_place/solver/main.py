"""Task router: dispatch a trial to its family-specific routine."""

from __future__ import annotations

from typing import Any

from solver.tasks import pickcube, pushcube, stackcube

_ROUTES = {
    "pushcube": pushcube.solve,
    "pickcube": pickcube.solve,
    "stackcube": stackcube.solve,
}


def run(ctx: Any) -> dict[str, Any]:
    """Execute one trial. Returns a small status dict (informational only)."""
    route = _ROUTES.get(ctx.task_family)
    if route is None:
        return {"status": "unknown_task", "task_family": ctx.task_family}
    route(ctx)
    return {"status": "done", "task_family": ctx.task_family}
