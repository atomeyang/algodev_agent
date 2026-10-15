"""export_docs node: regenerate task overview + iteration reports from the DB."""

from __future__ import annotations

from typing import Any, Dict

from ...paths import TaskPaths
from ...schemas.enums import TaskPhase
from ..state import TaskState


def export_docs_node(state: TaskState, ctx) -> Dict[str, Any]:
    task_id = state["task_id"]
    tp = TaskPaths(ctx.settings, state["task_slug"])
    if ctx.docs is not None:
        ctx.docs.export_task_overview(ctx.settings, ctx.repo, state["task_slug"])
        ctx.docs.export_iteration_report(ctx.settings, ctx.repo, state["task_slug"],
                                         int(state.get("iteration", 1)) or 1)
    ctx.repo.append_event(task_id, "stage.export_docs", "task overview and iteration report regenerated")
    snap = ctx.repo.get_state(task_id)
    if snap.phase not in (TaskPhase.ACCEPTED.value, TaskPhase.STOPPED.value, TaskPhase.FAILED.value):
        ctx.repo.update_state(task_id, phase=TaskPhase.ANALYZING.value)
    return {"stage": "export_docs"}


__all__ = ["export_docs_node"]
