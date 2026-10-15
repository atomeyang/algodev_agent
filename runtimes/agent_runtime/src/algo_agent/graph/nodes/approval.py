"""approval_gate node: pure LangGraph interrupt; the control layer validates the
plan hash/state version before resuming (guide 6).
"""

from __future__ import annotations

from typing import Any, Dict

from langgraph.types import interrupt

from ...schemas import contracts as C
from ...schemas.enums import TaskPhase
from ..state import TaskState


def approval_node(state: TaskState, ctx) -> Dict[str, Any]:
    pending = state.get("pending_approval") or {}
    spec = ctx.repo.get_spec(state["task_id"])
    resume = ({"approved": True, "plan_id": state["plan_id"], "plan_hash": state["plan_hash"]}
              if spec.auto_process else interrupt({
        "type": "approval_request",
        "task_id": state["task_id"],
        "plan_id": state.get("plan_id"),
        "plan_hash": state.get("plan_hash"),
        "plan_version": state.get("plan_version"),
        "summary": pending.get("summary", ""),
        "candidates": pending.get("candidates", []),
        "budget": pending.get("budget", {}),
        "message": "Reply with an explicit approval action to freeze this plan hash.",
    }))
    approved = bool(resume.get("approved")) if isinstance(resume, dict) else bool(resume)
    plan_id = resume.get("plan_id") if isinstance(resume, dict) else state.get("plan_id")
    plan_hash = resume.get("plan_hash") if isinstance(resume, dict) else state.get("plan_hash")
    scope = resume.get("scope") if isinstance(resume, dict) else None

    valid = approved and plan_id == state.get("plan_id") and plan_hash == state.get("plan_hash")
    if not valid:
        ctx.repo.append_event(state["task_id"], "stage.approval_gate",
                              "approval rejected or hash mismatch", level="warning",
                              data={"approved": approved, "plan_id": plan_id, "plan_hash": plan_hash})
        return {"stage": "approval_gate", "approved": False, "decision": "needs_input",
                "stop_reason": "approval pending or hash mismatch"}

    plan = ctx.repo.get_plan(plan_id)
    approval = C.Approval(
        approval_id=f"appr_{plan_id}", task_id=state["task_id"], plan_id=plan_id, plan_hash=plan_hash,
        scope=C.ApprovalScope(**(scope or plan.change_scope.model_dump())),
        mode=(scope or {}).get("approval_mode", plan.change_scope.approval_mode),
        state_version=ctx.repo.get_state(state["task_id"]).state_version)
    ctx.repo.save_approval(approval)
    ctx.repo.append_event(state["task_id"], "stage.approval_gate",
                          f"plan {plan_id} approved (hash {plan_hash[:12]})",
                          data={"approval_id": approval.approval_id, "mode": approval.mode,
                                "auto_process": spec.auto_process})
    ctx.repo.update_state(state["task_id"], phase=TaskPhase.EXECUTING.value)
    return {"stage": "approval_gate", "approved": True, "approval_id": approval.approval_id,
            "decision": "continue"}


__all__ = ["approval_node"]
