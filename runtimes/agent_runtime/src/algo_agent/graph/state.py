"""LangGraph task state. Large artifacts (weights, data, logs, chat) are kept
out of the checkpoint; only IDs and control fields are persisted (guide 6).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class TaskState(TypedDict, total=False):
    task_id: str
    task_slug: str
    iteration: int
    stage: str

    plan_id: str
    plan_hash: str
    plan_version: str
    pending_approval: Optional[Dict[str, Any]]
    approved: bool
    approval_id: str

    # research
    candidates: List[Dict[str, Any]]
    evidence_refs: List[str]
    research_report_ref: str
    unknown_items: List[str]

    # assets & coding
    dataset_version: str
    model_pool_entries: Dict[str, str]
    coding_ids: List[str]

    # experiments
    run_ids: List[str]
    job_ids: List[str]

    # models & evaluation
    model_ids: List[str]
    eval_ids: List[str]
    best_feasible_model: Optional[str]

    # feedback loop
    feedback_ids: List[str]
    needs_plan_change: bool
    change_request_ids: List[str]

    # control
    budget_used: Dict[str, Any]
    stop_reason: Optional[str]
    decision: str          # continue | stop | accepted | needs_input
    last_metrics: Dict[str, float]
    no_improve: int
    best_score: Optional[float]
    dataset_problems: List[str]
    raw_text: str
    final_eval_id: str
    evaluation_fingerprint: str


def initial_state(task_id: str, task_slug: str) -> TaskState:
    return TaskState(
        task_id=task_id, task_slug=task_slug, iteration=0, stage="start", approved=False,
        candidates=[], evidence_refs=[], coding_ids=[], run_ids=[], job_ids=[], model_ids=[],
        eval_ids=[], feedback_ids=[], needs_plan_change=False, change_request_ids=[],
        model_pool_entries={}, best_feasible_model=None, budget_used={}, decision="continue",
        last_metrics={}, unknown_items=[], no_improve=0, best_score=None,
    )


__all__ = ["TaskState", "initial_state"]
