"""End-to-end offline demo orchestration (guide 21-22, 36).

Runs the full happy-path loop against a real task so that a first-time user can
see every stage work without any cloud credentials::

    create task -> research/plan -> approve -> train+eval -> iterate
                -> user feedback -> re-plan -> approve -> accept -> export docs

The demo is honest about what it is: it uses synthetic data and the offline
coding executor, so every metric is labelled ``synthetic_demo``. Nothing is
faked: approvals really gate execution, jobs really run in subprocesses, and
metrics really come from the evaluator.
"""

from __future__ import annotations

import shutil
from typing import Any, Dict, List, Optional

from .cli import render
from .cli.client import ClientError, ControlClient
from .config import Settings

DEMO_REQUIREMENT = "离线合成数据上预测目标 y，RMSE 不超过 0.5。"
DEMO_TASK_ID = "demo_task"
DEMO_METHODS: List[str] = ["baseline_mean", "ridge", "gbr"]
DEMO_FEEDBACK: Dict[str, Any] = {
    "source": "user",
    "scenario": "low_light slice",
    "expected": "rmse (low_light slice) <= 1.0",
    "actual": "rmse (low_light slice) = 1.14",
    "severity": "high",
    "model_id": "m_gbr",
    "repro": "low-light samples are systematically under-predicted",
}


def run_demo(settings: Settings, *, client: Optional[ControlClient] = None,
             task_id: str = DEMO_TASK_ID, offline: bool = True, fresh: bool = False,
             json_mode: bool = False) -> Dict[str, Any]:
    """Drive one complete offline iteration loop and return a JSON report."""
    client = client or ControlClient(settings)
    steps: List[str] = []

    if fresh:
        n = _reset_local_state(settings, task_id)
        steps.append(f"--fresh removed {n} stale state path(s)")

    if not json_mode:
        render.banner("offline demo: synthetic tabular regression")

    _ensure_task(client, task_id)
    steps.append(f"task {task_id} ready")
    if not json_mode:
        render.info(f"task {task_id} created -- target: RMSE <= 0.5")

    # 1) first plan -> approval gate
    boundary = client.run(task_id, raw_text=DEMO_REQUIREMENT)
    pending = _require_approval(boundary, "initial plan")
    plan1 = pending.get("plan_id")
    steps.append(f"iteration 1 plan {plan1} awaiting approval")
    if not json_mode:
        render.approval_panel(pending)

    # 2) approve -> agents design + jobs run
    boundary = client.approve(task_id, plan1, pending["plan_hash"],
                              expected_state_version=pending.get("state_version"))
    plan2: Optional[str] = None
    feedback_model: Optional[str] = None

    if _is_terminal(boundary):
        steps.append("iteration 1 already met the target; skipping feedback")
    else:
        evals1 = client.evals(task_id)
        steps.append(f"iteration 1 produced {len(evals1)} evaluation(s)")
        if not json_mode:
            render.metrics_table(evals1, "iteration 1 evaluations")

        if boundary.get("type") != "approval_request":
            # 3) user feedback on the best-but-insufficient model -> re-plan
            feedback = _choose_feedback_model(evals1, DEMO_FEEDBACK)
            feedback_model = feedback.get("model_id")
            resp = client.submit_feedback(task_id, feedback)
            boundary = (resp or {}).get("boundary") or client.wait_boundary(task_id)
            steps.append(f"feedback on {feedback_model} produced a follow-up plan")
            if not json_mode:
                render.info(f"feedback on {feedback_model} -> new plan")
        if boundary.get("type") == "approval_request":
            pending = boundary["interrupt"]
            plan2 = pending.get("plan_id")
            if not json_mode:
                render.approval_panel(pending)
            # 4) approve iteration 2 -> should reach the target and accept
            boundary = client.approve(task_id, plan2, pending["plan_hash"],
                                      expected_state_version=pending.get("state_version"))
        boundary = _settle(client, task_id, boundary, json_mode)

    state = client.state(task_id)
    evals = client.evals(task_id)
    models = client.models(task_id)
    passing = [e for e in evals if e.get("passed")]
    steps.append(f"final phase={state.get('phase')} best={state.get('best_model_id')} "
                 f"passing_evals={len(passing)}")
    if not json_mode:
        render.metrics_table(evals, "all evaluations")

    # 5) regenerate the human-readable records
    docs = client.export(task_id)
    steps.append(f"exported {len(docs)} document(s)")
    if not json_mode:
        _docs_note(docs)

    report = {
        "ok": state.get("phase") in {"accepted", "stopped"} and bool(passing),
        "mode": "offline_synthetic_demo" if offline else client.mode,
        "disclaimer": ("synthetic data + offline coding executor; metrics are labelled "
                       "synthetic_demo and are NOT cloud/GPU/business validation"),
        "root": str(settings.root),
        "task_id": task_id,
        "task_slug": state.get("task_slug") or (state.get("task") or {}).get("task_slug"),
        "phase": state.get("phase"),
        "best_model_id": state.get("best_model_id"),
        "iterations": state.get("iteration"),
        "plans": [p for p in (plan1, plan2) if p],
        "feedback_model": feedback_model,
        "models": [{"model_id": m.get("model_id"), "status": m.get("status"),
                    "run_id": m.get("run_id"), "recommended": m.get("recommended")}
                   for m in models],
        "evaluations": [{"eval_id": e.get("eval_id"), "model_id": e.get("model_id"),
                         "metrics": e.get("metrics"), "passed": e.get("passed"),
                         "synthetic": e.get("synthetic", False),
                         "resources": e.get("resources", {})} for e in evals],
        "documents": docs,
        "steps": steps,
    }
    if not json_mode:
        render.banner("demo summary")
        render.print_json(report)
    return report


# ------------------------------------------------------------------------- #
# helpers
# ------------------------------------------------------------------------- #
def _reset_local_state(settings: Settings, task_id: str) -> int:
    """Remove local runtime state so ``--fresh`` is deterministic.

    This clears the shared sqlite state + checkpoints (all local tasks) and the
    per-task workspace; it never touches ``shared_assets/``.
    """
    removed = 0
    for path in (settings.db_path, settings.checkpoint_path):
        try:
            if path.exists():
                path.unlink()
                removed += 1
        except OSError:
            pass
    slug = f"{task_id}_experiment"
    for cand in (settings.root / slug, settings.coding_workspaces / slug):
        try:
            if cand.exists():
                shutil.rmtree(cand)
                removed += 1
        except OSError:
            pass
    return removed


def _ensure_task(client: ControlClient, task_id: str) -> Dict[str, Any]:
    try:
        return client.create_task(DEMO_REQUIREMENT, task_id=task_id, demo=True,
                                  synthetic=True, methods_hint=list(DEMO_METHODS))
    except ClientError as exc:
        if exc.status != 409:
            raise
        # A task with this id already exists and has a different requirement.
        raise ClientError(
            f"task {task_id!r} already exists with a different requirement",
            recovery="re-run with --fresh to reset it, or use --task <new_id>",
            status=409) from exc


def _require_approval(boundary: Dict[str, Any], what: str) -> Dict[str, Any]:
    if not boundary or boundary.get("type") != "approval_request":
        got = (boundary or {}).get("type")
        raise ClientError(f"expected approval for {what}, got {got!r}",
                          recovery="run: algo-agent events --task <id>")
    return boundary.get("interrupt") or {}


def _is_terminal(boundary: Dict[str, Any]) -> bool:
    return bool((boundary or {}).get("done")) or (boundary or {}).get("type") == "end"


def _choose_feedback_model(evals: List[Dict[str, Any]], base: Dict[str, Any]) -> Dict[str, Any]:
    fb = dict(base)
    ids = {e.get("model_id") for e in evals if e.get("model_id")}
    if fb.get("model_id") not in ids and ids:
        fb["model_id"] = next(iter(sorted(ids)))
    return fb


def _settle(client: ControlClient, task_id: str, boundary: Dict[str, Any],
            json_mode: bool) -> Dict[str, Any]:
    """Loop through any further approval gates until the task terminates."""
    for _ in range(6):
        if _is_terminal(boundary):
            return boundary
        if boundary.get("type") == "approval_request" and not json_mode:
            render.approval_panel(boundary["interrupt"])
        if boundary.get("type") == "approval_request":
            pending = boundary["interrupt"]
            boundary = client.approve(task_id, pending.get("plan_id"), pending.get("plan_hash"),
                                      expected_state_version=pending.get("state_version"))
        elif boundary.get("type") == "iteration_boundary":
            boundary = client.continue_task(task_id, stop=False)
        else:
            break
    return boundary


def _docs_note(docs: Dict[str, Any]) -> None:
    if not docs:
        render.warn("no documents were produced")
        return
    render.info("documents written:")
    for name, path in docs.items():
        render.console.print(f"  [cyan]{name}[/cyan] -> {path}")


__all__ = ["run_demo", "DEMO_REQUIREMENT", "DEMO_TASK_ID", "DEMO_METHODS"]
