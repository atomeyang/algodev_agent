"""analyze / decide / feedback_gate / diagnose_feedback nodes."""

from __future__ import annotations

from typing import Any, Dict, List

from langgraph.types import interrupt

from ...paths import TaskPaths, write_json
from ...schemas import contracts as C
from ...schemas.enums import TaskPhase
from ...policy import acceptance_failures
from ..state import TaskState


def _targets(spec: C.TaskSpec) -> List[C.MetricTarget]:
    return [spec.primary_metric] + list(spec.secondary_metrics)


def analyze_node(state: TaskState, ctx) -> Dict[str, Any]:
    task_id = state["task_id"]
    tp = TaskPaths(ctx.settings, state["task_slug"])
    spec = ctx.repo.get_spec(task_id)
    plan = ctx.repo.get_plan(state["plan_id"])
    evals = ctx.repo.list_evals(task_id)
    # diagnose the *best feasible* model, not whichever run finished last
    best_id = state.get("best_feasible_model") or ""
    best_evals = [e for e in evals if e.model_id == best_id]
    chosen = best_evals[-1] if best_evals else None
    best_metrics = chosen.metrics if chosen else {}

    # evidence packet with deltas vs the first (baseline) model
    models = ctx.repo.list_models(task_id)
    delta = {}
    if len(models) >= 2:
        delta = ctx.delta.compare_runs(task_id, models[0].model_id, models[-1].model_id)["delta"]
    packet = ctx.packets.build_packet(
        task_id=task_id, requirement=spec.business, plan_id=plan.plan_id,
        baseline_id=models[0].model_id if models else "", metric_delta=delta,
        slices=chosen.slices if chosen else {},
        failures=chosen.failures if chosen else [],
        counterexamples=[],
        unknowns=spec.unknowns)
    write_json(tp.docs / "iterations" / f"packet_iter{int(state.get('iteration', 1))}.json", packet.model_dump())

    failures = acceptance_failures(spec, chosen)
    failures.extend(state.get("dataset_problems", []))

    diag = ctx.providers["local"].diagnose({"summary": "; ".join(failures),
                                            "slices": chosen.slices if chosen else {},
                                            "metrics": best_metrics, "spec": spec.model_dump(),
                                            "domain": spec.domain})
    ctx.repo.append_event(task_id, "stage.analyze",
                          f"failures={len(failures)}", data={"failures": failures, "diagnosis": diag,
                                                             "delta": delta, "packet_id": packet.packet_id})
    score = best_metrics.get(spec.primary_metric.name)
    previous = state.get("best_score")
    from .experiments import evaluation_fingerprint
    fingerprint = evaluation_fingerprint(tp, chosen.dataset_version) if chosen else ""
    if spec.auto_process and fingerprint != state.get("evaluation_fingerprint"):
        previous = None
    improved = score is not None and (previous is None or
               (spec.primary_metric.better(score, previous) and
                abs(score - previous) > plan.stop_rules.min_metric_delta))
    return {"stage": "analyze", "last_metrics": best_metrics,
            "needs_plan_change": bool(failures),
            "no_improve": 0 if improved else int(state.get("no_improve", 0)) + 1,
            "best_score": score if improved else previous,
            "evaluation_fingerprint": fingerprint,
            "unknown_items": list(state.get("unknown_items", []))}


def decide_node(state: TaskState, ctx) -> Dict[str, Any]:
    task_id = state["task_id"]
    spec = ctx.repo.get_spec(task_id)
    iteration = int(state.get("iteration", 1)) or 1
    evaluations = ctx.repo.list_evals(task_id, state.get("best_feasible_model") or "__missing__")
    failures = acceptance_failures(spec, evaluations[-1] if evaluations else None)
    no_improve = int(state.get("no_improve", 0))

    if not failures:
        if spec.auto_process:
            from .experiments import evaluate_final
            final = evaluate_final(state, ctx)
            if final.passed is not True:
                reason = "final holdout failed; no further tuning on this holdout is permitted"
                ctx.repo.update_state(task_id, phase="stopped", stop_reason=reason)
                return {"stage": "decide", "decision": "stop", "stop_reason": reason,
                        "final_eval_id": final.eval_id}
        ctx.repo.append_event(task_id, "stage.decide", "primary and slice targets met; accepting",
                              data={"best_model": state.get("best_feasible_model")})
        ctx.repo.update_state(task_id, phase=TaskPhase.ACCEPTED.value,
                              best_model_id=state.get("best_feasible_model"))
        return {"stage": "decide", "decision": "accepted", "stop_reason": "acceptance criteria met"}

    if iteration >= spec.budget.max_iterations:
        reason = f"max_iterations ({spec.budget.max_iterations}) reached without meeting targets"
    elif no_improve >= spec.budget.max_no_improve:
        reason = f"no improvement for {no_improve} iterations"
    elif ctx.repo.get_budget(task_id)["used"].get("research_calls", 0) >= spec.budget.max_research_calls:
        reason = "research budget exhausted without meeting targets"
    elif ctx.repo.get_budget(task_id)["used"].get("experiments", 0) >= (
            spec.budget.max_experiments - (1 if spec.auto_process else 0)):
        reason = "experiment budget exhausted (final evaluation slot reserved)"
    else:
        ctx.repo.append_event(task_id, "stage.decide",
                              f"targets not met; continuing to iteration {iteration+1}",
                              data={"failures": failures})
        ctx.repo.update_state(task_id, phase=TaskPhase.EXECUTING.value)
        return {"stage": "decide", "decision": "continue", "stop_reason": None}

    ctx.repo.append_event(task_id, "stage.decide", f"stopping: {reason}", level="warning")
    ctx.repo.update_state(task_id, phase=TaskPhase.STOPPED.value, stop_reason=reason,
                          best_model_id=state.get("best_feasible_model"))
    return {"stage": "decide", "decision": "stop", "stop_reason": reason}


def feedback_gate_node(state: TaskState, ctx) -> Dict[str, Any]:
    iteration = int(state.get("iteration", 1)) or 1
    spec = ctx.repo.get_spec(state["task_id"])
    if not spec.auto_process and (ctx.extras.get("pause_between_iterations") or ctx.repo.list_feedback(state["task_id"])):
        resume = interrupt({
            "type": "iteration_boundary",
            "task_id": state["task_id"],
            "iteration": iteration,
            "metrics": state.get("last_metrics", {}),
            "best_model": state.get("best_feasible_model"),
            "message": "Iteration finished. Submit /feedback or continue to the next planning round.",
        })
        if isinstance(resume, dict) and resume.get("stop"):
            return {"stage": "feedback_gate", "decision": "stop", "stop_reason": "user stopped at iteration boundary"}
    ctx.repo.update_state(state["task_id"], iteration=iteration + 1)
    return {"stage": "feedback_gate", "iteration": iteration + 1}


def diagnose_feedback_node(state: TaskState, ctx) -> Dict[str, Any]:
    task_id = state["task_id"]
    feedback = ctx.repo.list_feedback(task_id)
    ids: List[str] = []
    for fb in feedback:
        diag = ctx.providers["local"].diagnose({"summary": f"{fb.scenario}: {fb.actual}", "slices": {}})
        # A diagnostic model's opinion is not an executed reproduction.
        if fb.reproduced is None and "reproduction has not been independently executed" not in fb.missing_evidence:
            fb.missing_evidence.append("reproduction has not been independently executed")
        ctx.repo.save_feedback(fb)
        ids.append(fb.feedback_id)
        ctx.repo.append_event(task_id, "stage.diagnose_feedback",
                              f"feedback {fb.feedback_id}: category={diag.get('category', 'unknown')}",
                              data={"diagnosis": diag, "reproduced": fb.reproduced})
    return {"stage": "diagnose_feedback", "feedback_ids": ids}


__all__ = ["analyze_node", "decide_node", "feedback_gate_node", "diagnose_feedback_node"]
