"""research node: run the autonomous research/design unit for the current iter."""

from __future__ import annotations

from typing import Any, Dict, List

from algorithm_plugins.registry import hook, resolve_plugin

from ...agents import ResearchDesignRequest
from ...paths import TaskPaths, write_json, read_json
from ...schemas.enums import TaskPhase
from ..state import TaskState


def _history_with_metrics(ctx, task_id: str, spec) -> List[Dict[str, Any]]:
    """Plan history enriched with each iteration's best evaluation.

    Passing the previous metrics/slices into the design request lets a domain
    plugin close the loop on measurement (e.g. pick a class-balanced variant
    when a rare defect class is weak) even without user feedback.
    """
    evals = [e for e in ctx.repo.list_evals(task_id) if e.stage != "final"]
    primary = spec.primary_metric.name
    history: List[Dict[str, Any]] = []
    for p in ctx.repo.list_plans(task_id):
        entry: Dict[str, Any] = {"plan_id": p["plan_id"], "version": p["version"]}
        evs = [e for e in evals if e.dataset_version == p["version"]]
        if evs:
            def _key(e):
                # Ties on the primary metric are common on the frozen benchmark
                # (e.g. the rule head and the adapter share the same defect
                # head). Break them with the other acceptance gates so the
                # "current status" reflects the model the agent will recommend,
                # not whichever run happened to be evaluated first.
                ranked = tuple(float(e.metrics.get(k) or 0.0) for k in
                               ("defect_macro_f1_cov", "hardcase_macro_f1", "plcc", "srcc"))
                value = e.metrics.get(primary)
                score = (value if spec.primary_metric.direction == "max" else -value) if value is not None else float("-inf")
                return (e.passed is True, score, ranked)

            best_ev = max(evs, key=_key)
            entry.update({"best_model": best_ev.model_id, "passed": best_ev.passed,
                          "metrics": best_ev.metrics, "slices": best_ev.slices,
                          "failures": best_ev.failures})
        try:
            plan = ctx.repo.get_plan(p["plan_id"])
            if plan is not None:
                entry["methods"] = [c.method for c in plan.candidates]
                entry["candidates"] = [c.model_dump() for c in plan.candidates]
                entry["data_operations"] = [op.model_dump() for op in plan.data_recipe.operations]
        except Exception:
            pass
        entry["jobs"] = [{"run_id": j.run_id, "status": j.status, "error": j.result.get("error")}
                         for j in ctx.repo.list_jobs(task_id)
                         if (ctx.repo.get_experiment(j.run_id) and
                             ctx.repo.get_experiment(j.run_id).plan_id == p["plan_id"] and
                             ctx.repo.get_experiment(j.run_id).env.get("stage") != "final")]
        tp = TaskPaths(ctx.settings, spec.task_slug)
        entry["data_failure"] = read_json(tp.plan_dir(p["version"]) / "data_failure.json", {})
        entry["coding_failures"] = read_json(tp.plan_dir(p["version"]) / "coding_failures.json", {})
        history.append(entry)
    return history[-8:]


def research_node(state: TaskState, ctx) -> Dict[str, Any]:
    task_id = state["task_id"]
    tp = TaskPaths(ctx.settings, state["task_slug"])
    spec = ctx.repo.get_spec(task_id)
    iteration = int(state.get("iteration", 1)) or 1
    ctx.repo.update_state(task_id, phase=TaskPhase.RESEARCHING.value)

    feedback = [f.model_dump() for f in ctx.repo.list_feedback(task_id)]
    history = _history_with_metrics(ctx, task_id, spec)
    slices_fn = hook(resolve_plugin(spec), "slices")
    slices = list(slices_fn(spec)) if slices_fn is not None else (
        ["low_light", "high_light"] if spec.demo else ["overall"])

    request = ResearchDesignRequest(
        task_spec=spec, task_slug=state["task_slug"], iteration=iteration,
        parent_plan_id=state.get("plan_id", ""), budget=spec.budget, slices=slices,
        methods_hint=spec.methods_hint or ctx.extras.get("methods_hint"),
        feedback=feedback, history=history,
    )
    result = ctx.research_agent.run(request)
    research_dir = tp.docs / "research"
    research_dir.mkdir(parents=True, exist_ok=True)
    (research_dir / f"research_iter{iteration}.md").write_text(result.report_markdown, encoding="utf-8")
    payload = {
        "report_markdown": result.report_markdown,
        "candidates": [c.model_dump() for c in result.candidates],
        "hypotheses": result.hypotheses, "controls": result.controls,
        "counterexamples": result.counterexamples, "unknowns": result.unknowns,
        "evidence_refs": result.evidence_refs,
        "data_recipe": result.data_recipe.model_dump() if result.data_recipe else None,
        "eval_protocol": result.eval_protocol.model_dump() if result.eval_protocol else None,
        "cost": result.cost, "status": result.status, "offline": result.offline,
        "tool_calls": result.tool_calls,
        "extras": getattr(result, "extras", {}) or {},
    }
    write_json(research_dir / f"research_iter{iteration}.json", payload)

    if result.status != "completed" or not result.candidates:
        from ...policy import TaskHalted
        reason = "research did not produce an executable plan: " + "; ".join(result.unknowns)
        ctx.repo.update_state(task_id, phase="stopped", stop_reason=reason)
        ctx.repo.append_event(task_id, "research.blocked", reason)
        raise TaskHalted(reason)

    ctx.repo.append_event(task_id, "stage.research",
                          f"research iteration {iteration}: {len(result.candidates)} candidates, "
                          f"status={result.status}",
                          data={"offline": result.offline, "candidates": [c.method for c in result.candidates],
                                "unknowns": result.unknowns})
    return {
        "stage": "research",
        "candidates": payload["candidates"],
        "evidence_refs": result.evidence_refs,
        "research_report_ref": str(research_dir / f"research_iter{iteration}.md"),
        "unknown_items": list(state.get("unknown_items", [])) + result.unknowns,
    }


__all__ = ["research_node"]
