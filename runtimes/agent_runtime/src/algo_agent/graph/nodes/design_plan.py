"""design_plan node: turn research output into a hash-frozen PlanSpec."""

from __future__ import annotations

from typing import Any, Dict, List

import yaml

from algorithm_plugins.registry import hook, resolve_plugin

from ...paths import TaskPaths, read_json, sha256_obj, write_json
from ...schemas import contracts as C
from ...schemas.enums import ApprovalMode, TaskPhase
from ...policy import plan_problems, task_key
from ..state import TaskState


def build_plan_spec(spec: C.TaskSpec, research: Dict[str, Any], iteration: int,
                    parent_plan_id: str, settings) -> C.PlanSpec:
    candidates = [C.Candidate(**c) for c in research.get("candidates", [])]
    recipe = C.DataRecipe(**research["data_recipe"]) if research.get("data_recipe") else C.DataRecipe(
        recipe_id=f"recipe_v{iteration}")
    protocol = C.EvalProtocol(**research["eval_protocol"]) if research.get("eval_protocol") else C.EvalProtocol(
        protocol_id=f"proto_v{iteration}", primary_metric=spec.primary_metric.name,
        direction=spec.primary_metric.direction, threshold=spec.primary_metric.threshold)
    if spec.auto_process:
        operations = research.get("extras", {}).get("data_operations", [])
        recipe.operations = [C.DataOperation(**op) for op in operations] if operations else recipe.operations
        protocol.stage = "development"
    protocol.dataset_version = f"v{iteration}"

    methods = [c.method for c in candidates]
    # Domain plugins may widen the approvable search space; without a hook the
    # synthetic-tabular defaults are used so the demo plan is unchanged.
    plugin = resolve_plugin(spec)
    param_space_fn = hook(plugin, "parameter_space")
    if param_space_fn is not None:
        parameter_space = dict(param_space_fn(spec, methods))
    else:
        parameter_space = {m: {"alpha": [0.1, 1.0, 10.0]} if m.startswith("ridge") else {}
                           for m in methods}
    for candidate in candidates:
        if candidate.params:
            parameter_space[candidate.method] = candidate.params
        if spec.auto_process:
            cfg = parameter_space[candidate.method]
            cfg.setdefault("epochs", spec.budget.max_epochs)
            cfg.setdefault("mlp_max_iter", spec.budget.max_epochs)
            for key in ("epochs", "max_epochs", "mlp_max_iter"):
                if key in cfg and key not in candidate.params:
                    cfg[key] = min(int(cfg[key]), spec.budget.max_epochs)
    data_sources_fn = hook(plugin, "data_sources")
    allowed_data_sources = (list(data_sources_fn(spec)) if data_sources_fn is not None
                            else (["synthetic_demo"] if spec.synthetic else ["task_local"]))
    allowed_data_sources += [source.source_id for source in spec.data_sources]
    scope = C.ApprovalScope(
        allowed_methods=methods,
        allowed_data_sources=allowed_data_sources,
        allowed_recipe_changes=["augment", "resample", "filter", "annotate", "relabel", "extend"]
        if spec.auto_process else [],
        parameter_space=parameter_space,
        write_paths=["model_pool"],
        evaluation_protocol_id=protocol.protocol_id,
        upload_policy=settings.workflow.get("upload_policy", {}).get("default", "restricted"),
        resource_budget=spec.budget,
        max_iterations=spec.budget.max_iterations,
        approval_mode=settings.workflow.get("approval", {}).get("default_mode", ApprovalMode.SCOPED_AUTO.value),
    )
    stop = settings.stop_rules()
    dag = [C.DagStep(step_id="prepare_assets", kind="experiment"),
           C.DagStep(step_id="coding", kind="coding", depends_on=["prepare_assets"])]
    for c in candidates:
        dag.append(C.DagStep(step_id=f"run_{c.method}", kind="experiment", depends_on=["coding"],
                             params={"method": c.method}))
    dag.append(C.DagStep(step_id="evaluate", kind="evaluation", depends_on=[f"run_{c.method}" for c in candidates]))
    dag.append(C.DagStep(step_id="analyze", kind="analysis", depends_on=["evaluate"]))

    return C.PlanSpec(
        plan_id=f"p_{task_key(spec.task_id)}_{iteration:03d}", task_id=spec.task_id, version=f"v{iteration}",
        parent_plan_id=parent_plan_id or None, summary=research.get("report_markdown", "")[:400],
        evidence_refs=research.get("evidence_refs", []), candidates=candidates,
        hypotheses=research.get("hypotheses", []), controls=research.get("controls", []),
        data_recipe=recipe, eval_protocol=protocol, execution_dag=dag, change_scope=scope,
        cost_estimate=str(research.get("cost", {})),
        stop_rules=C.StopRules(require_all_hard_constraints=stop.get("require_all_hard_constraints", True),
                               min_metric_delta=stop.get("min_metric_delta", 0.0),
                               max_no_improve=spec.budget.max_no_improve),
    )


def design_plan_node(state: TaskState, ctx) -> Dict[str, Any]:
    task_id = state["task_id"]
    tp = TaskPaths(ctx.settings, state["task_slug"])
    spec = ctx.repo.get_spec(task_id)
    iteration = int(state.get("iteration", 1)) or 1
    research = read_json(tp.docs / "research" / f"research_iter{iteration}.json", {}) or {}
    parent = state.get("plan_id", "")
    plan = build_plan_spec(spec, research, iteration, parent, ctx.settings)
    problems = plan_problems(spec, plan, ctx.repo.get_plan(parent) if parent else None)
    if problems:
        raise ValueError("plan validation failed: " + "; ".join(problems))
    content_hash = sha256_obj(plan.model_dump())
    ctx.repo.save_plan(plan, content_hash, status="proposed")

    pdir = tp.plan_dir(plan.version)
    pdir.mkdir(parents=True, exist_ok=True)
    write_json(pdir / "requirement.json", spec.model_dump())
    (pdir / "execution.yaml").write_text(
        yaml.safe_dump({"plan_id": plan.plan_id, "version": plan.version,
                        "dag": [s.model_dump() for s in plan.execution_dag],
                        "change_scope": plan.change_scope.model_dump()}, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    write_json(pdir / "proposal.json", plan.model_dump())
    write_json(pdir / "approval.json", {"status": "pending", "plan_id": plan.plan_id,
                                        "plan_hash": content_hash, "scope": plan.change_scope.model_dump()})

    ctx.repo.append_event(task_id, "stage.design_plan",
                          f"plan {plan.plan_id} ({plan.version}) proposed; hash={content_hash[:12]}",
                          data={"plan_id": plan.plan_id, "plan_hash": content_hash,
                                "candidates": [c.method for c in plan.candidates]})
    ctx.repo.update_state(task_id, phase=TaskPhase.AWAITING_APPROVAL.value, active_plan_id=plan.plan_id)
    return {
        "stage": "design_plan", "plan_id": plan.plan_id, "plan_hash": content_hash,
        "plan_version": plan.version,
        "pending_approval": {"plan_id": plan.plan_id, "plan_hash": content_hash,
                             "version": plan.version,
                             "summary": plan.summary[:300],
                             "candidates": [c.method for c in plan.candidates],
                             "budget": plan.change_scope.resource_budget.model_dump()},
        "approved": False,
        "model_pool_entries": {},
    }


__all__ = ["design_plan_node", "build_plan_spec"]
