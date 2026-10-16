"""coding node: obtain a runnable method entry for every approved candidate.

Two paths:

* the domain plugin exposes a canonical ``method_entry`` hook, so the plugin
  materialises ``model_pool/<method>/method.py`` itself (guide 4.1: reference a
  versioned implementation instead of re-generating it). No coding budget is
  spent because no LLM code generation happens;
* otherwise the Codex coding unit writes the implementation (demo path).
"""

from __future__ import annotations

from typing import Any, Dict
import json

from algorithm_plugins.registry import hook, resolve_plugin

from ...paths import TaskPaths, write_json
from ...schemas import contracts as C
from ...schemas.enums import ResultStatus
from ..state import TaskState


def coding_node(state: TaskState, ctx) -> Dict[str, Any]:
    task_id = state["task_id"]
    tp = TaskPaths(ctx.settings, state["task_slug"])
    spec = ctx.repo.get_spec(task_id)
    plan = ctx.repo.get_plan(state["plan_id"])
    budget = plan.change_scope.resource_budget
    coding_ids = list(state.get("coding_ids", []))
    entries: Dict[str, str] = dict(state.get("model_pool_entries", {}))
    open_crs = []
    new_calls = 0
    failures = []

    plugin = resolve_plugin(spec)
    materialise = hook(plugin, "method_entry")
    context_files_fn = hook(plugin, "coding_context_files")
    context_files = (list(context_files_fn(spec)) if context_files_fn is not None
                     else ["algorithm_plugins/demo_tabular/methods.py"])
    materialised = 0

    for c in plan.candidates:
        method = c.method
        entry_rel = f"model_pool/{method}/method.py"
        if c.action == "reuse" and (tp.root / entry_rel).exists():
            entries[method] = entry_rel
            continue
        if c.action == "reuse" and materialise is not None:
            rel = materialise(tp.root, method, spec)
            if rel:
                entries[method] = str(rel)
                materialised += 1
                continue
        if ctx.repo.get_budget(task_id).get("used", {}).get("coding_calls", 0) >= budget.max_coding_calls:
            open_crs.append(C.ChangeRequest(reason=f"coding call budget exhausted before {method}",
                                            suggested_next_action="raise budget or drop candidate"))
            break
        spec_task = C.CodingTaskSpec(
            coding_id=f"code_{plan.plan_id}_{method}", task_id=task_id, plan_id=plan.plan_id,
            plan_hash=state.get("plan_hash", ""),
            objective=json.dumps({
                "candidate": c.model_dump(), "task": spec.model_dump(),
                "data_contract": ("run(ctx, config) trains only on ctx.data_dir train/dev labels; "
                                  "predict(ctx, config, weights_path) loads trained weights without refitting. "
                                  "Return weights_ref pointing to an artifact in ctx.out_dir. "
                                  "Tabular predictions: pred_test.npy; IQA: pred_test.json. "
                                  "Do not read evaluation/protocols, other releases or test labels."),
                "entry": entry_rel, "hypotheses": plan.hypotheses,
                "history": [{"model_id": e.model_id, "metrics": e.metrics, "failures": e.failures}
                            for e in ctx.repo.list_evals(task_id)[-6:] if e.stage != "final"],
            }, ensure_ascii=False),
            workspace=str(tp.coding_workspaces), write_paths=[f"model_pool/{method}"],
            acceptance_commands=[
                f"python -c \"import ast,pathlib; ast.parse(pathlib.Path('{entry_rel}').read_text())\"",
                f"python -c \"from algorithm_plugins.common.plugin import load_method; "
                f"m=load_method('{entry_rel}'); assert callable(m.run)"
                + ("; assert callable(m.predict)" if spec.auto_process else "") + "\""],
            budget=budget, context_files=context_files, entry=entry_rel,
            require_real=c.action != "reuse")
        if spec.auto_process and spec.domain == "demo_tabular" and c.action != "reuse":
            spec_task.acceptance_commands.append(
                "python -m algo_agent.execution.contract_probe coding_contract.json")
        result = ctx.coding_agent.submit(spec_task, {"task_slug": tp.slug, "method": method,
                                                     "dependency_lock": "pyproject.toml"})
        new_calls += 1
        coding_ids.append(result.coding_id)
        if result.status == ResultStatus.COMPLETED.value:
            entries[method] = entry_rel
        else:
            open_crs.extend(result.change_requests)
            failures.append({"method": method, "status": result.status, "issues": result.open_issues,
                             "verifications": [v.model_dump() for v in result.verifications]})

    cr_ids = list(state.get("change_request_ids", []))
    for cr in open_crs:
        cr_ids.append(ctx.repo.save_change_request(task_id, plan.plan_id, "", cr))
    write_json(tp.plan_dir(plan.version) / "coding_failures.json",
               {"failures": failures, "change_requests": [c.model_dump() for c in open_crs]})
    ctx.repo.append_event(task_id, "stage.coding",
                          f"coding complete for {len(entries)} methods",
                          data={"methods": list(entries), "new_coding_calls": new_calls,
                                "materialised_from_plugin": materialised,
                                "change_requests": len(open_crs)})
    return {"stage": "coding", "coding_ids": coding_ids, "model_pool_entries": entries,
            "change_request_ids": cr_ids}


__all__ = ["coding_node"]
