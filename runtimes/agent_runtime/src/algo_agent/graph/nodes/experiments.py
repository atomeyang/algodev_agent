"""submit_experiments / collect_results / evaluate nodes.

Jobs are submitted with idempotent keys; in offline/inline mode the worker is
drained so the loop is deterministic. Evaluation always uses frozen protocol
code, never the method's own metric (guide 8).

Domain plugins may supply the experiment environment, the held-out split, the
resource declaration and the frozen evaluator; without those hooks the
synthetic-tabular behaviour is used unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List
import math

from algorithm_plugins.registry import hook, resolve_plugin

from ...execution import evaluate_run
from ...paths import TaskPaths, read_json, sha256_file, safe_join
from ...policy import acceptance_failures, remaining_seconds, task_key, check_running
from ...schemas import contracts as C
from ...schemas.enums import JobStatus, TaskPhase
from ..state import TaskState


def _experiment_env(plugin, spec: C.TaskSpec) -> Dict[str, Any]:
    fn = hook(plugin, "experiment_env")
    if fn is not None:
        return dict(fn(spec))
    return {"python": "local", "backend": "sklearn"}


def _experiment_split(plugin, spec: C.TaskSpec) -> str:
    fn = hook(plugin, "experiment_split")
    return str(fn(spec)) if fn is not None else "test"


def _experiment_synthetic(plugin, spec: C.TaskSpec) -> bool:
    fn = hook(plugin, "experiment_synthetic")
    return bool(fn(spec)) if fn is not None else True


def _experiment_resources(plugin, spec: C.TaskSpec) -> Dict[str, Any]:
    fn = hook(plugin, "experiment_resources")
    if fn is not None:
        return dict(fn(spec))
    return {"device": "cpu", "synthetic_demo": True}


def submit_experiments_node(state: TaskState, ctx) -> Dict[str, Any]:
    task_id = state["task_id"]
    tp = TaskPaths(ctx.settings, state["task_slug"])
    spec_task = ctx.repo.get_spec(task_id)
    plugin = resolve_plugin(spec_task)
    plan = ctx.repo.get_plan(state["plan_id"])
    entries: Dict[str, str] = dict(state.get("model_pool_entries", {}))
    job_ids: List[str] = list(state.get("job_ids", []))
    run_ids: List[str] = list(state.get("run_ids", []))
    max_experiments = plan.change_scope.resource_budget.max_experiments
    split = _experiment_split(plugin, spec_task)
    env = _experiment_env(plugin, spec_task)
    synthetic = _experiment_synthetic(plugin, spec_task)
    submitted = 0

    for c in plan.candidates:
        entry_rel = entries.get(c.method)
        if not entry_rel:
            continue
        run_id = f"run_{task_key(task_id)}_{plan.version}_{c.method}"
        spec = C.ExperimentSpec(
            run_id=run_id, task_id=task_id, plan_id=plan.plan_id, candidate_id=c.candidate_id,
            method=c.method, code_ref=entry_rel, data_version=state["dataset_version"],
            config=plan.change_scope.parameter_space.get(c.method, {}) or {},
            env={**env, "stage": plan.eval_protocol.stage}, seed=plan.data_recipe.seed,
            hypothesis="; ".join(plan.hypotheses[:1]), synthetic=synthetic)
        used = ctx.repo.get_budget(task_id)["used"].get("experiments", 0)
        if used >= max_experiments - (1 if spec_task.auto_process else 0):
            break
        job = ctx.runner.submit(spec, tp.slug, str((tp.root / entry_rel)), split=split)
        if job.job_id not in job_ids:
            job_ids.append(job.job_id)
        if run_id not in run_ids:
            run_ids.append(run_id)
        submitted += 1

    ctx.repo.append_event(task_id, "stage.submit_experiments",
                          f"submitted {submitted} experiment jobs for {plan.plan_id}",
                          data={"submitted": submitted, "split": split, "env": env,
                                "run_ids": run_ids})
    if ctx.inline:
        ctx.worker.drain(task_id=task_id)
    elif job_ids:
        ctx.worker.await_jobs(job_ids, timeout=max(1, remaining_seconds(spec_task)))
    check_running(ctx.repo, task_id)
    ctx.repo.update_state(task_id, phase=TaskPhase.EXECUTING.value)
    return {"stage": "submit_experiments", "job_ids": job_ids, "run_ids": run_ids}


def collect_results_node(state: TaskState, ctx) -> Dict[str, Any]:
    task_id = state["task_id"]
    results = []
    for job_id in state.get("job_ids", []):
        job = ctx.repo.get_job(job_id)
        if not job:
            continue
        results.append({"job_id": job_id, "run_id": job.run_id, "status": job.status,
                        "metrics": job.result.get("metrics", {})})
    failed = [r for r in results if r["status"] != JobStatus.SUCCEEDED.value]
    ctx.repo.append_event(task_id, "stage.collect_results",
                          f"collected {len(results)} runs ({len(failed)} not succeeded)",
                          data={"results": results})
    return {"stage": "collect_results"}


def evaluate_node(state: TaskState, ctx) -> Dict[str, Any]:
    task_id = state["task_id"]
    tp = TaskPaths(ctx.settings, state["task_slug"])
    spec_task = ctx.repo.get_spec(task_id)
    plugin = resolve_plugin(spec_task)
    plan = ctx.repo.get_plan(state["plan_id"])
    protocol = plan.eval_protocol
    evaluate_fn = hook(plugin, "evaluate")
    resources = _experiment_resources(plugin, spec_task)
    synthetic = _experiment_synthetic(plugin, spec_task)
    manifest_fn = hook(plugin, "model_manifest")
    select_best_fn = hook(plugin, "select_best")
    model_ids: List[str] = []
    eval_ids: List[str] = []
    collected: List[Dict[str, Any]] = []
    best = None
    best_metric = None
    last_metrics: Dict[str, float] = {}

    for c in plan.candidates:
        run_id = f"run_{task_key(task_id)}_{plan.version}_{c.method}"
        model_id = f"m_{task_key(task_id)}_{plan.version}_{c.method}"
        job = next((j for j in ctx.repo.list_jobs(task_id) if j.run_id == run_id), None)
        if job is None or job.status != JobStatus.SUCCEEDED.value:
            continue
        try:
            if evaluate_fn is not None:
                ev = evaluate_fn(tp, run_id, protocol, model_id, spec=spec_task,
                                 resources=resources, synthetic=synthetic)
            else:
                ev = evaluate_run(tp, run_id, protocol, model_id=model_id,
                                  resources=resources, synthetic=synthetic)
        except (FileNotFoundError, ValueError, TypeError) as exc:
            ctx.repo.append_event(task_id, "stage.evaluate", f"evaluation skipped: {exc}", level="warning")
            continue
        ev.task_id = task_id
        ev.stage = protocol.stage
        ev.failures = acceptance_failures(spec_task, ev)
        ev.passed = not ev.failures
        experiment = ctx.repo.get_experiment(run_id)
        result = read_json(tp.runs / run_id / "result.json", {})
        fields = {"code_ref": experiment.code_ref,
                  "env": "local_cpu", "weights_ref": ""}
        if manifest_fn is not None:
            fields.update(dict(manifest_fn(spec_task, tp, run_id, protocol) or {}))
        weights_ref, weights_hash = "", ""
        if result.get("weights_ref"):
            weights = safe_join(tp.runs / run_id, result["weights_ref"])
            if weights.is_file():
                weights_ref = str(weights.relative_to(tp.root))
                weights_hash = sha256_file(weights)
        if spec_task.auto_process and not weights_ref:
            ev.failures.append("trained model artifact missing; predictions are not weights")
            ev.passed = False
        ctx.repo.save_eval(ev)
        manifest = C.ModelManifest(
            model_id=model_id, task_id=task_id, run_id=run_id,
            code_ref=experiment.code_ref, data_version=state["dataset_version"],
            config_ref=f"model_pool/{c.method}/config.yaml", env=fields["env"],
            seed=plan.data_recipe.seed, weights_ref=weights_ref,
            weights_hash=weights_hash, export={"format": "trained_model" if weights_ref else "predictions_only"},
            status="candidate")
        ctx.repo.save_model(manifest)
        model_ids.append(model_id)
        eval_ids.append(ev.eval_id)
        collected.append({"model_id": model_id, "metrics": ev.metrics, "slices": ev.slices,
                          "passed": ev.passed, "failures": ev.failures})
        last_metrics = ev.metrics
        score = ev.metrics.get(protocol.primary_metric)
        if score is not None and math.isfinite(float(score)):
            ranked = (ev.passed, -len(ev.failures), -score if protocol.direction == "min" else score)
            if best_metric is None or ranked > best_metric:
                best_metric, best = ranked, model_id
        ctx.repo.append_event(task_id, "stage.evaluate",
                              f"{model_id}: {protocol.primary_metric}={score} passed={ev.passed}",
                              data={"metrics": ev.metrics, "slices": ev.slices, "eval_id": ev.eval_id})

    # A domain plugin may rank candidates on all of its acceptance gates
    # (e.g. IQA needs Macro-F1 + PLCC + hardcase + ROM), not just the single
    # primary metric the generic loop tracks.
    if select_best_fn is not None and collected and not spec_task.auto_process:
        # Pool this iteration's candidates with every previously evaluated model
        # so a weaker later iteration cannot silently replace a better earlier
        # one (regression guard). Newest eval per model wins.
        pool: Dict[str, Dict[str, Any]] = {}
        for e in ctx.repo.list_evals(task_id):
            pool[e.model_id] = {"model_id": e.model_id, "metrics": e.metrics, "slices": e.slices,
                                "passed": e.passed, "failures": e.failures}
        for c in collected:
            pool[c["model_id"]] = c
        try:
            picked = select_best_fn(protocol, list(pool.values()))
        except Exception as exc:
            picked = None
            ctx.repo.append_event(task_id, "stage.evaluate",
                                  f"select_best hook failed: {exc}", level="warning")
        if picked:
            best = picked

    if spec_task.auto_process and collected:
        current_fp = evaluation_fingerprint(tp, protocol.dataset_version)
        for ev in ctx.repo.list_evals(task_id):
            if ev.stage != "development" or evaluation_fingerprint(tp, ev.dataset_version) != current_fp:
                continue
            score = ev.metrics.get(protocol.primary_metric)
            if score is None or not math.isfinite(score):
                continue
            ranked = (ev.passed, -len(ev.failures), -score if protocol.direction == "min" else score)
            if best_metric is None or ranked > best_metric:
                best_metric, best = ranked, ev.model_id

    if best:
        ctx.repo.set_recommended(task_id, best)
        ctx.repo.update_state(task_id, best_model_id=best, phase=TaskPhase.ANALYZING.value)
    return {"stage": "evaluate", "model_ids": model_ids, "eval_ids": eval_ids,
            "best_feasible_model": best, "last_metrics": last_metrics}


def evaluation_fingerprint(tp, version: str) -> str:
    manifest = read_json(tp.release_dir(version) / "manifest.json", {})
    checksums = manifest.get("checksums", {})
    return checksums.get("protocol/development_labels.npz", checksums.get("protocol/eval_labels.jsonl", version))


def evaluate_final(state: TaskState, ctx) -> C.EvalResult:
    """One holdout use per task; only reload the exact selected weights."""
    from ...execution.data_pipeline import prepare_final
    task_id = state["task_id"]
    spec = ctx.repo.get_spec(task_id)
    tp = TaskPaths(ctx.settings, state["task_slug"])
    model = ctx.repo.get_model(state["best_feasible_model"])
    if model is None or not model.weights_ref:
        raise ValueError("final evaluation requires a trained model artifact")
    # A retry may recover an already completed final evaluation, never rerun it.
    existing = [e for e in ctx.repo.list_evals(task_id, model.model_id) if e.stage == "final"]
    if existing:
        return existing[-1]
    ctx.repo.claim_budget(task_id, "final_evaluations", 1, 1)
    weights = safe_join(tp.root, model.weights_ref)
    if sha256_file(weights) != model.weights_hash:
        raise ValueError("selected model weights changed")
    experiment = ctx.repo.get_experiment(model.run_id)
    original_job = next(j for j in ctx.repo.list_jobs(task_id) if j.run_id == model.run_id)
    for rel, digest in original_job.payload.get("code_hashes", {}).items():
        if sha256_file(safe_join(tp.runs / model.run_id / "code", rel)) != digest:
            raise ValueError("selected experiment code changed before final evaluation")
    plan = ctx.repo.get_plan(experiment.plan_id)
    version = prepare_final(tp, spec, model.data_version)
    protocol = plan.eval_protocol.model_copy(update={"stage": "final", "dataset_version": version,
                                                     "protocol_id": plan.eval_protocol.protocol_id + "_final"})
    run_id = model.run_id + "_final"
    final = experiment.model_copy(update={
        "run_id": run_id, "parent_run_id": model.run_id, "data_version": version,
        "env": {**experiment.env, "stage": "final", "inference_weights": str(weights)}})
    job = ctx.runner.submit(final, tp.slug, str(safe_join(tp.root, model.code_ref)), split="test")
    if ctx.inline:
        ctx.worker.drain(task_id=task_id)
    else:
        ctx.worker.await_jobs([job.job_id], timeout=max(1, remaining_seconds(spec)))
    check_running(ctx.repo, task_id)
    if ctx.repo.get_job(job.job_id).status != "succeeded":
        raise ValueError("one-shot final inference failed; holdout cannot be reused in this task")
    if sha256_file(weights) != model.weights_hash:
        raise ValueError("final inference modified the selected weights")
    plugin = resolve_plugin(spec)
    evaluator = hook(plugin, "evaluate") or evaluate_run
    kwargs = {"spec": spec} if hook(plugin, "evaluate") else {}
    result = evaluator(tp, run_id, protocol, model.model_id, synthetic=spec.synthetic,
                       resources=_experiment_resources(plugin, spec), **kwargs)
    result.task_id, result.stage = task_id, "final"
    result.failures = acceptance_failures(spec, result)
    result.passed = not result.failures
    ctx.repo.save_eval(result)
    ctx.repo.append_event(task_id, "stage.final_evaluation", "one-shot holdout evaluation completed",
                          data={"eval_id": result.eval_id, "passed": result.passed,
                                "weights_hash": model.weights_hash})
    return result


__all__ = ["submit_experiments_node", "collect_results_node", "evaluate_node"]
