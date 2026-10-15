"""Autonomous lifecycle regressions. All providers are local test doubles."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from algo_agent.agents import ResearchDesignRequest
from algo_agent.config import Settings
from algo_agent.execution.data_pipeline import prepare_development, verify_release
from algo_agent.graph.nodes.design_plan import build_plan_spec
from algo_agent.paths import TaskPaths, read_json, sha256_file, write_json
from algo_agent.policy import acceptance_failures, plan_problems
from algo_agent.providers.base import CodingOutcome, ProviderStatus, ResearchDesignResult
from algo_agent.providers.offline import OfflineResearchProvider
from algo_agent.schemas import contracts as C
from algo_agent.service.controller import Controller, ControllerError
from algo_agent.storage.repository import BudgetExceeded


@pytest.fixture()
def ctrl(tmp_path):
    settings = Settings(root=tmp_path, env={"ALGO_OFFLINE": "1", "ALGO_CODEX_ALLOW_REAL": "0"})
    return Controller(settings, extras={"pause_between_iterations": True})


def _task(ctrl, tid="auto", threshold=0.5, **budget):
    return ctrl.create_task("", tid, auto_process=True, task_spec={
        "task_id": tid, "task_slug": tid, "business": "autonomous regression",
        "demo": True, "synthetic": True,
        "primary_metric": {"name": "rmse", "direction": "min", "threshold": threshold},
        "budget": {"max_coding_calls": 8, **budget},
    })


def _plan(ctrl, spec, operations=()):
    provider = OfflineResearchProvider()
    result = provider.design({"task_spec": spec.model_dump(), "iteration": 1})
    research = {"candidates": [c.model_dump() for c in result.candidates],
                "data_recipe": result.data_recipe.model_dump(),
                "eval_protocol": result.eval_protocol.model_dump(),
                "extras": {"data_operations": list(operations)}}
    return build_plan_spec(spec, research, 1, "", ctrl.settings)


def test_auto_multiround_reloads_weights_on_one_shot_holdout(ctrl):
    spec = _task(ctrl)
    boundary = ctrl.run(spec.task_id)
    assert boundary["done"] and boundary["state"]["phase"] == "accepted", boundary
    assert ctrl.pending_interrupt(spec.task_id) is None
    assert len(ctrl.repo.list_plans(spec.task_id)) >= 2
    evaluations = ctrl.repo.list_evals(spec.task_id)
    finals = [e for e in evaluations if e.stage == "final"]
    assert len(finals) == 1 and finals[0].passed
    model = ctrl.repo.get_model(finals[0].model_id)
    tp = TaskPaths(ctrl.settings, spec.task_slug)
    assert sha256_file(tp.root / model.weights_ref) == model.weights_hash
    assert read_json(tp.runs / f"{model.run_id}_final" / "result.json")["inference_only"]
    assert not (tp.release_dir(f"{model.data_version}_final") / "train.npz").exists()
    jobs_before = len(ctrl.repo.list_jobs(spec.task_id))
    ctrl.run(spec.task_id)
    assert len(ctrl.repo.list_jobs(spec.task_id)) == jobs_before


def test_no_successful_runs_never_accept(ctrl):
    spec = _task(ctrl, max_iterations=1)
    spec.methods_hint = ["does_not_exist"]
    ctrl.repo.save_spec(spec)
    boundary = ctrl.run(spec.task_id)
    assert boundary["state"]["phase"] == "stopped"
    assert not ctrl.repo.list_evals(spec.task_id)
    assert not ctrl.repo.list_models(spec.task_id)


def test_zero_research_budget_prevents_provider_call(ctrl, monkeypatch):
    spec = _task(ctrl, max_research_calls=0)
    monkeypatch.setattr(ctrl.rt.research_agent.provider, "search_sources",
                        lambda *a: pytest.fail("research started with zero budget"))
    assert ctrl.run(spec.task_id)["state"]["phase"] == "stopped"
    assert not ctrl.repo.list_plans(spec.task_id)
    assert ctrl.repo.get_budget(spec.task_id)["used"].get("research_calls", 0) == 0


def test_plan_and_model_ids_are_task_scoped(ctrl):
    first = _task(ctrl, "first", threshold=10, max_iterations=1)
    second = _task(ctrl, "second", threshold=10, max_iterations=1)
    p1, p2 = _plan(ctrl, first), _plan(ctrl, second)
    assert p1.plan_id != p2.plan_id
    for spec in (first, second):
        ctrl.run(spec.task_id)
    ids = [{m.model_id for m in ctrl.repo.list_models(s.task_id)} for s in (first, second)]
    assert ids[0] and ids[1] and not ids[0] & ids[1]
    assert ctrl.repo.get_plan(p1.plan_id).task_id == first.task_id


def test_pause_does_not_restart_or_approve(ctrl):
    spec = ctrl.create_task("rmse <= 1", "manual", demo=True, synthetic=True)
    first = ctrl.run(spec.task_id)
    pending = first["interrupt"]
    ctrl.pause(spec.task_id)
    with pytest.raises(ControllerError, match="paused"):
        ctrl.approve(spec.task_id, pending["plan_id"], pending["plan_hash"])
    assert ctrl.run(spec.task_id)["type"] == "paused"
    ctrl.resume_task(spec.task_id)
    again = ctrl.run(spec.task_id)
    assert again["interrupt"]["plan_id"] == pending["plan_id"]
    assert len(ctrl.repo.list_plans(spec.task_id)) == 1


def test_data_experiments_are_versioned_and_holdout_is_label_free(ctrl):
    spec = _task(ctrl)
    plan = _plan(ctrl, spec, [
        {"operation_id": "boost_low", "kind": "resample", "split": "train",
         "hypothesis": "more low-light examples improve slice error", "params": {"count": 500, "slice": "low_light"}},
        {"operation_id": "jitter", "kind": "augment", "split": "train",
         "hypothesis": "small jitter regularises", "params": {"count": 40, "noise_std": 0.001}},
    ])
    tp = TaskPaths(ctrl.settings, spec.task_slug)
    manifest = prepare_development(tp, spec, plan, "v1", ctrl.repo)
    assert manifest.splits["train"] == 540
    assert len(manifest.provenance) == 2
    assert len(np.load(tp.release_dir("_auto_base") / "train.npz")["y"]) == 360
    assert np.all(np.load(tp.release_dir("v1") / "test.npz")["y"] == 0)
    before = ctrl.repo.get_budget(spec.task_id)
    prepare_development(tp, spec, plan, "v1", ctrl.repo)
    assert ctrl.repo.get_budget(spec.task_id) == before
    (tp.release_dir("v1") / "recipe.json").write_text("{}")
    with pytest.raises(ValueError, match="immutable"):
        verify_release(tp, "v1")


def test_final_labels_cannot_be_generated_or_filtered(ctrl):
    spec = _task(ctrl)
    plan = _plan(ctrl, spec, [{"operation_id": "cheat", "kind": "annotate", "split": "final_test",
                              "hypothesis": "generate labels"}])
    assert any("independently labelled" in p for p in plan_problems(spec, plan))
    with pytest.raises(ValueError, match="final test"):
        prepare_development(TaskPaths(ctrl.settings, spec.task_slug), spec, plan, "v1", ctrl.repo)


def test_budget_reservation_is_atomic(ctrl):
    spec = _task(ctrl)
    successes = []
    def reserve():
        try:
            ctrl.repo.claim_budget(spec.task_id, "experiments", 1, 1)
            successes.append(True)
        except BudgetExceeded:
            pass
    threads = [threading.Thread(target=reserve) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(successes) == 1


def test_research_can_choose_tools_and_read_results(ctrl):
    spec = _task(ctrl)
    base = OfflineResearchProvider()
    class Research:
        seen = []
        search_sources = base.search_sources
        def design(self, request):
            self.seen.append(request.copy())
            if len(self.seen) == 1:
                return ResearchDesignResult(status="needs_tools",
                                            extras={"tool_requests": [{"tool": "inspect_data", "arguments": {}}]})
            assert request["observations"][0]["result"] == {"releases": []}
            return base.design(request)
    provider = Research()
    ctrl.rt.research_agent.provider = provider
    result = ctrl.rt.research_agent.run(ResearchDesignRequest(
        task_spec=spec, task_slug=spec.task_slug, budget=spec.budget))
    assert result.status == "completed" and len(provider.seen) == 2
    assert any(call["tool"] == "inspect_data" for call in result.tool_calls)


def test_research_tool_loop_is_bounded(ctrl):
    spec = _task(ctrl, max_research_steps=2)
    base = OfflineResearchProvider()
    class Research:
        search_sources = base.search_sources
        def design(self, request):
            return ResearchDesignResult(status="needs_tools", extras={
                "tool_requests": [{"tool": "inspect_data", "arguments": {}}]})
    ctrl.rt.research_agent.provider = Research()
    result = ctrl.rt.research_agent.run(ResearchDesignRequest(
        task_spec=spec, task_slug=spec.task_slug, budget=spec.budget))
    assert result.status == "blocked"


def test_coding_harvests_actual_diff_and_repairs_failed_verification(ctrl):
    spec = _task(ctrl, max_fix_attempts=1)
    class Executor:
        name = "test_coder"
        calls = 0
        def status(self):
            return ProviderStatus(self.name, True)
        def execute(self, task, context):
            self.calls += 1
            target = Path(context["workspace"]) / "model_pool/new_method/method.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("broken (" if self.calls == 1 else "def run(ctx, config):\n    return {}\n")
            return CodingOutcome(status="completed", changed_files=[], tool=self.name)
    executor = Executor()
    ctrl.rt.coding_agent.executor = executor
    task = C.CodingTaskSpec(
        coding_id="new", task_id=spec.task_id, plan_id="p", objective="implement",
        write_paths=["model_pool/new_method"], entry="model_pool/new_method/method.py",
        require_real=True, budget=spec.budget,
        acceptance_commands=["python -c \"import ast,pathlib; ast.parse(pathlib.Path('model_pool/new_method/method.py').read_text())\""])
    result = ctrl.rt.coding_agent.submit(task, {"task_slug": spec.task_slug})
    assert result.status == "completed" and executor.calls == 2
    assert result.changed_files == ["model_pool/new_method/method.py"]
    assert ctrl.repo.get_budget(spec.task_id)["used"]["fix_attempts"] == 1


def test_scope_prefix_escape_blocks_entire_coding_result(ctrl):
    spec = _task(ctrl, max_fix_attempts=0)
    class Executor:
        name = "test_coder"
        def status(self):
            return ProviderStatus(self.name, True)
        def execute(self, task, context):
            root = Path(context["workspace"])
            for folder in ("model_pool/new_method", "model_pool/new_method_evil"):
                path = root / folder / "method.py"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("def run(ctx, config): return {}")
            return CodingOutcome(status="completed")
    ctrl.rt.coding_agent.executor = Executor()
    task = C.CodingTaskSpec(coding_id="escape", task_id=spec.task_id, plan_id="p", objective="test",
                            write_paths=["model_pool/new_method"], budget=spec.budget,
                            acceptance_commands=["python -c 'pass'"])
    result = ctrl.rt.coding_agent.submit(task, {"task_slug": spec.task_slug})
    assert result.status == "failed" and not result.changed_files
    assert not (TaskPaths(ctrl.settings, spec.task_slug).model_pool / "new_method/method.py").exists()


def test_nan_or_missing_measurement_fails_acceptance(ctrl):
    spec = _task(ctrl)
    assert acceptance_failures(spec, None)
    result = C.EvalResult(eval_id="e", task_id=spec.task_id, model_id="m", protocol_id="p",
                          metrics={"rmse": float("nan")}, passed=True)
    assert acceptance_failures(spec, result)


def test_cli_auto_process_requires_acceptance_and_returns_nonzero(tmp_path):
    from algo_agent.cli.main import app
    spec = tmp_path / "task.json"
    write_json(spec, {"task_id": "cli", "task_slug": "cli", "demo": True, "synthetic": True,
                      "primary_metric": {"name": "rmse", "threshold": 0.1},
                      "budget": {"max_research_calls": 0}})
    result = CliRunner().invoke(app, ["--root", str(tmp_path / "root"), "--embedded", "--json",
                                     "auto-process", "--spec", str(spec)],
                                env={"ALGO_OFFLINE": "1"})
    assert result.exit_code == 2, result.output
    assert '"phase": "stopped"' in result.output


def test_cancellation_terminates_running_process(ctrl):
    spec = _task(ctrl)
    tp = TaskPaths(ctrl.settings, spec.task_slug)
    method = tp.model_pool / "sleep/method.py"
    method.parent.mkdir(parents=True)
    method.write_text("import time\ndef run(ctx, config):\n    time.sleep(60)\n    return {}\n")
    plan = _plan(ctrl, spec)
    prepare_development(tp, spec, plan, "v1", ctrl.repo)
    experiment = C.ExperimentSpec(run_id="sleep", task_id=spec.task_id, plan_id=plan.plan_id,
                                  candidate_id="sleep", method="sleep", code_ref=str(method), data_version="v1")
    job = ctrl.rt.runner.submit(experiment, spec.task_slug, str(method))
    leased = ctrl.repo.lease_job(spec.task_id, "test")
    thread = threading.Thread(target=ctrl.rt.runner.execute_job, args=(leased,))
    thread.start()
    deadline = time.time() + 10
    while not ctrl.repo.get_job(job.job_id).pid and time.time() < deadline:
        time.sleep(0.05)
    ctrl.cancel(spec.task_id)
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert ctrl.repo.get_job(job.job_id).status == "cancelled"


def test_new_method_is_coded_executed_and_validated(ctrl):
    spec = _task(ctrl)
    base = OfflineResearchProvider()
    class Research:
        search_sources = base.search_sources
        def design(self, request):
            result = base.design(request)
            result.candidates = [C.Candidate(
                candidate_id="new", method="new_algorithm", action="create",
                implementation_goal="fit a train-only interaction model and persist inference weights")]
            return result
    class Executor:
        name = "test_code_generator"
        def status(self):
            return ProviderStatus(self.name, True)
        def execute(self, task, context):
            target = Path(context["workspace"]) / task.entry
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "from algorithm_plugins.demo_tabular.methods import run as train, predict\n"
                "def run(ctx, config):\n    return train('ridge_interaction', ctx, config)\n")
            return CodingOutcome(status="completed", tool=self.name)
    ctrl.rt.research_agent.provider = Research()
    ctrl.rt.coding_agent.executor = Executor()
    boundary = ctrl.run(spec.task_id)
    assert boundary["state"]["phase"] == "accepted", boundary
    assert ctrl.repo.get_budget(spec.task_id)["used"]["coding_calls"] == 1
    assert any("new_algorithm" in m.model_id for m in ctrl.repo.list_models(spec.task_id))


def test_final_failure_is_terminal_and_not_researched(ctrl, monkeypatch):
    spec = _task(ctrl, threshold=10)
    execute = ctrl.rt.runner.execute_job
    def broken_final(job):
        result = execute(job)
        if job.payload.get("stage") == "final":
            tp = TaskPaths(ctrl.settings, spec.task_slug)
            path = tp.runs / job.run_id / "pred_test.npy"
            np.save(path, np.load(path) + 100)
        return result
    monkeypatch.setattr(ctrl.rt.runner, "execute_job", broken_final)
    boundary = ctrl.run(spec.task_id)
    assert boundary["state"]["phase"] == "stopped"
    assert "holdout failed" in boundary["state"]["stop_reason"]
    assert ctrl.repo.get_budget(spec.task_id)["used"]["research_calls"] == 1
    assert len([e for e in ctrl.repo.list_evals(spec.task_id) if e.stage == "final"]) == 1
    from algo_agent.graph.nodes.research import _history_with_metrics
    history = _history_with_metrics(ctrl.rt.ctx, spec.task_id, spec)
    assert history[0]["metrics"]["rmse"] < 10


def test_failed_data_experiment_drives_a_new_plan(ctrl):
    spec = _task(ctrl, threshold=10)
    base = OfflineResearchProvider()
    class Research:
        search_sources = base.search_sources
        def design(self, request):
            result = base.design(request)
            if request["iteration"] == 1:
                result.extras["data_operations"] = [
                    {"operation_id": "bad_noise", "kind": "augment", "hypothesis": "test invalid noise",
                     "params": {"count": 1, "noise_std": -1}}]
            else:
                assert "noise_std" in request["history"][0]["data_failure"]["error"]
            return result
    ctrl.rt.research_agent.provider = Research()
    boundary = ctrl.run(spec.task_id)
    assert boundary["state"]["phase"] == "accepted", boundary
    assert len(ctrl.repo.list_plans(spec.task_id)) == 2


def test_http_persists_auto_mode_and_structured_budget(ctrl, monkeypatch):
    from fastapi.testclient import TestClient
    from algo_agent.service import app as service
    monkeypatch.setattr(service, "_CONTROLLER", ctrl)
    with TestClient(service.create_app(ctrl.settings)) as client:
        response = client.post("/tasks", json={
            "task_id": "api", "auto_process": True, "run": False,
            "task_spec": {"task_id": "api", "task_slug": "api",
                          "primary_metric": {"name": "rmse", "threshold": 0.3},
                          "budget": {"max_research_calls": 1}, "synthetic": True}})
        assert response.status_code == 200
        stored = ctrl.repo.get_spec("api")
        assert stored.auto_process and stored.budget.max_research_calls == 1
        assert client.get("/tasks/api/state").json()["task"]["auto_process"]


def test_cloud_budget_is_reserved_before_call(ctrl, monkeypatch):
    from algo_agent.providers.deepseek import DeepSeekResearchProvider
    spec = _task(ctrl, max_cloud_tokens=10)
    provider = DeepSeekResearchProvider(Settings(root=ctrl.settings.root, env={"ALGO_CLOUD_API_KEY": "test"}))
    monkeypatch.setattr(provider, "_chat", lambda *a, **k: pytest.fail("unbudgeted request"))
    ctrl.rt.research_agent.provider = provider
    with pytest.raises(BudgetExceeded):
        ctrl.rt.research_agent.run(ResearchDesignRequest(task_spec=spec, task_slug=spec.task_slug,
                                                        budget=spec.budget))


def test_cloud_can_propose_novel_methods_and_data_operations(ctrl, monkeypatch):
    from algo_agent.providers.deepseek import DeepSeekResearchProvider
    spec = _task(ctrl)
    provider = DeepSeekResearchProvider(Settings(root=ctrl.settings.root, env={"ALGO_CLOUD_API_KEY": "test"}))
    monkeypatch.setattr(provider, "_chat", lambda *a, **k: {
        "content": json.dumps({"candidates": [{"method": "novel", "action": "create",
                                              "implementation_goal": "new model with export"}],
                              "data_operations": [{"operation_id": "balance", "kind": "resample",
                                                   "hypothesis": "improve rare slice", "params": {"count": 400}}]}),
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}})
    ctrl.rt.research_agent.provider = provider
    result = ctrl.rt.research_agent.run(ResearchDesignRequest(
        task_spec=spec, task_slug=spec.task_slug, budget=spec.budget))
    assert result.candidates[0].method == "novel" and result.candidates[0].action == "create"
    assert result.extras["data_operations"][0]["kind"] == "resample"
    assert ctrl.repo.get_budget(spec.task_id)["used"]["cloud_tokens"] == 7
    assert ctrl.repo.get_budget(spec.task_id)["reserved"]["cloud_tokens"] == 0


def test_import_side_effect_is_checked_again_before_applying(ctrl):
    spec = _task(ctrl, max_fix_attempts=0)
    class Executor:
        name = "test_coder"
        def status(self):
            return ProviderStatus(self.name, True)
        def execute(self, task, context):
            target = Path(context["workspace"]) / "model_pool/side_effect/method.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("from pathlib import Path\nPath('coding_contract.json').write_text('{}')\n")
            return CodingOutcome(status="completed")
    ctrl.rt.coding_agent.executor = Executor()
    result = ctrl.rt.coding_agent.submit(C.CodingTaskSpec(
        coding_id="side_effect", task_id=spec.task_id, plan_id="p", objective="test", budget=spec.budget,
        write_paths=["model_pool/side_effect"],
        acceptance_commands=["python model_pool/side_effect/method.py"]), {"task_slug": spec.task_slug})
    assert result.status == "failed" and not result.changed_files


def test_auto_iqa_does_not_inherit_legacy_zero_coding_budget(ctrl, monkeypatch):
    monkeypatch.setenv("IQA_BENCHMARK", "koniq")
    spec = ctrl.create_task("IQA primary metric PLCC 0.85", "iqa_budget", domain="iqa", auto_process=True)
    assert spec.budget.max_coding_calls > 0


def test_unknown_blocking_constraint_cannot_hide_behind_a_plugin(ctrl, monkeypatch):
    monkeypatch.setenv("IQA_BENCHMARK", "koniq")
    spec = _task(ctrl)
    spec.domain = "iqa"
    spec.hard_constraints = [C.HardConstraint(name="unmeasured_mobile_latency", value=12)]
    result = C.EvalResult(eval_id="e", task_id=spec.task_id, model_id="m", protocol_id="p",
                          metrics={"rmse": 0.1}, passed=True)
    assert any("no constraint validator" in f for f in acceptance_failures(spec, result))
