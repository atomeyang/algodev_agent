"""Small regressions for failures observed during the real IQA dialogue trial."""

import json

from algo_agent.config import Settings
from algo_agent.providers.deepseek import DeepSeekResearchProvider
from algo_agent.schemas import contracts as C
from algo_agent.service.controller import Controller


def test_research_prompt_exposes_exact_schema_and_rejects_truncation(tmp_path, monkeypatch):
    provider = DeepSeekResearchProvider(Settings(tmp_path, env={"ALGO_CLOUD_API_KEY": "fake"}))
    monkeypatch.setattr(provider, "_domain_grounding", lambda request: None)
    calls = []

    def chat(system, user, **kwargs):
        calls.append((system, json.loads(user), kwargs))
        return {"content": '{"candidates":[{"method":"ridge"}]}', "finish_reason": "length"}

    monkeypatch.setattr(provider, "_chat", chat)
    result = provider.design({"task_spec": {"auto_process": True},
                              "reference_code_files": ["algorithm_plugins/iqa/methods/common.py"]})
    assert result.status == "failed"
    assert "truncated" in result.unknowns[0]
    schema = calls[0][1]["output_schema"]
    assert schema["properties"]["hypotheses"]["items"]["type"] == "string"
    assert "candidate_schema" in calls[0][1]
    assert calls[0][2]["max_tokens"] == 16384
    assert calls[0][1]["reference_code_files"]


def test_first_research_data_preview_does_not_open_final_labels(tmp_path, monkeypatch):
    from algorithm_plugins.iqa import data, config
    from algo_agent.agents.research_design import ResearchDesignRequest
    calls = []
    monkeypatch.setenv("IQA_BENCHMARK", "pseudo")
    monkeypatch.setattr(config, "dev_jsonl", lambda: tmp_path / "dev.jsonl")

    def rows(path):
        calls.append(path.name)
        assert path.name == "dev.jsonl"
        return [{"subject": str(i), "path": f"private-{i}", "score": 50 + i,
                 "defects": {"blur": {"present": i % 2}}} for i in range(12)]

    monkeypatch.setattr(data, "load_rows", rows)
    ctrl = Controller(Settings(tmp_path, env={"ALGO_OFFLINE": "1"}))
    spec = ctrl.create_task("", "preview", task_spec={
        "domain": "iqa", "primary_metric": {"name": "plcc", "threshold": .85}})
    result = ctrl.rt.research_agent._tool(
        ResearchDesignRequest(task_spec=spec, task_slug=spec.task_slug), "inspect_data", {}, "test")
    preview = result["source_preview"]
    assert preview["splits"]["train"]["count"] + preview["splits"]["development_test"]["count"] == 12
    assert not preview["final_labels_inspected"]
    assert "private-" not in json.dumps(result)
    assert calls == ["dev.jsonl"]


def test_dialogue_context_has_limits_controls_and_stop_reason(tmp_path):
    ctrl = Controller(Settings(tmp_path, env={"ALGO_OFFLINE": "1"}))
    spec = ctrl.create_task("", "context", task_spec={
        "business": "regression", "primary_metric": {"name": "rmse", "threshold": .5},
        "secondary_metrics": [{"name": "mae", "threshold": .25}],
        "unknowns": ["packaged ROM not measured"], "demo": True, "synthetic": True})
    ctrl.repo.update_state(spec.task_id, phase="stopped", stop_reason="research output truncated")
    context = ctrl._answer_context(spec.task_id)
    assert "max_wall_clock_s" in context
    assert "暂停任务" in context
    assert '"mae"' in context and "0.25" in context
    assert "research output truncated" in context
    assert "packaged ROM not measured" in context


def test_failed_research_is_archived_before_task_halts(tmp_path):
    import pytest
    from algo_agent.graph.nodes.research import research_node
    from algo_agent.paths import TaskPaths, read_json
    from algo_agent.policy import TaskHalted
    from algo_agent.providers.base import ResearchDesignResult, SearchResult

    ctrl = Controller(Settings(tmp_path, env={"ALGO_OFFLINE": "1"}))
    spec = ctrl.create_task("", "failure", task_spec={
        "business": "regression", "demo": True, "synthetic": True,
        "primary_metric": {"name": "rmse", "threshold": .5}})

    class Provider:
        name = "fixture"

        def search_sources(self, *args):
            return SearchResult()

        def design(self, request):
            return ResearchDesignResult(status="blocked", unknowns=["cannot form a valid plan"])

    ctrl.rt.research_agent.provider = Provider()
    with pytest.raises(TaskHalted):
        research_node({"task_id": spec.task_id, "task_slug": spec.task_slug, "iteration": 1}, ctrl.rt.ctx)
    record = read_json(TaskPaths(ctrl.settings, spec.task_slug).docs / "research/research_iter1.json")
    assert record["status"] == "blocked"
    assert record["tool_calls"][0]["tool"] == "search_sources"


def test_tool_batch_reserves_final_plan_synthesis(tmp_path):
    from algo_agent.agents.research_design import ResearchDesignRequest
    from algo_agent.providers.base import ResearchDesignResult, SearchResult
    ctrl = Controller(Settings(tmp_path, env={"ALGO_OFFLINE": "1"}))
    spec = ctrl.create_task("", "bounded", auto_process=True, task_spec={
        "business": "regression", "demo": True, "synthetic": True,
        "primary_metric": {"name": "rmse", "threshold": .5}, "budget": {"max_research_steps": 4}})
    calls = []

    class Provider:
        name = "fixture"

        def search_sources(self, *args):
            return SearchResult()

        def design(self, request):
            calls.append(dict(request))
            if len(calls) == 1:
                return ResearchDesignResult(status="needs_tools", extras={
                    "tool_requests": [{"tool": "inspect_data"} for _ in range(10)]})
            assert request["available_tools"] == {}
            assert request["finalize_now"]
            return ResearchDesignResult(candidates=[C.Candidate(candidate_id="r", method="ridge")])

    ctrl.rt.research_agent.provider = Provider()
    result = ctrl.rt.research_agent.run(ResearchDesignRequest(
        task_spec=spec, task_slug=spec.task_slug, budget=spec.budget))
    assert result.status == "completed"
    assert len(calls) == 2
    assert sum(t["tool"] == "inspect_data" for t in result.tool_calls) == 1


def test_data_plan_checks_aggregate_cost_and_remaining_budget(tmp_path):
    from algo_agent.graph.nodes.design_plan import build_plan_spec
    from algo_agent.policy import data_budget_problems, plan_problems
    settings = Settings(tmp_path, env={"ALGO_OFFLINE": "1"})
    spec = C.TaskSpec(task_id="b", task_slug="b", auto_process=True,
                      primary_metric=C.MetricTarget(name="rmse", threshold=.5),
                      budget=C.Budget(max_added_samples=240))
    result = {"candidates": [{"candidate_id": "r", "method": "ridge"}],
              "extras": {"data_operations": [
                  {"operation_id": "flip", "kind": "augment", "hypothesis": "test", "params": {"count": 240}},
                  {"operation_id": "sample", "kind": "resample", "hypothesis": "test", "params": {"count": 24}},
              ]}}
    plan = build_plan_spec(spec, result, 1, "", settings)
    assert any("added_samples=264" in p for p in plan_problems(spec, plan))
    plan.data_recipe.operations = plan.data_recipe.operations[:1]
    assert not data_budget_problems(spec, plan)
    assert data_budget_problems(spec, plan, used={"added_samples": 1})


def test_live_autonomous_report_does_not_inherit_offline_claim():
    from algo_agent.providers.base import ResearchDesignResult
    base = ResearchDesignResult(report_markdown="OFFLINE MODE: no cloud was used")
    text = DeepSeekResearchProvider._merge_report(base, "New live design", "model", autonomous=True)
    assert "OFFLINE MODE" not in text
    assert "New live design" in text


def test_research_restart_replays_budget_snapshot_without_rebilling(tmp_path):
    import pytest
    from algo_agent.agents.research_design import ResearchDesignRequest
    from algo_agent.policy import TaskHalted
    from algo_agent.providers.base import ResearchDesignResult, SearchResult

    ctrl = Controller(Settings(tmp_path, env={"ALGO_OFFLINE": "1"}))
    spec = ctrl.create_task("", "replay", auto_process=True, task_spec={
        "business": "regression", "demo": True, "synthetic": True,
        "primary_metric": {"name": "rmse", "threshold": .5}})
    calls = []

    class Provider:
        name, model = "fixture", "fixture-model"
        halt_once = True

        def search_sources(self, *args):
            return SearchResult()

        def _chat(self, *args, **kwargs):
            calls.append(args)
            return {"content": "inspect data", "usage": {"total_tokens": 12}}

        def design(self, request):
            if request["observations"]:
                if self.halt_once:
                    self.halt_once = False
                    raise TaskHalted("simulated process interruption after the tool")
                return ResearchDesignResult(candidates=[C.Candidate(candidate_id="r", method="ridge")])
            request["_chat"]("system", json.dumps(request["budget_ledger"]), max_tokens=20)
            return ResearchDesignResult(status="needs_tools", extras={
                "tool_requests": [{"tool": "inspect_data"}]})

    ctrl.rt.research_agent.provider = Provider()
    request = ResearchDesignRequest(task_spec=spec, task_slug=spec.task_slug, budget=spec.budget)
    with pytest.raises(TaskHalted):
        ctrl.rt.research_agent.run(request)
    assert ctrl.repo.get_budget(spec.task_id)["used"]["cloud_tokens"] == 12
    assert ctrl.rt.research_agent.run(request).status == "completed"
    assert len(calls) == 1
    assert ctrl.repo.get_budget(spec.task_id)["used"]["cloud_tokens"] == 12


def test_codex_namespace_failure_is_blocked_even_with_zero_cli_exit(tmp_path, monkeypatch):
    from algo_agent.providers import codex_cli

    class Child:
        returncode = 0

        def __init__(self, *args, **kwargs):
            pass

        def communicate(self, timeout):
            return json.dumps({"type": "item.completed", "item": {
                "type": "command_execution", "exit_code": 1,
                "aggregated_output": "bwrap: No permissions to create new namespace"}}) + "\n", ""

    monkeypatch.setattr(codex_cli.subprocess, "Popen", Child)
    monkeypatch.setattr(codex_cli.subprocess, "run",
                        lambda *a, **kw: type("Completed", (), {"stdout": ""})())
    executor = codex_cli.CodexCliExecutor(Settings(tmp_path), allow_real=True)
    result = executor.execute(
        C.CodingTaskSpec(coding_id="c", task_id="t", plan_id="p", objective="test"),
        {"workspace": str(tmp_path)})
    assert result.status == "blocked"
    assert "sandbox unavailable" in result.open_issues[0]


def test_blocked_coding_does_not_consume_repair_retries(tmp_path):
    from algo_agent.providers.base import CodingOutcome, ProviderStatus
    from algo_agent.paths import TaskPaths, read_json

    ctrl = Controller(Settings(tmp_path, env={"ALGO_OFFLINE": "1"}))
    spec = ctrl.create_task("", "blocked_code", task_spec={
        "business": "regression", "demo": True, "synthetic": True,
        "primary_metric": {"name": "rmse", "threshold": .5}})
    calls = []

    class Executor:
        name = "fixture_codex"

        def status(self):
            return ProviderStatus(self.name, True, "coding")

        def execute(self, *args):
            calls.append(1)
            return CodingOutcome(status="blocked", tool="codex_cli",
                                 open_issues=["execution sandbox unavailable"])

    ctrl.rt.coding_agent.executor = Executor()
    request = C.CodingTaskSpec(
        coding_id="blocked", task_id=spec.task_id, plan_id="p", objective="test",
        write_paths=["model_pool/new"], entry="model_pool/new/method.py",
        require_real=True, budget=spec.budget)
    result = ctrl.rt.coding_agent.submit(request, {"task_slug": spec.task_slug})
    assert result.status == "blocked"
    assert len(calls) == 1
    assert not ctrl.repo.get_budget(spec.task_id)["used"].get("fix_attempts")
    saved = read_json(TaskPaths(ctrl.settings, spec.task_slug).coding_records / "blocked/result.json")
    assert "execution sandbox unavailable" in saved["open_issues"]
