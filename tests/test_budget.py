"""Budget accounting and stop rules (guide 9, 10)."""

from __future__ import annotations

import uuid

from algo_agent.graph.nodes.analysis import decide_node
from algo_agent.schemas.enums import TaskPhase


def test_reserve_and_settle_are_bookkept(repo):
    task_id = f"budget_{uuid.uuid4().hex[:8]}"
    repo.reserve_budget(task_id, "llm_tokens", 100.0)
    assert repo.get_budget(task_id)["reserved"]["llm_tokens"] == 100.0

    repo.settle_budget(task_id, "llm_tokens", used=30.0, reserved=100.0)
    budget = repo.get_budget(task_id)
    assert budget["reserved"].get("llm_tokens", 0.0) == 0.0
    assert budget["used"]["llm_tokens"] == 30.0


def test_unknown_usage_is_flagged_not_zeroed(repo):
    task_id = f"budget_{uuid.uuid4().hex[:8]}"
    repo.settle_budget(task_id, "gpu_hours", used=None)
    used = repo.get_budget(task_id)["used"]
    assert used["gpu_hours__unknown"] is True
    assert "gpu_hours" not in used


def test_ledger_entries_are_listed(repo):
    task_id = f"budget_{uuid.uuid4().hex[:8]}"
    entry_id = repo.add_ledger_entry(task_id, "researcher", "llm", 120, 40, 0.02, "USD",
                                     "prices-2026-01", {"note": "unit"})
    rows = repo.list_ledger(task_id)
    assert [r["entry_id"] for r in rows] == [entry_id]
    assert rows[0]["tokens_in"] == 120


def test_controller_budget_exposes_limits_and_usage(controller, demo_report):
    budget = controller.budget("demo_task")
    assert budget["limits"]["max_iterations"] == 6
    assert set(budget) >= {"limits", "used", "reserved", "ledger"}


def test_decide_node_continues_below_the_iteration_limit(controller):
    task_id = f"decide_{uuid.uuid4().hex[:8]}"
    controller.create_task("decide stop-rule probe", task_id=task_id, synthetic=True)
    out = decide_node({"task_id": task_id, "iteration": 1,
                       "needs_plan_change": [{"reason": "target not met"}]}, controller.rt.ctx)
    assert out["decision"] == "continue"
    assert out["stop_reason"] is None


def test_decide_node_honours_spec_max_iterations(controller):
    task_id = f"decide_{uuid.uuid4().hex[:8]}"
    spec = controller.create_task("decide stop-rule probe", task_id=task_id, synthetic=True)
    limit = spec.budget.max_iterations
    out = decide_node({"task_id": task_id, "iteration": limit,
                       "needs_plan_change": [{"reason": "target not met"}]}, controller.rt.ctx)
    assert out["decision"] == "stop"
    assert "max_iterations" in out["stop_reason"]
    assert controller.repo.get_state(task_id).phase == TaskPhase.STOPPED.value


def test_decide_node_stops_after_no_improve_limit(controller):
    task_id = f"decide_{uuid.uuid4().hex[:8]}"
    spec = controller.create_task("decide stop-rule probe", task_id=task_id, synthetic=True)
    out = decide_node({"task_id": task_id, "iteration": 2,
                       "needs_plan_change": [{"reason": "stuck"}],
                       "no_improve": spec.budget.max_no_improve}, controller.rt.ctx)
    assert out["decision"] == "stop"
    assert "no improvement" in out["stop_reason"]
