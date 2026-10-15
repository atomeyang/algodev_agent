"""Recovery: a fresh Controller on the same root sees persisted task + graph state."""

from __future__ import annotations

from algo_agent.service.controller import Controller

EXTRAS = {"methods_hint": ["baseline_mean", "ridge"], "pause_between_iterations": True}


def test_new_controller_reads_the_persisted_task(settings, demo_report):
    fresh = Controller(settings, inline=True, extras=dict(EXTRAS))
    state = fresh.state("demo_task")
    assert state["task"]["task_id"] == "demo_task"
    assert state["phase"] not in {"", None}


def test_new_controller_reads_the_checkpointed_graph_state(settings, demo_report):
    fresh = Controller(settings, inline=True, extras=dict(EXTRAS))
    snapshot = fresh.rt.graph.get_state({"configurable": {"thread_id": "demo_task"}})
    assert snapshot.values.get("task_id") == "demo_task"
    # the demo completed the loop locally, so the checkpoint chain is not empty
    assert snapshot.values.get("stage")


def test_recovery_does_not_re_execute_a_finished_task(settings, demo_report):
    fresh = Controller(settings, inline=True, extras=dict(EXTRAS))
    before = len(fresh.repo.list_models("demo_task"))
    state = fresh.state("demo_task")
    assert state["phase"] in {"accepted", "stopped", "paused", "executing"}
    assert len(fresh.repo.list_models("demo_task")) == before
