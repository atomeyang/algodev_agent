"""Approval gate: frozen plan hashes and stale-approval rejection (guide 3, 6, 11).

The graph freezes a plan and interrupts. The control layer must refuse any
approval whose hash or expected state version does not match, and it must not be
possible to approve when the graph is not actually waiting at the gate.
"""

from __future__ import annotations

import uuid

import pytest

from algo_agent.service.controller import ControllerError

REQ = "predict a tabular quality score, RMSE <= 0.5, desktop CPU"
HINTS = ["baseline_mean", "ridge"]


@pytest.fixture()
def pending(controller):
    task_id = f"appr_{uuid.uuid4().hex[:8]}"
    controller.create_task(REQ, task_id=task_id, synthetic=True, methods_hint=HINTS)
    boundary = controller.run(task_id)
    assert boundary["type"] == "approval_request", boundary
    info = boundary["interrupt"]
    yield {"task_id": task_id, "plan_id": info["plan_id"], "plan_hash": info["plan_hash"]}
    # end the interrupted thread deterministically instead of leaking it
    try:
        controller.reject(task_id, info["plan_id"], reason="test cleanup")
    except Exception:
        pass


def test_approval_requires_the_frozen_plan_hash(controller, pending):
    with pytest.raises(ControllerError) as exc:
        controller.approve(pending["task_id"], pending["plan_id"], "deadbeef")
    assert exc.value.code == "stale_plan_hash"
    assert exc.value.status == 409
    assert "hash" in exc.value.recovery.lower()


def test_approval_requires_the_expected_state_version(controller, pending):
    with pytest.raises(ControllerError) as exc:
        controller.approve(pending["task_id"], pending["plan_id"], pending["plan_hash"],
                           expected_state_version=9999)
    assert exc.value.code == "state_conflict"
    assert exc.value.status == 409


def test_approval_of_unknown_plan_is_not_found(controller, pending):
    with pytest.raises(ControllerError) as exc:
        controller.approve(pending["task_id"], "plan_does_not_exist", pending["plan_hash"])
    assert exc.value.code == "unknown_plan"
    assert exc.value.status == 404


def test_plan_hash_is_stable_across_views(controller, pending):
    view = controller.plan_view(pending["task_id"], pending["plan_id"])
    assert view["plan_hash"] == pending["plan_hash"]
    assert controller.repo.get_plan_hash(pending["plan_id"]) == pending["plan_hash"]


def test_approval_outside_a_gate_is_rejected(controller):
    task_id = f"appr_nogate_{uuid.uuid4().hex[:8]}"
    controller.create_task("a task that never started", task_id=task_id, synthetic=True)
    plans = controller.repo.list_plans(task_id)
    plan = plans[-1] if plans else None
    # no plan and no interrupt: either way the gate must refuse
    with pytest.raises(ControllerError) as exc:
        controller.approve(task_id, plan["plan_id"] if plan else "x", "hash")
    assert exc.value.code in {"unknown_plan", "not_at_approval"}
    assert exc.value.status in {404, 409}
