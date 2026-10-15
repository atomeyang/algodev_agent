"""User feedback must be attributed to an explicit model version (guide 8).

No default to the newest model: an ambiguous report is stored with a marker and
returns the candidate list, and an unknown model id is refused.
"""

from __future__ import annotations

import uuid

import pytest

from algo_agent.schemas import contracts as C
from algo_agent.service.controller import ControllerError


@pytest.fixture()
def feedback_task(controller):
    task_id = f"fb_{uuid.uuid4().hex[:8]}"
    controller.create_task("feedback attribution probe task", task_id=task_id, synthetic=True)
    controller.repo.save_model(C.ModelManifest(model_id="m1", task_id=task_id, run_id="r1"))
    controller.repo.save_model(C.ModelManifest(model_id="m2", task_id=task_id, run_id="r2"))
    return task_id


def test_missing_model_id_is_not_defaulted(controller, feedback_task):
    result = controller.submit_feedback(feedback_task, {"scenario": "low light"}, auto_continue=False)
    assert result["needs_model_confirmation"] is True
    assert set(result["model_candidates"]) == {"m1", "m2"}
    assert result["model_id"] is None
    stored = [f for f in controller.repo.list_feedback(feedback_task)]
    assert stored and stored[-1].missing_evidence


def test_unknown_model_id_is_refused(controller, feedback_task):
    with pytest.raises(ControllerError) as exc:
        controller.submit_feedback(feedback_task, {"model_id": "ghost"}, auto_continue=False)
    assert exc.value.code == "unknown_model"
    assert exc.value.status == 409


def test_valid_feedback_round_trips_and_stays_unverified(controller, feedback_task):
    result = controller.submit_feedback(
        feedback_task,
        {"model_id": "m1", "scenario": "low light", "expected": "0.4", "actual": "0.8",
         "severity": "high", "repro": "python eval.py --slice low_light"},
        auto_continue=False)
    assert result["queued"] is True
    assert result["needs_model_confirmation"] is False
    stored = controller.repo.list_feedback(feedback_task)[-1]
    assert stored.model_id == "m1"
    assert stored.severity == "high"
    assert "low_light" in stored.repro
    # a report is not a verified reproduction until it is actually re-run
    assert stored.reproduced is None
