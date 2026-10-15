"""Chat driving: free text answers (never silently becomes feedback), /improve wakes the loop.

These lock in the behaviour that the operator-facing REPL relies on:
* plain text is a *question* answered by the live chat model,
* ``/improve`` on a live task submits feedback to wake diagnose -> research -> design,
* ``/improve`` on a terminal task starts a new task that references the best model,
  so a finished graph is never resumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from algo_agent.cli.chat import ChatSession
from algo_agent.cli.client import ClientError
from algo_agent.config import Settings


class FakeClient:
    """Minimal stand-in exposing exactly what ChatSession touches."""

    def __init__(self, *, phase: str = "accepted", best: str = "m_best",
                 answer: Optional[Dict[str, Any]] = None,
                 feedback: Optional[Dict[str, Any]] = None):
        self.state_ = {"task_id": "t1", "phase": phase, "best_model_id": best,
                       "active_plan_id": None, "pending_approval": None,
                       "paused": False, "iteration": 0}
        self.answer_result = answer if answer is not None else {
            "answer": "grounded live answer", "source": "deepseek",
            "model": "deepseek-flash", "offline": False}
        self.feedback_result = feedback or {"feedback_id": "fb_1", "continued": False}
        self.calls: List[str] = []
        self.answer_calls: List[str] = []
        self.feedback_calls: List[Dict[str, Any]] = []
        self.created: List[Dict[str, Any]] = []

    def state(self, task_id: str) -> Dict[str, Any]:
        self.calls.append("state")
        return dict(self.state_)

    def evals(self, task_id: str):
        return [{"model_id": self.state_["best_model_id"], "passed": True,
                 "metrics": {"plcc": 0.8886, "srcc": 0.8712}}]

    def plan_view(self, task_id: str, plan_id: str):
        return {"plan": {"summary": "s"}, "plan_hash": "h"}

    def answer(self, task_id: str, question: str) -> Dict[str, Any]:
        self.answer_calls.append(question)
        return dict(self.answer_result)

    def submit_feedback(self, task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.feedback_calls.append(payload)
        return dict(self.feedback_result)

    def create_task(self, text: str, *, demo: bool = False, synthetic: bool = False,
                    methods_hint=None) -> Dict[str, Any]:
        self.created.append({"text": text, "demo": demo, "methods_hint": methods_hint})
        self.state_["task_id"] = "t2"
        self.state_["phase"] = "research"
        return {"task_id": "t2", "task_slug": "t2"}

    def run(self, task_id: str, *, raw_text: str = "") -> Dict[str, Any]:
        return {"type": "end", "interrupt": None, "state": self.state_, "done": True}


def _session(client: FakeClient, *, task_id: Optional[str] = "t1") -> ChatSession:
    import tempfile
    root = Path(tempfile.mkdtemp())
    return ChatSession(Settings(root=root, workflow={}, env={}), client, task_id=task_id)


def test_free_text_asks_the_live_model_and_never_records_feedback():
    client = FakeClient()
    s = _session(client)
    s._free_text("现在状态如何了,实现了哪些优化了?")
    assert client.answer_calls == ["现在状态如何了,实现了哪些优化了?"]
    assert client.feedback_calls == []          # regression: was silently saved as feedback


def test_key_value_text_is_still_explicit_feedback():
    client = FakeClient()
    s = _session(client)
    s._free_text("scenario=improvement expected=higher_plcc model_id=m_best")
    assert client.feedback_calls and client.feedback_calls[0]["scenario"] == "improvement"
    assert client.answer_calls == []


def test_improve_on_live_task_submits_feedback_with_best_model():
    client = FakeClient(phase="research", best="m_best")
    s = _session(client)
    s._improve("用更强的头把 PLCC 提到 0.90")
    assert client.created == []                  # not a terminal task -> no new task
    assert len(client.feedback_calls) == 1
    payload = client.feedback_calls[0]
    assert payload["model_id"] == "m_best"
    assert payload["scenario"] == "improvement"


def test_improve_on_terminal_task_starts_a_new_task_referencing_best():
    client = FakeClient(phase="accepted", best="m_mobileclip2_mos_head")
    s = _session(client)
    s._improve("加 LoRA + rank loss 继续提 PLCC")
    assert len(client.created) == 1
    req = client.created[0]["text"]
    assert "m_mobileclip2_mos_head" in req
    assert "LoRA" in req
    assert s.task_id == "t2"                      # switched to the new task
    assert client.feedback_calls == []           # never resumes/touches the finished graph


def test_llm_answer_falls_back_when_provider_reports_no_answer():
    client = FakeClient(answer={"answer": "", "source": "offline", "offline": True})
    s = _session(client)
    assert s._llm_answer("你好") is False          # caller then uses deterministic grounding
    assert client.answer_calls == ["你好"]


def test_controller_answer_is_offline_without_a_chat_provider(controller):
    """With no cloud/local credentials the controller must say so, not invent a reply."""
    res = controller.answer("demo_task", "现在怎么样了?")
    assert res["answer"] == ""
    assert res["offline"] is True
    assert res["source"] == "offline"
