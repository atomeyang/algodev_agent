"""Typed intent proposals with deterministic authorization of state-changing actions."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class IntentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["query", "status", "pause", "resume", "cancel", "improve", "clarify"] = "query"
    confidence: float = Field(default=1, ge=0, le=1)
    reason: str = ""
    goal: str = ""
    source: str = "rules"


_QUESTION = re.compile(r"[?？]|如何|怎么|为什么|是否|能否|解释|说明|what |why |how |can you", re.I)
_NEGATED_CONTROL = re.compile(
    r"(不要|别|不必|不需要|do not|don't|never)\s*(暂停|停止|取消|恢复|pause|stop|cancel|resume)", re.I)
_IMPROVE = re.compile(
    r"^(?:请|帮我|麻烦|please\s+)?\s*(?:继续|进一步)?\s*"
    r"(?:优化|改进|尝试|试验|研究|调整|增加|训练|improve\b|optimi[sz]e\b|try\b|research\b)", re.I)
_CONTROLS = {
    "pause": r"暂停|pause",
    "resume": r"恢复|继续运行|resume",
    "cancel": r"取消|停止|cancel|stop",
}


def route_intent(text: str, proposal=None) -> IntentDecision:
    text = text.strip()
    if not text or _QUESTION.search(text) or _NEGATED_CONTROL.search(text):
        return IntentDecision(action="query")
    for action, pattern in _CONTROLS.items():
        if re.fullmatch(
            rf"(?:请|帮我|please\s+)?\s*(?:{pattern})\s*"
            r"(?:(?:当前|这个|本|the current|this)\s*)?(?:任务|训练|实验|task|run|training)?"
            r"\s*(?:吧|一下)?[。.!！]*", text, re.I):
            return IntentDecision(action=action)
    if re.fullmatch(r"(?:查看|显示|show\s+)?(?:当前)?(?:状态|进度|status)[。.!！]*", text, re.I):
        return IntentDecision(action="status")
    if _IMPROVE.search(text):
        return IntentDecision(action="improve", goal=text)
    if proposal is not None:
        try:
            proposed = IntentDecision.model_validate(proposal)
        except (ValueError, TypeError):
            return IntentDecision(action="clarify", confidence=0, reason="invalid intent proposal")
        if proposed.confidence < 0.85:
            return IntentDecision(action="clarify", confidence=proposed.confidence, reason="ambiguous intent")
        # A model cannot authorize approvals, target changes, cross-task operations
        # or ambiguous destructive commands. The original utterance grants authority.
        if proposed.action == "improve" and re.search(r"请|帮我|继续|尝试|please|let.s", text, re.I):
            return proposed.model_copy(update={"goal": text, "source": "model"})
        if proposed.action in {"query", "status", "clarify"}:
            return proposed.model_copy(update={"source": "model"})
        return IntentDecision(action="clarify", reason="use an explicit control command")
    return IntentDecision(action="query", reason="no explicit operation")
