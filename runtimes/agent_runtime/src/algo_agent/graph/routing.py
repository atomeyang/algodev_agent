"""Conditional routing between stage nodes (guide 6)."""

from __future__ import annotations

from ..schemas.enums import TaskPhase
from .state import TaskState


def route_after_intake(state: TaskState) -> str:
    return "END" if state.get("decision") == "needs_input" else "research"


def route_after_approval(state: TaskState) -> str:
    return "prepare_assets" if state.get("approved") else "END"


def route_after_assets(state: TaskState) -> str:
    if state.get("decision") == "data_failed":
        return "analyze"
    return "END" if state.get("decision") == "needs_input" else "coding"


def route_after_evaluate(state: TaskState) -> str:
    return "analyze"


def route_after_decide(state: TaskState) -> str:
    decision = state.get("decision")
    if decision in {"accepted", "stop"}:
        return "export_docs"
    return "feedback_gate"


def route_after_feedback_gate(state: TaskState) -> str:
    if state.get("decision") == "stop":
        return "export_docs"
    return "diagnose_feedback"


def route_after_diagnose(state: TaskState) -> str:
    return "research"


__all__ = [
    "route_after_intake", "route_after_approval", "route_after_assets", "route_after_evaluate",
    "route_after_decide", "route_after_feedback_gate", "route_after_diagnose",
]
