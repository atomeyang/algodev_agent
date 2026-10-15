"""Build and compile the LangGraph research lifecycle."""

from __future__ import annotations

from functools import partial
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from . import nodes as N
from .context import GraphContext
from .routing import (
    route_after_approval,
    route_after_assets,
    route_after_decide,
    route_after_diagnose,
    route_after_evaluate,
    route_after_feedback_gate,
    route_after_intake,
)
from .state import TaskState
from ..policy import check_running

STAGES = ["intake", "research", "design_plan", "approval_gate", "prepare_assets", "coding",
          "submit_experiments", "collect_results", "evaluate", "analyze", "decide",
          "feedback_gate", "diagnose_feedback", "export_docs"]


def build_graph(ctx: GraphContext, checkpointer: Optional[Any] = None):
    g = StateGraph(TaskState)
    def guarded(fn):
        def run(state):
            check_running(ctx.repo, state["task_id"])
            return fn(state, ctx=ctx)
        return run
    for name, fn in [
        ("intake", N.intake_node), ("research", N.research_node),
        ("design_plan", N.design_plan_node), ("approval_gate", N.approval_node),
        ("prepare_assets", N.prepare_assets_node), ("coding", N.coding_node),
        ("submit_experiments", N.submit_experiments_node), ("collect_results", N.collect_results_node),
        ("evaluate", N.evaluate_node), ("analyze", N.analyze_node), ("decide", N.decide_node),
        ("feedback_gate", N.feedback_gate_node), ("diagnose_feedback", N.diagnose_feedback_node),
    ]:
        g.add_node(name, guarded(fn))
    g.add_node("export_docs", partial(N.export_docs_node, ctx=ctx))

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route_after_intake, {"research": "research", "END": END})
    g.add_edge("research", "design_plan")
    g.add_edge("design_plan", "approval_gate")
    g.add_conditional_edges("approval_gate", route_after_approval,
                            {"prepare_assets": "prepare_assets", "END": END})
    g.add_conditional_edges("prepare_assets", route_after_assets,
                            {"coding": "coding", "END": END, "analyze": "analyze"})
    g.add_edge("coding", "submit_experiments")
    g.add_edge("submit_experiments", "collect_results")
    g.add_edge("collect_results", "evaluate")
    g.add_conditional_edges("evaluate", route_after_evaluate, {"analyze": "analyze", "decide": "decide"})
    g.add_edge("analyze", "decide")
    g.add_conditional_edges("decide", route_after_decide, {"export_docs": "export_docs",
                                                           "feedback_gate": "feedback_gate"})
    g.add_conditional_edges("feedback_gate", route_after_feedback_gate,
                            {"diagnose_feedback": "diagnose_feedback", "export_docs": "export_docs"})
    g.add_conditional_edges("diagnose_feedback", route_after_diagnose, {"research": "research"})
    g.add_edge("export_docs", END)

    return g.compile(checkpointer=checkpointer)


__all__ = ["build_graph", "STAGES"]
