"""Programmatic DOCX generation. Structured events + asset manifests are the
source of truth; the cloud only contributes research/analysis prose (guide 14).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..config import Settings
from ..paths import TaskPaths, read_json
from ..schemas import contracts as C
from ..storage.repository import Repository
from .. import __version__
from .docx_utils import add_kv_table, add_markdown, new_document, save


def _task_id(repo: Repository, slug: str) -> str:
    return next((row["task_id"] for row in repo.list_tasks() if row["task_slug"] == slug), slug)


def _task_summary_rows(repo: Repository, slug: str) -> List[Dict[str, Any]]:
    task_id = _task_id(repo, slug)
    state = repo.get_state(task_id)
    spec = repo.get_spec(task_id)
    return [
        {"field": "task_id", "value": spec.task_id},
        {"field": "task_slug", "value": spec.task_slug},
        {"field": "business", "value": spec.business},
        {"field": "phase", "value": state.phase},
        {"field": "iteration", "value": state.iteration},
        {"field": "state_version", "value": state.state_version},
        {"field": "active_plan", "value": state.active_plan_id},
        {"field": "best_model", "value": state.best_model_id},
        {"field": "stop_reason", "value": state.stop_reason},
        {"field": "primary_metric", "value": f"{spec.primary_metric.name} ({spec.primary_metric.direction}) "
                                             f"threshold={spec.primary_metric.threshold}"},
        {"field": "unknowns", "value": "; ".join(spec.unknowns)},
        {"field": "synthetic_demo", "value": spec.synthetic or spec.demo},
    ]


def export_task_overview(settings: Settings, repo: Repository, slug: str) -> str:
    task_id = _task_id(repo, slug)
    spec = repo.get_spec(task_id)
    tp = TaskPaths(settings, slug)
    doc = new_document(f"{slug} - Algorithm R&D Overview")
    doc.add_heading("Task summary", level=1)
    add_kv_table(doc, _task_summary_rows(repo, slug), ["field", "value"])

    doc.add_heading("Plans", level=1)
    add_kv_table(doc, repo.list_plans(task_id),
                 ["plan_id", "version", "parent_plan_id", "content_hash", "status", "created_at"])

    doc.add_heading("Models", level=1)
    models = repo.list_models(task_id)
    add_kv_table(doc, [{"model_id": m.model_id, "run_id": m.run_id, "status": m.status,
                        "recommended": m.recommended, "data_version": m.data_version,
                        "weights_ref": m.weights_ref} for m in models],
                 ["model_id", "run_id", "status", "recommended", "data_version", "weights_ref"])

    doc.add_heading("Evaluations", level=1)
    evals = repo.list_evals(task_id)
    metric_names = list(dict.fromkeys([spec.primary_metric.name, *[m.name for m in spec.secondary_metrics]]))
    add_kv_table(doc, [{"eval_id": e.eval_id, "model_id": e.model_id, "protocol": e.protocol_id,
                        "stage": e.stage, **{m: e.metrics.get(m) for m in metric_names},
                        "passed": e.passed, "synthetic": e.synthetic} for e in evals],
                 ["eval_id", "model_id", "protocol", "stage", *metric_names, "passed", "synthetic"])

    doc.add_heading("Recent events", level=1)
    events = repo.read_events(task_id, after=0, limit=10000)[-40:]
    add_kv_table(doc, [{"event_seq": e["event_seq"], "kind": e["kind"], "message": e["message"],
                        "created_at": e["created_at"]} for e in events],
                 ["event_seq", "kind", "message", "created_at"])

    doc.add_heading("Feedback", level=1)
    add_kv_table(doc, [{"feedback_id": f.feedback_id, "model_id": f.model_id, "scenario": f.scenario,
                        "actual": f.actual, "reproduced": f.reproduced} for f in repo.list_feedback(task_id)],
                 ["feedback_id", "model_id", "scenario", "actual", "reproduced"])

    doc.add_paragraph("Structured events, the asset registry and the database are the source of truth. "
                      "Offline/synthetic results are labelled and are not business validation.")
    return save(doc, tp.overview)


def export_iteration_report(settings: Settings, repo: Repository, slug: str, iteration: int) -> str:
    tp = TaskPaths(settings, slug)
    doc = new_document(f"{slug} - Iteration {iteration} Report")
    research_path = tp.docs / "research" / f"research_iter{iteration}.md"
    if research_path.exists():
        doc.add_heading("Research & design", level=1)
        add_markdown(doc, research_path.read_text(encoding="utf-8"))
    packet = read_json(tp.docs / "iterations" / f"packet_iter{iteration}.json", {})
    if packet:
        doc.add_heading("Evidence packet", level=1)
        add_kv_table(doc, [{"field": "packet_id", "value": packet.get("packet_id")},
                           {"field": "baseline_id", "value": packet.get("baseline_id")},
                           {"field": "metric_delta", "value": packet.get("metric_delta")},
                           {"field": "failures", "value": packet.get("failures")},
                           {"field": "unknowns", "value": packet.get("unknowns")}],
                     ["field", "value"])
        if packet.get("slices"):
            doc.add_heading("Slices", level=2)
            add_kv_table(doc, [{"slice": k, **v} for k, v in packet["slices"].items()])
    doc.add_heading("Models & evaluations", level=1)
    add_kv_table(doc, [{"model_id": m.model_id, "run_id": m.run_id, "status": m.status}
                       for m in repo.list_models(_task_id(repo, slug))
                       if m.data_version == f"v{iteration}"], ["model_id", "run_id", "status"])
    out = tp.docs / "iterations" / f"iteration_{iteration}.docx"
    return save(doc, out)


def export_plan_docs(settings: Settings, repo: Repository, slug: str, plan_id: str) -> Dict[str, str]:
    tp = TaskPaths(settings, slug)
    plan = repo.get_plan(plan_id)
    if not plan:
        return {}
    pdir = tp.plan_dir(plan.version)
    doc = new_document(f"{slug} - Plan {plan.version} Proposal")
    doc.add_heading("Summary", level=1)
    doc.add_paragraph(plan.summary or "(no summary)")
    doc.add_heading("Candidates", level=1)
    add_kv_table(doc, [{"candidate_id": c.candidate_id, "method": c.method, "family": c.family,
                        "description": c.description, "expected_gain": c.expected_gain,
                        "cost": c.cost_estimate, "baseline": c.is_baseline} for c in plan.candidates],
                 ["candidate_id", "method", "family", "description", "expected_gain", "cost", "baseline"])
    doc.add_heading("Hypotheses & controls", level=1)
    for h in plan.hypotheses:
        doc.add_paragraph(h, style="List Bullet")
    for c in plan.controls:
        doc.add_paragraph(c, style="List Bullet")
    doc.add_heading("Evaluation protocol", level=1)
    add_kv_table(doc, [{"field": "protocol_id", "value": plan.eval_protocol.protocol_id},
                       {"field": "primary_metric", "value": plan.eval_protocol.primary_metric},
                       {"field": "direction", "value": plan.eval_protocol.direction},
                       {"field": "threshold", "value": plan.eval_protocol.threshold},
                       {"field": "slices", "value": ", ".join(plan.eval_protocol.slices)},
                       {"field": "frozen", "value": plan.eval_protocol.frozen}], ["field", "value"])
    doc.add_heading("Execution DAG", level=1)
    add_kv_table(doc, [{"step": s.step_id, "kind": s.kind, "depends_on": ", ".join(s.depends_on)}
                       for s in plan.execution_dag], ["step", "kind", "depends_on"])
    doc.add_heading("Approval scope", level=1)
    add_kv_table(doc, [{"field": k, "value": str(v)} for k, v in plan.change_scope.model_dump().items()],
                 ["field", "value"])
    proposal = save(doc, pdir / "proposal.docx")

    notes = new_document(f"{slug} - Plan {plan.version} Release Notes")
    notes.add_heading("Changes in this plan", level=1)
    notes.add_paragraph(f"Plan {plan.plan_id} ({plan.version}), parent={plan.parent_plan_id or 'none'}.")
    for c in plan.candidates:
        notes.add_paragraph(f"{c.method}: {c.description}", style="List Bullet")
    rel = save(notes, pdir / "release_notes.docx")
    return {"proposal": proposal, "release_notes": rel}


def export_dataset_doc(settings: Settings, repo: Repository, slug: str,
                       dataset_id: str = "demo_tabular", version: str = "v1") -> str:
    tp = TaskPaths(settings, slug)
    manifest = repo.get_dataset(dataset_id, version)
    doc = new_document(f"{slug} - Dataset {dataset_id}:{version}")
    if manifest:
        add_kv_table(doc, [{"field": k, "value": str(v)} for k, v in manifest.model_dump().items()],
                     ["field", "value"])
    else:
        doc.add_paragraph("(dataset manifest not found)")
    return save(doc, tp.data / "DATASET.docx")


def export_research_docx(settings: Settings, repo: Repository, slug: str, iteration: int) -> str:
    tp = TaskPaths(settings, slug)
    src = tp.docs / "research" / f"research_iter{iteration}.md"
    doc = new_document(f"{slug} - Research Report (iteration {iteration})")
    add_markdown(doc, src.read_text(encoding="utf-8") if src.exists() else "(no report)")
    return save(doc, tp.docs / "research" / f"research_iter{iteration}.docx")


def export_system_status(settings: Settings, repo: Repository) -> str:
    from ..storage.db import DB_SCHEMA_VERSION
    doc = new_document("Algo-Agent System Status")
    doc.add_heading("Platform", level=1)
    add_kv_table(doc, [{"field": "root", "value": str(settings.root)},
                       {"field": "package_version", "value": __version__},
                       {"field": "db_schema_version", "value": str(DB_SCHEMA_VERSION)},
                       {"field": "workflow_config", "value": str(settings.config_path)}], ["field", "value"])
    doc.add_heading("Milestones", level=1)
    add_kv_table(doc, [
        {"milestone": "M0 installable skeleton", "status": "implemented + offline verified"},
        {"milestone": "M1 dialogue control", "status": "implemented + offline verified"},
        {"milestone": "M2 offline closed loop", "status": "implemented + offline verified (synthetic_demo)"},
        {"milestone": "M3 real tools (cloud/local/search/Codex)", "status": "adapters implemented; "
         "blocked on credentials/hardware - not accepted"},
        {"milestone": "M4 problem-driven iteration", "status": "offline evidence chain verified"},
        {"milestone": "M5 recovery & delivery", "status": "recovery/budget/upload/doc implemented; "
         "real business acceptance pending"},
    ], ["milestone", "status"])
    doc.add_heading("Tasks", level=1)
    add_kv_table(doc, repo.list_tasks(), ["task_id", "task_slug", "phase", "iteration",
                                          "state_version", "updated_at"])
    doc.add_paragraph("Real cloud/GPU/business acceptance is NOT claimed. Offline demo results are "
                      "labelled synthetic_demo. See 14_required_acceptance and the milestones doc.")
    return save(doc, settings.docs_dir / "agent_status.docx")


def export_version_notes(settings: Settings, repo: Repository, version: str = "v1.0") -> str:
    doc = new_document(f"Algo-Agent Version Notes {version}")
    doc.add_heading("Scope of this implementation", level=1)
    doc.add_paragraph("Implements the Codex development guide v1.2 against architecture design v0.4: "
                      "conversational CLI, control service, LangGraph lifecycle with interrupt-based "
                      "approval, research-design and Codex coding sub-agents, evidence service, "
                      "asset registry, reproduction/migration interfaces, token budget and DOCX records.")
    doc.add_heading("Verified offline", level=1)
    for line in ["M0-M2 end-to-end offline loop on deterministic synthetic tabular data",
                 "approval hash invalidation, idempotent job submission, data immutability",
                 "feedback version attribution, budget control, run recovery"]:
        doc.add_paragraph(line, style="List Bullet")
    doc.add_heading("Pending real acceptance", level=1)
    for line in ["cloud research provider requires ALGO_CLOUD_API_KEY",
                 "local model requires ALGO_LOCAL_BASE_URL",
                 "runtime Codex requires ALGO_CODEX_ALLOW_REAL=1 and CLI auth",
                 "target-hardware latency/ROM metrics cannot be measured without the device"]:
        doc.add_paragraph(line, style="List Bullet")
    return save(doc, settings.docs_dir / f"version_notes_{version}.docx")


__all__ = [
    "export_task_overview", "export_iteration_report", "export_plan_docs", "export_dataset_doc",
    "export_research_docx", "export_system_status", "export_version_notes",
]
