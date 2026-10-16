"""Controller: the single task-state writer for start/approve/pause/resume/cancel.

It validates the plan hash, the state version and the approval scope *before*
resuming the LangGraph interrupt, so a stale client can never approve a changed
plan (guide 3, 6, 11).
"""

from __future__ import annotations

import threading
import uuid
import json
from typing import Any, Dict, List, Optional

from ..agents import ResearchDesignRequest
from ..config import Settings
from ..graph import initial_state
from ..graph.builder import STAGES
from ..paths import TaskPaths, ensure_task_dirs, register_task_member, sha256_obj, slugify
from ..schemas import contracts as C
from ..schemas.enums import ApprovalMode, Intent, JobStatus, ResultStatus, TaskPhase
from ..storage.repository import Repository, StateConflict, new_id
from ..policy import TaskHalted, check_running, task_key
from ..storage.repository import BudgetExceeded
from .runtime import Runtime, build_runtime

TERMINAL_JOB = {JobStatus.SUCCEEDED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value}


class ControllerError(RuntimeError):
    def __init__(self, code: str, message: str, *, recovery: str = "", status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.recovery = recovery
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "recovery": self.recovery}}


class Controller:
    def __init__(self, settings: Settings, inline: bool = True,
                 extras: Optional[Dict[str, Any]] = None):
        self.settings = settings
        self.rt: Runtime = build_runtime(settings, inline=inline, extras=extras)
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @property
    def repo(self) -> Repository:
        return self.rt.repo

    def _cfg(self, task_id: str) -> Dict[str, Any]:
        return {"configurable": {"thread_id": task_id},
                "recursion_limit": 30 + 16 * self.repo.get_spec(task_id).budget.max_iterations}

    def pending_interrupt(self, task_id: str) -> Optional[Dict[str, Any]]:
        try:
            snap = self.rt.graph.get_state(self._cfg(task_id))
        except Exception:
            return None
        for task in getattr(snap, "tasks", ()) or ():
            for itr in getattr(task, "interrupts", ()) or ():
                return getattr(itr, "value", None)
        return None

    def at_end(self, task_id: str) -> bool:
        try:
            snap = self.rt.graph.get_state(self._cfg(task_id))
        except Exception:
            return True
        return not snap.next

    # ------------------------------------------------------------------ #
    # task lifecycle
    # ------------------------------------------------------------------ #
    def create_task(self, text: str, task_id: Optional[str] = None, *, task_slug: str = "",
                    demo: bool = False, synthetic: bool = False, domain: str = "",
                    methods_hint: Optional[List[str]] = None, auto_process: bool = False,
                    task_spec: Optional[Dict[str, Any]] = None,
                    parent_task: str = "", track: str = "") -> C.TaskSpec:
        from ..providers.offline import OfflineResearchProvider  # noqa: F401  (kept for clarity)
        from ..paths import slugify

        tid = task_id or (task_spec or {}).get("task_id") or slugify(text[:24])
        slug = task_slug or (task_spec or {}).get("task_slug") or slugify(tid)
        if slug != slugify(slug) or slug.endswith("_experiment"):
            raise ControllerError("invalid_slug", "task slug must be a normalised identifier")
        if self.repo.task_exists(tid):
            # explicit recovery, never silent overwrite (guide 4)
            spec = self.repo.get_spec(tid)
            if text and spec.business != text:
                raise ControllerError(
                    "task_exists",
                    f"task {tid} already exists with a different requirement",
                    recovery=f"use: algo-agent chat --task {tid}", status=409)
            if auto_process and not spec.auto_process:
                if self.repo.list_plans(tid):
                    raise ControllerError(
                        "legacy_task", "create a new task for auto-process; existing manual plans may use test feedback",
                        recovery="use a new task ID and an independent holdout", status=409)
                spec.auto_process = True
                self.repo.save_spec(spec)
            return spec
        if any(t["task_slug"] == slug for t in self.repo.list_tasks()):
            slug = f"{slug[:32]}_{task_key(tid)}"
        # parse structured fields from the text
        from ..graph.nodes.intake import parse_requirement
        spec = (C.TaskSpec(**{**task_spec, "task_id": tid, "task_slug": slug})
                if task_spec else parse_requirement(text, tid, slug, demo=demo, synthetic=synthetic, domain=domain))
        spec.auto_process = auto_process or spec.auto_process
        spec.budget = C.Budget(**{**self.settings.budget_defaults(),
                                  **spec.budget.model_dump(include=spec.budget.model_fields_set)})
        if spec.auto_process and task_spec is None and spec.budget.max_coding_calls == 0:
            spec.budget.max_coding_calls = self.settings.budget_defaults().get(
                "max_coding_calls", C.Budget().max_coding_calls)
        spec.methods_hint = list(methods_hint or spec.methods_hint)
        from pathlib import Path
        from ..paths import sha256_file
        for source in spec.data_sources:
            path = Path(source.path).expanduser().resolve()
            if not path.is_file():
                raise ControllerError("missing_data_source", f"source file missing: {source.source_id}")
            digest = sha256_file(path)
            if source.checksum and source.checksum != digest:
                raise ControllerError("source_hash_mismatch", f"source changed: {source.source_id}")
            source.path, source.checksum = str(path), digest
        if len({s.source_id for s in spec.data_sources}) != len(spec.data_sources):
            raise ControllerError("duplicate_data_source", "data source IDs must be unique")
        tp = TaskPaths(self.settings, slug)
        if parent_task:
            owner_slug = self.repo.get_spec(parent_task).task_slug
            register_task_member(self.settings, owner_slug, slug, track or slug)
        ensure_task_dirs(tp)
        self.repo.create_task(spec)
        self.rt.bus.publish(tid, "task.created", f"task {tid} created (slug={slug})",
                            data={"demo": demo, "synthetic": synthetic, "unknowns": spec.unknowns})
        self._persist_task_files(tid)
        return spec

    def _persist_task_files(self, task_id: str) -> None:
        """Task-local recovery/audit export; the live shared DB remains authoritative."""
        import yaml
        from ..paths import write_json
        spec = self.repo.get_spec(task_id)
        tp = TaskPaths(self.settings, spec.task_slug)
        tp.output_path("task.yaml").write_text(
            yaml.safe_dump(spec.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
            encoding="utf-8")
        write_json(tp.runtime / "state" / "snapshot.json",
                   {"task": spec.model_dump(mode="json"), "state": self.repo.get_state(task_id).model_dump(),
                    "jobs": [j.model_dump() for j in self.repo.list_jobs(task_id)],
                    "events": self.repo.read_events(task_id, limit=10000),
                    "control_state_dir": str(self.settings.state_dir),
                    "note": "audit snapshot, not a replacement for the live scheduler/checkpoint database"})

    def start(self, task_id: str, text: str = "") -> Dict[str, Any]:
        return self.run(task_id, raw_text=text)

    def run(self, task_id: str, *, resume: Optional[Any] = None,
            raw_text: str = "") -> Dict[str, Any]:
        import fcntl
        lock_path = self.settings.state_dir / f"task_{task_key(task_id)}.lock"
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ControllerError("task_busy", "task is already running", status=409)
            try:
                return self._run(task_id, resume=resume, raw_text=raw_text)
            except (TaskHalted, BudgetExceeded, ValueError) as exc:
                current = self.repo.get_state(task_id)
                if not current.paused and current.phase not in {"stopped", "accepted", "awaiting_input"}:
                    self.repo.update_state(task_id, phase="stopped", stop_reason=str(exc))
                self.repo.append_event(task_id, "task.halted", str(exc), level="warning")
                result = self._boundary(task_id, {})
                result["done"] = not current.paused
                result["type"] = "paused" if current.paused else "end"
                return result
            except ControllerError:
                raise
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                current = self.repo.get_state(task_id)
                if not current.paused and current.phase not in {"stopped", "accepted"}:
                    self.repo.update_state(task_id, phase="failed", stop_reason=reason)
                self.repo.append_event(task_id, "task.failed", reason, level="error")
                return self._boundary(task_id, {})
            finally:
                try:
                    if self.repo.task_exists(task_id):
                        self._persist_task_files(task_id)
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def _run(self, task_id: str, *, resume: Optional[Any] = None,
             raw_text: str = "") -> Dict[str, Any]:
        if not self.repo.task_exists(task_id):
            raise ControllerError("unknown_task", f"task {task_id} not found",
                                  recovery="create it first with: algo-agent chat / /new", status=404)
        cfg = self._cfg(task_id)
        current = self.repo.get_state(task_id)
        if current.paused:
            raise TaskHalted("task is paused")
        if current.phase in {"accepted", "stopped", "failed"}:
            return self._boundary(task_id, {})
        if resume is not None:
            from langgraph.types import Command
            payload: Any = Command(resume=resume)
        else:
            spec = self.repo.get_spec(task_id)
            snapshot = self.rt.graph.get_state(cfg)
            if snapshot.values:
                pending = self.pending_interrupt(task_id)
                if spec.auto_process and pending and pending.get("type") == "approval_request":
                    from langgraph.types import Command
                    payload = Command(resume={"approved": True, "plan_id": pending["plan_id"],
                                              "plan_hash": pending["plan_hash"]})
                elif spec.auto_process and pending and pending.get("type") == "iteration_boundary":
                    from langgraph.types import Command
                    payload = Command(resume={"stop": False})
                elif pending:
                    return {"type": pending.get("type"), "interrupt": pending,
                            "state": self.state(task_id), "done": False}
                elif snapshot.next:
                    payload = None
                else:
                    return self._boundary(task_id, {})
            else:
                st = initial_state(task_id, spec.task_slug)
                st["raw_text"] = raw_text or spec.business
                payload = st
        try:
            result = self.rt.graph.invoke(payload, cfg)
        except StateConflict as exc:
            raise ControllerError("state_conflict", str(exc),
                                  recovery="re-read state and retry", status=409)
        return self._boundary(task_id, result)

    def _boundary(self, task_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        interrupts = result.get("__interrupt__") or []
        state = self.state(task_id)
        if interrupts:
            value = getattr(interrupts[0], "value", {}) or {}
            kind = value.get("type", "interrupt") if isinstance(value, dict) else "interrupt"
            self.rt.bus.publish(task_id, "boundary", f"waiting at {kind}",
                                data={"interrupt": value})
            return {"type": kind, "interrupt": value, "state": state, "done": False}
        return {"type": "end", "interrupt": None, "state": state, "done": True}

    # ------------------------------------------------------------------ #
    # approval
    # ------------------------------------------------------------------ #
    def approve(self, task_id: str, plan_id: str, plan_hash: str, *,
                expected_state_version: Optional[int] = None,
                scope: Optional[Dict[str, Any]] = None, mode: Optional[str] = None,
                conversation_id: str = "", idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        if not self.repo.task_exists(task_id):
            raise ControllerError("unknown_task", f"task {task_id} not found", status=404)
        plan = self.repo.get_plan(plan_id)
        if plan is None:
            raise ControllerError("unknown_plan", f"plan {plan_id} not found",
                                  recovery="run: algo-agent plan show", status=404)
        if plan.task_id != task_id:
            raise ControllerError("plan_task_mismatch",
                                  f"plan {plan_id} belongs to task {plan.task_id}", status=409)
        current_hash = self.repo.get_plan_hash(plan_id)
        if not plan_hash or plan_hash != current_hash:
            raise ControllerError(
                "stale_plan_hash",
                "approval hash does not match the frozen plan content",
                recovery="re-read the plan id and hash (algo-agent plan show), then approve again",
                status=409)
        snap = self.repo.get_state(task_id)
        if snap.paused or snap.phase in {"stopped", "accepted", "failed"}:
            raise ControllerError("task_not_running", f"task is {snap.phase}", status=409)
        if expected_state_version is not None and expected_state_version != snap.state_version:
            raise ControllerError(
                "state_conflict",
                f"expected_state_version={expected_state_version} but current={snap.state_version}",
                recovery="a newer state exists; re-read and retry", status=409)

        pending = self.pending_interrupt(task_id)
        if not pending or pending.get("type") != "approval_request":
            raise ControllerError("not_at_approval",
                                  "graph is not waiting at an approval boundary",
                                  recovery="start the task first: algo-agent chat --task " + task_id,
                                  status=409)

        cmd = C.Command(command_id=new_id("cmd"), task_id=task_id, conversation_id=conversation_id,
                        intent=Intent.APPROVAL.value, text=f"approve {plan_id}",
                        expected_state_version=expected_state_version,
                        idempotency_key=idempotency_key,
                        payload={"plan_id": plan_id, "plan_hash": plan_hash, "scope": scope, "mode": mode})
        cmd = self.repo.add_command(cmd)
        self.rt.bus.publish(task_id, "command.approval", f"approval accepted for {plan_id}",
                            data={"command_id": cmd.command_id, "plan_hash": plan_hash[:12]})
        resume = {"approved": True, "plan_id": plan_id, "plan_hash": plan_hash,
                  "scope": scope, "mode": mode}
        boundary = self.run(task_id, resume=resume)
        boundary["command_id"] = cmd.command_id
        return boundary

    def reject(self, task_id: str, plan_id: str, reason: str = "") -> Dict[str, Any]:
        self.rt.bus.publish(task_id, "command.reject", f"plan {plan_id} rejected: {reason}")
        return self.run(task_id, resume={"approved": False, "plan_id": plan_id,
                                         "plan_hash": self.repo.get_plan_hash(plan_id)})

    # ------------------------------------------------------------------ #
    # feedback / control
    # ------------------------------------------------------------------ #
    def continue_task(self, task_id: str, *, stop: bool = False) -> Dict[str, Any]:
        """Resume a graph paused at an iteration boundary (feedback loop)."""
        if not self.repo.task_exists(task_id):
            raise ControllerError("unknown_task", f"task {task_id} not found", status=404)
        pending = self.pending_interrupt(task_id)
        if not pending or pending.get("type") != "iteration_boundary":
            raise ControllerError("no_iteration_boundary",
                                  "task is not waiting at an iteration boundary",
                                  recovery="check: algo-agent status --task " + task_id, status=409)
        self.rt.bus.publish(task_id, "control.continue",
                            "continuing from iteration boundary" + (" (stop)" if stop else ""))
        return self.run(task_id, resume={"stop": stop})

    def submit_feedback(self, task_id: str, feedback: Dict[str, Any],
                        *, auto_continue: bool = True) -> Dict[str, Any]:
        if not self.repo.task_exists(task_id):
            raise ControllerError("unknown_task", f"task {task_id} not found", status=404)
        models = self.repo.list_models(task_id)
        model_ids = [m.model_id for m in models]
        fb = C.Feedback(
            feedback_id=feedback.get("feedback_id") or new_id("fb"),
            task_id=task_id, model_id=feedback.get("model_id"), source=feedback.get("source", "user"),
            scenario=feedback.get("scenario", ""), expected=feedback.get("expected", ""),
            actual=feedback.get("actual", ""), severity=feedback.get("severity", "medium"),
            repro=feedback.get("repro", ""), conditions=feedback.get("conditions", {}) or {},
            upload_scope=feedback.get("upload_scope", "restricted"))
        need_confirm = False
        if fb.model_id is None:
            need_confirm = True
            fb.missing_evidence.append(
                "model version not supplied; candidates listed, no default to newest (guide 8)")
        elif fb.model_id not in model_ids:
            raise ControllerError("unknown_model",
                                  f"model {fb.model_id} not registered for {task_id}",
                                  recovery="choose from: " + ", ".join(model_ids), status=409)
        self.repo.save_feedback(fb)
        self.rt.bus.publish(task_id, "feedback.received",
                            f"feedback {fb.feedback_id} queued (severity={fb.severity})",
                            data={"model_id": fb.model_id, "scenario": fb.scenario})
        result: Dict[str, Any] = {
            "feedback_id": fb.feedback_id, "model_id": fb.model_id,
            "model_candidates": model_ids, "needs_model_confirmation": need_confirm,
            "queued": True}
        pending = self.pending_interrupt(task_id)
        if auto_continue and pending and pending.get("type") == "iteration_boundary":
            boundary = self.continue_task(task_id)
            result["continued"] = True
            result["boundary"] = boundary
        return result

    def pause(self, task_id: str) -> Dict[str, Any]:
        if self.repo.get_state(task_id).phase in {"accepted", "stopped", "failed"}:
            raise ControllerError("task_terminal", "terminal tasks are immutable", status=409)
        snap = self.repo.update_state(task_id, paused=True, phase=TaskPhase.PAUSED.value)
        self.rt.bus.publish(task_id, "control.pause", "pause requested; new steps will not start")
        return snap.model_dump()

    def resume_task(self, task_id: str) -> Dict[str, Any]:
        if self.repo.get_state(task_id).phase in {"accepted", "stopped", "failed"}:
            raise ControllerError("task_terminal", "terminal tasks are immutable; create a new task", status=409)
        snap = self.repo.update_state(task_id, paused=False, phase=TaskPhase.EXECUTING.value)
        self.rt.bus.publish(task_id, "control.resume", "resume requested; new steps may start")
        return snap.model_dump()

    def cancel(self, task_id: str, job_id: Optional[str] = None) -> Dict[str, Any]:
        if self.repo.get_state(task_id).phase in {"accepted", "stopped", "failed"}:
            raise ControllerError("task_terminal", "terminal tasks are immutable", status=409)
        cancelled: List[str] = []
        if job_id:
            jobs = [self.repo.get_job(job_id)] if self.repo.get_job(job_id) else []
            if jobs and jobs[0].task_id != task_id:
                raise ControllerError("job_task_mismatch", "job belongs to another task", status=409)
        else:
            jobs = [j for j in self.repo.list_jobs(task_id) if j.status not in TERMINAL_JOB]
        for job in jobs:
            if job is None:
                continue
            self.rt.runner.cancel(job.job_id)
            cancelled.append(job.job_id)
        if not job_id:
            self.repo.update_state(task_id, phase=TaskPhase.STOPPED.value,
                                   stop_reason="cancelled by user")
        self.rt.bus.publish(task_id, "control.cancel",
                            f"cancel requested for {len(cancelled)} job(s)",
                            data={"job_ids": cancelled})
        return {"cancelled": cancelled, "note": "saved assets are never deleted"}

    # ------------------------------------------------------------------ #
    # reads
    # ------------------------------------------------------------------ #
    def state(self, task_id: str) -> Dict[str, Any]:
        snap = self.repo.get_state(task_id)
        d = snap.model_dump()
        d["pending_approval"] = self.pending_interrupt(task_id)
        spec = self.repo.get_spec(task_id)
        d["task"] = {"task_id": spec.task_id, "task_slug": spec.task_slug,
                     "business": spec.business, "unknowns": spec.unknowns,
                     "primary_metric": spec.primary_metric.model_dump(),
                     "demo": spec.demo, "synthetic": spec.synthetic}
        d["task"]["auto_process"] = spec.auto_process
        d["stage_count"] = len(STAGES)
        return d

    def events(self, task_id: str, after: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
        return self.repo.read_events(task_id, after=after, limit=limit)

    def plan_view(self, task_id: str, plan_id: str) -> Dict[str, Any]:
        plan = self.repo.get_plan(plan_id)
        if plan is None or plan.task_id != task_id:
            raise ControllerError("unknown_plan", f"plan {plan_id} not found for {task_id}", status=404)
        slug = self.repo.get_spec(task_id).task_slug
        approval = self.repo.get_active_approval(task_id)
        return {"plan": plan.model_dump(), "plan_hash": self.repo.get_plan_hash(plan_id),
                "dir": str(TaskPaths(self.settings, slug).plan_dir(plan.version)),
                "approval": approval.model_dump() if approval else None}

    def answer(self, task_id: str, question: str, conversation_id: str = "") -> Dict[str, Any]:
        """Answer an operator question from the live task context via the chat model.

        Returns ``{"answer": "", "source": "offline", "offline": True}`` when no
        chat-capable provider is configured, so the caller can fall back to the
        deterministic grounded renderer without inventing anything.
        """
        if not self.repo.task_exists(task_id):
            raise ControllerError("unknown_task", f"task {task_id} not found", status=404)
        if not question.strip() or len(question) > 16000:
            raise ControllerError("invalid_message", "message must contain 1-16000 characters")
        context = self._answer_context(task_id, conversation_id)
        self.repo.add_message(task_id, conversation_id, "user", question)
        provider = self.rt.providers.get("chat")
        if provider is None or not hasattr(provider, "answer"):
            self.repo.add_message(task_id, conversation_id, "assistant", "",
                                  {"source": "offline", "fallback": "task_state"})
            return {"answer": "", "source": "offline", "offline": True}
        try:
            from ..providers.usage import tracked_call
            fn = getattr(provider, "answer_with_usage", provider.answer)
            out = tracked_call(self.repo, self.repo.get_spec(task_id), "chat",
                               context + question, 3072, lambda: fn(question, context))
            text = out.get("content", "") if isinstance(out, dict) else out
        except Exception as exc:  # a provider failure is diagnosable, never fatal
            self.repo.add_message(task_id, conversation_id, "assistant", "",
                                  {"source": "error", "error_type": type(exc).__name__})
            return {"answer": "", "source": "error", "offline": False, "error": str(exc)}
        self.repo.add_message(task_id, conversation_id, "assistant", text or "",
                              {"source": getattr(provider, "name", "chat"),
                               "model": getattr(provider, "model", "")})
        if not text:
            return {"answer": "", "source": getattr(provider, "name", "chat"), "offline": False,
                    "model": getattr(provider, "model", "")}
        return {"answer": text, "source": getattr(provider, "name", "chat"), "offline": False,
                "model": getattr(provider, "model", "")}

    def _answer_context(self, task_id: str, conversation_id: str = "") -> str:
        """Compact, grounded context for a conversational reply (no raw data)."""
        st = self.state(task_id)
        lines = [
            f"task_id: {task_id}",
            f"phase: {st.get('phase')}  iteration: {st.get('iteration')}  paused: {st.get('paused')}",
            f"active_plan: {st.get('active_plan_id')}  best_model_id: {st.get('best_model_id')}",
            f"budget_used: {st.get('budget_used')}",
            f"stop_reason: {st.get('stop_reason')}",
            "Supported dialogue controls: say '暂停任务' to pause, '恢复任务' to resume, "
            "'取消任务' to cancel, '查看状态' to inspect. Asking how to pause is read-only. "
            "Pause prevents new steps; it does not promise to interrupt/checkpoint a running training job. "
            "Stopped/accepted/failed tasks cannot be resumed; create a separate task for a fresh trial. "
            "Do not edit state files directly. Historical messages describe past states, not current state.",
        ]
        spec = self.repo.get_spec(task_id)
        if spec is not None:
            pm = spec.primary_metric
            lines.append(f"primary_metric: {pm.name} {pm.direction} floor={pm.threshold}")
            lines.append("frozen_acceptance_targets: " + json.dumps(
                [t.model_dump() for t in [pm, *spec.secondary_metrics]], ensure_ascii=False))
            lines.append("acceptance_expressions: " + "; ".join(
                f"{t.name} {('>' if t.strict else '>=') if t.direction == 'max' else ('<' if t.strict else '<=')} "
                f"{t.threshold}" for t in [pm, *spec.secondary_metrics]))
            lines.append("budget_limits: " + json.dumps(spec.budget.model_dump()))
            lines.append("declared_unknowns: " + json.dumps(spec.unknowns, ensure_ascii=False))
            lines.append("hard_constraints: " + json.dumps(
                [c.model_dump() for c in spec.hard_constraints], ensure_ascii=False))
            if spec.business:
                lines.append(f"business: {spec.business[:600]}")
        evals = self.repo.list_evals(task_id)
        if evals:
            lines.append("evaluations (model_id | passed | metrics):")
            for e in evals[-12:]:
                names = {t.name for t in [spec.primary_metric, *spec.secondary_metrics]}
                names.update(("rom_mb", "n_scored"))
                m = {k: e.metrics[k] for k in names if k in e.metrics}
                lines.append(f"- {e.model_id} | stage={e.stage} | passed={e.passed} | {m}")
        feedback = self.repo.list_feedback(task_id)
        if feedback:
            lines.append("feedback:")
            for fb in feedback[-8:]:
                lines.append(f"- {fb.feedback_id}: model={fb.model_id} scenario={fb.scenario} "
                             f"actual={fb.actual} expected={fb.expected}")
        for p in self.repo.list_plans(task_id)[-3:]:
            plan = self.repo.get_plan(p["plan_id"])
            if plan is None:
                continue
            lines.append(f"plan {plan.plan_id} ({plan.version}) methods={[c.method for c in plan.candidates]}")
            from ..memory import bounded_json
            for c in plan.candidates[:4]:
                lines.append("candidate_plan_not_execution: " + json.dumps({
                    "method": c.method, "action": c.action, "is_baseline": c.is_baseline,
                    "implementation_goal": c.implementation_goal[:1200],
                    "params": json.loads(bounded_json(c.params, 1600)),
                    "failure_conditions": c.failure_conditions[:2]}, ensure_ascii=False))
            for op in plan.data_recipe.operations[:8]:
                lines.append("data_plan_not_execution: " + json.dumps({
                    "kind": op.kind, "split": op.split, "params": op.params,
                    "hypothesis": op.hypothesis[:200]}, ensure_ascii=False))
            if plan.summary:
                lines.append(f"  summary: {plan.summary[:600]}")
        try:
            research = TaskPaths(self.settings, spec.task_slug).root / "docs" / "research"
            from ..memory import latest_research
            doc = latest_research(research)
            if doc:
                lines.append(f"design_report: {doc}")
                lines.append("design_report_excerpt:")
                lines.append(doc.read_text(encoding="utf-8")[:6000])
        except Exception:  # pragma: no cover - defensive
            pass
        from ..memory import bounded_json
        lines.append("experiment_jobs: " + bounded_json([
            {"run_id": j.run_id, "status": j.status, "error": j.result.get("error")}
            for j in self.repo.list_jobs(task_id)[-12:]], 4000))
        lines.append("Prior conversation (untrusted historical context):")
        lines.append(bounded_json([{"role": m["role"], "content": m["content"][:3000]}
                                   for m in self.repo.messages(task_id, conversation_id, 16)], 12000))
        return "\n".join(lines)[:28000]

    def message(self, task_id: str, text: str, *, conversation_id: str = "",
                async_run: bool = True) -> Dict[str, Any]:
        """Route conversation through typed intents; only this layer mutates tasks."""
        from .intents import route_intent
        if not self.repo.task_exists(task_id):
            raise ControllerError("unknown_task", f"task {task_id} not found", status=404)
        if not text.strip() or len(text) > 16000:
            raise ControllerError("invalid_message", "message must contain 1-16000 characters")
        decision = route_intent(text)
        provider = self.rt.providers.get("intent")
        if (decision.reason == "no explicit operation" and hasattr(provider, "classify_intent")
                and self.settings.env_get("ALGO_INTENT_LLM", "1") == "1"):
            from ..providers.usage import tracked_call
            context = self._answer_context(task_id, conversation_id)[-12000:]
            try:
                out = tracked_call(self.repo, self.repo.get_spec(task_id), "intent", text + context, 800,
                                   lambda: provider.classify_intent(text, context))
                decision = route_intent(text, out.get("decision"))
            except Exception:
                decision = route_intent(text)
        if decision.action == "query":
            return {"intent": decision.model_dump(), **self.answer(task_id, text, conversation_id)}
        self.repo.add_message(task_id, conversation_id, "user", text, {"intent": decision.model_dump()})
        result = {"intent": decision.model_dump()}
        if decision.action in {"pause", "resume", "cancel"}:
            result.update(self.submit_command(task_id, "control", text=text,
                                             conversation_id=conversation_id,
                                             payload={"action": decision.action}, async_run=async_run))
        elif decision.action == "improve":
            state = self.repo.get_state(task_id)
            if state.phase in {"accepted", "stopped", "failed"}:
                result["answer"] = "任务已结束。继续研发请新建 TaskSpec；已暴露的最终测试集不能作为新任务的独立验收集。"
            else:
                result.update(self.submit_command(
                    task_id, "feedback", text=text, conversation_id=conversation_id, async_run=async_run,
                    payload={"scenario": "improvement", "expected": text,
                             "actual": f"best model: {state.best_model_id}", "model_id": state.best_model_id}))
                result["answer"] = "改进目标已记录，将在下一研究轮次评估；现有验收目标和数据授权保持冻结。"
        elif decision.action == "status":
            result["state"] = self.state(task_id)
        else:
            result["answer"] = "这条消息的操作意图不明确。请明确说明要查询、改进、暂停、恢复还是取消当前任务。"
        result["intent"] = decision.model_dump()
        import json
        self.repo.add_message(task_id, conversation_id, "assistant",
                              result.get("answer") or json.dumps(result.get("state", {}), ensure_ascii=False),
                              {"intent": decision.action})
        return result

    def readiness(self, *, probe_coding: bool = False) -> Dict[str, Any]:
        from ..providers import readiness_report
        return readiness_report(self.settings, probe=probe_coding)

    def recover_tasks(self) -> List[str]:
        """Re-drive nonterminal autonomous graphs after service restart."""
        recovered = []
        self.repo.reconcile_jobs()
        for row in self.repo.list_tasks():
            tid = row["task_id"]
            spec, state = self.repo.get_spec(tid), self.repo.get_state(tid)
            if not spec.auto_process or state.paused or state.phase in {
                    "accepted", "stopped", "failed", "awaiting_input"}:
                continue
            try:
                self.run_async(tid)
                recovered.append(tid)
            except ControllerError:
                pass
        return recovered

    def budget(self, task_id: str) -> Dict[str, Any]:
        spec = self.repo.get_spec(task_id)
        used = self.repo.get_budget(task_id)
        return {"limits": spec.budget.model_dump(), "used": used.get("used", {}),
                "reserved": used.get("reserved", {}),
                "ledger": self.repo.list_ledger(task_id)}

    def jobs(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return [j.model_dump() for j in self.repo.list_jobs(task_id)]

    # ------------------------------------------------------------------ #
    # async execution (for the control service)
    # ------------------------------------------------------------------ #
    def run_async(self, task_id: str, *, resume: Optional[Any] = None,
                  raw_text: str = "") -> str:
        with self._lock:
            th = self._threads.get(task_id)
            if th and th.is_alive():
                raise ControllerError("already_running", f"task {task_id} is running",
                                      recovery="wait or cancel", status=409)
            th = threading.Thread(target=self._safe_run, kwargs=dict(
                task_id=task_id, resume=resume, raw_text=raw_text), daemon=True)
            self._threads[task_id] = th
            th.start()
        return task_id

    def _safe_run(self, task_id: str, resume: Optional[Any], raw_text: str) -> None:
        try:
            self.run(task_id, resume=resume, raw_text=raw_text)
        except Exception as exc:
            self.rt.bus.publish(task_id, "error", f"graph run failed: {exc}", level="error",
                                data={"error": repr(exc)})

    def wait(self, task_id: str, timeout: Optional[float] = None) -> None:
        th = self._threads.get(task_id)
        if th:
            th.join(timeout=timeout)

    # ------------------------------------------------------------------ #
    # dispatch a user command (shared by CLI and POST /commands)
    # ------------------------------------------------------------------ #
    def submit_command(self, task_id: str, intent: str, *, payload: Optional[Dict[str, Any]] = None,
                       text: str = "", conversation_id: str = "",
                       expected_state_version: Optional[int] = None,
                       idempotency_key: Optional[str] = None, async_run: bool = True) -> Dict[str, Any]:
        payload = payload or {}
        if idempotency_key:
            existing = self.repo.get_command_by_key(idempotency_key)
            if existing:
                return {"command_id": existing.command_id, "intent": existing.intent,
                        "idempotent_replay": True}
        cmd = C.Command(command_id=new_id("cmd"), task_id=task_id, conversation_id=conversation_id,
                        intent=intent, text=text, expected_state_version=expected_state_version,
                        idempotency_key=idempotency_key, payload=payload)
        cmd = self.repo.add_command(cmd)
        result: Dict[str, Any] = {"command_id": cmd.command_id, "intent": intent}

        if intent == Intent.APPROVAL.value:
            result.update(self.approve(task_id, payload.get("plan_id", ""), payload.get("plan_hash", ""),
                                       expected_state_version=expected_state_version,
                                       scope=payload.get("scope"), mode=payload.get("mode"),
                                       conversation_id=conversation_id,
                                       idempotency_key=idempotency_key))
        elif intent == Intent.FEEDBACK.value:
            result.update(self.submit_feedback(task_id, payload, auto_continue=False))
            if async_run and (self.pending_interrupt(task_id) or {}).get("type") == "iteration_boundary":
                self.run_async(task_id, resume={"stop": False})
                result["continued"] = True
        elif intent in (Intent.CONTROL.value, "control"):
            action = payload.get("action", "")
            if action == "pause":
                result.update({"state": self.pause(task_id)})
            elif action == "resume":
                result.update({"state": self.resume_task(task_id)})
                if async_run:
                    self.run_async(task_id)
            elif action == "cancel":
                result.update(self.cancel(task_id, payload.get("job_id")))
            else:
                raise ControllerError("bad_action", f"unknown control action {action!r}")
        elif intent == Intent.NEW_REQUIREMENT.value:
            spec = self.create_task(text, task_id=task_id, demo=payload.get("demo", False),
                                    synthetic=payload.get("synthetic", False),
                                    methods_hint=payload.get("methods_hint"),
                                    auto_process=payload.get("auto_process", False),
                                    domain=payload.get("domain", ""), task_spec=payload.get("task_spec"))
            result.update({"task_id": spec.task_id, "task_slug": spec.task_slug})
            if async_run:
                self.run_async(task_id, raw_text=text)
        elif intent in (Intent.QUERY.value, Intent.UNKNOWN.value):
            result.update({"state": self.state(task_id)})
        else:
            raise ControllerError("bad_intent", f"unsupported intent {intent!r}")
        return result

    # ------------------------------------------------------------------ #
    # exports
    # ------------------------------------------------------------------ #
    def export(self, task_id: str, *, iteration: Optional[int] = None) -> Dict[str, Any]:
        from .. import docs_export
        spec = self.repo.get_spec(task_id)
        slug = spec.task_slug
        out: Dict[str, str] = {}
        from ..memory import ResearchMemory
        out["knowledge_index"] = ResearchMemory(self.settings, self.repo).export(task_id)
        out["overview"] = docs_export.export_task_overview(self.settings, self.repo, slug)
        it = iteration or max(1, self.repo.get_state(task_id).iteration)
        try:
            out["iteration_report"] = docs_export.export_iteration_report(self.settings, self.repo, slug, it)
        except Exception:
            pass
        try:
            out["research"] = docs_export.export_research_docx(self.settings, self.repo, slug, it)
        except Exception:
            pass
        try:
            out["dataset"] = docs_export.export_dataset_doc(self.settings, self.repo, slug,
                                                            version=f"v{it}")
        except Exception:
            pass
        for p in self.repo.list_plans(task_id):
            try:
                out.update({f"plan_{p['version']}_{k}": v for k, v in
                            docs_export.export_plan_docs(self.settings, self.repo, slug, p["plan_id"]).items()})
            except Exception:
                pass
        self.rt.bus.publish(task_id, "export.docs", "DOCX records regenerated",
                            data={"files": list(out.values())})
        return out


__all__ = ["Controller", "ControllerError"]
