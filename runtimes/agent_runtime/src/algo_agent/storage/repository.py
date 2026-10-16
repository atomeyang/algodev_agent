"""Transactional repository over SQLite with compare-and-set state versions.

A single task has exactly one writer (guide 3); concurrent clients use the
expected_state_version so stale writes are rejected with a conflict instead of
silently clobbering state.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..schemas import contracts as C
from ..schemas.enums import JobStatus, TaskPhase
from . import db as _db


class StateConflict(RuntimeError):
    def __init__(self, task_id: str, expected: Optional[int], actual: int):
        super().__init__(
            f"state_version conflict for {task_id}: expected={expected} actual={actual}"
        )
        self.task_id = task_id
        self.expected = expected
        self.actual = actual


class BudgetExceeded(RuntimeError):
    pass


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def _loads(raw: Optional[str], default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


class Repository:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        _db.init_db(self.db_path)

    # ------------------------------------------------------------------ #
    def conn(self) -> sqlite3.Connection:
        return _db.connect(self.db_path)

    # ---- tasks / state ------------------------------------------------- #
    def create_task(self, spec: C.TaskSpec, phase: str = TaskPhase.DRAFT.value) -> C.TaskStateSnapshot:
        with self.conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO tasks(task_id, task_slug, phase, state_version, iteration, "
                "spec_json, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?)",
                (spec.task_id, spec.task_slug, phase, 0, 0, _dumps(spec.model_dump()),
                 spec.created_at, spec.created_at),
            )
            c.commit()
        return self.get_state(spec.task_id)

    def get_state(self, task_id: str) -> C.TaskStateSnapshot:
        with self.conn() as c:
            row = c.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown task {task_id}")
        return self._row_to_state(row)

    def task_exists(self, task_id: str) -> bool:
        with self.conn() as c:
            return c.execute("SELECT 1 FROM tasks WHERE task_id=?", (task_id,)).fetchone() is not None

    def list_tasks(self) -> List[Dict[str, Any]]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT task_id, task_slug, phase, iteration, state_version, updated_at FROM tasks "
                "ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def _row_to_state(self, row: sqlite3.Row) -> C.TaskStateSnapshot:
        spec = _loads(row["spec_json"], {})
        return C.TaskStateSnapshot(
            task_id=row["task_id"], phase=row["phase"], iteration=row["iteration"] or 0,
            state_version=row["state_version"] or 0, active_plan_id=row["active_plan_id"],
            best_model_id=row["best_model_id"], stop_reason=row["stop_reason"],
            paused=bool(row["paused"]), budget_used=self.get_budget(row["task_id"]),
        )

    def get_spec(self, task_id: str) -> C.TaskSpec:
        with self.conn() as c:
            row = c.execute("SELECT spec_json FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown task {task_id}")
        return C.TaskSpec(**_loads(row["spec_json"], {}))

    def save_spec(self, spec: C.TaskSpec) -> None:
        with self.conn() as c:
            c.execute("UPDATE tasks SET spec_json=?, updated_at=? WHERE task_id=?",
                      (_dumps(spec.model_dump()), C.utcnow(), spec.task_id))
            c.commit()

    def update_state(
        self,
        task_id: str,
        expected_state_version: Optional[int] = None,
        **fields: Any,
    ) -> C.TaskStateSnapshot:
        allowed = {"phase", "iteration", "active_plan_id", "best_model_id", "stop_reason", "paused"}
        fields = {k: v for k, v in fields.items() if k in allowed}
        with self.conn() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT state_version FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown task {task_id}")
            current = row["state_version"] or 0
            if expected_state_version is not None and expected_state_version != current:
                raise StateConflict(task_id, expected_state_version, current)
            sets = ", ".join(f"{k}=?" for k in fields) if fields else ""
            params: List[Any] = list(fields.values())
            new_version = current + 1
            if sets:
                params += [new_version, C.utcnow(), task_id]
                c.execute(f"UPDATE tasks SET {sets}, state_version=?, updated_at=? WHERE task_id=?", params)
            else:
                c.execute("UPDATE tasks SET state_version=?, updated_at=? WHERE task_id=?",
                          (new_version, C.utcnow(), task_id))
            c.commit()
        return self.get_state(task_id)

    # ---- events -------------------------------------------------------- #
    def append_event(self, task_id: str, kind: str, message: str = "", level: str = "info",
                     data: Optional[Dict[str, Any]] = None) -> int:
        with self.conn() as c:
            cur = c.execute(
                "INSERT INTO events(task_id, kind, level, message, data_json, created_at) VALUES(?,?,?,?,?,?)",
                (task_id, kind, level, message, _dumps(data or {}), C.utcnow()),
            )
            c.commit()
            return int(cur.lastrowid)

    def read_events(self, task_id: str, after: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM events WHERE task_id=? AND event_seq>? ORDER BY event_seq LIMIT ?",
                (task_id, after, limit),
            ).fetchall()
        out = []
        for r in rows:
            out.append({
                "event_seq": r["event_seq"], "task_id": r["task_id"], "kind": r["kind"],
                "level": r["level"], "message": r["message"], "data": _loads(r["data_json"], {}),
                "created_at": r["created_at"],
            })
        return out

    # ---- plans / approvals -------------------------------------------- #
    def save_plan(self, plan: C.PlanSpec, content_hash: str, status: str = "proposed") -> None:
        with self.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO plans(plan_id, task_id, version, parent_plan_id, content_hash, "
                "status, spec_json, created_at) VALUES(?,?,?,?,?,?,?,?)",
                (plan.plan_id, plan.task_id, plan.version, plan.parent_plan_id, content_hash,
                 status, _dumps(plan.model_dump()), plan.created_at),
            )
            c.commit()

    def get_plan(self, plan_id: str) -> Optional[C.PlanSpec]:
        with self.conn() as c:
            row = c.execute("SELECT spec_json FROM plans WHERE plan_id=?", (plan_id,)).fetchone()
        return C.PlanSpec(**_loads(row["spec_json"], {})) if row else None

    def get_plan_hash(self, plan_id: str) -> Optional[str]:
        with self.conn() as c:
            row = c.execute("SELECT content_hash FROM plans WHERE plan_id=?", (plan_id,)).fetchone()
        return row["content_hash"] if row else None

    def list_plans(self, task_id: str) -> List[Dict[str, Any]]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT plan_id, version, parent_plan_id, content_hash, status, created_at FROM plans "
                "WHERE task_id=? ORDER BY created_at", (task_id,)).fetchall()
        return [dict(r) for r in rows]

    def save_approval(self, approval: C.Approval) -> None:
        with self.conn() as c:
            c.execute("UPDATE approvals SET revoked=1 WHERE task_id=? AND plan_id<>?",
                      (approval.task_id, approval.plan_id))
            c.execute(
                "INSERT OR REPLACE INTO approvals(approval_id, task_id, plan_id, plan_hash, scope_json, "
                "mode, approved_at, state_version, revoked) VALUES(?,?,?,?,?,?,?,?,0)",
                (approval.approval_id, approval.task_id, approval.plan_id, approval.plan_hash,
                 _dumps(approval.scope.model_dump()), approval.mode, approval.approved_at,
                 approval.state_version),
            )
            c.execute("UPDATE plans SET status='approved' WHERE plan_id=?", (approval.plan_id,))
            c.commit()

    def get_active_approval(self, task_id: str) -> Optional[C.Approval]:
        with self.conn() as c:
            row = c.execute(
                "SELECT * FROM approvals WHERE task_id=? AND revoked=0 ORDER BY approved_at DESC LIMIT 1",
                (task_id,)).fetchone()
        if not row:
            return None
        return C.Approval(
            approval_id=row["approval_id"], task_id=row["task_id"], plan_id=row["plan_id"],
            plan_hash=row["plan_hash"], scope=C.ApprovalScope(**_loads(row["scope_json"], {})),
            mode=row["mode"], approved_at=row["approved_at"], state_version=row["state_version"] or 0)

    # ---- commands ------------------------------------------------------ #
    def add_command(self, cmd: C.Command) -> C.Command:
        if cmd.idempotency_key:
            existing = self.get_command_by_key(cmd.idempotency_key)
            if existing:
                return existing
        with self.conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO commands(command_id, task_id, conversation_id, intent, text, "
                "expected_state_version, payload_json, idempotency_key, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (cmd.command_id, cmd.task_id, cmd.conversation_id, cmd.intent, cmd.text,
                 cmd.expected_state_version, _dumps(cmd.payload), cmd.idempotency_key, cmd.created_at),
            )
            c.commit()
        return cmd

    def get_command_by_key(self, key: str) -> Optional[C.Command]:
        with self.conn() as c:
            row = c.execute("SELECT * FROM commands WHERE idempotency_key=?", (key,)).fetchone()
        return self._row_to_command(row) if row else None

    def get_command(self, command_id: str) -> Optional[C.Command]:
        with self.conn() as c:
            row = c.execute("SELECT * FROM commands WHERE command_id=?", (command_id,)).fetchone()
        return self._row_to_command(row) if row else None

    def _row_to_command(self, row: sqlite3.Row) -> C.Command:
        return C.Command(
            command_id=row["command_id"], task_id=row["task_id"], conversation_id=row["conversation_id"] or "",
            intent=row["intent"], text=row["text"] or "", expected_state_version=row["expected_state_version"],
            payload=_loads(row["payload_json"], {}), idempotency_key=row["idempotency_key"],
            created_at=row["created_at"])

    # ---- jobs (idempotent) -------------------------------------------- #
    def create_job(self, task_id: str, kind: str, idempotency_key: str, run_id: str = "",
                   payload: Optional[Dict[str, Any]] = None) -> C.JobRecord:
        job_id = "job_" + uuid.uuid5(uuid.NAMESPACE_URL, idempotency_key).hex[:12]
        with self.conn() as c:
            existing = c.execute("SELECT * FROM jobs WHERE idempotency_key=?", (idempotency_key,)).fetchone()
            if existing:
                return self._row_to_job(existing)
            c.execute(
                "INSERT INTO jobs(job_id, task_id, kind, status, idempotency_key, run_id, payload_json, "
                "result_json, attempts, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,0,?,?)",
                (job_id, task_id, kind, JobStatus.QUEUED.value, idempotency_key, run_id,
                 _dumps(payload or {}), "{}", C.utcnow(), C.utcnow()),
            )
            c.commit()
            row = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._row_to_job(row)

    def _row_to_job(self, row: sqlite3.Row) -> C.JobRecord:
        return C.JobRecord(
            job_id=row["job_id"], task_id=row["task_id"], kind=row["kind"], status=row["status"],
            idempotency_key=row["idempotency_key"] or "", run_id=row["run_id"] or "",
            payload=_loads(row["payload_json"], {}), result=_loads(row["result_json"], {}),
            lease_owner=row["lease_owner"], attempts=row["attempts"] or 0, pid=row["pid"],
            pid_start=row["pid_start"],
            heartbeat_at=row["heartbeat_at"], created_at=row["created_at"], updated_at=row["updated_at"])

    def get_job(self, job_id: str) -> Optional[C.JobRecord]:
        with self.conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(self, task_id: Optional[str] = None) -> List[C.JobRecord]:
        with self.conn() as c:
            if task_id:
                rows = c.execute("SELECT * FROM jobs WHERE task_id=? ORDER BY created_at", (task_id,)).fetchall()
            else:
                rows = c.execute("SELECT * FROM jobs ORDER BY created_at").fetchall()
        return [self._row_to_job(r) for r in rows]

    _JOB_FIELD_MAP = {
        "result": ("result_json", _dumps),
        "payload": ("payload_json", _dumps),
    }

    def update_job(self, job_id: str, **fields: Any) -> Optional[C.JobRecord]:
        if not fields:
            return self.get_job(job_id)
        cols: List[str] = []
        values: List[Any] = []
        for key, value in fields.items():
            col, encoder = self._JOB_FIELD_MAP.get(key, (key, None))
            cols.append(col)
            values.append(encoder(value) if encoder and not isinstance(value, str) else value)
        sets = ", ".join(f"{k}=?" for k in cols)
        with self.conn() as c:
            c.execute(f"UPDATE jobs SET {sets}, updated_at=? WHERE job_id=?",
                      values + [C.utcnow(), job_id])
            c.commit()
        return self.get_job(job_id)

    def lease_job(self, task_id: Optional[str], owner: str) -> Optional[C.JobRecord]:
        with self.conn() as c:
            c.execute("BEGIN IMMEDIATE")
            if task_id:
                row = c.execute(
                    "SELECT * FROM jobs WHERE status=? AND task_id=? "
                    "AND NOT EXISTS (SELECT 1 FROM tasks t WHERE t.task_id=jobs.task_id "
                    "AND (t.paused=1 OR t.phase IN ('stopped','failed','accepted'))) "
                    "ORDER BY created_at LIMIT 1",
                    (JobStatus.QUEUED.value, task_id)).fetchone()
            else:
                row = c.execute("SELECT * FROM jobs WHERE status=? "
                                "AND NOT EXISTS (SELECT 1 FROM tasks t WHERE t.task_id=jobs.task_id "
                                "AND (t.paused=1 OR t.phase IN ('stopped','failed','accepted'))) "
                                "ORDER BY created_at LIMIT 1",
                                (JobStatus.QUEUED.value,)).fetchone()
            if not row:
                return None
            c.execute("UPDATE jobs SET status=?, lease_owner=?, attempts=attempts+1, heartbeat_at=?, "
                      "updated_at=? WHERE job_id=? AND status=?",
                      (JobStatus.RUNNING.value, owner, C.utcnow(), C.utcnow(), row["job_id"],
                       JobStatus.QUEUED.value))
            c.commit()
            row = c.execute("SELECT * FROM jobs WHERE job_id=?", (row["job_id"],)).fetchone()
        return self._row_to_job(row)

    def heartbeat_job(self, job: C.JobRecord, *, pid: Optional[int] = None) -> bool:
        """A stale attempt cannot renew or take ownership of a newer attempt."""
        with self.conn() as c:
            fields = "heartbeat_at=?, updated_at=?"
            values = [C.utcnow(), C.utcnow()]
            if pid is not None:
                fields += ", pid=?, pid_start=?"
                values += [pid, process_identity(pid)]
            cur = c.execute(
                f"UPDATE jobs SET {fields} WHERE job_id=? AND lease_owner=? AND attempts=? "
                "AND status IN ('running','cancel_requested')",
                [*values, job.job_id, job.lease_owner, job.attempts])
            c.commit()
            return cur.rowcount == 1

    def finish_job(self, job: C.JobRecord, status: str, result: Dict[str, Any]) -> bool:
        with self.conn() as c:
            cur = c.execute(
                "UPDATE jobs SET status=CASE WHEN status='cancel_requested' THEN 'cancelled' ELSE ? END, "
                "result_json=?, updated_at=? WHERE job_id=? AND lease_owner=? AND attempts=? "
                "AND status IN ('running','cancel_requested')",
                (status, _dumps(result), C.utcnow(), job.job_id, job.lease_owner, job.attempts))
            c.commit()
            return cur.rowcount == 1

    def reconcile_jobs(self, lease_timeout_s: float = 60.0) -> List[C.JobRecord]:
        """Recover dead/no-PID leases, never start a duplicate of a live process."""
        requeued: List[C.JobRecord] = []
        with self.conn() as c:
            c.execute("BEGIN IMMEDIATE")
            rows = c.execute("SELECT * FROM jobs WHERE status IN ('running','cancel_requested')").fetchall()
            for r in rows:
                stamp = r["heartbeat_at"] or r["updated_at"]
                expired = (datetime.now(timezone.utc) - datetime.fromisoformat(stamp)).total_seconds() >= lease_timeout_s
                pid = r["pid"]
                dead = pid is not None and (
                    not _pid_alive(pid) or (r["pid_start"] and process_identity(pid) != r["pid_start"]))
                if dead or (pid is None and expired):
                    status = "cancelled" if r["status"] == "cancel_requested" else "queued"
                    c.execute("UPDATE jobs SET status=?, lease_owner=NULL, pid=NULL, pid_start=NULL, "
                              "updated_at=? WHERE job_id=?",
                              (status, C.utcnow(), r["job_id"]))
                    requeued.append(self._row_to_job(r))
            c.commit()
        return requeued

    # ---- conversation / resumable research / measured lessons ---------- #
    def add_message(self, task_id: str, conversation_id: str, role: str, content: str,
                    metadata: Optional[Dict[str, Any]] = None) -> int:
        if role not in {"user", "assistant"}:
            raise ValueError("only user/assistant conversation records are allowed")
        with self.conn() as c:
            cur = c.execute(
                "INSERT INTO messages(task_id,conversation_id,role,content,metadata_json,created_at) "
                "VALUES(?,?,?,?,?,?)",
                (task_id, conversation_id or task_id, role, content, _dumps(metadata or {}), C.utcnow()))
            c.commit()
            return cur.lastrowid

    def messages(self, task_id: str, conversation_id: str = "", limit: int = 20) -> List[Dict[str, Any]]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM messages WHERE task_id=? AND conversation_id=? "
                "ORDER BY message_id DESC LIMIT ?", (task_id, conversation_id or task_id, max(1, min(limit, 100)))).fetchall()
        return [{"role": r["role"], "content": r["content"], "message_id": r["message_id"],
                 "metadata": _loads(r["metadata_json"], {}), "created_at": r["created_at"]}
                for r in reversed(rows)]

    def research_step(self, key: str, input_hash: str) -> Optional[Dict[str, Any]]:
        with self.conn() as c:
            row = c.execute("SELECT result_json FROM research_steps WHERE step_key=? AND input_hash=?",
                            (key, input_hash)).fetchone()
        return _loads(row["result_json"], {}) if row else None

    def save_research_step(self, key: str, task_id: str, input_hash: str, result: Dict[str, Any]) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO research_steps VALUES(?,?,?,?,?)",
                      (key, task_id, input_hash, _dumps(result), C.utcnow()))
            c.commit()

    def save_lesson(self, lesson_id: str, task_id: str, plan_id: str, domain: str,
                    fingerprint: str, content: Dict[str, Any], *, shared: bool = False) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO lessons VALUES(?,?,?,?,?,?,?,?) "
                      "ON CONFLICT(lesson_id) DO UPDATE SET content_json=excluded.content_json",
                      (lesson_id, task_id, plan_id, domain, fingerprint, _dumps(content),
                       int(shared), C.utcnow()))
            c.commit()

    def lessons(self, task_id: str, domain: str, *, include_shared: bool = True) -> List[Dict[str, Any]]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM lessons WHERE domain=? AND (task_id=? OR shared=?) ORDER BY created_at DESC",
                (domain, task_id, 1 if include_shared else -1)).fetchall()
        return [{**dict(r), "content": _loads(r["content_json"], {})} for r in rows]

    # ---- models / evals / experiments / datasets ---------------------- #
    def save_model(self, m: C.ModelManifest) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO models(model_id, task_id, run_id, spec_json, status, "
                      "recommended, created_at) VALUES(?,?,?,?,?,?,?)",
                      (m.model_id, m.task_id, m.run_id, _dumps(m.model_dump()), m.status,
                       1 if m.recommended else 0, m.created_at))
            c.commit()

    def get_model(self, model_id: str) -> Optional[C.ModelManifest]:
        with self.conn() as c:
            row = c.execute("SELECT spec_json FROM models WHERE model_id=?", (model_id,)).fetchone()
        return C.ModelManifest(**_loads(row["spec_json"], {})) if row else None

    def list_models(self, task_id: str) -> List[C.ModelManifest]:
        with self.conn() as c:
            rows = c.execute("SELECT spec_json FROM models WHERE task_id=? ORDER BY created_at", (task_id,)).fetchall()
        return [C.ModelManifest(**_loads(r["spec_json"], {})) for r in rows]

    def set_recommended(self, task_id: str, model_id: str) -> None:
        # rewrite the manifests (not just the flag columns) so readers that
        # reconstruct ModelManifest from spec_json see the same values.
        for m in self.list_models(task_id):
            m.recommended = m.model_id == model_id
            m.status = "recommended" if m.recommended else "candidate"
            self.save_model(m)

    def save_eval(self, e: C.EvalResult) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO evaluations(eval_id, task_id, model_id, protocol_id, "
                      "spec_json, created_at) VALUES(?,?,?,?,?,?)",
                      (e.eval_id, e.task_id, e.model_id, e.protocol_id, _dumps(e.model_dump()), e.created_at))
            c.commit()

    def list_evals(self, task_id: str, model_id: Optional[str] = None) -> List[C.EvalResult]:
        with self.conn() as c:
            if model_id:
                rows = c.execute("SELECT spec_json FROM evaluations WHERE task_id=? AND model_id=? ORDER BY created_at",
                                 (task_id, model_id)).fetchall()
            else:
                rows = c.execute("SELECT spec_json FROM evaluations WHERE task_id=? ORDER BY created_at",
                                 (task_id,)).fetchall()
        return [C.EvalResult(**_loads(r["spec_json"], {})) for r in rows]

    def save_experiment(self, x: C.ExperimentSpec) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO experiments(run_id, task_id, plan_id, candidate_id, spec_json, "
                      "created_at) VALUES(?,?,?,?,?,?)",
                      (x.run_id, x.task_id, x.plan_id, x.candidate_id, _dumps(x.model_dump()), C.utcnow()))
            c.commit()

    def get_experiment(self, run_id: str) -> Optional[C.ExperimentSpec]:
        with self.conn() as c:
            row = c.execute("SELECT spec_json FROM experiments WHERE run_id=?", (run_id,)).fetchone()
        return C.ExperimentSpec(**_loads(row["spec_json"], {})) if row else None

    def save_dataset(self, d: C.DatasetManifest) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO datasets(dataset_id, version, manifest_json, created_at) "
                      "VALUES(?,?,?,?)", (d.dataset_id, d.version, _dumps(d.model_dump()), d.created_at))
            c.commit()

    def get_dataset(self, dataset_id: str, version: str) -> Optional[C.DatasetManifest]:
        with self.conn() as c:
            row = c.execute("SELECT manifest_json FROM datasets WHERE dataset_id=? AND version=?",
                            (dataset_id, version)).fetchone()
        return C.DatasetManifest(**_loads(row["manifest_json"], {})) if row else None

    # ---- feedback ------------------------------------------------------ #
    def save_feedback(self, f: C.Feedback) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO feedback(feedback_id, task_id, model_id, spec_json, created_at) "
                      "VALUES(?,?,?,?,?)", (f.feedback_id, f.task_id, f.model_id, _dumps(f.model_dump()),
                                            f.created_at))
            c.commit()

    def list_feedback(self, task_id: str) -> List[C.Feedback]:
        with self.conn() as c:
            rows = c.execute("SELECT spec_json FROM feedback WHERE task_id=? ORDER BY created_at",
                             (task_id,)).fetchall()
        return [C.Feedback(**_loads(r["spec_json"], {})) for r in rows]

    # ---- budget ledger ------------------------------------------------- #
    def get_budget(self, task_id: str) -> Dict[str, Any]:
        with self.conn() as c:
            row = c.execute("SELECT reserved_json, used_json FROM budget WHERE task_id=?", (task_id,)).fetchone()
        if not row:
            return {"reserved": {}, "used": {}}
        return {"reserved": _loads(row["reserved_json"], {}), "used": _loads(row["used_json"], {})}

    def claim_budget(self, task_id: str, key: str, amount: float, limit: float) -> None:
        """Atomically charge an attempt before starting its side effect."""
        with self.conn() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT reserved_json, used_json FROM budget WHERE task_id=?", (task_id,)).fetchone()
            used = _loads(row["used_json"], {}) if row else {}
            reserved = _loads(row["reserved_json"], {}) if row else {}
            if used.get(key, 0) + reserved.get(key, 0) + amount > limit:
                raise BudgetExceeded(f"{key} budget exhausted (limit={limit})")
            used[key] = used.get(key, 0) + amount
            c.execute("INSERT INTO budget VALUES(?,?,?,?) ON CONFLICT(task_id) DO UPDATE SET "
                      "used_json=excluded.used_json, updated_at=excluded.updated_at",
                      (task_id, _dumps(reserved), _dumps(used), C.utcnow()))
            c.commit()

    def reserve_budget(self, task_id: str, key: str, amount: float, limit: Optional[float] = None) -> None:
        with self.conn() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT reserved_json, used_json FROM budget WHERE task_id=?", (task_id,)).fetchone()
            reserved = _loads(row["reserved_json"], {}) if row else {}
            used = _loads(row["used_json"], {}) if row else {}
            if limit is not None and used.get(key, 0) + reserved.get(key, 0) + amount > limit:
                raise BudgetExceeded(f"{key} budget exhausted (limit={limit})")
            reserved[key] = reserved.get(key, 0) + amount
            c.execute("INSERT INTO budget(task_id, reserved_json, used_json, updated_at) VALUES(?,?,?,?) "
                      "ON CONFLICT(task_id) DO UPDATE SET reserved_json=excluded.reserved_json, updated_at=excluded.updated_at",
                      (task_id, _dumps(reserved), _dumps(_loads(row["used_json"], {}) if row else {}), C.utcnow()))
            c.commit()

    def settle_budget(self, task_id: str, key: str, used: Optional[float], reserved: float = 0.0) -> None:
        with self.conn() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT reserved_json, used_json FROM budget WHERE task_id=?", (task_id,)).fetchone()
            reserved_d = _loads(row["reserved_json"], {}) if row else {}
            used_d = _loads(row["used_json"], {}) if row else {}
            if reserved:
                reserved_d[key] = max(0.0, reserved_d.get(key, 0) - reserved)
            if used is None:
                used_d.setdefault(f"{key}__unknown", True)
            else:
                used_d[key] = used_d.get(key, 0) + used
            c.execute("INSERT INTO budget(task_id, reserved_json, used_json, updated_at) VALUES(?,?,?,?) "
                      "ON CONFLICT(task_id) DO UPDATE SET reserved_json=excluded.reserved_json, "
                      "used_json=excluded.used_json, updated_at=excluded.updated_at",
                      (task_id, _dumps(reserved_d), _dumps(used_d), C.utcnow()))
            c.commit()

    def add_ledger_entry(self, task_id: str, role: str, kind: str, tokens_in: Optional[int],
                         tokens_out: Optional[int], cost: Optional[float], currency: str,
                         price_version: str, usage: Dict[str, Any]) -> str:
        entry_id = new_id("ledger")
        with self.conn() as c:
            c.execute("INSERT INTO ledger(entry_id, task_id, role, kind, tokens_in, tokens_out, cost, "
                      "currency, price_version, usage_json, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                      (entry_id, task_id, role, kind, tokens_in, tokens_out, cost, currency,
                       price_version, _dumps(usage), C.utcnow()))
            c.commit()
        return entry_id

    def list_ledger(self, task_id: str) -> List[Dict[str, Any]]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM ledger WHERE task_id=? ORDER BY created_at", (task_id,)).fetchall()
        return [dict(r) for r in rows]

    # ---- sessions / invocations --------------------------------------- #
    def upsert_session(self, session_id: str, task_id: str, role: str, plan_id: str = "",
                       baseline_version: str = "") -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO sessions(session_id, task_id, role, plan_id, baseline_version, turns, "
                "created_at, updated_at) VALUES(?,?,?,?,?,0,?,?) "
                "ON CONFLICT(session_id) DO UPDATE SET turns=turns+1, updated_at=excluded.updated_at, "
                "plan_id=excluded.plan_id, baseline_version=excluded.baseline_version",
                (session_id, task_id, role, plan_id, baseline_version, C.utcnow(), C.utcnow()))
            c.commit()

    def find_session(self, task_id: str, role: str, plan_id: str, baseline_version: str) -> Optional[str]:
        with self.conn() as c:
            row = c.execute(
                "SELECT session_id FROM sessions WHERE task_id=? AND role=? AND plan_id=? AND baseline_version=?",
                (task_id, role, plan_id, baseline_version)).fetchone()
        return row["session_id"] if row else None

    def save_invocation(self, inv: C.AgentInvocation) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO invocations(invocation_id, task_id, role, plan_id, session_id, "
                      "spec_json, created_at) VALUES(?,?,?,?,?,?,?)",
                      (inv.invocation_id, inv.task_id, inv.role, inv.plan_id, inv.session_id,
                       _dumps(inv.model_dump()), inv.created_at))
            c.commit()

    # ---- assets / evidence -------------------------------------------- #
    def register_asset(self, manifest: C.AssetManifest, body_ref: str = "") -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO assets(asset_id, asset_version, task_id, asset_type, status, "
                      "manifest_json, body_ref, created_at) VALUES(?,?,?,?,?,?,?,?)",
                      (manifest.asset_id, manifest.asset_version, manifest.source_task, manifest.asset_type,
                       manifest.validation_status, _dumps(manifest.model_dump()), body_ref, manifest.created_at))
            c.commit()

    def get_asset(self, asset_id: str, version: str) -> Optional[C.AssetManifest]:
        with self.conn() as c:
            row = c.execute("SELECT manifest_json FROM assets WHERE asset_id=? AND asset_version=?",
                            (asset_id, version)).fetchone()
        return C.AssetManifest(**_loads(row["manifest_json"], {})) if row else None

    def search_assets(self, asset_type: Optional[str] = None, status: Optional[str] = None) -> List[C.AssetManifest]:
        q = "SELECT manifest_json FROM assets WHERE 1=1"
        params: List[Any] = []
        if asset_type:
            q += " AND asset_type=?"; params.append(asset_type)
        if status:
            q += " AND status=?"; params.append(status)
        with self.conn() as c:
            rows = c.execute(q, params).fetchall()
        return [C.AssetManifest(**_loads(r["manifest_json"], {})) for r in rows]

    def set_asset_status(self, asset_id: str, version: str, status: str) -> None:
        with self.conn() as c:
            row = c.execute("SELECT manifest_json FROM assets WHERE asset_id=? AND asset_version=?",
                            (asset_id, version)).fetchone()
            if not row:
                return
            m = _loads(row["manifest_json"], {})
            m["validation_status"] = status
            m["validated_at"] = C.utcnow() if status != "researched" else m.get("validated_at")
            c.execute("UPDATE assets SET status=?, manifest_json=? WHERE asset_id=? AND asset_version=?",
                      (status, _dumps(m), asset_id, version))
            c.commit()
        # A status change alters compatibility judgements (e.g. deprecated is
        # incompatible); drop any cached verdict so reuse never reads a stale one.
        self.cache_invalidate_by_dependency(f"{asset_id}:{version}")

    def register_evidence(self, ref: C.EvidenceRef, task_id: str) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO evidence(evidence_id, task_id, uri, source_type, version, "
                      "checksum, scope, upload_policy, trust, locator, ref_json, accessed_at) "
                      "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                      (ref.evidence_id, task_id, ref.uri, ref.source_type, ref.version, ref.checksum,
                       ref.scope, ref.upload_policy, ref.trust, ref.locator, _dumps(ref.model_dump()),
                       ref.accessed_at))
            c.commit()

    def get_evidence(self, evidence_id: str) -> Optional[C.EvidenceRef]:
        with self.conn() as c:
            row = c.execute("SELECT ref_json FROM evidence WHERE evidence_id=?", (evidence_id,)).fetchone()
        return C.EvidenceRef(**_loads(row["ref_json"], {})) if row else None

    def list_evidence(self, task_id: str) -> List[C.EvidenceRef]:
        with self.conn() as c:
            rows = c.execute("SELECT ref_json FROM evidence WHERE task_id=?", (task_id,)).fetchall()
        return [C.EvidenceRef(**_loads(r["ref_json"], {})) for r in rows]

    def add_read_audit(self, audit: C.ReadAudit) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO read_audits(audit_id, task_id, role, invocation_id, refs_json, "
                      "requested_range, actual_range, upload_bytes, status, usage_json, created_at) "
                      "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                      (audit.audit_id, "", audit.role, audit.invocation_id, _dumps(audit.refs),
                       audit.requested_range, audit.actual_range, audit.upload_bytes, audit.status,
                       _dumps(audit.usage), C.utcnow()))
            c.commit()

    def cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        with self.conn() as c:
            row = c.execute("SELECT value_json FROM cache WHERE cache_key=?", (key,)).fetchone()
        return _loads(row["value_json"], None) if row else None

    def cache_put(self, key: str, value: Any, dependencies: List[str]) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO cache(cache_key, dependencies_json, value_json, created_at) "
                      "VALUES(?,?,?,?)", (key, _dumps(dependencies), _dumps(value), C.utcnow()))
            c.commit()

    def cache_invalidate_by_dependency(self, dependency: str) -> int:
        removed = 0
        with self.conn() as c:
            rows = c.execute("SELECT cache_key, dependencies_json FROM cache").fetchall()
            for r in rows:
                deps = _loads(r["dependencies_json"], [])
                if dependency in deps:
                    c.execute("DELETE FROM cache WHERE cache_key=?", (r["cache_key"],))
                    removed += 1
            c.commit()
        return removed

    # ---- change requests ---------------------------------------------- #
    def save_change_request(self, task_id: str, plan_id: str, invocation_id: str, cr: C.ChangeRequest) -> str:
        cr_id = new_id("cr")
        with self.conn() as c:
            c.execute("INSERT INTO change_requests(cr_id, task_id, plan_id, invocation_id, spec_json, status, "
                      "created_at) VALUES(?,?,?,?,?,?,?)",
                      (cr_id, task_id, plan_id, invocation_id, _dumps(cr.model_dump()), "open", C.utcnow()))
            c.commit()
        return cr_id

    def list_change_requests(self, task_id: str) -> List[Dict[str, Any]]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM change_requests WHERE task_id=? ORDER BY created_at", (task_id,)).fetchall()
        return [{"cr_id": r["cr_id"], "spec": _loads(r["spec_json"], {}), "status": r["status"],
                 "created_at": r["created_at"]} for r in rows]


def _pid_alive(pid: int) -> bool:
    import os
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def process_identity(pid: int) -> Optional[str]:
    """Linux boot + process start ticks, so PID reuse is not a surviving job."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        ticks = stat.rsplit(")", 1)[1].split()[19]
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() + ":" + ticks
    except (OSError, IndexError):
        return None


__all__ = ["Repository", "StateConflict", "new_id"]
