"""CodexCodingAgent: schedules an isolated coding unit and hands back results.

Creates an isolated workspace snapshot, dispatches to a CodingExecutor (real
Codex CLI or the offline template executor), enforces the write scope, applies
the diff back to the task method pool, and archives the coding record.
"""

from __future__ import annotations

import shutil
import uuid
import shlex
import subprocess
import sys
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Settings
from ..paths import DIRECTORY_RULES, TaskPaths, relpath, sha256_file, sha256_obj, write_json
from ..providers import CodingExecutor, CodingOutcome
from ..schemas import contracts as C
from ..schemas.enums import AgentRole, ResultStatus
from ..storage.repository import Repository
from ..paths import safe_join
from ..policy import check_running, remaining_seconds
from ..execution.runner import _child_env


class WriteScopeError(RuntimeError):
    pass


class CodexCodingAgent:
    role = AgentRole.CODEX_CODING.value

    def __init__(self, settings: Settings, repo: Repository, executor: CodingExecutor,
                 fallback: Optional[CodingExecutor] = None):
        self.settings = settings
        self.repo = repo
        self.executor = executor
        self.fallback = fallback

    def _select_executor(self, task_id: str, require_real: bool = False) -> CodingExecutor:
        status = self.executor.status()
        if status.available:
            return self.executor
        if self.fallback is not None and not require_real:
            self.repo.append_event(task_id, "coding.fallback",
                                   f"primary executor unavailable ({status.detail}); using offline fallback")
            return self.fallback
        raise RuntimeError(f"no coding executor available: {status.detail}")

    def submit(self, spec: C.CodingTaskSpec, context: Dict[str, Any]) -> C.CodingResult:
        tp = TaskPaths(self.settings, context.get("task_slug", spec.task_id))
        workspace = safe_join(tp.coding_workspaces, f"{spec.coding_id}_{uuid.uuid4().hex[:8]}")
        workspace.mkdir(parents=True, exist_ok=True)
        temp = safe_join(tp.runtime, "tmp", workspace.name)
        temp.mkdir(parents=True, exist_ok=True)
        child_env = {**_child_env(), "TMPDIR": str(temp), "TMP": str(temp), "TEMP": str(temp)}

        # snapshot the current method pool as the baseline
        baseline_hash = self._snapshot(tp, workspace, spec)
        before = self._files(workspace)
        baseline_text = {p: (workspace / p).read_text(errors="replace") for p in before}
        invocation_id = f"inv_{uuid.uuid4().hex[:10]}"
        session_id = f"sess_{uuid.uuid4().hex[:10]}"
        self.repo.upsert_session(session_id, spec.task_id, self.role, spec.plan_id, baseline_hash)
        self.repo.save_invocation(C.AgentInvocation(
            invocation_id=invocation_id, task_id=spec.task_id, role=self.role, plan_id=spec.plan_id,
            session_id=session_id, budget=spec.budget))

        try:
            executor = self._select_executor(spec.task_id, spec.require_real)
        except RuntimeError as exc:
            return C.CodingResult(coding_id=spec.coding_id, status="blocked",
                                  open_issues=[str(exc)], change_requests=[C.ChangeRequest(reason=str(exc))])
        if spec.require_real and executor.name in {"offline", "offline_template", "offline_coding"}:
            return C.CodingResult(coding_id=spec.coding_id, status="blocked",
                                  open_issues=["novel code requires a real coding executor"])
        ctx = {**context, "workspace": str(workspace), "task_slug": tp.slug,
               "invocation_id": invocation_id, "directory_contract": DIRECTORY_RULES,
               "temp_dir": str(temp),
               "check_running": lambda: check_running(self.repo, spec.task_id)}
        applied: List[str] = []
        rejected: List[str] = []
        outcome = CodingOutcome(status="failed")
        verification = []
        for attempt in range(spec.budget.max_fix_attempts + 1):
            check_running(self.repo, spec.task_id)
            self.repo.claim_budget(spec.task_id, "coding_calls", 1, spec.budget.max_coding_calls)
            if attempt:
                self.repo.claim_budget(spec.task_id, "fix_attempts", 1, spec.budget.max_fix_attempts)
            limited = spec.model_copy(update={"budget": spec.budget.model_copy(
                update={"max_wall_clock_s": max(1, remaining_seconds(self.repo.get_spec(spec.task_id)))})})
            try:
                outcome = executor.execute(limited, ctx)
            except (OSError, subprocess.TimeoutExpired) as exc:
                outcome = CodingOutcome(status="failed", tool=executor.name, open_issues=[str(exc)])
            blocked = outcome.status == "blocked"
            after = self._files(workspace)
            changed = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
            rejected = [p for p in changed if not self._in_scope(p, spec.write_paths)]
            rejected += [str(p.relative_to(workspace)) for p in workspace.rglob("*") if p.is_symlink()]
            verification = []
            if not rejected and outcome.status == "completed":
                for command in spec.acceptance_commands:
                    check_running(self.repo, spec.task_id)
                    argv = shlex.split(command)
                    if argv and argv[0] in {"python", "python3"}:
                        argv[0] = sys.executable
                    try:
                        proc = subprocess.run(argv, cwd=workspace, env=child_env, capture_output=True,
                                              text=True, timeout=max(1, min(120, remaining_seconds(
                                                  self.repo.get_spec(spec.task_id)))))
                        verification.append(C.Verification(command=command, exit_code=proc.returncode,
                                                            stdout_tail=proc.stdout[-2000:],
                                                            stderr_tail=proc.stderr[-2000:]))
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        verification.append(C.Verification(command=command, exit_code=1, stderr_tail=str(exc)))
                if spec.entry and not safe_join(workspace, spec.entry).is_file():
                    verification.append(C.Verification(command="entry exists", exit_code=1))
            valid = (outcome.status == "completed" and not rejected
                     and bool(verification) and all(v.exit_code == 0 for v in verification))
            if valid:
                after = self._files(workspace)
                changed = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
                rejected = [p for p in changed if not self._in_scope(p, spec.write_paths)]
                rejected += [str(p.relative_to(workspace)) for p in workspace.rglob("*") if p.is_symlink()]
                valid = not rejected
            if spec.require_real and not changed:
                valid = False
                outcome.open_issues.append("no implementation changes detected")
            if valid:
                check_running(self.repo, spec.task_id)
                # Apply only a completely validated transaction. Failed attempts
                # stay in their workspace for diagnosis and never replace baseline code.
                for rel in changed:
                    src = safe_join(workspace, rel)
                    dst = safe_join(tp.root, rel)
                    if not src.exists():
                        if dst.exists():
                            dst.unlink()
                    else:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
                    applied.append(rel)
                break
            outcome.status = "blocked" if blocked else "failed"
            ctx["repair_feedback"] = {"attempt": attempt, "rejected_files": rejected,
                                      "verifications": [v.model_dump() for v in verification],
                                      "open_issues": outcome.open_issues}
            self.repo.append_event(spec.task_id, "coding.verification_failed",
                                   "coding attempt failed independent verification", data=ctx["repair_feedback"])
            if blocked or rejected:
                break
        outcome.verifications = verification
        outcome.diff = "\n".join("".join(difflib.unified_diff(
            baseline_text.get(p, "").splitlines(True),
            (workspace / p).read_text(errors="replace").splitlines(True) if (workspace / p).is_file() else [],
            fromfile=f"a/{p}", tofile=f"b/{p}")) for p in changed if p not in rejected)

        change_requests = list(outcome.change_requests)
        if rejected:
            change_requests.append(C.ChangeRequest(
                reason="coding unit wrote outside the approved write scope",
                affected_constraints=["write_paths"], new_fields={"rejected_files": rejected},
                suggested_next_action="return to planning to widen scope explicitly"))

        coding_id = spec.coding_id
        record_dir = tp.coding_records / coding_id
        record_dir.mkdir(parents=True, exist_ok=True)
        write_json(record_dir / "task.json", spec.model_dump())
        result = C.CodingResult(
            coding_id=coding_id, status=outcome.status, diff=outcome.diff, changed_files=applied,
            tool=outcome.tool, model=outcome.model, base_version=baseline_hash,
            dependency_lock=context.get("dependency_lock", ""),
            verifications=outcome.verifications, cost=outcome.cost, open_issues=outcome.open_issues,
            change_requests=change_requests, archive_ref=relpath(self.settings.root, record_dir))
        write_json(record_dir / "result.json", result.model_dump())
        (record_dir / "diff.patch").write_text(outcome.diff, encoding="utf-8")
        write_json(record_dir / "events.json", outcome.events)
        write_json(record_dir / "tool.json", {"executor": outcome.tool, "model": outcome.model,
                                              "capabilities": executor.capabilities() if hasattr(executor, "capabilities") else {}})

        self.repo.append_event(spec.task_id, "agent.codex_coding",
                               f"coding {coding_id}: status={outcome.status} files={len(applied)}",
                               data={"invocation_id": invocation_id, "session_id": session_id,
                                     "rejected": rejected, "tool": outcome.tool})
        return result

    def _snapshot(self, tp: TaskPaths, workspace: Path, spec: C.CodingTaskSpec) -> str:
        pool = tp.model_pool
        if pool.exists():
            shutil.copytree(pool, workspace / "model_pool", dirs_exist_ok=True, symlinks=True)
        import algorithm_plugins
        source_root = Path(algorithm_plugins.__file__).parent.parent
        for rel in spec.context_files:
            if not rel.startswith("algorithm_plugins/") or not rel.endswith(".py"):
                continue
            source = safe_join(source_root, rel)
            if source.is_file():
                dest = safe_join(workspace, rel)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
        write_json(workspace / "coding_contract.json", spec.model_dump())
        hashes = {}
        for f in sorted((workspace / "model_pool").rglob("*")):
            if f.is_file():
                hashes[relpath(workspace, f)] = sha256_file(f)[:16]
        return sha256_obj(hashes)

    @staticmethod
    def _files(workspace: Path) -> Dict[str, str]:
        return {str(p.relative_to(workspace)): sha256_file(p)
                for p in workspace.rglob("*") if p.is_file() and not p.is_symlink()
                and "__pycache__" not in p.parts and ".git" not in p.parts}

    @staticmethod
    def _in_scope(rel: str, paths: List[str]) -> bool:
        path = Path(rel)
        if path.is_absolute() or ".." in path.parts:
            return False
        return any(path == Path(root) or Path(root) in path.parents for root in paths)


__all__ = ["CodexCodingAgent", "WriteScopeError"]
