"""Background worker: leases queued jobs and executes them.

Runs inside the control service process (or in-process for the offline CLI).
Only job registration and results are persisted; long work happens in the
Runner's child process, never on the FastAPI request thread (guide 2, 7).
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Callable, Optional

from ..config import Settings
from ..storage.repository import Repository
from .runner import Runner


class Worker:
    def __init__(self, settings: Settings, repo: Repository, runner: Optional[Runner] = None,
                 poll_s: float = 0.5, task_filter: Optional[str] = None):
        self.settings = settings
        self.repo = repo
        self.runner = runner or Runner(settings, repo)
        self.poll_s = poll_s
        self.task_filter = task_filter
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._owner = f"worker_{uuid.uuid4().hex[:8]}"
        self.completed: int = 0
        self._listeners: list[Callable[[dict], None]] = []

    def add_listener(self, cb: Callable[[dict], None]) -> None:
        self._listeners.append(cb)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="algo-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def drain(self, max_jobs: int = 100, task_id: Optional[str] = None) -> int:
        """Execute currently queued jobs synchronously (offline/demo/test path)."""
        done = 0
        self.repo.reconcile_jobs()
        for _ in range(max_jobs):
            job = self.repo.lease_job(task_id or self.task_filter, self._owner)
            if not job:
                break
            self._run(job)
            done += 1
        return done

    def await_jobs(self, job_ids, timeout: float = 1800.0, poll_s: float = 0.25) -> bool:
        """Block until the given jobs reach a terminal state.

        In service mode a background worker thread leases the jobs; in
        standalone/offline mode we drain them ourselves so the loop still makes
        deterministic progress without an external daemon.
        """
        terminal = {"succeeded", "failed", "cancelled"}
        deadline = time.time() + timeout
        while time.time() < deadline:
            jobs = [self.repo.get_job(j) for j in job_ids]
            from ..policy import check_running
            for tid in {j.task_id for j in jobs if j is not None}:
                check_running(self.repo, tid)
            if all(j is not None and j.status in terminal for j in jobs):
                return True
            if not self.running():
                for tid in {j.task_id for j in jobs if j is not None}:
                    self.drain(task_id=tid)
                time.sleep(poll_s)
                continue
            time.sleep(poll_s)
        return False

    def _loop(self) -> None:
        self.repo.reconcile_jobs()
        reconciled = time.monotonic()
        while not self._stop.is_set():
            if time.monotonic() - reconciled >= 15:
                self.repo.reconcile_jobs()
                reconciled = time.monotonic()
            job = self.repo.lease_job(self.task_filter, self._owner)
            if not job:
                time.sleep(self.poll_s)
                continue
            self._run(job)

    def _run(self, job) -> None:
        finished = threading.Event()
        def renew():
            while not finished.wait(5):
                if not self.repo.heartbeat_job(job):
                    break
        heartbeat = threading.Thread(target=renew, name="job-heartbeat", daemon=True)
        heartbeat.start()
        try:
            outcome = self.runner.execute_job(job)
        except Exception as exc:  # never kill the worker loop
            outcome = {"status": "failed", "error": repr(exc)}
            self.repo.finish_job(job, "failed", result=outcome)
        finally:
            finished.set()
            heartbeat.join(timeout=6)
        self.completed += 1
        payload = {"job_id": job.job_id, "task_id": job.task_id, **outcome}
        for cb in self._listeners:
            try:
                cb(payload)
            except Exception:
                pass


__all__ = ["Worker"]
