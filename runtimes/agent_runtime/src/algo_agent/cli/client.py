"""Control client used by every CLI command and the chat REPL.

There is a single source of business logic (:class:`~algo_agent.service.controller.Controller`).
The client either drives that controller in-process ("embedded") or talks to a
running control service over HTTP. Both paths return the same contract dicts, so
the renderers and the chat commands never need to know which one is active.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..config import Settings

DEFAULT_PAUSE_EXTRAS: Dict[str, Any] = {"pause_between_iterations": True}


class ClientError(RuntimeError):
    def __init__(self, message: str, *, recovery: str = "", status: int = 400):
        super().__init__(message)
        self.message = message
        self.recovery = recovery
        self.status = status


def default_service_url(settings: Settings) -> str:
    host = settings.env_get("ALGO_SERVICE_HOST") or "127.0.0.1"
    port = settings.env_get("ALGO_SERVICE_PORT") or "8765"
    return f"http://{host}:{port}"


class ControlClient:
    def __init__(self, settings: Settings, *, service_url: Optional[str] = None,
                 force_mode: Optional[str] = None, extras: Optional[Dict[str, Any]] = None,
                 verbose: bool = False):
        self.settings = settings
        self.service_url = (service_url or default_service_url(settings)).rstrip("/")
        self.force_mode = force_mode  # "service" | "embedded" | None (auto)
        self.extras = dict(DEFAULT_PAUSE_EXTRAS if extras is None else extras)
        self.verbose = verbose
        self._ctrl = None
        self._mode: Optional[str] = None

    # ---- mode selection ----------------------------------------------- #
    def _probe_service(self) -> bool:
        try:
            import httpx
            r = httpx.get(f"{self.service_url}/health", timeout=1.5)
            return r.status_code == 200
        except Exception:
            return False

    @property
    def mode(self) -> str:
        if self._mode is not None:
            return self._mode
        if self.force_mode == "embedded":
            self._mode = "embedded"
        elif self.force_mode == "service":
            self._mode = "service"
        else:
            self._mode = "service" if self._probe_service() else "embedded"
        return self._mode

    def _controller(self):
        if self._ctrl is None:
            from ..service.controller import Controller
            self._ctrl = Controller(self.settings, inline=True, extras=self.extras)
        return self._ctrl

    # ---- generic helpers ---------------------------------------------- #
    def _req(self, method: str, path: str, **kwargs) -> Any:
        import httpx
        url = f"{self.service_url}{path}"
        try:
            resp = httpx.request(method, url, timeout=kwargs.pop("timeout", 60.0), **kwargs)
        except Exception as exc:
            raise ClientError(f"cannot reach service at {self.service_url}: {exc}",
                              recovery="start it with: algo-agent service start", status=503)
        if resp.status_code >= 400:
            try:
                err = resp.json().get("error", {})
            except Exception:
                err = {"message": resp.text}
            raise ClientError(err.get("message", resp.text), recovery=err.get("recovery", ""),
                              status=resp.status_code)
        if not resp.content:
            return {}
        return resp.json()

    @staticmethod
    def _call(fn, *args, **kwargs):
        from ..service.controller import ControllerError
        try:
            return fn(*args, **kwargs)
        except ControllerError as exc:
            raise ClientError(exc.message, recovery=exc.recovery, status=exc.status)

    # ---- health / reads ----------------------------------------------- #
    def health(self) -> Dict[str, Any]:
        if self.mode == "service":
            return self._req("GET", "/health", timeout=3.0)
        ctrl = self._controller()
        return {"status": "ok", "root": str(self.settings.root), "offline": self.settings.offline(),
                "mode": "embedded", "worker_running": ctrl.rt.worker.running()}

    def list_tasks(self) -> List[Dict[str, Any]]:
        if self.mode == "service":
            return self._req("GET", "/tasks")
        return self._controller().repo.list_tasks()

    def state(self, task_id: str) -> Dict[str, Any]:
        if self.mode == "service":
            return self._req("GET", f"/tasks/{task_id}/state")
        return self._call(self._controller().state, task_id)

    def events(self, task_id: str, after: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
        if self.mode == "service":
            return self._req("GET", f"/tasks/{task_id}/events", params={"after": after, "limit": limit})
        return self._controller().events(task_id, after=after, limit=limit)

    def answer(self, task_id: str, question: str) -> Dict[str, Any]:
        """LLM-backed grounded answer; empty answer means fall back to deterministic."""
        if self.mode == "service":
            return self._req("POST", f"/tasks/{task_id}/ask", timeout=300.0,
                             json={"question": question})
        return self._call(self._controller().answer, task_id, question)

    def plan_view(self, task_id: str, plan_id: str) -> Dict[str, Any]:
        if self.mode == "service":
            return self._req("GET", f"/tasks/{task_id}/plans/{plan_id}")
        return self._call(self._controller().plan_view, task_id, plan_id)

    def budget(self, task_id: str) -> Dict[str, Any]:
        if self.mode == "service":
            return self._req("GET", f"/tasks/{task_id}/state")  # state carries budget_used
        return self._call(self._controller().budget, task_id)

    def jobs(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.mode == "service":
            st = self._req("GET", "/tasks")
            return st
        return self._call(self._controller().jobs, task_id)

    def models(self, task_id: str) -> List[Dict[str, Any]]:
        if self.mode == "service":
            return []
        return [m.model_dump() for m in self._controller().repo.list_models(task_id)]

    def evals(self, task_id: str) -> List[Dict[str, Any]]:
        if self.mode == "service":
            return []
        return [e.model_dump() for e in self._controller().repo.list_evals(task_id)]

    # ---- task lifecycle ----------------------------------------------- #
    def create_task(self, text: str, task_id: Optional[str] = None, *, demo: bool = False,
                    synthetic: bool = False, methods_hint: Optional[List[str]] = None,
                    run: bool = False, auto_process: bool = False, domain: str = "",
                    task_spec: Optional[Dict[str, Any]] = None,
                    parent_task: str = "", track: str = "") -> Dict[str, Any]:
        if self.mode == "service":
            return self._req("POST", "/tasks", json={
                "text": text, "task_id": task_id, "demo": demo, "synthetic": synthetic,
                "methods_hint": methods_hint, "run": run, "auto_process": auto_process,
                "domain": domain, "task_spec": task_spec, "parent_task": parent_task, "track": track})
        ctrl = self._controller()
        spec = self._call(ctrl.create_task, text, task_id, demo=demo, synthetic=synthetic,
                          methods_hint=methods_hint, auto_process=auto_process,
                          domain=domain, task_spec=task_spec, parent_task=parent_task, track=track)
        result = {"task_id": spec.task_id, "task_slug": spec.task_slug, "accepted": True}
        if run:
            result["boundary"] = self._call(ctrl.run, spec.task_id)
        return result

    def run(self, task_id: str, *, raw_text: str = "") -> Dict[str, Any]:
        if self.mode == "service":
            # ask the running service to (re)drive the graph, then wait for a boundary
            self._req("POST", f"/tasks/{task_id}/control", json={"action": "resume"})
            return self.wait_boundary(task_id)
        return self._call(self._controller().run, task_id, raw_text=raw_text)

    def approve(self, task_id: str, plan_id: str, plan_hash: str, *,
                expected_state_version: Optional[int] = None,
                scope: Optional[Dict[str, Any]] = None, mode: Optional[str] = None,
                idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        if self.mode == "service":
            return self._req("POST", f"/tasks/{task_id}/approvals", timeout=600.0, json={
                "plan_id": plan_id, "plan_hash": plan_hash,
                "expected_state_version": expected_state_version, "scope": scope, "mode": mode,
                "idempotency_key": idempotency_key})
        return self._call(self._controller().approve, task_id, plan_id, plan_hash,
                          expected_state_version=expected_state_version, scope=scope, mode=mode,
                          idempotency_key=idempotency_key)

    def reject(self, task_id: str, plan_id: str, reason: str = "") -> Dict[str, Any]:
        if self.mode == "service":
            return self._req("POST", f"/tasks/{task_id}/approvals", timeout=600.0,
                             json={"plan_id": plan_id, "plan_hash": "reject", "reject": True,
                                   "reason": reason})
        return self._call(self._controller().reject, task_id, plan_id, reason)

    def submit_feedback(self, task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode == "service":
            return self._req("POST", f"/tasks/{task_id}/feedback", timeout=600.0, json=payload)
        return self._call(self._controller().submit_feedback, task_id, payload)

    def continue_task(self, task_id: str, *, stop: bool = False) -> Dict[str, Any]:
        if self.mode == "service":
            if stop:
                self._req("POST", f"/tasks/{task_id}/control", json={"action": "cancel"})
                return self.state(task_id)
            return self._req("POST", f"/tasks/{task_id}/feedback", json={"continue": True})
        return self._call(self._controller().continue_task, task_id, stop=stop)

    def control(self, task_id: str, action: str, *, job_id: Optional[str] = None) -> Dict[str, Any]:
        if self.mode == "service":
            return self._req("POST", f"/tasks/{task_id}/control", json={"action": action, "job_id": job_id})
        ctrl = self._controller()
        if action == "pause":
            return {"state": self._call(ctrl.pause, task_id)}
        if action == "resume":
            st = self._call(ctrl.resume_task, task_id)
            try:
                boundary = self._call(ctrl.run, task_id)
                return {"state": st, "boundary": boundary}
            except ClientError:
                return {"state": st}
        if action == "cancel":
            return self._call(ctrl.cancel, task_id, job_id)
        raise ClientError(f"unknown control action {action!r}")

    def export(self, task_id: str, *, iteration: Optional[int] = None) -> Dict[str, Any]:
        if self.mode == "service":
            return self._req("POST", f"/tasks/{task_id}/export", timeout=600.0,
                             json={"iteration": iteration})
        return self._call(self._controller().export, task_id, iteration=iteration)

    # ---- waiting ------------------------------------------------------ #
    TERMINAL_PHASES = {"accepted", "stopped", "failed"}

    def wait_boundary(self, task_id: str, timeout: float = 1800.0, poll_s: float = 0.4) -> Dict[str, Any]:
        """Poll until the task reaches an interrupt or a terminal phase (service mode)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            st = self.state(task_id)
            pending = st.get("pending_approval")
            if pending:
                return {"type": pending.get("type", "interrupt"), "interrupt": pending,
                        "state": st, "done": False}
            if st.get("phase") in self.TERMINAL_PHASES:
                return {"type": "end", "interrupt": None, "state": st, "done": True}
            if st.get("phase") in {"awaiting_input", "paused"}:
                return {"type": st["phase"], "interrupt": None, "state": st, "done": False}
            time.sleep(poll_s)
        raise ClientError("timed out waiting for the task to reach a boundary",
                          recovery="check: algo-agent status", status=504)


__all__ = ["ControlClient", "ClientError", "default_service_url"]
