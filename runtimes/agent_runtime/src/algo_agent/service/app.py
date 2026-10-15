"""FastAPI control service (guide 5). Binds to 127.0.0.1 by default.

Long operations (graph runs, training) never block the request thread: command
endpoints return a command_id/job_id immediately and the graph runs in a worker
thread over the persistent checkpointer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import load_settings
from ..schemas.enums import Intent
from .controller import Controller, ControllerError

_CONTROLLER: Optional[Controller] = None


def get_controller(settings=None) -> Controller:
    global _CONTROLLER
    if _CONTROLLER is None:
        settings = settings or load_settings()
        _CONTROLLER = Controller(settings, inline=False)
        _CONTROLLER.rt.start_workers()
    return _CONTROLLER


def create_app(settings=None) -> FastAPI:
    app = FastAPI(title="algo-agent control service", version="0.4.0")

    @app.exception_handler(ControllerError)
    async def _ctrl_err(_request, exc: ControllerError):  # pragma: no cover - thin wrapper
        return JSONResponse(status_code=exc.status, content=exc.to_dict())

    @app.get("/health")
    def health() -> Dict[str, Any]:
        ctrl = get_controller(settings)
        return {"status": "ok", "root": str(ctrl.settings.root),
                "offline": ctrl.settings.offline(),
                "worker_running": ctrl.rt.worker.running(),
                "providers": [s.name for s in ctrl.rt.providers.values()
                              if hasattr(s, "status") and s.status().available]}

    @app.get("/tasks")
    def list_tasks() -> List[Dict[str, Any]]:
        return get_controller(settings).repo.list_tasks()

    @app.post("/tasks")
    def create_task(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        ctrl = get_controller(settings)
        text = body.get("text", "")
        task_id = body.get("task_id")
        spec = ctrl.create_task(text, task_id=task_id, demo=body.get("demo", False),
                                synthetic=body.get("synthetic", False),
                                methods_hint=body.get("methods_hint"),
                                auto_process=body.get("auto_process", False),
                                domain=body.get("domain", ""), task_spec=body.get("task_spec"),
                                parent_task=body.get("parent_task", ""), track=body.get("track", ""))
        if body.get("run", True):
            ctrl.run_async(spec.task_id, raw_text=text)
        return {"task_id": spec.task_id, "task_slug": spec.task_slug, "accepted": True}

    @app.get("/tasks/{task_id}/state")
    def state(task_id: str) -> Dict[str, Any]:
        return get_controller(settings).state(task_id)

    @app.get("/tasks/{task_id}/events")
    def events(task_id: str, after: int = Query(0, ge=0), limit: int = Query(500, ge=1, le=2000)) -> List[Dict[str, Any]]:
        return get_controller(settings).events(task_id, after=after, limit=limit)

    @app.get("/tasks/{task_id}/events/stream")
    def events_stream(task_id: str, after: int = Query(0, ge=0)) -> StreamingResponse:
        ctrl = get_controller(settings)
        return StreamingResponse(ctrl.rt.bus.sse(task_id, after=after), media_type="text/event-stream")

    @app.get("/tasks/{task_id}/plans/{plan_id}")
    def get_plan(task_id: str, plan_id: str) -> Dict[str, Any]:
        return get_controller(settings).plan_view(task_id, plan_id)

    @app.post("/tasks/{task_id}/ask")
    def ask(task_id: str, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        return get_controller(settings).answer(task_id, str(body.get("question", "")))

    @app.post("/tasks/{task_id}/commands")
    def commands(task_id: str, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        ctrl = get_controller(settings)
        return ctrl.submit_command(
            task_id, body.get("intent", Intent.UNKNOWN.value),
            payload=body.get("payload", {}) or body.get("args", {}), text=body.get("text", ""),
            conversation_id=body.get("conversation_id", ""),
            expected_state_version=body.get("expected_state_version"),
            idempotency_key=body.get("idempotency_key"), async_run=body.get("async", True))

    @app.post("/tasks/{task_id}/approvals")
    def approve(task_id: str, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        ctrl = get_controller(settings)
        return ctrl.approve(task_id, body.get("plan_id", ""), body.get("plan_hash", ""),
                            expected_state_version=body.get("expected_state_version"),
                            scope=body.get("scope"), mode=body.get("mode"),
                            conversation_id=body.get("conversation_id", ""),
                            idempotency_key=body.get("idempotency_key"))

    @app.post("/tasks/{task_id}/feedback")
    def feedback(task_id: str, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        return get_controller(settings).submit_feedback(task_id, body)

    @app.post("/tasks/{task_id}/control")
    def control(task_id: str, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        ctrl = get_controller(settings)
        action = body.get("action", "")
        if action == "pause":
            return {"state": ctrl.pause(task_id)}
        if action == "resume":
            st = ctrl.resume_task(task_id)
            ctrl.run_async(task_id)
            return {"state": st}
        if action == "cancel":
            return ctrl.cancel(task_id, body.get("job_id"))
        raise ControllerError("bad_action", f"unknown control action {action!r}")

    @app.post("/tasks/{task_id}/export")
    def export(task_id: str, body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        return get_controller(settings).export(task_id, iteration=body.get("iteration"))

    @app.get("/jobs/{job_id}")
    def job(job_id: str) -> Dict[str, Any]:
        ctrl = get_controller(settings)
        j = ctrl.repo.get_job(job_id)
        if not j:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
        return j.model_dump()

    return app


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - runtime entry
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="algo-agent control service")
    parser.add_argument("--root", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.root)
    host = args.host or settings.env_get("ALGO_SERVICE_HOST") or "127.0.0.1"
    port = args.port or int(settings.env_get("ALGO_SERVICE_PORT") or 8765)
    app = create_app(settings)
    print(f"algo-agent service on http://{host}:{port}  (root={settings.root}, offline={settings.offline()})")
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
