"""Start / stop / inspect the local control service (127.0.0.1 by default)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..config import Settings


def _child_env() -> Dict[str, str]:
    env = dict(os.environ)
    here = Path(__file__).resolve()
    candidates = [str(here.parents[2])]
    for ancestor in here.parents:
        if (ancestor / "algorithm_plugins" / "common" / "plugin.py").exists():
            candidates.append(str(ancestor))
            break
    parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    for cand in candidates:
        if cand not in parts:
            parts.insert(0, cand)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def service_paths(settings: Settings) -> Tuple[Path, Path]:
    return settings.state_dir / "service.pid", settings.state_dir / "service.log"


def _read_pid(settings: Settings) -> Optional[int]:
    pidfile, _ = service_paths(settings)
    if not pidfile.exists():
        return None
    try:
        return int(pidfile.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def service_status(settings: Settings, url: str) -> Dict[str, Any]:
    pid = _read_pid(settings)
    reachable, detail = False, ""
    try:
        import httpx
        r = httpx.get(f"{url}/health", timeout=1.5)
        reachable = r.status_code == 200
        detail = r.json().get("status", "") if reachable else r.text[:120]
    except Exception as exc:
        detail = str(exc)[:120]
    running = bool(pid and _alive(pid))
    return {"running": running, "reachable": reachable, "pid": pid, "url": url, "detail": detail}


def start_service(settings: Settings, *, host: str, port: int, wait_s: float = 15.0,
                  foreground: bool = False) -> Dict[str, Any]:
    url = f"http://{host}:{port}"
    st = service_status(settings, url)
    if st["reachable"]:
        return {**st, "started": False, "note": "service already running"}

    settings.state_dir.mkdir(parents=True, exist_ok=True)
    pidfile, logfile = service_paths(settings)
    argv = [sys.executable, "-m", "algo_agent.service.app",
            "--root", str(settings.root), "--host", host, "--port", str(port)]
    if foreground:
        subprocess.call(argv, env=_child_env())
        return {"started": False, "foreground": True, "url": url}

    log = open(logfile, "ab")
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, env=_child_env())
    pidfile.write_text(str(proc.pid), encoding="utf-8")

    deadline = time.time() + wait_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return {"started": False, "error": f"service exited early (code {proc.returncode})",
                    "log": str(logfile), "url": url}
        st = service_status(settings, url)
        if st["reachable"]:
            return {**st, "started": True, "pid": proc.pid, "url": url, "log": str(logfile)}
        time.sleep(0.4)
    return {"started": True, "reachable": False, "pid": proc.pid, "url": url,
            "log": str(logfile), "note": "started but health check not yet reachable"}


def stop_service(settings: Settings, *, timeout: float = 10.0) -> Dict[str, Any]:
    pid = _read_pid(settings)
    pidfile, _ = service_paths(settings)
    if not pid or not _alive(pid):
        if pidfile.exists():
            pidfile.unlink()
        return {"stopped": False, "note": "no running service pid"}
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        return {"stopped": False, "error": str(exc)}
    deadline = time.time() + timeout
    while time.time() < deadline and _alive(pid):
        time.sleep(0.2)
    if _alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        return {"stopped": True, "pid": pid, "forced": True}
    if pidfile.exists():
        pidfile.unlink()
    return {"stopped": True, "pid": pid}


__all__ = ["start_service", "stop_service", "service_status", "service_paths", "_child_env"]
