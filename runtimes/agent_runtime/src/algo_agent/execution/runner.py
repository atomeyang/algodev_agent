"""Runner: submit / status / collect / cancel with idempotent job registration.

Each run executes in its own directory with a frozen input manifest, a dedicated
process (exit code + log captured) and a resource/time cap. Recovery reconciles
DB jobs against real processes before re-queueing (guide 7).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import signal
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import Settings
from ..paths import TaskPaths, read_json, write_json, sha256_file, safe_join
from ..schemas import contracts as C
from ..schemas.enums import JobStatus
from ..storage.repository import Repository
from ..policy import check_running, remaining_seconds


_PROBE: Dict[str, Any] = {}


def probe_compute() -> Dict[str, Any]:
    """Lazily detect the compute actually available to experiment jobs.

    Importing ``torch`` is expensive, so the result is cached for the process.
    A missing torch/GPU degrades gracefully to CPU so the demo path is
    unaffected.
    """
    if "compute" in _PROBE:
        return _PROBE["compute"]
    info: Dict[str, Any] = {"torch": None, "gpu": False, "gpu_devices": []}
    try:
        import torch

        info["torch"] = getattr(torch, "__version__", None)
        info["gpu"] = bool(torch.cuda.is_available())
        if info["gpu"]:
            info["gpu_devices"] = [torch.cuda.get_device_name(i)
                                   for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    _PROBE["compute"] = info
    return info


def _child_env() -> Dict[str, str]:
    """Environment for experiment subprocesses.

    The worker runs as ``python -m algo_agent...`` in its own cwd, so the
    package must remain importable. We prepend the directory that contains the
    ``algo_agent`` package (and any existing PYTHONPATH) so runs work whether or
    not the project was pip-installed.
    """
    env = {key: value for key, value in os.environ.items()
           if not any(word in key.upper() for word in
                      ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTHORIZATION"))}
    here = Path(__file__).resolve()
    candidates = [str(here.parents[2])]  # .../src, so `import algo_agent` works
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
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


class Runner:
    name = "local_process"

    def __init__(self, settings: Settings, repo: Repository):
        self.settings = settings
        self.repo = repo

    def capabilities(self) -> Dict[str, Any]:
        compute = probe_compute()
        return {"backend": "local_subprocess", "gpu": compute["gpu"],
                "gpu_devices": compute["gpu_devices"], "torch": compute["torch"],
                "python": sys.executable, "cancel": "process_terminate",
                "isolation": "per_run_dir",
                "max_wall_clock_s": self.settings.budget_defaults().get("max_wall_clock_s", 3600)}

    # ------------------------------------------------------------------ #
    def submit(self, spec: C.ExperimentSpec, task_slug: str, method_entry: str,
               split: str = "val") -> C.JobRecord:
        for job in self.repo.list_jobs(spec.task_id):
            if job.run_id == spec.run_id:
                return job
        check_running(self.repo, spec.task_id)
        task_spec = self.repo.get_spec(spec.task_id)
        self.repo.claim_budget(spec.task_id, "experiments", 1, task_spec.budget.max_experiments)
        tp = TaskPaths(self.settings, task_slug)
        code = safe_join(tp.runs, spec.run_id, "code")
        code.mkdir(parents=True, exist_ok=True)
        source = tp.resolve_reference(method_entry)
        shutil.copytree(source.parent, code / "method", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        import algorithm_plugins
        plugin_source = Path(algorithm_plugins.__file__).parent
        if spec.env.get("stage") == "final":
            plugin_source = source.parent.parent / "plugins" / "algorithm_plugins"
        shutil.copytree(plugin_source, code / "plugins" / "algorithm_plugins",
                        dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        frozen_entry = code / "method" / source.name
        hashes = {str(p.relative_to(code)): sha256_file(p) for p in code.rglob("*") if p.is_file()}
        spec.code_ref = str(frozen_entry.relative_to(tp.root))
        runtime_env = {key: os.environ[key] for key in
                       ("QAA_ROOT", "IQA_BENCHMARK", "KONIQ_DIR", "IQA_FEATURE_NPZ")
                       if key in os.environ}
        if task_spec.domain == "iqa":
            from algorithm_plugins.iqa import config as iqa_config
            runtime_env.setdefault("QAA_ROOT", str(iqa_config.qaa_root()))
            runtime_env.setdefault("IQA_BENCHMARK", iqa_config.benchmark_source())
        if spec.env.get("stage") == "final" and spec.parent_run_id:
            previous = next(j for j in self.repo.list_jobs(spec.task_id) if j.run_id == spec.parent_run_id)
            runtime_env = previous.payload.get("runtime_env", runtime_env)
        idem = f"{spec.task_id}:{spec.run_id}"
        payload = {
            "run_id": spec.run_id, "task_slug": task_slug, "method": spec.method,
            "method_entry": str(frozen_entry), "split": split, "config": spec.config, "seed": spec.seed,
            "data_version": spec.data_version, "code_hashes": hashes,
            "inference_weights": spec.env.get("inference_weights", ""),
            "stage": spec.env.get("stage", "legacy"),
            "runtime_env": runtime_env,
            "resource_limits": {"memory_mb": task_spec.budget.max_memory_mb,
                                "output_file_mb": task_spec.budget.max_output_file_mb},
        }
        self.repo.save_experiment(spec)
        return self.repo.create_job(spec.task_id, "experiment", idem, run_id=spec.run_id, payload=payload)

    def status(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.repo.get_job(job_id)
        return job.model_dump() if job else None

    def collect(self, job_id: str) -> Dict[str, Any]:
        job = self.repo.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        return {"job": job.model_dump(), "result": job.result}

    def cancel(self, job_id: str) -> bool:
        job = self.repo.get_job(job_id)
        if not job:
            return False
        if job.status == JobStatus.QUEUED.value:
            self.repo.update_job(job_id, status=JobStatus.CANCELLED.value)
            return True
        # running: flag for the worker; the worker terminates the child process.
        self.repo.update_job(job_id, status=JobStatus.CANCEL_REQUESTED.value)
        return True

    # ------------------------------------------------------------------ #
    def execute_job(self, job: C.JobRecord) -> Dict[str, Any]:
        assert job.kind == "experiment", f"unsupported job kind {job.kind}"
        if not self.repo.heartbeat_job(job):
            return {"status": "failed", "error": "job lease lost before execution"}
        payload = job.payload
        tp = TaskPaths(self.settings, payload["task_slug"])
        release = tp.release_dir(payload["data_version"])
        out_dir = safe_join(tp.runs, payload["run_id"])
        out_dir.mkdir(parents=True, exist_ok=True)
        method_entry = str(tp.resolve_reference(payload["method_entry"]))
        check_running(self.repo, job.task_id)
        code = out_dir / "code"
        for rel, digest in payload.get("code_hashes", {}).items():
            if sha256_file(safe_join(code, rel)) != digest:
                raise ValueError("frozen experiment code changed")
        task_spec = self.repo.get_spec(job.task_id)
        if task_spec.auto_process:
            from .data_pipeline import verify_release
            verify_release(tp, payload["data_version"])

        job_file = out_dir / "job.json"
        write_json(job_file, {**payload, "method_entry": method_entry,
                              "release_dir": str(release), "out_dir": str(out_dir),
                              "task_root": str(tp.root)})
        log_path = out_dir / "run.log"
        argv = [sys.executable, "-m", "algo_agent.execution.method_worker", str(job_file)]
        timeout = max(0, remaining_seconds(task_spec))
        env = _child_env()
        env.update(payload.get("runtime_env", {}))
        scratch = safe_join(out_dir, "tmp")
        scratch.mkdir(exist_ok=True)
        env.update(TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch),
                   PYTHONDONTWRITEBYTECODE="1")
        with open(log_path, "w", encoding="utf-8") as log:
            try:
                proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                        cwd=str(out_dir), env=env, start_new_session=True)
            except Exception as exc:
                self.repo.finish_job(job, "failed", {"error": f"spawn failed: {exc}"})
                return {"status": "failed", "error": str(exc)}
            owns_lease = self.repo.heartbeat_job(job, pid=proc.pid)
            deadline = time.monotonic() + timeout
            heartbeat_at = time.monotonic()
            reason = "" if owns_lease else "lease_lost"
            while proc.poll() is None:
                state = self.repo.get_state(job.task_id)
                current = self.repo.get_job(job.job_id)
                if time.monotonic() - heartbeat_at >= 5:
                    if not self.repo.heartbeat_job(job):
                        reason = "lease_lost"
                    heartbeat_at = time.monotonic()
                if state.phase in {"stopped", "failed"} or current.status == JobStatus.CANCEL_REQUESTED.value:
                    reason = "cancelled"
                elif time.monotonic() >= deadline:
                    reason = "timeout"
                if reason:
                    break
                time.sleep(0.1)
            if reason:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                status = "cancelled" if reason == "cancelled" else "failed"
                self.repo.finish_job(job, status, {"error": reason})
                return {"status": status, "error": reason}
            code = proc.wait()

        # a cancel requested mid-run wins over the child exit code
        if self.repo.get_job(job.job_id).status == JobStatus.CANCEL_REQUESTED.value:
            self.repo.finish_job(job, "cancelled", {"cancelled": True})
            return {"status": "cancelled"}

        result = read_json(out_dir / "result.json", {})
        if task_spec.auto_process:
            from .data_pipeline import verify_release
            verify_release(tp, payload["data_version"])
        status = (JobStatus.SUCCEEDED.value if code == 0 and result.get("status") in {"succeeded", "completed"}
                  else JobStatus.FAILED.value)
        if not self.repo.finish_job(job, status, {**result, "exit_code": code}):
            return {"status": "failed", "error": "stale attempt result rejected"}
        return {"status": status, "exit_code": code, "result": result}


__all__ = ["Runner"]
